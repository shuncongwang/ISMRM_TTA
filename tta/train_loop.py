"""Training loop shared by source training and supervised fine-tuning."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from .common import LOG, amp_dtype, load_case
from .data import augment, make_loader
from .infer import sliding_window
from .metrics import dice
from .model import ResEncUNet, deep_supervision_loss, patch_volume_target


def device_of(cfg):
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_model(cfg):
    return ResEncUNet(in_ch=len(cfg["modalities"]), preset=cfg["preset"], deep_supervision=True)


def load_model(cfg, ckpt_path, device=None):
    device = device or device_of(cfg)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    mcfg = dict(cfg)
    mcfg["preset"] = ck.get("preset", cfg["preset"])
    model = build_model(mcfg)
    model.load_state_dict(ck["model"])
    return model.to(device), ck


@torch.no_grad()
def validate(cfg, model, case_ids, device):
    scores = []
    for cid in case_ids[: cfg["val_max_cases"]]:
        c = load_case(cfg, cid)
        prob = sliding_window(model, c["img"], cfg["patch_size"], cfg["sw_overlap"], cfg["sw_batch"],
                              False, device, cfg["amp"])
        scores.append(dice(prob > 0.5, np.asarray(c["seg"]) > 0))
    model.train()
    return float(np.mean(scores)) if scores else float("nan")


def train(cfg, train_ids, val_ids, out_dir: Path, epochs, iters_per_epoch, lr, init_ckpt=None, resume=False,
          tag="train"):
    out_dir.mkdir(parents=True, exist_ok=True)
    device = device_of(cfg)
    model = build_model(cfg).to(device)
    if init_ckpt:
        model.load_state_dict(torch.load(init_ckpt, map_location="cpu", weights_only=False)["model"])
        LOG.info(f"initialised from {init_ckpt}")
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.99, nesterov=True, weight_decay=cfg["weight_decay"])
    start_epoch, best = 0, -1.0
    last = out_dir / "checkpoint_last.pt"
    if resume and last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        start_epoch, best = ck["epoch"] + 1, ck.get("best", -1.0)
        LOG.info(f"resumed at epoch {start_epoch}")
    loader = make_loader(cfg, train_ids, cfg["patch_size"], cfg["batch_size"], cfg["fg_oversample"],
                         cfg["seed"] + start_epoch, cfg["num_workers"])
    it = iter(loader)
    dt = amp_dtype(cfg["amp"], device)
    scaler = torch.amp.GradScaler("cuda", enabled=(dt == torch.float16))
    LOG.info(f"device {device}" + (f" ({torch.cuda.get_device_name()})" if device == "cuda" else "")
             + f", mixed precision: {dt}")
    log_path = out_dir / f"{tag}_log.csv"
    new_log = not log_path.exists() or not resume
    logf = open(log_path, "w" if new_log else "a", newline="")
    wr = csv.writer(logf)
    if new_log:
        wr.writerow(["epoch", "lr", "loss", "seg_loss", "vol_loss", "val_dice", "seconds"])
    (out_dir / "config_used.json").write_text(json.dumps(cfg, indent=1, default=str))
    model.train()
    for ep in range(start_epoch, epochs):
        t0 = time.time()
        cur_lr = lr * (1 - ep / epochs) ** 0.9
        for g in opt.param_groups:
            g["lr"] = cur_lr
        losses = []
        for _ in range(iters_per_epoch):
            x, y, b = next(it)
            x, y, b = x.to(device, non_blocking=True), y.to(device, non_blocking=True), b.to(device, non_blocking=True)
            x, y, b = augment(x, y, b, cfg["modality_dropout"])
            with torch.autocast("cuda", dtype=dt or torch.bfloat16, enabled=dt is not None):
                outs, vol = model(x, return_volume=True)
            seg_loss = deep_supervision_loss(outs, y)
            vol_loss = torch.nn.functional.smooth_l1_loss(vol.float(), patch_volume_target(y))
            loss = seg_loss + cfg["vol_loss_weight"] * vol_loss
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 12)
            scaler.step(opt)
            scaler.update()
            losses.append((loss.item(), seg_loss.item(), vol_loss.item()))
        val = float("nan")
        if val_ids and ((ep + 1) % cfg["val_every"] == 0 or ep + 1 == epochs):
            val = validate(cfg, model, val_ids, device)
            if val > best:
                best = val
                torch.save({"model": model.state_dict(), "epoch": ep, "val_dice": val, "preset": cfg["preset"]},
                           out_dir / "checkpoint_best.pt")
        m = np.mean(losses, 0)
        wr.writerow([ep, f"{cur_lr:.2e}", f"{m[0]:.4f}", f"{m[1]:.4f}", f"{m[2]:.4f}",
                     "" if np.isnan(val) else f"{val:.4f}", f"{time.time() - t0:.0f}"]); logf.flush()
        LOG.info(f"[{tag}] epoch {ep + 1}/{epochs} loss {m[0]:.4f} (seg {m[1]:.4f}, vol {m[2]:.4f})"
                 + ("" if np.isnan(val) else f" val Dice {val:.4f} (best {best:.4f})") + f" {time.time() - t0:.0f}s")
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "epoch": ep, "best": best,
                    "preset": cfg["preset"]}, last)
    torch.save({"model": model.state_dict(), "epoch": epochs - 1, "preset": cfg["preset"]}, out_dir / "checkpoint_final.pt")
    if not (out_dir / "checkpoint_best.pt").exists():
        torch.save({"model": model.state_dict(), "epoch": epochs - 1, "preset": cfg["preset"]},
                   out_dir / "checkpoint_best.pt")
    logf.close()
    return out_dir / "checkpoint_best.pt"
