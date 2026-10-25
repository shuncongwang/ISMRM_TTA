#!/usr/bin/env python3
"""
Step 5 - run the source model and the test-time adaptation methods on target (and source-test)
cases, and write one row of metrics per case x method.

Evaluate (normal use):
    python scripts/05_run_tta.py --config config.yaml --split exp1_cohort_src-UCSF-PDGM \
        --methods source tent sar meanteacher voltta
    # parallel jobs: --shard 0/4, 1/4, 2/4, 3/4  (each writes its own CSV)

Choose hyper-parameters (ONCE, on the tune cohort, with the Exp 1 model):
    python scripts/05_run_tta.py --config config.yaml --split exp1_cohort_src-UCSF-PDGM --tune \
        --methods tent sar meanteacher voltta
    -> writes <work_dir>/tuned_hparams.json, used automatically by every later run

Fine-tuning upper bound (after 04_finetune.py):
    python scripts/05_run_tta.py --config config.yaml --split exp1_cohort_src-UCSF-PDGM --methods source \
        --domains EGD --ckpt <work_dir>/models/exp1_cohort_src-UCSF-PDGM/finetune_EGD/checkpoint_final.pt \
        --label finetune10

Ablations: --hp voltta.w_vol=0  --label voltta_noVol    (any hyper-parameter, method.key=value)

Output: <work_dir>/results/<split>/<label or 'tta'>[_shardI].csv  (resumes if the file exists)
For methods with a safety gate, two rows are written: <method> (gated) and <method>_nogate.
"""
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from tta.common import LOG, load_case, load_config, read_split, seed_everything, setup_logging
from tta.infer import sliding_window
from tta.methods import METHODS, sample_patches
from tta.metrics import all_metrics, dice
from tta.train_loop import device_of, load_model


def parse_hp_overrides(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        meth, key = k.split(".", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        out.setdefault(meth, {})[key] = v
    return out


def hparams(cfg, method, tuned, overrides):
    hp = dict(cfg["tta"].get(method, {}))
    hp.setdefault("steps", cfg["tta"]["steps"])
    hp.update(tuned.get(method, {}))
    hp.update(overrides.get(method, {}))
    return hp


def save_pred(cfg, meta, prob, path):
    full = np.zeros(meta["processed_shape_full"], np.uint8)
    lo, hi = meta["crop_lo"], meta["crop_hi"]
    full[lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]] = (prob > 0.5).astype(np.uint8)
    aff = np.array(meta["affine"]) if meta.get("affine") else np.diag(list(meta["spacing"]) + [1])
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(full, aff), str(path))


def main():
    def extra(ap):
        ap.add_argument("--split", required=True)
        ap.add_argument("--methods", nargs="+", default=["source", "tent", "sar", "meanteacher", "voltta"])
        ap.add_argument("--roles", nargs="+", default=["target_test", "source_test"])
        ap.add_argument("--domains", nargs="+", default=["all"])
        ap.add_argument("--ckpt", default=None)
        ap.add_argument("--model_tag", default="")
        ap.add_argument("--label", default=None, help="name for this run / method label (e.g. finetune10)")
        ap.add_argument("--tune", action="store_true")
        ap.add_argument("--tune_max_cases", type=int, default=40)
        ap.add_argument("--hp", nargs="*", default=[])
        ap.add_argument("--no_tuned", action="store_true")
        ap.add_argument("--max_cases_per_domain", type=int, default=0)
        ap.add_argument("--shard", default="0/1")
        ap.add_argument("--save_pred", action="store_true")
    cfg, a = load_config(extra_args=extra)
    wd = Path(cfg["work_dir"])
    model_dir = wd / "models" / (a.split + (f"_{a.model_tag}" if a.model_tag else ""))
    setup_logging(wd / "results" / a.split / "run_tta.log")
    seed_everything(cfg["seed"])
    device = device_of(cfg)

    sp = read_split(wd / "splits" / f"{a.split}.csv")
    if a.tune:
        cases = sp[sp.role == "tune"]
        if cases.empty:
            sys.exit("No 'tune' cases in this split - use the Exp 1 split, which holds the tune cohort.")
        cases = cases.sample(min(len(cases), a.tune_max_cases), random_state=cfg["seed"])
    else:
        cases = sp[sp.role.isin(a.roles)]
        if a.domains != ["all"]:
            cases = cases[cases.domain.isin(a.domains)]
        if a.max_cases_per_domain:
            cases = cases.groupby("domain", group_keys=False).head(a.max_cases_per_domain)
    i_sh, n_sh = map(int, a.shard.split("/"))
    cases = cases.sort_values("case_id").iloc[i_sh::n_sh]

    ckpt = Path(a.ckpt) if a.ckpt else model_dir / "checkpoint_best.pt"
    model, _ = load_model(cfg, ckpt, device)
    model.eval()
    src_model = copy.deepcopy(model).eval()
    for p in src_model.parameters():
        p.requires_grad_(False)
    src_state = {k: v.clone() for k, v in model.state_dict().items()}

    tuned_path = wd / "tuned_hparams.json"
    tuned = {} if (a.no_tuned or a.tune or not tuned_path.exists()) else json.loads(tuned_path.read_text())
    overrides = parse_hp_overrides(a.hp)
    adaptive = [m for m in a.methods if m != "source"]
    for m in adaptive:
        if m not in METHODS:
            sys.exit(f"unknown method {m}; choose from source {' '.join(METHODS)}")

    label = a.label or ("tune" if a.tune else "tta")
    out_csv = wd / "results" / a.split / (f"{label}" + (f"_shard{i_sh}" if n_sh > 1 else "") + ".csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out_csv.exists():
        prev = pd.read_csv(out_csv, dtype=str)
        done = set(zip(prev.case_id, prev.method))
    LOG.info(f"{a.split}: {len(cases)} cases, methods {a.methods}, ckpt {ckpt.name}, out {out_csv.name}"
             + (f" (resuming, {len(done)} rows done)" if done else ""))
    rng = np.random.default_rng(cfg["seed"])
    patch = cfg["patch_size"]                                   # inference window
    tpatch = cfg.get("tta_patch_size") or cfg["tta"].get("patch_size") or patch  # adaptation patches (smaller on 24 GB GPUs)
    n_patch = cfg["tta"]["patches_per_step"]

    def predict(net, img):
        return sliding_window(net, img, patch, cfg["sw_overlap"], cfg["sw_batch"], cfg["mirror"], device, cfg["amp"])

    rows_buffer = []

    def flush():
        if rows_buffer:
            pd.DataFrame(rows_buffer).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
            rows_buffer.clear()

    # --label renames the method when ONE method is run (fine-tuned model, ablations)
    if a.label and not a.tune and len(a.methods) > 1:
        sys.exit("--label renames a method: use it with exactly one method in --methods")

    def lab_of(m):
        return a.label if (a.label and not a.tune) else m

    for n_case, r in enumerate(cases.itertuples(index=False), 1):
        if all((r.case_id, lab_of(m)) in done for m in a.methods) and not a.tune:
            continue
        c = load_case(cfg, r.case_id, mmap=False)
        img, gt, brain, meta = c["img"].astype(np.float32), c["seg"] > 0, c["brain"], c["meta"]
        spacing = meta["spacing"]
        base = {"experiment": a.split, "domain": r.domain, "role": r.role, "case_id": r.case_id,
                "cohort": r.cohort, "grade": r.grade, "IDH": r.IDH, "cell": r.cell, "ckpt": str(ckpt)}

        model.load_state_dict(src_state)
        t0 = time.time()
        src_prob = predict(model, img)
        t_src = time.time() - t0
        src_metrics = all_metrics(src_prob > 0.5, gt, spacing)
        if "source" in a.methods:
            lab = lab_of("source")
            rows_buffer.append({**base, "method": lab, **src_metrics, "dice_source": src_metrics["dice"],
                                "time_s": round(t_src, 2)})
            if a.save_pred:
                save_pred(cfg, meta, src_prob, wd / "preds" / a.split / lab / f"{r.case_id}.nii.gz")

        fg = np.argwhere(src_prob > 0.5)
        fg = fg[rng.choice(len(fg), min(len(fg), 5000), replace=False)] if len(fg) else None
        eval_x, eval_b = sample_patches(img, brain, fg, tpatch, n_patch, np.random.default_rng(0), device)

        for m in adaptive:
            if (r.case_id, lab_of(m)) in done and not a.tune:
                continue
            lab = lab_of(m)
            hp = hparams(cfg, m, tuned, overrides)
            lrs = cfg["tta"]["tune_grid"]["lr"] if a.tune else [hp["lr"]]
            checkpoints = sorted(cfg["tta"]["tune_grid"]["steps"]) if a.tune else [int(hp["steps"])]
            for lr in lrs:
                hp_run = dict(hp, lr=lr)
                model.load_state_dict(src_state)
                meth = METHODS[m](hp_run, src_model, cfg)
                meth.setup(model)
                obj_before = meth.objective(model, eval_x, eval_b)
                t_adapt = 0.0
                for s in range(1, checkpoints[-1] + 1):
                    t0 = time.time()
                    x, b = sample_patches(img, brain, fg, tpatch, n_patch, rng, device)
                    meth.step(model, x, b)
                    t_adapt += time.time() - t0
                    if s not in checkpoints:
                        continue
                    t0 = time.time()
                    net = meth.predictor(model)
                    prob = predict(net, img)
                    t_pred = time.time() - t0
                    obj_after = meth.objective(model, eval_x, eval_b)
                    overlap = dice(prob > 0.5, src_prob > 0.5)
                    gate_on = bool(hp_run.get("gate", False))
                    accepted = (overlap >= cfg["tta"]["gate_min_overlap"]) and (obj_after <= obj_before)
                    common = {**base, "lr": lr, "steps": s, "dice_source": src_metrics["dice"],
                              "overlap_with_source": round(overlap, 4), "obj_before": obj_before,
                              "obj_after": obj_after, "time_adapt_s": round(t_adapt, 2),
                              "time_s": round(t_adapt + t_pred, 2)}
                    mt = all_metrics(prob > 0.5, gt, spacing)
                    if gate_on:
                        final = mt if accepted else src_metrics
                        rows_buffer.append({**common, "method": lab, **final, "gate_accepted": accepted})
                        rows_buffer.append({**common, "method": f"{lab}_nogate", **mt, "gate_accepted": np.nan})
                    else:
                        rows_buffer.append({**common, "method": lab, **mt, "gate_accepted": np.nan})
                    if a.save_pred and not a.tune:
                        save_pred(cfg, meta, prob if (accepted or not gate_on) else src_prob,
                                  wd / "preds" / a.split / lab / f"{r.case_id}.nii.gz")
        flush()
        if n_case % 10 == 0 or n_case == len(cases):
            LOG.info(f"  {n_case}/{len(cases)} cases")
    flush()

    if a.tune:
        res = pd.read_csv(out_csv)
        res = res[~res.method.str.endswith("_nogate")]
        summ = res.groupby(["method", "lr", "steps"]).dice.mean().reset_index()
        best = {}
        for m, g in summ.groupby("method"):
            g = g.sort_values(["dice", "steps"], ascending=[False, True])
            best[m] = {"lr": float(g.iloc[0].lr), "steps": int(g.iloc[0].steps)}
            LOG.info(f"tuned {m}: lr={best[m]['lr']:g} steps={best[m]['steps']} mean Dice {g.iloc[0].dice:.4f} "
                     f"(source {res.dice_source.mean():.4f})")
        old = json.loads(tuned_path.read_text()) if tuned_path.exists() else {}
        old.update(best)
        tuned_path.write_text(json.dumps(old, indent=1))
        summ.to_csv(out_csv.with_name("tune_summary.csv"), index=False)
        LOG.info(f"wrote {tuned_path}")


if __name__ == "__main__":
    main()
