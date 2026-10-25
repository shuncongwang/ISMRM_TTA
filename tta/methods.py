"""Test-time adaptation methods for binary whole-tumour segmentation.

All methods are episodic: the model is reset to the source weights before every case.
Interface (used by scripts/05_run_tta.py):
    m = METHODS[name](hp, src_model, cfg)
    m.setup(model)                 # choose parameters, build optimiser
    m.step(model, x, b)            # one update on a batch of patches (x: B,C,X,Y,Z ; b: brain mask)
    m.predictor(model)             # network used for the final prediction
    m.objective(model, x, b)       # label-free objective (no grad) used by the safety gate

Methods
    tent         entropy minimisation of instance-norm affine parameters (Wang et al., ICLR 2021)
    sar          reliable-entropy filtering + sharpness-aware update (Niu et al., ICLR 2023)
    meanteacher  EMA teacher with flip-averaged, confidence-filtered pseudo-labels (CoTTA-style, Wang et al., CVPR 2022)
    voltta       proposed: entropy + volume consistency with the source volume head + FLAIR boundary
                 alignment + out-of-brain prior, with a safety gate
"""
from __future__ import annotations

import copy
import math

import numpy as np
import torch
import torch.nn.functional as F

from .model import patch_volume_target, select_parameters

EPS = 1e-6


def bin_entropy(p):
    p = p.clamp(EPS, 1 - EPS)
    return -(p * p.log() + (1 - p) * (1 - p).log())


def masked_mean(v, m):
    return (v * m).sum() / (m.sum() + EPS)


def forward(model, x, cfg, volume=False):
    use_amp = cfg["amp"] and x.is_cuda and torch.cuda.get_device_capability()[0] >= 8
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
        out = model(x, return_volume=volume)
    if volume:
        return out[0].float(), out[1].float()
    return out.float()


class Base:
    default_params = "norm"

    def __init__(self, hp, src_model, cfg):
        self.hp, self.src, self.cfg = hp, src_model, cfg

    def setup(self, model):
        model.eval()  # instance norm has no running stats; eval = single full-res output
        self.params = select_parameters(model, self.hp.get("params", self.default_params))
        self.opt = torch.optim.Adam(self.params, lr=self.hp["lr"], betas=(0.9, 0.999))

    def loss(self, model, x, b):
        raise NotImplementedError

    def step(self, model, x, b):
        loss = self.loss(model, x, b)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 5.0)
        self.opt.step()
        return float(loss.detach())

    def predictor(self, model):
        return model

    @torch.no_grad()
    def objective(self, model, x, b):
        return float(self.loss(self.predictor(model), x, b))


class Tent(Base):
    def loss(self, model, x, b):
        p = torch.sigmoid(forward(model, x, self.cfg))
        return masked_mean(bin_entropy(p), b)


class SAR(Base):
    def setup(self, model):
        model.eval()
        self.params = select_parameters(model, self.hp.get("params", "norm"))
        self.opt = torch.optim.SGD(self.params, lr=self.hp["lr"], momentum=0.9)

    def loss(self, model, x, b):
        p = torch.sigmoid(forward(model, x, self.cfg))
        ent = bin_entropy(p)
        reliable = (ent < self.hp.get("e_margin", 0.4) * math.log(2)).float() * b
        return masked_mean(ent, reliable)

    def step(self, model, x, b):
        rho = self.hp.get("rho", 0.05)
        loss = self.loss(model, x, b)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = [p.grad for p in self.params]
        if all(g is None for g in grads):
            return float(loss.detach())
        gnorm = torch.norm(torch.stack([g.norm() for g in grads if g is not None])) + 1e-12
        eps = []
        with torch.no_grad():
            for p in self.params:
                e = rho * p.grad / gnorm if p.grad is not None else torch.zeros_like(p)
                p.add_(e)
                eps.append(e)
        self.opt.zero_grad(set_to_none=True)
        self.loss(model, x, b).backward()
        with torch.no_grad():
            for p, e in zip(self.params, eps):
                p.sub_(e)
        torch.nn.utils.clip_grad_norm_(self.params, 5.0)
        self.opt.step()
        return float(loss.detach())


class MeanTeacher(Base):
    default_params = "all"

    def setup(self, model):
        super().setup(model)
        self.teacher = copy.deepcopy(model).eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def pseudo(self, x):
        probs = torch.sigmoid(forward(self.teacher, x, self.cfg))
        n = 1
        for ax in (2, 3, 4):
            probs = probs + torch.sigmoid(forward(self.teacher, x.flip(ax), self.cfg)).flip(ax)
            n += 1
        return probs / n

    def loss(self, model, x, b):
        tp = self.pseudo(x)
        conf = self.hp.get("conf", 0.9)
        mask = (((tp > conf) | (tp < 1 - conf)).float() * b)
        target = (tp > 0.5).float()
        # photometric perturbation only (labels stay aligned)
        xs = x * (0.9 + 0.2 * torch.rand(x.shape[0], x.shape[1], 1, 1, 1, device=x.device)) \
            + 0.05 * torch.randn_like(x)
        logits = forward(model, xs * (b > 0), self.cfg)
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * mask).sum() / (mask.sum() + EPS)
        p = torch.sigmoid(logits) * mask
        dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + (target * mask).sum() + 1)
        return bce + dice

    def step(self, model, x, b):
        loss = super().step(model, x, b)
        a = self.hp.get("ema", 0.99)
        with torch.no_grad():
            for pt, ps in zip(self.teacher.parameters(), model.parameters()):
                pt.mul_(a).add_(ps.detach(), alpha=1 - a)
        return loss

    def predictor(self, model):
        return self.teacher


class VolTTA(Base):
    """Proposed method. Terms (weights in hp):
        w_ent  entropy of the foreground probability inside the brain
        w_vol  |log1p(predicted tumour volume in patch) - source volume-head estimate|  (smooth L1)
        w_bnd  boundary alignment: 1 - <|grad p|, |grad FLAIR|> / <|grad p|, 1>
        w_out  mean foreground probability outside the brain
    """

    def __init__(self, hp, src_model, cfg):
        super().__init__(hp, src_model, cfg)
        self.flair = cfg.get("flair_channel", 3)

    @staticmethod
    def grad_mag(v):
        gx = F.pad(v[:, :, 1:] - v[:, :, :-1], (0, 0, 0, 0, 0, 1))
        gy = F.pad(v[:, :, :, 1:] - v[:, :, :, :-1], (0, 0, 0, 1))
        gz = F.pad(v[:, :, :, :, 1:] - v[:, :, :, :, :-1], (0, 1))
        return torch.sqrt(gx ** 2 + gy ** 2 + gz ** 2 + EPS)

    def loss(self, model, x, b):
        hp = self.hp
        logits, _ = forward(model, x, self.cfg, volume=True)
        p = torch.sigmoid(logits)
        loss = hp.get("w_ent", 1.0) * masked_mean(bin_entropy(p), b)
        if hp.get("w_vol", 0) > 0:
            with torch.no_grad():
                _, anchor = forward(self.src, x, self.cfg, volume=True)
            vol = patch_volume_target(p * b)
            loss = loss + hp["w_vol"] * F.smooth_l1_loss(vol, anchor)
        if hp.get("w_bnd", 0) > 0:
            g = self.grad_mag(x[:, self.flair:self.flair + 1])
            scale = torch.quantile(g.flatten(1)[:, ::7], 0.95, dim=1).view(-1, 1, 1, 1, 1) + EPS
            g = (g / scale).clamp(0, 1)
            gp = self.grad_mag(p)
            loss = loss + hp["w_bnd"] * (1 - (gp * g * b).sum() / ((gp * b).sum() + EPS))
        if hp.get("w_out", 0) > 0:
            loss = loss + hp["w_out"] * masked_mean(p, 1 - b)
        return loss


METHODS = {"tent": Tent, "sar": SAR, "meanteacher": MeanTeacher, "voltta": VolTTA}


def sample_patches(img, brain, centres_fg, patch, n, rng, device):
    """n patches; half centred on (source-predicted) tumour voxels, half on random brain voxels."""
    from .data import crop_at, random_patch
    xs, bs = [], []
    for i in range(n):
        use_fg = (i % 2 == 0) and centres_fg is not None and len(centres_fg)
        if use_fg:
            x, _, b = crop_at(img, None, brain, centres_fg[rng.integers(len(centres_fg))], patch)
        else:
            x, _, b = random_patch(img, None, brain, patch, rng, fg_prob=0.0)
        xs.append(x); bs.append(b)
    return torch.from_numpy(np.stack(xs)).to(device), torch.from_numpy(np.stack(bs)).to(device)
