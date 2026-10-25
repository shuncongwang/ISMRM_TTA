"""Patch sampling from the preprocessed cache + GPU augmentation."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from .common import load_case


def pad_to(arr: np.ndarray, size, value=0):
    """Pad the last 3 dims of arr to at least `size` (centred). Returns arr, pad offsets."""
    shape = arr.shape[-3:]
    pads, offs = [], []
    for s, p in zip(shape, size):
        tot = max(p - s, 0)
        pads.append((tot // 2, tot - tot // 2))
        offs.append(tot // 2)
    if any(sum(p) for p in pads):
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 3) + pads, constant_values=value)
    return arr, offs


def sample_coords(mask, max_n=20000, step=1, seed=0):
    """Subsampled voxel coordinates of a mask (computed once per case and cached)."""
    idx = np.argwhere(np.asarray(mask[::step, ::step, ::step]) > 0) * step
    if len(idx) > max_n:
        idx = idx[np.random.default_rng(seed).choice(len(idx), max_n, replace=False)]
    return idx


def random_patch(img, seg, brain, patch, rng, fg_prob, fg_coords=None, brain_coords=None):
    """Crop one patch; with probability fg_prob centre it on a random tumour voxel."""
    shape = np.array(img.shape[1:])
    patch = np.array(patch)
    centre = None
    if rng.random() < fg_prob and seg is not None:
        fg = fg_coords if fg_coords is not None else sample_coords(seg)
        centre = fg[rng.integers(len(fg))] if len(fg) else None
    if centre is None:
        bz = brain_coords if brain_coords is not None else sample_coords(brain, step=4)
        centre = bz[rng.integers(len(bz))] if len(bz) else shape // 2
    return crop_at(img, seg, brain, centre, patch)


def crop_at(img, seg, brain, centre, patch):
    shape = np.array(img.shape[1:])
    patch = np.array(patch)
    lo = np.clip(np.asarray(centre) - patch // 2, 0, np.maximum(shape - patch, 0))
    sl = tuple(slice(int(a), int(a + p)) for a, p in zip(lo, patch))
    x = np.asarray(img[(slice(None),) + sl], dtype=np.float32)
    y = np.asarray(seg[sl], dtype=np.float32)[None] if seg is not None else None
    b = np.asarray(brain[sl], dtype=np.float32)[None]
    x, _ = pad_to(x, patch)
    b, _ = pad_to(b, patch)
    if y is not None:
        y, _ = pad_to(y, patch)
    return x, y, b


class PatchStream(IterableDataset):
    """Endless stream of (image, seg, brain) patches from a list of cached cases."""

    def __init__(self, cfg, case_ids, patch, fg_prob, seed=0):
        self.cfg, self.case_ids, self.patch, self.fg_prob, self.seed = cfg, list(case_ids), patch, fg_prob, seed

    def __iter__(self):
        wi = get_worker_info()
        rng = np.random.default_rng(self.seed + (wi.id if wi else 0) + np.random.randint(1 << 30))
        cache = {}
        while True:
            cid = self.case_ids[rng.integers(len(self.case_ids))]
            if cid not in cache:
                if len(cache) > 64:
                    cache.pop(next(iter(cache)))
                c = load_case(self.cfg, cid)
                c["fg"] = sample_coords(c["seg"])
                c["bz"] = sample_coords(c["brain"], step=4)
                cache[cid] = c
            c = cache[cid]
            x, y, b = random_patch(c["img"], c["seg"], c["brain"], self.patch, rng, self.fg_prob, c["fg"], c["bz"])
            yield torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(b)


@torch.no_grad()
def augment(x, y, b, modality_dropout=0.1, gen=None):
    """Batch-wise GPU augmentation: flips, intensity scale/shift, gamma, noise, modality dropout."""
    B, C = x.shape[:2]
    dev = x.device

    def rand(*s):
        return torch.rand(*s, device=dev, generator=gen)

    for ax in (2, 3, 4):
        flip = rand(B) < 0.5
        if flip.any():
            x[flip] = x[flip].flip(ax)
            y[flip] = y[flip].flip(ax)
            b[flip] = b[flip].flip(ax)
    scale = 0.9 + 0.2 * rand(B, C, 1, 1, 1)
    shift = 0.1 * (rand(B, C, 1, 1, 1) - 0.5)
    x = x * scale + shift
    do_gamma = rand(B, C, 1, 1, 1) < 0.3
    gamma = 0.7 + 0.8 * rand(B, C, 1, 1, 1)
    mn = x.amin(dim=(2, 3, 4), keepdim=True)
    mx = x.amax(dim=(2, 3, 4), keepdim=True)
    xn = ((x - mn) / (mx - mn + 1e-6)).clamp(0, 1) ** gamma * (mx - mn) + mn
    x = torch.where(do_gamma, xn, x)
    noise = rand(B, 1, 1, 1, 1) < 0.15
    x = x + noise * 0.1 * torch.randn(x.shape, device=dev, generator=gen)
    if modality_dropout > 0:
        drop = (rand(B, C) < modality_dropout / C * 1.0)
        drop[drop.all(1)] = False
        x = x * (~drop)[..., None, None, None]
    x = x * (b > 0)
    return x, y, b


def make_loader(cfg, case_ids, patch, batch, fg_prob, seed, workers):
    ds = PatchStream(cfg, case_ids, patch, fg_prob, seed)
    return torch.utils.data.DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=torch.cuda.is_available(),
                                       persistent_workers=workers > 0, prefetch_factor=4 if workers > 0 else None)


def downsample_like(t, ref):
    return F.interpolate(t, size=ref.shape[2:], mode="nearest") if t.shape[2:] != ref.shape[2:] else t
