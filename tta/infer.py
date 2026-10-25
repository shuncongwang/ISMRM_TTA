"""Sliding-window inference with Gaussian weighting."""
from __future__ import annotations

import itertools

import numpy as np
import torch

from .common import amp_dtype
from .data import pad_to


def gaussian_map(patch, sigma_scale=0.125):
    coords = [np.arange(p) - (p - 1) / 2 for p in patch]
    g = np.ones(patch, dtype=np.float32)
    for ax, (c, p) in enumerate(zip(coords, patch)):
        shape = [1, 1, 1]
        shape[ax] = p
        g = g * np.exp(-(c ** 2) / (2 * (p * sigma_scale) ** 2)).reshape(shape)
    g = g / g.max()
    g[g < 1e-3] = 1e-3
    return torch.from_numpy(g)


def window_starts(size, patch, overlap):
    if size <= patch:
        return [0]
    step = max(int(patch * (1 - overlap)), 1)
    starts = list(range(0, size - patch + 1, step))
    if starts[-1] != size - patch:
        starts.append(size - patch)
    return starts


@torch.no_grad()
def sliding_window(model, img: np.ndarray, patch, overlap=0.5, batch=2, mirror=False, device="cuda", amp=True):
    """img: (C,X,Y,Z) numpy -> foreground probability (X,Y,Z) float32 numpy."""
    model.eval()
    x_np, offs = pad_to(np.asarray(img, dtype=np.float32), patch)
    x = torch.from_numpy(x_np).to(device)
    shape = x.shape[1:]
    out = torch.zeros(shape, device=device)
    norm = torch.zeros(shape, device=device)
    g = gaussian_map(patch).to(device)
    starts = list(itertools.product(*[window_starts(s, p, overlap) for s, p in zip(shape, patch)]))
    dt = amp_dtype(amp, device)
    for i in range(0, len(starts), batch):
        chunk = starts[i:i + batch]
        xb = torch.stack([x[:, a:a + patch[0], b:b + patch[1], c:c + patch[2]] for a, b, c in chunk])
        with torch.autocast("cuda", dtype=dt or torch.bfloat16, enabled=dt is not None):
            logits = model(xb).float()
            if mirror:
                n = 1
                for axes in [(2,), (3,), (4,), (2, 3), (2, 4), (3, 4), (2, 3, 4)]:
                    logits = logits + model(xb.flip(axes)).float().flip(axes)
                    n += 1
                logits = logits / n
        probs = torch.sigmoid(logits)[:, 0]
        for (a, b, c), p in zip(chunk, probs):
            out[a:a + patch[0], b:b + patch[1], c:c + patch[2]] += p * g
            norm[a:a + patch[0], b:b + patch[1], c:c + patch[2]] += g
    prob = (out / norm).cpu().numpy()
    orig = img.shape[1:]
    return prob[offs[0]:offs[0] + orig[0], offs[1]:offs[1] + orig[1], offs[2]:offs[2] + orig[2]]
