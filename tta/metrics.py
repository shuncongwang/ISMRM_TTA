"""Segmentation metrics on binary masks."""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p, g = pred.astype(bool), gt.astype(bool)
    s = p.sum() + g.sum()
    return 1.0 if s == 0 else float(2 * (p & g).sum() / s)


def _surface(m):
    return m & ~ndimage.binary_erosion(m, structure=np.ones((3, 3, 3)), border_value=0)


def hd95(pred: np.ndarray, gt: np.ndarray, spacing) -> float:
    """95th-percentile symmetric surface distance in mm.
    Empty prediction with non-empty GT (or vice versa) -> image diagonal in mm (worst case, never dropped)."""
    p, g = pred.astype(bool), gt.astype(bool)
    if not p.any() and not g.any():
        return 0.0
    if not p.any() or not g.any():
        return float(np.sqrt(np.sum((np.array(p.shape) * np.array(spacing)) ** 2)))
    # crop to the joint bounding box (+2) for speed
    nz = np.argwhere(p | g)
    lo = np.maximum(nz.min(0) - 2, 0)
    hi = np.minimum(nz.max(0) + 3, p.shape)
    sl = tuple(slice(a, b) for a, b in zip(lo, hi))
    p, g = p[sl], g[sl]
    sp, sg = _surface(p), _surface(g)
    dt_g = ndimage.distance_transform_edt(~sg, sampling=spacing)
    dt_p = ndimage.distance_transform_edt(~sp, sampling=spacing)
    d = np.concatenate([dt_g[sp], dt_p[sg]])
    return float(np.percentile(d, 95))


def volume_ml(mask: np.ndarray, spacing) -> float:
    return float(mask.astype(bool).sum() * np.prod(spacing) / 1000)


def all_metrics(pred, gt, spacing) -> dict:
    vp, vg = volume_ml(pred, spacing), volume_ml(gt, spacing)
    return {"dice": dice(pred, gt), "hd95": hd95(pred, gt, spacing), "vol_pred_ml": vp, "vol_gt_ml": vg,
            "vol_abs_err_ml": abs(vp - vg), "vol_rel_err": abs(vp - vg) / vg if vg > 0 else np.nan}
