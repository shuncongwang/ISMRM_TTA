#!/usr/bin/env python3
"""
Step 1 - preprocess every usable case into a fast cache (<work_dir>/cache).

Per case: load T1, T1GD, T2, FLAIR + mask -> (optional) reorient to RAS -> (optional) resample
-> crop to the non-zero brain region (+margin) -> z-score each channel inside the brain
(background set to 0) -> binarise the mask to whole tumour.
Saves  <id>_img.npy (4,X,Y,Z float16), <id>_seg.npy (uint8), <id>_brain.npy (uint8), <id>_meta.json
(the meta file holds what is needed to write predictions back as NIfTI).

    python scripts/01_preprocess.py --config config.yaml            # all cases (skips done ones)
    python scripts/01_preprocess.py --config config.yaml --overwrite true

Only cases with complete_4 == True and a mask path are processed. Failures are listed in
<work_dir>/reports/preprocess_failures.csv.
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

from tta.common import cache_dir, case_files, load_config, read_table, truthy


def load_nii(path, canonical):
    im = nib.load(str(path))
    if canonical:
        im = nib.as_closest_canonical(im)
    return im


def resample(arr, zooms, target, order):
    factors = [z / t for z, t in zip(zooms, target)]
    if np.allclose(factors, 1, atol=1e-3):
        return arr
    return ndimage.zoom(arr, factors, order=order)


def process(job):
    cid, paths, seg_rel, cfg = job
    f = case_files(cfg, cid)
    if f["meta"].exists() and not cfg["overwrite"]:
        return cid, "skipped"
    try:
        root = Path(cfg["data_root"])
        ims = [load_nii(root / paths[m], cfg["reorient_canonical"]) for m in cfg["modalities"]]
        seg_im = load_nii(root / seg_rel, cfg["reorient_canonical"])
        shape = ims[0].shape[:3]
        for im in ims[1:] + [seg_im]:
            if im.shape[:3] != shape:
                raise ValueError(f"shape mismatch {[i.shape for i in ims]} vs seg {seg_im.shape}")
        zooms = tuple(float(z) for z in ims[0].header.get_zooms()[:3])
        data = [np.asarray(im.dataobj, dtype=np.float32).reshape(shape) for im in ims]
        seg = np.asarray(seg_im.dataobj).reshape(shape)
        wt = np.isin(np.round(seg), cfg["wt_labels"]) if cfg["wt_labels"] else (seg > 0.5)

        spacing = zooms
        if cfg["target_spacing"]:
            tgt = [float(t) for t in cfg["target_spacing"]]
            data = [resample(d, zooms, tgt, 3) for d in data]
            wt = resample(wt.astype(np.uint8), zooms, tgt, 0).astype(bool)
            spacing = tuple(tgt)

        img = np.stack(data)
        brain = np.any(img != 0, axis=0)
        if brain.sum() < 1000:  # images not skull-stripped / no zero background -> use everything
            brain = np.ones_like(brain)
        brain = ndimage.binary_fill_holes(brain)
        nz = np.argwhere(brain)
        m = cfg["crop_margin"]
        lo = np.maximum(nz.min(0) - m, 0)
        hi = np.minimum(nz.max(0) + 1 + m, brain.shape)
        sl = tuple(slice(a, b) for a, b in zip(lo, hi))
        img, brain_c, wt = img[(slice(None),) + sl], brain[sl], wt[sl]

        for c in range(img.shape[0]):
            v = img[c][brain_c]
            lo_p, hi_p = np.percentile(v, [0.5, 99.5])
            v = np.clip(v, lo_p, hi_p)
            mu, sd = float(v.mean()), float(v.std()) or 1.0
            img[c] = np.clip((img[c] - mu) / sd, (lo_p - mu) / sd, (hi_p - mu) / sd)
            img[c][~brain_c] = 0

        cache_dir(cfg).mkdir(parents=True, exist_ok=True)
        np.save(f["img"], img.astype(np.float16))
        np.save(f["seg"], wt.astype(np.uint8))
        np.save(f["brain"], brain_c.astype(np.uint8))
        meta = {"case_id": cid, "spacing": list(spacing), "orig_shape": list(shape),
                "processed_shape_full": list(brain.shape), "crop_lo": lo.tolist(), "crop_hi": hi.tolist(),
                "affine": ims[0].affine.tolist() if not cfg["target_spacing"] else None,
                "resampled": bool(cfg["target_spacing"]), "canonical": cfg["reorient_canonical"],
                "wt_voxels": int(wt.sum()), "wt_ml": float(wt.sum() * np.prod(spacing) / 1000)}
        f["meta"].write_text(json.dumps(meta))
        return cid, "ok"
    except Exception as e:  # noqa: BLE001
        return cid, f"FAILED: {e.__class__.__name__}: {e}"


def main():
    def extra(ap):
        ap.add_argument("--jobs", type=int, default=8)
        ap.add_argument("--overwrite", type=lambda s: s.lower() in {"1", "true", "yes"}, default=False)
        ap.add_argument("--max_per_cohort", type=int, default=0, help="only the first N cases per cohort (smoke tests)")
    cfg, a = load_config(extra_args=extra)
    cfg["overwrite"] = a.overwrite
    df = read_table(cfg["merged_xlsx"], cfg["merged_sheet"])
    ok = df[cfg["complete_col"]].map(truthy) if cfg["complete_col"] in df else pd.Series(True, index=df.index)
    has_seg = df[cfg["seg_col"]].notna() & (df[cfg["seg_col"]].astype(str).str.strip() != "")
    use = df[ok & has_seg]
    if a.max_per_cohort:
        use = pd.concat([g.head(a.max_per_cohort) for _, g in use.groupby(cfg["cohort_col"])])
    print(f"{len(df)} rows; {len(use)} with 4 modalities + mask -> preprocessing")
    jobs = [(str(r[cfg["id_col"]]).strip(), {m: r[m] for m in cfg["modalities"]}, r[cfg["seg_col"]], cfg)
            for _, r in use.iterrows()]
    results = []
    t0 = time.time()
    with ProcessPoolExecutor(a.jobs) as ex:
        for i, res in enumerate(ex.map(process, jobs, chunksize=4), 1):
            results.append(res)
            if i % 100 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)}")
    st = pd.DataFrame(results, columns=["case_id", "status"])
    fails = st[st.status.str.startswith("FAILED")]
    rep = Path(cfg["work_dir"]) / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    fails.to_csv(rep / "preprocess_failures.csv", index=False)
    print(st.status.str.split(":").str[0].value_counts().to_string())
    n_done = int((st.status == "ok").sum())
    if n_done:
        sec = time.time() - t0
        (rep / "preprocess_timing.json").write_text(json.dumps(
            {"cases": n_done, "jobs": a.jobs, "wall_s": sec, "s_per_case_per_job": sec * a.jobs / n_done}))
        print(f"{sec / 60:.1f} min for {n_done} cases with {a.jobs} processes")
    if len(fails):
        print(f"{len(fails)} failures -> {rep / 'preprocess_failures.csv'}; first: {fails.iloc[0].status}")


if __name__ == "__main__":
    main()
