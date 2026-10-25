#!/usr/bin/env python3
"""
Step 0 - check the images before anything else.

For every case in merged_xlsx it reads the NIfTI headers (and the segmentation) and reports:
missing files, shape / spacing / orientation mismatches between the modalities and the mask,
spacing and shape distributions per cohort, the label values in the masks, empty masks and
whole-tumour volumes.  Nothing is modified.

    python scripts/00_check_data.py --config config.yaml
    python scripts/00_check_data.py --config config.yaml --max_cases 50     # quick look

Output: <work_dir>/reports/data_check.xlsx  (+ a printed summary)
Send me the "Per_cohort" and "Problems" sheets.
"""
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import nibabel as nib
import numpy as np
import pandas as pd

from tta.common import load_config, read_table, truthy


def check_one(args):
    cid, cohort, paths, seg_path, root = args
    row = {"case_id": cid, "cohort": cohort, "problems": []}
    shapes, spacings, orients = {}, {}, {}
    for mod, rel in list(paths.items()) + [("SEG", seg_path)]:
        if not rel or str(rel) == "nan":
            row["problems"].append(f"{mod}: no path")
            continue
        p = Path(root) / rel
        if not p.exists():
            row["problems"].append(f"{mod}: file not found")
            continue
        try:
            im = nib.load(str(p))
            shapes[mod] = tuple(int(s) for s in im.shape[:3])
            spacings[mod] = tuple(round(float(z), 3) for z in im.header.get_zooms()[:3])
            orients[mod] = "".join(nib.aff2axcodes(im.affine))
            if len(im.shape) > 3 and im.shape[3] > 1:
                row["problems"].append(f"{mod}: 4D image {im.shape}")
        except Exception as e:  # noqa: BLE001
            row["problems"].append(f"{mod}: unreadable ({e.__class__.__name__})")
    if shapes:
        ref = next(iter(shapes))
        row["shape"], row["spacing"], row["orientation"] = shapes[ref], spacings[ref], orients[ref]
        if len(set(shapes.values())) > 1:
            row["problems"].append(f"shape mismatch {shapes}")
        if len(set(spacings.values())) > 1:
            row["problems"].append(f"spacing mismatch {spacings}")
        if len(set(orients.values())) > 1:
            row["problems"].append(f"orientation mismatch {orients}")
    if seg_path and str(seg_path) != "nan" and (Path(root) / seg_path).exists():
        try:
            seg = np.asanyarray(nib.load(str(Path(root) / seg_path)).dataobj)
            vals, counts = np.unique(seg, return_counts=True)
            row["seg_labels"] = ",".join(f"{v:g}" for v in vals)
            vox = float(np.prod(spacings.get("SEG", (1, 1, 1))))
            row["wt_volume_ml"] = round(float(counts[vals > 0].sum()) * vox / 1000, 2)
            if row["wt_volume_ml"] == 0:
                row["problems"].append("empty mask")
            if not np.allclose(vals, np.round(vals)):
                row["problems"].append("non-integer mask values")
        except Exception as e:  # noqa: BLE001
            row["problems"].append(f"SEG: unreadable ({e.__class__.__name__})")
    row["problems"] = "; ".join(row["problems"])
    return row


def main():
    def extra(ap):
        ap.add_argument("--max_cases", type=int, default=0)
        ap.add_argument("--jobs", type=int, default=8)
    cfg, a = load_config(extra_args=extra)
    df = read_table(cfg["merged_xlsx"], cfg["merged_sheet"])
    if a.max_cases:
        k = max(1, a.max_cases // df[cfg["cohort_col"]].nunique())
        df = pd.concat([g.head(k) for _, g in df.groupby(cfg["cohort_col"])])
    jobs = [(str(r[cfg["id_col"]]).strip(), str(r.get(cfg["cohort_col"], "")),
             {m: r.get(m) for m in cfg["modalities"]}, r.get(cfg["seg_col"]), cfg["data_root"])
            for _, r in df.iterrows()]
    print(f"Checking {len(jobs)} cases with {a.jobs} processes ...")
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(check_one, jobs, chunksize=8))
    res = pd.DataFrame(rows)

    def most_common(s):
        c = Counter(map(str, s.dropna()))
        return "; ".join(f"{k} (n={v})" for k, v in c.most_common(3))

    per = res.groupby("cohort").agg(
        n=("case_id", "size"),
        n_with_problems=("problems", lambda s: (s != "").sum()),
        shapes=("shape", most_common), spacings=("spacing", most_common),
        orientations=("orientation", most_common), seg_label_sets=("seg_labels", most_common),
        wt_ml_median=("wt_volume_ml", "median"), wt_ml_min=("wt_volume_ml", "min"),
        wt_ml_max=("wt_volume_ml", "max"))
    probs = res[res["problems"] != ""][["case_id", "cohort", "problems"]]
    kinds = Counter(p.split(":")[0].split(" (")[0] for s in probs["problems"] for p in s.split("; "))

    out = Path(cfg["work_dir"]) / "reports"
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / "data_check.xlsx") as w:
        per.reset_index().to_excel(w, sheet_name="Per_cohort", index=False)
        probs.to_excel(w, sheet_name="Problems", index=False)
        res.to_excel(w, sheet_name="All_cases", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 60):
        print(per.to_string())
    print(f"\nCases with problems: {len(probs)} / {len(res)}")
    for k, v in kinds.most_common(15):
        print(f"  {v:5d}  {k}")
    complete = res["problems"].eq("") if cfg["complete_col"] not in df else (
        res["problems"].eq("") & df[cfg["complete_col"]].map(truthy).values)
    print(f"Usable (4 modalities + mask, no problems): {int(complete.sum())}")
    print(f"Wrote {out / 'data_check.xlsx'}")


if __name__ == "__main__":
    main()
