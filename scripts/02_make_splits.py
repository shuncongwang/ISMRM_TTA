#!/usr/bin/env python3
"""
Step 2 - build the experiment splits (only cases that were preprocessed successfully).

    Exp 1  cohort : train on source_cohorts (one or several), adapt/test on every other (unseen) cohort
    Exp 2/3 run only if the table has grade / IDH columns (grade_col / idh_col in config.yaml)
    Exp 2  grade  : train on grade 4 -> test grade 3, grade 2   (within each cohort in within_cohorts, + pooled)
    Exp 3  IDH    : train on IDH-wt  -> test IDH-mut G4, IDH-mut G2-3  (within each cohort, + pooled)
    --reverse adds LGG -> G4 and IDH-mut -> IDH-wt

The tune_cohort is put aside (role "tune") and used only to choose TTA hyper-parameters with
the Exp 1 model; it never appears as a test domain.  Each target domain gets n_finetune labelled
cases (role target_finetune) for the supervised fine-tuning upper bound; the rest is target_test.

    python scripts/02_make_splits.py --config config.yaml --reverse

Output: <work_dir>/splits/*.csv  and  <work_dir>/splits/splits_overview.xlsx
Columns: case_id, cohort, grade, IDH, cell, domain, role
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from tta.common import case_files, load_config, read_table

MISSING = {"", "na", "n/a", "nan", "none", "unknown", "onbekend", "nvt", "-", "?", "not available"}
ROMAN = {"i": 1, "ii": 2, "iii": 3, "iv": 4}
MIN_TEST = 30


def norm_grade(v):
    t = str(v).strip().lower()
    if t in MISSING:
        return np.nan
    m = re.search(r"\b(iv|iii|ii|i)\b", t)
    if m:
        return float(ROMAN[m.group(1)])
    m = re.search(r"([1-4])", t)
    return float(m.group(1)) if m else np.nan


def norm_idh(v):
    t = str(v).strip().lower()
    if t in MISSING:
        return np.nan
    if re.search(r"(^|[^a-z])(wt|wildtype|wild-type|wild type|negative|neg|no|0|0\.0|false|intact|"
                 r"not mutated|non-mutated|not mutant)([^a-z]|$)", t):
        return "IDH-wt"
    if re.search(r"(^|[^a-z0-9])(mut|mutant|mutated|positive|pos|r132|r172|yes|1|1\.0|true)", t):
        return "IDH-mut"
    return np.nan


def split_source(d, rng, strata):
    roles = pd.Series(index=d.index, dtype=object)
    key = d[strata].astype(str).agg("|".join, axis=1)
    for _, idx in key.groupby(key).groups.items():
        idx = rng.permutation(np.array(idx))
        n = len(idx)
        n_test, n_val = int(round(0.2 * n)), int(round(0.1 * n))
        roles.loc[idx[:n_test]] = "source_test"
        roles.loc[idx[n_test:n_test + n_val]] = "source_val"
        roles.loc[idx[n_test + n_val:]] = "source_train"
    return roles


def split_target(d, rng, n_ft, strata):
    roles = pd.Series("target_test", index=d.index, dtype=object)
    if n_ft == 0 or len(d) < n_ft + MIN_TEST // 2:
        return roles
    key = d[strata].astype(str).agg("|".join, axis=1)
    picked = []
    for k, p in key.value_counts(normalize=True).items():
        pool = key.index[key == k]
        picked += list(rng.choice(pool, min(len(pool), int(round(p * n_ft))), replace=False))
    if len(picked) < n_ft:
        picked += list(rng.choice(d.index.difference(picked), n_ft - len(picked), replace=False))
    roles.loc[picked[:n_ft]] = "target_finetune"
    return roles


def main():
    def extra(ap):
        ap.add_argument("--reverse", action="store_true")
    cfg, a = load_config(extra_args=extra)
    rng = np.random.default_rng(cfg["seed"])
    df = read_table(cfg["merged_xlsx"], cfg["merged_sheet"])
    d = pd.DataFrame({"case_id": df[cfg["id_col"]].astype(str).str.strip(),
                      "cohort": df[cfg["cohort_col"]].astype(str).str.strip(),
                      "grade_num": df[cfg["grade_col"]].map(norm_grade) if cfg.get("grade_col") in df else np.nan,
                      "IDH": df[cfg["idh_col"]].map(norm_idh) if cfg.get("idh_col") in df else np.nan})
    has_clinical = d.grade_num.notna().any() or d.IDH.notna().any()
    if not has_clinical:
        print("No grade / IDH columns found -> only the cohort experiment is built")
    d = d[~d.case_id.duplicated()]
    d["cached"] = d.case_id.map(lambda c: case_files(cfg, c)["meta"].exists())
    print(f"{len(d)} cases in table, {int(d.cached.sum())} preprocessed")
    d = d[d.cached & ~d.cohort.isin(cfg["exclude_cohorts"])].copy()
    d["grade"] = d.grade_num.map(lambda g: f"G{int(g)}" if pd.notna(g) else "G?")
    d["IDH"] = d.IDH.fillna("IDH?")
    d["cell"] = d.IDH + " " + d.grade

    tune = d[d.cohort == cfg["tune_cohort"]]
    if tune.empty:
        print(f"WARNING: tune_cohort '{cfg['tune_cohort']}' has no cases; set tune_cohort in config.yaml")
    d = d[d.cohort != cfg["tune_cohort"]]

    out = Path(cfg["work_dir"]) / "splits"
    out.mkdir(parents=True, exist_ok=True)
    cols = ["case_id", "cohort", "grade", "IDH", "cell", "domain", "role"]
    cells, written = [], []

    def save(name, src, targets, strata, include_tune=False):
        if len(src) < 20:
            print(f"SKIP {name}: only {len(src)} source cases")
            return
        parts = []
        s = src.copy(); s["domain"] = "source"; s["role"] = split_source(s, rng, strata); parts.append(s)
        for dom, t in targets.items():
            if len(t) == 0:
                continue
            t = t.copy(); t["domain"] = dom; t["role"] = split_target(t, rng, cfg["n_finetune"], strata)
            parts.append(t)
        if include_tune and len(tune):
            t = tune.copy(); t["domain"] = f"TUNE:{cfg['tune_cohort']}"; t["role"] = "tune"; parts.append(t)
        p = pd.concat(parts)[cols]
        assert not p.case_id.duplicated().any()
        p.to_csv(out / f"{name}.csv", index=False)
        written.append(name)
        for (dom, role), n in p.groupby(["domain", "role"]).size().items():
            cells.append({"experiment": name, "domain": dom, "role": role, "n": int(n),
                          "warning": "TOO SMALL (<30)" if role == "target_test" and n < MIN_TEST else ""})

    # Exp 1 - cohort
    src = cfg.get("source_cohorts") or [cfg["source_cohort"]]
    src = [src] if isinstance(src, str) else list(src)
    missing_src = [s for s in src if s not in set(d.cohort)]
    if not missing_src:
        save(f"exp1_cohort_src-{'+'.join(src)}", d[d.cohort.isin(src)],
             {c: g for c, g in d[~d.cohort.isin(src)].groupby("cohort")}, ["cohort", "IDH", "grade"],
             include_tune=True)
    else:
        print(f"WARNING: source cohort(s) {missing_src} not found. Cohorts: {sorted(set(d.cohort))}")

    # Exp 2 / 3 - within cohort and pooled
    scopes = {c: d[d.cohort == c] for c in cfg["within_cohorts"] if c in set(d.cohort)} if has_clinical else {}
    if has_clinical:
        scopes["pooled"] = d
    for sc, x in scopes.items():
        g = x[x.grade_num.notna()]
        save(f"exp2_grade_{sc}_G4-to-LGG", g[g.grade_num == 4],
             {"G3": g[g.grade_num == 3], "G2": g[g.grade_num == 2]}, ["IDH"])
        i = x[x.IDH != "IDH?"]
        wt, mut = i[i.IDH == "IDH-wt"], i[i.IDH == "IDH-mut"]
        save(f"exp3_idh_{sc}_wt-to-mut", wt,
             {"IDH-mut G4": mut[mut.grade == "G4"], "IDH-mut G2-3": mut[mut.grade.isin(["G2", "G3"])]}, ["grade"])
        if a.reverse:
            save(f"exp2_grade_{sc}_LGG-to-G4", g[g.grade_num.isin([2, 3])], {"G4": g[g.grade_num == 4]}, ["IDH"])
            save(f"exp3_idh_{sc}_mut-to-wt", mut, {"IDH-wt": wt}, ["grade"])

    xt = pd.crosstab(pd.concat([d, tune]).cohort, pd.concat([d, tune]).cell, margins=True, margins_name="Total")
    cells_df = pd.DataFrame(cells)
    with pd.ExcelWriter(out / "splits_overview.xlsx") as w:
        xt.to_excel(w, sheet_name="Cohort_x_IDH_grade")
        cells_df.to_excel(w, sheet_name="Cells", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(xt)
        small = cells_df[cells_df.warning != ""]
        print("\nTarget test sets < 30 cases:\n" + (small.to_string(index=False) if len(small) else "(none)"))
    print(f"\nWrote {len(written)} split files to {out}")
    (out / "experiments.txt").write_text("\n".join(written) + "\n")


if __name__ == "__main__":
    main()
