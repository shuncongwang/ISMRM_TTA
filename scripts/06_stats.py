#!/usr/bin/env python3
"""
Step 6 - statistics with 95% confidence intervals over all per-case results.

    python scripts/06_stats.py --config config.yaml
    python scripts/06_stats.py --config config.yaml --proposed voltta --reference finetune10 --experiments exp1_cohort_src-UCSF-PDGM

Unit of analysis = patient. All comparisons are paired on the same cases.

Sheets in <work_dir>/stats/stats.xlsx
  Descriptive         per experiment x domain x method: mean Dice (BCa 95% CI), median [IQR], HD95, volume error
  Paired_vs_source    ΔDice = method - source: mean (BCa CI), median (percentile CI), Wilcoxon signed-rank,
                      Holm-adjusted p (within experiment), harm rate (Δ < -harm_threshold, Wilson CI),
                      ΔHD95 median (CI)
  Proposed_vs_others  proposed - each other TTA method: ΔDice (BCa CI), Wilcoxon + Holm, harm-rate McNemar
  Recovery_NI         recovery = (method - source)/(finetune - source) (bootstrap CI);
                      non-inferiority vs finetune: lower 95% bound of mean(method - finetune) > -ni_margin
  Pooled              macro-average ΔDice over target domains with a two-level bootstrap
                      (resample domains, then patients within domains)
  Subgroups_cell      ΔDice per IDH x grade cell (within domain and pooled over target domains)
  MixedModel          logit(Dice) ~ method * (IDH + grade) + log(GT volume) + cohort, random intercept per patient
  Volume_agreement    Bland-Altman bias and 95% limits of agreement, ICC(A,1) with bootstrap CI
Figures: <work_dir>/stats/forest_<experiment>.png
"""
import glob as globmod
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from tta.common import load_config

warnings.filterwarnings("ignore", category=RuntimeWarning)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]


# ----------------------------------------------------------------------------- helpers
def boot_ci(x, stat=np.mean, n=10000, method="BCa", seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 3 or np.allclose(x, x[0]):
        v = stat(x) if len(x) else np.nan
        return v, v, v
    try:
        r = stats.bootstrap((x,), stat, n_resamples=n, method=method, random_state=seed, vectorized=False)
        return stat(x), r.confidence_interval.low, r.confidence_interval.high
    except Exception:  # noqa: BLE001  (BCa can fail on degenerate data)
        r = stats.bootstrap((x,), stat, n_resamples=n, method="percentile", random_state=seed, vectorized=False)
        return stat(x), r.confidence_interval.low, r.confidence_interval.high


def wilson(k, n, z=1.959964):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n
    den = 1 + z ** 2 / n
    c = (p + z ** 2 / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / den
    return p, max(0, c - h), min(1, c + h)


def wilcoxon_p(d):
    d = np.asarray(d, float)
    d = d[~np.isnan(d)]
    if len(d) < 5 or np.allclose(d, 0):
        return np.nan
    return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)


def holm(p):
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() == 0:
        return out
    pv = p[ok]
    order = np.argsort(pv)
    m = len(pv)
    adj = np.maximum.accumulate((m - np.arange(m)) * pv[order])
    res = np.empty(m)
    res[order] = np.minimum(adj, 1)
    out[ok] = res
    return out


def mcnemar_exact(a_bad, b_bad):
    a_bad, b_bad = np.asarray(a_bad, bool), np.asarray(b_bad, bool)
    n01, n10 = int((~a_bad & b_bad).sum()), int((a_bad & ~b_bad).sum())
    if n01 + n10 == 0:
        return np.nan
    return float(stats.binomtest(n10, n01 + n10, 0.5).pvalue)


def icc_a1(x, y):
    """ICC(A,1): two-way, absolute agreement, single measurement (McGraw & Wong)."""
    data = np.column_stack([x, y]).astype(float)
    data = data[~np.isnan(data).any(1)]
    n, k = data.shape
    if n < 3:
        return np.nan
    gm = data.mean()
    msr = k * ((data.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((data.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((data - data.mean(1, keepdims=True) - data.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n)


def fmt_ci(v, lo, hi, d=3):
    return f"{v:.{d}f} ({lo:.{d}f} to {hi:.{d}f})" if not np.isnan(v) else ""


def paired_frame(df, keys, m1, m2, metric="dice"):
    a = df[df.method == m1].set_index(keys + ["case_id"])[metric]
    b = df[df.method == m2].set_index(keys + ["case_id"])[metric]
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner").reset_index()
    j["d"] = j["a"] - j["b"]
    return j


# ----------------------------------------------------------------------------- main
def main():
    def extra(ap):
        ap.add_argument("--results", default=None, help="glob of result CSVs (default <work_dir>/results/*/*.csv)")
        ap.add_argument("--baseline", default="source")
        ap.add_argument("--proposed", default="voltta")
        ap.add_argument("--reference", default="finetune10")
        ap.add_argument("--experiments", nargs="*", default=None)
        ap.add_argument("--out_name", default="stats")
        ap.add_argument("--exclude_domains", nargs="*", default=[], help="drop these target domains from evaluation, e.g. IMAGO (outputs get the suffix _excl-<names>)")
    cfg, a = load_config(extra_args=extra)
    wd = Path(cfg["work_dir"])
    files = [Path(f) for f in sorted(globmod.glob(a.results))] if a.results else sorted((wd / "results").glob("*/*.csv"))
    files = [f for f in files if not f.name.startswith("tune")]
    if not files:
        sys.exit("no result CSVs found")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df[df.role != "tune"].drop_duplicates(["experiment", "case_id", "method"], keep="last")
    if a.experiments:
        df = df[df.experiment.isin(a.experiments)]
    sfx = ""
    if a.exclude_domains:
        n0 = len(df)
        df = df[~df.domain.isin(a.exclude_domains)]
        sfx = "_excl-" + "-".join(a.exclude_domains)
        if a.out_name == "stats":
            a.out_name = "stats" + sfx
        print(f"excluded domains {a.exclude_domains}: {n0 - len(df)} rows dropped")
    B, P, R = a.baseline, a.proposed, a.reference
    NB, thr, margin = cfg["n_boot"], cfg["harm_threshold"], cfg["ni_margin"]
    print(f"{len(files)} files, {len(df)} rows, experiments: {sorted(df.experiment.unique())}")
    print(f"methods: {sorted(df.method.unique())}")

    # A. descriptive
    rows = []
    for (e, dom, role, m), g in df.groupby(["experiment", "domain", "role", "method"]):
        mu, lo, hi = boot_ci(g.dice, n=NB)
        rows.append({"experiment": e, "domain": dom, "role": role, "method": m, "n": len(g),
                     "dice_mean": mu, "dice_ci_low": lo, "dice_ci_high": hi, "dice_mean_CI": fmt_ci(mu, lo, hi),
                     "dice_median": g.dice.median(), "dice_q1": g.dice.quantile(.25), "dice_q3": g.dice.quantile(.75),
                     "hd95_median": g.hd95.median(), "vol_abs_err_median_ml": g.vol_abs_err_ml.median()})
    desc = pd.DataFrame(rows)

    # B. paired vs baseline
    rows = []
    for e, ge in df.groupby("experiment"):
        for m in sorted(set(ge.method) - {B}):
            j = paired_frame(ge, ["domain", "role"], m, B)
            jh = paired_frame(ge, ["domain", "role"], m, B, "hd95")
            for (dom, role), g in j.groupby(["domain", "role"]):
                mu, lo, hi = boot_ci(g.d, n=NB)
                md, mlo, mhi = boot_ci(g.d, np.median, n=NB, method="percentile")
                h = jh[(jh.domain == dom) & (jh.role == role)].d
                hmd, hlo, hhi = boot_ci(h, np.median, n=NB, method="percentile")
                harm, hl, hh = wilson(int((g.d < -thr).sum()), len(g))
                rows.append({"experiment": e, "domain": dom, "role": role, "method": m, "n": len(g),
                             "dDice_mean": mu, "ci_low": lo, "ci_high": hi, "dDice_mean_CI": fmt_ci(mu, lo, hi),
                             "dDice_median_CI": fmt_ci(md, mlo, mhi), "wilcoxon_p": wilcoxon_p(g.d),
                             "harm_rate": harm, "harm_CI": fmt_ci(harm, hl, hh), "improved_rate": float((g.d > thr).mean()),
                             "dHD95_median_CI": fmt_ci(hmd, hlo, hhi, 2)})
    paired = pd.DataFrame(rows)
    if len(paired):
        paired["p_holm"] = paired.groupby("experiment").wilcoxon_p.transform(lambda s: holm(s.values))

    # C. proposed vs other TTA methods
    rows = []
    others_all = set(df.method) - {B, P, R}
    for e, ge in df[df.role == "target_test"].groupby("experiment"):
        if P not in set(ge.method):
            continue
        for o in sorted(others_all & set(ge.method)):
            j = paired_frame(ge, ["domain"], P, o)
            for dom, g in list(j.groupby("domain")) + [("ALL target domains (pooled cases)", j)]:
                mu, lo, hi = boot_ci(g.d, n=NB)
                rows.append({"experiment": e, "domain": dom, "comparison": f"{P} - {o}", "n": len(g),
                             "dDice_mean": mu, "dDice_mean_CI": fmt_ci(mu, lo, hi), "wilcoxon_p": wilcoxon_p(g.d)})
            # harm-rate McNemar vs baseline: is proposed harmed less often than o?
            jb_p = paired_frame(ge, ["domain"], P, B)
            jb_o = paired_frame(ge, ["domain"], o, B)
            hh = jb_p.merge(jb_o, on=["domain", "case_id"], suffixes=("_p", "_o"))
            if len(hh):
                rows.append({"experiment": e, "domain": "ALL target domains (pooled cases)",
                             "comparison": f"harm rate {P} vs {o} (McNemar)", "n": len(hh),
                             "dDice_mean": np.nan,
                             "dDice_mean_CI": f"{(hh.d_p < -thr).mean():.3f} vs {(hh.d_o < -thr).mean():.3f}",
                             "wilcoxon_p": mcnemar_exact(hh.d_p < -thr, hh.d_o < -thr)})
    prop = pd.DataFrame(rows)
    if len(prop):
        prop["p_holm"] = prop.groupby("experiment").wilcoxon_p.transform(lambda s: holm(s.values))

    # D. recovery and non-inferiority vs reference (fine-tuning)
    rows = []
    for (e, dom), g in df[df.role == "target_test"].groupby(["experiment", "domain"]):
        if R not in set(g.method) or B not in set(g.method):
            continue
        w = g.pivot_table(index="case_id", columns="method", values="dice")
        for m in sorted(set(w.columns) - {B, R}):
            x = w[[m, B, R]].dropna()
            if len(x) < 5:
                continue
            arr = x.values

            def rec(idx):
                s = arr[idx]
                den = s[:, 2].mean() - s[:, 1].mean()
                return (s[:, 0].mean() - s[:, 1].mean()) / den if abs(den) > 1e-6 else np.nan
            rng = np.random.default_rng(0)
            bs = np.array([rec(rng.integers(0, len(arr), len(arr))) for _ in range(min(NB, 5000))])
            ni_mu, ni_lo, ni_hi = boot_ci(x[m] - x[R], n=NB)
            rows.append({"experiment": e, "domain": dom, "method": m, "n": len(x),
                         "recovery": rec(np.arange(len(arr))),
                         "recovery_CI": f"{np.nanpercentile(bs, 2.5):.2f} to {np.nanpercentile(bs, 97.5):.2f}",
                         f"mean(method - {R})": ni_mu, "NI_CI": fmt_ci(ni_mu, ni_lo, ni_hi),
                         "non_inferior": bool(ni_lo > -margin), "margin": margin})
    reco = pd.DataFrame(rows)

    # E. pooled two-level bootstrap (macro average over target domains)
    rows = []
    rng = np.random.default_rng(1)
    for e, ge in df[df.role == "target_test"].groupby("experiment"):
        for m in sorted(set(ge.method) - {B}):
            j = paired_frame(ge, ["domain"], m, B)
            groups = [g.d.values for _, g in j.groupby("domain") if len(g) >= 3]
            if len(groups) < 2:
                continue
            est = np.mean([g.mean() for g in groups])
            bs = []
            for _ in range(min(NB, 5000)):
                pick = rng.integers(0, len(groups), len(groups))
                bs.append(np.mean([groups[i][rng.integers(0, len(groups[i]), len(groups[i]))].mean() for i in pick]))
            rows.append({"experiment": e, "method": m, "n_domains": len(groups), "n_cases": int(sum(map(len, groups))),
                         "macro_dDice": est, "CI": fmt_ci(est, np.percentile(bs, 2.5), np.percentile(bs, 97.5)),
                         "domains_improved": int(sum(g.mean() > 0 for g in groups))})
    pooled = pd.DataFrame(rows)

    # F. subgroups by IDH x grade cell
    rows = []
    for e, ge in df[df.role == "target_test"].groupby("experiment"):
        for m in sorted(set(ge.method) - {B}):
            j = paired_frame(ge, ["domain", "cell"], m, B)
            for (dom, cell), g in list(j.groupby(["domain", "cell"])) + \
                    [(("ALL target", c), g) for c, g in j.groupby("cell")]:
                if len(g) < 5:
                    continue
                mu, lo, hi = boot_ci(g.d, n=NB)
                rows.append({"experiment": e, "domain": dom, "cell": cell, "method": m, "n": len(g),
                             "dDice_mean_CI": fmt_ci(mu, lo, hi), "wilcoxon_p": wilcoxon_p(g.d)})
    subg = pd.DataFrame(rows)

    # G. mixed model
    mm_rows = []
    try:
        import statsmodels.formula.api as smf
        for e, ge in df[df.role == "target_test"].groupby("experiment"):
            d = ge[ge.method.isin([B, P])].copy()
            if d.method.nunique() < 2:
                continue
            d["y"] = np.log(np.clip(d.dice, 0.001, 0.999) / (1 - np.clip(d.dice, 0.001, 0.999)))
            d["logvol"] = np.log1p(d.vol_gt_ml)
            d["method"] = pd.Categorical(d.method, categories=[B, P])
            terms = ["C(method)"]
            for col in ["IDH", "grade"]:
                d = d[~d[col].astype(str).str.contains(r"\?")]
                if d[col].nunique() > 1:
                    terms.append(f"C(method):C({col})")
                    terms.append(f"C({col})")
            terms.append("logvol")
            if d.cohort.nunique() > 1:
                terms.append("C(cohort)")
            formula = "y ~ " + " + ".join(terms)
            fit = smf.mixedlm(formula, d, groups=d["case_id"]).fit(reml=True)
            ci = fit.conf_int()
            for name in fit.params.index:
                if name == "Group Var":
                    continue
                mm_rows.append({"experiment": e, "formula": formula, "term": name, "coef_logit": fit.params[name],
                                "ci_low": ci.loc[name, 0], "ci_high": ci.loc[name, 1], "p": fit.pvalues[name],
                                "n_obs": int(fit.nobs)})
    except Exception as ex:  # noqa: BLE001
        mm_rows.append({"experiment": "ERROR", "term": str(ex)})
    mixed = pd.DataFrame(mm_rows)

    # H. volume agreement
    rows = []
    for (e, m), g in df[df.role == "target_test"].groupby(["experiment", "method"]):
        diff = (g.vol_pred_ml - g.vol_gt_ml).values
        bias, sd = diff.mean(), diff.std(ddof=1)
        icc = icc_a1(g.vol_pred_ml.values, g.vol_gt_ml.values)
        rng = np.random.default_rng(2)
        arr = g[["vol_pred_ml", "vol_gt_ml"]].values
        bs = [icc_a1(*arr[rng.integers(0, len(arr), len(arr))].T) for _ in range(1000)]
        rows.append({"experiment": e, "method": m, "n": len(g), "bias_ml": bias,
                     "LoA_low_ml": bias - 1.96 * sd, "LoA_high_ml": bias + 1.96 * sd,
                     "ICC_A1": icc, "ICC_CI": f"{np.nanpercentile(bs, 2.5):.3f} to {np.nanpercentile(bs, 97.5):.3f}"})
    vol = pd.DataFrame(rows)

    out = wd / "stats"
    out.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out / f"{a.out_name}.xlsx") as w:
        for name, t in [("Descriptive", desc), ("Paired_vs_source", paired), ("Proposed_vs_others", prop),
                        ("Recovery_NI", reco), ("Pooled", pooled), ("Subgroups_cell", subg),
                        ("MixedModel", mixed), ("Volume_agreement", vol)]:
            (t if len(t) else pd.DataFrame({"note": ["not enough data / methods for this analysis"]})) \
                .to_excel(w, sheet_name=name, index=False)

    # forest plots
    for e, g in paired[paired.role == "target_test"].groupby("experiment") if len(paired) else []:
        methods = [m for m in sorted(g.method.unique()) if not m.endswith("_nogate")]
        doms = sorted(g.domain.unique())
        fig, ax = plt.subplots(figsize=(8.5, 0.45 * len(doms) * max(1, len(methods)) / 2 + 1.5))
        for k, m in enumerate(methods):
            gm = g[g.method == m].set_index("domain").reindex(doms)
            ypos = np.arange(len(doms)) + (k - (len(methods) - 1) / 2) * 0.8 / max(len(methods), 1)
            ax.errorbar(gm.dDice_mean, ypos, xerr=[gm.dDice_mean - gm.ci_low, gm.ci_high - gm.dDice_mean],
                        fmt="o", ms=5, color=PALETTE[k % len(PALETTE)], label=m, capsize=0, lw=1.5)
        ax.axvline(0, color="#52514e", lw=1)
        ax.set_yticks(range(len(doms)))
        ax.set_yticklabels(doms, fontsize=8)
        ax.set_xlabel(f"ΔDice vs {B} (mean, 95% CI)")
        ax.set_title(e, loc="left", fontsize=10)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
        fig.tight_layout()
        fig.savefig(out / f"forest_{e}{sfx}.png", dpi=150)
        plt.close(fig)

    with pd.option_context("display.width", 250, "display.max_columns", 20, "display.max_colwidth", 40):
        if len(paired):
            print("\nΔDice vs source (target_test):")
            print(paired[paired.role == "target_test"][["experiment", "domain", "method", "n", "dDice_mean_CI",
                                                        "p_holm", "harm_CI"]].to_string(index=False))
        if len(pooled):
            print("\nPooled (two-level bootstrap):")
            print(pooled.to_string(index=False))
    print(f"\nWrote {out / (a.out_name + '.xlsx')} and forest plots")


if __name__ == "__main__":
    main()
