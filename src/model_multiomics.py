"""
Methylation and immune summaries: do they add to the genomic model?

Fixed ridge Cox (alpha = 10), 5 folds x 5 repeats, corrected repeated t-test.
All feature sets within a cohort use the SAME patients (complete for every block
included in the run) and the same splits.

Blocks:  BASE (cleaned legacy baseline) | GEN = Desmedt + Hallmark + drivers (77)
         METH = methylation features with <= 1% missing among methylation-profiled
                tumours (fixed list, 14 of 20; the rule uses predictors only and was
                applied once before CV)
         IMM  = MCP-counter marker-based z-score summaries (10; mean of marker-gene
                z-scores, NOT the original MCP-counter abundance estimator)

Two testing families per cohort (Holm within each):
  vs BASE:      BASE+GEN, BASE+METH, BASE+IMM
  vs BASE+GEN:  BASE+GEN+METH, BASE+GEN+IMM, BASE+GEN+METH+IMM
Options: NO_METH=1 (immune only, full genomic cohort), NO_IMM=1 (methylation only).
"""
import os, warnings
import numpy as np, pandas as pd
from model_compare_shap import load, baselines, corrected_repeated_t, holm
from model_er_late import cox_cv

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2")


def main():
    co, DM, HM, DRV = load()           # complete legacy baseline + genomic, n = 1,295
    full = pd.read_csv(os.path.join(ROOT, "data", "processed", "metabric_recurrence_multiomics.csv"), low_memory=False)
    extra = [c for c in full.columns if c.startswith(("METH_", "IMM_"))]
    co = co.drop(columns=[c for c in extra if c in co.columns]).merge(
        full[["Patient ID"] + extra], on="Patient ID", how="left")
    prof = full[full["METH_mean"].notna()]
    METH = [c for c in extra if c.startswith("METH_") and prof[c].isna().mean() <= 0.01]
    IMM = [c for c in extra if c.startswith("IMM_")]
    if os.environ.get("NO_METH") == "1": METH = []
    if os.environ.get("NO_IMM") == "1": IMM = []
    GEN = DM + HM + DRV
    co = co.dropna(subset=METH + IMM).reset_index(drop=True)
    rows, tests = [], []
    for label, df in {"All patients": co, "ER+ only": co[co.er_binary == 1].reset_index(drop=True)}.items():
        B = baselines(df)["BASE"]
        sets = {"BASE": B, "BASE+GEN": B + GEN}
        fam1, fam2 = ["BASE+GEN"], []
        if METH:
            sets["BASE+METH"] = B + METH; sets["BASE+GEN+METH"] = B + GEN + METH
            fam1.append("BASE+METH"); fam2.append("BASE+GEN+METH")
        if IMM:
            sets["BASE+IMM"] = B + IMM; sets["BASE+GEN+IMM"] = B + GEN + IMM
            fam1.append("BASE+IMM"); fam2.append("BASE+GEN+IMM")
        if METH and IMM:
            sets["BASE+GEN+METH+IMM"] = B + GEN + METH + IMM; fam2.append("BASE+GEN+METH+IMM")
        res = cox_cv(df, sets)
        for k, v in res.items():
            rows.append({"analysis": label, "feature_set": k, "n_features": len(sets[k]), "n": len(df),
                         "events": int(df.y.sum()), "cindex_mean": v.mean(), "cindex_sd": v.std(ddof=1)})
        for ref, fam in [("BASE", fam1), ("BASE+GEN", fam2)]:
            tr = []
            for k in fam:
                m, lo, hi, p = corrected_repeated_t(res[k] - res[ref])
                tr.append({"analysis": label, "comparison": f"{k} vs {ref}", "delta": m, "ci_low": lo,
                           "ci_high": hi, "p_raw": p, "splits_improved": int((res[k] > res[ref]).sum())})
            for r, a in zip(tr, holm([r["p_raw"] for r in tr])): r["p_holm"] = a
            tests += tr
    tag = "_".join([x for x, on in [("meth", bool(METH)), ("imm", bool(IMM))] if on])
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"multiomics_{tag}_summary.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(OUT, f"multiomics_{tag}_tests.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(tests).round(4).to_string())


if __name__ == "__main__":
    main()
