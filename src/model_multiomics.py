"""
Phase 2, Step 3d: does adding METHYLATION and IMMUNE cell features help?

Same fold-safe design as before: penalised Cox (time to recurrence, C-index),
5 folds x 5 repeats, corrected repeated t-test, Holm across feature sets.
All feature sets are compared on the SAME patients (complete for every block
that is available), for all patients and for ER+ patients.

Blocks:  CLIN (22) | GEN = Desmedt + Hallmark + drivers (77) | METH (20) | IMM (MCP-counter, if built)
"""
import os, warnings
import numpy as np, pandas as pd
from model_compare_shap import load, CLIN, corrected_repeated_t, holm
from model_er_late import cox_cv

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2")


def main():
    co, DM, HM, DRV = load()           # complete CLIN + genomic, n = 1,295
    full = pd.read_csv(os.path.join(ROOT, "data", "processed", "metabric_recurrence_multiomics.csv"))
    extra = [c for c in full.columns if c.startswith(("METH_", "IMM_"))]
    co = co.drop(columns=[c for c in extra if c in co.columns]).merge(
        full[["Patient ID"] + extra], on="Patient ID", how="left")
    # keep methylation features with <=1% missing among methylation-profiled patients
    # (single-gene promoters such as BRCA1/ESR1 are missing in up to 19% and would halve the cohort)
    prof = full[full["METH_mean"].notna()]
    METH = [c for c in extra if c.startswith("METH_") and prof[c].isna().mean() <= 0.01]
    if os.environ.get("NO_METH") == "1":
        METH = []
    IMM = [c for c in extra if c.startswith("IMM_")]
    GEN = DM + HM + DRV
    blocks = METH + IMM
    co = co.dropna(subset=blocks).reset_index(drop=True)
    sets = {"CLIN": CLIN, "CLIN+GEN": CLIN + GEN}
    if METH:
        sets["CLIN+METH"] = CLIN + METH
    if IMM:
        sets["CLIN+IMM"] = CLIN + IMM
        sets["CLIN+GEN+IMM"] = CLIN + GEN + IMM
    if METH:
        sets["CLIN+GEN+METH" + ("+IMM" if IMM else "")] = CLIN + GEN + METH + IMM
    rows, tests = [], []
    for label, df in {"All patients": co, "ER+ only": co[co.er_binary == 1].reset_index(drop=True)}.items():
        res = cox_cv(df, sets)
        for k, v in res.items():
            rows.append({"analysis": label, "feature_set": k, "n_features": len(sets[k]), "n": len(df),
                         "events": int(df.y.sum()), "cindex_mean": v.mean(), "cindex_sd": v.std(ddof=1)})
        tr = []
        for k in list(sets)[1:]:
            m, lo, hi, p = corrected_repeated_t(res[k] - res["CLIN"])
            tr.append({"analysis": label, "feature_set": k, "delta": m, "ci_low": lo, "ci_high": hi, "p_raw": p,
                       "splits_improved": int((res[k] > res["CLIN"]).sum())})
        for r, a in zip(tr, holm([r["p_raw"] for r in tr])): r["p_holm"] = a
        tests += tr
    tag = ("meth_" if METH else "") + ("imm" if IMM else "")
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"multiomics_{tag}_summary.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(OUT, f"multiomics_{tag}_tests.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(tests).round(4).to_string())


if __name__ == "__main__":
    main()
