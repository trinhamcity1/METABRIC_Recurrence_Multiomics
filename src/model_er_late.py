"""
Fixed-penalty ridge Cox (alpha = 10), 5 folds x 5 repeats (25 splits):
all patients, ER+ patients, and a five-year landmark analysis.

Landmark: patients recurrence-free and still under follow-up at 60 months;
time is measured from month 60 and the event is any recorded relapse
(the endpoint does not distinguish local from distant relapse).

Baselines are cleaned within each analysis cohort (model_compare_shap.clean_cols):
  BASE   legacy EGETC baseline without constant / duplicated columns
  CLIN   BASE without the two mutation-count summaries (clinical + treatment only)
  LEGACY the original 22-column baseline, kept only to compare the cleaned and
         legacy specifications on identical splits (descriptive, not tested).

Testing family per analysis (Holm): BASE+DM, BASE+HM, BASE+DRIVERS, BASE+GEN,
BASE+CORE4 vs BASE, and CLIN+GEN vs CLIN. CORE4 (AURKA proliferation, ESR1
signalling, STAT1 immune, copy-number fraction) was defined post hoc after
SHAP results and is exploratory.
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from model_compare_shap import load, LEGACY, baselines, corrected_repeated_t, holm

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2")
SEED = 42
CORE4 = ["DM_proliferation_AURKA", "DM_ER_signaling_ESR1", "DM_immune_response_STAT1", "CNA_frac_genes_altered"]


def cox_cv(df, sets):
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    res = {k: [] for k in sets}
    for tr, te in rskf.split(np.zeros(len(df)), df.y):
        for name, cols in sets.items():
            X = df[cols].astype(float).values
            sc = StandardScaler().fit(X[tr])
            m = CoxPHSurvivalAnalysis(alpha=10.0, n_iter=200).fit(
                sc.transform(X[tr]), Surv.from_arrays(df.y.values[tr].astype(bool), df.t.values[tr]))
            r = m.predict(sc.transform(X[te]))
            res[name].append(concordance_index_censored(df.y.values[te].astype(bool), df.t.values[te], r)[0])
    return {k: np.array(v) for k, v in res.items()}


def main():
    co, DM, HM, DRV = load()
    GEN = DM + HM + DRV
    er = co[co.er_binary == 1].reset_index(drop=True)
    late = co[co.t > 60].copy(); late["t"] = late.t - 60
    late_er = late[late.er_binary == 1].reset_index(drop=True)
    analyses = {"All patients": co, "ER+ only": er, "Late (>5y), all": late.reset_index(drop=True),
                "Late (>5y), ER+": late_er}
    family = [("BASE+DM", "BASE"), ("BASE+HM", "BASE"), ("BASE+DRIVERS", "BASE"), ("BASE+GEN", "BASE"),
              ("BASE+CORE4", "BASE"), ("CLIN+GEN", "CLIN")]
    rows, tests = [], []
    for label, df in analyses.items():
        b = baselines(df); B, C = b["BASE"], b["CLIN"]
        sets = {"LEGACY": LEGACY, "BASE": B, "BASE+DM": B + DM, "BASE+HM": B + HM, "BASE+DRIVERS": B + DRV,
                "BASE+GEN": B + GEN, "BASE+CORE4": B + CORE4, "CLIN": C, "CLIN+GEN": C + GEN}
        res = cox_cv(df, sets)
        for k, v in res.items():
            rows.append({"analysis": label, "feature_set": k, "n_features": len(sets[k]), "n": len(df),
                         "events": int(df.y.sum()), "cindex_mean": v.mean(), "cindex_sd": v.std(ddof=1)})
        trows = []
        for k, base in family:
            m, lo, hi, p = corrected_repeated_t(res[k] - res[base])
            trows.append({"analysis": label, "comparison": f"{k} vs {base}", "delta": m, "ci_low": lo,
                          "ci_high": hi, "p_raw": p, "splits_improved": int((res[k] > res[base]).sum())})
        for r, a in zip(trows, holm([r["p_raw"] for r in trows])): r["p_holm"] = a
        m, lo, hi, p = corrected_repeated_t(res["BASE"] - res["LEGACY"])
        trows.append({"analysis": label, "comparison": "BASE vs LEGACY (descriptive)", "delta": m, "ci_low": lo,
                      "ci_high": hi, "p_raw": p, "splits_improved": int((res["BASE"] > res["LEGACY"]).sum()),
                      "p_holm": np.nan})
        tests += trows
        print(label, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "er_late_summary.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(OUT, "er_late_tests.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(tests).round(4).to_string())


if __name__ == "__main__":
    main()
