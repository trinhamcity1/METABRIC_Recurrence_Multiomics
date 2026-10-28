"""
Phase 2, Step 3b: the headline question. Do genomic features help for
ER-positive patients and for LATE recurrence (after 5 years)?

Same fold-safe design as model_compare_shap.py (5x5 repeated CV, corrected
repeated t-test, Holm). Penalised Cox (C-index) for ER+ time-to-recurrence,
and for late recurrence a landmark analysis: patients recurrence-free and still
followed at 60 months, time measured from month 60.
"""
import os, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from model_compare_shap import load, CLIN, corrected_repeated_t, holm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2")
SEED = 42


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
    core = ["DM_proliferation_AURKA", "DM_ER_signaling_ESR1", "DM_immune_response_STAT1", "CNA_fraction_altered"]
    sets = {"CLIN": CLIN, "CLIN+DM": CLIN + DM, "CLIN+HM": CLIN + HM, "CLIN+DRIVERS": CLIN + DRV,
            "CLIN+ALL": CLIN + DM + HM + DRV, "CLIN+CORE4": CLIN + core}
    er = co[co.er_binary == 1].reset_index(drop=True)
    late = co[co.t > 60].copy(); late["t"] = late.t - 60
    late_er = late[late.er_binary == 1].reset_index(drop=True)
    analyses = {"All patients": co, "ER+ only": er, "Late (>5y), all": late.reset_index(drop=True),
                "Late (>5y), ER+": late_er}
    rows, tests = [], []
    for label, df in analyses.items():
        res = cox_cv(df, sets)
        for k, v in res.items():
            rows.append({"analysis": label, "feature_set": k, "n": len(df), "events": int(df.y.sum()),
                         "cindex_mean": v.mean(), "cindex_sd": v.std(ddof=1)})
        trows = []
        for k in list(sets)[1:]:
            m, lo, hi, p = corrected_repeated_t(res[k] - res["CLIN"])
            trows.append({"analysis": label, "feature_set": k, "delta": m, "ci_low": lo, "ci_high": hi,
                          "p_raw": p, "splits_improved": int((res[k] > res["CLIN"]).sum())})
        for r, a in zip(trows, holm([r["p_raw"] for r in trows])): r["p_holm"] = a
        tests += trows
        print(label, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "er_late_summary.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(OUT, "er_late_tests.csv"), index=False)
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(tests).round(4).to_string())


if __name__ == "__main__":
    main()
