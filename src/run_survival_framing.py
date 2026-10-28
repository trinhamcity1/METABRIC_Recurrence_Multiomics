"""
METABRIC genomic extension — Phase 1b: does the ENDPOINT definition hide the
molecular signal?

The EGETC paper (and run_recurrence_molecular.py) uses a binary "recurred yes/no"
label with no follow-up time. Biology says molecular subgroups matter most for
WHEN a tumour recurs (ER- early, ER+ IntClust 1/2/6/9 late). This script tests
three domain-motivated reframings with the same CLIN vs CLIN+MOL comparison:

  A. Time-to-event: penalised Cox model on relapse-free months, 5-fold CV
     Harrell C-index (handles censoring properly).
  B. ER-positive patients only, time-to-event (where IntClust risk groups
     were defined).
  C. Late distant-type recurrence: among patients still recurrence-free and
     followed at 5 years, did they recur afterwards? (logistic regression AUC)

All comparisons use the same patients and seeded folds; corrected paired t-test.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from run_recurrence_molecular import load, corrected_t, holm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results", "pilot"))
os.makedirs(OUT, exist_ok=True)
SEED = 42


def drop_constant(X):
    keep = X.std(axis=0) > 0
    return X.loc[:, keep]


def cox_cv(df, cols, time_col, event_col, folds):
    cs = []
    for tr, te in folds:
        tr_df, te_df = df.iloc[tr], df.iloc[te]
        X_tr = drop_constant(tr_df[cols].astype(float))
        use = list(X_tr.columns)
        mu, sd = X_tr.mean(), X_tr.std().replace(0, 1)
        Z_tr = (X_tr - mu) / sd
        Z_te = (te_df[use].astype(float) - mu) / sd
        cph = CoxPHFitter(penalizer=0.1, l1_ratio=0.0)
        cph.fit(pd.concat([Z_tr, tr_df[[time_col, event_col]]], axis=1), time_col, event_col)
        risk = cph.predict_partial_hazard(Z_te).values.ravel()
        cs.append(concordance_index(te_df[time_col], -risk, te_df[event_col]))
    return np.array(cs)


def lr_cv(X, y, folds):
    aucs = []
    for tr, te in folds:
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000))
        m.fit(X[tr], y[tr]); aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    return np.array(aucs)


def main():
    d, clin, mol, _ = load()
    d["rfs_months"] = d["Relapse Free Status (Months)"]
    d["rfs_event"] = d["y"]
    base = d.dropna(subset=["rfs_months", "rfs_event"] + clin + mol)
    base = base[base.rfs_months > 0].reset_index(drop=True)
    sets = {"CLIN": clin, "CLIN+MOL": clin + mol}
    rows, tests = [], []

    def run_cox(label, df):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
        folds = list(skf.split(np.zeros(len(df)), df.rfs_event.astype(int)))
        res = {fs: cox_cv(df, cols, "rfs_months", "rfs_event", folds) for fs, cols in sets.items()}
        for fs, v in res.items():
            rows.append({"analysis": label, "feature_set": fs, "metric": "C-index",
                         "mean": v.mean(), "sd": v.std(ddof=1), "n": len(df), "events": int(df.rfs_event.sum())})
        diff, ci, p = corrected_t(res["CLIN+MOL"] - res["CLIN"])
        tests.append({"analysis": label, "delta": diff, "ci_low": ci[0], "ci_high": ci[1], "p_raw": p,
                      "folds_improved": int((res["CLIN+MOL"] > res["CLIN"]).sum())})

    run_cox("A. All patients, Cox (time-to-recurrence)", base)
    er_pos = base[base.er_binary == 1].reset_index(drop=True)
    run_cox("B. ER+ only, Cox (time-to-recurrence)", er_pos)

    # C. late recurrence: recurrence-free AND followed beyond 60 months
    late = base[(base.rfs_months > 60)].copy()
    late["late_y"] = late.rfs_event.astype(int)  # recurred after month 60
    late = late.reset_index(drop=True)
    y = late.late_y.values
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    res = {}
    for fs, cols in sets.items():
        Xdf = late[cols].astype(float)
        res[fs] = lr_cv(Xdf.loc[:, Xdf.std() > 0].values, y, folds)
        rows.append({"analysis": "C. Late recurrence (>5y), LR", "feature_set": fs, "metric": "AUC",
                     "mean": res[fs].mean(), "sd": res[fs].std(ddof=1), "n": len(late), "events": int(y.sum())})
    diff, ci, p = corrected_t(res["CLIN+MOL"] - res["CLIN"])
    tests.append({"analysis": "C. Late recurrence (>5y), LR", "delta": diff, "ci_low": ci[0],
                  "ci_high": ci[1], "p_raw": p, "folds_improved": int((res["CLIN+MOL"] > res["CLIN"]).sum())})

    tests = pd.DataFrame(tests); tests["p_holm"] = holm(tests.p_raw.values)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "survival_framing_summary.csv"), index=False)
    tests.to_csv(os.path.join(OUT, "survival_framing_tests.csv"), index=False)

    # descriptive: among ER+ patients recurrence-free at 5y, later-recurrence rate by IntClust
    er_late = late[late.er_binary == 1]
    ic_cols = [c for c in mol if c.startswith("intclust_")]
    desc = []
    for c in ic_cols:
        sub = er_late[er_late[c] == 1]
        if len(sub) >= 15:
            desc.append({"intclust": c.replace("intclust_", ""), "n_ER+_at_risk_5y": len(sub),
                         "recurred_after_5y_pct": round(100 * sub.late_y.mean(), 1)})
    pd.DataFrame(desc).sort_values("recurred_after_5y_pct", ascending=False).to_csv(
        os.path.join(OUT, "late_recurrence_by_intclust_ERpos.csv"), index=False)
    print(pd.DataFrame(desc).sort_values("recurred_after_5y_pct", ascending=False).to_string())
    print(pd.DataFrame(rows).round(3).to_string()); print(tests.round(4).to_string())


if __name__ == "__main__":
    main()
