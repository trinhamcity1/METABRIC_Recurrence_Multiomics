"""
Phase 2, Step 3a: fold-safe comparison of feature sets + SHAP stability.

Question: which genomic features add to the clinical model for recurrence, and
are they consistently important?

Design (everything that learns from data happens inside the training fold):
  * Cohort: recorded recurrence outcome + complete clinical (22) + expression,
    CNA and mutation data.
  * Repeated stratified CV: 5 folds x 5 repeats (25 train/test splits).
  * Feature sets:
      CLIN            22 clinical features (EGETC paper)
      CLIN+DM         + 7 Desmedt breast-cancer module scores
      CLIN+HM         + 50 Hallmark scores
      CLIN+PC         + first 9 PCs of the 57 scores (PCA fit on training fold)
      CLIN+DRIVERS    + 11 copy-number + 9 mutation features
      CLIN+ALL        + all 77 genomic features
      CLIN+SHAPSEL    + top-10 genomic features chosen by SHAP INSIDE the
                        training fold (honest test of "SHAP-selected" features)
  * Models: L2 logistic regression and LightGBM (binary recurrence, AUC);
            penalised Cox (time to recurrence, Harrell C-index).
  * Statistics: corrected repeated k-fold t-test (Bouckaert & Frank 2004),
    Holm-adjusted, each set vs CLIN.
  * SHAP: TreeSHAP of LightGBM on CLIN+ALL in every test fold; we record mean
    |SHAP| per feature and how often each feature is in the top 15.

Run in chunks with SPLITS=0-4 etc. if your environment has a time limit; each
split is checkpointed. SUMMARIZE=1 builds the tables.
"""
import os, json, warnings, glob, time
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import shap
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2"); CK = os.path.join(OUT, "ck_compare")
os.makedirs(CK, exist_ok=True)
SEED = 42
N_PCS = 9
TOPK = 10
LGB_PARAMS = dict(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=30,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_lambda=1.0,
                  random_state=SEED, verbose=-1, n_jobs=4)

CLIN = ["Age at Diagnosis", "Neoplasm Histologic Grade", "Lymph nodes examined positive",
        "Mutation Count", "Nottingham prognostic index", "Tumor Size", "Tumor Stage",
        "er_binary", "pr_binary", "her2_pos", "chemo", "hormone_rx", "radio_rx",
        "mastectomy", "postmeno", "ln_pos_flag", "high_grade", "log_mutation",
        "tumor_size_cm", "npi_group", "age_group", "er_x_her2"]


def load():
    d = pd.read_csv(os.path.join(ROOT, "data", "processed", "metabric_recurrence_multiomics.csv"))
    d["y"] = d["Relapse Free Status"].map({"Recurred": 1, "Not Recurred": 0})
    d["t"] = d["Relapse Free Status (Months)"]
    def b(col, pos):
        s = d[col]; return np.where(s.isna(), np.nan, (s == pos).astype(float))
    d["er_binary"] = b("ER Status", "Positive"); d["pr_binary"] = b("PR Status", "Positive")
    d["her2_pos"] = b("HER2 Status", "Positive"); d["chemo"] = b("Chemotherapy", "Yes")
    d["hormone_rx"] = b("Hormone Therapy", "Yes"); d["radio_rx"] = b("Radio Therapy", "Yes")
    d["mastectomy"] = b("Type of Breast Surgery", "Mastectomy"); d["postmeno"] = b("Inferred Menopausal State", "Post")
    ln = d["Lymph nodes examined positive"]; d["ln_pos_flag"] = np.where(ln.isna(), np.nan, (ln > 0).astype(float))
    g = d["Neoplasm Histologic Grade"]; d["high_grade"] = np.where(g.isna(), np.nan, (g == 3).astype(float))
    d["log_mutation"] = np.log1p(d["Mutation Count"]); d["tumor_size_cm"] = d["Tumor Size"] / 10
    npi = d["Nottingham prognostic index"]
    d["npi_group"] = np.where(npi.isna(), np.nan, np.where(npi <= 3.4, 0, np.where(npi <= 5.4, 1, 2)))
    a = d["Age at Diagnosis"]; d["age_group"] = np.where(a.isna(), np.nan, np.where(a < 50, 0, np.where(a <= 65, 1, 2)))
    d["er_x_her2"] = d["er_binary"] * d["her2_pos"]
    DM = [c for c in d.columns if c.startswith("DM_")]
    HM = [c for c in d.columns if c.startswith("HM_")]
    DRV = [c for c in d.columns if c.startswith(("CNA_", "MUT_"))]
    co = d.dropna(subset=["y", "t"] + CLIN + DM + HM + DRV)
    co = co[co.t > 0].reset_index(drop=True)
    return co, DM, HM, DRV


def corrected_repeated_t(diffs, k=5, r=5, test_over_train=0.25):
    m, s2 = np.mean(diffs), np.var(diffs, ddof=1)
    se = np.sqrt((1 / (k * r) + test_over_train) * s2)
    t = m / se if se > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=k * r - 1)
    h = stats.t.ppf(0.975, k * r - 1) * se
    return m, m - h, m + h, p


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j]); adj[j] = min(run, 1)
    return adj


def build_matrix(co, tr, te, fs, DM, HM, DRV):
    """Return Xtr, Xte, column names for a feature set; PCA/SHAP selection fit on tr only."""
    G = DM + HM
    if fs == "CLIN": cols = CLIN
    elif fs == "CLIN+DM": cols = CLIN + DM
    elif fs == "CLIN+HM": cols = CLIN + HM
    elif fs == "CLIN+DRIVERS": cols = CLIN + DRV
    elif fs == "CLIN+ALL": cols = CLIN + G + DRV
    elif fs == "CLIN+PC":
        sc = StandardScaler().fit(co.loc[tr, G]); pca = PCA(N_PCS, random_state=SEED).fit(sc.transform(co.loc[tr, G]))
        ptr = pca.transform(sc.transform(co.loc[tr, G])); pte = pca.transform(sc.transform(co.loc[te, G]))
        names = [f"PC{i+1}" for i in range(N_PCS)]
        Xtr = np.hstack([co.loc[tr, CLIN].values, ptr]); Xte = np.hstack([co.loc[te, CLIN].values, pte])
        return Xtr, Xte, CLIN + names
    elif fs == "CLIN+SHAPSEL":
        allc = CLIN + G + DRV
        m = lgb.LGBMClassifier(**LGB_PARAMS).fit(co.loc[tr, allc], co.loc[tr, "y"])
        sv = shap.TreeExplainer(m).shap_values(co.loc[tr, allc])
        sv = sv[1] if isinstance(sv, list) else sv
        imp = pd.Series(np.abs(sv).mean(0), index=allc).drop(CLIN)
        sel = list(imp.sort_values(ascending=False).head(TOPK).index)
        cols = CLIN + sel
        return co.loc[tr, cols].values, co.loc[te, cols].values, cols
    return co.loc[tr, cols].values.astype(float), co.loc[te, cols].values.astype(float), cols


def run_split(co, DM, HM, DRV, s, tr, te):
    rec = {"split": s, "res": [], "shap": None, "selected": None}
    ytr, yte = co.loc[tr, "y"].values, co.loc[te, "y"].values
    for fs in ["CLIN", "CLIN+DM", "CLIN+HM", "CLIN+PC", "CLIN+DRIVERS", "CLIN+ALL", "CLIN+SHAPSEL"]:
        Xtr, Xte, cols = build_matrix(co, tr, te, fs, DM, HM, DRV)
        if fs == "CLIN+SHAPSEL": rec["selected"] = cols[len(CLIN):]
        lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(Xtr, ytr)
        gb = lgb.LGBMClassifier(**LGB_PARAMS).fit(Xtr, ytr)
        sc = StandardScaler().fit(Xtr)
        ytr_s = Surv.from_arrays(co.loc[tr, "y"].astype(bool), co.loc[tr, "t"])
        cox = CoxPHSurvivalAnalysis(alpha=10.0, n_iter=200).fit(sc.transform(Xtr), ytr_s)
        risk = cox.predict(sc.transform(Xte))
        cidx = concordance_index_censored(co.loc[te, "y"].astype(bool), co.loc[te, "t"], risk)[0]
        rec["res"].append({"split": s, "feature_set": fs, "n_features": len(cols),
                           "auc_lr": roc_auc_score(yte, lr.predict_proba(Xte)[:, 1]),
                           "auc_lgbm": roc_auc_score(yte, gb.predict_proba(Xte)[:, 1]),
                           "cindex_cox": cidx})
        if fs == "CLIN+ALL":
            sv = shap.TreeExplainer(gb).shap_values(Xte)
            sv = sv[1] if isinstance(sv, list) else sv
            rec["shap"] = dict(zip(cols, map(float, np.abs(sv).mean(0))))
    return rec


def summarize(co, DM, HM, DRV):
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CK, "split_*.json")))]
    res = pd.DataFrame([r for rec in recs for r in rec["res"]])
    res.to_csv(os.path.join(OUT, "compare_folds.csv"), index=False)
    summ = res.groupby("feature_set")[["n_features", "auc_lr", "auc_lgbm", "cindex_cox"]].agg(["mean", "std"])
    summ.columns = ["_".join(c) for c in summ.columns]
    order = ["CLIN", "CLIN+DM", "CLIN+HM", "CLIN+PC", "CLIN+DRIVERS", "CLIN+ALL", "CLIN+SHAPSEL"]
    summ = summ.reindex(order); summ.to_csv(os.path.join(OUT, "compare_summary.csv"))
    tests = []
    for metric in ["auc_lr", "auc_lgbm", "cindex_cox"]:
        base = res[res.feature_set == "CLIN"].sort_values("split")[metric].values
        rows = []
        for fs in order[1:]:
            v = res[res.feature_set == fs].sort_values("split")[metric].values
            m, lo, hi, p = corrected_repeated_t(v - base)
            rows.append({"metric": metric, "feature_set": fs, "delta": m, "ci_low": lo, "ci_high": hi,
                         "p_raw": p, "splits_improved": int((v > base).sum()), "n_splits": len(v)})
        adj = holm([r["p_raw"] for r in rows])
        for r, a in zip(rows, adj): r["p_holm"] = a
        tests += rows
    tests = pd.DataFrame(tests); tests.to_csv(os.path.join(OUT, "compare_tests.csv"), index=False)
    sh = pd.DataFrame([rec["shap"] for rec in recs])
    ranks = sh.rank(axis=1, ascending=False)
    shs = pd.DataFrame({"mean_abs_shap": sh.mean(), "sd": sh.std(), "pct_splits_top15": (ranks <= 15).mean() * 100,
                        "median_rank": ranks.median()}).sort_values("mean_abs_shap", ascending=False)
    shs["type"] = ["clinical" if f in CLIN else "expression" if f.startswith(("HM_", "DM_")) else
                   "copy number" if f.startswith("CNA_") else "mutation" for f in shs.index]
    shs.to_csv(os.path.join(OUT, "shap_stability_CLIN_ALL.csv"))
    sel = pd.Series([f for rec in recs for f in rec["selected"]]).value_counts()
    (sel / len(recs) * 100).rename("pct_splits_selected").to_csv(os.path.join(OUT, "shap_selected_frequency.csv"))
    print("cohort n =", len(co), "| events =", int(co.y.sum()), "| splits =", len(recs))
    print(summ.round(3).to_string()); print(tests.round(4).to_string())
    print(shs.head(20).round(4).to_string()); print((sel / len(recs) * 100).head(15).round(0).to_string())


def main():
    co, DM, HM, DRV = load()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)
    splits = list(rskf.split(np.zeros(len(co)), co.y))
    lo, hi = map(int, os.environ.get("SPLITS", f"0-{len(splits)-1}").split("-"))
    json.dump({"n": len(co), "events": int(co.y.sum()), "n_DM": len(DM), "n_HM": len(HM), "n_DRV": len(DRV)},
              open(os.path.join(OUT, "compare_cohort.json"), "w"))
    for s in range(lo, hi + 1):
        f = os.path.join(CK, f"split_{s:02d}.json")
        if os.path.exists(f): continue
        t0 = time.time(); tr, te = splits[s]
        rec = run_split(co, DM, HM, DRV, s, tr, te)
        json.dump(rec, open(f, "w"))
        print(f"split {s} done ({time.time()-t0:.0f}s)", flush=True)
    if os.environ.get("SUMMARIZE", "0") == "1":
        summarize(co, DM, HM, DRV)


if __name__ == "__main__":
    main()
