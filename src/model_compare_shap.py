"""
Feature-set comparison in all patients + SHAP stability (5 folds x 5 repeats, 25 splits).

Steps that learn from data (scaling, PCA, SHAP-based selection) are fitted on the
training fold only. Note: the expression input is the cBioPortal mRNA z-score file,
which was normalized by the data provider against a reference population of the
whole cohort BEFORE this pipeline; that upstream step is not repeated inside folds.

Feature sets (keys used in all result files):
  BASE           cleaned legacy EGETC baseline (clinical, treatment, mutation count;
                 constant / duplicated columns removed: 21 columns in all patients)
  BASE+DM        + 7 Desmedt module scores
  BASE+HM        + 50 Hallmark scores
  BASE+PC        + 9 PCs of the 57 scores (PCA fit on the training fold; 9 is fixed)
  BASE+DRIVERS   + 11 copy-number + 9 mutation features
  BASE+GEN       + all 77 genomic features (called CLIN+ALL in earlier versions)
  BASE+SHAPSEL   + top-10 genomic features chosen by SHAP inside the training fold
Models: ridge Cox (alpha = 10; time to recurrence, Harrell C-index) is the primary
endpoint analysis. L2 logistic regression and LightGBM classify the recorded
Recurred / Not Recurred status WITHOUT using follow-up or censoring; their AUCs are
descriptive classification of observed status, not risk discrimination at a horizon.
Statistics: corrected repeated k-fold t-test (Bouckaert & Frank 2004), Holm, vs BASE.
SHAP: TreeSHAP of the binary LightGBM classifier on BASE+GEN in every test fold
(log-odds scale); describes that classifier in all patients only.

Run in chunks with SPLITS=0-4 etc.; each split is checkpointed. SUMMARIZE=1 builds tables.
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

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2"); CK = os.path.join(OUT, "ck_compare_v2")
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
# CLIN above is the LEGACY baseline of the EGETC paper (22 columns). It is not purely
# clinical: it contains treatment indicators and two mutation-count summaries.
LEGACY = CLIN
MUTCOLS = ["Mutation Count", "log_mutation"]
CLIN_ONLY = [c for c in LEGACY if c not in MUTCOLS]   # clinical + treatment, no mutation count


def clean_cols(df, cols):
    """Drop columns that are constant in df, or exact duplicates of an earlier column
    after standardization (e.g. Tumor Size vs tumor_size_cm; in ER+ patients er_binary
    is constant and er_x_her2 equals her2_pos). Uses predictors only, never the outcome."""
    keep, seen = [], []
    for c in cols:
        x = df[c].astype(float).values
        sd = x.std()
        if sd == 0:
            continue
        z = (x - x.mean()) / sd
        if any(np.allclose(z, w) for w in seen):
            continue
        keep.append(c); seen.append(z)
    return keep


def baselines(df):
    """BASE = cleaned legacy baseline (clinical, treatment and mutation count);
    CLIN = cleaned strictly clinical/treatment comparator (no mutation-count summaries)."""
    return {"BASE": clean_cols(df, LEGACY), "CLIN": clean_cols(df, CLIN_ONLY)}


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
    """Corrected repeated k-fold t-test (Bouckaert & Frank 2004).

    diffs: paired per-split differences (same split identities for both models).
    Degenerate cases: all differences zero -> (0, 0, 0, p = 1); a constant
    non-zero difference has no variance estimate -> (d, d, d, p = nan) and is
    reported, not tested.
    """
    diffs = np.asarray(diffs, dtype=float)
    if diffs.size != k * r:
        raise ValueError(f"expected {k * r} paired differences, got {diffs.size}")
    if not np.all(np.isfinite(diffs)):
        raise ValueError("non-finite paired differences")
    m = float(np.mean(diffs))
    s2 = float(np.var(diffs, ddof=1))
    if s2 == 0.0:
        return (0.0, 0.0, 0.0, 1.0) if m == 0.0 else (m, m, m, float("nan"))
    se = np.sqrt((1 / (k * r) + test_over_train) * s2)
    t = m / se
    p = 2 * stats.t.sf(abs(t), df=k * r - 1)
    h = stats.t.ppf(0.975, k * r - 1) * se
    return m, m - h, m + h, p


def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, j in enumerate(o):
        run = max(run, (len(p) - i) * p[j]); adj[j] = min(run, 1)
    return adj


ORDER = ["BASE", "BASE+DM", "BASE+HM", "BASE+PC", "BASE+DRIVERS", "BASE+GEN", "BASE+SHAPSEL"]


def build_matrix(co, tr, te, fs, DM, HM, DRV):
    """Return Xtr, Xte, column names for a feature set; PCA/SHAP selection fit on tr only.
    Feature-set keys: BASE (cleaned legacy baseline), BASE+DM, BASE+HM, BASE+PC,
    BASE+DRIVERS, BASE+GEN (= all 77 genomic features), BASE+SHAPSEL."""
    G = DM + HM
    CLIN = baselines(co)["BASE"]
    if fs == "BASE": cols = CLIN
    elif fs == "BASE+DM": cols = CLIN + DM
    elif fs == "BASE+HM": cols = CLIN + HM
    elif fs == "BASE+DRIVERS": cols = CLIN + DRV
    elif fs == "BASE+GEN": cols = CLIN + G + DRV
    elif fs == "BASE+PC":
        sc = StandardScaler().fit(co.loc[tr, G]); pca = PCA(N_PCS, random_state=SEED).fit(sc.transform(co.loc[tr, G]))
        ptr = pca.transform(sc.transform(co.loc[tr, G])); pte = pca.transform(sc.transform(co.loc[te, G]))
        names = [f"PC{i+1}" for i in range(N_PCS)]
        Xtr = np.hstack([co.loc[tr, CLIN].values, ptr]); Xte = np.hstack([co.loc[te, CLIN].values, pte])
        return Xtr, Xte, CLIN + names
    elif fs == "BASE+SHAPSEL":
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
    nb = len(baselines(co)["BASE"])
    for fs in ORDER:
        Xtr, Xte, cols = build_matrix(co, tr, te, fs, DM, HM, DRV)
        if fs == "BASE+SHAPSEL": rec["selected"] = cols[nb:]
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
        if fs == "BASE+GEN":
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
    order = ORDER
    summ = summ.reindex(order); summ.to_csv(os.path.join(OUT, "compare_summary.csv"))
    tests = []
    for metric in ["auc_lr", "auc_lgbm", "cindex_cox"]:
        base = res[res.feature_set == "BASE"].sort_values("split")[metric].values
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
    shs["type"] = ["clinical" if f in LEGACY else "expression" if f.startswith(("HM_", "DM_")) else
                   "copy number" if f.startswith("CNA_") else "mutation" for f in shs.index]
    shs.to_csv(os.path.join(OUT, "shap_stability_BASE_GEN.csv"))
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
