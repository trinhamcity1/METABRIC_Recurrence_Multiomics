"""
METABRIC genomic extension — Phase 1
Does adding molecular (genomics-derived) features improve RECURRENCE prediction
over the clinical feature set used in the EGETC 2026 paper?

Feature sets (all evaluated on the SAME patients and SAME seeded folds):
  CLIN      : the 22 clinical/treatment features from the EGETC 2026 paper
  CLIN+MOL  : CLIN + PAM50, Integrative Cluster (IntClust), 3-gene classifier,
              HER2 copy-number state (SNP6), tumour cellularity
  MOL       : molecular features + age only (how far does biology alone get?)

Methods: Logistic Regression (fixed), FLAML AutoML, MLJAR AutoML.
Outer CV: StratifiedKFold(5, shuffle, seed 42). AutoML internal tuning sees
only the outer-training fold.
Statistics: Nadeau-Bengio corrected paired t-test (df=4, var*(1/5+1/4)),
Holm-adjusted across methods.
"""
import tempfile
import os, sys, json, time, warnings, shutil
import numpy as np, pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score, brier_score_loss, average_precision_score

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("METABRIC_CSV", os.path.join(ROOT, "data", "Breast_Cancer_METABRIC.csv"))
OUT = os.environ.get("OUT_DIR", os.path.join(ROOT, "results", "pilot"))
os.makedirs(OUT, exist_ok=True)
SEED = 42
MLJAR_SECONDS = int(os.environ.get("MLJAR_SECONDS", 90))
FLAML_SECONDS = int(os.environ.get("FLAML_SECONDS", 45))
METHODS = os.environ.get("METHODS", "LR,FLAML,MLJAR").split(",")

# ------------------------------------------------------------------ features
def load():
    d = pd.read_csv(DATA)
    d.columns = [c.strip() for c in d.columns]
    rec = d["Relapse Free Status"].map({"Recurred": 1, "Not Recurred": 0})
    d["y"] = rec  # unknown outcome stays NaN (never treated as negative)

    def binmap(col, pos):
        s = d[col]
        return np.where(s.isna(), np.nan, (s == pos).astype(float))

    d["er_binary"] = binmap("ER Status", "Positive")
    d["pr_binary"] = binmap("PR Status", "Positive")
    d["her2_pos"] = binmap("HER2 Status", "Positive")
    d["chemo"] = binmap("Chemotherapy", "Yes")
    d["hormone_rx"] = binmap("Hormone Therapy", "Yes")
    d["radio_rx"] = binmap("Radio Therapy", "Yes")
    d["mastectomy"] = binmap("Type of Breast Surgery", "Mastectomy")
    d["postmeno"] = binmap("Inferred Menopausal State", "Post")
    ln = d["Lymph nodes examined positive"]
    d["ln_pos_flag"] = np.where(ln.isna(), np.nan, (ln > 0).astype(float))
    g = d["Neoplasm Histologic Grade"]
    d["high_grade"] = np.where(g.isna(), np.nan, (g == 3).astype(float))
    d["log_mutation"] = np.log1p(d["Mutation Count"])
    d["tumor_size_cm"] = d["Tumor Size"] / 10
    npi = d["Nottingham prognostic index"]
    d["npi_group"] = np.where(npi.isna(), np.nan, np.where(npi <= 3.4, 0, np.where(npi <= 5.4, 1, 2)))
    age = d["Age at Diagnosis"]
    d["age_group"] = np.where(age.isna(), np.nan, np.where(age < 50, 0, np.where(age <= 65, 1, 2)))
    d["er_x_her2"] = d["er_binary"] * d["her2_pos"]

    clin = ["Age at Diagnosis", "Neoplasm Histologic Grade", "Lymph nodes examined positive",
            "Mutation Count", "Nottingham prognostic index", "Tumor Size", "Tumor Stage",
            "er_binary", "pr_binary", "her2_pos", "chemo", "hormone_rx", "radio_rx",
            "mastectomy", "postmeno", "ln_pos_flag", "high_grade", "log_mutation",
            "tumor_size_cm", "npi_group", "age_group", "er_x_her2"]

    # molecular (genomics-derived) features already shipped in the cBioPortal clinical table
    pam = d["Pam50 + Claudin-low subtype"].replace({"NC": np.nan})
    ic = d["Integrative Cluster"]
    tg = d["3-Gene classifier subtype"]
    snp = d["HER2 status measured by SNP6"].replace({"Undef": np.nan})
    cell = d["Cellularity"].map({"Low": 0, "Moderate": 1, "High": 2})
    mol_parts = []
    for name, s in [("pam50", pam), ("intclust", ic), ("threegene", tg), ("her2cna", snp)]:
        dummies = pd.get_dummies(s, prefix=name, dtype=float)
        dummies.loc[s.isna(), :] = np.nan
        mol_parts.append(dummies)
    mol_df = pd.concat(mol_parts, axis=1)
    mol_df["cellularity"] = cell
    mol_df.columns = [c.replace(" ", "_").replace("/", "_").replace("+", "pos").replace("-", "neg")
                      for c in mol_df.columns]
    d = pd.concat([d, mol_df], axis=1)
    mol = list(mol_df.columns)

    groups = {
        "PAM50": [c for c in mol if c.startswith("pam50_")],
        "IntClust": [c for c in mol if c.startswith("intclust_")],
        "3-gene": [c for c in mol if c.startswith("threegene_")],
        "HER2 CNA": [c for c in mol if c.startswith("her2cna_")],
        "Cellularity": ["cellularity"],
    }
    return d, clin, mol, groups


def corrected_t(diffs, n_train_over_test=4):
    k = len(diffs)
    m, s2 = np.mean(diffs), np.var(diffs, ddof=1)
    se = np.sqrt(s2 * (1 / k + 1 / n_train_over_test))
    t = m / se if se > 0 else np.inf
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    half = stats.t.ppf(0.975, k - 1) * se
    return m, (m - half, m + half), p


def holm(pvals):
    p = np.asarray(pvals); order = np.argsort(p); adj = np.empty_like(p); run = 0
    for i, idx in enumerate(order):
        run = max(run, (len(p) - i) * p[idx]); adj[idx] = min(run, 1.0)
    return adj

# ------------------------------------------------------------------ models
def fit_predict(method, Xtr, ytr, Xte, fold, fs):
    if method == "LR":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(Xtr, ytr)
        return m.predict_proba(Xte)[:, 1], m, "LogReg C=1"
    if method == "FLAML":
        from flaml import AutoML
        m = AutoML()
        m.fit(Xtr, ytr, task="classification", metric="roc_auc", time_budget=FLAML_SECONDS,
              estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "lrl1"],
              eval_method="cv", n_splits=3, seed=SEED + fold, verbose=0, n_jobs=4)
        return m.predict_proba(Xte)[:, 1], m, m.best_estimator
    if method == "MLJAR":
        from supervised.automl import AutoML
        path = os.path.join(tempfile.gettempdir(), f"mljar_{fs}_{fold}")
        shutil.rmtree(path, ignore_errors=True)
        m = AutoML(results_path=path, mode="Explain", eval_metric="auc",
                   total_time_limit=MLJAR_SECONDS, explain_level=0, random_state=SEED + fold,
                   algorithms=["Baseline", "Linear", "Decision Tree", "Random Forest", "Xgboost", "LightGBM"],
                   verbose=0, n_jobs=4)
        Xtr_df = pd.DataFrame(Xtr); Xte_df = pd.DataFrame(Xte)
        m.fit(Xtr_df, ytr)
        p = m.predict_proba(Xte_df)
        p = p[:, 1] if p.ndim == 2 else p
        return p, m, str(m._best_model.get_name()) if hasattr(m, "_best_model") else "mljar"
    raise ValueError(method)


def group_perm_importance(model, method, Xte, yte, cols, groups, rng, n_rep=10):
    base = roc_auc_score(yte, predict(model, method, Xte))
    out = {}
    for g, gcols in groups.items():
        idx = [cols.index(c) for c in gcols if c in cols]
        if not idx: continue
        drops = []
        for _ in range(n_rep):
            Xp = Xte.copy(); perm = rng.permutation(len(Xp))
            Xp[:, idx] = Xte[perm][:, idx]   # permute the whole group jointly
            drops.append(base - roc_auc_score(yte, predict(model, method, Xp)))
        out[g] = float(np.mean(drops))
    return out


def predict(model, method, X):
    if method == "MLJAR":
        p = model.predict_proba(pd.DataFrame(X)); return p[:, 1] if p.ndim == 2 else p
    return model.predict_proba(X)[:, 1]

# ------------------------------------------------------------------ main
def main():
    d, clin, mol, mgroups = load()
    feature_sets = {"CLIN": clin, "CLIN+MOL": clin + mol, "MOL": ["Age at Diagnosis"] + mol}
    # same patients for every feature set: complete on the union + recorded outcome
    cohort = d.dropna(subset=["y"] + clin + mol).reset_index(drop=True)
    y = cohort["y"].astype(int).values
    info = {"n": int(len(y)), "n_recurred": int(y.sum()), "prevalence": float(y.mean()),
            "n_clin": len(clin), "n_mol": len(mol), "mljar_s": MLJAR_SECONDS, "flaml_s": FLAML_SECONDS}
    print(json.dumps(info)); json.dump(info, open(os.path.join(OUT, "cohort.json"), "w"), indent=2)

    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    folds = list(skf.split(np.zeros(len(y)), y))
    rows, oof, imps = [], {}, []
    rng = np.random.default_rng(SEED)
    # clinical groups for importance on the combined set
    cgroups = {"NPI / grade / size / stage": ["Nottingham prognostic index", "npi_group", "Neoplasm Histologic Grade",
                                               "high_grade", "Tumor Size", "tumor_size_cm", "Tumor Stage"],
               "Lymph nodes": ["Lymph nodes examined positive", "ln_pos_flag"],
               "Receptors (ER/PR/HER2 IHC)": ["er_binary", "pr_binary", "her2_pos", "er_x_her2"],
               "Treatment / surgery": ["chemo", "hormone_rx", "radio_rx", "mastectomy"],
               "Age / menopause": ["Age at Diagnosis", "age_group", "postmeno"],
               "Mutation count": ["Mutation Count", "log_mutation"]}
    all_groups = {**cgroups, **mgroups}

    FS_SEL = os.environ.get("FEATURE_SETS", "CLIN,CLIN+MOL,MOL").split(",")
    FOLD_SEL = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
    ck = os.path.join(OUT, "checkpoints"); os.makedirs(ck, exist_ok=True)
    plan = [(m, fs) for m in METHODS for fs in feature_sets
            if m in ("LR", "FLAML", "MLJAR") and fs in FS_SEL and not (m == "MLJAR" and fs == "MOL")]
    for method, fs in plan:
        cols = feature_sets[fs]
        X = cohort[cols].astype(float).values
        for k, (tr, te) in enumerate(folds):
            f = os.path.join(ck, f"{method}__{fs}__{k}.json")
            if k not in FOLD_SEL or os.path.exists(f):
                continue
            t0 = time.time()
            p, model, best = fit_predict(method, X[tr], y[tr], X[te], k, fs)
            auc = roc_auc_score(y[te], p)
            rec = {"method": method, "feature_set": fs, "fold": k, "auc": auc, "best_model": str(best),
                   "seconds": round(time.time() - t0, 1), "test_idx": te.tolist(), "pred": list(map(float, p))}
            if fs == "CLIN+MOL" and method in ("LR", "FLAML"):
                rec["importance"] = group_perm_importance(model, method, X[te], y[te], cols, all_groups, rng)
            json.dump(rec, open(f, "w"))
            print(f"{method:6s} {fs:9s} fold{k} AUC={auc:.3f} best={best} ({time.time()-t0:.0f}s)", flush=True)

    if os.environ.get("SUMMARIZE", "1") != "1":
        return
    import glob
    for f in sorted(glob.glob(os.path.join(ck, "*.json"))):
        r = json.load(open(f))
        rows.append({k2: r[k2] for k2 in ("method", "feature_set", "fold", "auc", "best_model", "seconds")})
        key = f"{r['method']}|{r['feature_set']}"
        oof.setdefault(key, np.full(len(y), np.nan))[np.array(r["test_idx"])] = r["pred"]
        for g, v in r.get("importance", {}).items():
            imps.append({"method": r["method"], "fold": r["fold"], "group": g, "auc_drop": v})
    pd.DataFrame(rows).sort_values(["method", "feature_set", "fold"]).to_csv(os.path.join(OUT, "fold_auc.csv"), index=False)
    pd.DataFrame(oof).assign(y=y).to_csv(os.path.join(OUT, "oof_predictions.csv"), index=False)
    METHODS_DONE = sorted({r["method"] for r in rows})

    # ---------------- summaries
    fa = pd.DataFrame(rows)
    summ = []
    for (m, fs), g in fa.groupby(["method", "feature_set"]):
        if len(g) < 5: continue
        p = oof[f"{m}|{fs}"]
        summ.append({"method": m, "feature_set": fs, "mean_auc": g.auc.mean(), "sd_auc": g.auc.std(ddof=1),
                     "pooled_auc": roc_auc_score(y, p), "avg_precision": average_precision_score(y, p),
                     "brier": brier_score_loss(y, p)})
    summ = pd.DataFrame(summ); summ.to_csv(os.path.join(OUT, "summary.csv"), index=False)

    tests = []
    for m in METHODS_DONE:
        a = fa[(fa.method == m) & (fa.feature_set == "CLIN+MOL")].sort_values("fold").auc.values
        b = fa[(fa.method == m) & (fa.feature_set == "CLIN")].sort_values("fold").auc.values
        if len(a) == 5 and len(b) == 5:
            diff, ci, p = corrected_t(a - b)
            tests.append({"method": m, "comparison": "CLIN+MOL - CLIN", "delta_auc": diff,
                          "ci_low": ci[0], "ci_high": ci[1], "p_raw": p,
                          "folds_improved": int((a > b).sum())})
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["p_holm"] = holm(tests.p_raw.values)
    tests.to_csv(os.path.join(OUT, "paired_tests.csv"), index=False)
    if imps:
        gi = pd.DataFrame(imps).groupby(["method", "group"]).auc_drop.agg(["mean", "std"]).reset_index()
        gi.sort_values(["method", "mean"], ascending=[True, False]).to_csv(
            os.path.join(OUT, "group_importance.csv"), index=False)
    print(summ.round(3).to_string()); print(tests.round(4).to_string())


if __name__ == "__main__":
    main()
