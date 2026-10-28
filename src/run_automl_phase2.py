"""
Phase 2, Step 3c: AutoML (MLJAR + FLAML) on the clinical + genomic feature sets.

AutoML libraries do classification, not survival, so the endpoint here is
binary recurrence (recurred yes/no), for all patients and for ER+ patients.
Same cohort as model_compare_shap.py (n = 1,295 complete cases).

  Feature sets : CLIN, CLIN+DM, CLIN+HM, CLIN+DRIVERS, CLIN+ALL
  Outer CV     : StratifiedKFold(5, shuffle, seed 42); AutoML tunes inside the
                 training fold only
  MLJAR        : Explain mode, eval AUC (as in the EGETC paper)
  FLAML        : 40 s per fold, LightGBM / XGBoost / RF / ExtraTrees / L1-LR,
                 internal 3-fold CV
  Statistics   : Nadeau-Bengio corrected paired t-test vs CLIN, Holm-adjusted

Run in chunks: METHOD=FLAML ENDPOINT=all FS=CLIN FOLDS=0,1,2 SUMMARIZE=0
Summarise only: METHOD=NONE SUMMARIZE=1
"""
import tempfile
import os, json, glob, time, shutil, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from model_compare_shap import load, CLIN
from run_recurrence_molecular import corrected_t, holm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2"); CK = os.path.join(OUT, "ck_automl")
os.makedirs(CK, exist_ok=True)
SEED = 42
FLAML_S = int(os.environ.get("FLAML_SECONDS", 40))
FSETS = ["CLIN", "CLIN+DM", "CLIN+HM", "CLIN+DRIVERS", "CLIN+ALL"]


def fit_predict(method, Xtr, ytr, Xte, tag, fold):
    if method == "FLAML":
        from flaml import AutoML
        m = AutoML()
        m.fit(Xtr, ytr, task="classification", metric="roc_auc", time_budget=FLAML_S,
              estimator_list=["lgbm", "xgboost", "rf", "extra_tree", "lrl1"],
              eval_method="cv", n_splits=3, seed=SEED + fold, verbose=0, n_jobs=4)
        return m.predict_proba(Xte)[:, 1], str(m.best_estimator)
    from supervised.automl import AutoML
    path = os.path.join(tempfile.gettempdir(), f"mljar2_{tag}")
    shutil.rmtree(path, ignore_errors=True)
    m = AutoML(results_path=path, mode="Explain", eval_metric="auc", total_time_limit=120,
               explain_level=0, random_state=SEED + fold, verbose=0, n_jobs=4,
               algorithms=["Baseline", "Linear", "Decision Tree", "Random Forest", "Xgboost", "LightGBM"])
    m.fit(Xtr, ytr)
    p = m.predict_proba(Xte); p = p[:, 1] if p.ndim == 2 else p
    best = m._best_model.get_name() if hasattr(m, "_best_model") else "mljar"
    shutil.rmtree(path, ignore_errors=True)
    return p, str(best)


def main():
    co, DM, HM, DRV = load()
    sets = {"CLIN": CLIN, "CLIN+DM": CLIN + DM, "CLIN+HM": CLIN + HM,
            "CLIN+DRIVERS": CLIN + DRV, "CLIN+ALL": CLIN + DM + HM + DRV}
    cohorts = {"all": co, "ERpos": co[co.er_binary == 1].reset_index(drop=True)}
    methods = [m for m in os.environ.get("METHOD", "MLJAR,FLAML").split(",") if m in ("MLJAR", "FLAML")]
    ends = os.environ.get("ENDPOINT", "all,ERpos").split(",")
    fs_sel = os.environ.get("FS", ",".join(FSETS)).split(",")
    fold_sel = [int(f) for f in os.environ.get("FOLDS", "0,1,2,3,4").split(",")]
    max_jobs = int(os.environ.get("MAXJOBS", 10**6)); done_now = 0
    for ep in ends:
        df = cohorts[ep]; y = df.y.astype(int).values
        folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(np.zeros(len(y)), y))
        for method in methods:
            for fs in fs_sel:
                cols = sets[fs]
                X = df[cols].astype(float).reset_index(drop=True)
                Xn = X.copy(); Xn.columns = [f"f{i}" for i in range(len(cols))]  # safe names for AutoML
                for k in fold_sel:
                    f = os.path.join(CK, f"{method}__{ep}__{fs}__{k}.json")
                    if os.path.exists(f): continue
                    if done_now >= max_jobs: break
                    done_now += 1
                    tr, te = folds[k]; t0 = time.time()
                    p, best = fit_predict(method, Xn.iloc[tr], y[tr], Xn.iloc[te], f"{ep}_{fs}_{k}".replace("+", "p"), k)
                    auc = roc_auc_score(y[te], p)
                    json.dump({"method": method, "endpoint": ep, "feature_set": fs, "fold": k, "auc": auc,
                               "best": best, "n": len(y), "events": int(y.sum()),
                               "seconds": round(time.time() - t0, 1)}, open(f, "w"))
                    print(f"{method} {ep} {fs} fold{k} AUC={auc:.3f} best={best} ({time.time()-t0:.0f}s)", flush=True)

    if os.environ.get("SUMMARIZE", "0") != "1":
        return
    r = pd.DataFrame([json.load(open(f)) for f in glob.glob(os.path.join(CK, "*.json"))])
    r.sort_values(["endpoint", "method", "feature_set", "fold"]).to_csv(os.path.join(OUT, "automl_folds.csv"), index=False)
    summ = (r.groupby(["endpoint", "method", "feature_set"]).auc.agg(["mean", "std", "count"]).reset_index())
    summ.to_csv(os.path.join(OUT, "automl_summary.csv"), index=False)
    tests = []
    for (ep, m), g in r.groupby(["endpoint", "method"]):
        base = g[g.feature_set == "CLIN"].sort_values("fold").auc.values
        rows = []
        for fs in FSETS[1:]:
            v = g[g.feature_set == fs].sort_values("fold").auc.values
            if len(v) != 5 or len(base) != 5: continue
            d, ci, p = corrected_t(v - base)
            rows.append({"endpoint": ep, "method": m, "feature_set": fs, "delta_auc": d, "ci_low": ci[0],
                         "ci_high": ci[1], "p_raw": p, "folds_improved": int((v > base).sum())})
        if rows:
            for row, a in zip(rows, holm([x["p_raw"] for x in rows])): row["p_holm"] = a
        tests += rows
    tests = pd.DataFrame(tests); tests.to_csv(os.path.join(OUT, "automl_tests.csv"), index=False)
    piv = summ.pivot_table(index=["endpoint", "method"], columns="feature_set", values="mean")[FSETS]
    print(piv.round(3).to_string()); print(tests.round(4).to_string())
    print(r.groupby(["method"]).best.value_counts().head(20).to_string())


if __name__ == "__main__":
    main()
