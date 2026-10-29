"""
Tuned survival models, 10 repeats x 5 folds (50 outer splits).

Revised specification (October 2026 review):
  * Baselines are cleaned of constant / exactly duplicated columns within each
    cohort (model_compare_shap.clean_cols):
      BASE  legacy EGETC baseline (clinical, treatment and mutation count)
            -> 21 columns in all patients, 19 in ER+ patients
      CLIN  strictly clinical/treatment comparator, BASE without Mutation Count
            and log_mutation -> 19 / 17 columns
  * Feature sets: X, X+GEN (all 77 genomic features), X+PC+DRV (9 principal
    components of the 57 gene-set scores + 20 driver features), for X in
    {BASE, CLIN}. The number of PCs (9) is a FIXED design choice taken from the
    descriptive PCA of all profiled tumours (80% variance); it is not tuned.
  * Every data-dependent step of the pipeline (scaling, PCA, model fit) is
    refitted inside each INNER training fold during hyperparameter selection,
    and again on the full outer training fold for the final model.
      - Ridge Cox: alpha in {1, 10, 100, 1000}, inner 3-fold CV C-index.
      - Random survival forest: 150 trees, sqrt(p) features, min_samples_leaf
        in {15, 40} chosen by inner 3-fold CV C-index (not out-of-bag).
  * Tests: corrected repeated k-fold t-test (r = 10, k = 5) against the matching
    baseline; Holm adjustment within each cohort x model family of 4 contrasts.
  * Out-of-fold risk predictions are saved with patient IDs.

Chunked + checkpointed (time-limited environments): rerun until it prints
"ALL SPLITS DONE"; then SUMMARIZE=1 builds the tables. Checkpoints carry a
specification hash and are ignored if the specification changes.
"""
import os, json, glob, time, hashlib, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from model_compare_shap import load, baselines, corrected_repeated_t, holm

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2"); CK = os.path.join(OUT, "ck_tuned_v2")
os.makedirs(CK, exist_ok=True)
SEED, K, R, N_PCS = 42, 5, 10, 9
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
LEAVES = [15, 40]
MAXSEC = float(os.environ.get("MAXSEC", 140))
FS = ["BASE", "BASE+GEN", "BASE+PC+DRV", "CLIN", "CLIN+GEN", "CLIN+PC+DRV"]
SPEC = hashlib.md5(json.dumps([FS, ALPHAS, LEAVES, N_PCS, K, R, SEED, "v2-innerpipe"]).encode()).hexdigest()[:10]


def cidx(y, t, risk):
    return concordance_index_censored(y.astype(bool), t, risk)[0]


def make_X(df, tr, te, cols, G, use_pca):
    """Design matrices; scaler and PCA are fitted on rows `tr` only."""
    if not use_pca:
        Xtr, Xte = df.loc[tr, cols].values.astype(float), df.loc[te, cols].values.astype(float)
    else:
        sg = StandardScaler().fit(df.loc[tr, G])
        pca = PCA(N_PCS, random_state=SEED).fit(sg.transform(df.loc[tr, G]))
        ptr, pte = pca.transform(sg.transform(df.loc[tr, G])), pca.transform(sg.transform(df.loc[te, G]))
        Xtr = np.hstack([df.loc[tr, cols].values.astype(float), ptr])
        Xte = np.hstack([df.loc[te, cols].values.astype(float), pte])
    sc = StandardScaler().fit(Xtr)
    return sc.transform(Xtr), sc.transform(Xte)


def fit_cox(Xtr, y, t, a):
    return CoxPHSurvivalAnalysis(alpha=a, n_iter=200).fit(Xtr, Surv.from_arrays(y.astype(bool), t))


def fit_rsf(Xtr, y, t, leaf):
    return RandomSurvivalForest(n_estimators=150, min_samples_leaf=leaf, max_features="sqrt",
                                n_jobs=4, random_state=SEED).fit(Xtr, Surv.from_arrays(y.astype(bool), t))


def tuned_split(df, tr, te, cols, G, use_pca):
    y, t = df.y.values.astype(int), df.t.values
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    sc_cox = {a: [] for a in ALPHAS}; sc_rsf = {l: [] for l in LEAVES}
    for itr, iva in inner.split(np.zeros(len(tr)), y[tr]):
        a_tr, a_va = tr[itr], tr[iva]
        Xa, Xv = make_X(df, a_tr, a_va, cols, G, use_pca)
        for a in ALPHAS:
            sc_cox[a].append(cidx(y[a_va], t[a_va], fit_cox(Xa, y[a_tr], t[a_tr], a).predict(Xv)))
        for l in LEAVES:
            sc_rsf[l].append(cidx(y[a_va], t[a_va], fit_rsf(Xa, y[a_tr], t[a_tr], l).predict(Xv)))
    a_best = max(ALPHAS, key=lambda a: np.mean(sc_cox[a]))
    l_best = max(LEAVES, key=lambda l: np.mean(sc_rsf[l]))
    Xtr, Xte = make_X(df, tr, te, cols, G, use_pca)
    rc = fit_cox(Xtr, y[tr], t[tr], a_best).predict(Xte)
    rr = fit_rsf(Xtr, y[tr], t[tr], l_best).predict(Xte)
    return rc, rr, a_best, l_best, Xtr.shape[1]


def fs_cols(fs, b, G, DRV):
    base = b["BASE"] if fs.startswith("BASE") else b["CLIN"]
    if fs.endswith("+GEN"):
        return base + G + DRV, False
    if fs.endswith("+PC+DRV"):
        return base + DRV, True
    return base, False


def main():
    co, DM, HM, DRV = load()
    G = DM + HM
    cohorts = {"ERpos": co[co.er_binary == 1].reset_index(drop=True), "all": co}
    t0 = time.time()
    for name, df in cohorts.items():
        b = baselines(df)
        y = df.y.values.astype(int)
        splits = list(RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=SEED).split(np.zeros(len(df)), y))
        for s, (tr, te) in enumerate(splits):
            f = os.path.join(CK, f"{name}_{s:02d}.json")
            if os.path.exists(f) and json.load(open(f)).get("spec") == SPEC:
                continue
            if time.time() - t0 > MAXSEC:
                continue
            rec = {"spec": SPEC, "cohort": name, "split": s, "test_ids": df.loc[te, "Patient ID"].tolist(), "res": []}
            for fs in FS:
                cols, use_pca = fs_cols(fs, b, G, DRV)
                rc, rr, a, l, p = tuned_split(df, tr, te, cols, G, use_pca)
                rec["res"].append({"feature_set": fs, "n_features": p,
                                   "cox_cindex": cidx(y[te], df.t.values[te], rc), "cox_alpha": a,
                                   "rsf_cindex": cidx(y[te], df.t.values[te], rr), "rsf_leaf": l,
                                   "cox_risk": [float(v) for v in rc], "rsf_risk": [float(v) for v in rr]})
            tmp = f + ".tmp"; json.dump(rec, open(tmp, "w")); os.replace(tmp, f)   # atomic write
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CK, "*.json")))]
    recs = [r for r in recs if r.get("spec") == SPEC]
    done = len(recs)
    print(f"splits saved: {done}/{2 * K * R}" + ("  ALL SPLITS DONE" if done == 2 * K * R else ""), flush=True)
    if os.environ.get("SUMMARIZE") != "1" or done < 2 * K * R:
        return
    rows, preds = [], []
    for rec in recs:
        for r in rec["res"]:
            rows.append({"cohort": rec["cohort"], "split": rec["split"],
                         **{k: v for k, v in r.items() if not k.endswith("_risk")}})
            for pid, a, b_ in zip(rec["test_ids"], r["cox_risk"], r["rsf_risk"]):
                preds.append({"cohort": rec["cohort"], "split": rec["split"], "feature_set": r["feature_set"],
                              "Patient ID": pid, "cox_risk": a, "rsf_risk": b_})
    r = pd.DataFrame(rows)
    # completeness check: every cohort x split x feature set exactly once
    cnt = r.groupby(["cohort", "feature_set"]).split.nunique()
    assert (cnt == K * R).all() and not r.duplicated(["cohort", "split", "feature_set"]).any(), cnt
    r.to_csv(os.path.join(OUT, "tuned_survival_folds.csv"), index=False)
    pd.DataFrame(preds).to_csv(os.path.join(OUT, "tuned_survival_oof_predictions.csv.gz"), index=False)
    summ = r.groupby(["cohort", "feature_set"]).agg(n_features=("n_features", "first"),
            cox_mean=("cox_cindex", "mean"), cox_sd=("cox_cindex", "std"),
            rsf_mean=("rsf_cindex", "mean"), rsf_sd=("rsf_cindex", "std")).reset_index()
    summ.to_csv(os.path.join(OUT, "tuned_survival_summary.csv"), index=False)
    tests = []
    for coh, g in r.groupby("cohort"):
        for model in ["cox_cindex", "rsf_cindex"]:
            fam = []
            for base, fss in [("BASE", ["BASE+GEN", "BASE+PC+DRV"]), ("CLIN", ["CLIN+GEN", "CLIN+PC+DRV"])]:
                bv = g[g.feature_set == base].sort_values("split")[model].values
                for fs in fss:
                    v = g[g.feature_set == fs].sort_values("split")[model].values
                    m, lo, hi, p = corrected_repeated_t(v - bv, k=K, r=R)
                    fam.append({"cohort": coh, "model": model.split("_")[0], "comparison": f"{fs} vs {base}",
                                "delta": m, "ci_low": lo, "ci_high": hi, "p_raw": p,
                                "splits_improved": int((v > bv).sum()), "n_splits": len(v)})
            for x, a in zip(fam, holm([x["p_raw"] for x in fam])): x["p_holm"] = a
            tests += fam
    tests = pd.DataFrame(tests); tests.to_csv(os.path.join(OUT, "tuned_survival_tests.csv"), index=False)
    sel = r.groupby(["cohort", "feature_set"]).agg(
        alpha=("cox_alpha", lambda s: s.value_counts().to_dict()), leaf=("rsf_leaf", lambda s: s.value_counts().to_dict()))
    sel.to_csv(os.path.join(OUT, "tuned_survival_selected_params.csv"))
    print(summ.round(3).to_string()); print(tests.round(4).to_string()); print(sel.to_string())


if __name__ == "__main__":
    main()
