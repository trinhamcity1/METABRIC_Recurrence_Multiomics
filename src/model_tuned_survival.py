"""
Phase 2, Step 3e: TUNED survival models with more CV repeats.

Everything we can tune ourselves, done fold-safe:
  * Penalised Cox (ridge): penalty alpha chosen by inner 3-fold CV (C-index) on
    the training fold only; grid alpha in {1, 10, 100, 1000}.
  * Random survival forest (RSF): 150 trees; min_samples_leaf chosen from
    {15, 40} by out-of-bag C-index on the training fold only.
  * Feature sets:
      CLIN          22 clinical features
      CLIN+GEN      + all 77 genomic features (57 gene-set scores + 20 drivers)
      CLIN+PC+DRV   reduced: + 9 PCs of the 57 gene-set scores (PCA fit on the
                    training fold) + 20 driver features  -> 51 features
  * Patients: ER+ (main question) and all patients.
  * CV: 10 repeats x 5 folds = 50 outer splits; corrected repeated k-fold
    t-test (Bouckaert & Frank 2004) with r = 10, Holm across comparisons.

Chunked + checkpointed: re-run until it prints "ALL SPLITS DONE"; then
SUMMARIZE=1 builds the tables.
"""
import os, json, glob, time, warnings
import numpy as np, pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from model_compare_shap import load, CLIN, corrected_repeated_t, holm

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2"); CK = os.path.join(OUT, "ck_tuned")
os.makedirs(CK, exist_ok=True)
SEED, K, R = 42, 5, 10
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
LEAVES = [15, 40]
MAXSEC = float(os.environ.get("MAXSEC", 140))


def cidx(y, t, risk):
    return concordance_index_censored(y.astype(bool), t, risk)[0]


def design(df, tr, te, fs, DM, HM, DRV):
    G = DM + HM
    if fs == "CLIN":
        return df.loc[tr, CLIN].values, df.loc[te, CLIN].values
    if fs == "CLIN+GEN":
        c = CLIN + G + DRV
        return df.loc[tr, c].values, df.loc[te, c].values
    sc = StandardScaler().fit(df.loc[tr, G])
    pca = PCA(9, random_state=SEED).fit(sc.transform(df.loc[tr, G]))
    ptr, pte = pca.transform(sc.transform(df.loc[tr, G])), pca.transform(sc.transform(df.loc[te, G]))
    c = CLIN + DRV
    return np.hstack([df.loc[tr, c].values, ptr]), np.hstack([df.loc[te, c].values, pte])


def tuned_cox(Xtr, ytr, ttr, Xte):
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    scores = {a: [] for a in ALPHAS}
    for itr, iva in inner.split(Xtr, ytr):
        sc = StandardScaler().fit(Xtr[itr])
        for a in ALPHAS:
            m = CoxPHSurvivalAnalysis(alpha=a, n_iter=200).fit(sc.transform(Xtr[itr]), Surv.from_arrays(ytr[itr].astype(bool), ttr[itr]))
            scores[a].append(cidx(ytr[iva], ttr[iva], m.predict(sc.transform(Xtr[iva]))))
    best = max(ALPHAS, key=lambda a: np.mean(scores[a]))
    sc = StandardScaler().fit(Xtr)
    m = CoxPHSurvivalAnalysis(alpha=best, n_iter=200).fit(sc.transform(Xtr), Surv.from_arrays(ytr.astype(bool), ttr))
    return m.predict(sc.transform(Xte)), best


def tuned_rsf(Xtr, ytr, ttr, Xte):
    best, best_s, best_m = None, -1, None
    for leaf in LEAVES:
        m = RandomSurvivalForest(n_estimators=150, min_samples_leaf=leaf, max_features="sqrt", oob_score=True,
                                 n_jobs=4, random_state=SEED).fit(Xtr, Surv.from_arrays(ytr.astype(bool), ttr))
        if m.oob_score_ > best_s:
            best, best_s, best_m = leaf, m.oob_score_, m
    return best_m.predict(Xte), best


def main():
    co, DM, HM, DRV = load()
    cohorts = {"ERpos": co[co.er_binary == 1].reset_index(drop=True), "all": co}
    FS = ["CLIN", "CLIN+GEN", "CLIN+PC+DRV"]
    t0 = time.time(); pending = 0
    for name, df in cohorts.items():
        y, t = df.y.values.astype(int), df.t.values
        splits = list(RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=SEED).split(np.zeros(len(df)), y))
        for s, (tr, te) in enumerate(splits):
            f = os.path.join(CK, f"{name}_{s:02d}.json")
            if os.path.exists(f): continue
            if time.time() - t0 > MAXSEC: pending += 1; continue
            rec = {"cohort": name, "split": s, "res": []}
            for fs in FS:
                Xtr, Xte = design(df, tr, te, fs, DM, HM, DRV)
                rc, a = tuned_cox(Xtr, y[tr], t[tr], Xte)
                rr, leaf = tuned_rsf(Xtr, y[tr], t[tr], Xte)
                rec["res"].append({"feature_set": fs, "n_features": Xtr.shape[1],
                                   "cox_cindex": cidx(y[te], t[te], rc), "cox_alpha": a,
                                   "rsf_cindex": cidx(y[te], t[te], rr), "rsf_leaf": leaf})
            json.dump(rec, open(f, "w"))
    done = len(glob.glob(os.path.join(CK, "*.json")))
    print(f"splits saved: {done}/{2 * K * R}" + ("  ALL SPLITS DONE" if done == 2 * K * R else ""))
    if os.environ.get("SUMMARIZE") != "1" or done < 2 * K * R:
        return
    rows = []
    for f in glob.glob(os.path.join(CK, "*.json")):
        rec = json.load(open(f))
        for r in rec["res"]: rows.append({"cohort": rec["cohort"], "split": rec["split"], **r})
    r = pd.DataFrame(rows); r.to_csv(os.path.join(OUT, "tuned_survival_folds.csv"), index=False)
    summ = r.groupby(["cohort", "feature_set"]).agg(n_features=("n_features", "first"),
            cox_mean=("cox_cindex", "mean"), cox_sd=("cox_cindex", "std"),
            rsf_mean=("rsf_cindex", "mean"), rsf_sd=("rsf_cindex", "std")).reset_index()
    summ.to_csv(os.path.join(OUT, "tuned_survival_summary.csv"), index=False)
    tests = []
    for (coh), g in r.groupby("cohort"):
        rows_t = []
        for model in ["cox_cindex", "rsf_cindex"]:
            base = g[g.feature_set == "CLIN"].sort_values("split")[model].values
            for fs in FS[1:]:
                v = g[g.feature_set == fs].sort_values("split")[model].values
                m, lo, hi, p = corrected_repeated_t(v - base, k=K, r=R)
                rows_t.append({"cohort": coh, "model": model.split("_")[0], "feature_set": fs, "delta": m,
                               "ci_low": lo, "ci_high": hi, "p_raw": p, "splits_improved": int((v > base).sum()),
                               "n_splits": len(v)})
        for x, a in zip(rows_t, holm([x["p_raw"] for x in rows_t])): x["p_holm"] = a
        tests += rows_t
    tests = pd.DataFrame(tests); tests.to_csv(os.path.join(OUT, "tuned_survival_tests.csv"), index=False)
    print(summ.round(3).to_string()); print(tests.round(4).to_string())
    print("chosen Cox alpha:", r.cox_alpha.value_counts().to_dict(), "| chosen RSF leaf:", r.rsf_leaf.value_counts().to_dict())


if __name__ == "__main__":
    main()
