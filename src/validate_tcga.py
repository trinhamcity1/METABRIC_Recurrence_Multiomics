"""
Exploratory transport assessment: METABRIC -> TCGA-BRCA.

This is NOT external validation of an unchanged, frozen predictor. It assesses a
separately specified HARMONIZED model with unsupervised target-cohort adaptation:
  * Models are refitted in METABRIC using 6 harmonized predictors (HCLIN) instead
    of the internal 21-column baseline.
  * Continuous molecular features (gene-set scores, copy-number fraction, mutation
    count) are standardized within each cohort, using the whole unlabeled TCGA
    cohort to estimate TCGA means and SDs (no outcomes used).
  * TCGA ER status is inferred from ESR1 expression by a two-component Gaussian
    mixture fitted to TCGA expression (build_tcga_features.py).
A prospective application to a single new patient would require these
transformations to be fixed before that patient is seen; that is not evaluated.

Train: METABRIC modelling cohort, relapse-free survival.
Test : TCGA-BRCA, progression-free interval (PFI). Endpoint definitions, follow-up
and case mix differ and may change the estimated increment in either direction.

Feature sets:
  HCLIN          age, T category (METABRIC: from tumour size; TCGA: pathological T,
                 T4 recoded as T3), N category (METABRIC: from positive-node count;
                 TCGA: pathological N), ER status (METABRIC IHC; TCGA ESR1-inferred),
                 ERBB2 high-level amplification (proxy for clinical HER2), radiotherapy
  HCLIN+GEN      + 57 gene-set scores + 19 driver features (ERBB2 amp already in HCLIN)
  HCLIN+PC+DRV   + 9 PCs of the gene-set scores (PCA fit on METABRIC) + 19 drivers

Tuning (METABRIC only, same rules as model_tuned_survival.py): ridge Cox alpha and
RSF leaf size chosen by inner 3-fold CV of the whole pipeline (scaling, PCA, model).
Uncertainty: 1,000 paired bootstrap resamples of the TCGA set.
Follow-up: median by reverse Kaplan-Meier, plus the median of observed times.

Also writes metabric_internal_harmonized_cox.csv: the same HCLIN vs HCLIN+GEN
comparison inside METABRIC (ridge Cox alpha = 10, 5 x 5 repeated CV).
Fitted Cox models, all scalers and PCA objects, and the chosen RSF settings are saved to\nresults/external_tcga/models/ (RSF forests are refit exactly from the seed).
"""
import os, json, warnings
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sksurv.nonparametric import kaplan_meier_estimator
from model_compare_shap import load, corrected_repeated_t, holm
from model_tuned_survival import fit_cox, fit_rsf, cidx, ALPHAS, LEAVES
from model_er_late import cox_cv

warnings.filterwarnings("ignore", category=FutureWarning)
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "external_tcga"); os.makedirs(os.path.join(OUT, "models"), exist_ok=True)
SEED, NBOOT, N_PCS = 42, 1000, 9
HCLIN = ["age", "T_cat", "N_cat", "er_binary", "her2_amp", "radio_rx"]


def harmonize_metabric(co):
    d = co.copy()
    d["age"] = d["Age at Diagnosis"]
    sz = d["Tumor Size"]
    d["T_cat"] = np.where(sz <= 20, 1, np.where(sz <= 50, 2, 3)).astype(float)
    ln = d["Lymph nodes examined positive"]
    d["N_cat"] = np.where(ln == 0, 0, np.where(ln <= 3, 1, np.where(ln <= 9, 2, 3))).astype(float)
    d["her2_amp"] = d["CNA_ERBB2_amp"]
    return d


def standardize_within(df, cols):
    df = df.copy()
    mu, sd = df[cols].mean(), df[cols].std(ddof=0)
    df[cols] = (df[cols] - mu) / sd
    return df, {"mean": mu.to_dict(), "sd": sd.to_dict()}


def reverse_km_median(t, e):
    """Median follow-up by reverse Kaplan-Meier (censoring treated as the event)."""
    x, s = kaplan_meier_estimator((1 - e).astype(bool), t)
    below = np.where(s <= 0.5)[0]
    return float(x[below[0]]) if len(below) else float("nan")


class Pipe:
    """Scaler (+ optional PCA on gene-set columns) + model, fitted on training rows only."""
    def __init__(self, base_cols, G, use_pca):
        self.base_cols, self.G, self.use_pca = base_cols, G, use_pca

    def fit_transform(self, df):
        X = df[self.base_cols].values.astype(float)
        if self.use_pca:
            self.sg = StandardScaler().fit(df[self.G]); self.pca = PCA(N_PCS, random_state=SEED).fit(self.sg.transform(df[self.G]))
            X = np.hstack([X, self.pca.transform(self.sg.transform(df[self.G]))])
        self.sc = StandardScaler().fit(X)
        return self.sc.transform(X)

    def transform(self, df):
        X = df[self.base_cols].values.astype(float)
        if self.use_pca:
            X = np.hstack([X, self.pca.transform(self.sg.transform(df[self.G]))])
        return self.sc.transform(X)


def tune_and_fit(tr, cols, G, use_pca, model):
    y, t = tr.y.values.astype(int), tr.t.values
    grid = ALPHAS if model == "Cox" else LEAVES
    fit = fit_cox if model == "Cox" else fit_rsf
    sc = {g: [] for g in grid}
    for itr, iva in StratifiedKFold(3, shuffle=True, random_state=SEED).split(np.zeros(len(tr)), y):
        p = Pipe(cols, G, use_pca); Xa = p.fit_transform(tr.iloc[itr]); Xv = p.transform(tr.iloc[iva])
        for g in grid:
            sc[g].append(cidx(y[iva], t[iva], fit(Xa, y[itr], t[itr], g).predict(Xv)))
    best = max(grid, key=lambda g: np.mean(sc[g]))
    p = Pipe(cols, G, use_pca); X = p.fit_transform(tr)
    return p, fit(X, y, t, best), best


def boot_ci(y, t, risks, base, n=NBOOT):
    rng = np.random.default_rng(SEED)
    out = {k: [] for k in risks}; dlt = {k: [] for k in risks if k != base}
    for _ in range(n):
        i = rng.choice(len(y), len(y), replace=True)
        if y[i].sum() < 5: continue
        cb = cidx(y[i], t[i], risks[base][i])
        for k, r in risks.items():
            c = cidx(y[i], t[i], r[i]); out[k].append(c)
            if k != base: dlt[k].append(c - cb)
    return out, dlt


def main():
    co, DM, HM, DRV = load()
    mb = harmonize_metabric(co)
    tc = pd.read_csv(os.path.join(ROOT, "data", "processed", "tcga_brca_features.csv"))
    G = DM + HM
    DRV2 = [c for c in DRV if c != "CNA_ERBB2_amp"]
    # sensitivity: drop the mutation count (173-gene panel vs exome are not comparable)
    SFX = ""
    if os.environ.get("EXCL_MUTCOUNT") == "1":
        DRV = [c for c in DRV if c != "MUT_nonsilent_count_log"]; DRV2 = [c for c in DRV2 if c != "MUT_nonsilent_count_log"]
        SFX = "_no_mutcount"
    cont = G + ["CNA_frac_genes_altered"] + (["MUT_nonsilent_count_log"] if not SFX else [])
    tc = tc.dropna(subset=["y", "t"] + HCLIN + G + DRV); tc = tc[tc.t > 0].reset_index(drop=True)
    mb, mb_par = standardize_within(mb, cont); tc, tc_par = standardize_within(tc, cont)
    joblib.dump({"metabric": mb_par, "tcga": tc_par}, os.path.join(OUT, "models", f"within_cohort_standardization{SFX}.joblib"))

    # ---- internal harmonized comparison in METABRIC (5 x 5 CV, alpha = 10)
    irows = []
    for sub in ["ER+", "All"]:
        d = (mb if sub == "All" else mb[mb.er_binary == 1]).reset_index(drop=True)
        hc = [c for c in HCLIN if d[c].std() > 0]          # er_binary is constant in ER+
        res = cox_cv(d, {"HCLIN": hc, "HCLIN+GEN": hc + G + DRV2})
        m, lo, hi, p = corrected_repeated_t(res["HCLIN+GEN"] - res["HCLIN"])
        irows.append({"subgroup": sub, "n": len(d), "events": int(d.y.sum()), "hclin": res["HCLIN"].mean(),
                      "hclin_gen": res["HCLIN+GEN"].mean(), "delta": m, "ci_low": lo, "ci_high": hi, "p_raw": p,
                      "splits_improved": int((res["HCLIN+GEN"] > res["HCLIN"]).sum())})
    pd.DataFrame(irows).to_csv(os.path.join(OUT, f"metabric_internal_harmonized_cox{SFX}.csv"), index=False)
    print(pd.DataFrame(irows).round(4).to_string(), flush=True)

    rows, tests, info = [], [], {}
    for sub in ["ER+", "All"]:
        tr = (mb if sub == "All" else mb[mb.er_binary == 1]).reset_index(drop=True)
        te = (tc if sub == "All" else tc[tc.er_binary == 1]).reset_index(drop=True)
        hc = [c for c in HCLIN if tr[c].std() > 0]
        yte, tte = te.y.values.astype(int), te.t.values
        info[sub] = {"label": "ESR1-inferred ER-positive" if sub == "ER+" else "all eligible",
                     "train_n": len(tr), "train_events": int(tr.y.sum()), "test_n": len(te),
                     "test_events": int(yte.sum()),
                     "test_median_followup_reverse_km_months": reverse_km_median(tte, yte),
                     "test_median_observed_time_months": float(np.median(tte))}
        designs = {"HCLIN": (hc, False), "HCLIN+GEN": (hc + G + DRV2, False), "HCLIN+PC+DRV": (hc + DRV2, True)}
        for model in ["Cox", "RSF"]:
            risks = {}
            for fs, (cols, use_pca) in designs.items():
                pipe, m, par = tune_and_fit(tr, cols, G, use_pca, model)
                r = m.predict(pipe.transform(te)); risks[fs] = r
                # RSF objects are ~100 MB each; store the fitted preprocessing and the chosen
                # parameter (the forest is refit exactly with random_state=42), full Cox models.
                joblib.dump({"pipeline": pipe, "model": m if model == "Cox" else None, "param": par,
                             "refit": "fit_rsf(X, y, t, param) with model_tuned_survival.fit_rsf (seed 42)" if model == "RSF" else None},
                            os.path.join(OUT, "models", f"{sub.replace('+', 'pos')}_{model}_{fs.replace('+', '_')}{SFX}.joblib"))
                rows.append({"subgroup": sub, "model": model, "feature_set": fs,
                             "n_features": len(cols) + (N_PCS if use_pca else 0),
                             "tuned_param": par, "tcga_cindex": cidx(yte, tte, r)})
            bc, bd = boot_ci(yte, tte, risks, "HCLIN")
            for r_ in rows:
                if r_["subgroup"] == sub and r_["model"] == model:
                    v = np.array(bc[r_["feature_set"]]); r_["ci_low"], r_["ci_high"] = np.percentile(v, [2.5, 97.5])
            for fs, v in bd.items():
                v = np.array(v)
                tests.append({"subgroup": sub, "model": model, "feature_set": fs,
                              "delta": cidx(yte, tte, risks[fs]) - cidx(yte, tte, risks["HCLIN"]),
                              "ci_low": np.percentile(v, 2.5), "ci_high": np.percentile(v, 97.5),
                              "pct_boot_positive": float((v > 0).mean() * 100)})
            pd.DataFrame({"Patient ID": te["Patient ID"], "y": yte, "t": tte,
                          **{f"risk_{k}": v for k, v in risks.items()}}).to_csv(
                os.path.join(OUT, f"tcga_predictions_{sub.replace('+', 'pos')}_{model}{SFX}.csv"), index=False)
        print(sub, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, f"tcga_validation_cindex{SFX}.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(OUT, f"tcga_validation_gain{SFX}.csv"), index=False)
    json.dump(info, open(os.path.join(OUT, f"tcga_validation_cohorts{SFX}.json"), "w"), indent=2)
    print(json.dumps(info, indent=1))
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(tests).round(4).to_string())


if __name__ == "__main__":
    main()
