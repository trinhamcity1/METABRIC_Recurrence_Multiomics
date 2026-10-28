"""
External validation, Step 2: train on METABRIC, test on TCGA-BRCA.

Train: METABRIC modelling cohort (complete clinical + genomic), relapse-free survival.
Test : TCGA-BRCA (built by build_tcga_features.py), progression-free interval.

Feature sets (shared by both cohorts):
  HCLIN          6 harmonized clinical features: age, T category, N category,
                 ER status, HER2 (ERBB2) amplification, radiotherapy
  HCLIN+GEN      + 57 gene-set scores + 19 driver features (ERBB2 amp already in HCLIN)
  HCLIN+PC+DRV   + 9 PCs of the gene-set scores (PCA fit on METABRIC only) + 19 drivers

Cross-platform harmonization: continuous molecular features (gene-set scores,
fraction of genome altered, mutation burden) are standardized within each cohort
(microarray vs RNA-seq; targeted panel vs exome). This uses no outcome data.

Models: ridge Cox (penalty tuned by inner 3-fold CV on METABRIC) and random
survival forest (leaf size tuned by out-of-bag C-index on METABRIC). Final models
are fit on all METABRIC patients of the subgroup, then applied once to TCGA.

Uncertainty: 1,000 paired bootstrap resamples of the TCGA test set give 95% CIs
for each C-index and for the gain over HCLIN.
"""
import os, json, warnings
import numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from model_compare_shap import load
from model_tuned_survival import tuned_cox, tuned_rsf, cidx

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "external_tcga"); os.makedirs(OUT, exist_ok=True)
SEED, NBOOT = 42, 1000
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
    df[cols] = (df[cols] - df[cols].mean()) / df[cols].std(ddof=0)
    return df


def boot_ci(y, t, risks, base, n=NBOOT):
    rng = np.random.default_rng(SEED)
    idx_all = np.arange(len(y))
    out = {k: [] for k in risks}; dlt = {k: [] for k in risks if k != base}
    for _ in range(n):
        i = rng.choice(idx_all, len(idx_all), replace=True)
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
    cont = G + ["CNA_fraction_altered", "MUT_nonsilent_count_log"]
    tc = tc.dropna(subset=["y", "t"] + HCLIN + G + DRV); tc = tc[tc.t > 0].reset_index(drop=True)
    mb = standardize_within(mb, cont); tc = standardize_within(tc, cont)

    rows, tests, info = [], [], {}
    for sub in ["ER+", "All"]:
        tr = mb if sub == "All" else mb[mb.er_binary == 1]
        te = tc if sub == "All" else tc[tc.er_binary == 1]
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        info[sub] = {"train_n": len(tr), "train_events": int(tr.y.sum()), "test_n": len(te),
                     "test_events": int(te.y.sum()), "test_median_followup_months": float(te.t.median())}
        sc = StandardScaler().fit(tr[G]); pca = PCA(9, random_state=SEED).fit(sc.transform(tr[G]))
        pcs_tr, pcs_te = pca.transform(sc.transform(tr[G])), pca.transform(sc.transform(te[G]))
        designs = {
            "HCLIN": (tr[HCLIN].values, te[HCLIN].values),
            "HCLIN+GEN": (tr[HCLIN + G + DRV2].values, te[HCLIN + G + DRV2].values),
            "HCLIN+PC+DRV": (np.hstack([tr[HCLIN + DRV2].values, pcs_tr]), np.hstack([te[HCLIN + DRV2].values, pcs_te])),
        }
        ytr, ttr, yte, tte = tr.y.values.astype(int), tr.t.values, te.y.values.astype(int), te.t.values
        for model in ["Cox", "RSF"]:
            risks = {}
            for fs, (Xtr, Xte) in designs.items():
                r, par = (tuned_cox if model == "Cox" else tuned_rsf)(Xtr, ytr, ttr, Xte)
                risks[fs] = r
                rows.append({"subgroup": sub, "model": model, "feature_set": fs, "n_features": Xtr.shape[1],
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
                              "p_boot_two_sided": float(min(1, 2 * min((v <= 0).mean(), (v >= 0).mean()))),
                              "pct_boot_positive": float((v > 0).mean() * 100)})
        print(sub, "done", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "tcga_validation_cindex.csv"), index=False)
    pd.DataFrame(tests).to_csv(os.path.join(OUT, "tcga_validation_gain.csv"), index=False)
    json.dump(info, open(os.path.join(OUT, "tcga_validation_cohorts.json"), "w"), indent=2)
    print(json.dumps(info, indent=1))
    print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(tests).round(4).to_string())


if __name__ == "__main__":
    main()
