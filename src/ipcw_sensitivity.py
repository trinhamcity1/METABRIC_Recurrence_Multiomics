"""
Censoring-sensitive sensitivity analysis: Uno's IPCW concordance (truncated at tau)
recomputed from the SAVED predictions (no refitting).

Internal (METABRIC, tuned survival models, 50 splits): for each outer split the
censoring distribution is estimated by Kaplan-Meier on that split's TRAINING patients;
tau = 120 months (median reverse-KM follow-up is ~160 months).
External (TCGA, harmonized models): censoring distribution estimated on the TCGA set;
tau = 36 months (median reverse-KM follow-up ~29 months; 36 months keeps the
censoring-weight estimate supported).
Paired differences use the same corrected repeated t-test (internal) or the same
1,000 paired bootstrap resamples (external) as the Harrell analyses.
Output: results/phase2/ipcw_internal.csv, results/external_tcga/ipcw_tcga.csv
"""
import os, glob, json
import numpy as np, pandas as pd
from sksurv.metrics import concordance_index_ipcw
from sksurv.util import Surv
from model_compare_shap import load, corrected_repeated_t, holm

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P2, EX = os.path.join(ROOT, "results", "phase2"), os.path.join(ROOT, "results", "external_tcga")
TAU_INT, TAU_EXT, SEED, NBOOT = 120.0, 36.0, 42, 1000


def uno(ytr, ttr, yte, tte, risk, tau):
    return concordance_index_ipcw(Surv.from_arrays(ytr.astype(bool), ttr),
                                  Surv.from_arrays(yte.astype(bool), tte), risk, tau=tau)[0]


def main():
    co, *_ = load()
    pr = pd.read_csv(os.path.join(P2, "tuned_survival_oof_predictions.csv.gz"))
    cohorts = {"ERpos": co[co.er_binary == 1], "all": co}
    rows = []
    for coh, df in cohorts.items():
        df = df.set_index("Patient ID")
        for (s, fs), g in pr[pr.cohort == coh].groupby(["split", "feature_set"]):
            te = g["Patient ID"].values; trn = df.drop(index=te)
            yte, tte = df.loc[te, "y"].values, df.loc[te, "t"].values
            for model in ["cox", "rsf"]:
                rows.append({"cohort": coh, "split": s, "feature_set": fs, "model": model,
                             "uno_c": uno(trn.y.values, trn.t.values, yte, tte, g[f"{model}_risk"].values, TAU_INT)})
    r = pd.DataFrame(rows)
    tests = []
    for (coh, model), g in r.groupby(["cohort", "model"]):
        fam = []
        for base, fss in [("BASE", ["BASE+GEN", "BASE+PC+DRV"]), ("CLIN", ["CLIN+GEN", "CLIN+PC+DRV"])]:
            bv = g[g.feature_set == base].sort_values("split").uno_c.values
            for fs in fss:
                v = g[g.feature_set == fs].sort_values("split").uno_c.values
                m, lo, hi, p = corrected_repeated_t(v - bv, k=5, r=10)
                fam.append({"cohort": coh, "model": model, "comparison": f"{fs} vs {base}", "tau_months": TAU_INT,
                            "uno_base": bv.mean(), "uno_new": v.mean(), "delta": m, "ci_low": lo, "ci_high": hi,
                            "p_raw": p, "splits_improved": int((v > bv).sum())})
        for x, a in zip(fam, holm([x["p_raw"] for x in fam])): x["p_holm"] = a
        tests += fam
    pd.DataFrame(tests).to_csv(os.path.join(P2, "ipcw_internal.csv"), index=False)
    print(pd.DataFrame(tests).round(4).to_string())

    ext = []
    for sub in ["ERpos", "All"]:
        for model in ["Cox", "RSF"]:
            f = os.path.join(EX, f"tcga_predictions_{sub}_{model}.csv")
            if not os.path.exists(f): continue
            d = pd.read_csv(f); y, t = d.y.values, d.t.values
            rng = np.random.default_rng(SEED)
            base = d["risk_HCLIN"].values
            for fs in ["HCLIN+GEN", "HCLIN+PC+DRV"]:
                rk = d[f"risk_{fs}"].values
                c0, c1 = uno(y, t, y, t, base, TAU_EXT), uno(y, t, y, t, rk, TAU_EXT)
                bd = []
                for _ in range(NBOOT):
                    i = rng.choice(len(y), len(y), replace=True)
                    if y[i].sum() < 5: continue
                    try:
                        bd.append(uno(y[i], t[i], y[i], t[i], rk[i], TAU_EXT) - uno(y[i], t[i], y[i], t[i], base[i], TAU_EXT))
                    except ValueError:
                        continue
                bd = np.array(bd)
                ext.append({"subgroup": sub, "model": model, "feature_set": fs, "tau_months": TAU_EXT,
                            "uno_base": c0, "uno_new": c1, "delta": c1 - c0,
                            "ci_low": np.percentile(bd, 2.5), "ci_high": np.percentile(bd, 97.5), "n_boot": len(bd)})
    pd.DataFrame(ext).to_csv(os.path.join(EX, "ipcw_tcga.csv"), index=False)
    print(pd.DataFrame(ext).round(4).to_string())


if __name__ == "__main__":
    main()
