"""
RSNet pilot, step 7: external check on TCGA-BRCA with direction-B modules (learned on METABRIC).

Usage (from pilot_rsnet/src):
  python 03_external_tcga.py --modules ../rsnet_output/B_METABRIC_top1000_it100/modules.csv \
                             --pool B_METABRIC_top1000 --tag B1000_external

Ridge Cox (penalty tuned by inner CV on METABRIC) fitted on all METABRIC patients of the
population, applied once to TCGA. Clinical block = the study's 6 harmonized features.
Eigengene PCA fitted on METABRIC, applied to TCGA (expression z-scored within each cohort).
GS scores standardized within each cohort, as in the study's external validation.
1,000 paired bootstrap resamples of TCGA for confidence intervals.
"""
import os, sys, json, argparse, warnings
import numpy as np, pandas as pd
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE); ROOT = os.path.dirname(PILOT)
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
from model_compare_shap import load                                        # noqa: E402
from model_tuned_survival import tuned_cox, cidx                            # noqa: E402
from validate_tcga import harmonize_metabric, standardize_within, boot_ci, HCLIN   # noqa: E402

SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", required=True); ap.add_argument("--pool", required=True)
    ap.add_argument("--tag", required=True)
    a = ap.parse_args()
    OUT = os.path.join(PILOT, "results", a.tag); os.makedirs(OUT, exist_ok=True)

    co, DM, HM, _ = load(); GS = DM + HM
    mb = standardize_within(harmonize_metabric(co), GS)
    tc = pd.read_csv(os.path.join(ROOT, "data", "processed", "tcga_brca_features.csv"))
    tc = tc.dropna(subset=["y", "t"] + HCLIN + GS); tc = tc[tc.t > 0]
    Em = pd.read_csv(os.path.join(PILOT, "inputs", "expr_z_METABRIC_pool.csv"), index_col=0)
    Et = pd.read_csv(os.path.join(PILOT, "inputs", "expr_z_TCGA_pool.csv"), index_col=0)
    tc = tc[tc["Patient ID"].isin(Et.index)].reset_index(drop=True)
    tc = standardize_within(tc, GS)
    pool = json.load(open(os.path.join(PILOT, "inputs", "gene_pools.json")))[a.pool]
    m = pd.read_csv(a.modules); m = m[m.gene.isin(pool)]
    mods = {k: list(g.gene) for k, g in m.groupby("module") if len(g) >= 2}

    rows, gains, info = [], [], {}
    for pop in ["All", "ER-"]:
        tr = mb if pop == "All" else mb[mb.er_binary == 0]
        te = tc if pop == "All" else tc[tc.er_binary == 0]
        tr, te = tr.reset_index(drop=True), te.reset_index(drop=True)
        info[pop] = {"train_n": len(tr), "train_events": int(tr.y.sum()), "test_n": len(te), "test_events": int(te.y.sum())}
        if te.y.sum() < 20:
            info[pop]["skipped"] = "fewer than 20 events"; continue
        Atr, Ate = Em.loc[tr["Patient ID"]], Et.loc[te["Patient ID"]]
        Ntr, Nte = [], []
        for k, genes in mods.items():
            p = PCA(1, random_state=SEED).fit(Atr[genes].values)
            s_tr, s_te = p.transform(Atr[genes].values)[:, 0], p.transform(Ate[genes].values)[:, 0]
            if np.corrcoef(s_tr, Atr[genes].values.mean(1))[0, 1] < 0: s_tr, s_te = -s_tr, -s_te
            Ntr.append(s_tr); Nte.append(s_te)
        Ntr, Nte = np.column_stack(Ntr), np.column_stack(Nte)
        C_tr, C_te, G_tr, G_te = tr[HCLIN].values, te[HCLIN].values, tr[GS].values, te[GS].values
        des = {"CLIN+GS": (np.hstack([C_tr, G_tr]), np.hstack([C_te, G_te])),
               "CLIN+NET": (np.hstack([C_tr, Ntr]), np.hstack([C_te, Nte])),
               "CLIN+GS+NET": (np.hstack([C_tr, G_tr, Ntr]), np.hstack([C_te, G_te, Nte]))}
        ytr, ttr, yte, tte = tr.y.values.astype(int), tr.t.values, te.y.values.astype(int), te.t.values
        risks = {}
        for fs, (Xtr, Xte) in des.items():
            r, alpha = tuned_cox(Xtr, ytr, ttr, Xte); risks[fs] = r
            rows.append({"pop": pop, "set": fs, "tcga_cindex": cidx(yte, tte, r), "alpha": alpha})
        bc, bd = boot_ci(yte, tte, risks, "CLIN+GS")
        for r_ in rows:
            if r_["pop"] == pop:
                r_["ci_low"], r_["ci_high"] = np.percentile(bc[r_["set"]], [2.5, 97.5])
        for fs, v in bd.items():
            v = np.array(v)
            gains.append({"pop": pop, "comparison": f"{fs} vs CLIN+GS", "dC": cidx(yte, tte, risks[fs]) - cidx(yte, tte, risks["CLIN+GS"]),
                          "ci_low": np.percentile(v, 2.5), "ci_high": np.percentile(v, 97.5),
                          "pct_boot_positive": float((v > 0).mean() * 100)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "external_cindex.csv"), index=False)
    pd.DataFrame(gains).to_csv(os.path.join(OUT, "external_gain.csv"), index=False)
    json.dump(info, open(os.path.join(OUT, "external_cohorts.json"), "w"), indent=1)
    print(json.dumps(info)); print(pd.DataFrame(rows).round(3).to_string()); print(pd.DataFrame(gains).round(4).to_string())


if __name__ == "__main__":
    main()
