"""
RSNet pilot, step 2: build network-learning inputs (pre-registered settings).

Outputs (pilot_rsnet/inputs/):
  rsnet_input_A_TCGA_top1000.csv / _top500.csv       direction A: TCGA samples x genes (z-scored)
  rsnet_input_B_METABRIC_top1000.csv / _top500.csv   direction B: METABRIC 1,295 x genes (z-scored)
  expr_z_METABRIC_pool.csv, expr_z_TCGA_pool.csv     z-scored expression of the union of all
                                                     selected genes, for module scoring
  gene_pools.json                                    the gene lists per direction / size
"""
import os, json
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE)
ROOT = os.path.dirname(PILOT)
RAW_M = os.path.join(ROOT, "data", "brca_metabric")
RAW_T = os.path.join(ROOT, "data", "brca_tcga_pan_can_atlas_2018")
OUT = os.path.join(PILOT, "inputs"); os.makedirs(OUT, exist_ok=True)


def zscore(df):  # samples x genes, within cohort
    return (df - df.mean()) / df.std(ddof=0)


def main():
    # METABRIC modelling cohort IDs (the 1,295)
    import sys; sys.path.insert(0, os.path.join(ROOT, "src"))
    from model_compare_shap import load
    co, *_ = load()
    ids_m = list(co["Patient ID"])

    m = pd.read_csv(os.path.join(RAW_M, "data_mrna_illumina_microarray.txt"), sep="\t", low_memory=False)
    m = m.dropna(subset=["Hugo_Symbol"]).drop(columns=["Entrez_Gene_Id"]).groupby("Hugo_Symbol").mean()
    M = m[[c for c in ids_m if c in m.columns]].T.astype(float)           # 1,295 x genes, log2 intensity

    t = pd.read_csv(os.path.join(RAW_T, "data_mrna_seq_v2_rsem.txt"), sep="\t", low_memory=False)
    t = t.dropna(subset=["Hugo_Symbol"]).drop(columns=["Entrez_Gene_Id"]).groupby("Hugo_Symbol").mean()
    tcols = [c for c in t.columns if c.endswith("-01")]                    # primary tumours
    T = np.log2(t[tcols].T.astype(float).clip(lower=0) + 1)               # samples x genes, log2(RSEM+1)
    T.index = [c[:12] for c in T.index]; T = T[~T.index.duplicated()]

    M = M.loc[:, M.notna().all() & (M.var() > 0)]
    T = T.loc[:, T.notna().all() & (T.var() > 0)]
    universe = sorted(set(M.columns) & set(T.columns))
    M, T = M[universe], T[universe]

    pools = {}
    for d, X in [("A_TCGA", T), ("B_METABRIC", M)]:
        order = X.var().sort_values(ascending=False).index
        for p in (1000, 500):
            pools[f"{d}_top{p}"] = list(order[:p])
    union = sorted(set().union(*pools.values()))

    Mz, Tz = zscore(M), zscore(T)
    for d, Xz in [("A_TCGA", Tz), ("B_METABRIC", Mz)]:
        for p in (1000, 500):
            Xz[pools[f"{d}_top{p}"]].round(5).to_csv(os.path.join(OUT, f"rsnet_input_{d}_top{p}.csv"), index_label="sample")
    Mz[union].round(5).to_csv(os.path.join(OUT, "expr_z_METABRIC_pool.csv"), index_label="Patient ID")
    Tz[union].round(5).to_csv(os.path.join(OUT, "expr_z_TCGA_pool.csv"), index_label="Patient ID")
    json.dump(pools, open(os.path.join(OUT, "gene_pools.json"), "w"), indent=1)
    info = {"metabric_samples": M.shape[0], "tcga_samples": T.shape[0], "shared_genes": len(universe),
            "overlap_A1000_B1000": len(set(pools["A_TCGA_top1000"]) & set(pools["B_METABRIC_top1000"])),
            "union_genes_scored": len(union)}
    json.dump(info, open(os.path.join(OUT, "inputs_summary.json"), "w"), indent=1)
    print(info)


if __name__ == "__main__":
    main()
