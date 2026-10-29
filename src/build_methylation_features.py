"""
Phase 2, Step 2c: DNA methylation features (RRBS promoter methylation, METABRIC).

Source: brca_metabric/data_methylation_promoters_rrbs.txt
  13,188 gene promoters x 1,418 tumours, beta values 0 (unmethylated) - 1 (methylated).
  No normal-tissue columns are included, so instability is measured WITHIN each
  tumour (no reference built from other patients).

Features (one row per patient, joined to the master table on Patient ID):
  METH_mean             mean promoter methylation (global level)
  METH_sd               spread of promoter methylation within the tumour
  METH_hyper_frac       fraction of promoters with beta > 0.5 (strongly methylated);
                        cBioPortal picked, for each gene, the promoter LEAST methylated in
                        normal breast, so high values largely reflect tumour-acquired gain
  METH_intermediate_frac fraction of promoters with 0.2 < beta < 0.8; intermediate values
                        reflect mixed / unstable methylation across cells (epigenomic disorder)
  METH_DM_<module>      mean promoter methylation of each Desmedt module's genes
  METH_<gene>           promoter methylation of selected breast-cancer genes
Motivation: Batra et al. 2021 (Nat Commun) found epigenomic instability in METABRIC
correlates with grade, stage, TP53 mutation and poorer prognosis.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "brca_metabric")
OUT = os.path.join(ROOT, "data", "processed")
KEY_GENES = ["ESR1", "PGR", "GATA3", "FOXA1", "CDH1", "BRCA1", "RASSF1", "CDKN2A", "GSTP1", "PTEN"]


def main():
    m = pd.read_csv(os.path.join(RAW, "data_methylation_promoters_rrbs.txt"), sep="\t", low_memory=False)
    m = m.groupby("Hugo_Symbol").mean(numeric_only=True)          # 4 duplicated symbols -> average
    samples = [c for c in m.columns if c.startswith("MB-")]
    B = m[samples]
    feats = pd.DataFrame(index=pd.Index(samples, name="Patient ID"))
    feats["METH_mean"] = B.mean(axis=0)
    feats["METH_sd"] = B.std(axis=0)
    feats["METH_hyper_frac"] = (B > 0.5).sum(axis=0) / B.notna().sum(axis=0)
    feats["METH_intermediate_frac"] = ((B > 0.2) & (B < 0.8)).sum(axis=0) / B.notna().sum(axis=0)

    # Desmedt modules: map Entrez -> symbol via the expression file
    z = pd.read_csv(os.path.join(RAW, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"),
                    sep="\t", usecols=[0, 1]).dropna()
    ent2sym = dict(zip(z.Entrez_Gene_Id.astype(int), z.Hugo_Symbol))
    mods = pd.read_csv(os.path.join(ROOT, "data", "gene_sets", "desmedt2008_modules.csv"))
    label = {"AURKA": "proliferation", "ESR1": "ER_signaling", "ERBB2": "HER2_signaling", "STAT1": "immune_response",
             "PLAU": "tumor_invasion", "VEGF": "angiogenesis", "CASP3": "apoptosis"}
    meta = [dict(feature=c, source="METABRIC RRBS promoter methylation", type="methylation") for c in feats.columns]
    for mod, g in mods.groupby("module"):
        genes = [ent2sym.get(int(e)) for e in g["EntrezGene.ID"]]
        genes = [x for x in genes if x in B.index]
        col = f"METH_DM_{label[mod]}"
        feats[col] = B.loc[genes].mean(axis=0)
        meta.append(dict(feature=col, source="METABRIC RRBS + Desmedt 2008", type="methylation",
                         genes_used=len(genes)))
    for gname in KEY_GENES:
        if gname in B.index:
            feats[f"METH_{gname}"] = B.loc[gname]
            meta.append(dict(feature=f"METH_{gname}", source="METABRIC RRBS promoter methylation", type="methylation"))

    master = pd.read_csv(os.path.join(OUT, "metabric_recurrence_multiomics.csv"))
    master = master.drop(columns=[c for c in master.columns if c.startswith("METH_")], errors="ignore")
    master = master.merge(feats, left_on="Patient ID", right_index=True, how="left")
    master.to_csv(os.path.join(OUT, "metabric_recurrence_multiomics.csv"), index=False)
    d = pd.read_csv(os.path.join(OUT, "feature_dictionary.csv"))
    d = d[~d.feature.str.startswith("METH_")]
    pd.concat([d, pd.DataFrame(meta)]).to_csv(os.path.join(OUT, "feature_dictionary.csv"), index=False)
    print("methylation features:", feats.shape[1], "| patients:", len(feats))
    print(feats.describe().T[["mean", "std", "min", "max"]].round(3).to_string())


if __name__ == "__main__":
    main()
