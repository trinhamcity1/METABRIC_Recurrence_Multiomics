"""
Phase 2, Step 2: build the clinical + genomic feature table for METABRIC.

Inputs (all inside this project folder):
  data/Breast_Cancer_METABRIC.csv                     main clinical table (2,509 patients)
  brca_metabric/data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt
  brca_metabric/data_cna.txt
  brca_metabric/data_mutations.txt
  brca_metabric/case_lists/cases_sequenced.txt
  gene_sets/h.all.v2026.1.Hs.symbols.gmt              MSigDB Hallmark (50 sets)
  gene_sets/desmedt2008_modules.csv                   Desmedt 2008 breast modules (7)

Output:
  data/processed/metabric_recurrence_multiomics.csv   one row per patient, joined on Patient ID
  data/processed/feature_dictionary.csv               what every new column means

All gene-set scores are computed per patient from that patient's own values and a
fixed, published gene list, so no information from other patients (or outcomes)
is used. That keeps the scores leakage-free before cross-validation.
"""
import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "brca_metabric")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)

CNA_AMP_GENES = ["CCND1", "PAK1", "FGFR1", "ZNF703", "MYC", "RPS6KB1", "PPM1D", "ERBB2", "MDM2"]
CNA_DEL_GENES = ["CDKN2A", "PTEN", "RB1", "MAP2K4"]
MUT_GENES = ["PIK3CA", "TP53", "GATA3", "KMT2C", "MAP3K1", "CDH1", "AKT1", "PTEN"]
SILENT = {"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "RNA", "IGR"}


def read_cbio(path, **kw):
    return pd.read_csv(path, sep="\t", comment="#", low_memory=False, **kw)


def expression_scores():
    z = read_cbio(os.path.join(RAW, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"))
    samples = [c for c in z.columns if c.startswith("MB-")]
    # average duplicate probes of the same gene
    by_sym = z.groupby("Hugo_Symbol")[samples].mean()
    by_ent = z.dropna(subset=["Entrez_Gene_Id"]).assign(
        Entrez_Gene_Id=lambda d: d.Entrez_Gene_Id.astype(int)).groupby("Entrez_Gene_Id")[samples].mean()

    scores, meta = {}, []
    # Hallmark: mean z-score of member genes present in METABRIC
    for line in open(os.path.join(ROOT, "data", "gene_sets", "h.all.v2026.1.Hs.symbols.gmt")):
        parts = line.rstrip("\n").split("\t")
        name, genes = parts[0], [g for g in parts[2:] if g]
        present = [g for g in genes if g in by_sym.index]
        col = "HM_" + name.replace("HALLMARK_", "")
        scores[col] = by_sym.loc[present].mean(axis=0)
        meta.append({"feature": col, "source": "MSigDB Hallmark", "type": "expression score",
                     "genes_in_set": len(genes), "genes_used": len(present),
                     "description": f"Mean expression z-score of {len(present)} {name} genes"})
    # Desmedt 2008: weighted score sum(w*z)/sum(|w|) (genefu sig.score style)
    mods = pd.read_csv(os.path.join(ROOT, "data", "gene_sets", "desmedt2008_modules.csv"))
    label = {"AURKA": "proliferation", "ESR1": "ER_signaling", "ERBB2": "HER2_signaling",
             "STAT1": "immune_response", "PLAU": "tumor_invasion", "VEGF": "angiogenesis",
             "CASP3": "apoptosis"}
    for mod, g in mods.groupby("module"):
        g = g.drop_duplicates("EntrezGene.ID")
        g = g[g["EntrezGene.ID"].isin(by_ent.index)]
        w = g.set_index("EntrezGene.ID")["coefficient"]
        col = f"DM_{label[mod]}_{mod}"
        scores[col] = (by_ent.loc[w.index].mul(w, axis=0)).sum(axis=0) / w.abs().sum()
        meta.append({"feature": col, "source": "Desmedt et al. 2008", "type": "expression score",
                     "genes_in_set": int((mods.module == mod).sum()), "genes_used": len(w),
                     "description": f"Weighted {label[mod].replace('_', ' ')} module score (prototype {mod})"})
    df = pd.DataFrame(scores)
    df.index.name = "Patient ID"
    return df, meta


def cna_features():
    c = read_cbio(os.path.join(RAW, "data_cna.txt"))
    samples = [x for x in c.columns if x.startswith("MB-")]
    c = c.drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")[samples]
    out, meta = {}, []
    for g in CNA_AMP_GENES:
        out[f"CNA_{g}_amp"] = (c.loc[g] == 2).astype(float)
        meta.append({"feature": f"CNA_{g}_amp", "source": "cBioPortal CNA (GISTIC)", "type": "copy number",
                     "description": f"1 if high-level amplification of {g} (value +2)"})
    out["CNA_TSG_deep_del_any"] = (c.loc[CNA_DEL_GENES] == -2).any(axis=0).astype(float)
    meta.append({"feature": "CNA_TSG_deep_del_any", "source": "cBioPortal CNA (GISTIC)", "type": "copy number",
                 "description": "1 if deep deletion (-2) of any of " + ", ".join(CNA_DEL_GENES)})
    out["CNA_fraction_altered"] = (c != 0).mean(axis=0)
    meta.append({"feature": "CNA_fraction_altered", "source": "cBioPortal CNA (GISTIC)", "type": "copy number",
                 "description": "Fraction of genes with any copy-number change (genome instability)"})
    df = pd.DataFrame(out); df.index.name = "Patient ID"
    return df, meta


def mutation_features():
    m = read_cbio(os.path.join(RAW, "data_mutations.txt"))
    seq = []
    for line in open(os.path.join(RAW, "case_lists", "cases_sequenced.txt")):
        if line.startswith("case_list_ids:"):
            seq = line.split(":", 1)[1].strip().split("\t")
    seq = [s.strip() for s in seq if s.strip()]
    nonsilent = m[~m.Variant_Classification.isin(SILENT)]
    out = pd.DataFrame(index=pd.Index(seq, name="Patient ID"))
    meta = []
    for g in MUT_GENES:
        hit = set(nonsilent.loc[nonsilent.Hugo_Symbol == g, "Tumor_Sample_Barcode"])
        out[f"MUT_{g}"] = out.index.isin(hit).astype(float)
        meta.append({"feature": f"MUT_{g}", "source": "cBioPortal mutations (173-gene panel)", "type": "mutation",
                     "description": f"1 if any protein-changing (non-silent) {g} mutation"})
    counts = nonsilent.groupby("Tumor_Sample_Barcode").size()
    out["MUT_nonsilent_count_log"] = np.log1p(out.index.map(counts).fillna(0).astype(float))
    meta.append({"feature": "MUT_nonsilent_count_log", "source": "cBioPortal mutations (173-gene panel)",
                 "type": "mutation", "description": "log(1 + number of protein-changing mutations in the panel)"})
    return out, meta


def main():
    clin = pd.read_csv(os.path.join(ROOT, "data", "Breast_Cancer_METABRIC.csv"))
    clin.columns = [c.strip() for c in clin.columns]
    expr, m1 = expression_scores()
    cna, m2 = cna_features()
    mut, m3 = mutation_features()
    master = (clin.merge(expr, left_on="Patient ID", right_index=True, how="left")
                  .merge(cna, left_on="Patient ID", right_index=True, how="left")
                  .merge(mut, left_on="Patient ID", right_index=True, how="left"))
    assert len(master) == len(clin) == master["Patient ID"].nunique()
    master.to_csv(os.path.join(OUT, "metabric_recurrence_multiomics.csv"), index=False)
    pd.DataFrame(m1 + m2 + m3).to_csv(os.path.join(OUT, "feature_dictionary.csv"), index=False)
    print("master:", master.shape)
    print("patients with expression:", master[expr.columns[0]].notna().sum(),
          "| CNA:", master[cna.columns[0]].notna().sum(), "| sequenced:", master[mut.columns[0]].notna().sum(),
          "| all three:", master[[expr.columns[0], cna.columns[0], mut.columns[0]]].notna().all(axis=1).sum())
    print("new columns:", len(expr.columns), "expression,", len(cna.columns), "CNA,", len(mut.columns), "mutation")


if __name__ == "__main__":
    main()
