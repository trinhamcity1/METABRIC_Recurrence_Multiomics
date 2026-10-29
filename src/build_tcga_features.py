"""
Exploratory transport assessment, Step 1: build the TCGA-BRCA feature table with the
same feature FORMULAS as METABRIC, plus harmonized clinical features shared by both
cohorts. Measurement differs between cohorts: RNA-seq vs microarray; exome vs a
173-gene panel (so the non-silent mutation COUNT covers a different gene universe and
is not tumour mutational burden per megabase); ER inferred from ESR1 expression vs IHC;
ERBB2 amplification as a proxy for clinical HER2. Variant classifications not in the
SILENT list are counted as non-silent.

Input : data/brca_tcga_pan_can_atlas_2018/ (cBioPortal "Breast Invasive Carcinoma
        (TCGA, PanCancer Atlas)"), data/gene_sets/
Output: data/processed/tcga_brca_features.csv

Why harmonized clinical features: TCGA does not provide histologic grade, NPI,
positive-node counts, chemotherapy/hormone flags, or receptor IHC in this release.
Both cohorts are therefore described with features they share:
  age, T category (1-3), N category (0-3), ER status, HER2 amplification, radiotherapy.
  - METABRIC T from tumour size (<=20 mm T1, <=50 mm T2, >50 mm T3); TCGA T from
    pathologic T stage (T4 capped at 3, since T4 is defined by invasion, not size).
  - METABRIC N from positive-node count (0, 1-3, 4-9, >=10); TCGA from pathologic N.
  - ER: METABRIC IHC; TCGA from a 2-component Gaussian mixture on log2 ESR1 RNA-seq
    expression (75.5% ER+, 99% of PAM50 luminal tumours called ER+).
  - HER2: ERBB2 high-level amplification (GISTIC +2) in both cohorts.
Outcome: TCGA progression-free interval (PFS_STATUS / PFS_MONTHS), the endpoint
recommended for TCGA by the PanCancer Clinical Data Resource.
Exclusions (TCGA): male patients, M1 (metastatic at diagnosis), neoadjuvant = Yes.
Patients with unknown M stage (MX) or unknown neoadjuvant status are RETAINED and are
not confirmed M0 / untreated; counts are in results/cohort/tcga_exclusions.csv.
Samples: barcodes are truncated to the 12-character patient ID. In this release every
expression, copy-number and mutation sample is a primary tumour (sample type 01) and no
patient has more than one sample per modality (verified; sample map written by
make_cohort_tables.py to results/cohort/tcga_sample_map.csv), so the duplicate guard
never discards a sample.
"""
import os
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "brca_tcga_pan_can_atlas_2018")
GS = os.path.join(ROOT, "data", "gene_sets")
OUT = os.path.join(ROOT, "data", "processed")

CNA_AMP_GENES = ["CCND1", "PAK1", "FGFR1", "ZNF703", "MYC", "RPS6KB1", "PPM1D", "ERBB2", "MDM2"]
CNA_DEL_GENES = ["CDKN2A", "PTEN", "RB1", "MAP2K4"]
MUT_GENES = ["PIK3CA", "TP53", "GATA3", "KMT2C", "MAP3K1", "CDH1", "AKT1", "PTEN"]
SILENT = {"Silent", "Intron", "3'UTR", "5'UTR", "3'Flank", "5'Flank", "RNA", "IGR"}
DM_LABEL = {"AURKA": "proliferation", "ESR1": "ER_signaling", "ERBB2": "HER2_signaling",
            "STAT1": "immune_response", "PLAU": "tumor_invasion", "VEGF": "angiogenesis", "CASP3": "apoptosis"}


def rd(name, **kw):
    return pd.read_csv(os.path.join(RAW, name), sep="\t", comment="#", low_memory=False, **kw)


def pid(sample):
    return sample[:12]


def main():
    # ---------------- clinical + outcome
    p = rd("data_clinical_patient.txt").set_index("PATIENT_ID")
    clin = pd.DataFrame(index=p.index)
    clin["age"] = p["AGE"]
    t = p["PATH_T_STAGE"].astype(str).str.upper().str.extract(r"T(\d)")[0].astype(float)
    clin["T_cat"] = t.clip(upper=3)
    n = p["PATH_N_STAGE"].astype(str).str.upper().str.extract(r"N(\d)")[0].astype(float)
    clin["N_cat"] = n
    clin["radio_rx"] = p["RADIATION_THERAPY"].map({"Yes": 1.0, "No": 0.0})
    clin["y"] = p["PFS_STATUS"].astype(str).str.startswith("1").astype(float).where(p["PFS_STATUS"].notna())
    clin["t"] = p["PFS_MONTHS"]
    clin["pam50"] = p["SUBTYPE"]
    keep = (p["SEX"] == "Female") & (p["PATH_M_STAGE"].astype(str) != "M1") & (p["HISTORY_NEOADJUVANT_TRTYN"] != "Yes")
    clin = clin[keep]

    # ---------------- expression scores + ER call
    z = rd("data_mrna_seq_v2_rsem_zscores_ref_diploid_samples.txt")
    samples = [c for c in z.columns if c.startswith("TCGA-")]
    by_sym = z.groupby("Hugo_Symbol")[samples].mean()
    by_ent = z.dropna(subset=["Entrez_Gene_Id"]).assign(Entrez_Gene_Id=lambda d: d.Entrez_Gene_Id.astype(int)) \
              .groupby("Entrez_Gene_Id")[samples].mean()
    sc = {}
    for line in open(os.path.join(GS, "h.all.v2026.1.Hs.symbols.gmt")):
        parts = line.rstrip("\n").split("\t")
        genes = [g for g in parts[2:] if g in by_sym.index]
        sc["HM_" + parts[0].replace("HALLMARK_", "")] = by_sym.loc[genes].mean(axis=0)
    mods = pd.read_csv(os.path.join(GS, "desmedt2008_modules.csv"))
    for mod, g in mods.groupby("module"):
        g = g.drop_duplicates("EntrezGene.ID"); g = g[g["EntrezGene.ID"].isin(by_ent.index)]
        w = g.set_index("EntrezGene.ID")["coefficient"]
        sc[f"DM_{DM_LABEL[mod]}_{mod}"] = by_ent.loc[w.index].mul(w, axis=0).sum(axis=0) / w.abs().sum()
    expr = pd.DataFrame(sc); expr.index = [pid(s) for s in expr.index]
    expr = expr[~expr.index.duplicated()]

    raw = rd("data_mrna_seq_v2_rsem.txt")
    esr1 = np.log2(raw.loc[raw.Hugo_Symbol == "ESR1"].iloc[0, 2:].astype(float) + 1)
    gm = GaussianMixture(2, random_state=0).fit(esr1.values.reshape(-1, 1))
    er = pd.Series((gm.predict(esr1.values.reshape(-1, 1)) == np.argmax(gm.means_.ravel())).astype(float),
                   index=[pid(s) for s in esr1.index])
    er = er[~er.index.duplicated()]

    # ---------------- copy number
    c = rd("data_cna.txt")
    cs = [x for x in c.columns if x.startswith("TCGA-")]
    c = c.drop_duplicates("Hugo_Symbol").set_index("Hugo_Symbol")[cs]
    cna = pd.DataFrame({f"CNA_{g}_amp": (c.loc[g] == 2).astype(float).where(c.loc[g].notna()) for g in CNA_AMP_GENES})
    dd = c.loc[CNA_DEL_GENES]
    cna["CNA_TSG_deep_del_any"] = (dd == -2).any(axis=0).astype(float).where((dd == -2).any(axis=0) | dd.notna().all(axis=0))  # missing if no deletion seen and a call is missing
    # fraction of assayed genes with a non-zero GISTIC call (missing calls excluded; no NaN in this release)
    cna["CNA_frac_genes_altered"] = (c != 0).astype(float).where(c.notna()).mean(axis=0)
    cna.index = [pid(s) for s in cna.index]; cna = cna[~cna.index.duplicated()]

    # ---------------- mutations
    m = rd("data_mutations.txt")
    seq = []
    for line in open(os.path.join(RAW, "case_lists", "cases_sequenced.txt")):
        if line.startswith("case_list_ids:"):
            seq = [s.strip() for s in line.split(":", 1)[1].strip().split("\t") if s.strip()]
    ns = m[~m.Variant_Classification.isin(SILENT)].copy()
    ns["pid"] = ns.Tumor_Sample_Barcode.str[:12]
    mut = pd.DataFrame(index=pd.Index(sorted({pid(s) for s in seq})))
    for g in MUT_GENES:
        mut[f"MUT_{g}"] = mut.index.isin(set(ns.loc[ns.Hugo_Symbol == g, "pid"])).astype(float)
    mut["MUT_nonsilent_count_log"] = np.log1p(mut.index.map(ns.groupby("pid").size()).fillna(0).astype(float))

    d = clin.join(er.rename("er_binary")).join(expr).join(cna).join(mut)
    d["her2_amp"] = d["CNA_ERBB2_amp"]
    d.index.name = "Patient ID"
    d.to_csv(os.path.join(OUT, "tcga_brca_features.csv"))
    gen = [x for x in d.columns if x.startswith(("HM_", "DM_", "CNA_", "MUT_"))]
    comp = d.dropna(subset=["y", "t", "age", "T_cat", "N_cat", "er_binary", "radio_rx"] + gen)
    comp = comp[comp.t > 0]
    print("TCGA patients after exclusions:", len(d), "| complete for validation:", len(comp),
          "| events:", int(comp.y.sum()), "| ER+:", int(comp.er_binary.sum()),
          "ER+ events:", int(comp.loc[comp.er_binary == 1, "y"].sum()),
          "| median follow-up (months):", round(comp.t.median(), 1))


if __name__ == "__main__":
    main()
