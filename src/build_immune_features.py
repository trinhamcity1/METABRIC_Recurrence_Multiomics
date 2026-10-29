"""
Phase 2, Step 2d: immune / stromal cell-population scores (MCP-counter).

Source of marker genes: Becht et al. 2016, Genome Biology (MCP-counter),
file gene_sets/genes.txt from https://github.com/ebecht/MCPcounter (Signatures/).
Each population score = mean expression z-score of its marker genes in that
patient (outcome-free; input z-scores are cohort-normalized upstream, see build_features.py).
These are MCP-counter MARKER-BASED Z-SCORE SUMMARIES, an adaptation of the marker
signatures, not the original MCP-counter abundance estimator. 10 populations:
T cells, CD8 T cells, cytotoxic lymphocytes, NK cells, B lineage,
monocytic lineage, myeloid dendritic cells, neutrophils, endothelial cells, fibroblasts.
"""
import os, re
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "brca_metabric")
OUT = os.path.join(ROOT, "data", "processed")


def read_markers(path):
    g = pd.read_csv(path, sep="\t")
    cols = {c.lower(): c for c in g.columns}
    pop = next(cols[c] for c in cols if "population" in c)
    sym = next((cols[c] for c in cols if "hugo" in c or "symbol" in c), None)
    ent = next((cols[c] for c in cols if "entrez" in c), None)
    return g, pop, sym, ent


def main():
    path = os.path.join(ROOT, "data", "gene_sets", "genes.txt")
    g, pop, sym, ent = read_markers(path)
    z = pd.read_csv(os.path.join(RAW, "data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt"),
                    sep="\t", low_memory=False)
    samples = [c for c in z.columns if c.startswith("MB-")]
    by_sym = z.groupby("Hugo_Symbol")[samples].mean()
    by_ent = z.dropna(subset=["Entrez_Gene_Id"]).assign(Entrez_Gene_Id=lambda d: d.Entrez_Gene_Id.astype(int)) \
              .groupby("Entrez_Gene_Id")[samples].mean()
    feats, meta = {}, []
    for p, grp in g.groupby(pop):
        if ent is not None:
            ids = [int(x) for x in grp[ent].dropna() if str(x).strip().isdigit() and int(x) in by_ent.index]
            M = by_ent.loc[sorted(set(ids))]
        else:
            ids = [s for s in grp[sym].dropna().astype(str) if s in by_sym.index]
            M = by_sym.loc[sorted(set(ids))]
        if len(M) == 0:
            continue
        col = "IMM_" + re.sub(r"[^A-Za-z0-9]+", "_", str(p)).strip("_")
        feats[col] = M.mean(axis=0)
        meta.append(dict(feature=col, source="MCP-counter (Becht 2016)", type="immune/stromal score",
                         genes_in_set=len(grp), genes_used=len(M),
                         description=f"Mean expression z-score of {len(M)} {p} marker genes"))
    feats = pd.DataFrame(feats); feats.index.name = "Patient ID"
    master = pd.read_csv(os.path.join(OUT, "metabric_recurrence_multiomics.csv"))
    master = master.drop(columns=[c for c in master.columns if c.startswith("IMM_")], errors="ignore")
    master.merge(feats, left_on="Patient ID", right_index=True, how="left").to_csv(
        os.path.join(OUT, "metabric_recurrence_multiomics.csv"), index=False)
    d = pd.read_csv(os.path.join(OUT, "feature_dictionary.csv"))
    d = d[~d.feature.str.startswith("IMM_")]
    pd.concat([d, pd.DataFrame(meta)]).to_csv(os.path.join(OUT, "feature_dictionary.csv"), index=False)
    print(pd.DataFrame(meta)[["feature", "genes_in_set", "genes_used"]].to_string(index=False))


if __name__ == "__main__":
    main()
