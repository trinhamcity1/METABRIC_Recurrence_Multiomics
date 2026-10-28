"""
Phase 2, Step 2b: PCA of the 57 expression scores (descriptive, unsupervised).

PCA never sees the recurrence outcome, so this tells us about REDUNDANCY between
gene sets, not about which ones predict recurrence. For modelling, PCA is refit
inside each training fold (see model_compare_shap.py).
"""
import os
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, fcluster, leaves_list
from scipy.spatial.distance import squareform

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "phase2"); os.makedirs(OUT, exist_ok=True)

d = pd.read_csv(os.path.join(ROOT, "data", "processed", "metabric_recurrence_multiomics.csv"))
cols = [c for c in d.columns if c.startswith(("HM_", "DM_"))]
X = d[cols].dropna()
Z = StandardScaler().fit_transform(X)

pca = PCA().fit(Z)
ev = pca.explained_variance_ratio_
cum = np.cumsum(ev)
n80, n90 = int(np.searchsorted(cum, 0.80) + 1), int(np.searchsorted(cum, 0.90) + 1)
load = pd.DataFrame(pca.components_[:10].T, index=cols, columns=[f"PC{i+1}" for i in range(10)])
load.to_csv(os.path.join(OUT, "pca_loadings_top10.csv"))
pd.DataFrame({"PC": np.arange(1, len(ev) + 1), "explained": ev, "cumulative": cum}).to_csv(
    os.path.join(OUT, "pca_explained_variance.csv"), index=False)

# redundancy clusters: hierarchical clustering on 1 - |r|
r = X.corr()
dist = 1 - r.abs().values
np.fill_diagonal(dist, 0)
L = linkage(squareform(dist, checks=False), method="average")
cl = fcluster(L, t=0.3, criterion="distance")   # sets with |r| >= ~0.7 group together
clusters = pd.DataFrame({"feature": cols, "cluster": cl}).sort_values("cluster")
clusters.to_csv(os.path.join(OUT, "redundancy_clusters.csv"), index=False)

summary = [f"n patients = {len(X)}, n scores = {len(cols)}",
           f"PCs for 80% variance: {n80}; for 90%: {n90}",
           f"PC1 explains {ev[0]:.1%}, PC2 {ev[1]:.1%}, PC3 {ev[2]:.1%}",
           f"redundancy clusters (|r| >= ~0.7): {clusters.cluster.nunique()} groups from {len(cols)} scores"]
for k in range(3):
    top = load[f"PC{k+1}"].abs().sort_values(ascending=False).head(6).index
    summary.append(f"PC{k+1} top: " + ", ".join(f"{t}({load.loc[t, f'PC{k+1}']:+.2f})" for t in top))
big = clusters.groupby("cluster").feature.apply(list)
for c, feats in big.items():
    if len(feats) > 1:
        summary.append(f"cluster {c} ({len(feats)}): " + ", ".join(feats))
open(os.path.join(OUT, "pca_summary.txt"), "w").write("\n".join(summary))
print("\n".join(summary))

# figure
order = leaves_list(L)
fig, ax = plt.subplots(1, 2, figsize=(15, 6.8), gridspec_kw={"width_ratios": [1, 1.6]})
ax[0].bar(range(1, 21), ev[:20] * 100, color="#1A2B4A")
ax[0].plot(range(1, 21), cum[:20] * 100, color="#E8749A", marker="o", ms=3)
ax[0].axhline(80, ls="--", c="grey", lw=1); ax[0].axhline(90, ls=":", c="grey", lw=1)
ax[0].set_xlabel("Principal component"); ax[0].set_ylabel("% variance explained")
ax[0].set_title(f"A. 57 scores compress to {n80} PCs (80%) / {n90} PCs (90%)", fontsize=10)
lab = [cols[i].replace("HM_", "").replace("DM_", "D:")[:28] for i in order]
im = ax[1].imshow(r.values[np.ix_(order, order)], cmap="RdBu_r", vmin=-1, vmax=1)
ax[1].set_xticks(range(len(cols))); ax[1].set_yticks(range(len(cols)))
ax[1].set_xticklabels(lab, rotation=90, fontsize=5); ax[1].set_yticklabels(lab, fontsize=5)
ax[1].set_title("B. Correlation between gene-set scores (clustered)", fontsize=10)
plt.colorbar(im, ax=ax[1], fraction=0.03)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig_pca_redundancy.png"), dpi=170)
