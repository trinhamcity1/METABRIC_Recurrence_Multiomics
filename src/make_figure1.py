"""
Manuscript Figure 1: gain in C-index from adding genomic features, built from the
saved result tables (no numbers are typed in by hand).
Inputs : results/phase2/er_late_tests.csv, tuned_survival_tests.csv,
         results/external_tcga/metabric_internal_harmonized_cox.csv, tcga_validation_gain.csv
Output : results/figures/figure1_gain_forest.png
"""
import os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
P2, EX = os.path.join(ROOT, "results", "phase2"), os.path.join(ROOT, "results", "external_tcga")
OUT = os.path.join(ROOT, "results", "figures"); os.makedirs(OUT, exist_ok=True)


def main():
    el = pd.read_csv(os.path.join(P2, "er_late_tests.csv"))
    tu = pd.read_csv(os.path.join(P2, "tuned_survival_tests.csv"))
    ih = pd.read_csv(os.path.join(EX, "metabric_internal_harmonized_cox.csv"))
    tg = pd.read_csv(os.path.join(EX, "tcga_validation_gain.csv"))
    g = lambda d, **k: d.loc[np.logical_and.reduce([d[a] == b for a, b in k.items()])].iloc[0]
    rows = []
    def add(label, r, kind): rows.append((label, r["delta"], r["ci_low"], r["ci_high"], kind))
    add("All patients · Cox · BASE+GEN (25 splits)", g(el, analysis="All patients", comparison="BASE+GEN vs BASE"), "int")
    add("ER+ · Cox · BASE+GEN (25 splits)", g(el, analysis="ER+ only", comparison="BASE+GEN vs BASE"), "int")
    add("Landmark >5 y, ER+ · Cox · BASE+GEN", g(el, analysis="Late (>5y), ER+", comparison="BASE+GEN vs BASE"), "int")
    add("ER+ · tuned Cox · BASE+GEN (50 splits)", g(tu, cohort="ERpos", model="cox", comparison="BASE+GEN vs BASE"), "int")
    add("ER+ · tuned RSF · BASE+PC+DRV (50 splits)", g(tu, cohort="ERpos", model="rsf", comparison="BASE+PC+DRV vs BASE"), "int")
    add("ER+ · tuned RSF · CLIN+PC+DRV vs CLIN", g(tu, cohort="ERpos", model="rsf", comparison="CLIN+PC+DRV vs CLIN"), "int")
    add("ER+ · Cox · harmonized HCLIN+GEN", g(ih, subgroup="ER+"), "int")
    add("TCGA ESR1-inferred ER+ · Cox · +GEN", g(tg, subgroup="ER+", model="Cox", feature_set="HCLIN+GEN"), "ext")
    add("TCGA ESR1-inferred ER+ · RSF · +GEN", g(tg, subgroup="ER+", model="RSF", feature_set="HCLIN+GEN"), "ext")
    add("TCGA ESR1-inferred ER+ · RSF · +PC+DRV", g(tg, subgroup="ER+", model="RSF", feature_set="HCLIN+PC+DRV"), "ext")
    add("TCGA all · Cox · +GEN", g(tg, subgroup="All", model="Cox", feature_set="HCLIN+GEN"), "ext")

    fmt = lambda v: f"{v:+.3f}".replace("-", "−") if abs(v) >= 0.0005 else "0.000"
    fig, ax = plt.subplots(figsize=(10.2, 4.6), dpi=200)
    n = len(rows)
    for i, (lab, d, lo, hi, kind) in enumerate(rows):
        y = n - 1 - i
        col, mk = ("#2E75D6", "o") if kind == "int" else ("#E8683A", "s")
        ax.plot([lo, hi], [y, y], color=col, lw=2.2, solid_capstyle="round")
        ax.plot(d, y, mk, color=col, ms=7, mec="white", mew=1)
        ax.text(1.01, y, f"{fmt(d)} ({fmt(lo)} to {fmt(hi)})", transform=ax.get_yaxis_transform(), va="center", fontsize=8.5)
    nint = sum(r[4] == "int" for r in rows)
    ax.axhline(n - nint - 0.5, color="#cccccc", lw=0.8)
    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks(range(n)); ax.set_yticklabels([r[0] for r in rows][::-1], fontsize=8.5)
    ax.set_xlim(-0.14, 0.15); ax.set_xlabel("Change in C-index vs matching baseline (95% CI)", fontsize=9)
    ax.grid(axis="x", color="#eeeeee"); ax.set_axisbelow(True)
    for s in ["top", "right", "left"]: ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    h1, = ax.plot([], [], "o-", color="#2E75D6", label="METABRIC internal CV (corrected CI)")
    h2, = ax.plot([], [], "s-", color="#E8683A", label="TCGA-BRCA exploratory transport (bootstrap CI)")
    ax.legend(handles=[h1, h2], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, fontsize=8.5)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figure1_gain_forest.png"), bbox_inches="tight")
    pd.DataFrame(rows, columns=["label", "delta", "ci_low", "ci_high", "kind"]).to_csv(os.path.join(OUT, "figure1_data.csv"), index=False)
    print(pd.DataFrame(rows).round(4).to_string())


if __name__ == "__main__":
    main()
