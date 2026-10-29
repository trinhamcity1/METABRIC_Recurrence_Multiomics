"""
RSNet pilot, steps 4-6 and 8: cross-validated comparison in METABRIC (pre-registered).

Usage (from pilot_rsnet/src):
  python 02_evaluate_cv.py --modules ../rsnet_output/A_TCGA_top1000_it100/modules.csv \
                           --degree  ../rsnet_output/A_TCGA_top1000_it100/degree.csv \
                           --pool A_TCGA_top1000 --tag A1000_eig [--scoring eigengene|meanz]
  Re-run until it prints ALL SPLITS DONE (checkpointed), then add --summarize.

Design (unchanged from the study's tuned analysis): same 1,295-patient cohort, RFS endpoint,
RepeatedStratifiedKFold(5 x 10, seed 42) within each population (All, ER-, ER+), ridge Cox with
penalty tuned by inner 3-fold CV (primary), RSF with leaf tuned by OOB (secondary), corrected
repeated k-fold t-test, Holm across the 6 pre-registered primary comparisons.
"""
import os, sys, json, glob, time, argparse, warnings
import numpy as np, pandas as pd
from scipy.stats import hypergeom
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.util import Surv
from sksurv.metrics import cumulative_dynamic_auc

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE); ROOT = os.path.dirname(PILOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
from model_compare_shap import load, CLIN, corrected_repeated_t, holm   # noqa: E402
from model_tuned_survival import tuned_rsf, cidx                          # noqa: E402

SEED, K, R, N_RAND = 42, 5, 10, 20
ALPHAS = [1.0, 10.0, 100.0, 1000.0]
POPS = ["All", "ER-", "ER+"]
SETS = ["CLIN", "CLIN+GS", "CLIN+NET", "CLIN+GS+NET", "CLIN+RAND"]


# ------------------------------------------------------------------ modules
def read_modules(path, pool_genes):
    m = pd.read_csv(path)
    m = m[m.gene.isin(pool_genes)]
    mods = {k: list(g.gene) for k, g in m.groupby("module")}
    return {k: v for k, v in mods.items() if len(v) >= 2}


def random_modules(sizes, pool, seed):
    rng = np.random.default_rng(seed)
    genes = list(rng.permutation(pool)); out, i = {}, 0
    for j, s in enumerate(sizes):
        out[f"R{j+1}"] = genes[i:i + s]; i += s
    return out


def score_modules(E, mods, tr, te, scoring):
    """Return train/test score matrices; PCA fitted on training rows only."""
    Str, Ste = [], []
    for k, genes in mods.items():
        A = E[genes].values
        if scoring == "meanz":
            Str.append(A[tr].mean(1)); Ste.append(A[te].mean(1)); continue
        p = PCA(1, random_state=SEED).fit(A[tr])
        s_tr, s_te = p.transform(A[tr])[:, 0], p.transform(A[te])[:, 0]
        if np.corrcoef(s_tr, A[tr].mean(1))[0, 1] < 0:
            s_tr, s_te = -s_tr, -s_te
        Str.append(s_tr); Ste.append(s_te)
    return np.column_stack(Str), np.column_stack(Ste)


# ------------------------------------------------------------------ models
def tuned_cox_coef(Xtr, ytr, ttr, Xte):
    inner = StratifiedKFold(3, shuffle=True, random_state=SEED)
    sc = {a: [] for a in ALPHAS}
    for itr, iva in inner.split(Xtr, ytr):
        s = StandardScaler().fit(Xtr[itr])
        for a in ALPHAS:
            m = CoxPHSurvivalAnalysis(alpha=a, n_iter=200).fit(s.transform(Xtr[itr]), Surv.from_arrays(ytr[itr].astype(bool), ttr[itr]))
            sc[a].append(cidx(ytr[iva], ttr[iva], m.predict(s.transform(Xtr[iva]))))
    best = max(ALPHAS, key=lambda a: np.mean(sc[a]))
    s = StandardScaler().fit(Xtr)
    m = CoxPHSurvivalAnalysis(alpha=best, n_iter=200).fit(s.transform(Xtr), Surv.from_arrays(ytr.astype(bool), ttr))
    return m.predict(s.transform(Xte)), best, m.coef_


def td_auc60(ytr, ttr, yte, tte, risk):
    try:
        a, _ = cumulative_dynamic_auc(Surv.from_arrays(ytr.astype(bool), ttr), Surv.from_arrays(yte.astype(bool), tte), risk, [60.0])
        return float(a[0])
    except Exception:
        return np.nan


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", required=True); ap.add_argument("--degree", default=None)
    ap.add_argument("--pool", required=True); ap.add_argument("--tag", required=True)
    ap.add_argument("--scoring", default="eigengene", choices=["eigengene", "meanz"])
    ap.add_argument("--maxsec", type=float, default=float(os.environ.get("MAXSEC", 140)))
    ap.add_argument("--summarize", action="store_true"); ap.add_argument("--no-rsf", action="store_true")
    ap.add_argument("--pops", default="All,ER-,ER+", help="subset of populations to run (for parallel chunks)")
    a = ap.parse_args()
    OUT = os.path.join(PILOT, "results", a.tag); CK = os.path.join(OUT, "ck"); os.makedirs(CK, exist_ok=True)

    co, DM, HM, _ = load()
    GS = DM + HM
    pools = json.load(open(os.path.join(PILOT, "inputs", "gene_pools.json")))
    pool = pools[a.pool]
    E = pd.read_csv(os.path.join(PILOT, "inputs", "expr_z_METABRIC_pool.csv"), index_col=0)
    E = E.loc[co["Patient ID"]].reset_index(drop=True)
    mods = read_modules(a.modules, pool)
    sizes = [len(v) for v in mods.values()]
    rand_sets = [random_modules(sizes, pool, seed) for seed in range(1, N_RAND + 1)]
    json.dump({"modules": {k: v for k, v in mods.items()}, "sizes": sizes, "pool": a.pool, "scoring": a.scoring},
              open(os.path.join(OUT, "modules_used.json"), "w"), indent=1)

    t0 = time.time()
    for pop in [p for p in POPS if p in a.pops.split(",")]:
        idx = {"All": co.index, "ER-": co.index[co.er_binary == 0], "ER+": co.index[co.er_binary == 1]}[pop]
        d = co.loc[idx].reset_index(drop=True); Ep = E.loc[idx].reset_index(drop=True)
        y, t = d.y.values.astype(int), d.t.values
        splits = list(RepeatedStratifiedKFold(n_splits=K, n_repeats=R, random_state=SEED).split(np.zeros(len(d)), y))
        for s, (tr, te) in enumerate(splits):
            f = os.path.join(CK, f"{pop}_{s:02d}.json")
            if os.path.exists(f) or time.time() - t0 > a.maxsec: continue
            if int(os.environ.get("SHARD_N", 1)) > 1 and s % int(os.environ["SHARD_N"]) != int(os.environ.get("SHARD_I", 0)): continue
            C, G = d[CLIN].values, d[GS].values
            Ntr, Nte = score_modules(Ep, mods, tr, te, a.scoring)
            des = {"CLIN": (C[tr], C[te]), "CLIN+GS": (np.hstack([C[tr], G[tr]]), np.hstack([C[te], G[te]])),
                   "CLIN+NET": (np.hstack([C[tr], Ntr]), np.hstack([C[te], Nte])),
                   "CLIN+GS+NET": (np.hstack([C[tr], G[tr], Ntr]), np.hstack([C[te], G[te], Nte]))}
            rec = {"pop": pop, "split": s, "res": {}}
            for fs, (Xtr, Xte) in des.items():
                r, alpha, coef = tuned_cox_coef(Xtr, y[tr], t[tr], Xte)
                out = {"cox": cidx(y[te], t[te], r), "cox_alpha": alpha, "cox_tdauc60": td_auc60(y[tr], t[tr], y[te], t[te], r)}
                if fs == "CLIN+NET":
                    out["net_coef"] = dict(zip(mods.keys(), map(float, coef[len(CLIN):])))
                if not a.no_rsf:
                    rr, leaf = tuned_rsf(Xtr, y[tr], t[tr], Xte); out["rsf"] = cidx(y[te], t[te], rr)
                rec["res"][fs] = out
            rc = []
            for rm in rand_sets:
                Rtr, Rte = score_modules(Ep, rm, tr, te, a.scoring)
                r, _, _ = tuned_cox_coef(np.hstack([C[tr], Rtr]), y[tr], t[tr], np.hstack([C[te], Rte]))
                rc.append(cidx(y[te], t[te], r))
            rec["res"]["CLIN+RAND"] = {"cox": float(np.mean(rc)), "cox_draws": list(map(float, rc))}
            json.dump(rec, open(f, "w"))
    done = len(glob.glob(os.path.join(CK, "*.json"))); total = len(POPS) * K * R
    print(f"[{a.tag}] splits saved: {done}/{total}" + ("  ALL SPLITS DONE" if done == total else ""))
    if not a.summarize or done < total:
        return
    summarize(OUT, CK, mods, a)


def summarize(OUT, CK, mods, a):
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(CK, "*.json")))]
    rows = []
    for r in recs:
        for fs, v in r["res"].items():
            rows.append({"pop": r["pop"], "split": r["split"], "set": fs, "cox": v.get("cox"),
                         "rsf": v.get("rsf"), "tdauc60": v.get("cox_tdauc60")})
    df = pd.DataFrame(rows); df.to_csv(os.path.join(OUT, "cv_folds.csv"), index=False)
    summ = df.groupby(["pop", "set"]).agg(cox_mean=("cox", "mean"), cox_sd=("cox", "std"),
                                          rsf_mean=("rsf", "mean"), tdauc60_mean=("tdauc60", "mean")).reset_index()
    summ.to_csv(os.path.join(OUT, "cv_summary.csv"), index=False)

    def cmp(pop, A, B, metric="cox"):
        g = df[df["pop"] == pop]
        va = g[g.set == A].sort_values("split")[metric].values; vb = g[g.set == B].sort_values("split")[metric].values
        dlt = va - vb; m, lo, hi, p = corrected_repeated_t(dlt, k=K, r=R)
        return {"pop": pop, "comparison": f"{A} vs {B}", "metric": metric, "mean_dC": m, "ci_low": lo, "ci_high": hi,
                "min_dC": dlt.min(), "max_dC": dlt.max(), "splits_won": int((dlt > 0).sum()), "n": len(dlt), "p_corrected": p}

    prim = [cmp("All", "CLIN+NET", "CLIN+GS"), cmp("ER-", "CLIN+NET", "CLIN+GS"),
            cmp("All", "CLIN+GS+NET", "CLIN+GS"), cmp("ER-", "CLIN+GS+NET", "CLIN+GS"),
            cmp("All", "CLIN+NET", "CLIN+RAND"), cmp("ER-", "CLIN+NET", "CLIN+RAND")]
    for x, h in zip(prim, holm([x["p_corrected"] for x in prim])): x["p_holm"] = h
    pd.DataFrame(prim).to_csv(os.path.join(OUT, "primary_comparisons.csv"), index=False)
    sec = [cmp("ER+", "CLIN+NET", "CLIN+GS"), cmp("ER+", "CLIN+GS+NET", "CLIN+GS"), cmp("ER+", "CLIN+NET", "CLIN+RAND")]
    for pop in POPS:
        sec += [cmp(pop, "CLIN+GS", "CLIN"), cmp(pop, "CLIN+NET", "CLIN")]
        if df.rsf.notna().any():
            sec += [cmp(pop, "CLIN+NET", "CLIN+GS", "rsf"), cmp(pop, "CLIN+GS+NET", "CLIN+GS", "rsf")]
        sec += [cmp(pop, "CLIN+NET", "CLIN+GS", "tdauc60")]
    pd.DataFrame(sec).to_csv(os.path.join(OUT, "secondary_comparisons.csv"), index=False)

    # biology readout: top modules by mean |standardized coef| in CLIN+NET
    deg = pd.read_csv(a.degree).set_index("gene")["degree"] if a.degree and os.path.exists(a.degree) else pd.Series(dtype=float)
    hm = {}
    gmt = os.path.join(ROOT, "data", "gene_sets", "h.all.v2026.1.Hs.symbols.gmt")
    if os.path.exists(gmt):
        for line in open(gmt):
            p = line.rstrip("\n").split("\t"); hm[p[0]] = set(p[2:])
    pool = set(json.load(open(os.path.join(PILOT, "inputs", "gene_pools.json")))[a.pool])
    bio = []
    for pop in ["All", "ER-"]:
        coefs = pd.DataFrame([r["res"]["CLIN+NET"]["net_coef"] for r in recs if r["pop"] == pop])
        ranks = coefs.abs().rank(axis=1, ascending=False)
        for k in coefs.abs().mean().sort_values(ascending=False).index[:5]:
            genes = mods[k]
            hubs = deg.reindex(genes).dropna().sort_values(ascending=False).head(5).index.tolist() if len(deg) else []
            best, bp = None, 1.0
            for name, hs in hm.items():
                hs_p = hs & pool; ov = len(set(genes) & hs_p)
                if ov == 0: continue
                pv = hypergeom.sf(ov - 1, len(pool), len(hs_p), len(genes))
                if pv < bp: best, bp = f"{name} ({ov} genes)", pv
            bio.append({"pop": pop, "module": k, "size": len(genes), "hub_genes": ", ".join(hubs),
                        "mean_coef": coefs[k].mean(), "hazard_direction": "higher score = higher risk" if coefs[k].mean() > 0 else "higher score = lower risk",
                        "pct_splits_top5": float((ranks[k] <= 5).mean() * 100), "top_hallmark": best,
                        "hallmark_p_raw": bp, "hallmark_p_bonf50": min(1.0, bp * 50)})
    pd.DataFrame(bio).to_csv(os.path.join(OUT, "biology_top_modules.csv"), index=False)
    print(summ.round(3).to_string()); print(pd.DataFrame(prim).round(4).to_string())


if __name__ == "__main__":
    main()
