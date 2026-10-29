"""
Cohort flow, included-vs-excluded comparison, Table 2 (cohort characteristics with
reverse Kaplan-Meier follow-up) and TCGA exclusion counts.

Outputs (results/cohort/):
  cohort_flow.csv            sequential exclusions 2,509 -> 1,295 (-> 945 methylation)
  included_vs_excluded.csv   modelling cohort vs patients with a recorded outcome who were excluded
  table2_characteristics.csv
  tcga_exclusions.csv        sequential TCGA exclusions, unknown M stage / neoadjuvant counted separately
"""
import os
import numpy as np, pandas as pd
from sksurv.nonparametric import kaplan_meier_estimator
from model_compare_shap import load, LEGACY

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(ROOT, "results", "cohort"); os.makedirs(OUT, exist_ok=True)


def rkm(t, e):
    x, s = kaplan_meier_estimator((1 - np.asarray(e)).astype(bool), np.asarray(t))
    b = np.where(s <= 0.5)[0]
    return float(x[b[0]]) if len(b) else float("nan")


def main():
    co, DM, HM, DRV = load()
    raw = pd.read_csv(os.path.join(ROOT, "data", "processed", "metabric_recurrence_multiomics.csv"), low_memory=False)
    ids_model = set(co["Patient ID"])
    # rebuild the derived legacy columns on all patients by re-using load() logic without the dropna
    import model_compare_shap as mcs
    d = raw.copy()
    d["y"] = d["Relapse Free Status"].map({"Recurred": 1, "Not Recurred": 0})
    d["t"] = d["Relapse Free Status (Months)"]
    clin_raw = ["Age at Diagnosis", "Neoplasm Histologic Grade", "Lymph nodes examined positive", "Mutation Count",
                "Nottingham prognostic index", "Tumor Size", "Tumor Stage", "ER Status", "PR Status", "HER2 Status",
                "Chemotherapy", "Hormone Therapy", "Radio Therapy", "Type of Breast Surgery", "Inferred Menopausal State"]
    G = DM + HM
    steps = [("All patients in the clinical table", pd.Series(True, index=d.index)),
             ("Recorded relapse-free status and time", d.y.notna() & d.t.notna()),
             ("Follow-up time > 0", d.t > 0),
             ("Complete legacy clinical/treatment variables", d[clin_raw].notna().all(axis=1)),
             ("Expression gene-set scores available", d[G].notna().all(axis=1)),
             ("Copy-number features available", d[[c for c in DRV if c.startswith("CNA_")]].notna().all(axis=1)),
             ("Mutation features available (sequenced)", d[[c for c in DRV if c.startswith("MUT_")]].notna().all(axis=1))]
    keep = pd.Series(True, index=d.index); rows = []
    for label, cond in steps:
        prev = int(keep.sum()); keep &= cond.fillna(False)
        rows.append({"step": label, "remaining": int(keep.sum()), "excluded_at_step": prev - int(keep.sum()),
                     "events_remaining": int(d.loc[keep, "y"].sum())})
    assert set(d.loc[keep, "Patient ID"]) == ids_model, "flow does not reproduce the modelling cohort"
    prof = raw[raw["METH_mean"].notna()]
    METH = [c for c in raw.columns if c.startswith("METH_") and prof[c].isna().mean() <= 0.01]
    meth_ok = keep & d[METH].notna().all(axis=1)
    rows.append({"step": f"Methylation subset ({len(METH)} features with <=1% missing among profiled)",
                 "remaining": int(meth_ok.sum()), "excluded_at_step": int(keep.sum() - meth_ok.sum()),
                 "events_remaining": int(d.loc[meth_ok, "y"].sum())})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "cohort_flow.csv"), index=False)
    print(pd.DataFrame(rows).to_string())

    # included vs excluded among patients with a recorded outcome
    base = d.y.notna() & d.t.notna() & (d.t > 0)
    inc, exc = d[keep], d[base & ~keep]
    def summ(x):
        return {"n": len(x), "recurrence_pct": 100 * x.y.mean(), "age_median": x["Age at Diagnosis"].median(),
                "er_pos_pct": 100 * (x["ER Status"] == "Positive").mean(),
                "grade3_pct": 100 * (x["Neoplasm Histologic Grade"] == 3).mean(),
                "node_pos_pct": 100 * (x["Lymph nodes examined positive"] > 0).mean(),
                "tumor_size_median": x["Tumor Size"].median(), "chemo_pct": 100 * (x["Chemotherapy"] == "Yes").mean(),
                "followup_rkm_months": rkm(x.t, x.y)}
    ie = pd.DataFrame({"included": summ(inc), "excluded_with_outcome": summ(exc)})
    ie.to_csv(os.path.join(OUT, "included_vs_excluded.csv")); print(ie.round(1).to_string())

    # Table 2
    def t2(x):
        return {"n": len(x), "recurrences": int(x.y.sum()), "recurrence_pct": 100 * x.y.mean(),
                "age_median": x["Age at Diagnosis"].median(), "age_q1": x["Age at Diagnosis"].quantile(.25),
                "age_q3": x["Age at Diagnosis"].quantile(.75), "her2_pos_pct": 100 * (x["HER2 Status"] == "Positive").mean(),
                "node_pos_pct": 100 * (x["Lymph nodes examined positive"] > 0).mean(),
                "grade3_pct": 100 * (x["Neoplasm Histologic Grade"] == 3).mean(),
                "size_median": x["Tumor Size"].median(), "size_q1": x["Tumor Size"].quantile(.25),
                "size_q3": x["Tumor Size"].quantile(.75), "chemo_pct": 100 * (x["Chemotherapy"] == "Yes").mean(),
                "hormone_pct": 100 * (x["Hormone Therapy"] == "Yes").mean(),
                "radio_pct": 100 * (x["Radio Therapy"] == "Yes").mean(),
                "censored_before_60m": int(((x.y == 0) & (x.t < 60)).sum()),
                "followup_rkm_months": rkm(x.t, x.y)}
    tab = pd.DataFrame({"All": t2(inc), "ER-positive": t2(inc[inc["ER Status"] == "Positive"]),
                        "ER-negative": t2(inc[inc["ER Status"] == "Negative"])})
    tab.to_csv(os.path.join(OUT, "table2_characteristics.csv")); print(tab.round(1).to_string())

    # TCGA exclusions
    rp = os.path.join(ROOT, "data", "brca_tcga_pan_can_atlas_2018", "data_clinical_patient.txt")
    if os.path.exists(rp):
        p = pd.read_csv(rp, sep="\t", comment="#", low_memory=False)
        m = p["PATH_M_STAGE"].astype(str); nj = p["HISTORY_NEOADJUVANT_TRTYN"].astype(str)
        tr = [("TCGA-BRCA patients", len(p)), ("Male (excluded)", int((p.SEX != "Female").sum())),
              ("Female, M1 (excluded)", int(((p.SEX == "Female") & (m == "M1")).sum())),
              ("Female, not M1, neoadjuvant = Yes (excluded)", int(((p.SEX == "Female") & (m != "M1") & (nj == "Yes")).sum())),
              ("Retained", int(((p.SEX == "Female") & (m != "M1") & (nj != "Yes")).sum())),
              ("  of retained: M stage not M0 or M1 (unknown/MX/missing; retained, not confirmed M0)",
               int(((p.SEX == "Female") & (m != "M1") & (nj != "Yes") & ~m.isin(["M0", "CM0 (I+)"])).sum())),
              ("  of retained: neoadjuvant status not 'No' (unknown/missing; retained)",
               int(((p.SEX == "Female") & (m != "M1") & (nj != "Yes") & (nj != "No")).sum()))]
        pd.DataFrame(tr, columns=["step", "n"]).to_csv(os.path.join(OUT, "tcga_exclusions.csv"), index=False)
        print(pd.DataFrame(tr).to_string())
        # sample map: patient -> sample barcode per modality, with sample type
        rawd = os.path.dirname(rp); maps = {}
        for mod, f in [("expression", "data_mrna_seq_v2_rsem.txt"), ("copy_number", "data_cna.txt")]:
            cols = [c for c in pd.read_csv(os.path.join(rawd, f), sep="\t", nrows=0).columns if c.startswith("TCGA-")]
            maps[mod] = pd.Series(cols, index=[c[:12] for c in cols])
        mut = pd.read_csv(os.path.join(rawd, "data_mutations.txt"), sep="\t", comment="#", usecols=["Tumor_Sample_Barcode"])
        ms = mut.Tumor_Sample_Barcode.drop_duplicates(); maps["mutation"] = pd.Series(ms.values, index=ms.str[:12].values)
        sm = pd.DataFrame({k: v[~v.index.duplicated()] for k, v in maps.items()})
        for k in maps:
            sm[k + "_sample_type"] = sm[k].str[13:15]
            sm[k + "_n_samples"] = maps[k].index.value_counts().reindex(sm.index)
        sm.index.name = "patient_id"; sm.to_csv(os.path.join(OUT, "tcga_sample_map.csv"))
        print("TCGA sample map:", len(sm), "patients; max samples per patient:",
              {k: int(sm[k + "_n_samples"].max()) for k in maps})


if __name__ == "__main__":
    main()
