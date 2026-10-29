# Response to "Improvements Required Before Submission" (H. B. Arteaga-Arteaga)

Manuscript and repository were revised together. Every change to preprocessing, predictors or model selection was rerun, and all tables, Figure 1, the abstract, conclusion and README were regenerated from the new outputs (`docs/TABLE_FIGURE_MANIFEST.md`). Status: **Done**, **Partly** (what remains is stated as a limitation), or **Authors** (needs input from the author team).

## Headline numbers after the rerun

| | Before | After |
|---|---|---|
| ER+ tuned RSF, reduced set vs baseline | 0.633 → 0.661, +0.028 (0.009 to 0.046), Holm p = 0.016 | 0.629 → 0.662, +0.033 (0.015 to 0.051), Holm p = 0.002 |
| Same vs strictly clinical comparator (new) | — | +0.043 (0.024 to 0.061) |
| ER+ tuned Cox, all genomic | +0.025, Holm p = 0.14 | +0.026 (−0.002 to 0.054), Holm p = 0.28 |
| TCGA ESR1-inferred ER+, Cox | +0.014 (−0.036 to 0.064) | unchanged; IPCW at 36 mo: −0.009 |
| TCGA ER+, full RSF | −0.050 | −0.029 (now tuned like the internal models) |
| TCGA follow-up | "28 months" for all 902 | reverse KM 29.9 (all) and 28.4 months (ER+) |

## Item by item

| # | Status | What changed |
|---|---|---|
| 01 Normalization | Partly | "No information from other patients" / "leakage-free" removed from paper, code and README. Paper states that expression z-scores were provider-normalized against a cohort reference, which was not re-estimated in folds, so the evaluation uses previously cohort-normalized data. The file name and download date are documented. A strictly inductive rebuild is listed as a limitation and as future work. |
| 02 PCA in inner CV | Done | Scaling, PCA and model fitting are refitted inside every inner fold. RSF leaf size is chosen by inner CV of the whole pipeline instead of OOB. 9 PCs is documented as a fixed choice taken from the descriptive PCA. All tuned models were rerun on the same outer splits. |
| 03 "Clinical only" | Done | The legacy baseline is renamed BASE and described as clinical + treatment + mutation count. A new CLIN comparator (no mutation-count summaries) was run on identical patients and splits, with uncertainty, in the fixed Cox, tuned Cox and tuned RSF analyses. HCLIN components (ERBB2 amplification, ESR1-inferred ER) are explained. |
| 04 Duplicates | Done | Constant and duplicated columns are removed within each cohort: 21 columns in all patients, 19 in ER+. Cleaned vs legacy on identical splits changes the C-index by < 0.001. Retention of NPI and categories is justified. "51 non-redundant features" was removed. |
| 05 Binary vs survival | Done | Survival is the primary analysis. LR/LightGBM AUCs are labeled descriptive classification of observed status, with the censoring counts (88 / 73 before 60 months). The AutoML table was removed from the paper; its archived results are in `results/legacy_v1/`. |
| 06 TCGA framing | Done | Section renamed "Exploratory assessment". It distinguishes the internal model, the refitted harmonized model and target-cohort transformations. Cox models, all scalers and PCA objects, standardization parameters and predictions are saved. "Validated" wording removed. |
| 07 Endpoint direction | Done | Your replacement sentence is used. Event, time origin and censoring are defined for each cohort. Non-significance is no longer attributed only to the event count. |
| 08 Exploratory labeling | Done | Primary, secondary and post hoc analyses are declared (Section 2.4). The reduced RSF result is labeled internal exploratory, and Holm is described as within-family only. CORE4 is in the testing family and reported. "Reproducible" removed. Your title is adopted. The abstract and conclusion use your wording with updated numbers. |
| 09 CNA feature | Done | Renamed to fraction of assayed genes with a non-zero GISTIC call. Missing calls are now excluded (82 METABRIC tumors, 2,356 calls, change ≤ 0.003; TCGA has none). Indicators keep missingness (none at driver genes). All models and SHAP were rerun. |
| 10 Immune naming | Done | Renamed "MCP-counter marker-based z-score summaries" in the paper, feature dictionary, code and README. |
| 11 Mutation count | Done | Panel vs exome difference stated; described as a variant count, not TMB/Mb. TCGA sensitivity analysis without the count gives Cox +0.016. |
| 12 Harmonization | Partly | Proxies, T4→T3 recoding, ERBB2-for-HER2 and "ESR1-inferred ER+" are stated. The influence of T4 recoding was not assessed (stated). |
| 13 Sample map | Done | `results/cohort/tcga_sample_map.csv`: all samples are type 01, one per patient per modality, so the duplicate rule never drops a sample. Unknown M stage (158) and neoadjuvant status (1) are counted separately. |
| 14 Cohort flow | Done | Sequential flow 2,509 → 2,380 → 2,376 → 1,295 → 945 methylation, with an included-vs-excluded comparison, in the text and `results/cohort/`. The methylation list is documented as frozen. |
| 15 Prediction time | Done | New "Prediction time" paragraph: post-treatment-planning prognosis, not pretreatment; no causal reading of treatment. The landmark uses baseline information only. |
| 16 Censoring / calibration | Partly | Uno's IPCW C-index was added (internal τ = 120 mo, which agrees; TCGA τ = 36 mo, where the Cox gain becomes −0.009). Claims are limited to discrimination. Calibration and PH checks are not done (stated). |
| 17 METH/IMM vs GEN | Done | Paired contrasts against BASE+GEN with CIs and declared Holm families (new Table 6). "Add nothing" replaced. NO_IMM option added and filename tag fixed. |
| 18 Landmark wording | Done | Your sentence is used; the overfitting explanation was removed and the event is defined. "Late distant-type" was removed from the pilot code. |
| 19 ER− claim | Done | ER− claims removed; your sentence is used. |
| 20 SHAP scope | Partly | SHAP is tied to the binary LightGBM classifier in all patients, on the log-odds scale. "Primarily through instability/proliferation" was removed. Block-addition results show most of the ER+ Cox gain comes from Hallmark scores (drivers +0.008). Paired block ablation is listed as future work. |
| 21 Proliferation cluster | Done | Your sentence is used, and the clustering description is corrected. |
| 22 Linear models transfer | Done | Your sentence is used; reduced RSF is included; no superiority claim. |
| 23 Ceiling / equivalence | Done | "Realistic ceiling" and "matches commercial signatures" were removed. |
| 24 TCGA follow-up | Done | `validate_tcga.py` now computes reverse KM; observed-time median kept as a separate field. |
| 25 Labels | Done | Splits per method stated. CSV keys, paper and README use the same names (CLIN+ALL → BASE+GEN). README Holm values regenerated from the files. |
| 26 Zero-variance test | Done | All-zero differences return (0, 0, 0, p = 1); a constant non-zero difference is reported, not tested. Inputs are validated, and regression tests are in `src/test_stats.py`. |
| 27 Checkpoints | Done | Bounded retries with a progress check, per-step logs, a spec-hash in checkpoints, atomic writes, and completeness/duplicate checks before aggregation. |
| 28 Generators | Done | New `make_figure1.py`, `make_cohort_tables.py` (Table 2 with reverse KM), harmonized internal Cox inside `validate_tcga.py`, NO_IMM branch, manifest. |
| 29 Environment | Partly | `requirements-lock.txt` added (Python 3.10.12, Linux, 4 cores; the paper previously said 3.11, now corrected). Seeds documented (42; ESR1 mixture 0). Out-of-fold and TCGA predictions saved. Blanket warning suppression narrowed to FutureWarning. Grid expansion is not done (stated). |
| 30 References / declarations | Partly / Authors | Refs 13 and 40 corrected; "Best Paper Award" removed; human Hallmark release named; TRIPOD+AI adherence claim removed. Ethics/data-use statement added. **Funding, conflicts of interest (KlinicSync) and author contributions are yellow placeholders for the authors; the EGETC proceedings identifier still needs to be added.** |
| 31 Language | Done | Absolute claims replaced; your two grammar fixes applied. |
| 32 C-index meaning | Done | Your sentence is used. |
| 33 ACET requirements | Authors | Page limit, abstract limit and template need confirming in the ACET portal or at info@acetweb.org. The manuscript is 18 pages; the abstract is 251 words. |

## Not done before the deadline (stated as limitations or future work)

- Strictly inductive renormalization of expression.
- Calibration, Brier score and decision curves.
- PH and functional-form checks.
- T4-recoding sensitivity.
- Paired block ablation for SHAP-free attribution.
- Wider Cox penalty grid.
- A length-weighted copy-number measure.
- An AutoML rerun under the new specification (removed from the paper).
