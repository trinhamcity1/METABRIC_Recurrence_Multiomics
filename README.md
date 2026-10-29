# Genomic Features for Recurrence Risk Stratification in METABRIC With an Exploratory TCGA Assessment

Code, processed data and results for the ACET 2026 manuscript of the same title (Nham, Arteaga-Arteaga, Bravo-Ortiz, Tabares-Soto, Hernandez, Arenas, Guillen-Rondon). Follow-up to *Multi-Target Machine Learning for Breast Cancer Prognosis and Molecular Classification Using METABRIC* (EGETC 2026).

**Question.** Do gene-level molecular summaries (expression gene-set scores, copy-number and mutation drivers; methylation and immune marker summaries as secondary blocks) improve recurrence discrimination in METABRIC, and is a similar signal seen in TCGA-BRCA?

## Main results (from the files in `results/`)

All internal comparisons use the same patients and seeded splits. Every step that learns from data inside this pipeline (scaling, PCA, SHAP selection, hyperparameter tuning) is fitted inside training folds, and inside inner folds during tuning. **Exception:** the expression input is the cBioPortal mRNA z-score file, which the data provider normalized against a reference population from the cohort before this pipeline; that step is not re-estimated inside folds, so the evaluation uses previously cohort-normalized data and is not strictly inductive.

Baselines: **BASE** = the EGETC legacy baseline (clinical, treatment and mutation-count variables) with constant / duplicated columns removed (21 columns in all patients, 19 in ER+); **CLIN** = BASE without the two mutation-count summaries (strictly clinical and treatment). **GEN** = all 77 genomic features; **PC+DRV** = 9 principal components of the 57 gene-set scores + 20 drivers.

| ER+ patients (n = 996, 408 events), tuned models, 50 splits | Baseline | + genomic | Gain (95% CI) | Holm p |
|---|---|---|---|---|
| Random survival forest, BASE+PC+DRV vs BASE | 0.629 | 0.662 | +0.033 (+0.015 to +0.051) | 0.002 |
| Random survival forest, CLIN+PC+DRV vs CLIN | 0.620 | 0.662 | +0.043 (+0.024 to +0.061) | <0.001 |
| Ridge Cox, BASE+GEN vs BASE | 0.632 | 0.658 | +0.026 (−0.002 to +0.054) | 0.28 |

These are **internal, exploratory** model-development results (the reduced representation, the tuned models, the CLIN comparator and CORE4 were added after an earlier analysis round). Holm adjustment covers each declared comparison family, not every adaptive decision. Uno's IPCW C-index at 120 months gives similar internal increments (`results/phase2/ipcw_internal.csv`).

- All patients combined: gains +0.004 to +0.017, none significant after Holm adjustment.
- Five-year landmark (ER+, 188 events): the evaluated genomic feature sets did not improve discrimination (−0.025, 95% CI −0.078 to +0.027).
- Methylation / immune summaries: no improvement over BASE+GEN was demonstrated (paired contrasts in `multiomics_*_tests.csv`).
- **Exploratory TCGA assessment** (refitted harmonized model with 6 predictors, within-cohort molecular standardization, ESR1-inferred ER; 679 ER+ patients, 66 PFI events; median reverse-KM follow-up 28.4 months): Cox gain +0.014 (−0.036 to +0.064); full RSF −0.029; all gain intervals include zero; Uno's C at 36 months gives −0.009 for Cox. This is not external validation of a frozen predictor.
- SHAP results describe the binary LightGBM classifier in all patients (most stable molecular attribution: fraction of assayed genes with a copy-number call), not the ER+ survival models.

Binary Recurred / Not Recurred classification (logistic regression, LightGBM) ignores censoring and is reported only as descriptive classification of observed status. AutoML (FLAML, MLJAR) results from the earlier specification are archived in `results/legacy_v1/` and are not used in the manuscript.

## Reproducing the manuscript

```
# Python 3.10/3.11 recommended
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-lock.txt      # exact versions used (Python 3.10.12, Linux)
#   or: pip install -r requirements.txt   # broader ranges, e.g. Python 3.11 on macOS (LightGBM needs: brew install libomp)
bash run_all.sh                            # ~40-60 min on 4 cores; logs in results/logs/
RUN_EXTRA=1 bash run_all.sh                # also AutoML + Phase 1 pilot (not in the manuscript)
```

`run_all.sh` rebuilds features from raw cBioPortal data if present, otherwise starts from `data/processed/`. Each step stops the pipeline on failure; the checkpointed tuned-survival step has a retry limit and a progress check, and checkpoints carry a specification hash so stale results are never mixed in. `docs/TABLE_FIGURE_MANIFEST.md` maps every manuscript table and figure to its script and output file.

## Data

| Block | Prefix | Count | Definition |
|---|---|---|---|
| Clinical table | (as in METABRIC) | 34 | cBioPortal clinical data |
| Hallmark gene-set scores | `HM_` | 50 | mean of member-gene expression z-scores (MSigDB human Hallmark, release 2026.1, `h.all.v2026.1.Hs.symbols.gmt`) |
| Desmedt modules | `DM_` | 7 | weighted average of member-gene z-scores, published coefficients (genefu `mod1`) |
| Copy-number drivers | `CNA_` | 11 | 9 driver amplifications (GISTIC +2), any deep deletion of 4 tumour suppressors, and `CNA_frac_genes_altered` = fraction of **assayed genes** with a non-zero GISTIC call (missing calls excluded; not length-weighted, not "fraction of genome altered") |
| Mutation drivers | `MUT_` | 9 | non-silent mutation in 8 driver genes + log count of non-silent variants (173-gene panel in METABRIC, exome in TCGA: a variant count, not TMB per Mb) |
| Methylation | `METH_` | 20 (14 modelled) | RRBS promoter methylation summaries; the 14 modelled features (≤1% missing among profiled tumours) are a frozen list |
| Immune / stromal | `IMM_` | 10 | **MCP-counter marker-based z-score summaries** (mean of marker-gene z-scores); not the original MCP-counter abundance estimator, not cell proportions |

Cohort flow (2,509 → 1,295 modelling patients → 945 with methylation) and the included-vs-excluded comparison are in `results/cohort/`. TCGA exclusions (unknown M stage and neoadjuvant status counted separately) and a patient-to-sample map are in `results/cohort/tcga_exclusions.csv` and `tcga_sample_map.csv`.

**Raw data** (not in GitHub): METABRIC (`brca_metabric`) and TCGA-BRCA PanCancer Atlas (`brca_tcga_pan_can_atlas_2018`) from https://www.cbioportal.org/datasets, unzipped into `data/`; Hallmark `.gmt` from MSigDB; `mod1.rda` from https://github.com/bhklab/genefu; MCP-counter `genes.txt` from https://github.com/ebecht/MCPcounter. The analysed METABRIC release was downloaded in May 2025; its expression file is `data_mrna_illumina_microarray_zscores_ref_diploid_samples.txt`.

## Script index (`src/`)

| Script | Purpose |
|---|---|
| `build_features.py`, `build_methylation_features.py`, `build_immune_features.py`, `extract_desmedt_modules.py` | METABRIC feature construction |
| `build_tcga_features.py` | TCGA features with the same formulas + harmonized clinical predictors |
| `make_cohort_tables.py` | cohort flow, Table 2, included vs excluded, TCGA exclusions and sample map |
| `pca_expression_scores.py` | descriptive PCA and redundancy clusters (Figure 2) |
| `model_compare_shap.py` | all-patient feature-set comparison (Table 3) and SHAP (Table 5); shared helpers (`clean_cols`, `baselines`, corrected t-test) |
| `model_er_late.py` | fixed-penalty Cox: all, ER+, five-year landmark (Table 4) |
| `model_multiomics.py` | methylation / immune paired contrasts (Table 6); `NO_METH=1`, `NO_IMM=1` options |
| `model_tuned_survival.py` | tuned Cox and RSF, whole pipeline inside inner folds, 50 splits (Tables 7–8); saves out-of-fold predictions |
| `validate_tcga.py` | exploratory TCGA assessment (Table 9), harmonized internal comparison, saved preprocessing/models; `EXCL_MUTCOUNT=1` sensitivity |
| `ipcw_sensitivity.py` | Uno's IPCW C-index from saved predictions |
| `make_figure1.py` | Figure 1 from the result tables |
| `run_automl_phase2.py`, `run_recurrence_molecular.py`, `run_survival_framing.py` | not used in the manuscript (AutoML, Phase 1 pilot) |

## Known limitations

Previously cohort-normalized expression input; complete-case cohort (51.6% of patients); adaptive, exploratory model development on one cohort; discrimination only (no calibration or decision analysis; proportional-hazards assumptions not assessed); treatment variables are post-planning information; TCGA endpoint, platform, ER ascertainment and follow-up differ from METABRIC. See the manuscript, Section 4.4.
