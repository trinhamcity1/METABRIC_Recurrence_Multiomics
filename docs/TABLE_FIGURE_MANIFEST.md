# Manuscript table and figure manifest

All paths relative to the repository root. `run_all.sh` runs every command below in order.
Commands are run from `src/`.

| Manuscript element | Command | Input | Output file(s) |
|---|---|---|---|
| Table 1 (data sources) | descriptive; counts from raw cBioPortal files | `data/brca_metabric/` | — |
| Section 2.1 cohort flow, included vs excluded; Table 2 | `python make_cohort_tables.py` | `data/processed/metabric_recurrence_multiomics.csv` | `results/cohort/cohort_flow.csv`, `included_vs_excluded.csv`, `table2_characteristics.csv` |
| Section 2.8 TCGA exclusions, sample map | `python make_cohort_tables.py` | `data/brca_tcga_pan_can_atlas_2018/` | `results/cohort/tcga_exclusions.csv`, `tcga_sample_map.csv` |
| Section 3.2, Figure 2 | `python pca_expression_scores.py` | processed METABRIC | `results/phase2/fig_pca_redundancy.png`, `pca_explained_variance.csv`, `redundancy_clusters.csv` |
| Table 3 | `SUMMARIZE=1 python model_compare_shap.py` | processed METABRIC | `results/phase2/compare_summary.csv`, `compare_tests.csv` |
| Table 5, Section 3.5 | same | same | `results/phase2/shap_stability_BASE_GEN.csv`, `shap_selected_frequency.csv` |
| Table 4, Section 3.4 | `python model_er_late.py` | processed METABRIC | `results/phase2/er_late_summary.csv`, `er_late_tests.csv` |
| Table 6, Section 3.6 | `python model_multiomics.py`; `NO_METH=1 python model_multiomics.py`; `NO_IMM=1 python model_multiomics.py` | processed METABRIC | `results/phase2/multiomics_meth_imm_*.csv`, `multiomics_imm_*.csv`, `multiomics_meth_*.csv` |
| Tables 7–8, Section 3.7 | `python model_tuned_survival.py` (repeat until "ALL SPLITS DONE"), then `SUMMARIZE=1 python model_tuned_survival.py` | processed METABRIC | `results/phase2/tuned_survival_summary.csv`, `tuned_survival_tests.csv`, `tuned_survival_selected_params.csv`, `tuned_survival_folds.csv`, `tuned_survival_oof_predictions.csv.gz` |
| Table 9, Section 3.8 | `python validate_tcga.py` | processed METABRIC + `data/processed/tcga_brca_features.csv` | `results/external_tcga/tcga_validation_cindex.csv`, `tcga_validation_gain.csv`, `tcga_validation_cohorts.json` (reverse-KM follow-up), `metabric_internal_harmonized_cox.csv`, `tcga_predictions_*.csv`, `models/` |
| Section 3.8 mutation-count sensitivity | `EXCL_MUTCOUNT=1 python validate_tcga.py` | same | `results/external_tcga/*_no_mutcount.*` |
| Sections 3.7–3.8 IPCW | `python ipcw_sensitivity.py` | saved predictions | `results/phase2/ipcw_internal.csv`, `results/external_tcga/ipcw_tcga.csv` |
| Figure 1 | `python make_figure1.py` | the result tables above | `results/figures/figure1_gain_forest.png`, `figure1_data.csv` |

Archived, not used in the manuscript: `results/legacy_v1/` (AutoML and SHAP outputs from the earlier predictor specification, before the October 2026 revision).
