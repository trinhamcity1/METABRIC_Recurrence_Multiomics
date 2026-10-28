# METABRIC Recurrence: Clinical + Multi-Omic Prediction

Follow-up to *Multi-Target Machine Learning for Breast Cancer Prognosis and Molecular Classification Using METABRIC* (EGETC 2026, Best Paper Award).
Target venue: ACET Annual Conference 2026, followed by a journal paper.

**Question.** Recurrence was the hardest target in the EGETC paper (AUC 0.666 with clinical features). Do genomic, epigenomic, and immune features improve recurrence prediction, and for which patients?

## Documentation

The full write-up of methods, the reasoning behind each choice, all results, and limitations is in **`docs/Phase2_Methods_and_Findings.md`** (also as .docx). The Phase 1 pilot is in `docs/Phase1_Pilot_Report.md`.

## Key findings:

All comparisons use the same patients and the same seeded folds; anything learned from data (scaling, PCA, feature selection, AutoML tuning) is fitted inside the training fold only.

| Analysis (penalised Cox, 5x5 repeated CV, C-index) | Clinical only | + Genomic | Change | Splits improved | Holm p |
|---|---|---|---|---|---|
| All patients (n = 1,295, 545 recurred) | 0.649 | 0.653 | +0.004 | 15/25 | 1.00 |
| **ER+ patients (n = 996, 408 recurred)** | **0.627** | **0.659** | **+0.032** | **23/25** | 0.15 |
| Late recurrence > 5 y, ER+ (n = 703) | 0.596 | 0.571 | -0.025 | 10/25 | 1.00 |

**Tuned survival models (10 repeats x 5 folds = 50 splits, tuning inside training folds only):**

| ER+ patients (n = 996) | Clinical only | + All genomic (99 features) | + Reduced genomic (9 PCs + 20 drivers) |
|---|---|---|---|
| Penalised Cox, tuned penalty | 0.632 | 0.657 (+0.025, 46/50 splits, Holm p = 0.14) | 0.645 (+0.012) |
| **Random survival forest, tuned** | 0.633 | 0.656 (+0.023, 43/50, Holm p = 0.14) | **0.661 (+0.028, 45/50, Holm p = 0.016)** |

All patients: gains of +0.009 to +0.012, not significant. Results: `results/phase2/tuned_survival_summary.csv`, `tuned_survival_tests.csv`.

**External validation (train on METABRIC, test on TCGA-BRCA; 6 harmonized clinical features):**

| TCGA ER+ (679 patients, 66 events) | Clinical | + Genomic | Gain (95% bootstrap CI) |
|---|---|---|---|
| Penalised Cox | 0.608 | 0.623 | +0.014 (−0.036 to +0.064) |
| Random survival forest | 0.604 | 0.554 | −0.050 (−0.128 to +0.029) |

The ER+ gain keeps its direction in TCGA for Cox but is smaller and not significant (short follow-up, few events); the random survival forest does not transfer. Results: `results/external_tcga/`.

- The genomic gain is concentrated in **ER+ patients** and is consistent across Cox, random survival forests, FLAML (+0.021 AUC) and MLJAR (+0.027 AUC); it reaches statistical significance with the tuned random survival forest on the reduced feature set.
- No gain for all patients combined, or for late recurrence (too few events for ~100 features).
- **Methylation** (+0.003) and **immune cell scores** (-0.001) add nothing beyond the genomic features; immune scores are highly correlated with existing gene-set scores (r = 0.83-0.93).
- The 57 gene-set scores carry about 9 independent signals (PCA: 9 PCs = 80% variance).
- Most stable genomic predictor (SHAP, all 25 splits): **fraction of genome altered** (genome instability), followed by proliferation gene sets.

Main results files: `results/phase2/er_late_summary.csv`, `er_late_tests.csv`, `compare_summary.csv`, `compare_tests.csv`, `shap_stability_CLIN_ALL.csv`, `automl_summary.csv`, `multiomics_*_summary.csv`, `fig_pca_redundancy.png`.

## Folder structure

```
METABRIC_Recurrence_Multiomics/
├── README.md, requirements.txt, run_all.sh, .gitignore
├── data/
│   ├── Breast_Cancer_METABRIC.csv            main clinical table (2,509 patients, 34 columns)
│   ├── brca_metabric/                        raw cBioPortal METABRIC study (1.3 GB, not in GitHub - see below)
│   ├── brca_tcga_pan_can_atlas_2018/         raw cBioPortal TCGA-BRCA study (1.5 GB, not in GitHub)
│   ├── gene_sets/                            Hallmark .gmt, Desmedt modules, MCP-counter markers
│   └── processed/
│       ├── metabric_recurrence_multiomics.csv   FINAL DATASET: 2,509 patients x 141 columns
│       ├── tcga_brca_features.csv               TCGA-BRCA external validation features
│       └── feature_dictionary.csv               what every engineered column means
├── src/                                      all scripts (run in the order of run_all.sh)
├── results/
│   ├── pilot/                                Phase 1: molecular labels already in the clinical table
│   ├── phase2/                               Phase 2: gene-level features, PCA, SHAP, AutoML, multi-omics, tuned survival
│   └── external_tcga/                        external validation on TCGA-BRCA
└── docs/
    ├── Phase2_Methods_and_Findings (.md/.docx)   full methods, rationale, results, interpretation, limitations
    └── Phase1_Pilot_Report (.md/.docx)           Phase 1 pilot: molecular labels from the clinical table
```

### The final dataset: `metabric_recurrence_multiomics.csv`

One row per patient, joined on `Patient ID` (MB-xxxx). Columns:

| Block | Prefix | Count | Source |
|---|---|---|---|
| Original clinical table | (as in METABRIC) | 34 | cBioPortal clinical data |
| Hallmark gene-set scores | `HM_` | 50 | MSigDB Hallmark (Liberzon 2015), mean expression z-score |
| Desmedt breast modules | `DM_` | 7 | Desmedt et al. 2008, weighted module scores |
| Copy-number drivers | `CNA_` | 11 | 9 driver amplifications, tumour-suppressor deep deletion, fraction of genome altered |
| Mutation drivers | `MUT_` | 9 | 8 driver genes (non-silent mutations) + mutation burden |
| Methylation | `METH_` | 20 | RRBS promoter methylation: global level, within-tumour instability, module and key genes |
| Immune / stromal cells | `IMM_` | 10 | MCP-counter (Becht 2016) cell-population scores |

Not every patient has every data type (expression 1,980; copy number 1,985; mutations 2,433; methylation 1,418); missing blocks are left empty. The modelling cohort is the 1,295 patients with a recorded recurrence outcome and complete clinical + genomic data.

## Getting the data

1. **METABRIC study (cBioPortal):** download `brca_metabric.tar.gz` from https://www.cbioportal.org/datasets ("Breast Cancer (METABRIC, Nature 2012 & Nat Commun 2016)") and unzip it into `data/brca_metabric/`.
2. **MSigDB Hallmark gene sets:** free login at https://www.gsea-msigdb.org/gsea/msigdb/human/collections.jsp, download "H: hallmark gene sets", Gene Symbols (`h.all.v2026.1.Hs.symbols.gmt`), into `data/gene_sets/`. If your version number differs, update the file name in `src/build_features.py`.
3. **Desmedt modules:** `mod1.rda` from https://github.com/bhklab/genefu (data folder) into `data/gene_sets/`.
4. **MCP-counter markers:** `genes.txt` from https://github.com/ebecht/MCPcounter (Signatures folder) into `data/gene_sets/`.

5. **TCGA-BRCA (external validation):** download "Breast Invasive Carcinoma (TCGA, PanCancer Atlas)" (`brca_tcga_pan_can_atlas_2018.tar.gz`) from https://www.cbioportal.org/datasets and unzip into `data/brca_tcga_pan_can_atlas_2018/`.

`data/Breast_Cancer_METABRIC.csv` is the same clinical table used in the EGETC paper.

## How to run (Mac)

```bash
# 1. Open Terminal and go to the project folder
cd ~/Downloads/Paper1/METABRIC_Recurrence_Multiomics

# 2. Create and activate a virtual environment with Python 3.11
#    (Python 3.13/3.14 lack prebuilt packages for some libraries; pip then fails
#    trying to compile SciPy). Install 3.11 once with: brew install python@3.11
python3.11 -m venv .venv
source .venv/bin/activate

# 3. Install packages (LightGBM on Mac needs OpenMP: brew install libomp)
pip install --upgrade pip
pip install -r requirements.txt

# 4. Run everything except the slow AutoML steps (~10-20 minutes)
QUICK=1 bash run_all.sh

# 5. Or run the full pipeline including AutoML (~1-1.5 hours)
bash run_all.sh
```

**Two ways to run, chosen automatically by `run_all.sh`:**

- **From the processed data (no download needed).** The shared project includes the processed datasets (`data/processed/metabric_recurrence_multiomics.csv`, `data/processed/tcga_brca_features.csv`). If the raw cBioPortal folders are not present, `run_all.sh` skips the feature-building steps and runs every model, test, and figure from these files. This reproduces all reported results.
- **From scratch (raw data downloaded).** If `data/brca_metabric/` (and optionally `data/brca_tcga_pan_can_atlas_2018/`) are present, `run_all.sh` first rebuilds all feature tables from the raw files, then runs the models. Rebuilding gives the same processed datasets.

To run one step only: `cd src` then e.g. `python model_er_late.py`. Scripts skip work that is already saved in `results/**/ck_*` checkpoint folders; delete those folders to recompute from scratch.

## Methods summary

- **Endpoints:** binary recurrence (relapse-free status) and time to recurrence (relapse-free months, censoring handled by Cox models); ER+ subgroup; landmark analysis for recurrence after 5 years.
- **Models:** penalised Cox (scikit-survival), L2 logistic regression, LightGBM, FLAML and MLJAR AutoML. Cox penalty and RSF leaf size tuned inside training folds; other models use fixed settings or short AutoML budgets.
- **Validation:** 5-fold x 5-repeat stratified CV (Cox / fixed models), 5-fold CV (AutoML); corrected repeated k-fold t-test (Bouckaert & Frank 2004) or Nadeau-Bengio corrected t-test, Holm adjustment.
- **Interpretability:** TreeSHAP on LightGBM in every test fold, with top-15 stability; SHAP-based selection also tested inside training folds.

## Script index (`src/`)

| Script | What it does |
|---|---|
| `extract_desmedt_modules.py` | Converts Desmedt modules from R format to CSV |
| `build_features.py` | Builds gene-set scores, copy-number and mutation features; creates the final dataset |
| `build_methylation_features.py` | Adds methylation features |
| `build_immune_features.py` | Adds MCP-counter immune / stromal scores |
| `pca_expression_scores.py` | PCA and redundancy clusters of the 57 gene-set scores |
| `model_compare_shap.py` | Feature-set comparison (LR, LightGBM, Cox) + SHAP stability |
| `model_er_late.py` | ER+ and late-recurrence Cox analyses |
| `model_multiomics.py` | Tests methylation and immune blocks |
| `model_tuned_survival.py` | Tuned Cox and random survival forest, reduced feature set, 10x5 CV |
| `build_tcga_features.py` | Builds TCGA-BRCA features with identical definitions + harmonized clinical features |
| `validate_tcga.py` | Trains on METABRIC, tests on TCGA, bootstrap confidence intervals |
| `run_automl_phase2.py` | MLJAR + FLAML AutoML on the feature sets |
| `run_recurrence_molecular.py`, `run_survival_framing.py` | Phase 1 pilot (molecular labels from the clinical table) |

## Next steps

1. Deep survival model (e.g. DeepSurv) with careful tuning, compared with the tuned Cox and random survival forest.
2. Wider penalty grid and elastic-net Cox.
3. Calibration of the ER+ survival models (calibration curves, integrated Brier score at 5 and 10 years).
4. Clinical interpretation of the leading molecular features (genomic instability, proliferation, oxidative stress).
5. Validation in a larger external cohort with longer follow-up (TCGA is directionally supportive but underpowered).

## Key references

Curtis 2012 (Nature); Pereira 2016 (Nat Commun); Rueda 2019 (Nature); Desmedt 2008 (Clin Cancer Res); Liberzon 2015 (Cell Systems); Becht 2016 (Genome Biol); Batra 2021 (Nat Commun); Ali 2016 (PLoS Med); Sestak 2018 (JAMA Oncol); Dowsett 2013 (JCO); Nadeau & Bengio 2003 (Mach Learn); Bouckaert & Frank 2004 (PAKDD).
