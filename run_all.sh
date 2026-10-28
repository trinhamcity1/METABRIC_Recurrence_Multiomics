#!/usr/bin/env bash
# Run the full pipeline in order. From the project folder:  bash run_all.sh
#
# Two modes, chosen automatically:
#   * Raw cBioPortal data present (data/brca_metabric, data/brca_tcga_pan_can_atlas_2018):
#     rebuilds every feature table from scratch, then runs all models.
#   * Raw data absent (e.g. from the shared zip): skips the feature-building steps and
#     runs all models from the processed datasets in data/processed/.
#
# Fast steps take ~10-20 minutes; AutoML adds ~45-60 minutes.
# Set QUICK=1 to skip AutoML:  QUICK=1 bash run_all.sh
set -e
cd "$(dirname "$0")/src"

if [ -d ../data/brca_metabric ]; then
  echo "== Raw METABRIC data found: rebuilding features from scratch"
  echo "== Step 0: Desmedt modules (R data -> CSV)";          python extract_desmedt_modules.py
  echo "== Step 1: build clinical + genomic feature table";    python build_features.py
  echo "== Step 2: add methylation features";                  python build_methylation_features.py
  echo "== Step 3: add immune / stromal cell scores";          python build_immune_features.py
else
  echo "== Raw METABRIC data not found: using data/processed/metabric_recurrence_multiomics.csv"
fi
if [ -d ../data/brca_tcga_pan_can_atlas_2018 ]; then
  echo "== Raw TCGA data found: rebuilding TCGA features";      python build_tcga_features.py
else
  echo "== Raw TCGA data not found: using data/processed/tcga_brca_features.csv"
fi

echo "== Step 4: PCA redundancy of the 57 gene-set scores";     python pca_expression_scores.py
echo "== Step 5: feature-set comparison + SHAP (5x5 CV)";      SUMMARIZE=1 python model_compare_shap.py
echo "== Step 6: ER+ and late-recurrence Cox models";          python model_er_late.py
echo "== Step 7: methylation + immune added";                  python model_multiomics.py
echo "== Step 7b: immune only (full cohort)";                  NO_METH=1 python model_multiomics.py
echo "== Step 7c: tuned Cox + random survival forest (10x5 CV)"
until python model_tuned_survival.py | grep -q "ALL SPLITS DONE"; do :; done
SUMMARIZE=1 python model_tuned_survival.py
echo "== Step 8: TCGA external validation";                    python validate_tcga.py

if [ "${QUICK:-0}" != "1" ]; then
  echo "== Step 9: AutoML (MLJAR + FLAML) - slow";            SUMMARIZE=1 python run_automl_phase2.py
  echo "== Pilot (Phase 1): molecular labels, AutoML - slow";  python run_recurrence_molecular.py
  echo "== Pilot (Phase 1): survival framing";                 python run_survival_framing.py
fi
echo "Done. Results are in results/."
