#!/usr/bin/env bash
# Run the pipeline that produces every table and figure in the ACET 2026 manuscript.
# From the project folder:  bash run_all.sh
#
# Two modes, chosen automatically:
#   * Raw cBioPortal data present (data/brca_metabric, data/brca_tcga_pan_can_atlas_2018):
#     rebuilds every feature table from scratch, then runs all models.
#   * Raw data absent: skips the feature-building steps and runs all models from the
#     processed datasets in data/processed/.
#
# Manuscript analyses take ~40-60 minutes on 4 CPU cores (the tuned survival models
# dominate). Optional, NOT used in the manuscript:
#   RUN_EXTRA=1 bash run_all.sh   also runs AutoML (FLAML + MLJAR; descriptive binary
#                                 classification, ~1-2 h) and the Phase 1 pilot analyses.
# See docs/TABLE_FIGURE_MANIFEST.md for which script/output feeds each manuscript element.
set -euo pipefail
cd "$(dirname "$0")/src"
mkdir -p ../results/logs
LOG=../results/logs

run() {  # run a step, keep its log, stop on failure
  local name="$1"; shift
  echo "== $name"
  "$@" > "$LOG/${name}.log" 2>&1 || { echo "FAILED: $name (see results/logs/${name}.log)"; exit 1; }
}

if [ -d ../data/brca_metabric ]; then
  echo "== Raw METABRIC data found: rebuilding features from scratch"
  run 00_desmedt_modules    python extract_desmedt_modules.py
  run 01_build_features     python build_features.py
  run 02_methylation        python build_methylation_features.py
  run 03_immune             python build_immune_features.py
else
  echo "== Raw METABRIC data not found: using data/processed/metabric_recurrence_multiomics.csv"
fi
if [ -d ../data/brca_tcga_pan_can_atlas_2018 ]; then
  run 04_build_tcga         python build_tcga_features.py
else
  echo "== Raw TCGA data not found: using data/processed/tcga_brca_features.csv"
fi

run 05_cohort_tables        python make_cohort_tables.py
run 06_pca_redundancy       python pca_expression_scores.py
run 07_compare_shap         env SUMMARIZE=1 python model_compare_shap.py
run 08_er_late              python model_er_late.py
run 09_multiomics_meth_imm  python model_multiomics.py
run 10_multiomics_imm       env NO_METH=1 python model_multiomics.py
run 11_multiomics_meth      env NO_IMM=1 python model_multiomics.py

# Tuned survival models are checkpointed; each call runs until MAXSEC, then exits.
# Retry with a hard limit, and stop if a retry makes no progress.
echo "== 12_tuned_survival (checkpointed)"
prev=-1
for attempt in $(seq 1 200); do
  out=$(MAXSEC=${MAXSEC:-600} python model_tuned_survival.py 2>>"$LOG/12_tuned_survival.err") \
    || { echo "FAILED: tuned survival (see results/logs/12_tuned_survival.err)"; exit 1; }
  echo "$out"
  echo "$out" | grep -q "ALL SPLITS DONE" && break
  done_n=$(echo "$out" | sed -n 's/.*splits saved: \([0-9]*\).*/\1/p')
  if [ "$done_n" = "$prev" ]; then echo "FAILED: tuned survival made no progress"; exit 1; fi
  prev=$done_n
done
run 12_tuned_survival_summary env SUMMARIZE=1 python model_tuned_survival.py
run 13_tcga_transport       python validate_tcga.py
run 13b_tcga_no_mutcount    env EXCL_MUTCOUNT=1 python validate_tcga.py
run 14_ipcw_sensitivity     python ipcw_sensitivity.py
run 15_figure1              python make_figure1.py

if [ "${RUN_EXTRA:-0}" = "1" ]; then
  run 90_automl             env SUMMARIZE=1 python run_automl_phase2.py
  run 91_pilot_molecular    python run_recurrence_molecular.py
  run 92_pilot_survival     python run_survival_framing.py
fi
echo "Done. Results are in results/, logs in results/logs/."
