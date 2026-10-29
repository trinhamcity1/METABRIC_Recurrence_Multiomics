# RSNet pilot — pre-registered settings

Written 2026-10-14, **before any network was learned and before any RSNet result was seen.**
Source request: `Monti_RSNet_pilot_runbook.md` (S. Monti, Oct 14 2026).
Any change after this point is recorded in `DEVIATIONS.md` with date and reason.

## Question

Do RSNet network-module scores add recurrence-prediction value beyond the gene-set scores already used in the study, in all patients and in ER-negative patients?

## Data and cohort (unchanged from the study)

- METABRIC modelling cohort: 1,295 patients, 545 recurrences. ER+ 996 (408 events); ER− 299 (137 events).
- Endpoint: relapse-free months / relapse-free status (as in the study).
- External: TCGA-BRCA PanCancer Atlas, progression-free interval, same exclusions as `build_tcga_features.py` (female, not M1, no neoadjuvant therapy).

## Gene universe and network input

- Expression: METABRIC `data_mrna_illumina_microarray.txt` (log2 intensity); TCGA `data_mrna_seq_v2_rsem.txt` transformed to log2(RSEM + 1). Duplicate HGNC symbols averaged.
- Universe: HGNC symbols present in both cohorts with non-zero variance in both.
- Top-variance genes chosen **in the network-learning cohort only**: 1,000 (primary), 500 (sensitivity).
- Each gene z-scored within each cohort separately (no outcome data used).
- Direction A (primary, for METABRIC CV): network learned on TCGA-BRCA expression only (all primary-tumour RNA-seq samples).
- Direction B (for TCGA external check): network learned on METABRIC expression only (the 1,295 modelling patients).

## RSNet settings

- Package: montilab/RSNet, commit `f8c2f360e613b0759d9c72731fe2ecd712f0070b`. RHugin not installed (not needed).
- `ensemble_ggm(num_iteration = 100, boot = TRUE, estimate_CI = TRUE, method = "D-S_NW_SL")`; timing pilot with 20 iterations first.
- `consensus_net_ggm(CI = 0.95, filter = "fdr", threshold = 0.05)`.
- `community_detection(method = "leiden", resolution = 1, small_community = 10, mode = "total")`.
- If 1,000 genes x 100 iterations is impractical on the available machine, the run uses 500 genes as primary; this is logged.
- Leiden in igraph requires non-negative edge weights. If the consensus network carries signed weights (partial correlations), absolute values are used as weights; this is logged.

## Module scores

- Primary: module eigengene = first principal component of the module's genes, **fitted on training patients only** in every CV split and applied to test patients; sign set so the score correlates positively with the module's mean expression (in training data).
- Sensitivity: mean z-score of module genes (no fitting).
- Random-module control: same number of modules with the same sizes, genes drawn at random from the same top-variance pool, 20 draws (seeds 1–20), scored identically. CLIN+RAND C-index per split = mean over the 20 draws.

## Feature sets

| ID | Features |
|---|---|
| CLIN | 22 clinical features of the study |
| CLIN+GS | + 57 gene-set scores (50 MSigDB Hallmark + 7 Desmedt modules) — comparator |
| CLIN+NET | + RSNet module scores |
| CLIN+GS+NET | + both |
| CLIN+RAND | + random-module scores |

Copy-number and mutation driver features are **not** included in any set, because RSNet modules are expression-based and the comparator is the expression gene-set block.

## Models, splits, tuning (unchanged from the study's tuned analysis)

- Splits: `RepeatedStratifiedKFold(n_splits = 5, n_repeats = 10, random_state = 42)` within each population, stratified on the event indicator → 50 splits.
- Primary model: ridge Cox (scikit-survival), penalty from {1, 10, 100, 1000} by inner 3-fold CV on the training fold (C-index).
- Secondary model: random survival forest, 150 trees, min leaf from {15, 40} by out-of-bag C-index. The random-module control is run for Cox only (compute cost).
- Populations: All, ER− (focus), ER+ (reference).

## Metrics and tests

- Primary metric: Harrell's C-index on the held-out fold.
- For each comparison: mean ΔC, range, splits won (x/50), corrected repeated k-fold t-test (k = 5, r = 10, test/train = 1/4; Nadeau–Bengio-type correction as in the study), Holm-adjusted p.
- **Primary comparisons (Holm across these 6, ridge Cox, 1,000 genes, eigengene scoring, direction A modules):**
  1. CLIN+NET vs CLIN+GS, All
  2. CLIN+NET vs CLIN+GS, ER−
  3. CLIN+GS+NET vs CLIN+GS, All
  4. CLIN+GS+NET vs CLIN+GS, ER−
  5. CLIN+NET vs CLIN+RAND, All
  6. CLIN+NET vs CLIN+RAND, ER−
- Secondary (no multiplicity claims): same comparisons in ER+; RSF; 500-gene run; mean-z scoring; time-dependent AUC at 60 months (Cox).
- ER− has about 137 events; confidence intervals are reported and small differences are not over-interpreted.

## External check (TCGA-BRCA)

- Direction B modules (learned on METABRIC). Ridge Cox tuned and fitted on all METABRIC patients of the population, applied once to TCGA.
- Clinical block: the 6 harmonized clinical features of the study's external validation (age, T, N, ER, ERBB2 amplification, radiotherapy), since TCGA lacks grade, NPI, node counts and treatment flags.
- Eigengene PCA fitted on METABRIC, applied to TCGA (each cohort z-scored separately).
- Report C-index for CLIN+GS, CLIN+NET, CLIN+GS+NET; ΔC with 1,000 paired bootstrap resamples; All, and ER− if it has at least 20 events.

## Biology readout

- Top 3–5 modules ranked by mean absolute standardized Cox coefficient in CLIN+NET across the 50 splits (All and ER−), with how often each is in the top 5 (stability).
- For each: size, top hub genes (highest degree in the consensus network), hazard direction (sign of coefficient), Hallmark overlap (hypergeometric test, universe = the top-variance gene pool, BH-adjusted).

## Rules

- No tuning to make the result "work". A null result is reported as such.
