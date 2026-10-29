# RSNet pilot: network modules vs gene-set scores for recurrence

Request: S. Monti, Oct 14 2026 — *do RSNet module scores add recurrence-prediction value beyond the gene-set scores, in all patients and in ER− patients?*
Settings are fixed in `PREREGISTRATION.md` (written before any result). Changes go in `DEVIATIONS.md`.

## Status

| Step | What | Where it runs | Status |
|---|---|---|---|
| 2 | Inputs: shared gene universe (17,072 genes), top 1,000/500 variable genes per learning cohort, within-cohort z-scores | analysis environment | done → `inputs/` |
| 3 | RSNet network + Leiden modules (direction A on TCGA, direction B on METABRIC) | **your Mac (R)** | to do |
| 4–6, 8 | In-fold eigengene scores, random-module control, CV comparison, biology readout | analysis environment | code ready and tested on stand-in modules |
| 7 | External check on TCGA with direction-B modules | analysis environment | code ready and tested |

Input facts: METABRIC 1,295 patients; TCGA 1,082 primary tumours; the TCGA and METABRIC top-1,000 gene lists share 350 genes.

## What to run on your Mac

Open Terminal:

```bash
cd ~/Downloads/Paper1/METABRIC_Recurrence_Multiomics/pilot_rsnet

# 1. Install R >= 4.3 if needed (https://cran.r-project.org/bin/macosx/), then once:
Rscript R/install_rsnet.R

# 2. Timing pilot (20 iterations, direction A, 1,000 genes)
Rscript R/run_rsnet.R A_TCGA_top1000 20

# 3. Full runs (100 iterations)
Rscript R/run_rsnet.R A_TCGA_top1000 100
Rscript R/run_rsnet.R B_METABRIC_top1000 100

# 4. Sensitivity (500 genes)
Rscript R/run_rsnet.R A_TCGA_top500 100
Rscript R/run_rsnet.R B_METABRIC_top500 100
```

Each run writes `rsnet_output/<input>_it<N>/` with `modules.csv`, `degree.csv`, `edges.csv`, `summary.json`.

- If the 20-iteration pilot suggests the 100-iteration run would take far too long, run the 500-gene inputs as primary and note it in `DEVIATIONS.md`.
- Keep the Mac awake during long runs (e.g. `caffeinate -i Rscript R/run_rsnet.R A_TCGA_top1000 100`).
- Nothing needs to be copied anywhere: the output lands in this folder, which the analysis environment can read.

## Then (analysis environment)

```bash
cd src
python 02_evaluate_cv.py --modules ../rsnet_output/A_TCGA_top1000_it100/modules.csv \
  --degree ../rsnet_output/A_TCGA_top1000_it100/degree.csv --pool A_TCGA_top1000 --tag A1000_eig
#   repeat until "ALL SPLITS DONE", then add --summarize
python 03_external_tcga.py --modules ../rsnet_output/B_METABRIC_top1000_it100/modules.csv \
  --pool B_METABRIC_top1000 --tag B1000_external
```

Secondary runs: `--scoring meanz` (mean-z scoring) and the `top500` modules.

Results land in `results/<tag>/`: `cv_summary.csv`, `primary_comparisons.csv` (the 6 Holm-adjusted comparisons), `secondary_comparisons.csv`, `biology_top_modules.csv`, and for the external check `external_cindex.csv`, `external_gain.csv`.
