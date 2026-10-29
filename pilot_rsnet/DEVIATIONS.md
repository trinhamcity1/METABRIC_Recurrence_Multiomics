# RSNet pilot — deviations log

Record every change from `PREREGISTRATION.md` here, with date and reason, before looking at the affected results.

| Date | Change | Reason |
|---|---|---|
| 2026-10-14 | RSNet (steps 2–4 of the runbook) run on the user's Mac in R; evaluation (steps 5–8) run in the analysis environment. | The analysis environment cannot install R or reach CRAN/GitHub downloads. This is the runbook's stated fallback. |
| 2026-10-14 | RSNet runs completed on the Mac (R 4.5.1, RSNet 1.0.0 installed from the pinned commit via `remotes::install_github`; `devtools` 2.5 required `remotes`). 1,000 genes x 100 iterations took ~9–11 min per direction, so the pre-registered primary setting was kept (no switch to 500 genes). | Install fix only; no analysis change. |
| 2026-10-14 | The consensus networks carry no `weight` edge attribute, so Leiden ran unweighted; the pre-registered absolute-weight rule was not triggered. R printed "50 or more warnings" during each run; their content was not captured. | Logged for transparency; no analysis change. |
| 2026-10-14 | Secondary mean-z scoring run uses ridge Cox only (`--no-rsf`); the 500-gene sensitivity run is pending (not yet run in R). Decided after the primary run finished, before the mean-z results were seen. | Compute time; RSF is secondary and was reported for the primary eigengene run. |
