# RSNet pilot — results (filled runbook template)

Settings as in `PREREGISTRATION.md`; changes in `DEVIATIONS.md`. All numbers from `results/` and `rsnet_output/`.

```
RSNet commit:            f8c2f360e613b0759d9c72731fe2ecd712f0070b (RSNet 1.0.0, R 4.5.1, macOS, 11 cores)
Genes (p):               1000 (primary); 500-gene sensitivity not yet run
Iterations:              100      Runtime: 9.1 min (A, TCGA) / 10.8 min (B, METABRIC); 20-iteration pilot 1.4 min
Network A (TCGA):        edges 1,055, modules 21 (sizes 46, 44, 43, 42, 42, 42, 40, 39, 37, 36, 35, 34, 34, 32, 31, 31, 30, 26, 21, 19, 15); 719 genes in modules; modularity 0.85
Network B (METABRIC):    edges 2,085, modules 15 (sizes 103, 96, 96, 84, 78, 77, 70, 62, 60, 58, 56, 50, 26, 23, 14); 953 genes in modules; modularity 0.66

CV results, penalised Cox, C-index mean (sd), 50 splits (direction-A modules, eigengene scores)
Population | CLIN          | CLIN+GS       | CLIN+NET      | CLIN+GS+NET   | CLIN+RAND
All        | 0.653 (0.021) | 0.660 (0.024) | 0.654 (0.022) | 0.659 (0.023) | 0.654 (0.020)
ER-        | 0.666 (0.042) | 0.655 (0.043) | 0.649 (0.047) | 0.648 (0.043) | 0.646 (0.045)
ER+        | 0.632 (0.031) | 0.660 (0.030) | 0.636 (0.026) | 0.657 (0.028) | 0.634 (0.029)

Primary comparisons (Holm across these 6)
Comparison                 | Pop | mean dC | range            | splits won | p (corrected) | p (Holm)
NET vs GS                  | All | -0.006  | -0.035 to +0.030 | 15/50      | 0.37          | 1.00
NET vs GS                  | ER- | -0.006  | -0.075 to +0.056 | 20/50      | 0.72          | 1.00
GS+NET vs GS               | All | -0.001  | -0.017 to +0.024 | 18/50      | 0.71          | 1.00
GS+NET vs GS               | ER- | -0.007  | -0.032 to +0.039 | 13/50      | 0.38          | 1.00
NET vs RAND                | All | -0.001  | -0.017 to +0.010 | 26/50      | 0.82          | 1.00
NET vs RAND                | ER- | +0.003  | -0.026 to +0.025 | 33/50      | 0.64          | 1.00

External TCGA-BRCA (direction-B modules; 6 harmonized clinical features; All: 918 patients / 102 events; ER-: 223 / 34)
All: C-index CLIN+GS 0.672, CLIN+NET 0.636, CLIN+GS+NET 0.663
     dC NET vs GS -0.036 (95% CI -0.068 to -0.007); GS+NET vs GS -0.009 (-0.023 to +0.003)
ER-: C-index CLIN+GS 0.708, CLIN+NET 0.701, CLIN+GS+NET 0.716
     dC NET vs GS -0.007 (-0.067 to +0.049); GS+NET vs GS +0.008 (-0.010 to +0.028)

Top modules (CLIN+NET Cox, ranked by mean |standardized coefficient| across 50 splits)
Pop | Module | Size | Hub genes                              | Hazard direction        | In top 5 | Best Hallmark overlap
All | M7     | 44   | MS4A1, IL6, UBD, AQP9, FCER2           | higher score, lower risk  | 94%      | Interferon-gamma response (6 genes; Bonferroni p < 0.001)
All | M3     | 42   | ADCY5, HPX, NDP, TPSG1, FGFR3          | higher score, lower risk  | 96%      | none significant
All | M29    | 31   | CST5, CST6, MESP1, KCNF1, LRRC26       | higher score, higher risk | 44%      | none significant
All | M34    | 34   | LRRIQ1, CLGN, NMU, RNF183, DQX1        | higher score, higher risk | 74%      | none significant
All | M5     | 21   | PRSS33, SPDEF, TFF1, ROPN1B, TMEM45B   | higher score, higher risk | 70%      | Estrogen response early (4 genes; Bonferroni p = 0.22)
ER- | M7     | 44   | (as above)                             | higher score, lower risk  | 98%      | Interferon-gamma response
ER- | M5     | 21   | (as above)                             | higher score, higher risk | 84%      | Estrogen response early (n.s.)
```

## Secondary results

- **ER+ (reference):** CLIN+NET was worse than CLIN+GS (ΔC −0.024, 5/50 splits won, corrected p = 0.02); GS+NET vs GS −0.003; NET vs RAND +0.002.
- **Random survival forest:** same pattern. NET vs GS −0.007 (All), +0.005 (ER−), −0.019 (ER+); GS+NET vs GS +0.001 (All), −0.003 (ER−), −0.004 (ER+).
- **Time-dependent AUC at 60 months (Cox):** NET vs GS −0.010 (All), +0.004 (ER−), −0.031 (ER+).
- **Mean-z scoring instead of eigengenes (Cox):** NET vs GS −0.005 (All), +0.005 (ER−); GS+NET vs GS +0.002 (All), −0.004 (ER−); NET vs RAND +0.003 (All), +0.001 (ER−); all Holm p = 1.00.
- **500-gene sensitivity:** not yet run (requires two more RSNet runs in R).

## Reading

1. **Null result on the pre-registered question.** RSNet module scores did not add recurrence-prediction value beyond the gene-set scores in all patients or in ER− patients, either alone or combined with the gene-set scores (all six primary comparisons Holm p = 1.00, mean ΔC between −0.007 and +0.003).
2. **Network structure did not beat random modules.** Modules learned by RSNet performed the same as random gene blocks of the same sizes drawn from the same 1,000-gene pool (ΔC −0.001 in All, +0.003 in ER−).
3. **External check agrees.** On TCGA, METABRIC-learned modules were worse than gene-set scores in all patients (ΔC −0.036, CI excludes 0) and indistinguishable in ER− (34 events, wide CI).
4. **ER− is hard for every molecular block in this cohort.** In ER− patients the clinical model alone had the highest C-index (0.666); adding gene-set scores (0.655), network modules (0.649) or random modules (0.646) all lowered it. With 299 patients and 137 events, extra features mostly add variance.
5. **Biology.** The most stable module (M7; hubs MS4A1, IL6, UBD, AQP9, FCER2; in the top 5 in 94–98% of splits) is an immune / B-cell and interferon module associated with lower recurrence risk, consistent with known immune prognostic signal, but it did not translate into better discrimination than the gene-set scores, which already contain immune and interferon sets.

## Things that broke, changed, or surprised

- RSNet had to run on the Mac (no R / CRAN access in the analysis environment); install needed `remotes` because `devtools` 2.5 deprecated `install_github`.
- The full 1,000-gene, 100-iteration runs took only ~10 minutes each, so no reduction to 500 genes was needed.
- The consensus networks are sparse and unweighted, so Leiden ran unweighted. Network A: 1,055 edges among 1,000 genes, median degree 2, 129 isolated genes. Network B: 2,085 edges, median degree 4, 32 isolated genes.
- Several TCGA-learned modules contain genes with on/off expression in RNA-seq (for example cystatins, spermatogenesis-related genes). Top-variance selection on log2(RSEM+1) favours such genes, and the TCGA and METABRIC top-1,000 lists share only 350 genes. This may limit how well TCGA-learned modules transfer to METABRIC microarray data; a variance filter restricted to well-expressed genes, or learning and evaluating within the same platform, would be reasonable follow-ups (not pre-registered).
- R printed "50 or more warnings" in each run; their content was not captured.
