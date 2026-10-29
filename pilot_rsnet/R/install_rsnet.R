# One-time install of RSNet at the pinned commit (R >= 4.3). RHugin is NOT needed.
# Safe to re-run: packages already installed are skipped.
options(repos = c(CRAN = "https://cloud.r-project.org"))
need <- function(p) !requireNamespace(p, quietly = TRUE)
if (need("BiocManager")) install.packages("BiocManager")
if (need("Biobase")) BiocManager::install("Biobase", update = FALSE, ask = FALSE)
pkgs <- c("remotes", "SILGGM", "BDgraph", "orca", "igraph", "kinship2",
          "clustAnalytics", "proxy", "visNetwork", "DT", "jsonlite")
missing <- pkgs[vapply(pkgs, need, logical(1))]
if (length(missing)) install.packages(missing)
remotes::install_github("montilab/RSNet@f8c2f360e613b0759d9c72731fe2ecd712f0070b",
                        upgrade = "never", dependencies = TRUE)
library(RSNet); cat("RSNet", as.character(packageVersion("RSNet")), "installed\n")
