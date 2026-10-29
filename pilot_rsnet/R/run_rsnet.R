# RSNet pilot, steps 3: learn network + modules (run on your Mac in R >= 4.3).
#
# Usage (from the pilot_rsnet folder, in Terminal):
#   Rscript R/install_rsnet.R                         # once
#   Rscript R/run_rsnet.R A_TCGA_top1000 20           # timing pilot: direction A, 20 iterations
#   Rscript R/run_rsnet.R A_TCGA_top1000 100          # full run, direction A
#   Rscript R/run_rsnet.R B_METABRIC_top1000 100      # full run, direction B
#   (500-gene sensitivity: A_TCGA_top500 / B_METABRIC_top500)
#
# Outputs (pilot_rsnet/rsnet_output/<input>_it<N>/):
#   modules.csv        gene, module  (modules >= 10 genes after Leiden)
#   degree.csv         gene, degree in the consensus network
#   edges.csv          consensus network edge list with attributes
#   summary.json       settings, runtime, edges, modules, sizes, weight handling, session info
#
# Settings follow PREREGISTRATION.md. Do not change them without logging in DEVIATIONS.md.

suppressPackageStartupMessages({
  library(RSNet); library(igraph); library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
input_id <- ifelse(length(args) >= 1, args[1], "A_TCGA_top1000")
n_iter   <- ifelse(length(args) >= 2, as.integer(args[2]), 20L)
n_cores  <- max(1L, parallel::detectCores() - 1L)

in_file <- file.path("inputs", paste0("rsnet_input_", input_id, ".csv"))
out_dir <- file.path("rsnet_output", paste0(input_id, "_it", n_iter))
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

X <- as.matrix(read.csv(in_file, row.names = 1, check.names = FALSE))
cat(sprintf("Input %s: %d samples x %d genes | iterations %d | cores %d\n",
            input_id, nrow(X), ncol(X), n_iter, n_cores))

set.seed(2026)
t0 <- Sys.time()
ens <- capture_all(ensemble_ggm(dat = X,
                                num_iteration = n_iter,
                                boot = TRUE,
                                sub_ratio = NULL,
                                sample_class = NULL,
                                correlated = FALSE,
                                cluster_ratio = 1,
                                estimate_CI = TRUE,
                                method = "D-S_NW_SL",
                                n_cores = n_cores))
t_ens <- as.numeric(difftime(Sys.time(), t0, units = "mins"))
cat(sprintf("ensemble_ggm done in %.1f min\n", t_ens))

cons <- consensus_net_ggm(ggm_networks = ens, CI = 0.95, filter = "fdr", threshold = 0.05)
ig <- cons$consensus_network
cat(sprintf("Consensus network: %d nodes, %d edges\n", vcount(ig), ecount(ig)))

# Leiden in igraph needs non-negative weights. Pre-registered rule: if weights are signed,
# use absolute values (logged in summary.json).
weight_note <- "unweighted"
if (is_weighted(ig)) {
  w <- E(ig)$weight
  if (any(w < 0, na.rm = TRUE)) {
    E(ig)$weight <- abs(w); weight_note <- "signed weights -> absolute values used for Leiden"
  } else weight_note <- "non-negative weights used as-is"
}

deg <- degree(ig)
write.csv(data.frame(gene = names(deg), degree = as.integer(deg)),
          file.path(out_dir, "degree.csv"), row.names = FALSE)
write.csv(igraph::as_data_frame(ig, what = "edges"), file.path(out_dir, "edges.csv"), row.names = FALSE)

comm <- community_detection(ig = ig, mode = "total", vertex_symbol = "name",
                            method = "leiden", resolution = 1, small_community = 10)
g2 <- comm$ig
memb <- data.frame(gene = V(g2)$name, module = paste0("M", V(g2)$community), stringsAsFactors = FALSE)
memb <- memb[!is.na(V(g2)$community), ]
write.csv(memb, file.path(out_dir, "modules.csv"), row.names = FALSE)
sizes <- sort(table(memb$module), decreasing = TRUE)

summary <- list(
  input = input_id, samples = nrow(X), genes = ncol(X), iterations = n_iter, cores = n_cores,
  rsnet_version = as.character(packageVersion("RSNet")),
  rsnet_commit_expected = "f8c2f360e613b0759d9c72731fe2ecd712f0070b",
  runtime_ensemble_min = round(t_ens, 1),
  runtime_total_min = round(as.numeric(difftime(Sys.time(), t0, units = "mins")), 1),
  consensus_nodes = vcount(ig), consensus_edges = ecount(ig),
  weight_handling = weight_note,
  n_modules = length(sizes), module_sizes = as.integer(sizes),
  genes_in_modules = nrow(memb),
  modularity = comm$modularity,
  settings = list(boot = TRUE, estimate_CI = TRUE, method = "D-S_NW_SL", CI = 0.95,
                  filter = "fdr", threshold = 0.05, community = "leiden", resolution = 1,
                  small_community = 10),
  session = paste(capture.output(sessionInfo()), collapse = "\n"))
write_json(summary, file.path(out_dir, "summary.json"), auto_unbox = TRUE, pretty = TRUE)
cat(sprintf("Modules: %d (sizes %s) | genes in modules: %d\nSaved to %s\n",
            length(sizes), paste(as.integer(sizes), collapse = ", "), nrow(memb), out_dir))
