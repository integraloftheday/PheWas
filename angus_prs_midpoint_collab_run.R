#!/usr/bin/env Rscript
# Angus PRS midpoint collaborator run script
#
# Paste the whole file into a single notebook cell (R kernel) or run directly:
#   Rscript angus_prs_midpoint_collab_run.R
#
# Steps:
#   1. Score the collaborator midpoint PRS with PLINK via 02_plink_PGRS_Generator_all.py
#   2. Run association + PheWAS analysis via prs_midpoint_collab_analysis.R
#   3. Print the generated markdown summary
#
# W&B tracking logs only aggregate progress, plots, and result tables.
# No patient-level rows are uploaded.

# ==============================================================================
# CONFIGURATION — edit these before running
# ==============================================================================

score_pattern    <- "METAL_midp_all_pst_eff_a1_b0.5_phi1e-02_ALL"
output_dir       <- "results/angus_midpoint_prs_analysis"
ancestry_filter  <- "all"
score_ids_parquet <- "processed_data/ready_for_analysis.parquet"
nightly_parquet  <- "processed_data/ready_for_analysis.parquet"
phewas_parquet   <- "processed_data/master/master_phewas_wide.parquet"
covariates_parquet <- "processed_data/fitbit_cohort_covariates.parquet"
reuse_existing_score <- TRUE

enable_wandb   <- TRUE
wandb_project  <- "aou-prs-midpoint"
wandb_entity   <- ""
wandb_run_name <- paste0("prs-midpoint-", format(Sys.time(), "%Y%m%d-%H%M%S"))

options(width = 120)

# ==============================================================================
# PRE-FLIGHT CHECKS
# ==============================================================================

required_paths <- c(
  nightly_parquet,
  score_ids_parquet,
  phewas_parquet,
  "prs_midpoint_collab_analysis.R",
  "wandb_progress_logger.py",
  "02_plink_PGRS_Generator_all.py"
)

missing_paths <- required_paths[!file.exists(required_paths)]
if (length(missing_paths) > 0) {
  stop("Missing required files: ", paste(missing_paths, collapse = ", "))
}

score_file <- file.path(
  "processed_data", "PGRS", score_pattern,
  paste0(score_pattern, "_PGRS.txt")
)
weights_file <- file.path("analysis_inputs", paste0(score_pattern, ".txt"))
if (!reuse_existing_score || !file.exists(score_file)) {
  if (!file.exists(weights_file)) {
    stop("Missing weight file required for scoring: ", weights_file)
  }
}

if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite")

if (!file.exists(covariates_parquet)) {
  message(
    "Covariates parquet not found at ", covariates_parquet,
    "; the analysis script will fall back automatically."
  )
}

resolve_covariates_path_runner <- function(explicit_path = "") {
  candidates <- c(
    explicit_path,
    "processed_data/fitbit_cohort_covariates.parquet",
    "processed_data/master/master_covariates_only.parquet",
    "processed_data/master/master_phewas_wide.parquet"
  )
  candidates <- candidates[nzchar(candidates)]
  hit <- candidates[file.exists(candidates)]
  if (length(hit) == 0) {
    stop("No covariates parquet found. Checked: ", paste(candidates, collapse = ", "))
  }
  hit[[1]]
}

read_parquet_ids <- function(path) {
  as.character(unique(as.data.frame(arrow::read_parquet(path, col_select = "person_id"))$person_id))
}

read_score_ids <- function(path) {
  score_df <- read.table(path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  if ("IID" %in% names(score_df)) return(as.character(unique(score_df$IID)))
  if ("person_id" %in% names(score_df)) return(as.character(unique(score_df$person_id)))
  stop("Score file must contain IID or person_id column: ", path)
}

print_cohort_preview <- function(nightly_parquet, score_file, covariates_path, phewas_parquet) {
  if (!requireNamespace("arrow", quietly = TRUE)) install.packages("arrow")

  sleep_ids <- read_parquet_ids(nightly_parquet)
  score_ids <- read_score_ids(score_file)
  cov_ids <- read_parquet_ids(covariates_path)
  phewas_ids <- read_parquet_ids(phewas_parquet)

  sleep_genetics_ids <- intersect(sleep_ids, score_ids)
  association_ids <- intersect(sleep_genetics_ids, cov_ids)
  phewas_overlap_ids <- intersect(sleep_genetics_ids, phewas_ids)

  preview_df <- data.frame(
    step = c(
      "Sleep phenotype cohort",
      "Scored genetics cohort",
      "Selected association covariates cohort",
      "PheWAS cohort",
      "Sleep + scored genetics",
      "Sleep + scored genetics + association covariates",
      "Sleep + scored genetics + PheWAS"
    ),
    n = c(
      length(sleep_ids),
      length(score_ids),
      length(cov_ids),
      length(phewas_ids),
      length(sleep_genetics_ids),
      length(association_ids),
      length(phewas_overlap_ids)
    )
  )

  cat("\n============================================================\n")
  cat("Quick cohort preview\n")
  cat("============================================================\n")
  cat("Association covariates source: ", covariates_path, "\n", sep = "")
  print(preview_df, row.names = FALSE)
  cat("============================================================\n\n")

  append_progress_event(
    "notebook",
    "cohort_preview",
    status = "completed",
    metrics = list(
      sleep_cohort = length(sleep_ids),
      scored_genetics = length(score_ids),
      association_covariates = length(cov_ids),
      sleep_scored_genetics = length(sleep_genetics_ids),
      association_overlap = length(association_ids),
      phewas_overlap = length(phewas_overlap_ids)
    ),
    details = list(covariates_path = covariates_path)
  )
}

# ==============================================================================
# PROGRESS FILE SETUP
# ==============================================================================

dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
progress_file <- file.path(output_dir, "wandb_progress.jsonl")
logger_log    <- file.path(output_dir, "wandb_logger.log")
if (file.exists(progress_file)) file.remove(progress_file)
if (file.exists(logger_log))    file.remove(logger_log)

append_progress_event <- function(stage, event, status = "running",
                                  metrics = list(), details = list()) {
  payload <- c(
    list(
      timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      stage = stage,
      event = event,
      status = status
    ),
    if (length(metrics) > 0) list(metrics = metrics) else list(),
    if (length(details) > 0) list(details = details) else list()
  )
  cat(
    jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null"),
    "\n",
    file = progress_file,
    append = TRUE
  )
}

# ==============================================================================
# START W&B LOGGER (background process)
# ==============================================================================

if (enable_wandb) {
  wandb_ready <- system2(
    "python",
    c("-c", "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('wandb') else 1)"),
    stdout = FALSE, stderr = FALSE
  ) == 0

  if (!wandb_ready) {
    install_status <- system2("python", c("-m", "pip", "install", "--quiet", "wandb"))
    if (install_status != 0) stop("Failed to install wandb.")
  }

  logger_args <- c(
    "wandb_progress_logger.py",
    "--progress-file", progress_file,
    "--out-dir", output_dir,
    "--project", wandb_project,
    "--run-name", wandb_run_name
  )
  if (nzchar(wandb_entity)) logger_args <- c(logger_args, "--entity", wandb_entity)

  system2("python", args = logger_args, stdout = logger_log, stderr = logger_log, wait = FALSE)
  append_progress_event(
    "notebook", "logger_started", status = "completed",
    details = list(project = wandb_project, run_name = wandb_run_name)
  )
  message("Started W&B logger; logs -> ", logger_log)
}

# ==============================================================================
# PIPELINE EXECUTION
# ==============================================================================

Sys.setenv(
  PGRS_ANCESTRY_FILTER   = ancestry_filter,
  PGRS_FILE_PATTERN      = score_pattern,
  PGRS_ID_SOURCE_PARQUET = score_ids_parquet,
  WANDB_PROGRESS_FILE    = progress_file
)

tryCatch({

  # ---- Step 1: Score PRS ----
  if (reuse_existing_score && file.exists(score_file)) {
    message("Reusing existing score file: ", score_file)
    append_progress_event(
      "notebook", "score_phase_skipped", status = "completed",
      details = list(score_file = score_file, reason = "existing_score_file_reused")
    )
  } else {
    append_progress_event("notebook", "score_phase_started",
                          details = list(score_pattern = score_pattern))
    score_status <- system2("Rscript", "02_plink_PGRS_Generator_all.py")
    if (score_status != 0) stop("PRS scoring failed.")
    append_progress_event("notebook", "score_phase_completed", status = "completed",
                          details = list(score_file = score_file))
  }

  # ---- Step 2: Association + PheWAS ----
  selected_covariates_path <- resolve_covariates_path_runner(covariates_parquet)
  print_cohort_preview(
    nightly_parquet = nightly_parquet,
    score_file = score_file,
    covariates_path = selected_covariates_path,
    phewas_parquet = phewas_parquet
  )

  analysis_args <- c(
    "prs_midpoint_collab_analysis.R",
    paste0("--nightly_parquet=",    nightly_parquet),
    paste0("--phewas_parquet=",     phewas_parquet),
    paste0("--covariates_parquet=", selected_covariates_path),
    paste0("--score_file=",         score_file),
    paste0("--ancestry_tsv=",       file.path("processed_data", "PGRS", "shared", "ancestry_preds.tsv")),
    paste0("--out_dir=",            output_dir)
  )

  append_progress_event("notebook", "analysis_phase_started")
  analysis_status <- system2("Rscript", analysis_args)
  if (analysis_status != 0) stop("Collaborator analysis failed.")

  append_progress_event("notebook", "run_finished", status = "completed",
                        details = list(output_dir = output_dir))

}, error = function(e) {
  append_progress_event("notebook", "run_failed", status = "failed",
                        details = list(message = conditionMessage(e)))
  stop(e)
})

# ==============================================================================
# PRINT SUMMARY
# ==============================================================================

summary_path <- file.path(output_dir, "summary.md")
if (!file.exists(summary_path)) stop("Expected summary.md not found at ", summary_path)
cat(readLines(summary_path), sep = "\n")

show_png_inline <- function(path) {
  if (!file.exists(path)) return(invisible(FALSE))
  if (!interactive() && !nzchar(Sys.getenv("JPY_PARENT_PID"))) return(invisible(FALSE))
  if (!requireNamespace("IRdisplay", quietly = TRUE)) return(invisible(FALSE))
  IRdisplay::display_png(file = path)
  invisible(TRUE)
}

plot_paths <- c(
  file.path(output_dir, "plots", "cohort_flow.png"),
  file.path(output_dir, "plots", "association_forest_per_sd.png"),
  file.path(output_dir, "plots", "midpoint_by_prs_tertile.png"),
  file.path(output_dir, "plots", "phewas_manhattan.png"),
  file.path(output_dir, "plots", "phewas_volcano.png")
)

for (plot_path in plot_paths) {
  shown <- show_png_inline(plot_path)
  if (!shown) message("Plot written: ", plot_path)
}
