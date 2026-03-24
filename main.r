#!/usr/bin/env Rscript

# main.r
# Orchestrator for the Data Processing and Model Fitting pipeline.
#
# Reads config.yaml, creates a timestamped output directory,
# creates data subsets if requested, and then sequentially runs the pipeline.

suppressPackageStartupMessages({
  library(yaml)
})

`%||%` <- function(a, b) {
  if (!is.null(a) && length(a) > 0 && !all(is.na(a))) a else b
}

# Prefer project-local R library when present.
if (dir.exists(".r_libs")) {
  .libPaths(c(".r_libs", .libPaths()))
}

log_msg <- function(txt) {
  message(sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), txt))
}

run_step <- function(script_path, step_name) {
  if (!file.exists(script_path)) {
    stop(sprintf("Missing script for %s: %s", step_name, script_path))
  }
  log_msg(sprintf("Starting %s (%s)", step_name, script_path))
  source(script_path, local = new.env(parent = globalenv()))
  log_msg(sprintf("Finished %s", step_name))
}

# 1. Load config
config_path <- "config.yaml"
if (!file.exists(config_path)) stop("config.yaml not found!")
config <- yaml::read_yaml(config_path)

# 2. Setup Output Directory
timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
run_dir <- file.path(config$output_base_dir, paste0("run_", timestamp))
dir.create(run_dir, recursive = TRUE, showWarnings = FALSE)

models_dir <- file.path(run_dir, "models")
tables_dir <- file.path(run_dir, "tables")
plots_dir <- file.path(run_dir, "plots")

dir.create(models_dir, showWarnings = FALSE)
dir.create(tables_dir, showWarnings = FALSE)
dir.create(plots_dir, showWarnings = FALSE)

log_msg(sprintf("Created run directory: %s", run_dir))

# Copy config to run directory for provenance
file.copy(config_path, file.path(run_dir, "config.yaml"))

# 3. Determine Dataset
dataset_type <- config$dataset_type
input_data <- config$data_paths[[dataset_type]]

if (is.null(input_data) || !file.exists(input_data)) {
  stop(sprintf("Input data not found for dataset_type '%s': %s", dataset_type, input_data))
}

# 4. Handle Subsetting
if (isTRUE(config$use_subset)) {
  log_msg("use_subset is TRUE, invoking subset creator...")
  subset_engine <- tolower(config$subset_engine %||% "python")
  py_script <- "02b_Create_Analysis_Subset.py"
  r_script <- "02b_Create_Analysis_Subset.r"

  if (subset_engine == "python" && file.exists(py_script)) {
    py_bin <- Sys.getenv("PYTHON_BIN", "python3")
    subset_cmd <- sprintf(
      "%s %s %s %s",
      shQuote(py_bin),
      shQuote(py_script),
      shQuote(config_path),
      shQuote(run_dir)
    )
    log_msg(sprintf("Running subset creator (python): %s", subset_cmd))
    ret <- system(subset_cmd)
  } else {
    subset_cmd <- sprintf(
      "Rscript %s %s %s",
      shQuote(r_script),
      shQuote(config_path),
      shQuote(run_dir)
    )
    log_msg(sprintf("Running subset creator (R): %s", subset_cmd))
    ret <- system(subset_cmd)
  }

  if (ret != 0) stop("Subset creation failed.")
  
  # Update input data to point to the new subset
  input_data <- file.path(run_dir, "balanced_subset.parquet")
  log_msg(sprintf("Using subset for analysis: %s", input_data))
} else {
  log_msg(sprintf("Using full dataset for analysis: %s", input_data))
}

# 5. Export Environment Variables for Downstream Scripts
Sys.setenv(
  INPUT_PARQUET_04 = input_data,
  MODEL_DIR_04 = models_dir,
  SUMMARY_DIR_04 = file.path(run_dir, "model_summaries"),
  AIC_REPORT_FILE_04 = file.path(run_dir, "model_comparison_aic.md"),
  OUTPUT_DIR_05 = run_dir,
  PLOT_DIR = plots_dir
)

# Optional threading optimizations
if (Sys.getenv("OMP_NUM_THREADS") == "") Sys.setenv(OMP_NUM_THREADS = "1")
if (Sys.getenv("OPENBLAS_NUM_THREADS") == "") Sys.setenv(OPENBLAS_NUM_THREADS = "1")
if (Sys.getenv("MKL_NUM_THREADS") == "") Sys.setenv(MKL_NUM_THREADS = "1")

# 6. Execute Pipeline
log_msg("Executing pipeline steps defined in config.yaml...")

steps <- config$run_steps
for (i in seq_along(steps)) {
  script_name <- steps[[i]]
  run_step(script_name, sprintf("Step %d: %s", i, script_name))
}

log_msg(sprintf("Pipeline complete. All results saved to %s", run_dir))
