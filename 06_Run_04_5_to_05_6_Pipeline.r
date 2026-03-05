#!/usr/bin/env Rscript

# 06_Run_04_5_to_05_6_Pipeline.r
# Master runner for:
#   1) 04_5_LLM_Regression.r
#   2) 05_5_Precompute_Predictions.r
#   3) 05_6_LMM_Results.r
#
# Notes:
# - 04_5 already has resume-friendly skipping via SKIP_EXISTING_MODELS <- TRUE.
# - This script preserves that behavior and runs the full chain sequentially.

set.seed(123)

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

# Shared defaults (override via shell env vars as needed).
if (Sys.getenv("MODEL_DIR_04_5") == "") Sys.setenv(MODEL_DIR_04_5 = "models_04_5")
if (Sys.getenv("OUTPUT_DIR_05_5") == "") Sys.setenv(OUTPUT_DIR_05_5 = "results_05_5")
if (Sys.getenv("OUTPUT_DIR_05_6") == "") Sys.setenv(OUTPUT_DIR_05_6 = "results_05_6")

log_msg("Pipeline start: 04_5 -> 05_5 -> 05_6")
log_msg(sprintf("MODEL_DIR_04_5=%s", Sys.getenv("MODEL_DIR_04_5")))
log_msg(sprintf("OUTPUT_DIR_05_5=%s", Sys.getenv("OUTPUT_DIR_05_5")))
log_msg(sprintf("OUTPUT_DIR_05_6=%s", Sys.getenv("OUTPUT_DIR_05_6")))

run_step("04_5_LLM_Regression.r", "Step 1 / 3: model fitting")
run_step("05_5_Precompute_Predictions.r", "Step 2 / 3: precompute predictions")
run_step("05_6_LMM_Results.r", "Step 3 / 3: result tables and plots")

log_msg("Pipeline complete.")
