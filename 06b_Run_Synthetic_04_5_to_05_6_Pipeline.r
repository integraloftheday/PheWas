#!/usr/bin/env Rscript

# Synthetic smoke test runner for:
#   1) 04_4_LMM_Synthetic_Data.py
#   2) 04_5_LLM_Regression.r
#   3) 05_5_Precompute_Predictions.r
#   4) 05_6_LMM_Results.r
#   5) 07_Validate_Precompute_Outputs.py

set.seed(123)

if (dir.exists(".r_libs")) {
  .libPaths(c(".r_libs", .libPaths()))
}

log_msg <- function(txt) {
  message(sprintf("[%s] %s", format(Sys.time(), "%Y-%m-%d %H:%M:%S"), txt))
}

run_cmd <- function(cmd, step_name) {
  log_msg(sprintf("Starting %s", step_name))
  status <- system(cmd)
  if (!identical(status, 0L)) {
    stop(sprintf("%s failed with exit status %s", step_name, status))
  }
  log_msg(sprintf("Finished %s", step_name))
}

Sys.setenv(
  INPUT_PARQUET_04_5 = "processed_data/synthetic/LMM_analysis_synthetic_small.parquet",
  MODEL_DIR_04_5 = "models_04_5_synth",
  SUMMARY_DIR_04_5 = "model_summaries_04_5_synth",
  AIC_REPORT_FILE_04_5 = "model_comparison_aic_04_5_synth.md",
  OUTPUT_DIR_05_5 = "results_05_5_synth",
  OUTPUT_DIR_05_6 = "results_05_6_synth",
  RESUME_CHECKPOINTS_05_5 = "false",
  MEM_DIAGNOSTICS_05_5 = "false",
  NOTEBOOK_INLINE_05_6 = "false"
)

log_msg("Synthetic smoke test start")

run_cmd("./venv/bin/python 04_4_LMM_Synthetic_Data.py", "synthetic data generation")
source("06_Run_04_5_to_05_6_Pipeline.r", local = new.env(parent = globalenv()))
run_cmd("./venv/bin/python 07_Validate_Precompute_Outputs.py results_05_5_synth", "05_5 export validation")

log_msg("Synthetic smoke test complete")
