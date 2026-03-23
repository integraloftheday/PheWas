#!/usr/bin/env Rscript

# 02b_Create_Analysis_Subset.r
# Creates a balanced subset of the main analysis dataset for quick testing.

if (dir.exists(".r_libs")) {
  .libPaths(c(normalizePath(".r_libs"), .libPaths()))
}

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: 02b_Create_Analysis_Subset.r <path_to_config.yaml> <run_dir>")
}

config_path <- args[1]
run_dir <- args[2]

config <- yaml::read_yaml(config_path)

dataset_type <- config$dataset_type
input_path <- config$data_paths[[dataset_type]]
subset_size <- config$subset_size
balance_vars <- config$subset_balance_vars

if (!file.exists(input_path)) {
  stop(sprintf("Input dataset not found: %s", input_path))
}

cat(sprintf("Loading full dataset from %s...\n", input_path))
df <- arrow::read_parquet(input_path)

cat(sprintf("Creating balanced subset of size ~%d persons...\n", subset_size))

# We want to balance by person, so we first get person-level demographics
person_df <- df %>%
  group_by(person_id) %>%
  summarise(across(all_of(balance_vars), first)) %>%
  ungroup()

# Calculate how many people we need per group
group_counts <- person_df %>%
  group_by(across(all_of(balance_vars))) %>%
  tally()

num_groups <- nrow(group_counts)
target_per_group <- ceiling(subset_size / num_groups)

# Sample persons
sampled_persons <- person_df %>%
  group_by(across(all_of(balance_vars))) %>%
  sample_n(size = min(n(), target_per_group)) %>%
  ungroup() %>%
  pull(person_id)

cat(sprintf("Selected %d unique persons.\n", length(sampled_persons)))

# Filter original dataframe
subset_df <- df %>%
  filter(person_id %in% sampled_persons)

output_path <- file.path(run_dir, "balanced_subset.parquet")
cat(sprintf("Writing subset to %s...\n", output_path))
arrow::write_parquet(subset_df, output_path)
cat("Subset creation complete.\n")
