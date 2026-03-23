#!/usr/bin/env Rscript

# 02b_Create_Analysis_Subset.r
# Creates a balanced subset of the main analysis dataset for quick testing.

if (dir.exists(".r_libs")) {
  .libPaths(c(normalizePath(".r_libs"), .libPaths()))
}

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(stringr)
  library(yaml)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) == 0) {
  stop("Usage: 02b_Create_Analysis_Subset.r <path_to_config.yaml> <run_dir>")
}

config_path <- args[1]
run_dir <- args[2]

config <- yaml::read_yaml(config_path)

`%||%` <- function(x, y) if (is.null(x)) y else x

dataset_type <- config$dataset_type
input_path <- config$data_paths[[dataset_type]]
subset_size <- config$subset_size
balance_vars <- config$subset_balance_vars
subset_seed <- config$subset_seed %||% 42

default_no_dst_zip3 <- unique(c(
  sprintf("%03d", 850:865),
  "967", "968",
  "006", "007", "008", "009", "969"
))

normalize_zip3 <- function(x) {
  x_chr <- as.character(x)
  vapply(x_chr, function(z) {
    if (is.na(z) || z == "") return(NA_character_)
    digits <- gsub("[^0-9]", "", z)
    if (nchar(digits) == 0) return(NA_character_)
    if (nchar(digits) <= 3) return(sprintf("%03d", as.integer(digits)))
    if (nchar(digits) == 4) return(substr(sprintf("%05d", as.integer(digits)), 1, 3))
    substr(digits, 1, 3)
  }, FUN.VALUE = character(1))
}

first_non_missing_chr <- function(x) {
  x_chr <- as.character(x)
  x_chr <- x_chr[!is.na(x_chr) & x_chr != ""]
  if (length(x_chr) == 0) return(NA_character_)
  x_chr[[1]]
}

if (!file.exists(input_path)) {
  stop(sprintf("Input dataset not found: %s", input_path))
}

cat(sprintf("Loading full dataset from %s...\n", input_path))
df <- arrow::read_parquet(input_path)

if (!"person_id" %in% names(df)) {
  stop("Input dataset must contain 'person_id'.")
}

if (!is.character(balance_vars) || length(balance_vars) == 0) {
  stop("config$subset_balance_vars must be a non-empty character vector.")
}

cat(sprintf("Creating balanced subset of size ~%d persons...\n", subset_size))

zip_source <- if ("zip3" %in% names(df)) {
  "zip3"
} else if ("zip_code" %in% names(df)) {
  "zip_code"
} else {
  NA_character_
}

if (is.na(zip_source)) {
  warning("zip3/zip_code not found; dst_observed will be set to 'Unknown'.")
}

person_df <- df %>%
  group_by(person_id) %>%
  summarise(
    sex_raw = if ("sex_concept" %in% names(df)) first_non_missing_chr(sex_concept) else NA_character_,
    age_raw = if ("age_at_sleep" %in% names(df)) mean(age_at_sleep, na.rm = TRUE) else NA_real_,
    race_raw = if ("race" %in% names(df)) first_non_missing_chr(race) else NA_character_,
    zip_raw = if (!is.na(zip_source)) first_non_missing_chr(.data[[zip_source]]) else NA_character_,
    .groups = "drop"
  ) %>%
  mutate(
    age_raw = if_else(is.nan(age_raw), as.numeric(NA), age_raw),
    sex_binary = case_when(
      str_detect(str_to_lower(sex_raw), "female|woman|girl") ~ "Female",
      str_detect(str_to_lower(sex_raw), "male|man|boy") ~ "Male",
      TRUE ~ "Other/Unknown"
    ),
    age_bin = case_when(
      is.na(age_raw) ~ "Unknown",
      age_raw < 40 ~ "18-39",
      age_raw < 55 ~ "40-54",
      age_raw < 70 ~ "55-69",
      TRUE ~ "70+"
    ),
    race_collapsed = case_when(
      str_detect(str_to_lower(race_raw), "white") ~ "White",
      str_detect(str_to_lower(race_raw), "black|african") ~ "Black",
      str_detect(str_to_lower(race_raw), "asian") ~ "Asian",
      str_detect(str_to_lower(race_raw), "hispanic|latino") ~ "Hispanic/Latino",
      str_detect(str_to_lower(race_raw), "american indian|alaska native") ~ "AI/AN",
      str_detect(str_to_lower(race_raw), "native hawaiian|pacific islander") ~ "NH/PI",
      str_detect(str_to_lower(race_raw), "more than one|multiracial|multiple") ~ "Multiracial",
      is.na(race_raw) | race_raw == "" ~ "Unknown",
      TRUE ~ "Other"
    ),
    zip3_norm = normalize_zip3(zip_raw),
    dst_observed = case_when(
      is.na(zip3_norm) ~ "Unknown",
      zip3_norm %in% default_no_dst_zip3 ~ "NoDST",
      TRUE ~ "DST"
    )
  )

requested_vars <- dplyr::case_match(
  balance_vars,
  c("sex", "sex_concept", "sex_binary") ~ "sex_binary",
  c("age", "age_at_sleep", "age_bin") ~ "age_bin",
  c("race", "race_collapsed") ~ "race_collapsed",
  c("dst_observed", "dst_observes", "dst_group") ~ "dst_observed",
  .default = balance_vars
)

missing_balance_vars <- setdiff(requested_vars, names(person_df))
if (length(missing_balance_vars) > 0) {
  stop(sprintf(
    "Missing balance variables after derivation: %s",
    paste(missing_balance_vars, collapse = ", ")
  ))
}

cat(sprintf("Balancing over: %s\n", paste(requested_vars, collapse = ", ")))
set.seed(subset_seed)

# Calculate how many people we need per group
group_counts <- person_df %>%
  group_by(across(all_of(requested_vars))) %>%
  tally()

num_groups <- nrow(group_counts)
target_per_group <- ceiling(subset_size / num_groups)

# Sample persons
sampled_persons <- person_df %>%
  group_by(across(all_of(requested_vars))) %>%
  slice_sample(n = min(n(), target_per_group)) %>%
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
