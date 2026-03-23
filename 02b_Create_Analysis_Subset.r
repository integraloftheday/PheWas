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

span_days <- function(x) {
  x_date <- as.Date(x)
  x_date <- x_date[!is.na(x_date)]
  if (length(x_date) == 0) return(NA_real_)
  as.numeric(max(x_date) - min(x_date)) + 1
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

has_sleep_date <- "sleep_date" %in% names(df)
if (has_sleep_date) {
  df <- df %>% mutate(sleep_date_std = as.Date(sleep_date))
}

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
    nights_recorded = if (has_sleep_date) n_distinct(sleep_date_std, na.rm = TRUE) else NA_integer_,
    recording_span_days = if (has_sleep_date) span_days(sleep_date_std) else NA_real_,
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
    ),
    recording_span_bin = case_when(
      is.na(recording_span_days) ~ "Unknown",
      recording_span_days >= 365 ~ ">=12m",
      recording_span_days >= 183 ~ "6-12m",
      TRUE ~ "<6m"
    ),
    nights_bin = case_when(
      is.na(nights_recorded) ~ "Unknown",
      nights_recorded >= 300 ~ ">=300 nights",
      nights_recorded >= 180 ~ "180-299 nights",
      nights_recorded >= 90 ~ "90-179 nights",
      TRUE ~ "<90 nights"
    ),
    longitudinal_weight = case_when(
      recording_span_bin == ">=12m" ~ 3,
      recording_span_bin == "6-12m" ~ 2,
      recording_span_bin == "<6m" ~ 1,
      TRUE ~ 1
    )
  )

requested_vars <- dplyr::case_match(
  balance_vars,
  c("sex", "sex_concept", "sex_binary") ~ "sex_binary",
  c("age", "age_at_sleep", "age_bin") ~ "age_bin",
  c("race", "race_collapsed") ~ "race_collapsed",
  c("dst_observed", "dst_observes", "dst_group") ~ "dst_observed",
  c("recording_span", "recording_span_bin", "longitudinal_bin") ~ "recording_span_bin",
  c("nights", "nights_bin", "nights_recorded_bin") ~ "nights_bin",
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

if (has_sleep_date) {
  coverage_counts <- person_df %>%
    count(recording_span_bin, nights_bin, name = "n_people") %>%
    arrange(desc(n_people))
  cat("Coverage summary (person-level):\n")
  print(coverage_counts, n = Inf)
}

# Calculate how many people we need per group
group_counts <- person_df %>%
  group_by(across(all_of(requested_vars))) %>%
  tally()

num_groups <- nrow(group_counts)
target_per_group <- ceiling(subset_size / num_groups)

# Sample persons
sampled_persons <- person_df %>%
  group_by(across(all_of(requested_vars))) %>%
  slice_sample(n = min(n(), target_per_group), weight_by = longitudinal_weight) %>%
  ungroup() %>%
  pull(person_id)

cat(sprintf("Selected %d unique persons.\n", length(sampled_persons)))

sampled_person_df <- person_df %>%
  filter(person_id %in% sampled_persons)

# Write person-level profile table for sampled participants.
person_profile_path <- file.path(run_dir, "subset_person_profile.csv")
utils::write.csv(sampled_person_df, person_profile_path, row.names = FALSE)

# Write grouped summary table (demographics + longitudinal coverage bins).
summary_group_cols <- intersect(
  c("sex_binary", "age_bin", "race_collapsed", "dst_observed", "recording_span_bin", "nights_bin"),
  names(sampled_person_df)
)

subset_group_summary <- sampled_person_df %>%
  group_by(across(all_of(summary_group_cols))) %>%
  summarise(
    n_persons = n(),
    mean_nights_recorded = mean(nights_recorded, na.rm = TRUE),
    median_nights_recorded = median(nights_recorded, na.rm = TRUE),
    mean_recording_span_days = mean(recording_span_days, na.rm = TRUE),
    median_recording_span_days = median(recording_span_days, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_persons))

summary_path <- file.path(run_dir, "subset_group_summary.csv")
utils::write.csv(subset_group_summary, summary_path, row.names = FALSE)

# Write counts for the exact balancing strata used by the sampler.
subset_balance_summary <- sampled_person_df %>%
  group_by(across(all_of(requested_vars))) %>%
  summarise(n_persons = n(), .groups = "drop") %>%
  arrange(desc(n_persons))

balance_summary_path <- file.path(run_dir, "subset_balance_strata_counts.csv")
utils::write.csv(subset_balance_summary, balance_summary_path, row.names = FALSE)

cat(sprintf("Wrote person-level subset profile to %s\n", person_profile_path))
cat(sprintf("Wrote grouped subset summary to %s\n", summary_path))
cat(sprintf("Wrote balance-strata counts to %s\n", balance_summary_path))

# Filter original dataframe
subset_df <- df %>%
  filter(person_id %in% sampled_persons)

output_path <- file.path(run_dir, "balanced_subset.parquet")
cat(sprintf("Writing subset to %s...\n", output_path))
arrow::write_parquet(subset_df, output_path)
cat("Subset creation complete.\n")
