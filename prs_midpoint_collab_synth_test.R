#!/usr/bin/env Rscript

if (dir.exists(".r_libs")) {
  .libPaths(c(normalizePath(".r_libs"), .libPaths()))
}

required_packages <- c("arrow", "dplyr", "jsonlite", "purrr", "readr", "tibble")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install them before running prs_midpoint_collab_synth_test.R."
    )
  )
}

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(jsonlite)
  library(purrr)
  library(readr)
  library(tibble)
})

set.seed(42)

script_path <- normalizePath("prs_midpoint_collab_analysis.R", mustWork = TRUE)
work_dir <- file.path(tempdir(), paste0("prs_midpoint_collab_test_", Sys.getpid()))
dir.create(work_dir, recursive = TRUE, showWarnings = FALSE)

person_n <- 240
person_ids <- sprintf("SYN_%05d", seq_len(person_n))
scores <- rnorm(person_n)
ages <- sample(25:75, person_n, replace = TRUE)
sexes <- sample(c("Female", "Male"), person_n, replace = TRUE)
dob <- as.Date("2022-01-01") - round(ages * 365.25)

covariates <- tibble(
  person_id = person_ids,
  date_of_birth = dob,
  sex_concept = sexes,
  race = sample(c("White", "Black", "Asian", "Hispanic/Latino"), person_n, replace = TRUE),
  employment_status = sample(c("Working", "Retired", "Student", "Not Working"), person_n, replace = TRUE),
  bmi = round(rnorm(person_n, mean = 28, sd = 4), 1)
)

nightly_rows <- purrr::map_dfr(seq_along(person_ids), function(i) {
  dates <- seq.Date(as.Date("2022-01-01"), by = "day", length.out = 42)
  is_free <- weekdays(dates) %in% c("Saturday", "Sunday")
  start_base <- 23.4 + (0.12 * scores[[i]]) + rnorm(1, 0, 0.2)
  dur_work <- 7.4 + rnorm(1, 0, 0.2)
  dur_free <- 8.1 + (0.10 * scores[[i]]) + rnorm(1, 0, 0.25)
  start_work <- start_base
  start_free <- start_base + 0.35 + rnorm(1, 0, 0.15)

  tibble(
    person_id = person_ids[[i]],
    sleep_date = dates,
    is_weekend_or_holiday = is_free,
    is_weekend = is_free,
    daily_start_hour = ifelse(is_free, start_free, start_work) %% 24,
    daily_duration_mins = ifelse(is_free, dur_free, dur_work) * 60
  ) %>%
    mutate(
      daily_end_hour = (daily_start_hour + (daily_duration_mins / 60)) %% 24,
      daily_midpoint_hour = (daily_start_hour + (daily_duration_mins / 120)) %% 24,
      onset_linear = ifelse(daily_start_hour < 12, daily_start_hour + 24, daily_start_hour),
      offset_linear = ifelse(daily_end_hour < 12, daily_end_hour + 24, daily_end_hour),
      midpoint_linear = ifelse(daily_midpoint_hour < 12, daily_midpoint_hour + 24, daily_midpoint_hour),
      daily_sleep_window_mins = daily_duration_mins
    )
})

score_file <- file.path(work_dir, "scores.txt")
readr::write_tsv(
  tibble(FID = person_ids, IID = person_ids, PHENO = 1, SCORESUM = scores),
  score_file
)

ancestry_file <- file.path(work_dir, "ancestry_preds.tsv")
ancestry_df <- tibble(
  research_id = person_ids,
  ancestry_pred = sample(c("eur", "afr", "amr", "eas"), person_n, replace = TRUE),
  pca_features = purrr::map_chr(seq_len(person_n), function(i) {
    jsonlite::toJSON(round(rnorm(17, sd = 0.8), 4), auto_unbox = TRUE)
  })
)
readr::write_tsv(ancestry_df, ancestry_file)

phecodes <- sprintf("%03d", seq_len(24))
phewas <- covariates
for (j in seq_along(phecodes)) {
  lp <- -2.2 + (if (j <= 4) 0.8 * scores else 0.0) + rnorm(person_n, 0, 0.15)
  has_val <- stats::rbinom(person_n, size = 1, prob = plogis(lp))
  had_val <- stats::rbinom(person_n, size = 1, prob = ifelse(has_val == 1, 0.15, 0.03))
  phewas[[paste0("has_phe_", phecodes[[j]])]] <- as.logical(has_val)
  phewas[[paste0("had_phe_", phecodes[[j]])]] <- as.logical(had_val)
}

phewas_file <- file.path(work_dir, "master_phewas_wide.parquet")
cov_file <- file.path(work_dir, "master_covariates_only.parquet")
nightly_file <- file.path(work_dir, "ready_for_analysis.parquet")
mapping_file <- file.path(work_dir, "ICD_to_Phecode_mapping.csv")
out_dir <- file.path(work_dir, "outputs")

arrow::write_parquet(nightly_rows, nightly_file)
arrow::write_parquet(covariates, cov_file)
arrow::write_parquet(phewas, phewas_file)

readr::write_csv(
  tibble(
    ICD = sprintf("%03d", seq_along(phecodes)),
    ICD_STR = paste("ICD", seq_along(phecodes)),
    PHECODE = phecodes,
    PHENOTYPE = paste("Synthetic phenotype", seq_along(phecodes))
  ),
  mapping_file
)

args <- c(
  script_path,
  paste0("--nightly_parquet=", nightly_file),
  paste0("--covariates_parquet=", cov_file),
  paste0("--phewas_parquet=", phewas_file),
  paste0("--score_file=", score_file),
  paste0("--ancestry_tsv=", ancestry_file),
  paste0("--phecode_map_csv=", mapping_file),
  paste0("--out_dir=", out_dir),
  "--min_case_count=20"
)

status <- system2("Rscript", args = args)
if (status != 0) {
  stop("Synthetic collaborator analysis run failed.")
}

expected_outputs <- c(
  file.path(out_dir, "summary.md"),
  file.path(out_dir, "tables", "association_continuous_models.csv"),
  file.path(out_dir, "tables", "association_tertile_models.csv"),
  file.path(out_dir, "tables", "phewas_results.csv"),
  file.path(out_dir, "plots", "association_forest_per_sd.png"),
  file.path(out_dir, "plots", "midpoint_by_prs_tertile.png"),
  file.path(out_dir, "plots", "phewas_manhattan.png")
)

missing_outputs <- expected_outputs[!file.exists(expected_outputs)]
if (length(missing_outputs) > 0) {
  stop("Synthetic run completed but missing outputs: ", paste(missing_outputs, collapse = ", "))
}

cat("PASS\n")
cat("Synthetic workflow validated at:", out_dir, "\n")
