#!/usr/bin/env Rscript

if (dir.exists(".r_libs")) {
  .libPaths(c(normalizePath(".r_libs"), .libPaths()))
}

required_packages <- c("arrow", "dplyr", "readr", "stringr", "tibble")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install them before running prs_midpoint_collab_local_plink_test.R."
    )
  )
}

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(readr)
  library(stringr)
  library(tibble)
})

parse_args <- function(args) {
  defaults <- list(
    plink = file.path("plink_mac_20250819", "plink"),
    weights = file.path("analysis_inputs", "METAL_midp_all_pst_eff_a1_b0.5_phi1e-02_ALL.txt"),
    out_dir = file.path("results", "angus_midpoint_prs_bundle"),
    n_snps = 12L,
    n_people = 240L
  )

  out <- defaults
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    key <- kv[[1]]
    value <- if (length(kv) > 1) paste(kv[-1], collapse = "=") else "true"
    out[[key]] <- value
  }
  out$n_snps <- as.integer(out$n_snps)
  out$n_people <- as.integer(out$n_people)
  out
}

write_review_md <- function(bundle_dir, score_file, summary_file, plink_path, n_people, n_snps) {
  lines <- c(
    "# Angus midpoint PRS pipeline review bundle",
    "",
    "## What was tested",
    "",
    sprintf("- Local PLINK binary: `%s`", plink_path),
    sprintf("- Synthetic participants: %d", n_people),
    sprintf("- Collaborator weight file subset used for scoring: %d SNPs", n_snps),
    "- End-to-end path exercised: synthetic genotype -> PLINK score -> one-off midpoint/PheWAS analysis -> collaborator-style outputs",
    "",
    "## Review files",
    "",
    sprintf("- PLINK score output: `%s`", score_file),
    sprintf("- Analysis summary: `%s`", summary_file),
    "- Association tables: `outputs/tables/association_continuous_models.csv`, `outputs/tables/association_tertile_models.csv`",
    "- PheWAS table: `outputs/tables/phewas_results.csv`",
    "- Plot source tables: `outputs/tables/association_forest_plot_data.csv`, `outputs/tables/midpoint_by_prs_tertile_plot_data.csv`, `outputs/tables/phewas_manhattan_plot_data.csv`",
    "- Plots: `outputs/plots/association_forest_per_sd.png`, `outputs/plots/midpoint_by_prs_tertile.png`, `outputs/plots/phewas_manhattan.png`",
    "",
    "## Notes",
    "",
    "- This local run uses synthetic participants and a small subset of the collaborator weight file for compatibility testing.",
    "- The AoU production scorer still depends on remote data access (`gsutil`, PLINK reference downloads, and AoU genetics), but the local scoring and downstream analysis interfaces were exercised here."
  )

  writeLines(lines, file.path(bundle_dir, "review.md"))
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  plink_path <- normalizePath(args$plink, mustWork = TRUE)
  weights_path <- normalizePath(args$weights, mustWork = TRUE)
  analysis_script <- normalizePath("prs_midpoint_collab_analysis.R", mustWork = TRUE)

  bundle_dir <- args$out_dir
  inputs_dir <- file.path(bundle_dir, "inputs")
  plink_dir <- file.path(bundle_dir, "plink")
  outputs_dir <- file.path(bundle_dir, "outputs")

  dir.create(bundle_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(inputs_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(plink_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(outputs_dir, recursive = TRUE, showWarnings = FALSE)

  snps <- read.table(weights_path, header = FALSE, stringsAsFactors = FALSE, nrows = args$n_snps)
  colnames(snps) <- c("chr", "rsid", "pos", "effect_allele", "other_allele", "weight")

  person_ids <- sprintf("PLINK_SYN_%05d", seq_len(args$n_people))
  sex_code <- sample(c(1, 2), args$n_people, replace = TRUE)
  ped_prefix <- file.path(plink_dir, "synthetic_genotypes")
  map_path <- paste0(ped_prefix, ".map")
  ped_path <- paste0(ped_prefix, ".ped")

  map_df <- snps %>%
    transmute(chr = chr, rsid = rsid, cm = 0, pos = pos)
  write.table(map_df, file = map_path, quote = FALSE, row.names = FALSE, col.names = FALSE, sep = "\t")

  maf <- runif(nrow(snps), min = 0.1, max = 0.45)
  ped_lines <- vector("character", length = args$n_people)

  for (i in seq_len(args$n_people)) {
    genotype_fields <- character(nrow(snps) * 2)
    for (j in seq_len(nrow(snps))) {
      dosage <- rbinom(1, size = 2, prob = maf[[j]])
      alleles <- c(snps$other_allele[[j]], snps$effect_allele[[j]])
      if (dosage == 0) {
        pair <- c(alleles[[1]], alleles[[1]])
      } else if (dosage == 1) {
        pair <- c(alleles[[1]], alleles[[2]])
      } else {
        pair <- c(alleles[[2]], alleles[[2]])
      }
      genotype_fields[((j - 1) * 2 + 1):((j - 1) * 2 + 2)] <- pair
    }

    ped_lines[[i]] <- paste(
      c(person_ids[[i]], person_ids[[i]], 0, 0, sex_code[[i]], -9, genotype_fields),
      collapse = " "
    )
  }

  writeLines(ped_lines, ped_path)

  make_bed_prefix <- file.path(plink_dir, "synthetic_bed")
  status <- system2(
    plink_path,
    args = c("--file", ped_prefix, "--make-bed", "--out", make_bed_prefix),
    stdout = TRUE,
    stderr = TRUE
  )
  if (!file.exists(paste0(make_bed_prefix, ".bed"))) {
    stop("PLINK bed creation failed:\n", paste(status, collapse = "\n"))
  }

  score_prefix <- file.path(plink_dir, "midpoint_score")
  score_status <- system2(
    plink_path,
    args = c(
      "--bfile", make_bed_prefix,
      "--score", weights_path, "2", "4", "6", "sum",
      "--out", score_prefix
    ),
    stdout = TRUE,
    stderr = TRUE
  )
  score_file <- paste0(score_prefix, ".profile")
  if (!file.exists(score_file)) {
    stop("PLINK score failed:\n", paste(score_status, collapse = "\n"))
  }

  score_df <- read.table(score_file, header = TRUE, stringsAsFactors = FALSE) %>%
    as_tibble() %>%
    transmute(person_id = IID, score = as.numeric(SCORESUM))

  ages <- sample(25:75, args$n_people, replace = TRUE)
  sexes <- ifelse(sex_code == 1, "Male", "Female")
  dob <- as.Date("2022-01-01") - round(ages * 365.25)

  covariates <- tibble(
    person_id = person_ids,
    date_of_birth = dob,
    sex_concept = sexes,
    race = sample(c("White", "Black", "Asian", "Hispanic/Latino"), args$n_people, replace = TRUE),
    employment_status = sample(c("Working", "Retired", "Student", "Not Working"), args$n_people, replace = TRUE),
    bmi = round(rnorm(args$n_people, mean = 28, sd = 4), 1)
  )

  nightly_rows <- purrr::map_dfr(seq_along(person_ids), function(i) {
    dates <- seq.Date(as.Date("2022-01-01"), by = "day", length.out = 42)
    is_free <- weekdays(dates) %in% c("Saturday", "Sunday")
    person_score <- score_df$score[score_df$person_id == person_ids[[i]]][[1]]
    start_base <- 23.4 + (0.9 * person_score) + rnorm(1, 0, 0.15)
    dur_work <- 7.2 + rnorm(1, 0, 0.2)
    dur_free <- 8.0 + (0.25 * person_score) + rnorm(1, 0, 0.2)
    start_work <- start_base
    start_free <- start_base + 0.4 + rnorm(1, 0, 0.1)

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

  ancestry_df <- tibble(
    research_id = person_ids,
    ancestry_pred = sample(c("eur", "afr", "amr", "eas"), args$n_people, replace = TRUE),
    pca_features = vapply(
      seq_len(args$n_people),
      function(i) jsonlite::toJSON(round(rnorm(17, sd = 0.8), 4), auto_unbox = TRUE),
      character(1)
    )
  )

  phecodes <- sprintf("%03d", seq_len(24))
  phewas <- covariates
  scaled_score <- as.numeric(scale(score_df$score))
  for (j in seq_along(phecodes)) {
    lp <- -2.2 + (if (j <= 4) 1.0 * scaled_score else 0.0) + rnorm(args$n_people, 0, 0.1)
    has_val <- rbinom(args$n_people, size = 1, prob = plogis(lp))
    had_val <- rbinom(args$n_people, size = 1, prob = ifelse(has_val == 1, 0.18, 0.03))
    phewas[[paste0("has_phe_", phecodes[[j]])]] <- as.logical(has_val)
    phewas[[paste0("had_phe_", phecodes[[j]])]] <- as.logical(had_val)
  }

  nightly_file <- file.path(inputs_dir, "ready_for_analysis.parquet")
  cov_file <- file.path(inputs_dir, "master_covariates_only.parquet")
  phewas_file <- file.path(inputs_dir, "master_phewas_wide.parquet")
  ancestry_file <- file.path(inputs_dir, "ancestry_preds.tsv")
  mapping_file <- file.path(inputs_dir, "ICD_to_Phecode_mapping.csv")

  arrow::write_parquet(nightly_rows, nightly_file)
  arrow::write_parquet(covariates, cov_file)
  arrow::write_parquet(phewas, phewas_file)
  readr::write_tsv(ancestry_df, ancestry_file)
  readr::write_csv(
    tibble(
      ICD = sprintf("%03d", seq_along(phecodes)),
      ICD_STR = paste("ICD", seq_along(phecodes)),
      PHECODE = phecodes,
      PHENOTYPE = paste("Synthetic phenotype", seq_along(phecodes))
    ),
    mapping_file
  )

  analysis_status <- system2(
    "Rscript",
    args = c(
      analysis_script,
      paste0("--nightly_parquet=", nightly_file),
      paste0("--covariates_parquet=", cov_file),
      paste0("--phewas_parquet=", phewas_file),
      paste0("--score_file=", score_file),
      paste0("--ancestry_tsv=", ancestry_file),
      paste0("--phecode_map_csv=", mapping_file),
      paste0("--out_dir=", outputs_dir),
      "--min_case_count=20"
    )
  )
  if (analysis_status != 0) {
    stop("Collaborator analysis script failed after PLINK scoring.")
  }

  summary_file <- file.path(outputs_dir, "summary.md")
  if (!file.exists(summary_file)) {
    stop("Expected analysis summary file was not created: ", summary_file)
  }

  write_review_md(
    bundle_dir = bundle_dir,
    score_file = normalizePath(score_file),
    summary_file = normalizePath(summary_file),
    plink_path = plink_path,
    n_people = args$n_people,
    n_snps = args$n_snps
  )

  cat("PASS\n")
  cat("Review bundle:", normalizePath(bundle_dir), "\n")
}

main()
