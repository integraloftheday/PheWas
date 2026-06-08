#!/usr/bin/env Rscript

if (dir.exists(".r_libs")) {
  .libPaths(c(normalizePath(".r_libs"), .libPaths()))
}

required_packages <- c(
  "arrow", "dplyr", "ggplot2", "jsonlite", "purrr",
  "readr", "stringr", "tibble", "tidyr"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install them before running prs_midpoint_collab_analysis.R."
    )
  )
}

suppressPackageStartupMessages({
  library(arrow)
  library(dplyr)
  library(ggplot2)
  library(jsonlite)
  library(purrr)
  library(readr)
  library(stringr)
  library(tibble)
  library(tidyr)
})

phenotype_label_map <- c(
  person_weekend_avg_midpoint = "Weekend average midpoint",
  MSF = "MSF",
  MSFsc = "MSFsc"
)

`%||%` <- function(x, y) {
  if (is.null(x) || length(x) == 0 || all(is.na(x))) y else x
}

wandb_progress_file <- Sys.getenv("WANDB_PROGRESS_FILE", "")

log_progress_event <- function(stage, event, status = "running", metrics = list(), details = list()) {
  if (!nzchar(wandb_progress_file)) return(invisible(NULL))
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
  dir.create(dirname(wandb_progress_file), recursive = TRUE, showWarnings = FALSE)
  cat(
    jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null"),
    "\n",
    file = wandb_progress_file,
    append = TRUE
  )
}

parse_args <- function(args) {
  defaults <- list(
    nightly_parquet = "processed_data/ready_for_analysis.parquet",
    covariates_parquet = "",
    phewas_parquet = "processed_data/master/master_phewas_wide.parquet",
    score_file = file.path(
      "processed_data",
      "PGRS",
      "METAL_midp_all_pst_eff_a1_b0.5_phi1e-02_ALL",
      "METAL_midp_all_pst_eff_a1_b0.5_phi1e-02_ALL_PGRS.txt"
    ),
    ancestry_tsv = file.path("processed_data", "PGRS", "shared", "ancestry_preds.tsv"),
    phecode_map_csv = "analysis_inputs/ICD_to_Phecode_mapping.csv",
    out_dir = "results/angus_midpoint_prs_analysis",
    skip_phewas = FALSE,
    pc_count = 10L,
    min_case_count = 20L
  )

  out <- defaults
  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    key <- kv[[1]]
    value <- if (length(kv) > 1) paste(kv[-1], collapse = "=") else "true"
    out[[key]] <- value
  }

  as_bool <- function(x) {
    tolower(as.character(x)) %in% c("1", "true", "yes", "y")
  }

  out$skip_phewas <- as_bool(out$skip_phewas)
  out$pc_count <- as.integer(out$pc_count)
  out$min_case_count <- as.integer(out$min_case_count)
  out
}

format_clock <- function(x) {
  vals <- suppressWarnings(as.numeric(x))
  out <- rep(NA_character_, length(vals))
  ok <- is.finite(vals)
  total_minutes <- round((vals[ok] %% 24) * 60) %% (24 * 60)
  out[ok] <- sprintf("%02d:%02d", total_minutes %/% 60, total_minutes %% 60)
  out
}

clock_to_linear <- function(x) {
  vals <- suppressWarnings(as.numeric(x))
  out <- ifelse(vals < 12, vals + 24, vals)
  out[!is.finite(vals)] <- NA_real_
  out
}

linear_to_clock <- function(x) {
  vals <- suppressWarnings(as.numeric(x))
  out <- vals %% 24
  out[!is.finite(vals)] <- NA_real_
  out
}

circular_mean_time <- function(x) {
  vals <- suppressWarnings(as.numeric(x))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) return(NA_real_)
  radians <- vals * (2 * pi / 24)
  mean_rad <- atan2(mean(sin(radians)), mean(cos(radians)))
  (mean_rad * 24 / (2 * pi)) %% 24
}

safe_mean <- function(x) {
  vals <- suppressWarnings(as.numeric(x))
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) return(NA_real_)
  mean(vals)
}

phenotype_label <- function(x) {
  unname(phenotype_label_map[x] %||% x)
}

normalize_phecode <- function(x) {
  out <- as.character(x)
  out <- str_trim(out)
  out <- str_replace(out, "(\\.\\d*?[1-9])0+$", "\\1")
  out <- str_replace(out, "\\.0+$", "")
  out
}

theme_research <- function() {
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "gray30"),
      axis.title = element_text(face = "bold"),
      legend.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom"
    )
}

resolve_covariates_path <- function(explicit_path = "") {
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

load_score_file <- function(path) {
  if (!file.exists(path)) stop("Score file not found: ", path)
  df <- read.table(path, header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  df <- as_tibble(df)

  if (all(c("IID", "SCORESUM") %in% names(df))) {
    out <- df %>% transmute(person_id = as.character(IID), score_raw = as.numeric(SCORESUM))
  } else if (all(c("person_id", "score") %in% names(df))) {
    out <- df %>% transmute(person_id = as.character(person_id), score_raw = as.numeric(score))
  } else {
    stop("Score file must contain either IID/SCORESUM or person_id/score columns.")
  }

  distinct(out, person_id, .keep_all = TRUE)
}

parse_pca_json <- function(x, n_pcs) {
  out <- rep(NA_real_, n_pcs)
  vals <- tryCatch(as.numeric(jsonlite::fromJSON(x)), error = function(e) numeric())
  if (length(vals) == 0) return(out)
  keep <- seq_len(min(length(vals), n_pcs))
  out[keep] <- vals[keep]
  out
}

load_ancestry_features <- function(path, n_pcs) {
  if (!file.exists(path)) stop("Ancestry TSV not found: ", path)
  ancestry <- readr::read_tsv(path, show_col_types = FALSE)
  if (!"research_id" %in% names(ancestry)) {
    stop("Ancestry TSV must contain research_id.")
  }

  ancestry_pred <- ancestry$ancestry_pred %||% rep(NA_character_, nrow(ancestry))
  out <- tibble(
    person_id = as.character(ancestry$research_id),
    ancestry_pred = ancestry_pred
  )

  if ("pca_features" %in% names(ancestry)) {
    pca_mat <- t(vapply(ancestry$pca_features, parse_pca_json, numeric(n_pcs), n_pcs = n_pcs))
    colnames(pca_mat) <- paste0("pc_tmp_", seq_len(ncol(pca_mat)))
    pca_df <- as_tibble(pca_mat)
    names(pca_df) <- paste0("pca_", seq_len(n_pcs))
    out <- bind_cols(out, pca_df)
  } else {
    existing_pcs <- intersect(names(ancestry), paste0("pca_", seq_len(n_pcs)))
    if (length(existing_pcs) > 0) {
      out <- bind_cols(out, ancestry[, existing_pcs, drop = FALSE])
    }
  }

  distinct(out, person_id, .keep_all = TRUE)
}

load_covariates <- function(path) {
  cov_df <- arrow::read_parquet(path)
  cov_df <- as_tibble(cov_df) %>% mutate(person_id = as.character(person_id))

  wanted <- c(
    "person_id", "date_of_birth", "sex_concept", "race",
    "employment_status", "bmi", "min_date", "max_date", "duration"
  )
  keep <- intersect(wanted, names(cov_df))
  cov_df %>% select(any_of(keep))
}

build_midpoint_phenotypes <- function(nightly_df) {
  if (!"person_id" %in% names(nightly_df)) stop("Nightly data must contain person_id.")
  if (!"sleep_date" %in% names(nightly_df)) stop("Nightly data must contain sleep_date.")

  free_day_col <- if ("is_weekend_or_holiday" %in% names(nightly_df)) {
    "is_weekend_or_holiday"
  } else if ("is_weekend" %in% names(nightly_df)) {
    "is_weekend"
  } else {
    stop("Nightly data must contain is_weekend_or_holiday or is_weekend.")
  }

  duration_col <- if ("daily_duration_mins" %in% names(nightly_df)) {
    "daily_duration_mins"
  } else if ("daily_sleep_window_mins" %in% names(nightly_df)) {
    "daily_sleep_window_mins"
  } else {
    stop("Nightly data must contain daily_duration_mins or daily_sleep_window_mins.")
  }

  nightly_df %>%
    as_tibble() %>%
    mutate(
      person_id = as.character(person_id),
      sleep_date = as.Date(sleep_date),
      free_day = as.logical(.data[[free_day_col]]),
      daily_duration_hours = as.numeric(.data[[duration_col]]) / 60,
      daily_start_linear = if ("onset_linear" %in% names(.)) {
        as.numeric(onset_linear)
      } else {
        clock_to_linear(daily_start_hour)
      }
    ) %>%
    group_by(person_id) %>%
    summarise(
      mean_sleep_date = as.Date(mean(as.numeric(sleep_date), na.rm = TRUE), origin = "1970-01-01"),
      n_total_nights = n(),
      n_free_nights = sum(free_day %in% TRUE, na.rm = TRUE),
      n_work_nights = sum(free_day %in% FALSE, na.rm = TRUE),
      person_weekend_avg_midpoint_clock = circular_mean_time(daily_midpoint_hour[free_day %in% TRUE]),
      SO_f = safe_mean(daily_start_linear[free_day %in% TRUE]),
      SO_w = safe_mean(daily_start_linear[free_day %in% FALSE]),
      SD_f = safe_mean(daily_duration_hours[free_day %in% TRUE]),
      SD_w = safe_mean(daily_duration_hours[free_day %in% FALSE]),
      .groups = "drop"
    ) %>%
    mutate(
      person_weekend_avg_midpoint = if_else(
        n_free_nights > 0 & is.finite(person_weekend_avg_midpoint_clock),
        clock_to_linear(person_weekend_avg_midpoint_clock),
        NA_real_
      ),
      MSF_linear = if_else(
        n_free_nights > 0 & is.finite(SO_f) & is.finite(SD_f),
        ((SO_f + (SD_f / 2) - 12) %% 24) + 12,
        NA_real_
      ),
      SD_week = if_else(
        n_free_nights > 0 & n_work_nights > 0 & is.finite(SD_f) & is.finite(SD_w),
        ((5 * SD_w) + (2 * SD_f)) / 7,
        NA_real_
      ),
      MSFsc_linear = if_else(
        n_free_nights > 0 & n_work_nights > 0 & is.finite(SO_f) & is.finite(SD_week),
        if_else(SD_f > SD_w, ((SO_f + (SD_week / 2) - 12) %% 24) + 12, MSF_linear),
        NA_real_
      ),
      MSF = MSF_linear,
      MSFsc = MSFsc_linear
    ) %>%
    select(
      person_id, mean_sleep_date, n_total_nights, n_free_nights, n_work_nights,
      person_weekend_avg_midpoint, MSF, MSFsc
    )
}

prepare_analysis_base <- function(phenotypes, covariates, scores, ancestry) {
  out <- phenotypes %>%
    left_join(covariates, by = "person_id") %>%
    left_join(scores, by = "person_id") %>%
    left_join(ancestry, by = "person_id") %>%
    mutate(
      date_of_birth = if ("date_of_birth" %in% names(.)) as.Date(date_of_birth) else as.Date(NA),
      age = if ("date_of_birth" %in% names(.)) {
        as.numeric(mean_sleep_date - date_of_birth) / 365.25
      } else {
        NA_real_
      },
      sex_concept = if ("sex_concept" %in% names(.)) as.character(sex_concept) else NA_character_,
      sex_concept = dplyr::case_when(
        is.na(sex_concept) ~ NA_character_,
        str_detect(str_to_lower(sex_concept), "female|woman|girl") ~ "Female",
        str_detect(str_to_lower(sex_concept), "male|man|boy") ~ "Male",
        TRUE ~ "Other/Unknown"
      ),
      sex_concept = factor(sex_concept, levels = c("Female", "Male", "Other/Unknown"))
    )

  out
}

prepare_phewas_base_data <- function(
  phewas_parquet,
  scores,
  ancestry,
  phenotypes,
  pc_cols
) {
  if (!file.exists(phewas_parquet)) stop("PheWAS parquet not found: ", phewas_parquet)

  raw <- arrow::read_parquet(phewas_parquet) %>%
    as_tibble() %>%
    mutate(person_id = as.character(person_id)) %>%
    left_join(scores, by = "person_id") %>%
    left_join(ancestry, by = "person_id") %>%
    left_join(phenotypes %>% select(person_id, mean_sleep_date), by = "person_id") %>%
    mutate(
      date_of_birth = if ("date_of_birth" %in% names(.)) as.Date(date_of_birth) else as.Date(NA),
      age = if ("date_of_birth" %in% names(.)) {
        as.numeric(mean_sleep_date - date_of_birth) / 365.25
      } else {
        NA_real_
      },
      sex_concept = if ("sex_concept" %in% names(.)) as.character(sex_concept) else NA_character_,
      sex_concept = dplyr::case_when(
        is.na(sex_concept) ~ NA_character_,
        str_detect(str_to_lower(sex_concept), "female|woman|girl") ~ "Female",
        str_detect(str_to_lower(sex_concept), "male|man|boy") ~ "Male",
        TRUE ~ "Other/Unknown"
      ),
      sex_concept = factor(sex_concept, levels = c("Female", "Male", "Other/Unknown"))
    )

  base_required <- c("score_raw", "age", "sex_concept", pc_cols)
  eligible <- raw %>%
    filter(complete.cases(across(all_of(intersect(base_required, names(.)))))) %>%
    mutate(score_z = as.numeric(scale(score_raw)))

  list(raw = raw, eligible = eligible)
}

build_cohort_flow_data <- function(
  phenotypes,
  scores,
  analysis_df,
  pc_cols,
  phewas_base = NULL
) {
  assoc_required <- c("score_raw", "age", "sex_concept", pc_cols)

  association_complete_n <- analysis_df %>%
    filter(complete.cases(across(all_of(intersect(assoc_required, names(.)))))) %>%
    distinct(person_id) %>%
    nrow()

  association_outcome_n <- function(outcome_col) {
    analysis_df %>%
      filter(complete.cases(across(all_of(intersect(c(assoc_required, outcome_col), names(.)))))) %>%
      distinct(person_id) %>%
      nrow()
  }

  flow_rows <- tibble(
    step_id = c(
      "sleep_cohort",
      "sleep_genetics",
      "association_complete",
      "assoc_weekend",
      "assoc_msf",
      "assoc_msfsc"
    ),
    parent_step_id = c(NA, "sleep_cohort", "sleep_genetics", "association_complete", "association_complete", "association_complete"),
    branch = c("Sleep", "Sleep", "Association", "Association", "Association", "Association"),
    step_label = c(
      "Sleep phenotype cohort",
      "Sleep + scored genetics",
      "Association covariate-complete",
      "Weekend midpoint model",
      "MSF model",
      "MSFsc model"
    ),
    n = c(
      nrow(phenotypes),
      analysis_df %>% filter(!is.na(score_raw)) %>% distinct(person_id) %>% nrow(),
      association_complete_n,
      association_outcome_n("person_weekend_avg_midpoint"),
      association_outcome_n("MSF"),
      association_outcome_n("MSFsc")
    ),
    x = c(1, 2, 3, 4, 4, 4),
    y = c(2.2, 2.2, 2.2, 3.2, 2.2, 1.2)
  )

  if (!is.null(phewas_base)) {
    phewas_rows <- tibble(
      step_id = c("phewas_cohort", "sleep_genetics_phewas", "phewas_complete"),
      parent_step_id = c(NA, "phewas_cohort", "sleep_genetics_phewas"),
      branch = c("PheWAS", "PheWAS", "PheWAS"),
      step_label = c(
        "PheWAS cohort",
        "Sleep + genetics + PheWAS",
        "PheWAS covariate-complete"
      ),
      n = c(
        phewas_base$raw %>% distinct(person_id) %>% nrow(),
        phewas_base$raw %>% filter(!is.na(score_raw), !is.na(mean_sleep_date)) %>% distinct(person_id) %>% nrow(),
        phewas_base$eligible %>% distinct(person_id) %>% nrow()
      ),
      x = c(1, 3, 4),
      y = c(0.4, 0.4, 0.4)
    )
    flow_rows <- bind_rows(flow_rows, phewas_rows)
  }

  flow_rows %>%
    rowwise() %>%
    mutate(
      pct_of_sleep = round(100 * n / max(flow_rows$n[flow_rows$step_id == "sleep_cohort"], 1), 2),
      pct_of_parent = if (is.na(parent_step_id)) {
        NA_real_
      } else {
        round(100 * n / max(flow_rows$n[flow_rows$step_id == parent_step_id], 1), 2)
      },
      label = sprintf("%s\nn = %s", step_label, format(n, big.mark = ","))
    ) %>%
    ungroup()
}

write_cohort_flow_outputs <- function(out_dir, flow_df) {
  tables_dir <- file.path(out_dir, "tables")
  plots_dir <- file.path(out_dir, "plots")

  readr::write_csv(flow_df, file.path(tables_dir, "cohort_flow_counts.csv"))

  edge_df <- flow_df %>%
    filter(!is.na(parent_step_id)) %>%
    left_join(
      flow_df %>% select(step_id, x_parent = x, y_parent = y),
      by = c("parent_step_id" = "step_id")
    )

  p <- ggplot() +
    geom_segment(
      data = edge_df,
      aes(x = x_parent + 0.18, y = y_parent, xend = x - 0.18, yend = y, color = branch),
      arrow = grid::arrow(length = grid::unit(0.18, "cm")),
      linewidth = 0.7,
      lineend = "round"
    ) +
    geom_label(
      data = flow_df,
      aes(x = x, y = y, label = label, fill = branch),
      linewidth = 0.25,
      size = 3.4,
      label.padding = grid::unit(0.18, "lines")
    ) +
    scale_fill_manual(values = c("Sleep" = "#d9edf7", "Association" = "#d5e8d4", "PheWAS" = "#fce5cd")) +
    scale_color_manual(values = c("Sleep" = "#2c7fb8", "Association" = "#238b45", "PheWAS" = "#d95f0e")) +
    coord_cartesian(xlim = c(0.7, 4.35), ylim = c(0, 3.6), clip = "off") +
    labs(
      title = "Participant flow for Angus midpoint PRS analysis",
      subtitle = "Largest available sleep-genetics cohort is used for association; PheWAS uses the sleep-genetics-EHR overlap",
      caption = "Source table: tables/cohort_flow_counts.csv"
    ) +
    theme_void() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "gray30"),
      plot.caption = element_text(size = 9, color = "gray35"),
      legend.position = "none",
      plot.margin = margin(15, 25, 15, 25)
    )

  ggsave(
    filename = file.path(plots_dir, "cohort_flow.png"),
    plot = p,
    width = 12,
    height = 6,
    dpi = 320,
    bg = "white"
  )
}

make_score_tertiles <- function(x) {
  idx <- dplyr::ntile(x, 3)
  factor(c("Low", "Medium", "High")[idx], levels = c("Low", "Medium", "High"))
}

safe_confint <- function(model, term) {
  out <- tryCatch(stats::confint.default(model, parm = term), error = function(e) NULL)
  if (is.null(out)) return(c(NA_real_, NA_real_))
  as.numeric(out[1, ])
}

extract_linear_term <- function(model, term, outcome, model_name, n_obs) {
  coef_mat <- summary(model)$coefficients
  if (!term %in% rownames(coef_mat)) {
    return(tibble(
      phenotype = outcome, model = model_name, term = term, n = n_obs,
      estimate_hours = NA_real_, estimate_minutes = NA_real_,
      std_error = NA_real_, p_value = NA_real_,
      ci_low_hours = NA_real_, ci_high_hours = NA_real_
    ))
  }
  ci <- safe_confint(model, term)
  tibble(
    phenotype = outcome,
    model = model_name,
    term = term,
    n = n_obs,
    estimate_hours = as.numeric(coef_mat[term, "Estimate"]),
    estimate_minutes = as.numeric(coef_mat[term, "Estimate"]) * 60,
    std_error = as.numeric(coef_mat[term, "Std. Error"]),
    p_value = as.numeric(coef_mat[term, "Pr(>|t|)"]),
    ci_low_hours = ci[[1]],
    ci_high_hours = ci[[2]]
  )
}

run_association_models <- function(analysis_df, out_dir, pc_cols) {
  outcome_cols <- c("person_weekend_avg_midpoint", "MSF", "MSFsc")
  plots_dir <- file.path(out_dir, "plots")
  tables_dir <- file.path(out_dir, "tables")

  assoc_results <- list()
  tertile_results <- list()
  tertile_stats <- list()
  plot_rows <- list()

  for (outcome in outcome_cols) {
    required <- c("score_raw", outcome, "age", "sex_concept", pc_cols)
    available_required <- intersect(required, names(analysis_df))
    df <- analysis_df %>%
      filter(complete.cases(across(all_of(available_required))))

    if (nrow(df) < 30) next

    df <- df %>%
      mutate(
        score_z = as.numeric(scale(score_raw)),
        score_tertile = make_score_tertiles(score_raw)
      )

    model_specs <- list(
      "PRS only" = as.formula(paste(outcome, "~ score_z")),
      "PRS + age + sex" = as.formula(paste(outcome, "~ score_z + age + sex_concept"))
    )

    if (length(pc_cols) > 0) {
      model_specs[["PRS + age + sex + PC1-PC10"]] <- as.formula(
        paste(outcome, "~ score_z + age + sex_concept +", paste(pc_cols, collapse = " + "))
      )
    }

    for (model_name in names(model_specs)) {
      fit <- stats::lm(model_specs[[model_name]], data = df)
      assoc_results[[length(assoc_results) + 1]] <- extract_linear_term(
        fit,
        term = "score_z",
        outcome = outcome,
        model_name = model_name,
        n_obs = nrow(df)
      )
    }

    tertile_formula_rhs <- c("score_tertile", "age", "sex_concept", pc_cols)
    tertile_formula <- as.formula(paste(outcome, "~", paste(tertile_formula_rhs, collapse = " + ")))
    tertile_fit <- stats::lm(tertile_formula, data = df)
    tertile_coef <- summary(tertile_fit)$coefficients

    for (term in c("score_tertileMedium", "score_tertileHigh")) {
      if (term %in% rownames(tertile_coef)) {
        ci <- safe_confint(tertile_fit, term)
        tertile_results[[length(tertile_results) + 1]] <- tibble(
          phenotype = outcome,
          term = term,
          comparison = sub("score_tertile", "", term),
          reference = "Low",
          n = nrow(df),
          estimate_hours = as.numeric(tertile_coef[term, "Estimate"]),
          estimate_minutes = as.numeric(tertile_coef[term, "Estimate"]) * 60,
          ci_low_hours = ci[[1]],
          ci_high_hours = ci[[2]],
          ci_low_minutes = ci[[1]] * 60,
          ci_high_minutes = ci[[2]] * 60,
          std_error = as.numeric(tertile_coef[term, "Std. Error"]),
          p_value = as.numeric(tertile_coef[term, "Pr(>|t|)"])
        )
      }
    }

    tertile_stats[[length(tertile_stats) + 1]] <- df %>%
      group_by(score_tertile) %>%
      summarise(
        phenotype = outcome,
        n = n(),
        mean_hours = mean(.data[[outcome]], na.rm = TRUE),
        sd_hours = sd(.data[[outcome]], na.rm = TRUE),
        mean_clock = format_clock(mean_hours),
        .groups = "drop"
      )

    plot_rows[[length(plot_rows) + 1]] <- df %>%
      transmute(
        phenotype = outcome,
        score_tertile = score_tertile,
        midpoint_hours = .data[[outcome]]
      )

    log_progress_event(
      "associations",
      "phenotype_completed",
      metrics = list(
        participants_used = nrow(df),
        models_fit = length(model_specs)
      ),
      details = list(phenotype = outcome)
    )
  }

  assoc_table <- bind_rows(assoc_results)
  tertile_table <- bind_rows(tertile_results)
  tertile_stats_table <- bind_rows(tertile_stats)
  plot_df <- bind_rows(plot_rows)

  forest_plot_df <- assoc_table %>%
    mutate(
      phenotype_label = factor(
        vapply(phenotype, phenotype_label, character(1)),
        levels = phenotype_label(c("person_weekend_avg_midpoint", "MSF", "MSFsc"))
      ),
      model = factor(model, levels = c("PRS only", "PRS + age + sex", "PRS + age + sex + PC1-PC10")),
      ci_low_minutes = ci_low_hours * 60,
      ci_high_minutes = ci_high_hours * 60
    )

  tertile_plot_df <- plot_df %>%
    mutate(
      phenotype_label = factor(
        vapply(phenotype, phenotype_label, character(1)),
        levels = phenotype_label(c("person_weekend_avg_midpoint", "MSF", "MSFsc"))
      ),
      midpoint_clock = format_clock(midpoint_hours)
    )

  readr::write_csv(assoc_table, file.path(tables_dir, "association_continuous_models.csv"))
  readr::write_csv(tertile_table, file.path(tables_dir, "association_tertile_models.csv"))
  readr::write_csv(tertile_stats_table, file.path(tables_dir, "association_tertile_summary.csv"))
  readr::write_csv(forest_plot_df, file.path(tables_dir, "association_forest_plot_data.csv"))
  readr::write_csv(tertile_plot_df, file.path(tables_dir, "midpoint_by_prs_tertile_plot_data.csv"))
  log_progress_event(
    "associations",
    "tables_written",
    status = "completed",
    metrics = list(
      continuous_rows = nrow(assoc_table),
      tertile_rows = nrow(tertile_table),
      tertile_summary_rows = nrow(tertile_stats_table)
    )
  )

  if (nrow(tertile_plot_df) > 0) {
    p_tertile <- ggplot(tertile_plot_df, aes(x = score_tertile, y = midpoint_hours, fill = score_tertile)) +
      geom_boxplot(outlier.alpha = 0.2, width = 0.68, color = "gray25") +
      stat_summary(fun = mean, geom = "point", shape = 23, size = 2.5, fill = "gold", color = "black") +
      facet_wrap(~ phenotype_label, scales = "free_y") +
      scale_fill_manual(values = c("Low" = "#c6dbef", "Medium" = "#6baed6", "High" = "#2171b5")) +
      labs(
        title = "Midpoint phenotypes by PRS tertile",
        subtitle = "Boxes show the interquartile range; diamonds mark phenotype means",
        x = "PRS tertile",
        y = "Midpoint (clock time, HH:MM)",
        caption = "Source table: tables/midpoint_by_prs_tertile_plot_data.csv"
      ) +
      theme_research() +
      theme(legend.position = "none")
    ggsave(
      filename = file.path(plots_dir, "midpoint_by_prs_tertile.png"),
      plot = p_tertile,
      width = 11,
      height = 6,
      dpi = 320,
      bg = "white"
    )
  }

  if (nrow(forest_plot_df) > 0) {
    p_forest <- forest_plot_df %>%
      ggplot(aes(x = estimate_minutes, y = phenotype_label, color = model)) +
      geom_point(position = position_dodge(width = 0.5), size = 2) +
      geom_errorbar(
        aes(xmin = ci_low_minutes, xmax = ci_high_minutes),
        position = position_dodge(width = 0.5),
        width = 0.2,
        orientation = "y"
      ) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
      labs(
        title = "Association of PRS per SD with midpoint phenotypes",
        subtitle = "Effect estimates are shown in minutes per 1 SD higher PRS",
        x = "Beta (minutes per SD higher PRS)",
        y = NULL,
        color = "Model",
        caption = "Source table: tables/association_forest_plot_data.csv"
      ) +
      theme_research()

    ggsave(
      filename = file.path(plots_dir, "association_forest_per_sd.png"),
      plot = p_forest,
      width = 10,
      height = 5,
      dpi = 320,
      bg = "white"
    )
  }

  list(
    continuous = assoc_table,
    tertiles = tertile_table,
    tertile_summary = tertile_stats_table
  )
}

run_phewas <- function(
  phewas_base,
  phecode_map_csv,
  out_dir,
  pc_cols,
  min_case_count = 20L
) {
  if (!file.exists(phecode_map_csv)) stop("Phecode map not found: ", phecode_map_csv)
  phe_df <- phewas_base$eligible

  phe_cols <- grep("^has_phe_", names(phe_df), value = TRUE)
  results <- vector("list", length(phe_cols))
  idx <- 1L
  total_phecodes <- length(phe_cols)
  processed_phecodes <- 0L
  log_progress_event(
    "phewas",
    "started",
    metrics = list(total_phecodes = total_phecodes)
  )

  for (phe_col in phe_cols) {
    processed_phecodes <- processed_phecodes + 1L
    had_col <- sub("^has_phe_", "had_phe_", phe_col)
    outcome <- phe_df[[phe_col]]
    if (had_col %in% names(phe_df)) {
      had_vals <- phe_df[[had_col]]
      outcome[!is.na(had_vals) & had_vals %in% c(TRUE, 1)] <- NA
    }

    dat <- phe_df %>%
      transmute(
        outcome = as.logical(outcome),
        score_z = score_z,
        age = age,
        sex_concept = sex_concept,
        across(any_of(pc_cols))
      ) %>%
      filter(!is.na(outcome))

    n_events <- sum(dat$outcome %in% TRUE, na.rm = TRUE)
    n_non_events <- sum(dat$outcome %in% FALSE, na.rm = TRUE)
    if (n_events < min_case_count || n_non_events < min_case_count) next

    formula_terms <- c("score_z", "age", "sex_concept", pc_cols)
    fit <- tryCatch(
      glm(
        as.formula(paste("outcome ~", paste(formula_terms, collapse = " + "))),
        family = binomial(link = "logit"),
        data = dat
      ),
      error = function(e) NULL
    )
    if (is.null(fit)) next

    coef_mat <- summary(fit)$coefficients
    if (!"score_z" %in% rownames(coef_mat)) next
    ci <- safe_confint(fit, "score_z")

    results[[idx]] <- tibble(
      phecode = sub("^has_phe_", "", phe_col),
      estimate = as.numeric(coef_mat["score_z", "Estimate"]),
      std_error = as.numeric(coef_mat["score_z", "Std. Error"]),
      z_value = as.numeric(coef_mat["score_z", "z value"]),
      p_value = as.numeric(coef_mat["score_z", "Pr(>|z|)"]),
      odds_ratio = exp(as.numeric(coef_mat["score_z", "Estimate"])),
      ci_lower_2_5 = exp(ci[[1]]),
      ci_upper_97_5 = exp(ci[[2]]),
      n = nrow(dat),
      n_events = n_events
    )
    idx <- idx + 1L

    if (processed_phecodes %% 100L == 0L || processed_phecodes == total_phecodes) {
      log_progress_event(
        "phewas",
        "progress",
        metrics = list(
          phecodes_processed = processed_phecodes,
          total_phecodes = total_phecodes,
          retained_results = idx - 1L,
          progress_pct = round(100 * processed_phecodes / max(total_phecodes, 1L), 2)
        )
      )
    }
  }

  results_df <- bind_rows(results)
  if (nrow(results_df) == 0) return(tibble())

  phemap <- readr::read_csv(phecode_map_csv, show_col_types = FALSE) %>%
    transmute(
      phecode_raw = as.character(PHECODE),
      phecode_join = normalize_phecode(PHECODE),
      concept_name = PHENOTYPE
    ) %>%
    distinct(phecode_join, .keep_all = TRUE)

  results_df <- results_df %>%
    mutate(phecode_join = normalize_phecode(phecode)) %>%
    left_join(phemap %>% select(-phecode_raw), by = "phecode_join") %>%
    mutate(
      fdr = p.adjust(p_value, method = "BH"),
      minus_log10_p = -log10(p_value),
      concept_name = if_else(is.na(concept_name) | !nzchar(concept_name), paste("Phecode", phecode), concept_name)
    ) %>%
    select(-phecode_join) %>%
    arrange(p_value) %>%
    mutate(phecode_index = row_number())

  tables_dir <- file.path(out_dir, "tables")
  plots_dir <- file.path(out_dir, "plots")

  readr::write_csv(results_df, file.path(tables_dir, "phewas_results.csv"))
  readr::write_csv(results_df, file.path(tables_dir, "phewas_manhattan_plot_data.csv"))

  volcano_plot_df <- results_df %>%
    mutate(
      label = if_else(p_value < 0.001, str_replace(concept_name, ",.*$", ""), NA_character_),
      siglevel = case_when(
        p_value < 1e-5 ~ "p < .00001",
        p_value < 1e-4 ~ "p < .0001",
        p_value < 1e-3 ~ "p < .001",
        TRUE ~ NA_character_
      ),
      siglevel = factor(siglevel, levels = c("p < .00001", "p < .0001", "p < .001"))
    ) %>%
    filter(is.finite(odds_ratio))

  readr::write_csv(volcano_plot_df, file.path(tables_dir, "phewas_volcano_plot_data.csv"))
  log_progress_event(
    "phewas",
    "results_written",
    status = "completed",
    metrics = list(
      phewas_rows = nrow(results_df),
      fdr_hits = sum(results_df$fdr < 0.05, na.rm = TRUE)
    )
  )

  p <- ggplot(results_df, aes(x = phecode_index, y = minus_log10_p, color = fdr < 0.05)) +
    geom_point(alpha = 0.85, size = 1.8) +
    geom_hline(
      yintercept = -log10(0.05 / max(nrow(results_df), 1)),
      linetype = "dashed",
      color = "firebrick"
    ) +
    scale_color_manual(values = c("FALSE" = "gray55", "TRUE" = "#1b9e77")) +
    labs(
      title = "Continuous PRS per SD PheWAS",
      subtitle = "Green points pass FDR < 0.05; dashed line shows Bonferroni threshold",
      x = "Phecode index",
      y = expression(-log[10](p)),
      color = "FDR < 0.05",
      caption = "Source table: tables/phewas_manhattan_plot_data.csv"
    ) +
    theme_research()

  ggsave(
    filename = file.path(plots_dir, "phewas_manhattan.png"),
    plot = p,
    width = 11,
    height = 5.5,
    dpi = 320,
    bg = "white"
  )

  if (nrow(volcano_plot_df) > 0) {
    p_volcano <- ggplot(volcano_plot_df, aes(x = odds_ratio, y = minus_log10_p)) +
      geom_point(color = "grey70", size = 1.8, alpha = 0.85) +
      geom_point(
        data = volcano_plot_df %>% filter(!is.na(siglevel)),
        aes(color = siglevel),
        size = 2.2,
        alpha = 0.95
      ) +
      geom_vline(xintercept = 1, color = "gray35", linewidth = 0.6) +
      geom_hline(yintercept = -log10(0.05), color = "#b2182b", linetype = "dashed", linewidth = 0.6) +
      geom_hline(
        yintercept = -log10(0.05 / max(nrow(results_df), 1)),
        color = "#2166ac",
        linetype = "dashed",
        linewidth = 0.6
      ) +
      geom_text(
        data = volcano_plot_df %>% filter(!is.na(label)),
        aes(label = label),
        check_overlap = TRUE,
        nudge_y = 0.1,
        size = 3
      ) +
      scale_color_manual(
        values = c("p < .00001" = "#7f0000", "p < .0001" = "#cb181d", "p < .001" = "#fb6a4a"),
        na.translate = FALSE,
        drop = FALSE
      ) +
      labs(
        title = "PheWAS volcano plot for continuous PRS",
        subtitle = "Legacy-style odds-ratio display; labels shown for p < 0.001; all finite ORs plotted",
        x = "Odds ratio per 1 SD higher PRS",
        y = expression(-log[10](p)),
        color = "Sig. level",
        caption = "Source table: tables/phewas_volcano_plot_data.csv"
      ) +
      theme_research()

    ggsave(
      filename = file.path(plots_dir, "phewas_volcano.png"),
      plot = p_volcano,
      width = 10,
      height = 8,
      dpi = 320,
      bg = "white"
    )
  }

  results_df
}

summary_flow_lookup <- function(flow_df, step_id) {
  out <- flow_df$n[match(step_id, flow_df$step_id)]
  if (length(out) == 0 || is.na(out)) return(NA_integer_)
  as.integer(out)
}

existing_summary_figures <- function(out_dir) {
  figure_specs <- tribble(
    ~path, ~heading, ~caption,
    "plots/cohort_flow.png", "Cohort flow", "Participant flow from the raw sleep cohort into the association and PheWAS branches.",
    "plots/association_forest_per_sd.png", "Primary association forest plot", "Continuous PRS associations for the all-ancestry analysis.",
    "plots/association_forest_per_sd_eur.png", "EUR association forest plot", "Continuous PRS associations for the EUR-restricted sensitivity analysis.",
    "plots/midpoint_by_prs_tertile.png", "PRS tertile boxplots", "All-ancestry tertile distributions for the three midpoint phenotypes, with y-axis labels shown as clock time.",
    "plots/midpoint_by_prs_tertile_no_outliers.png", "PRS tertile boxplots without outlier points", "All-ancestry tertile distributions with outlier points hidden for display and the y-axis cropped to the non-outlier whisker range.",
    "plots/midpoint_regression_lines.png", "PRS regression lines", "Unadjusted regression-line views of the three midpoint phenotypes, with y-axis labels shown as clock time.",
    "plots/phewas_manhattan.png", "PheWAS Manhattan plot", "Phenome-wide signal strength across tested phecodes.",
    "plots/phewas_volcano.png", "PheWAS volcano plot", "Odds-ratio view of the phenome-wide scan."
  )

  figure_specs %>%
    filter(file.exists(file.path(out_dir, path)))
}

read_summary_tertile_trends <- function(out_dir) {
  tertile_path <- file.path(out_dir, "tables", "midpoint_by_prs_tertile_plot_data_by_cohort.csv")
  if (!file.exists(tertile_path)) return(tibble())

  readr::read_csv(tertile_path, show_col_types = FALSE) %>%
    filter(cohort == "All") %>%
    mutate(score_tertile = factor(score_tertile, levels = c("Low", "Medium", "High"))) %>%
    group_by(phenotype, phenotype_label, score_tertile) %>%
    summarise(
      n = n(),
      mean_midpoint = mean(midpoint_hours, na.rm = TRUE),
      median_midpoint = median(midpoint_hours, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    tidyr::pivot_wider(
      names_from = score_tertile,
      values_from = c(n, mean_midpoint, median_midpoint),
      names_glue = "{.value}_{score_tertile}"
    ) %>%
    mutate(
      low_to_high_minutes = (mean_midpoint_High - mean_midpoint_Low) * 60,
      direction = dplyr::case_when(
        mean_midpoint_High > mean_midpoint_Low ~ "increase",
        mean_midpoint_High < mean_midpoint_Low ~ "decrease",
        TRUE ~ "flat"
      ),
      mean_clock_Low = format_clock(mean_midpoint_Low),
      mean_clock_Medium = format_clock(mean_midpoint_Medium),
      mean_clock_High = format_clock(mean_midpoint_High)
    ) %>%
    arrange(match(phenotype, c("person_weekend_avg_midpoint", "MSF", "MSFsc")))
}

build_summary_md_lines <- function(
  out_dir,
  association_continuous,
  phewas_results,
  pc_count,
  flow_df
) {
  top_assoc <- association_continuous %>%
    arrange(p_value) %>%
    slice_head(n = 6)

  primary_assoc <- association_continuous %>%
    filter(model == "PRS + age + sex + PC1-PC10") %>%
    arrange(match(phenotype, c("person_weekend_avg_midpoint", "MSF", "MSFsc")))

  if (nrow(primary_assoc) == 0) {
    primary_assoc <- association_continuous %>%
      arrange(match(phenotype, c("person_weekend_avg_midpoint", "MSF", "MSFsc")), p_value)
  }

  strongest_primary <- primary_assoc %>%
    arrange(p_value) %>%
    slice_head(n = 1)

  top_phewas <- phewas_results %>%
    arrange(p_value) %>%
    slice_head(n = 10)

  best_phewas <- top_phewas %>%
    slice_head(n = 1)

  phewas_sig_n <- sum(!is.na(phewas_results$fdr) & phewas_results$fdr < 0.05)
  figures <- existing_summary_figures(out_dir)
  tertile_trends <- read_summary_tertile_trends(out_dir)

  lines <- c(
    "# PRS midpoint collaborator summary",
    "",
    "## Key findings",
    ""
  )

  if (nrow(strongest_primary) > 0) {
    row <- strongest_primary[1, ]
    lines <- c(
      lines,
      sprintf(
        "- The strongest primary adjusted association was **%s**: **%.2f minutes later per 1 SD higher PRS** (`p = %.3g`, `n = %s`).",
        phenotype_label(row$phenotype),
        row$estimate_minutes,
        row$p_value,
        format(row$n, big.mark = ",", trim = TRUE)
      )
    )
  }

  if (nrow(primary_assoc) > 0 && all(primary_assoc$estimate_minutes > 0, na.rm = TRUE)) {
    lines <- c(
      lines,
      "- All three midpoint phenotypes moved in the same direction: higher PRS corresponded to later sleep timing, with the clearest signal for weekend average midpoint."
    )
  }

  if (nrow(tertile_trends) > 0 && all(tertile_trends$direction == "increase", na.rm = TRUE)) {
    lines <- c(
      lines,
      sprintf(
        "- The tertile plots showed the same overall pattern: **Low < Medium < High** for all three midpoint phenotypes, with mean Low-to-High shifts ranging from **%.2f to %.2f minutes**.",
        min(tertile_trends$low_to_high_minutes, na.rm = TRUE),
        max(tertile_trends$low_to_high_minutes, na.rm = TRUE)
      )
    )
  }

  if (nrow(best_phewas) > 0) {
    phe <- best_phewas[1, ]
    if (phewas_sig_n == 0) {
      lines <- c(
        lines,
        sprintf(
          "- The PheWAS produced **no FDR-significant hits** across %s tested phecodes; the smallest p-value was for **%s (%s)** with **OR %.2f** (`p = %.3g`, `FDR = %.3g`).",
          format(nrow(phewas_results), big.mark = ",", trim = TRUE),
          phe$phecode,
          phe$concept_name %||% "Unknown phenotype",
          phe$odds_ratio,
          phe$p_value,
          phe$fdr
        )
      )
    } else {
      lines <- c(
        lines,
        sprintf(
          "- The PheWAS produced **%d FDR-significant hits** across %s tested phecodes; the top signal was **%s (%s)** with **OR %.2f** (`p = %.3g`, `FDR = %.3g`).",
          phewas_sig_n,
          format(nrow(phewas_results), big.mark = ",", trim = TRUE),
          phe$phecode,
          phe$concept_name %||% "Unknown phenotype",
          phe$odds_ratio,
          phe$p_value,
          phe$fdr
        )
      )
    }
  }

  lines <- c(
    lines,
    "",
    "## Cohort summary",
    "",
    sprintf("- Sleep phenotype cohort: %s", format(summary_flow_lookup(flow_df, "sleep_cohort"), big.mark = ",", trim = TRUE)),
    sprintf("- Sleep + scored genetics overlap: %s", format(summary_flow_lookup(flow_df, "sleep_genetics"), big.mark = ",", trim = TRUE)),
    sprintf("- Association covariate-complete cohort: %s", format(summary_flow_lookup(flow_df, "association_complete"), big.mark = ",", trim = TRUE)),
    sprintf("- Weekend midpoint model n: %s", format(summary_flow_lookup(flow_df, "assoc_weekend"), big.mark = ",", trim = TRUE)),
    sprintf("- MSF model n: %s", format(summary_flow_lookup(flow_df, "assoc_msf"), big.mark = ",", trim = TRUE)),
    sprintf("- MSFsc model n: %s", format(summary_flow_lookup(flow_df, "assoc_msfsc"), big.mark = ",", trim = TRUE)),
    if (!is.na(summary_flow_lookup(flow_df, "sleep_genetics_phewas"))) {
      sprintf("- Sleep + genetics + PheWAS overlap: %s", format(summary_flow_lookup(flow_df, "sleep_genetics_phewas"), big.mark = ",", trim = TRUE))
    } else {
      character(0)
    },
    sprintf("- Available ancestry PCs in primary adjusted models: %d", pc_count),
    "- Midpoint phenotypes compared: `person_weekend_avg_midpoint`, `MSF`, `MSFsc`",
    "",
    "## Phenotype definitions",
    "",
    "- `person_weekend_avg_midpoint`: participant-level circular mean of free-day sleep midpoints, then converted to the noon-to-noon linearized scale used downstream.",
    "- `MSF`: midpoint of sleep on free days, calculated as `SO_f + SD_f / 2`, where `SO_f` is mean free-day sleep onset and `SD_f` is mean free-day sleep duration.",
    "- `MSFsc`: sleep-corrected midpoint of sleep on free days. This uses `SO_f + SD_week / 2`, where `SD_week = (5 * SD_w + 2 * SD_f) / 7`, and applies the correction when free-day sleep duration exceeds work-day sleep duration.",
    "",
    "## Methods",
    "",
    "- **Association models:** linear regression of each midpoint phenotype on PRS. The report includes PRS-only, PRS + age + sex, and the primary adjusted model PRS + age + sex + PC1-PC10; the PRS exposure is analyzed per 1 SD higher score.",
    "- **PheWAS models:** logistic regression of each phecode outcome on the continuous PRS, adjusted for age, sex, and PC1-PC10, using the sleep + genetics + PheWAS overlap cohort. Only phecodes meeting the configured minimum case count are tested.",
    "",
    "## Main figures",
    ""
  )

  if (nrow(figures) > 0) {
    for (i in seq_len(nrow(figures))) {
      fig <- figures[i, ]
      lines <- c(
        lines,
        paste0("### ", fig$heading),
        "",
        fig$caption,
        "",
        sprintf("![%s](%s)", fig$heading, fig$path),
        ""
      )
    }
  } else {
    lines <- c(lines, "_No figures were found in `plots/` at summary-generation time._", "")
  }

  lines <- c(
    lines,
    "## Output files",
    "",
    "- Continuous per-SD model table: `tables/association_continuous_models.csv`",
    "- Tertile model table: `tables/association_tertile_models.csv`",
    "- Tertile descriptive table: `tables/association_tertile_summary.csv`",
    "- Cohort flow table: `tables/cohort_flow_counts.csv`"
  )

  if (file.exists(file.path(out_dir, "regenerate_figures.R"))) {
    lines <- c(lines, "- Figure regeneration script: `regenerate_figures.R`")
  }
  if (file.exists(file.path(out_dir, "tables", "association_forest_plot_data_by_cohort.csv"))) {
    lines <- c(lines, "- Cohort-specific association plot data: `tables/association_forest_plot_data_by_cohort.csv`")
  }
  if (file.exists(file.path(out_dir, "tables", "midpoint_by_prs_tertile_plot_data_by_cohort.csv"))) {
    lines <- c(lines, "- Cohort-specific tertile plot data: `tables/midpoint_by_prs_tertile_plot_data_by_cohort.csv`")
  }
  if (file.exists(file.path(out_dir, "tables", "midpoint_by_prs_tertile_plot_data_by_cohort_no_outliers.csv"))) {
    lines <- c(lines, "- No-outlier tertile plot data: `tables/midpoint_by_prs_tertile_plot_data_by_cohort_no_outliers.csv`")
  }
  if (file.exists(file.path(out_dir, "tables", "midpoint_regression_equations_by_cohort.csv"))) {
    lines <- c(lines, "- Regression equation table: `tables/midpoint_regression_equations_by_cohort.csv`")
  }
  if (nrow(phewas_results) > 0) {
    lines <- c(
      lines,
      "- PheWAS results table: `tables/phewas_results.csv`",
      "- PheWAS volcano plot data: `tables/phewas_volcano_plot_data.csv`"
    )
  }

  if (nrow(primary_assoc) > 0) {
    lines <- c(lines, "", "## Primary adjusted association results", "")
    for (i in seq_len(nrow(primary_assoc))) {
      row <- primary_assoc[i, ]
      lines <- c(
        lines,
        sprintf(
          "- %s | beta = %.2f minutes per SD | p = %.3g | n = %s",
          row$phenotype,
          row$estimate_minutes,
          row$p_value,
          format(row$n, big.mark = ",", trim = TRUE)
        )
      )
    }
  }

  if (nrow(top_assoc) > 0) {
    lines <- c(lines, "", "## Top association rows", "")
    for (i in seq_len(nrow(top_assoc))) {
      row <- top_assoc[i, ]
      lines <- c(
        lines,
        sprintf(
          "- %s | %s | beta = %.2f minutes per SD | p = %.3g | n = %s",
          row$phenotype,
          row$model,
          row$estimate_minutes,
          row$p_value,
          format(row$n, big.mark = ",", trim = TRUE)
        )
      )
    }
  }

  if (nrow(tertile_trends) > 0) {
    lines <- c(lines, "", "## Tertile trend check", "", "| Phenotype | Low mean | Medium mean | High mean | Low-to-High shift (min) | Direction |", "| --- | --- | --- | --- | ---: | --- |")
    for (i in seq_len(nrow(tertile_trends))) {
      row <- tertile_trends[i, ]
      lines <- c(
        lines,
        sprintf(
          "| %s | %s | %s | %s | %.2f | %s |",
          row$phenotype_label,
          row$mean_clock_Low,
          row$mean_clock_Medium,
          row$mean_clock_High,
          row$low_to_high_minutes,
          row$direction
        )
      )
    }
  }

  if (nrow(top_phewas) > 0) {
    lines <- c(lines, "", "## Top PheWAS rows", "", "| Phecode | Phenotype | OR per SD | p | FDR | Cases |", "| --- | --- | ---: | ---: | ---: | ---: |")
    for (i in seq_len(nrow(top_phewas))) {
      row <- top_phewas[i, ]
      lines <- c(
        lines,
        sprintf(
          "| %s | %s | %.2f | %.3g | %.3g | %s |",
          row$phecode,
          row$concept_name %||% "Unknown phenotype",
          row$odds_ratio,
          row$p_value,
          row$fdr,
          format(row$n_events, big.mark = ",", trim = TRUE)
        )
      )
    }
  }

  lines
}

write_summary_md_from_tables <- function(
  out_dir,
  association_continuous,
  phewas_results,
  pc_count,
  flow_df
) {
  summary_path <- file.path(out_dir, "summary.md")
  writeLines(
    build_summary_md_lines(
      out_dir = out_dir,
      association_continuous = association_continuous,
      phewas_results = phewas_results,
      pc_count = pc_count,
      flow_df = flow_df
    ),
    summary_path
  )
}

write_summary_md <- function(
  out_dir,
  association_results,
  phewas_results,
  analysis_df,
  pc_cols,
  flow_df
) {
  write_summary_md_from_tables(
    out_dir = out_dir,
    association_continuous = association_results$continuous,
    phewas_results = phewas_results,
    pc_count = length(pc_cols),
    flow_df = flow_df
  )
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  out_dir <- args$out_dir
  tables_dir <- file.path(out_dir, "tables")
  plots_dir <- file.path(out_dir, "plots")
  dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(tables_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

  covariates_path <- resolve_covariates_path(args$covariates_parquet)

  nightly <- arrow::read_parquet(args$nightly_parquet)
  phenotypes <- build_midpoint_phenotypes(nightly)
  scores <- load_score_file(args$score_file)
  ancestry <- load_ancestry_features(args$ancestry_tsv, args$pc_count)
  covariates <- load_covariates(covariates_path)
  log_progress_event(
    "analysis",
    "inputs_loaded",
    metrics = list(
      phenotype_rows = nrow(phenotypes),
      score_rows = nrow(scores),
      ancestry_rows = nrow(ancestry),
      covariate_rows = nrow(covariates)
    )
  )

  analysis_df <- prepare_analysis_base(phenotypes, covariates, scores, ancestry)
  pc_cols <- intersect(names(analysis_df), paste0("pca_", seq_len(args$pc_count)))
  log_progress_event(
    "analysis",
    "analysis_dataset_ready",
    metrics = list(
      participants = nrow(analysis_df),
      pc_count = length(pc_cols)
    )
  )

  phewas_base <- NULL
  if (!isTRUE(args$skip_phewas)) {
    phewas_base <- prepare_phewas_base_data(
      phewas_parquet = args$phewas_parquet,
      scores = scores,
      ancestry = ancestry,
      phenotypes = phenotypes,
      pc_cols = pc_cols
    )
  }

  association_results <- run_association_models(analysis_df, out_dir, pc_cols)
  phewas_results <- tibble()
  if (!isTRUE(args$skip_phewas)) {
    phewas_results <- run_phewas(
      phewas_base = phewas_base,
      phecode_map_csv = args$phecode_map_csv,
      out_dir = out_dir,
      pc_cols = pc_cols,
      min_case_count = args$min_case_count
    )
  }

  flow_df <- build_cohort_flow_data(
    phenotypes = phenotypes,
    scores = scores,
    analysis_df = analysis_df,
    pc_cols = pc_cols,
    phewas_base = phewas_base
  )
  write_cohort_flow_outputs(out_dir, flow_df)
  write_summary_md(out_dir, association_results, phewas_results, analysis_df, pc_cols, flow_df)
  log_progress_event(
    "analysis",
    "completed",
    status = "completed",
    metrics = list(
      participants = nrow(analysis_df),
      continuous_rows = nrow(association_results$continuous),
      tertile_rows = nrow(association_results$tertiles),
      phewas_rows = nrow(phewas_results)
    ),
    details = list(output_dir = out_dir)
  )
  message("Done. Outputs written to ", normalizePath(out_dir))
}

if (sys.nframe() == 0L) {
  main()
}
