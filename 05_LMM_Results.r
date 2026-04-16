# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .r
#       format_name: percent
#       format_version: "1.3"
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# # 05_6_LMM_Results.r
# Plotting/report layer for precomputed 04_5 model predictions.
#
# This script intentionally does NOT load model .rds files. It consumes
# cached outputs from 05_5_Precompute_Predictions.r and builds figures/tables
# in the structure of legacy 05_LMM_Results.r, extended for new model outputs.

# %%
required_packages <- c(
  "dplyr", "tidyr", "readr", "stringr", "ggplot2", "scales", "tibble", "purrr"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install them before running 05_6_LMM_Results.r."
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(ggplot2)
  library(scales)
  library(tibble)
  library(purrr)
})

set.seed(123)

# %% [markdown]
# ## Configuration

# %%
PRECOMP_OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR_05", "results_05")
PRECOMP_TABLE_DIR <- file.path(PRECOMP_OUTPUT_DIR, "tables")

OUTPUT_DIR <- Sys.getenv("PLOT_DIR", file.path(PRECOMP_OUTPUT_DIR, "plots"))
TABLE_DIR <- file.path(OUTPUT_DIR, "tables")
PLOT_DIR <- OUTPUT_DIR

NOTEBOOK_INLINE <- tolower(Sys.getenv("NOTEBOOK_INLINE_05", "true")) %in% c("true", "1", "yes")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)
if (!dir.exists(TABLE_DIR)) dir.create(TABLE_DIR, recursive = TRUE)
if (!dir.exists(PLOT_DIR)) dir.create(PLOT_DIR, recursive = TRUE)

analysis_log <- character()
log_msg <- function(txt) {
  message(txt)
  analysis_log <<- c(analysis_log, paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), txt))
}

`%||%` <- function(a, b) {
  if (!is.null(a) && length(a) > 0 && !all(is.na(a))) a else b
}

log_msg("Configuration loaded.")
log_msg(paste("PRECOMP_TABLE_DIR:", PRECOMP_TABLE_DIR))
log_msg(paste("OUTPUT_DIR:", OUTPUT_DIR))
log_msg(paste("NOTEBOOK_INLINE:", NOTEBOOK_INLINE))

# %% [markdown]
# ## Helpers

# %%
format_time_axis <- function(x) {
  x_norm <- x %% 24
  hrs <- floor(x_norm)
  mins <- round((x_norm - hrs) * 60)
  sprintf("%02d:%02d", hrs, mins)
}

format_duration_axis <- function(x) {
  hrs <- floor(x)
  mins <- round((x - hrs) * 60)
  sprintf("%dh %02dm", hrs, mins)
}

normalize_weekend_level <- function(x) {
  sx <- as.character(x)
  sx <- tolower(sx)
  sx <- gsub("[^a-z0-9]+", "", sx)
  out <- rep(NA_character_, length(sx))
  out[grepl("weekend", sx) | sx %in% c("true", "t", "1", "yes", "y")] <- "Weekend"
  out[grepl("weekday", sx) | sx %in% c("false", "f", "0", "no", "n")] <- "Weekday"
  out
}

clean_level_label <- function(x, unknown = "All") {
  sx <- as.character(x)
  sx[is.na(sx) | sx == "" | tolower(sx) == "na"] <- unknown
  sx
}

clean_employment <- function(x) {
  sx <- x %>%
    as.character() %>%
    str_replace_all("_", " ") %>%
    str_squish()

  dplyr::case_when(
    sx %in% c("Working", "Employed For Wages", "Self Employed") ~ "Working",
    sx %in% c("Student") ~ "Student",
    sx %in% c("Retired") ~ "Retired",
    sx %in% c("Not Working", "Out Of Work", "Out Of Work Less Than One", "Out Of Work One Or More", "Homemaker", "Unable To Work") ~ "Not Working",
    TRUE ~ str_to_title(sx)
  )
}

employment_levels_ordered <- function(x) {
  factor(clean_employment(x), levels = c("Working", "Student", "Retired", "Not Working"))
}

normalize_outcome_variant <- function(x) {
  sx <- tolower(trimws(as.character(x)))
  sx[is.na(sx) | sx == "" | sx == "na"] <- "primary"
  sx
}

robust_limits <- function(
  x,
  lower_q = 0.01,
  upper_q = 0.99,
  pad_frac = 0.05,
  hard_min = NA_real_,
  hard_max = NA_real_
) {
  vals <- as.numeric(x)
  vals <- vals[is.finite(vals)]
  if (length(vals) == 0) {
    return(c(ifelse(is.finite(hard_min), hard_min, 0), ifelse(is.finite(hard_max), hard_max, 1)))
  }

  q <- suppressWarnings(stats::quantile(vals, probs = c(lower_q, upper_q), na.rm = TRUE, names = FALSE))
  lo <- q[[1]]
  hi <- q[[2]]

  if (!is.finite(lo) || !is.finite(hi)) {
    lo <- min(vals, na.rm = TRUE)
    hi <- max(vals, na.rm = TRUE)
  }

  if (!is.finite(lo) || !is.finite(hi)) {
    lo <- 0
    hi <- 1
  }

  if (lo == hi) {
    lo <- lo - 0.5
    hi <- hi + 0.5
  }

  pad <- (hi - lo) * pad_frac
  lims <- c(lo - pad, hi + pad)

  if (is.finite(hard_min)) lims[[1]] <- max(lims[[1]], hard_min)
  if (is.finite(hard_max)) lims[[2]] <- min(lims[[2]], hard_max)

  if (!is.finite(lims[[1]]) || !is.finite(lims[[2]]) || lims[[1]] >= lims[[2]]) {
    lims <- c(min(vals, na.rm = TRUE), max(vals, na.rm = TRUE))
    if (lims[[1]] == lims[[2]]) lims <- lims + c(-0.5, 0.5)
  }

  lims
}

safe_read_csv <- function(names, required = FALSE) {
  names <- as.character(names)
  candidates <- file.path(PRECOMP_TABLE_DIR, names)
  hit <- candidates[file.exists(candidates)]
  if (length(hit) == 0) {
    txt <- paste("Missing input table(s):", paste(candidates, collapse = " | "))
    if (required) stop(txt)
    log_msg(txt)
    return(tibble())
  }
  path <- hit[[1]]
  out <- readr::read_csv(path, show_col_types = FALSE)
  attr(out, "source_path") <- path
  log_msg(paste("Loaded:", path, "| rows:", nrow(out)))
  out
}

require_cols <- function(df, cols, table_name) {
  missing <- setdiff(cols, names(df))
  if (length(missing) > 0) {
    log_msg(paste("Skipping", table_name, "- missing columns:", paste(missing, collapse = ", ")))
    return(FALSE)
  }
  TRUE
}

write_table <- function(df, csv_name) {
  path <- file.path(TABLE_DIR, csv_name)
  readr::write_csv(df, path)
  saveRDS(df, sub("\\.csv$", ".rds", path))
  if (requireNamespace("arrow", quietly = TRUE)) {
    arrow::write_parquet(df, sub("\\.csv$", ".parquet", path))
  }
  log_msg(paste("Wrote table:", path))
}

save_plot <- function(plot_obj, filename, width = 12, height = 7, dpi = 300) {
  if (exists("PLOT_ORIGIN_LABEL", inherits = TRUE) && nzchar(PLOT_ORIGIN_LABEL)) {
    existing_caption <- plot_obj$labels$caption %||% ""
    merged_caption <- if (nzchar(existing_caption)) {
      paste(existing_caption, PLOT_ORIGIN_LABEL, sep = "\n")
    } else {
      PLOT_ORIGIN_LABEL
    }
    plot_obj <- plot_obj +
      labs(caption = merged_caption) +
      theme(
        plot.caption = element_text(hjust = 0, size = 9, color = "gray35"),
        plot.caption.position = "plot"
      )
  }

  out <- file.path(PLOT_DIR, filename)
  ggsave(filename = out, plot = plot_obj, width = width, height = height, dpi = dpi)
  log_msg(paste("Saved plot:", out))
  if (isTRUE(NOTEBOOK_INLINE)) print(plot_obj)
}

# %% [markdown]
# ## Load Precomputed Inputs

# %%
predictions_all <- safe_read_csv(c("predictions_all.csv"), required = TRUE)
model_inventory <- safe_read_csv(c("model_inventory.csv"), required = FALSE)

derived_main <- safe_read_csv(c("derived_duration_midpoint_main_grid.csv"), required = FALSE)
derived_main_compare <- safe_read_csv(c("derived_vs_direct_main_grid.csv"), required = FALSE)
derived_age <- safe_read_csv(c("derived_duration_midpoint_age.csv"), required = FALSE)
derived_age_compare <- safe_read_csv(c("derived_vs_direct_age.csv"), required = FALSE)

weekend_contrasts <- safe_read_csv(c("weekend_contrasts.csv"), required = FALSE)
dst_contrasts <- safe_read_csv(c("dst_contrasts.csv"), required = FALSE)
emmeans_marginalized <- safe_read_csv(c("emmeans_marginalized.csv"), required = FALSE)

write_table(predictions_all, "predictions_all_input_copy.csv")
if (nrow(model_inventory) > 0) write_table(model_inventory, "model_inventory.csv")

if (!"outcome_variant" %in% names(predictions_all)) {
  predictions_all$outcome_variant <- "primary"
}
predictions_all$outcome_variant <- normalize_outcome_variant(predictions_all$outcome_variant)
primary_predictions <- predictions_all %>% filter(outcome_variant == "primary")

build_plot_origin_label <- function(pred_df, inventory_df, precomp_dir) {
  src <- attr(pred_df, "source_path") %||% file.path(precomp_dir, "tables", "predictions_all.csv")
  batch_vals <- if ("batch" %in% names(pred_df)) sort(unique(as.character(pred_df$batch))) else character()

  method_vals <- character()
  if ("model_method" %in% names(pred_df)) {
    method_vals <- sort(unique(na.omit(as.character(pred_df$model_method))))
  }
  if (length(method_vals) == 0 && "model_method" %in% names(inventory_df)) {
    method_vals <- sort(unique(na.omit(as.character(inventory_df$model_method))))
  }

  model_file_n <- if ("model_file" %in% names(pred_df)) {
    length(unique(na.omit(as.character(pred_df$model_file))))
  } else if ("model_file" %in% names(inventory_df)) {
    length(unique(na.omit(as.character(inventory_df$model_file))))
  } else {
    NA_integer_
  }

  paste0(
    "Model origin: ", src,
    " | Methods: ", if (length(method_vals) > 0) paste(method_vals, collapse = ", ") else "unknown",
    " | Batches: ", if (length(batch_vals) > 0) paste(batch_vals, collapse = ", ") else "unknown",
    " | Model files: ", if (!is.na(model_file_n)) as.character(model_file_n) else "unknown"
  )
}

PLOT_ORIGIN_LABEL <- build_plot_origin_label(predictions_all, model_inventory, PRECOMP_OUTPUT_DIR)
log_msg(paste("Plot provenance label:", PLOT_ORIGIN_LABEL))

# %% [markdown]
# ## Figure 1: Employment x Weekend (Legacy-style Core Figure)

# %%
if (require_cols(
  primary_predictions,
  c("analysis", "outcome", "estimate", "conf.low", "conf.high", "employment_status", "is_weekend_factor", "batch"),
  "predictions_all"
)) {
  employment_weekend <- primary_predictions %>%
    filter(analysis == "employment_x_weekend") %>%
    mutate(
      outcome = factor(outcome, levels = c("onset", "midpoint", "offset", "duration")),
      day_type = factor(normalize_weekend_level(is_weekend_factor), levels = c("Weekday", "Weekend")),
      employment_clean = employment_levels_ordered(employment_status),
      employment_clean = str_wrap(employment_clean, width = 25),
      batch = factor(batch, levels = c("base", "dst"))
    )

  write_table(employment_weekend, "employment_x_weekend_plot_data.csv")

  timing_df <- employment_weekend %>% filter(outcome %in% c("onset", "midpoint", "offset"))
  if (nrow(timing_df) > 0) {
    p_timing <- ggplot(timing_df, aes(x = estimate, y = employment_clean, color = day_type)) +
      geom_line(aes(group = interaction(batch, employment_clean)), color = "gray85", linewidth = 0.6) +
      geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0, linewidth = 0.9, alpha = 0.9) +
      geom_point(size = 2.4) +
      facet_grid(batch ~ outcome, scales = "free_x") +
      scale_color_manual(values = c("Weekday" = "#0072B2", "Weekend" = "#D55E00"), drop = FALSE) +
      scale_x_continuous(labels = format_time_axis) +
      labs(
        title = "Sleep Timing by Employment and Weekend Status",
        subtitle = "Marginal predictions from precomputed grids",
        x = "Predicted Clock Time",
        y = NULL,
        color = NULL
      ) +
      theme_classic(base_size = 13)

    save_plot(p_timing, "employment_weekend_timing.png", width = 15, height = 8)
  }

  duration_df <- employment_weekend %>% filter(outcome == "duration")
  if (nrow(duration_df) > 0) {
    p_duration <- ggplot(duration_df, aes(x = estimate, y = employment_clean, color = day_type)) +
      geom_line(aes(group = interaction(batch, employment_clean)), color = "gray85", linewidth = 0.6) +
      geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0, linewidth = 0.9, alpha = 0.9) +
      geom_point(size = 2.4) +
      facet_wrap(~ batch, ncol = 1, scales = "free_x") +
      scale_color_manual(values = c("Weekday" = "#0072B2", "Weekend" = "#D55E00"), drop = FALSE) +
      scale_x_continuous(labels = format_duration_axis) +
      labs(
        title = "Sleep Duration by Employment and Weekend Status",
        x = "Predicted Sleep Duration",
        y = NULL,
        color = NULL
      ) +
      theme_classic(base_size = 13)

    save_plot(p_duration, "employment_weekend_duration.png", width = 11, height = 8)
  }

}

# %% [markdown]
# ## Figure 2: Main Effects Panel (Demographics, Employment, Weekend)

# %%
main_effects <- tibble()

sex_main <- tibble()
if (require_cols(primary_predictions, c("analysis", "sex_concept", "outcome", "batch", "estimate", "conf.low", "conf.high"), "predictions_all")) {
  sex_main <- primary_predictions %>%
    filter(analysis == "sex_main") %>%
    transmute(
      outcome,
      batch,
      category = "Demographics",
      term = clean_employment(sex_concept),
      estimate,
      conf.low,
      conf.high
    )
}

employment_main <- tibble()
if (require_cols(primary_predictions, c("analysis", "employment_status", "outcome", "batch", "estimate", "conf.low", "conf.high"), "predictions_all")) {
  employment_main <- primary_predictions %>%
    filter(analysis == "employment_x_weekend") %>%
    group_by(outcome, batch, employment_status) %>%
    summarize(
      estimate = mean(estimate, na.rm = TRUE),
      conf.low = mean(conf.low, na.rm = TRUE),
      conf.high = mean(conf.high, na.rm = TRUE),
      .groups = "drop"
    ) %>%
    transmute(
      outcome,
      batch,
      category = "Employment",
      term = employment_levels_ordered(employment_status),
      estimate,
      conf.low,
      conf.high
    )
}

weekend_main <- tibble()
if (require_cols(primary_predictions, c("analysis", "is_weekend_factor", "outcome", "batch", "estimate", "conf.low", "conf.high"), "predictions_all")) {
  weekend_main <- primary_predictions %>%
    filter(analysis == "weekend_main") %>%
    transmute(
      outcome,
      batch,
      category = "Weekend Effect",
      term = normalize_weekend_level(is_weekend_factor),
      estimate,
      conf.low,
      conf.high
    )
}

main_effects <- bind_rows(sex_main, employment_main, weekend_main) %>%
  mutate(
    outcome = factor(outcome, levels = c("onset", "midpoint", "offset", "duration")),
    category = factor(category, levels = c("Demographics", "Employment", "Weekend Effect")),
    batch = factor(batch, levels = c("base", "dst"))
  )

if (nrow(main_effects) > 0) {
  write_table(main_effects, "main_effects.csv")

  main_timing <- main_effects %>% filter(outcome %in% c("onset", "midpoint", "offset"))
  if (nrow(main_timing) > 0) {
    p_main_timing <- ggplot(main_timing, aes(x = estimate, y = term, color = batch)) +
      geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, linewidth = 0.8) +
      geom_point(size = 2.5) +
      facet_grid(category ~ outcome, scales = "free_y", space = "free_y") +
      scale_color_manual(values = c("base" = "#0072B2", "dst" = "#009E73"), drop = FALSE) +
      scale_x_continuous(labels = format_time_axis) +
      labs(
        title = "Main Effects: Sleep Timing",
        subtitle = "Demographics, employment, and weekend effects",
        x = "Predicted Clock Time",
        y = NULL,
        color = "Batch"
      ) +
      theme_bw(base_size = 12)

    save_plot(p_main_timing, "main_effects_timing.png", width = 15, height = 9)
  }

  main_duration <- main_effects %>% filter(outcome == "duration")
  if (nrow(main_duration) > 0) {
    p_main_duration <- ggplot(main_duration, aes(x = estimate, y = term, color = batch)) +
      geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0.2, linewidth = 0.8) +
      geom_point(size = 2.5) +
      facet_wrap(~ category, scales = "free_y", ncol = 1) +
      scale_color_manual(values = c("base" = "#D55E00", "dst" = "#CC79A7"), drop = FALSE) +
      scale_x_continuous(labels = format_duration_axis) +
      labs(
        title = "Main Effects: Sleep Duration",
        x = "Predicted Sleep Duration",
        y = NULL,
        color = "Batch"
      ) +
      theme_bw(base_size = 12)

    save_plot(p_main_duration, "main_effects_duration.png", width = 12, height = 10)
  }

}

# %% [markdown]
# ## Figure 2B: Continuous Main Effects (Deviation, Duration Covariate)

# %%
deviation_main <- predictions_all %>% filter(analysis == "deviation_main")
if (nrow(deviation_main) > 0 && require_cols(deviation_main, c("deviation", "outcome", "batch", "estimate", "conf.low", "conf.high"), "deviation_main")) {
  deviation_main <- deviation_main %>%
    mutate(
      outcome = factor(outcome, levels = c("onset", "midpoint", "offset", "duration")),
      batch = factor(batch, levels = c("base", "dst"))
    )

  write_table(deviation_main, "deviation_main.csv")

  p_dev <- ggplot(deviation_main, aes(x = deviation, y = estimate, color = batch, fill = batch)) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0) +
    geom_line(linewidth = 1.0) +
    facet_wrap(~ outcome, scales = "free_y", ncol = 2) +
    labs(
      title = "Deviation Main Effect",
      subtitle = "Marginal predictions across observed deviation range",
      x = "Deviation",
      y = "Predicted value",
      color = "Batch",
      fill = "Batch"
    ) +
    theme_classic(base_size = 13)

  save_plot(p_dev, "deviation_main_effect.png", width = 13, height = 8)
}

duration_cov_main <- predictions_all %>% filter(analysis == "duration_covariate_main")
if (nrow(duration_cov_main) > 0 && require_cols(duration_cov_main, c("duration_hours", "outcome", "batch", "estimate", "conf.low", "conf.high"), "duration_covariate_main")) {
  duration_cov_main <- duration_cov_main %>%
    mutate(
      outcome = factor(outcome, levels = c("onset", "midpoint", "offset")),
      batch = factor(batch, levels = c("base", "dst"))
    )

  write_table(duration_cov_main, "duration_covariate_main.csv")

  p_dur_cov <- ggplot(duration_cov_main, aes(x = duration_hours, y = estimate, color = batch, fill = batch)) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0) +
    geom_line(linewidth = 1.0) +
    facet_wrap(~ outcome, scales = "free_y", ncol = 3) +
    scale_y_continuous(labels = format_time_axis) +
    labs(
      title = "Duration Covariate Main Effect",
      subtitle = "Timing outcomes as a function of sleep duration covariate",
      x = "Duration covariate (hours)",
      y = "Predicted Clock Time",
      color = "Batch",
      fill = "Batch"
    ) +
    theme_classic(base_size = 13)

  save_plot(p_dur_cov, "duration_covariate_main_effect.png", width = 13, height = 6)
}

# %% [markdown]
# ## Figure 3: Seasonality and DST

# %%
month_main <- primary_predictions %>% filter(analysis == "month_main")
if (nrow(month_main) > 0 && require_cols(month_main, c("month", "outcome", "batch", "estimate", "conf.low", "conf.high"), "month_main")) {
  month_main <- month_main %>%
    mutate(
      month = factor(sprintf("%02d", as.integer(as.character(month))), levels = sprintf("%02d", 1:12)),
      outcome = factor(outcome, levels = c("onset", "midpoint", "offset", "duration")),
      batch = factor(batch, levels = c("base", "dst"))
    )

  p_month <- ggplot(month_main, aes(x = month, y = estimate, color = batch, group = batch, fill = batch)) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0) +
    geom_line(linewidth = 1.0) +
    geom_point(size = 1.7) +
    facet_wrap(~ outcome, scales = "free_y", ncol = 2) +
    scale_color_manual(values = c("base" = "#0072B2", "dst" = "#009E73"), drop = FALSE) +
    scale_fill_manual(values = c("base" = "#0072B2", "dst" = "#009E73"), drop = FALSE) +
    labs(
      title = "Seasonality Across Outcomes",
      subtitle = "Month-specific marginal predictions",
      x = "Month",
      y = "Predicted value",
      color = "Batch",
      fill = "Batch"
    ) +
    theme_classic(base_size = 13)

  save_plot(p_month, "seasonality_month_main.png", width = 13, height = 9)

}

month_weekend <- primary_predictions %>% filter(analysis == "month_x_weekend")
if (nrow(month_weekend) > 0 && require_cols(month_weekend, c("month", "is_weekend_factor", "outcome", "batch", "estimate"), "month_x_weekend")) {
  month_weekend <- month_weekend %>%
    mutate(
      month = factor(sprintf("%02d", as.integer(as.character(month))), levels = sprintf("%02d", 1:12)),
      day_type = factor(normalize_weekend_level(is_weekend_factor), levels = c("Weekday", "Weekend")),
      outcome = factor(outcome, levels = c("onset", "midpoint", "offset", "duration")),
      batch = factor(batch, levels = c("base", "dst"))
    )

  write_table(month_weekend, "month_x_weekend.csv")

  p_month_weekend <- ggplot(month_weekend, aes(x = month, y = estimate, color = day_type, group = day_type)) +
    geom_line(linewidth = 0.9) +
    geom_point(size = 1.5) +
    facet_grid(batch ~ outcome, scales = "free_y") +
    scale_color_manual(values = c("Weekday" = "#0072B2", "Weekend" = "#D55E00"), drop = FALSE) +
    labs(
      title = "Seasonality by Weekend Status",
      x = "Month",
      y = "Predicted value",
      color = NULL
    ) +
    theme_classic(base_size = 12)

  save_plot(p_month_weekend, "seasonality_month_x_weekend.png", width = 15, height = 8)

}

month_dst <- primary_predictions %>% filter(analysis == "month_x_dst", outcome %in% c("onset", "offset"))
if (nrow(month_dst) > 0 && require_cols(month_dst, c("month", "dst_observes", "outcome", "batch", "estimate", "conf.low", "conf.high"), "month_x_dst")) {
  month_dst <- month_dst %>%
    mutate(
      month = factor(sprintf("%02d", as.integer(as.character(month))), levels = sprintf("%02d", 1:12)),
      outcome = factor(outcome, levels = c("onset", "offset")),
      batch = factor(batch, levels = c("base", "dst"))
    )

  write_table(month_dst, "onset_offset_month_x_dst.csv")

  p_month_dst <- ggplot(month_dst, aes(x = month, y = estimate, color = dst_observes, group = dst_observes, fill = dst_observes)) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0) +
    geom_line(linewidth = 1.0) +
    geom_point(size = 1.7) +
    facet_grid(batch ~ outcome, scales = "free_y") +
    scale_y_continuous(labels = format_time_axis) +
    labs(
      title = "Onset and Offset vs Month by DST Group",
      x = "Month",
      y = "Predicted Clock Time",
      color = "DST Group",
      fill = "DST Group"
    ) +
    theme_classic(base_size = 13)

  save_plot(p_month_dst, "onset_offset_month_x_dst.png", width = 13, height = 8)

}

# %% [markdown]
# ## Figure 4: Age Trends (Global + Controlled)

# %%
age_main <- primary_predictions %>% filter(analysis == "age_main")
if (nrow(age_main) > 0 && require_cols(age_main, c("age_at_sleep", "outcome", "batch", "estimate", "conf.low", "conf.high"), "age_main")) {
  age_main <- age_main %>%
    mutate(
      outcome = factor(outcome, levels = c("onset", "midpoint", "offset", "duration")),
      batch = factor(batch, levels = c("base", "dst"))
    )

  write_table(age_main, "age_main.csv")

  age_timing <- age_main %>% filter(outcome %in% c("onset", "midpoint", "offset"))
  if (nrow(age_timing) > 0) {
    p_age_timing <- ggplot(age_timing, aes(x = age_at_sleep, y = estimate, color = batch, fill = batch)) +
      geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.14, linewidth = 0) +
      geom_line(linewidth = 1.0) +
      facet_wrap(~ outcome, scales = "free_y", ncol = 3) +
      scale_y_continuous(labels = format_time_axis) +
      labs(
        title = "Age-related Shifts in Sleep Timing",
        x = "Age (years)",
        y = "Predicted Clock Time",
        color = "Batch",
        fill = "Batch"
      ) +
      theme_classic(base_size = 13)

    save_plot(p_age_timing, "age_trends_timing.png", width = 15, height = 6)

  }

  age_duration <- age_main %>% filter(outcome == "duration")
  if (nrow(age_duration) > 0) {
    p_age_duration <- ggplot(age_duration, aes(x = age_at_sleep, y = estimate, color = batch, fill = batch)) +
      geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.14, linewidth = 0) +
      geom_line(linewidth = 1.0) +
      facet_wrap(~ batch, scales = "free_y", ncol = 1) +
      scale_y_continuous(labels = format_duration_axis) +
      labs(
        title = "Age-related Shifts in Sleep Duration",
        x = "Age (years)",
        y = "Predicted Sleep Duration",
        color = "Batch",
        fill = "Batch"
      ) +
      theme_classic(base_size = 13)

    save_plot(p_age_duration, "age_trends_duration.png", width = 11, height = 8)

  }
}

age_x_employment_duration <- primary_predictions %>% filter(analysis == "age_x_employment", outcome == "duration")
if (nrow(age_x_employment_duration) > 0 && require_cols(age_x_employment_duration, c("age_at_sleep", "employment_status", "batch", "estimate", "conf.low", "conf.high"), "age_x_employment_duration")) {
  age_x_employment_duration <- age_x_employment_duration %>%
    mutate(
      employment_clean = employment_levels_ordered(employment_status),
      batch = factor(batch, levels = c("base", "dst"))
    )

  write_table(age_x_employment_duration, "duration_age_x_employment.csv")

  p_age_emp <- ggplot(age_x_employment_duration, aes(x = age_at_sleep, y = estimate, color = employment_clean, fill = employment_clean)) +
    geom_line(linewidth = 0.9) +
    facet_wrap(~ batch, ncol = 1, scales = "free_y") +
    scale_y_continuous(labels = format_duration_axis) +
    labs(
      title = "Duration vs Age Controlling for Employment",
      subtitle = "Precomputed age x employment grid",
      x = "Age (years)",
      y = "Predicted Sleep Duration",
      color = "Employment",
      fill = "Employment"
    ) +
    theme_classic(base_size = 12)

  save_plot(p_age_emp, "duration_age_x_employment.png", width = 12, height = 9)

}

age_x_dst_onoff <- primary_predictions %>% filter(analysis == "age_x_dst", outcome %in% c("onset", "offset"))
if (nrow(age_x_dst_onoff) > 0 && require_cols(age_x_dst_onoff, c("age_at_sleep", "dst_observes", "outcome", "batch", "estimate", "conf.low", "conf.high"), "age_x_dst")) {
  age_x_dst_onoff <- age_x_dst_onoff %>%
    mutate(
      outcome = factor(outcome, levels = c("onset", "offset")),
      batch = factor(batch, levels = c("base", "dst"))
    )

  write_table(age_x_dst_onoff, "onset_offset_age_x_dst.csv")

  p_age_dst <- ggplot(age_x_dst_onoff, aes(x = age_at_sleep, y = estimate, color = dst_observes, fill = dst_observes)) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0) +
    geom_line(linewidth = 1.0) +
    facet_grid(batch ~ outcome, scales = "free_y") +
    scale_y_continuous(labels = format_time_axis) +
    labs(
      title = "Onset/Offset vs Age by DST Group",
      x = "Age (years)",
      y = "Predicted Clock Time",
      color = "DST Group",
      fill = "DST Group"
    ) +
    theme_classic(base_size = 13)

  save_plot(p_age_dst, "onset_offset_age_x_dst.png", width = 13, height = 8)

}

# %% [markdown]
# ## Figure 4B: EMMeans-based Marginalized Effects (DST model)

# %%
if (nrow(emmeans_marginalized) > 0 && require_cols(emmeans_marginalized, c("analysis", "outcome", "estimate", "conf.low", "conf.high"), "emmeans_marginalized")) {
  emm <- emmeans_marginalized %>%
    filter(outcome_variant == "primary", batch == "dst") %>%
    mutate(
      outcome = factor(outcome, levels = c("onset", "midpoint", "offset", "duration"))
    )

  write_table(emm, "emmeans_marginalized_primary.csv")

  emm_emp_wk <- emm %>% filter(analysis == "emm_employment_x_weekend")
  if (nrow(emm_emp_wk) > 0 && require_cols(emm_emp_wk, c("employment_status", "is_weekend_factor", "outcome"), "emm_employment_x_weekend")) {
    emm_emp_wk <- emm_emp_wk %>%
      mutate(
        employment_clean = employment_levels_ordered(employment_status),
        employment_clean = str_wrap(employment_clean, width = 25),
        day_type = factor(normalize_weekend_level(is_weekend_factor), levels = c("Weekday", "Weekend"))
      )

    write_table(emm_emp_wk, "employment_x_weekend_marginalized.csv")

    emm_timing <- emm_emp_wk %>% filter(outcome %in% c("onset", "midpoint", "offset"))
    if (nrow(emm_timing) > 0) {
      p_emm_timing <- ggplot(emm_timing, aes(x = estimate, y = employment_clean, color = day_type)) +
        geom_line(aes(group = employment_clean), color = "gray85", linewidth = 0.6) +
        geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0, linewidth = 0.9, alpha = 0.9) +
        geom_point(size = 2.4) +
        facet_wrap(~ outcome, scales = "free_x", ncol = 3) +
        scale_color_manual(values = c("Weekday" = "#0072B2", "Weekend" = "#D55E00"), drop = FALSE) +
        scale_x_continuous(labels = format_time_axis) +
        labs(
          title = "Sleep Timing by Employment and Weekend Status (EMMeans marginalized)",
          subtitle = "DST interaction model, proportional weighting over DST groups",
          x = "Predicted Clock Time",
          y = NULL,
          color = NULL
        ) +
        theme_classic(base_size = 13)

      save_plot(p_emm_timing, "employment_weekend_timing_marginalized.png", width = 15, height = 7)
    }

    emm_duration <- emm_emp_wk %>% filter(outcome == "duration")
    if (nrow(emm_duration) > 0) {
      p_emm_duration <- ggplot(emm_duration, aes(x = estimate, y = employment_clean, color = day_type)) +
        geom_line(aes(group = employment_clean), color = "gray85", linewidth = 0.6) +
        geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0, linewidth = 0.9, alpha = 0.9) +
        geom_point(size = 2.4) +
        scale_color_manual(values = c("Weekday" = "#0072B2", "Weekend" = "#D55E00"), drop = FALSE) +
        scale_x_continuous(labels = format_duration_axis) +
        labs(
          title = "Sleep Duration by Employment and Weekend Status (EMMeans marginalized)",
          x = "Predicted Sleep Duration",
          y = NULL,
          color = NULL
        ) +
        theme_classic(base_size = 13)

      save_plot(p_emm_duration, "employment_weekend_duration_marginalized.png", width = 11, height = 7)
    }
  }

  emm_age_main <- emm %>% filter(analysis == "emm_age_main")
  if (nrow(emm_age_main) > 0 && require_cols(emm_age_main, c("age_at_sleep", "outcome"), "emm_age_main")) {
    write_table(emm_age_main, "age_main_marginalized.csv")

    emm_age_timing <- emm_age_main %>% filter(outcome %in% c("onset", "midpoint", "offset"))
    if (nrow(emm_age_timing) > 0) {
      p_emm_age_timing <- ggplot(emm_age_timing, aes(x = age_at_sleep, y = estimate)) +
        geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.14, linewidth = 0, fill = "#8DA0CB") +
        geom_line(linewidth = 1.0, color = "#1F78B4") +
        facet_wrap(~ outcome, scales = "free_y", ncol = 3) +
        scale_y_continuous(labels = format_time_axis) +
        labs(
          title = "Age-related Shifts in Sleep Timing (EMMeans marginalized)",
          x = "Age (years)",
          y = "Predicted Clock Time"
        ) +
        theme_classic(base_size = 13)

      save_plot(p_emm_age_timing, "age_trends_timing_marginalized.png", width = 15, height = 6)
    }

    emm_age_duration <- emm_age_main %>% filter(outcome == "duration")
    if (nrow(emm_age_duration) > 0) {
      p_emm_age_duration <- ggplot(emm_age_duration, aes(x = age_at_sleep, y = estimate)) +
        geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.14, linewidth = 0, fill = "#FC8D62") +
        geom_line(linewidth = 1.0, color = "#D95F02") +
        scale_y_continuous(labels = format_duration_axis) +
        labs(
          title = "Age-related Shifts in Sleep Duration (EMMeans marginalized)",
          x = "Age (years)",
          y = "Predicted Sleep Duration"
        ) +
        theme_classic(base_size = 13)

      save_plot(p_emm_age_duration, "age_trends_duration_marginalized.png", width = 11, height = 6)
    }
  }

  emm_age_emp <- emm %>% filter(analysis == "emm_age_x_employment", outcome == "duration")
  if (nrow(emm_age_emp) > 0 && require_cols(emm_age_emp, c("age_at_sleep", "employment_status"), "emm_age_x_employment")) {
    emm_age_emp <- emm_age_emp %>% mutate(employment_clean = employment_levels_ordered(employment_status))
    write_table(emm_age_emp, "duration_age_x_employment_marginalized.csv")

    p_emm_age_emp <- ggplot(emm_age_emp, aes(x = age_at_sleep, y = estimate, color = employment_clean, fill = employment_clean)) +
      geom_line(linewidth = 0.9) +
      scale_y_continuous(labels = format_duration_axis) +
      labs(
        title = "Duration vs Age Controlling for Employment (EMMeans marginalized)",
        subtitle = "DST interaction model, proportional weighting over DST groups",
        x = "Age (years)",
        y = "Predicted Sleep Duration",
        color = "Employment",
        fill = "Employment"
      ) +
      theme_classic(base_size = 12)

    save_plot(p_emm_age_emp, "duration_age_x_employment_marginalized.png", width = 12, height = 7)
  }

  emm_month <- emm %>% filter(analysis == "emm_month_main")
  if (nrow(emm_month) > 0 && require_cols(emm_month, c("month", "outcome"), "emm_month_main")) {
    emm_month <- emm_month %>% mutate(month = factor(sprintf("%02d", as.integer(as.character(month))), levels = sprintf("%02d", 1:12)))
    write_table(emm_month, "month_main_marginalized.csv")

    p_emm_month <- ggplot(emm_month, aes(x = month, y = estimate, group = 1)) +
      geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0, fill = "#8DA0CB") +
      geom_line(linewidth = 1.0, color = "#1F78B4") +
      geom_point(size = 1.7, color = "#1F78B4") +
      facet_wrap(~ outcome, scales = "free_y", ncol = 2) +
      labs(
        title = "Seasonality Across Outcomes (EMMeans marginalized)",
        x = "Month",
        y = "Predicted value"
      ) +
      theme_classic(base_size = 13)

    save_plot(p_emm_month, "seasonality_month_main_marginalized.png", width = 13, height = 9)

    emm_month_onoff <- emm_month %>% filter(outcome %in% c("onset", "offset"))
    if (nrow(emm_month_onoff) > 0) {
      p_emm_month_onoff <- ggplot(emm_month_onoff, aes(x = month, y = estimate, group = outcome, color = outcome, fill = outcome)) +
        geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0) +
        geom_line(linewidth = 1.0) +
        geom_point(size = 1.7) +
        facet_wrap(~ outcome, scales = "free_y", ncol = 2) +
        scale_y_continuous(labels = format_time_axis) +
        labs(
          title = "Onset/Offset vs Month (EMMeans marginalized)",
          x = "Month",
          y = "Predicted Clock Time",
          color = "Outcome",
          fill = "Outcome"
        ) +
        theme_classic(base_size = 13)

      save_plot(p_emm_month_onoff, "onset_offset_month_marginalized.png", width = 13, height = 8)
    }
  }

  emm_month_wk <- emm %>% filter(analysis == "emm_month_x_weekend")
  if (nrow(emm_month_wk) > 0 && require_cols(emm_month_wk, c("month", "is_weekend_factor", "outcome"), "emm_month_x_weekend")) {
    emm_month_wk <- emm_month_wk %>%
      mutate(
        month = factor(sprintf("%02d", as.integer(as.character(month))), levels = sprintf("%02d", 1:12)),
        day_type = factor(normalize_weekend_level(is_weekend_factor), levels = c("Weekday", "Weekend"))
      )
    write_table(emm_month_wk, "month_x_weekend_marginalized.csv")

    p_emm_month_wk <- ggplot(emm_month_wk, aes(x = month, y = estimate, color = day_type, group = day_type)) +
      geom_line(linewidth = 0.9) +
      geom_point(size = 1.5) +
      facet_wrap(~ outcome, scales = "free_y", ncol = 2) +
      scale_color_manual(values = c("Weekday" = "#0072B2", "Weekend" = "#D55E00"), drop = FALSE) +
      labs(
        title = "Seasonality by Weekend Status (EMMeans marginalized)",
        x = "Month",
        y = "Predicted value",
        color = NULL
      ) +
      theme_classic(base_size = 12)

    save_plot(p_emm_month_wk, "seasonality_month_x_weekend_marginalized.png", width = 15, height = 8)
  }
}

# %% [markdown]
# ## Figure 5: Reconstruction Quality (Derived vs Direct)

# %%
if (nrow(derived_age_compare) > 0) {
  if (require_cols(derived_age_compare, c("age_at_sleep", "batch", "derived_duration_hours"), "derived_age_compare")) {
    duration_series_cols <- intersect(
      c("derived_duration_hours", "duration_direct_estimate", "duration_adjusted_estimate"),
      names(derived_age_compare)
    )
    if (length(duration_series_cols) >= 2) {
      derived_age_duration <- derived_age_compare %>%
        select(any_of(c("batch", "age_at_sleep", "is_weekend_factor", "dst_observes", duration_series_cols))) %>%
        pivot_longer(
          cols = all_of(duration_series_cols),
          names_to = "series",
          values_to = "estimate"
        ) %>%
        filter(!is.na(estimate)) %>%
        mutate(
          series = recode(
            series,
            derived_duration_hours = "Derived from Onset+Offset",
            duration_direct_estimate = "Direct Duration Model",
            duration_adjusted_estimate = "Duration Model Adjusted for Midpoint"
          ),
          facet_key = "All"
        )

      if ("is_weekend_factor" %in% names(derived_age_duration)) {
        wk_clean <- clean_level_label(normalize_weekend_level(derived_age_duration$is_weekend_factor), unknown = "AllDays")
        derived_age_duration$facet_key <- paste0(derived_age_duration$facet_key, " | ", wk_clean)
      }
      if ("dst_observes" %in% names(derived_age_duration)) {
        dst_clean <- clean_level_label(derived_age_duration$dst_observes, unknown = "AllDST")
        derived_age_duration$facet_key <- paste0(derived_age_duration$facet_key, " | ", dst_clean)
      }

      derived_age_duration <- derived_age_duration %>%
        mutate(
          batch = factor(batch, levels = c("base", "dst")),
          facet_key = factor(facet_key)
        ) %>%
        arrange(batch, facet_key, series, age_at_sleep)

      write_table(derived_age_duration, "derived_vs_direct_duration_age_plot_data.csv")

      duration_ylim <- robust_limits(
        derived_age_duration$estimate,
        lower_q = 0.01,
        upper_q = 0.99,
        hard_min = 2,
        hard_max = 14
      )

      p_derived_age_duration <- ggplot(
        derived_age_duration,
        aes(x = age_at_sleep, y = estimate, color = series, group = interaction(series, batch, facet_key, drop = TRUE))
      ) +
        geom_line(linewidth = 1.0) +
        facet_grid(batch ~ facet_key, scales = "free_y") +
        coord_cartesian(ylim = duration_ylim) +
        scale_y_continuous(labels = format_duration_axis) +
        labs(
          title = "Duration: Derived vs Direct Models",
          subtitle = "Age-grid comparison including midpoint-adjusted sensitivity model (y-axis robustly clipped)",
          x = "Age (years)",
          y = "Predicted Sleep Duration",
          color = NULL
        ) +
        theme_classic(base_size = 12)

      save_plot(p_derived_age_duration, "derived_vs_direct_duration_age.png", width = 16, height = 8)
    }
  }

  if (require_cols(derived_age_compare, c("age_at_sleep", "batch", "derived_midpoint_linear"), "derived_age_compare")) {
    midpoint_series_cols <- intersect(
      c("derived_midpoint_linear", "midpoint_direct_estimate", "midpoint_adjusted_estimate"),
      names(derived_age_compare)
    )
    if (length(midpoint_series_cols) >= 2) {
      derived_age_midpoint <- derived_age_compare %>%
        select(any_of(c("batch", "age_at_sleep", "is_weekend_factor", "dst_observes", midpoint_series_cols))) %>%
        pivot_longer(
          cols = all_of(midpoint_series_cols),
          names_to = "series",
          values_to = "estimate"
        ) %>%
        filter(!is.na(estimate)) %>%
        mutate(
          series = recode(
            series,
            derived_midpoint_linear = "Derived from Onset+Offset",
            midpoint_direct_estimate = "Direct Midpoint Model",
            midpoint_adjusted_estimate = "Midpoint Model Adjusted for Duration"
          ),
          facet_key = "All"
        )

      if ("is_weekend_factor" %in% names(derived_age_midpoint)) {
        wk_clean <- clean_level_label(normalize_weekend_level(derived_age_midpoint$is_weekend_factor), unknown = "AllDays")
        derived_age_midpoint$facet_key <- paste0(derived_age_midpoint$facet_key, " | ", wk_clean)
      }
      if ("dst_observes" %in% names(derived_age_midpoint)) {
        dst_clean <- clean_level_label(derived_age_midpoint$dst_observes, unknown = "AllDST")
        derived_age_midpoint$facet_key <- paste0(derived_age_midpoint$facet_key, " | ", dst_clean)
      }

      derived_age_midpoint <- derived_age_midpoint %>%
        mutate(
          batch = factor(batch, levels = c("base", "dst")),
          facet_key = factor(facet_key)
        ) %>%
        arrange(batch, facet_key, series, age_at_sleep)

      write_table(derived_age_midpoint, "derived_vs_direct_midpoint_age_plot_data.csv")

      midpoint_ylim <- robust_limits(
        derived_age_midpoint$estimate,
        lower_q = 0.01,
        upper_q = 0.99,
        hard_min = 12,
        hard_max = 36
      )

      p_derived_age_midpoint <- ggplot(
        derived_age_midpoint,
        aes(x = age_at_sleep, y = estimate, color = series, group = interaction(series, batch, facet_key, drop = TRUE))
      ) +
        geom_line(linewidth = 1.0) +
        facet_grid(batch ~ facet_key, scales = "free_y") +
        coord_cartesian(ylim = midpoint_ylim) +
        scale_y_continuous(labels = format_time_axis) +
        labs(
          title = "Midpoint: Derived vs Direct Models",
          subtitle = "Age-grid comparison including duration-adjusted sensitivity model (y-axis robustly clipped)",
          x = "Age (years)",
          y = "Predicted Clock Time",
          color = NULL
        ) +
        theme_classic(base_size = 12)

      save_plot(p_derived_age_midpoint, "derived_vs_direct_midpoint_age.png", width = 16, height = 8)
    }
  }
}

if (nrow(derived_main_compare) > 0 && require_cols(derived_main_compare, c("batch", "derived_duration_hours", "derived_midpoint_linear"), "derived_main_compare")) {
  derived_main_compare <- derived_main_compare %>%
    mutate(strata = "Overall")

  if ("employment_status" %in% names(derived_main_compare)) {
    derived_main_compare$strata <- paste0(derived_main_compare$strata, " | ", clean_employment(derived_main_compare$employment_status))
  }
  if ("is_weekend_factor" %in% names(derived_main_compare)) {
    wk_clean <- clean_level_label(normalize_weekend_level(derived_main_compare$is_weekend_factor), unknown = "AllDays")
    derived_main_compare$strata <- paste0(derived_main_compare$strata, " | ", wk_clean)
  }
  if ("dst_observes" %in% names(derived_main_compare)) {
    dst_clean <- clean_level_label(derived_main_compare$dst_observes, unknown = "AllDST")
    derived_main_compare$strata <- paste0(derived_main_compare$strata, " | ", dst_clean)
  }

  duration_diff_cols <- intersect(
    c("duration_direct_minus_derived", "duration_adjusted_minus_derived"),
    names(derived_main_compare)
  )
  if (length(duration_diff_cols) > 0) {
    duration_main_diff <- derived_main_compare %>%
      select(any_of(c("batch", "strata", duration_diff_cols))) %>%
      pivot_longer(cols = all_of(duration_diff_cols), names_to = "series", values_to = "difference") %>%
      filter(!is.na(difference)) %>%
      mutate(
        series = recode(
          series,
          duration_direct_minus_derived = "Direct Duration Model",
          duration_adjusted_minus_derived = "Duration Model Adjusted for Midpoint"
        )
      )

    write_table(duration_main_diff, "derived_vs_direct_duration_main_grid_plot_data.csv")

    duration_diff_xlim <- robust_limits(
      duration_main_diff$difference,
      lower_q = 0.01,
      upper_q = 0.99,
      hard_min = -4,
      hard_max = 4
    )

    p_main_duration_diff <- ggplot(duration_main_diff, aes(x = difference, y = reorder(strata, difference), color = series)) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +
      geom_point(size = 2.3) +
      facet_wrap(~ batch, ncol = 1, scales = "free_y") +
      coord_cartesian(xlim = duration_diff_xlim) +
      labs(
        title = "Duration Models minus Derived Duration (Main Grid)",
        x = "Hours (Model - Derived; robustly clipped)",
        y = NULL,
        color = NULL
      ) +
      theme_classic(base_size = 12)

    save_plot(p_main_duration_diff, "derived_vs_direct_duration_main_grid.png", width = 12, height = 8)
  }

  midpoint_diff_cols <- intersect(
    c("midpoint_direct_minus_derived", "midpoint_adjusted_minus_derived"),
    names(derived_main_compare)
  )
  if (length(midpoint_diff_cols) > 0) {
    midpoint_main_diff <- derived_main_compare %>%
      select(any_of(c("batch", "strata", midpoint_diff_cols))) %>%
      pivot_longer(cols = all_of(midpoint_diff_cols), names_to = "series", values_to = "difference") %>%
      filter(!is.na(difference)) %>%
      mutate(
        series = recode(
          series,
          midpoint_direct_minus_derived = "Direct Midpoint Model",
          midpoint_adjusted_minus_derived = "Midpoint Model Adjusted for Duration"
        )
      )

    write_table(midpoint_main_diff, "derived_vs_direct_midpoint_main_grid_plot_data.csv")

    midpoint_diff_xlim <- robust_limits(
      midpoint_main_diff$difference,
      lower_q = 0.01,
      upper_q = 0.99,
      hard_min = -4,
      hard_max = 4
    )

    p_main_midpoint_diff <- ggplot(midpoint_main_diff, aes(x = difference, y = reorder(strata, difference), color = series)) +
      geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +
      geom_point(size = 2.3) +
      facet_wrap(~ batch, ncol = 1, scales = "free_y") +
      coord_cartesian(xlim = midpoint_diff_xlim) +
      scale_x_continuous(labels = function(x) sprintf("%.2fh", x)) +
      labs(
        title = "Midpoint Models minus Derived Midpoint (Main Grid)",
        x = "Hours (Model - Derived; robustly clipped)",
        y = NULL,
        color = NULL
      ) +
      theme_classic(base_size = 12)

    save_plot(p_main_midpoint_diff, "derived_vs_direct_midpoint_main_grid.png", width = 12, height = 8)
  }

  write_table(derived_main_compare, "derived_vs_direct_main_grid.csv")

  if (require_cols(derived_main_compare, c("batch", "derived_midpoint_linear"), "derived_main_compare_duration_vs_midpoint")) {
    duration_y_cols <- intersect(c("duration_direct_estimate", "duration_adjusted_estimate"), names(derived_main_compare))
    if (length(duration_y_cols) > 0) {
      duration_vs_midpoint <- derived_main_compare %>%
        select(any_of(c("batch", "derived_midpoint_linear", duration_y_cols))) %>%
        pivot_longer(cols = all_of(duration_y_cols), names_to = "series", values_to = "duration_estimate") %>%
        filter(!is.na(derived_midpoint_linear), !is.na(duration_estimate)) %>%
        mutate(
          series = recode(
            series,
            duration_direct_estimate = "Direct Duration Model",
            duration_adjusted_estimate = "Duration Model Adjusted for Midpoint"
          ),
          batch = factor(batch, levels = c("base", "dst"))
        )

      write_table(duration_vs_midpoint, "duration_vs_midpoint_plot_data.csv")

      x_mid_lims <- robust_limits(duration_vs_midpoint$derived_midpoint_linear, hard_min = 12, hard_max = 36)
      y_dur_lims <- robust_limits(duration_vs_midpoint$duration_estimate, hard_min = 2, hard_max = 14)

      p_duration_vs_midpoint <- ggplot(
        duration_vs_midpoint,
        aes(x = derived_midpoint_linear, y = duration_estimate, color = series)
      ) +
        geom_point(alpha = 0.30, size = 1.2) +
        geom_smooth(method = "loess", se = FALSE, linewidth = 1.1) +
        facet_wrap(~ batch, ncol = 1) +
        coord_cartesian(xlim = x_mid_lims, ylim = y_dur_lims) +
        scale_x_continuous(labels = format_time_axis) +
        scale_y_continuous(labels = format_duration_axis) +
        labs(
          title = "Duration ~ Midpoint",
          subtitle = "Main-grid predictions from direct and midpoint-adjusted duration models (robustly clipped axes)",
          x = "Midpoint (derived from onset+offset)",
          y = "Predicted Sleep Duration",
          color = NULL
        ) +
        theme_classic(base_size = 12)

      save_plot(p_duration_vs_midpoint, "duration_vs_midpoint.png", width = 12, height = 8)
    }
  }

  if (require_cols(derived_main_compare, c("batch", "derived_duration_hours"), "derived_main_compare_midpoint_vs_duration")) {
    midpoint_y_cols <- intersect(c("midpoint_direct_estimate", "midpoint_adjusted_estimate"), names(derived_main_compare))
    if (length(midpoint_y_cols) > 0) {
      midpoint_vs_duration <- derived_main_compare %>%
        select(any_of(c("batch", "derived_duration_hours", midpoint_y_cols))) %>%
        pivot_longer(cols = all_of(midpoint_y_cols), names_to = "series", values_to = "midpoint_estimate") %>%
        filter(!is.na(derived_duration_hours), !is.na(midpoint_estimate)) %>%
        mutate(
          series = recode(
            series,
            midpoint_direct_estimate = "Direct Midpoint Model",
            midpoint_adjusted_estimate = "Midpoint Model Adjusted for Duration"
          ),
          batch = factor(batch, levels = c("base", "dst"))
        )

      write_table(midpoint_vs_duration, "midpoint_vs_duration_plot_data.csv")

      x_dur_lims <- robust_limits(midpoint_vs_duration$derived_duration_hours, hard_min = 2, hard_max = 14)
      y_mid_lims <- robust_limits(midpoint_vs_duration$midpoint_estimate, hard_min = 12, hard_max = 36)

      p_midpoint_vs_duration <- ggplot(
        midpoint_vs_duration,
        aes(x = derived_duration_hours, y = midpoint_estimate, color = series)
      ) +
        geom_point(alpha = 0.30, size = 1.2) +
        geom_smooth(method = "loess", se = FALSE, linewidth = 1.1) +
        facet_wrap(~ batch, ncol = 1) +
        coord_cartesian(xlim = x_dur_lims, ylim = y_mid_lims) +
        scale_x_continuous(labels = format_duration_axis) +
        scale_y_continuous(labels = format_time_axis) +
        labs(
          title = "Midpoint ~ Duration",
          subtitle = "Main-grid predictions from direct and duration-adjusted midpoint models (robustly clipped axes)",
          x = "Duration (derived from onset+offset)",
          y = "Predicted Midpoint",
          color = NULL
        ) +
        theme_classic(base_size = 12)

      save_plot(p_midpoint_vs_duration, "midpoint_vs_duration.png", width = 12, height = 8)
    }
  }
}

if (nrow(derived_main) > 0) write_table(derived_main, "derived_midpoint_duration_main_grid.csv")
if (nrow(derived_age) > 0) write_table(derived_age, "derived_midpoint_duration_age.csv")
if (nrow(derived_age_compare) > 0) write_table(derived_age_compare, "derived_vs_direct_age.csv")

# %% [markdown]
# ## Figure 6: Contrast Summaries

# %%
if (nrow(weekend_contrasts) > 0 && require_cols(weekend_contrasts, c("outcome", "contrast_scope", "weekend_minus_weekday"), "weekend_contrasts")) {
  if (!"outcome_variant" %in% names(weekend_contrasts)) weekend_contrasts$outcome_variant <- "primary"
  weekend_contrasts <- weekend_contrasts %>%
    mutate(outcome_variant = normalize_outcome_variant(outcome_variant)) %>%
    filter(outcome_variant == "primary") %>%
    mutate(
      outcome = factor(outcome, levels = c("onset", "midpoint", "offset", "duration")),
      term = ifelse("employment_status" %in% names(weekend_contrasts) & !is.na(employment_status), clean_employment(employment_status), "Overall")
    )

  write_table(weekend_contrasts, "weekend_contrasts.csv")

  p_weekend_ctr <- ggplot(weekend_contrasts, aes(x = weekend_minus_weekday, y = reorder(term, weekend_minus_weekday), color = batch)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +
    geom_point(size = 2.2) +
    facet_grid(contrast_scope ~ outcome, scales = "free_y", space = "free_y") +
    labs(
      title = "Weekend minus Weekday Contrasts",
      x = "Difference in predicted value",
      y = NULL,
      color = "Batch"
    ) +
    theme_bw(base_size = 11)

  save_plot(p_weekend_ctr, "weekend_contrasts.png", width = 16, height = 10)
}

if (nrow(dst_contrasts) > 0 && require_cols(dst_contrasts, c("outcome", "contrast_scope", "nodst_minus_dst"), "dst_contrasts")) {
  if (!"outcome_variant" %in% names(dst_contrasts)) dst_contrasts$outcome_variant <- "primary"
  dst_contrasts <- dst_contrasts %>%
    mutate(outcome_variant = normalize_outcome_variant(outcome_variant)) %>%
    filter(outcome_variant == "primary") %>%
    mutate(
      outcome = factor(outcome, levels = c("onset", "midpoint", "offset", "duration")),
      term = case_when(
        "month" %in% names(dst_contrasts) & !is.na(month) ~ paste("Month", month),
        "is_weekend_factor" %in% names(dst_contrasts) & !is.na(is_weekend_factor) ~ normalize_weekend_level(is_weekend_factor),
        TRUE ~ "Overall"
      )
    )

  write_table(dst_contrasts, "dst_contrasts.csv")

  p_dst_ctr <- ggplot(dst_contrasts, aes(x = nodst_minus_dst, y = reorder(term, nodst_minus_dst), color = batch)) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray60") +
    geom_point(size = 2.2) +
    facet_grid(contrast_scope ~ outcome, scales = "free_y", space = "free_y") +
    labs(
      title = "NoDST minus DST Contrasts",
      x = "Difference in predicted value",
      y = NULL,
      color = "Batch"
    ) +
    theme_bw(base_size = 11)

  save_plot(p_dst_ctr, "dst_contrasts.png", width = 16, height = 10)
}

# %% [markdown]
# ## Final Export

# %%
writeLines(analysis_log, con = file.path(OUTPUT_DIR, "analysis_log.txt"))
log_msg("Done.")
log_msg(paste("Tables:", TABLE_DIR))
log_msg(paste("Plots:", PLOT_DIR))
