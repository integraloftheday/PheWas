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
# # 05_6_DST_OnsetOffset_Results.r
# Focused post-fit plotting for DST onset/offset models:
# - Onset vs Month by DST group
# - Offset vs Month by DST group
# - Derived Duration vs Age from onset + offset models

# %%
required_packages <- c(
  "dplyr", "tidyr", "readr", "stringr",
  "marginaleffects", "ggplot2", "scales", "tibble"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install them before running 05_6_DST_OnsetOffset_Results.r."
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(readr)
  library(stringr)
  library(marginaleffects)
  library(ggplot2)
  library(scales)
  library(tibble)
})

set.seed(123)

# %% [markdown]
# ## Configuration

# %%
MODEL_DIR <- Sys.getenv("MODEL_DIR_04_5", "models_04_5")
OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR_05_6", "results_05_6")
TABLE_DIR <- file.path(OUTPUT_DIR, "tables")
PLOT_DIR <- file.path(OUTPUT_DIR, "plots")

PREFERRED_METHOD <- toupper(Sys.getenv("MODEL_METHOD_05_6", "REML"))
SUPPORTED_METHODS <- c("REML", "ML")
METHOD_PRIORITY <- unique(c(PREFERRED_METHOD, setdiff(SUPPORTED_METHODS, PREFERRED_METHOD)))

AGE_MIN <- as.numeric(Sys.getenv("AGE_MIN_05_6", "18"))
AGE_MAX <- as.numeric(Sys.getenv("AGE_MAX_05_6", "85"))
AGE_BY <- as.numeric(Sys.getenv("AGE_BY_05_6", "1"))
NOTEBOOK_INLINE <- tolower(Sys.getenv("NOTEBOOK_INLINE_05_6", "true")) %in% c("true", "1", "yes")

# Quick validation mode: thin prediction grids to speed up script checks.
QUICK_PREVIEW <- tolower(Sys.getenv("QUICK_PREVIEW_05_6", "false")) %in% c("true", "1", "yes")
PREVIEW_MONTH_COUNT <- as.integer(Sys.getenv("PREVIEW_MONTH_COUNT_05_6", "4"))
PREVIEW_AGE_POINTS <- as.integer(Sys.getenv("PREVIEW_AGE_POINTS_05_6", "12"))

# CI calculation can be expensive; disable by default in quick preview mode.
COMPUTE_CI <- tolower(Sys.getenv(
  "COMPUTE_CI_05_6",
  ifelse(QUICK_PREVIEW, "false", "true")
)) %in% c("true", "1", "yes")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)
if (!dir.exists(TABLE_DIR)) dir.create(TABLE_DIR, recursive = TRUE)
if (!dir.exists(PLOT_DIR)) dir.create(PLOT_DIR, recursive = TRUE)

analysis_log <- character()
log_msg <- function(txt) {
  message(txt)
  analysis_log <<- c(analysis_log, paste(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), txt))
}

log_msg("Configuration loaded.")
log_msg(paste("MODEL_DIR:", MODEL_DIR))
log_msg(paste("OUTPUT_DIR:", OUTPUT_DIR))
log_msg(paste("METHOD_PRIORITY:", paste(METHOD_PRIORITY, collapse = " -> ")))
log_msg(paste("Age sequence:", paste(c(AGE_MIN, AGE_MAX, AGE_BY), collapse = ", ")))
log_msg(paste("NOTEBOOK_INLINE:", NOTEBOOK_INLINE))
log_msg(paste("QUICK_PREVIEW:", QUICK_PREVIEW))
log_msg(paste("COMPUTE_CI:", COMPUTE_CI))
if (QUICK_PREVIEW) {
  log_msg(paste("PREVIEW_MONTH_COUNT:", PREVIEW_MONTH_COUNT))
  log_msg(paste("PREVIEW_AGE_POINTS:", PREVIEW_AGE_POINTS))
}

# %% [markdown]
# ## Helpers

# %%
resolve_model_file <- function(stem) {
  for (m in METHOD_PRIORITY) {
    candidate <- file.path(MODEL_DIR, paste0(stem, "_", m, ".rds"))
    if (file.exists(candidate)) {
      return(list(path = candidate, method = m, exists = TRUE))
    }
  }

  list(
    path = file.path(MODEL_DIR, paste0(stem, "_", METHOD_PRIORITY[[1]], ".rds")),
    method = NA_character_,
    exists = FALSE
  )
}

safe_read_model <- function(path) {
  tryCatch(
    readRDS(path),
    error = function(e) {
      log_msg(paste("Failed to read model:", path, "|", e$message))
      NULL
    }
  )
}

has_vars <- function(model, vars) {
  all(vars %in% names(model@frame))
}

get_model_values <- function(model, var_name) {
  if (!var_name %in% names(model@frame)) return(NULL)
  x <- model@frame[[var_name]]
  if (is.factor(x)) return(levels(x))
  sort(unique(x))
}

safe_predictions <- function(model, grid_args, keep_cols, analysis_name) {
  tryCatch({
    newdata <- do.call(datagrid, c(list(model = model), grid_args))
    pred <- predictions(
      model,
      newdata = newdata,
      re.form = NA,
      allow.new.levels = TRUE,
      vcov = if (COMPUTE_CI) TRUE else FALSE
    ) %>%
      as_tibble()

    # Maintain a stable schema when CI computation is disabled.
    if (!"conf.low" %in% names(pred)) pred$conf.low <- pred$estimate
    if (!"conf.high" %in% names(pred)) pred$conf.high <- pred$estimate

    cols <- unique(c(keep_cols, "estimate", "conf.low", "conf.high"))
    cols <- cols[cols %in% names(pred)]

    pred %>%
      select(all_of(cols)) %>%
      mutate(analysis = analysis_name, .before = 1)
  }, error = function(e) {
    log_msg(paste("Prediction failed for", analysis_name, "|", e$message))
    tibble()
  })
}

write_table <- function(df, csv_name) {
  csv_path <- file.path(TABLE_DIR, csv_name)
  if (nrow(df) == 0) {
    log_msg(paste("No rows produced for:", csv_name))
    return(invisible(NULL))
  }
  readr::write_csv(df, csv_path)
  saveRDS(df, sub("\\.csv$", ".rds", csv_path))
  log_msg(paste("Wrote:", csv_path))
}

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

clock_diff_shortest <- function(a, b) {
  ((a - b + 12) %% 24) - 12
}

print_plot_inline <- function(title, plot_obj) {
  if (!NOTEBOOK_INLINE) return(invisible(NULL))
  cat("\n----------------------------------------------------\n")
  cat("Figure:", title, "\n")
  cat("----------------------------------------------------\n")
  print(plot_obj)
  invisible(NULL)
}

run_gc <- function(label = NULL) {
  invisible(gc(verbose = FALSE))
  if (!is.null(label)) log_msg(paste("Memory cleanup:", label))
}

thin_levels <- function(vals, n_target) {
  if (is.null(vals)) return(vals)
  vals <- unique(vals)
  if (length(vals) <= n_target) return(vals)
  idx <- unique(round(seq(1, length(vals), length.out = n_target)))
  vals[idx]
}

get_values_from_model_path <- function(path, var_name) {
  model <- safe_read_model(path)
  if (is.null(model)) return(NULL)
  vals <- get_model_values(model, var_name)
  rm(model)
  run_gc(paste("loaded values for", basename(path), "|", var_name))
  vals
}

predict_from_model_path <- function(path, grid_args, keep_cols, analysis_name) {
  model <- safe_read_model(path)
  if (is.null(model)) return(tibble())
  pred <- safe_predictions(
    model = model,
    grid_args = grid_args,
    keep_cols = keep_cols,
    analysis_name = analysis_name
  )
  rm(model)
  run_gc(paste("predictions complete for", basename(path), "|", analysis_name))
  pred
}

# %% [markdown]
# ## Model Resolution

# %%
onset_dst_info <- resolve_model_file("onset_poly_interact_04_5_dst")
offset_dst_info <- resolve_model_file("offset_poly_interact_04_5_dst")

onset_base_info <- resolve_model_file("onset_poly_interact_04_5")
offset_base_info <- resolve_model_file("offset_poly_interact_04_5")

model_inventory <- tibble::tribble(
  ~model_key, ~model_path, ~model_method, ~exists,
  "onset_dst", onset_dst_info$path, onset_dst_info$method, onset_dst_info$exists,
  "offset_dst", offset_dst_info$path, offset_dst_info$method, offset_dst_info$exists,
  "onset_base", onset_base_info$path, onset_base_info$method, onset_base_info$exists,
  "offset_base", offset_base_info$path, offset_base_info$method, offset_base_info$exists
)

write_table(model_inventory, "model_inventory_05_6.csv")

use_dst <- onset_dst_info$exists && offset_dst_info$exists
if (use_dst) {
  log_msg("Using DST onset/offset models.")
  onset_info <- onset_dst_info
  offset_info <- offset_dst_info
  batch_label <- "dst"
} else {
  log_msg("DST onset/offset pair not fully available. Falling back to base onset/offset models.")
  onset_info <- onset_base_info
  offset_info <- offset_base_info
  batch_label <- "base"
}

if (!(onset_info$exists && offset_info$exists)) {
  stop("No usable onset/offset model pair found in MODEL_DIR_04_5.")
}

# %% [markdown]
# ## Plot 1: Onset + Offset vs Month by DST Group

# %%
onset_month_vals <- get_values_from_model_path(onset_info$path, "month")
offset_month_vals <- get_values_from_model_path(offset_info$path, "month")
shared_month <- intersect(onset_month_vals, offset_month_vals)

onset_dst_vals <- get_values_from_model_path(onset_info$path, "dst_observes")
offset_dst_vals <- get_values_from_model_path(offset_info$path, "dst_observes")
shared_dst <- intersect(onset_dst_vals, offset_dst_vals)

if (QUICK_PREVIEW && !is.null(shared_month) && length(shared_month) > 0) {
  shared_month <- thin_levels(shared_month, PREVIEW_MONTH_COUNT)
  log_msg(paste("Quick preview month levels:", paste(shared_month, collapse = ", ")))
}

month_dst_preds <- tibble()

if (!is.null(shared_month) && length(shared_month) > 0) {
  grid_month <- list(month = shared_month)
  key_month <- c("month")

  if (length(shared_dst) > 0) {
    grid_month$dst_observes <- shared_dst
    key_month <- c(key_month, "dst_observes")
  }

  onset_month <- predict_from_model_path(
    path = onset_info$path,
    grid_args = grid_month,
    keep_cols = key_month,
    analysis_name = "onset_month_dst"
  ) %>%
    mutate(outcome = "onset")

  offset_month <- predict_from_model_path(
    path = offset_info$path,
    grid_args = grid_month,
    keep_cols = key_month,
    analysis_name = "offset_month_dst"
  ) %>%
    mutate(outcome = "offset")

  month_dst_preds <- bind_rows(onset_month, offset_month) %>%
    mutate(
      batch = batch_label,
      month = factor(month, levels = sprintf("%02d", 1:12)),
      outcome = factor(outcome, levels = c("onset", "offset"))
    )
}

write_table(month_dst_preds, "onset_offset_month_dst_05_6.csv")

p_month <- NULL
if (nrow(month_dst_preds) > 0) {
  month_dst_preds <- month_dst_preds %>%
    mutate(
      month = suppressWarnings(as.integer(as.character(month))),
      month = factor(sprintf("%02d", month), levels = sprintf("%02d", 1:12))
    )

  has_dst_col <- "dst_observes" %in% names(month_dst_preds)

  if (has_dst_col) {
    p_month <- ggplot(
      month_dst_preds,
      aes(x = month, y = estimate, color = dst_observes, group = dst_observes, fill = dst_observes)
    ) +
      geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0) +
      geom_line(linewidth = 1.0) +
      geom_point(size = 1.8) +
      facet_wrap(~ outcome, ncol = 1, scales = "free_y") +
      scale_y_continuous(labels = format_time_axis) +
      labs(
        title = "Onset and Offset vs Month",
        subtitle = ifelse(batch_label == "dst", "DST models", "Base models (DST pair unavailable)"),
        x = "Month",
        y = "Predicted Clock Time",
        color = "DST Group",
        fill = "DST Group"
      ) +
      theme_classic(base_size = 13)
  } else {
    p_month <- ggplot(
      month_dst_preds,
      aes(x = month, y = estimate, group = 1)
    ) +
      geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0, fill = "#56B4E9") +
      geom_line(linewidth = 1.0, color = "#0072B2") +
      geom_point(size = 1.8, color = "#0072B2") +
      facet_wrap(~ outcome, ncol = 1, scales = "free_y") +
      scale_y_continuous(labels = format_time_axis) +
      labs(
        title = "Onset and Offset vs Month",
        subtitle = "DST grouping unavailable in selected model pair",
        x = "Month",
        y = "Predicted Clock Time"
      ) +
      theme_classic(base_size = 13)
  }

  ggsave(
    filename = file.path(PLOT_DIR, "onset_offset_vs_month_dst_05_6.png"),
    plot = p_month,
    width = 11,
    height = 9,
    dpi = 300
  )
  log_msg("Saved plot: onset_offset_vs_month_dst_05_6.png")
}

# %% [markdown]
# ## Plot 2: Derived Duration vs Age from Onset + Offset

# %%
age_seq <- if (QUICK_PREVIEW) {
  unique(seq(AGE_MIN, AGE_MAX, length.out = PREVIEW_AGE_POINTS))
} else {
  seq(AGE_MIN, AGE_MAX, by = AGE_BY)
}
log_msg(paste("Age grid points:", length(age_seq)))
grid_age <- list(age_at_sleep = age_seq)
key_age <- c("age_at_sleep")

if (length(shared_dst) > 0) {
  grid_age$dst_observes <- shared_dst
  key_age <- c(key_age, "dst_observes")
}

onset_age <- predict_from_model_path(
  path = onset_info$path,
  grid_args = grid_age,
  keep_cols = key_age,
  analysis_name = "onset_age_for_derived"
) %>%
  rename(onset_estimate = estimate)

offset_age <- predict_from_model_path(
  path = offset_info$path,
  grid_args = grid_age,
  keep_cols = key_age,
  analysis_name = "offset_age_for_derived"
) %>%
  rename(offset_estimate = estimate)

derived_duration_age <- tibble()
if (nrow(onset_age) > 0 && nrow(offset_age) > 0) {
  derived_duration_age <- onset_age %>%
    select(all_of(c(key_age, "onset_estimate"))) %>%
    inner_join(offset_age %>% select(all_of(c(key_age, "offset_estimate"))), by = key_age) %>%
    mutate(
      batch = batch_label,
      derived_duration_hours = (offset_estimate - onset_estimate) %% 24,
      derived_midpoint_linear = (onset_estimate + derived_duration_hours / 2) %% 24
    ) %>%
    relocate(batch, .before = 1)
}

write_table(derived_duration_age, "derived_duration_vs_age_05_6.csv")

p_age <- NULL
if (nrow(derived_duration_age) > 0) {
  has_dst_col <- "dst_observes" %in% names(derived_duration_age)

  if (has_dst_col) {
    p_age <- ggplot(
      derived_duration_age,
      aes(x = age_at_sleep, y = derived_duration_hours, color = dst_observes)
    ) +
      geom_line(linewidth = 1.1) +
      scale_y_continuous(labels = format_duration_axis) +
      scale_x_continuous(breaks = pretty_breaks(8)) +
      labs(
        title = "Derived Sleep Duration vs Age",
        subtitle = "Duration derived from onset + offset predictions",
        x = "Age (years)",
        y = "Predicted Sleep Duration",
        color = "DST Group"
      ) +
      theme_classic(base_size = 13)
  } else {
    p_age <- ggplot(
      derived_duration_age,
      aes(x = age_at_sleep, y = derived_duration_hours)
    ) +
      geom_line(linewidth = 1.1, color = "#0072B2") +
      scale_y_continuous(labels = format_duration_axis) +
      scale_x_continuous(breaks = pretty_breaks(8)) +
      labs(
        title = "Derived Sleep Duration vs Age",
        subtitle = "Duration derived from onset + offset predictions (no DST grouping available)",
        x = "Age (years)",
        y = "Predicted Sleep Duration"
      ) +
      theme_classic(base_size = 13)
  }

  ggsave(
    filename = file.path(PLOT_DIR, "derived_duration_vs_age_05_6.png"),
    plot = p_age,
    width = 11,
    height = 6.5,
    dpi = 300
  )
  log_msg("Saved plot: derived_duration_vs_age_05_6.png")
}

# %% [markdown]
# ## Notebook Inline Output

# %%
if (NOTEBOOK_INLINE) {
  if (!is.null(p_month)) print_plot_inline("onset_offset_vs_month_dst_05_6", p_month)
  if (!is.null(p_age)) print_plot_inline("derived_duration_vs_age_05_6", p_age)
}

# %% [markdown]
# ## Final Export

# %%
writeLines(analysis_log, con = file.path(OUTPUT_DIR, "run_log_05_6.txt"))
log_msg("Done.")
log_msg(paste("Tables:", TABLE_DIR))
log_msg(paste("Plots:", PLOT_DIR))
