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
# # 05_5_LMM_Results.r
# Comprehensive post-fit analysis for 04_5 mixed models:
# - robust model discovery (skip if .rds missing)
# - marginal means focus across key covariates
# - DST-focused controlled summaries
# - derived midpoint/duration from onset + offset to avoid coupled-model confounding
# - duration vs age diagnostics (direct vs derived)

# %%

required_packages <- c(
  "dplyr", "tidyr", "purrr", "readr", "stringr",
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
      ". Install them before running 05_5_LMM_Results.r."
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
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
# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
MODEL_DIR <- Sys.getenv("MODEL_DIR_04_5", "models_04_5")
OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR_05_5", "results_05_5")
TABLE_DIR <- file.path(OUTPUT_DIR, "tables")
PLOT_DIR <- file.path(OUTPUT_DIR, "plots")

PREFERRED_METHOD <- toupper(Sys.getenv("MODEL_METHOD_05_5", "REML"))
SUPPORTED_METHODS <- c("REML", "ML")
METHOD_PRIORITY <- unique(c(PREFERRED_METHOD, setdiff(SUPPORTED_METHODS, PREFERRED_METHOD)))

AGE_MIN <- as.numeric(Sys.getenv("AGE_MIN_05_5", "18"))
AGE_MAX <- as.numeric(Sys.getenv("AGE_MAX_05_5", "85"))
AGE_BY <- as.numeric(Sys.getenv("AGE_BY_05_5", "1"))
NOTEBOOK_INLINE <- tolower(Sys.getenv("NOTEBOOK_INLINE_05_5", "true")) %in% c("true", "1", "yes")
INLINE_TABLE_MAX_ROWS <- as.numeric(Sys.getenv("INLINE_TABLE_MAX_ROWS_05_5", "0"))
if (is.na(INLINE_TABLE_MAX_ROWS)) INLINE_TABLE_MAX_ROWS <- 0

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)
if (!dir.exists(TABLE_DIR)) dir.create(TABLE_DIR, recursive = TRUE)
if (!dir.exists(PLOT_DIR)) dir.create(PLOT_DIR, recursive = TRUE)

outcome_specs <- tibble::tribble(
  ~outcome, ~stem, ~outcome_type,
  "onset", "onset_poly_interact_04_5", "ClockTime",
  "offset", "offset_poly_interact_04_5", "ClockTime",
  "midpoint", "midpoint_poly_interact_04_5", "ClockTime",
  "duration", "duration_poly_interact_04_5", "Duration"
)

batch_specs <- tibble::tribble(
  ~batch, ~suffix,
  "base", "",
  "dst", "_dst"
)

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

# %% [markdown]
# ## Helpers

# %%
# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
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

resolve_model_file <- function(stem_with_suffix) {
  for (m in METHOD_PRIORITY) {
    candidate <- file.path(MODEL_DIR, paste0(stem_with_suffix, "_", m, ".rds"))
    if (file.exists(candidate)) {
      return(list(path = candidate, method = m, exists = TRUE))
    }
  }
  list(
    path = file.path(MODEL_DIR, paste0(stem_with_suffix, "_", METHOD_PRIORITY[[1]], ".rds")),
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

get_model_values <- function(model, var_name) {
  meta <- NULL
  if (inherits(model, "phewas_marginal_model")) {
    meta <- model$grid_meta
  } else {
    meta <- attr(model, "phewas_grid_meta", exact = TRUE)
  }
  if (!is.null(meta$vars) && var_name %in% names(meta$vars)) {
    var_info <- meta$vars[[var_name]]
    if (!is.null(var_info$values) && length(var_info$values) > 0) {
      return(var_info$values)
    }
    if (!is.null(var_info$min) && !is.null(var_info$max)) {
      return(sort(unique(c(var_info$min, var_info$max))))
    }
  }

  frm <- tryCatch(model@frame, error = function(e) NULL)
  if (is.null(frm) || !var_name %in% names(frm)) return(NULL)
  x <- frm[[var_name]]
  if (is.factor(x)) return(levels(x))
  if (is.character(x)) return(sort(unique(x)))
  if (is.logical(x)) return(sort(unique(x)))
  sort(unique(x))
}

ordered_intersection <- function(a, b) {
  if (is.null(a) || is.null(b)) return(NULL)
  a[a %in% b]
}

has_vars <- function(model, vars) {
  frm_names <- character()
  if (!inherits(model, "phewas_marginal_model")) {
    frm_names <- tryCatch(names(model@frame), error = function(e) character())
  }
  meta <- NULL
  if (inherits(model, "phewas_marginal_model")) {
    meta <- model$grid_meta
  } else {
    meta <- attr(model, "phewas_grid_meta", exact = TRUE)
  }
  meta_names <- if (!is.null(meta$vars) && !is.null(names(meta$vars))) names(meta$vars) else character()
  all(vars %in% unique(c(frm_names, meta_names)))
}

safe_predictions <- function(model, grid_args, keep_cols, analysis_name) {
  tryCatch({
    if (inherits(model, "phewas_marginal_model")) {
      if (length(grid_args) == 0) return(tibble())
      grid_df <- do.call(
        expand.grid,
        c(
          lapply(grid_args, function(x) {
            if (is.factor(x)) as.character(x) else x
          }),
          list(KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
        )
      )
      if (nrow(grid_df) == 0) return(tibble())

      defaults <- model$default_row
      if (is.null(defaults) || !is.data.frame(defaults) || nrow(defaults) == 0) return(tibble())
      newdata <- defaults[rep(1, nrow(grid_df)), , drop = FALSE]
      for (nm in names(grid_df)) newdata[[nm]] <- grid_df[[nm]]

      meta_vars <- model$grid_meta$vars
      for (nm in names(newdata)) {
        if (!is.null(meta_vars[[nm]])) {
          info <- meta_vars[[nm]]
          typ <- if (!is.null(info$type)) info$type else ""
          if (typ == "factor" && !is.null(info$values)) {
            newdata[[nm]] <- factor(as.character(newdata[[nm]]), levels = as.character(info$values))
          } else if (typ == "logical") {
            newdata[[nm]] <- as.logical(newdata[[nm]])
          } else if (typ == "integer") {
            newdata[[nm]] <- as.integer(round(as.numeric(newdata[[nm]])))
          } else if (typ == "numeric") {
            newdata[[nm]] <- as.numeric(newdata[[nm]])
          } else if (typ == "character") {
            newdata[[nm]] <- as.character(newdata[[nm]])
          }
        }
      }

      mf <- model.frame(model$fixed_terms, data = newdata, xlev = model$xlevels, na.action = na.pass)
      X <- model.matrix(model$fixed_terms, data = mf)

      beta <- model$coefficients
      needed <- names(beta)
      missing_cols <- setdiff(needed, colnames(X))
      if (length(missing_cols) > 0) {
        X <- cbind(X, matrix(0, nrow = nrow(X), ncol = length(missing_cols), dimnames = list(NULL, missing_cols)))
      }
      X <- X[, needed, drop = FALSE]

      est <- as.numeric(X %*% beta)
      vc <- model$vcov_beta
      if (!is.null(vc) && all(needed %in% rownames(vc)) && all(needed %in% colnames(vc))) {
        vc <- vc[needed, needed, drop = FALSE]
        se <- sqrt(pmax(rowSums((X %*% vc) * X), 0))
      } else {
        se <- rep(0, length(est))
      }
      z <- qnorm(0.975)

      pred <- tibble::as_tibble(grid_df) %>%
        mutate(
          estimate = est,
          conf.low = estimate - z * se,
          conf.high = estimate + z * se
        )
    } else {
      newdata <- do.call(datagrid, c(list(model = model), grid_args))
      pred <- predictions(
        model,
        newdata = newdata,
        re.form = NA,
        allow.new.levels = TRUE
      ) %>%
        as_tibble()
    }

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

write_table <- function(df, csv_path, rds_path) {
  if (nrow(df) == 0) {
    log_msg(paste("No rows produced for:", basename(csv_path)))
    return(invisible(NULL))
  }
  readr::write_csv(df, csv_path)
  saveRDS(df, rds_path)
  log_msg(paste("Wrote:", csv_path))
}

add_meta <- function(df, reg_row) {
  if (nrow(df) == 0) return(df)
  df %>%
    mutate(
      outcome = reg_row$outcome,
      outcome_type = reg_row$outcome_type,
      batch = reg_row$batch,
      model_method = reg_row$model_method,
      model_file = basename(reg_row$model_path),
      .before = 1
    )
}

clock_diff_shortest <- function(a, b) {
  ((a - b + 12) %% 24) - 12
}

find_model_row <- function(registry, outcome_name, batch_name) {
  registry %>%
    filter(outcome == outcome_name, batch == batch_name, exists) %>%
    slice(1)
}

print_table_inline <- function(title, df) {
  if (!NOTEBOOK_INLINE) return(invisible(NULL))
  cat("\n====================================================\n")
  cat(title, "\n")
  cat("====================================================\n")

  if (nrow(df) == 0) {
    cat("[0 rows]\n")
    return(invisible(NULL))
  }

  if (INLINE_TABLE_MAX_ROWS <= 0 || nrow(df) <= INLINE_TABLE_MAX_ROWS) {
    print(df, n = nrow(df), width = Inf)
  } else {
    print(df, n = INLINE_TABLE_MAX_ROWS, width = Inf)
    cat(paste0(
      "\n[Truncated inline print: showing ",
      INLINE_TABLE_MAX_ROWS,
      " of ",
      nrow(df),
      " rows. Set INLINE_TABLE_MAX_ROWS_05_5=0 for all rows.]\n"
    ))
  }

  invisible(NULL)
}

print_plot_inline <- function(title, plot_obj) {
  if (!NOTEBOOK_INLINE) return(invisible(NULL))
  cat("\n----------------------------------------------------\n")
  cat("Figure:", title, "\n")
  cat("----------------------------------------------------\n")
  print(plot_obj)
  invisible(NULL)
}

# %% [markdown]
# ## Model Inventory

# %%
# -------------------------------------------------------------------
# Model inventory
# -------------------------------------------------------------------
model_registry <- tidyr::crossing(outcome_specs, batch_specs) %>%
  mutate(stem_with_suffix = paste0(stem, suffix)) %>%
  mutate(resolved = map(stem_with_suffix, resolve_model_file)) %>%
  mutate(
    model_path = map_chr(resolved, "path"),
    model_method = map_chr(resolved, "method"),
    exists = map_lgl(resolved, "exists")
  ) %>%
  select(outcome, outcome_type, batch, stem_with_suffix, model_path, model_method, exists)

readr::write_csv(model_registry, file.path(TABLE_DIR, "model_inventory_05_5.csv"))
saveRDS(model_registry, file.path(TABLE_DIR, "model_inventory_05_5.rds"))

log_msg(paste("Models found:", sum(model_registry$exists), "of", nrow(model_registry)))
inline_plots <- list()

# %% [markdown]
# ## Section 1: Comprehensive Marginal Means

# %%
# -------------------------------------------------------------------
# Section 1: Comprehensive marginal means
# -------------------------------------------------------------------
all_marginal_means <- list()
idx <- 1

for (i in seq_len(nrow(model_registry))) {
  reg_row <- model_registry[i, ]

  if (!reg_row$exists) {
    log_msg(paste("Skipping missing model:", reg_row$model_path))
    next
  }

  log_msg(paste("Running marginal means for:", reg_row$stem_with_suffix, "[", reg_row$model_method, "]"))
  model <- safe_read_model(reg_row$model_path)
  if (is.null(model)) next

  # Employment x Weekend
  if (has_vars(model, c("employment_status", "is_weekend_factor"))) {
    emp_levels <- get_model_values(model, "employment_status")
    wk_levels <- get_model_values(model, "is_weekend_factor")

    preds <- safe_predictions(
      model = model,
      grid_args = list(employment_status = emp_levels, is_weekend_factor = wk_levels),
      keep_cols = c("employment_status", "is_weekend_factor"),
      analysis_name = "employment_x_weekend"
    )
    all_marginal_means[[idx]] <- add_meta(preds, reg_row)
    idx <- idx + 1
  }

  # Sex
  if (has_vars(model, c("sex_concept"))) {
    sex_levels <- get_model_values(model, "sex_concept")
    preds <- safe_predictions(
      model = model,
      grid_args = list(sex_concept = sex_levels),
      keep_cols = c("sex_concept"),
      analysis_name = "sex_main"
    )
    all_marginal_means[[idx]] <- add_meta(preds, reg_row)
    idx <- idx + 1
  }

  # Month
  if (has_vars(model, c("month"))) {
    month_levels <- get_model_values(model, "month")
    preds <- safe_predictions(
      model = model,
      grid_args = list(month = month_levels),
      keep_cols = c("month"),
      analysis_name = "month_main"
    )
    all_marginal_means[[idx]] <- add_meta(preds, reg_row)
    idx <- idx + 1
  }

  # DST main and DST x weekend (if available in the formula/model frame)
  if (has_vars(model, c("dst_observes"))) {
    dst_levels <- get_model_values(model, "dst_observes")

    preds_dst <- safe_predictions(
      model = model,
      grid_args = list(dst_observes = dst_levels),
      keep_cols = c("dst_observes"),
      analysis_name = "dst_main"
    )
    all_marginal_means[[idx]] <- add_meta(preds_dst, reg_row)
    idx <- idx + 1

    if (has_vars(model, c("is_weekend_factor"))) {
      wk_levels <- get_model_values(model, "is_weekend_factor")
      preds_dst_wk <- safe_predictions(
        model = model,
        grid_args = list(dst_observes = dst_levels, is_weekend_factor = wk_levels),
        keep_cols = c("dst_observes", "is_weekend_factor"),
        analysis_name = "dst_x_weekend"
      )
      all_marginal_means[[idx]] <- add_meta(preds_dst_wk, reg_row)
      idx <- idx + 1
    }
  }

  # Age trend
  if (has_vars(model, c("age_at_sleep"))) {
    age_seq <- seq(AGE_MIN, AGE_MAX, by = AGE_BY)
    preds <- safe_predictions(
      model = model,
      grid_args = list(age_at_sleep = age_seq),
      keep_cols = c("age_at_sleep"),
      analysis_name = "age_main"
    )
    all_marginal_means[[idx]] <- add_meta(preds, reg_row)
    idx <- idx + 1
  }

  rm(model)
  gc(verbose = FALSE)
}

marginal_means_template <- tibble(
  outcome = character(),
  outcome_type = character(),
  batch = character(),
  model_method = character(),
  model_file = character(),
  analysis = character(),
  estimate = numeric(),
  conf.low = numeric(),
  conf.high = numeric()
)

marginal_means_all <- if (length(all_marginal_means) == 0) {
  marginal_means_template
} else {
  bind_rows(all_marginal_means)
}
write_table(
  marginal_means_all,
  file.path(TABLE_DIR, "marginal_means_all_05_5.csv"),
  file.path(TABLE_DIR, "marginal_means_all_05_5.rds")
)

# %% [markdown]
# ## Section 2: DST-focused Controlled Summaries

# %%
# -------------------------------------------------------------------
# Section 2: DST-focused controlled summaries
# -------------------------------------------------------------------
dst_controlled_means <- marginal_means_all %>%
  filter(batch == "dst", analysis %in% c("dst_main", "dst_x_weekend"))

write_table(
  dst_controlled_means,
  file.path(TABLE_DIR, "dst_controlled_means_05_5.csv"),
  file.path(TABLE_DIR, "dst_controlled_means_05_5.rds")
)

dst_weekend_contrast <- tibble()

if (nrow(dst_controlled_means) > 0 &&
    all(c("dst_observes", "is_weekend_factor") %in% names(dst_controlled_means))) {
  dst_weekend_contrast <- dst_controlled_means %>%
    filter(analysis == "dst_x_weekend") %>%
    mutate(
      dst_observes = as.character(dst_observes),
      is_weekend_factor = as.character(is_weekend_factor)
    ) %>%
    select(outcome, outcome_type, batch, model_method, model_file, is_weekend_factor, dst_observes, estimate) %>%
    tidyr::pivot_wider(names_from = dst_observes, values_from = estimate, names_prefix = "estimate_")
}

if (nrow(dst_weekend_contrast) > 0 && all(c("estimate_DST", "estimate_NoDST") %in% names(dst_weekend_contrast))) {
  dst_weekend_contrast <- dst_weekend_contrast %>%
    mutate(diff_NoDST_minus_DST = estimate_NoDST - estimate_DST)
}

write_table(
  dst_weekend_contrast,
  file.path(TABLE_DIR, "dst_weekend_contrasts_05_5.csv"),
  file.path(TABLE_DIR, "dst_weekend_contrasts_05_5.rds")
)

# Age x DST x Weekend for duration model (if available)
duration_dst_row <- find_model_row(model_registry, "duration", "dst")
duration_dst_age <- tibble()

if (nrow(duration_dst_row) == 1) {
  log_msg("Running duration age profile with DST controls.")
  model <- safe_read_model(duration_dst_row$model_path)
  if (!is.null(model) && has_vars(model, c("age_at_sleep", "dst_observes"))) {
    grid_args <- list(
      age_at_sleep = seq(AGE_MIN, AGE_MAX, by = AGE_BY),
      dst_observes = get_model_values(model, "dst_observes")
    )
    keep_cols <- c("age_at_sleep", "dst_observes")

    if (has_vars(model, c("is_weekend_factor"))) {
      grid_args$is_weekend_factor <- get_model_values(model, "is_weekend_factor")
      keep_cols <- c(keep_cols, "is_weekend_factor")
    }

    duration_dst_age <- safe_predictions(
      model = model,
      grid_args = grid_args,
      keep_cols = keep_cols,
      analysis_name = "duration_age_dst"
    ) %>%
      add_meta(duration_dst_row)
  } else {
    log_msg("Duration DST model missing required variables for age/DST profile.")
  }
  rm(model)
  gc(verbose = FALSE)
} else {
  log_msg("No duration DST model available. Skipping duration DST age section.")
}

write_table(
  duration_dst_age,
  file.path(TABLE_DIR, "duration_age_dst_05_5.csv"),
  file.path(TABLE_DIR, "duration_age_dst_05_5.rds")
)

if (nrow(duration_dst_age) > 0) {
  p_dst_age <- ggplot(
    duration_dst_age,
    aes(x = age_at_sleep, y = estimate, color = dst_observes, fill = dst_observes)
  ) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0) +
    geom_line(linewidth = 1.0) +
    scale_y_continuous(labels = format_duration_axis) +
    scale_x_continuous(breaks = pretty_breaks(8)) +
    labs(
      title = "Duration vs Age with DST Controls",
      subtitle = "Marginal means from duration_poly_interact_04_5_dst",
      x = "Age (years)",
      y = "Predicted Sleep Duration",
      color = "DST Group",
      fill = "DST Group"
    ) +
    theme_classic(base_size = 13)

  if ("is_weekend_factor" %in% names(duration_dst_age)) {
    p_dst_age <- p_dst_age + facet_wrap(~is_weekend_factor)
  }

  ggsave(
    filename = file.path(PLOT_DIR, "duration_age_dst_05_5.png"),
    plot = p_dst_age,
    width = 12,
    height = 7,
    dpi = 300
  )
  log_msg("Saved plot: duration_age_dst_05_5.png")
  inline_plots[["duration_age_dst_05_5"]] <- p_dst_age
}

# %% [markdown]
# ## Section 3: Derived Midpoint/Duration from Onset + Offset

# %%
# -------------------------------------------------------------------
# Section 3: Derived midpoint/duration from onset + offset models
# -------------------------------------------------------------------
derived_means_all <- list()
derived_age_all <- list()
comparison_all <- list()

for (batch_name in c("base", "dst")) {
  onset_row <- find_model_row(model_registry, "onset", batch_name)
  offset_row <- find_model_row(model_registry, "offset", batch_name)

  if (nrow(onset_row) == 0 || nrow(offset_row) == 0) {
    log_msg(paste("Skipping derived section for", batch_name, "- onset/offset model missing."))
    next
  }

  onset_model <- safe_read_model(onset_row$model_path)
  offset_model <- safe_read_model(offset_row$model_path)
  if (is.null(onset_model) || is.null(offset_model)) {
    log_msg(paste("Skipping derived section for", batch_name, "- model load failed."))
    rm(onset_model, offset_model)
    gc(verbose = FALSE)
    next
  }

  # --- 3A. Derived means on employment/weekend (+ DST in dst batch)
  shared_emp <- ordered_intersection(
    get_model_values(onset_model, "employment_status"),
    get_model_values(offset_model, "employment_status")
  )
  shared_weekend <- ordered_intersection(
    get_model_values(onset_model, "is_weekend_factor"),
    get_model_values(offset_model, "is_weekend_factor")
  )

  grid_main <- list()
  key_main <- character()

  if (!is.null(shared_emp) && length(shared_emp) > 0) {
    grid_main$employment_status <- shared_emp
    key_main <- c(key_main, "employment_status")
  }
  if (!is.null(shared_weekend) && length(shared_weekend) > 0) {
    grid_main$is_weekend_factor <- shared_weekend
    key_main <- c(key_main, "is_weekend_factor")
  }

  if (batch_name == "dst" && has_vars(onset_model, c("dst_observes")) && has_vars(offset_model, c("dst_observes"))) {
    shared_dst <- ordered_intersection(
      get_model_values(onset_model, "dst_observes"),
      get_model_values(offset_model, "dst_observes")
    )
    if (!is.null(shared_dst) && length(shared_dst) > 0) {
      grid_main$dst_observes <- shared_dst
      key_main <- c(key_main, "dst_observes")
    }
  }

  if (length(grid_main) > 0) {
    onset_pred_main <- safe_predictions(
      model = onset_model,
      grid_args = grid_main,
      keep_cols = key_main,
      analysis_name = "onset_main_for_derived"
    ) %>%
      rename(
        onset_estimate = estimate,
        onset_conf_low = conf.low,
        onset_conf_high = conf.high
      )

    offset_pred_main <- safe_predictions(
      model = offset_model,
      grid_args = grid_main,
      keep_cols = key_main,
      analysis_name = "offset_main_for_derived"
    ) %>%
      rename(
        offset_estimate = estimate,
        offset_conf_low = conf.low,
        offset_conf_high = conf.high
      )

    if (nrow(onset_pred_main) > 0 && nrow(offset_pred_main) > 0) {
      derived_main <- onset_pred_main %>%
        select(all_of(c(key_main, "onset_estimate", "onset_conf_low", "onset_conf_high"))) %>%
        inner_join(
          offset_pred_main %>% select(all_of(c(key_main, "offset_estimate", "offset_conf_low", "offset_conf_high"))),
          by = key_main
        ) %>%
        mutate(
          batch = batch_name,
          derived_duration_hours = (offset_estimate - onset_estimate) %% 24,
          derived_midpoint_linear = (onset_estimate + derived_duration_hours / 2) %% 24
        ) %>%
        relocate(batch, .before = 1)

      derived_means_all[[paste0("means_", batch_name)]] <- derived_main

      # Optional comparison against direct midpoint/duration models on the same grid
      duration_row <- find_model_row(model_registry, "duration", batch_name)
      midpoint_row <- find_model_row(model_registry, "midpoint", batch_name)

      direct_compare <- derived_main

      if (nrow(duration_row) == 1) {
        duration_model <- safe_read_model(duration_row$model_path)
        if (!is.null(duration_model)) {
          duration_direct <- safe_predictions(
            model = duration_model,
            grid_args = grid_main,
            keep_cols = key_main,
            analysis_name = "duration_direct_for_compare"
          ) %>%
            rename(duration_direct_estimate = estimate)

          if (nrow(duration_direct) > 0) {
            direct_compare <- direct_compare %>%
              left_join(
                duration_direct %>% select(all_of(c(key_main, "duration_direct_estimate"))),
                by = key_main
              ) %>%
              mutate(duration_direct_minus_derived = duration_direct_estimate - derived_duration_hours)
          }
        }
        rm(duration_model)
      }

      if (nrow(midpoint_row) == 1) {
        midpoint_model <- safe_read_model(midpoint_row$model_path)
        if (!is.null(midpoint_model)) {
          midpoint_direct <- safe_predictions(
            model = midpoint_model,
            grid_args = grid_main,
            keep_cols = key_main,
            analysis_name = "midpoint_direct_for_compare"
          ) %>%
            rename(midpoint_direct_estimate = estimate)

          if (nrow(midpoint_direct) > 0) {
            direct_compare <- direct_compare %>%
              left_join(
                midpoint_direct %>% select(all_of(c(key_main, "midpoint_direct_estimate"))),
                by = key_main
              ) %>%
              mutate(midpoint_direct_minus_derived = clock_diff_shortest(midpoint_direct_estimate, derived_midpoint_linear))
          }
        }
        rm(midpoint_model)
      }

      comparison_all[[paste0("compare_", batch_name)]] <- direct_compare
    }
  } else {
    log_msg(paste("No shared categorical grid for derived means in", batch_name))
  }

  # --- 3B. Derived duration vs age (key diagnostic for odd age patterns)
  grid_age <- list(age_at_sleep = seq(AGE_MIN, AGE_MAX, by = AGE_BY))
  key_age <- c("age_at_sleep")

  if (!is.null(shared_weekend) && length(shared_weekend) > 0) {
    grid_age$is_weekend_factor <- shared_weekend
    key_age <- c(key_age, "is_weekend_factor")
  }

  if (batch_name == "dst" && has_vars(onset_model, c("dst_observes")) && has_vars(offset_model, c("dst_observes"))) {
    shared_dst <- ordered_intersection(
      get_model_values(onset_model, "dst_observes"),
      get_model_values(offset_model, "dst_observes")
    )
    if (!is.null(shared_dst) && length(shared_dst) > 0) {
      grid_age$dst_observes <- shared_dst
      key_age <- c(key_age, "dst_observes")
    }
  }

  onset_age <- safe_predictions(
    model = onset_model,
    grid_args = grid_age,
    keep_cols = key_age,
    analysis_name = "onset_age_for_derived"
  ) %>%
    rename(onset_estimate = estimate)

  offset_age <- safe_predictions(
    model = offset_model,
    grid_args = grid_age,
    keep_cols = key_age,
    analysis_name = "offset_age_for_derived"
  ) %>%
    rename(offset_estimate = estimate)

  if (nrow(onset_age) > 0 && nrow(offset_age) > 0) {
    derived_age <- onset_age %>%
      select(all_of(c(key_age, "onset_estimate"))) %>%
      inner_join(offset_age %>% select(all_of(c(key_age, "offset_estimate"))), by = key_age) %>%
      mutate(
        batch = batch_name,
        derived_duration_hours = (offset_estimate - onset_estimate) %% 24,
        derived_midpoint_linear = (onset_estimate + derived_duration_hours / 2) %% 24
      ) %>%
      relocate(batch, .before = 1)

    duration_row <- find_model_row(model_registry, "duration", batch_name)
    if (nrow(duration_row) == 1) {
      duration_model <- safe_read_model(duration_row$model_path)
      if (!is.null(duration_model)) {
        duration_age <- safe_predictions(
          model = duration_model,
          grid_args = grid_age,
          keep_cols = key_age,
          analysis_name = "duration_age_direct"
        ) %>%
          rename(duration_direct_estimate = estimate)

        if (nrow(duration_age) > 0) {
          derived_age <- derived_age %>%
            left_join(duration_age %>% select(all_of(c(key_age, "duration_direct_estimate"))), by = key_age) %>%
            mutate(duration_direct_minus_derived = duration_direct_estimate - derived_duration_hours)
        }
      }
      rm(duration_model)
    }

    derived_age_all[[paste0("age_", batch_name)]] <- derived_age

    # Plot: direct vs derived duration vs age
    if ("duration_direct_estimate" %in% names(derived_age)) {
      plot_df <- derived_age %>%
        select(all_of(key_age), batch, derived_duration_hours, duration_direct_estimate) %>%
        pivot_longer(
          cols = c(derived_duration_hours, duration_direct_estimate),
          names_to = "series",
          values_to = "estimate"
        ) %>%
        mutate(
          series = recode(
            series,
            derived_duration_hours = "Derived from Onset+Offset",
            duration_direct_estimate = "Direct Duration Model"
          )
        )

      p_age <- ggplot(plot_df, aes(x = age_at_sleep, y = estimate, color = series)) +
        geom_line(linewidth = 1.0) +
        scale_y_continuous(labels = format_duration_axis) +
        scale_x_continuous(breaks = pretty_breaks(8)) +
        labs(
          title = paste("Duration vs Age:", str_to_title(batch_name), "batch"),
          subtitle = "Comparing direct duration model with onset/offset-derived duration",
          x = "Age (years)",
          y = "Predicted Sleep Duration",
          color = NULL
        ) +
        theme_classic(base_size = 13)

      facet_vars <- c()
      if ("is_weekend_factor" %in% names(plot_df)) facet_vars <- c(facet_vars, "is_weekend_factor")
      if ("dst_observes" %in% names(plot_df)) facet_vars <- c(facet_vars, "dst_observes")
      if (length(facet_vars) > 0) {
        facet_formula <- as.formula(paste("~", paste(facet_vars, collapse = " + ")))
        p_age <- p_age + facet_wrap(facet_formula, scales = "free_y")
      }

      ggsave(
        filename = file.path(PLOT_DIR, paste0("duration_age_direct_vs_derived_", batch_name, "_05_5.png")),
        plot = p_age,
        width = 12,
        height = 7,
        dpi = 300
      )
      log_msg(paste("Saved plot: duration_age_direct_vs_derived_", batch_name, "_05_5.png", sep = ""))
      inline_plots[[paste0("duration_age_direct_vs_derived_", batch_name, "_05_5")]] <- p_age
    }
  } else {
    log_msg(paste("Skipping derived age section for", batch_name, "- no onset/offset age predictions."))
  }

  rm(onset_model, offset_model)
  gc(verbose = FALSE)
}

derived_means_tbl <- bind_rows(derived_means_all)
derived_age_tbl <- bind_rows(derived_age_all)
derived_compare_tbl <- bind_rows(comparison_all)

write_table(
  derived_means_tbl,
  file.path(TABLE_DIR, "derived_midpoint_duration_means_05_5.csv"),
  file.path(TABLE_DIR, "derived_midpoint_duration_means_05_5.rds")
)

write_table(
  derived_age_tbl,
  file.path(TABLE_DIR, "derived_midpoint_duration_age_05_5.csv"),
  file.path(TABLE_DIR, "derived_midpoint_duration_age_05_5.rds")
)

write_table(
  derived_compare_tbl,
  file.path(TABLE_DIR, "derived_vs_direct_comparison_05_5.csv"),
  file.path(TABLE_DIR, "derived_vs_direct_comparison_05_5.rds")
)

# %% [markdown]
# ## Section 4: Global Duration vs Age Plot

# %%
# -------------------------------------------------------------------
# Section 4: Global duration-vs-age plot (all available batches)
# -------------------------------------------------------------------
duration_age_direct <- marginal_means_all %>%
  filter(outcome == "duration", analysis == "age_main")

if (nrow(duration_age_direct) > 0) {
  p_duration_age <- ggplot(
    duration_age_direct,
    aes(x = age_at_sleep, y = estimate, color = batch, fill = batch)
  ) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), alpha = 0.12, linewidth = 0) +
    geom_line(linewidth = 1.1) +
    scale_y_continuous(labels = format_duration_axis) +
    scale_x_continuous(breaks = pretty_breaks(8)) +
    labs(
      title = "Duration vs Age (Direct Models)",
      subtitle = "Marginal means from all available duration models",
      x = "Age (years)",
      y = "Predicted Sleep Duration",
      color = "Batch",
      fill = "Batch"
    ) +
    theme_classic(base_size = 13)

  ggsave(
    filename = file.path(PLOT_DIR, "duration_age_direct_models_05_5.png"),
    plot = p_duration_age,
    width = 11,
    height = 6.5,
    dpi = 300
  )
  log_msg("Saved plot: duration_age_direct_models_05_5.png")
  inline_plots[["duration_age_direct_models_05_5"]] <- p_duration_age
} else {
  log_msg("No duration age_main predictions available for global duration-vs-age plot.")
}

# -------------------------------------------------------------------
# Notebook inline output
# -------------------------------------------------------------------
if (NOTEBOOK_INLINE) {
  print_table_inline("Model Inventory (resolved files)", model_registry)
  print_table_inline("Marginal Means (all analyses)", marginal_means_all)
  print_table_inline("DST-Controlled Means", dst_controlled_means)
  print_table_inline("DST Weekend Contrasts", dst_weekend_contrast)
  print_table_inline("Duration Age DST", duration_dst_age)
  print_table_inline("Derived Midpoint/Duration Means", derived_means_tbl)
  print_table_inline("Derived Midpoint/Duration Age", derived_age_tbl)
  print_table_inline("Derived vs Direct Comparison", derived_compare_tbl)
  print_table_inline("Direct Duration Age (from marginal means)", duration_age_direct)

  if (length(inline_plots) == 0) {
    cat("\nNo figures available to print inline for current model availability.\n")
  } else {
    for (nm in names(inline_plots)) {
      print_plot_inline(nm, inline_plots[[nm]])
    }
  }
}

# %% [markdown]
# ## Final Export

# %%
# -------------------------------------------------------------------
# Final export: run log and completion status
# -------------------------------------------------------------------
writeLines(analysis_log, con = file.path(OUTPUT_DIR, "run_log_05_5.txt"))

log_msg("Done.")
log_msg(paste("Tables:", TABLE_DIR))
log_msg(paste("Plots:", PLOT_DIR))
