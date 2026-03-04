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
# # 05_7_Precompute_Predictions.r
# Precompute and cache marginal predictions for fast re-plotting.
# - Resolves available 04_5 models (base + dst)
# - Generates common prediction grids per model
# - Saves long-form prediction tables for downstream plotting
# - Generates derived duration/midpoint age predictions from onset + offset

# %%
required_packages <- c(
  "dplyr", "tidyr", "purrr", "readr", "tibble", "marginaleffects"
)

missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install them before running 05_7_Precompute_Predictions.r."
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(tidyr)
  library(purrr)
  library(readr)
  library(tibble)
  library(marginaleffects)
})

set.seed(123)

# %% [markdown]
# ## Configuration

# %%
MODEL_DIR <- Sys.getenv("MODEL_DIR_04_5", "models_04_5")
OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR_05_7", "results_05_7")
TABLE_DIR <- file.path(OUTPUT_DIR, "tables")

PREFERRED_METHOD <- toupper(Sys.getenv("MODEL_METHOD_05_7", "REML"))
SUPPORTED_METHODS <- c("REML", "ML")
METHOD_PRIORITY <- unique(c(PREFERRED_METHOD, setdiff(SUPPORTED_METHODS, PREFERRED_METHOD)))

AGE_MIN <- as.numeric(Sys.getenv("AGE_MIN_05_7", "18"))
AGE_MAX <- as.numeric(Sys.getenv("AGE_MAX_05_7", "85"))
AGE_BY <- as.numeric(Sys.getenv("AGE_BY_05_7", "1"))

# all | base | dst
PRED_SCOPE <- tolower(Sys.getenv("PRED_SCOPE_05_7", "all"))
if (!PRED_SCOPE %in% c("all", "base", "dst")) {
  stop("PRED_SCOPE_05_7 must be one of: all, base, dst")
}

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)
if (!dir.exists(TABLE_DIR)) dir.create(TABLE_DIR, recursive = TRUE)

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
log_msg(paste("PRED_SCOPE:", PRED_SCOPE))

# %% [markdown]
# ## Helpers

# %%
run_gc <- function(label = NULL) {
  invisible(gc(verbose = FALSE))
  if (!is.null(label)) log_msg(paste("Memory cleanup:", label))
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
  if (!var_name %in% names(model@frame)) return(NULL)
  x <- model@frame[[var_name]]
  if (is.factor(x)) return(levels(x))
  sort(unique(x))
}

has_vars <- function(model, vars) {
  all(vars %in% names(model@frame))
}

safe_predictions <- function(model, grid_args, keep_cols, analysis_name) {
  tryCatch({
    newdata <- do.call(datagrid, c(list(model = model), grid_args))
    pred <- predictions(
      model,
      newdata = newdata,
      re.form = NA,
      allow.new.levels = TRUE
    ) %>%
      as_tibble()

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

write_table <- function(df, csv_name) {
  csv_path <- file.path(TABLE_DIR, csv_name)
  if (nrow(df) == 0) {
    log_msg(paste("No rows produced for:", csv_name))
    return(invisible(NULL))
  }
  readr::write_csv(df, csv_path)
  saveRDS(df, sub("\\.csv$", ".rds", csv_path))
  if (requireNamespace("arrow", quietly = TRUE)) {
    arrow::write_parquet(df, sub("\\.csv$", ".parquet", csv_path))
  }
  log_msg(paste("Wrote:", csv_path))
}

find_model_row <- function(registry, outcome_name, batch_name) {
  registry %>%
    filter(outcome == outcome_name, batch == batch_name, exists) %>%
    slice(1)
}

# %% [markdown]
# ## Model Inventory

# %%
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

model_registry <- tidyr::crossing(outcome_specs, batch_specs) %>%
  mutate(stem_with_suffix = paste0(stem, suffix)) %>%
  mutate(resolved = map(stem_with_suffix, resolve_model_file)) %>%
  mutate(
    model_path = map_chr(resolved, "path"),
    model_method = map_chr(resolved, "method"),
    exists = map_lgl(resolved, "exists")
  ) %>%
  select(outcome, outcome_type, batch, stem_with_suffix, model_path, model_method, exists)

if (PRED_SCOPE != "all") {
  model_registry <- model_registry %>% filter(batch == PRED_SCOPE)
}

write_table(model_registry, "model_inventory_05_7.csv")
log_msg(paste("Models found:", sum(model_registry$exists), "of", nrow(model_registry)))

if (sum(model_registry$exists) == 0) {
  stop("No models available for requested PRED_SCOPE_05_7.")
}

# %% [markdown]
# ## Section 1: Per-model Marginal Predictions

# %%
all_predictions <- list()
idx <- 1

for (i in seq_len(nrow(model_registry))) {
  reg_row <- model_registry[i, ]

  if (!reg_row$exists) {
    log_msg(paste("Skipping missing model:", reg_row$model_path))
    next
  }

  log_msg(paste("Precomputing:", reg_row$stem_with_suffix, "[", reg_row$model_method, "]"))
  model <- safe_read_model(reg_row$model_path)
  if (is.null(model)) next

  # Employment x Weekend
  if (has_vars(model, c("employment_status", "is_weekend_factor"))) {
    pred_emp_wk <- safe_predictions(
      model = model,
      grid_args = list(
        employment_status = get_model_values(model, "employment_status"),
        is_weekend_factor = get_model_values(model, "is_weekend_factor")
      ),
      keep_cols = c("employment_status", "is_weekend_factor"),
      analysis_name = "employment_x_weekend"
    )
    all_predictions[[idx]] <- add_meta(pred_emp_wk, reg_row)
    idx <- idx + 1
  }

  # Sex
  if (has_vars(model, c("sex_concept"))) {
    pred_sex <- safe_predictions(
      model = model,
      grid_args = list(sex_concept = get_model_values(model, "sex_concept")),
      keep_cols = c("sex_concept"),
      analysis_name = "sex_main"
    )
    all_predictions[[idx]] <- add_meta(pred_sex, reg_row)
    idx <- idx + 1
  }

  # Month
  if (has_vars(model, c("month"))) {
    pred_month <- safe_predictions(
      model = model,
      grid_args = list(month = get_model_values(model, "month")),
      keep_cols = c("month"),
      analysis_name = "month_main"
    )
    all_predictions[[idx]] <- add_meta(pred_month, reg_row)
    idx <- idx + 1
  }

  # DST main
  if (has_vars(model, c("dst_observes"))) {
    pred_dst <- safe_predictions(
      model = model,
      grid_args = list(dst_observes = get_model_values(model, "dst_observes")),
      keep_cols = c("dst_observes"),
      analysis_name = "dst_main"
    )
    all_predictions[[idx]] <- add_meta(pred_dst, reg_row)
    idx <- idx + 1
  }

  # Month x DST (useful for onset/offset DST month plots)
  if (has_vars(model, c("month", "dst_observes"))) {
    pred_month_dst <- safe_predictions(
      model = model,
      grid_args = list(
        month = get_model_values(model, "month"),
        dst_observes = get_model_values(model, "dst_observes")
      ),
      keep_cols = c("month", "dst_observes"),
      analysis_name = "month_x_dst"
    )
    all_predictions[[idx]] <- add_meta(pred_month_dst, reg_row)
    idx <- idx + 1
  }

  # DST x Weekend
  if (has_vars(model, c("dst_observes", "is_weekend_factor"))) {
    pred_dst_wk <- safe_predictions(
      model = model,
      grid_args = list(
        dst_observes = get_model_values(model, "dst_observes"),
        is_weekend_factor = get_model_values(model, "is_weekend_factor")
      ),
      keep_cols = c("dst_observes", "is_weekend_factor"),
      analysis_name = "dst_x_weekend"
    )
    all_predictions[[idx]] <- add_meta(pred_dst_wk, reg_row)
    idx <- idx + 1
  }

  # Age
  if (has_vars(model, c("age_at_sleep"))) {
    pred_age <- safe_predictions(
      model = model,
      grid_args = list(age_at_sleep = seq(AGE_MIN, AGE_MAX, by = AGE_BY)),
      keep_cols = c("age_at_sleep"),
      analysis_name = "age_main"
    )
    all_predictions[[idx]] <- add_meta(pred_age, reg_row)
    idx <- idx + 1
  }

  rm(model)
  run_gc(paste("finished", basename(reg_row$model_path)))
}

predictions_all <- if (length(all_predictions) == 0) tibble() else bind_rows(all_predictions)
write_table(predictions_all, "predictions_all_05_7.csv")

# %% [markdown]
# ## Section 2: Derived Predictions (Onset + Offset)

# %%
derived_age_list <- list()
derived_compare_list <- list()
idx_age <- 1
idx_cmp <- 1

available_batches <- unique(model_registry$batch)

for (b in available_batches) {
  onset_row <- find_model_row(model_registry, "onset", b)
  offset_row <- find_model_row(model_registry, "offset", b)

  if (nrow(onset_row) == 0 || nrow(offset_row) == 0) {
    log_msg(paste("Skipping derived predictions for", b, "- onset/offset missing."))
    next
  }

  onset_model <- safe_read_model(onset_row$model_path)
  offset_model <- safe_read_model(offset_row$model_path)
  if (is.null(onset_model) || is.null(offset_model)) {
    log_msg(paste("Skipping derived predictions for", b, "- failed to load onset/offset."))
    rm(onset_model, offset_model)
    run_gc(paste("failed load cleanup for", b))
    next
  }

  grid_age <- list(age_at_sleep = seq(AGE_MIN, AGE_MAX, by = AGE_BY))
  key_age <- c("age_at_sleep")

  shared_weekend <- intersect(
    get_model_values(onset_model, "is_weekend_factor"),
    get_model_values(offset_model, "is_weekend_factor")
  )
  if (!is.null(shared_weekend) && length(shared_weekend) > 0) {
    grid_age$is_weekend_factor <- shared_weekend
    key_age <- c(key_age, "is_weekend_factor")
  }

  shared_dst <- intersect(
    get_model_values(onset_model, "dst_observes"),
    get_model_values(offset_model, "dst_observes")
  )
  if (!is.null(shared_dst) && length(shared_dst) > 0) {
    grid_age$dst_observes <- shared_dst
    key_age <- c(key_age, "dst_observes")
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
        batch = b,
        derived_duration_hours = (offset_estimate - onset_estimate) %% 24,
        derived_midpoint_linear = (onset_estimate + derived_duration_hours / 2) %% 24,
        analysis = "derived_duration_midpoint_age"
      ) %>%
      relocate(batch, analysis, .before = 1)

    derived_age_list[[idx_age]] <- derived_age
    idx_age <- idx_age + 1

    # Optional comparison to direct models if available
    duration_row <- find_model_row(model_registry, "duration", b)
    midpoint_row <- find_model_row(model_registry, "midpoint", b)
    compare_df <- derived_age

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
          compare_df <- compare_df %>%
            left_join(duration_age %>% select(all_of(c(key_age, "duration_direct_estimate"))), by = key_age) %>%
            mutate(duration_direct_minus_derived = duration_direct_estimate - derived_duration_hours)
        }
      }
      rm(duration_model)
      run_gc(paste("duration compare cleanup for", b))
    }

    if (nrow(midpoint_row) == 1) {
      midpoint_model <- safe_read_model(midpoint_row$model_path)
      if (!is.null(midpoint_model)) {
        midpoint_age <- safe_predictions(
          model = midpoint_model,
          grid_args = grid_age,
          keep_cols = key_age,
          analysis_name = "midpoint_age_direct"
        ) %>%
          rename(midpoint_direct_estimate = estimate)

        if (nrow(midpoint_age) > 0) {
          compare_df <- compare_df %>%
            left_join(midpoint_age %>% select(all_of(c(key_age, "midpoint_direct_estimate"))), by = key_age) %>%
            mutate(midpoint_direct_minus_derived = clock_diff_shortest(midpoint_direct_estimate, derived_midpoint_linear))
        }
      }
      rm(midpoint_model)
      run_gc(paste("midpoint compare cleanup for", b))
    }

    derived_compare_list[[idx_cmp]] <- compare_df
    idx_cmp <- idx_cmp + 1
  } else {
    log_msg(paste("No onset/offset age predictions for derived section in", b))
  }

  rm(onset_model, offset_model)
  run_gc(paste("derived section finished for", b))
}

derived_age_tbl <- if (length(derived_age_list) == 0) tibble() else bind_rows(derived_age_list)
derived_compare_tbl <- if (length(derived_compare_list) == 0) tibble() else bind_rows(derived_compare_list)

write_table(derived_age_tbl, "derived_duration_midpoint_age_05_7.csv")
write_table(derived_compare_tbl, "derived_vs_direct_age_05_7.csv")

# %% [markdown]
# ## Final Export

# %%
writeLines(analysis_log, con = file.path(OUTPUT_DIR, "run_log_05_7.txt"))
log_msg("Done.")
log_msg(paste("Tables:", TABLE_DIR))
