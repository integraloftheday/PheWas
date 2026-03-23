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
# # 05_5_Precompute_Predictions.r
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
      ". Install them before running 05_5_Precompute_Predictions.r."
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
MODEL_DIR <- Sys.getenv("MODEL_DIR_04", "models_04")
OUTPUT_DIR <- Sys.getenv("OUTPUT_DIR_05", "results_05")
TABLE_DIR <- file.path(OUTPUT_DIR, "tables")
CHECKPOINT_DIR <- file.path(TABLE_DIR, "_checkpoints")

PREFERRED_METHOD <- toupper(Sys.getenv("MODEL_METHOD_05", "REML"))
SUPPORTED_METHODS <- c("REML", "ML")
METHOD_PRIORITY <- unique(c(PREFERRED_METHOD, setdiff(SUPPORTED_METHODS, PREFERRED_METHOD)))

AGE_MIN <- as.numeric(Sys.getenv("AGE_MIN_05", "18"))
AGE_MAX <- as.numeric(Sys.getenv("AGE_MAX_05", "85"))
AGE_BY <- as.numeric(Sys.getenv("AGE_BY_05", "1"))

# all | base | dst
PRED_SCOPE <- tolower(Sys.getenv("PRED_SCOPE_05", "all"))
if (!PRED_SCOPE %in% c("all", "base", "dst")) {
  stop("PRED_SCOPE_05 must be one of: all, base, dst")
}

RESUME_CHECKPOINTS <- tolower(Sys.getenv("RESUME_CHECKPOINTS_05", "false")) %in% c("true", "1", "yes")
MEM_DIAGNOSTICS <- tolower(Sys.getenv("MEM_DIAGNOSTICS_05", "true")) %in% c("true", "1", "yes")

if (!dir.exists(OUTPUT_DIR)) dir.create(OUTPUT_DIR, recursive = TRUE)
if (!dir.exists(TABLE_DIR)) dir.create(TABLE_DIR, recursive = TRUE)
if (!dir.exists(CHECKPOINT_DIR)) dir.create(CHECKPOINT_DIR, recursive = TRUE)

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
log_msg(paste("RESUME_CHECKPOINTS:", RESUME_CHECKPOINTS))
log_msg(paste("MEM_DIAGNOSTICS:", MEM_DIAGNOSTICS))

# %% [markdown]
# ## Helpers

# %%
run_gc <- function(label = NULL) {
  invisible(gc(verbose = FALSE))
  if (!is.null(label)) log_msg(paste("Memory cleanup:", label))
}

read_proc_kb <- function(path, field) {
  if (!file.exists(path)) return(NA_real_)
  lines <- tryCatch(readLines(path, warn = FALSE), error = function(e) character())
  hit <- grep(paste0("^", field, ":"), lines, value = TRUE)
  if (length(hit) == 0) return(NA_real_)
  as.numeric(gsub("[^0-9]", "", hit[[1]]))
}

fmt_mb_from_kb <- function(x) {
  if (is.na(x)) return("NA")
  sprintf("%.1fMB", x / 1024)
}

log_memory <- function(label) {
  if (!MEM_DIAGNOSTICS) return(invisible(NULL))
  gc_tbl <- gc(verbose = FALSE)
  r_used_mb <- sum(gc_tbl[, "used"], na.rm = TRUE)
  vmrss_kb <- read_proc_kb("/proc/self/status", "VmRSS")
  vmhwm_kb <- read_proc_kb("/proc/self/status", "VmHWM")
  avail_kb <- read_proc_kb("/proc/meminfo", "MemAvailable")
  log_msg(
    paste0(
      "MEM [", label, "] ",
      "R_used=", sprintf("%.1fMB", r_used_mb),
      " | VmRSS=", fmt_mb_from_kb(vmrss_kb),
      " | VmHWM=", fmt_mb_from_kb(vmhwm_kb),
      " | MemAvail=", fmt_mb_from_kb(avail_kb)
    )
  )
}

rm_if_exists <- function(...) {
  objs <- as.character(substitute(list(...)))[-1L]
  objs <- objs[objs %in% ls(envir = parent.frame(), all.names = TRUE)]
  if (length(objs) > 0) rm(list = objs, envir = parent.frame())
  invisible(NULL)
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

has_vars <- function(model, vars) {
  frm_names <- character()
  if (!inherits(model, "phewas_marginal_model")) {
    frm_names <- tryCatch(names(model@frame), error = function(e) character())
  }
  formula_vars <- tryCatch(all.vars(formula(model)), error = function(e) character())
  meta <- NULL
  if (inherits(model, "phewas_marginal_model")) {
    meta <- model$grid_meta
  } else {
    meta <- attr(model, "phewas_grid_meta", exact = TRUE)
  }
  meta_names <- if (!is.null(meta$vars) && !is.null(names(meta$vars))) names(meta$vars) else character()
  all(vars %in% unique(c(frm_names, formula_vars, meta_names)))
}

ordered_intersection <- function(a, b) {
  if (is.null(a) || is.null(b)) return(NULL)
  a <- as.character(a)
  b <- as.character(b)
  out <- a[a %in% b]
  unique(out)
}

normalize_weekend_level <- function(x) {
  sx <- gsub("[^a-z0-9]+", "", tolower(as.character(x)))
  out <- rep(NA_character_, length(sx))
  out[grepl("weekend", sx) | sx %in% c("true", "t", "1", "yes", "y")] <- "Weekend"
  out[grepl("weekday", sx) | sx %in% c("false", "f", "0", "no", "n")] <- "Weekday"
  out
}

normalize_dst_level <- function(x) {
  sx <- gsub("[^a-z0-9]+", "", tolower(as.character(x)))
  out <- rep(NA_character_, length(sx))
  out[grepl("^no", sx) & grepl("dst", sx)] <- "NoDST"
  out[sx %in% c("nodst", "nost", "no", "0", "false", "f")] <- "NoDST"
  out[is.na(out) & grepl("dst", sx)] <- "DST"
  out[sx %in% c("dst", "yes", "y", "1", "true", "t")] <- "DST"
  out
}

compact_tibble <- function(df) {
  if (is.null(df) || nrow(df) == 0) return(tibble())
  tibble::as_tibble(as.data.frame(df, stringsAsFactors = FALSE))
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
      mutate(analysis = analysis_name, .before = 1) %>%
      compact_tibble()
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
      outcome_variant = reg_row$outcome_variant,
      batch = reg_row$batch,
      model_method = reg_row$model_method,
      model_file = basename(reg_row$model_path),
      .before = 1
    ) %>%
    compact_tibble()
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

checkpoint_schemas <- list(
  predictions_all.csv = c(
    "outcome", "outcome_type", "batch", "model_method", "model_file",
    "outcome_variant",
    "analysis",
    "employment_status", "is_weekend_factor", "sex_concept", "month", "dst_observes", "age_at_sleep",
    "estimate", "conf.low", "conf.high"
  ),
  derived_duration_midpoint_main_grid.csv = c(
    "batch", "analysis", "employment_status", "is_weekend_factor", "dst_observes",
    "onset_estimate", "offset_estimate", "derived_duration_hours", "derived_midpoint_linear"
  ),
  derived_vs_direct_main_grid.csv = c(
    "batch", "analysis", "employment_status", "is_weekend_factor", "dst_observes",
    "onset_estimate", "offset_estimate", "derived_duration_hours", "derived_midpoint_linear",
    "duration_direct_estimate", "duration_direct_minus_derived",
    "duration_adjusted_estimate", "duration_adjusted_minus_derived",
    "midpoint_direct_estimate", "midpoint_direct_minus_derived",
    "midpoint_adjusted_estimate", "midpoint_adjusted_minus_derived"
  ),
  derived_duration_midpoint_age.csv = c(
    "batch", "analysis", "age_at_sleep", "is_weekend_factor", "dst_observes",
    "onset_estimate", "offset_estimate", "derived_duration_hours", "derived_midpoint_linear"
  ),
  derived_vs_direct_age.csv = c(
    "batch", "analysis", "age_at_sleep", "is_weekend_factor", "dst_observes",
    "onset_estimate", "offset_estimate", "derived_duration_hours", "derived_midpoint_linear",
    "duration_direct_estimate", "duration_direct_minus_derived",
    "duration_adjusted_estimate", "duration_adjusted_minus_derived",
    "midpoint_direct_estimate", "midpoint_direct_minus_derived",
    "midpoint_adjusted_estimate", "midpoint_adjusted_minus_derived"
  )
)

align_checkpoint_schema <- function(df, csv_name) {
  schema <- checkpoint_schemas[[csv_name]]
  if (is.null(schema) || nrow(df) == 0) return(df)

  extra_cols <- setdiff(names(df), schema)
  if (length(extra_cols) > 0) {
    log_msg(paste("Checkpoint schema has unexpected columns for", csv_name, ":", paste(extra_cols, collapse = ", ")))
    schema <- c(schema, extra_cols)
  }

  missing_cols <- setdiff(schema, names(df))
  for (nm in missing_cols) df[[nm]] <- NA

  df %>%
    select(all_of(schema)) %>%
    compact_tibble()
}

reset_table_files <- function(csv_name) {
  csv_path <- file.path(TABLE_DIR, csv_name)
  targets <- c(
    csv_path,
    sub("\\.csv$", ".rds", csv_path),
    sub("\\.csv$", ".parquet", csv_path)
  )
  for (p in targets) {
    if (file.exists(p)) file.remove(p)
  }
}

checkpoint_marker_path <- function(kind, key) {
  safe_key <- gsub("[^A-Za-z0-9._-]+", "_", key)
  file.path(CHECKPOINT_DIR, paste0(kind, "__", safe_key, ".done"))
}

checkpoint_is_complete <- function(kind, key) {
  file.exists(checkpoint_marker_path(kind, key))
}

mark_checkpoint_complete <- function(kind, key) {
  p <- checkpoint_marker_path(kind, key)
  writeLines(format(Sys.time(), "%Y-%m-%d %H:%M:%S"), p)
  log_msg(paste("Checkpoint complete:", basename(p)))
}

reset_checkpoints <- function(kind = NULL) {
  pat <- if (is.null(kind)) "\\.done$" else paste0("^", kind, "__.*\\.done$")
  files <- list.files(CHECKPOINT_DIR, pattern = pat, full.names = TRUE)
  if (length(files) > 0) file.remove(files)
}

append_checkpoint <- function(df, csv_name) {
  if (nrow(df) == 0) return(invisible(0L))
  df <- align_checkpoint_schema(df, csv_name)
  csv_path <- file.path(TABLE_DIR, csv_name)
  if (file.exists(csv_path)) {
    readr::write_csv(df, csv_path, append = TRUE)
  } else {
    readr::write_csv(df, csv_path)
  }
  log_msg(paste("Checkpoint append:", csv_path, "| rows:", nrow(df)))
  invisible(nrow(df))
}

read_checkpoint <- function(csv_name) {
  csv_path <- file.path(TABLE_DIR, csv_name)
  if (!file.exists(csv_path)) return(tibble())
  readr::read_csv(csv_path, show_col_types = FALSE)
}

find_model_row <- function(registry, outcome_name, batch_name, variant_name = "primary") {
  registry %>%
    filter(outcome == outcome_name, batch == batch_name, outcome_variant == variant_name, exists) %>%
    slice(1)
}

compute_two_level_difference <- function(df, group_cols, level_col, normalize_fn, high_level, low_level, diff_col) {
  if (nrow(df) == 0) return(tibble())
  if (!level_col %in% names(df)) return(tibble())

  tmp <- df %>%
    mutate(.level = normalize_fn(.data[[level_col]])) %>%
    filter(!is.na(.level))

  if (nrow(tmp) == 0) return(tibble())

  tmp <- tmp %>%
    group_by(across(all_of(c(group_cols, ".level")))) %>%
    summarize(estimate = mean(estimate, na.rm = TRUE), .groups = "drop") %>%
    tidyr::pivot_wider(
      names_from = .level,
      values_from = estimate,
      names_prefix = "estimate_"
    )

  high_col <- paste0("estimate_", high_level)
  low_col <- paste0("estimate_", low_level)
  if (!all(c(high_col, low_col) %in% names(tmp))) return(tibble())

  tmp[[diff_col]] <- tmp[[high_col]] - tmp[[low_col]]
  tmp
}

# %% [markdown]
# ## Model Inventory

# %%
outcome_specs <- tibble::tribble(
  ~outcome, ~stem, ~outcome_type, ~outcome_variant,
  "onset", "onset_poly_interact_04", "ClockTime", "primary",
  "offset", "offset_poly_interact_04", "ClockTime", "primary",
  "midpoint", "midpoint_poly_interact_04", "ClockTime", "primary",
  "duration", "duration_poly_interact_04", "Duration", "primary",
  "midpoint", "midpoint_poly_interact_duration_adjusted_04", "ClockTime", "adjusted",
  "duration", "duration_poly_interact_midpoint_adjusted_04", "Duration", "adjusted"
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
  select(outcome, outcome_type, outcome_variant, batch, stem_with_suffix, model_path, model_method, exists)

if (PRED_SCOPE != "all") {
  model_registry <- model_registry %>% filter(batch == PRED_SCOPE)
}

write_table(model_registry, "model_inventory.csv")
log_msg(paste("Models found:", sum(model_registry$exists), "of", nrow(model_registry)))

if (sum(model_registry$exists) == 0) {
  stop("No models available for requested PRED_SCOPE.")
}

# %% [markdown]
# ## Section 1: Per-model Marginal Predictions

# %%
predictions_csv <- "predictions_all.csv"
if (!RESUME_CHECKPOINTS) {
  reset_table_files(predictions_csv)
  reset_checkpoints("model")
}
pred_rows_written <- 0L

for (i in seq_len(nrow(model_registry))) {
  reg_row <- model_registry[i, ]
  model_key <- paste(reg_row$stem_with_suffix, reg_row$model_method, sep = "__")

  if (!reg_row$exists) {
    log_msg(paste("Skipping missing model:", reg_row$model_path))
    next
  }
  if (RESUME_CHECKPOINTS && checkpoint_is_complete("model", model_key)) {
    log_msg(paste("Skipping completed model:", reg_row$stem_with_suffix, "[", reg_row$model_method, "]"))
    next
  }

  log_msg(paste("Precomputing:", reg_row$stem_with_suffix, "[", reg_row$model_method, "]"))
  log_memory(paste("before_load", basename(reg_row$model_path)))
  model <- safe_read_model(reg_row$model_path)
  if (is.null(model)) next
  log_memory(paste("after_load", basename(reg_row$model_path)))

  append_pred <- function(pred_obj) {
    if (nrow(pred_obj) == 0) return(invisible(NULL))
    pred_out <- add_meta(pred_obj, reg_row)
    pred_size_mb <- as.numeric(object.size(pred_out)) / 1024^2
    log_msg(
      paste0(
        "Prediction chunk: analysis=", unique(pred_out$analysis)[1],
        " | rows=", nrow(pred_out),
        " | size=", sprintf("%.2fMB", pred_size_mb)
      )
    )
    pred_rows_written <<- pred_rows_written + append_checkpoint(pred_out, predictions_csv)
    rm(pred_out)
    run_gc(paste("after checkpoint append", reg_row$stem_with_suffix))
  }

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
    append_pred(pred_emp_wk)
  }

  # Weekend main effect
  if (has_vars(model, c("is_weekend_factor"))) {
    pred_weekend <- safe_predictions(
      model = model,
      grid_args = list(is_weekend_factor = get_model_values(model, "is_weekend_factor")),
      keep_cols = c("is_weekend_factor"),
      analysis_name = "weekend_main"
    )
    append_pred(pred_weekend)
  }

  # Sex
  if (has_vars(model, c("sex_concept"))) {
    pred_sex <- safe_predictions(
      model = model,
      grid_args = list(sex_concept = get_model_values(model, "sex_concept")),
      keep_cols = c("sex_concept"),
      analysis_name = "sex_main"
    )
    append_pred(pred_sex)
  }

  # Month
  if (has_vars(model, c("month"))) {
    pred_month <- safe_predictions(
      model = model,
      grid_args = list(month = get_model_values(model, "month")),
      keep_cols = c("month"),
      analysis_name = "month_main"
    )
    append_pred(pred_month)
  }

  # Month x Weekend
  if (has_vars(model, c("month", "is_weekend_factor"))) {
    pred_month_wk <- safe_predictions(
      model = model,
      grid_args = list(
        month = get_model_values(model, "month"),
        is_weekend_factor = get_model_values(model, "is_weekend_factor")
      ),
      keep_cols = c("month", "is_weekend_factor"),
      analysis_name = "month_x_weekend"
    )
    append_pred(pred_month_wk)
  }

  # DST main
  if (has_vars(model, c("dst_observes"))) {
    pred_dst <- safe_predictions(
      model = model,
      grid_args = list(dst_observes = get_model_values(model, "dst_observes")),
      keep_cols = c("dst_observes"),
      analysis_name = "dst_main"
    )
    append_pred(pred_dst)
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
    append_pred(pred_month_dst)
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
    append_pred(pred_dst_wk)
  }

  # Age
  if (has_vars(model, c("age_at_sleep"))) {
    pred_age <- safe_predictions(
      model = model,
      grid_args = list(age_at_sleep = seq(AGE_MIN, AGE_MAX, by = AGE_BY)),
      keep_cols = c("age_at_sleep"),
      analysis_name = "age_main"
    )
    append_pred(pred_age)
  }

  # Age x Employment
  if (has_vars(model, c("age_at_sleep", "employment_status"))) {
    pred_age_emp <- safe_predictions(
      model = model,
      grid_args = list(
        age_at_sleep = seq(AGE_MIN, AGE_MAX, by = AGE_BY),
        employment_status = get_model_values(model, "employment_status")
      ),
      keep_cols = c("age_at_sleep", "employment_status"),
      analysis_name = "age_x_employment"
    )
    append_pred(pred_age_emp)
  }

  # Age x Weekend
  if (has_vars(model, c("age_at_sleep", "is_weekend_factor"))) {
    pred_age_wk <- safe_predictions(
      model = model,
      grid_args = list(
        age_at_sleep = seq(AGE_MIN, AGE_MAX, by = AGE_BY),
        is_weekend_factor = get_model_values(model, "is_weekend_factor")
      ),
      keep_cols = c("age_at_sleep", "is_weekend_factor"),
      analysis_name = "age_x_weekend"
    )
    append_pred(pred_age_wk)
  }

  # Age x DST for onset/offset (DST-focused diagnostic)
  if (reg_row$outcome %in% c("onset", "offset") && has_vars(model, c("age_at_sleep", "dst_observes"))) {
    pred_age_dst <- safe_predictions(
      model = model,
      grid_args = list(
        age_at_sleep = seq(AGE_MIN, AGE_MAX, by = AGE_BY),
        dst_observes = get_model_values(model, "dst_observes")
      ),
      keep_cols = c("age_at_sleep", "dst_observes"),
      analysis_name = "age_x_dst"
    )
    append_pred(pred_age_dst)
  }

  rm(model)
  run_gc(paste("finished", basename(reg_row$model_path)))
  log_memory(paste("after_model", basename(reg_row$model_path)))
  mark_checkpoint_complete("model", model_key)
}

log_msg(paste("Total prediction rows checkpointed:", pred_rows_written))
predictions_all <- read_checkpoint(predictions_csv)
log_msg(paste("Predictions loaded from checkpoint rows:", nrow(predictions_all)))
log_memory("after_predictions_reload")
write_table(predictions_all, "predictions_all.csv")

# %% [markdown]
# ## Section 2: Derived Predictions (Onset + Offset)

# %%
derived_main_csv <- "derived_duration_midpoint_main_grid.csv"
derived_main_compare_csv <- "derived_vs_direct_main_grid.csv"
derived_age_csv <- "derived_duration_midpoint_age.csv"
derived_age_compare_csv <- "derived_vs_direct_age.csv"

if (!RESUME_CHECKPOINTS) {
  reset_table_files(derived_main_csv)
  reset_table_files(derived_main_compare_csv)
  reset_table_files(derived_age_csv)
  reset_table_files(derived_age_compare_csv)
  reset_checkpoints("derived")
}

available_batches <- unique(model_registry$batch)

for (b in available_batches) {
  if (RESUME_CHECKPOINTS && checkpoint_is_complete("derived", b)) {
    log_msg(paste("Skipping completed derived batch:", b))
    next
  }
  onset_row <- find_model_row(model_registry, "onset", b, "primary")
  offset_row <- find_model_row(model_registry, "offset", b, "primary")

  if (nrow(onset_row) == 0 || nrow(offset_row) == 0) {
    log_msg(paste("Skipping derived predictions for", b, "- onset/offset missing."))
    next
  }

  log_memory(paste("before_derived_load", b))
  onset_model <- safe_read_model(onset_row$model_path)
  offset_model <- safe_read_model(offset_row$model_path)
  log_memory(paste("after_derived_load", b))
  if (is.null(onset_model) || is.null(offset_model)) {
    log_msg(paste("Skipping derived predictions for", b, "- failed to load onset/offset."))
    rm(onset_model, offset_model)
    run_gc(paste("failed load cleanup for", b))
    next
  }

  shared_emp <- ordered_intersection(
    get_model_values(onset_model, "employment_status"),
    get_model_values(offset_model, "employment_status")
  )
  shared_weekend <- ordered_intersection(
    get_model_values(onset_model, "is_weekend_factor"),
    get_model_values(offset_model, "is_weekend_factor")
  )
  shared_dst <- ordered_intersection(
    get_model_values(onset_model, "dst_observes"),
    get_model_values(offset_model, "dst_observes")
  )

  # 2A. Derived main-grid predictions (employment/weekend and optional DST)
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
  if (!is.null(shared_dst) && length(shared_dst) > 0) {
    grid_main$dst_observes <- shared_dst
    key_main <- c(key_main, "dst_observes")
  }

  duration_row <- find_model_row(model_registry, "duration", b, "primary")
  duration_adjusted_row <- find_model_row(model_registry, "duration", b, "adjusted")
  midpoint_row <- find_model_row(model_registry, "midpoint", b, "primary")
  midpoint_adjusted_row <- find_model_row(model_registry, "midpoint", b, "adjusted")

  if (length(grid_main) > 0) {
    onset_main <- safe_predictions(
      model = onset_model,
      grid_args = grid_main,
      keep_cols = key_main,
      analysis_name = "onset_main_for_derived"
    ) %>%
      rename(onset_estimate = estimate)

    offset_main <- safe_predictions(
      model = offset_model,
      grid_args = grid_main,
      keep_cols = key_main,
      analysis_name = "offset_main_for_derived"
    ) %>%
      rename(offset_estimate = estimate)

    if (nrow(onset_main) > 0 && nrow(offset_main) > 0) {
      derived_main <- onset_main %>%
        select(all_of(c(key_main, "onset_estimate"))) %>%
        inner_join(offset_main %>% select(all_of(c(key_main, "offset_estimate"))), by = key_main) %>%
        mutate(
          batch = b,
          analysis = "derived_duration_midpoint_main_grid",
          derived_duration_hours = (offset_estimate - onset_estimate) %% 24,
          derived_midpoint_linear = (onset_estimate + derived_duration_hours / 2) %% 24
        ) %>%
        relocate(batch, analysis, .before = 1)

      append_checkpoint(derived_main, derived_main_csv)

      compare_main <- compact_tibble(derived_main)
      rm(derived_main)
    } else {
      log_msg(paste("No onset/offset main-grid predictions for derived section in", b))
      compare_main <- tibble()
    }
  } else {
    log_msg(paste("No shared categorical grid for derived main section in", b))
    compare_main <- tibble()
  }

  # 2B. Derived age-based predictions
  grid_age <- list(age_at_sleep = seq(AGE_MIN, AGE_MAX, by = AGE_BY))
  key_age <- c("age_at_sleep")

  if (!is.null(shared_weekend) && length(shared_weekend) > 0) {
    grid_age$is_weekend_factor <- shared_weekend
    key_age <- c(key_age, "is_weekend_factor")
  }

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

    append_checkpoint(derived_age, derived_age_csv)

    compare_age <- compact_tibble(derived_age)
    rm(derived_age)
  } else {
    log_msg(paste("No onset/offset age predictions for derived section in", b))
    compare_age <- tibble()
  }

  rm_if_exists(onset_main, offset_main, onset_age, offset_age, onset_model, offset_model)
  run_gc(paste("released onset/offset models for", b))
  log_memory(paste("after_releasing_onoff", b))

  if (nrow(duration_row) == 1 && (nrow(compare_main) > 0 || nrow(compare_age) > 0)) {
    duration_model <- safe_read_model(duration_row$model_path)
    if (!is.null(duration_model)) {
      if (nrow(compare_main) > 0 && length(grid_main) > 0) {
        duration_main <- safe_predictions(
          model = duration_model,
          grid_args = grid_main,
          keep_cols = key_main,
          analysis_name = "duration_main_direct"
        ) %>%
          rename(duration_direct_estimate = estimate)
        if (nrow(duration_main) > 0) {
          compare_main <- compare_main %>%
            left_join(duration_main %>% select(all_of(c(key_main, "duration_direct_estimate"))), by = key_main) %>%
            mutate(duration_direct_minus_derived = duration_direct_estimate - derived_duration_hours) %>%
            compact_tibble()
        }
      }
      if (nrow(compare_age) > 0) {
        duration_age <- safe_predictions(
          model = duration_model,
          grid_args = grid_age,
          keep_cols = key_age,
          analysis_name = "duration_age_direct"
        ) %>%
          rename(duration_direct_estimate = estimate)
        if (nrow(duration_age) > 0) {
          compare_age <- compare_age %>%
            left_join(duration_age %>% select(all_of(c(key_age, "duration_direct_estimate"))), by = key_age) %>%
            mutate(duration_direct_minus_derived = duration_direct_estimate - derived_duration_hours) %>%
            compact_tibble()
        }
      }
    }
    rm_if_exists(duration_main, duration_age, duration_model)
    run_gc(paste("duration compare cleanup for", b))
    log_memory(paste("after_duration_compare", b))
  }

  if (nrow(duration_adjusted_row) == 1 && (nrow(compare_main) > 0 || nrow(compare_age) > 0)) {
    duration_adjusted_model <- safe_read_model(duration_adjusted_row$model_path)
    if (!is.null(duration_adjusted_model)) {
      if (nrow(compare_main) > 0 && length(grid_main) > 0) {
        duration_adjusted_main <- safe_predictions(
          model = duration_adjusted_model,
          grid_args = grid_main,
          keep_cols = key_main,
          analysis_name = "duration_main_adjusted"
        ) %>%
          rename(duration_adjusted_estimate = estimate)
        if (nrow(duration_adjusted_main) > 0) {
          compare_main <- compare_main %>%
            left_join(duration_adjusted_main %>% select(all_of(c(key_main, "duration_adjusted_estimate"))), by = key_main) %>%
            mutate(duration_adjusted_minus_derived = duration_adjusted_estimate - derived_duration_hours) %>%
            compact_tibble()
        }
      }
      if (nrow(compare_age) > 0) {
        duration_adjusted_age <- safe_predictions(
          model = duration_adjusted_model,
          grid_args = grid_age,
          keep_cols = key_age,
          analysis_name = "duration_age_adjusted"
        ) %>%
          rename(duration_adjusted_estimate = estimate)
        if (nrow(duration_adjusted_age) > 0) {
          compare_age <- compare_age %>%
            left_join(duration_adjusted_age %>% select(all_of(c(key_age, "duration_adjusted_estimate"))), by = key_age) %>%
            mutate(duration_adjusted_minus_derived = duration_adjusted_estimate - derived_duration_hours) %>%
            compact_tibble()
        }
      }
    }
    rm_if_exists(duration_adjusted_main, duration_adjusted_age, duration_adjusted_model)
    run_gc(paste("duration adjusted compare cleanup for", b))
    log_memory(paste("after_duration_adjusted_compare", b))
  }

  if (nrow(midpoint_row) == 1 && (nrow(compare_main) > 0 || nrow(compare_age) > 0)) {
    midpoint_model <- safe_read_model(midpoint_row$model_path)
    if (!is.null(midpoint_model)) {
      if (nrow(compare_main) > 0 && length(grid_main) > 0) {
        midpoint_main <- safe_predictions(
          model = midpoint_model,
          grid_args = grid_main,
          keep_cols = key_main,
          analysis_name = "midpoint_main_direct"
        ) %>%
          rename(midpoint_direct_estimate = estimate)
        if (nrow(midpoint_main) > 0) {
          compare_main <- compare_main %>%
            left_join(midpoint_main %>% select(all_of(c(key_main, "midpoint_direct_estimate"))), by = key_main) %>%
            mutate(midpoint_direct_minus_derived = clock_diff_shortest(midpoint_direct_estimate, derived_midpoint_linear)) %>%
            compact_tibble()
        }
      }
      if (nrow(compare_age) > 0) {
        midpoint_age <- safe_predictions(
          model = midpoint_model,
          grid_args = grid_age,
          keep_cols = key_age,
          analysis_name = "midpoint_age_direct"
        ) %>%
          rename(midpoint_direct_estimate = estimate)
        if (nrow(midpoint_age) > 0) {
          compare_age <- compare_age %>%
            left_join(midpoint_age %>% select(all_of(c(key_age, "midpoint_direct_estimate"))), by = key_age) %>%
            mutate(midpoint_direct_minus_derived = clock_diff_shortest(midpoint_direct_estimate, derived_midpoint_linear)) %>%
            compact_tibble()
        }
      }
    }
    rm_if_exists(midpoint_main, midpoint_age, midpoint_model)
    run_gc(paste("midpoint compare cleanup for", b))
    log_memory(paste("after_midpoint_compare", b))
  }

  if (nrow(midpoint_adjusted_row) == 1 && (nrow(compare_main) > 0 || nrow(compare_age) > 0)) {
    midpoint_adjusted_model <- safe_read_model(midpoint_adjusted_row$model_path)
    if (!is.null(midpoint_adjusted_model)) {
      if (nrow(compare_main) > 0 && length(grid_main) > 0) {
        midpoint_adjusted_main <- safe_predictions(
          model = midpoint_adjusted_model,
          grid_args = grid_main,
          keep_cols = key_main,
          analysis_name = "midpoint_main_adjusted"
        ) %>%
          rename(midpoint_adjusted_estimate = estimate)
        if (nrow(midpoint_adjusted_main) > 0) {
          compare_main <- compare_main %>%
            left_join(midpoint_adjusted_main %>% select(all_of(c(key_main, "midpoint_adjusted_estimate"))), by = key_main) %>%
            mutate(midpoint_adjusted_minus_derived = clock_diff_shortest(midpoint_adjusted_estimate, derived_midpoint_linear)) %>%
            compact_tibble()
        }
      }
      if (nrow(compare_age) > 0) {
        midpoint_adjusted_age <- safe_predictions(
          model = midpoint_adjusted_model,
          grid_args = grid_age,
          keep_cols = key_age,
          analysis_name = "midpoint_age_adjusted"
        ) %>%
          rename(midpoint_adjusted_estimate = estimate)
        if (nrow(midpoint_adjusted_age) > 0) {
          compare_age <- compare_age %>%
            left_join(midpoint_adjusted_age %>% select(all_of(c(key_age, "midpoint_adjusted_estimate"))), by = key_age) %>%
            mutate(midpoint_adjusted_minus_derived = clock_diff_shortest(midpoint_adjusted_estimate, derived_midpoint_linear)) %>%
            compact_tibble()
        }
      }
    }
    rm_if_exists(midpoint_adjusted_main, midpoint_adjusted_age, midpoint_adjusted_model)
    run_gc(paste("midpoint adjusted compare cleanup for", b))
    log_memory(paste("after_midpoint_adjusted_compare", b))
  }

  if (nrow(compare_main) > 0) append_checkpoint(compare_main, derived_main_compare_csv)
  if (nrow(compare_age) > 0) append_checkpoint(compare_age, derived_age_compare_csv)

  rm_if_exists(compare_main, compare_age)
  run_gc(paste("derived section finished for", b))
  log_memory(paste("after_derived_batch", b))
  mark_checkpoint_complete("derived", b)
}

derived_main_tbl <- read_checkpoint(derived_main_csv)
derived_main_compare_tbl <- read_checkpoint(derived_main_compare_csv)
derived_age_tbl <- read_checkpoint(derived_age_csv)
derived_age_compare_tbl <- read_checkpoint(derived_age_compare_csv)

write_table(derived_main_tbl, "derived_duration_midpoint_main_grid.csv")
write_table(derived_main_compare_tbl, "derived_vs_direct_main_grid.csv")

write_table(derived_age_tbl, "derived_duration_midpoint_age.csv")
write_table(derived_age_compare_tbl, "derived_vs_direct_age.csv")

# %% [markdown]
# ## Section 3: Contrast Summaries

# %%
base_keys <- c("outcome", "outcome_type", "outcome_variant", "batch", "model_method", "model_file")

weekend_overall <- predictions_all %>%
  filter(analysis == "weekend_main") %>%
  compute_two_level_difference(
    group_cols = base_keys,
    level_col = "is_weekend_factor",
    normalize_fn = normalize_weekend_level,
    high_level = "Weekend",
    low_level = "Weekday",
    diff_col = "weekend_minus_weekday"
  ) %>%
  mutate(contrast_scope = "overall")

weekend_by_employment <- predictions_all %>%
  filter(analysis == "employment_x_weekend") %>%
  compute_two_level_difference(
    group_cols = c(base_keys, "employment_status"),
    level_col = "is_weekend_factor",
    normalize_fn = normalize_weekend_level,
    high_level = "Weekend",
    low_level = "Weekday",
    diff_col = "weekend_minus_weekday"
  ) %>%
  mutate(contrast_scope = "by_employment")

weekend_contrasts <- bind_rows(weekend_overall, weekend_by_employment)
write_table(weekend_contrasts, "weekend_contrasts.csv")

dst_overall <- predictions_all %>%
  filter(analysis == "dst_main") %>%
  compute_two_level_difference(
    group_cols = base_keys,
    level_col = "dst_observes",
    normalize_fn = normalize_dst_level,
    high_level = "NoDST",
    low_level = "DST",
    diff_col = "nodst_minus_dst"
  ) %>%
  mutate(contrast_scope = "overall")

dst_by_weekend <- predictions_all %>%
  filter(analysis == "dst_x_weekend") %>%
  compute_two_level_difference(
    group_cols = c(base_keys, "is_weekend_factor"),
    level_col = "dst_observes",
    normalize_fn = normalize_dst_level,
    high_level = "NoDST",
    low_level = "DST",
    diff_col = "nodst_minus_dst"
  ) %>%
  mutate(contrast_scope = "by_weekend")

dst_by_month <- predictions_all %>%
  filter(analysis == "month_x_dst") %>%
  compute_two_level_difference(
    group_cols = c(base_keys, "month"),
    level_col = "dst_observes",
    normalize_fn = normalize_dst_level,
    high_level = "NoDST",
    low_level = "DST",
    diff_col = "nodst_minus_dst"
  ) %>%
  mutate(contrast_scope = "by_month")

dst_contrasts <- bind_rows(dst_overall, dst_by_weekend, dst_by_month)
write_table(dst_contrasts, "dst_contrasts.csv")

# %% [markdown]
# ## Final Export

# %%
writeLines(analysis_log, con = file.path(OUTPUT_DIR, "run_log.txt"))
log_msg("Done.")
log_msg(paste("Tables:", TABLE_DIR))
