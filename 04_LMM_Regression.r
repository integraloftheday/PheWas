# 04_5_LLM_Regression.r
# Mixed-effects regression with:
# - spline age base model (df=3)
# - weekend interaction terms
# - reduced-rank age x employment interaction
# - month-based seasonality controls
# - deviation environmental control (PhotoPeriod excluded by design)
# - optional DST DiD-style event encoding
# - cross-adjustment between duration and midpoint outcomes

library(arrow)
library(dplyr)
library(lme4)
library(lmerTest)
if (requireNamespace("emmeans", quietly = TRUE)) {
  suppressPackageStartupMessages(library(emmeans))
} else {
  message("Package 'emmeans' not installed; EMMeans export will be skipped.")
}

set.seed(123)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
TEST_MODE <- tolower(Sys.getenv("TEST_MODE_04", "false")) %in% c("true", "1", "yes")
N_TEST_IDS <- as.integer(Sys.getenv("N_TEST_IDS_04", "25"))
FIT_ALL_REML <- tolower(Sys.getenv("FIT_ALL_REML_04", "true")) %in% c("true", "1", "yes")
# Restrict DST refits to selected outcomes only.
# Valid values: "onset", "offset", "midpoint", "duration"
DST_TARGET_OUTCOMES <- c("onset", "offset", "midpoint", "duration")
# Event window (days) used to encode post-transition indicators when available.
DST_EVENT_WINDOW_DAYS <- as.integer(Sys.getenv("DST_EVENT_WINDOW_DAYS_04", "14"))
# Resume-friendly behavior: skip fitting if target .rds already exists.
SKIP_EXISTING_MODELS <- tolower(Sys.getenv("SKIP_EXISTING_MODELS_04", "true")) %in% c("true", "1", "yes")

INPUT_PARQUET <- Sys.getenv("INPUT_PARQUET_04", "processed_data/LMM_analysis.parquet")
MODEL_DIR <- Sys.getenv("MODEL_DIR_04", "models_04")
SUMMARY_DIR <- Sys.getenv("SUMMARY_DIR_04", "model_summaries_04")
AIC_REPORT_FILE <- Sys.getenv("AIC_REPORT_FILE_04", "model_comparison_aic_04.md")

if (!dir.exists(MODEL_DIR)) dir.create(MODEL_DIR)
if (!dir.exists(SUMMARY_DIR)) dir.create(SUMMARY_DIR)

default_no_dst_zip3 <- unique(c(
  sprintf("%03d", 850:865),  # Arizona (most ZIP3; some tribal areas differ)
  "967", "968",              # Hawaii
  "006", "007", "008", "009",# Puerto Rico + USVI
  "969"                      # Guam / CNMI
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

collapse_employment_status <- function(x) {
  x_chr <- trimws(as.character(x))
  out <- rep(NA_character_, length(x_chr))

  out[x_chr %in% c("Employed For Wages", "Self Employed")] <- "Working"
  out[x_chr %in% c("Student")] <- "Student"
  out[x_chr %in% c("Retired")] <- "Retired"
  out[x_chr %in% c(
    "Homemaker",
    "Unable To Work",
    "Out Of Work Less Than One",
    "Out Of Work One Or More"
  )] <- "Not Working"

  out
}

get_dst_dates <- function(year) {
  # Spring: 2nd Sunday in March, Fall: 1st Sunday in November
  m_start <- as.Date(sprintf("%d-03-01", year))
  m_wday <- as.POSIXlt(m_start)$wday
  first_sun_march <- m_start + ((7 - m_wday) %% 7)
  spring_dst <- first_sun_march + 7

  n_start <- as.Date(sprintf("%d-11-01", year))
  n_wday <- as.POSIXlt(n_start)$wday
  fall_dst <- n_start + ((7 - n_wday) %% 7)

  list(spring = spring_dst, fall = fall_dst)
}

# -------------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------------
df_clean <- arrow::read_parquet(INPUT_PARQUET)

duration_source_col <- if ("daily_duration_mins" %in% names(df_clean)) {
  "daily_duration_mins"
} else if ("daily_sleep_window_mins" %in% names(df_clean)) {
  "daily_sleep_window_mins"
} else {
  NA_character_
}

if (is.na(duration_source_col)) {
  stop("Input parquet must contain either 'daily_duration_mins' or 'daily_sleep_window_mins'.")
}

if (TEST_MODE) {
  message("TEST MODE: subsetting by person_id")
  target_ids <- sample(unique(df_clean$person_id), N_TEST_IDS)
  df_clean <- df_clean %>% filter(person_id %in% target_ids)
} else {
  message("FULL MODE: using all rows")
}

if (!"zip3" %in% names(df_clean)) {
  if ("zip_code" %in% names(df_clean)) {
    df_clean$zip3 <- substr(as.character(df_clean$zip_code), 1, 3)
  } else {
    warning("zip3/zip_code not found; DST models will be skipped.")
    df_clean$zip3 <- NA_character_
  }
}

if (!"sleep_date" %in% names(df_clean)) {
  warning("sleep_date not found; DST event-time terms will only be used if precomputed days_to_spring/days_to_fall exist.")
}

if (!"days_to_spring" %in% names(df_clean) && !"days_to_fall" %in% names(df_clean) && "sleep_date" %in% names(df_clean)) {
  sleep_dates <- as.Date(df_clean$sleep_date)
  valid_idx <- which(!is.na(sleep_dates))
  if (length(valid_idx) > 0) {
    years <- as.integer(format(sleep_dates, "%Y"))
    unique_years <- sort(unique(years[!is.na(years)]))
    dst_lookup <- lapply(unique_years, get_dst_dates)
    names(dst_lookup) <- as.character(unique_years)

    spring_dates <- as.Date(rep(NA_character_, length(sleep_dates)))
    fall_dates <- as.Date(rep(NA_character_, length(sleep_dates)))
    for (yy in unique_years) {
      idx <- which(years == yy)
      spring_dates[idx] <- dst_lookup[[as.character(yy)]]$spring
      fall_dates[idx] <- dst_lookup[[as.character(yy)]]$fall
    }

    df_clean$days_to_spring <- as.numeric(sleep_dates - spring_dates)
    df_clean$days_to_fall <- as.numeric(sleep_dates - fall_dates)
  }
}

df_clean <- df_clean %>%
  mutate(
    duration_hours = .data[[duration_source_col]] / 60,
    sex_concept = as.factor(sex_concept),
    employment_status_original = employment_status,
    employment_status = collapse_employment_status(employment_status),
    employment_status = factor(
      employment_status,
      levels = c("Working", "Student", "Retired", "Not Working")
    ),
    month = as.factor(month),
    is_weekend_factor = factor(
      ifelse(is_weekend %in% c(TRUE, 1), "Weekend", "Weekday"),
      levels = c("Weekday", "Weekend")
    ),
    zip3_norm = normalize_zip3(zip3),
    dst_observes = case_when(
      is.na(zip3_norm) ~ NA_character_,
      zip3_norm %in% default_no_dst_zip3 ~ "NoDST",
      TRUE ~ "DST"
    ),
    dst_observes = factor(dst_observes, levels = c("DST", "NoDST")),
    post_spring = if ("days_to_spring" %in% names(df_clean)) {
      as.integer(days_to_spring >= 0 & days_to_spring <= DST_EVENT_WINDOW_DAYS)
    } else {
      NA_integer_
    },
    post_fall = if ("days_to_fall" %in% names(df_clean)) {
      as.integer(days_to_fall >= 0 & days_to_fall <= DST_EVENT_WINDOW_DAYS)
    } else {
      NA_integer_
    }
  )

age_center <- suppressWarnings(mean(df_clean$age_at_sleep, na.rm = TRUE))
if (!is.finite(age_center)) age_center <- 0

df_clean <- df_clean %>%
  mutate(age_centered = age_at_sleep - age_center)

message("Seasonality: using month factor only")

has_deviation <- "deviation" %in% names(df_clean) && any(!is.na(df_clean$deviation))
has_dst_observes <- "dst_observes" %in% names(df_clean) && n_distinct(na.omit(df_clean$dst_observes)) >= 2
has_dst_event <- all(c("post_spring", "post_fall") %in% names(df_clean)) &&
  any(!is.na(df_clean$post_spring)) && any(!is.na(df_clean$post_fall))

if (!has_deviation) message("deviation not available/non-varying; excluding from model RHS.")
if (!has_dst_observes) message("dst_observes not available/non-varying; excluding DST terms.")
if (!has_dst_event) message("DST event indicators not available; excluding post_spring/post_fall DiD terms.")

cols_needed <- c(
  "person_id",
  "onset_linear", "offset_linear", "midpoint_linear", "duration_hours",
  "age_at_sleep", "age_centered", "sex_concept", "employment_status", "is_weekend_factor",
  "month", "dst_observes", "deviation", "post_spring", "post_fall"
)

df_clean <- df_clean[, intersect(names(df_clean), cols_needed)]
gc()

# -------------------------------------------------------------------
# Model fitting helper
# -------------------------------------------------------------------
fit_save_summarize <- function(formula_obj, model_name, description, df, run_reml = FALSE) {
  message(paste0("Processing: ", model_name, " (", description, ")"))

  ml_path <- file.path(MODEL_DIR, paste0(model_name, "_ML.rds"))
  reml_path <- file.path(MODEL_DIR, paste0(model_name, "_REML.rds"))

  current_aic <- NA_real_
  current_bic <- NA_real_
  status_msg <- "Success"
  fit_ml <- NULL

  compute_vif_table <- function(fit_obj, method_label) {
    out_path <- file.path(SUMMARY_DIR, paste0(model_name, "_", method_label, "_vif.csv"))
    tryCatch({
      x <- tryCatch(lme4::getME(fit_obj, "X"), error = function(e) NULL)
      if (is.null(x)) {
        write.csv(data.frame(term = character(), vif = numeric()), out_path, row.names = FALSE)
        return(invisible(NULL))
      }

      cn <- colnames(x)
      if (is.null(cn) || length(cn) == 0) {
        write.csv(data.frame(term = character(), vif = numeric()), out_path, row.names = FALSE)
        return(invisible(NULL))
      }

      keep_idx <- which(!grepl("^\\(Intercept\\)$", cn))
      if (length(keep_idx) == 0) {
        write.csv(data.frame(term = character(), vif = numeric()), out_path, row.names = FALSE)
        return(invisible(NULL))
      }

      x_use <- x[, keep_idx, drop = FALSE]
      cn_use <- colnames(x_use)
      if (ncol(x_use) == 1) {
        vif_tbl <- data.frame(term = cn_use, vif = 1, stringsAsFactors = FALSE)
        write.csv(vif_tbl, out_path, row.names = FALSE)
        return(invisible(NULL))
      }

      vif_vals <- vapply(seq_len(ncol(x_use)), function(j) {
        y <- x_use[, j]
        others <- x_use[, -j, drop = FALSE]
        if (ncol(others) == 0) return(1)

        fit_aux <- tryCatch(stats::lm.fit(x = cbind(1, others), y = y), error = function(e) NULL)
        if (is.null(fit_aux)) return(NA_real_)

        rss <- sum(fit_aux$residuals^2)
        tss <- sum((y - mean(y))^2)
        if (!is.finite(tss) || tss <= 0) return(NA_real_)
        r2 <- 1 - (rss / tss)
        if (!is.finite(r2) || r2 >= 1) return(Inf)
        if (r2 < 0) r2 <- 0
        1 / (1 - r2)
      }, FUN.VALUE = numeric(1))

      vif_tbl <- data.frame(term = cn_use, vif = as.numeric(vif_vals), stringsAsFactors = FALSE)
      write.csv(vif_tbl, out_path, row.names = FALSE)
      invisible(NULL)
    }, error = function(e) {
      warning(paste("VIF export failed for", model_name, method_label, ":", e$message))
      invisible(NULL)
    })
  }

  export_emmeans_table <- function(fit_obj, method_label) {
    if (!requireNamespace("emmeans", quietly = TRUE)) return(invisible(NULL))

    fit_vars <- tryCatch(names(model.frame(fit_obj)), error = function(e) character())

    make_numeric_grid <- function(x, n = 21L) {
      xv <- suppressWarnings(as.numeric(x))
      xv <- xv[is.finite(xv)]
      if (length(xv) < 2) return(NULL)
      probs <- seq(0.05, 0.95, length.out = as.integer(n))
      vals <- suppressWarnings(as.numeric(stats::quantile(xv, probs = probs, na.rm = TRUE, names = FALSE)))
      vals <- unique(vals[is.finite(vals)])
      if (length(vals) < 2) {
        vals <- sort(unique(c(min(xv, na.rm = TRUE), max(xv, na.rm = TRUE))))
      }
      vals
    }

    age_grid <- seq(
      floor(suppressWarnings(min(df$age_at_sleep, na.rm = TRUE))),
      ceiling(suppressWarnings(max(df$age_at_sleep, na.rm = TRUE))),
      by = 1
    )
    age_grid <- age_grid[is.finite(age_grid)]
    if (length(age_grid) == 0) age_grid <- 18:85

    deviation_grid <- if ("deviation" %in% fit_vars && "deviation" %in% names(df)) make_numeric_grid(df$deviation, n = 21L) else NULL
    duration_grid <- if ("duration_hours" %in% fit_vars && "duration_hours" %in% names(df)) make_numeric_grid(df$duration_hours, n = 21L) else NULL

    safe_emm <- function(specs, analysis_name, at = NULL) {
      out <- tryCatch({
        as.data.frame(emmeans::emmeans(
          fit_obj,
          specs = specs,
          weights = "proportional",
          at = at,
          data = df
        ))
      }, error = function(e) {
        message(paste("EMMeans failed for", model_name, method_label, analysis_name, ":", e$message))
        NULL
      })
      if (is.null(out) || nrow(out) == 0) return(NULL)
      out$analysis <- analysis_name
      out
    }

    emm_parts <- list(
      safe_emm(~ employment_status * is_weekend_factor, "emm_employment_x_weekend"),
      safe_emm(~ employment_status, "emm_employment_main"),
      safe_emm(~ is_weekend_factor, "emm_weekend_main"),
      safe_emm(~ sex_concept, "emm_sex_main"),
      safe_emm(~ month, "emm_month_main"),
      safe_emm(~ month * is_weekend_factor, "emm_month_x_weekend"),
      safe_emm(~ age_at_sleep, "emm_age_main", at = list(age_at_sleep = age_grid)),
      safe_emm(~ age_at_sleep * employment_status, "emm_age_x_employment", at = list(age_at_sleep = age_grid))
    )

    if (!is.null(deviation_grid) && length(deviation_grid) >= 2) {
      emm_parts <- append(emm_parts, list(
        safe_emm(~ deviation, "emm_deviation_main", at = list(deviation = deviation_grid))
      ))
    }

    if (!is.null(duration_grid) && length(duration_grid) >= 2) {
      emm_parts <- append(emm_parts, list(
        safe_emm(~ duration_hours, "emm_duration_covariate_main", at = list(duration_hours = duration_grid))
      ))
    }

    emm_parts <- emm_parts[!vapply(emm_parts, is.null, logical(1))]
    if (length(emm_parts) == 0) return(invisible(NULL))

    emm_tbl <- bind_rows(emm_parts)
    names(emm_tbl)[names(emm_tbl) == "emmean"] <- "estimate"
    names(emm_tbl)[names(emm_tbl) == "lower.CL"] <- "conf.low"
    names(emm_tbl)[names(emm_tbl) == "upper.CL"] <- "conf.high"

    keep <- c(
      "analysis", "employment_status", "is_weekend_factor", "sex_concept", "month", "age_at_sleep", "deviation", "duration_hours",
      "estimate", "SE", "df", "conf.low", "conf.high", "t.ratio", "p.value"
    )
    emm_tbl <- emm_tbl[, intersect(keep, names(emm_tbl)), drop = FALSE]

    out_path <- file.path(SUMMARY_DIR, paste0(model_name, "_", method_label, "_emmeans.csv"))
    write.csv(emm_tbl, out_path, row.names = FALSE)
    invisible(NULL)
  }

  if (SKIP_EXISTING_MODELS && file.exists(ml_path)) {
    message(paste("ML exists, skipping fit:", ml_path))
    fit_ml <- tryCatch(readRDS(ml_path), error = function(e) NULL)
    if (is.null(fit_ml)) {
      message("Existing ML model could not be read. Re-fitting ML.")
    } else {
      status_msg <- "Success (ML reused)"
    }
  }

  if (is.null(fit_ml)) {
    fit_ml_result <- tryCatch({
      fit_obj <- lmer(
        formula_obj,
        data = df,
        REML = FALSE,
        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      )

      saveRDS(fit_obj, ml_path, compress = "xz")

      sink(file.path(SUMMARY_DIR, paste0(model_name, "_ML_summary.txt")))
      print(paste("Model:", model_name))
      print("Method: ML (for model comparison)")
      print(formula_obj)
      print(summary(fit_obj))
      sink()

      compute_vif_table(fit_obj, "ML")
      export_emmeans_table(fit_obj, "ML")

      list(success = TRUE, fit = fit_obj)
    }, error = function(e) {
      if (sink.number() > 0) sink()
      message(paste("ML fit failed:", e$message))
      list(success = FALSE, error = e$message)
    })

    if (!fit_ml_result$success) {
      gc()
      return(data.frame(
        Model_Name = model_name,
        Formula_Description = description,
        Formula = deparse(formula_obj),
        AIC = NA_real_,
        BIC = NA_real_,
        Status = paste("FAILED (ML):", fit_ml_result$error),
        stringsAsFactors = FALSE
      ))
    }
    fit_ml <- fit_ml_result$fit
  }

  current_aic <- AIC(fit_ml)
  current_bic <- BIC(fit_ml)

  if (run_reml) {
    rm(fit_ml)
    gc()

    if (SKIP_EXISTING_MODELS && file.exists(reml_path)) {
      message(paste("REML exists, skipping fit:", reml_path))
      if (status_msg == "Success") {
        status_msg <- "Success (REML reused)"
      } else {
        status_msg <- paste(status_msg, "+ REML reused")
      }
    } else {
      tryCatch({
        fit_reml <- lmer(
          formula_obj,
          data = df,
          REML = TRUE,
          control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
        )

        saveRDS(fit_reml, reml_path, compress = "xz")

        sink(file.path(SUMMARY_DIR, paste0(model_name, "_REML_summary.txt")))
        print(paste("Model:", model_name))
        print("Method: REML (for coefficient interpretation)")
        print(formula_obj)
        print(summary(fit_reml))
        sink()

        compute_vif_table(fit_reml, "REML")
        export_emmeans_table(fit_reml, "REML")

        rm(fit_reml)
      }, error = function(e) {
        if (sink.number() > 0) sink()
        message(paste("REML fit failed:", e$message))
        status_msg <<- paste("Partial Success (REML Failed:", e$message, ")")
      })
    }
  } else {
    rm(fit_ml)
  }

  gc()

  data.frame(
    Model_Name = model_name,
    Formula_Description = description,
    Formula = deparse(formula_obj),
    AIC = current_aic,
    BIC = current_bic,
    Status = status_msg,
    stringsAsFactors = FALSE
  )
}

# -------------------------------------------------------------------
# Formula builder
# -------------------------------------------------------------------
build_rhs <- function(outcome_name, include_dst = FALSE, adjusted_variant = FALSE) {
  rhs_terms <- c(
    # Base model: spline age + interaction structure
    "splines::ns(age_at_sleep, df = 3)",
    "employment_status",
    "is_weekend_factor",
    "age_centered:employment_status",            # reduced-rank age x employment interaction
    "splines::ns(age_at_sleep, df = 3):is_weekend_factor",  # weekend interaction with age
    "is_weekend_factor:employment_status",      # weekend interaction with employment
    "sex_concept",
    "month"                                     # month-based seasonality
  )

  if (has_deviation) {
    rhs_terms <- c(rhs_terms, "deviation")
  }

  if (include_dst && has_dst_observes) {
    rhs_terms <- c(
      rhs_terms,
      "dst_observes",
      "is_weekend_factor:dst_observes"
    )
  }

  if (include_dst && has_dst_observes && has_dst_event) {
    rhs_terms <- c(
      rhs_terms,
      "post_spring",
      "post_fall",
      "post_spring:dst_observes",
      "post_fall:dst_observes"
    )
  }

  # Optional cross-adjustment sensitivity models
  if (adjusted_variant && outcome_name == "duration_hours") {
    # Additive adjustment for midpoint (not an interaction term)
    rhs_terms <- c(rhs_terms, "midpoint_linear")
  }
  if (adjusted_variant && outcome_name == "midpoint_linear") {
    # Additive adjustment for duration (not an interaction term)
    rhs_terms <- c(rhs_terms, "duration_hours")
  }

  paste(c(rhs_terms, "(1 | person_id)"), collapse = " + ")
}

build_formula <- function(outcome_name, include_dst = FALSE, adjusted_variant = FALSE) {
  as.formula(paste(
    outcome_name,
    "~",
    build_rhs(
      outcome_name,
      include_dst = include_dst,
      adjusted_variant = adjusted_variant
    )
  ))
}

# -------------------------------------------------------------------
# Run models
# -------------------------------------------------------------------
model_specs <- list(
  list(
    key = "onset",
    name = "onset_poly_interact_04",
    outcome = "onset_linear",
    adjusted_variant = FALSE,
    description = "High-level: spline age(df=3) + reduced age-employment deviation + weekend/employment interactions + month factor + optional environment + DST DiD terms"
  ),
  list(
    key = "offset",
    name = "offset_poly_interact_04",
    outcome = "offset_linear",
    adjusted_variant = FALSE,
    description = "High-level: spline age(df=3) + reduced age-employment deviation + weekend/employment interactions + month factor + optional environment + DST DiD terms"
  ),
  list(
    key = "midpoint",
    name = "midpoint_poly_interact_04",
    outcome = "midpoint_linear",
    adjusted_variant = FALSE,
    description = "High-level: spline age(df=3) + reduced age-employment deviation + weekend/employment interactions + month factor + optional environment + DST DiD terms"
  ),
  list(
    key = "duration",
    name = "duration_poly_interact_04",
    outcome = "duration_hours",
    adjusted_variant = FALSE,
    description = "High-level: spline age(df=3) + reduced age-employment deviation + weekend/employment interactions + month factor + optional environment + DST DiD terms"
  ),
  list(
    key = "midpoint",
    name = "midpoint_poly_interact_duration_adjusted_04",
    outcome = "midpoint_linear",
    adjusted_variant = TRUE,
    description = "High-level: spline age(df=3) + reduced age-employment deviation + weekend/employment interactions + month factor + optional environment + DST DiD terms + duration adjustment"
  ),
  list(
    key = "duration",
    name = "duration_poly_interact_midpoint_adjusted_04",
    outcome = "duration_hours",
    adjusted_variant = TRUE,
    description = "High-level: spline age(df=3) + reduced age-employment deviation + weekend/employment interactions + month factor + optional environment + DST DiD terms + midpoint adjustment"
  )
)

model_performance <- data.frame(
  Model_Name = character(),
  Formula_Description = character(),
  Formula = character(),
  AIC = numeric(),
  BIC = numeric(),
  Status = character(),
  stringsAsFactors = FALSE
)

for (m in model_specs) {
  # Base variant (no explicit DST terms)
  fml_base <- build_formula(
    m$outcome,
    include_dst = FALSE,
    adjusted_variant = isTRUE(m$adjusted_variant)
  )

  res_base <- fit_save_summarize(
    formula_obj = fml_base,
    model_name = m$name,
    description = paste0(m$description, " [base: no DST terms]"),
    df = df_clean,
    run_reml = FIT_ALL_REML
  )
  model_performance <- rbind(model_performance, res_base)

  # DST-enhanced variant (explicit DST terms), restricted by configured outcomes
  if (m$key %in% DST_TARGET_OUTCOMES && has_dst_observes) {
    fml_dst <- build_formula(
      m$outcome,
      include_dst = TRUE,
      adjusted_variant = isTRUE(m$adjusted_variant)
    )

    res_dst <- fit_save_summarize(
      formula_obj = fml_dst,
      model_name = paste0(m$name, "_dst"),
      description = paste0(m$description, " [dst: includes DST terms]"),
      df = df_clean,
      run_reml = FIT_ALL_REML
    )
    model_performance <- rbind(model_performance, res_dst)
  }
}

# -------------------------------------------------------------------
# Export comparison table
# -------------------------------------------------------------------
model_performance <- model_performance[order(model_performance$AIC, na.last = TRUE), ]

md_header <- paste(
  "| Model Name | Description | AIC | BIC | Status |",
  "|---|---|---|---|---|",
  sep = "\n"
)

md_rows <- apply(model_performance, 1, function(x) {
  aic_disp <- if (is.na(x["AIC"])) "NA" else round(as.numeric(x["AIC"]), 2)
  bic_disp <- if (is.na(x["BIC"])) "NA" else round(as.numeric(x["BIC"]), 2)
  paste0(
    "| ", x["Model_Name"], " | ", x["Formula_Description"], " | ",
    aic_disp, " | ", bic_disp, " | ", x["Status"], " |"
  )
})

md_content <- paste(md_header, paste(md_rows, collapse = "\n"), sep = "\n")
writeLines(md_content, AIC_REPORT_FILE)

message("Done")
message(paste("AIC report:", AIC_REPORT_FILE))
message(paste("Model binaries:", MODEL_DIR))
message(paste("Model summaries:", SUMMARY_DIR))
