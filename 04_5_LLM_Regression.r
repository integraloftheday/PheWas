# 04_5_LLM_Regression.r
# Mixed-effects regression with:
# - polynomial age base model
# - weekend interaction terms
# - age x employment interaction
# - month-based seasonality controls
# - cross-adjustment between duration and midpoint outcomes

library(arrow)
library(dplyr)
library(lme4)
library(lmerTest)

set.seed(123)

# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------
TEST_MODE <- FALSE
N_TEST_IDS <- 25
FIT_ALL_REML <- TRUE

INPUT_PARQUET <- "processed_data/LMM_analysis.parquet"
MODEL_DIR <- "models_04_5"
SUMMARY_DIR <- "model_summaries_04_5"
AIC_REPORT_FILE <- "model_comparison_aic_04_5.md"

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

# -------------------------------------------------------------------
# Load and prepare data
# -------------------------------------------------------------------
df_clean <- arrow::read_parquet(INPUT_PARQUET)

if (TEST_MODE) {
  message("TEST MODE: subsetting by person_id")
  target_ids <- sample(unique(df_clean$person_id), N_TEST_IDS)
  df_clean <- df_clean %>% filter(person_id %in% target_ids)
} else {
  message("FULL MODE: using all rows")
}

if (!"zip3" %in% names(df_clean)) {
  warning("zip3 not found in LMM_analysis.parquet; DST models will be skipped until 02_Data_Preparation is rerun.")
  df_clean$zip3 <- NA_character_
}

df_clean <- df_clean %>%
  mutate(
    duration_hours = daily_sleep_window_mins / 60,
    sex_concept = as.factor(sex_concept),
    employment_status = as.factor(employment_status),
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
    dst_observes = factor(dst_observes, levels = c("DST", "NoDST"))
  )

message("Seasonality: using month factor only")

cols_needed <- c(
  "person_id",
  "onset_linear", "offset_linear", "midpoint_linear", "duration_hours",
  "age_at_sleep", "sex_concept", "employment_status", "is_weekend_factor",
  "month", "dst_observes"
)

df_clean <- df_clean[, intersect(names(df_clean), cols_needed)]
gc()

# -------------------------------------------------------------------
# Model fitting helper
# -------------------------------------------------------------------
fit_save_summarize <- function(formula_obj, model_name, description, df, run_reml = FALSE) {
  message(paste0("Processing: ", model_name, " (", description, ")"))

  current_aic <- NA_real_
  current_bic <- NA_real_
  status_msg <- "Success"

  fit_ml_result <- tryCatch({
    fit_obj <- lmer(
      formula_obj,
      data = df,
      REML = FALSE,
      control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
    )

    saveRDS(fit_obj, file.path(MODEL_DIR, paste0(model_name, "_ML.rds")), compress = "xz")

    sink(file.path(SUMMARY_DIR, paste0(model_name, "_ML_summary.txt")))
    print(paste("Model:", model_name))
    print("Method: ML (for model comparison)")
    print(formula_obj)
    print(summary(fit_obj))
    sink()

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
  current_aic <- AIC(fit_ml)
  current_bic <- BIC(fit_ml)

  if (run_reml) {
    rm(fit_ml)
    gc()

    tryCatch({
      fit_reml <- lmer(
        formula_obj,
        data = df,
        REML = TRUE,
        control = lmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
      )

      saveRDS(fit_reml, file.path(MODEL_DIR, paste0(model_name, "_REML.rds")), compress = "xz")

      sink(file.path(SUMMARY_DIR, paste0(model_name, "_REML_summary.txt")))
      print(paste("Model:", model_name))
      print("Method: REML (for coefficient interpretation)")
      print(formula_obj)
      print(summary(fit_reml))
      sink()

      rm(fit_reml)
    }, error = function(e) {
      if (sink.number() > 0) sink()
      message(paste("REML fit failed:", e$message))
      status_msg <<- paste("Partial Success (REML Failed:", e$message, ")")
    })
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
build_rhs <- function(outcome_name, include_dst = FALSE) {
  rhs_terms <- c(
    # Base model: polynomial age + interaction structure
    "poly(age_at_sleep, 2)",
    "employment_status",
    "is_weekend_factor",
    "poly(age_at_sleep, 2):employment_status",  # age x employment interaction
    "poly(age_at_sleep, 2):is_weekend_factor",  # weekend interaction with age
    "is_weekend_factor:employment_status",      # weekend interaction with employment
    "sex_concept",
    "month"                                     # month-based seasonality
  )

  if (include_dst) {
    rhs_terms <- c(
      rhs_terms,
      "dst_observes",
      "is_weekend_factor:dst_observes"
    )
  }

  # Requested cross-adjustment
  if (outcome_name == "duration_hours") {
    # Additive adjustment for midpoint (not an interaction term)
    rhs_terms <- c(rhs_terms, "midpoint_linear")
  }
  if (outcome_name == "midpoint_linear") {
    # Additive adjustment for duration (not an interaction term)
    rhs_terms <- c(rhs_terms, "duration_hours")
  }

  paste(c(rhs_terms, "(1 | person_id)"), collapse = " + ")
}

build_formula <- function(outcome_name, include_dst = FALSE) {
  as.formula(paste(outcome_name, "~", build_rhs(outcome_name, include_dst = include_dst)))
}

# -------------------------------------------------------------------
# Run models
# -------------------------------------------------------------------
model_specs <- list(
  list(name = "onset_poly_interact_04_5", outcome = "onset_linear"),
  list(name = "offset_poly_interact_04_5", outcome = "offset_linear"),
  list(name = "midpoint_poly_interact_04_5", outcome = "midpoint_linear"),
  list(name = "duration_poly_interact_04_5", outcome = "duration_hours")
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
  fml <- build_formula(m$outcome, include_dst = FALSE)
  desc <- "Poly age + weekend/employment interactions + month seasonality"

  res <- fit_save_summarize(
    formula_obj = fml,
    model_name = m$name,
    description = desc,
    df = df_clean,
    run_reml = FIT_ALL_REML
  )
  model_performance <- rbind(model_performance, res)
}

# -------------------------------------------------------------------
# Run second training batch with DST encoding
# -------------------------------------------------------------------
df_dst <- df_clean %>% filter(!is.na(dst_observes))
dst_levels_present <- n_distinct(df_dst$dst_observes)

if (nrow(df_dst) > 0 && dst_levels_present >= 2) {
  message("Running DST-encoded model batch...")
  for (m in model_specs) {
    fml <- build_formula(m$outcome, include_dst = TRUE)
    desc <- "Poly age + weekend/employment interactions + month seasonality + DST encoding"

    res <- fit_save_summarize(
      formula_obj = fml,
      model_name = paste0(m$name, "_dst"),
      description = desc,
      df = df_dst,
      run_reml = FIT_ALL_REML
    )
    model_performance <- rbind(model_performance, res)
  }
} else {
  message("Skipping DST-encoded batch: insufficient non-missing/variant DST data.")
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
