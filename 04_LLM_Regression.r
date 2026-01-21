# Load necessary libraries
library(arrow)
library(tidyverse)
library(lme4)
library(lmerTest) # For p-values
library(dplyr)
library(polars)
library(reticulate)
library(ggplot2)
library(patchwork)
library(broom.mixed)
library(ggeffects)

# Set seed for reproducibility
set.seed(123)


TEST_MODE <- FALSE  # Set to FALSE only when ready for the full 20M row run
MODEL_DIR <- "models"

# Load Data
# Load the processed data
df_clean <- arrow::read_parquet("processed_data/LMM_analysis.parquet")

if (TEST_MODE) {
  message("⚠️ TEST MODE ON: Subsetting to 1,000 users for rapid testing...")
  set.seed(123)
  
  # Sample unique person_ids to preserve the (1|person_id) structure
  target_ids <- sample(unique(df_clean$person_id), 10)
  
  # Filter directly into df_clean to save memory
  df_clean <- df_clean %>% 
    filter(person_id %in% target_ids)
    
  # Force garbage collection to free the memory of the un-subsetted data
  gc()
} else {
  message("✅ FULL MODE: Using entire dataset.")
}

df_clean <- df_clean %>%
  mutate(
    # 'month' might be string "01", "02", make it a factor
    duration_hours = daily_sleep_window_mins / 60,
    month = as.factor(month),
    # 'sex_concept' might be string, make it factor
    sex_concept = as.factor(sex_concept),
    # 'employment_status' - ensure it's a factor
    employment_status = as.factor(employment_status)
  )

# Check for missing values in employment_status
missing_emp <- sum(is.na(df_clean$employment_status))
if (missing_emp > 0) {
  message(paste("Note: Dropping", missing_emp, "rows due to missing employment_status."))
  # lmer will drop them automatically, but good to know
}# Ensure factors and CREATE DURATION IN HOURS


colnames(df_clean)

head(df_clean)

dim(df_clean)

# 2. Exploratory Plots (Only run in TEST_MODE)
if (TEST_MODE) {
  print("Generating exploratory plots...")
  
  if (!require(patchwork)) stop("Please install 'patchwork' library: install.packages('patchwork')")

  # --- JUPYTER NOTEBOOK SIZE FIX ---
  # This tells Jupyter to make the output image 16 inches wide by 6 inches tall
  options(repr.plot.width = 16, repr.plot.height = 6)

  # Prepare data 
  df_plot <- df_clean %>% 
    mutate(
      day_type = ifelse(is_weekend, "Weekend/Holiday", "Weekday"),
      day_type = factor(day_type, levels = c("Weekday", "Weekend/Holiday"))
    )

  # --- PLOT 1: Age vs. Onset ---
  # Note: base_size = 14 makes the text larger to match the wider plot
  p1 <- ggplot(df_plot, aes(x = age_at_sleep, y = onset_linear)) +
    geom_point(alpha = 0.05, size = 0.5, color = "gray50") +
    geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), 
                color = "#2E86C1", fill = "#AED6F1", size = 1.2) +
    theme_minimal(base_size = 14) + 
    labs(
      title = "Linearity Check",
      x = "Age",
      y = "Sleep Onset (Linearized)"
    ) +
    scale_y_continuous(breaks = seq(-4, 4, 1), limits = c(-6, 6))

  # --- PLOT 2: Interaction ---
  p2 <- ggplot(df_plot, aes(x = age_at_sleep, y = onset_linear, color = day_type)) +
    geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs"), size = 1.2) +
    theme_minimal(base_size = 14) +
    scale_color_manual(values = c("Weekday" = "#2E86C1", "Weekend/Holiday" = "#E74C3C")) +
    labs(
      title = "Social Jetlag Check",
      x = "Age",
      y = "", 
      color = "Day Type"
    ) +
    scale_y_continuous(breaks = seq(-4, 4, 1), limits = c(-2, 2)) +
    theme(legend.position = "bottom") 

  # --- COMBINE AND PRINT ---
  combined_plot <- p1 + p2
  print(combined_plot)
}

# Load necessary libraries
library(lme4)

# --- CONFIGURATION ---
MODEL_DIR <- "models"
SUMMARY_DIR <- "model_summaries" 
AIC_REPORT_FILE <- "model_comparison_aic.md"

# Toggle: Set to TRUE to fit the "Production" (REML) version for every single model immediately
FIT_ALL_REML <- TRUE 

# Ensure directories exist
if (!dir.exists(MODEL_DIR)) dir.create(MODEL_DIR)
if (!dir.exists(SUMMARY_DIR)) dir.create(SUMMARY_DIR)

# --- 0. OPTIMIZATION: SHRINK DATA ---
# This is the most critical step for Disk/RAM usage.
cols_needed <- c(
  "person_id", 
  "onset_linear", "offset_linear", "midpoint_linear", "duration_hours", # Outcomes
  "age_at_sleep", "sex_concept", "is_weekend", "month", "employment_status" # Predictors
)

message("🧹 Optimization: Subsetting dataframe to used columns only...")
# We assume 'df_clean' exists in your environment.
df_clean <- df_clean[, intersect(names(df_clean), cols_needed)]
gc() # Force garbage collection immediately

message(paste("✅ Dataframe shrunk. New size in RAM:", format(object.size(df_clean), units="Mb")))

# Initialize storage (Added Status column)
model_performance <- data.frame(
  Model_Name = character(),
  Formula_Description = character(),
  AIC = numeric(),
  BIC = numeric(),
  Status = character(),
  stringsAsFactors = FALSE
)

# --- HELPER FUNCTION WITH ERROR HANDLING ---
fit_save_summarize <- function(formula_obj, model_name, description, df, run_reml = FALSE) {
  
  message(paste0(">>> ⏳ Processing: ", model_name, " (", description, ")..."))
  
  # Initialize variables to NA in case of failure
  current_aic <- NA
  current_bic <- NA
  status_msg <- "Success"
  
  # --- STEP 1: ML FIT (Wrapped in tryCatch) ---
  fit_ml_result <- tryCatch({
    
    fit_obj <- lmer(formula_obj, data = df, REML = FALSE)
    
    # Save ML Binary & Summary
    saveRDS(fit_obj, file.path(MODEL_DIR, paste0(model_name, "_ML.rds")), compress = "xz")
    
    sink(file.path(SUMMARY_DIR, paste0(model_name, "_ML_summary.txt")))
    print(paste("Model:", model_name))
    print("Method: Maximum Likelihood (ML) - USE FOR AIC COMPARISON ONLY")
    print(summary(fit_obj))
    sink()
    
    # Return success list
    list(success = TRUE, fit = fit_obj)
    
  }, error = function(e) {
    # Emergency cleanup if crash happened during sink()
    if(sink.number() > 0) sink()
    message(paste("    ❌ ML Fit FAILED:", e$message))
    return(list(success = FALSE, error = e$message))
  })
  
  # Handle ML Failure
  if (!fit_ml_result$success) {
    gc()
    return(data.frame(
      Model_Name = model_name,
      Formula_Description = description,
      AIC = NA,
      BIC = NA,
      Status = paste("FAILED (ML):", fit_ml_result$error)
    ))
  }
  
  # If ML succeeded, extract stats
  fit_ml <- fit_ml_result$fit
  current_aic <- AIC(fit_ml)
  current_bic <- BIC(fit_ml)
  message(paste("    🔹 ML Fit Done. AIC:", round(current_aic, 2)))

  # --- MEMORY ESTIMATION (Only runs if ML succeeded) ---
  if (!exists("mem_check_done")) {
    obj_size_mb <- as.numeric(object.size(fit_ml)) / 1024 / 1024
    if (obj_size_mb > 500) {
      warning("⚠️  WARNING: Individual model objects are large (>500MB).")
    }
    assign("mem_check_done", TRUE, envir = .GlobalEnv)
  }

  # --- STEP 2: REML FIT (Wrapped in tryCatch) ---
  if (run_reml) {
    # Free ML memory
    rm(fit_ml) 
    gc() 
    
    tryCatch({
      fit_reml <- lmer(formula_obj, data = df, REML = TRUE)
      
      saveRDS(fit_reml, file.path(MODEL_DIR, paste0(model_name, "_REML.rds")), compress = "xz")
      
      sink(file.path(SUMMARY_DIR, paste0(model_name, "_REML_summary.txt")))
      print(paste("Model:", model_name))
      print("Method: REML - USE FOR COEFFICIENT INTERPRETATION")
      print(summary(fit_reml))
      sink()
      
      message("    🔸 REML Fit Done & Saved.")
      rm(fit_reml)
      
    }, error = function(e) {
      if(sink.number() > 0) sink()
      message(paste("    ⚠️ REML Fit FAILED (ML succeeded):", e$message))
      # We update status but don't crash, because ML stats are still valid for table
      status_msg <<- paste("Partial Success (REML Failed:", e$message, ")")
    })
  } else {
    rm(fit_ml)
  }
  
  gc()

  return(data.frame(
    Model_Name = model_name,
    Formula_Description = description,
    AIC = current_aic,
    BIC = current_bic,
    Status = status_msg
  ))
}

# --- DEFINE FORMULAS ---

base_controls <- "+ sex_concept + month + (1 | person_id)"

# List 1: Linear Age Models
formulas_linear <- list(
  list(name="onset_linear",   y="onset_linear",    desc="Linear Age"),
  list(name="offset_linear",  y="offset_linear",   desc="Linear Age"),
  list(name="midpoint_linear",y="midpoint_linear", desc="Linear Age"),
  list(name="duration_linear",y="duration_hours",  desc="Linear Age")
)

# List 2: Quadratic Age Models
formulas_quad <- list(
  list(name="onset_quad",   y="onset_linear",    desc="Quadratic Age"),
  list(name="offset_quad",  y="offset_linear",   desc="Quadratic Age"),
  list(name="midpoint_quad",y="midpoint_linear", desc="Quadratic Age"),
  list(name="duration_quad",y="duration_hours",  desc="Quadratic Age")
)

# List 3: Interaction Models
formulas_interact <- list(
  list(name="onset_interact",   y="onset_linear",    desc="Quad Age + Employ*Weekend"),
  list(name="offset_interact",  y="offset_linear",   desc="Quad Age + Employ*Weekend"),
  list(name="midpoint_interact",y="midpoint_linear", desc="Quad Age + Employ*Weekend"),
  list(name="duration_interact",y="duration_hours",  desc="Quad Age + Employ*Weekend")
)

# --- EXECUTION LOOP ---

run_batch <- function(model_list, custom_formula_part) {
  local_perf <- data.frame()
  for(m in model_list) {
    f <- as.formula(paste(m$y, custom_formula_part, base_controls))
    
    res <- fit_save_summarize(f, m$name, m$desc, df_clean, run_reml = FIT_ALL_REML)
    local_perf <- rbind(local_perf, res)
  }
  return(local_perf)
}

# 1. Run Linear Age Models
model_performance <- rbind(model_performance, 
                           run_batch(formulas_linear, "~ age_at_sleep + is_weekend + employment_status"))

# 2. Run Quadratic Age Models
model_performance <- rbind(model_performance, 
                           run_batch(formulas_quad, "~ poly(age_at_sleep, 2) + is_weekend + employment_status"))

# 3. Run Interaction Models
model_performance <- rbind(model_performance, 
                           run_batch(formulas_interact, "~ poly(age_at_sleep, 2) + is_weekend * employment_status"))


# --- GENERATE MARKDOWN TABLE ---

# Sort by AIC (Handling NAs by putting them last)
model_performance <- model_performance[order(model_performance$AIC, na.last = TRUE), ]

md_header <- "| Model Name | Description | AIC | BIC | Status |\n|---|---|---|---|---|\n"
md_rows <- apply(model_performance, 1, function(x) {
  
  # Handle NAs for display
  aic_disp <- if(is.na(x['AIC'])) "NA" else round(as.numeric(x['AIC']), 2)
  bic_disp <- if(is.na(x['BIC'])) "NA" else round(as.numeric(x['BIC']), 2)
  
  paste0("| ", x['Model_Name'], " | ", x['Formula_Description'], " | ", aic_disp, " | ", bic_disp, " | ", x['Status'], " |")
})

md_content <- paste(md_header, paste(md_rows, collapse = "\n"), sep = "")
writeLines(md_content, AIC_REPORT_FILE)

message("------------------------------------------------")
message(paste("✅ All processing complete."))
message(paste("✅ AIC Comparison Table:", AIC_REPORT_FILE))
if(FIT_ALL_REML) {
  message("✅ Both ML (Selection) and REML (Inference) models were attempted.")
} else {
  message("✅ Only ML models were saved.")
}


