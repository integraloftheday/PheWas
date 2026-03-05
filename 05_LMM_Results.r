# --- Block 0: Global Setup ---
suppressPackageStartupMessages({
  library(lme4)
  library(dplyr)
  library(performance) # Key package for efficient diagnostics
  library(tibble)
  library(knitr)       # For nice table rendering in notebooks
  library(doParallel)
  library(foreach)
library(lme4)
library(dplyr)
library(marginaleffects)
    library(ggplot2)
library(dplyr)
library(stringr)
library(marginaleffects)
library(purrr)     # For easy iteration
library(patchwork) # For combining plots with '+'
    library(ggplot2)
library(dplyr)
library(stringr)
library(patchwork)
library(scales)
library(tools)
library(dplyr)
library(purrr)
library(readr) # Part of tidyverse

})

# Define the directory where your binary models are stored
model_dir <- "models"

# Define your outcomes and model types for iteration
outcomes <- c("onset", "offset", "midpoint", "duration")
complexities <- c("linear", "quad", "interact")
model_dir <- "models"
target_complexity <- "interact" # or "linear", "quad"
model_type <- "REML"  

# Helper function to construct file path
get_model_path <- function(outcome, complexity, type = "REML") {
  # Type must be "ML" or "REML"
  model_name <- paste0(outcome, "_", complexity)
  file_path <- file.path(model_dir, paste0(model_name, "_", type, ".rds"))
  return(file_path)
}

print("Setup Complete. Ready for Phase 1 Analysis.")

# --- 2. HELPER: Time Formatting Function ---
# This converts linear hours (e.g., -1.5, 25.25) into HH:MM strings (22:30, 01:15)
# while preserving the continuous numeric scale for the axis.
format_time_axis <- function(x) {
  # 1. Normalize to 0-24 range for the label text
  #    R's modulo (%%) handles negatives correctly (-1 %% 24 = 23)
  normalized_hours <- x %% 24
  
  # 2. Extract hours and minutes
  hrs <- floor(normalized_hours)
  mins <- round((normalized_hours - hrs) * 60)
  
  # 3. Format as HH:MM string
  sprintf("%02d:%02d", hrs, mins)
}

get_model_values <- function(model, var_name) {
  meta <- attr(model, "phewas_grid_meta", exact = TRUE)
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

# --- 3. PROCESS: Batch Load & Predict ---
# Define the specific files/outcomes you want
outcomes <- c("onset", "midpoint", "offset", "duration")

message(">>> Batch processing models...")

# map_dfr iterates through 'outcomes', runs the code, and binds rows together
all_predictions <- map_dfr(outcomes, function(outcome_name) {
  
  # Construct path
  fname <- paste0(outcome_name, "_", target_complexity, "_", model_type, "_fast.rds")
  fpath <- file.path(model_dir, fname)
  
  if (!file.exists(fpath)) {
    warning(paste("Skipping missing model:", fpath))
    return(NULL)
  }
  message(paste("Processing:", fpath))
  
  # 1. Load the model
  model <- readRDS(fpath)
  
  message(paste("Predicting:", outcome_name))
  
  # 2. Generate predictions
  # Using datagrid for speed as requested
  preds <- predictions(
    model,
    newdata = datagrid(
      employment_status = get_model_values(model, "employment_status"),
      is_weekend = c(TRUE, FALSE)
    )
  )
  
  # 3. Process the result into a dataframe
  # We assign this to a variable first so we can run cleanup code after
  result_df <- preds %>%
    as.data.frame() %>%
    mutate(
      Outcome = outcome_name,
      # Identify if this is a 'Clock Time' metric or a 'Duration' metric
      Type = ifelse(outcome_name == "duration", "Quantity", "ClockTime")
    )
  
  # 4. GARBAGE COLLECTION
  # Remove the heavy model object from the environment
  rm(model) 
  
  # Explicitly call garbage collection to reclaim RAM to OS
  # verbose = FALSE keeps the console clean
  gc(verbose = FALSE) 
  
  # 5. Return the result
  return(result_df)
})

dim(all_predictions)

write_csv(all_predictions, "./results/interactions_sleep_data.csv")
saveRDS(all_predictions, "./results/interactions_sleep_data.rds")


# --- 1. SETTINGS & FORMATTERS ---

# TIMING PLOT SETTINGS (Onset/Midpoint/Offset)
# 0.5 = 30 mins, 0.25 = 15 mins
TIMING_TICK_INTERVAL <- 0.5 

# DURATION PLOT SETTINGS
# 0.25 = 15 mins (Best for readability of 7h 15m vs 7h 30m)
DURATION_TICK_INTERVAL <-0.16666666666/2

# Formatter for Time of Day (e.g., 23.5 -> "23:30")
format_time_axis <- function(x) {
  x_norm <- x %% 24
  hours <- floor(x_norm)
  minutes <- round((x_norm %% 1) * 60)
  sprintf("%02d:%02d", hours, minutes)
}

# Formatter for Duration (e.g., 7.5 -> "7h 30m")
format_duration_axis <- function(x) {
  hours <- floor(x)
  minutes <- round((x %% 1) * 60)
  sprintf("%dh %02dm", hours, minutes) 
}

# --- 2. DATA PREPARATION ---

plot_ready_data <- all_predictions %>%
  mutate(
    Clean_Term = as.character(employment_status),
    # Replace underscores with spaces
    Clean_Term = str_replace_all(Clean_Term, "_", " "),
    # Smart Title Casing (e.g., "Out Of Work" -> "Out of Work")
    Clean_Term = tools::toTitleCase(tolower(Clean_Term)), 
    # Wrap text to keep labels tidy
    Clean_Term = str_wrap(Clean_Term, width = 25),
    
    # Define Day Type Factor
    Day_Type = factor(ifelse(is_weekend == TRUE, "Weekend", "Weekday"), 
                      levels = c("Weekday", "Weekend")),
    
    # Define Outcome Factor
    Outcome = factor(Outcome, levels = c("onset", "midpoint", "offset", "duration"))
  )

# Sort Employment categories by Sleep Duration (Shortest to Longest on Weekdays)
# This creates the "staircase" visual effect
order_levels <- plot_ready_data %>%
  filter(Outcome == "duration", Day_Type == "Weekday") %>%
  arrange(estimate) %>% 
  pull(Clean_Term)

plot_ready_data$Clean_Term <- factor(plot_ready_data$Clean_Term, levels = order_levels)


# --- 3. PLOTTING FUNCTIONS ---

# Function A: For Time of Day (Onset, Midpoint, Offset)
create_timing_plot <- function(data_subset, title_text, show_y_axis = TRUE) {
  
  # Calculate limits to ensure ticks fit nicely
  min_val <- floor(min(data_subset$conf.low))
  max_val <- ceiling(max(data_subset$conf.high))
  
  p <- ggplot(data_subset, aes(x = estimate, y = Clean_Term, color = Day_Type)) +
    geom_line(aes(group = Clean_Term), color = "gray85", linewidth = 0.6) +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0, linewidth = 1.2, alpha = 1) +
    geom_point(size = 3.0, shape = 21, fill = "white", stroke = 1.5) + 
    geom_point(size = 3.0, alpha = 1) + 
    scale_color_manual(values = c("Weekday" = "#0072B2", "Weekend" = "#D55E00")) +
    labs(title = title_text, x = NULL, y = NULL) +
    theme_classic(base_size = 14) +
    theme(
      legend.position = "none",
      plot.title = element_text(size = 14, face = "bold", hjust = 0.5),
      axis.line.y = element_blank(), axis.ticks.y = element_blank(),
      axis.text.x = element_text(color = "black", size = 10),
      panel.grid.major.y = element_line(color = "gray92"),
      panel.grid.major.x = element_line(color = "gray90", linetype = "dotted"), 
      axis.line.x = element_line(color = "black", linewidth = 0.5)
    )

  # Time Axis Scaling
  breaks_seq <- seq(from = min_val, to = max_val, by = TIMING_TICK_INTERVAL)
  p <- p + scale_x_continuous(
      labels = format_time_axis, 
      breaks = breaks_seq,
      limits = c(min(breaks_seq), max(breaks_seq))
  )

  if (!show_y_axis) {
    p <- p + theme(axis.text.y = element_blank())
  } else {
    p <- p + theme(axis.text.y = element_text(color = "black", size = 12, margin = margin(r = 10)))
  }
  return(p)
}

# Function B: For Duration (Hours + Minutes)
create_duration_plot <- function(data_subset, title_text) {
  
  # Determine logical range for 15-minute breaks
  min_val <- min(data_subset$conf.low)
  max_val <- max(data_subset$conf.high)
  
  # Create breaks every 15 minutes (0.25) or 30 mins (0.5)
  breaks_seq <- seq(from = floor(min_val * 4)/4, 
                    to = ceiling(max_val * 4)/4, 
                    by = DURATION_TICK_INTERVAL) 
  
  p <- ggplot(data_subset, aes(x = estimate, y = Clean_Term, color = Day_Type)) +
    geom_line(aes(group = Clean_Term), color = "gray85", linewidth = 0.8) +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), height = 0, linewidth = 1.2, alpha = 1) +
    geom_point(size = 3.5, shape = 21, fill = "white", stroke = 1.5) + 
    geom_point(size = 3.5, alpha = 1) + 
    scale_color_manual(values = c("Weekday" = "#0072B2", "Weekend" = "#D55E00")) +
    labs(title = title_text, x = NULL, y = NULL) +
    theme_classic(base_size = 14) +
    theme(
      legend.position = "bottom",
      legend.title = element_blank(),
      plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
      axis.line.y = element_blank(), axis.ticks.y = element_blank(),
      axis.text.y = element_text(color = "black", size = 12, margin = margin(r = 10)),
      axis.text.x = element_text(color = "black", size = 11),
      panel.grid.major.y = element_line(color = "gray95"),
      panel.grid.major.x = element_line(color = "gray90", linetype = "dotted"), # Vital for reading minutes
      axis.line.x = element_line(color = "black", linewidth = 0.5)
    ) +
    scale_x_continuous(
      labels = format_duration_axis,
      breaks = breaks_seq,
      limits = c(min(breaks_seq), max(breaks_seq))
    )
  
  return(p)
}

# --- 4. GENERATE AND COMBINE ---

# FIGURE 1: TIMING (Onset + Midpoint + Offset)
p_on  <- create_timing_plot(filter(plot_ready_data, Outcome == "onset"), "Sleep Onset", show_y_axis = TRUE)
p_mid <- create_timing_plot(filter(plot_ready_data, Outcome == "midpoint"), "Sleep Midpoint", show_y_axis = FALSE)
p_off <- create_timing_plot(filter(plot_ready_data, Outcome == "offset"), "Sleep Offset", show_y_axis = FALSE)

fig_timing <- p_on + p_mid + p_off + 
  plot_layout(guides = "collect") & 
  theme(legend.position = "bottom", legend.title = element_blank())

fig_timing <- fig_timing + plot_annotation(
  title = "Sleep Timing: Social Jetlag Analysis",
  theme = theme(plot.title = element_text(size = 16, face = "bold"))
)

# FIGURE 2: DURATION
dat_dur <- filter(plot_ready_data, Outcome == "duration")
fig_duration <- create_duration_plot(dat_dur, "Sleep Duration") + 
  plot_annotation(
    title = "Sleep Duration by Employment Status",
    theme = theme(plot.title = element_text(size = 18, face = "bold"))
  )

# --- 5. VIEW PLOTS ---

# Print Figure 1 (Timing)
options(repr.plot.width = 16, repr.plot.height = 8)
print(fig_timing)

# Print Figure 2 (Duration) - Note the granular X-axis
options(repr.plot.width = 16, repr.plot.height = 7)
print(fig_duration)


# --- HELPER: Standardize Prediction Tables ---
# This renames the specific column (e.g., 'employment_status') to a generic 'Term'
# so we can stack them all into one big dataframe.
clean_pred <- function(pred_obj, term_col, category_name, outcome_name) {
  pred_obj %>%
    as.data.frame() %>%
    mutate(
      Term = as.character(.data[[term_col]]), # Rename specific col to generic 'Term'
      Category = category_name,
      Outcome = outcome_name,
      Type = ifelse(outcome_name == "duration", "Quantity", "ClockTime")
    ) %>%
    select(Term, Category, Outcome, Type, estimate, conf.low, conf.high)
}

# --- MAIN LOOP ---
outcomes <- c("onset", "midpoint", "offset", "duration")

all_main_effects <- map_dfr(outcomes, function(outcome_name) {
  
  # 1. Load Model
  fname <- paste0(outcome_name, "_", target_complexity, "_", model_type, "_fast.rds")
  fpath <- file.path(model_dir, fname)
  
  if (!file.exists(fpath)) return(NULL)
  model <- readRDS(fpath)
  
  message(paste("Predicting Main Effects for:", outcome_name))
  
  # --- PREDICTION 1: EMPLOYMENT EFFECT ---
  # Vary Employment, hold others at mean/mode
  p_emp <- predictions(
    model, 
    newdata = datagrid(employment_status = get_model_values(model, "employment_status"))
  ) %>% clean_pred("employment_status", "Employment", outcome_name)

  # --- PREDICTION 2: DEMOGRAPHICS (SEX) ---
  # Vary Sex, hold others at mean/mode
  # Note: Adjust 'sex_concept' if your variable name is different
  p_sex <- predictions(
    model, 
    newdata = datagrid(sex_concept = get_model_values(model, "sex_concept"))
  ) %>% clean_pred("sex_concept", "Demographics", outcome_name)

  # --- PREDICTION 3: WEEKEND EFFECT ---
  # Vary Weekend, hold others at mean/mode
  p_we <- predictions(
    model, 
    newdata = datagrid(is_weekend = c(TRUE, FALSE))
  ) %>% 
    mutate(is_weekend = ifelse(is_weekend, "Weekend", "Weekday")) %>% # Fix label
    clean_pred("is_weekend", "Weekend Effect", outcome_name)

  # Combine all three for this outcome
  bind_rows(p_emp, p_sex, p_we)
})

visualize_main_effects_means <- function(df) {
  
  plot_data <- df %>%
    # Filter "No Matching Concept" to clean up the scale
    filter(!str_detect(Term, "No Matching|Skip|Refused|None Of These")) %>%
    mutate(
      Term = str_remove(Term, "^.*:\\s*"), 
      Term = str_replace_all(Term, "_", " "),
      Term = str_to_title(Term),
      Term = str_wrap(Term, width = 25), 
      
      Outcome = factor(Outcome, levels = c("onset", "midpoint", "offset", "duration")),
      Category = factor(Category, levels = c("Demographics", "Employment", "Weekend Effect"))
    ) %>%
    group_by(Category) %>%
    mutate(Term = reorder(Term, estimate)) %>%
    ungroup()
  
  ggplot(plot_data, aes(x = estimate, y = Term)) +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), 
                   height = 0.2, linewidth = 0.8, color = "gray60") +
    geom_point(size = 4.5, color = "#0072B2") +
    
    facet_grid(Category ~ Outcome, scales = "free", space = "free_y", switch = "y") +
    
    scale_x_continuous(labels = format_time_axis) +
    
    labs(
      title = "Predicted Main Effects: Sleep Patterns",
      subtitle = "Marginal Means (sorted by estimate)",
      x = "Predicted Time (HH:MM) / Duration (Hours)",
      y = NULL
    ) +
    
    theme_bw(base_size = 18) +  # Increased from 14
    theme(
      strip.placement = "outside",
      strip.text.y = element_text(angle = 180, face = "bold", size = 16, 
                                  margin = margin(r = 10)),
      strip.text.x = element_text(face = "bold", size = 17, margin = margin(b = 10)),
      strip.background = element_rect(fill = "gray97", color = NA),
      
      axis.text.y = element_text(size = 15), 
      axis.text.x = element_text(size = 14, angle = 45, hjust = 1),
      axis.title.x = element_text(size = 16),
      
      plot.title = element_text(size = 22, face = "bold"),
      plot.subtitle = element_text(size = 18),
      
      plot.margin = margin(t = 10, r = 10, b = 10, l = 50, unit = "pt"),
      
      panel.spacing.x = unit(1.5, "lines"),
      panel.grid.minor = element_blank()
    )
}

options(repr.plot.width = 20, repr.plot.height = 18)
print(visualize_main_effects_means(all_main_effects))

dim(all_main_effects)

library(readr) # Part of tidyverse

# Save to the current working directory
write_csv(all_main_effects, "./results/main_effects_sleep_data.csv")
saveRDS(all_main_effects, "./results/main_effects_sleep_data.rds")

library(marginaleffects)
library(dplyr)
library(purrr)
library(stringr)

# --- HELPER: Standardize Output ---
clean_predictions <- function(pred_obj, term_col, category_name, outcome_name) {
  pred_obj %>%
    as.data.frame() %>%
    mutate(
      Term = as.character(.data[[term_col]]), 
      Category = category_name,
      Outcome = outcome_name,
      Type = ifelse(outcome_name == "duration", "Quantity", "ClockTime")
    ) %>%
    select(Term, Category, Outcome, Type, estimate, conf.low, conf.high)
}

# --- MAIN LOOP ---
outcomes <- c("onset", "midpoint", "offset", "duration")

all_seasonality_means <- map_dfr(outcomes, function(outcome_name) {
  
  fname <- paste0(outcome_name, "_", target_complexity, "_", model_type, "_fast.rds")
  fpath <- file.path(model_dir, fname)
  
  if (!file.exists(fpath)) return(NULL)
  model <- readRDS(fpath)
  
  message(paste("Predicting Monthly Means for:", outcome_name))
  
  # --- PREDICTION: SEASONALITY ---
  # distinct() ensures we get exactly one row per month found in the data
  # This is much faster than emmeans
  p_month <- predictions(
    model, 
    newdata = datagrid(month = get_model_values(model, "month")) 
  ) %>%
    clean_predictions("month", "Seasonality", outcome_name)
  
  return(p_month)
})

write_csv(all_seasonality_means, "./results/seasonality_sleep_data.csv")
saveRDS(all_seasonality_means, "./results/seasonality_sleep_data.rds")

library(ggplot2)
library(dplyr)
library(patchwork)
library(scales)
library(tools)

# --- 1. AXIS FORMATTERS ---
# Time: 23.5 -> 23:30
format_time_axis <- function(x) {
  x_norm <- x %% 24
  hours <- floor(x_norm)
  minutes <- round((x_norm %% 1) * 60)
  sprintf("%02d:%02d", hours, minutes)
}

# Duration: 7.5 -> 7h 30m
format_duration_axis <- function(x) {
  hours <- floor(x)
  minutes <- round((x %% 1) * 60)
  sprintf("%dh %02dm", hours, minutes) 
}

# --- 2. THE VISUALIZER FUNCTION ---
visualize_seasonality_means <- function(df) {
  
  # A. Data Cleaning & Factor Ordering
  # -------------------------------------------------------
  # Map numeric/string months to Abbreviations
  month_map <- setNames(month.abb, 1:12) # Standard R "Jan"..."Dec"
  month_map_str <- setNames(month.abb, sprintf("%02d", 1:12)) # "01"..."12"
  
  plot_data <- df %>%
    mutate(
      # Create clean month column
      Clean_Term = case_when(
        Term %in% names(month_map) ~ month_map[as.character(Term)],
        Term %in% names(month_map_str) ~ month_map_str[as.character(Term)],
        TRUE ~ as.character(Term)
      ),
      # REVERSE order so "Jan" appears at the TOP of the Y-axis in ggplot
      Clean_Term = factor(Clean_Term, levels = rev(month.abb)),
      
      # Ensure Outcomes are ordered logically
      Outcome = factor(Outcome, levels = c("onset", "midpoint", "offset", "duration"))
    )

  # B. Calculate Yearly Averages (The Red Dashed Line)
  # -------------------------------------------------------
  yearly_avg <- plot_data %>%
    group_by(Outcome) %>%
    summarise(Avg_Est = mean(estimate, na.rm = TRUE), .groups = "drop")

  # C. Common Theme Definition (DRY Principle)
  # -------------------------------------------------------
  paper_theme <- theme_bw(base_size = 14) +
    theme(
      strip.background = element_rect(fill = "white", color = "black"),
      strip.text = element_text(face = "bold", size = 12),
      axis.text.x = element_text(angle = 0, hjust = 0.5, color = "black"),
      axis.text.y = element_text(color = "black"),
      panel.grid.minor = element_blank(),
      panel.grid.major.y = element_line(color = "gray92"), # horizontal guides
      panel.grid.major.x = element_line(color = "gray90", linetype = "dotted"), # vertical guides
      plot.title = element_text(face = "bold", size = 14)
    )

  # D. Plot 1: TIMING (Onset, Midpoint, Offset)
  # -------------------------------------------------------
  df_timing <- filter(plot_data, Outcome != "duration")
  avg_timing <- filter(yearly_avg, Outcome != "duration")
  
  # Dynamic limits for timing to make ticks nice
  t_min <- floor(min(df_timing$conf.low))
  t_max <- ceiling(max(df_timing$conf.high))
  
  p_timing <- ggplot(df_timing, aes(x = estimate, y = Clean_Term, group = 1)) +
    # 1. Yearly Average Line
    geom_vline(data = avg_timing, aes(xintercept = Avg_Est), 
               linetype = "dashed", color = "#D55E00", linewidth = 0.8) +
    # 2. The Seasonal Trend Line (Connects the months)
    geom_path(color = "gray70", linewidth = 0.8) +
    # 3. Error Bars
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), 
                   height = 0, linewidth = 0.6, color = "gray40") +
    # 4. Points
    geom_point(size = 3, color = "#0072B2") +
    # 5. Faceting
    facet_grid(~ Outcome, scales = "free_x") +
    # 6. Formatting
    scale_x_continuous(labels = format_time_axis, breaks = scales::pretty_breaks(n=4)) +
    labs(title = "Seasonal Sleep Timing", x = NULL, y = NULL) +
    paper_theme

  # E. Plot 2: DURATION (Hours)
  # -------------------------------------------------------
  df_dur <- filter(plot_data, Outcome == "duration")
  avg_dur <- filter(yearly_avg, Outcome == "duration")
  
  # Dynamic limits for duration (force quarters if needed)
  d_min <- floor(min(df_dur$conf.low))
  d_max <- ceiling(max(df_dur$conf.high))
  
  p_duration <- ggplot(df_dur, aes(x = estimate, y = Clean_Term, group = 1)) +
    geom_vline(data = avg_dur, aes(xintercept = Avg_Est), 
               linetype = "dashed", color = "#D55E00", linewidth = 0.8) +
    geom_path(color = "gray70", linewidth = 0.8) +
    geom_errorbarh(aes(xmin = conf.low, xmax = conf.high), 
                   height = 0, linewidth = 0.6, color = "gray40") +
    geom_point(size = 3, color = "#0072B2") +
    facet_grid(~ Outcome) +
    # Use Duration Formatter here
    scale_x_continuous(labels = format_duration_axis, breaks = scales::pretty_breaks(n=5)) +
    labs(title = "Seasonal Sleep Duration", x = "Estimated Means (95% CI)", y = NULL) +
    paper_theme +
    theme(axis.text.y = element_blank(), axis.ticks.y = element_blank()) # Hide Y labels on 2nd plot

  # F. Combine with Patchwork
  # -------------------------------------------------------
  # Layout: Timing (3/4 width) | Duration (1/4 width)
  combined_plot <- p_timing + p_duration + 
    plot_layout(widths = c(3, 1)) +
    plot_annotation(
      title = "Seasonal Trends in Sleep Behavior",
      subtitle = "Monthly predicted means relative to yearly average (dashed red line)",
      theme = theme(plot.title = element_text(size = 18, face = "bold"),
                    plot.subtitle = element_text(size = 12, color = "gray50"))
    )
    
  return(combined_plot)
}

# --- 3. RUN IT ---
# (Assuming 'all_seasonality_means' exists from your previous code)
seasonality_plot <- visualize_seasonality_means(all_seasonality_means)

# View
options(repr.plot.width = 20, repr.plot.height = 8)
print(seasonality_plot)

# --- Block 5: Age Effect Analysis (Quadratic) ---

library(lme4)
library(dplyr)
library(marginaleffects)
library(ggplot2)
library(patchwork)
library(purrr)

# --- 1. SETTINGS ---
model_dir <- "models"
outcomes <- c("onset", "midpoint", "offset", "duration")
target_complexity <- "interact" 
model_type <- "REML"

# *** MANUAL OVERRIDE ***
# Since the model stores transformed polynomials (small decimals),
# we must specify the real-world age range we want to graph.
MIN_AGE <- 18
MAX_AGE <- 85 

# --- 2. PREDICTION LOOP ---
all_age_preds <- map_dfr(outcomes, function(outcome_name) {
  
  fname <- paste0(outcome_name, "_", target_complexity, "_", model_type, "_fast.rds")
  fpath <- file.path(model_dir, fname)
  
  if (!file.exists(fpath)) {
    warning(paste("Model not found:", fpath))
    return(NULL)
  }
  
  model <- readRDS(fpath)
  message(paste("Processing Age Trends for:", outcome_name))
  
  # Create a sequence of REAL ages
  age_seq <- seq(from = MIN_AGE, to = MAX_AGE, by = 1)
  
  # Generate Predictions
  # datagrid() is smart enough to take these real numbers (e.g., 45)
  # and pass them through the model's poly() function correctly.
  preds <- predictions(
    model,
    newdata = datagrid(age_at_sleep = age_seq)
  )
  
  # Clean and Format
  preds %>%
    as.data.frame() %>%
    mutate(
      Outcome = outcome_name,
      Type = ifelse(outcome_name == "duration", "Quantity", "ClockTime")
    ) %>%
    select(age_at_sleep, Outcome, Type, estimate, conf.low, conf.high)
})

# --- 3. PLOTTING FUNCTIONS ---

format_time_axis <- function(x) {
  x_norm <- x %% 24
  hours <- floor(x_norm)
  minutes <- round((x_norm %% 1) * 60)
  sprintf("%02d:%02d", hours, minutes)
}

format_duration_axis <- function(x) {
  hours <- floor(x)
  minutes <- round((x %% 1) * 60)
  sprintf("%dh %02dm", hours, minutes) 
}

visualize_age_trends <- function(df) {
  
  plot_data <- df %>%
    mutate(
      Outcome = factor(Outcome, levels = c("onset", "midpoint", "offset", "duration"))
    )
  
  common_theme <- theme_classic(base_size = 14) +
    theme(
      plot.title = element_text(face = "bold", size = 16),
      strip.background = element_rect(fill = "gray95", color = NA),
      strip.text = element_text(face = "bold", size = 12),
      panel.grid.major = element_line(color = "gray92"),
      panel.grid.minor = element_blank(),
      legend.position = "none"
    )

  # Plot 1: Timing
  df_timing <- filter(plot_data, Outcome != "duration")
  
  p_timing <- ggplot(df_timing, aes(x = age_at_sleep, y = estimate)) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), fill = "#0072B2", alpha = 0.2) +
    geom_line(color = "#0072B2", linewidth = 1.2) +
    facet_wrap(~ Outcome, scales = "free_y", ncol = 3) +
    scale_y_continuous(labels = format_time_axis) +
    labs(title = "Age-Related Shifts in Sleep Timing", x = "Age (Years)", y = "Time of Day") +
    common_theme

  # Plot 2: Duration
  df_dur <- filter(plot_data, Outcome == "duration")
  
  p_duration <- ggplot(df_dur, aes(x = age_at_sleep, y = estimate)) +
    geom_ribbon(aes(ymin = conf.low, ymax = conf.high), fill = "#D55E00", alpha = 0.2) +
    geom_line(color = "#D55E00", linewidth = 1.2) +
    facet_wrap(~ Outcome, scales = "free_y") + 
    scale_y_continuous(labels = format_duration_axis) +
    labs(title = "Age-Related Shifts in Duration", x = "Age (Years)", y = "Duration") +
    common_theme

  # Combine
  combined <- p_timing / p_duration + plot_layout(heights = c(2, 1))
  return(combined)
}

# --- 4. EXECUTE ---
if (exists("all_age_preds") && nrow(all_age_preds) > 0) {
  age_plot <- visualize_age_trends(all_age_preds)
  options(repr.plot.width = 16, repr.plot.height = 10)
  print(age_plot)
} else {
  message("No predictions generated.")
}

# Example of how to save the raw data
readr::write_csv(
    x = all_age_preds, 
    file = "results/Age_Effect_Quadratic_Predictions.csv"
)

# OR (if you prefer an R binary format for speed and precision)
saveRDS(
    object = all_age_preds, 
    file = "results/Age_Effect_Quadratic_Predictions.rds"
)

