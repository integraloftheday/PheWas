 #Install packages if needed (commented out)
 #install.packages(c("tidyverse", "arrow", "gtsummary", "mgcv", "viridis"))

library(tidyverse)
library(arrow)
library(viridis)
library(dplyr)
library(gtsummary)

# Load data
df <- arrow::read_parquet("processed_data/ready_for_analysis.parquet")

# Preview
#head(df)

colnames(df)

# ==============================================================================
# 1. AGGREGATE DATA FROM NIGHT-LEVEL TO PERSON-LEVEL
# ==============================================================================
person_df <- df %>%
  group_by(person_id) %>%
  summarise(
    # --- Demographics (Categorical) ---
    sex_concept = first(sex_concept),
    race = first(race),
    employment_status = first(employment_status),
    education = first(education),
    smoking = first(smoking),
    alcohol_freq = first(alcohol_freq),
    
    # --- Demographics (Continuous) ---
    age = mean(age_at_sleep, na.rm = TRUE),
    bmi = mean(bmi, na.rm = TRUE),
    deprivation_index = mean(deprivation_index, na.rm = TRUE),
    
    # --- Sleep Statistics (Aggregated) ---
    total_nights_recorded = n(),
    avg_nightly_duration = mean(daily_duration_mins, na.rm = TRUE)
  ) %>%
  ungroup()

# ==============================================================================
# 2. FILTERING AND MASKING LOGIC
# ==============================================================================
# Step A: Remove STRATIFICATION GROUPS (Table Columns) with < 20 people
person_df_groups_filtered <- person_df %>%
  group_by(sex_concept) %>%
  filter(n() >= 20) %>%
  ungroup()

# Step B: Remove DATA VARIABLES (Table Rows) that have < 20 valid observations
cols_to_keep <- names(person_df_groups_filtered)[colSums(!is.na(person_df_groups_filtered)) >= 20]
# Ensure we keep the grouping variable 'sex_concept' and ID
cols_to_keep <- unique(c(cols_to_keep, "sex_concept", "person_id"))

person_df_final <- person_df_groups_filtered %>%
  select(all_of(cols_to_keep))

# ==============================================================================
# 3. GENERATE TABLE WITH CELL SUPPRESSION
# ==============================================================================
# Define Chi-Sq arguments
chisq_args <- list(simulate.p.value = TRUE)

table1 <- person_df_final %>%
  # Select columns for the table (checking if they survived the filter)
  select(any_of(c(
    "age", "sex_concept", "race", "employment_status", 
    "education", "bmi", "total_nights_recorded", "avg_nightly_duration"
  ))) %>%
  tbl_summary(
    by = sex_concept, 
    
    # --- UPDATED SECTION: ADDED MEDIAN ---
    statistic = list(
      all_continuous() ~ "{mean} ({sd}) / {median} [{min}, {max}]",
      all_categorical() ~ "{n} ({p}%)"
    ),
    
    label = list(
      age ~ "Age (Years)",
      race ~ "Race",
      employment_status ~ "Employment Status",
      education ~ "Education Level",
      bmi ~ "BMI",
      total_nights_recorded ~ "Total Nights Recorded (n)",
      avg_nightly_duration ~ "Avg Nightly Duration (mins)"
    ),
    
    missing = "ifany", 
    missing_text = "(Missing)"
  ) %>%
  add_p(
    test = list(all_categorical() ~ "chisq.test"),
    test.args = list(all_categorical() ~ chisq_args)
  ) %>%
  # Step C: MASK SPECIFIC SQUARES (Cells with N < 20)
  modify_table_body(
    ~ .x %>%
      dplyr::mutate(
        across(
          starts_with("stat_"),
          ~ dplyr::case_when(
            # Logic: If row is categorical AND the leading number (count) is < 20
            var_type %in% c("categorical", "dichotomous") & 
            suppressWarnings(as.numeric(sub("^([0-9]+).*", "\\1", .))) < 20 ~ "<20",
            TRUE ~ .
          )
        )
      )
  ) %>%
  modify_header(label = "**Variable**") %>%
  # --- UPDATED FOOTNOTE TO REFLECT NEW FORMAT ---
  modify_footnote(
    all_stat_cols() ~ "Statistics: Mean (SD) / Median [Min, Max] for continuous; n (%) for categorical. Values <20 masked."
  ) %>%
  add_overall() %>%
  as_kable()

# ==============================================================================
# 4. RENDER
# ==============================================================================
if (requireNamespace("IRdisplay", quietly = TRUE)) {
  IRdisplay::display_markdown(paste(as.character(table1), collapse = "\n"))
} else {
  print(table1)
}

# Figure 1 (Chronotype)
# Histogram of midpoint
ggplot(df, aes(x = daily_midpoint_hour)) +
  geom_histogram(binwidth = 0.5, fill = "skyblue", color = "black") +
  labs(title = "Distribution of Sleep Midpoint (Chronotype)", x = "Midpoint Hour", y = "Count") +
  theme_minimal()

# Histogram of Number of Nights per Person
nights_per_person <- df %>%
  count(person_id, name = "n_nights")

ggplot(nights_per_person, aes(x = n_nights)) +
  geom_histogram(bins = 50, fill = "steelblue", color = "white") +
  scale_y_log10() +
  labs(
    title = "Data Density: Number of Nights per Person",
    subtitle = "Log-scaled Y-axis to show distribution balance",
    x = "Number of Nights",
    y = "Count of People (Log Scale)"
  ) +
  theme_minimal()

# Helper function for density plots
plot_density_by_weekend <- function(data, var, title) {
  ggplot(data, aes(x = .data[[var]], fill = as.factor(is_weekend))) +
    geom_density(alpha = 0.5) +
    labs(title = title, x = "Hour of Day", y = "Density", fill = "Is Weekend") +
    theme_minimal()
}

# Use full dataset for plotting/modeling (no sampling)
df_sample <- df

# Onset
p1 <- plot_density_by_weekend(df_sample, "daily_start_hour", "Sleep Onset Density: Weekday vs Weekend")
print(p1)

# Offset
p2 <- plot_density_by_weekend(df_sample, "daily_end_hour", "Sleep Offset Density: Weekday vs Weekend")
print(p2)

# Midpoint
p3 <- plot_density_by_weekend(df_sample, "daily_midpoint_hour", "Sleep Midpoint Density: Weekday vs Weekend")
print(p3)

colnames(df_sample)



# Figure 3 (Social Jetlag)
# Paired boxplot of Weekday vs Weekend midpoint
df_sjl <- df %>%
  group_by(person_id, is_weekend) %>%
  summarise(mean_midpoint = mean(daily_midpoint_hour, na.rm = TRUE)) %>%
  ungroup()

ggplot(df_sjl, aes(x = as.factor(is_weekend), y = mean_midpoint)) +
  geom_boxplot(fill = "lightgreen") +
  labs(title = "Social Jetlag: Weekday vs Weekend Midpoint", x = "Is Weekend", y = "Mean Midpoint Hour") +
  theme_minimal()

# One row per person, restrict to Female & Male
df_sjl_age <- df %>%
  group_by(person_id, sex_concept) %>%
  summarise(
    mean_age = mean(age_at_sleep, na.rm = TRUE),
    sjl_raw  = first(SJL_raw)   # constant per person from Phase 0
  ) %>%
  ungroup() %>%
  filter(sex_concept %in% c("Female", "Male"))

# GAM-only plot
ggplot(df_sjl_age, aes(x = mean_age, y = sjl_raw, color = sex_concept)) +
  geom_smooth(
    method = mgcv::bam,
    formula = y ~ s(x, bs = "cs"),
    method.args = list(method = "fREML", discrete = TRUE),
    se = TRUE
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  labs(
    title = "Social Jetlag across the Lifespan",
    subtitle = "Does the weekend effect disappear in retirement?",
    x = "Age",
    y = "Social Jetlag (Weekend - Weekday Midpoint)",
    color = "Sex"
  ) +
  scale_color_manual(
    values = c("Female" = "#E64B35", "Male" = "#4DBBD5"),
    labels = c("Female", "Male")
  ) +
  theme_minimal()

# Prepare one-row-per-person and keep only Female & Male
df_sjl_age <- df %>%
  group_by(person_id, sex_concept) %>%
  summarise(
    mean_age = mean(age_at_sleep, na.rm = TRUE),
    sjl_raw = first(SJL_raw)
  ) %>%
  ungroup() %>%
  filter(sex_concept %in% c("Female", "Male"))

# Compute 95% percentile range for cropping
y_limits <- quantile(df_sjl_age$sjl_raw, probs = c(0.025, 0.975), na.rm = TRUE)

# Plot
ggplot(df_sjl_age, aes(x = mean_age, y = sjl_raw, color = sex_concept)) +
  geom_jitter(alpha = 0.5, width = 0.3, height = 0, size = 1.5) +
  geom_smooth(
    method = mgcv::bam,
    formula = y ~ s(x, bs = "cs"),
    method.args = list(method = "fREML", discrete = TRUE),
    se = TRUE,
    size = 1
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray50") +
  coord_cartesian(ylim = y_limits) +  # zoom without affecting model fit
  labs(
    title = "Social Jetlag across the Lifespan",
    subtitle = "95% of observations shown (extreme outliers cropped)",
    x = "Age",
    y = "Social Jetlag (Weekend - Weekday Midpoint)",
    color = "Sex"
  ) +
  scale_color_manual(
    values = c("Female" = "#E64B35", "Male" = "#4DBBD5"),
    labels = c("Female", "Male")
  ) +
  theme_minimal() +
  theme(
    legend.position = "right",
    plot.title = element_text(face = "bold")
  )

library(dplyr)
library(ggplot2)
library(mgcv)
library(patchwork)
library(viridis)

# ---- Ensure Female/Male only (one row per person assumed already) ----
df_sjl_age <- df_sjl_age %>%
  filter(sex_concept %in% c("Female", "Male"))

# Consistent color palette
sex_colors <- c("Female" = "#E64B35", "Male" = "#4DBBD5")

# ---- 1) Raw Points + GAM ----
plot_sample <- df_sjl_age

p1 <- ggplot() +
  geom_jitter(data = plot_sample,
              aes(mean_age, sjl_raw, color = sex_concept),
              width = 0.25, height = 0,
              alpha = 0.35, size = 0.8) +
  geom_smooth(
    data = df_sjl_age,
    aes(mean_age, sjl_raw, color = sex_concept),
    method = mgcv::bam,
    formula = y ~ s(x, bs = "cs"),
    method.args = list(method = "fREML", discrete = TRUE),
    se = TRUE
  ) +
  scale_color_manual(values = sex_colors) +
  labs(title = "Raw Points + GAM",
       x = "Age", y = "Social Jetlag") +
  theme_minimal()

# ---- 2) Hexbin Density + GAM (Faceted) ----
p2 <- ggplot(df_sjl_age, aes(mean_age, sjl_raw)) +
  stat_bin_hex(bins = 40) +
  scale_fill_viridis_c(option = "magma", trans = "sqrt") +
  geom_smooth(
    aes(mean_age, sjl_raw, color = sex_concept),
    method = mgcv::bam,
    formula = y ~ s(x, bs = "cs"),
    method.args = list(method = "fREML", discrete = TRUE),
    se = FALSE
  ) +
  facet_wrap(~ sex_concept) +
  labs(title = "Hexbin Density + GAM",
       x = "Age", y = "Social Jetlag") +
  theme_minimal() +
  theme(legend.position = "none")

# ---- 3) Binned Mean ± 95% CI ----
df_bins <- df_sjl_age %>%
  mutate(age_bin = cut(mean_age, breaks = seq(20, 100, by = 5), right = FALSE)) %>%
  group_by(sex_concept, age_bin) %>%
  summarise(
    age_mid = mean(mean_age),
    mean_sjl = mean(sjl_raw),
    se = sd(sjl_raw)/sqrt(n()),
    .groups = "drop"
  )

p3 <- ggplot(df_bins, aes(age_mid, mean_sjl, color = sex_concept)) +
  geom_line(size = 1) +
  geom_ribbon(aes(ymin = mean_sjl - 1.96*se,
                  ymax = mean_sjl + 1.96*se,
                  fill = sex_concept),
              alpha = 0.2, color = NA) +
  scale_color_manual(values = sex_colors) +
  scale_fill_manual(values = sex_colors) +
  labs(title = "Binned Mean ± 95% CI",
       x = "Age (5-year bins)", y = "Mean Social Jetlag") +
  theme_minimal()

# ---- 4) 2D Density Contours + GAM ----
p4 <- ggplot(df_sjl_age, aes(mean_age, sjl_raw, color = sex_concept)) +
  geom_point(alpha = 0.15, size = 0.6) +
  geom_density_2d(size = 0.4) +
  geom_smooth(
    method = mgcv::bam,
    formula = y ~ s(x, bs = "cs"),
    method.args = list(method = "fREML", discrete = TRUE),
    se = TRUE
  ) +
  scale_color_manual(values = sex_colors) +
  labs(title = "2D Density Contours + GAM",
       x = "Age", y = "Social Jetlag") +
  theme_minimal()

# ---- Combine into one figure ----
combined_plot <- (p1 | p2) /
                 (p3 | p4) +
  plot_annotation(
    title = "Social Jetlag Across the Lifespan: Visualization Comparison",
    subtitle = "Female and Male Participants",
    theme = theme(plot.title = element_text(face = "bold", size = 16))
  )

combined_plot

df_sample$daily_sleep_window_hours = df_sample$daily_sleep_window_mins / 60

p_seas_offset <- plot_seasonal_trend(df_sample, "daily_sleep_window_hours", "Seasonal Trend: Duration")
print(p_seas_offset)

# Seasonal plots showing variation over day of year
# Using geom_smooth to show trends

plot_seasonal_trend <- function(data, y_var, title) {
  ggplot(data, aes(x = day_of_year, y = .data[[y_var]])) +
    geom_smooth(
      method = mgcv::bam,
      formula = y ~ s(x, bs = "cc"),
      method.args = list(method = "fREML", discrete = TRUE),
      color = "blue"
    ) +
    labs(title = title, x = "Day of Year", y = "Hour") +
    theme_minimal()
}

# Midpoint
p_seas_mid <- plot_seasonal_trend(df_sample, "daily_midpoint_hour", "Seasonal Trend: Sleep Midpoint")
print(p_seas_mid)

# Onset
p_seas_onset <- plot_seasonal_trend(df_sample, "daily_start_hour", "Seasonal Trend: Sleep Onset")
print(p_seas_onset)

# Offset
p_seas_offset <- plot_seasonal_trend(df_sample, "daily_end_hour", "Seasonal Trend: Sleep Offset")
print(p_seas_offset)

#
p_seas_offset <- plot_seasonal_trend(df_sample, "daily_end_hour", "Seasonal Trend: Sleep Offset")
print(p_seas_offset)

library(dplyr)
library(ggplot2)
library(mgcv)

# -----------------------------
# Settings & helpers
# -----------------------------
SHIFT <- 12  # hours; centres the axis around noon so midnight-crossing averages correctly

shift_hour    <- function(x) ((x + SHIFT) %% 24)
unshift_hour  <- function(x) ((x - SHIFT) %% 24)

week_from_day <- function(doy) pmin(52L, ceiling(as.integer(doy) / 7L))

# Convert shifted numeric hours back to "hh:mm" labels for the y-axis
hour_label_fun <- function(b) {
  h <- unshift_hour(b) %% 24          # back to real clock hours (0–23.999)
  hh <- floor(h)
  mm <- round((h - hh) * 60)
  sprintf("%02d:%02d", hh, mm)
}

# Format decimal hours as "Xh Ym" for duration plots
duration_label_fun <- function(b) {
  h  <- floor(b)
  m  <- round((b - h) * 60)
  ifelse(m == 0, sprintf("%dh", h), sprintf("%dh %dm", h, m))
}

# Month tick positions in week-space (week of the 1st of each month, approx.)
month_week_breaks <- c(1, 5, 9, 14, 18, 23, 27, 32, 36, 40, 45, 49)
month_labels      <- c("Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec")

col_weekly <- "#2b8cbe"   # blue for points / bars / ribbon
col_gam    <- "black"     # GAM trend line

# -----------------------------
# Main plotting function
# -----------------------------
plot_weekly_binned <- function(df, y_var, title, k = 8, label_type = c('clock', 'duration')) {

  # --- 1. Prepare data ---
  label_type <- match.arg(label_type)
  is_duration <- label_type == "duration"

  df2 <- df %>%
    filter(!is.na(.data[[y_var]]), !is.na(day_of_year)) %>%
    mutate(
      week    = week_from_day(day_of_year),
      # Duration values are not circular so skip the midnight-crossing shift
      shifted = if (is_duration) .data[[y_var]] else shift_hour(.data[[y_var]])
    )

  # --- 2. Weekly aggregation ---
  weekly <- df2 %>%
    group_by(week) %>%
    summarise(
      n      = n(),
      mean_y = mean(shifted, na.rm = TRUE),
      se_y   = sd(shifted, na.rm = TRUE) / sqrt(n),
      lower  = mean_y - 1.96 * se_y,
      upper  = mean_y + 1.96 * se_y,
      .groups = "drop"
    ) %>%
    arrange(week)

  # --- 3. Cyclic GAM on raw daily data ---
  # Fitting to individual nights rather than weekly means means the SE (and ribbon)
  # reflects genuine residual variation in daily sleep, not just uncertainty about
  # the weekly mean. k is capped at 20 — enough flexibility for a seasonal curve
  # without overfitting day-to-day noise.
  k_use   <- min(k, 20)
  gam_fit <- mgcv::bam(
    shifted ~ s(week, bs = "cc", k = k_use),
    data = df2,
    method = "fREML",
    discrete = TRUE
  )

  # Dense prediction grid
  pred_grid <- data.frame(week = seq(1, 52, length.out = 300))
  pred      <- predict(gam_fit, newdata = pred_grid, se.fit = TRUE)
  pred_grid <- pred_grid %>% mutate(
    fit   = pred$fit,
    lower = pred$fit - 1.96 * pred$se.fit,
    upper = pred$fit + 1.96 * pred$se.fit
  )

  # --- 4. Zoom window ---
  # Use IQR-based bounds on weekly means to ignore outlier weeks (e.g. a single
  # poorly-sampled week pulling the window open). Extend by 1.5*IQR each side,
  # with a hard floor of 15 min (0.25 h) padding.
  iqr      <- IQR(weekly$mean_y, na.rm = TRUE)
  q25      <- quantile(weekly$mean_y, 0.25, na.rm = TRUE)
  q75      <- quantile(weekly$mean_y, 0.75, na.rm = TRUE)
  padding  <- max(0.25, 1.5 * iqr)          # at least 15 min each side
  ylim_raw <- c(q25 - padding, q75 + padding)
  # For duration, don't clamp to clock range — just ensure non-negative
  ylim <- if (is_duration) c(max(ylim_raw[1], 0), ylim_raw[2])
          else              c(max(ylim_raw[1], 0), min(ylim_raw[2], 24))

  # --- 5. Build plot ---
  p <- ggplot() +

    # GAM 95 % CI ribbon (drawn first so it sits behind points)
    geom_ribbon(
      data = pred_grid,
      aes(x = week, ymin = lower, ymax = upper, fill = "GAM 95% CI"),
      alpha = 0.20
    ) +

    # GAM trend line
    geom_line(
      data = pred_grid,
      aes(x = week, y = fit, colour = "GAM Trend"),
      linewidth = 1.2
    ) +

    # Weekly mean ± 95 % CI error bars
    geom_errorbar(
      data = weekly,
      aes(x = week, ymin = lower, ymax = upper, colour = "Weekly Mean"),
      width = 0.4, alpha = 0.8, linewidth = 0.7
    ) +

    # Weekly mean points
    geom_point(
      data = weekly,
      aes(x = week, y = mean_y, colour = "Weekly Mean"),
      size = 2.2, alpha = 0.95
    ) +

    # Axes
    coord_cartesian(ylim = ylim) +
    scale_y_continuous(
      labels       = if (is_duration) duration_label_fun else hour_label_fun,
      breaks       = seq(floor(ylim[1]),  ceiling(ylim[2]),  by = 0.25),  # every 15 min
      minor_breaks = seq(floor(ylim[1]),  ceiling(ylim[2]),  by = 0.25 / 3)  # every 5 min
    ) +
    scale_x_continuous(
      breaks = month_week_breaks,
      labels = month_labels,
      expand = expansion(mult = 0.01)
    ) +

    # Colour scale — explicit breaks fixes override.aes ordering
    scale_colour_manual(
      name   = NULL,
      breaks = c("GAM Trend", "Weekly Mean"),   # controls key order; override.aes matches this
      values = c("Weekly Mean" = col_weekly, "GAM Trend" = col_gam),
      guide  = guide_legend(
        override.aes = list(
          linetype  = c("solid", "blank"),   # GAM Trend = line, Weekly Mean = point only
          shape     = c(NA,       16),
          linewidth = c(1.2,      0)
        )
      )
    ) +

    # Fill scale for ribbon — merged into same legend via same name
    scale_fill_manual(
      name   = NULL,
      values = c("GAM 95% CI" = col_weekly)
    ) +

    labs(
      title = title,
      x     = "Month",
      y     = if (is_duration) "Sleep Duration" else "Clock Time"
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = "bold"),
      legend.position  = "top",
      panel.grid.minor = element_line(colour = "grey93")
    )

  list(plot = p, weekly = weekly, gam = gam_fit, pred = pred_grid)
}

# -----------------------------
# Create the four figures
# -----------------------------
res_mid      <- plot_weekly_binned(df_sample, "daily_midpoint_hour",       "Weekly Seasonal Trend: Sleep Midpoint")
res_onset    <- plot_weekly_binned(df_sample, "daily_start_hour",          "Weekly Seasonal Trend: Sleep Onset")
res_offset   <- plot_weekly_binned(df_sample, "daily_end_hour",            "Weekly Seasonal Trend: Sleep Offset")

df_sample$daily_sleep_window_hours <- df_sample$daily_sleep_window_mins / 60
res_duration <- plot_weekly_binned(df_sample, "daily_sleep_window_hours",  "Weekly Seasonal Trend: Sleep Duration",
                                   label_type = "duration")

# -----------------------------
# Display inline (notebook)
# -----------------------------
print(res_mid$plot)
print(res_onset$plot)
print(res_offset$plot)
print(res_duration$plot)

# -----------------------------
# Save
# -----------------------------
ggsave("weekly_midpoint.png", res_mid$plot,      width = 9, height = 5, dpi = 300)
ggsave("weekly_onset.png",    res_onset$plot,    width = 9, height = 5, dpi = 300)
ggsave("weekly_offset.png",   res_offset$plot,   width = 9, height = 5, dpi = 300)
ggsave("weekly_duration.png", res_duration$plot, width = 9, height = 5, dpi = 300)

message("Saved: weekly_midpoint.png, weekly_onset.png, weekly_offset.png, weekly_duration.png")

# Plot of Sleep Regularity (Standard Deviation) vs. Age
# Calculate SD of daily_start_hour per person
# Handle midnight crossing: if hour < 12, add 24

df_variability <- df %>%
  mutate(
    start_hour_adj = ifelse(daily_start_hour < 12, daily_start_hour + 24, daily_start_hour)
  ) %>%
  group_by(person_id, sex_concept) %>%
  summarise(
    mean_age = mean(age_at_sleep, na.rm = TRUE),
    start_sd = sd(start_hour_adj, na.rm = TRUE),
    n_nights = n()
  ) %>%
  filter(n_nights >= 7) %>%
  ungroup()

ggplot(df_variability, aes(x = mean_age, y = start_sd)) +
  geom_hex(bins = 50) +
  geom_smooth(
    color = "red",
    method = mgcv::bam,
    formula = y ~ s(x, bs = "cs"),
    method.args = list(method = "fREML", discrete = TRUE)
  ) +
  labs(
    title = "Sleep Variability (Consistency) vs Age",
    subtitle = "Standard Deviation of Sleep Onset",
    x = "Age",
    y = "SD of Sleep Onset (Hours)"
  ) +
  theme_minimal()

# Calculate duration if not present (assuming end - start, handling day wrap if needed)
# Simple approximation: (end - start) %% 24
df_sample <- df_sample %>%
  mutate(duration = (daily_end_hour - daily_start_hour) %% 24)

ggplot(df_sample, aes(x = age_at_sleep, y = duration)) +
  geom_hex(bins = 50) +
  geom_smooth(color = "red") +
  labs(title = "Sleep Duration vs Age", x = "Age", y = "Sleep Duration (Hours)") +
  theme_minimal()

# Updated: only GAM (on person-level data) + 5-year binned means with ±1 SD; legend fixed.
library(dplyr)
library(ggplot2)
library(mgcv)
library(scales)

col_weekly <- "#2b8cbe"   # blue for points / bars / ribbon
col_gam    <- "black"     # GAM trend line

duration_label_fun <- function(b) {
  h  <- floor(b)
  m  <- round((b - h) * 60)
  ifelse(m == 0, sprintf("%dh", h), sprintf("%dh %dm", h, m))
}

# --- Prepare person-level summaries ---

# 1) Sleep onset variability (SD per person)
df_variability <- df %>%
  filter(!is.na(daily_start_hour), !is.na(age_at_sleep)) %>%
  mutate(start_hour_adj = ifelse(daily_start_hour < 12, daily_start_hour + 24, daily_start_hour)) %>%
  group_by(person_id, sex_concept) %>%
  summarise(
    mean_age = mean(age_at_sleep, na.rm = TRUE),
    start_sd  = sd(start_hour_adj, na.rm = TRUE),
    n_nights  = n(),
    .groups = "drop"
  ) %>%
  filter(n_nights >= 7, !is.na(start_sd), is.finite(start_sd))

# 2) Mean duration per person (compute nightly durations first if needed)
df_nightly <- df_sample %>%
  filter(!is.na(daily_start_hour), !is.na(daily_end_hour), !is.na(age_at_sleep)) %>%
  mutate(nightly_duration = (daily_end_hour - daily_start_hour) %% 24)

df_person_duration <- df_nightly %>%
  group_by(person_id) %>%
  summarise(
    mean_age = mean(age_at_sleep, na.rm = TRUE),
    mean_duration = mean(nightly_duration, na.rm = TRUE),
    n_nights = n(),
    .groups = "drop"
  ) %>%
  filter(n_nights >= 7, !is.na(mean_duration), is.finite(mean_duration))

# --- Helpers for binning ---
make_age_bins <- function(age_vector, bin_width = 5) {
  min_age <- floor(min(age_vector, na.rm = TRUE))
  p95_age <- quantile(age_vector, 0.95, na.rm = TRUE)
  max_bin <- ceiling(p95_age / bin_width) * bin_width
  breaks  <- seq(min_age, max_bin, by = bin_width)
  if (length(breaks) < 2) breaks <- seq(min_age, min_age + bin_width, by = bin_width)
  breaks
}

compute_bin_mid <- function(age_factor) {
  labs <- as.character(age_factor)
  sapply(labs, function(L) {
    nums <- as.numeric(unlist(regmatches(L, gregexpr("[0-9]+\\.?[0-9]*", L))))
    if (length(nums) >= 2) mean(nums[1:2]) else NA_real_
  })
}

# --- Plot function: GAM on person-level points, binned means ± SD, tidy legend ---
plot_age_binned_sd <- function(df_person, age_col, y_col, title, y_label,
                               bin_width = 5, k = 8, is_duration = FALSE, out_file = NULL) {

  # Fit GAM on person-level data (captures overall trend)
  k_use <- min(k, 20)
  fml <- as.formula(paste0(y_col, " ~ s(", age_col, ", bs = 'tp', k = ", k_use, ")"))
  gam_fit <- mgcv::bam(fml, data = df_person, method = "fREML", discrete = TRUE)

  # prediction grid across min..p95
  min_age <- min(df_person[[age_col]], na.rm = TRUE)
  p95_age  <- as.numeric(quantile(df_person[[age_col]], 0.95, na.rm = TRUE))
  x_grid  <- seq(min_age, p95_age, length.out = 300)
  newdata_df <- setNames(data.frame(x_grid), age_col)
  pred <- predict(gam_fit, newdata = newdata_df, se.fit = TRUE)
  pred_df <- data.frame(
    age = x_grid,
    fit = pred$fit,
    lower = pred$fit - 1.96 * pred$se.fit,
    upper = pred$fit + 1.96 * pred$se.fit
  )

  # Binned summary: compute mean and SD (we will show ±1 SD)
  breaks <- make_age_bins(df_person[[age_col]], bin_width)
  df_binned <- df_person %>%
    mutate(age_bin = cut(.data[[age_col]], breaks = c(breaks, Inf), right = FALSE)) %>%
    group_by(age_bin) %>%
    summarise(
      bin_n = n(),
      mean_age = mean(.data[[age_col]], na.rm = TRUE),
      mean_y  = mean(.data[[y_col]], na.rm = TRUE),
      sd_y    = sd(.data[[y_col]], na.rm = TRUE),
      lower   = mean_y - sd_y,
      upper   = mean_y + sd_y,
      .groups = "drop"
    ) %>%
    mutate(age_bin_mid = compute_bin_mid(age_bin)) %>%
    mutate(alpha_flag = ifelse(bin_n < 5, 0.5, 1))

  # --- Plot: NO individual dots, only GAM + ribbon + binned means ± SD ---
  p <- ggplot() +
    # GAM ribbon (95% CI)
    geom_ribbon(data = pred_df, aes(x = age, ymin = lower, ymax = upper, fill = "GAM 95% CI"), alpha = 0.20) +
    # GAM trend
    geom_line(data = pred_df, aes(x = age, y = fit, colour = "GAM Trend"), linewidth = 1.2) +
    # Binned mean errorbars = ±1 SD
    geom_errorbar(data = df_binned, aes(x = age_bin_mid, ymin = lower, ymax = upper, colour = "Binned Mean ± SD"),
                  width = bin_width * 0.6, linewidth = 0.9, alpha = 0.95) +
    # Binned mean points
    geom_point(data = df_binned, aes(x = age_bin_mid, y = mean_y, colour = "Binned Mean ± SD", alpha = alpha_flag),
               size = 3, show.legend = FALSE) +

    # Colour & fill scales — put everything under one legend title "Legend" and control order
    scale_colour_manual(
      name   = "Legend",
      breaks = c("GAM Trend", "Binned Mean ± SD"),
      values = c("Binned Mean ± SD" = col_weekly, "GAM Trend" = col_gam)
    ) +
    scale_fill_manual(
      name   = "Legend",
      breaks = "GAM 95% CI",
      values = c("GAM 95% CI" = col_weekly)
    ) +

    # Tidy guides: ensure the ribbon shows as a rectangle, line shows as line, binned mean shows as point
    guides(
      colour = guide_legend(order = 1, override.aes = list(
        linetype = c("solid", "blank"),
        shape    = c(NA, 16),
        size     = c(1.2, 3)
      )),
      fill = guide_legend(order = 2, override.aes = list(
        alpha = 0.6, shape = 22, size = 5
      ))
    ) +

    labs(title = title, x = "Age", y = y_label) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title      = element_text(face = "bold"),
      legend.position = "top",
      panel.grid.minor = element_line(colour = "grey93")
    )

  # y formatting
  if (is_duration) {
    p <- p + scale_y_continuous(labels = duration_label_fun, breaks = pretty_breaks(n = 6))
  } else {
    p <- p + scale_y_continuous(breaks = pretty_breaks(n = 6))
  }

  p <- p + coord_cartesian(xlim = c(min_age, p95_age))

  if (!is.null(out_file)) ggsave(out_file, p, width = 9, height = 5, dpi = 300)

  list(plot = p, binned = df_binned, gam = gam_fit, pred = pred_df)
}

# --- Produce the two updated figures ---

res_variability <- plot_age_binned_sd(
  df_person = df_variability,
  age_col   = "mean_age",
  y_col     = "start_sd",
  title     = "Sleep Variability (SD of Onset) vs Age (5-y bins)",
  y_label   = "SD of Sleep Onset (hours)",
  bin_width = 5,
  k = 8,
  is_duration = FALSE,
  out_file = "age_variability_binned_sd.png"
)

res_duration_age <- plot_age_binned_sd(
  df_person = df_person_duration,
  age_col   = "mean_age",
  y_col     = "mean_duration",
  title     = "Mean Sleep Duration vs Age (5-y bins)",
  y_label   = "Mean Sleep Duration",
  bin_width = 5,
  k = 8,
  is_duration = TRUE,
  out_file = "age_duration_binned_sd.png"
)

# print to notebook / viewer
print(res_variability$plot)
print(res_duration_age$plot)

message("Saved: age_variability_binned_sd.png, age_duration_binned_sd.png")

# Boxplots of Sleep Metrics vs Employment Status
# Metrics: Onset, Offset, Duration, Variability

# Prepare person-level summary with all metrics
df_employment_metrics <- df %>%
  mutate(
    start_hour_adj = ifelse(daily_start_hour < 12, daily_start_hour + 24, daily_start_hour),
    duration = (daily_end_hour - daily_start_hour) %% 24
  ) %>%
  group_by(person_id, employment_status) %>%
  summarise(
    mean_onset = mean(start_hour_adj, na.rm = TRUE),
    mean_offset = mean(daily_end_hour, na.rm = TRUE),
    mean_duration = mean(duration, na.rm = TRUE),
    onset_variability = sd(start_hour_adj, na.rm = TRUE),
    n_nights = n()
  ) %>%
  filter(n_nights >= 7) %>%
  ungroup()

# Helper for boxplots
plot_employment_boxplot <- function(data, y_var, title, y_label) {
  ggplot(data, aes(x = employment_status, y = .data[[y_var]], fill = employment_status)) +
    geom_boxplot() +
    coord_flip() +
    labs(title = title, x = "Employment Status", y = y_label) +
    theme_minimal() +
    theme(legend.position = "none")
}

# Onset
p_emp_onset <- plot_employment_boxplot(df_employment_metrics, "mean_onset", "Sleep Onset by Employment Status", "Mean Onset Hour (Adjusted >12pm)")
print(p_emp_onset)

# Offset
p_emp_offset <- plot_employment_boxplot(df_employment_metrics, "mean_offset", "Sleep Offset by Employment Status", "Mean Offset Hour")
print(p_emp_offset)

# Duration
p_emp_dur <- plot_employment_boxplot(df_employment_metrics, "mean_duration", "Sleep Duration by Employment Status", "Mean Duration (Hours)")
print(p_emp_dur)

# Variability
p_emp_var <- plot_employment_boxplot(df_employment_metrics, "onset_variability", "Sleep Variability by Employment Status", "SD of Onset (Hours)")
print(p_emp_var)

# weekly_binned_dst_split.R
# Modified plotting function that draws two seasonal GAM trends:
#  - one for ZIP3s that observe DST
#  - one for ZIP3s that do not observe DST
# If a ZIP3 is split (some rows flagged DST, some not), the majority rule is used.
# You can supply a vector of no-DST ZIP3 prefixes (e.g. c("850","967")).

library(dplyr)
library(ggplot2)
library(mgcv)
library(lubridate)

# -----------------------------
# Settings & helpers (copied from your original)
# -----------------------------
SHIFT <- 12  # hours; centres the axis around noon so midnight-crossing averages correctly

shift_hour    <- function(x) ((x + SHIFT) %% 24)
unshift_hour  <- function(x) ((x - SHIFT) %% 24)

week_from_day <- function(doy) pmin(52L, ceiling(as.integer(doy) / 7L))

# Convert shifted numeric hours back to "hh:mm" labels for the y-axis
hour_label_fun <- function(b) {
  h <- unshift_hour(b) %% 24          # back to real clock hours (0–23.999)
  hh <- floor(h)
  mm <- round((h - hh) * 60)
  sprintf("%02d:%02d", hh, mm)
}

# Format decimal hours as "Xh Ym" for duration plots
duration_label_fun <- function(b) {
  h  <- floor(b)
  m  <- round((b - h) * 60)
  ifelse(m == 0, sprintf("%dh", h), sprintf("%dh %dm", h, m))
}

# Month tick positions in week-space (week of the 1st of each month, approx.)
month_week_breaks <- c(1, 5, 9, 14, 18, 23, 27, 32, 36, 40, 45, 49)
month_labels      <- c("Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec")

col_weekly <- "#2b8cbe"   # blue for points / bars / ribbon (keeps one visual identity)
col_gam_dst_yes <- "#1b7837"   # green-ish for DST-observing GAM
col_gam_dst_no  <- "#7b3294"   # purple-ish for no-DST GAM

# -----------------------------
# Helper: build per-row DST flag using a hardcoded list of no-DST zip3 prefixes
# -----------------------------
# By default this contains the common no-DST regions: Arizona (most prefixes) and Hawaii.
# NOTE: This is only an example: edit/extend the vector to match your preferred mapping.
default_no_dst_zip3 <- c(
  # Arizona (major ranges) -- adapt as needed
  sprintf("%03d", 850:865),
  # Hawaii
  "967", "968",
  # U.S. territories (example prefixes) -- adjust/remove if not relevant
  "006", "007", "009",  # Puerto Rico (006-009)
  "969"                     # Guam/NMI
)
# make unique (character)
default_no_dst_zip3 <- unique(as.character(default_no_dst_zip3))

# normalize a ZIP or ZIP3 to a 3-digit prefix string
zip3_prefix <- function(x) {
  x_chr <- as.character(x)
  # if already 3-digit return, otherwise take first 3 characters of 5-digit zips
  sapply(x_chr, function(z) {
    if (is.na(z) || z == "") return(NA_character_)
    z_clean <- gsub("[^0-9]", "", z)
    if (nchar(z_clean) <= 3) sprintf("%03s", z_clean) else substr(z_clean, 1, 3)
  }, USE.NAMES = FALSE)
}

# -----------------------------
# Main plotting function (two-group version)
# -----------------------------
plot_weekly_binned_by_dst <- function(df, y_var, title,
                                      k = 8,
                                      label_type = c('clock', 'duration'),
                                      no_dst_zip3 = NULL,
                                      default_to_observes_dst = TRUE) {
  # label_type: 'clock' (circular) or 'duration' (not circular)
  label_type <- match.arg(label_type)
  is_duration <- label_type == 'duration'

  # --- Build per-row ZIP3 -> DST flag ---
  df <- df %>% mutate(.zip3 = zip3_prefix(zip3))

  if (is.null(no_dst_zip3)) {
    no_dst_zip3 <- default_no_dst_zip3
  }
  no_dst_zip3 <- as.character(no_dst_zip3)

  # per-row naive assignment: TRUE if observes DST, FALSE if in no_dst list
  df <- df %>% mutate(
    dst_assigned = case_when(
      is.na(.zip3) ~ NA, # unknown
      .zip3 %in% no_dst_zip3 ~ FALSE,
      TRUE ~ TRUE
    )
  )

  # If a ZIP3 has mixed dst_assigned values across rows, choose the majority
  zip3_majority <- df %>%
    group_by(.zip3) %>%
    summarise(
      n = n(),
      prop_true = mean(dst_assigned == TRUE, na.rm = TRUE),
      dst_majority = ifelse(is.nan(prop_true),
                            if (default_to_observes_dst) TRUE else FALSE,
                            prop_true >= 0.5),
      .groups = 'drop'
    )

  # merge majority back to df
  df <- df %>% left_join(zip3_majority %>% select(.zip3, dst_majority), by = '.zip3') %>%
    mutate(dst = dst_majority)

  # --- 1. Prepare data ---
  df2 <- df %>%
    filter(!is.na(.data[[y_var]]), !is.na(day_of_year), !is.na(dst)) %>%
    mutate(
      week    = week_from_day(day_of_year),
      shifted = if (is_duration) .data[[y_var]] else shift_hour(.data[[y_var]])
    )

  # --- 2. Weekly aggregation by dst group ---
  weekly <- df2 %>%
    group_by(dst, week) %>%
    summarise(
      n      = n(),
      mean_y = mean(shifted, na.rm = TRUE),
      se_y   = sd(shifted, na.rm = TRUE) / sqrt(n),
      lower  = mean_y - 1.96 * se_y,
      upper  = mean_y + 1.96 * se_y,
      .groups = 'drop'
    ) %>%
    arrange(dst, week)

  # --- 3. Fit cyclic GAM separately for the two groups ---
  k_use <- min(k, 20)

  fit_for_group <- function(df_group) {
    # if there are too few distinct weeks, skip fitting and return NULL
    if (nrow(df_group) < 6 || length(unique(df_group$week)) < 4) return(NULL)
    mgcv::bam(shifted ~ s(week, bs = 'cc', k = k_use), data = df_group, method = 'fREML', discrete = TRUE)
  }

  gam_yes <- fit_for_group(df2 %>% filter(dst == TRUE))
  gam_no  <- fit_for_group(df2 %>% filter(dst == FALSE))

  # Prediction grid for both groups
  pred_grid <- expand.grid(week = seq(1, 52, length.out = 300), dst = c(TRUE, FALSE))
  pred_grid$fit <- NA_real_
  pred_grid$lower <- NA_real_
  pred_grid$upper <- NA_real_

  if (!is.null(gam_yes)) {
    pr <- predict(gam_yes, newdata = pred_grid %>% filter(dst == TRUE), se.fit = TRUE)
    idx <- which(pred_grid$dst == TRUE)
    pred_grid$fit[idx]   <- pr$fit
    pred_grid$lower[idx] <- pr$fit - 1.96 * pr$se.fit
    pred_grid$upper[idx] <- pr$fit + 1.96 * pr$se.fit
  }
  if (!is.null(gam_no)) {
    pr <- predict(gam_no, newdata = pred_grid %>% filter(dst == FALSE), se.fit = TRUE)
    idx <- which(pred_grid$dst == FALSE)
    pred_grid$fit[idx]   <- pr$fit
    pred_grid$lower[idx] <- pr$fit - 1.96 * pr$se.fit
    pred_grid$upper[idx] <- pr$fit + 1.96 * pr$se.fit
  }

  # --- 4. Zoom window (computed using all weekly means pooled) ---
  weekly_pooled <- weekly %>% group_by(week) %>% summarise(mean_y = mean(mean_y, na.rm = TRUE), .groups = 'drop')
  iqr      <- IQR(weekly_pooled$mean_y, na.rm = TRUE)
  q25      <- quantile(weekly_pooled$mean_y, 0.25, na.rm = TRUE)
  q75      <- quantile(weekly_pooled$mean_y, 0.75, na.rm = TRUE)
  padding  <- max(0.25, 1.5 * iqr)
  ylim_raw <- c(q25 - padding, q75 + padding)
  ylim <- if (is_duration) c(max(ylim_raw[1], 0), ylim_raw[2]) else c(max(ylim_raw[1], 0), min(ylim_raw[2], 24))

  # --- 5. Build plot ---
  p <- ggplot() +
    # ribbons (drawn first)
    geom_ribbon(
      data = pred_grid %>% filter(dst == TRUE & !is.na(fit)),
      aes(x = week, ymin = lower, ymax = upper, fill = 'DST: yes'),
      alpha = 0.18
    ) +
    geom_ribbon(
      data = pred_grid %>% filter(dst == FALSE & !is.na(fit)),
      aes(x = week, ymin = lower, ymax = upper, fill = 'DST: no'),
      alpha = 0.18
    ) +

    # GAM lines
    geom_line(
      data = pred_grid %>% filter(dst == TRUE & !is.na(fit)),
      aes(x = week, y = fit, colour = 'DST: yes'),
      linewidth = 1.2
    ) +
    geom_line(
      data = pred_grid %>% filter(dst == FALSE & !is.na(fit)),
      aes(x = week, y = fit, colour = 'DST: no'),
      linewidth = 1.2
    ) +

    # Weekly means + error bars for both groups
    geom_errorbar(
      data = weekly,
      aes(x = week, ymin = lower, ymax = upper, colour = ifelse(dst, 'DST: yes', 'DST: no')),
      width = 0.4, alpha = 0.9, linewidth = 0.7
    ) +
    geom_point(
      data = weekly,
      aes(x = week, y = mean_y, colour = ifelse(dst, 'DST: yes', 'DST: no')),
      size = 2.2, alpha = 0.95
    ) +

    # Axes and scales
    coord_cartesian(ylim = ylim) +
    scale_y_continuous(
      labels       = if (is_duration) duration_label_fun else hour_label_fun,
      breaks       = seq(floor(ylim[1]),  ceiling(ylim[2]),  by = 0.25),
      minor_breaks = seq(floor(ylim[1]),  ceiling(ylim[2]),  by = 0.25 / 3)
    ) +
    scale_x_continuous(
      breaks = month_week_breaks,
      labels = month_labels,
      expand = expansion(mult = 0.01)
    ) +

    scale_colour_manual(
      name   = NULL,
      values = c('DST: yes' = col_gam_dst_yes, 'DST: no' = col_gam_dst_no)
    ) +
    scale_fill_manual(
      name   = NULL,
      values = c('DST: yes' = col_gam_dst_yes, 'DST: no' = col_gam_dst_no)
    ) +

    labs(
      title = title,
      x     = 'Month',
      y     = if (is_duration) 'Sleep Duration' else 'Clock Time'
    ) +
    theme_minimal(base_size = 12) +
    theme(
      plot.title       = element_text(face = 'bold'),
      legend.position  = 'top',
      panel.grid.minor = element_line(colour = 'grey93')
    )

  list(plot = p, weekly = weekly, gam_yes = gam_yes, gam_no = gam_no, pred = pred_grid)
}




no_dst_zip3 <- c(
  "850","851","852","853","855","856","857","859","860","863","864","865",
  "967","968",
  "006","007","008","009",
  "969"
)

res <- plot_weekly_binned_by_dst(df_sample, 'daily_midpoint_hour',
                           'Weekly Seasonal Trend: Midpoint by DST',
                           k = 12,
                           label_type = 'clock',
                           no_dst_zip3 = no_dst_zip3)
print(res$plot)


# improved_weekly_binned_dst_plots.R
# Full script: improved plotting for DST vs non-DST ZIP3 groups
#
# Requirements:
#   install.packages(c("dplyr","ggplot2","mgcv","lubridate"))
#
library(dplyr)
library(ggplot2)
library(mgcv)
library(lubridate)

# -----------------------------
# Settings & helpers
# -----------------------------
SHIFT <- 12  # hours; centres the axis around noon so midnight-crossing averages correctly

shift_hour    <- function(x) ((x + SHIFT) %% 24)
unshift_hour  <- function(x) ((x - SHIFT) %% 24)

week_from_day <- function(doy) pmin(52L, ceiling(as.integer(doy) / 7L))

# Convert shifted numeric hours back to "hh:mm" labels for the y-axis
hour_label_fun <- function(b) {
  h <- unshift_hour(b) %% 24          # back to real clock hours (0–23.999)
  hh <- floor(h)
  mm <- round((h - hh) * 60)
  sprintf("%02d:%02d", hh, mm)
}

# Format decimal hours as "Xh Ym" for duration plots
duration_label_fun <- function(b) {
  h  <- floor(b)
  m  <- round((b - h) * 60)
  ifelse(m == 0, sprintf("%dh", h), sprintf("%dh %dm", h, m))
}

# Month tick positions in week-space (week of the 1st of each month, approx.)
month_week_breaks <- c(1, 5, 9, 14, 18, 23, 27, 32, 36, 40, 45, 49)
month_labels      <- c("Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec")

col_weekly <- "#2b8cbe"   # blue for points / bars / ribbon (keeps one visual identity)
col_gam_dst_yes <- "#1b7837"   # green-ish for DST-observing GAM
col_gam_dst_no  <- "#7b3294"   # purple-ish for no-DST GAM

# -----------------------------
# Helper: hardcoded no-DST ZIP3 list (sensible default; not perfect)
# -----------------------------
default_no_dst_zip3 <- unique(c(
  sprintf("%03d", 850:865),  # Arizona
  "967","968",               # Hawaii
  "006","007","008","009",   # Puerto Rico + USVI
  "969"                      # Guam / Northern Mariana Islands
))

# normalize a ZIP or ZIP3 to a 3-digit prefix string
zip3_prefix <- function(x) {
  x_chr <- as.character(x)
  sapply(x_chr, function(z) {
    if (is.na(z) || z == "") return(NA_character_)
    z_clean <- gsub("[^0-9]", "", z)
    if (nchar(z_clean) == 0) return(NA_character_)
    if (nchar(z_clean) <= 3) sprintf("%03s", z_clean) else substr(z_clean, 1, 3)
  }, USE.NAMES = FALSE)
}

# -----------------------------
# Main plotting function (improved, less-cluttered display)
# -----------------------------
plot_weekly_binned_by_dst <- function(df, y_var, title,
                                     k = 8,
                                     label_type = c('clock', 'duration'),
                                     no_dst_zip3 = NULL,
                                     default_to_observes_dst = TRUE,
                                     min_n_for_errorbar = 8L,
                                     point_size_range = c(1.2, 4.0),
                                     dodge_width = 0.8) {
  label_type <- match.arg(label_type)
  is_duration <- label_type == 'duration'

  # Build per-row ZIP3 -> DST flag
  df <- df %>% mutate(.zip3 = zip3_prefix(zip3))

  if (is.null(no_dst_zip3)) no_dst_zip3 <- default_no_dst_zip3
  no_dst_zip3 <- as.character(no_dst_zip3)

  df <- df %>% mutate(
    dst_assigned = case_when(
      is.na(.zip3) ~ NA,
      .zip3 %in% no_dst_zip3 ~ FALSE,
      TRUE ~ TRUE
    )
  )

  # majority rule per zip3
  zip3_majority <- df %>%
    group_by(.zip3) %>%
    summarise(
      n = n(),
      prop_true = mean(dst_assigned == TRUE, na.rm = TRUE),
      dst_majority = ifelse(is.nan(prop_true),
                            if (default_to_observes_dst) TRUE else FALSE,
                            prop_true >= 0.5),
      .groups = "drop"
    )

  df <- df %>% left_join(zip3_majority %>% select(.zip3, dst_majority), by = ".zip3") %>%
    mutate(dst = dst_majority)

  # Prepare data
  df2 <- df %>%
    filter(!is.na(.data[[y_var]]), !is.na(day_of_year), !is.na(dst)) %>%
    mutate(
      week = week_from_day(day_of_year),
      shifted = if (is_duration) .data[[y_var]] else shift_hour(.data[[y_var]])
    )

  # Weekly aggregation by dst group
  weekly <- df2 %>%
    group_by(dst, week) %>%
    summarise(
      n = n(),
      mean_y = mean(shifted, na.rm = TRUE),
      se_y = sd(shifted, na.rm = TRUE) / sqrt(n),
      lower = mean_y - 1.96 * se_y,
      upper = mean_y + 1.96 * se_y,
      .groups = "drop"
    ) %>%
    arrange(dst, week)

  # Fit cyclic GAM separately for the two groups
  k_use <- min(k, 20)
  fit_for_group <- function(df_group) {
    if (nrow(df_group) < 6 || length(unique(df_group$week)) < 4) return(NULL)
    mgcv::bam(shifted ~ s(week, bs = "cc", k = k_use), data = df_group, method = "fREML", discrete = TRUE)
  }
  gam_yes <- fit_for_group(df2 %>% filter(dst == TRUE))
  gam_no  <- fit_for_group(df2 %>% filter(dst == FALSE))

  # Prediction grid for both groups
  pred_grid <- expand.grid(week = seq(1, 52, length.out = 300), dst = c(TRUE, FALSE))
  pred_grid$fit <- NA_real_
  pred_grid$lower <- NA_real_
  pred_grid$upper <- NA_real_

  if (!is.null(gam_yes)) {
    pr <- predict(gam_yes, newdata = pred_grid %>% filter(dst == TRUE), se.fit = TRUE)
    idx <- which(pred_grid$dst == TRUE)
    pred_grid$fit[idx]   <- pr$fit
    pred_grid$lower[idx] <- pr$fit - 1.96 * pr$se.fit
    pred_grid$upper[idx] <- pr$fit + 1.96 * pr$se.fit
  }
  if (!is.null(gam_no)) {
    pr <- predict(gam_no, newdata = pred_grid %>% filter(dst == FALSE), se.fit = TRUE)
    idx <- which(pred_grid$dst == FALSE)
    pred_grid$fit[idx]   <- pr$fit
    pred_grid$lower[idx] <- pr$fit - 1.96 * pr$se.fit
    pred_grid$upper[idx] <- pr$fit + 1.96 * pr$se.fit
  }

  # Zoom window computed from pooled weekly means
  weekly_pooled <- weekly %>% group_by(week) %>% summarise(mean_y = mean(mean_y, na.rm = TRUE), .groups = "drop")
  iqr <- IQR(weekly_pooled$mean_y, na.rm = TRUE)
  q25 <- quantile(weekly_pooled$mean_y, 0.25, na.rm = TRUE)
  q75 <- quantile(weekly_pooled$mean_y, 0.75, na.rm = TRUE)
  padding <- max(0.25, 1.5 * iqr)
  ylim_raw <- c(q25 - padding, q75 + padding)
  ylim <- if (is_duration) c(max(ylim_raw[1], 0), ylim_raw[2]) else c(max(ylim_raw[1], 0), min(ylim_raw[2], 24))

  # Build improved plot: dodge points, hide errorbars for small n, size ~ sqrt(n)
  weekly_plot_points <- weekly %>% mutate(group = ifelse(dst, "DST: yes", "DST: no"))
  weekly_errorbars    <- weekly_plot_points %>% filter(n >= min_n_for_errorbar)

  pos <- position_dodge(width = dodge_width)

  p <- ggplot() +

    # GAM ribbons
    geom_ribbon(
      data = pred_grid %>% filter(dst == TRUE & !is.na(fit)),
      aes(x = week, ymin = lower, ymax = upper),
      fill = col_gam_dst_yes, alpha = 0.12
    ) +
    geom_ribbon(
      data = pred_grid %>% filter(dst == FALSE & !is.na(fit)),
      aes(x = week, ymin = lower, ymax = upper),
      fill = col_gam_dst_no, alpha = 0.10
    ) +

    # GAM trend lines (distinct linetypes)
    geom_line(
      data = pred_grid %>% filter(dst == TRUE & !is.na(fit)),
      aes(x = week, y = fit),
      colour = col_gam_dst_yes, linewidth = 1.15, linetype = "solid"
    ) +
    geom_line(
      data = pred_grid %>% filter(dst == FALSE & !is.na(fit)),
      aes(x = week, y = fit),
      colour = col_gam_dst_no, linewidth = 1.15, linetype = "dashed"
    ) +

    # Weekly mean points (dodge to separate groups), size ~ sqrt(n)
    geom_point(
      data = weekly_plot_points,
      aes(x = week, y = mean_y, colour = group, size = sqrt(n)),
      position = pos, stroke = 0.2, alpha = 0.95
    ) +

    # Weekly errorbars only for reasonably-sampled weeks
    geom_errorbar(
      data = weekly_errorbars,
      aes(x = week, ymin = lower, ymax = upper, colour = ifelse(dst, "DST: yes", "DST: no")),
      width = 0.35, position = pos, linewidth = 0.6, alpha = 0.9
    ) +

    # Small marker for weeks with too few samples to show errorbar
    geom_point(
      data = weekly_plot_points %>% filter(n < min_n_for_errorbar),
      aes(x = week, y = mean_y, colour = group),
      position = pos, size = 1.2, alpha = 0.7
    ) +

    coord_cartesian(ylim = ylim) +
    scale_y_continuous(
      labels       = if (is_duration) duration_label_fun else hour_label_fun,
      breaks       = seq(floor(ylim[1]),  ceiling(ylim[2]),  by = 0.25),
      minor_breaks = NULL
    ) +
    scale_x_continuous(
      breaks = month_week_breaks,
      labels = month_labels,
      expand = expansion(mult = c(0.01, 0.01))
    ) +

    scale_colour_manual(
      name   = NULL,
      values = c('DST: yes' = col_gam_dst_yes, 'DST: no' = col_gam_dst_no)
    ) +
    scale_size_continuous(name = "√(n)", range = point_size_range, guide = "legend") +

    labs(
      title = title,
      x     = "Month",
      y     = if (is_duration) "Sleep Duration" else "Clock Time"
    ) +
    theme_minimal(base_size = 13) +
    theme(
      plot.title       = element_text(face = "bold", size = 14),
      legend.position  = "top",
      legend.box       = "horizontal",
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank()
    )

  list(plot = p, weekly = weekly, gam_yes = gam_yes, gam_no = gam_no, pred = pred_grid)
}

# -----------------------------
# Example usage
# -----------------------------
# Ensure df_sample exists and has the required columns: zip3, day_of_year, daily_midpoint_hour, etc.
# If your duration is in minutes, convert to hours:
# df_sample$daily_sleep_window_hours <- df_sample$daily_sleep_window_mins / 60

# Generate the four figures (midpoint, onset, offset, duration)
# Adjust k (smoothing basis size) if you want more/less flexibility in GAM
#
# Example (uncomment to run):
# res_mid <- plot_weekly_binned_by_dst(df_sample, "daily_midpoint_hour", "Weekly Seasonal Trend: Midpoint by DST", k = 12, label_type = "clock")
# res_onset <- plot_weekly_binned_by_dst(df_sample, "daily_start_hour", "Weekly Seasonal Trend: Onset by DST", k = 10, label_type = "clock")
# res_offset <- plot_weekly_binned_by_dst(df_sample, "daily_end_hour", "Weekly Seasonal Trend: Offset by DST", k = 10, label_type = "clock")
# df_sample$daily_sleep_window_hours <- df_sample$daily_sleep_window_mins / 60
# res_duration <- plot_weekly_binned_by_dst(df_sample, "daily_sleep_window_hours", "Weekly Seasonal Trend: Duration by DST", k = 8, label_type = "duration")
#
# Then print or save:
# print(res_mid$plot)
# ggsave("weekly_midpoint_dst_improved.png", res_mid$plot, width = 9, height = 5, dpi = 300)

# -----------------------------
# End of script
# -----------------------------

res_mid <- plot_weekly_binned_by_dst(df_sample, "daily_midpoint_hour", "Weekly Seasonal Trend: Midpoint by DST", k = 12, label_type = "clock")
print(res_mid)

# Define no-DST ZIP3
no_dst_zip3 <- unique(c(
  sprintf("%03d", 850:865),
  "967","968",
  "006","007","008","009",
  "969"
))

df_sample %>%
  mutate(
    zip3_chr = substr(sprintf("%05d", as.numeric(zip3)), 1, 3),
    dst = !zip3_chr %in% no_dst_zip3
  ) %>%
  count(dst) %>%
  mutate(percent = n / sum(n) * 100)

# 1) helper to create 3-digit prefix (robust)
zip3_prefix <- function(x) {
  x_chr <- as.character(x)
  sapply(x_chr, function(z) {
    if (is.na(z) || z == "") return(NA_character_)
    z_clean <- gsub("[^0-9]", "", z)
    if (nchar(z_clean) == 0) return(NA_character_)
    if (nchar(z_clean) <= 3) sprintf("%03s", z_clean) else substr(z_clean, 1, 3)
  }, USE.NAMES = FALSE)
}

# 2) default no-DST ZIP3s (sensible but not perfect)
no_dst_zip3 <- unique(c(
  sprintf("%03d", 850:865),  # Arizona
  "967","968",               # Hawaii
  "006","007","008","009",   # Puerto Rico + USVI
  "969"                      # Guam / N. Mariana
))

# 3) add zip3 (3-digit) and dst flag to df_sample
df_sample2 <- df_sample %>%
  mutate(
    zip3_3 = zip3_prefix(zip3),
    dst = case_when(
      is.na(zip3_3) ~ NA,                # unknown
      zip3_3 %in% no_dst_zip3 ~ FALSE,   # no DST
      TRUE ~ TRUE                        # observes DST
    )
  )

# 4) overall counts by DST
df_sample2 %>%
  count(dst) %>%
  mutate(percent = n / sum(n) * 100) %>%
  print()

# 5) weekly sample-size summary (diagnostic for large errorbars)
df_sample2 %>%
  mutate(week = ceiling(day_of_year / 7)) %>%
  count(dst, week) %>%
  group_by(dst) %>%
  summarise(
    min_week_n    = min(n, na.rm = TRUE),
    p10_week_n    = as.integer(quantile(n, 0.10, na.rm = TRUE)),
    median_week_n = as.integer(median(n, na.rm = TRUE)),
    mean_week_n   = mean(n, na.rm = TRUE),
    max_week_n    = max(n, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  print()

# 6) weekly sd of your primary variable (replace daily_midpoint_hour if different)
df_sample2 %>%
  filter(!is.na(daily_midpoint_hour)) %>%
  mutate(week = ceiling(day_of_year / 7)) %>%
  group_by(dst, week) %>%
  summarise(
    sd_week = sd(daily_midpoint_hour, na.rm = TRUE),
    n = n(),
    .groups = "drop"
  ) %>%
  group_by(dst) %>%
  summarise(
    median_sd = median(sd_week, na.rm = TRUE),
    mean_sd   = mean(sd_week, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  print()

# --- Safer ZIP3 extraction (handles 3-digit or 5-digit properly) ---

zip3_prefix <- function(x) {
  x_chr <- as.character(x)
  sapply(x_chr, function(z) {
    if (is.na(z) || z == "") return(NA_character_)
    z_clean <- gsub("[^0-9]", "", z)
    if (nchar(z_clean) <= 3) sprintf("%03s", z_clean) else substr(z_clean, 1, 3)
  }, USE.NAMES = FALSE)
}

no_dst_zip3 <- unique(c(
  sprintf("%03d", 850:865),
  "967","968",
  "006","007","008","009",
  "969"
))

df_sample %>%
  mutate(
    zip3_chr = zip3_prefix(zip3),
    dst = !zip3_chr %in% no_dst_zip3
  ) %>%
  count(dst) %>%
  mutate(percent = n / sum(n) * 100)

dim(df_sample)
