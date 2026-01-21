# Install packages if needed (commented out)
# install.packages(c("tidyverse", "arrow", "gtsummary", "mgcv", "viridis"))

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

# Sample for plotting if data is too large
set.seed(123)
df_sample <- df %>%
  sample_n(min(50000, n()))

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

# Line plot of Social Jetlag Magnitude vs. Age (stratified by Sex)
# Using SJL_raw calculated in Phase 0
# We need one row per person with their mean age and SJL

df_sjl_age <- df %>%
  group_by(person_id, sex_concept) %>%
  summarise(
    mean_age = mean(age_at_sleep, na.rm = TRUE),
    sjl_raw = first(SJL_raw) # SJL_raw is constant per person from Phase 0
  ) %>%
  ungroup()

ggplot(df_sjl_age, aes(x = mean_age, y = sjl_raw, color = sex_concept)) +
  geom_smooth(method = "gam", formula = y ~ s(x, bs = "cs")) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "gray") +
  labs(
    title = "Social Jetlag across the Lifespan",
    subtitle = "Does the weekend effect disappear in retirement?",
    x = "Age",
    y = "Social Jetlag (Weekend - Weekday Midpoint)",
    color = "Sex"
  ) +
  theme_minimal()

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
  geom_smooth(color = "red", method = "gam") +
  labs(
    title = "Sleep Variability (Consistency) vs Age",
    subtitle = "Standard Deviation of Sleep Onset",
    x = "Age",
    y = "SD of Sleep Onset (Hours)"
  ) +
  theme_minimal()

# Seasonal plots showing variation over day of year
# Using geom_smooth to show trends

plot_seasonal_trend <- function(data, y_var, title) {
  ggplot(data, aes(x = day_of_year, y = .data[[y_var]])) +
    geom_smooth(method = "gam", formula = y ~ s(x, bs = "cc"), color = "blue") +
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

df_sample$daily_sleep_window_hours = df_sample$daily_sleep_window_mins / 60

p_seas_offset <- plot_seasonal_trend(df_sample, "daily_sleep_window_hours", "Seasonal Trend: Duration")
print(p_seas_offset)

# Calculate duration if not present (assuming end - start, handling day wrap if needed)
# Simple approximation: (end - start) %% 24
df_sample <- df_sample %>%
  mutate(duration = (daily_end_hour - daily_start_hour) %% 24)

ggplot(df_sample, aes(x = age_at_sleep, y = duration)) +
  geom_hex(bins = 50) +
  geom_smooth(color = "red") +
  labs(title = "Sleep Duration vs Age", x = "Age", y = "Sleep Duration (Hours)") +
  theme_minimal()

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


