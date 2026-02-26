circular_mean_time <- function(hours) {
  # Convert hours to radians (multiply by 2π/24)
  radians <- hours * 2 * pi / 24
  # Calculate mean direction
  mean_rad <- mean.circular(circular(radians))
  # Convert back to hours (divide by 2π/24)
  mean_hour <- mean_rad * 24 / (2 * pi)
  # Ensure result is between 0 and 24
  mean_hour <- mean_hour %% 24
  return(mean_hour)
}

# Function to calculate circular standard deviation
circular_sd_time <- function(hours) {
  # Convert hours to radians
  radians <- hours * 2 * pi / 24
  # Calculate circular SD
  sd_rad <- sd.circular(circular(radians))
  # Convert back to hours
  sd_hours <- sd_rad * 24 / (2 * pi)
  return(sd_hours)
}

# Helper function to get US holidays
get_us_holidays <- function(years) {
  holidays_list <- c()
  
  for(year in years) {
    # Major US holidays
    year_holidays <- c(
      # New Year's Day
      as.Date(paste(year, "01-01", sep="-")),
      
      # Martin Luther King Jr. Day (3rd Monday in January)
      as.Date(paste(year, "01-01", sep="-")) + days(14 + (2 - wday(as.Date(paste(year, "01-01", sep="-")))) %% 7),
      
      # Presidents' Day (3rd Monday in February)
      as.Date(paste(year, "02-01", sep="-")) + days(14 + (2 - wday(as.Date(paste(year, "02-01", sep="-")))) %% 7),
      
      # Memorial Day (last Monday in May)
      as.Date(paste(year, "06-01", sep="-")) - days(wday(as.Date(paste(year, "06-01", sep="-"))) - 2),
      
      # Independence Day
      as.Date(paste(year, "07-04", sep="-")),
      
      # Labor Day (1st Monday in September)
      as.Date(paste(year, "09-01", sep="-")) + days((2 - wday(as.Date(paste(year, "09-01", sep="-")))) %% 7),
      
      # Columbus Day (2nd Monday in October)
      as.Date(paste(year, "10-01", sep="-")) + days(7 + (2 - wday(as.Date(paste(year, "10-01", sep="-")))) %% 7),
      
      # Veterans Day
      as.Date(paste(year, "11-11", sep="-")),
      
      # Thanksgiving (4th Thursday in November)
      as.Date(paste(year, "11-01", sep="-")) + days(21 + (5 - wday(as.Date(paste(year, "11-01", sep="-")))) %% 7),
      
      # Christmas Day
      as.Date(paste(year, "12-25", sep="-"))
    )
    
    holidays_list <- c(holidays_list, year_holidays)
  }
  
  return(unique(as.Date(holidays_list)))
}

library(lubridate)

# Enhanced version with weekend and holiday flags - FIXED
daily_sleep_metrics <- sleep_cluster %>%
  # Extract date for grouping
  mutate(sleep_date = as.Date(cluster_start_utc)) %>%
  
  # Add weekend and holiday flags - SIMPLIFIED VERSION
  mutate(
    # Weekend flag (Saturday = 7, Sunday = 1 in wday())
    is_weekend = wday(sleep_date) %in% c(1, 7),  # Sunday and Saturday
    
    # US Holiday flag - simplified approach
    is_holiday = sleep_date %in% get_us_holidays(unique(year(sleep_date))),
    
    # Combined weekend OR holiday flag
    is_weekend_or_holiday = is_weekend | is_holiday,
    
    # Day of week name for reference
    day_of_week = wday(sleep_date, label = TRUE, abbr = FALSE),
    
    # Month for seasonal analysis
    month = month(sleep_date, label = TRUE)
  ) %>%
  
  group_by(person_id, sleep_date) %>%
  summarize(
    # Keep the flags
    is_weekend = first(is_weekend),
    is_holiday = first(is_holiday),
    is_weekend_or_holiday = first(is_weekend_or_holiday),
    day_of_week = first(day_of_week),
    month = first(month),
    
    # Daily sleep timing (using circular stats)
    daily_start_hour = circular_mean_time(hour(cluster_start_utc) + minute(cluster_start_utc)/60),
    daily_end_hour = circular_mean_time(hour(cluster_end_utc) + minute(cluster_end_utc)/60),
    daily_midpoint_hour = circular_mean_time(c(
      hour(cluster_start_utc) + minute(cluster_start_utc)/60,
      hour(cluster_end_utc) + minute(cluster_end_utc)/60
    )),
    
    # Daily variability
    daily_start_sd = circular_sd_time(hour(cluster_start_utc) + minute(cluster_start_utc)/60),
    daily_end_sd = circular_sd_time(hour(cluster_end_utc) + minute(cluster_end_utc)/60),
    
    # Daily sleep quality metrics
    daily_duration_mins = sum(cluster_duration_mins),
    n_episodes_per_day = n(),
    
    .groups = 'drop'
  ) %>%
  
  # Add individual-level summaries (including separate weekend/weekday stats)
  group_by(person_id) %>%
  mutate(
    # Overall person averages
    person_avg_start = circular_mean_time(daily_start_hour),
    person_avg_end = circular_mean_time(daily_end_hour),
    person_avg_midpoint = circular_mean_time(daily_midpoint_hour),
    
    # Separate weekend vs weekday averages
    person_weekday_avg_start = circular_mean_time(daily_start_hour[!is_weekend_or_holiday]),
    person_weekend_avg_start = circular_mean_time(daily_start_hour[is_weekend_or_holiday]),
    
    person_weekday_avg_end = circular_mean_time(daily_end_hour[!is_weekend_or_holiday]),
    person_weekend_avg_end = circular_mean_time(daily_end_hour[is_weekend_or_holiday]),
    
    person_weekday_avg_midpoint = circular_mean_time(daily_midpoint_hour[!is_weekend_or_holiday]),
    person_weekend_avg_midpoint = circular_mean_time(daily_midpoint_hour[is_weekend_or_holiday]),
    
    # Weekend effect (how much sleep timing shifts on weekends)
    person_weekend_delay_start = person_weekend_avg_start - person_weekday_avg_start,
    person_weekend_delay_end = person_weekend_avg_end - person_weekday_avg_end,
    person_weekend_delay_midpoint = person_weekend_avg_midpoint - person_weekday_avg_midpoint,
    
    # Duration differences
    person_weekday_avg_duration = mean(daily_duration_mins[!is_weekend_or_holiday], na.rm = TRUE),
    person_weekend_avg_duration = mean(daily_duration_mins[is_weekend_or_holiday], na.rm = TRUE),
    person_weekend_extra_sleep = person_weekend_avg_duration - person_weekday_avg_duration,
    
    # Consistency metrics
    person_start_consistency = circular_sd_time(daily_start_hour),
    person_end_consistency = circular_sd_time(daily_end_hour),
    person_midpoint_consistency = circular_sd_time(daily_midpoint_hour),
    
    # Other person-level metrics
    person_avg_duration = mean(daily_duration_mins, na.rm = TRUE),
    person_total_days = n(),
    person_weekend_days = sum(is_weekend_or_holiday),
    person_weekday_days = sum(!is_weekend_or_holiday),
    
    # Daily deviations from appropriate baseline (weekday vs weekend)
    daily_start_deviation = ifelse(is_weekend_or_holiday,
                                  abs(daily_start_hour - person_weekend_avg_start),
                                  abs(daily_start_hour - person_weekday_avg_start)),
    
    daily_end_deviation = ifelse(is_weekend_or_holiday,
                                abs(daily_end_hour - person_weekend_avg_end),
                                abs(daily_end_hour - person_weekday_avg_end))
  ) %>%
  ungroup()

# Ensure output directory exists, then write
out_dir <- file.path(getwd(), "processed_data")
if (!dir.exists(out_dir)) dir.create(out_dir, recursive = TRUE)
write_parquet(daily_sleep_metrics, file.path(out_dir, "daily_sleep_metrics_enhanced.parquet"))


