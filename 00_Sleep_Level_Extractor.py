#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install holidays')


# In[ ]:


get_ipython().system('pip install polars')


# In[ ]:


import polars as pl
from google.cloud import bigquery
import os
import pandas as pd
import numpy as np
from scipy.stats import circmean, circstd
import holidays
from datetime import date
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats


# In[ ]:


dataset = os.getenv("WORKSPACE_CDR")
my_bucket = os.getenv("WORKSPACE_BUCKET")

# Constants for easy tuning
MAX_RAW_DURATION = 1080   # 18 hours: Max reliable sensor reading
MAX_GAP_SECONDS = 2700    # 2 hours: Max awake time to be considered same sleep
MIN_EPISODE_MINS = 20     # 20 mins: Minimum duration to keep (naps)
MAX_EPISODE_MINS = 1440   # 24 hours: Maximum cluster duration

sleep_level_clusters_sql = f"""
WITH cohort AS (
    SELECT DISTINCT person_id 
    FROM `{dataset}.sleep_daily_summary`
),
person_zip AS (
    SELECT person_id, value_as_string AS zip
    FROM `{dataset}.observation`
    WHERE person_id IN (SELECT person_id FROM cohort)
      AND observation_source_concept_id = 1585250
),
zip_ses AS (
    SELECT DISTINCT z.*EXCEPT(zip3, zip3_as_string), 
           CAST(zip3 AS STRING) as zip_code
    FROM `{dataset}.zip3_ses_map` z
),
socio_eco_data AS (
    SELECT 
        c.person_id,
        z.zip_code
    FROM cohort c
    LEFT JOIN person_zip p ON c.person_id = p.person_id
    LEFT JOIN zip_ses z ON LEFT(p.zip, 3) = z.zip_code
),
sleep_data AS (
  SELECT DISTINCT
    sl.person_id,
    sl.start_datetime AS utc_start_datetime,
    TIMESTAMP_ADD(sl.start_datetime, INTERVAL CAST(sl.duration_in_min AS INT64) MINUTE) as utc_end_datetime,
    DATETIME(sl.start_datetime) AS local_start_datetime,
    sl.level,
    sl.duration_in_min,
    CAST(sl.is_main_sleep AS BOOL) as is_main_sleep,
    sds.minute_asleep,
    DATE(sl.sleep_date) AS local_sleep_date
  FROM `{dataset}.sleep_level` sl
  INNER JOIN `{dataset}.sleep_daily_summary` sds
    ON sl.person_id = sds.person_id
    AND DATE(sl.sleep_date) = DATE(sds.sleep_date)
  LEFT JOIN socio_eco_data sed ON sl.person_id = sed.person_id
  WHERE 
    sds.minute_asleep > 90
    AND sl.duration_in_min < {MAX_RAW_DURATION} 
    AND sl.duration_in_min > 0
    AND sl.duration_in_min != 960
    AND CAST(sl.is_main_sleep AS BOOL) IS TRUE
),
-- [FIX 1] Strict Deduplication
-- If two rows start at the exact same second, keep the one with the longest duration.
deduped_sleep AS (
  SELECT * EXCEPT(rn)
  FROM (
    SELECT *,
      ROW_NUMBER() OVER (
          PARTITION BY person_id, utc_start_datetime 
          ORDER BY duration_in_min DESC, level
      ) as rn
    FROM sleep_data
  )
  WHERE rn = 1
),
time_diff_calc AS (
  SELECT 
    *,
    -- [FIX 2] Calculate Gap. If overlap occurs (negative), treat as 0.
    GREATEST(0, CAST(TIMESTAMP_DIFF(
      utc_start_datetime,
      LAG(utc_end_datetime) OVER (PARTITION BY person_id ORDER BY utc_start_datetime),
      SECOND
    ) AS INT64)) as seconds_since_last_end
  FROM deduped_sleep
),
clustered_data AS (
  SELECT 
    *,
    SUM(
      -- If gap > 2 hours (7200s), start new cluster
      IF(COALESCE(seconds_since_last_end, {MAX_GAP_SECONDS} + 1) > {MAX_GAP_SECONDS}, 1, 0)
    ) OVER (PARTITION BY person_id ORDER BY utc_start_datetime) as cluster_id
  FROM time_diff_calc
)
SELECT 
  person_id,
  cluster_id,
  MIN(utc_start_datetime) as cluster_start_utc,
  MAX(utc_end_datetime) as cluster_end_utc,
  MIN(local_start_datetime) as cluster_start_local,
  MAX(local_start_datetime) as cluster_end_local,
  TIMESTAMP_DIFF(MAX(utc_end_datetime), MIN(utc_start_datetime), MINUTE) as cluster_duration_mins,
  COUNT(*) as n_observations,
  STRING_AGG(CAST(level as STRING) ORDER BY utc_start_datetime) as level_sequence,
  MIN(local_sleep_date) as local_sleep_date,
  SUM(duration_in_min) as total_duration_mins, 
  AVG(duration_in_min) as avg_duration_mins,
  COUNT(DISTINCT level) as unique_levels,
  SUM(IF(is_main_sleep, 1, 0)) as main_sleep_count
FROM clustered_data
GROUP BY person_id, cluster_id
HAVING 
  cluster_duration_mins >= {MIN_EPISODE_MINS}
  AND cluster_duration_mins <= {MAX_EPISODE_MINS}
  AND n_observations >= 3
  -- [FIX 3] Sanity Check: Total summed sleep cannot exceed Wall Clock Time by >20%
  -- This filters out any remaining bad data with heavy overlaps
  AND total_duration_mins <= (cluster_duration_mins * 1.2)
ORDER BY person_id, cluster_start_utc
"""


# In[ ]:


# 1. Initialize the BigQuery client
#client = bigquery.Client()
client = bigquery.Client(project=os.environ["GOOGLE_PROJECT"])

# 2. Run the query (make sure sleep_level_clusters_sql is defined)
query_job = client.query(sleep_level_clusters_sql)

# 3. Get results as an Arrow table and load into Polars (zero-copy)
sleep_cluster = pl.from_arrow(query_job.result().to_arrow())


# In[ ]:


sleep_cluster.head()


# In[ ]:


DOWNSAMPLE = False 
# Down sample for testing
if(DOWNSAMPLE):
    unique_persons = sleep_cluster.select("person_id").unique()
    sampled_persons = unique_persons.sample(fraction=0.1, seed=42)
    sleep_cluster = sleep_cluster.join(sampled_persons, on="person_id", how="semi")


def circular_diff_expr(col_a: pl.Expr, col_b: pl.Expr) -> pl.Expr:
    """
    Calculates shortest signed difference on a 24h clock.
    Returns a value between -12 and +12.
    Example: 01:00 - 23:00 = +2.0 hours (not -22.0)
    """
    return ((col_a - col_b + 12) % 24) - 12

def circular_dist_expr(col_a: pl.Expr, col_b: pl.Expr) -> pl.Expr:
    """
    Calculates shortest absolute distance on a 24h clock.
    Returns a value between 0 and 12.
    """
    return circular_diff_expr(col_a, col_b).abs()


def get_major_us_holidays(years: list) -> set:
    """
    Get major US holidays that people typically get off work.
    Returns a set of date objects.
    """
    us_cal = holidays.US(years=years)
    
    major_holidays = {
        "New Year's Day",
        "Martin Luther King Jr. Day",
        "Presidents' Day",
        "Memorial Day",
        "Independence Day",
        "Labor Day",
        "Columbus Day",
        "Veterans Day",
        "Thanksgiving",
        "Christmas Day"
    }
    
    major_holiday_dates = {
        date_obj for date_obj, name in us_cal.items()
        if name in major_holidays
    }
    
    return major_holiday_dates


def circular_mean_expr(col_name: str) -> pl.Expr:
    """
    Pure Polars expression for circular mean of hours (0-24).
    Stays in Rust backend - much faster than map_elements!
    """
    radians = pl.col(col_name) * (2 * np.pi / 24)
    cos_sum = radians.cos().sum()
    sin_sum = radians.sin().sum()
    mean_rad = pl.arctan2(sin_sum, cos_sum)
    return (mean_rad * 24 / (2 * np.pi)) % 24


def circular_sd_expr(col_name: str) -> pl.Expr:
    """
    Pure Polars expression for circular standard deviation of hours (0-24).
    Stays in Rust backend - much faster than map_elements!
    """
    radians = pl.col(col_name) * (2 * np.pi / 24)
    n = pl.col(col_name).len()
    cos_sum = radians.cos().sum()
    sin_sum = radians.sin().sum()
    R = (cos_sum.pow(2) + sin_sum.pow(2)).sqrt() / n
    sd_rad = (-2 * R.log()).sqrt()
    return sd_rad * 24 / (2 * np.pi)


# Generate holidays
min_year = sleep_cluster.select(pl.col("cluster_start_utc").dt.year().min()).item()
max_year = sleep_cluster.select(pl.col("cluster_start_utc").dt.year().max()).item()
years_list = list(range(min_year, max_year + 1))
us_holidays_set = get_major_us_holidays(years_list)
print(f"Generated holiday list for years: {years_list}")
print(f"Number of holidays: {len(us_holidays_set)}")



daily_sleep_metrics = (
    sleep_cluster
    # ---------------------------------------------------------
    # 1. FILTER: Isolate "Nightly Sleep" candidates
    # ---------------------------------------------------------
    .filter(
        (pl.col("main_sleep_count") > 0) & 
        (pl.col("cluster_duration_mins") > 180)
    )
    .with_columns([
        # 2. Trust the Fixed SQL Datetimes
        pl.col("cluster_start_utc").cast(pl.Datetime),
        pl.col("cluster_end_utc").cast(pl.Datetime),
    ])
    .with_columns([
        # 3. Calculate linear duration using TRUSTED timestamps
        (pl.col("cluster_end_utc") - pl.col("cluster_start_utc"))
            .dt.total_minutes()
            .alias("duration_mins_precise"),

        # 4. Calculate linear midpoint (Start + Duration/2)
        (pl.col("cluster_start_utc") + 
            ((pl.col("cluster_end_utc") - pl.col("cluster_start_utc")) / 2)
        ).alias("midpoint_utc"),
        
        # 5. Logical Date (Shift back 6 hours)
        (pl.col("cluster_start_utc") - pl.duration(hours=6)).dt.date().alias("sleep_date")
    ])
    .with_columns([
        # 6. Convert to floats for Circular Aggregation
        (pl.col("cluster_start_utc").dt.hour() + 
         pl.col("cluster_start_utc").dt.minute() / 60).alias("start_hour"),
          
        (pl.col("cluster_end_utc").dt.hour() + 
         pl.col("cluster_end_utc").dt.minute() / 60).alias("end_hour"),
          
        (pl.col("midpoint_utc").dt.hour() + 
         pl.col("midpoint_utc").dt.minute() / 60).alias("midpoint_hour"),

        # Context columns
        pl.col("sleep_date").dt.weekday().is_in([5, 6]).alias("is_weekend"),
        pl.col("sleep_date").is_in(list(us_holidays_set)).alias("is_holiday"),
        pl.col("sleep_date").dt.weekday().alias("weekday_num"),
        pl.col("sleep_date").dt.month().alias("month"),
    ])
    .with_columns([
        (pl.col("is_weekend") | pl.col("is_holiday")).alias("is_weekend_or_holiday")
    ])
    # ---------------------------------------------------------
    # 7. FILTER: Keep ONLY the Largest Cluster per Night
    # ---------------------------------------------------------
    .with_columns([
        pl.col("duration_mins_precise")
          .rank(method="ordinal", descending=True)
          .over(["person_id", "sleep_date"])
          .alias("duration_rank")
    ])
    .filter(pl.col("duration_rank") == 1) 
    # ---------------------------------------------------------

    .group_by(["person_id", "sleep_date", "is_weekend", "is_holiday", 
               "is_weekend_or_holiday", "weekday_num", "month"])
    .agg([
        # Since we filtered to 1 row per day, min/max/sum just return that single row's value.
        
        # Start/End boundaries (Now specific to the largest cluster)
        pl.col("cluster_start_utc").min().alias("daily_start_datetime"),
        pl.col("cluster_end_utc").max().alias("daily_end_datetime"),

        # Circular means (On a single value, this just returns the value)
        circular_mean_expr("start_hour").alias("daily_start_hour"),
        circular_mean_expr("end_hour").alias("daily_end_hour"),
        circular_mean_expr("midpoint_hour").alias("daily_midpoint_hour"),
        
        # NOTE: Removed circular_sd (Standard Deviation) because 
        # you cannot calculate SD on a single value.
        
        # Total sleep duration (Specific to the largest cluster)
        pl.col("duration_mins_precise").sum().alias("daily_duration_mins"),
        
        # Metadata (Should always be 1 now)
        pl.len().alias("n_episodes_per_day"),
    ])
    # ---------------------------------------------------------
    # NEW FILTER: Keep only days with < 15.5 hours (930 mins) of sleep
    # ---------------------------------------------------------
    .filter(pl.col("daily_duration_mins") < (15.5 * 60))
    # ---------------------------------------------------------
    .with_columns([
        (pl.col("daily_end_datetime") - pl.col("daily_start_datetime"))
            .dt.total_minutes()
            .alias("daily_sleep_window_mins")
    ])
    .sort(["person_id", "sleep_date"])
)

# Step 2: Add overall person-level statistics (simple aggregations)
daily_sleep_metrics = daily_sleep_metrics.with_columns([
    circular_mean_expr("daily_start_hour").over("person_id").alias("person_avg_start"),
    circular_mean_expr("daily_end_hour").over("person_id").alias("person_avg_end"),
    circular_mean_expr("daily_midpoint_hour").over("person_id").alias("person_avg_midpoint"),
    
    circular_sd_expr("daily_start_hour").over("person_id").alias("person_start_consistency"),
    circular_sd_expr("daily_end_hour").over("person_id").alias("person_end_consistency"),
    circular_sd_expr("daily_midpoint_hour").over("person_id").alias("person_midpoint_consistency"),
    
    pl.col("daily_duration_mins").mean().over("person_id").alias("person_avg_duration"),
    pl.len().over("person_id").alias("person_total_days"),
    pl.col("is_weekend_or_holiday").sum().over("person_id").alias("person_weekend_days"),
    (~pl.col("is_weekend_or_holiday")).sum().over("person_id").alias("person_weekday_days"),
])

# Step 3: Calculate weekday/weekend splits using filtered aggregations
# For circular means with filtering, we need to manually compute sin/cos sums
daily_sleep_metrics = daily_sleep_metrics.with_columns([
    # Weekday start
    pl.when(~pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_start_hour") * (2 * np.pi / 24)).sin())
      .sum().over("person_id")
      .alias("_wd_start_sin"),
    pl.when(~pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_start_hour") * (2 * np.pi / 24)).cos())
      .sum().over("person_id")
      .alias("_wd_start_cos"),
    
    # Weekday end
    pl.when(~pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_end_hour") * (2 * np.pi / 24)).sin())
      .sum().over("person_id")
      .alias("_wd_end_sin"),
    pl.when(~pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_end_hour") * (2 * np.pi / 24)).cos())
      .sum().over("person_id")
      .alias("_wd_end_cos"),
    
    # Weekday midpoint
    pl.when(~pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_midpoint_hour") * (2 * np.pi / 24)).sin())
      .sum().over("person_id")
      .alias("_wd_mid_sin"),
    pl.when(~pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_midpoint_hour") * (2 * np.pi / 24)).cos())
      .sum().over("person_id")
      .alias("_wd_mid_cos"),
    
    # Weekend start
    pl.when(pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_start_hour") * (2 * np.pi / 24)).sin())
      .sum().over("person_id")
      .alias("_we_start_sin"),
    pl.when(pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_start_hour") * (2 * np.pi / 24)).cos())
      .sum().over("person_id")
      .alias("_we_start_cos"),
    
    # Weekend end
    pl.when(pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_end_hour") * (2 * np.pi / 24)).sin())
      .sum().over("person_id")
      .alias("_we_end_sin"),
    pl.when(pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_end_hour") * (2 * np.pi / 24)).cos())
      .sum().over("person_id")
      .alias("_we_end_cos"),
    
    # Weekend midpoint
    pl.when(pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_midpoint_hour") * (2 * np.pi / 24)).sin())
      .sum().over("person_id")
      .alias("_we_mid_sin"),
    pl.when(pl.col("is_weekend_or_holiday"))
      .then((pl.col("daily_midpoint_hour") * (2 * np.pi / 24)).cos())
      .sum().over("person_id")
      .alias("_we_mid_cos"),
    
    # Duration splits
    pl.when(~pl.col("is_weekend_or_holiday"))
      .then(pl.col("daily_duration_mins"))
      .mean().over("person_id")
      .alias("person_weekday_avg_duration"),
    pl.when(pl.col("is_weekend_or_holiday"))
      .then(pl.col("daily_duration_mins"))
      .mean().over("person_id")
      .alias("person_weekend_avg_duration"),
])



# Apply the updates
daily_sleep_metrics = daily_sleep_metrics.with_columns([
    # 1. Reconstruct angles to hours (0-24)
    (pl.arctan2(pl.col("_wd_start_sin"), pl.col("_wd_start_cos")) * 24 / (2 * np.pi) % 24).alias("person_weekday_avg_start"),
    (pl.arctan2(pl.col("_wd_end_sin"), pl.col("_wd_end_cos")) * 24 / (2 * np.pi) % 24).alias("person_weekday_avg_end"),
    (pl.arctan2(pl.col("_wd_mid_sin"), pl.col("_wd_mid_cos")) * 24 / (2 * np.pi) % 24).alias("person_weekday_avg_midpoint"),
    (pl.arctan2(pl.col("_we_start_sin"), pl.col("_we_start_cos")) * 24 / (2 * np.pi) % 24).alias("person_weekend_avg_start"),
    (pl.arctan2(pl.col("_we_end_sin"), pl.col("_we_end_cos")) * 24 / (2 * np.pi) % 24).alias("person_weekend_avg_end"),
    (pl.arctan2(pl.col("_we_mid_sin"), pl.col("_we_mid_cos")) * 24 / (2 * np.pi) % 24).alias("person_weekend_avg_midpoint"),
]).with_columns([
    # 2. Calculate Social Jetlag (Signed Difference)
    circular_diff_expr(pl.col("person_weekend_avg_start"), pl.col("person_weekday_avg_start")).alias("person_weekend_delay_start"),
    circular_diff_expr(pl.col("person_weekend_avg_end"), pl.col("person_weekday_avg_end")).alias("person_weekend_delay_end"),
    circular_diff_expr(pl.col("person_weekend_avg_midpoint"), pl.col("person_weekday_avg_midpoint")).alias("person_weekend_delay_midpoint"),
    
    # Duration is linear, so standard subtraction is correct
    (pl.col("person_weekend_avg_duration") - pl.col("person_weekday_avg_duration")).alias("person_weekend_extra_sleep"),
]).with_columns([
    # 3. Calculate Daily Deviation (Absolute Circular Distance)
    pl.when(pl.col("is_weekend_or_holiday"))
      .then(circular_dist_expr(pl.col("daily_start_hour"), pl.col("person_weekend_avg_start")))
      .otherwise(circular_dist_expr(pl.col("daily_start_hour"), pl.col("person_weekday_avg_start")))
      .alias("daily_start_deviation"),

    pl.when(pl.col("is_weekend_or_holiday"))
      .then(circular_dist_expr(pl.col("daily_end_hour"), pl.col("person_weekend_avg_end")))
      .otherwise(circular_dist_expr(pl.col("daily_end_hour"), pl.col("person_weekday_avg_end")))
      .alias("daily_end_deviation"),
]).drop([col for col in daily_sleep_metrics.columns if col.startswith("_")])

daily_sleep_metrics.head()

# Save to parquet
out_dir = Path.cwd() / "processed_data"
out_dir.mkdir(parents=True, exist_ok=True)
output_path = out_dir / "daily_sleep_metrics_enhanced.parquet"
daily_sleep_metrics.write_parquet(output_path)

print(f"Saved daily sleep metrics to: {output_path}")
print(f"Shape: {daily_sleep_metrics.shape}")
print(f"Columns: {daily_sleep_metrics.columns}")




# ## Daily Sleep Metrics Explained
# 
# This dataset is structured so that **one row represents one "night" of sleep for one person.**
# 
# It combines distinct daily observations with calculated long-term baselines (person-level stats) to allow for immediate analysis of sleep regularity and "Social Jetlag."
# 
# ---
# 
# ### **1. Identifiers & Temporal Context**
# These columns define *who* and *when*.
# 
# * **`person_id`**: The unique identifier for the participant.
# * **`sleep_date`**: The **logical date** of the sleep episode.
#     * *Note:* Calculated by shifting the start time back 6 hours. A sleep episode starting at 02:00 AM on Tuesday is assigned to **Monday**.
# * **`is_weekend`**: `True` if the sleep occurred on a Saturday or Sunday.
# * **`is_holiday`**: `True` if the date matches a major US holiday (e.g., Thanksgiving, New Year's).
# * **`is_weekend_or_holiday`**: The primary filter used for separating "Work/School days" from "Free days."
# * **`weekday_num`**: Integer (1-7) representing the day of the week.
# * **`month`**: Integer (1-12).
# 
# ### **2. Daily Sleep Metrics (The "Night" Level)**
# These columns describe the specific sleep behavior for that specific date.
# 
# * **`daily_start_datetime`**: The absolute timestamp (UTC) when the person first fell asleep that night.
# * **`daily_end_datetime`**: The absolute timestamp (UTC) when the person finally woke up.
# * **`daily_start_hour`**: The time falling asleep, expressed as a decimal hour (0–24) on a 24-hour clock.
#     * *Note:* Uses **Circular Mean**. If a person has two fragments (23:00 and 01:00), the mean is correctly calculated as 00:00 (midnight).
# * **`daily_end_hour`**: The wakeup time expressed as a decimal hour (0–24).
# * **`daily_midpoint_hour`**: The "center of gravity" for the sleep episode (Decimal Hour).
# * **`daily_duration_mins`**: The total minutes spent *actually asleep*. (Sum of sleep fragments).
# * **`daily_sleep_window_mins`**: The total wall-clock time passed between falling asleep and final wake up.
#     * *Difference:* If `window` > `duration`, the person spent time awake in bed (WASO) between sleep fragments.
# * **`n_episodes_per_day`**: How many distinct sleep clusters were merged to form this night (usually 1, but higher indicates fragmented sleep).
# 
# ### **3. Person-Level Baselines (The "Habit" Level)**
# These columns are repeated for every row of the same person. They represent the person's **overall average behavior** across the entire study period.
# 
# * **`person_avg_start`**: The person's typical bedtime (Circular Mean).
# * **`person_avg_end`**: The person's typical wake time (Circular Mean).
# * **`person_avg_duration`**: The person's average sleep duration in minutes.
# * **`person_total_days`**: How many nights of data we have for this person.
# * **`person_start_consistency` / `person_end_consistency`**: The **Circular Standard Deviation** of their bedtimes/wake times.
#     * *Interpretation:* A lower number means the person is very consistent (always goes to bed at the same time). A higher number indicates erratic schedules.
# 
# ### **4. Social Jetlag & Weekday vs. Weekend Splits**
# These metrics quantify the misalignment between a person's biological clock and their social obligations (Social Jetlag).
# 
# **The Averages**
# * **`person_weekday_avg_[start/end/midpoint]`**: Average times calculated *only* using Mon–Fri (non-holiday) data.
# * **`person_weekend_avg_[start/end/midpoint]`**: Average times calculated *only* using Sat–Sun (and holiday) data.
# 
# **The Deltas (Social Jetlag)**
# * **`person_weekend_delay_start`**: The shift in bedtime on weekends.
#     * *Example:* `+2.0` means they go to bed 2 hours later on weekends than weekdays.
# * **`person_weekend_delay_end`**: The shift in wake time on weekends.
#     * *Example:* `+3.0` means they sleep in 3 hours later on weekends.
# * **`person_weekend_delay_midpoint`**: **The clinical definition of Social Jetlag.** It measures how much the center of their sleep window shifts.
# * **`person_weekend_extra_sleep`**: The difference in duration (Weekend Duration - Weekday Duration). Positive values indicate "Catch-up Sleep."
# 
# ### **5. Daily Deviations**
# These columns measure how "weird" a specific night was compared to that person's specific norm for that type of day.
# 
# * **`daily_start_deviation`**: The absolute difference (in hours) between *tonight's* bedtime and the person's average baseline.
#     * *Logic:* If it is a Monday, this compares Monday's bedtime to `person_weekday_avg_start`. If Saturday, it compares to `person_weekend_avg_start`.
# * **`daily_end_deviation`**: Same as above, but for wake time.
# 
# ---
# 
# ### **Key Technical Concepts in this Data**
# 
# **1. Logical Date (The "6-hour shift")**
# Sleep data is messy because "Monday Night" usually ends on "Tuesday Morning."
# * **Raw:** Start `2023-01-01 23:00`, End `2023-01-02 07:00`
# * **Logic:** We subtract 6 hours from the start time. `23:00 - 6h = 17:00`.
# * **Result:** The date is logged as `2023-01-01`. This ensures the entire sleep bout belongs to the day the night *began*.
# 
# **2. Circular Statistics**
# Standard math fails on a 24-hour clock (the average of 23:00 and 01:00 is 12:00 noon using standard math, which is wrong).
# * **Circular Mean:** Converts time to coordinates on a circle (Sin/Cos), averages the coordinates, and converts back to time. Correctly identifies midnight (00:00) as the average of 23:00 and 01:00.
# * **Circular SD:** A robust measure of spread for angular data.
# 
# **3. Social Jetlag**
# This is the phenomenon where people live in one timezone during the week (School/Work) and a different biological timezone on the weekend.
# * High `person_weekend_delay_midpoint` is strongly correlated with metabolic health issues, mood disorders, and poor sleep hygiene.

# In[ ]:


print(daily_sleep_metrics.head())


# ## Visualization test

# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import numpy as np
import polars as pl

# ==========================================
# 0. CONFIGURATION
# ==========================================
# Set your filtering and visualization bounds here
MIN_SLEEP_HOURS = 1.0   # Minimum duration to include/show
MAX_SLEEP_HOURS = 14  # Maximum duration to include/show

# Visual Noise Threshold (Filter bins with fewer than X dots)
MIN_OBSERVATIONS = 110    
# ==========================================

# 1. Use the RAW sleep clusters
episode_data = (
    sleep_cluster
    # Filter using the variables defined above (converting hours to minutes)
    .filter(
        (pl.col("total_duration_mins") >= (MIN_SLEEP_HOURS * 60)) & 
        (pl.col("total_duration_mins") <= (MAX_SLEEP_HOURS * 60))
    )
    .with_columns([
        # Convert start time to decimal hours
        (pl.col("cluster_start_utc").dt.hour() + 
         pl.col("cluster_start_utc").dt.minute() / 60).alias("start_hour"),
        
        # Convert duration to hours
        (pl.col("total_duration_mins") / 60).alias("duration_hours")
    ])
    .with_columns([
        # Apply the "Noon-to-Noon" logic for X-axis
        pl.when(pl.col("start_hour") < 12)
          .then(pl.col("start_hour") + 24)
          .otherwise(pl.col("start_hour"))
          .alias("onset_plot")
    ])
    .drop_nulls(["onset_plot", "duration_hours"])
)

# 2. Get Data
x = episode_data["onset_plot"].to_numpy()
y = episode_data["duration_hours"].to_numpy()

# 3. Calculate Histogram Manually
# We dynamically set the range based on your variables so the heatmap 
# doesn't generate empty pixels for hours we don't care about.
counts, xedges, yedges = np.histogram2d(
    x, y, 
    bins=[120, 120], 
    range=[[12, 36], [MIN_SLEEP_HOURS, MAX_SLEEP_HOURS]] 
)

# 4. FILTER: Mask bins with fewer than N observations
counts_masked = np.ma.masked_where(counts < MIN_OBSERVATIONS, counts)

# 5. Plot
plt.figure(figsize=(10, 7), dpi=120)

pcm = plt.pcolormesh(xedges, yedges, counts_masked.T, cmap="turbo", norm=LogNorm())

# Formatting
cbar = plt.colorbar(pcm)
cbar.set_label('Count (Log Scale)')

ax = plt.gca()

# X-Axis (Time) Settings
ax.set_xlim(12, 36)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

def time_fmt(x, pos):
    return f"{int(x % 24)}:00"
ax.xaxis.set_major_formatter(ticker.FuncFormatter(time_fmt))

# Y-Axis (Duration) Settings
# Enforce the limits strictly based on your configuration
ax.set_ylim(MIN_SLEEP_HOURS, MAX_SLEEP_HOURS)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.title(f"Sleep Episodes: Onset vs. Cluster Duration (total_duration_mins) ({MIN_SLEEP_HOURS}h - {MAX_SLEEP_HOURS}h)")
plt.xlabel("Sleep Onset Time")
plt.ylabel("Sleep Duration (h)")

plt.tight_layout()
plt.show()


# In[ ]:


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
import numpy as np
import polars as pl

# ==========================================
# 0. CONFIGURATION
# ==========================================
# Set your filtering and visualization bounds here
MIN_SLEEP_HOURS = 1.0   # Minimum duration to include/show
MAX_SLEEP_HOURS = 24    # Maximum duration to include/show

# Visual Noise Threshold (Filter bins with fewer than X dots)
MIN_OBSERVATIONS = 70   
# ==========================================

# 1. Use the AGGREGATED daily metrics
episode_data = (
    daily_sleep_metrics # <--- CHANGED SOURCE
    # Filter using the variables defined above (converting hours to minutes)
    .filter(
        (pl.col("daily_duration_mins") >= (MIN_SLEEP_HOURS * 60)) & 
        (pl.col("daily_duration_mins") <= (MAX_SLEEP_HOURS * 60))
    )
    .with_columns([
        # Convert daily start datetime to decimal hours
        (pl.col("daily_start_datetime").dt.hour() + 
         pl.col("daily_start_datetime").dt.minute() / 60).alias("start_hour"),
         
        # Convert daily duration to hours
        (pl.col("daily_duration_mins") / 60).alias("duration_hours")
    ])
    .with_columns([
        # Apply the "Noon-to-Noon" logic for X-axis
        # If sleep starts before noon (e.g., 1 AM), add 24 to make it 25.0
        pl.when(pl.col("start_hour") < 12)
          .then(pl.col("start_hour") + 24)
          .otherwise(pl.col("start_hour"))
          .alias("onset_plot")
    ])
    .drop_nulls(["onset_plot", "duration_hours"])
)

# 2. Get Data
x = episode_data["onset_plot"].to_numpy()
y = episode_data["duration_hours"].to_numpy()

# 3. Calculate Histogram Manually
# We dynamically set the range based on your variables so the heatmap 
# doesn't generate empty pixels for hours we don't care about.
counts, xedges, yedges = np.histogram2d(
    x, y, 
    bins=[120, 120], 
    range=[[12, 36], [MIN_SLEEP_HOURS, MAX_SLEEP_HOURS]] 
)

# 4. FILTER: Mask bins with fewer than N observations
counts_masked = np.ma.masked_where(counts < MIN_OBSERVATIONS, counts)

# 5. Plot
plt.figure(figsize=(10, 7), dpi=120)

pcm = plt.pcolormesh(xedges, yedges, counts_masked.T, cmap="turbo", norm=LogNorm())

# Formatting
cbar = plt.colorbar(pcm)
cbar.set_label('Count (Log Scale)')

ax = plt.gca()

# X-Axis (Time) Settings
ax.set_xlim(12, 36)
ax.xaxis.set_major_locator(ticker.MultipleLocator(2))

def time_fmt(x, pos):
    return f"{int(x % 24)}:00"
ax.xaxis.set_major_formatter(ticker.FuncFormatter(time_fmt))

# Y-Axis (Duration) Settings
# Enforce the limits strictly based on your configuration
ax.set_ylim(MIN_SLEEP_HOURS, MAX_SLEEP_HOURS)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

plt.title(f"Nightly Sleep: Onset vs. Duration ({MIN_SLEEP_HOURS}h - {MAX_SLEEP_HOURS}h)")
plt.xlabel("Sleep Onset Time")
plt.ylabel("Sleep Duration (h)")

plt.tight_layout()
plt.show()


# ## Anomaly detection

# In[ ]:


# Filter for the "Monster" clusters (> 24 hours)
long_clusters = (
    daily_sleep_metrics
    .filter(pl.col("daily_sleep_window_mins") > 1440)
    .select([
        "person_id",  
        "sleep_date",
        "daily_start_datetime", 
        "daily_end_datetime",
        "daily_sleep_window_mins"
    ])
    .sort("daily_sleep_window_mins", descending=True)
)

print(f"Found {long_clusters.height} clusters longer than 24 hours.")
print(long_clusters.head(10))



# In[ ]:


# Filter for the "Monster" clusters (> 24 hours)
long_clusters = (
    sleep_cluster
    .filter(pl.col("total_duration_mins") > 1440)
    .select([
        "person_id", 
        "cluster_id", 
        "cluster_start_utc", 
        "cluster_end_utc",
        "total_duration_mins", 
        "n_observations"
    ])
    .sort("total_duration_mins", descending=True)
)

print(f"Found {long_clusters.height} clusters longer than 24 hours.")
print(long_clusters.head(10))

# METRIC: Calculate "Density"
# If a cluster is 48 hours long (2880 mins) but only has 5 observations, 
# it's a 'Sparse Bridge' error.
density_check = (
    long_clusters
    .with_columns([
        (pl.col("total_duration_mins") / pl.col("n_observations")).alias("mins_per_obs")
    ])
    .sort("mins_per_obs", descending=True)
)

print("\nLeast dense clusters (High minutes per observation = likely bridging errors):")
print(density_check.head(10))


# In[ ]:


# Filter for clusters around 16 hours (+/- 10 mins)
line_16h = (
    sleep_cluster
    .filter(
        (pl.col("total_duration_mins") >= 950) & 
        (pl.col("total_duration_mins") <= 970)
    )
)

# Check the exact mode (most common value)
print("\nMost frequent durations around 16h:")
print(line_16h.group_by("total_duration_mins").len().sort("len", descending=True).head(5))


# ## Cluster Visualization

# In[ ]:


import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import polars as pl
import numpy as np
import os
from google.cloud import bigquery
from datetime import datetime, timedelta

# ==========================================
# CONFIGURATION
# ==========================================
TARGET_PERSON_ID = 5169008  
TARGET_START_DATE = "2016-10-25" 
dataset = os.getenv("WORKSPACE_CDR")

# Define the window for plotting (Target +/- 2 days)
start_date_dt = datetime.strptime(TARGET_START_DATE, "%Y-%m-%d")
window_start = start_date_dt - timedelta(days=2)
window_end = start_date_dt + timedelta(days=3)

print(f"Inspecting")
print(f"Window: {window_start.date()} to {window_end.date()}")

# ==========================================
# 1. FETCH RAW SENSOR DATA (SQL)
# ==========================================
# We still need SQL for the specific 'levels' (hypnogram bars) 
# as those are not usually kept in the summary tables.
raw_sql = f"""
SELECT 
    person_id,
    start_datetime,
    duration_in_min,
    level,
    CAST(is_main_sleep AS BOOL) as is_main_sleep
FROM `{dataset}.sleep_level` 
WHERE person_id = {TARGET_PERSON_ID}
  AND DATE(start_datetime) BETWEEN '{window_start.strftime('%Y-%m-%d')}' 
                               AND '{window_end.strftime('%Y-%m-%d')}'
ORDER BY start_datetime
"""

print("Fetching raw sensor levels from BigQuery...")
client = bigquery.Client(project=os.environ["GOOGLE_PROJECT"])
df_raw = client.query(raw_sql).to_dataframe()

# Pre-calc end times for plotting
if not df_raw.empty:
    df_raw['start_datetime'] = pd.to_datetime(df_raw['start_datetime'])
    df_raw['end_datetime'] = df_raw['start_datetime'] + pd.to_timedelta(df_raw['duration_in_min'], unit='m')
    print(f" -> Found {len(df_raw)} raw sensor rows.")
else:
    print(" -> No raw data found for this window.")

# ==========================================
# 2. FILTER EXISTING DATA (Polars)
# ==========================================
print("Filtering existing 'sleep_cluster' and 'daily_sleep_metrics'...")

# A. Filter Clusters
# We use the 'sleep_cluster' variable from your previous steps
# Columns expected: cluster_start_utc, cluster_end_utc, total_duration_mins
try:
    df_clusters = (
        sleep_cluster
        .filter(pl.col("person_id") == TARGET_PERSON_ID)
        .filter(
            (pl.col("cluster_start_utc") >= window_start) & 
            (pl.col("cluster_start_utc") <= window_end)
        )
        .to_pandas()
    )
    # Ensure datetime objects
    df_clusters['cluster_start_utc'] = pd.to_datetime(df_clusters['cluster_start_utc'])
    df_clusters['cluster_end_utc'] = pd.to_datetime(df_clusters['cluster_end_utc'])
    print(f" -> Found {len(df_clusters)} sleep clusters.")

except Exception as e:
    print(f"Error filtering sleep_cluster (is it defined?): {e}")
    df_clusters = pd.DataFrame()


# B. Filter Daily Metrics
# We use the 'daily_sleep_metrics' variable from your previous steps
# Columns expected: daily_start_datetime, daily_end_datetime
try:
    df_metrics = (
        daily_sleep_metrics
        .filter(pl.col("person_id") == TARGET_PERSON_ID)
        .filter(
            (pl.col("daily_start_datetime") >= window_start) & 
            (pl.col("daily_start_datetime") <= window_end)
        )
        .to_pandas()
    )
    # Ensure datetime objects
    df_metrics['daily_start_datetime'] = pd.to_datetime(df_metrics['daily_start_datetime'])
    df_metrics['daily_end_datetime'] = pd.to_datetime(df_metrics['daily_end_datetime'])
    print(f" -> Found {len(df_metrics)} daily metric days.")

except Exception as e:
    print(f"Error filtering daily_sleep_metrics (is it defined?): {e}")
    df_metrics = pd.DataFrame()


# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================
def plot_sleep_debug(raw, clusters, metrics):
    fig, ax = plt.subplots(figsize=(18, 10))
    
    # A. Plot CLUSTERS (The Background Box)
    # Columns: cluster_start_utc, cluster_end_utc, total_duration_mins
    if not clusters.empty:
        for _, row in clusters.iterrows():
            start = mdates.date2num(row['cluster_start_utc'])
            end = mdates.date2num(row['cluster_end_utc'])
            width = end - start
            
            # Draw a grey box spanning the entire vertical area
            rect = mpatches.Rectangle(
                (start, 0.5), width, 3.0, 
                color='#e0e0e0', alpha=0.5, edgecolor='black', linestyle=':', linewidth=1,
                label='Cluster (Existing)' if _ == 0 else ""
            )
            ax.add_patch(rect)
            
            # Label
            midpoint = start + (width/2)
            dur = row.get('total_duration_mins', row.get('cluster_duration_mins', 0)) # Handle potential naming diffs
            ax.text(midpoint, 3.4, f"Cluster\n{int(dur)}m", 
                    ha='center', va='bottom', fontsize=9, color='dimgrey')

    # B. Plot RAW SENSOR DATA (The Bars)
    color_map = {'deep': '#000080', 'light': '#6495ED', 'rem': '#87CEEB', 'wake': '#FFA500', 'awake': '#FFA500'}
    
    if not raw.empty:
        for _, row in raw.iterrows():
            start = mdates.date2num(row['start_datetime'])
            duration_days = row['duration_in_min'] / (60 * 24)
            
            level = str(row['level']).lower()
            color = color_map.get(level, '#808080')
            
            # Lane Logic
            is_main = str(row['is_main_sleep']).lower() == 'true'
            y_pos = 2.0 if is_main else 1.0
            
            rect = mpatches.Rectangle(
                (start, y_pos), duration_days, 0.8,
                color=color, alpha=0.9, edgecolor='white'
            )
            ax.add_patch(rect)

    # C. Plot DAILY METRICS (The Calculated Boundaries)
    # Columns: daily_start_datetime, daily_end_datetime
    if not metrics.empty:
        for _, row in metrics.iterrows():
            d_start = row['daily_start_datetime']
            d_end = row['daily_end_datetime']
            
            # Start Line (Green)
            ax.axvline(d_start, color='green', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(mdates.date2num(d_start), 0.6, "Metric Start", rotation=90, color='green', fontsize=8, ha='right')
            
            # End Line (Red)
            ax.axvline(d_end, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(mdates.date2num(d_end), 0.6, "Metric End", rotation=90, color='red', fontsize=8, ha='left')

    # D. Formatting
    ax.set_ylim(0, 4)
    ax.set_yticks([1.4, 2.4])
    ax.set_yticklabels(['Naps/Secondary', 'Main Sleep'])
    
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.set_xlabel(f"Time (UTC)")
    
    # Top Axis for Dates
    # Set limits first based on window
    ax.set_xlim(mdates.date2num(window_start), mdates.date2num(window_end))
    
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    ax_top.xaxis.set_major_locator(mdates.DayLocator())
    ax_top.xaxis.set_major_formatter(mdates.DateFormatter('\n%Y-%m-%d'))
    
    # Legend
    patches = [mpatches.Patch(color=v, label=k) for k, v in color_map.items()]
    patches.append(mpatches.Patch(color='#e0e0e0', label='Cluster Extent'))
    ax.legend(handles=patches, loc='upper right')
    
    plt.title(f"Raw vs Clusters vs Metrics", y=1.1)
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Execute
plot_sleep_debug(df_raw, df_clusters, df_metrics)


# # Additional Visualizations and Tests 
