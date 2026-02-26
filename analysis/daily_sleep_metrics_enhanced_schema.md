# daily_sleep_metrics_enhanced.parquet — Schema & Notes ✅

## Summary
This document describes the data produced by `00_Sleep_Level_Extractor.ipynb` and saved to `processed_data/daily_sleep_metrics_enhanced.parquet`.

- Purpose: Produce person-day level summary metrics from Fitbit sleep level and daily summary data, compute circular statistics (start/end times and midpoints), and produce person-level and day-level aggregates for downstream analysis.
- Main outputs:
  - `sleep_cluster` (in-memory Polars DataFrame, derived from BigQuery SQL) — cluster-level sleep episodes (not persisted by notebook by default)
  - `daily_sleep_metrics` (saved) — daily aggregates with person-level stats, weekend/weekday splits, and deviation measures saved as `processed_data/daily_sleep_metrics_enhanced.parquet`.

---

## Where files are saved
- Parquet output: `processed_data/daily_sleep_metrics_enhanced.parquet` (in repo root / `processed_data` folder).
- The notebook also builds `sleep_cluster` as a Polars DataFrame in memory via a BigQuery job and uses `daily_sleep_metrics` for plotting/statistics.

---

## How the data was generated (key steps)
1. A BigQuery query generates clusters of `sleep_level` events (`sleep_cluster`) per person using a 4-hour gap threshold between events to create a `cluster_id`. The SQL query does the following:
   - Joins `sleep_level` and `sleep_daily_summary` using `person_id` and matching `sleep_date`.
   - Adds `socio_eco_data` using `person_id` -> `observation` (zip) -> zip3 SES mapping (not used in final output columns but available in the SQL join).
   - Filters out days with `minute_asleep` <= 90.
   - Computes `time_diff` via `TIMESTAMP_DIFF` and assigns `cluster_id` by checking if `time_diff` > 14400 (4 hours) — this establishes cluster boundaries.
   - Returns cluster-level quantities and filters clusters with
     - `cluster_duration_mins` between 60 and 720 (1 hour to 12 hours),
     - `n_observations` >= 3.
2. The query is run with Google BigQuery and the results are loaded into Polars (`sleep_cluster` DataFrame).
3. The notebook computes daily aggregates grouped by `person_id` + `sleep_date` and derives many person-level aggregates using `polars` expressions that remain within Rust for performance (circular mean/sd implementations).
4. An enhanced daily Parquet file is written to `processed_data/daily_sleep_metrics_enhanced.parquet`.

---

## `sleep_cluster` (SQL result) — cluster-level fields
This is the direct output of the SQL query (returned as a Polars DataFrame). If you want to persist it, export from the notebook with `write_parquet()`.

- `person_id` (int) — Person identifier (AoU pseudonymous id).
- `cluster_id` (int) — Cluster ID computed by the SQL query; clusters are contiguous episodes split when gap > 4 hours.
- `cluster_start_utc` (timestamp/TIMESTAMP) — UTC timestamp for the first event of the cluster.
- `cluster_end_utc` (timestamp/TIMESTAMP) — UTC timestamp for last event of cluster.
- `cluster_start_local` (datetime/DATETIME) — local datetime of cluster start.
- `cluster_end_local` (datetime/DATETIME) — local end datetime.
- `cluster_duration_mins` (int) — cluster duration in minutes (difference between max and min UTC cluster timestamps).
- `n_observations` (int) — number of level entries (sleep epochs) inside cluster.
- `level_sequence` (string) — string aggregation of the per-epoch level sequence (ordered by UTC).
- `local_sleep_date` (date/DATE) — date (local date) representing where the cluster falls.
- `total_duration_mins` (int) — sum of epoch durations for the cluster (duration_in_min sum for cluster).
- `avg_duration_mins` (float) — average epoch duration in minutes in the cluster.
- `unique_levels` (int) — number of distinct levels recorded in the cluster.
- `main_sleep_count` (int) — count of episodes flagged as `is_main_sleep` (value cast to boolean earlier).

Notes on cluster building & filtering
- Clustering uses 14400-second (4 hours) gap threshold.
- Only clusters with `cluster_duration_mins` between 60 and 720 are retained.
- Only `sleep_daily_summary` days with `minute_asleep` > 90 are included (filter in SQL in `sleep_data`).

---

## `daily_sleep_metrics_enhanced.parquet` (Saved schema & descriptions)
This is the dataset explicitly saved by the notebook. It contains person-day entries and additional person-level stats (some computed via `.over("person_id")` windows in Polars).

Important: All hour-based circular fields are in 0–24 hour scale (floating point) and represent times of day. Circular values wrap (e.g., 23.5 & 0.25 represent near-each-other times); computed circular statistics (mean, sd) are handled with trigonometric transforms.

Below is the field list (as saved) with type and description:

- Keys / grouping
  - `person_id` (integer) — pseudonymous person identifier.
  - `sleep_date` (date) — local calendar date for the daily summary.
  - `is_weekend` (bool) — `True` if `sleep_date` is a Saturday or Sunday.
  - `is_holiday` (bool) — `True` if `sleep_date` matches a major US holiday (holidays library, compiled between years available in dataset).
  - `is_weekend_or_holiday` (bool) — True when `is_weekend` OR `is_holiday`.
  - `weekday_num` (int 0–6) — weekday 0-based index (0=Monday if consistent with Polars; check runtime if in doubt).
  - `month` (int 1–12) — month number of `sleep_date`.

- Day-level derived (aggregations per day)
  - `daily_start_hour` (float 0–24) — circular mean of cluster start times on that day; hours decimal.
  - `daily_end_hour` (float 0–24) — circular mean of cluster end times on that day.
  - `daily_midpoint_hour` (float 0–24) — circular mean for the midpoint (computed by combining start and end times and computing circular mean); approximate mid-time across episodes.
  - `daily_start_sd` (float) — circular standard deviation of start times (in hours).
  - `daily_end_sd` (float) — circular standard deviation of end times (in hours).
  - `daily_duration_mins` (int) — total sleep duration on that day (minutes), computed as sum of cluster durations grouped by person/day.
  - `n_episodes_per_day` (int) — count of clusters/episodes for that person on that day.

- Person-level aggregates (computed via window `.over("person_id")`)
  - `person_avg_start` (float) — circular mean of `daily_start_hour` across that person’s days; a person-level habitual start time (hours).
  - `person_avg_end` (float) — circular mean of `daily_end_hour` across days.
  - `person_avg_midpoint` (float) — circular mean of `daily_midpoint_hour` across days.
  - `person_start_consistency` (float) — circular standard deviation of start times across days (hours). Lower = more consistent schedule.
  - `person_end_consistency` (float) — circular sd for end times (hours).
  - `person_midpoint_consistency` (float) — circular sd for midpoints (hours).
  - `person_avg_duration` (float minutes) — average daily duration across days.
  - `person_total_days` (int) — number of days in the dataset for that person (used in aggregation weighting).
  - `person_weekend_days` (int) — count of days that were weekend or holiday for the person.
  - `person_weekday_days` (int) — count of non-weekend/holiday days for the person.

- Weekday / weekend split aggregates (derived via sin/cos sums, then converted):
  - `person_weekday_avg_start` (float 0–24) — weekday-only circular mean start.
  - `person_weekday_avg_end` (float 0–24) — weekday-only circular mean end.
  - `person_weekday_avg_midpoint` (float 0–24) — weekday-only circ mean midpoint.
  - `person_weekend_avg_start` (float 0–24) — weekend/holiday circular mean start.
  - `person_weekend_avg_end` (float 0–24) — weekend/holiday circular mean end.
  - `person_weekend_avg_midpoint` (float 0–24) — weekend/holiday circular mean midpoint.
  - `person_weekday_avg_duration` (float minutes) — mean daily duration on weekdays.
  - `person_weekend_avg_duration` (float minutes) — mean daily duration on weekends/holidays.

- Derived differences / deviations
  - `person_weekend_delay_start` (float hours) — `person_weekend_avg_start - person_weekday_avg_start`; positive = later times on weekend vs weekday.
  - `person_weekend_delay_end` (float hours) — weekend end minus weekday end.
  - `person_weekend_delay_midpoint` (float hours) — weekend midpoint delay vs weekday.
  - `person_weekend_extra_sleep` (float minutes) — weekend average duration minus weekday average duration.
  - `daily_start_deviation` (float hours) — absolute deviation for that day compared to the relevant person-level average (weekday vs weekend average used accordingly).
  - `daily_end_deviation` (float hours) — absolute deviation for the end time for the day.

- Computational notes about units and types
  - Hours (0–24) are represented as float; be careful with circular interpretation at 0/24 hours.
  - Duration fields are in minutes, unless explicitly labeled hours (hours derived only when the code divides minutes by 60 for printing; the raw saved fields remain minutes).
  - Circular statistics are computed using trig functions and returned in hours. Standard deviations are circular standard deviations (in hours).

---

## Example: Loading the saved parquet
- Using Polars (as used in the notebook):

```python
import polars as pl
path = 'processed_data/daily_sleep_metrics_enhanced.parquet'
df = pl.read_parquet(path)
df.head()
```

- Using Pandas:

```python
import pandas as pd
path = 'processed_data/daily_sleep_metrics_enhanced.parquet'
df = pd.read_parquet(path)
df.head()
```

---

## Statistical Analysis & Visualization (done in notebook)
- The notebook performs:
  - Circular KDE plots for onset and offset (weekdays vs weekends/holidays).
  - Per-person paired t-tests for weekday vs weekend differences using `stats.ttest_rel`, with circular mean handling.
  - Cohort-level statistics printed for metrics: Sleep Onset, Sleep Offset, Sleep Midpoint, and Sleep Duration.
- The notebook also produces a `cluster_summary` for plotting (subsampled for plotting), and a `person_stats` table used when running the paired tests — these are derived from `daily_sleep_metrics`.

---

## Important caveats & notes ⚠️
- Holidays: A curated list of major US holidays is used and set by `holidays.US(years=...)`. If your analysis needs a different holiday set/definition, change the function `get_major_us_holidays`.
- Weekends vs Holidays: `is_weekend_or_holiday` uses an OR logic — if a holiday falls on a weekday, the day is not treated as a weekday.
- Filter thresholds in SQL: The initial SQL filters can exclude certain individuals/days — `minute_asleep > 90`, `cluster_duration_mins` between 60 and 720, and `n_observations >= 3`. Check these if you need more or fewer events.
- Cluster creation: `time_diff` threshold = 4 * 60 * 60 seconds (14400). Adjust this if you want different clustering behavior.
- Person ID: `person_id` is AoU pseudonymous ID; keep in mind AoU rules/data handling when sharing datasets.
- Performance: The notebook uses Polars and Rust-backed expressions to compute circular stats; this is efficient on large datasets.

---

## Recommended downstream tasks / suggestions 💡
- Persist `sleep_cluster` to Parquet for cluster-level analysis (if needed), e.g., `sleep_cluster.write_parquet('processed_data/sleep_cluster.parquet')`.
- Additional fields to consider computing:
  - `sleep_efficiency`: If `minute_asleep` or other flags are available per episode.
  - `onset_variability` and `offset_variability`: sigma across dates or derived percentiles for more robust metrics.
- Consider merging `daily_sleep_metrics` with EHR or PGS tables for PheWAS or PGS analyses.

---

## Quick reference
- Notebook: `00_Sleep_Level_Extractor.ipynb`
- BigQuery dataset used: `WORKSPACE_CDR` (populated from environment — do not hardcode)
- Saved Parquet: `processed_data/daily_sleep_metrics_enhanced.parquet`

---

If you’d like, I can also:
- add a small CLI or notebook cell to persist `sleep_cluster` to Parquet,
- create a one-page data dictionary CSV or TSV with the columns and descriptions,
- or include a sample `pandas`/`polars` script demonstrating joins with `ICD_to_Phecode_mapping.csv` for downstream analysis.

Let me know which of the above you'd like next and I’ll add it. ✅
