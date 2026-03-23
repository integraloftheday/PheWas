#!/usr/bin/env python
# coding: utf-8

"""
03_Descriptive_Analysis.py
Python port of key descriptive plotting workflow.
Uses linearized sleep timing columns consistently.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

INPUT_PATH = "processed_data/ready_for_analysis.parquet"
OUTPUT_DIR = Path("results/descriptive_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def linearize_noon_to_noon(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return np.where(s < 12, s + 24, s)


print(f"Loading {INPUT_PATH}...")
lf = pl.scan_parquet(INPUT_PATH)
schema_cols = set(lf.collect_schema().names())

# Build canonical analysis columns using linearized values where possible.
if "onset_linear" in schema_cols:
    onset_expr = pl.col("onset_linear").cast(pl.Float64)
else:
    onset_expr = pl.when(pl.col("daily_start_hour") < 12).then(pl.col("daily_start_hour") + 24).otherwise(pl.col("daily_start_hour")).cast(pl.Float64)

if "midpoint_linear" in schema_cols:
    midpoint_expr = pl.col("midpoint_linear").cast(pl.Float64)
else:
    midpoint_expr = pl.when(pl.col("daily_midpoint_hour") < 12).then(pl.col("daily_midpoint_hour") + 24).otherwise(pl.col("daily_midpoint_hour")).cast(pl.Float64)

if "offset_linear" in schema_cols:
    offset_expr = pl.col("offset_linear").cast(pl.Float64)
else:
    offset_expr = pl.when(pl.col("daily_end_hour") < 12).then(pl.col("daily_end_hour") + 24).otherwise(pl.col("daily_end_hour")).cast(pl.Float64)

select_cols = [
    "person_id",
    "sleep_date",
    "is_weekend",
    "age_at_sleep",
    "sex_concept",
    "daily_duration_mins",
    "daily_sleep_window_mins",
    "SJL_raw",
]
timing_source_cols = [
    c for c in [
        "onset_linear", "midpoint_linear", "offset_linear",
        "daily_start_hour", "daily_midpoint_hour", "daily_end_hour",
    ] if c in schema_cols
]
select_cols = select_cols + timing_source_cols
available_select = [c for c in select_cols if c in schema_cols]

df = (
    lf.select(available_select)
    .with_columns([
        pl.col("sleep_date").cast(pl.Date),
        pl.col("sleep_date").cast(pl.Date).dt.ordinal_day().alias("day_of_year"),
        onset_expr.alias("daily_start_hour"),
        midpoint_expr.alias("daily_midpoint_hour"),
        offset_expr.alias("daily_end_hour"),
        (pl.col("daily_sleep_window_mins") / 60.0).alias("daily_sleep_window_hours") if "daily_sleep_window_mins" in schema_cols else (pl.col("daily_duration_mins") / 60.0).alias("daily_sleep_window_hours"),
    ])
    .collect(engine="streaming")
    .to_pandas()
)

print("Using timing columns: onset=linearized, midpoint=linearized, offset=linearized")

# ---------- Plot 1: midpoint histogram ----------
plt.figure(figsize=(9, 5))
plt.hist(df["daily_midpoint_hour"].dropna(), bins=np.arange(12, 37, 0.5), color="skyblue", edgecolor="black")
plt.title("Distribution of Sleep Midpoint (Linearized)")
plt.xlabel("Midpoint Hour (Noon-to-Noon)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "01_midpoint_hist_linearized.png", dpi=300)
plt.close()

# ---------- Plot 2: nights/person ----------
nights_per_person = df.groupby("person_id", as_index=False).size().rename(columns={"size": "n_nights"})
plt.figure(figsize=(9, 5))
plt.hist(nights_per_person["n_nights"], bins=50, color="steelblue", edgecolor="white")
plt.yscale("log")
plt.title("Data Density: Number of Nights per Person")
plt.xlabel("Number of Nights")
plt.ylabel("Count of People (Log Scale)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "02_nights_per_person_log_hist.png", dpi=300)
plt.close()

# ---------- Plot 3/4/5: weekday/weekend densities ----------
def density_plot(col: str, title: str, filename: str):
    tmp = df[[col, "is_weekend"]].dropna()
    if tmp.empty:
        return
    plt.figure(figsize=(9, 5))
    sns.kdeplot(data=tmp, x=col, hue="is_weekend", fill=True, common_norm=False, alpha=0.4)
    plt.title(title)
    plt.xlabel("Hour (Noon-to-Noon)")
    plt.ylabel("Density")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


density_plot("daily_start_hour", "Sleep Onset Density: Weekday vs Weekend (Linearized)", "03_onset_density_weekend_linearized.png")
density_plot("daily_end_hour", "Sleep Offset Density: Weekday vs Weekend (Linearized)", "04_offset_density_weekend_linearized.png")
density_plot("daily_midpoint_hour", "Sleep Midpoint Density: Weekday vs Weekend (Linearized)", "05_midpoint_density_weekend_linearized.png")

# ---------- Plot 6: social jetlag vs age ----------
if {"SJL_raw", "age_at_sleep", "sex_concept"}.issubset(df.columns):
    sjl_age = (
        df.groupby(["person_id", "sex_concept"], as_index=False)
        .agg(mean_age=("age_at_sleep", "mean"), sjl_raw=("SJL_raw", "first"))
    )
    sjl_age = sjl_age[sjl_age["sex_concept"].isin(["Female", "Male"])].dropna(subset=["mean_age", "sjl_raw"])

    plt.figure(figsize=(10, 6))
    sns.regplot(data=sjl_age, x="mean_age", y="sjl_raw", scatter=False, lowess=True, color="black")
    sns.scatterplot(data=sjl_age.sample(min(len(sjl_age), 30000), random_state=42), x="mean_age", y="sjl_raw", hue="sex_concept", alpha=0.15, s=10)
    plt.axhline(0, linestyle="--", color="gray")
    plt.title("Social Jetlag across the Lifespan")
    plt.xlabel("Age")
    plt.ylabel("Social Jetlag (Weekend - Weekday Midpoint)")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "06_sjl_vs_age_linearized_context.png", dpi=300)
    plt.close()

# ---------- Plot 7/8/9/10: seasonal trends (weekly means) ----------
df["week"] = np.ceil(pd.to_numeric(df["day_of_year"], errors="coerce") / 7.0)


def weekly_plot(col: str, title: str, filename: str, ylabel: str):
    tmp = df[["week", col]].dropna()
    if tmp.empty:
        return
    agg = tmp.groupby("week", as_index=False).agg(mean=(col, "mean"), se=(col, lambda x: x.std(ddof=1) / np.sqrt(max(len(x), 1))))
    plt.figure(figsize=(10, 5))
    plt.errorbar(agg["week"], agg["mean"], yerr=1.96 * agg["se"].fillna(0), fmt="o", alpha=0.4, markersize=3)
    plt.plot(agg["week"], agg["mean"].rolling(3, min_periods=1, center=True).mean(), linewidth=2)
    plt.title(title)
    plt.xlabel("Week of Year")
    plt.ylabel(ylabel)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()


weekly_plot("daily_midpoint_hour", "Weekly Seasonal Trend: Sleep Midpoint (Linearized)", "07_weekly_midpoint_linearized.png", "Hour")
weekly_plot("daily_start_hour", "Weekly Seasonal Trend: Sleep Onset (Linearized)", "08_weekly_onset_linearized.png", "Hour")
weekly_plot("daily_end_hour", "Weekly Seasonal Trend: Sleep Offset (Linearized)", "09_weekly_offset_linearized.png", "Hour")
weekly_plot("daily_sleep_window_hours", "Weekly Seasonal Trend: Sleep Duration", "10_weekly_duration.png", "Hours")

print(f"Done. Saved outputs to {OUTPUT_DIR}")
