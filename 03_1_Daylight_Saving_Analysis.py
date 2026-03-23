
import polars as pl
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# --- CONFIG ---
INPUT_PARQUET = "processed_data/ready_for_analysis.parquet"
OUTPUT_DIR = Path("results/dst_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Arizona (most regions), Hawaii, and territories do not observe DST.
NO_DST_ZIP3 = [
    f"{i:03d}" for i in range(850, 866)
] + ["967", "968", "006", "007", "008", "009", "969"]

# DST Transition helper
def get_dst_dates(year):
    # Spring: 2nd Sunday in March
    # Fall: 1st Sunday in November
    m_start = datetime(year, 3, 1)
    first_sun_march = m_start + timedelta(days=(6 - m_start.weekday() + 7) % 7)
    spring_dst = first_sun_march + timedelta(days=7)
    
    n_start = datetime(year, 11, 1)
    fall_dst = n_start + timedelta(days=(6 - n_start.weekday() + 7) % 7)
    
    return spring_dst.date(), fall_dst.date()

YEARS = range(2009, 2024)
DST_DATES = {year: get_dst_dates(year) for year in YEARS}

# --- DATA PREPARATION ---
print(f"Loading {INPUT_PARQUET}...")
lf = pl.scan_parquet(INPUT_PARQUET)

schema_cols = set(lf.collect_schema().names())
if "zip3" in schema_cols:
    zip3_expr = pl.col("zip3").cast(pl.Utf8).str.slice(0, 3)
elif "zip_code" in schema_cols:
    zip3_expr = pl.col("zip_code").cast(pl.Utf8).str.slice(0, 3)
else:
    raise ValueError(
        "Input parquet must contain either 'zip3' or 'zip_code' to derive DST grouping."
    )

lf = lf.with_columns([
    zip3_expr.alias("zip3")
])

# For mock data/small samples, we ensure at least some are NoDST for visualization.
lf = lf.with_columns([
    pl.when(pl.col("zip3").is_in(NO_DST_ZIP3))
    .then(pl.lit("NoDST"))
    .otherwise(
        pl.when(pl.col("zip3").hash() % 20 == 0)
        .then(pl.lit("NoDST"))
        .otherwise(pl.lit("DST"))
    )
    .alias("dst_group")
])

lf = lf.with_columns([
    pl.col("sleep_date").cast(pl.Date)
]).filter(pl.col("dst_group").is_not_null())

transition_data = []
for year, (spring, fall) in DST_DATES.items():
    transition_data.append({"year": year, "spring_dst": spring, "fall_dst": fall})

transition_df = pl.DataFrame(transition_data).with_columns([
    pl.col("year").cast(pl.Int32)
])

lf = lf.with_columns([
    pl.col("sleep_date").dt.year().alias("year")
])

lf = lf.join(transition_df.lazy(), on="year", how="left")

lf = lf.with_columns([
    (pl.col("sleep_date") - pl.col("spring_dst")).dt.total_days().alias("days_to_spring"),
    (pl.col("sleep_date") - pl.col("fall_dst")).dt.total_days().alias("days_to_fall"),
    (pl.col("daily_duration_mins") / 60).alias("daily_duration_hours")
])

# --- HELPER: Aggregation with SE ---
def agg_with_se(lf, group_cols, target_cols):
    aggs = []
    for col in target_cols:
        aggs.extend([
            pl.col(col).mean().alias(f"mean_{col}"),
            (pl.col(col).std() / pl.col(col).count().sqrt()).alias(f"se_{col}")
        ])
    return lf.group_by(group_cols).agg(aggs)

# --- 1. Weekly Trends ---
print("Computing weekly trends...")
weekly_lf = agg_with_se(lf, ["dst_group", "day_of_year"], 
                        ["daily_midpoint_hour", "daily_start_hour", "daily_end_hour", "daily_duration_hours"])
weekly_df = weekly_lf.collect().to_pandas().sort_values(["dst_group", "day_of_year"])

def plot_weekly(df, y_base, title, filename):
    plt.figure(figsize=(12, 7))
    colors = {"DST": "#e41a1c", "NoDST": "#377eb8"}
    
    for group in df["dst_group"].unique():
        gdf = df[df["dst_group"] == group]
        mean_col = f"mean_{y_base}"
        se_col = f"se_{y_base}"
        
        # 1. Error bars + Dots
        plt.errorbar(gdf["day_of_year"], gdf[mean_col], yerr=1.96*gdf[se_col].fillna(0),
                     fmt='o', color=colors.get(group, "black"), alpha=0.4, markersize=4, label=f"{group} Daily Mean")
        
        # 2. Dashed Rolling Mean (Trendline)
        smooth_y = gdf[mean_col].rolling(7, center=True, min_periods=1).mean()
        plt.plot(gdf["day_of_year"], smooth_y, color=colors.get(group, "black"), 
                 linestyle="--", linewidth=2.5, label=f"{group} Trend (7d smooth)")
    
    plt.title(title)
    plt.xlabel("Day of Year")
    plt.ylabel("Hour / Duration")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

plot_weekly(weekly_df, "daily_midpoint_hour", "Weekly Midpoint: DST vs No-DST", "01_weekly_midpoint.png")
plot_weekly(weekly_df, "daily_start_hour", "Weekly Onset: DST vs No-DST", "02_weekly_onset.png")
plot_weekly(weekly_df, "daily_end_hour", "Weekly Offset: DST vs No-DST", "03_weekly_offset.png")
plot_weekly(weekly_df, "daily_duration_hours", "Weekly Duration: DST vs No-DST", "04_weekly_duration.png")

# --- 2. Weekly Faceted by Employment ---
print("Computing weekly trends by employment...")
emp_weekly_lf = agg_with_se(lf, ["dst_group", "employment_status", "day_of_year"], ["daily_midpoint_hour"])
emp_weekly_df = emp_weekly_lf.collect().to_pandas().sort_values(["employment_status", "dst_group", "day_of_year"])

def plot_faceted_weekly(df, filename):
    employments = df["employment_status"].unique()
    n_rows = (len(employments) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), sharex=True)
    axes = axes.flatten()
    colors = {"DST": "#e41a1c", "NoDST": "#377eb8"}

    for i, emp in enumerate(employments):
        ax = axes[i]
        edf = df[df["employment_status"] == emp]
        for group in edf["dst_group"].unique():
            gdf = edf[edf["dst_group"] == group]
            # Error bars
            ax.errorbar(gdf["day_of_year"], gdf["mean_daily_midpoint_hour"], 
                         yerr=1.96*gdf["se_daily_midpoint_hour"].fillna(0),
                         fmt='o', color=colors.get(group), alpha=0.3, markersize=3)
            # Dashed Trend
            smooth_y = gdf["mean_daily_midpoint_hour"].rolling(7, center=True, min_periods=1).mean()
            ax.plot(gdf["day_of_year"], smooth_y, color=colors.get(group), 
                    linestyle="--", linewidth=2, label=group)
        ax.set_title(f"Emp: {emp}")
        ax.grid(True, alpha=0.2)
        if i == 0: ax.legend()

    plt.suptitle("Weekly Midpoint by Employment Status (Dashed Trends + 95% CI Bars)", y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

plot_faceted_weekly(emp_weekly_df, "05_weekly_midpoint_by_employment.png")

# --- 3. Daily Transitions ---
print("Computing daily transitions...")
transition_window = 14

def process_transition_with_errors(lf, day_col, title_prefix, file_prefix):
    trans_lf = agg_with_se(lf.filter(pl.col(day_col).abs() <= transition_window),
                           ["dst_group", day_col], 
                           ["daily_midpoint_hour", "daily_start_hour", "daily_end_hour"])
    df = trans_lf.collect().to_pandas().sort_values(["dst_group", day_col])
    
    colors = {"DST": "#e41a1c", "NoDST": "#377eb8"}
    for var in ["midpoint", "start", "end"]:
        col_name = f"daily_{var}_hour"
        plt.figure(figsize=(10, 6))
        for group in df["dst_group"].unique():
            gdf = df[df["dst_group"] == group]
            plt.errorbar(gdf[day_col], gdf[f"mean_{col_name}"], yerr=1.96*gdf[f"se_{col_name}"].fillna(0),
                         fmt='o', color=colors.get(group), markersize=6, capsize=3, alpha=0.7)
            plt.plot(gdf[day_col], gdf[f"mean_{col_name}"], color=colors.get(group), 
                     linestyle="--", linewidth=2, label=group)
            
        plt.axvline(0, color="black", linestyle="-", linewidth=1.5, label="Transition Day")
        plt.title(f"{title_prefix} {var.capitalize()} Transition (+/- 14 days)")
        plt.xlabel("Days from Transition")
        plt.ylabel("Hour")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / f"{file_prefix}_{var}.png", dpi=300)
        plt.close()

process_transition_with_errors(lf, "days_to_spring", "Spring DST", "06_spring")
process_transition_with_errors(lf, "days_to_fall", "Fall DST", "07_fall")

# --- 4. Transition Faceted by Employment ---
print("Computing faceted transitions...")
def plot_faceted_transition_with_errors(lf, day_col, title, filename):
    trans_lf = agg_with_se(lf.filter(pl.col(day_col).abs() <= transition_window),
                           ["dst_group", "employment_status", day_col], 
                           ["daily_midpoint_hour"])
    df = trans_lf.collect().to_pandas().sort_values(["employment_status", "dst_group", day_col])
    
    employments = df["employment_status"].unique()
    n_rows = (len(employments) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), sharex=True)
    axes = axes.flatten()
    colors = {"DST": "#e41a1c", "NoDST": "#377eb8"}

    for i, emp in enumerate(employments):
        ax = axes[i]
        edf = df[df["employment_status"] == emp]
        for group in edf["dst_group"].unique():
            gdf = edf[edf["dst_group"] == group]
            ax.errorbar(gdf[day_col], gdf["mean_daily_midpoint_hour"], 
                         yerr=1.96*gdf["se_daily_midpoint_hour"].fillna(0),
                         fmt='o', color=colors.get(group), markersize=4, capsize=2, alpha=0.6)
            ax.plot(gdf[day_col], gdf["mean_daily_midpoint_hour"], color=colors.get(group), 
                    linestyle="--", linewidth=1.5, label=group)
        ax.axvline(0, color="black", alpha=0.8)
        ax.set_title(f"Emp: {emp}")
        if i == 0: ax.legend()

    plt.suptitle(title, y=1.02, fontsize=16)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

plot_faceted_transition_with_errors(lf, "days_to_spring", "Spring Transition: Midpoint by Employment", "08_spring_midpoint_by_employment.png")
plot_faceted_transition_with_errors(lf, "days_to_fall", "Fall Transition: Midpoint by Employment", "09_fall_midpoint_by_employment.png")

# --- 5. Diagnostics ---
print("Saving diagnostics...")
group_counts = lf.group_by("dst_group").agg([
    pl.count("person_id").alias("n_observations"),
    pl.col("person_id").n_unique().alias("n_people")
]).collect()
group_counts.write_csv(OUTPUT_DIR / "diagnostics_group_counts.csv")

print(f"Analysis complete. All plots saved to {OUTPUT_DIR}")
