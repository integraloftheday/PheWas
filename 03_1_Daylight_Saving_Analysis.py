
import polars as pl
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter
from scipy import stats
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

SPRING_DST_DOY = int(round(np.mean([spring.timetuple().tm_yday for spring, _ in DST_DATES.values()])))
FALL_DST_DOY = int(round(np.mean([fall.timetuple().tm_yday for _, fall in DST_DATES.values()])))

MONTH_STARTS_NON_LEAP = [
    (1, "Jan"), (32, "Feb"), (60, "Mar"), (91, "Apr"), (121, "May"), (152, "Jun"),
    (182, "Jul"), (213, "Aug"), (244, "Sep"), (274, "Oct"), (305, "Nov"), (335, "Dec")
]

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

MIDPOINT_COL = "midpoint_linear" if "midpoint_linear" in schema_cols else "daily_midpoint_hour"
ONSET_COL = "onset_linear" if "onset_linear" in schema_cols else "daily_start_hour"
OFFSET_COL = "offset_linear" if "offset_linear" in schema_cols else "daily_end_hour"

print(
    "Using timing columns for DST analysis: "
    f"onset={ONSET_COL}, midpoint={MIDPOINT_COL}, offset={OFFSET_COL}"
)

TIME_AXIS_LABEL = "Sleep timing (HH:MM; linearized/noon-to-noon scale when applicable)"
DURATION_AXIS_LABEL = "Sleep duration (HH:MM/night)"
TIME_METRIC_COLS = {MIDPOINT_COL, ONSET_COL, OFFSET_COL}


def decimal_hours_to_hhmm(value, wrap_24=False):
    if pd.isna(value) or np.isinf(value):
        return ""
    sign = "-" if value < 0 else ""
    total_minutes = int(round(abs(float(value)) * 60))
    hours, minutes = divmod(total_minutes, 60)

    if wrap_24:
        day_offset, hour24 = divmod(hours, 24)
        if day_offset == 0:
            return f"{sign}{hour24:02d}:{minutes:02d}"
        return f"{sign}{hour24:02d}:{minutes:02d} (+{day_offset}d)"

    return f"{sign}{hours:02d}:{minutes:02d}"


def apply_hhmm_axis_formatter(ax, metric_col):
    wrap_24 = metric_col in TIME_METRIC_COLS
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda x, pos: decimal_hours_to_hhmm(x, wrap_24=wrap_24))
    )


def apply_weekly_month_ticks(ax, max_day=366):
    ticks = [d for d, _ in MONTH_STARTS_NON_LEAP if d <= max_day]
    labels = [f"{d}\n{m}" for d, m in MONTH_STARTS_NON_LEAP if d <= max_day]
    if ticks:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)


def add_weekly_dst_transition_lines(ax, show_labels=False):
    spring_label = "Spring DST transition" if show_labels else None
    fall_label = "Fall DST transition" if show_labels else None
    ax.axvline(SPRING_DST_DOY, color="#636363", linestyle="--", linewidth=1.3, alpha=0.9, label=spring_label)
    ax.axvline(FALL_DST_DOY, color="#636363", linestyle="--", linewidth=1.3, alpha=0.9, label=fall_label)

lf = lf.with_columns([
    zip3_expr.alias("zip3")
])

# Assign DST group from known non-DST ZIP3 prefixes.
lf = lf.with_columns([
    pl.when(pl.col("zip3").is_in(NO_DST_ZIP3))
    .then(pl.lit("NoDST"))
    .otherwise(pl.lit("DST"))
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
                        [MIDPOINT_COL, ONSET_COL, OFFSET_COL, "daily_duration_hours"])
weekly_df = weekly_lf.collect().to_pandas().sort_values(["dst_group", "day_of_year"])

def plot_weekly(df, y_base, title, filename, y_label):
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

    add_weekly_dst_transition_lines(plt.gca(), show_labels=True)
    apply_weekly_month_ticks(plt.gca(), max_day=int(df["day_of_year"].max()))
    apply_hhmm_axis_formatter(plt.gca(), y_base)
    
    plt.title(title)
    plt.xlabel("Day of year + month")
    plt.ylabel(y_label)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

plot_weekly(
    weekly_df,
    MIDPOINT_COL,
    "Weekly sleep midpoint by DST-observing status (points=day mean ±95% CI, dashed line=7-day smoothed trend)",
    "01_weekly_midpoint.png",
    TIME_AXIS_LABEL,
)
plot_weekly(
    weekly_df,
    ONSET_COL,
    "Weekly sleep onset by DST-observing status (points=day mean ±95% CI, dashed line=7-day smoothed trend)",
    "02_weekly_onset.png",
    TIME_AXIS_LABEL,
)
plot_weekly(
    weekly_df,
    OFFSET_COL,
    "Weekly sleep offset by DST-observing status (points=day mean ±95% CI, dashed line=7-day smoothed trend)",
    "03_weekly_offset.png",
    TIME_AXIS_LABEL,
)
plot_weekly(
    weekly_df,
    "daily_duration_hours",
    "Weekly sleep duration by DST-observing status (points=day mean ±95% CI, dashed line=7-day smoothed trend)",
    "04_weekly_duration.png",
    DURATION_AXIS_LABEL,
)

# --- 2. Weekly Faceted by Employment ---
print("Computing weekly trends by employment...")
emp_weekly_lf = agg_with_se(
    lf,
    ["dst_group", "employment_status", "day_of_year"],
    [MIDPOINT_COL, ONSET_COL, OFFSET_COL, "daily_duration_hours"]
)
emp_weekly_df = emp_weekly_lf.collect().to_pandas().sort_values(["employment_status", "dst_group", "day_of_year"])

def plot_faceted_weekly(df, y_base, title, filename, y_label="Value"):
    employments = df["employment_status"].unique()
    n_rows = (len(employments) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), sharex=True)
    axes = axes.flatten()
    colors = {"DST": "#e41a1c", "NoDST": "#377eb8"}
    mean_col = f"mean_{y_base}"
    se_col = f"se_{y_base}"

    for i, emp in enumerate(employments):
        ax = axes[i]
        edf = df[df["employment_status"] == emp]
        for group in edf["dst_group"].unique():
            gdf = edf[edf["dst_group"] == group]
            # Error bars
            ax.errorbar(gdf["day_of_year"], gdf[mean_col], 
                         yerr=1.96*gdf[se_col].fillna(0),
                         fmt='o', color=colors.get(group), alpha=0.3, markersize=3)
            # Dashed Trend
            smooth_y = gdf[mean_col].rolling(7, center=True, min_periods=1).mean()
            ax.plot(gdf["day_of_year"], smooth_y, color=colors.get(group), 
                    linestyle="--", linewidth=2, label=group)
            add_weekly_dst_transition_lines(ax, show_labels=False)
            apply_weekly_month_ticks(ax, max_day=int(df["day_of_year"].max()))
            apply_hhmm_axis_formatter(ax, y_base)
        ax.set_title(f"Emp: {emp}")
        ax.grid(True, alpha=0.2)
        if i == 0: ax.legend()

    for j in range(len(employments), len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, y=1.02, fontsize=16)
    fig.supxlabel("Day of year + month")
    fig.supylabel(y_label)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

plot_faceted_weekly(
    emp_weekly_df,
    MIDPOINT_COL,
    "Weekly sleep midpoint by employment status and DST-observing status (points=day mean ±95% CI, dashed line=7-day smoothed trend)",
    "05_weekly_midpoint_by_employment.png",
    TIME_AXIS_LABEL
)
plot_faceted_weekly(
    emp_weekly_df,
    ONSET_COL,
    "Weekly sleep onset by employment status and DST-observing status (points=day mean ±95% CI, dashed line=7-day smoothed trend)",
    "05b_weekly_onset_by_employment.png",
    TIME_AXIS_LABEL
)
plot_faceted_weekly(
    emp_weekly_df,
    OFFSET_COL,
    "Weekly sleep offset by employment status and DST-observing status (points=day mean ±95% CI, dashed line=7-day smoothed trend)",
    "05c_weekly_offset_by_employment.png",
    TIME_AXIS_LABEL
)
plot_faceted_weekly(
    emp_weekly_df,
    "daily_duration_hours",
    "Weekly sleep duration by employment status and DST-observing status (points=day mean ±95% CI, dashed line=7-day smoothed trend)",
    "05d_weekly_duration_by_employment.png",
    DURATION_AXIS_LABEL
)

# --- 3. Daily Transitions ---
print("Computing daily transitions...")
transition_window = 14

def process_transition_with_errors(lf, day_col, title_prefix, file_prefix):
    trans_lf = agg_with_se(lf.filter(pl.col(day_col).abs() <= transition_window),
                           ["dst_group", day_col], 
                           [MIDPOINT_COL, ONSET_COL, OFFSET_COL, "daily_duration_hours"])
    df = trans_lf.collect().to_pandas().sort_values(["dst_group", day_col])
    
    colors = {"DST": "#e41a1c", "NoDST": "#377eb8"}
    metric_specs = [
        ("sleep midpoint", MIDPOINT_COL, TIME_AXIS_LABEL),
        ("sleep onset", ONSET_COL, TIME_AXIS_LABEL),
        ("sleep offset", OFFSET_COL, TIME_AXIS_LABEL),
        ("sleep duration", "daily_duration_hours", DURATION_AXIS_LABEL),
    ]
    for var, col_name, y_label in metric_specs:
        plt.figure(figsize=(10, 6))
        for group in df["dst_group"].unique():
            gdf = df[df["dst_group"] == group]
            plt.errorbar(gdf[day_col], gdf[f"mean_{col_name}"], yerr=1.96*gdf[f"se_{col_name}"].fillna(0),
                         fmt='o', color=colors.get(group), markersize=6, capsize=3, alpha=0.7)
            plt.plot(gdf[day_col], gdf[f"mean_{col_name}"], color=colors.get(group), 
                     linestyle="--", linewidth=2, label=group)
            
        plt.axvline(0, color="black", linestyle="-", linewidth=1.5, label="Transition Day")
        plt.title(
            f"{title_prefix} transition effect on {var} (±14 days around transition date; points=day mean ±95% CI, dashed line=trend)"
        )
        plt.xlabel("Days from DST transition date")
        plt.ylabel(y_label)
        apply_hhmm_axis_formatter(plt.gca(), col_name)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(OUTPUT_DIR / f"{file_prefix}_{var}.png", dpi=300)
        plt.close()

process_transition_with_errors(lf, "days_to_spring", "Spring DST", "06_spring")
process_transition_with_errors(lf, "days_to_fall", "Fall DST", "07_fall")

# --- 4. Transition Faceted by Employment ---
print("Computing faceted transitions...")
def plot_faceted_transition_with_errors(lf, day_col, y_base, title, filename, y_label="Value"):
    trans_lf = agg_with_se(lf.filter(pl.col(day_col).abs() <= transition_window),
                           ["dst_group", "employment_status", day_col], 
                           [y_base])
    df = trans_lf.collect().to_pandas().sort_values(["employment_status", "dst_group", day_col])
    
    employments = df["employment_status"].unique()
    n_rows = (len(employments) + 2) // 3
    fig, axes = plt.subplots(n_rows, 3, figsize=(18, 5 * n_rows), sharex=True)
    axes = axes.flatten()
    colors = {"DST": "#e41a1c", "NoDST": "#377eb8"}
    mean_col = f"mean_{y_base}"
    se_col = f"se_{y_base}"

    for i, emp in enumerate(employments):
        ax = axes[i]
        edf = df[df["employment_status"] == emp]
        for group in edf["dst_group"].unique():
            gdf = edf[edf["dst_group"] == group]
            ax.errorbar(gdf[day_col], gdf[mean_col], 
                         yerr=1.96*gdf[se_col].fillna(0),
                         fmt='o', color=colors.get(group), markersize=4, capsize=2, alpha=0.6)
            ax.plot(gdf[day_col], gdf[mean_col], color=colors.get(group), 
                    linestyle="--", linewidth=1.5, label=group)
        ax.axvline(0, color="black", alpha=0.8)
        apply_hhmm_axis_formatter(ax, y_base)
        ax.set_title(f"Emp: {emp}")
        if i == 0: ax.legend()

    for j in range(len(employments), len(axes)):
        axes[j].axis("off")

    plt.suptitle(title, y=1.02, fontsize=16)
    fig.supxlabel("Days from DST transition date")
    fig.supylabel(y_label)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300)
    plt.close()

plot_faceted_transition_with_errors(
    lf,
    "days_to_spring",
    MIDPOINT_COL,
    "Spring DST transition effect on sleep midpoint by employment and DST-observing status (±14 days; points=day mean ±95% CI, dashed line=trend)",
    "08_spring_midpoint_by_employment.png",
    TIME_AXIS_LABEL
)
plot_faceted_transition_with_errors(
    lf,
    "days_to_fall",
    MIDPOINT_COL,
    "Fall DST transition effect on sleep midpoint by employment and DST-observing status (±14 days; points=day mean ±95% CI, dashed line=trend)",
    "09_fall_midpoint_by_employment.png",
    TIME_AXIS_LABEL
)
plot_faceted_transition_with_errors(
    lf,
    "days_to_spring",
    ONSET_COL,
    "Spring DST transition effect on sleep onset by employment and DST-observing status (±14 days; points=day mean ±95% CI, dashed line=trend)",
    "08b_spring_onset_by_employment.png",
    TIME_AXIS_LABEL
)
plot_faceted_transition_with_errors(
    lf,
    "days_to_fall",
    ONSET_COL,
    "Fall DST transition effect on sleep onset by employment and DST-observing status (±14 days; points=day mean ±95% CI, dashed line=trend)",
    "09b_fall_onset_by_employment.png",
    TIME_AXIS_LABEL
)
plot_faceted_transition_with_errors(
    lf,
    "days_to_spring",
    OFFSET_COL,
    "Spring DST transition effect on sleep offset by employment and DST-observing status (±14 days; points=day mean ±95% CI, dashed line=trend)",
    "08c_spring_offset_by_employment.png",
    TIME_AXIS_LABEL
)
plot_faceted_transition_with_errors(
    lf,
    "days_to_fall",
    OFFSET_COL,
    "Fall DST transition effect on sleep offset by employment and DST-observing status (±14 days; points=day mean ±95% CI, dashed line=trend)",
    "09c_fall_offset_by_employment.png",
    TIME_AXIS_LABEL
)
plot_faceted_transition_with_errors(
    lf,
    "days_to_spring",
    "daily_duration_hours",
    "Spring DST transition effect on sleep duration by employment and DST-observing status (±14 days; points=day mean ±95% CI, dashed line=trend)",
    "08d_spring_duration_by_employment.png",
    DURATION_AXIS_LABEL
)
plot_faceted_transition_with_errors(
    lf,
    "days_to_fall",
    "daily_duration_hours",
    "Fall DST transition effect on sleep duration by employment and DST-observing status (±14 days; points=day mean ±95% CI, dashed line=trend)",
    "09d_fall_duration_by_employment.png",
    DURATION_AXIS_LABEL
)

# --- 5. ZIP3 Map of DST Filtering ---
print("Building ZIP3 DST filter map...")
def plot_dst_zip3_map(lf):
    gdb_path = Path("environmental_data/v108/zip3.gdb")
    if not gdb_path.exists():
        print(f"ZIP3 geodatabase not found at {gdb_path}; skipping map.")
        return

    try:
        import geopandas as gpd  # type: ignore[import-not-found]
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        from matplotlib import colors as mcolors
    except Exception as exc:
        print(f"Could not import geopandas ({exc}); skipping map.")
        return

    zip_present = (
        lf.select(pl.col("zip3").cast(pl.Utf8).str.slice(0, 3).alias("zip3"))
        .drop_nulls()
        .unique()
        .collect()
        .to_pandas()["zip3"]
        .astype(str)
        .str.zfill(3)
        .str.slice(0, 3)
    )
    observed_zip3 = set(zip_present.tolist())

    participant_counts = (
        lf.select([
            pl.col("zip3").cast(pl.Utf8).str.slice(0, 3).alias("zip3"),
            pl.col("person_id"),
        ])
        .drop_nulls()
        .group_by("zip3")
        .agg(pl.col("person_id").n_unique().alias("n_people"))
        .collect()
        .to_pandas()
    )
    participant_counts["zip3"] = participant_counts["zip3"].astype(str).str.zfill(3).str.slice(0, 3)

    try:
        gdf = gpd.read_file(str(gdb_path))
    except Exception as exc:
        print(f"Could not read {gdb_path} ({exc}); skipping map.")
        return

    zip_col_candidates = [
        col for col in gdf.columns
        if any(token in str(col).lower() for token in ["zip3", "zcta3", "zip", "geoid"])
    ]
    if not zip_col_candidates:
        print("No ZIP-like column found in ZIP3 geodatabase; skipping map.")
        return

    zip_col = zip_col_candidates[0]
    gdf["zip3"] = gdf[zip_col].astype(str).str.zfill(3).str.slice(0, 3)
    gdf = gdf.merge(participant_counts, on="zip3", how="left")
    gdf["n_people"] = gdf["n_people"].fillna(0)

    if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    gdf["filter_group"] = np.where(gdf["zip3"].isin(NO_DST_ZIP3), "NoDST filter", "DST filter")
    gdf["in_dataset"] = gdf["zip3"].isin(observed_zip3)

    pts = gdf.geometry.representative_point()
    lon = pts.x
    lat = pts.y

    contiguous_mask = lon.between(-125, -66) & lat.between(24, 50)
    hawaii_mask = lon.between(-161, -154) & lat.between(18, 23)
    us_map = gdf[contiguous_mask | hawaii_mask].copy()
    contig = us_map[contiguous_mask.loc[us_map.index]].copy()
    hawaii = us_map[hawaii_mask.loc[us_map.index]].copy()

    if us_map.empty:
        print("Filtered US ZIP3 geometry is empty; skipping map.")
        return

    fig = plt.figure(figsize=(14, 9))
    ax_main = fig.add_axes([0.04, 0.06, 0.76, 0.88])
    ax_hi = fig.add_axes([0.78, 0.12, 0.2, 0.24])

    def draw_region(ax, region_df, xlim, ylim):
        region_df.plot(ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.2, aspect="auto")

        nodst = region_df[region_df["filter_group"] == "NoDST filter"]
        if not nodst.empty:
            nodst.plot(
                ax=ax,
                color="#377eb8",
                edgecolor="#08306b",
                linewidth=0.35,
                alpha=0.9,
                aspect="auto",
            )

        observed = region_df[region_df["in_dataset"]]
        if not observed.empty:
            observed.boundary.plot(ax=ax, color="#1f1f1f", linewidth=0.35, alpha=0.7)

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_axis_off()
        ax.set_aspect("auto")

    draw_region(ax_main, contig, xlim=(-125, -66), ylim=(24, 50))
    draw_region(ax_hi, hawaii, xlim=(-161, -154), ylim=(18, 23))
    ax_hi.set_title("Hawaii", fontsize=10)

    legend_handles = [
        Patch(facecolor="#f0f0f0", edgecolor="#d0d0d0", label="ZIP3 outlines"),
        Patch(facecolor="#377eb8", edgecolor="#08306b", label="NoDST ZIP3 filter"),
        Line2D([0], [0], color="#1f1f1f", linewidth=1.2, label="ZIP3 present in dataset"),
    ]
    ax_main.legend(handles=legend_handles, loc="lower left", frameon=True)
    fig.suptitle("DST ZIP3 Filtering Map (Continental US + Hawaii)", y=0.98, fontsize=14)

    map_summary = pd.DataFrame([
        {
            "zip3_total_in_map": int(len(us_map)),
            "zip3_observed_in_dataset": int(us_map["in_dataset"].sum()),
            "zip3_nodst_filter_in_map": int((us_map["filter_group"] == "NoDST filter").sum()),
            "zip3_nodst_filter_observed": int(((us_map["filter_group"] == "NoDST filter") & us_map["in_dataset"]).sum()),
        }
    ])
    map_summary.to_csv(OUTPUT_DIR / "10_dst_zip3_filter_map_summary.csv", index=False)

    plt.savefig(OUTPUT_DIR / "10_dst_zip3_filter_map.png", dpi=300)
    plt.close()

    # --- Participant count map (same layout) ---
    fig2 = plt.figure(figsize=(14, 9))
    ax2_main = fig2.add_axes([0.04, 0.12, 0.76, 0.82])
    ax2_hi = fig2.add_axes([0.78, 0.18, 0.2, 0.24])
    cax = fig2.add_axes([0.12, 0.05, 0.58, 0.03])

    count_df = us_map.copy()
    count_df = count_df[count_df["n_people"] > 0]
    count_df["n_people_clipped"] = count_df["n_people"].clip(lower=20)

    vmax = float(count_df["n_people_clipped"].max()) if not count_df.empty else 21.0
    vmin = 20.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    cmap = plt.cm.viridis.copy()
    cmap.set_under("#440154")
    norm = mcolors.LogNorm(vmin=vmin, vmax=vmax, clip=False)

    def draw_count_region(ax, region_df, xlim, ylim):
        region_df.plot(ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.2, aspect="auto")
        filled = region_df[region_df["n_people"] > 0].copy()
        if not filled.empty:
            filled["n_people_clipped"] = filled["n_people"].clip(lower=20)
            filled.plot(
                ax=ax,
                column="n_people_clipped",
                cmap=cmap,
                norm=norm,
                edgecolor="#303030",
                linewidth=0.2,
                aspect="auto",
            )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_axis_off()
        ax.set_aspect("auto")

    contig_count = contig.copy()
    hawaii_count = hawaii.copy()
    contig_count["n_people"] = contig_count["n_people"].fillna(0)
    hawaii_count["n_people"] = hawaii_count["n_people"].fillna(0)

    draw_count_region(ax2_main, contig_count, xlim=(-125, -66), ylim=(24, 50))
    draw_count_region(ax2_hi, hawaii_count, xlim=(-161, -154), ylim=(18, 23))
    ax2_hi.set_title("Hawaii", fontsize=10)

    if count_df.empty:
        ax2_main.text(
            0.5,
            0.5,
            "No observed ZIP3 with participants\nin continental US + Hawaii",
            ha="center",
            va="center",
            transform=ax2_main.transAxes,
            fontsize=11,
            color="#333333",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#999999"),
        )

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cb = fig2.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_label("Unique participants per ZIP3 (log scale; all values < 20 use same color)")

    fig2.suptitle("Participant Count by ZIP3 (Continental US + Hawaii)", y=0.98, fontsize=14)
    plt.savefig(OUTPUT_DIR / "11_zip3_participant_count_map.png", dpi=300)
    plt.close(fig2)

    # --- Bubble map: participant count at ZIP3 representative points ---
    fig3 = plt.figure(figsize=(14, 9))
    ax3_main = fig3.add_axes([0.04, 0.12, 0.76, 0.82])
    ax3_hi = fig3.add_axes([0.78, 0.18, 0.2, 0.24])
    cax3 = fig3.add_axes([0.12, 0.05, 0.58, 0.03])

    def draw_bubble_region(ax, region_df, xlim, ylim):
        region_df.plot(ax=ax, color="#f0f0f0", edgecolor="#d0d0d0", linewidth=0.2, aspect="auto")

        points_df = region_df[region_df["n_people"] > 0].copy()
        if not points_df.empty:
            points_df["n_people_clipped"] = points_df["n_people"].clip(lower=20)
            rep_points = points_df.geometry.representative_point()
            points_df["lon"] = rep_points.x
            points_df["lat"] = rep_points.y

            scale = np.log10(points_df["n_people_clipped"]) - np.log10(vmin)
            denom = max(np.log10(vmax) - np.log10(vmin), 1e-6)
            scale = scale / denom
            sizes = 20 + 220 * scale

            ax.scatter(
                points_df["lon"],
                points_df["lat"],
                c=points_df["n_people_clipped"],
                s=sizes,
                cmap=cmap,
                norm=norm,
                alpha=0.8,
                edgecolors="#1f1f1f",
                linewidths=0.25,
            )

        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_axis_off()
        ax.set_aspect("auto")

    draw_bubble_region(ax3_main, contig_count, xlim=(-125, -66), ylim=(24, 50))
    draw_bubble_region(ax3_hi, hawaii_count, xlim=(-161, -154), ylim=(18, 23))
    ax3_hi.set_title("Hawaii", fontsize=10)

    if count_df.empty:
        ax3_main.text(
            0.5,
            0.5,
            "No observed ZIP3 with participants\nin continental US + Hawaii",
            ha="center",
            va="center",
            transform=ax3_main.transAxes,
            fontsize=11,
            color="#333333",
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="#999999"),
        )

    sm3 = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm3.set_array([])
    cb3 = fig3.colorbar(sm3, cax=cax3, orientation="horizontal")
    cb3.set_label("Unique participants per ZIP3 (log scale)")

    # Bubble size legend
    size_breaks = [20, 50, 100, 200, 500]
    size_breaks = [v for v in size_breaks if v <= vmax]
    if not size_breaks:
        size_breaks = [20]
    legend_sizes = []
    for v in size_breaks:
        v_clip = max(v, vmin)
        s = 20 + 220 * ((np.log10(v_clip) - np.log10(vmin)) / max(np.log10(vmax) - np.log10(vmin), 1e-6))
        legend_sizes.append(s)
    size_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#666666', markeredgecolor='#1f1f1f', markersize=np.sqrt(s))
        for s in legend_sizes
    ]
    ax3_main.legend(size_handles, [str(v) for v in size_breaks], title="Participants", loc="lower left", frameon=True)

    fig3.suptitle("Participant Count Bubble Map by ZIP3 (Continental US + Hawaii)", y=0.98, fontsize=14)
    plt.savefig(OUTPUT_DIR / "12_zip3_participant_bubble_map.png", dpi=300)
    plt.close(fig3)

plot_dst_zip3_map(lf)

# --- 6. Statistical testing (DiD-style, person-level pre/post deltas) ---
print("Running statistical DST tests (DiD-style)...")

def did_person_level_test(lf, day_col, metric_col, transition_name, window=14):
    # Build person-year level pre/post deltas: delta = post_mean - pre_mean
    # and compare deltas between DST vs NoDST groups.
    # This is a Difference-in-Differences style estimate:
    #   (post-pre in DST group) - (post-pre in NoDST group)
    base = (
        lf.filter(pl.col(day_col).is_between(-window, window))
        .select([
            "person_id",
            "year",
            "dst_group",
            pl.col(day_col).alias("day_from_transition"),
            pl.col(metric_col).alias("metric_value"),
        ])
        .drop_nulls(["person_id", "year", "dst_group", "day_from_transition", "metric_value"])
        .with_columns(
            pl.when(pl.col("day_from_transition") < 0)
            .then(pl.lit("pre"))
            .when(pl.col("day_from_transition") > 0)
            .then(pl.lit("post"))
            .otherwise(pl.lit(None))
            .alias("period")
        )
        .drop_nulls(["period"])
    )

    # Average within person-year-period, then pivot to get pre/post side-by-side.
    person_period = (
        base.group_by(["person_id", "year", "dst_group", "period"])
        .agg(pl.col("metric_value").mean().alias("period_mean"))
        .collect()
    )

    if person_period.height == 0:
        return {
            "transition": transition_name,
            "metric": metric_col,
            "n_dst": 0,
            "n_nodst": 0,
            "mean_delta_dst": np.nan,
            "mean_delta_nodst": np.nan,
            "did_estimate": np.nan,
            "se_diff": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "test": "Welch t-test on person-level deltas",
            "window_days": window,
        }

    person_period_pd = person_period.to_pandas()

    pivoted = (
        person_period_pd
        .pivot_table(
            index=["person_id", "year", "dst_group"],
            columns="period",
            values="period_mean",
            aggfunc="mean",
        )
        .reset_index()
    )

    if "pre" not in pivoted.columns or "post" not in pivoted.columns:
        return {
            "transition": transition_name,
            "metric": metric_col,
            "n_dst": 0,
            "n_nodst": 0,
            "mean_delta_dst": np.nan,
            "mean_delta_nodst": np.nan,
            "did_estimate": np.nan,
            "se_diff": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "p_value": np.nan,
            "test": "Welch t-test on person-level deltas",
            "window_days": window,
        }

    pivoted = pivoted.dropna(subset=["pre", "post"]).copy()
    pivoted["delta"] = pivoted["post"] - pivoted["pre"]

    dst_delta = pivoted.loc[pivoted["dst_group"] == "DST", "delta"].dropna().to_numpy()
    nodst_delta = pivoted.loc[pivoted["dst_group"] == "NoDST", "delta"].dropna().to_numpy()

    n_dst = int(dst_delta.shape[0])
    n_nodst = int(nodst_delta.shape[0])

    mean_dst = float(np.mean(dst_delta)) if n_dst > 0 else np.nan
    mean_nodst = float(np.mean(nodst_delta)) if n_nodst > 0 else np.nan
    did_est = mean_dst - mean_nodst if (n_dst > 0 and n_nodst > 0) else np.nan

    if n_dst >= 2 and n_nodst >= 2:
        t_res = stats.ttest_ind(dst_delta, nodst_delta, equal_var=False, nan_policy="omit")
        var_dst = float(np.var(dst_delta, ddof=1))
        var_nodst = float(np.var(nodst_delta, ddof=1))
        se_diff = float(np.sqrt((var_dst / n_dst) + (var_nodst / n_nodst)))
        ci95_low = did_est - 1.96 * se_diff
        ci95_high = did_est + 1.96 * se_diff
        t_stat = float(t_res.statistic)
        p_value = float(t_res.pvalue)
    else:
        se_diff = np.nan
        ci95_low = np.nan
        ci95_high = np.nan
        t_stat = np.nan
        p_value = np.nan

    return {
        "transition": transition_name,
        "metric": metric_col,
        "n_dst": n_dst,
        "n_nodst": n_nodst,
        "mean_delta_dst": mean_dst,
        "mean_delta_nodst": mean_nodst,
        "did_estimate": did_est,
        "se_diff": se_diff,
        "ci95_low": ci95_low,
        "ci95_high": ci95_high,
        "t_stat": t_stat,
        "p_value": p_value,
        "test": "Welch t-test on person-level deltas",
        "window_days": window,
    }


test_metrics = [MIDPOINT_COL, ONSET_COL, OFFSET_COL, "daily_duration_hours"]
stats_rows = []

for metric in test_metrics:
    stats_rows.append(did_person_level_test(lf, "days_to_spring", metric, "spring", window=14))
    stats_rows.append(did_person_level_test(lf, "days_to_fall", metric, "fall", window=14))

stats_df = pd.DataFrame(stats_rows)

# FDR correction across all tests (Benjamini-Hochberg)
if "p_value" in stats_df.columns:
    pvals = stats_df["p_value"].to_numpy(dtype=float)
    valid = np.isfinite(pvals)
    qvals = np.full_like(pvals, np.nan, dtype=float)
    if valid.sum() > 0:
        p_valid = pvals[valid]
        order = np.argsort(p_valid)
        ranks = np.arange(1, len(p_valid) + 1)
        bh = p_valid[order] * len(p_valid) / ranks
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        bh = np.clip(bh, 0, 1)
        qtmp = np.empty_like(p_valid)
        qtmp[order] = bh
        qvals[valid] = qtmp
    stats_df["q_value_bh"] = qvals

stats_df.to_csv(OUTPUT_DIR / "13_dst_did_person_level_stats.csv", index=False)
print(f"Saved DST statistical tests to {OUTPUT_DIR / '13_dst_did_person_level_stats.csv'}")

# --- 7. Descriptive table (Overall vs DST/NoDST) ---
print("Building DST descriptive table...")

TABLE_OUTPUT = Path("results") / "dst_descriptive_table_by_group.csv"
TABLE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)


def _safe_n_label(n):
    return f"{int(n):,}" if pd.notna(n) else "0"


def _fmt_pct(n, denom):
    if denom <= 0 or pd.isna(n):
        return "0 (0%)"
    pct = 100.0 * float(n) / float(denom)
    pct_str = f"{pct:.1f}".rstrip("0").rstrip(".")
    return f"{int(n):,} ({pct_str}%)"


def _fmt_categorical_cell(n, denom, suppress_lt20=True):
    if pd.isna(n) or n <= 0:
        return "0 (0%)"
    if suppress_lt20 and n < 20:
        return "<20"
    return _fmt_pct(n, denom)


def _fmt_continuous(values, decimals=0, suppress_lt20=True):
    arr = pd.Series(values).dropna()
    n = arr.shape[0]
    if n == 0:
        return "NA"
    if suppress_lt20 and n < 20:
        return "<20"
    mean_v = arr.mean()
    sd_v = arr.std(ddof=1) if n > 1 else 0.0
    min_v = arr.min()
    max_v = arr.max()
    if decimals == 0:
        return f"{mean_v:.0f} ({sd_v:.0f}) [{min_v:.0f}, {max_v:.0f}]"
    return f"{mean_v:.{decimals}f} ({sd_v:.{decimals}f}) [{min_v:.{decimals}f}, {max_v:.{decimals}f}]"


def _normalize_sex_concept(value):
    if pd.isna(value):
        return "Other/Unknown"
    txt = str(value).strip().lower()
    if "female" in txt:
        return "Female"
    if "male" in txt and "female" not in txt:
        return "Male"
    return "Other/Unknown"


person_agg_exprs = [
    pl.len().alias("n_nights"),
    pl.col("daily_duration_mins").mean().alias("avg_nightly_duration_mins"),
]

if "age_at_sleep" in schema_cols:
    person_agg_exprs.append(pl.col("age_at_sleep").mean().alias("age_years"))
if "sex_concept" in schema_cols:
    person_agg_exprs.append(pl.col("sex_concept").drop_nulls().first().alias("sex_concept"))
if "employment_status" in schema_cols:
    person_agg_exprs.append(pl.col("employment_status").drop_nulls().first().alias("employment_status"))

person_agg_exprs.append(pl.col("dst_group").drop_nulls().first().alias("dst_group"))

person_df = (
    lf.select([
        "person_id",
        "dst_group",
        "daily_duration_mins",
        *(["age_at_sleep"] if "age_at_sleep" in schema_cols else []),
        *(["sex_concept"] if "sex_concept" in schema_cols else []),
        *(["employment_status"] if "employment_status" in schema_cols else []),
    ])
    .drop_nulls(["person_id", "dst_group"])
    .group_by("person_id")
    .agg(person_agg_exprs)
    .collect()
    .to_pandas()
)

if "sex_concept" in person_df.columns:
    person_df["sex_group"] = person_df["sex_concept"].apply(_normalize_sex_concept)
else:
    person_df["sex_group"] = "Other/Unknown"

if "employment_status" in person_df.columns:
    person_df["employment_status"] = person_df["employment_status"].fillna("Unknown")
else:
    person_df["employment_status"] = "Unknown"

group_subsets = {
    "Overall": person_df,
    "DST": person_df[person_df["dst_group"] == "DST"].copy(),
    "NoDST": person_df[person_df["dst_group"] == "NoDST"].copy(),
}

column_order = ["Overall", "DST", "NoDST"]
col_headers = {
    k: f"{k} N = {_safe_n_label(v['person_id'].nunique() if not v.empty else 0)}"
    for k, v in group_subsets.items()
}

table_rows = []

def add_row(var_name, formatter):
    row = {"Variable": var_name}
    for col in column_order:
        row[col_headers[col]] = formatter(group_subsets[col])
    table_rows.append(row)


add_row("Age (Years)", lambda d: _fmt_continuous(d["age_years"], decimals=0) if "age_years" in d.columns else "NA")

table_rows.append({"Variable": "Sex Concept", **{col_headers[c]: "" for c in column_order}})
for sex_level in ["Female", "Male"]:
    add_row(
        f"  {sex_level}",
        lambda d, lvl=sex_level: _fmt_categorical_cell((d["sex_group"] == lvl).sum(), d["person_id"].nunique())
    )

table_rows.append({"Variable": "Employment Status", **{col_headers[c]: "" for c in column_order}})
preferred_emp_order = [
    "Employed For Wages",
    "Homemaker",
    "Out Of Work Less Than One",
    "Out Of Work One Or More",
    "Retired",
    "Self Employed",
    "Student",
    "Unable To Work",
    "Unknown",
]
observed_emp = set(person_df["employment_status"].dropna().astype(str).unique().tolist())
employment_levels = [e for e in preferred_emp_order if e in observed_emp]
employment_levels += sorted([e for e in observed_emp if e not in employment_levels])

for emp in employment_levels:
    add_row(
        f"  {emp}",
        lambda d, lvl=emp: _fmt_categorical_cell((d["employment_status"] == lvl).sum(), d["person_id"].nunique())
    )

add_row("Total Nights Recorded (n)", lambda d: _fmt_continuous(d["n_nights"], decimals=0))
add_row("Avg Nightly Duration (mins)", lambda d: _fmt_continuous(d["avg_nightly_duration_mins"], decimals=0))

table_df = pd.DataFrame(table_rows)
table_df.to_csv(TABLE_OUTPUT, index=False)
print(f"Saved DST descriptive table to {TABLE_OUTPUT}")

# --- 8. Diagnostics ---
print("Saving diagnostics...")
group_counts = lf.group_by("dst_group").agg([
    pl.count("person_id").alias("n_observations"),
    pl.col("person_id").n_unique().alias("n_people")
]).collect()
group_counts.write_csv(OUTPUT_DIR / "diagnostics_group_counts.csv")

print(f"Analysis complete. All plots saved to {OUTPUT_DIR}")
