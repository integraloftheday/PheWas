#!/usr/bin/env python3
"""
03_2_Environemntal_Analysis.py

Environmental analysis of sleep outcomes using Polars + plotting + GAMs.

Outputs:
- data quality diagnostics for environmental sources
- merged sleep + environmental parquet
- raw scatter plots (overall and by employment)
- binned mean plots with 95% CI error bars (overall and by employment)
- GAM smooth plots (overall and by employment)
- GAM fit summary tables
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns

from pygam import LinearGAM, s

# -----------------------------
# Configuration
# -----------------------------
SLEEP_PATH = Path("processed_data/ready_for_analysis.parquet")
PHOTO_PATH = Path("environmental_data/photo_info/all_photo_info.parquet")
WEATHER_GLOB = "environmental_data/prism_weather_daily/zip3_weather_*.parquet"

OUTPUT_DIR = Path("results/environmental_analysis")
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
for d in [OUTPUT_DIR, PLOT_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OUTCOMES = {
    "daily_midpoint_hour": "Sleep Midpoint (hour)",
    "daily_start_hour": "Sleep Onset (hour)",
    "daily_end_hour": "Sleep Offset (hour)",
    "daily_duration_hours": "Sleep Duration (hours)",
}

ENV_CANDIDATES = [
    "PhotoPeriod",
    "deviation",
    "tmean",
    "tmin",
    "tmax",
    "ppt",
    "vpdmin",
    "vpdmax",
    "tdmean",
]

MIN_GROUP_ROWS = 80
SCATTER_SAMPLE_N = 120_000
BINS = 16
MAX_ROWS_PER_PAIR = 400_000

sns.set_style("whitegrid")


# -----------------------------
# Helpers
# -----------------------------
def normalize_zip3(expr: pl.Expr) -> pl.Expr:
    """Extract first 3 numeric digits from a ZIP-like string."""
    digits = expr.cast(pl.Utf8).str.replace_all(r"[^0-9]", "")
    return (
        pl.when(digits.str.len_chars() >= 3)
        .then(digits.str.slice(0, 3))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def qcut_labels(values: pd.Series, bins: int = BINS) -> pd.Series:
    """Robust quantile bins even with ties."""
    valid = values.dropna()
    if valid.empty:
        return pd.Series([np.nan] * len(values), index=values.index)

    unique_n = valid.nunique()
    q = int(max(2, min(bins, unique_n)))
    return pd.qcut(values, q=q, duplicates="drop")


def mean_se_table(df: pd.DataFrame, x_col: str, y_col: str, group_col: str | None = None) -> pd.DataFrame:
    work = df[[x_col, y_col] + ([group_col] if group_col else [])].dropna().copy()
    if work.empty:
        return pd.DataFrame()

    work["x_bin"] = qcut_labels(work[x_col], bins=BINS)
    gb_cols = ["x_bin"] + ([group_col] if group_col else [])
    out = (
        work.groupby(gb_cols, observed=True)
        .agg(
            x_mean=(x_col, "mean"),
            y_mean=(y_col, "mean"),
            y_sd=(y_col, "std"),
            n=(y_col, "size"),
        )
        .reset_index()
    )
    out["se"] = out["y_sd"] / np.sqrt(out["n"].clip(lower=1))
    out["ci95"] = 1.96 * out["se"].fillna(0.0)
    return out.sort_values([c for c in [group_col, "x_mean"] if c is not None])


def fit_gam_1d(x: np.ndarray, y: np.ndarray) -> LinearGAM | None:
    """Fit a 1D GAM with mild regularization; return None if insufficient data."""
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]
    if x.shape[0] < 25:
        return None

    X = x.reshape(-1, 1)
    gam = LinearGAM(s(0, n_splines=12))
    gam.gridsearch(X, y, progress=False)
    return gam


def gam_curve(gam: LinearGAM, x_min: float, x_max: float, n: int = 200) -> pd.DataFrame:
    xs = np.linspace(x_min, x_max, n)
    Xg = xs.reshape(-1, 1)
    yhat = gam.predict(Xg)
    ci = gam.confidence_intervals(Xg, width=0.95)
    return pd.DataFrame({"x": xs, "yhat": yhat, "lo": ci[:, 0], "hi": ci[:, 1]})


# -----------------------------
# Load / verify data
# -----------------------------
print("Loading sleep data...")
sleep_lf = pl.scan_parquet(SLEEP_PATH)
sleep_cols = set(sleep_lf.collect_schema().names())

zip_expr = None
if "zip3" in sleep_cols:
    zip_expr = normalize_zip3(pl.col("zip3"))
elif "zip_code" in sleep_cols:
    zip_expr = normalize_zip3(pl.col("zip_code"))
else:
    raise ValueError("Sleep parquet missing both 'zip3' and 'zip_code'.")

required_sleep_cols = [
    "sleep_date",
    "daily_midpoint_hour",
    "daily_start_hour",
    "daily_end_hour",
    "daily_duration_mins",
]
for c in required_sleep_cols:
    if c not in sleep_cols:
        raise ValueError(f"Sleep parquet missing required column: {c}")

if PHOTO_PATH.exists() is False:
    raise FileNotFoundError(f"Missing photo info parquet: {PHOTO_PATH}")

weather_files = sorted(Path().glob(WEATHER_GLOB))
if not weather_files:
    raise FileNotFoundError(f"No weather files found with glob: {WEATHER_GLOB}")

print("Loading environmental data...")
photo_lf = (
    pl.scan_parquet(PHOTO_PATH)
    .with_columns(
        normalize_zip3(pl.col("ZIP3")).alias("ZIP3"),
        pl.col("DayOfYear").cast(pl.Int32),
    )
    .select(["ZIP3", "DayOfYear", "PhotoPeriod", "deviation"])
)

weather_lf = (
    pl.scan_parquet([str(p) for p in weather_files])
    .with_columns(
        normalize_zip3(pl.col("ZIP3")).alias("ZIP3"),
        pl.col("Date").cast(pl.Date),
    )
    .select(["ZIP3", "Date", "tmean", "tmin", "tmax", "ppt", "vpdmin", "vpdmax", "tdmean"])
)

# Environmental data reasonableness checks
print("Running environmental diagnostics...")
photo_diag = photo_lf.select(
    [
        pl.len().alias("rows"),
        pl.col("ZIP3").n_unique().alias("n_zip3"),
        pl.col("DayOfYear").min().alias("day_min"),
        pl.col("DayOfYear").max().alias("day_max"),
        pl.col("PhotoPeriod").min().alias("photoperiod_min"),
        pl.col("PhotoPeriod").max().alias("photoperiod_max"),
        pl.col("deviation").min().alias("deviation_min"),
        pl.col("deviation").max().alias("deviation_max"),
    ]
).collect()

weather_diag = weather_lf.select(
    [
        pl.len().alias("rows"),
        pl.col("ZIP3").n_unique().alias("n_zip3"),
        pl.col("Date").min().alias("date_min"),
        pl.col("Date").max().alias("date_max"),
        pl.col("tmean").min().alias("tmean_min"),
        pl.col("tmean").max().alias("tmean_max"),
        pl.col("ppt").quantile(0.99).alias("ppt_p99"),
        pl.col("ppt").max().alias("ppt_max"),
        pl.col("vpdmax").quantile(0.99).alias("vpdmax_p99"),
        pl.col("vpdmax").max().alias("vpdmax_max"),
    ]
).collect()

weather_years = (
    weather_lf.select(pl.col("Date").dt.year().alias("year")).unique().sort("year").collect()["year"].to_list()
)
if weather_years:
    missing_years = sorted(set(range(min(weather_years), max(weather_years) + 1)) - set(weather_years))
else:
    missing_years = []

consistency = weather_lf.select(
    [
        (pl.col("tmin") <= pl.col("tmean")).mean().alias("frac_tmin_le_tmean"),
        (pl.col("tmean") <= pl.col("tmax")).mean().alias("frac_tmean_le_tmax"),
        (pl.col("ppt") >= 0).mean().alias("frac_ppt_nonnegative"),
        (pl.col("vpdmin") >= 0).mean().alias("frac_vpdmin_nonnegative"),
        (pl.col("vpdmax") >= 0).mean().alias("frac_vpdmax_nonnegative"),
    ]
).collect()

diagnostics = pd.concat(
    [
        photo_diag.to_pandas().assign(dataset="photo_info"),
        weather_diag.to_pandas().assign(dataset="prism_weather_daily"),
        consistency.to_pandas().assign(dataset="weather_consistency"),
    ],
    ignore_index=True,
)
diagnostics.to_csv(TABLE_DIR / "00_environmental_diagnostics.csv", index=False)

with (TABLE_DIR / "00_environmental_diagnostics_notes.txt").open("w", encoding="utf-8") as handle:
    handle.write("Environmental data diagnostics\n")
    handle.write(f"Weather years present: {weather_years}\n")
    handle.write(f"Missing weather years in min-max span: {missing_years}\n")

# -----------------------------
# Merge sleep with environmental data
# -----------------------------
print("Merging sleep + environmental data...")
sleep_base_lf = sleep_lf.select(
    [
        "person_id",
        "sleep_date",
        "employment_status",
        "daily_midpoint_hour",
        "daily_start_hour",
        "daily_end_hour",
        "daily_duration_mins",
        "zip3" if "zip3" in sleep_cols else "zip_code",
    ]
)

sleep_env_lf = (
    sleep_base_lf
    .with_columns(
        [
            zip_expr.alias("ZIP3"),
            pl.col("sleep_date").cast(pl.Date).alias("Date"),
            pl.col("sleep_date").cast(pl.Date).dt.ordinal_day().alias("DayOfYear"),
            (pl.col("daily_duration_mins") / 60.0).alias("daily_duration_hours"),
            pl.when(pl.col("employment_status").is_null() | (pl.col("employment_status") == ""))
            .then(pl.lit("Unknown"))
            .otherwise(pl.col("employment_status"))
            .alias("employment_status"),
        ]
    )
    .join(weather_lf, on=["ZIP3", "Date"], how="left")
    .join(photo_lf, on=["ZIP3", "DayOfYear"], how="left")
)

coverage = sleep_env_lf.select(
    [
        pl.len().alias("n_rows"),
        pl.col("tmean").is_not_null().mean().alias("frac_weather_matched"),
        pl.col("PhotoPeriod").is_not_null().mean().alias("frac_photo_matched"),
        pl.col("ZIP3").is_not_null().mean().alias("frac_zip3_valid"),
    ]
).collect(engine="streaming").to_pandas()
coverage.to_csv(TABLE_DIR / "01_sleep_environment_coverage.csv", index=False)

# If key matching is near-zero (e.g., synthetic ZIP strings), stop after diagnostics.
if float(coverage.loc[0, "frac_weather_matched"]) < 0.01 and float(coverage.loc[0, "frac_photo_matched"]) < 0.01:
    raise RuntimeError(
        "Environmental join coverage is near zero. Check ZIP code fields in sleep data "
        "(synthetic/non-numeric ZIPs will prevent matches). Diagnostics were written to results/environmental_analysis/tables/."
    )

# Write merged table in streaming mode (lower memory than collect->write).
MERGED_PATH = OUTPUT_DIR / "sleep_environment_merged.parquet"
sleep_env_lf.sink_parquet(MERGED_PATH)

# Re-scan merged table and process one variable pair at a time.
merged_lf = pl.scan_parquet(MERGED_PATH)
merged_cols = set(merged_lf.collect_schema().names())
env_vars = [c for c in ENV_CANDIDATES if c in merged_cols]
if not env_vars:
    raise RuntimeError("No environmental variables available after merge.")

emp_counts_df = (
    merged_lf
    .select("employment_status")
    .group_by("employment_status")
    .agg(pl.len().alias("n"))
    .collect(engine="streaming")
    .to_pandas()
)
emp_counts_df = emp_counts_df.sort_values("n", ascending=False)
employment_groups = emp_counts_df.loc[emp_counts_df["n"] >= MIN_GROUP_ROWS, "employment_status"].tolist()
if not employment_groups:
    employment_groups = emp_counts_df["employment_status"].head(6).tolist()


def collect_pair_df(outcome: str, env: str) -> tuple[pd.DataFrame, int]:
    pair_lf = merged_lf.select(["person_id", "employment_status", outcome, env]).drop_nulls()
    n_full = int(pair_lf.select(pl.len().alias("n")).collect(engine="streaming").item())

    if n_full == 0:
        return pd.DataFrame(columns=[outcome, env, "employment_status"]), 0

    if n_full > MAX_ROWS_PER_PAIR:
        mod = int(np.ceil(n_full / MAX_ROWS_PER_PAIR))
        pair_lf = pair_lf.filter((pl.col("person_id").cast(pl.Utf8).hash(seed=2026) % mod) == 0)

    pdf = pair_lf.select([outcome, env, "employment_status"]).collect(engine="streaming").to_pandas()
    return pdf, n_full

# -----------------------------
# Plotting + GAM
# -----------------------------
print("Generating plots and GAM models...")

summary_rows: List[Dict[str, object]] = []

for outcome, y_label in OUTCOMES.items():
    if outcome not in merged_cols:
        continue

    for env in env_vars:
        d, n_full = collect_pair_df(outcome, env)
        if d.empty:
            continue

        # -------- Raw scatter (overall) --------
        n_sample = min(SCATTER_SAMPLE_N, len(d))
        d_sample = d.sample(n=n_sample, random_state=2026) if len(d) > n_sample else d

        plt.figure(figsize=(9, 6))
        plt.scatter(d_sample[env], d_sample[outcome], s=8, alpha=0.12, color="#1f77b4", edgecolor="none")
        plt.xlabel(env)
        plt.ylabel(y_label)
        plt.title(f"Raw sleep vs environmental: {outcome} ~ {env} (overall)")
        plt.tight_layout()
        plt.savefig(PLOT_DIR / f"raw_overall__{outcome}__{env}.png", dpi=220)
        plt.close()

        # -------- Raw scatter by employment --------
        d_emp = d[d["employment_status"].isin(employment_groups)].copy()
        if not d_emp.empty:
            g = sns.FacetGrid(
                d_emp.sample(n=min(len(d_emp), 80_000), random_state=2026) if len(d_emp) > 80_000 else d_emp,
                col="employment_status",
                col_wrap=3,
                sharex=True,
                sharey=True,
                height=3.1,
            )
            g.map_dataframe(sns.scatterplot, x=env, y=outcome, s=7, alpha=0.15, linewidth=0)
            g.set_axis_labels(env, y_label)
            g.fig.subplots_adjust(top=0.90)
            g.fig.suptitle(f"Raw sleep vs environmental by employment: {outcome} ~ {env}")
            g.savefig(PLOT_DIR / f"raw_by_employment__{outcome}__{env}.png", dpi=200)
            plt.close(g.fig)

        # -------- Average + error bars (overall) --------
        avg = mean_se_table(d, x_col=env, y_col=outcome)
        if not avg.empty:
            plt.figure(figsize=(9, 6))
            plt.errorbar(avg["x_mean"], avg["y_mean"], yerr=avg["ci95"], fmt="o-", capsize=3, alpha=0.9)
            plt.xlabel(env)
            plt.ylabel(y_label)
            plt.title(f"Binned mean ±95% CI: {outcome} ~ {env} (overall)")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"avg_overall__{outcome}__{env}.png", dpi=220)
            plt.close()

        # -------- Average + error bars (by employment) --------
        avg_emp = mean_se_table(
            d[d["employment_status"].isin(employment_groups)],
            x_col=env,
            y_col=outcome,
            group_col="employment_status",
        )
        if not avg_emp.empty:
            plt.figure(figsize=(11, 7))
            for grp, gdf in avg_emp.groupby("employment_status"):
                plt.errorbar(gdf["x_mean"], gdf["y_mean"], yerr=gdf["ci95"], fmt="o-", capsize=2, alpha=0.8, label=grp)
            plt.xlabel(env)
            plt.ylabel(y_label)
            plt.title(f"Binned mean ±95% CI by employment: {outcome} ~ {env}")
            plt.legend(loc="best", fontsize=8)
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"avg_by_employment__{outcome}__{env}.png", dpi=220)
            plt.close()

        # -------- GAM overall --------
        x = d[env].to_numpy(dtype=float)
        y = d[outcome].to_numpy(dtype=float)
        gam = fit_gam_1d(x, y)
        if gam is not None:
            curve = gam_curve(gam, np.nanpercentile(x, 1), np.nanpercentile(x, 99))

            plt.figure(figsize=(9, 6))
            plt.scatter(d_sample[env], d_sample[outcome], s=8, alpha=0.08, color="gray", edgecolor="none")
            plt.plot(curve["x"], curve["yhat"], color="#d62728", linewidth=2.3, label="GAM smooth")
            plt.fill_between(curve["x"], curve["lo"], curve["hi"], color="#d62728", alpha=0.2, label="95% CI")
            plt.xlabel(env)
            plt.ylabel(y_label)
            plt.title(f"GAM: {outcome} ~ s({env}) (overall)")
            plt.legend()
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"gam_overall__{outcome}__{env}.png", dpi=220)
            plt.close()

            stats = gam.statistics_
            summary_rows.append(
                {
                    "scope": "overall",
                    "outcome": outcome,
                    "environment_var": env,
                    "n": int(n_full),
                    "edof": float(stats.get("edof", np.nan)),
                    "AIC": float(stats.get("AIC", np.nan)),
                    "GCV": float(stats.get("GCV", np.nan)),
                    "pseudo_r2_explained_deviance": float(stats.get("pseudo_r2", {}).get("explained_deviance", np.nan)),
                    "p_value_smooth": float(gam.statistics_.get("p_values", [np.nan])[0]),
                }
            )

        # -------- GAM by employment --------
        for grp in employment_groups:
            dg = d[d["employment_status"] == grp]
            if len(dg) < MIN_GROUP_ROWS:
                continue

            xg = dg[env].to_numpy(dtype=float)
            yg = dg[outcome].to_numpy(dtype=float)
            gam_g = fit_gam_1d(xg, yg)
            if gam_g is None:
                continue

            curve_g = gam_curve(gam_g, np.nanpercentile(xg, 1), np.nanpercentile(xg, 99))
            plt.figure(figsize=(8.5, 5.5))
            ds = dg.sample(n=min(len(dg), 25_000), random_state=2026) if len(dg) > 25_000 else dg
            plt.scatter(ds[env], ds[outcome], s=8, alpha=0.10, color="gray", edgecolor="none")
            plt.plot(curve_g["x"], curve_g["yhat"], color="#2ca02c", linewidth=2.2)
            plt.fill_between(curve_g["x"], curve_g["lo"], curve_g["hi"], color="#2ca02c", alpha=0.2)
            plt.xlabel(env)
            plt.ylabel(y_label)
            plt.title(f"GAM by employment ({grp}): {outcome} ~ s({env})")
            plt.tight_layout()
            plt.savefig(PLOT_DIR / f"gam_by_employment__{outcome}__{env}__{grp[:40].replace('/', '-').replace(' ', '_')}.png", dpi=220)
            plt.close()

            stats_g = gam_g.statistics_
            summary_rows.append(
                {
                    "scope": "employment_group",
                    "employment_status": grp,
                    "outcome": outcome,
                    "environment_var": env,
                    "n": int(len(dg)),
                    "edof": float(stats_g.get("edof", np.nan)),
                    "AIC": float(stats_g.get("AIC", np.nan)),
                    "GCV": float(stats_g.get("GCV", np.nan)),
                    "pseudo_r2_explained_deviance": float(stats_g.get("pseudo_r2", {}).get("explained_deviance", np.nan)),
                    "p_value_smooth": float(gam_g.statistics_.get("p_values", [np.nan])[0]),
                }
            )

# Save GAM summaries
if summary_rows:
    pd.DataFrame(summary_rows).to_csv(TABLE_DIR / "02_gam_summary.csv", index=False)

print("Done.")
print(f"Outputs written to: {OUTPUT_DIR}")
