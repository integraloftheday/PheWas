#!/usr/bin/env python3
"""
03_2_1_Environmental_Analysis_plot.py

Create overall environmental-response plots for selected variables only:
- PhotoPeriod (linear regression)
- deviation (linear regression)
- tmin (quadratic regression)
- tmax (quadratic regression)

Regression equation is displayed directly on each plot.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


# -----------------------------
# Configuration
# -----------------------------
SLEEP_PATH = Path("processed_data/ready_for_analysis.parquet")
PHOTO_PATH = Path("environmental_data/photo_info/all_photo_info.parquet")
WEATHER_GLOB = "environmental_data/prism_weather_daily/zip3_weather_*.parquet"

OUTPUT_DIR = Path("results/environmental_analysis_03_2_1")
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
for d in [OUTPUT_DIR, PLOT_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

OUTCOME_SPECS = [
    {"name": "midpoint", "analysis_col": "midpoint_linear_use", "label": "Sleep Midpoint (HH:MM)", "is_clock": True},
    {"name": "onset", "analysis_col": "onset_linear_use", "label": "Sleep Onset (HH:MM)", "is_clock": True},
    {"name": "offset", "analysis_col": "offset_linear_use", "label": "Sleep Offset (HH:MM)", "is_clock": True},
    {"name": "duration", "analysis_col": "daily_duration_hours", "label": "Sleep Duration (hours)", "is_clock": False},
]

ENV_MODEL_SPECS = {
    "PhotoPeriod": {"model": "linear", "label": "Photoperiod"},
    "deviation": {"model": "linear", "label": "Photoperiod Deviation"},
    "tmin": {"model": "quadratic", "label": "Minimum Temperature"},
    "tmax": {"model": "quadratic", "label": "Maximum Temperature"},
}

BINS = 16

sns.set_style("whitegrid")


# -----------------------------
# Helpers
# -----------------------------
def normalize_zip3(expr: pl.Expr) -> pl.Expr:
    digits = expr.cast(pl.Utf8).str.replace_all(r"[^0-9]", "")
    return (
        pl.when(digits.str.len_chars() >= 3)
        .then(digits.str.slice(0, 3))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def linearize_noon_to_noon(expr: pl.Expr) -> pl.Expr:
    return pl.when(expr < 12).then(expr + 24).otherwise(expr)


def wrap_noon_linear(arr: np.ndarray | pd.Series) -> np.ndarray:
    vals = np.asarray(arr, dtype=float)
    return np.mod(vals - 12.0, 24.0) + 12.0


def hour_24_formatter(x: float, _pos: int) -> str:
    if not np.isfinite(x):
        return ""
    x = x % 24.0
    total_minutes = int(round(x * 60.0)) % (24 * 60)
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h:02d}:{m:02d}"


def fit_linear(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, float]:
    coef = np.polyfit(x, y, deg=1)
    p = np.poly1d(coef)
    yhat = p(x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
    eq = f"y = {coef[0]:.4f}x {'+' if coef[1] >= 0 else '-'} {abs(coef[1]):.4f}"
    return coef, yhat, eq, r2


def fit_quadratic(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, str, float]:
    coef = np.polyfit(x, y, deg=2)
    p = np.poly1d(coef)
    yhat = p(x)
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = np.nan if ss_tot <= 0 else (1.0 - ss_res / ss_tot)
    eq = (
        f"y = {coef[0]:.6f}x² "
        f"{'+' if coef[1] >= 0 else '-'} {abs(coef[1]):.4f}x "
        f"{'+' if coef[2] >= 0 else '-'} {abs(coef[2]):.4f}"
    )
    return coef, yhat, eq, r2


def qcut_labels(values: pd.Series, bins: int = BINS) -> pd.Series:
    valid = values.dropna()
    if valid.empty:
        return pd.Series([np.nan] * len(values), index=values.index)
    unique_n = valid.nunique()
    q = int(max(2, min(bins, unique_n)))
    return pd.qcut(values, q=q, duplicates="drop")


def mean_se_table(df: pd.DataFrame, x_col: str, y_col: str) -> pd.DataFrame:
    work = df[[x_col, y_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame()

    work["x_bin"] = qcut_labels(work[x_col], bins=BINS)
    out = (
        work.groupby("x_bin", observed=True)
        .agg(
            x_mean=(x_col, "mean"),
            y_mean=(y_col, "mean"),
            y_sd=(y_col, "std"),
            n=(y_col, "size"),
        )
        .reset_index()
        .sort_values("x_mean")
    )
    out["se"] = out["y_sd"] / np.sqrt(out["n"].clip(lower=1))
    out["ci95"] = 1.96 * out["se"].fillna(0.0)
    return out


def build_midpoint_split_person_df(merged_lf: pl.LazyFrame, env_col: str) -> pd.DataFrame:
    """
    Person-level midpoint split table for one environmental variable.
    Returns columns: person_id, env_mean, MSW_linear, MSF_linear, MSFsc_linear.
    """
    needed = {"person_id", "is_work_day", "onset_linear_use", "daily_duration_hours", env_col}
    schema_cols = set(merged_lf.collect_schema().names())
    if not needed.issubset(schema_cols):
        return pd.DataFrame()

    daytype = (
        merged_lf
        .select(["person_id", "is_work_day", "onset_linear_use", "daily_duration_hours"])
        .drop_nulls()
        .group_by(["person_id", "is_work_day"])
        .agg([
            pl.col("onset_linear_use").mean().alias("mean_onset"),
            pl.col("daily_duration_hours").mean().alias("mean_duration"),
            pl.len().alias("n_nights"),
        ])
        .collect(engine="streaming")
        .to_pandas()
    )
    if daytype.empty:
        return pd.DataFrame()

    work = daytype[daytype["is_work_day"] == 1].rename(
        columns={"mean_onset": "SO_w", "mean_duration": "SD_w", "n_nights": "n_work"}
    )
    free = daytype[daytype["is_work_day"] == 0].rename(
        columns={"mean_onset": "SO_f", "mean_duration": "SD_f", "n_nights": "n_free"}
    )

    ms = work[["person_id", "SO_w", "SD_w", "n_work"]].merge(
        free[["person_id", "SO_f", "SD_f", "n_free"]],
        on="person_id",
        how="inner",
    )
    if ms.empty:
        return pd.DataFrame()

    ms["MSW_linear"] = wrap_noon_linear(ms["SO_w"].to_numpy(dtype=float) + (ms["SD_w"].to_numpy(dtype=float) / 2.0))
    ms["MSF_linear"] = wrap_noon_linear(ms["SO_f"].to_numpy(dtype=float) + (ms["SD_f"].to_numpy(dtype=float) / 2.0))
    ms["SD_week"] = ((5.0 * ms["SD_w"]) + (2.0 * ms["SD_f"])) / 7.0

    msfsc = np.where(
        ms["SD_f"].to_numpy(dtype=float) > ms["SD_w"].to_numpy(dtype=float),
        ms["SO_f"].to_numpy(dtype=float) + (ms["SD_week"].to_numpy(dtype=float) / 2.0),
        ms["MSF_linear"].to_numpy(dtype=float),
    )
    ms["MSFsc_linear"] = wrap_noon_linear(msfsc)

    env_person = (
        merged_lf
        .select(["person_id", env_col])
        .drop_nulls()
        .group_by("person_id")
        .agg(pl.col(env_col).mean().alias("env_mean"))
        .collect(engine="streaming")
        .to_pandas()
    )
    if env_person.empty:
        return pd.DataFrame()

    out = ms.merge(env_person, on="person_id", how="inner")
    return out


def main() -> None:
    print("Loading sleep data...")
    sleep_lf = pl.scan_parquet(SLEEP_PATH)
    sleep_cols = set(sleep_lf.collect_schema().names())

    if "zip3" in sleep_cols:
        zip_expr = normalize_zip3(pl.col("zip3"))
        zip_col = "zip3"
    elif "zip_code" in sleep_cols:
        zip_expr = normalize_zip3(pl.col("zip_code"))
        zip_col = "zip_code"
    else:
        raise ValueError("Sleep parquet missing both 'zip3' and 'zip_code'.")

    weather_files = sorted(Path().glob(WEATHER_GLOB))
    if not weather_files:
        raise FileNotFoundError(f"No weather files found with glob: {WEATHER_GLOB}")

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
        .select(["ZIP3", "Date", "tmin", "tmax"])
    )

    sleep_select = [
        "person_id",
        "sleep_date",
        "is_weekend",
        "is_weekend_or_holiday",
        "daily_midpoint_hour",
        "daily_start_hour",
        "daily_end_hour",
        "daily_duration_mins",
        zip_col,
    ]
    for c in ["onset_linear", "midpoint_linear", "offset_linear"]:
        if c in sleep_cols:
            sleep_select.append(c)

    sleep_base = sleep_lf.select(sleep_select)

    if "is_weekend_or_holiday" in sleep_cols:
        is_work_day_expr = (
            pl.when(pl.col("is_weekend_or_holiday").is_null())
            .then(pl.lit(None, dtype=pl.Int8))
            .otherwise((~pl.col("is_weekend_or_holiday").cast(pl.Boolean)).cast(pl.Int8))
        )
    elif "is_weekend" in sleep_cols:
        is_work_day_expr = (
            pl.when(pl.col("is_weekend").is_null())
            .then(pl.lit(None, dtype=pl.Int8))
            .otherwise((~pl.col("is_weekend").cast(pl.Boolean)).cast(pl.Int8))
        )
    else:
        is_work_day_expr = pl.lit(None, dtype=pl.Int8)

    onset_linear_expr = (
        pl.col("onset_linear").cast(pl.Float64)
        if "onset_linear" in sleep_select
        else linearize_noon_to_noon(pl.col("daily_start_hour").cast(pl.Float64))
    )
    midpoint_linear_expr = (
        pl.col("midpoint_linear").cast(pl.Float64)
        if "midpoint_linear" in sleep_select
        else linearize_noon_to_noon(pl.col("daily_midpoint_hour").cast(pl.Float64))
    )
    offset_linear_expr = (
        pl.col("offset_linear").cast(pl.Float64)
        if "offset_linear" in sleep_select
        else linearize_noon_to_noon(pl.col("daily_end_hour").cast(pl.Float64))
    )

    merged = (
        sleep_base
        .with_columns(
            [
                zip_expr.alias("ZIP3"),
                pl.col("sleep_date").cast(pl.Date).alias("Date"),
                pl.col("sleep_date").cast(pl.Date).dt.ordinal_day().alias("DayOfYear"),
                (pl.col("daily_duration_mins") / 60.0).alias("daily_duration_hours"),
                onset_linear_expr.alias("onset_linear_use"),
                midpoint_linear_expr.alias("midpoint_linear_use"),
                offset_linear_expr.alias("offset_linear_use"),
                is_work_day_expr.alias("is_work_day"),
            ]
        )
        .join(weather_lf, on=["ZIP3", "Date"], how="left")
        .join(photo_lf, on=["ZIP3", "DayOfYear"], how="left")
        .select([
            "person_id",
            "is_work_day",
            "midpoint_linear_use",
            "onset_linear_use",
            "offset_linear_use",
            "daily_duration_hours",
            "PhotoPeriod",
            "deviation",
            "tmin",
            "tmax",
        ])
    )

    print("Collecting merged subset...")
    df = merged.collect(engine="streaming").to_pandas()

    model_rows: list[Dict[str, object]] = []

    for out_spec in OUTCOME_SPECS:
        y_col = out_spec["analysis_col"]
        y_label = out_spec["label"]
        is_clock = bool(out_spec["is_clock"])

        if y_col not in df.columns:
            continue

        for env_col, env_spec in ENV_MODEL_SPECS.items():
            if env_col not in df.columns:
                continue

            tmp = df[[env_col, y_col]].dropna().copy()
            if tmp.empty or tmp.shape[0] < 50:
                continue

            n_full = tmp.shape[0]

            x = tmp[env_col].to_numpy(dtype=float)
            y = tmp[y_col].to_numpy(dtype=float)

            x_lo = float(np.nanquantile(x, 0.10))
            x_hi = float(np.nanquantile(x, 0.90))

            reg_mask = np.isfinite(x) & np.isfinite(y) & (x >= x_lo) & (x <= x_hi)
            x_reg = x[reg_mask]
            y_reg = y[reg_mask]
            if x_reg.shape[0] < 25:
                continue

            avg = mean_se_table(tmp, env_col, y_col)
            if avg.empty:
                continue

            if env_spec["model"] == "linear":
                coef, _, eq, r2 = fit_linear(x_reg, y_reg)
                poly = np.poly1d(coef)
                effect_minutes_per_unit = float(coef[0] * 60.0)
                effect_text = f"Effect: {effect_minutes_per_unit:+.2f} min per +1 unit {env_col}"
            else:
                coef, _, eq, r2 = fit_quadratic(x_reg, y_reg)
                poly = np.poly1d(coef)
                x_mid = float(np.nanmedian(x_reg))
                slope_mid = float((2.0 * coef[0] * x_mid) + coef[1])
                effect_minutes_per_unit = float(slope_mid * 60.0)
                effect_text = f"Local effect @ median x: {effect_minutes_per_unit:+.2f} min per +1 unit {env_col}"

            x_grid = np.linspace(x_lo, x_hi, 300)
            y_grid = poly(x_grid)
            y_lo = poly(x_lo)
            y_hi = poly(x_hi)
            delta_10_90_minutes = float((y_hi - y_lo) * 60.0)

            fig, ax = plt.subplots(figsize=(9.8, 6.2))
            # Background variability ribbon: binned mean ± 1 SD
            ax.fill_between(
                avg["x_mean"],
                avg["y_mean"] - avg["y_sd"].fillna(0.0),
                avg["y_mean"] + avg["y_sd"].fillna(0.0),
                color="#9e9e9e",
                alpha=0.18,
                label="Binned mean ±1 SD",
                zorder=1,
            )
            ax.errorbar(
                avg["x_mean"],
                avg["y_mean"],
                yerr=avg["ci95"],
                fmt="o",
                capsize=3,
                alpha=0.95,
                color="#1f77b4",
                label="Binned mean ±95% CI",
                zorder=3,
            )
            ax.plot(
                x_grid,
                y_grid,
                color="#d62728",
                linewidth=2.4,
                label=f"{env_spec['model'].capitalize()} fit (10%-90% interval)",
                zorder=4,
            )
            ax.axvline(x_lo, color="#555555", linestyle="--", linewidth=1.6, label="10th pct x")
            ax.axvline(x_hi, color="#555555", linestyle="--", linewidth=1.6, label="90th pct x")

            ax.set_title(f"Overall: {out_spec['name']} vs {env_col}")
            ax.set_xlabel(env_spec["label"])
            ax.set_ylabel(y_label)

            if is_clock:
                ticks = np.arange(12, 37, 2)
                ax.set_yticks(ticks)
                ax.yaxis.set_major_formatter(FuncFormatter(hour_24_formatter))

            text = (
                f"{eq}\n"
                f"$R^2$ = {r2:.4f}\n"
                f"{effect_text}\n"
                f"Net 10-90% change: {delta_10_90_minutes:+.2f} min\n"
                f"N(all) = {n_full:,}\n"
                f"N(10-90%) = {x_reg.shape[0]:,}"
            )
            ax.text(
                0.02,
                0.98,
                text,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=10,
                bbox=dict(facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
            )

            ax.legend(loc="best", frameon=False)
            ax.grid(alpha=0.25)
            fig.tight_layout()
            out_name = f"overall__{out_spec['name']}__{env_col}__{env_spec['model']}.png"
            fig.savefig(PLOT_DIR / out_name, dpi=260)
            plt.close(fig)

            model_rows.append(
                {
                    "outcome": out_spec["name"],
                    "outcome_col": y_col,
                    "environment_var": env_col,
                    "model_type": env_spec["model"],
                    "n": int(n_full),
                    "n_reg_10_90": int(x_reg.shape[0]),
                    "equation": eq,
                    "r2": float(r2),
                    "effect_minutes_per_unit": float(effect_minutes_per_unit),
                    "delta_10_90_minutes": float(delta_10_90_minutes),
                    "x_lo_10pct": float(x_lo),
                    "x_hi_90pct": float(x_hi),
                    "coef_0": float(coef[0]) if len(coef) > 0 else np.nan,
                    "coef_1": float(coef[1]) if len(coef) > 1 else np.nan,
                    "coef_2": float(coef[2]) if len(coef) > 2 else np.nan,
                }
            )

    if model_rows:
        pd.DataFrame(model_rows).to_csv(TABLE_DIR / "overall_regression_models_03_2_1.csv", index=False)

    # Midpoint split regressions: work-day midpoint (MSW), free-day midpoint (MSF), adjusted midpoint (MSFsc)
    midpoint_split_rows: list[Dict[str, object]] = []
    midpoint_series = [
        ("MSW_linear", "Work midpoint (MSW)"),
        ("MSF_linear", "Free midpoint (MSF)"),
        ("MSFsc_linear", "Adjusted midpoint (MSFsc)"),
    ]

    for env_col, env_spec in ENV_MODEL_SPECS.items():
        person_mid = build_midpoint_split_person_df(merged, env_col)
        if person_mid.empty:
            continue

        fig, axes = plt.subplots(1, 3, figsize=(19, 5.8), sharex=True)

        for ax, (y_col, panel_label) in zip(axes, midpoint_series):
            tmp = person_mid[["env_mean", y_col]].dropna().copy()
            if tmp.shape[0] < 50:
                ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            x = tmp["env_mean"].to_numpy(dtype=float)
            y = tmp[y_col].to_numpy(dtype=float)

            x_lo = float(np.nanquantile(x, 0.10))
            x_hi = float(np.nanquantile(x, 0.90))
            reg_mask = np.isfinite(x) & np.isfinite(y) & (x >= x_lo) & (x <= x_hi)
            x_reg = x[reg_mask]
            y_reg = y[reg_mask]
            if x_reg.shape[0] < 25:
                ax.text(0.5, 0.5, "Insufficient 10-90% data", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            avg = mean_se_table(tmp.rename(columns={"env_mean": env_col}), env_col, y_col)
            if avg.empty:
                ax.text(0.5, 0.5, "No bins available", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                continue

            if env_spec["model"] == "linear":
                coef, _, eq, r2 = fit_linear(x_reg, y_reg)
                poly = np.poly1d(coef)
                effect_minutes_per_unit = float(coef[0] * 60.0)
                effect_text = f"Effect: {effect_minutes_per_unit:+.2f} min per +1 unit"
            else:
                coef, _, eq, r2 = fit_quadratic(x_reg, y_reg)
                poly = np.poly1d(coef)
                x_mid = float(np.nanmedian(x_reg))
                slope_mid = float((2.0 * coef[0] * x_mid) + coef[1])
                effect_minutes_per_unit = float(slope_mid * 60.0)
                effect_text = f"Local effect @ median x: {effect_minutes_per_unit:+.2f} min per +1 unit"

            x_grid = np.linspace(x_lo, x_hi, 300)
            y_grid = poly(x_grid)
            delta_10_90_minutes = float((poly(x_hi) - poly(x_lo)) * 60.0)

            ax.fill_between(
                avg["x_mean"],
                avg["y_mean"] - avg["y_sd"].fillna(0.0),
                avg["y_mean"] + avg["y_sd"].fillna(0.0),
                color="#9e9e9e",
                alpha=0.18,
                label="Binned mean ±1 SD",
                zorder=1,
            )
            ax.errorbar(
                avg["x_mean"],
                avg["y_mean"],
                yerr=avg["ci95"],
                fmt="o",
                capsize=3,
                alpha=0.95,
                color="#1f77b4",
                label="Binned mean ±95% CI",
                zorder=3,
            )
            ax.plot(x_grid, y_grid, color="#d62728", linewidth=2.3, label=f"{env_spec['model'].capitalize()} fit (10%-90%)", zorder=4)
            ax.axvline(x_lo, color="#555555", linestyle="--", linewidth=1.4, label="10th pct x")
            ax.axvline(x_hi, color="#555555", linestyle="--", linewidth=1.4, label="90th pct x")

            ticks = np.arange(12, 37, 2)
            ax.set_yticks(ticks)
            ax.yaxis.set_major_formatter(FuncFormatter(hour_24_formatter))
            ax.set_title(panel_label)
            ax.set_xlabel(env_spec["label"])
            ax.set_ylabel("Midpoint (HH:MM)")
            ax.grid(alpha=0.25)

            ann = (
                f"{eq}\n"
                f"$R^2$={r2:.4f}\n"
                f"{effect_text}\n"
                f"Δ(10→90%)={delta_10_90_minutes:+.2f} min"
            )
            ax.text(
                0.02,
                0.98,
                ann,
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="#aaaaaa", alpha=0.9),
            )

            midpoint_split_rows.append(
                {
                    "environment_var": env_col,
                    "series": y_col,
                    "series_label": panel_label,
                    "model_type": env_spec["model"],
                    "n": int(tmp.shape[0]),
                    "n_reg_10_90": int(x_reg.shape[0]),
                    "equation": eq,
                    "r2": float(r2),
                    "effect_minutes_per_unit": float(effect_minutes_per_unit),
                    "delta_10_90_minutes": float(delta_10_90_minutes),
                    "x_lo_10pct": float(x_lo),
                    "x_hi_90pct": float(x_hi),
                    "coef_0": float(coef[0]) if len(coef) > 0 else np.nan,
                    "coef_1": float(coef[1]) if len(coef) > 1 else np.nan,
                    "coef_2": float(coef[2]) if len(coef) > 2 else np.nan,
                }
            )

        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.suptitle(f"Overall midpoint split vs {env_col}", y=1.02, fontsize=14)
        fig.tight_layout()
        fig.savefig(PLOT_DIR / f"overall_midpoint_split__{env_col}__{env_spec['model']}.png", dpi=260, bbox_inches="tight")
        plt.close(fig)

    if midpoint_split_rows:
        pd.DataFrame(midpoint_split_rows).to_csv(TABLE_DIR / "overall_midpoint_split_regression_models_03_2_1.csv", index=False)

    print("Done.")
    print(f"Plots: {PLOT_DIR}")
    print(f"Tables: {TABLE_DIR}")


if __name__ == "__main__":
    main()
