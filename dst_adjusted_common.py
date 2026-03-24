#!/usr/bin/env python3
"""
Shared utilities for DST-adjusted event-study scripts.

Provides a single `run_analysis()` entry point with two methods:
- `low_memory`: mean-based DiD without FE regression
- `twfe`: two-way fixed effects event-study with clustered SE
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Literal, Tuple
import os

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


WINDOW = 14
BASELINE_START = -14
BASELINE_END = -7
REFERENCE_K = -1
PLOT_WINDOW = 14
TEST_SAMPLE_N = 250_000

# Arizona (most regions), Hawaii, and territories used as non-DST controls.
NO_DST_ZIP3 = [f"{i:03d}" for i in range(850, 866)] + ["967", "968", "006", "007", "008", "009", "969"]

METRIC_SPECS = [
    {
        "name": "midpoint",
        "candidates": ["midpoint_linear", "daily_midpoint_hour"],
        "label": "Sleep midpoint",
        "units": "hours",
    },
    {
        "name": "onset",
        "candidates": ["onset_linear", "daily_start_hour"],
        "label": "Sleep onset",
        "units": "hours",
    },
    {
        "name": "offset",
        "candidates": ["offset_linear", "daily_end_hour"],
        "label": "Sleep offset",
        "units": "hours",
    },
    {
        "name": "duration",
        "candidates": ["daily_duration_mins", "daily_sleep_window_mins"],
        "label": "Sleep duration",
        "units": "minutes",
    },
]

DAYTYPE_SPECS = [
    ("all_days", None),
    ("work_days", True),
    ("free_days", False),
]


@dataclass
class EventStudyResult:
    coef_df: pd.DataFrame
    pretrend_df: pd.DataFrame


def ensure_output_dirs(output_dir: Path) -> tuple[Path, Path]:
    plot_dir = output_dir / "plots"
    table_dir = output_dir / "tables"
    for d in [output_dir, plot_dir, table_dir]:
        d.mkdir(parents=True, exist_ok=True)
    return plot_dir, table_dir


def get_dst_dates(year: int) -> Tuple[datetime.date, datetime.date]:
    """Return spring and fall DST transition dates for a given year."""
    # Spring: 2nd Sunday in March
    march_1 = datetime(year, 3, 1)
    first_sunday_march = march_1 + timedelta(days=(6 - march_1.weekday() + 7) % 7)
    spring = first_sunday_march + timedelta(days=7)

    # Fall: 1st Sunday in November
    nov_1 = datetime(year, 11, 1)
    fall = nov_1 + timedelta(days=(6 - nov_1.weekday() + 7) % 7)

    return spring.date(), fall.date()


def choose_metric_columns(schema_cols: set[str]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}
    for spec in METRIC_SPECS:
        chosen = None
        for c in spec["candidates"]:
            if c in schema_cols:
                chosen = c
                break
        if chosen is None:
            raise ValueError(
                f"Could not find metric column for '{spec['name']}'. Tried: {spec['candidates']}"
            )
        out[spec["name"]] = {
            "col": chosen,
            "label": spec["label"],
            "units": spec["units"],
        }
    return out


def normalize_zip3_expr(schema_cols: set[str]) -> pl.Expr:
    if "zip3" in schema_cols:
        source = pl.col("zip3")
    elif "zip_code" in schema_cols:
        source = pl.col("zip_code")
    else:
        raise ValueError("Input parquet must contain either 'zip3' or 'zip_code'.")

    digits = source.cast(pl.Utf8).str.replace_all(r"[^0-9]", "")
    return (
        pl.when(digits.str.len_chars() >= 3)
        .then(digits.str.slice(0, 3))
        .otherwise(pl.lit(None, dtype=pl.String))
    )


def to_noon_linear_expr(expr: pl.Expr) -> pl.Expr:
    """Map clock-hour [0,24) to noon-linearized scale [12,36)."""
    return pl.when(expr < 12).then(expr + 24).otherwise(expr)


def wrap_noon_linear_array(arr: np.ndarray) -> np.ndarray:
    """Wrap any hour values to noon-linearized scale [12,36)."""
    return ((arr - 12.0) % 24.0) + 12.0


def linear_to_clock_array(arr: np.ndarray) -> np.ndarray:
    """Map noon-linearized [12,36) hours back to 0-24 clock hours."""
    return np.mod(arr, 24.0)


def save_chronotype_midpoint_figures(
    base_lf: pl.LazyFrame,
    onset_col: str,
    duration_col_mins: str,
    plot_dir: Path,
    table_dir: Path,
) -> None:
    """
    Save low-memory chronotype midpoint summaries:
    - MSW: midpoint of sleep on workdays
    - MSF: midpoint of sleep on free days
    - MSFsc: sleep-corrected midpoint (MCTQ style)
    """
    print("Building chronotype midpoint summaries (MSW, MSF, MSFsc)...")

    person_daytype = (
        base_lf
        .select([
            "person_id",
            "is_work_day",
            to_noon_linear_expr(pl.col(onset_col)).alias("onset_linear_use"),
            (pl.col(duration_col_mins) / 60.0).alias("duration_hours"),
        ])
        .drop_nulls(["person_id", "is_work_day", "onset_linear_use", "duration_hours"])
        .group_by(["person_id", "is_work_day"])
        .agg([
            pl.col("onset_linear_use").mean().alias("mean_onset_linear"),
            pl.col("duration_hours").mean().alias("mean_duration_hours"),
            pl.len().alias("n_nights"),
        ])
        .collect(engine="streaming")
    )

    daytype_diag = person_daytype.to_pandas() if person_daytype.height > 0 else pd.DataFrame(columns=["person_id", "is_work_day", "mean_onset_linear", "mean_duration_hours", "n_nights"])
    if not daytype_diag.empty:
        summary = (
            daytype_diag.groupby("is_work_day", as_index=False)
            .agg(n_people=("person_id", "nunique"), n_person_daytype_rows=("person_id", "size"), n_nights=("n_nights", "sum"))
            .sort_values("is_work_day")
        )
    else:
        summary = pd.DataFrame(columns=["is_work_day", "n_people", "n_person_daytype_rows", "n_nights"])
    summary.to_csv(table_dir / "04_chronotype_daytype_diagnostics.csv", index=False)

    if person_daytype.height == 0:
        fig = plt.figure(figsize=(9, 4.8))
        plt.text(0.5, 0.5, "No data available to compute MSW/MSF/MSFsc.", ha="center", va="center")
        plt.axis("off")
        plt.title("Chronotype midpoint distributions (not available)")
        plt.tight_layout()
        plt.savefig(plot_dir / "chronotype_midpoints_distribution.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig2 = plt.figure(figsize=(9, 4.8))
        plt.text(0.5, 0.5, "Insufficient work/free-day data for chronotype relationship plots.", ha="center", va="center")
        plt.axis("off")
        plt.title("Chronotype midpoint relationships (not available)")
        plt.tight_layout()
        plt.savefig(plot_dir / "chronotype_midpoints_relationships.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)
        return

    pd_day = person_daytype.to_pandas()
    work = pd_day[pd_day["is_work_day"] == 1].copy().rename(
        columns={
            "mean_onset_linear": "SO_w",
            "mean_duration_hours": "SD_w",
            "n_nights": "n_work_nights",
        }
    )
    free = pd_day[pd_day["is_work_day"] == 0].copy().rename(
        columns={
            "mean_onset_linear": "SO_f",
            "mean_duration_hours": "SD_f",
            "n_nights": "n_free_nights",
        }
    )

    ms = work[["person_id", "SO_w", "SD_w", "n_work_nights"]].merge(
        free[["person_id", "SO_f", "SD_f", "n_free_nights"]],
        on="person_id",
        how="inner",
    )
    if ms.empty:
        fig = plt.figure(figsize=(9, 4.8))
        plt.text(0.5, 0.5, "No overlapping persons with both work-day and free-day sleep.", ha="center", va="center")
        plt.axis("off")
        plt.title("Chronotype midpoint distributions (not available)")
        plt.tight_layout()
        plt.savefig(plot_dir / "chronotype_midpoints_distribution.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

        fig2 = plt.figure(figsize=(9, 4.8))
        plt.text(0.5, 0.5, "Cannot compute MSFsc without both work and free-day means.", ha="center", va="center")
        plt.axis("off")
        plt.title("Chronotype midpoint relationships (not available)")
        plt.tight_layout()
        plt.savefig(plot_dir / "chronotype_midpoints_relationships.png", dpi=300, bbox_inches="tight")
        plt.close(fig2)
        return

    ms["MSW_linear"] = wrap_noon_linear_array(ms["SO_w"].to_numpy(dtype=float) + (ms["SD_w"].to_numpy(dtype=float) / 2.0))
    ms["MSF_linear"] = wrap_noon_linear_array(ms["SO_f"].to_numpy(dtype=float) + (ms["SD_f"].to_numpy(dtype=float) / 2.0))
    ms["SD_week"] = ((ms["SD_w"] * 5.0) + (ms["SD_f"] * 2.0)) / 7.0

    msf_sc = np.where(
        ms["SD_f"].to_numpy(dtype=float) > ms["SD_w"].to_numpy(dtype=float),
        ms["SO_f"].to_numpy(dtype=float) + (ms["SD_week"].to_numpy(dtype=float) / 2.0),
        ms["MSF_linear"].to_numpy(dtype=float),
    )
    ms["MSFsc_linear"] = wrap_noon_linear_array(msf_sc)

    ms["MSW_clock"] = linear_to_clock_array(ms["MSW_linear"].to_numpy(dtype=float))
    ms["MSF_clock"] = linear_to_clock_array(ms["MSF_linear"].to_numpy(dtype=float))
    ms["MSFsc_clock"] = linear_to_clock_array(ms["MSFsc_linear"].to_numpy(dtype=float))

    ms.to_csv(table_dir / "04_chronotype_midpoints_person_level.csv", index=False)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    bins = np.arange(0.0, 24.25, 0.25)
    ax.hist(ms["MSW_clock"], bins=bins, alpha=0.45, label="MSW (workday midpoint)", density=True)
    ax.hist(ms["MSF_clock"], bins=bins, alpha=0.45, label="MSF (free-day midpoint)", density=True)
    ax.hist(ms["MSFsc_clock"], bins=bins, alpha=0.45, label="MSFsc (sleep-corrected midpoint)", density=True)
    ax.set_xlabel("Clock hour")
    ax.set_ylabel("Density")
    ax.set_title(
        "Chronotype midpoint distributions (person-level): MSW vs MSF vs MSFsc\n"
        "MSW = SO_w + SD_w/2; MSF = SO_f + SD_f/2; MSFsc adjusted for free-day oversleep"
    )
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(plot_dir / "chronotype_midpoints_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig2, axes = plt.subplots(1, 2, figsize=(13.5, 5.8), sharex=False, sharey=False)

    axes[0].scatter(ms["MSW_clock"], ms["MSF_clock"], s=8, alpha=0.2, color="#1f77b4", edgecolors="none")
    lo0 = min(ms["MSW_clock"].min(), ms["MSF_clock"].min())
    hi0 = max(ms["MSW_clock"].max(), ms["MSF_clock"].max())
    axes[0].plot([lo0, hi0], [lo0, hi0], linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("MSW (clock hour)")
    axes[0].set_ylabel("MSF (clock hour)")
    axes[0].set_title("Workday midpoint vs free-day midpoint")
    axes[0].grid(alpha=0.25)

    axes[1].scatter(ms["MSF_clock"], ms["MSFsc_clock"], s=8, alpha=0.2, color="#d62728", edgecolors="none")
    lo1 = min(ms["MSF_clock"].min(), ms["MSFsc_clock"].min())
    hi1 = max(ms["MSF_clock"].max(), ms["MSFsc_clock"].max())
    axes[1].plot([lo1, hi1], [lo1, hi1], linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("MSF (clock hour)")
    axes[1].set_ylabel("MSFsc (clock hour)")
    axes[1].set_title("Free-day midpoint vs sleep-corrected midpoint")
    axes[1].grid(alpha=0.25)

    fig2.suptitle(
        "Chronotype midpoint relationships (person-level)\n"
        "MSFsc correction applied when SD_f > SD_w",
        y=1.02,
    )
    plt.tight_layout()
    plt.savefig(plot_dir / "chronotype_midpoints_relationships.png", dpi=300, bbox_inches="tight")
    plt.close(fig2)


def _normal_approx_two_sided_p_from_z(z: np.ndarray) -> np.ndarray:
    from math import erf, sqrt

    return np.array([
        2.0 * (1.0 - 0.5 * (1.0 + erf(abs(val) / sqrt(2.0)))) if np.isfinite(val) else np.nan
        for val in z
    ])


def run_low_memory_mean_did(
    metric_lf: pl.LazyFrame,
    transition_name: str,
    metric_name: str,
    metric_label: str,
    units: str,
    daytype_name: str,
    daytype_filter: bool | None,
    metric_col: str,
) -> EventStudyResult:
    lf = metric_lf
    if daytype_filter is True:
        lf = lf.filter(pl.col("is_work_day") == 1)
    elif daytype_filter is False:
        lf = lf.filter(pl.col("is_work_day") == 0)

    n_rows = int(lf.select(pl.len().alias("n")).collect(engine="streaming").item())
    if n_rows == 0:
        return EventStudyResult(pd.DataFrame(), pd.DataFrame())

    people_df = lf.select(pl.col("person_id").n_unique().alias("n_people")).collect(engine="streaming")
    n_people = int(people_df["n_people"][0]) if people_df.height > 0 else 0
    dates_df = lf.select(pl.col("sleep_date").n_unique().alias("n_dates")).collect(engine="streaming")
    n_dates = int(dates_df["n_dates"][0]) if dates_df.height > 0 else 0

    agg = (
        lf.group_by(["dst_state", "event_day"])
        .agg([
            pl.col(metric_col).mean().alias("mean_metric"),
            pl.col(metric_col).std().alias("sd_metric"),
            pl.len().alias("n"),
        ])
        .with_columns((pl.col("sd_metric") / pl.col("n").cast(pl.Float64).sqrt()).alias("se_metric"))
        .collect(engine="streaming")
    )
    if agg.height == 0:
        return EventStudyResult(pd.DataFrame(), pd.DataFrame())

    ap = agg.to_pandas()

    base = (
        ap[(ap["event_day"] >= BASELINE_START) & (ap["event_day"] <= BASELINE_END)]
        .groupby("dst_state", as_index=False)["mean_metric"]
        .mean()
        .rename(columns={"mean_metric": "baseline_group_mean"})
    )
    ap = ap.merge(base, on="dst_state", how="left")
    ap["centered_mean"] = ap["mean_metric"] - ap["baseline_group_mean"]

    dst = ap[ap["dst_state"] == 1][["event_day", "centered_mean", "se_metric", "n"]].rename(
        columns={
            "centered_mean": "centered_dst",
            "se_metric": "se_dst",
            "n": "n_dst",
        }
    )
    nodst = ap[ap["dst_state"] == 0][["event_day", "centered_mean", "se_metric", "n"]].rename(
        columns={
            "centered_mean": "centered_nodst",
            "se_metric": "se_nodst",
            "n": "n_nodst",
        }
    )

    wide = dst.merge(nodst, on="event_day", how="inner")
    if wide.empty:
        return EventStudyResult(pd.DataFrame(), pd.DataFrame())

    wide = wide[(wide["event_day"] >= -WINDOW) & (wide["event_day"] <= WINDOW)].copy()
    wide = wide.sort_values("event_day")
    wide = wide[wide["event_day"] != REFERENCE_K].copy()
    if wide.empty:
        return EventStudyResult(pd.DataFrame(), pd.DataFrame())

    wide["beta"] = wide["centered_dst"] - wide["centered_nodst"]
    wide["se"] = np.sqrt(np.square(wide["se_dst"].astype(float)) + np.square(wide["se_nodst"].astype(float)))
    wide["ci95_low"] = wide["beta"] - 1.96 * wide["se"]
    wide["ci95_high"] = wide["beta"] + 1.96 * wide["se"]
    z = np.divide(wide["beta"].to_numpy(dtype=float), wide["se"].to_numpy(dtype=float), out=np.full(wide.shape[0], np.nan), where=wide["se"].to_numpy(dtype=float) > 0)

    try:
        from scipy.stats import norm  # type: ignore

        pvals = 2.0 * norm.sf(np.abs(z))
    except Exception:
        pvals = _normal_approx_two_sided_p_from_z(z)

    coef_df = pd.DataFrame(
        {
            "transition": transition_name,
            "metric": metric_name,
            "metric_label": metric_label,
            "units": units,
            "day_type": daytype_name,
            "event_day": wide["event_day"].to_numpy(dtype=int),
            "beta": wide["beta"].to_numpy(dtype=float),
            "se": wide["se"].to_numpy(dtype=float),
            "ci95_low": wide["ci95_low"].to_numpy(dtype=float),
            "ci95_high": wide["ci95_high"].to_numpy(dtype=float),
            "p_value": pvals,
            "n_rows": int(n_rows),
            "n_people": int(n_people),
            "n_dates": int(n_dates),
            "n_clusters_person": np.nan,
            "model_type": "low_memory_mean_did",
            "n_dst_day_mean": wide["n_dst"].to_numpy(dtype=float),
            "n_nodst_day_mean": wide["n_nodst"].to_numpy(dtype=float),
        }
    ).sort_values("event_day")

    pre = coef_df[coef_df["event_day"] < 0]
    if len(pre) >= 2 and np.isfinite(pre["beta"]).all():
        pre_mean = float(pre["beta"].mean())
        pre_sd = float(pre["beta"].std(ddof=1))
        pre_se = pre_sd / np.sqrt(len(pre)) if len(pre) > 1 else np.nan
        pre_z = pre_mean / pre_se if (np.isfinite(pre_se) and pre_se > 0) else np.nan
        try:
            from scipy.stats import norm  # type: ignore

            pre_p = float(2.0 * norm.sf(abs(pre_z))) if np.isfinite(pre_z) else np.nan
        except Exception:
            pre_p = float(_normal_approx_two_sided_p_from_z(np.array([pre_z]))[0]) if np.isfinite(pre_z) else np.nan
        pre_note = "Pretrend check uses mean(pre-event betas) ≈ 0 test (normal approximation)."
    else:
        pre_mean = np.nan
        pre_se = np.nan
        pre_z = np.nan
        pre_p = np.nan
        pre_note = "Insufficient pre-event points for pretrend check."

    pretrend_df = pd.DataFrame(
        [
            {
                "transition": transition_name,
                "metric": metric_name,
                "metric_label": metric_label,
                "units": units,
                "day_type": daytype_name,
                "pretrend_wald_stat": np.nan,
                "pretrend_df": len(pre),
                "pretrend_p_value": pre_p,
                "pretrend_z": pre_z,
                "pretrend_beta_mean": pre_mean,
                "pretrend_beta_se": pre_se,
                "pretrend_note": pre_note,
                "n_rows": int(n_rows),
                "n_people": int(n_people),
                "model_type": "low_memory_mean_did",
            }
        ]
    )

    return EventStudyResult(coef_df=coef_df, pretrend_df=pretrend_df)


def demean_two_way(v: np.ndarray, person_codes: np.ndarray, date_codes: np.ndarray, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Two-way demeaning by alternating projections (person FE + date FE)."""
    r = v.astype(np.float64, copy=True)

    n_person = int(person_codes.max()) + 1
    n_date = int(date_codes.max()) + 1

    for _ in range(max_iter):
        prev = r.copy()

        p_sum = np.bincount(person_codes, weights=r, minlength=n_person)
        p_cnt = np.bincount(person_codes, minlength=n_person)
        p_mean = np.divide(p_sum, np.maximum(p_cnt, 1), out=np.zeros_like(p_sum), where=p_cnt > 0)
        r = r - p_mean[person_codes]

        d_sum = np.bincount(date_codes, weights=r, minlength=n_date)
        d_cnt = np.bincount(date_codes, minlength=n_date)
        d_mean = np.divide(d_sum, np.maximum(d_cnt, 1), out=np.zeros_like(d_sum), where=d_cnt > 0)
        r = r - d_mean[date_codes]

        r = r - np.mean(r)

        delta = np.max(np.abs(r - prev))
        if delta < tol:
            break

    return r


def fit_twfe_clustered(
    y: np.ndarray,
    X: np.ndarray,
    person_codes: np.ndarray,
    date_codes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit TWFE via Frisch-Waugh residualization, clustered by person."""
    y_t = demean_two_way(y, person_codes, date_codes)
    X_t_cols = [demean_two_way(X[:, j], person_codes, date_codes) for j in range(X.shape[1])]
    X_t = np.column_stack(X_t_cols)

    XtX = X_t.T @ X_t
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X_t.T @ y_t)

    resid = y_t - X_t @ beta

    p_unique = np.unique(person_codes)
    k = X_t.shape[1]
    meat = np.zeros((k, k), dtype=float)

    for g in p_unique:
        idx = person_codes == g
        Xg = X_t[idx, :]
        ug = resid[idx]
        s = Xg.T @ ug
        meat += np.outer(s, s)

    vcov = XtX_inv @ meat @ XtX_inv

    n = X_t.shape[0]
    G = len(p_unique)
    if G > 1 and n > k:
        corr = (G / (G - 1)) * ((n - 1) / (n - k))
        vcov *= corr

    se = np.sqrt(np.clip(np.diag(vcov), a_min=0.0, a_max=None))
    return beta, se, vcov


def normal_approx_p(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    return _normal_approx_two_sided_p_from_z(z)


def run_event_study(
    df: pd.DataFrame,
    transition_name: str,
    metric_name: str,
    metric_col: str,
    metric_label: str,
    units: str,
    daytype_name: str,
    daytype_filter: bool | None,
) -> EventStudyResult:
    work = df.copy()

    if daytype_filter is True:
        work = work[work["is_work_day"] == 1].copy()
    elif daytype_filter is False:
        work = work[work["is_work_day"] == 0].copy()

    if work.empty:
        return EventStudyResult(
            coef_df=pd.DataFrame(),
            pretrend_df=pd.DataFrame(),
        )

    day_col = "event_day"
    work = work[(work[day_col] >= -WINDOW) & (work[day_col] <= WINDOW)].copy()

    baseline = (
        work[(work[day_col] >= BASELINE_START) & (work[day_col] <= BASELINE_END)]
        .groupby(["person_id", "year"], as_index=False)[metric_col]
        .mean()
        .rename(columns={metric_col: "baseline_value"})
    )

    work = work.merge(baseline, on=["person_id", "year"], how="left")
    work = work.dropna(subset=[metric_col, "baseline_value", "dst_state", "sleep_date", "person_id", "event_day"])

    if work.empty:
        return EventStudyResult(coef_df=pd.DataFrame(), pretrend_df=pd.DataFrame())

    work["y"] = work[metric_col] - work["baseline_value"]

    event_ks = [k for k in range(-WINDOW, WINDOW + 1) if k != REFERENCE_K]
    x_cols = []
    for k in event_ks:
        cname = f"dst_x_event_{k:+d}"
        work[cname] = ((work["event_day"] == k).astype(float) * work["dst_state"].astype(float))
        x_cols.append(cname)

    person_codes = pd.factorize(work["person_id"], sort=True)[0].astype(int)
    date_codes = pd.factorize(work["sleep_date"], sort=True)[0].astype(int)

    y = work["y"].to_numpy(dtype=float)
    X = work[x_cols].to_numpy(dtype=float)

    nonzero_mask = np.asarray((X != 0).any(axis=0)).ravel()
    X = X[:, nonzero_mask]
    kept_ks = [k for k, keep in zip(event_ks, nonzero_mask) if keep]

    if X.shape[1] == 0 or X.shape[0] < 100:
        return EventStudyResult(coef_df=pd.DataFrame(), pretrend_df=pd.DataFrame())

    beta, se, vcov = fit_twfe_clustered(y=y, X=X, person_codes=person_codes, date_codes=date_codes)
    pvals = normal_approx_p(beta, se)

    coef_df = pd.DataFrame(
        {
            "transition": transition_name,
            "metric": metric_name,
            "metric_label": metric_label,
            "units": units,
            "day_type": daytype_name,
            "event_day": kept_ks,
            "beta": beta,
            "se": se,
            "ci95_low": beta - 1.96 * se,
            "ci95_high": beta + 1.96 * se,
            "p_value": pvals,
            "n_rows": int(work.shape[0]),
            "n_people": int(work["person_id"].nunique()),
            "n_dates": int(work["sleep_date"].nunique()),
            "n_clusters_person": int(np.unique(person_codes).shape[0]),
            "model_type": "twfe",
        }
    ).sort_values("event_day")

    pre_mask = np.array([k < 0 for k in kept_ks], dtype=bool)
    if pre_mask.sum() > 0:
        b_pre = beta[pre_mask]
        V_pre = vcov[np.ix_(pre_mask, pre_mask)]
        V_pre_inv = np.linalg.pinv(V_pre)
        wald = float(b_pre.T @ V_pre_inv @ b_pre)
        df_wald = int(pre_mask.sum())
        try:
            from scipy.stats import chi2  # type: ignore

            pretrend_p = float(chi2.sf(wald, df_wald))
        except Exception:
            pretrend_p = np.nan
    else:
        wald = np.nan
        df_wald = 0
        pretrend_p = np.nan

    pretrend_df = pd.DataFrame(
        [
            {
                "transition": transition_name,
                "metric": metric_name,
                "metric_label": metric_label,
                "units": units,
                "day_type": daytype_name,
                "pretrend_wald_stat": wald,
                "pretrend_df": df_wald,
                "pretrend_p_value": pretrend_p,
                "pretrend_note": "Joint test over k<0 event-day coefficients.",
                "n_rows": int(work.shape[0]),
                "n_people": int(work["person_id"].nunique()),
                "model_type": "twfe",
            }
        ]
    )

    return EventStudyResult(coef_df=coef_df, pretrend_df=pretrend_df)


def maybe_downsample_by_person_hash(lf: pl.LazyFrame, target_rows: int) -> pl.LazyFrame:
    """
    Reduce rows by selecting a deterministic subset of persons using hash modulo.
    Keeps full within-person series for selected persons (better for panel structure).
    """
    if target_rows <= 0:
        return lf

    n_rows = int(lf.select(pl.len().alias("n")).collect(engine="streaming").item())
    if n_rows <= target_rows:
        return lf

    mod = int(np.ceil(n_rows / target_rows))
    print(f"Applying person-hash downsampling: rows={n_rows:,} -> target≈{target_rows:,} (mod={mod})")
    return lf.filter((pl.col("person_id").cast(pl.Utf8).hash(seed=2026) % mod) == 0)


def plot_metric_panels(
    coef_all: pd.DataFrame,
    transition: str,
    metric_name: str,
    metric_label: str,
    units: str,
    plot_dir: Path,
) -> None:
    subset = coef_all[(coef_all["transition"] == transition) & (coef_all["metric"] == metric_name)].copy()
    if subset.empty:
        return

    order = ["all_days", "work_days", "free_days"]
    pretty = {
        "all_days": "All days",
        "work_days": "Work days",
        "free_days": "Free days",
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for i, day_type in enumerate(order):
        ax = axes[i]
        sdf = subset[subset["day_type"] == day_type].sort_values("event_day")
        if sdf.empty:
            ax.text(0.5, 0.5, "No estimable coefficients", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{pretty[day_type]}\n(no data)")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axhline(0, color="gray", linestyle=":", linewidth=1)
            continue

        sdf = sdf[(sdf["event_day"] >= -PLOT_WINDOW) & (sdf["event_day"] <= PLOT_WINDOW)]

        ax.plot(sdf["event_day"], sdf["beta"], marker="o", linewidth=2, color="#1f77b4")
        ax.fill_between(sdf["event_day"], sdf["ci95_low"], sdf["ci95_high"], color="#1f77b4", alpha=0.2)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2, label="DST transition day")
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)

        n_people = int(sdf["n_people"].iloc[0]) if not sdf.empty else 0
        n_rows = int(sdf["n_rows"].iloc[0]) if not sdf.empty else 0

        ax.set_title(
            f"{pretty[day_type]}\nN people={n_people:,}, person-nights={n_rows:,}",
            fontsize=10,
        )
        ax.set_xlabel("Event day relative to DST transition")
        ax.grid(alpha=0.25)

    axes[0].set_ylabel(f"DiD effect on {metric_label} ({units})")
    model_desc = "Low-memory mean-based DiD" if (subset["model_type"] == "low_memory_mean_did").all() else "TWFE event-study DiD"
    fig.suptitle(
        f"Event-study DiD: {metric_label} around {transition.title()} DST transition\n"
        f"Model: {model_desc} (reference day = -1)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out_path = plot_dir / f"eventstudy_{transition}_{metric_name}_panels.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transition_overview(coef_all: pd.DataFrame, transition: str, metric_meta: Dict[str, Dict[str, str]], plot_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
    axes = axes.flatten()

    metric_order = ["midpoint", "onset", "offset", "duration"]

    for i, metric in enumerate(metric_order):
        ax = axes[i]
        sdf = coef_all[
            (coef_all["transition"] == transition)
            & (coef_all["metric"] == metric)
            & (coef_all["day_type"] == "all_days")
        ].sort_values("event_day")

        if sdf.empty:
            ax.text(0.5, 0.5, "No estimable coefficients", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{metric_meta[metric]['label']} (no data)")
            ax.axvline(0, color="black", linestyle="--", linewidth=1)
            ax.axhline(0, color="gray", linestyle=":", linewidth=1)
            continue

        sdf = sdf[(sdf["event_day"] >= -PLOT_WINDOW) & (sdf["event_day"] <= PLOT_WINDOW)]
        ax.plot(sdf["event_day"], sdf["beta"], marker="o", linewidth=2, color="#d62728")
        ax.fill_between(sdf["event_day"], sdf["ci95_low"], sdf["ci95_high"], color="#d62728", alpha=0.2)
        ax.axvline(0, color="black", linestyle="--", linewidth=1.2)
        ax.axhline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_title(f"{metric_meta[metric]['label']} ({metric_meta[metric]['units']})")
        ax.set_xlabel("Event day")
        ax.set_ylabel("DiD effect")
        ax.grid(alpha=0.25)

    tsub = coef_all[coef_all["transition"] == transition].copy()
    model_desc = "Low-memory mean-based DiD" if (tsub["model_type"] == "low_memory_mean_did").all() else "TWFE event-study DiD"
    fig.suptitle(
        f"DST event-study DiD overview ({transition.title()} transition, all days)\n"
        f"{model_desc}; baseline-centered on days {BASELINE_START} to {BASELINE_END}",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    out_path = plot_dir / f"eventstudy_{transition}_overview_2x2.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_base_lazyframe(input_parquet: Path) -> tuple[pl.LazyFrame, Dict[str, Dict[str, str]]]:
    lf = pl.scan_parquet(input_parquet)
    schema_cols = set(lf.collect_schema().names())

    metric_meta = choose_metric_columns(schema_cols)
    zip3_expr = normalize_zip3_expr(schema_cols)

    if "is_weekend_or_holiday" in schema_cols:
        weekend_expr = pl.col("is_weekend_or_holiday").cast(pl.Boolean)
    elif "is_weekend" in schema_cols:
        weekend_expr = pl.col("is_weekend").cast(pl.Boolean)
    else:
        weekend_expr = (pl.col("sleep_date").cast(pl.Date).dt.weekday() >= 5)

    years = range(2009, 2025)
    rows = []
    for y in years:
        spring, fall = get_dst_dates(y)
        rows.append({"year": y, "spring_dst": spring, "fall_dst": fall})
    transitions_df = pl.DataFrame(rows)

    needed_cols = ["person_id", "sleep_date"]
    for m in metric_meta.values():
        needed_cols.append(m["col"])

    if "zip3" in schema_cols:
        needed_cols.append("zip3")
    if "zip_code" in schema_cols:
        needed_cols.append("zip_code")
    if "is_weekend_or_holiday" in schema_cols:
        needed_cols.append("is_weekend_or_holiday")
    if "is_weekend" in schema_cols:
        needed_cols.append("is_weekend")

    needed_cols = list(dict.fromkeys(needed_cols))

    base = (
        lf.select([c for c in needed_cols if c in schema_cols])
        .with_columns(
            pl.col("sleep_date").cast(pl.Date),
            zip3_expr.alias("zip3"),
            weekend_expr.alias("is_weekend_or_holiday"),
            pl.col("sleep_date").dt.year().alias("year"),
        )
        .join(transitions_df.lazy(), on="year", how="left")
        .with_columns(
            pl.when(pl.col("zip3").is_in(NO_DST_ZIP3)).then(pl.lit(0)).otherwise(pl.lit(1)).alias("dst_state"),
            (pl.col("sleep_date") - pl.col("spring_dst")).dt.total_days().alias("days_to_spring"),
            (pl.col("sleep_date") - pl.col("fall_dst")).dt.total_days().alias("days_to_fall"),
            pl.when(pl.col("is_weekend_or_holiday") == True).then(pl.lit(0)).otherwise(pl.lit(1)).alias("is_work_day"),
        )
    )

    for key, meta in metric_meta.items():
        col = meta["col"]
        if key == "duration" and col == "daily_sleep_window_mins":
            base = base.with_columns(pl.col(col).cast(pl.Float64).alias("daily_duration_mins"))
            metric_meta[key]["col"] = "daily_duration_mins"
        elif key == "duration" and col == "daily_duration_mins":
            base = base.with_columns(pl.col("daily_duration_mins").cast(pl.Float64))
        else:
            base = base.with_columns(pl.col(col).cast(pl.Float64))

    base = base.filter(
        pl.col("person_id").is_not_null() &
        pl.col("sleep_date").is_not_null() &
        pl.col("dst_state").is_not_null()
    )

    return base, metric_meta


def run_analysis(
    method: Literal["low_memory", "twfe"],
    input_parquet: Path,
    output_dir: Path,
    test_mode: bool | None = None,
    max_model_rows: int | None = None,
) -> None:
    if method not in {"low_memory", "twfe"}:
        raise ValueError("method must be one of: low_memory, twfe")

    plot_dir, table_dir = ensure_output_dirs(output_dir)

    if test_mode is None:
        test_mode = str(os.getenv("TEST_MODE", "0")).strip() == "1"

    if max_model_rows is None:
        if method == "twfe":
            max_model_rows = int(str(os.getenv("DST_MAX_MODEL_ROWS", "2000000")).strip() or "0")
        else:
            max_model_rows = int(str(os.getenv("DST_MAX_MODEL_ROWS", "0")).strip() or "0")

    print(f"Loading data from {input_parquet}...")
    base, metric_meta = build_base_lazyframe(input_parquet)

    try:
        save_chronotype_midpoint_figures(
            base_lf=base,
            onset_col=metric_meta["onset"]["col"],
            duration_col_mins=metric_meta["duration"]["col"],
            plot_dir=plot_dir,
            table_dir=table_dir,
        )
    except Exception as exc:
        print(f"Skipping chronotype midpoint figures due to error: {exc}")

    coef_parts: List[pd.DataFrame] = []
    pretrend_parts: List[pd.DataFrame] = []
    total_rows_used = 0
    total_people = 0
    total_dates = 0

    for transition in ["spring", "fall"]:
        transition_day_col = "days_to_spring" if transition == "spring" else "days_to_fall"

        for metric_name, meta in metric_meta.items():
            metric_col = meta["col"]

            metric_lf = (
                base
                .with_columns(pl.col(transition_day_col).cast(pl.Int32).alias("event_day"))
                .filter(pl.col("event_day").is_between(-WINDOW, WINDOW))
                .select([
                    "person_id",
                    "sleep_date",
                    "year",
                    "dst_state",
                    "is_work_day",
                    "event_day",
                    metric_col,
                ])
                .drop_nulls([
                    "person_id",
                    "sleep_date",
                    "year",
                    "dst_state",
                    "is_work_day",
                    "event_day",
                    metric_col,
                ])
            )

            if test_mode:
                metric_lf = maybe_downsample_by_person_hash(metric_lf, TEST_SAMPLE_N)
            elif max_model_rows > 0:
                metric_lf = maybe_downsample_by_person_hash(metric_lf, max_model_rows)

            metric_n_rows = int(metric_lf.select(pl.len().alias("n")).collect(engine="streaming").item())
            if metric_n_rows == 0:
                continue

            metric_n_people_df = metric_lf.select(pl.col("person_id").n_unique().alias("n_people")).collect(engine="streaming")
            metric_n_dates_df = metric_lf.select(pl.col("sleep_date").n_unique().alias("n_dates")).collect(engine="streaming")

            total_rows_used += int(metric_n_rows)
            total_people += int(metric_n_people_df["n_people"][0]) if metric_n_people_df.height > 0 else 0
            total_dates += int(metric_n_dates_df["n_dates"][0]) if metric_n_dates_df.height > 0 else 0

            for daytype_name, daytype_filter in DAYTYPE_SPECS:
                if method == "low_memory":
                    res = run_low_memory_mean_did(
                        metric_lf=metric_lf,
                        transition_name=transition,
                        metric_name=metric_name,
                        metric_label=meta["label"],
                        units=meta["units"],
                        daytype_name=daytype_name,
                        daytype_filter=daytype_filter,
                        metric_col=metric_col,
                    )
                else:
                    metric_df = metric_lf.collect(engine="streaming")
                    if metric_df.height == 0:
                        continue

                    tdf = metric_df.to_pandas()
                    tdf["event_day"] = pd.to_numeric(tdf["event_day"], errors="coerce").astype("Int64")
                    tdf = tdf.dropna(subset=["event_day"]).copy()
                    if tdf.empty:
                        continue
                    tdf["event_day"] = tdf["event_day"].astype(int)
                    tdf["transition"] = transition

                    res = run_event_study(
                        df=tdf,
                        transition_name=transition,
                        metric_name=metric_name,
                        metric_col=metric_col,
                        metric_label=meta["label"],
                        units=meta["units"],
                        daytype_name=daytype_name,
                        daytype_filter=daytype_filter,
                    )

                if not res.coef_df.empty:
                    coef_parts.append(res.coef_df)
                if not res.pretrend_df.empty:
                    pretrend_parts.append(res.pretrend_df)

    coef_all = pd.concat(coef_parts, ignore_index=True) if coef_parts else pd.DataFrame()
    pretrend_all = pd.concat(pretrend_parts, ignore_index=True) if pretrend_parts else pd.DataFrame()

    if not coef_all.empty:
        coef_all = coef_all.sort_values(["transition", "metric", "day_type", "event_day"]).reset_index(drop=True)
        coef_all.to_csv(table_dir / "01_eventstudy_coefficients.csv", index=False)

    if not pretrend_all.empty:
        pretrend_all = pretrend_all.sort_values(["transition", "metric", "day_type"]).reset_index(drop=True)
        pretrend_all.to_csv(table_dir / "02_pretrend_diagnostics.csv", index=False)

    if not coef_all.empty:
        key_days = coef_all[coef_all["event_day"].isin([1, 3])].copy()
        key_days.to_csv(table_dir / "03_key_effect_days_1_and_3.csv", index=False)

    if not coef_all.empty:
        for transition in ["spring", "fall"]:
            for metric_name, meta in metric_meta.items():
                plot_metric_panels(
                    coef_all=coef_all,
                    transition=transition,
                    metric_name=metric_name,
                    metric_label=meta["label"],
                    units=meta["units"],
                    plot_dir=plot_dir,
                )
            plot_transition_overview(coef_all=coef_all, transition=transition, metric_meta=metric_meta, plot_dir=plot_dir)

    run_meta = {
        "method": method,
        "input_parquet": str(input_parquet),
        "test_mode": bool(test_mode),
        "dst_max_model_rows": int(max_model_rows),
        "window_days": WINDOW,
        "baseline_window_start": BASELINE_START,
        "baseline_window_end": BASELINE_END,
        "reference_day": REFERENCE_K,
        "plot_window": PLOT_WINDOW,
        "n_rows_used_across_transition_metric_runs": int(total_rows_used),
        "n_people_summed_across_runs": int(total_people),
        "n_dates_summed_across_runs": int(total_dates),
    }
    pd.DataFrame([run_meta]).to_csv(table_dir / "00_run_metadata.csv", index=False)

    print("Done.")
    print(f"Method: {method}")
    print(f"Tables: {table_dir}")
    print(f"Plots:  {plot_dir}")
