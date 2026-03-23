#!/usr/bin/env python3
"""
03_3_DLST_Adjusted.py

Event-study Difference-in-Differences (DiD) around DST transitions using person-night data.

Core design:
- Person fixed effects
- Calendar-date fixed effects
- Dynamic treatment effects by event day (DST_state x 1[event_time = k])
- Cluster-robust SE at person level

Outputs (results/dst_adjusted):
- event-study coefficient tables (CSV)
- pretrend diagnostics (CSV)
- standalone-labeled plots for all metrics and day-type splits

Notes:
- Uses days -14..+14, omits k=-1 as reference period.
- Baseline centering per person-year on days -14..-7.
- Works for both spring and fall transitions.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt


# -----------------------------
# Configuration
# -----------------------------
INPUT_PARQUET = Path("processed_data/ready_for_analysis.parquet")
OUTPUT_DIR = Path("results/dst_adjusted")
PLOT_DIR = OUTPUT_DIR / "plots"
TABLE_DIR = OUTPUT_DIR / "tables"
for d in [OUTPUT_DIR, PLOT_DIR, TABLE_DIR]:
    d.mkdir(parents=True, exist_ok=True)

WINDOW = 14
BASELINE_START = -14
BASELINE_END = -7
REFERENCE_K = -1
PLOT_WINDOW = 7

# Set TEST_MODE=1 for quick development runs.
TEST_MODE = str(__import__("os").getenv("TEST_MODE", "0")).strip() == "1"
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


# -----------------------------
# Utilities
# -----------------------------
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


def demean_two_way(v: np.ndarray, person_codes: np.ndarray, date_codes: np.ndarray, max_iter: int = 200, tol: float = 1e-10) -> np.ndarray:
    """Two-way demeaning by alternating projections (person FE + date FE)."""
    r = v.astype(np.float64, copy=True)

    n_person = int(person_codes.max()) + 1
    n_date = int(date_codes.max()) + 1

    for _ in range(max_iter):
        prev = r.copy()

        # Remove person means
        p_sum = np.bincount(person_codes, weights=r, minlength=n_person)
        p_cnt = np.bincount(person_codes, minlength=n_person)
        p_mean = np.divide(p_sum, np.maximum(p_cnt, 1), out=np.zeros_like(p_sum), where=p_cnt > 0)
        r = r - p_mean[person_codes]

        # Remove date means
        d_sum = np.bincount(date_codes, weights=r, minlength=n_date)
        d_cnt = np.bincount(date_codes, minlength=n_date)
        d_mean = np.divide(d_sum, np.maximum(d_cnt, 1), out=np.zeros_like(d_sum), where=d_cnt > 0)
        r = r - d_mean[date_codes]

        # Stabilize around global mean zero
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
    """
    Fit TWFE via Frisch-Waugh residualization, clustered by person.

    Returns:
    - beta
    - se_cluster
    - vcov_cluster
    """
    y_t = demean_two_way(y, person_codes, date_codes)
    X_t_cols = [demean_two_way(X[:, j], person_codes, date_codes) for j in range(X.shape[1])]
    X_t = np.column_stack(X_t_cols)

    XtX = X_t.T @ X_t
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X_t.T @ y_t)

    resid = y_t - X_t @ beta

    # Cluster-robust sandwich estimator by person
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

    # Small-sample style finite-cluster correction (HC1-like for clusters)
    n = X_t.shape[0]
    G = len(p_unique)
    if G > 1 and n > k:
        corr = (G / (G - 1)) * ((n - 1) / (n - k))
        vcov *= corr

    se = np.sqrt(np.clip(np.diag(vcov), a_min=0.0, a_max=None))
    return beta, se, vcov


def normal_approx_p(beta: np.ndarray, se: np.ndarray) -> np.ndarray:
    z = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    # Two-sided normal approximation via erf
    from math import erf, sqrt

    p = np.array([2.0 * (1.0 - 0.5 * (1.0 + erf(abs(val) / sqrt(2.0)))) if np.isfinite(val) else np.nan for val in z])
    return p


@dataclass
class EventStudyResult:
    coef_df: pd.DataFrame
    pretrend_df: pd.DataFrame


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

    # Keep event window and drop reference period from regressors only (rows kept)
    work = work[(work[day_col] >= -WINDOW) & (work[day_col] <= WINDOW)].copy()

    # Baseline per person-year: mean over [-14, -7]
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

    # Build event-time interaction design for k != reference
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

    # Drop all-zero columns defensively
    nonzero_mask = np.asarray((X != 0).any(axis=0)).ravel()
    X = X[:, nonzero_mask]
    kept_cols = [c for c, keep in zip(x_cols, nonzero_mask) if keep]
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
        }
    ).sort_values("event_day")

    # Pretrend joint Wald test: all k < 0 coefficients (excluding reference -1)
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
            }
        ]
    )

    return EventStudyResult(coef_df=coef_df, pretrend_df=pretrend_df)


# -----------------------------
# Plotting
# -----------------------------
def plot_metric_panels(coef_all: pd.DataFrame, transition: str, metric_name: str, metric_label: str, units: str) -> None:
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
    fig.suptitle(
        f"Event-study DiD: {metric_label} around {transition.title()} DST transition\n"
        f"Model: person FE + calendar-date FE + DST_state × event-day indicators (reference day = -1)",
        fontsize=12,
        y=1.02,
    )
    plt.tight_layout()
    out_path = PLOT_DIR / f"eventstudy_{transition}_{metric_name}_panels.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_transition_overview(coef_all: pd.DataFrame, transition: str, metric_meta: Dict[str, Dict[str, str]]) -> None:
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

    fig.suptitle(
        f"DST event-study DiD overview ({transition.title()} transition, all days)\n"
        f"Person FE + calendar-date FE; baseline-centered on days {BASELINE_START} to {BASELINE_END}",
        fontsize=13,
        y=1.01,
    )
    plt.tight_layout()
    out_path = PLOT_DIR / f"eventstudy_{transition}_overview_2x2.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    print(f"Loading data from {INPUT_PARQUET}...")
    lf = pl.scan_parquet(INPUT_PARQUET)
    schema_cols = set(lf.collect_schema().names())

    metric_meta = choose_metric_columns(schema_cols)
    zip3_expr = normalize_zip3_expr(schema_cols)

    # Pick weekend indicator robustly
    if "is_weekend_or_holiday" in schema_cols:
        weekend_expr = pl.col("is_weekend_or_holiday").cast(pl.Boolean)
    elif "is_weekend" in schema_cols:
        weekend_expr = pl.col("is_weekend").cast(pl.Boolean)
    else:
        weekend_expr = pl.lit(None, dtype=pl.Boolean)

    # Build transition lookup table
    years = range(2009, 2025)
    rows = []
    for y in years:
        spring, fall = get_dst_dates(y)
        rows.append({"year": y, "spring_dst": spring, "fall_dst": fall})
    transitions_df = pl.DataFrame(rows)

    needed_cols = ["person_id", "sleep_date"]
    for m in metric_meta.values():
        needed_cols.append(m["col"])

    # Ensure source columns used in derived expressions are retained.
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

    # Convert duration to minutes if needed
    for key, meta in metric_meta.items():
        col = meta["col"]
        if key == "duration" and col == "daily_sleep_window_mins":
            base = base.with_columns(pl.col(col).cast(pl.Float64).alias("daily_duration_mins"))
            metric_meta[key]["col"] = "daily_duration_mins"
        elif key == "duration" and col == "daily_duration_mins":
            base = base.with_columns(pl.col("daily_duration_mins").cast(pl.Float64))
        else:
            base = base.with_columns(pl.col(col).cast(pl.Float64))

    # Keep only needed analysis rows to reduce memory
    base = base.filter(
        pl.col("person_id").is_not_null() &
        pl.col("sleep_date").is_not_null() &
        pl.col("dst_state").is_not_null()
    )

    if TEST_MODE:
        print(f"TEST_MODE=1 active: sampling up to {TEST_SAMPLE_N:,} rows for quick run.")
        base_df = base.collect(engine="streaming")
        if base_df.height > TEST_SAMPLE_N:
            base_df = base_df.sample(n=TEST_SAMPLE_N, seed=42)
        pdf_all = base_df.to_pandas()
    else:
        pdf_all = base.collect(engine="streaming").to_pandas()

    if pdf_all.empty:
        raise RuntimeError("No rows available after preprocessing.")

    # Build two transition-specific long frames
    spring_df = pdf_all.copy()
    spring_df["transition"] = "spring"
    spring_df["event_day"] = pd.to_numeric(spring_df["days_to_spring"], errors="coerce")

    fall_df = pdf_all.copy()
    fall_df["transition"] = "fall"
    fall_df["event_day"] = pd.to_numeric(fall_df["days_to_fall"], errors="coerce")

    all_transition_df = pd.concat([spring_df, fall_df], ignore_index=True)
    all_transition_df = all_transition_df.dropna(subset=["event_day"]) 
    all_transition_df["event_day"] = all_transition_df["event_day"].astype(int)

    coef_parts: List[pd.DataFrame] = []
    pretrend_parts: List[pd.DataFrame] = []

    for transition in ["spring", "fall"]:
        tdf = all_transition_df[all_transition_df["transition"] == transition].copy()
        for metric_name, meta in metric_meta.items():
            metric_col = meta["col"]
            for daytype_name, daytype_filter in DAYTYPE_SPECS:
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

    if coef_parts:
        coef_all = pd.concat(coef_parts, ignore_index=True)
    else:
        coef_all = pd.DataFrame()

    if pretrend_parts:
        pretrend_all = pd.concat(pretrend_parts, ignore_index=True)
    else:
        pretrend_all = pd.DataFrame()

    if not coef_all.empty:
        coef_all = coef_all.sort_values(["transition", "metric", "day_type", "event_day"]).reset_index(drop=True)
        coef_all.to_csv(TABLE_DIR / "01_eventstudy_coefficients.csv", index=False)

    if not pretrend_all.empty:
        pretrend_all = pretrend_all.sort_values(["transition", "metric", "day_type"]).reset_index(drop=True)
        pretrend_all.to_csv(TABLE_DIR / "02_pretrend_diagnostics.csv", index=False)

    # Summary table at day +1 and +3 for quick interpretation
    if not coef_all.empty:
        key_days = coef_all[coef_all["event_day"].isin([1, 3])].copy()
        key_days.to_csv(TABLE_DIR / "03_key_effect_days_1_and_3.csv", index=False)

    # Plots
    if not coef_all.empty:
        for transition in ["spring", "fall"]:
            for metric_name, meta in metric_meta.items():
                plot_metric_panels(
                    coef_all=coef_all,
                    transition=transition,
                    metric_name=metric_name,
                    metric_label=meta["label"],
                    units=meta["units"],
                )
            plot_transition_overview(coef_all=coef_all, transition=transition, metric_meta=metric_meta)

    # Run metadata
    run_meta = {
        "input_parquet": str(INPUT_PARQUET),
        "test_mode": TEST_MODE,
        "window_days": WINDOW,
        "baseline_window_start": BASELINE_START,
        "baseline_window_end": BASELINE_END,
        "reference_day": REFERENCE_K,
        "plot_window": PLOT_WINDOW,
        "n_rows_input_after_prep": int(all_transition_df.shape[0]),
        "n_people": int(all_transition_df["person_id"].nunique()),
        "n_dates": int(all_transition_df["sleep_date"].nunique()),
    }
    pd.DataFrame([run_meta]).to_csv(TABLE_DIR / "00_run_metadata.csv", index=False)

    print("Done.")
    print(f"Tables: {TABLE_DIR}")
    print(f"Plots:  {PLOT_DIR}")


if __name__ == "__main__":
    main()
