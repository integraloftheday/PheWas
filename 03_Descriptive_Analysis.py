#!/usr/bin/env python
# coding: utf-8

"""
03_Descriptive_Analysis.py

Publication-ready descriptive plots for sleep timing/duration.
All timing figures use human-readable 24h clock formatting.
"""

from pathlib import Path
from matplotlib.ticker import FuncFormatter
import calendar
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


INPUT_PATH = Path("processed_data/ready_for_analysis.parquet")
OUTPUT_DIR = Path("results/descriptive_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def linear_to_clock(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    return np.mod(vals, 24.0)


def to_noon_linear(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    return np.where(vals < 12.0, vals + 24.0, vals)


def ensure_noon_linear(s: pd.Series) -> pd.Series:
    vals = pd.to_numeric(s, errors="coerce")
    out = vals.copy()
    # If a value looks like clock-hour [0,24), move to noon-linearized scale.
    mask = out.notna() & (out >= 0.0) & (out < 24.0)
    out.loc[mask] = np.where(out.loc[mask] < 12.0, out.loc[mask] + 24.0, out.loc[mask])
    # Wrap all finite values into [12, 36) for consistent centered plotting.
    finite_mask = out.notna()
    out.loc[finite_mask] = np.mod(out.loc[finite_mask] - 12.0, 24.0) + 12.0
    return out


def hour_24_formatter(x: float, _pos: int) -> str:
    if not np.isfinite(x):
        return ""
    x = x % 24.0
    total_minutes = int(round(x * 60.0)) % (24 * 60)
    h = total_minutes // 60
    m = total_minutes % 60
    return f"{h:02d}:{m:02d}"


def style_publication() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 400,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def apply_time_axis(ax: plt.Axes, axis: str = "x") -> None:
    ticks = np.arange(12, 37, 2)
    formatter = FuncFormatter(hour_24_formatter)
    if axis == "x":
        ax.set_xlim(12, 36)
        ax.set_xticks(ticks)
        ax.xaxis.set_major_formatter(formatter)
    else:
        ax.set_ylim(12, 36)
        ax.set_yticks(ticks)
        ax.yaxis.set_major_formatter(formatter)


def apply_time_axis_zoom(ax: plt.Axes, values: pd.Series | np.ndarray) -> None:
    vals = pd.to_numeric(pd.Series(values), errors="coerce")
    vals = vals[np.isfinite(vals)]
    if vals.empty:
        apply_time_axis(ax, axis="y")
        return

    vmin = float(vals.min())
    vmax = float(vals.max())
    if np.isclose(vmin, vmax):
        vmin -= 0.5
        vmax += 0.5

    span = vmax - vmin
    pad = max(0.20 * span, 0.20)
    lo = vmin - pad
    hi = vmax + pad

    total = hi - lo
    if total <= 2.0:
        step = 0.25
    elif total <= 4.0:
        step = 0.5
    elif total <= 8.0:
        step = 1.0
    else:
        step = 2.0

    t0 = np.floor(lo / step) * step
    t1 = np.ceil(hi / step) * step
    ticks = np.arange(t0, t1 + step * 0.5, step)

    ax.set_ylim(lo, hi)
    ax.set_yticks(ticks)
    ax.yaxis.set_major_formatter(FuncFormatter(hour_24_formatter))


def month_week_ticks() -> tuple[list[int], list[str]]:
    # Non-leap template year; used only for labeling week-of-year axis.
    year = 2021
    month_starts = [dt.date(year, m, 1).timetuple().tm_yday for m in range(1, 13)]
    week_positions = [int(np.ceil(day / 7.0)) for day in month_starts]
    labels = [calendar.month_abbr[m] for m in range(1, 13)]
    return week_positions, labels


def add_week_and_month_labels(ax: plt.Axes) -> None:
    # Bottom axis: week-of-year numeric labels
    week_ticks = np.arange(1, 54, 4)
    ax.set_xlim(1, 53)
    ax.set_xticks(week_ticks)
    ax.set_xticklabels([str(int(w)) for w in week_ticks])
    ax.set_xlabel("Week of year")

    # Top axis: month labels at month starts
    month_ticks, month_labels = month_week_ticks()
    top_ax = ax.twiny()
    top_ax.set_xlim(ax.get_xlim())
    top_ax.set_xticks(month_ticks)
    top_ax.set_xticklabels(month_labels)
    top_ax.set_xlabel("Month")
    top_ax.grid(False)
    top_ax.spines["top"].set_visible(False)


def pick_column(schema_cols: set[str], candidates: list[str], required_name: str) -> str:
    for c in candidates:
        if c in schema_cols:
            return c
    raise ValueError(f"Missing required column for {required_name}. Tried: {candidates}")


def normalize_bool_like(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.fillna(False)
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0).astype(int) != 0

    txt = s.astype("string").str.strip().str.lower()
    true_vals = {"1", "true", "t", "yes", "y", "weekend"}
    false_vals = {"0", "false", "f", "no", "n", "weekday", "workday", "work_day"}
    out = pd.Series(False, index=s.index)
    out = out.where(~txt.isin(true_vals), True)
    out = out.where(~txt.isin(false_vals), False)
    return out


def build_midpoint_person_level(df: pd.DataFrame) -> pd.DataFrame:
    req = ["person_id", "is_work_day", "onset_linear_use", "duration_hours"]
    tmp = df[req].dropna().copy()
    if tmp.empty:
        return pd.DataFrame()

    g = (
        tmp.groupby(["person_id", "is_work_day"], as_index=False)
        .agg(
            mean_onset_linear=("onset_linear_use", "mean"),
            mean_duration_hours=("duration_hours", "mean"),
            n_nights=("duration_hours", "size"),
        )
    )

    work = g[g["is_work_day"] == 1].rename(
        columns={
            "mean_onset_linear": "SO_w",
            "mean_duration_hours": "SD_w",
            "n_nights": "n_work_nights",
        }
    )
    free = g[g["is_work_day"] == 0].rename(
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
        return pd.DataFrame()

    ms["MSW_linear"] = np.mod(ms["SO_w"] + (ms["SD_w"] / 2.0) - 12.0, 24.0) + 12.0
    ms["MSF_linear"] = np.mod(ms["SO_f"] + (ms["SD_f"] / 2.0) - 12.0, 24.0) + 12.0
    ms["SD_week"] = ((5.0 * ms["SD_w"]) + (2.0 * ms["SD_f"])) / 7.0

    msfsc = np.where(
        ms["SD_f"] > ms["SD_w"],
        ms["SO_f"] + (ms["SD_week"] / 2.0),
        ms["MSF_linear"],
    )
    ms["MSFsc_linear"] = np.mod(msfsc - 12.0, 24.0) + 12.0

    ms["MSW_clock"] = linear_to_clock(ms["MSW_linear"])
    ms["MSF_clock"] = linear_to_clock(ms["MSF_linear"])
    ms["MSFsc_clock"] = linear_to_clock(ms["MSFsc_linear"])
    return ms


def build_midpoint_person_level_from_summary(df: pd.DataFrame) -> pd.DataFrame:
    needed = {
        "person_id",
        "person_weekday_avg_start",
        "person_weekend_avg_start",
        "person_weekday_avg_duration",
        "person_weekend_avg_duration",
    }
    if not needed.issubset(df.columns):
        return pd.DataFrame()

    tmp = (
        df[
            [
                "person_id",
                "person_weekday_avg_start",
                "person_weekend_avg_start",
                "person_weekday_avg_duration",
                "person_weekend_avg_duration",
            ]
        ]
        .dropna()
        .drop_duplicates(subset=["person_id"])
        .copy()
    )
    if tmp.empty:
        return pd.DataFrame()

    so_w = to_noon_linear(tmp["person_weekday_avg_start"])
    so_f = to_noon_linear(tmp["person_weekend_avg_start"])
    sd_w = pd.to_numeric(tmp["person_weekday_avg_duration"], errors="coerce") / 60.0
    sd_f = pd.to_numeric(tmp["person_weekend_avg_duration"], errors="coerce") / 60.0

    out = pd.DataFrame({
        "person_id": tmp["person_id"],
        "SO_w": so_w,
        "SO_f": so_f,
        "SD_w": sd_w,
        "SD_f": sd_f,
    }).dropna()
    if out.empty:
        return pd.DataFrame()

    out["MSW_linear"] = np.mod(out["SO_w"] + (out["SD_w"] / 2.0) - 12.0, 24.0) + 12.0
    out["MSF_linear"] = np.mod(out["SO_f"] + (out["SD_f"] / 2.0) - 12.0, 24.0) + 12.0
    out["SD_week"] = ((5.0 * out["SD_w"]) + (2.0 * out["SD_f"])) / 7.0

    msfsc = np.where(
        out["SD_f"] > out["SD_w"],
        out["SO_f"] + (out["SD_week"] / 2.0),
        out["MSF_linear"],
    )
    out["MSFsc_linear"] = np.mod(msfsc - 12.0, 24.0) + 12.0
    out["MSW_clock"] = linear_to_clock(out["MSW_linear"])
    out["MSF_clock"] = linear_to_clock(out["MSF_linear"])
    out["MSFsc_clock"] = linear_to_clock(out["MSFsc_linear"])
    return out


def weekly_plot(agg: pd.DataFrame, y_col: str, title: str, filename: str, ylabel: str, is_time: bool = False) -> None:
    if agg.empty:
        return

    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.errorbar(
        agg["week"],
        agg["mean"],
        yerr=1.96 * agg["se"].fillna(0),
        fmt="o",
        alpha=0.35,
        markersize=3,
        color="#4c72b0",
        ecolor="#9fbbe8",
        elinewidth=0.8,
        capsize=1,
    )
    ax.plot(
        agg["week"],
        agg[y_col],
        linewidth=2.4,
        color="#1f4e79",
        label="3-week smoothed",
    )
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    add_week_and_month_labels(ax)
    ax.grid(alpha=0.25)
    if is_time:
        apply_time_axis_zoom(ax, agg["mean"])
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    style_publication()

    print(f"Loading {INPUT_PATH}...")
    lf = pl.scan_parquet(INPUT_PATH)
    schema_cols = set(lf.collect_schema().names())

    onset_col = pick_column(schema_cols, ["onset_linear", "daily_start_hour"], "onset")
    midpoint_col = pick_column(schema_cols, ["midpoint_linear", "daily_midpoint_hour"], "midpoint")
    offset_col = pick_column(schema_cols, ["offset_linear", "daily_end_hour"], "offset")
    duration_col = pick_column(schema_cols, ["daily_sleep_window_mins", "daily_duration_mins"], "duration")

    base_select = [
        "person_id",
        "sleep_date",
        "is_weekend",
        "is_weekend_or_holiday",
        "age_at_sleep",
        "sex_concept",
        "employment_status",
        "SJL_raw",
        "person_weekday_avg_start",
        "person_weekend_avg_start",
        "person_weekday_avg_duration",
        "person_weekend_avg_duration",
        onset_col,
        midpoint_col,
        offset_col,
        duration_col,
    ]
    use_cols = [c for c in base_select if c in schema_cols]

    df = (
        lf.select(use_cols)
        .with_columns(
            [
                pl.col("sleep_date").cast(pl.Date),
                pl.col("sleep_date").cast(pl.Date).dt.ordinal_day().alias("day_of_year"),
                pl.col(onset_col).cast(pl.Float64).alias("onset_linear_use"),
                pl.col(midpoint_col).cast(pl.Float64).alias("midpoint_linear_use"),
                pl.col(offset_col).cast(pl.Float64).alias("offset_linear_use"),
                (pl.col(duration_col).cast(pl.Float64) / 60.0).alias("duration_hours"),
            ]
        )
        .collect(engine="streaming")
        .to_pandas()
    )

    if "is_weekend_or_holiday" in df.columns:
        is_free = normalize_bool_like(df["is_weekend_or_holiday"])
        df["is_work_day"] = (~is_free).astype(int)
    elif "is_weekend" in df.columns:
        is_free = normalize_bool_like(df["is_weekend"])
        df["is_work_day"] = (~is_free).astype(int)
    else:
        raise ValueError("Need either is_weekend_or_holiday or is_weekend to define work/free days.")

    df["onset_centered"] = ensure_noon_linear(df["onset_linear_use"])
    df["midpoint_centered"] = ensure_noon_linear(df["midpoint_linear_use"])
    df["offset_centered"] = ensure_noon_linear(df["offset_linear_use"])
    df["onset_clock"] = linear_to_clock(df["onset_centered"])
    df["midpoint_clock"] = linear_to_clock(df["midpoint_centered"])
    df["offset_clock"] = linear_to_clock(df["offset_centered"])
    df["week"] = np.ceil(pd.to_numeric(df["day_of_year"], errors="coerce") / 7.0)

    midpoint_person = build_midpoint_person_level(df)
    if midpoint_person.empty:
        midpoint_person = build_midpoint_person_level_from_summary(df)
    if not midpoint_person.empty:
        midpoint_person.to_csv(OUTPUT_DIR / "04_midpoint_work_free_adjusted_person_level.csv", index=False)

    print("Generating figures...")

    # ---------- Plot 1: Midpoint histogram in 24h ----------
    fig, ax = plt.subplots(figsize=(10.0, 5.4))
    bins = np.arange(12, 36.5, 0.5)
    if not midpoint_person.empty:
        ax.hist(midpoint_person["MSW_linear"].dropna(), bins=bins, alpha=0.45, label="Work midpoint (MSW)", color="#1f77b4", density=True)
        ax.hist(midpoint_person["MSF_linear"].dropna(), bins=bins, alpha=0.45, label="Free midpoint (MSF)", color="#ff7f0e", density=True)
        ax.hist(midpoint_person["MSFsc_linear"].dropna(), bins=bins, alpha=0.45, label="Adjusted midpoint (MSFsc)", color="#2ca02c", density=True)
        ax.set_ylabel("Density")
        ax.legend(frameon=False, title="")
        ax.set_title("Distribution of Midpoint (work, free, adjusted)")
    else:
        ax.hist(df["midpoint_centered"].dropna(), bins=bins, color="#86b0d9", edgecolor="white")
        ax.set_ylabel("N nights")
        ax.set_title("Distribution of Sleep Midpoint")
    ax.set_xlabel("Midpoint time (24h, midnight-centered)")
    apply_time_axis(ax, axis="x")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "01_midpoint_hist_24h.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- Plot 2: nights per person ----------
    nights_per_person = df.groupby("person_id", as_index=False).size().rename(columns={"size": "n_nights"})
    fig, ax = plt.subplots(figsize=(9.6, 5.4))
    ax.hist(nights_per_person["n_nights"], bins=50, color="#4c72b0", edgecolor="white")
    ax.set_yscale("log")
    ax.set_title("Data density: nights per person")
    ax.set_xlabel("Number of nights")
    ax.set_ylabel("Count of people (log scale)")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_nights_per_person_log_hist.png", bbox_inches="tight")
    plt.close(fig)

    # ---------- Plot 3/4: onset/offset work vs free ----------
    for col, title, out_name in [
        ("onset_centered", "Sleep onset density: work vs free days", "03_onset_density_work_free_24h.png"),
        ("offset_centered", "Sleep offset density: work vs free days", "04_offset_density_work_free_24h.png"),
    ]:
        tmp = df[[col, "is_work_day"]].dropna().copy()
        if tmp.empty:
            continue
        tmp["day_type"] = np.where(tmp["is_work_day"] == 1, "Work days", "Free days")

        fig, ax = plt.subplots(figsize=(10.0, 5.5))
        sns.kdeplot(
            data=tmp,
            x=col,
            hue="day_type",
            common_norm=False,
            fill=True,
            alpha=0.35,
            bw_adjust=0.8,
            ax=ax,
            palette=["#1f77b4", "#d62728"],
        )
        ax.set_title(title)
        ax.set_xlabel("Clock time (24h, midnight-centered)")
        ax.set_ylabel("Density")
        apply_time_axis(ax, axis="x")
        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(False)
            leg.set_title("")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / out_name, bbox_inches="tight")
        plt.close(fig)

    # ---------- Plot 5: midpoint density work/free/adjusted ----------
    if not midpoint_person.empty:
        msw_df = midpoint_person[["MSW_clock"]].copy()
        msw_df = msw_df.rename(columns={"MSW_clock": "midpoint_clock"})
        msw_df["series"] = "Work midpoint (MSW)"

        msf_df = midpoint_person[["MSF_clock"]].copy()
        msf_df = msf_df.rename(columns={"MSF_clock": "midpoint_clock"})
        msf_df["series"] = "Free midpoint (MSF)"

        msfsc_df = midpoint_person[["MSFsc_clock"]].copy()
        msfsc_df = msfsc_df.rename(columns={"MSFsc_clock": "midpoint_clock"})
        msfsc_df["series"] = "Adjusted midpoint (MSFsc)"

        dens_df = pd.concat([msw_df, msf_df, msfsc_df], ignore_index=True).dropna()
        dens_df["midpoint_centered"] = ensure_noon_linear(dens_df["midpoint_clock"])

        fig, ax = plt.subplots(figsize=(10.5, 5.7))
        sns.kdeplot(
            data=dens_df,
            x="midpoint_centered",
            hue="series",
            fill=True,
            common_norm=False,
            alpha=0.35,
            bw_adjust=0.85,
            palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
            ax=ax,
        )
        ax.set_title("Midpoint density: work, free, and sleep-corrected")
        ax.set_xlabel("Midpoint time (24h, midnight-centered)")
        ax.set_ylabel("Density")
        apply_time_axis(ax, axis="x")
        leg = ax.get_legend()
        if leg is not None:
            leg.set_frame_on(False)
            leg.set_title("")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "05_midpoint_density_work_free_adjusted_24h.png", bbox_inches="tight")
        plt.close(fig)

    # ---------- Plot 6: social jetlag vs age ----------
    if {"SJL_raw", "age_at_sleep", "sex_concept"}.issubset(df.columns):
        sjl_age = (
            df.groupby(["person_id", "sex_concept"], as_index=False)
            .agg(mean_age=("age_at_sleep", "mean"), sjl_raw=("SJL_raw", "first"))
            .dropna(subset=["mean_age", "sjl_raw"])
        )
        if not sjl_age.empty:
            fig, ax = plt.subplots(figsize=(10.0, 6.0))
            sns.regplot(data=sjl_age, x="mean_age", y="sjl_raw", scatter=False, lowess=True, color="black", ax=ax)
            sample_n = min(len(sjl_age), 30_000)
            sns.scatterplot(
                data=sjl_age.sample(sample_n, random_state=42),
                x="mean_age",
                y="sjl_raw",
                hue="sex_concept",
                alpha=0.16,
                s=10,
                ax=ax,
            )
            ax.axhline(0, linestyle="--", color="gray", linewidth=1)
            ax.set_title("Social jetlag across the lifespan")
            ax.set_xlabel("Age (years)")
            ax.set_ylabel("Social jetlag (hours)")
            ax.legend(frameon=False, title="Sex")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "06_sjl_vs_age.png", bbox_inches="tight")
            plt.close(fig)

    # ---------- Plot 7: weekly midpoint trend (work/free/adjusted together) ----------
    week_mid_frames: list[pd.DataFrame] = []
    week_work = df[df["is_work_day"] == 1][["week", "midpoint_centered"]].dropna()
    if not week_work.empty:
        week_mid_frames.append(week_work.assign(series="Work midpoint (MSW proxy)").rename(columns={"midpoint_centered": "value"}))

    week_free = df[df["is_work_day"] == 0][["week", "midpoint_centered"]].dropna()
    if not week_free.empty:
        week_mid_frames.append(week_free.assign(series="Free midpoint (MSF proxy)").rename(columns={"midpoint_centered": "value"}))

    if not midpoint_person.empty:
        person_week = (
            df[["person_id", "week"]]
            .dropna()
            .drop_duplicates()
            .merge(midpoint_person[["person_id", "MSFsc_linear"]], on="person_id", how="inner")
            .rename(columns={"MSFsc_linear": "value"})
            .assign(series="Adjusted midpoint (MSFsc)")
        )
        week_mid_frames.append(person_week[["week", "value", "series"]])

    if week_mid_frames:
        wm = pd.concat(week_mid_frames, ignore_index=True)
        weekly_mid = (
            wm.groupby(["week", "series"], as_index=False)
            .agg(
                mean=("value", "mean"),
                se=("value", lambda x: x.std(ddof=1) / np.sqrt(max(len(x), 1))),
            )
            .sort_values(["series", "week"])
        )
        weekly_mid["smooth"] = weekly_mid.groupby("series")["mean"].transform(lambda s: s.rolling(3, min_periods=1, center=True).mean())

        fig, ax = plt.subplots(figsize=(11.2, 6.0))
        palette = {
            "Work midpoint (MSW proxy)": "#1f77b4",
            "Free midpoint (MSF proxy)": "#ff7f0e",
            "Adjusted midpoint (MSFsc)": "#2ca02c",
        }
        for series, g in weekly_mid.groupby("series"):
            c = palette.get(series, "#333333")
            ax.plot(g["week"], g["smooth"], linewidth=2.4, color=c, label=series)
            ax.fill_between(g["week"], g["mean"] - 1.96 * g["se"].fillna(0), g["mean"] + 1.96 * g["se"].fillna(0), color=c, alpha=0.12)

        ax.set_title("Weekly seasonal trend: midpoint (work, free, adjusted)")
        ax.set_ylabel("Midpoint time (24h, midnight-centered)")
        add_week_and_month_labels(ax)
        apply_time_axis_zoom(ax, weekly_mid["mean"])
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, title="")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "07_weekly_midpoint_work_free_adjusted_24h.png", bbox_inches="tight")
        plt.close(fig)

    # ---------- Plot 8/9/10: weekly onset/offset/duration ----------
    for col, title, out_name, ylabel, is_time in [
        ("onset_centered", "Weekly seasonal trend: sleep onset", "08_weekly_onset_24h.png", "Onset time (24h, midnight-centered)", True),
        ("offset_centered", "Weekly seasonal trend: sleep offset", "09_weekly_offset_24h.png", "Offset time (24h, midnight-centered)", True),
        ("duration_hours", "Weekly seasonal trend: sleep duration", "10_weekly_duration_hours.png", "Duration (hours)", False),
    ]:
        tmp = df[["week", col]].dropna()
        if tmp.empty:
            continue
        agg = (
            tmp.groupby("week", as_index=False)
            .agg(mean=(col, "mean"), se=(col, lambda x: x.std(ddof=1) / np.sqrt(max(len(x), 1))))
            .sort_values("week")
        )
        agg["smooth"] = agg["mean"].rolling(3, min_periods=1, center=True).mean()
        weekly_plot(agg, "smooth", title, out_name, ylabel, is_time=is_time)

    # ---------- Plot 11: age-bin plots (5-year bins; midpoint includes all three) ----------
    age_person = (
        df[["person_id", "age_at_sleep", "duration_hours", "onset_centered", "offset_centered"]]
        .dropna(subset=["person_id", "age_at_sleep"])
        .groupby("person_id", as_index=False)
        .agg(
            age=("age_at_sleep", "mean"),
            duration=("duration_hours", "mean"),
            onset=("onset_centered", "mean"),
            offset=("offset_centered", "mean"),
        )
    )
    if not age_person.empty:
        low = int(np.floor(age_person["age"].min() / 5.0) * 5)
        high = int(np.ceil(age_person["age"].max() / 5.0) * 5 + 5)
        bins = np.arange(low, high + 1, 5)
        labels = [f"{a}-{a+4}" for a in bins[:-1]]
        age_person["age_bin"] = pd.cut(age_person["age"], bins=bins, labels=labels, right=False)
        age_person = age_person.dropna(subset=["age_bin"])

        midpoint_age = pd.DataFrame()
        if not midpoint_person.empty:
            midpoint_age = midpoint_person[["person_id", "MSW_linear", "MSF_linear", "MSFsc_linear"]].merge(
                age_person[["person_id", "age_bin"]], on="person_id", how="inner"
            )

        fig, axs = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)

        dur_agg = age_person.groupby("age_bin", observed=True, as_index=False).agg(
            mean=("duration", "mean"),
            se=("duration", lambda x: x.std(ddof=1) / np.sqrt(max(len(x), 1))),
        )
        x_dur = np.arange(len(dur_agg))
        axs[0, 0].errorbar(
            x_dur,
            dur_agg["mean"],
            yerr=1.96 * dur_agg["se"].fillna(0),
            marker="o",
            linestyle="-",
            color="#4c72b0",
            ecolor="#9fbbe8",
            elinewidth=1,
            capsize=2,
        )
        axs[0, 0].set_xticks(x_dur)
        axs[0, 0].set_xticklabels(dur_agg["age_bin"].astype(str))
        axs[0, 0].set_title("Duration vs age (5-year bins)")
        axs[0, 0].set_xlabel("Age bin (years)")
        axs[0, 0].set_ylabel("Duration (hours)")
        axs[0, 0].tick_params(axis="x", rotation=45)

        if not midpoint_age.empty:
            mid_long = midpoint_age.melt(
                id_vars=["age_bin"],
                value_vars=["MSW_linear", "MSF_linear", "MSFsc_linear"],
                var_name="series",
                value_name="midpoint_clock",
            )
            mid_map = {
                "MSW_linear": "Work midpoint",
                "MSF_linear": "Free midpoint",
                "MSFsc_linear": "Adjusted midpoint",
            }
            mid_long["series"] = mid_long["series"].map(mid_map)
            mid_agg = mid_long.groupby(["age_bin", "series"], observed=True, as_index=False).agg(
                mean=("midpoint_clock", "mean"),
                se=("midpoint_clock", lambda x: x.std(ddof=1) / np.sqrt(max(len(x), 1))),
            )

            palette = {
                "Work midpoint": "#1f77b4",
                "Free midpoint": "#ff7f0e",
                "Adjusted midpoint": "#2ca02c",
            }
            age_levels = [str(v) for v in mid_agg["age_bin"].drop_duplicates().tolist()]
            x_map = {lvl: i for i, lvl in enumerate(age_levels)}

            for series, g in mid_agg.groupby("series"):
                g = g.copy()
                g["age_str"] = g["age_bin"].astype(str)
                x_vals = g["age_str"].map(x_map).to_numpy(dtype=float)
                axs[0, 1].errorbar(
                    x_vals,
                    g["mean"],
                    yerr=1.96 * g["se"].fillna(0),
                    marker="o",
                    linestyle="-",
                    color=palette.get(series, "#333333"),
                    ecolor=palette.get(series, "#333333"),
                    elinewidth=0.9,
                    capsize=2,
                    label=series,
                )

            axs[0, 1].set_xticks(np.arange(len(age_levels)))
            axs[0, 1].set_xticklabels(age_levels)
            axs[0, 1].legend(frameon=False, title="")
            apply_time_axis_zoom(axs[0, 1], mid_agg["mean"])
        axs[0, 1].set_title("Midpoint vs age (work, free, adjusted)")
        axs[0, 1].set_xlabel("Age bin (years)")
        axs[0, 1].set_ylabel("Midpoint time (24h)")
        axs[0, 1].tick_params(axis="x", rotation=45)

        onset_agg = age_person.groupby("age_bin", observed=True, as_index=False).agg(
            mean=("onset", "mean"),
            se=("onset", lambda x: x.std(ddof=1) / np.sqrt(max(len(x), 1))),
        )
        x_on = np.arange(len(onset_agg))
        axs[1, 0].errorbar(
            x_on,
            onset_agg["mean"],
            yerr=1.96 * onset_agg["se"].fillna(0),
            marker="o",
            linestyle="-",
            color="#dd8452",
            ecolor="#e9b99c",
            elinewidth=1,
            capsize=2,
        )
        axs[1, 0].set_xticks(x_on)
        axs[1, 0].set_xticklabels(onset_agg["age_bin"].astype(str))
        axs[1, 0].set_title("Onset vs age (5-year bins)")
        axs[1, 0].set_xlabel("Age bin (years)")
        axs[1, 0].set_ylabel("Onset time (24h)")
        axs[1, 0].tick_params(axis="x", rotation=45)
        apply_time_axis_zoom(axs[1, 0], onset_agg["mean"])

        offset_agg = age_person.groupby("age_bin", observed=True, as_index=False).agg(
            mean=("offset", "mean"),
            se=("offset", lambda x: x.std(ddof=1) / np.sqrt(max(len(x), 1))),
        )
        x_off = np.arange(len(offset_agg))
        axs[1, 1].errorbar(
            x_off,
            offset_agg["mean"],
            yerr=1.96 * offset_agg["se"].fillna(0),
            marker="o",
            linestyle="-",
            color="#55a868",
            ecolor="#9fd2ae",
            elinewidth=1,
            capsize=2,
        )
        axs[1, 1].set_xticks(x_off)
        axs[1, 1].set_xticklabels(offset_agg["age_bin"].astype(str))
        axs[1, 1].set_title("Offset vs age (5-year bins)")
        axs[1, 1].set_xlabel("Age bin (years)")
        axs[1, 1].set_ylabel("Offset time (24h)")
        axs[1, 1].tick_params(axis="x", rotation=45)
        apply_time_axis_zoom(axs[1, 1], offset_agg["mean"])

        fig.savefig(OUTPUT_DIR / "11_age_bin_sleep_metrics_5y.png", bbox_inches="tight")
        plt.close(fig)

    # ---------- Plot 12: violin plot employment vs midpoint (work/free/adjusted) ----------
    if "employment_status" in df.columns and not midpoint_person.empty:
        person_emp = (
            df[["person_id", "employment_status"]]
            .dropna()
            .groupby("person_id", as_index=False)["employment_status"]
            .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else x.iloc[0])
        )
        emp_mid = midpoint_person[["person_id", "MSW_linear", "MSF_linear", "MSFsc_linear"]].merge(
            person_emp,
            on="person_id",
            how="inner",
        )
        if not emp_mid.empty:
            order = emp_mid["employment_status"].value_counts().index.tolist()
            emp_long = emp_mid.melt(
                id_vars=["employment_status"],
                value_vars=["MSW_linear", "MSF_linear", "MSFsc_linear"],
                var_name="series",
                value_name="midpoint_clock",
            )
            emp_long["series"] = emp_long["series"].map(
                {
                    "MSW_linear": "Work midpoint",
                    "MSF_linear": "Free midpoint",
                    "MSFsc_linear": "Adjusted midpoint",
                }
            )

            fig, ax = plt.subplots(figsize=(12.8, 6.2))
            sns.violinplot(
                data=emp_long,
                x="employment_status",
                y="midpoint_clock",
                hue="series",
                split=False,
                inner="quart",
                linewidth=0.8,
                order=order,
                palette=["#1f77b4", "#ff7f0e", "#2ca02c"],
                ax=ax,
            )
            ax.set_title("Midpoint by employment status (work, free, adjusted)")
            ax.set_xlabel("Employment status")
            ax.set_ylabel("Midpoint time (24h, midnight-centered)")
            ax.tick_params(axis="x", rotation=25)
            apply_time_axis_zoom(ax, emp_long["midpoint_clock"])
            ax.legend(frameon=False, title="")
            fig.tight_layout()
            fig.savefig(OUTPUT_DIR / "12_employment_vs_midpoint_violin_24h.png", bbox_inches="tight")
            plt.close(fig)

    print(f"Done. Saved outputs to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
