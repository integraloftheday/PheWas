#!/usr/bin/env python
# coding: utf-8

"""
03_5_Descriptive_Analysis_Age_Colinear.py

Creates an age density plot (x = age, y = density) stratified by employment cluster.
The visual style mirrors density figures in 03_Descriptive_Analysis.py.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns


INPUT_CANDIDATES = [
    Path("processed_data/ready_for_analysis.parquet"),
    Path("processed_data/LMM_analysis.parquet"),
]
OUTPUT_DIR = Path("results/descriptive_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DENSITY_BW_ADJUST = 1.10
DENSITY_GRIDSIZE = 1024


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


def pick_column(schema_cols: set[str], candidates: list[str], required_name: str) -> str:
    for c in candidates:
        if c in schema_cols:
            return c
    raise ValueError(f"Missing required column for {required_name}. Tried: {candidates}")


def choose_input_path() -> tuple[Path, str, str]:
    required_age = ["age_at_sleep"]
    required_emp = ["employment_cluster", "employment_status"]

    for path in INPUT_CANDIDATES:
        if not path.exists():
            continue
        schema_cols = set(pl.scan_parquet(path).collect_schema().names())
        try:
            age_col = pick_column(schema_cols, required_age, "age")
            emp_col = pick_column(schema_cols, required_emp, "employment cluster")
            return path, age_col, emp_col
        except ValueError:
            continue

    raise ValueError(
        "Could not find a parquet input containing both age and employment cluster/status columns. "
        f"Tried: {[str(p) for p in INPUT_CANDIDATES]}"
    )


def clean_employment_value(value: object) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Unknown"
    txt = str(value).strip()
    if not txt:
        return "Unknown"
    return txt


def modal_or_first_nonnull(values: pd.Series) -> str:
    cleaned = values.map(clean_employment_value)
    if cleaned.empty:
        return "Unknown"
    mode = cleaned.mode(dropna=False)
    if not mode.empty:
        return str(mode.iloc[0])
    return str(cleaned.iloc[0])


def main() -> None:
    style_publication()

    input_path, age_col, employment_col = choose_input_path()
    print(f"Loading {input_path}...")
    lf = pl.scan_parquet(input_path)

    df = (
        lf.select(["person_id", age_col, employment_col])
        .with_columns(
            [
                pl.col(age_col).cast(pl.Float64).alias("age_at_sleep"),
                pl.col(employment_col).cast(pl.Utf8).alias("employment_cluster"),
            ]
        )
        .collect(engine="streaming")
        .to_pandas()
    )

    person_level = (
        df.groupby("person_id", as_index=False)
        .agg(
            age=("age_at_sleep", "mean"),
            employment_cluster=("employment_cluster", modal_or_first_nonnull),
        )
        .dropna(subset=["age"])
    )

    person_level = person_level[np.isfinite(person_level["age"])].copy()
    if person_level.empty:
        raise ValueError("No person-level rows with valid age available for plotting.")

    if person_level["employment_cluster"].nunique(dropna=False) < 1:
        raise ValueError("No employment cluster values available for plotting.")

    # Use true employment classes directly (no collapsing into "Other").
    person_level["employment_cluster_plot"] = person_level["employment_cluster"]

    # Stable legend order: largest groups first, then Other if present.
    order = (
        person_level["employment_cluster_plot"]
        .value_counts()
        .sort_values(ascending=False)
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(10.0, 5.5))
    sns.kdeplot(
        data=person_level,
        x="age",
        hue="employment_cluster_plot",
        hue_order=order,
        common_norm=False,
        fill=True,
        alpha=0.35,
        bw_adjust=DENSITY_BW_ADJUST,
        gridsize=DENSITY_GRIDSIZE,
        cut=0,
        ax=ax,
    )

    ax.set_title("Age density by employment cluster")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Density")

    leg = ax.get_legend()
    handles, labels = ax.get_legend_handles_labels()
    if leg is not None:
        leg.remove()
    if handles and labels:
        if labels[0] == "employment_cluster_plot":
            handles = handles[1:]
            labels = labels[1:]
        ax.legend(
            handles,
            labels,
            title="Employment cluster",
            loc="upper center",
            bbox_to_anchor=(0.5, -0.28),
            ncol=3,
            frameon=False,
            columnspacing=1.1,
            handlelength=1.6,
            borderaxespad=0.0,
        )

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.3)
    out_png = OUTPUT_DIR / "03_5_age_density_by_employment_cluster.png"
    fig.savefig(out_png, bbox_inches="tight")
    plt.close(fig)

    out_csv = OUTPUT_DIR / "03_5_age_density_by_employment_cluster_source.csv"
    person_level[["person_id", "age", "employment_cluster", "employment_cluster_plot"]].to_csv(out_csv, index=False)

    print(f"Saved: {out_png}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
