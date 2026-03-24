#!/usr/bin/env python3
"""
02b_Create_Analysis_Subset.py
Creates a balanced subset of the main analysis dataset for quick testing.

Usage:
  python 02b_Create_Analysis_Subset.py <path_to_config.yaml> <run_dir>
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml


NO_DST_ZIP3 = [f"{i:03d}" for i in range(850, 866)] + ["967", "968", "006", "007", "008", "009", "969"]


def normalize_zip3(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    digits = re.sub(r"[^0-9]", "", s)
    if not digits:
        return None
    if len(digits) <= 3:
        return f"{int(digits):03d}"
    if len(digits) == 4:
        return f"{int(digits):05d}"[:3]
    return digits[:3]


def to_sex_binary(value: Any) -> str:
    s = "" if value is None else str(value).lower()
    if re.search(r"female|woman|girl", s):
        return "Female"
    if re.search(r"male|man|boy", s):
        return "Male"
    return "Other/Unknown"


def to_age_bin(value: Any) -> str:
    if value is None:
        return "Unknown"
    try:
        x = float(value)
    except Exception:
        return "Unknown"
    if math.isnan(x):
        return "Unknown"
    if x < 40:
        return "18-39"
    if x < 55:
        return "40-54"
    if x < 70:
        return "55-69"
    return "70+"


def to_race_collapsed(value: Any) -> str:
    s = "" if value is None else str(value).lower().strip()
    if not s:
        return "Unknown"
    if re.search(r"white", s):
        return "White"
    if re.search(r"black|african", s):
        return "Black"
    if re.search(r"asian", s):
        return "Asian"
    if re.search(r"hispanic|latino", s):
        return "Hispanic/Latino"
    if re.search(r"american indian|alaska native", s):
        return "AI/AN"
    if re.search(r"native hawaiian|pacific islander", s):
        return "NH/PI"
    if re.search(r"more than one|multiracial|multiple", s):
        return "Multiracial"
    return "Other"


def to_span_bin(span_days: Any) -> str:
    if span_days is None:
        return "Unknown"
    try:
        x = float(span_days)
    except Exception:
        return "Unknown"
    if math.isnan(x):
        return "Unknown"
    if x >= 365:
        return ">=12m"
    if x >= 183:
        return "6-12m"
    return "<6m"


def to_nights_bin(nights: Any) -> str:
    if nights is None:
        return "Unknown"
    try:
        x = float(nights)
    except Exception:
        return "Unknown"
    if math.isnan(x):
        return "Unknown"
    if x >= 300:
        return ">=300 nights"
    if x >= 180:
        return "180-299 nights"
    if x >= 90:
        return "90-179 nights"
    return "<90 nights"


def to_longitudinal_weight(span_bin: str) -> float:
    if span_bin == ">=12m":
        return 3.0
    if span_bin == "6-12m":
        return 2.0
    return 1.0


def normalize_balance_var(v: str) -> str:
    if v in {"sex", "sex_concept", "sex_binary"}:
        return "sex_binary"
    if v in {"age", "age_at_sleep", "age_bin"}:
        return "age_bin"
    if v in {"race", "race_collapsed"}:
        return "race_collapsed"
    if v in {"dst_observed", "dst_observes", "dst_group"}:
        return "dst_observed"
    if v in {"recording_span", "recording_span_bin", "longitudinal_bin"}:
        return "recording_span_bin"
    if v in {"nights", "nights_bin", "nights_recorded_bin"}:
        return "nights_bin"
    return v


def weighted_group_sample(person_df: pl.DataFrame, group_cols: list[str], target_per_group: int, seed: int) -> pl.DataFrame:
    pdf = person_df.to_pandas()
    rng = np.random.default_rng(seed)

    sampled_parts = []
    for _, grp in pdf.groupby(group_cols, dropna=False, sort=False):
        n_take = min(len(grp), target_per_group)
        if n_take <= 0:
            continue
        w = grp["longitudinal_weight"].astype(float).to_numpy()
        if np.isfinite(w).sum() == 0 or float(w.sum()) <= 0:
            w = np.ones_like(w, dtype=float)
        probs = w / w.sum()
        idx = rng.choice(grp.index.to_numpy(), size=n_take, replace=False, p=probs)
        sampled_parts.append(grp.loc[idx])

    if not sampled_parts:
        return person_df.head(0)

    sampled_pdf = pl.from_pandas(
        __import__("pandas").concat(sampled_parts, ignore_index=True)
    )
    return sampled_pdf


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: 02b_Create_Analysis_Subset.py <path_to_config.yaml> <run_dir>")

    config_path = Path(sys.argv[1])
    run_dir = Path(sys.argv[2])
    run_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    dataset_type = config.get("dataset_type")
    input_path = Path(config.get("data_paths", {}).get(dataset_type, ""))
    subset_size = int(config.get("subset_size", 500))
    balance_vars = config.get("subset_balance_vars", [])
    subset_seed = int(config.get("subset_seed", 42))
    include_other_unknown_sex = bool(config.get("subset_include_other_unknown_sex", False))

    if not input_path.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_path}")
    if not isinstance(balance_vars, list) or not balance_vars:
        raise ValueError("config.subset_balance_vars must be a non-empty list")

    print(f"Loading full dataset from {input_path}...")
    df = pl.read_parquet(str(input_path))

    if "person_id" not in df.columns:
        raise ValueError("Input dataset must contain 'person_id'.")

    print(f"Creating balanced subset of size ~{subset_size} persons...")

    has_sleep_date = "sleep_date" in df.columns
    if has_sleep_date:
        df = df.with_columns(pl.col("sleep_date").cast(pl.Date).alias("sleep_date_std"))

    if "zip3" in df.columns:
        zip_source = "zip3"
    elif "zip_code" in df.columns:
        zip_source = "zip_code"
    else:
        zip_source = None
        print("Warning: zip3/zip_code not found; dst_observed will be set to 'Unknown'.")

    agg_exprs: list[pl.Expr] = []
    if "sex_concept" in df.columns:
        agg_exprs.append(pl.col("sex_concept").drop_nulls().first().alias("sex_raw"))
    else:
        agg_exprs.append(pl.lit(None).alias("sex_raw"))

    if "age_at_sleep" in df.columns:
        agg_exprs.append(pl.col("age_at_sleep").mean().alias("age_raw"))
    else:
        agg_exprs.append(pl.lit(None).alias("age_raw"))

    if "race" in df.columns:
        agg_exprs.append(pl.col("race").drop_nulls().first().alias("race_raw"))
    else:
        agg_exprs.append(pl.lit(None).alias("race_raw"))

    if zip_source is not None:
        agg_exprs.append(pl.col(zip_source).drop_nulls().first().alias("zip_raw"))
    else:
        agg_exprs.append(pl.lit(None).alias("zip_raw"))

    if has_sleep_date:
        agg_exprs.extend(
            [
                pl.col("sleep_date_std").n_unique().alias("nights_recorded"),
                (
                    (pl.col("sleep_date_std").max().cast(pl.Int64) - pl.col("sleep_date_std").min().cast(pl.Int64))
                    + 1
                ).cast(pl.Float64).alias("recording_span_days"),
            ]
        )
    else:
        agg_exprs.extend([
            pl.lit(None).alias("nights_recorded"),
            pl.lit(None).alias("recording_span_days"),
        ])

    person_df = df.group_by("person_id").agg(agg_exprs)

    person_df = person_df.with_columns(
        [
            pl.col("sex_raw").map_elements(to_sex_binary, return_dtype=pl.Utf8).alias("sex_binary"),
            pl.col("age_raw").map_elements(to_age_bin, return_dtype=pl.Utf8).alias("age_bin"),
            pl.col("race_raw").map_elements(to_race_collapsed, return_dtype=pl.Utf8).alias("race_collapsed"),
            pl.col("zip_raw").map_elements(normalize_zip3, return_dtype=pl.Utf8).alias("zip3_norm"),
        ]
    )

    person_df = person_df.with_columns(
        [
            pl.when(pl.col("zip3_norm").is_null())
            .then(pl.lit("Unknown"))
            .when(pl.col("zip3_norm").is_in(NO_DST_ZIP3))
            .then(pl.lit("NoDST"))
            .otherwise(pl.lit("DST"))
            .alias("dst_observed"),
            pl.col("recording_span_days").map_elements(to_span_bin, return_dtype=pl.Utf8).alias("recording_span_bin"),
            pl.col("nights_recorded").map_elements(to_nights_bin, return_dtype=pl.Utf8).alias("nights_bin"),
        ]
    )

    person_df = person_df.with_columns(
        pl.col("recording_span_bin").map_elements(to_longitudinal_weight, return_dtype=pl.Float64).alias("longitudinal_weight")
    )

    if not include_other_unknown_sex:
        n_before = person_df.height
        person_df = person_df.filter(pl.col("sex_binary").is_in(["Female", "Male"]))
        n_removed = n_before - person_df.height
        print(f"Filtered out {n_removed} participants with sex_binary == 'Other/Unknown'.")

    requested_vars = [normalize_balance_var(str(v)) for v in balance_vars]
    missing = [v for v in requested_vars if v not in person_df.columns]
    if missing:
        raise ValueError(f"Missing balance variables after derivation: {', '.join(missing)}")

    print(f"Balancing over: {', '.join(requested_vars)}")

    if has_sleep_date:
        coverage_counts = (
            person_df.group_by(["recording_span_bin", "nights_bin"]).len(name="n_people").sort("n_people", descending=True)
        )
        print("Coverage summary (person-level):")
        print(coverage_counts)

    group_counts = person_df.group_by(requested_vars).len(name="n")
    num_groups = max(group_counts.height, 1)
    target_per_group = int(math.ceil(subset_size / num_groups))

    sampled_person_df = weighted_group_sample(person_df, requested_vars, target_per_group, subset_seed)
    sampled_person_ids = sampled_person_df.select("person_id").unique()

    print(f"Selected {sampled_person_df.height} unique persons.")

    person_profile_path = run_dir / "subset_person_profile.csv"
    sampled_person_df.write_csv(str(person_profile_path))

    summary_group_cols = [
        c
        for c in ["sex_binary", "age_bin", "race_collapsed", "dst_observed", "recording_span_bin", "nights_bin"]
        if c in sampled_person_df.columns
    ]

    subset_group_summary = (
        sampled_person_df.group_by(summary_group_cols)
        .agg(
            [
                pl.len().alias("n_persons"),
                pl.col("nights_recorded").mean().alias("mean_nights_recorded"),
                pl.col("nights_recorded").median().alias("median_nights_recorded"),
                pl.col("recording_span_days").mean().alias("mean_recording_span_days"),
                pl.col("recording_span_days").median().alias("median_recording_span_days"),
            ]
        )
        .sort("n_persons", descending=True)
    )
    summary_path = run_dir / "subset_group_summary.csv"
    subset_group_summary.write_csv(str(summary_path))

    subset_balance_summary = (
        sampled_person_df.group_by(requested_vars)
        .len(name="n_persons")
        .sort("n_persons", descending=True)
    )
    balance_summary_path = run_dir / "subset_balance_strata_counts.csv"
    subset_balance_summary.write_csv(str(balance_summary_path))

    print(f"Wrote person-level subset profile to {person_profile_path}")
    print(f"Wrote grouped subset summary to {summary_path}")
    print(f"Wrote balance-strata counts to {balance_summary_path}")

    subset_df = df.join(sampled_person_ids, on="person_id", how="semi")

    output_path = run_dir / "balanced_subset.parquet"
    print(f"Writing subset to {output_path}...")
    subset_df.write_parquet(str(output_path))
    print("Subset creation complete.")


if __name__ == "__main__":
    main()
