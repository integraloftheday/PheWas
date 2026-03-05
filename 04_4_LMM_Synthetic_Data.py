#!/usr/bin/env python
# coding: utf-8

"""
Create synthetic input data for 04_5_LLM_Regression.r and downstream scripts.

Self-contained workflow:
1) Optionally profile an existing parquet to capture schema/ranges/levels.
2) Generate a synthetic parquet compatible with processed_data/LMM_analysis.parquet.

Run:
  python 04_4_LMM_Synthetic_Data.py
"""

import json
import os
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

import numpy as np
import polars as pl


REQUIRED_COLS = [
    "person_id",
    "onset_linear",
    "offset_linear",
    "midpoint_linear",
    "daily_sleep_window_mins",
    "age_at_sleep",
    "sex_concept",
    "employment_status",
    "month",
    "is_weekend",
    "zip3",
]

DEFAULT_NO_DST_ZIP3 = sorted(
    set([f"{z:03d}" for z in range(850, 866)] + ["967", "968", "006", "007", "008", "009", "969"])
)

# -------------------------------------------------------------------
# Configuration (edit these values directly; no CLI arguments needed)
# -------------------------------------------------------------------
INPUT_PARQUET = "processed_data/LMM_analysis.parquet"
PROFILE_REAL_DATA = True
PROFILE_OUT = "processed_data/synthetic/LMM_analysis_profile.json"

# Fast defaults for quick model iteration.
OUTPUT_SYNTHETIC = "processed_data/synthetic/LMM_analysis_synthetic_small.parquet"
N_PEOPLE = 120
MIN_NIGHTS = 10
MAX_NIGHTS = 25
SEED = 20260304


@dataclass
class SyntheticConfig:
    n_people: int
    min_nights: int
    max_nights: int
    seed: int


def _jsonable_value(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (np.floating, float)):
        if not np.isfinite(v):
            return None
        return float(v)
    if isinstance(v, (np.integer, int)):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    return str(v)


def profile_parquet(input_path: str, profile_out: str) -> dict:
    df = pl.read_parquet(input_path)
    profile: dict[str, Any] = {
        "file": input_path,
        "row_count": int(df.height),
        "column_count": int(df.width),
        "required_columns_present": [c for c in REQUIRED_COLS if c in df.columns],
        "required_columns_missing": [c for c in REQUIRED_COLS if c not in df.columns],
        "columns": {},
    }

    for c in df.columns:
        s = df[c]
        info: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": int(s.null_count()),
            "n_unique": int(s.n_unique()),
        }
        if s.dtype.is_numeric():
            info["min"] = _jsonable_value(s.min())
            info["max"] = _jsonable_value(s.max())
            info["mean"] = _jsonable_value(s.mean())
            info["std"] = _jsonable_value(s.std())
        elif s.dtype == pl.Boolean:
            vals = s.drop_nulls().unique().to_list()
            info["values"] = [_jsonable_value(v) for v in vals]
        else:
            vals = s.drop_nulls().unique().head(20).to_list()
            info["sample_values"] = [_jsonable_value(v) for v in vals]
        profile["columns"][c] = info

    os.makedirs(os.path.dirname(profile_out) or ".", exist_ok=True)
    with open(profile_out, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    return profile


def _pick_with_fallback(rng: np.random.Generator, vals: list[str], probs: list[float], fallback: str) -> str:
    if not vals:
        return fallback
    p = np.array(probs, dtype=float)
    p = p / p.sum()
    return str(rng.choice(vals, p=p))


def generate_synthetic_lmm(cfg: SyntheticConfig) -> pl.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    # Employment categories aligned to 01b_Fitbit_Cohort_Covariates.py normalization.
    employment_levels = [
        "Employed For Wages",
        "Self Employed",
        "Student",
        "Homemaker",
        "Retired",
        "Unable To Work",
        "Out Of Work Less Than One",
        "Out Of Work One Or More",
    ]
    employment_probs = [0.42, 0.10, 0.08, 0.07, 0.18, 0.06, 0.05, 0.04]

    # Reflect observed AoU-style sex categories from your profile.
    sex_levels = [
        "Female",
        "Male",
        "PMI: Skip",
        "I prefer not to answer",
        "Sex At Birth: Sex At Birth None Of These",
        "No matching concept",
        "Intersex",
    ]
    sex_probs = [0.50, 0.47, 0.005, 0.01, 0.005, 0.007, 0.003]

    dst_zip_pool = [f"{z:03d}" for z in range(100, 999) if f"{z:03d}" not in DEFAULT_NO_DST_ZIP3]
    no_dst_zip_pool = DEFAULT_NO_DST_ZIP3

    rows: list[dict[str, Any]] = []
    start_date = date(2022, 1, 1)

    for i in range(cfg.n_people):
        person_id = f"SYN_{i + 1:06d}"
        n_nights = int(rng.integers(cfg.min_nights, cfg.max_nights + 1))

        # Match wide observed span while keeping most participants in adult range.
        base_age = float(np.clip(rng.normal(53, 15), 13, 95))
        sex = _pick_with_fallback(rng, sex_levels, sex_probs, "Female")
        employment = _pick_with_fallback(rng, employment_levels, employment_probs, "Employed For Wages")

        # person-level random effect in hours
        person_re = float(rng.normal(0, 0.8))
        base_onset = float(rng.normal(-0.4, 0.9)) + person_re * 0.2
        base_duration = float(np.clip(rng.normal(7.2, 0.7), 4.5, 10.5)) + person_re * 0.15

        # 15% participants in NoDST geographies to guarantee DST level variation.
        if rng.uniform() < 0.15:
            zip3 = str(rng.choice(no_dst_zip_pool))
        else:
            zip3 = str(rng.choice(dst_zip_pool))

        # Sparse missing zip3 values to test DST filtering path.
        if rng.uniform() < 0.02:
            zip3 = None

        person_start = start_date + timedelta(days=int(rng.integers(0, 365)))

        for d in range(n_nights):
            this_date = person_start + timedelta(days=d)
            month = this_date.strftime("%m")
            is_weekend = this_date.weekday() >= 5
            age_at_sleep = base_age + (d / 365.25)

            weekend_onset_shift = 0.55 if is_weekend else 0.0
            weekend_duration_shift = 0.30 if is_weekend else 0.0

            emp_shift_map = {
                "Student": 0.50,
                "Retired": -0.15,
                "Out Of Work Less Than One": 0.30,
                "Out Of Work One Or More": 0.40,
                "Unable To Work": 0.35,
                "Homemaker": 0.10,
                "Self Employed": 0.05,
                "Employed For Wages": -0.05,
            }
            emp_dur_map = {
                "Student": 0.20,
                "Retired": 0.25,
                "Out Of Work Less Than One": 0.05,
                "Out Of Work One Or More": 0.05,
                "Unable To Work": 0.10,
                "Homemaker": 0.10,
                "Self Employed": 0.00,
                "Employed For Wages": -0.05,
            }

            sex_shift_map = {"Female": 0.00, "Male": -0.10, "Other": 0.05}
            sex_dur_map = {"Female": 0.00, "Male": 0.05, "Other": 0.00}

            age_onset_effect = 0.012 * (age_at_sleep - 50.0)
            age_dur_effect = -0.003 * (age_at_sleep - 50.0)

            onset = (
                base_onset
                + weekend_onset_shift
                + emp_shift_map.get(employment, 0.0)
                + sex_shift_map.get(sex, 0.0)
                + age_onset_effect
                + float(rng.normal(0, 0.55))
            )
            onset = float(np.clip(onset, -8.0, 11.5))

            duration_hours = (
                base_duration
                + weekend_duration_shift
                + emp_dur_map.get(employment, 0.0)
                + sex_dur_map.get(sex, 0.0)
                + age_dur_effect
                + float(rng.normal(0, 0.45))
            )
            duration_hours = float(np.clip(duration_hours, 3.5, 12.0))

            offset = onset + duration_hours + float(rng.normal(0, 0.20))
            midpoint = onset + duration_hours / 2 + float(rng.normal(0, 0.15))
            sleep_window_mins = int(np.clip(np.round(duration_hours * 60 + rng.normal(0, 18)), 180, 929))

            rows.append(
                {
                    "person_id": person_id,
                    "onset_linear": onset,
                    "offset_linear": float(offset),
                    "midpoint_linear": float(midpoint),
                    "daily_sleep_window_mins": sleep_window_mins,
                    "age_at_sleep": float(age_at_sleep),
                    "sex_concept": sex,
                    "employment_status": employment if rng.uniform() > 0.005 else None,
                    "month": month,
                    "is_weekend": bool(is_weekend),
                    "zip3": zip3,
                }
            )

    df = pl.from_dicts(rows).with_columns(
        pl.col("person_id").cast(pl.Utf8),
        pl.col("onset_linear").cast(pl.Float64),
        pl.col("offset_linear").cast(pl.Float64),
        pl.col("midpoint_linear").cast(pl.Float64),
        pl.col("daily_sleep_window_mins").cast(pl.Int64),
        pl.col("age_at_sleep").cast(pl.Float64),
        pl.col("sex_concept").cast(pl.Utf8),
        pl.col("employment_status").cast(pl.Utf8),
        pl.col("month").cast(pl.Utf8),
        pl.col("is_weekend").cast(pl.Boolean),
        pl.col("zip3").cast(pl.Utf8),
    )
    return df


def main() -> None:
    if MIN_NIGHTS < 3:
        raise ValueError("MIN_NIGHTS must be >= 3")
    if MAX_NIGHTS < MIN_NIGHTS:
        raise ValueError("MAX_NIGHTS must be >= MIN_NIGHTS")
    if N_PEOPLE < 10:
        raise ValueError("N_PEOPLE must be >= 10 for stable mixed-model fitting")

    if PROFILE_REAL_DATA and os.path.exists(INPUT_PARQUET):
        profile = profile_parquet(INPUT_PARQUET, PROFILE_OUT)
        print(f"[profile] wrote: {PROFILE_OUT}")
        print(f"[profile] rows={profile['row_count']} cols={profile['column_count']}")
        if profile["required_columns_missing"]:
            print(f"[profile] missing required columns: {profile['required_columns_missing']}")
        else:
            print("[profile] all required 04_5 columns are present")
    else:
        if not PROFILE_REAL_DATA:
            print("[profile] skipped: PROFILE_REAL_DATA=False")
        else:
            print(f"[profile] skipped: input file not found at {INPUT_PARQUET}")

    cfg = SyntheticConfig(
        n_people=N_PEOPLE,
        min_nights=MIN_NIGHTS,
        max_nights=MAX_NIGHTS,
        seed=SEED,
    )
    syn = generate_synthetic_lmm(cfg)

    os.makedirs(os.path.dirname(OUTPUT_SYNTHETIC) or ".", exist_ok=True)
    syn.write_parquet(OUTPUT_SYNTHETIC)
    print(f"[synthetic] wrote: {OUTPUT_SYNTHETIC}")
    print(f"[synthetic] rows={syn.height} people={syn.select(pl.col('person_id').n_unique()).item()}")
    print(f"[synthetic] columns={syn.columns}")


if __name__ == "__main__":
    main()
