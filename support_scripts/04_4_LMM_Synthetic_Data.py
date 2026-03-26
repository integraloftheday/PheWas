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

NUMERIC_QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
CATEGORICAL_TOP_K = 200

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
N_PEOPLE = 60
MIN_NIGHTS = 8
MAX_NIGHTS = 16
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
        "numeric_quantiles": NUMERIC_QUANTILES,
        "columns": {},
    }

    for c in df.columns:
        s = df[c]
        non_null_count = int(s.len() - s.null_count())
        info: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": int(s.null_count()),
            "non_null_count": non_null_count,
            "null_fraction": float(s.null_count() / max(1, s.len())),
            "n_unique": int(s.n_unique()),
        }
        if s.dtype.is_numeric():
            info["min"] = _jsonable_value(s.min())
            info["max"] = _jsonable_value(s.max())
            info["mean"] = _jsonable_value(s.mean())
            info["std"] = _jsonable_value(s.std())
            nn = s.drop_nulls().to_numpy()
            q_vals = np.quantile(nn, NUMERIC_QUANTILES) if nn.size else np.array([None] * len(NUMERIC_QUANTILES))
            info["quantiles"] = {
                f"{q:.2f}": _jsonable_value(v) for q, v in zip(NUMERIC_QUANTILES, q_vals.tolist(), strict=False)
            }
        elif s.dtype == pl.Boolean:
            non_null = s.drop_nulls().cast(pl.Boolean)
            true_count = int(non_null.sum()) if non_null_count else 0
            false_count = int(non_null_count - true_count)
            info["value_distribution"] = [
                {"value": True, "count": true_count, "prob": float(true_count / max(1, non_null_count))},
                {"value": False, "count": false_count, "prob": float(false_count / max(1, non_null_count))},
            ]
        else:
            vc = (
                pl.DataFrame({"v": s})
                .drop_nulls()
                .group_by("v")
                .len()
                .sort("len", descending=True)
                .head(CATEGORICAL_TOP_K)
            )
            values = vc["v"].to_list() if "v" in vc.columns else []
            counts = vc["len"].to_list() if "len" in vc.columns else []
            info["value_distribution"] = [
                {
                    "value": _jsonable_value(v),
                    "count": int(c),
                    "prob": float(c / max(1, non_null_count)),
                }
                for v, c in zip(values, counts, strict=False)
            ]
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


def _column_profile(profile: dict, col: str) -> dict[str, Any]:
    return (profile.get("columns") or {}).get(col, {})


def _distribution_from_profile(col_profile: dict[str, Any], fallback_vals: list[str], fallback_probs: list[float]) -> tuple[list[str], list[float]]:
    value_dist = col_profile.get("value_distribution") or []
    vals: list[str] = []
    probs: list[float] = []
    for rec in value_dist:
        if rec is None:
            continue
        v = rec.get("value")
        if v is None:
            continue
        vals.append(str(v))
        probs.append(float(rec.get("prob", 0.0)))

    if vals:
        p = np.array(probs, dtype=float)
        if np.isfinite(p).all() and p.sum() > 0:
            p = p / p.sum()
            return vals, p.tolist()

    return fallback_vals, fallback_probs


def _null_fraction_from_profile(col_profile: dict[str, Any], fallback: float = 0.0) -> float:
    v = col_profile.get("null_fraction")
    try:
        fv = float(v)
        return float(np.clip(fv, 0.0, 0.95))
    except Exception:
        return fallback


def _sample_numeric_from_quantiles(
    rng: np.random.Generator,
    col_profile: dict[str, Any],
    n: int,
    fallback_mean: float,
    fallback_sd: float,
    lo: float,
    hi: float,
) -> np.ndarray:
    q_map = col_profile.get("quantiles") or {}
    points = sorted((float(k), float(v)) for k, v in q_map.items() if v is not None)
    if len(points) >= 2:
        q_probs = np.array([q for q, _ in points], dtype=float)
        q_vals = np.array([v for _, v in points], dtype=float)
        u = rng.uniform(0.01, 0.99, size=n)
        arr = np.interp(u, q_probs, q_vals)
    else:
        arr = rng.normal(fallback_mean, fallback_sd, size=n)
    return np.clip(arr, lo, hi)


def generate_synthetic_lmm(cfg: SyntheticConfig) -> pl.DataFrame:
    rng = np.random.default_rng(cfg.seed)

    profile: dict[str, Any] = {}
    if os.path.exists(PROFILE_OUT):
        with open(PROFILE_OUT, "r", encoding="utf-8") as f:
            profile = json.load(f)

    # Employment categories aligned to 01b_Fitbit_Cohort_Covariates.py normalization.
    employment_levels_default = [
        "Employed For Wages",
        "Self Employed",
        "Student",
        "Homemaker",
        "Retired",
        "Unable To Work",
        "Out Of Work Less Than One",
        "Out Of Work One Or More",
    ]
    employment_probs_default = [0.42, 0.10, 0.08, 0.07, 0.18, 0.06, 0.05, 0.04]

    # Reflect observed AoU-style sex categories from your profile.
    sex_levels_default = [
        "Female",
        "Male",
        "PMI: Skip",
        "I prefer not to answer",
        "Sex At Birth: Sex At Birth None Of These",
        "No matching concept",
        "Intersex",
    ]
    sex_probs_default = [0.50, 0.47, 0.005, 0.01, 0.005, 0.007, 0.003]

    employment_levels, employment_probs = _distribution_from_profile(
        _column_profile(profile, "employment_status"),
        employment_levels_default,
        employment_probs_default,
    )
    sex_levels, sex_probs = _distribution_from_profile(
        _column_profile(profile, "sex_concept"),
        sex_levels_default,
        sex_probs_default,
    )

    zip_vals, zip_probs = _distribution_from_profile(
        _column_profile(profile, "zip3"),
        [f"{z:03d}" for z in range(100, 999)],
        [1.0 / 899.0] * 899,
    )
    month_vals, month_probs = _distribution_from_profile(
        _column_profile(profile, "month"),
        [f"{m:02d}" for m in range(1, 13)],
        [1.0 / 12.0] * 12,
    )

    zip_null_fraction = _null_fraction_from_profile(_column_profile(profile, "zip3"), fallback=0.02)
    emp_null_fraction = _null_fraction_from_profile(_column_profile(profile, "employment_status"), fallback=0.005)

    age_prof = _column_profile(profile, "age_at_sleep")
    onset_prof = _column_profile(profile, "onset_linear")
    duration_prof = _column_profile(profile, "daily_sleep_window_mins")

    dst_zip_pool = [f"{z:03d}" for z in range(100, 999) if f"{z:03d}" not in DEFAULT_NO_DST_ZIP3]
    no_dst_zip_pool = DEFAULT_NO_DST_ZIP3

    rows: list[dict[str, Any]] = []
    start_date = date(2022, 1, 1)

    for i in range(cfg.n_people):
        person_id = f"SYN_{i + 1:06d}"
        n_nights = int(rng.integers(cfg.min_nights, cfg.max_nights + 1))

        # Draw person-level latent traits from empirical marginal distributions when available.
        base_age = float(
            _sample_numeric_from_quantiles(
                rng,
                age_prof,
                n=1,
                fallback_mean=53.0,
                fallback_sd=15.0,
                lo=13.0,
                hi=95.0,
            )[0]
        )
        sex = _pick_with_fallback(rng, sex_levels, sex_probs, "Female")
        employment = _pick_with_fallback(rng, employment_levels, employment_probs, "Employed For Wages")

        # person-level random effect in hours
        person_re = float(rng.normal(0, 0.8))
        base_onset = float(
            _sample_numeric_from_quantiles(
                rng,
                onset_prof,
                n=1,
                fallback_mean=-0.4,
                fallback_sd=0.9,
                lo=-8.0,
                hi=11.5,
            )[0]
        ) + person_re * 0.2

        duration_mins = float(
            _sample_numeric_from_quantiles(
                rng,
                duration_prof,
                n=1,
                fallback_mean=430.0,
                fallback_sd=42.0,
                lo=180.0,
                hi=929.0,
            )[0]
        )
        base_duration = float(np.clip(duration_mins / 60.0, 3.5, 12.0)) + person_re * 0.15

        zip3 = _pick_with_fallback(rng, zip_vals, zip_probs, str(rng.choice(dst_zip_pool)))
        zip3_digits = "".join(ch for ch in zip3 if ch.isdigit())
        zip3 = (zip3_digits[:3] if len(zip3_digits) >= 3 else zip3.zfill(3)) if zip3 else None

        if rng.uniform() < zip_null_fraction:
            zip3 = None

        person_start = start_date + timedelta(days=int(rng.integers(0, 365)))

        for d in range(n_nights):
            this_date = person_start + timedelta(days=d)
            # Keep temporal coherence but align month marginals to observed data.
            month = _pick_with_fallback(rng, month_vals, month_probs, this_date.strftime("%m"))
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
                    "employment_status": employment if rng.uniform() > emp_null_fraction else None,
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
