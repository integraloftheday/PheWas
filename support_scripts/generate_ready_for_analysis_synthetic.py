#!/usr/bin/env python
"""
Generate a DST-compatible synthetic processed_data/ready_for_analysis.parquet.

Guarantees panel structure with repeated nights per person and both
work-day/free-day coverage for chronotype midpoint plots.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import polars as pl

OUT_PATH = Path("processed_data/ready_for_analysis.parquet")
SEED = 20260325
N_PEOPLE = 40
MIN_NIGHTS = 330
MAX_NIGHTS = 365

NO_DST_ZIP3 = [f"{i:03d}" for i in range(850, 866)] + ["967", "968", "006", "007", "008", "009", "969"]
DST_POOL = [f"{i:03d}" for i in range(100, 999) if f"{i:03d}" not in set(NO_DST_ZIP3)]


def main() -> None:
    rng = np.random.default_rng(SEED)
    rows = []
    start = date(2022, 1, 1)

    for i in range(N_PEOPLE):
        pid = f"SYN_{i+1:06d}"
        nights = int(rng.integers(MIN_NIGHTS, MAX_NIGHTS + 1))
        start_offset = int(rng.integers(0, 21))
        person_start = start + timedelta(days=start_offset)

        person_re = float(rng.normal(0, 0.8))
        base_onset = float(np.clip(rng.normal(-0.4, 0.8) + person_re * 0.2, -8, 11.5))
        base_duration = float(np.clip(rng.normal(430, 40) + person_re * 18, 180, 900))

        zip3 = str(rng.choice(NO_DST_ZIP3 if rng.uniform() < 0.15 else DST_POOL))

        for d in range(nights):
            sleep_date = person_start + timedelta(days=d)
            is_weekend = sleep_date.weekday() >= 5

            onset = float(np.clip(base_onset + (0.55 if is_weekend else 0.0) + rng.normal(0, 0.45), -8, 11.5))
            duration_mins = float(np.clip(base_duration + (25 if is_weekend else 0) + rng.normal(0, 24), 180, 929))
            offset = onset + duration_mins / 60.0 + float(rng.normal(0, 0.15))
            midpoint = onset + duration_mins / 120.0 + float(rng.normal(0, 0.1))

            rows.append(
                {
                    "person_id": pid,
                    "sleep_date": sleep_date,
                    "zip3": zip3,
                    "is_weekend_or_holiday": bool(is_weekend),
                    "is_weekend": bool(is_weekend),
                    "daily_start_hour": float(np.mod(onset, 24.0)),
                    "daily_end_hour": float(np.mod(offset, 24.0)),
                    "daily_midpoint_hour": float(np.mod(midpoint, 24.0)),
                    "onset_linear": onset,
                    "offset_linear": float(offset),
                    "midpoint_linear": float(midpoint),
                    "daily_duration_mins": float(duration_mins),
                    "daily_sleep_window_mins": int(np.round(duration_mins)),
                }
            )

    df = pl.from_dicts(rows).sort(["person_id", "sleep_date"])
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(OUT_PATH)

    overlap = (
        df.select(["person_id", "is_weekend_or_holiday"])
        .group_by("person_id")
        .agg(pl.col("is_weekend_or_holiday").n_unique().alias("n_types"))
        .select((pl.col("n_types") >= 2).sum().alias("with_both"))
        .item()
    )

    print(f"[synthetic-ready] wrote: {OUT_PATH}")
    print(f"[synthetic-ready] rows={df.height} people={df.select(pl.col('person_id').n_unique()).item()}")
    print(f"[synthetic-ready] persons_with_both_daytypes={int(overlap)}")


if __name__ == "__main__":
    main()
