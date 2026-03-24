#!/usr/bin/env python3
"""
03_3_DLST_Adjusted_twfe.py

Regression-based DST event-study using TWFE (person/date FE) with person-clustered SE.
"""

from __future__ import annotations

from pathlib import Path

from dst_adjusted_common import run_analysis


INPUT_PARQUET = Path("processed_data/ready_for_analysis.parquet")
OUTPUT_DIR = Path("results/dst_adjusted_twfe")


def main() -> None:
    run_analysis(
        method="twfe",
        input_parquet=INPUT_PARQUET,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
