#!/usr/bin/env python3
"""
03_3_DLST_Adjusted_low_memory.py

Low-memory DST event-study using mean-based DiD (no FE regression).
"""

from __future__ import annotations

from pathlib import Path

from dst_adjusted_common import run_analysis


INPUT_PARQUET = Path("processed_data/ready_for_analysis.parquet")
OUTPUT_DIR = Path("results/dst_adjusted_low_memory")


def main() -> None:
    run_analysis(
        method="low_memory",
        input_parquet=INPUT_PARQUET,
        output_dir=OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
