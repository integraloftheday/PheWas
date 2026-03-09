#!/usr/bin/env python3
"""
Validate 05_5 precompute outputs for schema drift and obviously corrupted rows.

Usage:
  python 07_Validate_Precompute_Outputs.py
  python 07_Validate_Precompute_Outputs.py results_05_5
"""

from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path


EXPECTED_PREDICTION_COLUMNS = [
    "outcome",
    "outcome_type",
    "batch",
    "model_method",
    "model_file",
    "analysis",
    "employment_status",
    "is_weekend_factor",
    "sex_concept",
    "month",
    "dst_observes",
    "age_at_sleep",
    "estimate",
    "conf.low",
    "conf.high",
]

EXPECTED_ANALYSES = {
    "employment_x_weekend",
    "weekend_main",
    "sex_main",
    "month_main",
    "month_x_weekend",
    "dst_main",
    "month_x_dst",
    "dst_x_weekend",
    "age_main",
    "age_x_employment",
    "age_x_weekend",
    "age_x_dst",
}


def is_number(value: str) -> bool:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(x)


def normalize_weekend(value: str) -> str | None:
    s = "".join(ch.lower() for ch in str(value) if ch.isalnum())
    if "weekend" in s or s in {"true", "t", "1", "yes", "y"}:
        return "Weekend"
    if "weekday" in s or s in {"false", "f", "0", "no", "n"}:
        return "Weekday"
    return None


def normalize_dst(value: str) -> str | None:
    s = "".join(ch.lower() for ch in str(value) if ch.isalnum())
    if s in {"nodst", "nost", "no", "0", "false", "f"}:
        return "NoDST"
    if s in {"dst", "yes", "y", "1", "true", "t"}:
        return "DST"
    if s.startswith("no") and "dst" in s:
        return "NoDST"
    if "dst" in s:
        return "DST"
    return None


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def validate_predictions(path: Path) -> list[str]:
    errors: list[str] = []
    columns, rows = read_csv_rows(path)

    missing = [col for col in EXPECTED_PREDICTION_COLUMNS if col not in columns]
    if missing:
      errors.append(f"{path.name}: missing columns: {', '.join(missing)}")

    counts = Counter(row.get("analysis", "") for row in rows)
    missing_analyses = sorted(a for a in EXPECTED_ANALYSES if counts[a] == 0)
    if missing_analyses:
        errors.append(f"{path.name}: missing analyses: {', '.join(missing_analyses)}")

    weekend_rows = [row for row in rows if row.get("analysis") == "weekend_main"]
    if weekend_rows:
        bad = [row.get("is_weekend_factor", "") for row in weekend_rows if normalize_weekend(row.get("is_weekend_factor", "")) is None]
        if bad:
            errors.append(f"{path.name}: weekend_main has non-weekend labels in is_weekend_factor, sample={bad[:4]}")

    dst_rows = [row for row in rows if row.get("analysis") == "dst_main"]
    if dst_rows:
        bad = [row.get("dst_observes", "") for row in dst_rows if normalize_dst(row.get("dst_observes", "")) is None]
        if bad:
            errors.append(f"{path.name}: dst_main has non-DST labels in dst_observes, sample={bad[:4]}")

    month_rows = [row for row in rows if row.get("analysis") == "month_main"]
    if month_rows:
        bad = [row.get("month", "") for row in month_rows if not str(row.get("month", "")).isdigit()]
        if bad:
            errors.append(f"{path.name}: month_main has non-numeric month values, sample={bad[:4]}")

    age_rows = [row for row in rows if row.get("analysis") == "age_main"]
    if age_rows:
        bad = [row.get("age_at_sleep", "") for row in age_rows if not is_number(row.get("age_at_sleep", ""))]
        if bad:
            errors.append(f"{path.name}: age_main has non-numeric age_at_sleep values, sample={bad[:4]}")

    bad_estimate = [row.get("estimate", "") for row in rows[: min(100, len(rows))] if not is_number(row.get("estimate", ""))]
    if bad_estimate:
        errors.append(f"{path.name}: estimate column contains non-numeric values near top of file, sample={bad_estimate[:4]}")

    return errors


def validate_table_presence(base_dir: Path, filename: str) -> list[str]:
    path = base_dir / "tables" / filename
    if not path.exists():
        return [f"missing required output: {path}"]
    if path.stat().st_size == 0:
        return [f"empty required output: {path}"]
    return []


def main() -> int:
    result_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results_05_5")
    table_dir = result_dir / "tables"
    errors: list[str] = []

    pred_path = table_dir / "predictions_all_05_5.csv"
    if not pred_path.exists():
        print(f"FAIL: missing {pred_path}")
        return 1

    errors.extend(validate_predictions(pred_path))
    errors.extend(validate_table_presence(result_dir, "model_inventory_05_5.csv"))
    errors.extend(validate_table_presence(result_dir, "derived_duration_midpoint_main_grid_05_5.csv"))
    errors.extend(validate_table_presence(result_dir, "derived_duration_midpoint_age_05_5.csv"))

    weekend_path = table_dir / "weekend_contrasts_05_5.csv"
    dst_path = table_dir / "dst_contrasts_05_5.csv"
    if not weekend_path.exists():
        errors.append(f"{weekend_path.name}: not written")
    if not dst_path.exists():
        errors.append(f"{dst_path.name}: not written")

    if errors:
        print("FAIL")
        for err in errors:
            print(f"- {err}")
        return 1

    print("PASS")
    print(f"- predictions file: {pred_path}")
    print(f"- rows validated: schema and analysis coverage look consistent")
    print(f"- weekend contrast file: {weekend_path}")
    print(f"- dst contrast file: {dst_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
