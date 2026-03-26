#!/usr/bin/env python
"""
Create an aggregate-only profile from a real parquet dataset for synthetic generation.

This script exports only distribution summaries (no person-level rows).

Example:
  python support_scripts/characterize_real_data.py \
    --input processed_data/ready_for_analysis.parquet \
    --output processed_data/synthetic/ready_for_analysis_profile.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np
import polars as pl

NUMERIC_QUANTILES = [0.01, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]


def _jsonable(v: Any) -> Any:
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


def build_profile(df: pl.DataFrame, top_k: int) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": int(df.height),
        "column_count": int(df.width),
        "numeric_quantiles": NUMERIC_QUANTILES,
        "categorical_top_k": int(top_k),
        "columns": {},
    }

    for c in df.columns:
        s = df[c]
        non_null_count = int(s.len() - s.null_count())
        col: dict[str, Any] = {
            "dtype": str(s.dtype),
            "null_count": int(s.null_count()),
            "non_null_count": non_null_count,
            "null_fraction": float(s.null_count() / max(1, s.len())),
            "n_unique": int(s.n_unique()),
        }

        if s.dtype.is_numeric():
            nn = s.drop_nulls().to_numpy()
            col["min"] = _jsonable(s.min())
            col["max"] = _jsonable(s.max())
            col["mean"] = _jsonable(s.mean())
            col["std"] = _jsonable(s.std())
            q_vals = np.quantile(nn, NUMERIC_QUANTILES) if nn.size else np.array([None] * len(NUMERIC_QUANTILES))
            col["quantiles"] = {
                f"{q:.2f}": _jsonable(v) for q, v in zip(NUMERIC_QUANTILES, q_vals.tolist(), strict=False)
            }
        elif s.dtype == pl.Boolean:
            non_null = s.drop_nulls().cast(pl.Boolean)
            true_count = int(non_null.sum()) if non_null_count else 0
            false_count = int(non_null_count - true_count)
            col["value_distribution"] = [
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
                .head(top_k)
            )
            vals = vc["v"].to_list() if "v" in vc.columns else []
            cnts = vc["len"].to_list() if "len" in vc.columns else []
            col["value_distribution"] = [
                {"value": _jsonable(v), "count": int(n), "prob": float(n / max(1, non_null_count))}
                for v, n in zip(vals, cnts, strict=False)
            ]

        profile["columns"][c] = col

    return profile


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregate-only dataset profile for synthetic generation")
    parser.add_argument("--input", required=True, help="Input parquet path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--top-k", type=int, default=200, help="Top categories to keep per categorical column")
    args = parser.parse_args()

    df = pl.read_parquet(args.input)
    profile = build_profile(df, top_k=max(10, int(args.top_k)))
    profile["file"] = args.input

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)

    print(f"[profile] wrote: {args.output}")
    print(f"[profile] rows={profile['row_count']} cols={profile['column_count']}")


if __name__ == "__main__":
    main()
