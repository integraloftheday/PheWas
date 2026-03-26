import polars as pl
import numpy as np
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
INPUT_SCHEMA_FILE = "processed_data/dataset_metadata.json"
OUTPUT_DIR = "processed_data"
PROFILE_FILE = "processed_data/synthetic/ready_for_analysis_profile.json"

rng = np.random.default_rng(20260325)


def _load_zip3_pool() -> list[str]:
    """Load a realistic ZIP3 pool, preferring environmental ZIP3 coverage."""
    default_pool = [f"{i:03d}" for i in range(1000)]

    env_root = Path("environmental_data")
    weather_glob = env_root / "prism_weather_daily" / "zip3_weather_*.parquet"
    photo_file = env_root / "photo_info" / "all_photo_info.parquet"

    zip_sets = []

    try:
        weather_files = sorted(Path().glob(str(weather_glob)))
        if weather_files:
            w = (
                pl.scan_parquet([str(p) for p in weather_files])
                .select(
                    pl.col("ZIP3")
                    .cast(pl.Utf8)
                    .str.replace_all(r"[^0-9]", "")
                    .str.slice(0, 3)
                    .alias("zip3")
                )
                .drop_nulls()
                .filter(pl.col("zip3").str.len_chars() == 3)
                .unique()
                .collect()["zip3"]
                .to_list()
            )
            zip_sets.append(set(w))
    except Exception:
        pass

    try:
        if photo_file.exists():
            p = (
                pl.scan_parquet(str(photo_file))
                .select(
                    pl.col("ZIP3")
                    .cast(pl.Utf8)
                    .str.replace_all(r"[^0-9]", "")
                    .str.slice(0, 3)
                    .alias("zip3")
                )
                .drop_nulls()
                .filter(pl.col("zip3").str.len_chars() == 3)
                .unique()
                .collect()["zip3"]
                .to_list()
            )
            zip_sets.append(set(p))
    except Exception:
        pass

    if zip_sets:
        intersection = set.intersection(*zip_sets) if len(zip_sets) > 1 else zip_sets[0]
        if intersection:
            return sorted(intersection)

        union = set.union(*zip_sets)
        if union:
            return sorted(union)

    return default_pool


ZIP3_POOL = _load_zip3_pool()


def _sample_from_value_distribution(col_name: str, profile_meta: dict, n_rows: int):
    if not profile_meta:
        return None

    dist = profile_meta.get("value_distribution") or []
    vals = []
    probs = []
    for item in dist:
        if not item:
            continue
        v = item.get("value")
        p = float(item.get("prob", 0.0))
        if v is None or p <= 0:
            continue
        vals.append(v)
        probs.append(p)

    if not vals:
        return None

    p = np.array(probs, dtype=float)
    p = p / p.sum()
    sampled = rng.choice(vals, size=n_rows, p=p)
    return pl.Series(col_name, sampled)


def _sample_numeric_quantiles(col_name: str, profile_meta: dict, n_rows: int):
    if not profile_meta:
        return None

    quantiles = profile_meta.get("quantiles") or {}
    points = sorted((float(k), float(v)) for k, v in quantiles.items() if v is not None)
    if len(points) < 2:
        return None

    q_probs = np.array([q for q, _ in points], dtype=float)
    q_vals = np.array([v for _, v in points], dtype=float)
    u = rng.uniform(0.01, 0.99, size=n_rows)
    sampled = np.interp(u, q_probs, q_vals)
    return pl.Series(col_name, sampled)

def generate_series(col_name, meta, n_rows, profile_meta=None):
    """Generates a Polars Series based on metadata specs."""
    dtype_str = meta["type"]

    series = None

    if dtype_str == "int":
        q_series = _sample_numeric_quantiles(col_name, profile_meta or {}, n_rows)
        if q_series is not None:
            series = q_series.cast(pl.Int64)
        else:
            min_val = int(meta.get("min", 0))
            max_val = int(meta.get("max", 100))
            if min_val >= max_val:
                max_val = min_val + 1
            data = rng.integers(min_val, max_val, size=n_rows)
            series = pl.Series(col_name, data, dtype=pl.Int64)

    elif dtype_str == "float":
        q_series = _sample_numeric_quantiles(col_name, profile_meta or {}, n_rows)
        if q_series is not None:
            series = q_series.cast(pl.Float64)
        else:
            min_val = meta.get("min", 0.0)
            max_val = meta.get("max", 1.0)
            safe_max = 1e12
            safe_min = -1e12

            if not np.isfinite(min_val):
                min_val = safe_min
            if not np.isfinite(max_val):
                max_val = safe_max

            if min_val > max_val:
                min_val, max_val = max_val, min_val
            if min_val == max_val:
                max_val += 1.0

            data = rng.uniform(min_val, max_val, size=n_rows)
            series = pl.Series(col_name, data, dtype=pl.Float64)

    elif dtype_str == "boolean":
        sampled = _sample_from_value_distribution(col_name, profile_meta or {}, n_rows)
        if sampled is not None:
            series = sampled.cast(pl.Boolean)
        else:
            data = rng.choice([True, False], size=n_rows)
            series = pl.Series(col_name, data, dtype=pl.Boolean)

    elif dtype_str == "string":
        name_lower = col_name.lower()

        if "zip" in name_lower:
            sampled = _sample_from_value_distribution(col_name, profile_meta or {}, n_rows)
            if sampled is not None:
                series = (
                    sampled.cast(pl.Utf8)
                    .str.replace_all(r"[^0-9]", "")
                    .str.slice(0, 3)
                    .str.zfill(3)
                )
            else:
                data = rng.choice(ZIP3_POOL, size=n_rows)
                series = pl.Series(col_name, data, dtype=pl.Utf8)
        elif meta.get("is_categorical", False):
            sampled = _sample_from_value_distribution(col_name, profile_meta or {}, n_rows)
            if sampled is not None:
                series = sampled.cast(pl.Utf8)
            else:
                categories = meta.get("categories") or []
                clean_cats = [c for c in categories if c is not None]
                if not clean_cats:
                    clean_cats = ["Unknown"]
                data = rng.choice(clean_cats, size=n_rows)
                series = pl.Series(col_name, data, dtype=pl.Utf8)
        else:
            sampled = _sample_from_value_distribution(col_name, profile_meta or {}, n_rows)
            if sampled is not None:
                series = sampled.cast(pl.Utf8)
            elif "id" in name_lower:
                data = np.array([f"SYN_{i:08d}" for i in range(n_rows)], dtype=object)
                series = pl.Series(col_name, data, dtype=pl.Utf8)
            else:
                series = pl.Series(col_name, ["Unknown"] * n_rows, dtype=pl.Utf8)

    elif dtype_str in ["date", "datetime"]:
        fmt = "%Y-%m-%d" if dtype_str == "date" else "%Y-%m-%d %H:%M:%S"
        try:
            start_date = datetime.strptime(meta["min"], fmt) if meta.get("min") else datetime.now() - timedelta(days=365)
            end_date = datetime.strptime(meta["max"], fmt) if meta.get("max") else datetime.now()
        except Exception:
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()

        delta_seconds = int((end_date - start_date).total_seconds())
        if delta_seconds <= 0:
            delta_seconds = 86400

        random_seconds = rng.integers(0, delta_seconds + 1, size=n_rows)
        base_dates = [start_date + timedelta(seconds=int(s)) for s in random_seconds]

        if dtype_str == "date":
            base_dates = [d.date() for d in base_dates]
            series = pl.Series(col_name, base_dates, dtype=pl.Date)
        else:
            series = pl.Series(col_name, base_dates, dtype=pl.Datetime)

    else:
        series = pl.Series(col_name, [None] * n_rows, dtype=pl.Utf8)

    # 2. Apply Nulls
    null_pct = meta.get("null_percentage", 0)
    if profile_meta and isinstance(profile_meta.get("null_fraction"), (int, float)):
        null_pct = float(profile_meta["null_fraction"])
    if null_pct > 0:
        mask = rng.random(n_rows) < null_pct
        
        # We create a temporary DataFrame to run the "when/then" expression
        # and then extract the series back out. This forces evaluation.
        temp_df = pl.DataFrame({"temp": series})
        series = temp_df.select(
            pl.when(pl.Series(mask)).then(None).otherwise(pl.col("temp")).alias(col_name)
        ).to_series()

    return series

def main():
    # FIX: Changed OUTPUT_SCHEMA_FILE to INPUT_SCHEMA_FILE
    if not os.path.exists(INPUT_SCHEMA_FILE):
        print(f"Error: {INPUT_SCHEMA_FILE} not found.")
        return

    with open(INPUT_SCHEMA_FILE, "r") as f:
        full_metadata = json.load(f)

    profile = {}
    if os.path.exists(PROFILE_FILE):
        with open(PROFILE_FILE, "r") as f:
            profile = json.load(f)
        print(f"Loaded profile: {PROFILE_FILE}")
    else:
        print(f"Profile not found, falling back to metadata-only sampling: {PROFILE_FILE}")

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename, file_data in full_metadata.items():
        print(f"Generating mock data for: {filename}")
        
        n_rows = file_data["row_count"]
        columns = []
        
        for col_name, col_meta in file_data["columns"].items():
            # generate_series now guarantees a materialized Series object
            profile_meta = (profile.get("columns") or {}).get(col_name, {})
            col_series = generate_series(col_name, col_meta, n_rows, profile_meta=profile_meta)
            columns.append(col_series)
        
        df = pl.DataFrame(columns)
        
        output_path = file_data.get("filename") or os.path.join(OUTPUT_DIR, filename)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        df.write_parquet(output_path)
        print(f"Saved {output_path} ({n_rows} rows)")

if __name__ == "__main__":
    main()