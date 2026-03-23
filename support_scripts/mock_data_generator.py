import polars as pl
import numpy as np
import json
import os
from datetime import datetime, timedelta
from faker import Faker
import random

# Configuration
INPUT_SCHEMA_FILE = "processed_data/dataset_metadata.json"
OUTPUT_DIR = "processed_data"

fake = Faker()

def generate_series(col_name, meta, n_rows):
    """Generates a Polars Series based on metadata specs."""
    dtype_str = meta["type"]
    
    # 1. Generate Data
    
    data = []
    series = None
    
    if dtype_str == "int":
        min_val = int(meta.get("min", 0))
        max_val = int(meta.get("max", 100))
        
        # Numpy randint can fail if the range is too massive for the C-long type
        # or if min/max are essentially the same.
        if min_val >= max_val:
            max_val = min_val + 1
            
        data = np.random.randint(min_val, max_val, size=n_rows)
        series = pl.Series(col_name, data, dtype=pl.Int64)

    elif dtype_str == "float":
        min_val = meta.get("min", 0.0)
        max_val = meta.get("max", 1.0)
        
        # FIX: Handle Infinity/NaN which cause OverflowError in np.random.uniform
        # We clamp infinite values to a "safe large number" for mocking purposes.
        safe_max = 1e12 
        safe_min = -1e12

        if not np.isfinite(min_val):
            min_val = safe_min
        if not np.isfinite(max_val):
            max_val = safe_max
            
        # Ensure constraints are logical
        if min_val > max_val:
            min_val, max_val = max_val, min_val
        if min_val == max_val:
            max_val += 1.0 # Ensure there is a range
            
        data = np.random.uniform(min_val, max_val, size=n_rows)
        series = pl.Series(col_name, data, dtype=pl.Float64)

    elif dtype_str == "boolean":
        data = np.random.choice([True, False], size=n_rows)
        series = pl.Series(col_name, data, dtype=pl.Boolean)

    elif dtype_str == "string":
        if meta.get("is_categorical", False):
            categories = meta["categories"]
            clean_cats = [c for c in categories if c is not None]
            if not clean_cats: clean_cats = ["MockVal"]
            data = np.random.choice(clean_cats, size=n_rows)
        else:
            name_lower = col_name.lower()
            if "id" in name_lower:
                data = [str(fake.uuid4()) for _ in range(n_rows)]
            elif "name" in name_lower:
                data = [fake.name() for _ in range(n_rows)]
            elif "email" in name_lower:
                data = [fake.email() for _ in range(n_rows)]
            elif "zip" in name_lower:
                # Generate realistic 3-digit zip prefixes
                data = [f"{random.randint(0, 999):03d}" for _ in range(n_rows)]
            else:
                data = [fake.word() for _ in range(n_rows)]
        
        series = pl.Series(col_name, data, dtype=pl.Utf8)

    elif dtype_str in ["date", "datetime"]:
        fmt = "%Y-%m-%d" if dtype_str == "date" else "%Y-%m-%d %H:%M:%S"
        try:
            start_date = datetime.strptime(meta["min"], fmt) if meta.get("min") else datetime.now() - timedelta(days=365)
            end_date = datetime.strptime(meta["max"], fmt) if meta.get("max") else datetime.now()
        except:
            start_date = datetime.now() - timedelta(days=365)
            end_date = datetime.now()

        delta_seconds = int((end_date - start_date).total_seconds())
        
        # Ensure delta is positive
        if delta_seconds <= 0:
            delta_seconds = 86400 # Default to 1 day range if dates are weird

        random_seconds = np.random.randint(0, delta_seconds + 1, size=n_rows)
        base_dates = [start_date + timedelta(seconds=int(s)) for s in random_seconds]
        
        if dtype_str == "date":
            base_dates = [d.date() for d in base_dates]
            series = pl.Series(col_name, base_dates, dtype=pl.Date)
        else:
            series = pl.Series(col_name, base_dates, dtype=pl.Datetime)

    else:
        data = [None] * n_rows
        series = pl.Series(col_name, data, dtype=pl.Utf8)

    # 2. Apply Nulls
    null_pct = meta.get("null_percentage", 0)
    if null_pct > 0:
        mask = np.random.random(n_rows) < null_pct
        
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

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    for filename, file_data in full_metadata.items():
        print(f"Generating mock data for: {filename}")
        
        n_rows = file_data["row_count"]
        columns = []
        
        for col_name, col_meta in file_data["columns"].items():
            # generate_series now guarantees a materialized Series object
            col_series = generate_series(col_name, col_meta, n_rows)
            columns.append(col_series)
        
        df = pl.DataFrame(columns)
        
        output_path = os.path.join(OUTPUT_DIR, filename)
        df.write_parquet(output_path)
        print(f"Saved {output_path} ({n_rows} rows)")

if __name__ == "__main__":
    main()