#!/usr/bin/env python
# coding: utf-8

# # Phase 0: Data Preparation & Engineering
# 
# **Notebook**: `02_Data_Preparation.ipynb`  
# **Language**: Python (Polars)
# 
# This notebook implements the data preparation phase of the Fitbit Sleep Analysis pipeline.
# 
# ## Steps:
# 1. **Load Data**: Sleep metrics and covariates.
# 2. **Merge & Clean**: Join datasets and filter invalid entries.
# 3. **Feature Engineering**: Calculate Age, Seasonality, Circular Encoding, and Social Jetlag.
# 4. **Variable Mapping**: Consolidate categories.
# 5. **Output**: Save the cleaned dataset for analysis.

# In[ ]:


import polars as pl
import numpy as np
import os
import polars as pl
import matplotlib.pyplot as plt
import numpy as np


# ## 1. Load Data

# In[ ]:


# Define paths
SLEEP_DATA_PATH = "processed_data/daily_sleep_metrics_enhanced.parquet"
COVARIATES_PATH = "processed_data/fitbit_cohort_covariates.parquet"
OUTPUT_PATH = "processed_data/ready_for_analysis.parquet"

# Load data
try:
    df_sleep = pl.scan_parquet(SLEEP_DATA_PATH)
    if os.path.exists(COVARIATES_PATH):
        df_covariates = pl.scan_parquet(COVARIATES_PATH)
    else:
        print(f"Warning: {COVARIATES_PATH} not found. Using master_covariates_only.parquet as fallback.")
        df_covariates = pl.scan_parquet("processed_data/master/master_covariates_only.parquet")
    print("Data loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading data: {e}")


# ## 2. Merge & Clean

# In[ ]:


# Join Sleep Data with Covariates
df = df_sleep.join(df_covariates, on="person_id", how="left")

# Fix for mock data where zip_code might be in df_covariates but not sleep data
# or vice-versa. We ensure it's selected properly.

# Filtering
# Drop rows with missing date_of_birth or sex_concept
df = df.filter(
    pl.col("date_of_birth").is_not_null() &
    pl.col("sex_concept").is_not_null()
)

# Optional: Filter person_total_nights >= 7 if preferred
# df = df.filter(pl.col("person_total_nights") >= 7)

# Keep specific variables
cols_to_keep = [
    "person_id", "sleep_date", "daily_midpoint_hour", "daily_start_hour", "daily_end_hour",
    "date_of_birth", "sex_concept", "race", "employment_status", "zip_code", "bmi", "menstral_stop_reason",
    "is_weekend" # Assuming this exists in sleep data or needs to be calculated
]

# Select columns if they exist, otherwise keep all for now to avoid errors during dev
# df = df.select(cols_to_keep)


# In[ ]:


#df.shape does not work on lazyFrame


# ## 3. Feature Engineering

# In[ ]:


# Age & Seasonality
# We can chain these, but keeping your structure of separate steps works fine in Lazy mode too.
df = df.with_columns([
    ((pl.col("sleep_date") - pl.col("date_of_birth")).dt.total_days() / 365.25).alias("age_at_sleep"),
    pl.col("sleep_date").dt.ordinal_day().alias("day_of_year")
])

# Circular Encoding
# Optimization: Swapped "np.sin(col)" for "col.sin()". 
# This allows Polars to execute the math in Rust without moving data back and forth to Python/NumPy.
import polars as pl
import numpy as np

df = df.with_columns([
    # --- Existing Circular Transformations ---
    (pl.col("daily_midpoint_hour") * 2 * np.pi / 24).sin().alias("midpoint_sin"),
    (pl.col("daily_midpoint_hour") * 2 * np.pi / 24).cos().alias("midpoint_cos"),
    
    (pl.col("daily_start_hour") * 2 * np.pi / 24).sin().alias("onset_sin"),
    (pl.col("daily_start_hour") * 2 * np.pi / 24).cos().alias("onset_cos"),
    
    (pl.col("daily_end_hour") * 2 * np.pi / 24).sin().alias("offset_sin"),
    (pl.col("daily_end_hour") * 2 * np.pi / 24).cos().alias("offset_cos"),

    # --- NEW: Linearized Columns (Noon-to-Noon) ---
    # Logic: If hour < 12, add 24. Otherwise keep as is.
    pl.when(pl.col("daily_start_hour") < 12)
      .then(pl.col("daily_start_hour") + 24)
      .otherwise(pl.col("daily_start_hour"))
      .alias("onset_linear"),

    pl.when(pl.col("daily_end_hour") < 12)
      .then(pl.col("daily_end_hour") + 24)
      .otherwise(pl.col("daily_end_hour"))
      .alias("offset_linear"),

    pl.when(pl.col("daily_midpoint_hour") < 12)
      .then(pl.col("daily_midpoint_hour") + 24)
      .otherwise(pl.col("daily_midpoint_hour"))
      .alias("midpoint_linear")
])


# In[ ]:


# Social Jetlag
# In Lazy mode, 'weekend_means' and 'weekday_means' become branches of the 
# execution graph rather than immediate DataFrames.
weekend_means = (
    df.filter(pl.col("is_weekend") == True)
    .group_by("person_id")
    .agg(pl.col("daily_midpoint_hour").mean().alias("mean_weekend"))
)

weekday_means = (
    df.filter(pl.col("is_weekend") == False)
    .group_by("person_id")
    .agg(pl.col("daily_midpoint_hour").mean().alias("mean_weekday"))
)

# Join the lazy branches back to the main trunk
df = df.join(weekend_means, on="person_id", how="left") \
       .join(weekday_means, on="person_id", how="left")

df = df.with_columns(
    (pl.col("mean_weekend") - pl.col("mean_weekday")).alias("SJL_raw")
)

# EXECUTE
# streaming=True is the key to saving memory. It processes data in chunks 
# rather than loading the whole result into RAM at once.
#df = df.collect(engine="streaming")


# ## 4. Variable Mapping

# In[ ]:


# Menopause: Create binary is_postmenopausal
# Example logic: if menstral_stop_reason is "Natural Menopause" or "Surgery"
df = df.with_columns(
    pl.when(pl.col("menstral_stop_reason").is_in(["Natural Menopause"]))
    .then(1)
    .otherwise(0)
    .alias("is_postmenopausal")
)


# ## 5. Output

# In[ ]:


print(f"Streaming processed data directly to {OUTPUT_PATH}...")
df.sink_parquet(OUTPUT_PATH)
print("Processing and save complete.")


# In[ ]:


df.show_graph()
# This is pretty cool


# In[ ]:


q = pl.scan_parquet("processed_data/ready_for_analysis.parquet")
print(q.columns)


# ## Make subset for LMM modeling

# In[ ]:


import polars as pl

q = pl.scan_parquet("processed_data/ready_for_analysis.parquet")

# 2. Process (Filter, Clean, Linearize & Select)
df_clean = (
    q.filter(
        pl.col("age_at_sleep").is_not_null() &
        pl.col("sex_concept").is_not_null()
    )
    .with_columns(
        month = pl.col("sleep_date").cast(pl.Date).dt.strftime("%m"),
        person_id = pl.col("person_id").cast(pl.String).cast(pl.Categorical),
        sex_concept = pl.col("sex_concept").cast(pl.String).cast(pl.Categorical),
        zip3 = (
            pl.when(pl.col("zip_code").cast(pl.Utf8).str.replace_all(r"[^0-9]", "").str.len_chars() == 0)
            .then(None)
            # Already a ZIP3 (or ZIP3 with dropped leading zeros): keep as 3-digit
            .when(pl.col("zip_code").cast(pl.Utf8).str.replace_all(r"[^0-9]", "").str.len_chars() <= 3)
            .then(
                pl.col("zip_code")
                .cast(pl.Utf8)
                .str.replace_all(r"[^0-9]", "")
                .str.zfill(3)
            )
            # Possible ZIP5 with one dropped leading zero (length 4): restore then take prefix
            .when(pl.col("zip_code").cast(pl.Utf8).str.replace_all(r"[^0-9]", "").str.len_chars() == 4)
            .then(
                pl.col("zip_code")
                .cast(pl.Utf8)
                .str.replace_all(r"[^0-9]", "")
                .str.zfill(5)
                .str.slice(0, 3)
            )
            # ZIP5 or longer: take first 3 digits
            .otherwise(
                pl.col("zip_code")
                .cast(pl.Utf8)
                .str.replace_all(r"[^0-9]", "")
                .str.slice(0, 3)
            )
        ),
        is_weekend = pl.col("is_weekend_or_holiday").cast(pl.Boolean),
        
        # --- Create Linearized Variables ---
        # Standardized Noon-to-Noon transform used across the project.
        # Logic: if clock hour < 12, add 24; otherwise keep as-is.
        # Resulting domain is approximately [12, 36).

        # 1. Onset
        onset_linear = pl.when(pl.col("daily_start_hour") < 12)
                 .then(pl.col("daily_start_hour") + 24)
                         .otherwise(pl.col("daily_start_hour")),

        # 2. Midpoint
        midpoint_linear = pl.when(pl.col("daily_midpoint_hour") < 12)
                    .then(pl.col("daily_midpoint_hour") + 24)
                            .otherwise(pl.col("daily_midpoint_hour")),
        
        # 3. Offset
        offset_linear = pl.when(pl.col("daily_end_hour") < 12)
                  .then(pl.col("daily_end_hour") + 24)
                          .otherwise(pl.col("daily_end_hour"))
    )
    .select([
        "midpoint_sin", "midpoint_cos", 
        "onset_sin", "onset_cos", 
        "offset_sin", "offset_cos", 
        "onset_linear", "offset_linear", "midpoint_linear",
        "daily_sleep_window_mins", "is_postmenopausal",
        "person_id", "sex_concept", "age_at_sleep", "month", "is_weekend", "employment_status", "zip3"
    ])
)

df_clean.sink_parquet("processed_data/LMM_analysis.parquet")


# ## Validate Linearlization

# In[ ]:


df = pl.read_parquet("processed_data/LMM_analysis.parquet")

print("--- 1. Summary Statistics for Linear Variables ---")
stats = df.select([
    pl.col("onset_linear"),
    pl.col("midpoint_linear"),
    pl.col("offset_linear")
]).describe()
print(stats)

print("\n--- 2. Logic Check: Raw vs Linear (Midnight Crossover) ---")
# To do this, we need to infer what the 'raw' value was approximately 
# (since we didn't save the raw columns in the final file, we reconstruct logic for display)

# We define a helper to reverse-engineer the display for verification
# If linear >= 24, it means raw was (linear - 24). If linear < 24, raw was same.
check_df = df.select([
    "onset_linear",
    "midpoint_linear",
    "offset_linear"
]).with_columns(
    inferred_raw_onset = pl.when(pl.col("onset_linear") >= 24)
                           .then(pl.col("onset_linear") - 24)
                           .otherwise(pl.col("onset_linear")),
    label = pl.when(pl.col("onset_linear") >= 24).then(pl.lit("Post-Midnight (shifted +24)"))
              .otherwise(pl.lit("Evening/Pre-Midnight"))
)

# Show examples of evening vs post-midnight rows in noon-to-noon encoding.
print("\nSample of Post-Midnight Onsets (Shifted to >=24):")
print(check_df.filter(pl.col("onset_linear") >= 24).head(5))

print("\nSample of Evening Onsets (<24):")
print(check_df.filter(pl.col("onset_linear") < 24).head(5))

print("\n--- 3. Range Sanity Check ---")
# Check for strange outliers for noon-to-noon representation
outliers = df.filter(
    (pl.col("onset_linear") < 12) | (pl.col("onset_linear") > 36)
)

if outliers.height > 0:
    print(f"WARNING: Found {outliers.height} rows with potentially weird linear values outside [12, 36].")
    print(outliers.select(["person_id", "onset_linear", "midpoint_linear"]).head())
else:
    print("SUCCESS: All onset_linear values are within [12, 36] for noon-to-noon encoding.")


# In[ ]:


import polars as pl
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the data
print("Loading data...")
df = pl.read_parquet("processed_data/LMM_analysis.parquet")

# 2. Subsample
# Plotting 20 million points is slow and unnecessary for checking shape.
# 500,000 points is statistically sufficient to see the distribution shape clearly.
SAMPLE_SIZE = 500_000

if df.height > SAMPLE_SIZE:
    print(f"Subsampling {SAMPLE_SIZE} rows from {df.height} total rows...")
    df_plot = df.sample(n=SAMPLE_SIZE, seed=42)
else:
    df_plot = df

# 3. Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Define the columns and their "Cutoff" points (where we wrapped the data)
# We want to check the wrap boundary near 12/36 and continuity near midnight (24).
variables = [
    {
        "col": "onset_linear", 
        "title": "Sleep Onset (Linearized)", 
        "cutoff_low": 12,
        "cutoff_high": 36
    },
    {
        "col": "midpoint_linear", 
        "title": "Sleep Midpoint (Linearized)", 
        "cutoff_low": 12,
        "cutoff_high": 36
    },
    {
        "col": "offset_linear", 
        "title": "Wake Up / Offset (Linearized)", 
        "cutoff_low": 12,
        "cutoff_high": 36
    }
]

for i, var in enumerate(variables):
    ax = axes[i]
    col_name = var["col"]
    data = df_plot[col_name].to_numpy()
    
    # Plot Histogram
    # FIXED: Changed 'slatebytes' (typo) to 'black'
    ax.hist(data, bins=100, color='skyblue', edgecolor='black', alpha=0.7)
    
    # Add Midnight Reference Line (24.0 in noon-to-noon scale)
    ax.axvline(24, color='red', linestyle='--', linewidth=1.5, label='Midnight (24.0)')
    
    # Add Cutoff Lines (Where the wrap-around logic happens)
    # Ideally, the bars should be empty/flat near these green lines.
    ax.axvline(var["cutoff_low"], color='green', linestyle=':', linewidth=2, label='Wrap Boundary')
    ax.axvline(var["cutoff_high"], color='green', linestyle=':', linewidth=2)
    
    # Formatting
    ax.set_title(var["title"], fontsize=12, fontweight='bold')
    ax.set_xlabel("Linearized Hour (Noon-to-Noon)")
    ax.set_ylabel("Frequency (Sampled)")
    
    if i == 0:
        ax.legend()

plt.suptitle("Distribution Check: Are the shapes Bell-like? (Ensure no cliffs at Green lines)", fontsize=14)
plt.tight_layout()
plt.show()


# In[ ]:


