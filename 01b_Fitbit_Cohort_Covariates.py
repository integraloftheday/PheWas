#!/usr/bin/env python
# coding: utf-8

# # Fitbit Cohort Covariate Database Builder
# This notebook builds a covariate database for the Fitbit cohort (defined by presence in `sleep_daily_summary`).
# It removes all EHR clinical data (PheWAS, conditions, etc.) and focuses on:
# - Demographics
# - SES (Zip3)
# - Survey Covariates
# - Anthropometrics (BMI, Height, Weight)
# 
# **Output**:
# - `processed_data/fitbit_cohort_covariates.parquet`

# In[ ]:


get_ipython().system('pip install polars')


# In[ ]:


# Environment and package setup
import os
import polars as pl
from google.cloud import bigquery
from pathlib import Path

GOOGLE_PROJECT = os.getenv("GOOGLE_PROJECT")
WORKSPACE_CDR = os.getenv("WORKSPACE_CDR")

assert GOOGLE_PROJECT, "GOOGLE_PROJECT env var is required"
assert WORKSPACE_CDR, "WORKSPACE_CDR env var is required"

dataset = WORKSPACE_CDR

# BigQuery helpers
def sql_with_dataset(sql: str) -> str:
    return sql.format(dataset=WORKSPACE_CDR)

def bq_to_polars(SQL_QUERY: str) -> pl.DataFrame:
    client = bigquery.Client()
    query_job = client.query(SQL_QUERY)
    return pl.from_arrow(query_job.result().to_arrow())


# ## Fitbit Cohort Definition
# Define the cohort as anyone present in the `sleep_daily_summary` table.

# In[ ]:


cohort_query = sql_with_dataset(
    """
    WITH cohort AS (
        SELECT DISTINCT person_id 
        FROM `{dataset}.sleep_daily_summary`
    )
    SELECT * FROM cohort
    """
)

cohort = bq_to_polars(cohort_query)
print(f"Fitbit Cohort Size: {cohort.shape[0]}")


# ## Demographics and SES

# In[ ]:


# Demographics
dem_query = sql_with_dataset(
    """
    SELECT 
        p.person_id,
        p.birth_datetime AS date_of_birth,
        rc.concept_name AS race,
        gc.concept_name AS gender,
        COALESCE(sab.concept_name, 'Unknown') AS sex_concept
    FROM `{dataset}.person` p
    LEFT JOIN `{dataset}.concept` rc ON p.race_concept_id = rc.concept_id
    LEFT JOIN `{dataset}.concept` gc ON p.gender_concept_id = gc.concept_id
    LEFT JOIN `{dataset}.concept` sab ON p.sex_at_birth_concept_id = sab.concept_id
    """
)
dem = bq_to_polars(dem_query)

# SES and ZIP3
ses_query = sql_with_dataset(
    """
    SELECT DISTINCT z.* EXCEPT(zip3, zip3_as_string), z.zip3 AS zip3
    FROM `{dataset}.zip3_ses_map` z
    """
)
participant_zip_query = sql_with_dataset(
    """
    SELECT c.person_id, CAST(SUBSTR(TRIM(o.value_as_string), 1, 3) AS INT64) AS zip3
    FROM `{dataset}.observation` o
    JOIN `{dataset}.sleep_daily_summary` c ON o.person_id = c.person_id -- Filter to relevant people implicitly or join later
    WHERE o.observation_source_concept_id = 1585250
    GROUP BY c.person_id, zip3
    """
)

ses_df = bq_to_polars(ses_query)
participant_zip_df = bq_to_polars(participant_zip_query)
socio_eco_df = participant_zip_df.join(ses_df, on="zip3", how="left")


# ## Survey Covariates

# In[ ]:


codes = [1585857, 1585860, 1585940, 1586198, 1586201, 1585789, 1585952]
codes_str = ', '.join(map(str, codes))

survey_query = sql_with_dataset("""
SELECT s.person_id,
       MAX(IF(s.question_concept_id = 1585857, s.answer, NULL)) AS cigs_100,
       MAX(IF(s.question_concept_id = 1585860, s.answer, NULL)) AS smoke_freq,
       MAX(IF(s.question_concept_id = 1585940, s.answer, NULL)) AS education,
       MAX(IF(s.question_concept_id = 1586198, s.answer, NULL)) AS alcohol,
       MAX(IF(s.question_concept_id = 1586201, s.answer, NULL)) AS alcohol_freq,
       MAX(IF(s.question_concept_id = 1585789, s.answer, NULL)) AS menstral_stop_reason,
       MAX(IF(s.question_concept_id = 1585952, s.answer, NULL)) AS employment_status
FROM `{{dataset}}.ds_survey` s
WHERE s.question_concept_id IN ({codes})
GROUP BY s.person_id
""".format(codes=codes_str))

survey = bq_to_polars(survey_query)

survey = survey.select([
    pl.col("person_id"),
    pl.col("cigs_100").alias("smoking"),
    pl.col("education"),
    pl.col("alcohol"),
    pl.col("alcohol_freq"),
    pl.when(pl.col("menstral_stop_reason").is_in([
        'Menstrual Stopped Reason: Endometrial Ablation',
        'Menstrual Stopped Reason: Medication Therapy',
        'Menstrual Stopped Reason: Natural Menopause',
        'Menstrual Stopped Reason: Surgery'
    ]))
    .then(pl.col("menstral_stop_reason").str.replace("Menstrual Stopped Reason: ", ""))
    .otherwise(pl.lit(None))
    .alias("menstral_stop_reason"),

    pl.when(pl.col("employment_status").is_in([
        'Employment Status: Employed For Wages',
        'Employment Status: Homemaker',
        'Employment Status: Out Of Work Less Than One',
        'Employment Status: Out Of Work One Or More',
        'Employment Status: Retired',
        'Employment Status: Self Employed',
        'Employment Status: Student',
        'Employment Status: Unable To Work'
    ]))
    .then(pl.col("employment_status").str.replace("Employment Status: ", ""))
    .otherwise(pl.lit(None))
    .alias("employment_status"),
])


# ## BMI and Anthropometrics

# In[ ]:


bmi_query = sql_with_dataset(
    """
    SELECT m.person_id, m.measurement_date, m.value_as_number AS bmi
    FROM `{dataset}.measurement` m
    WHERE m.measurement_concept_id = 3038553
    """
)
bmi2_query = sql_with_dataset(
    """
    SELECT m.person_id, m.measurement_date, m.value_as_number AS bmi
    FROM `{dataset}.measurement` m
    WHERE m.measurement_concept_id = 4245997
    """
)
height_query = sql_with_dataset(
    """
    SELECT m.person_id, m.measurement_date, m.value_as_number AS height
    FROM `{dataset}.measurement` m
    JOIN `{dataset}.concept` c ON m.measurement_concept_id = c.concept_id
    WHERE c.concept_id IN (3036277, 42527659, 3023540, 3019171)
    """
)
weight_query = sql_with_dataset(
    """
    SELECT m.person_id, m.measurement_date, m.value_as_number AS weight
    FROM `{dataset}.measurement` m
    JOIN `{dataset}.concept` c ON m.measurement_concept_id = c.concept_id
    WHERE c.concept_id IN (3025315, 3013762)
    """
)

bmi_df = bq_to_polars(bmi_query)
bmi2_df = bq_to_polars(bmi2_query)
height_df = bq_to_polars(height_query)
weight_df = bq_to_polars(weight_query)

# Filter sensible ranges
bmi_df = bmi_df.filter((pl.col("bmi") > 10) & (pl.col("bmi") < 200))
bmi2_df = bmi2_df.filter((pl.col("bmi") > 10) & (pl.col("bmi") < 200))

# Calculate BMI from Height/Weight
hw = height_df.join(weight_df, on=["person_id", "measurement_date"], how="inner")
hw = hw.with_columns((pl.col("weight") / (pl.col("height") * 0.01) ** 2).alias("bmi")).filter(pl.col("bmi") < 200).select(["person_id", "measurement_date", "bmi"])

bmi_all = pl.concat([bmi_df, bmi2_df, hw], how="diagonal")

# Get latest BMI
bmi_latest = (
    bmi_all
      .sort(["person_id", "measurement_date"], descending=[False, True])
      .group_by("person_id")
      .agg([pl.first("bmi").alias("bmi")])
)


# ## Join and Save
# Join all covariates to the Fitbit cohort and save.

# In[ ]:


final_df = cohort.join(dem, on="person_id", how="left")
final_df = final_df.join(socio_eco_df, on="person_id", how="left")
final_df = final_df.join(survey, on="person_id", how="left")
final_df = final_df.join(bmi_latest, on="person_id", how="left")

print(f"Final Shape: {final_df.shape}")
print(final_df.head())



# In[ ]:


final_df.head()


# In[ ]:


output_path = Path("processed_data/fitbit_cohort_covariates.parquet")
output_path.parent.mkdir(parents=True, exist_ok=True)
final_df.write_parquet(output_path)
print(f"Saved to {output_path}")


# In[ ]:


final_df.shape


# In[ ]:




