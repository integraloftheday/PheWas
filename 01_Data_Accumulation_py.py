#!/usr/bin/env python
# coding: utf-8

# # PheWAS Dataset Builder (Python)
# This notebook builds the PheWAS-ready datasets from AoU EHR plus Fitbit-derived baseline coverage. Delivered outputs:
# 
# - `processed_data/master/master_phewas_wide.parquet`: person-level PheWAS wide table with `has_phe_*`, `had_phe_*`, `date_*`, demographics, SES, Fitbit baseline (`min_date`, `max_date`, `duration`), overall sleep aggregates (`average_daily_sleep`, `sd_daily_sleep`), survey covariates (including sleep patterns from questions 1585789 and 1585952), BMI, cancer, and CAD flags.
# - `processed_data/master/master_covariates_only.parquet`: same as above but WITHOUT PheWAS flags (`has_phe_*`, `had_phe_*`, `date_*`), containing only covariates for analysis.
# - `processed_data/person_ids.parquet`: unique `person_id`s represented in the final wide dataset.
# - `processed_data/master/master_summary_stats.parquet`: basic summary statistics.
# 
# Requirements:
# - Env vars: `GOOGLE_PROJECT`, `WORKSPACE_CDR`
# - BigQuery auth for the selected project
# - Python packages: `polars`, `google-cloud-bigquery`
# 
# Notes:
# - Baseline and sleep aggregates use the same naming as the R workflow for downstream compatibility.
# - `had_phe_*` flags indicate a phecode occurred within 6 months after baseline start.
# - Survey questions 1585789 and 1585952 are included as `sleep_pattern_1585789` and `sleep_pattern_1585952`.

# In[ ]:


get_ipython().system('pip install polars')


# ## Setup
# 
# Initialize environment, libraries, helpers, and required env vars.
# 

# In[ ]:


# Environment and package setup
import os
import polars as pl
from google.cloud import bigquery
import os
from pathlib import Path
import shutil

GOOGLE_PROJECT = os.getenv("GOOGLE_PROJECT")
WORKSPACE_CDR = os.getenv("WORKSPACE_CDR")

assert GOOGLE_PROJECT, "GOOGLE_PROJECT env var is required"
assert WORKSPACE_CDR, "WORKSPACE_CDR env var is required (e.g., fc-aou-XXXX.C2024Q2R3)"

dataset=WORKSPACE_CDR

# User inputs
# Path to phecode mapping CSV (second column is expected to be condition_source_value or we'll rename)
phemap_path = os.getenv("PHEMAP_PATH", "analysis_inputs/ICD_to_Phecode_mapping.csv")

# Optional: limit rows for quick validation (None for full run)
sample_limit: int | None = None

# BigQuery helpers using Arrow → Polars zero-copy

def sql_with_dataset(sql: str) -> str:
    return sql.format(dataset=WORKSPACE_CDR)


def bq_to_polars(SQL_QUERY: str) -> pl.DataFrame:
    # 1. Initialize the BigQuery client
    client = bigquery.Client()

    # 2. Run the query
    query_job = client.query(SQL_QUERY)

    # 3. Get results as an Arrow table and load into Polars (zero-copy)
    return pl.from_arrow(query_job.result().to_arrow())


def maybe_limit(sql: str) -> str:
    if sample_limit is not None:
        return sql + f"\nLIMIT {int(sample_limit)}"
    return sql


# ## EHR Cohort
# 
# Define the EHR cohort using AoU EHR-sourced tables (OMOP domains).

# In[ ]:


# EHR Cohort
ehr_query = sql_with_dataset(
    """
    WITH ehr AS (
        SELECT DISTINCT person_id
        FROM `{dataset}.measurement` m
        LEFT JOIN `{dataset}.measurement_ext` mm USING (measurement_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.condition_occurrence` m
        LEFT JOIN `{dataset}.condition_occurrence_ext` mm USING (condition_occurrence_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.device_exposure` m
        LEFT JOIN `{dataset}.device_exposure_ext` mm USING (device_exposure_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.drug_exposure` m
        LEFT JOIN `{dataset}.drug_exposure_ext` mm USING (drug_exposure_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.observation` m
        LEFT JOIN `{dataset}.observation_ext` mm USING (observation_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.procedure_occurrence` m
        LEFT JOIN `{dataset}.procedure_occurrence_ext` mm USING (procedure_occurrence_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.visit_occurrence` m
        LEFT JOIN `{dataset}.visit_occurrence_ext` mm USING (visit_occurrence_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'
    )
    SELECT DISTINCT person_id FROM ehr
    """
)

ehr_cohort = bq_to_polars(maybe_limit(ehr_query))
print(ehr_cohort.shape)


# ## Demographics and SES
# 
# Pull demographics and participant ZIP3 to join with SES metrics.

# In[ ]:


# Demographics and Sex
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

dem = bq_to_polars(maybe_limit(dem_query))
print(dem.shape)

# SES and participant ZIP3
ses_query = sql_with_dataset(
    """
    SELECT DISTINCT z.* EXCEPT(zip3, zip3_as_string), z.zip3 AS zip3
    FROM `{dataset}.zip3_ses_map` z
    """
)

participant_zip_query = sql_with_dataset(
    """
    WITH cohort AS (
        SELECT DISTINCT person_id FROM `{dataset}.activity_summary`
    )
    SELECT c.person_id, CAST(SUBSTR(TRIM(o.value_as_string), 1, 3) AS INT64) AS zip3
    FROM cohort c
    JOIN `{dataset}.observation` o ON c.person_id = o.person_id
    WHERE o.observation_source_concept_id = 1585250
    """
)

ses_df = bq_to_polars(maybe_limit(ses_query))
participant_zip_df = bq_to_polars(maybe_limit(participant_zip_query))

socio_eco_df = participant_zip_df.join(ses_df, on="zip3", how="left")
print(socio_eco_df.shape)


# ## Survey + Anthropometrics
# 
# Derive survey covariates and compute latest BMI/BP within EHR coverage.

# ## Fitbit Baseline (min/max/duration)
# 
# We compute the per-person Fitbit coverage window from `sleep_daily_summary`:
# 
# - `min_fitbit_date`: first day with valid sleep data
# 
# - `max_fitbit_date`: last day with valid sleep data
# 
# - `duration_days`: day span between min and max
# 
# 
# 
# This baseline anchors `had_*` (occurrence within 180 days from baseline start) and restricts the cohort to EHR∩Fitbit.
# 

# In[ ]:


# Fitbit baseline window

baseline_query = sql_with_dataset(

    """

    SELECT

      person_id,

      MIN(sleep_date) AS min_fitbit_date,

      MAX(sleep_date) AS max_fitbit_date,

      DATE_DIFF(MAX(sleep_date), MIN(sleep_date), DAY) AS duration_days

    FROM `{dataset}.sleep_daily_summary`

    WHERE minute_asleep > 0

    GROUP BY person_id

    """

)



baseline = bq_to_polars(maybe_limit(baseline_query))

print("baseline:", baseline.shape)


# In[ ]:


# Survey covariates

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



survey = bq_to_polars(maybe_limit(survey_query))



survey = survey.select([
    pl.col("person_id"),
    pl.col("cigs_100").alias("smoking"),
    
    # ... [Education and Alcohol logic from before] ...

    # 1. Menstrual: Transforms text, turns PMI/others into null, but KEEPS the row
    pl.when(pl.col("menstral_stop_reason").is_in([
        'Menstrual Stopped Reason: Endometrial Ablation',
        'Menstrual Stopped Reason: Medication Therapy',
        'Menstrual Stopped Reason: Natural Menopause',
        'Menstrual Stopped Reason: Surgery'
    ]))
    .then(pl.col("menstral_stop_reason").str.replace("Menstrual Stopped Reason: ", ""))
    .otherwise(pl.lit(None)) # Non-matching values become null
    .alias("menstral_stop_reason"),

    # 2. Employment: Transforms text, turns PMI/others into null, but KEEPS the row
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
    .otherwise(pl.lit(None)) # Non-matching values become null
    .alias("employment_status"),

]) 


print(survey.shape)



# Height, Weight, BMI

bmi_query = sql_with_dataset(

    """

    WITH cohort AS (

        SELECT DISTINCT person_id FROM `{dataset}.sleep_daily_summary`

    )

    SELECT m.person_id, m.measurement_date, m.value_as_number AS bmi

    FROM cohort

    JOIN `{dataset}.measurement` m USING (person_id)

    WHERE m.measurement_concept_id = 3038553

    """

)



bmi2_query = sql_with_dataset(

    """

    WITH cohort AS (

        SELECT DISTINCT person_id FROM `{dataset}.sleep_daily_summary`

    )

    SELECT m.person_id, m.measurement_date, m.value_as_number AS bmi

    FROM cohort

    JOIN `{dataset}.measurement` m USING (person_id)

    WHERE m.measurement_concept_id = 4245997

    """

)



height_query = sql_with_dataset(

    """

    WITH cohort AS (

        SELECT DISTINCT person_id FROM `{dataset}.sleep_daily_summary`

    )

    SELECT m.person_id, m.measurement_date, m.value_as_number AS height

    FROM cohort

    JOIN `{dataset}.measurement` m USING (person_id)

    JOIN `{dataset}.concept` c ON m.measurement_concept_id = c.concept_id

    WHERE c.concept_id IN (3036277, 42527659, 3023540, 3019171)

    """

)



weight_query = sql_with_dataset(

    """

    WITH cohort AS (

        SELECT DISTINCT person_id FROM `{dataset}.sleep_daily_summary`

    )

    SELECT m.person_id, m.measurement_date, m.value_as_number AS weight

    FROM cohort

    JOIN `{dataset}.measurement` m USING (person_id)

    JOIN `{dataset}.concept` c ON m.measurement_concept_id = c.concept_id

    WHERE c.concept_id IN (3025315, 3013762)

    """

)



bmi_df = bq_to_polars(maybe_limit(bmi_query))

bmi2_df = bq_to_polars(maybe_limit(bmi2_query))

height_df = bq_to_polars(maybe_limit(height_query))

weight_df = bq_to_polars(maybe_limit(weight_query))



# Filter sensible ranges and compute fallback BMI

bmi_df = bmi_df.lazy().filter((pl.col("bmi") > 10) & (pl.col("bmi") < 200)).collect()

bmi2_df = bmi2_df.lazy().filter((pl.col("bmi") > 10) & (pl.col("bmi") < 200)).collect()



hw = height_df.join(weight_df, on=["person_id", "measurement_date"], how="inner")

hw = hw.lazy().with_columns((pl.col("weight") / (pl.col("height") * 0.01) ** 2).alias("bmi")).filter(pl.col("bmi") < 200).select(["person_id", "measurement_date", "bmi"]).collect()



bmi_all = pl.concat([bmi_df, bmi2_df, hw], how="diagonal")



bmi_latest = (

    bmi_all.lazy()

      .sort(["person_id", "measurement_date"], descending=[False, True])

      .group_by("person_id")

      .agg([pl.first("bmi").alias("bmi")])

      .collect(streaming=True)

)

print(bmi_latest.shape)


# ## Cancer and CAD
# 
# Flag cancer and coronary artery disease from ICD/CPT definitions (ever).

# In[ ]:


# Cancer and CAD flags

def like_any(col: str, codes: list[str]) -> str:
    parts = [f"{col} LIKE '{code}'" for code in codes]
    return " OR ".join(parts)

# Cancer definitions
icd9_cancer = [
    '104','104.%','105','105.%','106','106.%','107','107.%','108','108.%','109','109.%','110','110.%','111','111.%','112','112.%',
    '113','113.%','114','114.%','115','115.%','116','116.%','117','117.%','118','118.%','119','119.%','120','120.%','121','121.%',
    '122','122.%','123','123.%','124','124.%','125','125.%','126','126.%','127','127.%','128','128.%','129','129.%','130','130.%',
    '131','131.%','132','132.%','133','133.%','134','134.%','135','135.%','136','136.%','137','137.%','138','138.%','139','139.%',
    '140','140.%','141','141.%','142','142.%','143','143.%','144','144.%','145','145.%','146','146.%','147','147.%','148','148.%',
    '149','149.%','150','150.%','151','151.%','152','152.%','153','153.%','154','154.%','155','155.%','156','156.%','157','157.%',
    '158','158.%','159','159.%','160','160.%','161','161.%','162','162.%','163','163.%','164','164.%','165','165.%','166','166.%',
    '167','167.%','168','168.%','169','169.%','170','170.%','171','171.%','172','172.%','173','173.%','174','174.%','175','175.%',
    '176','176.%','177','177.%','178','178.%','179','179.%','180','180.%','181','181.%','182','182.%','183','183.%','184','184.%',
    '185','185.%','186','186.%','187','187.%','188','188.%','189','189.%','190','190.%','191','191.%','192','192.%','193','193.%',
    '194','194.%','195','195.%','196','196.%','197','197.%','198','198.%','199','199.%','200','200.%','201','201.%','202','202.%',
    '203','203.%','204','204.%','205','205.%','206','206.%','207','207.%','208','208.%','209','209.%'
]
icd10_cancer = [
    'C00%','C01%','C02%','C03%','C04%','C05%','C06%','C07%','C08%','C09%','C10%','C11%','C12%','C13%','C14%','C15%','C16%','C17%',
    'C18%','C19%','C20%','C21%','C22%','C23%','C24%','C25%','C26%','C27%','C28%','C29%','C30%','C31%','C32%','C33%','C34%','C35%',
    'C36%','C37%','C38%','C39%','C40%','C41%','C42%','C43%','C44%','C45%','C46%','C47%','C48%','C49%','C50%','C51%','C52%','C53%',
    'C54%','C55%','C56%','C57%','C58%','C59%','C60%','C61%','C62%','C63%','C64%','C65%','C66%','C67%','C68%','C69%','C70%','C71%',
    'C72%','C73%','C74%','C75%','C76%','C77%','C78%','C79%','C80%','C81%','C82%','C83%','C84%','C85%','C86%','C87%','C88%','C89%',
    'C90%','C91%','C92%','C93%','C94%','C95%','C96%','D00%','D01%','D02%','D03%','D04%','D05%','D06%','D07%','D08%','D09%','D10%',
    'D11%','D12%','D13%','D14%','D15%','D16%','D17%','D18%','D19%','D20%','D21%','D22%','D23%','D24%','D25%','D26%','D27%','D28%',
    'D29%','D30%','D31%','D32%','D33%','D34%','D35%','D36%','D37%','D38%','D39%','D40%','D41%','D42%','D43%','D44%','D45%','D46%',
    'D47%','D48%','D49%'
]

cancer_query = sql_with_dataset(
    f"""
    SELECT DISTINCT co.person_id, co.condition_start_date
    FROM `{dataset}.condition_occurrence` co
    JOIN `{dataset}.concept` c ON co.condition_source_concept_id = c.concept_id
    WHERE (c.vocabulary_id = 'ICD9CM' AND ({like_any('co.condition_source_value', icd9_cancer)}))
       OR (c.vocabulary_id = 'ICD10CM' AND ({like_any('co.condition_source_value', icd10_cancer)}))
    """
)

cancer_hits = bq_to_polars(maybe_limit(cancer_query))
cancer = (
    cancer_hits.group_by("person_id")
    .agg([
        pl.col("condition_start_date").min().alias("cancer_date"),
    ])
    .with_columns((pl.col("cancer_date").is_not_null()).alias("cancer"))
    .select(["person_id", "cancer"])
)
print(cancer.shape)

# CAD
icd9_cad = ['410','410.%','411','411.%','412','412.%','413','413.%','414','414.%','V45.82']
icd10_cad = ['I25.1%']
cpt_cad = ['33534','33535','33536','33510','33511','33512','33513','33514','33515','33516','33517','33518','33519','33520','33521','33522','33523','92980','92981','92982','92984','92995','92996']

cad_icd_query = sql_with_dataset(
    f"""
    SELECT DISTINCT co.person_id, co.condition_start_date
    FROM `{dataset}.condition_occurrence` co
    JOIN `{dataset}.concept` c ON co.condition_source_concept_id = c.concept_id
    WHERE (c.vocabulary_id = 'ICD9CM' AND ({like_any('co.condition_source_value', icd9_cad)}))
       OR (c.vocabulary_id = 'ICD10CM' AND ({like_any('co.condition_source_value', icd10_cad)}))
    """
)

cad_cpt_query = sql_with_dataset(
    f"""
    SELECT DISTINCT p.person_id, p.procedure_date AS entry_date
    FROM `{dataset}.concept` c
    JOIN `{dataset}.procedure_occurrence` p ON c.concept_id = p.procedure_source_concept_id
    WHERE c.vocabulary_id = 'CPT4' AND ({like_any('c.concept_code', cpt_cad)})
    """
)

cad_icd = bq_to_polars(maybe_limit(cad_icd_query))
cad_cpt = bq_to_polars(maybe_limit(cad_cpt_query))

cad = (
    pl.concat([
        cad_icd.select(["person_id"]),
        cad_cpt.select(["person_id"]),
    ], how="diagonal")
    .unique()
    .with_columns(pl.lit(True).alias("cad"))
)
print(cad.shape)


# ## PheWAS Mapping
# 
# Load ICD→Phecode map from `analysis_inputs/ICD_to_Phecode_mapping.csv`.

# In[ ]:


# PheWAS mapping and wide pivot
# Load PheWAS mapping file
phemap = pl.read_csv(
    phemap_path,
    schema_overrides={"ICD": pl.String}
)
# Normalize expected column names
if "condition_source_value" not in phemap.columns and len(phemap.columns) >= 2:
    phemap = phemap.rename({phemap.columns[0]: "condition_source_value"})


# In[ ]:


phemap.head()


# In[ ]:


# PheWAS mapping and wide pivot — LOCAL JOIN + CHUNKED PIVOT (with had flags)



# Ensure all output directories exist

os.makedirs("processed_data/master", exist_ok=True)

os.makedirs("processed_data/master/phewas_chunks", exist_ok=True)



# ICD events restricted to EHR participants

icd_query = sql_with_dataset(

    """

    WITH ehr AS (

        SELECT DISTINCT person_id

        FROM `{dataset}.measurement` m

        LEFT JOIN `{dataset}.measurement_ext` mm USING (measurement_id)

        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT

        SELECT DISTINCT person_id

        FROM `{dataset}.condition_occurrence` m

        LEFT JOIN `{dataset}.condition_occurrence_ext` mm USING (condition_occurrence_id)

        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT

        SELECT DISTINCT person_id

        FROM `{dataset}.device_exposure` m

        LEFT JOIN `{dataset}.device_exposure_ext` mm USING (device_exposure_id)

        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT

        SELECT DISTINCT person_id

        FROM `{dataset}.drug_exposure` m

        LEFT JOIN `{dataset}.drug_exposure_ext` mm USING (drug_exposure_id)

        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT

        SELECT DISTINCT person_id

        FROM `{dataset}.observation` m

        LEFT JOIN `{dataset}.observation_ext` mm USING (observation_id)

        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT

        SELECT DISTINCT person_id

        FROM `{dataset}.procedure_occurrence` m

        LEFT JOIN `{dataset}.procedure_occurrence_ext` mm USING (procedure_occurrence_id)

        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT

        SELECT DISTINCT person_id

        FROM `{dataset}.visit_occurrence` m

        LEFT JOIN `{dataset}.visit_occurrence_ext` mm USING (visit_occurrence_id)

        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

    )

    SELECT DISTINCT cohort.person_id, co.condition_start_date, co.condition_source_value

    FROM ehr AS cohort

    JOIN `{dataset}.condition_occurrence` co USING (person_id)

    JOIN `{dataset}.concept` c ON co.condition_source_concept_id = c.concept_id

    WHERE c.vocabulary_id IN ('ICD9CM','ICD10CM')

    """

)



print("Fetching ICD events from BigQuery...")

icd_events = bq_to_polars(maybe_limit(icd_query))

print(f"ICD events: {icd_events.shape}")



# Join to phemap locally

print("Joining ICD events to phemap...")

phecodes = icd_events.join(phemap, on="condition_source_value", how="left")

print(f"After phemap join: {phecodes.shape}")



# Aggregate per person+phecode (compute earliest date)

print("Aggregating per person+phecode...")

phewas_long = (

    phecodes.lazy()

    .filter(pl.col("PHECODE").is_not_null())

    .group_by(["person_id", "PHECODE"])

    .agg([pl.col("condition_start_date").min().alias("entry_date")])

    .collect()

)

print(f"PheWAS long format: {phewas_long.shape}")



# Merge Fitbit baseline and compute had (≤180 days from baseline start)

phewas_long = (

    phewas_long

      .join(baseline, on="person_id", how="inner")

      .with_columns([

          (pl.col("entry_date") - pl.col("min_fitbit_date")).dt.total_days().alias("days_from_base")

      ])

      .with_columns([

          (pl.col("days_from_base") <= 180).alias("had")

      ])

      .drop("days_from_base")

)



# Save long format (useful for debugging and some analyses)

phewas_long.select(["person_id","PHECODE","entry_date","had"]).write_parquet("processed_data/master/master_phewas_long.parquet")

print("Wrote: processed_data/master/master_phewas_long.parquet")



# Chunked pivot: one phecode at a time to avoid memory explosion

unique_phecodes = phewas_long.select("PHECODE").unique().to_series().to_list()

print(f"Pivoting {len(unique_phecodes)} phecodes in chunks...")



# Get all unique person_ids from the long table (already restricted to Fitbit∩EHR)

all_persons = phewas_long.select("person_id").unique()

print(f"Total participants with baseline and phecode data: {all_persons.height}")



# Build presence/date/had columns one phecode at a time

for i, phecode in enumerate(unique_phecodes):

    subset = phewas_long.filter(pl.col("PHECODE") == phecode).select(["person_id", "entry_date", "had"])    

    subset = subset.with_columns([

        pl.lit(True).alias(f"phe_{phecode}"),

        pl.col("entry_date").alias(f"date_{phecode}"),

        pl.col("had").alias(f"had_{phecode}")

    ]).select(["person_id", f"phe_{phecode}", f"date_{phecode}", f"had_{phecode}"])

    subset.write_parquet(f"processed_data/master/phewas_chunks/phecode_{phecode}.parquet")

    if (i + 1) % 100 == 0:

        print(f"  Processed {i+1}/{len(unique_phecodes)} phecodes...")



print("Joining all chunks into wide format...")



# Start with all persons

phewas_wide = all_persons



# Join all phecode chunks (left join to preserve all participants)

chunk_files = sorted(Path("processed_data/master/phewas_chunks").glob("phecode_*.parquet"))

print(f"Found {len(chunk_files)} chunk files")



for i, chunk_file in enumerate(chunk_files):

    chunk_df = pl.read_parquet(chunk_file)

    phewas_wide = phewas_wide.join(chunk_df, on="person_id", how="left")

    if (i + 1) % 100 == 0:

        print(f"  Joined {i+1}/{len(chunk_files)} chunks...")



# Fill null boolean presence columns with False

phe_cols = [col for col in phewas_wide.columns if col.startswith("phe_")]

had_cols = [col for col in phewas_wide.columns if col.startswith("had_")]

print(f"Filling {len(phe_cols)} presence and {len(had_cols)} had columns with False for missing values...")

phewas_wide = phewas_wide.with_columns([

    *[pl.col(col).fill_null(False) for col in phe_cols],

    *[pl.col(col).fill_null(False) for col in had_cols],

])



print(f"Final phewas_wide shape (pre-covariate merge): {phewas_wide.shape}")



# Clean up temp chunks

shutil.rmtree("processed_data/master/phewas_chunks")

print("Cleaned up temporary chunk files")


# In[ ]:


phewas_wide.head()


# ## Overall Sleep Aggregates
# 
# We compute per-person overall average sleep duration (minutes) and standard deviation across all observed days.
# 

# In[ ]:


# Compute per-person overall sleep aggregates

sleep_agg_query = sql_with_dataset(

    """

    SELECT

      person_id,

      AVG(minute_asleep) AS avg_sleep_duration,

      STDDEV_SAMP(minute_asleep) AS sd_sleep_duration

    FROM `{dataset}.sleep_daily_summary`

    WHERE minute_asleep > 0

    GROUP BY person_id

    """

)



sleep_agg = bq_to_polars(maybe_limit(sleep_agg_query))

print(f"Sleep aggregates: {sleep_agg.shape}")


# ## Assemble Outputs
# 
# We merge PheWAS wide with Fitbit baseline, sleep aggregates, demographics, and SES, then write:
# 
# - `master_phewas_wide.parquet`
# 
# - `person_ids.parquet`
# 
# - `master_summary_stats.parquet`

# In[ ]:


# Assemble covariates and write outputs

# Clean/prepare demographics
cov_dat = (
    dem.lazy()
    .with_columns([
        pl.when(pl.col("race").is_in([
            "I prefer not to answer","No matching concept","More than one population","Another single population","Asian","None Indicated","None of these","PMI: Skip"
        ])).then(pl.lit(None)).otherwise(pl.col("race")).alias("race"),
        pl.when(pl.col("gender") == "Not man only, not woman only, prefer not to answer, or skipped").then(pl.lit(None))
          .when(pl.col("gender") == "No matching concept").then(pl.lit(None))
          .otherwise(pl.col("gender")).alias("gender"),
    ])
    .select(["person_id","date_of_birth","race","gender","sex_concept"])
    .collect()
)

# SES renames to match R naming vernacular
ses_renamed = (
    socio_eco_df
      .with_columns([pl.col("zip3").cast(pl.Utf8).alias("zip_code")])
      .rename({
          "median_income": "median_income",
          "fraction_assisted_income": "perc_with_assisted_income",
          "fraction_high_school_edu": "perc_with_high_school_education",
          "fraction_no_health_ins": "perc_with_no_health_insurance",
          "fraction_poverty": "perc_in_poverty",
          "fraction_vacant_housing": "perc_vacant_housing",
      })
      .select([
          "person_id","zip_code","median_income","perc_with_assisted_income",
          "perc_with_high_school_education","perc_with_no_health_insurance",
          "perc_in_poverty","perc_vacant_housing"
      ])
)

os.makedirs("processed_data/master", exist_ok=True)

# Final wide: PheWAS + baseline + sleep aggregates + demographics + SES + survey + BMI + cancer + cad
final_wide = (
    phewas_wide
      .join(baseline, on="person_id", how="left")
      .join(sleep_agg, on="person_id", how="left")
      .join(cov_dat, on="person_id", how="left")
      .join(ses_renamed, on="person_id", how="left")
      .join(survey, on="person_id", how="left")
      .join(bmi_latest, on="person_id", how="left")
      .join(cancer, on="person_id", how="left")
      .join(cad, on="person_id", how="left")
)

# R-compat renames
# - Baseline & sleep aggregates
final_wide = final_wide.rename({
    "min_fitbit_date": "min_date",
    "max_fitbit_date": "max_date",
    "duration_days": "duration",
    "avg_sleep_duration": "average_daily_sleep",
    "sd_sleep_duration": "sd_daily_sleep",
})
# - PheWAS flags: phe_* -> has_phe_*, had_* -> had_phe_*
rename_map = {}
for c in final_wide.columns:
    if c.startswith("phe_"):
        rename_map[c] = f"has_{c}"
    elif c.startswith("had_"):
        rename_map[c] = f"had_phe_{c[4:]}"
if rename_map:
    final_wide = final_wide.rename(rename_map)

# Write master wide (full dataset with PheWAS)
final_wide.write_parquet("processed_data/master/master_phewas_wide.parquet")
print("Wrote: processed_data/master/master_phewas_wide.parquet")

# Create covariates-only dataset (exclude PheWAS columns)
covariate_cols = [
    c for c in final_wide.columns 
    if c == "date_of_birth" or not (c.startswith("has_phe_") or c.startswith("had_phe_") or c.startswith("date_"))
]
covariates_only = final_wide.select(covariate_cols)
covariates_only.write_parquet("processed_data/master/master_covariates_only.parquet")
print("Wrote: processed_data/master/master_covariates_only.parquet")

# Person IDs from final_wide
person_ids = final_wide.select(["person_id"]).unique()
person_ids.write_parquet("processed_data/person_ids.parquet")
print("Wrote: processed_data/person_ids.parquet")

# Summary stats (basic)
summary_stats = pl.DataFrame({
    "total_participants": [person_ids.height],
    "bmi_missing": [int(bmi_latest.join(person_ids, on="person_id", how="right").select(pl.col("bmi").is_null().sum()).item())],
    "cancer_true": [int(cancer.join(person_ids, on="person_id", how="right").select(pl.col("cancer") == True).sum().row(0)[0])],
    "cad_true": [int(cad.join(person_ids, on="person_id", how="right").select(pl.col("cad") == True).sum().row(0)[0])],
})
summary_stats.write_parquet("processed_data/master/master_summary_stats.parquet")
print("Wrote: processed_data/master/master_summary_stats.parquet")


# In[ ]:


# Recompute BMI and BP restricted to EHR cohort
# Define EHR cohort CTE for reuse in measurement queries
ehr_cte = sql_with_dataset(
    """
    WITH ehr AS (
        SELECT DISTINCT person_id
        FROM `{dataset}.measurement` m
        LEFT JOIN `{dataset}.measurement_ext` mm USING (measurement_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.condition_occurrence` m
        LEFT JOIN `{dataset}.condition_occurrence_ext` mm USING (condition_occurrence_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.device_exposure` m
        LEFT JOIN `{dataset}.device_exposure_ext` mm USING (device_exposure_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.drug_exposure` m
        LEFT JOIN `{dataset}.drug_exposure_ext` mm USING (drug_exposure_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.observation` m
        LEFT JOIN `{dataset}.observation_ext` mm USING (observation_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.procedure_occurrence` m
        LEFT JOIN `{dataset}.procedure_occurrence_ext` mm USING (procedure_occurrence_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'

        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.visit_occurrence` m
        LEFT JOIN `{dataset}.visit_occurrence_ext` mm USING (visit_occurrence_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'
    )
    SELECT * FROM ehr
    """
)

# Helper to wrap a SELECT with the EHR CTE

def with_ehr(select_sql: str) -> str:
    # ehr_cte already includes full CTE + SELECT * FROM ehr; we only need the CTE part
    # Rebuild: take ehr_cte up to the last line before SELECT * FROM ehr and append provided select
    cte_prefix = ehr_cte.split("SELECT * FROM ehr")[0]
    return cte_prefix + select_sql

# BMI-related queries using EHR cohort
bmi_query = with_ehr(sql_with_dataset(
    """
    SELECT m.person_id, m.measurement_date, m.value_as_number AS bmi
    FROM ehr
    JOIN `{dataset}.measurement` m USING (person_id)
    WHERE m.measurement_concept_id = 3038553
    """
))

bmi2_query = with_ehr(sql_with_dataset(
    """
    SELECT m.person_id, m.measurement_date, m.value_as_number AS bmi
    FROM ehr
    JOIN `{dataset}.measurement` m USING (person_id)
    WHERE m.measurement_concept_id = 4245997
    """
))

height_query = with_ehr(sql_with_dataset(
    """
    SELECT m.person_id, m.measurement_date, m.value_as_number AS height
    FROM ehr
    JOIN `{dataset}.measurement` m USING (person_id)
    JOIN `{dataset}.concept` c ON m.measurement_concept_id = c.concept_id
    WHERE c.concept_id IN (3036277, 42527659, 3023540, 3019171)
    """
))

weight_query = with_ehr(sql_with_dataset(
    """
    SELECT m.person_id, m.measurement_date, m.value_as_number AS weight
    FROM ehr
    JOIN `{dataset}.measurement` m USING (person_id)
    JOIN `{dataset}.concept` c ON m.measurement_concept_id = c.concept_id
    WHERE c.concept_id IN (3025315, 3013762)
    """
))

# Run updated queries and recompute latest BMI
bmi_df = bq_to_polars(maybe_limit(bmi_query))
bmi2_df = bq_to_polars(maybe_limit(bmi2_query))
height_df = bq_to_polars(maybe_limit(height_query))
weight_df = bq_to_polars(maybe_limit(weight_query))

bmi_df = bmi_df.lazy().filter((pl.col("bmi") > 10) & (pl.col("bmi") < 200)).collect()
bmi2_df = bmi2_df.lazy().filter((pl.col("bmi") > 10) & (pl.col("bmi") < 200)).collect()

hw = height_df.join(weight_df, on=["person_id", "measurement_date"], how="inner")
hw = hw.lazy().with_columns((pl.col("weight") / (pl.col("height") * 0.01) ** 2).alias("bmi")).filter(pl.col("bmi") < 200).select(["person_id", "measurement_date", "bmi"]).collect()

bmi_all = pl.concat([bmi_df, bmi2_df, hw], how="diagonal")

bmi_latest = (
    bmi_all.lazy()
      .sort(["person_id", "measurement_date"], descending=[False, True])
      .group_by("person_id")
      .agg([pl.first("bmi").alias("bmi")])
      .collect(streaming=True)
)
print("bmi_latest (EHR cohort)", bmi_latest.shape)

# BP query using EHR cohort
bp_query = sql_with_dataset(
    """
    WITH ehr AS (
        SELECT DISTINCT person_id
        FROM `{dataset}.measurement` m
        LEFT JOIN `{dataset}.measurement_ext` mm USING (measurement_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'
        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.condition_occurrence` m
        LEFT JOIN `{dataset}.condition_occurrence_ext` mm USING (condition_occurrence_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'
        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.device_exposure` m
        LEFT JOIN `{dataset}.device_exposure_ext` mm USING (device_exposure_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'
        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.drug_exposure` m
        LEFT JOIN `{dataset}.drug_exposure_ext` mm USING (drug_exposure_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'
        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.observation` m
        LEFT JOIN `{dataset}.observation_ext` mm USING (observation_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'
        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.procedure_occurrence` m
        LEFT JOIN `{dataset}.procedure_occurrence_ext` mm USING (procedure_occurrence_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'
        UNION DISTINCT
        SELECT DISTINCT person_id
        FROM `{dataset}.visit_occurrence` m
        LEFT JOIN `{dataset}.visit_occurrence_ext` mm USING (visit_occurrence_id)
        WHERE LOWER(mm.src_id) LIKE 'ehr site%'
    ),
    diatb AS (
        SELECT person_id, measurement_datetime, value_as_number AS dia
        FROM `{dataset}.measurement` m
        JOIN `{dataset}.concept` c ON m.measurement_concept_id = c.concept_id
        WHERE LOWER(c.concept_name) LIKE '%diastolic blood pressure%'
    ),
    systb AS (
        SELECT person_id, measurement_datetime, value_as_number AS sys
        FROM `{dataset}.measurement` m
        JOIN `{dataset}.concept` c ON m.measurement_concept_id = c.concept_id
        WHERE LOWER(c.concept_name) LIKE '%systolic blood pressure%'
    )
    SELECT ehr.person_id, d.measurement_datetime, sys, dia
    FROM ehr
    JOIN diatb d USING (person_id)
    JOIN systb s USING (person_id, measurement_datetime)
    """
)

bp = bq_to_polars(maybe_limit(bp_query))

bp_latest = (
    bp.lazy()
      .sort(["person_id", "measurement_datetime"], descending=[False, True])
      .group_by("person_id")
      .agg([pl.col("sys").first().alias("sys"), pl.col("measurement_datetime").first().alias("bp_date")])
      .collect(streaming=True)
)
print("bp_latest (EHR cohort)", bp_latest.shape)


# In[ ]:




