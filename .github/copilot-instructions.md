# Copilot Instructions for PheWas

## Build, test, and lint commands

- There is no package build system, no centralized lint configuration, and no formal unit-test framework checked into this repository.
- The main runnable pipeline entry points are script-based:

```bash
python3 00_Sleep_Level_Extractor.py
python3 01b_Fitbit_Cohort_Covariates.py
python3 02_Data_Preparation.py
Rscript main.r
```

- You can also run the model/report stages directly when the expected parquet inputs and model directories already exist:

```bash
Rscript 04_LMM_Regression.r
Rscript 05_Precompute_Predictions.r
Rscript 05_LMM_Results.r
```

- Fast smoke-style runs use the repo's built-in subset/test switches rather than a separate test suite:

```bash
TEST_MODE_04=true N_TEST_IDS_04=25 Rscript 04_LMM_Regression.r
python3 02b_Create_Analysis_Subset.py config.yaml results/tmp_run
Rscript 02b_Create_Analysis_Subset.r config.yaml results/tmp_run
```

- The closest thing to a targeted validation command is the precompute-output validator. It checks one result bundle at a time:

```bash
python3 07_Validate_Precompute_Outputs.py results_05_5
```

## High-level architecture

- The repository is a mixed Python/R pipeline for Fitbit sleep, EHR, environment, and PGRS/PheWAS analysis on All of Us data.
- `00_Sleep_Level_Extractor.py` is the Fitbit sleep extraction layer. It queries AoU BigQuery tables, de-duplicates/merges sleep episodes, computes nightly sleep metrics, and writes `processed_data/daily_sleep_metrics_enhanced.parquet`.
- `01b_Fitbit_Cohort_Covariates.py` builds Fitbit-cohort covariates only and writes `processed_data/fitbit_cohort_covariates.parquet`.
- `01_Data_Accumulation_py.py` is the broader EHR/PheWAS accumulation path. It produces `processed_data/master/master_phewas_wide.parquet`, `processed_data/master/master_covariates_only.parquet`, and `processed_data/person_ids.parquet`.
- `02_Data_Preparation.py` is the bridge into analysis. It joins nightly sleep with cohort covariates, creates analysis features, writes `processed_data/ready_for_analysis.parquet`, then derives the reduced mixed-model input `processed_data/LMM_analysis.parquet`.
- `main.r` is the orchestrator for the modeling/report pipeline. It reads `config.yaml`, creates a timestamped `results/run_*` directory, optionally calls `02b_Create_Analysis_Subset.py` or `02b_Create_Analysis_Subset.r`, exports environment variables for downstream scripts, then sources the configured run steps.
- `04_LMM_Regression.r` fits the mixed-effects models and writes `.rds` model objects plus summary/VIF/emmeans artifacts.
- `05_Precompute_Predictions.r` turns model objects into cached CSV/RDS/parquet tables for plotting, and `05_LMM_Results.r` consumes those cached tables to render final tables and plots without reopening every model.
- `03_Descriptive_Analysis.py` and `03_Descriptive_Analysis.r` operate on `processed_data/ready_for_analysis.parquet` for descriptive outputs outside the main orchestrated model pipeline.
- `02_plink_PGRS_Generator_all.py` is part of the genetics branch, but despite the `.py` suffix it contains R code and behaves like an R notebook export. Treat it accordingly.

## Key conventions

- Always read AoU dataset identifiers from the environment. The core variables are `GOOGLE_PROJECT` and `WORKSPACE_CDR`; do not hardcode CDR/project IDs in code or docs.
- Prefer Parquet for large intermediates and keep them under `processed_data/`. Static mappings and score files live under `analysis_inputs/`. Results and model artifacts live under `results*/`, `models_*`, and `model_summaries_*`.
- The Python ETL scripts are notebook exports and often keep notebook-style cells plus inline package install commands such as `get_ipython().system(...)`. Preserve that style when editing those files instead of refactoring them into a package structure.
- `main.r` and `02b_Create_Analysis_Subset.r` prefer a project-local `.r_libs` when present. Avoid assuming only system-wide R libraries are available.
- Sleep timing is standardized to a noon-to-noon linearized scale in `02_Data_Preparation.py`: values before noon are shifted by +24, and downstream R/Python analysis expects `onset_linear`, `midpoint_linear`, and `offset_linear` on roughly the `[12, 36)` scale.
- The "weekend" flag used downstream is really free-day logic: `00_Sleep_Level_Extractor.py` builds `is_weekend_or_holiday`, and `02_Data_Preparation.py` casts that into the modeling `is_weekend` field.
- Environmental seasonality for the mixed-model pipeline uses month as the main seasonal control. `02_Data_Preparation.py` can join `environmental_data/photo_info/all_photo_info.parquet` to add `deviation`, and `04_LMM_Regression.r` explicitly uses month and may include `deviation` when available.
- In the LMM workflow, ML fits are for model comparison/AIC and REML fits are for coefficient interpretation and downstream marginal summaries. Preserve both outputs when changing `04_LMM_Regression.r`.
- Employment and ZIP3 normalization logic is duplicated across Python and R paths (`02b_*`, `04_LMM_Regression.r`, `05_LMM_Results.r`). Keep those transformations aligned when modifying one side.
- `02_Data_Preparation.py` falls back to `processed_data/master/master_covariates_only.parquet` when `processed_data/fitbit_cohort_covariates.parquet` is absent. Do not remove that fallback unless you also update the broader pipeline assumptions.
- Do not commit, print, or summarize PHI/PII or AoU-restricted row-level data. Shared outputs should stay aggregated/masked.
