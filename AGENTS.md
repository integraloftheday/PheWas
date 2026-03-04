# Agent Guidelines for PheWas

This repository focuses on Phenome-wide analyses (PheWAS) using All of Us (AoU) data, specifically integrating Fitbit sleep metrics with electronic health records (EHR) and Polygenic Risk Scores (PGS).

## 🛠 Build, Test, and Execution

The project follows a sequential workflow using Python (for BigQuery/Polars) and R (for statistics/visualization).

### Execution Flow
1. **Extraction**: `00_Sleep_Level_Extractor.py` (BigQuery -> Polars)
2. **Accumulation**: `01_Data_Accumulation_py.py` & `01b_Fitbit_Cohort_Covariates.py` (EHR Joining)
3. **PGRS Generation**: `02_plink_PGRS_Generator_all.py` (Integration of genomic data)
4. **Analysis**: `03_Descriptive_Analysis.r`, `04_LLM_Regression.r`, `05_LMM_Results.r`

### Environment Requirements
- **GCP Environment Variables**:
  - `GOOGLE_PROJECT`: GCP billing project ID.
  - `WORKSPACE_CDR`: Full AoU CDR dataset ID (e.g., `fc-aou-XXXX.C2024Q2R3`).
- **Python Dependencies**: `polars`, `google-cloud-bigquery`, `pandas`, `holidays`, `scipy`, `plotly`.
- **R Dependencies**: `tidyverse`, `arrow`, `nanoparquet`, `bigrquery`, `rms`, `gtsummary`, `MungeSumstats`.

### Running Scripts
Most scripts are converted from notebooks. Run them directly via the CLI:
- **Python**: `python 00_Sleep_Level_Extractor.py`
- **R**: `Rscript 03_Descriptive_Analysis.r`

### Testing
There is no formal testing framework (like pytest or testthat) currently implemented.
- **Manual Verification**: Check if output files are generated in `processed_data/` (e.g., `ready_for_analysis.parquet`).
- **Dry Runs**: Use `LIMIT` in BigQuery queries to test extraction logic.

---

## 🎨 Code Style and Conventions

### Python Guidelines
- **Libraries**: Prefer **Polars** over Pandas for large-scale data manipulation. Use `polars.from_arrow(query_job.result().to_arrow())` for efficient BigQuery ingestion.
- **Naming**: Use `snake_case` for variables and functions.
- **SQL**: Format SQL strings using f-strings with `{dataset}` placeholders for the CDR.
- **Types**: Use Python type hints where possible (e.g., `def my_func(df: pl.DataFrame) -> pl.DataFrame:`).

### R Guidelines
- **Libraries**: Use the **Tidyverse** ecosystem (`dplyr`, `ggplot2`, `readr`).
- **Data I/O**: Use `arrow` or `nanoparquet` for reading/writing Parquet files.
- **Pipes**: Use the native pipe `|>` (R 4.1+) or `%>%` (magrittr/dplyr).
- **Functions**: Define helper functions for repetitive plotting or regression tasks to maintain DRY principles.

### Data Management
- **Intermediates**: Store large datasets in `.parquet` format within `processed_data/`.
- **Static Assets**: Place mapping files (e.g., ICD to Phecode) in `analysis_inputs/`.
- **Privacy**: **NEVER** commit or log PHI/PII. Ensure all shared results use aggregation/masking (e.g., filter groups with n < 20).

### Error Handling
- **Python**: Use `try...except` blocks around BigQuery requests and file I/O.
- **R**: Use `tryCatch` for complex statistical models (like LMMs) that may fail to converge.

---

## 🤖 AI Guardrails
- **Environment Variables**: Always use `os.getenv("WORKSPACE_CDR")`. Never hardcode dataset IDs.
- **Codebase Consistency**: Mimic the "notebook-style" script structure where blocks are clearly separated by comments or `In [ ]` markers.
- **BigQuery**: Be mindful of query costs; always select only necessary columns.
