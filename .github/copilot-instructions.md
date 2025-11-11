# Copilot Instructions for PheWas

Purpose and Scope
- Focus: Phenome-wide analyses on All of Us (AoU) data, extracting Fitbit sleep metrics and building PGRS/PGS-driven PheWAS tables.
- Primary workflow is R-based and organized as three notebooks executed in order: `00_`, `01_`, `02_`.

Repo Structure and Flow
- Notebooks:
  - `00_sleep_level_extractor.ipynb`: Query AoU BigQuery CDR for Fitbit sleep levels/daily summary; derive sleep features.
  - `01_data_accumulation.ipynb`: Join AoU EHR/measurement/concepts with sleep features; build analysis tables.
  - `02_plink_PGRS_Generator.ipynb`: Integrate PGS/PGRS resources and ancestry filters; prepare PGRS tables for downstream PheWAS.
- Data dirs:
  - `aou_raw_data/`, `raw_data/`: external inputs or dumps (never commit PHI).
  - `processed_data/avg_duration/`, `processed_data/midpoint/`: notebook outputs (Parquet/TSV/CSV).
  - `analysis_inputs/`: static inputs (e.g., `ICD_to_Phecode_mapping.csv`, `PGS002196-average.txt`, `PGS002209_hmPOS_GRCh38.txt`, `chronotype_meta_PRS_model.txt`).
  - `results/`: figures/tables produced by notebooks.

Execution Environment
- Language: R (R kernel in .ipynb). Ensure IRkernel is available in VS Code/Jupyter.
- R packages frequently used: `bigrquery`, `data.table`, `readr`, `stringr`, `dplyr`, `tidyr`, `tidyverse`, `lubridate`, `nanoparquet`, `ggplot2`, `ggrepel`, `viridis`, `ggsci`, `rms`, `ggpubr`, `jsonlite`, `furrr`, `purrr`, `speedglm`, `progressr`, `circular`, `lutz`, `usa`, `Hmisc`.
- External services: Google BigQuery (AoU CDR). Auth and billing required.

Required Environment Variables
- `GOOGLE_PROJECT`: GCP billing project for BigQuery jobs (used by `bq_project_query(...)` and as `billing` in `bq_dataset_query(...)`).
- `WORKSPACE_CDR`: Full AoU CDR dataset id (e.g., `fc-aou-XXXX.C2024Q2R3`), used to format queries like ``FROM `{dataset}.table` ``.

Critical Patterns and Conventions
- BigQuery access:
  - Queries are built as strings with `{dataset}` placeholders; executed via `bigrquery::bq_project_query(Sys.getenv("GOOGLE_PROJECT"), query)` or `bq_dataset_query(dataset, query, billing=...)`.
  - Expect long-running queries; code batches reads and sometimes uses intermediate Parquet via `nanoparquet`.
- Data I/O:
  - Prefer Parquet for large intermediates (`nanoparquet::read_parquet`/`write_parquet`), TSV/CSV for compatibility.
  - Outputs organized under `processed_data/` by metric (e.g., `avg_duration`, `midpoint`).
- PGS/PGRS integration:
  - `02_` notebook reads precomputed PGRS/PGS artifacts (e.g., under `PGRS_Average/plinkfiles/...` and `analysis_inputs/`).
  - PLINK execution is assumed external; the repo consumes its outputs rather than invoking PLINK directly.

Running the Workflow (local, minimal)
- Configure GCP auth and env vars, then run notebooks in order: `00_` → `01_` → `02_`.
- Example shell (zsh):
  - `export GOOGLE_PROJECT="your-gcp-project"`
  - `export WORKSPACE_CDR="your-project.your_dataset"`
- In VS Code, open each notebook and use the R kernel; install missing R packages as prompted.

Key Files to Reference
- Notebooks: `00_sleep_level_extractor.ipynb`, `01_data_accumulation.ipynb`, `02_plink_PGRS_Generator.ipynb`
- Inputs: `analysis_inputs/ICD_to_Phecode_mapping.csv`, `analysis_inputs/PGS002196-average.txt`, `analysis_inputs/PGS002209_hmPOS_GRCh38.txt`, `analysis_inputs/chronotype_meta_PRS_model.txt`
- Data roots: `processed_data/`, `raw_data/`, `aou_raw_data/`, `results/`

Guardrails for Agents
- Do not commit or print PHI/PII or AoU-restricted data.
- Avoid hardcoding CDR ids; always read `WORKSPACE_CDR`/`GOOGLE_PROJECT` from the environment.
- Maintain R-first workflow; do not introduce Python unless explicitly requested.
- Keep intermediate files under `processed_data/` and large static mappings under `analysis_inputs/`.
