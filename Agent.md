# **Implementation Strategy: Fitbit Sleep Analysis**

## General Instructions 
After each implementation add a breif implementation results section below the general instructions. 

Note the data included in the processed_data is just randomly generated for testing DO NOT expect to see any signal. 

Operational note: Production analyses and model runs are executed remotely in the All of Us secure platform. Local workspace files are for development/reference only and should not be used as the source of truth for remote run completion status.

This guide outlines the full execution plan, from raw data to genetic phenotypes, using a **hybrid Python/R notebook workflow**.

## **Project Setup & Environment**

**1\. Directory Structure:**  
project\_root/  
├── analysis\_inputs/                \# Raw inputs (PGRS files, master\_covariates\_only.parquet)  
├── processed\_data/                 \# Intermediate files (Parquet, cleaned CSVs)  
├── results/                        \# Final figures, tables, and model summaries  
├── 00\_Sleep\_Level\_Extractor.ipynb        \# (Python) Upstream Extractor  
├── 01\_Data\_Accumulation\_py.ipynb         \# (Python) Existing Upstream Accumulation  
├── 02\_Data\_Preparation.ipynb             \# (Python) NEW: Merge, Clean & Feature Eng  
├── 03\_Descriptive\_Analysis.ipynb         \# (R) Table 1 & Visualization  
├── 04\_Statistical\_Modeling.ipynb         \# (R) GAMMs & Mixed Models  
├── 05\_Phenotype\_Export.ipynb             \# (R) BLUP Extraction  
├── 06\_plink\_PGRS\_Generator\_all.ipynb     \# (Python) Downstream Genetics  
├── mock\_data\_generator.py          \# Local support script  
└── venv/                           \# Python virtual environment

**2\. Environment Management:**

* **Python (Data Prep):** Use venv with polars, numpy, pandas.  
* **R (Modeling):** Ensure your Jupyter environment has an R kernel installed (IRkernel). Key packages: tidyverse, arrow, mgcv, gtsummary.

## **Phase 0: Data Preparation & Engineering**

Notebook: 02\_Data\_Preparation.ipynb  
Language: Python (Polars)  
**1\. Load Data:**

* **Sleep Data:** Load processed\_data/daily\_sleep\_metrics\_enhanced.parquet (Output from 00/01) using pl.read\_parquet().  
* **Covariates:** Load analysis\_inputs/master\_covariates\_only.parquet.

**2\. Merge & Clean:**

* **Join:** Left join Sleep Data with Covariates on person\_id.  
* **Pregnancy:** *Note limitation (V8 data).*  
* **Filtering:**  
  * Drop rows with missing date\_of\_birth or sex\_concept.  
  * *(Optional)* Filter person\_total\_nights \>= 7 if preferred, or keep all for LMM shrinkage.  
* **Variables to Keep (from metadata):**  
  * date\_of\_birth (for dynamic age)  
  * sex\_concept (Male/Female)  
  * race  
  * employment\_status (e.g., "Retired", "Employed For Wages")  
  * zip\_code (for latitude proxy)  
  * bmi  
  * menstral\_stop\_reason (for menopause status)  
  * *(Optional)* Disease flags: depression, type\_2\_diabetes, cad (available for sensitivity analysis).

**3\. Feature Engineering (Python Implementation):**

* **Age:** Calculate age\_at\_sleep \= (sleep\_date \- date\_of\_birth) / 365.25.  
* **Seasonality:** Calculate day\_of\_year (1-366) using .dt.ordinal\_day().  
* **Circular Encoding:**  
  import numpy as np  
  \# Apply to Start, End, and Midpoint  
  df \= df.with\_columns(\[  
      (np.sin(2 \* np.pi \* pl.col("daily\_midpoint\_hour") / 24)).alias("midpoint\_sin"),  
      (np.cos(2 \* np.pi \* pl.col("daily\_midpoint\_hour") / 24)).alias("midpoint\_cos")  
  \])

* **Social Jetlag:** Calculate raw difference: (mean\_weekend \- mean\_weekday) per person.

**4\. Variable Mapping:**

* **Employment:** Consolidate if necessary (e.g., Group "Out of Work..." categories).  
* **Menopause:** Create binary is\_postmenopausal from menstral\_stop\_reason (e.g., "Natural Menopause", "Surgery").

**5\. Output:**

* Save processed\_data/ready\_for\_analysis.parquet.

## **Phase 1: Descriptive Analysis**

Notebook: 03\_Descriptive\_Analysis.ipynb  
Language: R (ggplot2)  
**1\. Setup:**

* Load data: df \<- arrow::read\_parquet("processed\_data/ready\_for\_analysis.parquet").

**2\. Table 1 (Demographics):**

* **Tool:** gtsummary.  
* Create a person-level subset (1 row per person).  
* Generate summary table stratified by sex\_concept.  
* Include: Age, race, employment\_status, bmi.

**3\. Visualization (ggplot2):**

* **Figure 1 (Chronotype):** geom\_histogram(aes(x \= mean\_midpoint))  
* **Figure 2 (Biological Window):**  
  * Create a balanced subsample (e.g., 30 nights/person).  
  * geom\_density(aes(x \= daily\_start\_hour), fill="blue") \+ geom\_density(aes(x \= daily\_end\_hour), fill="orange").  
* **Figure 3 (Social Jetlag):**  
  * Paired boxplot of Weekday vs Weekend midpoint.

## **Phase 2: Statistical Modeling**

Notebook: 04\_Statistical\_Modeling.ipynb  
Language: R (mgcv)  
1\. Strategy:  
Use Generalized Additive Mixed Models (GAMMs) to handle the non-linear Age and cyclic Seasonality.  
**2\. Models to Run:**

* **Model A: Sleep Onset (Bedtime)**  
  gam(list(  
      onset\_sin \~ s(age\_at\_sleep, by=sex\_concept) \+ employment\_status \+ bmi \+ is\_weekend \+ s(day\_of\_year, bs='cc', k=12),  
      onset\_cos \~ s(age\_at\_sleep, by=sex\_concept) \+ employment\_status \+ bmi \+ is\_weekend \+ s(day\_of\_year, bs='cc', k=12)  
  ), family \= mvn(d=2), data \= df, random \= list(person\_id=\~1))

* **Model B: Sleep Offset (Wake time)**  
  * Same structure, targeting end\_hour.  
* **Model C: Sleep Midpoint (Chronotype)**  
  * Same structure, targeting midpoint\_hour.  
* **Model D: Social Jetlag (Person-Level)**  
  * Target: SJL\_raw (difference score).  
  * Model: Linear regression (not mixed) checking association with Age, Sex, Employment, BMI.

## **Phase 3: Phenotype Generation (Export)**

Notebook: 05\_Phenotype\_Export.ipynb  
Language: R  
**1\. BLUP Extraction:**

* Fit "Null Models" (Intercept \+ Random Effect only) for Onset, Offset, and Midpoint.  
* Extract the random intercepts ($u\_i$) using ranef().

**2\. Filtering:**

* **Strict Rule:** Filter to person\_id with **N \> 7 nights**.

**3\. Export:**

* Save results/sleep\_phenotypes\_for\_gwas.csv.  
* Columns: person\_id, u\_onset, u\_offset, u\_midpoint, SJL\_raw.

## **Phase 4: Downstream Genetics**

Notebook: 06\_plink\_PGRS\_Generator\_all.ipynb  
Language: Python

* **Input:** Reads results/sleep\_phenotypes\_for\_gwas.csv.  
* **Action:** Runs the genomic association pipeline (PLINK/PGRS) using the phenotypes generated in Phase 3\.  
* 
