# --- Load Required Libraries ---
# Ensure necessary packages are installed
if (!requireNamespace("BiocManager", quietly = TRUE))
    install.packages("BiocManager")
if (!requireNamespace("remotes", quietly = TRUE))
    install.packages("remotes")
if (!requireNamespace("readr", quietly = TRUE))
    install.packages("readr")
if (!requireNamespace("dplyr", quietly = TRUE))
    install.packages("dplyr")
if (!requireNamespace("tidyverse", quietly = TRUE))
    install.packages("tidyverse")
if (!requireNamespace("MungeSumstats", quietly = TRUE))
    BiocManager::install('MungeSumstats')
if (!requireNamespace("GWASBrewer", quietly = TRUE)) {
    # Need remotes/devtools for GitHub installs
    if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
    devtools::install_github("jean997/GWASBrewer", 
                         ref = "fdea10e7c4d86585a42c918a2ab828103e07b5ae",
                         build_vignettes = TRUE)
}
if (!requireNamespace("janitor", quietly = TRUE))
    install.packages("janitor")
if (!requireNamespace("SNPlocs.Hsapiens.dbSNP155.GRCh38", quietly = TRUE))
    BiocManager::install("SNPlocs.Hsapiens.dbSNP155.GRCh38")
if (!requireNamespace("BSgenome.Hsapiens.NCBI.GRCh38", quietly = TRUE))
    BiocManager::install("BSgenome.Hsapiens.NCBI.GRCh38")
if (!requireNamespace("arrow", quietly = TRUE))
    install.packages("arrow") # For reading Parquet files

library(readr)
library(dplyr)
library(tidyverse)
library(MungeSumstats)
library(GWASBrewer)
library(janitor)
library(nanoparquet) # Load arrow package

