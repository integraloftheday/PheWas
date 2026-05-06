#!/usr/bin/env python
# coding: utf-8

# In[ ]:


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
# Install GWASBrewer without building vignettes to avoid heavy Suggests like hapsim/DiagrammeR
if (!requireNamespace("GWASBrewer", quietly = TRUE)) {
    if (!requireNamespace("remotes", quietly = TRUE)) install.packages("remotes")
    # Avoid vignette build to prevent optional deps errors
    remotes::install_github(
        repo = "jean997/GWASBrewer",
        ref = "fdea10e7c4d86585a42c918a2ab828103e07b5ae",
        build_vignettes = FALSE,
        dependencies = TRUE,
        upgrade = "never"
    )
}
if (!requireNamespace("janitor", quietly = TRUE))
    install.packages("janitor")
if (!requireNamespace("SNPlocs.Hsapiens.dbSNP155.GRCh38", quietly = TRUE))
    BiocManager::install("SNPlocs.Hsapiens.dbSNP155.GRCh38")
if (!requireNamespace("BSgenome.Hsapiens.NCBI.GRCh38", quietly = TRUE))
    BiocManager::install("BSgenome.Hsapiens.NCBI.GRCh38")
if (!requireNamespace("nanoparquet", quietly = TRUE))
    install.packages("nanoparquet")
if (!requireNamespace("jsonlite", quietly = TRUE))
    install.packages("jsonlite")

library(readr)
library(dplyr)
library(tidyverse)
library(MungeSumstats)
library(GWASBrewer)
library(janitor)
library(nanoparquet)
library(jsonlite)

cat("✓ All libraries loaded successfully\n")


# ## Configuration and Setup

# In[ ]:


# --- Configuration ---
# Define input/output paths
parquet_file <- "processed_data/person_ids.parquet"
analysis_inputs_dir <- "analysis_inputs"
output_base_dir <- "processed_data/PGRS"

# Ensure base directories exist
for (d in c("processed_data", output_base_dir, analysis_inputs_dir)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# GCS paths - update these if dataset version changes
gcs_plink_base <- "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/clinvar/plink_bed"
gcs_ancestry_path <- "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv"
ancestry_filter <- tolower(Sys.getenv("PGRS_ANCESTRY_FILTER", "all"))
if (!ancestry_filter %in% c("all", "eur")) {
  stop("PGRS_ANCESTRY_FILTER must be one of: all, eur")
}
ancestry_filter_label <- if (ancestry_filter == "eur") "EUR-only" else "all-ancestry"
shared_ref_tag <- paste0("plink_ref_", ancestry_filter)
file_pattern <- Sys.getenv("PGRS_FILE_PATTERN", "")
wandb_progress_file <- Sys.getenv("WANDB_PROGRESS_FILE", "")

log_progress_event <- function(stage, event, status = "running", metrics = list(), details = list()) {
  if (!nzchar(wandb_progress_file)) return(invisible(NULL))
  payload <- c(
    list(
      timestamp = format(Sys.time(), "%Y-%m-%dT%H:%M:%SZ", tz = "UTC"),
      stage = stage,
      event = event,
      status = status
    ),
    if (length(metrics) > 0) list(metrics = metrics) else list(),
    if (length(details) > 0) list(details = details) else list()
  )
  dir.create(dirname(wandb_progress_file), recursive = TRUE, showWarnings = FALSE)
  cat(
    jsonlite::toJSON(payload, auto_unbox = TRUE, null = "null"),
    "\n",
    file = wandb_progress_file,
    append = TRUE
  )
}

cat("Output directory:", output_base_dir, "\n")
cat("GCS PLINK base:", gcs_plink_base, "\n")
cat("Participant filter:", ancestry_filter_label, "\n")

# Discover all PGRS files in analysis_inputs (exclude ICD mapping)
pgrs_files <- list.files(analysis_inputs_dir,
                        pattern = "\\.(txt|txt\\.gz)$",
                        full.names = TRUE)
pgrs_files <- pgrs_files[!grepl("ICD_to_Phecode", pgrs_files)]
if (nzchar(file_pattern)) {
  pgrs_files <- pgrs_files[grepl(file_pattern, basename(pgrs_files))]
}
if (length(pgrs_files) == 0) {
  stop("No PGRS files matched the current selection in analysis_inputs/.")
}

cat("\n✓ Found", length(pgrs_files), "PGRS files to process:\n")
for (f in pgrs_files) {
  cat("  -", basename(f), "\n")
}

# Define chromosomes to process
chromosomes <- c("chr1","chr2","chr3","chr4","chr5","chr6","chr7","chr8","chr9","chr10",
                 "chr11","chr12","chr13","chr14","chr15","chr16","chr17","chr18","chr19",
                 "chr20","chr21","chr22","chrX","chrY")
cat("\n✓ Will process", length(chromosomes), "chromosomes\n")
log_progress_event(
  "pgrs_scoring",
  "configuration_loaded",
  metrics = list(
    pgrs_files_total = length(pgrs_files),
    chromosomes_total = length(chromosomes)
  ),
  details = list(
    ancestry_filter = ancestry_filter,
    file_pattern = file_pattern
  )
)


# ## Helper Functions

# In[ ]:


# Function to detect PGRS file format and return column positions
detect_pgrs_format <- function(file_path) {
  cat("\nDetecting format for:", basename(file_path), "\n")
  
  # Read first lines to detect format
  if (grepl("\\.gz$", file_path)) {
    con <- gzfile(file_path, "r")
    lines <- readLines(con, n = 30)
    close(con)
  } else {
    lines <- readLines(file_path, n = 30)
  }
  
  # Remove comment lines
  header_lines <- lines[grepl("^#", lines)]
  data_lines <- lines[!grepl("^#", lines)]
  
  # Get first data line to detect format
  first_data <- data_lines[1]
  fields <- strsplit(first_data, "\\s+")[[1]]
  
  format_info <- list(
    file = file_path,
    has_header = FALSE,
    rsid_col = NA,
    effect_allele_col = NA,
    weight_col = NA,
    header_lines = header_lines
  )
  
  # Detect PGS Catalog format (has rsID header)
  if (any(grepl("^rsID", data_lines[1:2]))) {
    cat("  Format: PGS Catalog (with rsID column header)\n")
    format_info$has_header <- TRUE
    header_line <- data_lines[grepl("^rsID", data_lines)][1]
    headers <- strsplit(header_line, "\\s+")[[1]]
    format_info$rsid_col <- which(headers == "rsID")
    format_info$effect_allele_col <- which(headers == "effect_allele")
    format_info$weight_col <- which(headers == "effect_weight")
  } 
  # Detect chronotype format (chr pos rsid allele1 allele2 weight)
  else if (length(fields) >= 6 && grepl("^(chr)?[0-9XY]+$", fields[1])) {
    cat("  Format: Chronotype-style (chr pos rsid alleles weight)\n")
    format_info$has_header <- FALSE
    format_info$rsid_col <- 2  # rsID is 2nd column
    format_info$effect_allele_col <- 4  # effect allele
    format_info$weight_col <- 6  # weight
  }
  else {
    stop("Unknown PGRS file format for: ", basename(file_path))
  }
  
  cat("  Columns: rsID=", format_info$rsid_col, 
      " effect_allele=", format_info$effect_allele_col,
      " weight=", format_info$weight_col, "\n")
  
  return(format_info)
}

# Function to prepare PGRS file for PLINK (remove headers, extract needed columns)
prepare_pgrs_file <- function(pgrs_path, output_dir) {
  cat("\nPreparing PGRS file:", basename(pgrs_path), "\n")
  
  format_info <- detect_pgrs_format(pgrs_path)
  
  # Read all lines
  if (grepl("\\.gz$", pgrs_path)) {
    con <- gzfile(pgrs_path, "r")
    all_lines <- readLines(con)
    close(con)
  } else {
    all_lines <- readLines(pgrs_path)
  }
  
  # Remove comment lines (starting with #)
  data_lines <- all_lines[!grepl("^#", all_lines)]
  
  # If has header, remove it
  if (format_info$has_header) {
    data_lines <- data_lines[-1]
  }
  
  # Write prepared file
  prepared_file <- file.path(output_dir, paste0(tools::file_path_sans_ext(basename(pgrs_path)), "_prepared.txt"))
  writeLines(data_lines, prepared_file)
  
  cat("  Prepared file saved:", prepared_file, "\n")
  cat("  Lines:", length(data_lines), "\n")
  
  return(list(
    prepared_file = prepared_file,
    format_info = format_info,
    variant_count = length(data_lines)
  ))
}

cat("✓ Helper functions defined\n")


# ## Load Shared Data (Person IDs and Ancestry)

# In[ ]:


cat("\n=== Loading shared data (person IDs and ancestry) ===\n")

# Load unique person IDs from parquet
cat("Loading person IDs from:", parquet_file, "\n")
if (!file.exists(parquet_file)) {
    stop("Parquet file not found: ", parquet_file)
}
all_data <- read_parquet(parquet_file, col_select = "person_id")
person_ids_df <- data.frame(person_id = unique(all_data$person_id))
cat("✓ Loaded", nrow(person_ids_df), "unique person IDs\n")
rm(all_data)
gc()

# Download/load ancestry data (shared location in processed_data)
ancestry_dir <- file.path(output_base_dir, "shared")
dir.create(ancestry_dir, showWarnings = FALSE, recursive = TRUE)
ancestry_dest_path <- file.path(ancestry_dir, "ancestry_preds.tsv")

cat("\nChecking/Downloading ancestry data...\n")
if (!file.exists(ancestry_dest_path)) {
  gsutil_ancestry_args <- c("-u", Sys.getenv("GOOGLE_PROJECT"),
                            "cp",
                            gcs_ancestry_path,
                            ancestry_dest_path)
  result <- system2("gsutil", args = gsutil_ancestry_args, stdout = TRUE, stderr = TRUE)
  if (!file.exists(ancestry_dest_path)) {
      stop("Failed to download ancestry data")
  }
  cat("  ✓ Downloaded successfully\n")
} else {
  cat("  ✓ Ancestry data already exists\n")
}

# Download chr21.fam (shared)
chr21_fam_path <- file.path(ancestry_dir, "chr21.fam")
chr21_fam_gcs <- paste0(gcs_plink_base, "/chr21.fam")

cat("\nChecking/Downloading chr21.fam...\n")
if (!file.exists(chr21_fam_path)) {
  gsutil_fam_args <- c("-u", Sys.getenv("GOOGLE_PROJECT"), "cp", chr21_fam_gcs, chr21_fam_path)
  result <- system2("gsutil", args = gsutil_fam_args, stdout = TRUE, stderr = TRUE)
  if (!file.exists(chr21_fam_path)) {
      stop("Failed to download chr21.fam")
  }
  cat("  ✓ Downloaded successfully\n")
} else {
  cat("  ✓ chr21.fam already exists\n")
}

# Create filtered patient list
cat("\nCreating patient list...\n")
fam_data <- read_table(chr21_fam_path, col_names = FALSE, show_col_types = FALSE)
colnames(fam_data) <- c("FID", "IID", "PID", "MID", "SEX", "PHENOTYPE")

ancestry <- read_tsv(ancestry_dest_path, show_col_types = FALSE)
eligible_ids <- if (ancestry_filter == "eur") {
  ancestry %>% filter(ancestry_pred == "eur") %>% pull(research_id)
} else {
  ancestry %>% pull(research_id)
}

filtered_fam <- fam_data[fam_data$IID %in% eligible_ids & fam_data$IID %in% person_ids_df$person_id, ]
patient_list <- filtered_fam[, c("FID", "IID")]

patient_list_file <- file.path(ancestry_dir, paste0("patient_list_", ancestry_filter, ".txt"))
write_tsv(patient_list, patient_list_file, col_names = FALSE)

cat("✓ Patient list created:", nrow(patient_list), ancestry_filter_label, "individuals\n")
cat("  File saved:", patient_list_file, "\n")
log_progress_event(
  "pgrs_scoring",
  "patient_list_created",
  metrics = list(participants_targeted = nrow(patient_list)),
  details = list(ancestry_filter = ancestry_filter)
)

rm(fam_data, ancestry, eligible_ids, filtered_fam)
gc()


# ## Chromosome Processing Functions

# ## Build Shared PLINK Reference (once)
# Build per-chromosome filtered PLINK files for the selected participant set and merge into a shared `all_chromosomes` reference to reuse across all PGRS scoring runs.

# In[ ]:


# Build shared reference under processed_data/PGRS/shared/plink_ref_<filter>
shared_ref_dir <- file.path(output_base_dir, "shared", shared_ref_tag)
raw_dir <- file.path(shared_ref_dir, "plink_raw")
filtered_dir <- file.path(shared_ref_dir, "filtered")
all_prefix <- file.path(shared_ref_dir, "all_chromosomes")
for (d in c(shared_ref_dir, raw_dir, filtered_dir)) {
  if (!dir.exists(d)) dir.create(d, recursive = TRUE, showWarnings = FALSE)
}

# Helper: update rsIDs via MungeSumstats using BIM content (once per chromosome)
update_rsid_for_chr <- function(chr) {
  bim_file_path <- file.path(raw_dir, paste0(chr, ".bim"))
  if (!file.exists(bim_file_path)) return(NULL)
  bim.raw <- read_tsv(bim_file_path, col_names = FALSE, show_col_types = FALSE) %>%
    janitor::clean_names() %>%
    dplyr::rename(chrom = x1, id = x2, cm = x3, pos = x4, ref = x6, alt = x5)
  dat_simple <- GWASBrewer::sim_mv(G = 1, N = 2578, J = nrow(bim.raw), h2 = 0.5, pi = 0.01,
                                   est_s = TRUE, af = function(n){rbeta(n, 1, 5)})
  bim <- dplyr::bind_cols(bim.raw, dat_simple$beta_hat, dat_simple$se_beta_hat, dat_simple$snp_info) %>%
    tibble::as_tibble() %>%
    dplyr::rename(beta = ...7, se = ...8) %>%
    dplyr::select(-SNP) %>%
    dplyr::mutate(Z = beta / se, P = 2 * (1 - stats::pt(abs(Z), Inf)))
  reformatted <- MungeSumstats::format_sumstats(path = bim, ref_genome = "GRCh38", return_data = TRUE,
                                                bi_allelic_filter = FALSE, N_dropNA = FALSE, rmv_chr = NULL,
                                                allele_flip_check = FALSE, allele_flip_drop = FALSE,
                                                allele_flip_frq = FALSE, dbSNP = 155) %>% tibble::as_tibble()
  rsid_map <- reformatted %>% dplyr::select(ID, SNP)
  rsid_file <- file.path(filtered_dir, paste0(chr, "_rsid_update.txt"))
  readr::write_tsv(rsid_map, rsid_file, col_names = FALSE)
  return(rsid_file)
}

# Build per-chromosome filtered files once
shared_reference_exists <- file.exists(paste0(all_prefix, ".bed"))
if (shared_reference_exists) {
  log_progress_event(
    "shared_reference",
    "reused_existing_reference",
    status = "completed",
    metrics = list(chromosomes_total = length(chromosomes))
  )
} else {
  log_progress_event(
    "shared_reference",
    "build_started",
    metrics = list(chromosomes_total = length(chromosomes))
  )
  shared_chr_completed <- 0L
  for (chr in chromosomes) {
    cat("\n[Shared] Processing", chr, "\n")
    # Download once if needed
    for (ext in c("bim","bed","fam")) {
      dest <- file.path(raw_dir, paste0(chr, ".", ext))
      if (!file.exists(dest)) {
        src <- paste0(gcs_plink_base, "/", chr, ".", ext)
        res <- system2("gsutil", args = c("-m","-u", Sys.getenv("GOOGLE_PROJECT"), "cp", src, dest), stdout = TRUE, stderr = TRUE)
      }
      if (!file.exists(dest)) stop("Failed to obtain ", dest)
    }
    # Filter to selected participant list and update rsIDs once
    rsid_file <- update_rsid_for_chr(chr)
    out_prefix <- file.path(filtered_dir, chr)
    plink2_args <- c("--bfile", file.path(raw_dir, chr),
                     "--keep", patient_list_file,
                     "--geno", "0.05", "--snps-only", "just-acgt", "--rm-dup", "exclude-all",
                     if (!is.null(rsid_file)) c("--update-name", rsid_file) else NULL,
                     "--keep-allele-order", "--make-bed", "--out", out_prefix)
    res <- system2("plink2", args = plink2_args, stdout = TRUE, stderr = TRUE)
    if (!file.exists(paste0(out_prefix, ".bed"))) stop("plink2 failed for shared ", chr)

    shared_chr_completed <- shared_chr_completed + 1L
    log_progress_event(
      "shared_reference",
      "chromosome_completed",
      metrics = list(
        chromosomes_completed = shared_chr_completed,
        chromosomes_total = length(chromosomes),
        progress_pct = round(100 * shared_chr_completed / length(chromosomes), 2)
      ),
      details = list(chromosome = chr)
    )

    # Cleanup: delete raw chromosome files immediately after filtering
    for (ext in c(".bed", ".bim", ".fam")) {
      raw_file <- paste0(file.path(raw_dir, chr), ext)
      if (file.exists(raw_file)) file.remove(raw_file)
    }
  }
}

# Merge all into a single shared reference once
if (!file.exists(paste0(all_prefix, ".bed"))) {
  first_chr <- chromosomes[1]
  file.copy(paste0(file.path(filtered_dir, first_chr), ".bed"), paste0(all_prefix, ".bed"), overwrite = TRUE)
  file.copy(paste0(file.path(filtered_dir, first_chr), ".bim"), paste0(all_prefix, ".bim"), overwrite = TRUE)
  file.copy(paste0(file.path(filtered_dir, first_chr), ".fam"), paste0(all_prefix, ".fam"), overwrite = TRUE)
  if (length(chromosomes) > 1) {
    for (chr in chromosomes[-1]) {
      temp_merge_prefix <- paste0(all_prefix, "_temp_merge")
      merge_args <- c("--bfile", all_prefix, "--bmerge", file.path(filtered_dir, chr), "--make-bed", "--out", temp_merge_prefix)
      res <- system2("plink", args = merge_args, stdout = TRUE, stderr = TRUE)
      if (!file.exists(paste0(temp_merge_prefix, ".bed"))) stop("Shared merge failed at ", chr)
      file.rename(paste0(temp_merge_prefix, ".bed"), paste0(all_prefix, ".bed"))
      file.rename(paste0(temp_merge_prefix, ".bim"), paste0(all_prefix, ".bim"))
      file.rename(paste0(temp_merge_prefix, ".fam"), paste0(all_prefix, ".fam"))
      for (ext in c(".log", ".nosex", "-merge.missnp")) {
        tf <- paste0(temp_merge_prefix, ext)
        if (file.exists(tf)) file.remove(tf)
      }
    }
  }
  
  # Cleanup: delete all filtered chromosome files and rsid_update files after merge
  cat("\n[Cleanup] Removing intermediate filtered files...\n")
  for (chr in chromosomes) {
    for (ext in c(".bed", ".bim", ".fam", ".log")) {
      filtered_file <- paste0(file.path(filtered_dir, chr), ext)
      if (file.exists(filtered_file)) file.remove(filtered_file)
    }
    # Also remove rsid_update file
    rsid_file <- file.path(filtered_dir, paste0(chr, "_rsid_update.txt"))
    if (file.exists(rsid_file)) file.remove(rsid_file)
  }
  
  # Remove empty directories
  if (dir.exists(raw_dir) && length(list.files(raw_dir, all.files = TRUE, no.. = TRUE)) == 0) {
    unlink(raw_dir, recursive = TRUE)
  }
  if (dir.exists(filtered_dir) && length(list.files(filtered_dir, all.files = TRUE, no.. = TRUE)) == 0) {
    unlink(filtered_dir, recursive = TRUE)
  }
  
  cat("[Cleanup] Complete - only all_chromosomes files remain\n")
}

cat("\n✓ Shared PLINK reference ready:", all_prefix, "\n")
log_progress_event(
  "shared_reference",
  "reference_ready",
  status = "completed",
  metrics = list(chromosomes_total = length(chromosomes)),
  details = list(shared_reference = all_prefix)
)


# In[ ]:


# Process a single chromosome: download, filter, update rsIDs, extract variants
process_chromosome <- function(chr, pgrs_work_dir, pgrs_prep_file, patient_list_path) {
  cat("Starting processing for", chr, "\n")
  
  # Define subdirectories within this PGRS work directory
  raw_dir <- file.path(pgrs_work_dir, "plink_raw")
  filter_dir <- file.path(pgrs_work_dir, "filter")
  filtered_dir <- file.path(pgrs_work_dir, "filtered")
  habshd_dir <- file.path(pgrs_work_dir, "habshd")
  
  dir.create(raw_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(filter_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(filtered_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(habshd_dir, showWarnings = FALSE, recursive = TRUE)
  
  # Download chromosome files
  cat("  Downloading files for", chr, "\n")
  for (file_type in c("bim", "bed", "fam")) {
    dest_file <- file.path(raw_dir, paste0(chr, ".", file_type))
    source_file <- paste0(gcs_plink_base, "/", chr, ".", file_type)
    
    if (!file.exists(dest_file)) {
      result <- system2("gsutil", args = c("-m", "-u", Sys.getenv("GOOGLE_PROJECT"), "cp", source_file, dest_file), 
                       stdout = TRUE, stderr = TRUE)
      if (!file.exists(dest_file)) {
         warning("Failed to download ", dest_file)
         return(NULL)
      }
    }
  }
  
  # Process .bim file
  bim_file_path <- file.path(raw_dir, paste0(chr, ".bim"))
  bim.raw <- read_tsv(bim_file_path, col_names = FALSE, show_col_types = FALSE) %>%
    janitor::clean_names() %>%
    dplyr::rename(chrom = x1, id = x2, cm = x3, pos = x4, ref = x6, alt = x5)
  
  # Simulate GWAS data for MungeSumstats
  dat_simple <- GWASBrewer::sim_mv(G = 1, N = 2578, J = nrow(bim.raw), h2 = 0.5, pi = 0.01, 
                                   est_s = TRUE, af = function(n){rbeta(n, 1, 5)})
  
  bim <- dplyr::bind_cols(bim.raw, dat_simple$beta_hat, dat_simple$se_beta_hat, dat_simple$snp_info) %>%
    tibble::as_tibble() %>%
    dplyr::rename(beta = ...7, se = ...8) %>%
    dplyr::select(-SNP) %>%
    dplyr::mutate(Z = beta / se, P = 2 * (1 - stats::pt(abs(Z), Inf)))
  
  # MungeSumstats
  reformatted <- MungeSumstats::format_sumstats(path = bim, ref_genome = "GRCh38", return_data = TRUE,
                                                bi_allelic_filter = FALSE, N_dropNA = FALSE, rmv_chr = NULL,
                                                allele_flip_check = FALSE, allele_flip_drop = FALSE,
                                                allele_flip_frq = FALSE, dbSNP = 155) %>% tibble::as_tibble()
  
  rsid_test_file <- file.path(filter_dir, paste0(chr, "_rsid_test.txt"))
  reformatted %>% dplyr::select(ID, SNP) %>% write_tsv(rsid_test_file, col_names = FALSE)
  
  # PLINK2: filter and update rsIDs
  plink2_args <- c("--bfile", file.path(raw_dir, chr), "--keep", patient_list_path, 
                   "--geno", "0.05", "--snps-only", "just-acgt", "--rm-dup", "exclude-all",
                   "--update-name", rsid_test_file, "--keep-allele-order", "--make-bed",
                   "--out", file.path(filtered_dir, chr))
  result <- system2("plink2", args = plink2_args, stdout = TRUE, stderr = TRUE)
  if (!file.exists(paste0(file.path(filtered_dir, chr), ".bed"))) {
      warning("PLINK2 failed for ", chr)
      return(NULL)
  }
  
  # Extract variants in this PGRS file
  variant_list <- read_tsv(pgrs_prep_file, col_names = FALSE, show_col_types = FALSE)
  rsids_to_keep_file <- file.path(filter_dir, "rsids_to_keep.txt")
  write.table(variant_list[[2]], file = rsids_to_keep_file, row.names = FALSE, col.names = FALSE, quote = FALSE)
  
  # PLINK: extract and create final bed
  plink_args <- c("--bfile", file.path(filtered_dir, chr), "--extract", rsids_to_keep_file, 
                  "--make-bed", "--out", file.path(habshd_dir, chr))
  result <- system2("plink", args = plink_args, stdout = TRUE, stderr = TRUE)
  if (!file.exists(paste0(file.path(habshd_dir, chr), ".bed"))) {
      warning("Final PLINK failed for ", chr)
      return(NULL)
  }
  
  cat("  ✓ Processing complete for", chr, "\n")
  return(file.path(habshd_dir, chr))
}

cat("✓ process_chromosome() defined\n")


# In[ ]:


# Merge a chromosome file with the cumulative 'all' file
merge_chromosome_file <- function(chr_file_prefix, pgrs_work_dir) {
  all_dir <- file.path(pgrs_work_dir, "all")
  dir.create(all_dir, showWarnings = FALSE, recursive = TRUE)
  all_file_prefix <- file.path(all_dir, "all_chromosomes")
  
  if (!file.exists(paste0(chr_file_prefix, ".bed"))) {
    warning("Input file for merging not found:", paste0(chr_file_prefix, ".bed"))
    return()
  }
  
  if (!file.exists(paste0(all_file_prefix, ".bed"))) {
    # Create initial merged file
    file.copy(paste0(chr_file_prefix, ".bed"), paste0(all_file_prefix, ".bed"), overwrite = TRUE)
    file.copy(paste0(chr_file_prefix, ".bim"), paste0(all_file_prefix, ".bim"), overwrite = TRUE)
    file.copy(paste0(chr_file_prefix, ".fam"), paste0(all_file_prefix, ".fam"), overwrite = TRUE)
    cat("  Created initial 'all' file\n")
  } else {
    # Merge with existing
    temp_merge_prefix <- paste0(all_file_prefix, "_temp_merge")
    merge_args <- c("--bfile", all_file_prefix, "--bmerge", chr_file_prefix, "--make-bed", "--out", temp_merge_prefix)
    result <- system2("plink", args = merge_args, stdout = TRUE, stderr = TRUE)
    
    if (file.exists(paste0(temp_merge_prefix, ".bed"))) {
        file.rename(paste0(temp_merge_prefix, ".bed"), paste0(all_file_prefix, ".bed"))
        file.rename(paste0(temp_merge_prefix, ".bim"), paste0(all_file_prefix, ".bim"))
        file.rename(paste0(temp_merge_prefix, ".fam"), paste0(all_file_prefix, ".fam"))
        # Cleanup temp files
        for (ext in c(".log", ".nosex", "-merge.missnp")) {
          temp_file <- paste0(temp_merge_prefix, ext)
          if (file.exists(temp_file)) file.remove(temp_file)
        }
        cat("  Merged successfully\n")
    } else {
        warning("Merge failed for ", basename(chr_file_prefix))
    }
  }
}

cat("✓ merge_chromosome_file() defined\n")


# In[ ]:


# Clean up intermediate chromosome files to save space
clean_up_files <- function(chr, pgrs_work_dir) {
  for (subdir in c("plink_raw", "filtered", "habshd", "filter")) {
    chr_prefix <- file.path(pgrs_work_dir, subdir, chr)
    for (ext in c(".bed", ".bim", ".fam", ".log", ".nosex")) {
      f <- paste0(chr_prefix, ext)
      if (file.exists(f)) file.remove(f)
    }
  }
  # Remove rsid files
  rsid_file <- file.path(pgrs_work_dir, "filter", paste0(chr, "_rsid_test.txt"))
  if (file.exists(rsid_file)) file.remove(rsid_file)
}

cat("✓ clean_up_files() defined\n")


# ## Main Processing Loop - Score Each PGRS Against Shared Reference
# This loop prepares each PGRS weight file and computes scores using the shared `all_chromosomes` PLINK reference. No per-PGRS chromosome downloads or merges.

# In[ ]:


cat("\n=== STARTING MAIN PGRS PROCESSING LOOP ===\n\n")

shared_all_prefix <- file.path(output_base_dir, "shared", shared_ref_tag, "all_chromosomes")
if (!file.exists(paste0(shared_all_prefix, ".bed"))) {
  stop("Shared reference not found: ", shared_all_prefix, ".bed — run the shared build cell above.")
}

completed_pgrs_files <- 0L
for (pgrs_file in pgrs_files) {
  pgrs_name <- tools::file_path_sans_ext(basename(pgrs_file))
  if (grepl("\\.gz$", pgrs_name)) pgrs_name <- tools::file_path_sans_ext(pgrs_name)
  
  cat("\n############################################\n")
  cat("### Processing PGRS:", pgrs_name, "###\n")
  cat("############################################\n\n")
  
  # Create work directory for this PGRS (ensure subdirs for future extensions)
  pgrs_work_dir <- file.path(output_base_dir, pgrs_name)
  if (!dir.exists(pgrs_work_dir)) dir.create(pgrs_work_dir, recursive = TRUE, showWarnings = FALSE)
  temp_dir <- file.path(pgrs_work_dir, "temp")
  if (!dir.exists(temp_dir)) dir.create(temp_dir, recursive = TRUE, showWarnings = FALSE)
  
  # Prepare PGRS file (header stripping; detect column indices)
  prep_result <- tryCatch({
    prepare_pgrs_file(pgrs_file, pgrs_work_dir)
  }, error = function(e) {
    cat("ERROR preparing PGRS file:", conditionMessage(e), "\n")
    return(NULL)
  })
  if (is.null(prep_result)) {
    cat("Skipping", pgrs_name, "due to preparation error\n")
    next
  }
  pgrs_prep_file <- prep_result$prepared_file
  format_info <- prep_result$format_info
  variant_count <- prep_result$variant_count
  log_progress_event(
    "pgrs_scoring",
    "score_started",
    metrics = list(
      files_completed = completed_pgrs_files,
      files_total = length(pgrs_files),
      participants_targeted = nrow(patient_list),
      variant_count = variant_count
    ),
    details = list(pgrs_name = pgrs_name)
  )

  # Calculate PGRS
  cat("\n--- Calculating PGRS Score ---\n")
  pgrs_output_file <- file.path(pgrs_work_dir, paste0(pgrs_name, "_PGRS.txt"))
  plink_score_args <- c(
    "--bfile", shared_all_prefix,
    "--score", pgrs_prep_file,
    as.character(format_info$rsid_col),
    as.character(format_info$effect_allele_col),
    as.character(format_info$weight_col),
    "sum",
    "--out", file.path(temp_dir, paste0(pgrs_name, "_PGRS_temp"))
  )
  result <- system2("plink", args = plink_score_args, stdout = TRUE, stderr = TRUE)
  profile_file <- paste0(file.path(temp_dir, paste0(pgrs_name, "_PGRS_temp")), ".profile")
  if (file.exists(profile_file)) {
    file.rename(profile_file, pgrs_output_file)
    cat("✓ PGRS calculated successfully:", pgrs_output_file, "\n")
    completed_pgrs_files <- completed_pgrs_files + 1L
    log_progress_event(
      "pgrs_scoring",
      "score_completed",
      status = "completed",
      metrics = list(
        files_completed = completed_pgrs_files,
        files_total = length(pgrs_files),
        participants_targeted = nrow(patient_list),
        variant_count = variant_count,
        progress_pct = round(100 * completed_pgrs_files / length(pgrs_files), 2)
      ),
      details = list(
        pgrs_name = pgrs_name,
        score_file = pgrs_output_file
      )
    )
    for (ext in c(".log", ".nosex")) {
      f <- paste0(file.path(temp_dir, paste0(pgrs_name, "_PGRS_temp")), ext)
      if (file.exists(f)) file.remove(f)
    }
  } else {
    cat("✗ PGRS calculation failed\n")
    log_progress_event(
      "pgrs_scoring",
      "score_failed",
      status = "failed",
      metrics = list(
        files_completed = completed_pgrs_files,
        files_total = length(pgrs_files),
        participants_targeted = nrow(patient_list),
        variant_count = variant_count
      ),
      details = list(pgrs_name = pgrs_name)
    )
  }
  
  cat("\n✓ Finished processing:", pgrs_name, "\n")
}

cat("\n=== ALL PGRS FILES PROCESSED ===\n")
log_progress_event(
  "pgrs_scoring",
  "all_scores_completed",
  status = "completed",
  metrics = list(
    files_completed = completed_pgrs_files,
    files_total = length(pgrs_files),
    participants_targeted = nrow(patient_list)
  )
)


# ## Summary
# 
# Check the `processed_data/PGRS/` directory for results:
# - `processed_data/PGRS/shared/` - Shared resources (ancestry, patient lists)
# - `processed_data/PGRS/PGS002196-average/PGS002196-average_PGRS.txt` - PGRS scores
# - `processed_data/PGRS/PGS002209_hmPOS_GRCh38/PGS002209_hmPOS_GRCh38_PGRS.txt` - PGRS scores
# - `processed_data/PGRS/chronotype_meta_PRS_model/chronotype_meta_PRS_model_PGRS.txt` - PGRS scores
