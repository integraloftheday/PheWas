source("./scripts/dependencies.r")



plink_analysis <- function(config = list(
    base_path = "PGRS_Average",
    parquet_file = "person_ids.parquet",
    pgs_url = "https://ftp.ebi.ac.uk/pub/databases/spot/pgs/scores/PGS002196/ScoringFiles/PGS002196.txt.gz",
    chromosomes = paste0("chr", c(1:22, "X", "Y"))
)) {


source("./scripts/plink-analysis-helpers.r")

# --- Configuration ---
# Define the base path for all generated files and directories
base_path <- config$base_path

# Define the location of the input Parquet file (adjust path if necessary)
parquet_file <- config$parquet_file
# Ensure this file exists in your working directory or provide the full path.

# Define PGS Score File URL and standardized names
pgs_url <- config$pgs_url
pgs_gz_file <- file.path(base_path, "resources", "pgs_score_raw.txt.gz")
pgs_unzipped_file <- file.path(base_path, "resources", "pgs_score_unzipped.txt") # Intermediate, can be removed later if needed
pgs_no_header_file <- file.path(base_path, "data", "pgs_score_no_header.txt") # Used as input for PLINK score

# --- Create Necessary Directories ---
cat("Creating required directories under:", base_path, "\n")
dir.create(base_path, showWarnings = FALSE)
dir.create(file.path(base_path, "resources"), showWarnings = FALSE)
dir.create(file.path(base_path, "data"), showWarnings = FALSE)
dir.create(file.path(base_path, "plinkfiles"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(base_path, "plinkfiles", "filter"), showWarnings = FALSE, recursive = TRUE)
dir.create(file.path(base_path, "plinkfiles", "filtered"), showWarnings = FALSE, recursive = TRUE) # For intermediate PLINK outputs
dir.create(file.path(base_path, "plinkfiles", "habshd"), showWarnings = FALSE, recursive = TRUE)   # For intermediate PLINK outputs
dir.create(file.path(base_path, "plinkfiles", "all"), showWarnings = FALSE, recursive = TRUE)      # For merged PLINK files
dir.create(file.path(base_path, "plinkfiles", "PGRS"), showWarnings = FALSE, recursive = TRUE)    # For final PGRS score output
# --- Download and Prepare PGS Score File ---
cat("Downloading PGS score file...\n")
# Check if file exists before downloading
if (!file.exists(pgs_gz_file)) {
  download.file(pgs_url, destfile = pgs_gz_file)
  cat("Download complete.\n")
} else {
  cat("PGS score file already exists:", pgs_gz_file, "\n")
}

cat("Unzipping and preparing PGS score file...\n")
# Unzip the file using R's built-in gzfile function
gzfile_con <- gzfile(pgs_gz_file, "r")
unzipped_lines <- readLines(gzfile_con)
close(gzfile_con)

# Write unzipped content to an intermediate file (optional step, could process directly)
# writeLines(unzipped_lines, pgs_unzipped_file)

# Remove header lines (starting with '#') and save to the final input file
lines_no_header <- unzipped_lines[!grepl("^#", unzipped_lines)]
writeLines(lines_no_header, pgs_no_header_file)
cat("PGS score file prepared (header removed):", pgs_no_header_file, "\n")

# Clean up intermediate unzipped lines object
rm(unzipped_lines, lines_no_header)
gc()

# --- Obtain Ancestry Data ---
ancestry_dest_path <- file.path(base_path, "plinkfiles", "filter", "ancestry_preds.tsv")
cat("Checking/Downloading Ancestry data...\n")
if (!file.exists(ancestry_dest_path)) {
  # Now, execute the gsutil cp command
  gsutil_ancestry_args <- c("-u", Sys.getenv("GOOGLE_PROJECT"),
                            "cp",
                            "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/aux/ancestry/ancestry_preds.tsv",
                            ancestry_dest_path)
  cat("Running gsutil command:", paste("gsutil", paste(gsutil_ancestry_args, collapse=" ")), "\n")                      
  result_ancestry <- system2("gsutil", 
                             args = gsutil_ancestry_args,
                             stdout = TRUE,
                             stderr = TRUE)
  # Check result (basic check: does the file exist now?)
  if (!file.exists(ancestry_dest_path)) {
      stop("Failed to download ancestry data. gsutil output:\n", paste(result_ancestry, collapse="\n"))
  } else {
      cat("Ancestry data downloaded successfully.\n")
  }
} else {
  cat("Ancestry data already exists:", ancestry_dest_path, "\n")
}
# --- Prepare Patient List ---
cat("Preparing patient list...\n")

# Step 1: Load unique person_ids from the Parquet file
cat("Loading unique person IDs from:", parquet_file, "\n")
if (!file.exists(parquet_file)) {
    stop("Parquet file not found: ", parquet_file)
}
all_data <- read_parquet(parquet_file, col_select = "person_id")
person_ids_df <- data.frame(person_id = unique(all_data$person_id)) # Ensure it's a dataframe
cat("Loaded", nrow(person_ids_df), "unique person IDs.\n")
# Remove large object immediately and call garbage collector
rm(all_data)
gc() 
cat("Removed large Parquet object from memory.\n")
# Step 2: Read the chr21.fam file (Assuming this is a template or one example)
# Note: This uses chr21.fam for structure, but filters based on all unique person IDs and EUR ancestry.
# Ensure chr21.fam is available or adjust logic if needed.
# Let's assume it needs downloading like other chromosome files.
chr21_fam_local_path <- file.path(base_path, "plinkfiles", "chr21.fam") # Store it in base plinkfiles dir
chr21_fam_gcs_path <- "gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/clinvar/plink_bed/chr21.fam" # Example GCS path

cat("Checking/Downloading chr21.fam for patient list filtering...\n")
if (!file.exists(chr21_fam_local_path)) {
  gsutil_fam_args <- c("-u", Sys.getenv("GOOGLE_PROJECT"),
                       "cp",
                       chr21_fam_gcs_path,
                       chr21_fam_local_path)
  cat("Running gsutil command:", paste("gsutil", paste(gsutil_fam_args, collapse=" ")), "\n")                       
  result_fam <- system2("gsutil", 
                        args = gsutil_fam_args,
                        stdout = TRUE,
                        stderr = TRUE)
  if (!file.exists(chr21_fam_local_path)) {
      stop("Failed to download chr21.fam. gsutil output:\n", paste(result_fam, collapse="\n"))
  } else {
      cat("chr21.fam downloaded successfully.\n")
  }
} else {
  cat("chr21.fam already exists:", chr21_fam_local_path, "\n")
}

fam_data <- read_table(chr21_fam_local_path, col_names = FALSE)
colnames(fam_data) <- c("FID", "IID", "PID", "MID", "SEX", "PHENOTYPE")

# Load ancestry data
ancestry <- read_tsv(ancestry_dest_path)

# Filter only ancestry = eur 
eur_ids <- ancestry %>%
  filter(ancestry_pred == "eur") %>%
  pull(research_id)

# Step 3: Filter fam_data: Keep IIDs present in both EUR ancestry list AND the unique person_id list from Parquet
filtered_fam <- fam_data[fam_data$IID %in% eur_ids & fam_data$IID %in% person_ids_df$person_id, ]

# Step 4: Create the patient_list dataframe with FID and IID
patient_list <- filtered_fam[, c("FID", "IID")]

# Step 5: Write the patient_list to a file
patient_list_file <- file.path(base_path, "plinkfiles", "filter", "patient_list.txt")
write_tsv(patient_list, patient_list_file, col_names = FALSE)

cat("Patient list created with", nrow(patient_list), "individuals (EUR ancestry and in Parquet data).\n")
cat("File saved as:", patient_list_file, "\n")

# Clean up intermediate objects
rm(fam_data, ancestry, eur_ids, filtered_fam, person_ids_df)
gc()
# Define the list of chromosomes you want to process
chromosomes <- config$chromosomes


# --- Main Processing Loop ---
cat("\n=== Starting Chromosome Processing ===\n")
total_chromosomes <- length(chromosomes)
all_chromosomes_prefix <- file.path(base_path, "plinkfiles", "all", "all_chromosomes")
final_habshd_files <- list() # To store paths of successfully processed chromosome files
total_chromosomes
cat("\n=== Starting Chromosome Processing For Loop===\n")
for (i in seq_along(chromosomes)) {
    cat("\n=== Starting Chromosome Processing inner Loop ===\n")
  chr <- chromosomes[i]
  cat("\n--- Processing chromosome", i, "of", total_chromosomes, ":", chr, "---\n")
  
  # Process the chromosome
  # Pass necessary file paths to the function
  chr_final_prefix <- tryCatch({
      process_chromosome(chr = chr, 
                         base_path = base_path, 
                         pgs_input_file = pgs_no_header_file, 
                         patient_list_path = patient_list_file)
  }, error = function(e) {
      cat("Error processing chromosome", chr, ": ", conditionMessage(e), "\n")
      return(NULL) # Indicate failure
  })

  # If processing was successful, proceed to merge and cleanup
  if (!is.null(chr_final_prefix)) {
      final_habshd_files[[chr]] <- chr_final_prefix # Store prefix if successful
      cat("\nMerging", chr, "into all file\n")
      merge_chromosome_file(chr_final_prefix, base_path)
      
      cat("\nCleaning up intermediate files for", chr, "\n")
      clean_up_files(chr, base_path)
  } else {
      cat("\nSkipping merge and cleanup for failed chromosome:", chr, "\n")
  }

  cat("\n--- Finished processing attempt for", chr, "---\n")
  cat("\nProgress:", i, "out of", total_chromosomes, "chromosomes attempted\n")
}

cat("\n=== Chromosome Processing Loop Finished ===\n")

# Check if the final merged file exists
if (file.exists(paste0(all_chromosomes_prefix, ".bed"))) {
    cat("\nAll successfully processed chromosomes have been merged into:", all_chromosomes_prefix, "\n")

    # --- Calculate Polygenic Risk Score using PLINK ---
    cat("\nCalculating Polygenic Risk Score using the merged file...\n")
    
    # Define PLINK command for scoring
    plink_score_command <- "plink"
    plink_score_out_prefix <- file.path(base_path, "plinkfiles", "PGRS", "temp_PGRS_calc") # Temporary prefix
    final_pgrs_output_file <- file.path(base_path, "plinkfiles", "PGRS", "PGRS.txt") # Final desired name

    # Arguments for --score: 
    # file, rsid col, effect allele col, effect weight col, options (header, sum)
    # Check columns in pgs_no_header_file: Needs rsID, effect_allele, effect_weight
    # Assuming standard PGS Catalog format: rsID=col 2, effect_allele=col 3, effect_weight=col 5 (adjust if different!)
    # Let's re-read the header of the *original* unzipped file to be sure
    gzfile_con_check <- gzfile(pgs_gz_file, "r")
    header_lines <- readLines(gzfile_con_check, n=20) # Read first 20 lines
    close(gzfile_con_check)
    score_header <- header_lines[grepl("^rsID", header_lines) | grepl("^effect_allele", header_lines)] # Find the header line
    print("Detected Score File Header:")
    print(score_header)
    # Based on PGS002196.txt.gz format:
    # #chr_name chr_position rsID effect_allele other_allele effect_weight ...
    # So: rsID=col 3, effect_allele=col 4, effect_weight=col 6
    # HOWEVER, the original script used 1 4 6. Let's stick to that assuming it was correct for the *intended* file (PGS002209?)
    # Let's use 1 4 6 as in the original script, but add a note.
    # **IMPORTANT**: Verify these column numbers (1, 4, 6) are correct for the actual score file used (PGS002196)!
    # Column 1: rsID (Assumed, check file)
    # Column 4: effect_allele (Assumed, check file)
    # Column 6: effect_weight (Assumed, check file)
    
    plink_score_args <- c(
      "--bfile", all_chromosomes_prefix, # Use the final merged file
      "--score", pgs_no_header_file, "1", "4", "6", # Use prepared score file & ASSUMED columns 1,4,6
      "header", # Indicate the score file has a header (though we removed '#' lines, the first data line is header)
      "sum",    # Calculate sum score per individual
      "--out", plink_score_out_prefix # Output prefix for PLINK score results
    )

    cat("Executing: plink", paste(plink_score_args, collapse=" "), "\n")
    result_score <- system2(plink_score_command, args = plink_score_args, stdout = TRUE, stderr = TRUE)

    # Print the output/log from PLINK scoring
    cat("PLINK scoring command output:\n")
    cat(paste(result_score, collapse = "\n"), "\n")

    # Rename the output file (PLINK typically creates .sscore for sum scores)
     original_plink_score_output <- paste0(plink_score_out_prefix, ".profile")
    cat("Checking for PLINK output file:", original_plink_score_output, "\n")
    if (file.exists(original_plink_score_output)) {
      file.rename(original_plink_score_output, final_pgrs_output_file)
      cat("Successfully calculated PGRS. Renamed PLINK output to:", final_pgrs_output_file, "\n")
      # Clean up log file if exists
      log_file <- paste0(plink_score_out_prefix, ".log")
      if(file.exists(log_file)) file.remove(log_file)
      nosex_file <- paste0(plink_score_out_prefix, ".nosex")
       if(file.exists(nosex_file)) file.remove(nosex_file)

    } else {
      cat("Warning: Expected PLINK score output file not found:", original_plink_score_output, "\n")
      cat("Check PLINK output above for errors.\n")
    }

} else {
    cat("\nError: The final merged file", paste0(all_chromosomes_prefix, ".bed"), "was not created. Cannot calculate PGRS.\n")
    cat("Review the logs from the chromosome processing and merging steps.\n")
}


cat("\n--- Script Finished ---\n")
}