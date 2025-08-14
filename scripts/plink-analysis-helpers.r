# Function to process a single chromosome
process_chromosome <- function(chr, base_path, pgs_input_file, patient_list_path) {
  cat("Starting processing for", chr, "\n")

  # Define subdirectories
  raw_dir <- file.path(base_path)
  filter_dir <- file.path(base_path, "filter")
  filtered_dir <- file.path(base_path, "filtered")
  habshd_dir <- file.path(base_path, "habshd")

  # Create directories
  dir.create(filter_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(filtered_dir, showWarnings = FALSE, recursive = TRUE)
  dir.create(habshd_dir, showWarnings = FALSE, recursive = TRUE)

  # Download files
  cat("Downloading files for", chr, "\n")
  file_types <- c("bim", "bed", "fam")
  for (file_type in file_types) {
    cat("  Downloading", file_type, "file\n")
    dest_file <- file.path(raw_dir, paste0(chr, ".", file_type))
    source_file <- paste0("gs://fc-aou-datasets-controlled/v8/wgs/short_read/snpindel/clinvar/plink_bed/", chr, ".", file_type)
    
    result <- system2("gsutil",
                      args = c("-m","-u", Sys.getenv("GOOGLE_PROJECT"),
                               "cp",
                               source_file,
                               dest_file),
                      stdout = TRUE,
                      stderr = TRUE)
    # Add basic error check for download
    if (!file.exists(dest_file)) {
       warning("Failed to download ", dest_file, " from ", source_file)
       # Consider stopping execution if download fails: stop("Download failed for ", chr)
    }
    cat("  Download complete\n")
  }

  # Process .bim file
  cat("Processing .bim file for", chr, "\n")
  bim_file_path <- file.path(raw_dir, paste0(chr, ".bim"))
  bim.raw <- readr::read_tsv(bim_file_path, col_names = FALSE, show_col_types = FALSE) %>%
    janitor::clean_names() %>%
    dplyr::rename(chrom = x1, id = x2, cm = x3, pos = x4, ref = x6, alt = x5)
  cat("  .bim file processed\n")

  # Simulate GWAS data
  cat("Simulating GWAS data for", chr, "\n")
  # Ensure GWASBrewer is available and loaded
  if (!requireNamespace("GWASBrewer", quietly = TRUE)) {
      stop("Package 'GWASBrewer' is required but not installed.")
  }
  dat_simple <- GWASBrewer::sim_mv(G = 1, N = 2578, J = nrow(bim.raw), h2 = 0.5, pi = 0.01, est_s = TRUE, af = function(n){rbeta(n, 1, 5)})
  cat("  GWAS data simulation complete\n")

  bim <- dplyr::bind_cols(
    bim.raw, dat_simple$beta_hat, dat_simple$se_beta_hat, dat_simple$snp_info
  ) %>%
    tibble::as_tibble() %>%
    dplyr::rename(beta = ...7, se = ...8) %>% # Adjusted renaming based on typical bind_cols behavior
    dplyr::select(-SNP) %>%
    dplyr::mutate(
      Z = beta / se,
      P = 2 * (1 - stats::pt(abs(Z), Inf)) # Use stats::pt for clarity
    )

  # MungeSumstats
  cat("Running MungeSumstats for", chr, "\n")
  # Ensure MungeSumstats is available and loaded
  if (!requireNamespace("MungeSumstats", quietly = TRUE)) {
      stop("Package 'MungeSumstats' is required but not installed.")
  }
  # Check if input 'bim' has required columns for MungeSumstats
  required_cols <- c("id", "chrom", "pos", "alt", "ref", "beta", "se", "P")
  if (!all(required_cols %in% names(bim))) {
      stop("Input data frame 'bim' is missing required columns for MungeSumstats. Found: ", paste(names(bim), collapse=", "))
  }
  
  # Create a temporary file for MungeSumstats if needed, or pass data directly
  # MungeSumstats::format_sumstats can accept a data.frame directly
  reformatted <- MungeSumstats::format_sumstats(path = bim, # Pass the data frame directly
                                                ref_genome = "GRCh38",
                                                return_data = TRUE,
                                                bi_allelic_filter = FALSE, # Explicitly set to FALSE
                                                N_dropNA = FALSE, # Explicitly set to FALSE
                                                rmv_chr = NULL,   # Explicitly set to NULL
                                                allele_flip_check = FALSE,
                                                allele_flip_drop = FALSE,
                                                allele_flip_frq = FALSE,
                                                dbSNP = 155) %>%
    tibble::as_tibble()
  cat("  MungeSumstats complete\n")

  rsid_test_file <- file.path(filter_dir, paste0(chr, "_rsid_test.txt"))
  reformatted %>%
    dplyr::select(ID, SNP) %>% # Assuming 'ID' is the original ID and 'SNP' is the updated rsID from MungeSumstats
    readr::write_tsv(rsid_test_file, col_names = FALSE)

  # PLINK2 command
  cat("Running PLINK2 command for", chr, "\n")
  plink2_args <- c(
    "--bfile", file.path(raw_dir, chr),
    "--keep", patient_list_path, # Use parameterized patient list path
    "--geno", "0.05",
    "--snps-only", "just-acgt", # Often recommended with --snps-only
    "--rm-dup", "exclude-all",
    "--update-name", rsid_test_file, # Path to the rsID update file
    "--keep-allele-order",
    "--make-bed",
    "--out", file.path(filtered_dir, chr) # Output to filtered directory
  )
  result_plink2 <- system2("plink2", args = plink2_args, stdout = TRUE, stderr = TRUE)
  # Add basic check for plink2 output files
  if (!file.exists(paste0(file.path(filtered_dir, chr), ".bed"))) {
      warning("PLINK2 output file not found for ", chr, ". Check PLINK2 log.")
      print(result_plink2) # Print output/error from system2 call
  }
  cat("  PLINK2 command complete\n")

  # Filter by required variants
  cat("Filtering by required variants for", chr, "\n")
  variant_list <- readr::read_tsv(pgs_input_file, show_col_types = FALSE) # Use parameterized PGS input file path
  rsids_to_keep_file <- file.path(filter_dir, "rsids_to_keep.txt") # Place temp file in filter dir
  # Check if rsID column exists before writing
  if (!"rsID" %in% names(variant_list)) {
      stop("Column 'rsID' not found in the file: ", pgs_input_file)
  }
  write.table(variant_list$rsID, file = rsids_to_keep_file, row.names = FALSE, col.names = FALSE, quote = FALSE)

  # PLINK command (using plink 1.9 often for basic filtering)
  cat("Running final PLINK command for", chr, "\n")
  plink_args <- c(
    "--bfile", file.path(filtered_dir, chr), # Input from filtered directory
    "--extract", rsids_to_keep_file,         # Use the rsID list file
    "--make-bed",
    "--out", file.path(habshd_dir, chr)       # Final output to habshd directory
  )
  result_plink1 <- system2("plink", args = plink_args, stdout = TRUE, stderr = TRUE)
   # Add basic check for plink output files
  if (!file.exists(paste0(file.path(habshd_dir, chr), ".bed"))) {
      warning("Final PLINK output file not found for ", chr, ". Check PLINK log.")
      print(result_plink1) # Print output/error from system2 call
  }
  cat("  Final PLINK command complete\n")

  cat("Processing complete for", chr, "\n")
  # Return the path prefix of the final bed file
  return(file.path(habshd_dir, chr))
}


# Function to merge chromosome file with existing 'all' file
merge_chromosome_file <- function(chr_file_prefix, base_path) {
  #all_dir <- file.path(base_path, "all")
  all_dir <- file.path(base_path, "plinkfiles", "all")
  dir.create(all_dir, showWarnings = FALSE, recursive = TRUE)
  all_file_prefix <- file.path(all_dir, "all_chromosomes")

  # Check if the input chromosome file exists before proceeding
  if (!file.exists(paste0(chr_file_prefix, ".bed"))) {
    warning("Input file for merging not found: ", paste0(chr_file_prefix, ".bed"))
    return() # Stop merging if input is missing
  }

  if (!file.exists(paste0(all_file_prefix, ".bed"))) {
    cat("Creating initial 'all' file from", basename(chr_file_prefix), "\n")
    # If 'all' file doesn't exist, just copy the chromosome file components
    file.copy(paste0(chr_file_prefix, ".bed"), paste0(all_file_prefix, ".bed"), overwrite = TRUE)
    file.copy(paste0(chr_file_prefix, ".bim"), paste0(all_file_prefix, ".bim"), overwrite = TRUE)
    file.copy(paste0(chr_file_prefix, ".fam"), paste0(all_file_prefix, ".fam"), overwrite = TRUE)
    cat("Initial 'all' file created\n")
  } else {
    cat("Merging", basename(chr_file_prefix), "with existing 'all' file\n")
    # Define temporary output prefix for merging
    temp_merge_prefix <- paste0(all_file_prefix, "_temp_merge")

    # Construct plink merge command arguments
    merge_args <- c(
      "--bfile", all_file_prefix,     # Existing merged file
      "--bmerge", chr_file_prefix,    # New chromosome file to merge
      "--make-bed",
      "--out", temp_merge_prefix      # Temporary output prefix
    )

    merge_command <- "plink" # Assuming plink 1.9 for merging
    result <- system2(merge_command, args = merge_args, stdout = TRUE, stderr = TRUE)
    cat("Merge command intermediate step complete\n")

    # Check if merge was successful (plink creates .bed, .bim, .fam)
    if (file.exists(paste0(temp_merge_prefix, ".bed"))) {
        # Replace old 'all' files with new merged files
        file.rename(paste0(temp_merge_prefix, ".bed"), paste0(all_file_prefix, ".bed"))
        file.rename(paste0(temp_merge_prefix, ".bim"), paste0(all_file_prefix, ".bim"))
        file.rename(paste0(temp_merge_prefix, ".fam"), paste0(all_file_prefix, ".fam"))
        # Clean up temporary log/nosex files if they exist
        if (file.exists(paste0(temp_merge_prefix, ".log"))) file.remove(paste0(temp_merge_prefix, ".log"))
        if (file.exists(paste0(temp_merge_prefix, ".nosex"))) file.remove(paste0(temp_merge_prefix, ".nosex"))
         # Check for plink's automatic handling of mismatched SNPs (-merge-list)
        if (file.exists(paste0(temp_merge_prefix, "-merge.missnp"))) file.remove(paste0(temp_merge_prefix, "-merge.missnp"))
        cat("'all' file updated successfully\n")
    } else {
        warning("PLINK merge failed. Check output/log:\n", paste(result, collapse="\n"))
        # Clean up potential merge failure artifacts like -merge.missnp
        if (file.exists(paste0(all_file_prefix, "-merge.missnp"))) {
             cat("Attempting to handle merge failure (e.g. mismatched SNPs). Check ", paste0(all_file_prefix, "-merge.missnp"), "\n")
             # NOTE: Automatic handling of merge failures is complex.
             # User might need to manually intervene based on plink output/log.
             # Basic cleanup: remove the missnp file if it exists from a failed direct merge attempt
             file.remove(paste0(all_file_prefix, "-merge.missnp"))
        }
         # Clean up potential temp files even on failure
        if (file.exists(paste0(temp_merge_prefix, ".log"))) file.remove(paste0(temp_merge_prefix, ".log"))
        if (file.exists(paste0(temp_merge_prefix, ".nosex"))) file.remove(paste0(temp_merge_prefix, ".nosex"))
    }
  }
}


# Function to clean up intermediate chromosome files
clean_up_files <- function(chr, base_path) {
  cat("Cleaning up intermediate files for", chr, "\n")

  # Define expected file prefixes and directories
  raw_prefix <- file.path(base_path, chr)
  filtered_prefix <- file.path(base_path, "filtered", chr)
  habshd_prefix <- file.path(base_path, "habshd", chr)
  filter_files <- c(
      file.path(base_path, "filter", paste0(chr, "_rsid_test.txt"))
      # Note: rsids_to_keep.txt is common, maybe don't delete it here? Or handle outside loop.
      # file.path(base_path, "filter", "rsids_to_keep.txt")
  )

  # List of file extensions to remove for plink file sets
  extensions <- c(".bed", ".bim", ".fam", ".log", ".nosex") # Include common plink output files

  # Function to safely remove a file if it exists
  safe_remove <- function(file_path) {
    if (file.exists(file_path)) {
      file.remove(file_path)
      # cat("  Removed:", file_path, "\n") # Optional: uncomment for verbose cleanup
    }
  }

  # Remove files for each prefix and extension combination
  for (prefix in c(raw_prefix, filtered_prefix, habshd_prefix)) {
    for (ext in extensions) {
      safe_remove(paste0(prefix, ext))
    }
  }

  # Remove specific filter files
  for (f_file in filter_files) {
      safe_remove(f_file)
  }

  # Optional: Remove plink2 intermediate files if any (.pgen, .pvar, .psam if --make-pgen used)
  # safe_remove(paste0(filtered_prefix,".pgen"))
  # safe_remove(paste0(filtered_prefix,".pvar"))
  # safe_remove(paste0(filtered_prefix,".psam"))

  cat("Cleanup attempt complete for", chr, "\n")
}