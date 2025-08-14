# Required libraries
library(bigrquery)
library(readr)
library(dplyr)
library(purrr)
library(stringr)
library(nanoparquet)  # Alternative: library(arrow) then use arrow::read_parquet

#These packages with version numbers are need to be installed

need_package <- function(pkg,version)
{
    need <- FALSE
    if (!require(pkg,character.only = TRUE))
    {
        need <- TRUE
    } else {
        if (packageVersion(pkg) != version)
        {
            need <- TRUE
        }
    }
    return(need)
}

using_version <- function(pkg,version)
{
   if (need_package(pkg,version))
   {
        devtools::install_version(pkg, version = version,upgrade=FALSE)
   }
}

# Helper functions
get_dataset <- function() Sys.getenv("WORKSPACE_CDR")
get_project <- function() Sys.getenv("GOOGLE_PROJECT")

# Enhanced read_bucket that automatically detects and handles both parquet and CSV
read_bucket <- function(export_path)
{
    # Get list of files
    files <- system2("gsutil", args = c("ls", export_path), stdout = TRUE, stderr = TRUE)
    
    # Check if we have parquet files
    if (any(grepl("\\.parquet$", files))) {
        message("Detected parquet files, using parquet reader...")
        
        # Read parquet files
        bind_rows(map(files, function(file) {
            message(str_glue("Loading {file}"))
            temp_file <- tempfile(fileext = ".parquet")
            system2("gsutil", args = c("cp", file, temp_file), stdout = FALSE, stderr = FALSE)
            data <- read_parquet(temp_file)
            unlink(temp_file)
            return(data)
        }))
        
    } else {
        message("Using CSV reader...")
        
        # Original CSV reading logic
        col_types <- NULL
        bind_rows(map(files, function(csv) {
            message(str_glue("Loading {csv}."))
            chunk <- read_csv(pipe(str_glue("gsutil cat {csv}")),
                col_types = col_types, show_col_types = FALSE)
            if (is.null(col_types)) {
                col_types <- spec(chunk)
            }
            chunk
        }))
    }
}

# Enhanced download_data that can optionally export as parquet
download_data <- function(query)
{
    tb <- bq_project_query(Sys.getenv("GOOGLE_PROJECT"), query)
    bq_table_download(tb, page_size = 1e+05)
}

# Additional utility functions for parquet workflow
# These functions can be used to replace bq_table_save + read_bucket pattern

# Function to export query results as parquet
export_query_parquet <- function(query, export_path) {
    bq_table_save(
        bq_dataset_query(get_dataset(), query, billing = get_project()),
        export_path,
        destination_format = "PARQUET"
    )
}

# Function to export query results as CSV (for backward compatibility)
export_query_csv <- function(query, export_path) {
    bq_table_save(
        bq_dataset_query(get_dataset(), query, billing = get_project()),
        export_path,
        destination_format = "CSV"
    )
}

# Unified export function that defaults to parquet
export_query <- function(query, export_path, format = "PARQUET") {
    bq_table_save(
        bq_dataset_query(get_dataset(), query, billing = get_project()),
        export_path,
        destination_format = format
    )
}