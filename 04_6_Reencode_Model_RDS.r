# ---
# jupyter:
#   jupytext:
#     formats: ipynb,R:percent
#     text_representation:
#       extension: .r
#       format_name: percent
#       format_version: "1.3"
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# # 04_6_Reencode_Model_RDS.r
# Re-encode fitted model .rds files to faster-read format for downstream analysis.
#
# Default behavior:
# - Reads from models_04_5
# - Writes to models_04_5_fast
# - Saves uncompressed .rds (fastest load, largest disk use)
#
# Environment variables:
# - MODEL_SOURCE_DIR_04_6       (default: "models_04_5")
# - MODEL_OUTPUT_DIR_04_6       (default: "models_04_5_fast")
# - TARGET_COMPRESSION_04_6     (default: "none"; one of: none, gzip, bzip2, xz)
# - OVERWRITE_04_6              (default: "false")
# - VERIFY_READ_04_6            (default: "true")
# - GC_EACH_FILE_04_6           (default: "true")

# %%
SOURCE_DIR <- Sys.getenv("MODEL_SOURCE_DIR_04_6", "models_04_5")
OUTPUT_DIR <- Sys.getenv("MODEL_OUTPUT_DIR_04_6", "models_04_5_fast")
TARGET_COMPRESSION <- tolower(Sys.getenv("TARGET_COMPRESSION_04_6", "none"))
OVERWRITE <- tolower(Sys.getenv("OVERWRITE_04_6", "false")) %in% c("true", "1", "yes")
VERIFY_READ <- tolower(Sys.getenv("VERIFY_READ_04_6", "true")) %in% c("true", "1", "yes")
GC_EACH_FILE <- tolower(Sys.getenv("GC_EACH_FILE_04_6", "true")) %in% c("true", "1", "yes")

if (!TARGET_COMPRESSION %in% c("none", "gzip", "bzip2", "xz")) {
  stop("TARGET_COMPRESSION_04_6 must be one of: none, gzip, bzip2, xz")
}

if (!dir.exists(SOURCE_DIR)) {
  stop(paste("Source directory does not exist:", SOURCE_DIR))
}

if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

log_lines <- character()
log_msg <- function(txt) {
  message(txt)
  stamp <- format(Sys.time(), "%Y-%m-%d %H:%M:%S")
  log_lines <<- c(log_lines, paste(stamp, txt))
}

format_bytes <- function(n) {
  if (is.na(n)) return(NA_character_)
  units <- c("B", "KB", "MB", "GB", "TB")
  v <- as.numeric(n)
  idx <- 1
  while (v >= 1024 && idx < length(units)) {
    v <- v / 1024
    idx <- idx + 1
  }
  sprintf("%.2f %s", v, units[idx])
}

compression_arg <- switch(
  TARGET_COMPRESSION,
  none = FALSE,
  gzip = "gzip",
  bzip2 = "bzip2",
  xz = "xz"
)

model_files <- list.files(SOURCE_DIR, pattern = "\\.rds$", full.names = TRUE)
if (length(model_files) == 0) {
  stop(paste("No .rds files found in", SOURCE_DIR))
}

log_msg("Starting model re-encoding.")
log_msg(paste("SOURCE_DIR:", SOURCE_DIR))
log_msg(paste("OUTPUT_DIR:", OUTPUT_DIR))
log_msg(paste("TARGET_COMPRESSION:", TARGET_COMPRESSION))
log_msg(paste("OVERWRITE:", OVERWRITE))
log_msg(paste("VERIFY_READ:", VERIFY_READ))
log_msg(paste("Files found:", length(model_files)))

results <- vector("list", length(model_files))

for (i in seq_along(model_files)) {
  src <- model_files[[i]]
  base <- basename(src)
  dst <- file.path(OUTPUT_DIR, base)

  src_size <- file.info(src)$size

  if (file.exists(dst) && !OVERWRITE) {
    log_msg(paste("Skipping existing:", dst))
    dst_size <- file.info(dst)$size
    results[[i]] <- data.frame(
      file = base,
      status = "skipped_exists",
      source_size_bytes = src_size,
      output_size_bytes = dst_size,
      read_source_sec = NA_real_,
      write_output_sec = NA_real_,
      read_output_sec = NA_real_,
      size_ratio_output_over_source = ifelse(is.na(src_size) || src_size == 0, NA_real_, dst_size / src_size),
      stringsAsFactors = FALSE
    )
    next
  }

  log_msg(paste("Processing:", base))

  read_src_time <- system.time({
    obj <- readRDS(src)
  })["elapsed"]

  write_time <- system.time({
    saveRDS(obj, dst, compress = compression_arg)
  })["elapsed"]

  read_out_time <- NA_real_
  if (VERIFY_READ) {
    read_out_time <- system.time({
      obj_check <- readRDS(dst)
      rm(obj_check)
    })["elapsed"]
  }

  dst_size <- file.info(dst)$size
  ratio <- ifelse(is.na(src_size) || src_size == 0, NA_real_, dst_size / src_size)

  log_msg(
    paste0(
      "Done: ", base,
      " | src=", format_bytes(src_size),
      " -> out=", format_bytes(dst_size),
      " | read_src=", sprintf("%.2fs", read_src_time),
      " | write=", sprintf("%.2fs", write_time),
      if (VERIFY_READ) paste0(" | read_out=", sprintf("%.2fs", read_out_time)) else ""
    )
  )

  results[[i]] <- data.frame(
    file = base,
    status = "converted",
    source_size_bytes = src_size,
    output_size_bytes = dst_size,
    read_source_sec = as.numeric(read_src_time),
    write_output_sec = as.numeric(write_time),
    read_output_sec = as.numeric(read_out_time),
    size_ratio_output_over_source = as.numeric(ratio),
    stringsAsFactors = FALSE
  )

  rm(obj)
  if (GC_EACH_FILE) invisible(gc(verbose = FALSE))
}

report <- do.call(rbind, results)
report <- report[order(report$file), ]
report_path_csv <- file.path(OUTPUT_DIR, "reencode_report_04_6.csv")
write.csv(report, report_path_csv, row.names = FALSE)

converted <- report[report$status == "converted", , drop = FALSE]
summary_lines <- c()
summary_lines <- c(summary_lines, paste("Files total:", nrow(report)))
summary_lines <- c(summary_lines, paste("Files converted:", nrow(converted)))

if (nrow(converted) > 0) {
  total_src <- sum(converted$source_size_bytes, na.rm = TRUE)
  total_out <- sum(converted$output_size_bytes, na.rm = TRUE)
  avg_read_src <- mean(converted$read_source_sec, na.rm = TRUE)
  avg_read_out <- mean(converted$read_output_sec, na.rm = TRUE)
  ratio <- ifelse(total_src == 0, NA_real_, total_out / total_src)

  summary_lines <- c(summary_lines, paste("Total source size:", format_bytes(total_src)))
  summary_lines <- c(summary_lines, paste("Total output size:", format_bytes(total_out)))
  summary_lines <- c(summary_lines, paste("Output/Source size ratio:", sprintf("%.3f", ratio)))
  summary_lines <- c(summary_lines, paste("Avg source read time (s):", sprintf("%.3f", avg_read_src)))
  if (VERIFY_READ) {
    summary_lines <- c(summary_lines, paste("Avg output read time (s):", sprintf("%.3f", avg_read_out)))
  }
}

summary_path <- file.path(OUTPUT_DIR, "reencode_summary_04_6.txt")
writeLines(summary_lines, con = summary_path)

log_path <- file.path(OUTPUT_DIR, "reencode_log_04_6.txt")
writeLines(log_lines, con = log_path)

log_msg("Re-encoding complete.")
log_msg(paste("Report:", report_path_csv))
log_msg(paste("Summary:", summary_path))
log_msg(paste("Log:", log_path))

