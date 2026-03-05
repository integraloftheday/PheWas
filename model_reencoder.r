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
# - FRAME_MODE_04_6             (default: "keep"; one of: keep, compact, marginal)
#   - keep: preserve model@frame as-is
#   - compact: replace model@frame with a 1-row defaults frame + attach metadata
#   - marginal: save a lightweight fixed-effects bundle for marginal means
# - OVERWRITE_04_6              (default: "false")
# - VERIFY_READ_04_6            (default: "true")
# - GC_EACH_FILE_04_6           (default: "true")

# %%
SOURCE_DIR <- Sys.getenv("MODEL_SOURCE_DIR_04_6", "models_04_5")
OUTPUT_DIR <- Sys.getenv("MODEL_OUTPUT_DIR_04_6", "models_04_5_fast")
TARGET_COMPRESSION <- tolower(Sys.getenv("TARGET_COMPRESSION_04_6", "none"))
FRAME_MODE <- tolower(Sys.getenv("FRAME_MODE_04_6", "keep"))
OVERWRITE <- tolower(Sys.getenv("OVERWRITE_04_6", "false")) %in% c("true", "1", "yes")
VERIFY_READ <- tolower(Sys.getenv("VERIFY_READ_04_6", "true")) %in% c("true", "1", "yes")
GC_EACH_FILE <- tolower(Sys.getenv("GC_EACH_FILE_04_6", "true")) %in% c("true", "1", "yes")

if (!TARGET_COMPRESSION %in% c("none", "gzip", "bzip2", "xz")) {
  stop("TARGET_COMPRESSION_04_6 must be one of: none, gzip, bzip2, xz")
}
if (!FRAME_MODE %in% c("keep", "compact", "marginal")) {
  stop("FRAME_MODE_04_6 must be one of: keep, compact, marginal")
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

mode_value <- function(x) {
  x <- x[!is.na(x)]
  if (length(x) == 0) return(NA)
  ux <- unique(x)
  ux[[which.max(tabulate(match(x, ux)))]]
}

safe_names <- function(x) {
  out <- names(x)
  if (is.null(out)) character() else out
}

compact_model_frame <- function(obj) {
  if (!inherits(obj, "merMod")) {
    return(list(
      object = obj,
      compacted = FALSE,
      frame_rows_before = NA_integer_,
      frame_rows_after = NA_integer_,
      vars_tracked = NA_integer_
    ))
  }

  frm <- tryCatch(obj@frame, error = function(e) NULL)
  if (is.null(frm) || !is.data.frame(frm) || ncol(frm) == 0) {
    return(list(
      object = obj,
      compacted = FALSE,
      frame_rows_before = NA_integer_,
      frame_rows_after = NA_integer_,
      vars_tracked = NA_integer_
    ))
  }

  vars <- vector("list", ncol(frm))
  names(vars) <- names(frm)
  default_row <- vector("list", ncol(frm))
  names(default_row) <- names(frm)

  for (nm in names(frm)) {
    x <- frm[[nm]]
    x_no_na <- x[!is.na(x)]
    klass <- class(x)
    var_info <- list(
      class = klass,
      has_na = anyNA(x),
      values = NULL,
      min = NULL,
      max = NULL,
      default = NULL
    )

    if (is.factor(x)) {
      lev <- levels(x)
      m <- mode_value(as.character(x_no_na))
      if (is.na(m) && length(lev) > 0) m <- lev[[1]]
      if (length(lev) > 0 && !(m %in% lev)) m <- lev[[1]]
      default_row[[nm]] <- factor(m, levels = lev, ordered = is.ordered(x))
      var_info$values <- lev
      var_info$default <- m
      var_info$type <- "factor"
    } else if (is.character(x)) {
      vals <- sort(unique(x_no_na))
      m <- mode_value(x_no_na)
      if (is.na(m) && length(vals) > 0) m <- vals[[1]]
      default_row[[nm]] <- as.character(m)
      var_info$values <- vals
      var_info$default <- m
      var_info$type <- "character"
    } else if (is.logical(x)) {
      vals <- sort(unique(x_no_na))
      m <- mode_value(x_no_na)
      if (is.na(m)) m <- FALSE
      default_row[[nm]] <- as.logical(m)
      var_info$values <- as.logical(vals)
      var_info$default <- as.logical(m)
      var_info$type <- "logical"
    } else if (is.numeric(x) || is.integer(x)) {
      v <- as.numeric(x_no_na)
      m <- if (length(v) > 0) mean(v) else NA_real_
      default_row[[nm]] <- if (is.integer(x)) as.integer(round(m)) else as.numeric(m)
      var_info$default <- m
      if (length(v) > 0) {
        var_info$min <- min(v)
        var_info$max <- max(v)
      }
      uniq <- sort(unique(v))
      if (length(uniq) > 0 && length(uniq) <= 100) {
        var_info$values <- uniq
      }
      var_info$type <- if (is.integer(x)) "integer" else "numeric"
    } else {
      m <- mode_value(x_no_na)
      default_row[[nm]] <- if (length(x) > 0) x[[1]] else m
      var_info$default <- m
      var_info$type <- "other"
    }

    vars[[nm]] <- var_info
  }

  compact_row <- as.data.frame(default_row, stringsAsFactors = FALSE)
  for (nm in names(frm)) {
    if (is.factor(frm[[nm]])) {
      compact_row[[nm]] <- factor(
        as.character(compact_row[[nm]]),
        levels = levels(frm[[nm]]),
        ordered = is.ordered(frm[[nm]])
      )
    }
  }

  attr(obj, "phewas_grid_meta") <- list(
    version = 1L,
    source_frame_rows = nrow(frm),
    source_frame_cols = ncol(frm),
    vars = vars
  )
  obj@frame <- compact_row

  list(
    object = obj,
    compacted = TRUE,
    frame_rows_before = nrow(frm),
    frame_rows_after = nrow(compact_row),
    vars_tracked = length(safe_names(vars))
  )
}

build_marginal_bundle <- function(obj) {
  if (!inherits(obj, "merMod")) {
    return(list(
      object = obj,
      bundled = FALSE,
      frame_rows_before = NA_integer_,
      frame_rows_after = NA_integer_,
      vars_tracked = NA_integer_
    ))
  }

  frm <- tryCatch(obj@frame, error = function(e) NULL)
  if (is.null(frm) || !is.data.frame(frm) || ncol(frm) == 0) {
    return(list(
      object = obj,
      bundled = FALSE,
      frame_rows_before = NA_integer_,
      frame_rows_after = NA_integer_,
      vars_tracked = NA_integer_
    ))
  }

  compact_res <- compact_model_frame(obj)
  obj_compact <- compact_res$object
  meta <- attr(obj_compact, "phewas_grid_meta", exact = TRUE)
  vars <- if (!is.null(meta$vars)) meta$vars else list()

  xlevels <- list()
  for (nm in names(vars)) {
    v <- vars[[nm]]
    if (!is.null(v$type) && v$type %in% c("factor", "character") &&
        !is.null(v$values) && length(v$values) > 0) {
      xlevels[[nm]] <- as.character(v$values)
    }
  }

  fixed_formula <- tryCatch(lme4::nobars(formula(obj_compact)), error = function(e) NULL)
  if (is.null(fixed_formula)) {
    return(list(
      object = obj_compact,
      bundled = FALSE,
      frame_rows_before = compact_res$frame_rows_before,
      frame_rows_after = compact_res$frame_rows_after,
      vars_tracked = compact_res$vars_tracked
    ))
  }
  environment(fixed_formula) <- baseenv()
  fixed_terms <- stats::delete.response(terms(fixed_formula))
  attr(fixed_terms, ".Environment") <- baseenv()

  beta <- tryCatch(as.numeric(lme4::fixef(obj_compact)), error = function(e) NULL)
  beta_names <- tryCatch(names(lme4::fixef(obj_compact)), error = function(e) NULL)
  if (is.null(beta) || is.null(beta_names) || length(beta) == 0) {
    return(list(
      object = obj_compact,
      bundled = FALSE,
      frame_rows_before = compact_res$frame_rows_before,
      frame_rows_after = compact_res$frame_rows_after,
      vars_tracked = compact_res$vars_tracked
    ))
  }
  names(beta) <- beta_names

  vc <- tryCatch(as.matrix(stats::vcov(obj_compact)), error = function(e) NULL)
  if (!is.null(vc) && all(beta_names %in% rownames(vc)) && all(beta_names %in% colnames(vc))) {
    vc <- vc[beta_names, beta_names, drop = FALSE]
  } else {
    vc <- NULL
  }

  marginal_obj <- list(
    version = 1L,
    model_class = class(obj_compact),
    fixed_terms = fixed_terms,
    coefficients = beta,
    vcov_beta = vc,
    sigma = tryCatch(stats::sigma(obj_compact), error = function(e) NA_real_),
    default_row = obj_compact@frame,
    xlevels = xlevels,
    grid_meta = list(
      version = 1L,
      source_frame_rows = compact_res$frame_rows_before,
      source_frame_cols = if (!is.null(meta$source_frame_cols)) meta$source_frame_cols else NA_integer_,
      vars = vars
    )
  )
  class(marginal_obj) <- c("phewas_marginal_model", "list")

  list(
    object = marginal_obj,
    bundled = TRUE,
    frame_rows_before = compact_res$frame_rows_before,
    frame_rows_after = compact_res$frame_rows_after,
    vars_tracked = compact_res$vars_tracked
  )
}

model_files <- list.files(SOURCE_DIR, pattern = "\\.rds$", full.names = TRUE)
if (length(model_files) == 0) {
  stop(paste("No .rds files found in", SOURCE_DIR))
}

log_msg("Starting model re-encoding.")
log_msg(paste("SOURCE_DIR:", SOURCE_DIR))
log_msg(paste("OUTPUT_DIR:", OUTPUT_DIR))
log_msg(paste("TARGET_COMPRESSION:", TARGET_COMPRESSION))
log_msg(paste("FRAME_MODE:", FRAME_MODE))
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

  frame_rows_before <- NA_integer_
  frame_rows_after <- NA_integer_
  vars_tracked <- NA_integer_
  frame_status <- "kept"
  if (FRAME_MODE == "compact") {
    compact_res <- compact_model_frame(obj)
    obj <- compact_res$object
    frame_rows_before <- compact_res$frame_rows_before
    frame_rows_after <- compact_res$frame_rows_after
    vars_tracked <- compact_res$vars_tracked
    frame_status <- if (isTRUE(compact_res$compacted)) "compacted" else "unchanged"
  } else if (FRAME_MODE == "marginal") {
    bundle_res <- build_marginal_bundle(obj)
    obj <- bundle_res$object
    frame_rows_before <- bundle_res$frame_rows_before
    frame_rows_after <- bundle_res$frame_rows_after
    vars_tracked <- bundle_res$vars_tracked
    frame_status <- if (isTRUE(bundle_res$bundled)) "marginal_bundle" else "unchanged"
  }

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
      if (!is.na(frame_rows_before)) paste0(" | frame_rows=", frame_rows_before, "->", frame_rows_after) else "",
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
    frame_mode = FRAME_MODE,
    frame_status = frame_status,
    frame_rows_before = frame_rows_before,
    frame_rows_after = frame_rows_after,
    vars_tracked = vars_tracked,
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
