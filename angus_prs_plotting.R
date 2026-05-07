#!/usr/bin/env Rscript

if (dir.exists(".r_libs")) {
  .libPaths(c(normalizePath(".r_libs"), .libPaths()))
}

required_packages <- c("dplyr", "ggplot2", "readr", "stringr", "tibble")
missing_packages <- required_packages[
  !vapply(required_packages, requireNamespace, logical(1), quietly = TRUE)
]

if (length(missing_packages) > 0) {
  stop(
    paste0(
      "Missing required packages: ",
      paste(missing_packages, collapse = ", "),
      ". Install them before running angus_prs_plotting.R."
    )
  )
}

suppressPackageStartupMessages({
  library(dplyr)
  library(ggplot2)
  library(readr)
  library(stringr)
  library(tibble)
})

theme_research <- function() {
  theme_minimal(base_size = 12) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "gray30"),
      axis.title = element_text(face = "bold"),
      legend.title = element_text(face = "bold"),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom"
    )
}

normalize_phecode <- function(x) {
  out <- as.character(x)
  out <- str_trim(out)
  out <- str_replace(out, "(\\.\\d*?[1-9])0+$", "\\1")
  out <- str_replace(out, "\\.0+$", "")
  out
}

show_png_inline <- function(path) {
  if (!file.exists(path)) return(invisible(FALSE))
  if (!interactive() && !nzchar(Sys.getenv("JPY_PARENT_PID"))) return(invisible(FALSE))
  if (!requireNamespace("IRdisplay", quietly = TRUE)) return(invisible(FALSE))
  IRdisplay::display_png(file = path)
  invisible(TRUE)
}

parse_cli_args <- function(args) {
  out <- list(
    results_dir = "",
    results_root = "results",
    results_prefix = "angus_midpoint_prs_analysis",
    phecode_map_csv = "analysis_inputs/ICD_to_Phecode_mapping.csv",
    display_plots = TRUE
  )

  for (arg in args) {
    if (!startsWith(arg, "--")) next
    kv <- strsplit(sub("^--", "", arg), "=", fixed = TRUE)[[1]]
    key <- kv[[1]]
    value <- if (length(kv) > 1) paste(kv[-1], collapse = "=") else "true"
    out[[key]] <- value
  }

  out$display_plots <- tolower(as.character(out$display_plots)) %in% c("1", "true", "yes", "y")
  out
}

find_latest_results_dir <- function(results_root, results_prefix) {
  if (!dir.exists(results_root)) stop("Results root not found: ", results_root)

  all_dirs <- list.dirs(results_root, recursive = FALSE, full.names = TRUE)
  candidates <- all_dirs[basename(all_dirs) == results_prefix | startsWith(basename(all_dirs), paste0(results_prefix, "_"))]
  candidates <- candidates[file.exists(file.path(candidates, "tables", "phewas_results.csv"))]
  if (length(candidates) == 0) {
    stop("No Angus result directories found under ", results_root, " matching ", results_prefix)
  }

  summaries <- file.path(candidates, "summary.md")
  mtimes <- file.info(ifelse(file.exists(summaries), summaries, candidates))$mtime
  candidates[[order(mtimes, decreasing = TRUE)[[1]]]]
}

refresh_phewas_labels <- function(results_df, phecode_map_csv) {
  if (!file.exists(phecode_map_csv)) stop("Phecode map not found: ", phecode_map_csv)

  phemap <- readr::read_csv(phecode_map_csv, show_col_types = FALSE) %>%
    transmute(
      phecode_join = normalize_phecode(PHECODE),
      mapped_label = PHENOTYPE
    ) %>%
    distinct(phecode_join, .keep_all = TRUE)

  results_df %>%
    mutate(
      phecode = as.character(phecode),
      phecode_join = normalize_phecode(phecode)
    ) %>%
    left_join(phemap, by = "phecode_join") %>%
    mutate(
      concept_name = if_else(
        is.na(concept_name) | !nzchar(concept_name) | str_detect(concept_name, "^Phecode\\s+"),
        coalesce(mapped_label, concept_name),
        concept_name
      ),
      concept_name = if_else(is.na(concept_name) | !nzchar(concept_name), paste("Phecode", phecode), concept_name),
      label = if_else(p_value < 0.001, str_replace(concept_name, ",.*$", ""), NA_character_),
      siglevel = case_when(
        p_value < 1e-5 ~ "p < .00001",
        p_value < 1e-4 ~ "p < .0001",
        p_value < 1e-3 ~ "p < .001",
        TRUE ~ NA_character_
      ),
      siglevel = factor(siglevel, levels = c("p < .00001", "p < .0001", "p < .001"))
    ) %>%
    select(-phecode_join, -mapped_label)
}

write_cohort_flow_plot <- function(flow_df, plots_dir) {
  edge_df <- flow_df %>%
    filter(!is.na(parent_step_id)) %>%
    left_join(
      flow_df %>% select(step_id, x_parent = x, y_parent = y),
      by = c("parent_step_id" = "step_id")
    )

  p <- ggplot() +
    geom_segment(
      data = edge_df,
      aes(x = x_parent + 0.18, y = y_parent, xend = x - 0.18, yend = y, color = branch),
      arrow = grid::arrow(length = grid::unit(0.18, "cm")),
      linewidth = 0.7,
      lineend = "round"
    ) +
    geom_label(
      data = flow_df,
      aes(x = x, y = y, label = label, fill = branch),
      linewidth = 0.25,
      size = 3.4,
      label.padding = grid::unit(0.18, "lines")
    ) +
    scale_fill_manual(values = c("Sleep" = "#d9edf7", "Association" = "#d5e8d4", "PheWAS" = "#fce5cd")) +
    scale_color_manual(values = c("Sleep" = "#2c7fb8", "Association" = "#238b45", "PheWAS" = "#d95f0e")) +
    coord_cartesian(xlim = c(0.7, 4.35), ylim = c(0, 3.6), clip = "off") +
    labs(
      title = "Participant flow for Angus midpoint PRS analysis",
      subtitle = "Largest available sleep-genetics cohort is used for association; PheWAS uses the sleep-genetics-EHR overlap",
      caption = "Source table: tables/cohort_flow_counts.csv"
    ) +
    theme_void() +
    theme(
      plot.title = element_text(face = "bold", size = 14),
      plot.subtitle = element_text(size = 11, color = "gray30"),
      plot.caption = element_text(size = 9, color = "gray35"),
      legend.position = "none",
      plot.margin = margin(15, 25, 15, 25)
    )

  out <- file.path(plots_dir, "cohort_flow.png")
  ggsave(out, plot = p, width = 12, height = 6, dpi = 320, bg = "white")
  out
}

write_association_forest_plot <- function(forest_df, plots_dir) {
  p <- forest_df %>%
    ggplot(aes(x = estimate_minutes, y = phenotype_label, color = model)) +
    geom_point(position = position_dodge(width = 0.5), size = 2) +
    geom_errorbar(
      aes(xmin = ci_low_minutes, xmax = ci_high_minutes),
      position = position_dodge(width = 0.5),
      width = 0.2,
      orientation = "y"
    ) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "gray40") +
    labs(
      title = "Association of PRS per SD with midpoint phenotypes",
      subtitle = "Effect estimates are shown in minutes per 1 SD higher PRS",
      x = "Beta (minutes per SD higher PRS)",
      y = NULL,
      color = "Model",
      caption = "Source table: tables/association_forest_plot_data.csv"
    ) +
    theme_research()

  out <- file.path(plots_dir, "association_forest_per_sd.png")
  ggsave(out, plot = p, width = 10, height = 5, dpi = 320, bg = "white")
  out
}

write_tertile_plot <- function(tertile_df, plots_dir) {
  p <- ggplot(tertile_df, aes(x = score_tertile, y = midpoint_hours, fill = score_tertile)) +
    geom_boxplot(outlier.alpha = 0.2, width = 0.68, color = "gray25") +
    stat_summary(fun = mean, geom = "point", shape = 23, size = 2.5, fill = "gold", color = "black") +
    facet_wrap(~ phenotype_label, scales = "free_y") +
    scale_fill_manual(values = c("Low" = "#c6dbef", "Medium" = "#6baed6", "High" = "#2171b5")) +
    labs(
      title = "Midpoint phenotypes by PRS tertile",
      subtitle = "Boxes show the interquartile range; diamonds mark phenotype means",
      x = "PRS tertile",
      y = "Midpoint (decimal hours)",
      caption = "Source table: tables/midpoint_by_prs_tertile_plot_data.csv"
    ) +
    theme_research() +
    theme(legend.position = "none")

  out <- file.path(plots_dir, "midpoint_by_prs_tertile.png")
  ggsave(out, plot = p, width = 11, height = 6, dpi = 320, bg = "white")
  out
}

write_phewas_plots <- function(phewas_df, plots_dir) {
  manhattan <- ggplot(phewas_df, aes(x = phecode_index, y = minus_log10_p, color = fdr < 0.05)) +
    geom_point(alpha = 0.85, size = 1.8) +
    geom_hline(
      yintercept = -log10(0.05 / max(nrow(phewas_df), 1)),
      linetype = "dashed",
      color = "firebrick"
    ) +
    scale_color_manual(values = c("FALSE" = "gray55", "TRUE" = "#1b9e77")) +
    labs(
      title = "Continuous PRS per SD PheWAS",
      subtitle = "Green points pass FDR < 0.05; dashed line shows Bonferroni threshold",
      x = "Phecode index",
      y = expression(-log[10](p)),
      color = "FDR < 0.05",
      caption = "Source table: tables/phewas_manhattan_plot_data.csv"
    ) +
    theme_research()

  manhattan_path <- file.path(plots_dir, "phewas_manhattan.png")
  ggsave(manhattan_path, plot = manhattan, width = 11, height = 5.5, dpi = 320, bg = "white")

  volcano_df <- phewas_df %>% filter(is.finite(odds_ratio))
  volcano_path <- file.path(plots_dir, "phewas_volcano.png")
  if (nrow(volcano_df) > 0) {
    volcano <- ggplot(volcano_df, aes(x = odds_ratio, y = minus_log10_p)) +
      geom_point(color = "grey70", size = 1.8, alpha = 0.85) +
      geom_point(
        data = volcano_df %>% filter(!is.na(siglevel)),
        aes(color = siglevel),
        size = 2.2,
        alpha = 0.95
      ) +
      geom_vline(xintercept = 1, color = "gray35", linewidth = 0.6) +
      geom_hline(yintercept = -log10(0.05), color = "#b2182b", linetype = "dashed", linewidth = 0.6) +
      geom_hline(
        yintercept = -log10(0.05 / max(nrow(phewas_df), 1)),
        color = "#2166ac",
        linetype = "dashed",
        linewidth = 0.6
      ) +
      geom_text(
        data = volcano_df %>% filter(!is.na(label)),
        aes(label = label),
        check_overlap = TRUE,
        nudge_y = 0.1,
        size = 3
      ) +
      scale_color_manual(
        values = c("p < .00001" = "#7f0000", "p < .0001" = "#cb181d", "p < .001" = "#fb6a4a"),
        na.translate = FALSE,
        drop = FALSE
      ) +
      labs(
        title = "PheWAS volcano plot for continuous PRS",
        subtitle = "Legacy-style odds-ratio display; labels shown for p < 0.001; all finite ORs plotted",
        x = "Odds ratio per 1 SD higher PRS",
        y = expression(-log[10](p)),
        color = "Sig. level",
        caption = "Source table: tables/phewas_volcano_plot_data.csv"
      ) +
      theme_research()

    ggsave(volcano_path, plot = volcano, width = 10, height = 8, dpi = 320, bg = "white")
  }

  c(manhattan_path, volcano_path)
}

main <- function(
  results_dir = "",
  results_root = "results",
  results_prefix = "angus_midpoint_prs_analysis",
  phecode_map_csv = "analysis_inputs/ICD_to_Phecode_mapping.csv",
  display_plots = TRUE
) {
  chosen_results_dir <- if (nzchar(results_dir)) results_dir else find_latest_results_dir(results_root, results_prefix)
  tables_dir <- file.path(chosen_results_dir, "tables")
  plots_dir <- file.path(chosen_results_dir, "plots")

  required_files <- c(
    file.path(tables_dir, "cohort_flow_counts.csv"),
    file.path(tables_dir, "association_forest_plot_data.csv"),
    file.path(tables_dir, "midpoint_by_prs_tertile_plot_data.csv"),
    file.path(tables_dir, "phewas_results.csv")
  )
  missing <- required_files[!file.exists(required_files)]
  if (length(missing) > 0) {
    stop("Missing required plotting inputs: ", paste(missing, collapse = ", "))
  }

  dir.create(plots_dir, recursive = TRUE, showWarnings = FALSE)

  flow_df <- readr::read_csv(file.path(tables_dir, "cohort_flow_counts.csv"), show_col_types = FALSE)
  forest_df <- readr::read_csv(file.path(tables_dir, "association_forest_plot_data.csv"), show_col_types = FALSE)
  tertile_df <- readr::read_csv(file.path(tables_dir, "midpoint_by_prs_tertile_plot_data.csv"), show_col_types = FALSE)
  phewas_df <- readr::read_csv(file.path(tables_dir, "phewas_results.csv"), show_col_types = FALSE)

  phewas_df <- refresh_phewas_labels(phewas_df, phecode_map_csv) %>%
    arrange(p_value) %>%
    mutate(
      minus_log10_p = -log10(p_value),
      phecode_index = row_number()
    )

  readr::write_csv(phewas_df, file.path(tables_dir, "phewas_results.csv"))
  readr::write_csv(phewas_df, file.path(tables_dir, "phewas_manhattan_plot_data.csv"))
  readr::write_csv(phewas_df %>% filter(is.finite(odds_ratio)), file.path(tables_dir, "phewas_volcano_plot_data.csv"))

  plot_paths <- c(
    write_cohort_flow_plot(flow_df, plots_dir),
    write_association_forest_plot(forest_df, plots_dir),
    write_tertile_plot(tertile_df, plots_dir),
    write_phewas_plots(phewas_df, plots_dir)
  )

  cat("Using results directory:", chosen_results_dir, "\n")
  cat("Updated labels and rewrote:\n")
  cat(" -", file.path(tables_dir, "phewas_results.csv"), "\n")
  cat(" -", file.path(tables_dir, "phewas_manhattan_plot_data.csv"), "\n")
  cat(" -", file.path(tables_dir, "phewas_volcano_plot_data.csv"), "\n")
  cat("Plots regenerated:\n")
  for (plot_path in plot_paths) {
    if (file.exists(plot_path)) cat(" -", plot_path, "\n")
  }

  top_rows <- phewas_df %>%
    select(phecode, concept_name, odds_ratio, p_value, fdr) %>%
    slice_head(n = 10)
  print(top_rows, n = 10)

  if (display_plots) {
    for (plot_path in plot_paths) {
      shown <- show_png_inline(plot_path)
      if (!shown) message("Plot written: ", plot_path)
    }
  }

  invisible(list(
    results_dir = chosen_results_dir,
    phewas_results = phewas_df,
    plot_paths = plot_paths
  ))
}

if (sys.nframe() == 0L) {
  cli <- parse_cli_args(commandArgs(trailingOnly = TRUE))
  main(
    results_dir = cli$results_dir,
    results_root = cli$results_root,
    results_prefix = cli$results_prefix,
    phecode_map_csv = cli$phecode_map_csv,
    display_plots = cli$display_plots
  )
}
