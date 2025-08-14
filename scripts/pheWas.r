# Define some helper functions for running the PheWAS

run_glm <- function(y, dat, model_equation = "sleep.dev_abs+race+age+sex_concept", impute = TRUE) {
    dat <- as.data.frame(dat)
    dat <- dat[which(!is.na(dat[, y])), ]
    dat[, y] <- dat[, y] >= 1

    # Check if the outcome variable is constant
    if (length(unique(dat[, y])) <= 1) {
        print(paste("Skipping", y, "because it's constant"))
        return(NULL)
    }

    # Process model equation to handle interaction terms properly
    processed_equation <- model_equation

    # Extract ALL variable names from model equation, including interaction terms
    # First, handle explicit interaction terms (e.g., "var1:var2")
    temp_formula <- as.formula(paste("~", model_equation))

    # Get all base variable names (excluding transformations like I(age^2))
    all_vars <- all.vars(temp_formula)

    # For interaction terms in the formula (e.g., "var1 * var2"),
    # we need to ensure both main effects and interaction are available
    if (grepl("\\*", model_equation)) {
        # Extract interaction patterns
        interactions <- regmatches(model_equation, gregexpr("\\w+\\s*\\*\\s*\\w+", model_equation))[[1]]
        for (interaction in interactions) {
            # Split interaction into components
            parts <- trimws(strsplit(interaction, "\\*")[[1]])
            # Add both main effect variables to our list
            all_vars <- c(all_vars, parts)
        }
    }

    # Remove duplicates
    all_vars <- unique(all_vars)

    # Ensure we have all necessary columns in the dataset
    available_vars <- intersect(all_vars, colnames(dat))
    missing_vars <- setdiff(all_vars, colnames(dat))

    if (length(missing_vars) > 0) {
        warning(paste("Missing variables in dataset:", paste(missing_vars, collapse = ", ")))
        # Return NULL if critical variables are missing
        return(NULL)
    }

    # Select only the necessary columns
    dat <- dat[, c(y, available_vars), drop = FALSE]

    # Remove rows with missing values in any of the required variables
    # This is crucial for interaction terms to work properly
    complete_rows <- complete.cases(dat[, available_vars, drop = FALSE])
    dat <- dat[complete_rows, , drop = FALSE]

    # Check if we have enough data after removing incomplete cases
    if (nrow(dat) < 10) {  # Minimum threshold for meaningful analysis
        print(paste("Skipping", y, "because too few complete cases:", nrow(dat)))
        return(NULL)
    }

    tryCatch({
        f <- paste0(y, "~", model_equation)

        if (impute) {
            # For imputation, we need to be more careful with interaction terms
            # The imputation formula should include all main effects
            main_effects <- setdiff(available_vars, y)

            # Create imputation formula with main effects only
            # Interaction terms will be computed from imputed main effects
            fimp <- paste0(" ~ ", y, "+", paste(main_effects, collapse = "+"))

            # Check if we have sufficient variation for imputation
            var_check <- sapply(dat[, main_effects, drop = FALSE], function(x) {
                if (is.numeric(x)) {
                    return(var(x, na.rm = TRUE) > 0)
                } else {
                    return(length(unique(x[!is.na(x)])) > 1)
                }
            })

            if (!all(var_check)) {
                print(paste("Skipping", y, "because of constant variables in imputation"))
                return(NULL)
            }

            # Run imputation
            xtrans <- Hmisc::aregImpute(as.formula(fimp), data = dat, pr = FALSE, n.impute = 5)

            # Fit model with imputation
            m <- Hmisc::fit.mult.impute(
                as.formula(f),
                glm,
                xtrans = xtrans,
                data = dat,
                family = binomial(link = "logit"),
                pr = FALSE
            )
        } else {
            # Use complete case analysis (no imputation)
            dat_complete <- dat[complete.cases(dat), ]
            if (nrow(dat_complete) < 10) {  # Minimum threshold
                print(paste("Skipping", y, "because insufficient complete cases:", nrow(dat_complete)))
                return(NULL)
            }

            # Check for perfect separation or other issues
            outcome_by_predictors <- tryCatch({
                temp_model <- glm(as.formula(f), data = dat_complete, family = binomial(link = "logit"))
                TRUE
            }, warning = function(w) {
                if (grepl("fitted probabilities numerically 0 or 1", w$message)) {
                    print(paste("Warning for", y, "- perfect separation detected"))
                }
                TRUE
            }, error = function(e) {
                FALSE
            })

            if (!outcome_by_predictors) {
                print(paste("Skipping", y, "because of model fitting issues"))
                return(NULL)
            }

            m <- glm(
                as.formula(f),
                data = dat_complete,
                family = binomial(link = "logit")
            )
        }

        return(m)
    }, error = function(cond) {
        print(paste("Error in glm for:", y))
        print(cond)
        NULL
    })
}



summary_glm <- function(m, exposure_var = "sleep.dev_abs")
{
    if (!is.null(m))
    {
        or <- cbind("Odds Ratio"=exp(coef(m)),exp(suppressMessages(confint.default(m,ci=.95,method="Wald")) ))
        s <- summary(m)
        out <- as.data.frame(cbind(s$coefficients,or,nobs(m),sum(m$y)))
        out$coeff <- rownames(out)
        out <- out[out$coeff == exposure_var,]
        colnames(out) <- c("estimate","std_error",
                          "z_value","p_value","odds_ratio",
                          "ci_lower_2.5","ci_higher_97.5","n","n_events",
                          "coeff")
        return(out)
    } else {
        data.frame("estimate"=NA,"std_error"=NA, "z_value"=NA,"p_value"=NA,"odds_ratio"=NA,
                          "ci_lower_2.5"=NA,"ci_higher_97.5"=NA,"n"=NA,"n_events"=NA,
                          "coeff"=NA)
    }
}

run_phewas <- function(final_dat,
                      model_equation = "sleep.dev_abs+race+age+sex_concept",
                      exposure_var = "sleep.dev_abs",
                      impute = TRUE,
                      save_file_path = "./phewas_results.csv",
                      save_models = FALSE,
                      model_save_path = "./models/",
                      phemap = NULL) {

    # Create phecode to concept mapping
    if (is.null(phemap)) {
        stop("phemap parameter is required for concept mapping")
    }

    phe_to_concept_map <- phemap[,c("phecode","description")]
    phe_to_concept_map <- phe_to_concept_map[!duplicated(phe_to_concept_map),]

    # Identify phenotype columns
    phe_cols <- grep("has_phe", colnames(final_dat), value=TRUE)
    fitbit_cols <- grep("average", colnames(final_dat), value=TRUE)

    # Convert to data frame for easier manipulation
    final_dat <- as.data.frame(final_dat)

    # Set phenotype to NA if patient had the condition before baseline
    for (i in 1:length(phe_cols)) {
        had_col <- gsub("has","had", phe_cols[i])
        if (had_col %in% colnames(final_dat)) {
            final_dat[which(final_dat[,had_col] >= 1), phe_cols[i]] <- NA
        }
    }

    # Prepare data types
    options(warn = -1)
    set.seed(1)
    final_dat$age <- as.numeric(final_dat$age)
    final_dat$race <- as.factor(final_dat$race)
    final_dat$sex_concept <- as.factor(final_dat$sex_concept)

# Recode rare categories
    final_dat$sex_concept[final_dat$sex_concept %in% c("Intersex",
                                                   "Sex At Birth: Sex At Birth None Of These",
                                                   "I prefer not to answer",
                                                   "PMI: Skip")] <- "Other/Unknown"

    # Convert back to factor and drop unused levels
    final_dat$sex_concept <- factor(final_dat$sex_concept)
    final_dat$sex_concept <- droplevels(final_dat$sex_concept)


    # Run models for each phenotype
   # print("Running GLM models...")
    model_out <- plyr::llply(phe_cols,
                            function(x) run_glm(x, final_dat,
                                              model_equation = model_equation,
                                              impute = impute),
                            .progress = "text")

    # Save models if requested
    if (save_models) {
        print("Saving model objects...")
        if (!dir.exists(model_save_path)) {
            dir.create(model_save_path, recursive = TRUE)
        }

        # Name models by phenotype
        names(model_out) <- gsub("has_phe_", "", phe_cols)

        # Save as RDS for efficient storage
        saveRDS(model_out, file = paste0(model_save_path, "phewas_models.rds"))

        # Also save a summary of which models were successfully fit
        model_success <- sapply(model_out, function(x) !is.null(x))
        model_summary <- data.frame(
            phecode = gsub("has_phe_", "", phe_cols),
            model_fit_success = model_success,
            stringsAsFactors = FALSE
        )
        write.csv(model_summary, file = paste0(model_save_path, "model_fit_summary.csv"),
                 row.names = FALSE)
    }

    # Summarize results
    print("Summarizing results...")
    summary_out <- plyr::llply(model_out,
                              function(x) summary_glm(x, exposure_var = exposure_var),
                              .progress = "text")

    # Add phecode information
    summary_out <- Map(cbind, summary_out, phecode = gsub("has_phe_", "", phe_cols))

    # Combine results
    df <- do.call(rbind, summary_out)
    rownames(df) <- NULL
    df_all <- data.frame(df)

    # Merge with concept mapping
    df_all <- merge(df_all, phe_to_concept_map, by = "phecode")

    # Standardize column names
    colnames(df_all) <- c("phecode", "estimate", "std_error",
                         "z_value", "p_value", "odds_ratio",
                         "ci_lower_2.5", "ci_higher_97.5", "n", "n_events",
                         "coeff", "concept_name")

    # Create directory for results if it doesn't exist
    result_dir <- dirname(save_file_path)
    if (!dir.exists(result_dir) && result_dir != ".") {
        dir.create(result_dir, recursive = TRUE)
    }

    # Save results
    print(paste("Saving results to:", save_file_path))
    data.table::fwrite(df_all, file = save_file_path)

    # Print summary statistics
    n_total <- nrow(df_all)
    n_successful <- sum(!is.na(df_all$p_value))
    n_significant <- sum(df_all$p_value < 0.05, na.rm = TRUE)

    cat("\n=== PheWAS Analysis Summary ===\n")
    cat("Total phenotypes analyzed:", n_total, "\n")
    cat("Successful model fits:", n_successful, "\n")
    cat("Significant associations (p < 0.05):", n_significant, "\n")
    cat("Results saved to:", save_file_path, "\n")
    if (save_models) {
        cat("Models saved to:", paste0(model_save_path, "phewas_models.rds"), "\n")
    }
    cat("===============================\n\n")

    # Reset warnings
    options(warn = 0)

    # Return results
    return(df_all)
}


# Function to extract all results for all covariates across all phenotypes
extract_all_results <- function(model_list, phe_cols) {
    all_results <- list()

    for (i in seq_along(model_list)) {
        if (!is.null(model_list[[i]])) {
            # Get all coefficients
            s <- summary(model_list[[i]])
            coef_df <- as.data.frame(s$coefficients)
            coef_df$coeff <- rownames(coef_df)
            coef_df$phenotype <- gsub("has_phe_", "", phe_cols[i])
            coef_df$odds_ratio <- exp(coef_df$Estimate)

            all_results[[i]] <- coef_df
        }
    }

    # Combine all results
    combined_results <- do.call(rbind, all_results)
    return(combined_results)
}