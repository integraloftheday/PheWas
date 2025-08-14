
source("./scripts/general-utils.r")
source("./scripts/data_accumulation_dependencies.r")


dataset <- Sys.getenv("WORKSPACE_CDR")
my_bucket <- Sys.getenv("WORKSPACE_BUCKET")

data_bucket='gs://fc-secure-353614d4-9462-4a97-b1a5-d90627e50025'


phemap <- read.csv("phemap.csv", header = TRUE, sep = ",")
colnames(phemap)[2] <- "condition_source_value"
#save.image(file = "full_workspace.RData")

icd_query <- str_glue("
    WITH cohort AS (
          SELECT 
              DISTINCT person_id 
          FROM `{dataset}.sleep_daily_summary`

          UNION DISTINCT
          SELECT 
              DISTINCT person_id 
          FROM `{dataset}.sleep_daily_summary`)
    SELECT DISTINCT cohort.person_id,co.condition_start_date,co.condition_source_value,c.concept_name
    FROM 
        cohort
        INNER JOIN
        `{dataset}.condition_occurrence` co
        ON (cohort.person_id = co.person_id)
        INNER JOIN
        `{dataset}.concept` c
        ON (co.condition_source_concept_id = c.concept_id)
    WHERE
        c.VOCABULARY_ID LIKE 'ICD10CM' OR c.VOCABULARY_ID LIKE 'ICD9CM'
    ")

path <- file.path(
      Sys.getenv("WORKSPACE_BUCKET"),
      "bq_exports",
      "icd_codes",
      "icd_codes_*.csv")

if (!load_stored_result) {
    bq_table_save(
      bq_dataset_query(dataset, icd_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      path,
      destination_format = "CSV")
}
icd_result <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/icd_codes/icd_codes_*.csv")))
icd_result <- icd_result[!duplicated(icd_result)]
#write_parquet(icd_result,"")
#write_parquet(icd_result, "starter_id.parquet")
#save.image(file = "full_workspace.RData")

length(unique(icd_result$person_id))

#fitbit_query <- str_glue("
#    SELECT person_id,start_datetime
#    FROM 
#    `{dataset}.sleep_level`
#")

fitbit_query <- str_glue("
    SELECT person_id, sleep_date, minute_asleep
    FROM 
    `{dataset}.sleep_daily_summary`
    WHERE minute_asleep > 0
")
path <- file.path(
      Sys.getenv("WORKSPACE_BUCKET"),
      "bq_exports",
      "minute_asleep",
      "all_sleep_t_*.csv")

if (!load_stored_result) {
    bq_table_save(
      bq_dataset_query(dataset, fitbit_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      path,
      destination_format = "CSV")
}
all_sleep <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/minute_asleep/all_sleep_t_*.csv")))
#save.image(file = "full_workspace.RData")

length(unique(all_sleep$person_id))

#unique_ids <- unique(all_sleep$person_id)

# Create a data frame with all IDs in a single row
# We'll use a named list where each element becomes a column
#id_df <- data.frame(t(unique_ids))

# Rename columns to be person_id_1, person_id_2, etc.
#colnames(id_df) <- paste0("person_id_", 1:ncol(id_df))

# Save as parquet file
#write_parquet(id_df, "starter_id.parquet")

# Confirm the file was created
cat("File 'starter_id.parquet' has been created successfully!\n")

fitbit_query <- str_glue("
    SELECT DISTINCT
        sl.person_id,
        sl.start_datetime,
        sl.level,
        sl.duration_in_min,
        sl.is_main_sleep,
        sds.minute_asleep
    FROM `{dataset}.sleep_level` sl
        INNER JOIN `{dataset}.sleep_daily_summary` sds
    ON sl.person_id = sds.person_id
    AND DATE(sl.sleep_date) = DATE(sds.sleep_date)
    WHERE sds.minute_asleep > 0
        AND sl.is_main_sleep = 'TRUE'
")


path <- file.path(
      Sys.getenv("WORKSPACE_BUCKET"),
      "bq_exports",
      "minute_asleep",
      "all_sleep_temporal_*.csv")

if (!load_stored_result) {
    bq_table_save(
      bq_dataset_query(dataset, fitbit_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      path,
      destination_format = "CSV")
}
all_sleep_temp <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/minute_asleep/all_sleep_temporal_*.csv")))
#save.image(file = "full_workspace.RData")

all_sleep_temp <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/minute_asleep/all_sleep_temporal_000000000000.csv")))

all_sleep_temp

#save.image(file = "full_workspace_sleep_data.RData")

if (!load_stored_result) {
    ses_query = str_glue("
        SELECT DISTINCT z.*EXCEPT(zip3, zip3_as_string), zip3 as zip3
        FROM `{dataset}.zip3_ses_map` z
        ")

    # Export the results
    ses_path <- file.path(
      Sys.getenv("WORKSPACE_BUCKET"),
      "bq_exports",
      "ses_data",
      "ses_data_*.csv")

    bq_table_save(
      bq_dataset_query(dataset, ses_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      ses_path,
      destination_format = "CSV")
}

# Read the exported data
ses_df <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/ses_data/ses_data_*.csv")))

if (!load_stored_result) {
participant_zip_query <- str_glue("
WITH cohort AS (
    SELECT DISTINCT person_id 
    FROM `{dataset}.activity_summary`
)
SELECT
    c.person_id,
    LEFT(TRIM(o.value_as_string), 3) AS zip3
FROM cohort c
JOIN `{dataset}.observation` o ON c.person_id = o.person_id
WHERE o.observation_source_concept_id = 1585250  -- Zip code concept
")

# Export the results
participant_zip_path <- file.path(
  Sys.getenv("WORKSPACE_BUCKET"),
  "bq_exports",
  "participant_zip",
  "participant_zip_*.csv")

bq_table_save(
  bq_dataset_query(dataset, participant_zip_query, billing = Sys.getenv("GOOGLE_PROJECT")),
  participant_zip_path,
  destination_format = "CSV")
}
# Read the exported data
participant_zip_df <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/participant_zip/participant_zip_*.csv")))


# Convert the zip3 column in participant_zip_df to numeric
participant_zip_df$zip3 <- as.numeric(participant_zip_df$zip3)

socio_eco_df <- participant_zip_df %>%
  left_join(ses_df, by = c("zip3" = "zip3"))

head(socio_eco_df)

length(unique(socio_eco_df$person_id))

#compliance query
#should this be modified for sleep? 

comp_query <- str_glue("
SELECT person_id, sleep_date, minute_asleep>=60 AS compliant
    FROM 
    `{dataset}.sleep_daily_summary`
    WHERE minute_asleep > 0
")

path <- file.path(
      Sys.getenv("WORKSPACE_BUCKET"),
      "bq_exports",
      "compliance",
      "compliance_*.csv")
    
if (!load_stored_result) {
    bq_table_save(
      bq_dataset_query(dataset, comp_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      path,
      destination_format = "CSV")
}
compliance <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/compliance/compliance_*.csv")))
#save.image(file = "full_workspace.RData")



dem_query <- str_glue("
    SELECT 
        person.PERSON_ID AS person_id,
        person.BIRTH_DATETIME as date_of_birth, 
        p_race_concept.concept_name as race, 
        p_gender_concept.concept_name as gender
    FROM 
    `{dataset}.person` person
    LEFT JOIN `{dataset}.concept` p_race_concept on person.race_concept_id = p_race_concept.CONCEPT_ID 
    LEFT JOIN `{dataset}.concept` p_gender_concept on person.gender_concept_id = p_gender_concept.CONCEPT_ID
")


path <- file.path(
      Sys.getenv("WORKSPACE_BUCKET"),
      "bq_exports",
      "dem",
      "dem_*.csv")

if (!load_stored_result) {
    bq_table_save(
      bq_dataset_query(dataset, dem_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      path,
      destination_format = "CSV")
}
dem <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/dem/dem_*.csv")))
#save.image(file = "full_workspace.RData")

ehr_query <- str_glue("
    WITH ehr AS (
    SELECT
       distinct person_id
    FROM `{dataset}.measurement` AS m
    LEFT JOIN `{dataset}.measurement_ext` AS mm on m.measurement_id = mm.measurement_id
    WHERE LOWER(mm.src_id) LIKE 'ehr site%'

    UNION DISTINCT

    SELECT
       DISTINCT person_id
    FROM `{dataset}.condition_occurrence` AS m
    LEFT JOIN `{dataset}.condition_occurrence_ext` AS mm on m.condition_occurrence_id = mm.condition_occurrence_id
    WHERE LOWER(mm.src_id) LIKE 'ehr site%'

    UNION DISTINCT

    SELECT
       DISTINCT person_id
    FROM `{dataset}.device_exposure` AS m
    LEFT JOIN `{dataset}.device_exposure_ext` AS mm on m.device_exposure_id = mm.device_exposure_id
    WHERE LOWER(mm.src_id) LIKE 'ehr site%'

    UNION DISTINCT

    SELECT
       DISTINCT person_id
    FROM `{dataset}.drug_exposure` AS m
    LEFT JOIN `{dataset}.drug_exposure_ext` AS mm on m.drug_exposure_id = mm.drug_exposure_id
    WHERE LOWER(mm.src_id) LIKE 'ehr site%'

    UNION DISTINCT

    SELECT
       DISTINCT person_id
    FROM `{dataset}.observation` AS m
    LEFT JOIN `{dataset}.observation_ext` AS mm on m.observation_id = mm.observation_id
    WHERE LOWER(mm.src_id) LIKE 'ehr site%'

    union distinct

    Select
       distinct person_id
    from `{dataset}.procedure_occurrence` as m
    left join `{dataset}.procedure_occurrence_ext` as mm on m.procedure_occurrence_id = mm.procedure_occurrence_id
    where lower(mm.src_id) like 'ehr site%'

    union distinct

    Select
       distinct person_id
    from `{dataset}.visit_occurrence` as m
    left join `{dataset}.visit_occurrence_ext` as mm on m.visit_occurrence_id = mm.visit_occurrence_id
    where lower(mm.src_id) like 'ehr site%'
    )

    select DISTINCT person_id
    FROM ehr
")
path <- file.path(
      Sys.getenv("WORKSPACE_BUCKET"),
      "bq_exports",
      "ehr_cohort",
      "ehr_cohort_*.csv")

if (!load_stored_result) {
    bq_table_save(
      bq_dataset_query(dataset, ehr_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      path,
      destination_format = "CSV")
}
ehr_cohort <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/ehr_cohort/ehr_cohort_*.csv")))
#save.image(file = "full_workspace.RData")

query <- str_glue("
    WITH ehr AS (
    SELECT person_id, MAX(m.measurement_date) AS date
    FROM `{dataset}.measurement` AS m
    LEFT JOIN `{dataset}.measurement_ext` AS mm on m.measurement_id = mm.measurement_id
    WHERE LOWER(mm.src_id) LIKE 'ehr site%'
    GROUP BY person_id

    UNION DISTINCT

    SELECT person_id, MAX(m.condition_start_date) AS date
    FROM `{dataset}.condition_occurrence` AS m
    LEFT JOIN `{dataset}.condition_occurrence_ext` AS mm on m.condition_occurrence_id = mm.condition_occurrence_id
    WHERE LOWER(mm.src_id) LIKE 'ehr site%'
    GROUP BY person_id

    UNION DISTINCT

    SELECT person_id, MAX(m.procedure_date) AS date
    FROM `{dataset}.procedure_occurrence` AS m
    LEFT JOIN `{dataset}.procedure_occurrence_ext` AS mm on m.procedure_occurrence_id = mm.procedure_occurrence_id
    WHERE LOWER(mm.src_id) LIKE 'ehr site%'
    GROUP BY person_id

    UNION DISTINCT

    SELECT person_id, max(m.visit_end_date) AS date
    FROM `{dataset}.visit_occurrence` AS m
    LEFT JOIN `{dataset}.visit_occurrence_ext` AS mm on m.visit_occurrence_id = mm.visit_occurrence_id
    WHERE LOWER(mm.src_id) LIKE 'ehr site%'
    GROUP BY person_id
    )

    SELECT person_id, MAX(date)
    FROM ehr
    GROUP BY person_id
    ")
    path <- file.path(
      Sys.getenv("WORKSPACE_BUCKET"),
      "bq_exports",
      "last_enc",
      "last_enc_*.csv")

if (!load_stored_result) {
    bq_table_save(
      bq_dataset_query(dataset, query, billing = Sys.getenv("GOOGLE_PROJECT")),
      path,
      destination_format = "CSV")
}
last_enc <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/last_enc/last_enc_*.csv")))
colnames(last_enc) <- c("person_id","last_encounter")
#save.image(file = "full_workspace.RData")

# Assuming 'all_sleep' dataframe and 'hours12' variable (e.g., hours12 <- 720) are already defined
# And dplyr is loaded: library(dplyr)
hours12 <- 60*24
# 1. Count of unique people with AT LEAST ONE sleep record > 12 hours (this part you had working)
people_with_any_long_sleep <- all_sleep %>%
  filter(minute_asleep > hours12) %>%
  distinct(person_id)

count_people_any_sleep_over_12h <- nrow(people_with_any_long_sleep)

print(paste("Number of people with at least one sleep record > 12 hours:", count_people_any_sleep_over_12h))

# 2. Count of unique people where ALL their sleep records are > 12 hours (this part you had working)
people_only_long_sleep <- all_sleep %>%
  group_by(person_id) %>%
  summarise(
    all_records_over_12h = all(minute_asleep > hours12),
    .groups = 'drop'
  ) %>%
  filter(all_records_over_12h == TRUE)

count_people_only_sleep_over_12h <- nrow(people_only_long_sleep)

print(paste("Number of people where ALL their sleep records are > 12 hours:", count_people_only_sleep_over_12h))


# 3. For people who have *at least one day* with over 12 hours of sleep,
#    count how many such days each of those people has.
days_over_12h_per_person <- all_sleep %>%
  filter(minute_asleep > hours12) %>% # First, consider only the days with >12h sleep
  group_by(person_id) %>%             # Then, group by person
  summarise(
    count_days_over_12h = n(),        # Count the number of rows (days) for each person
    .groups = 'drop'                  # Drop grouping
  )

# Display the result
print("Number of days with over 12 hours of sleep for each person who had at least one such day:")
print(days_over_12h_per_person)

# If you want to merge this information back to the list of people who had *any* long sleep
# (though `days_over_12h_per_person` already contains only those people):
# people_with_any_long_sleep_and_counts <- people_with_any_long_sleep %>%
#   left_join(days_over_12h_per_person, by = "person_id")
# print(people_with_any_long_sleep_and_counts)

# Let's also consider a slight variation: what if you want a table showing ALL people,
# and for those who never had a >12h sleep day, it shows 0?
all_people_long_sleep_counts <- all_sleep %>%
  group_by(person_id) %>%
  summarise(
    count_days_over_12h = sum(minute_asleep > hours12, na.rm = TRUE), # Summing TRUEs for days > 12h
    .groups = 'drop'
  )

print("Number of days with over 12 hours of sleep for ALL people (0 if none):")
print(all_people_long_sleep_counts)

# And if you specifically want to filter this list to only show those with >0 days:
all_people_long_sleep_counts_filtered <- all_people_long_sleep_counts %>%
  filter(count_days_over_12h > 0)

print("Number of days with over 12 hours of sleep for people who had at least one such day (alternative way):")
print(all_people_long_sleep_counts_filtered)



sorted_days_over_12h_per_person <- days_over_12h_per_person %>%
  arrange(desc(count_days_over_12h))

# Print the sorted dataframe
print("Sorted list of people by the number of days with over 12 hours of sleep (descending):")
print(sorted_days_over_12h_per_person)


nrow(all_sleep)

# How should sleep data be organized
# Increase to 120 minutes asleep as a minimum number
# Exclude participants with < 90 days of monitoring

hours12 <- 60*24 

# Initial data
initial_participants <- length(unique(all_sleep$person_id))
initial_rows <- nrow(all_sleep)
cat(sprintf("Initial number of participants: %d\n", initial_participants))

# Filter sleep records > 12 hours
fitbit <- all_sleep[minute_asleep < hours12]
participants_after_hour_filter <- length(unique(fitbit$person_id))
nights_after_filter <- nrow(fitbit)
cat(sprintf("After removing sleep records > 24 hours: %d participants (removed %d)\n", 
            participants_after_hour_filter, initial_participants - participants_after_hour_filter))
cat(sprintf("After removing sleep records > 24 hours: %d observations (removed %d)\n", 
            nights_after_filter, initial_rows - nights_after_filter))

# Merge with compliance data
fitbit_merge <- merge(fitbit, compliance, by=c("person_id", "sleep_date"), all.x=TRUE)
fitbit_merge <- fitbit_merge[compliant==TRUE]
participants_after_compliance <- length(unique(fitbit_merge$person_id))
cat(sprintf("After keeping only compliant records: %d participants (removed %d)\n", 
            participants_after_compliance, participants_after_hour_filter - participants_after_compliance))


# Remove sleep records < 120 minutes
merged_cleaned <- fitbit_merge[minute_asleep >= 0]
participants_after_min_sleep <- length(unique(merged_cleaned$person_id))
nights_after_filter <- nrow(fitbit)
cat(sprintf("After removing sleep records < 0 minutes: %d participants (removed %d)\n", 
            participants_after_min_sleep, participants_after_compliance - participants_after_min_sleep))
cat(sprintf("After removing sleep records < 0 hours: %d observations (removed %d)\n", 
            nights_after_filter, initial_rows - nights_after_filter))

# Keep only necessary columns
merged_cleaned <- merged_cleaned[, c("person_id", "sleep_date", "minute_asleep")]

# Merge with demographic data
fitbit <- merge(merged_cleaned, dem, by="person_id", all.x=TRUE)
participants_after_dem_merge <- length(unique(fitbit$person_id))
cat(sprintf("After merging with demographic data: %d participants (removed %d)\n", 
            participants_after_dem_merge, participants_after_min_sleep - participants_after_dem_merge))

# Filter for age >= 18
fitbit[, age := as.numeric(as.Date(sleep_date) - as.Date(date_of_birth)) / 365.25]
fitbit <- fitbit[age >= 18]
participants_after_age <- length(unique(fitbit$person_id))
cat(sprintf("After removing participants < 18 years: %d participants (removed %d)\n", 
            participants_after_age, participants_after_dem_merge - participants_after_age))

# Extract necessary data
fitbit_dat <- fitbit[, c("person_id", "sleep_date", "minute_asleep")]
write_parquet(fitbit_dat, "fitbit_sleep_dat.parquet")

# Calculate monitoring duration and filter for >= 90 days
fitbit_dat[, duration := as.numeric(max(sleep_date) - min(sleep_date)), by=.(person_id)]
participants_before_duration <- length(unique(fitbit_dat$person_id))
fitbit_dat <- fitbit_dat[duration >= 1] # 90 days
participants_after_duration <- length(unique(fitbit_dat$person_id))
cat(sprintf("After removing participants with < 1 days monitoring: %d participants (removed %d)\n", 
            participants_after_duration, participants_before_duration - participants_after_duration))

# Merge with EHR cohort
fitbit_dat <- merge(fitbit_dat, ehr_cohort, by="person_id")
participants_after_ehr <- length(unique(fitbit_dat$person_id))
cat(sprintf("After merging with EHR cohort: %d participants (removed %d)\n", 
            participants_after_ehr, participants_after_duration - participants_after_ehr))

# Get minimum fitbit date per participant
fitbit_min_date <- fitbit_dat[, .(min_fitbit_date = min(sleep_date)), .(person_id)]

# Summary of filtering process
cat(sprintf("\nSUMMARY OF FILTERING:\n"))
cat(sprintf("Started with %d participants\n", initial_participants))
cat(sprintf("Ended with %d participants\n", participants_after_ehr))
cat(sprintf("Total participants removed: %d (%.1f%%)\n", 
            initial_participants - participants_after_ehr, 
            (initial_participants - participants_after_ehr) / initial_participants * 100))

# save.image(file = "full_workspace.RData")


if (load_stored_result)
{
    system(str_glue("gsutil cp {my_bucket}/bq_exports/survey/survey.csv survey.csv"))
    survey <- data.table::fread("survey.csv")
} else {
    codes <- c("1585857","1585860","1585940","1586198","1586201")
    survey_query <- function(code) {
        query <- sprintf("
        SELECT s.person_id, answer
        FROM `%s.ds_survey` s
        WHERE question_concept_id = %s",
        dataset,code)
        result <- download_data(query)
        return(result)
    }

    result <- lapply(codes,function(x) survey_query(x))
    survey <- Reduce(function(x,y) merge(x,y,by="person_id",all.x=TRUE,all.y=TRUE),result)
    colnames(survey) <- c("person_id","cigs_100","smoke_freq","education","alcohol","alcohol_freq")
    data.table::fwrite(survey,"survey.csv")
    system(str_glue("gsutil cp survey.csv {my_bucket}/bq_exports/survey/survey.csv"),intern=T)  
}
survey <- setDT(survey)[,c("person_id","cigs_100","education","alcohol")]
colnames(survey)[2] <- "smoking"
survey[,education := ifelse(education == "Highest Grade: Twelve Or GED" | 
                            education == "Less than a high school degree or equivalent",
                            "no college",
                     ifelse(education == "Highest Grade: College One to Three",
                            "some college",
                     ifelse(education == "College graduate or advanced degree",
                            "college degree",
                            NA)))]
survey[,alcohol := ifelse(alcohol == "Alcohol Participant: Yes",
                          "Yes",
                    ifelse(alcohol == "Alcohol Participant: No",
                           "No",
                          NA))]    
survey[,smoking := ifelse(smoking == "PMI: Skip" | smoking == "PMI: Dont Know" |
                          smoking == "PMI: Prefer Not To Answer",NA,smoking)]

#Phase synchronization, mutual information# look into results if more than one BMI 

bmi_query <- str_glue("
        WITH cohort AS (
              SELECT 
                  DISTINCT person_id 
              FROM `{dataset}.sleep_daily_summary`)
    SELECT 
        m.PERSON_ID, 
        m.measurement_date,
        m.value_as_number
    FROM 
    cohort
    INNER JOIN
    `{dataset}.measurement` m
    ON (m.person_id = cohort.person_id)
    WHERE m.measurement_concept_id = 3038553")

bmi_path <- file.path(
  Sys.getenv("WORKSPACE_BUCKET"),
  "bq_exports",
  "bmi",
  "bmi_*.csv")
if (!load_stored_result) 
{
    bq_table_save(
      bq_dataset_query(dataset, bmi_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      bmi_path,
      destination_format = "CSV")
}
bmi <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/bmi/bmi_*.csv")))
bmi <- bmi[value_as_number > 0 & value_as_number < 200]
bmi <- bmi[!is.na(value_as_number)]

bmi2_query <- str_glue("
        WITH cohort AS (
              SELECT 
                  DISTINCT person_id 
              FROM `{dataset}.sleep_daily_summary`)
    SELECT 
        m.PERSON_ID, 
        m.measurement_date,
        m.value_as_number
    FROM 
    cohort
    INNER JOIN
    `{dataset}.measurement` m
    ON (m.person_id = cohort.person_id)
    WHERE m.measurement_concept_id = 4245997")

bmi2_path <- file.path(
  Sys.getenv("WORKSPACE_BUCKET"),
  "bq_exports",
  "bmi2",
  "bmi2_*.csv")
if (!load_stored_result) 
{
    bq_table_save(
      bq_dataset_query(dataset, bmi2_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      bmi2_path,
      destination_format = "CSV")
}
bmi2 <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/bmi2/bmi2_*.csv")))

#Body height 	3036277
#Body height method 	42527659
#Body height Measured 	3023540
#Body height Stated 	3019171

height_query <- str_glue("
        WITH cohort AS (
              SELECT 
                  DISTINCT person_id 
              FROM `{dataset}.sleep_daily_summary`)
    SELECT 
        m.person_id, 
        m.measurement_date,
        m.value_as_number
    FROM
        cohort
        INNER JOIN
        `{dataset}.measurement` m
        ON (cohort.person_id = m.person_id)
        INNER JOIN
        `{dataset}.concept` c
        ON (m.measurement_concept_id = c.concept_id)
    WHERE 
        c.concept_id = 3036277 OR c.concept_id = 42527659 OR c.concept_id = 3023540 OR c.concept_id = 3019171
     ")

height_path <- file.path(
  Sys.getenv("WORKSPACE_BUCKET"),
  "bq_exports",
  "height",
  "height_*.csv")
if (!load_stored_result)
{
    bq_table_save(
      bq_dataset_query(dataset, height_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      height_path,
      destination_format = "CSV")
}
height <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/height/height_*.csv")))

# Body weight = 3025315
# Body weight Measured = 3013762

weight_query <- str_glue("
        WITH cohort AS (
              SELECT 
                  DISTINCT person_id 
              FROM `{dataset}.sleep_daily_summary`)
    SELECT 
        m.person_id, 
        m.measurement_date,
        m.value_as_number
    FROM
        cohort
        INNER JOIN
        `{dataset}.measurement` m
        ON (cohort.person_id = m.person_id)
        INNER JOIN
        `{dataset}.concept` c
        ON (m.measurement_concept_id = c.concept_id)
    WHERE 
        c.concept_id = 3025315 OR c.concept_id = 3013762
")

weight_path <- file.path(
  Sys.getenv("WORKSPACE_BUCKET"),
  "bq_exports",
  "weight",
  "weight_*.csv")
if (!load_stored_result)
{
    bq_table_save(
      bq_dataset_query(dataset, weight_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      weight_path,
      destination_format = "CSV")
}
weight <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/weight/weight_*.csv")))

#Baseline BMI
colnames(bmi) <- tolower(colnames(bmi))
colnames(bmi2) <- tolower(colnames(bmi2))
colnames(height)[3] <- "height"
colnames(weight)[3] <- "weight"

hw <- merge(height,weight,by=c("person_id","measurement_date"))
setDT(hw)[,bmi := weight / (height*.01)^2]
hw <- hw[bmi < 200]
hw <- hw[,c("person_id","measurement_date","bmi")]
colnames(hw)[3] <- "value_as_number"

bmi <- rbind(bmi,bmi2,hw)
bmi <- bmi[bmi$value_as_number > 10,]
colnames(bmi) <- tolower(colnames(bmi))

bmi <- as.data.table(merge(fitbit_min_date,bmi,by="person_id",all.x=TRUE))
bmi[,time_to_min_date := measurement_date - min_fitbit_date ]
bmi <- bmi[time_to_min_date <= 0 & time_to_min_date >= -365*2]
bmi[,closest := abs(time_to_min_date) == min(abs(time_to_min_date)),.(person_id)]
bmi <- bmi[closest == TRUE]
bmi <- bmi[!duplicated(bmi)]
bmi <- bmi[,.(bmi = value_as_number[closest==TRUE][1],
              time_to_min_date = time_to_min_date[closest==TRUE][1],
              measurement_date = measurement_date[closest==TRUE][1]),.(person_id)]


icd9_query <- function(dataset,icd9_codes)
{
    icd9_terms <- paste('co.CONDITION_SOURCE_VALUE LIKE ',"'",icd9_codes,"'",collapse=' OR ',sep="")
    query <- str_glue("
    SELECT DISTINCT co.person_id, co.condition_start_date,co.condition_source_value 
    FROM 
        `{dataset}.condition_occurrence` co
        INNER JOIN
        `{dataset}.concept` c
        ON (co.condition_source_concept_id = c.concept_id)
    WHERE
        c.VOCABULARY_ID LIKE 'ICD9CM' AND
        ({icd9_terms})
    ")
    download_data(query)
}

icd10_query <- function(dataset,icd10_codes)
{
    icd10_terms <- paste('co.CONDITION_SOURCE_VALUE LIKE ',"'",icd10_codes,"'",collapse=' OR ',sep="")
    query <- str_glue("
    SELECT DISTINCT co.person_id,co.condition_start_date,co.condition_source_value 
    FROM 
        `{dataset}.condition_occurrence` co
        INNER JOIN
        `{dataset}.concept` c
        ON (co.condition_source_concept_id = c.concept_id)
    WHERE
        c.VOCABULARY_ID LIKE 'ICD10CM' AND
        ({icd10_terms})
    ")
    download_data(query)
}

cpt_query <- function(dataset,cpt_codes)
{
    cpt_terms <- paste('c.CONCEPT_CODE LIKE ',"'",cpt_codes,"'",collapse=' OR ',sep="")
    query <- str_glue("
    SELECT DISTINCT p.person_id,c.CONCEPT_CODE AS CODE,p.PROCEDURE_DATE AS ENTRY_DATE
    FROM 
        {dataset}.concept c,
        {dataset}.procedure_occurrence p
        WHERE
        c.VOCABULARY_ID like 'CPT4' AND
        c.CONCEPT_ID = p.PROCEDURE_SOURCE_CONCEPT_ID AND
        ({cpt_terms})
    ")
    download_data(query)
}


download_data <- function(query) {
    tb <- bq_project_query(Sys.getenv('GOOGLE_PROJECT'), query)
    bq_table_download(tb,page_size=50000)
}
if (load_stored_result)
{
    system(str_glue("gsutil cp {my_bucket}/cancer/cancer.csv cancer.csv"))
    cancer <- data.table::fread("cancer.csv")
} else {
    icd9_codes <- c("104", "104.%", "105", "105.%", "106", "106.%", "107", "107.%", "108", "108.%", "109", "109.%", "110", "110.%", "111", "111.%", "112", "112.%", "113", "113.%", "114", "114.%", "115", "115.%", "116", "116.%", "117", "117.%", "118", "118.%", "119", "119.%", "120", "120.%", "121", "121.%", "122", "122.%", "123", "123.%", "124", "124.%", "125", "125.%", "126", "126.%", "127", "127.%", "128", "128.%", "129", "129.%", "130", "130.%", "131", "131.%", "132", "132.%", "133", "133.%", "134", "134.%", "135", "135.%", "136", "136.%", "137", "137.%", "138", "138.%", "139", "139.%", "140", "140.%", "141", "141.%", "142", "142.%", "143", "143.%", "144", "144.%", "145", "145.%", "146", "146.%", "147", "147.%", "148", "148.%", "149", "149.%", "150", "150.%", "151", "151.%", "152", "152.%", "153", "153.%", "154", "154.%", "155", "155.%", "156", "156.%", "157", "157.%", "158", "158.%", "159", "159.%", "160", "160.%", "161", "161.%", "162", "162.%", "163", "163.%", "164", "164.%", "165", "165.%", "166", "166.%", "167", "167.%", "168", "168.%", "169", "169.%", "170", "170.%", "171", "171.%", "172", "172.%", "173", "173.%", "174", "174.%", "175", "175.%", "176", "176.%", "177", "177.%", "178", "178.%", "179", "179.%", "180", "180.%", "181", "181.%", "182", "182.%", "183", "183.%", "184", "184.%", "185", "185.%", "186", "186.%", "187", "187.%", "188", "188.%", "189", "189.%", "190", "190.%", "191", "191.%", "192", "192.%", "193", "193.%", "194", "194.%", "195", "195.%", "196", "196.%", "197", "197.%", "198", "198.%", "199", "199.%", "200", "200.%", "201", "201.%", "202", "202.%", "203", "203.%", "204", "204.%", "205", "205.%", "206", "206.%", "207", "207.%", "208", "208.%", "209", "209.%") 
    icd10_codes <- c("C00", "C00.%", "C01", "C01.%", "C02", "C02.%", "C03", "C03.%", "C04", "C04.%", "C05", "C05.%", "C06", "C06.%", "C07", "C07.%", "C08", "C08.%", "C09", "C09.%", "C10", "C10.%", "C11", "C11.%", "C12", "C12.%", "C13", "C13.%", "C14", "C14.%", "C15", "C15.%", "C16", "C16.%", "C17", "C17.%", "C18", "C18.%", "C19", "C19.%", "C20", "C20.%", "C21", "C21.%", "C22", "C22.%", "C23", "C23.%", "C24", "C24.%", "C25", "C25.%", "C26", "C26.%", "C27", "C27.%", "C28", "C28.%", "C29", "C29.%", "C30", "C30.%", "C31", "C31.%", "C32", "C32.%", "C33", "C33.%", "C34", "C34.%", "C35", "C35.%", "C36", "C36.%", "C37", "C37.%", "C38", "C38.%", "C39", "C39.%", "C40", "C40.%", "C41", "C41.%", "C42", "C42.%", "C43", "C43.%", "C44", "C44.%", "C45", "C45.%", "C46", "C46.%", "C47", "C47.%", "C48", "C48.%", "C49", "C49.%", "C50", "C50.%", "C51", "C51.%", "C52", "C52.%", "C53", "C53.%", "C54", "C54.%", "C55", "C55.%", "C56", "C56.%", "C57", "C57.%", "C58", "C58.%", "C59", "C59.%", "C60", "C60.%", "C61", "C61.%", "C62", "C62.%", "C63", "C63.%", "C64", "C64.%", "C65", "C65.%", "C66", "C66.%", "C67", "C67.%", "C68", "C68.%", "C69", "C69.%", "C70", "C70.%", "C71", "C71.%", "C72", "C72.%", "C73", "C73.%", "C74", "C74.%", "C75", "C75.%", "C76", "C76.%", "C77", "C77.%", "C78", "C78.%", "C79", "C79.%", "C80", "C80.%", "C81", "C81.%", "C82", "C82.%", "C83", "C83.%", "C84", "C84.%", "C85", "C85.%", "C86", "C86.%", "C87", "C87.%", "C88", "C88.%", "C89", "C89.%", "C90", "C90.%", "C91", "C91.%", "C92", "C92.%", "C93", "C93.%", "C94", "C94.%", "C95", "C95.%", "C96", "C96.%", "D00", "D00.%", "D01", "D01.%", "D02", "D02.%", "D03", "D03.%", "D04", "D04.%", "D05", "D05.%", "D06", "D06.%", "D07", "D07.%", "D08", "D08.%", "D09", "D09.%", "D10", "D10.%", "D11", "D11.%", "D12", "D12.%", "D13", "D13.%", "D14", "D14.%", "D15", "D15.%", "D16", "D16.%", "D17", "D17.%", "D18", "D18.%", "D19", "D19.%", "D20", "D20.%", "D21", "D21.%", "D22", "D22.%", "D23", "D23.%", "D24", "D24.%", "D25", "D25.%", "D26", "D26.%", "D27", "D27.%", "D28", "D28.%", "D29", "D29.%", "D30", "D30.%", "D31", "D31.%", "D32", "D32.%", "D33", "D33.%", "D34", "D34.%", "D35", "D35.%", "D36", "D36.%", "D37", "D37.%", "D38", "D38.%", "D39", "D39.%", "D40", "D40.%", "D41", "D41.%", "D42", "D42.%", "D43", "D43.%", "D44", "D44.%", "D45", "D45.%", "D46", "D46.%", "D47", "D47.%", "D48", "D48.%", "D49", "D49.%")
    result_icd9 <- icd9_query(dataset,icd9_codes)
    result_icd10 <- icd10_query(dataset,icd10_codes)
    result_icd <- rbind(result_icd9,result_icd10)
    cancer <- setDT(result_icd)[,.(cancer_date = min(condition_start_date)),.(person_id)]
    data.table::fwrite(cancer,file="cancer.csv")
    system(str_glue("gsutil cp cancer.csv {my_bucket}/cancer/cancer.csv"),intern=T)
}
cancer <- merge(fitbit_min_date,cancer,by="person_id",all.x=TRUE)
cancer[,cancer := ifelse(!is.na(cancer_date),cancer_date < min_fitbit_date,FALSE)]
cancer <- cancer[,c("person_id","cancer")]

if (load_stored_result)
{
    system(str_glue("gsutil cp {my_bucket}/cad/cad.csv cad.csv"))
    cad <- data.table::fread("cad.csv")
} else {
    icd9_codes = c("410","410.%","411","411.%","412","412.%","413","413.%","414","414.%","V45.82")
    cpt_codes = c("33534","33535","33536","33510","33511","33512","33513","33514","33515","33516","33517","33518","33519","33520","33521","33522","33523","92980","92981","92982","92984","92995","92996")
    icd10_codes = c("I25.1%")
    result_icd9 <- icd9_query(dataset,icd9_codes)
    result_icd10 <- icd10_query(dataset,icd10_codes)
    result_cpt <- cpt_query(dataset,cpt_codes)
    
    icd9_merged <- merge(fitbit_min_date,result_icd9,by="person_id")
    icd9_merged <- setDT(icd9_merged)[condition_start_date < min_fitbit_date]
    icd9_agg <- icd9_merged[,.(icd9_length = length(condition_start_date)),.(person_id)]

    icd10_merged <- merge(fitbit_min_date,result_icd10,by="person_id")
    icd10_merged <- setDT(icd10_merged)[condition_start_date < min_fitbit_date]
    icd10_agg <- icd10_merged[,.(icd10_length = length(condition_start_date)),.(person_id)]

    cpt_merged <- merge(fitbit_min_date,result_cpt,by="person_id")
    cpt_merged <- setDT(cpt_merged)[ENTRY_DATE < min_fitbit_date]
    cpt_agg <- cpt_merged[,.(cpt_length = length(ENTRY_DATE)),.(person_id)]

    cad <- merge(fitbit_min_date,icd9_agg,by="person_id",all.x=TRUE)
    cad <- merge(cad,icd10_agg,by="person_id",all.x=TRUE)
    cad <- as.data.table(merge(cad,cpt_agg,by="person_id",all.x=TRUE))

    cad$icd9_length[is.na(cad$icd9_length)] <- 0
    cad$icd10_length[is.na(cad$icd10_length)] <- 0
    cad$cpt_length[is.na(cad$cpt_length)] <- 0

    cad[,cad_flag := icd9_length + icd10_length > 1 | cpt_length > 0]

    cad <- cad[,c("person_id","cad_flag")]
    colnames(cad)[2] <- "cad"
    
    data.table::fwrite(cad,"cad.csv")
    system(str_glue("gsutil cp cad.csv {my_bucket}/cad/cad.csv"))
}

dem_merged <- merge(fitbit_min_date,dem,by="person_id")
dem_merged[,age := as.numeric(as_date(min_fitbit_date) - as_date(date_of_birth)) / 365.25]
colnames(dem_merged) <- tolower(colnames(dem_merged))
dem_merged$race[dem_merged$race == "I prefer not to answer"] <- "Unspecified"
dem_merged$race[dem_merged$race == "No matching concept"] <- "Unspecified"
dem_merged$race[dem_merged$race == "More than one population"] <- "Other"
dem_merged$race[dem_merged$race == "Another single population"] <- "Other"
dem_merged$race[dem_merged$race == "Asian"] <- "Other"
dem_merged$race[dem_merged$race == "None Indicated"] <- "Unspecified"
dem_merged$race[dem_merged$race == "None of these"] <- "Unspecified"
dem_merged$race[dem_merged$race == "PMI: Skip"] <- "Unspecified"
dem_merged$gender[dem_merged$gender == "Not man only, not woman only, prefer not to answer, or skipped"] <- "Unspecified"
dem_merged$gender[dem_merged$gender == "No matching concept"] <- "Unspecified"
dem_merged[,c("race","gender")] <- lapply(dem_merged[,c("race","gender")], 
                                        gsub, pattern = "Unspecified", replacement = NA, fixed = TRUE)
#save.image(file = "full_workspace.RData")

bp_query <- str_glue("
WITH cohort AS (
    SELECT 
        DISTINCT person_id 
        FROM `{dataset}.sleep_daily_summary`),
diatb AS (SELECT
    person_id, measurement_datetime, value_as_number AS dia
    FROM `{dataset}.measurement` m
    INNER JOIN
    `{dataset}.concept` c
    ON (m.measurement_concept_id = c.concept_id)
WHERE 
    lower(c.concept_name) LIKE '%diastolic blood pressure%'),
systb AS (SELECT
    person_id, measurement_datetime, value_as_number AS sys
    FROM `{dataset}.measurement` m
    INNER JOIN
    `{dataset}.concept` c
    ON (m.measurement_concept_id = c.concept_id)
WHERE 
    lower(c.concept_name) LIKE '%systolic blood pressure%')
SELECT DISTINCT cohort.person_id, d.measurement_datetime, sys, dia
FROM cohort
INNER JOIN
diatb d
ON (cohort.person_id = d.person_id)
INNER JOIN systb s
ON (d.person_id = s.person_id)
WHERE 
d.measurement_datetime = s.measurement_datetime")

bp_path <- file.path(
      Sys.getenv("WORKSPACE_BUCKET"),
      "bp",
      "bp_*.csv")

if (!load_stored_result)
{
    bq_table_save(
      bq_dataset_query(dataset, bp_query, billing = Sys.getenv("GOOGLE_PROJECT")),
      bp_path,
      destination_format = "CSV")
}
bp <- as.data.table(read_bucket(str_glue("{my_bucket}/bp/bp_*.csv")))

bp <- as.data.table(merge(fitbit_min_date,bp,by="person_id",all.x=TRUE))
bp[,time_to_min_date := lubridate::as_date(measurement_datetime) - min_fitbit_date ]
bp <- bp[time_to_min_date <= 0 & time_to_min_date >= -365*2]
bp[,closest := abs(time_to_min_date) == min(abs(time_to_min_date)),.(person_id)]
bp <-  bp[closest == TRUE]
bp <- bp[!duplicated(bp)]
bp <- bp[,.(sys = sys[closest==TRUE][1],
              time_to_min_date = time_to_min_date[closest==TRUE][1],
              measurement_date = measurement_datetime[closest==TRUE][1]),.(person_id)]
bp <- bp[,c("person_id","sys","measurement_date")]
#save.image(file = "full_workspace.RData")

head(weight)

#system(str_glue("gsutil cp {my_bucket}/covariates/covariates_cleaned.csv covariates_cleaned.csv"))
#covariates_cleaned <- data.table::fread("covariates_cleaned.csv")
#bmi_baseline <- covariates_cleaned[,c("person_id","bmi")]
#bp <- covariates_cleaned[,c("person_id","sys")]
#height <- covariates_cleaned[,c("person_id","height")]
#weight <- covariates_cleaned[,c("person_id","weight")]

cov_dat <- merge(fitbit_min_date,dem_merged,by="person_id",all.x=TRUE)
#cov_dat <- merge(cov_dat,cad,by="person_id",all.x=TRUE)
cov_dat <- merge(cov_dat,cancer,by="person_id",all.x=TRUE)
cov_dat <- merge(cov_dat,bmi,by="person_id",all.x=TRUE)
cov_dat <- merge(cov_dat,bp,by="person_id",all.x=TRUE)
cov_dat <- merge(cov_dat,survey,by="person_id",all.x=TRUE)
# Aggregate height data by taking the mean for each person_id
height_avg <- height[, .(height = mean(height, na.rm = TRUE)), by = person_id]
cov_dat <- merge(cov_dat,height_avg,by="person_id",all.x=TRUE)
# Aggregate weight data by taking the mean for each person_id
weight_avg <- weight[, .(weight = mean(weight, na.rm = TRUE)), by = person_id]
cov_dat <- merge(cov_dat,weight_avg,by="person_id",all.x=TRUE)
fitbit_agg <- fitbit_dat[,.(mean_sleep = mean(minute_asleep)),.(person_id)]
fitbit_agg$mean_sleep = fitbit_agg$mean_sleep / 60
fitbit_agg_sd <- fitbit_dat[,.(sd_sleep = sd(minute_asleep)),.(person_id)] 
fitbit_agg_sd$sd_sleep = fitbit_agg_sd$sd_sleep / 60
fitbit_agg_total <-fitbit_dat[,.(total = sum(minute_asleep)),.(person_id)] 
cov_dat <- merge(fitbit_agg,cov_dat,by="person_id",all.x=TRUE)
cov_dat <- merge(fitbit_agg_sd,cov_dat,by="person_id",all.x=TRUE)
cov_dat <- cov_dat[,c("person_id","race","gender","age",
                      "cancer","bmi","sys","smoking","education","alcohol","mean_sleep","sd_sleep","height","weight")]
#save.image(file = "full_workspace.RData")

tb1 <- summaryM(age + race + gender + mean_sleep + sd_sleep + bmi + cancer + smoking + education + alcohol + sys ~ 1, data=cov_dat)
out <- html(tb1, caption='',
     exclude1=F, npct='both', digits=5,long=T,
     prmsd=TRUE, brmsd=T, msdsize=mu$smaller2, longtable=T, middle.bold=T, vnames = c('names'))
IRdisplay::display_html(out)

phecodes <- merge(icd_result,phemap,by="condition_source_value",all.x=TRUE)
head(phecodes)

dim(phecodes)

#save.image(file = "full_workspace.RData")

phecodes <- merge(icd_result,phemap,by="condition_source_value",all.x=TRUE)

#update to 90 days 

#fitbit_dat <- fitbit_dat_time

fitbit_summary <- fitbit_dat[,.(min_date = min(sleep_date),
                                max_date = max(sleep_date),
                                duration = as.numeric(max(sleep_date) - min(sleep_date))),
                            .(person_id)]
fitbit_dat_min_date <- fitbit_summary[duration >= 0,c("person_id","min_date","max_date","duration")]
merged <- merge(fitbit_dat_min_date,phecodes,by='person_id',all.x=TRUE)

fitbit_dat_time <- fitbit_dat

fitbit_dat_time <- copy(fitbit_dat)

length(unique(fitbit_dat_min_date$person_id))

write_parquet(fitbit_dat_min_date,"PERSON_TEST.parquet")

#Analysis type:

Analysis <- function(input_table){
    result <- input_table[, .(
        average_daily_sleep = mean(minute_asleep),
        sd_daily_sleep = sd(minute_asleep)
    ), .(person_id)]
    return(result)
}

calculate_daily_sleep_stats <- function(input_table) {
    input_table[, day_of_year := yday(sleep_date)]
    
    result <- input_table[, .(
        avg_sleep = mean(minute_asleep),
        sd_sleep = sd(minute_asleep)
    ), by = .(person_id, day_of_year)]
    
    # Ensure all 366 days are present for each person by creating a complete grid
    full_days <- CJ(person_id = unique(result$person_id), day_of_year = 1:366)
    result <- merge(full_days, result, by = c("person_id", "day_of_year"), all.x = TRUE)
    
    # Reshape to wide format, filling missing values with NA
    avg_wide <- dcast(result, person_id ~ day_of_year, value.var = "avg_sleep", 
                      fun.aggregate = mean, na.rm = FALSE, fill = NA)
    sd_wide <- dcast(result, person_id ~ day_of_year, value.var = "sd_sleep", 
                     fun.aggregate = mean, na.rm = FALSE, fill = NA)
    
    # Rename columns consistently for 366 days
    setnames(avg_wide, as.character(1:366), paste0("avg_sleep_day_", 1:366))
    setnames(sd_wide, as.character(1:366), paste0("sd_sleep_day_", 1:366))
    
    # Merge average and sd data
    final_result <- merge(avg_wide, sd_wide, by = "person_id")
    
    return(final_result)
}

# Calculate overall average and sd
overall_sleep_stats <- Analysis(fitbit_dat)

daily_sleep_stats <- calculate_daily_sleep_stats(fitbit_dat)

fitbit_dat <- overall_sleep_stats #merge(overall_sleep_stats, daily_sleep_stats, by = "person_id")

all_ids <- merged[,c("person_id")]
all_ids <- all_ids[!duplicated(all_ids)]

merged[, post_month_6 := ((condition_start_date - min_date) > 180) & 
                         (condition_start_date == min(condition_start_date)),
                         by=.(person_id,phecode)]
merged[, prior_month_6 := ((condition_start_date - min_date) <= 180) & 
                          (condition_start_date == min(condition_start_date)),
                         by=.(person_id,phecode)]
head(merged[!is.na(condition_source_value)])
#save.image(file = "full_workspace.RData")

dim(merged)

# merge 
library(data.table)
# Function to print object size
print_size <- function(obj_name) {
  size <- object.size(get(obj_name))
  print(paste(obj_name, "size:", format(size, units = "auto")))
}

# Load workspace
print("Loading working space")
#load("full_workspace.RData")
print("Loaded workspace")

# Print sizes of loaded objects
print_size("all_ids")
print_size("merged")
print_size("ehr_cohort")
print_size("socio_eco_df")
print_size("fitbit_dat")

# Initialize dataframes as data.tables
print("Initializing data.tables")
setDT(all_ids)
setDT(merged)
setDT(ehr_cohort)
setDT(socio_eco_df)
setDT(fitbit_dat)
print("Done initializing")

# Perform merges
print("Performing merges")

print("Merge 1: all_ids and merged")
merged <- merge(all_ids, merged, by="person_id", all.x=TRUE)
dim(merged)
print_size("merged")
rm(all_ids)
gc()

print("Merge 2: merged and fitbit_dat")
merged <- merge(merged, fitbit_dat, by="person_id")
dim(merged)
print_size("merged")
rm(fitbit_dat)
gc()

print("Merge 3: ehr_cohort and merged")
merged <- merge(ehr_cohort, merged, by='person_id')
dim(merged)
print_size("merged")
rm(ehr_cohort)
gc()

print("Merge 4: merged and socio_eco_df")
merged <- merge(merged, socio_eco_df, by='person_id', all.x=TRUE)
dim(merged)
print_size("merged")
rm(socio_eco_df)
gc()

print("Done")



head(merged)

length(unique(merged$person_id))

#save.image(file = "full_workspace_all_data_2.RData")

#load("full_workspace_all_data_2.RData")

# Rename the columns and select the desired ones
merged <- merged %>%
  rename(
    # Assuming 'zip_code' was already renamed from 'zip3' in the previous step
    zip_code = zip3,
    perc_with_assisted_income = fraction_assisted_income,
    perc_with_high_school_education = fraction_high_school_edu,
    perc_with_no_health_insurance = fraction_no_health_ins,
    perc_in_poverty = fraction_poverty,
    perc_vacant_housing = fraction_vacant_housing
    # Add any other renames if needed, format: new_name = old_name
  )

length(unique(merged$person_id))

print("Additional operations")
merged[,has := post_month_6]
merged[,entry := condition_start_date]
merged[,had := prior_month_6]
print_size("merged")

print("Performing dcast operation")
daily_sleep_cols <- grep("^avg_sleep_day_", names(merged), value = TRUE)
daily_sleep_cols_sd <- grep("^sd_sleep_day_", names(merged), value = TRUE)


# Create the left-hand side of the formula
lhs <- paste(
  "person_id", "max_date", "min_date", "duration", "average_daily_sleep",
  "sd_daily_sleep", "zip_code", "median_income", "perc_with_assisted_income",
  "perc_with_high_school_education", "perc_with_no_health_insurance", 
  "perc_in_poverty", "perc_vacant_housing",
  sep = " + "
)

# Combine the left-hand side with the right-hand side
formula_string <- paste(lhs, "~ phecode")

# Convert to formula
sleep_formula <- as.formula(formula_string)

# Perform the dcast operation
dt_icd <- dcast(
  merged, 
  formula = sleep_formula,
  value.var = c("has", "entry", "had"),
  fun.aggregate = list(
    phe = function(x) length(which(x == TRUE)) > 0,
    date = function(x) as.character(ifelse(any(!is.na(x)), as.character(min(x, na.rm = TRUE)), NA))
  )
)


print_size("dt_icd")

print("Done")



dim(dt_icd)

# pull sex data 
# look into results if more than one BMI 


sex_query <- str_glue("
    WITH cohort AS (
        SELECT DISTINCT person_id 
        FROM `{dataset}.sleep_daily_summary`
    )
    SELECT
        c.person_id,
        COALESCE(con.concept_name, 'Unknown') AS sex_concept
    FROM cohort c
    LEFT JOIN `{dataset}.person` p ON c.person_id = p.person_id
    LEFT JOIN `{dataset}.concept` con ON p.sex_at_birth_concept_id = con.concept_id
    ORDER BY c.person_id
")



sex_path <- file.path(
  Sys.getenv("WORKSPACE_BUCKET"),
  "bq_exports",
  "sex",
  "sex_*.csv")

bq_table_save(
  bq_dataset_query(dataset, sex_query, billing = Sys.getenv("GOOGLE_PROJECT")),
  sex_path,
  destination_format = "CSV")

sex <- as.data.table(read_bucket(str_glue("{my_bucket}/bq_exports/sex/sex_*.csv")))

head(sex)

#final_dat <- read_parquet("final_data_v8_5_30_no_daily_one_day_less_strict_hours.parquet")

final_dat <- merge(dt_icd, dem, by = "person_id", all.x = TRUE)


dim(final_dat)

final_dat_with_sex <- left_join(final_dat, sex, by = "person_id")

dim(final_dat_with_sex)

write_parquet(final_dat_with_sex,"final_data_v8_5_30_no_daily_one_day_less_strict_hours.parquet")

final_dat_steps <- merge(final_dat, daily_sleep_stats, by = "person_id", all.x = TRUE)
write_parquet(final_dat_steps,"final_data_v8_5_13_daily_one_day.parquet")



