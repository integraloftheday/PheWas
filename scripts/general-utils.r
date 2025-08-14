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

read_bucket <- function (export_path)
{
    col_types <- NULL
    bind_rows(map(system2("gsutil", args = c("ls", export_path),
        stdout = TRUE, stderr = TRUE), function(csv) {
        message(str_glue("Loading {csv}."))
        chunk <- read_csv(pipe(str_glue("gsutil cat {csv}")),
            col_types = col_types, show_col_types = FALSE)
        if (is.null(col_types)) {
            col_types <- spec(chunk)
        }
        chunk
    }))
}

download_data <- function (query)
{
    tb <- bq_project_query(Sys.getenv("GOOGLE_PROJECT"), query)
    bq_table_download(tb, page_size = 1e+05)
}