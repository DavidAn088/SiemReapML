# Shared helpers for the R scripts. Open the project folder in RStudio (or setwd() to it)
# and source each script; paths are resolved from the folder that holds config/project.toml.

find_root <- function(start = getwd()) {
  p <- normalizePath(start, winslash = "/")
  repeat {
    if (file.exists(file.path(p, "config", "project.toml"))) return(p)
    parent <- dirname(p)
    if (parent == p) stop("config/project.toml not found above ", start)
    p <- parent
  }
}

# Settings exported by the preprocessing notebook from the project configuration, so R and
# Python read one configuration. The binary project writes data/processed/r_settings.json;
# another configuration names its folder in the SIEMREAP_PROCESSED environment variable
# (the 5-class notebook sets it; in RStudio: Sys.setenv(SIEMREAP_PROCESSED = "data/processed_5class")).
read_settings <- function(root) {
  proc <- Sys.getenv("SIEMREAP_PROCESSED", "data/processed")
  f <- file.path(root, proc, "r_settings.json")
  if (!file.exists(f)) stop("Run the preprocessing notebook first (it writes ", f, ").")
  jsonlite::fromJSON(f)
}

out_dir <- function(root, settings, stage, kind = "aggregate") {
  outputs <- if (is.null(settings$outputs_dir)) "outputs" else settings$outputs_dir
  base <- if (identical(settings$mode, "smoke")) file.path(root, outputs, "smoke", stage)
          else file.path(root, outputs, stage)
  d <- file.path(base, kind)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

read_cells <- function(root, settings) {
  cells <- read.csv(file.path(root, settings$files$cells), stringsAsFactors = FALSE)
  # multiclass: the class name balances the folds; binary: Clay/Sand
  cells$cls <- if ("class_name" %in% names(cells)) cells$class_name else ifelse(cells$y == 1, "Clay", "Sand")
  cells
}

write_session_info <- function(dir, name) {
  writeLines(c(capture.output(sessionInfo()),
               paste("blockCV", as.character(utils::packageVersion("blockCV")))),
             file.path(dir, paste0("session_info_", name, ".txt")))
}
