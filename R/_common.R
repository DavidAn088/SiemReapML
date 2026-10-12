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

# Settings exported by notebooks/01_preprocessing.ipynb from config/project.toml, so R and
# Python read one configuration.
read_settings <- function(root) {
  f <- file.path(root, "data", "processed", "r_settings.json")
  if (!file.exists(f)) stop("Run notebooks/01_preprocessing.ipynb first (it writes ", f, ").")
  jsonlite::fromJSON(f)
}

out_dir <- function(root, settings, stage, kind = "aggregate") {
  base <- if (identical(settings$mode, "smoke")) file.path(root, "outputs", "smoke", stage)
          else file.path(root, "outputs", stage)
  d <- file.path(base, kind)
  dir.create(d, recursive = TRUE, showWarnings = FALSE)
  d
}

read_cells <- function(root, settings) {
  cells <- read.csv(file.path(root, settings$files$cells), stringsAsFactors = FALSE)
  cells$cls <- ifelse(cells$y == 1, "Clay", "Sand")
  cells
}

write_session_info <- function(dir, name) {
  writeLines(c(capture.output(sessionInfo()),
               paste("blockCV", as.character(utils::packageVersion("blockCV")))),
             file.path(dir, paste0("session_info_", name, ".txt")))
}
