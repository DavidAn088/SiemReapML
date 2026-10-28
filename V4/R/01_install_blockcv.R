# Install blockCV exactly as published on GitHub (rvalavi/blockCV), pinned to one commit.
# Nothing in blockCV is changed or re-written.
#
# Two ways, tried in this order:
#   1. from the GitHub ZIP of that commit, saved in data/raw/ (works without internet in R):
#      download https://github.com/rvalavi/blockCV/archive/<commit>.zip in a browser
#   2. from GitHub with the 'remotes' package (needs internet in R)
# On Windows it also needs Rtools that matches your R version (blockCV contains C++ code).
# Notebook 02 runs this script for you.

# Install into a writable library (R's own folder is often read-only on Windows).
lib <- .libPaths()[1]
if (file.access(lib, 2) != 0) {
  lib <- strsplit(Sys.getenv("R_LIBS_USER"), .Platform$path.sep)[[1]][1]
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}
cat("R library:", lib, "\n")
if (!requireNamespace("jsonlite", quietly = TRUE)) install.packages("jsonlite", lib = lib, repos = "https://cloud.r-project.org")

s <- jsonlite::fromJSON("data/r_settings.json")
repo <- s$blockcv$github_repo
commit <- s$blockcv$github_commit
want_version <- s$blockcv$version
# Any blockCV*.zip in data/raw (the browser may rename it, e.g. "blockCV-....zip (1)" or ".zip.zip").
# The commit is checked from inside the ZIP: GitHub names its top folder blockCV-<commit>.
zip_file <- ""
for (z in list.files(file.path("data", "raw"), pattern = "^blockCV.*zip", ignore.case = TRUE, full.names = TRUE)) {
  top <- unique(sub("/.*", "", unzip(z, list = TRUE)$Name))
  cat("Found", basename(z), "-> folder inside:", top, "\n")
  if (identical(top, paste0("blockCV-", commit))) zip_file <- z
}
marker <- file.path(lib, "blockCV", "INSTALLED_FROM_GITHUB_COMMIT")   # written after a ZIP install

test_source <- Sys.getenv("V4_BLOCKCV_SOURCE")   # software tests only: a git clone of the same commit

installed_commit <- function() {
  if (!requireNamespace("blockCV", quietly = TRUE)) return(NA)
  d <- packageDescription("blockCV")
  if (!is.null(d$RemoteSha)) return(d$RemoteSha)
  m <- file.path(find.package("blockCV"), "INSTALLED_FROM_GITHUB_COMMIT")
  if (file.exists(m)) return(trimws(readLines(m, warn = FALSE)[1]))
  if (nzchar(test_source)) return(commit)
  NA
}

needed <- c("sf", "terra", "ggplot2", "cowplot", "automap", "Rcpp")
missing <- setdiff(needed, rownames(installed.packages()))
if (length(missing)) {
  cat("Installing the packages blockCV needs:", missing, "\n")
  install.packages(missing, lib = lib, repos = "https://cloud.r-project.org")
}

if (identical(installed_commit(), commit) && as.character(packageVersion("blockCV")) == want_version) {
  cat("blockCV", want_version, "from", repo, "commit", commit, "is already installed.\n")
} else if (nzchar(test_source)) {
  cat("TEST INSTALL from the local clone", test_source, "\n")
  install.packages(test_source, repos = NULL, type = "source", lib = lib)
} else if (nzchar(zip_file)) {
  cat("Installing blockCV from", zip_file, "\n")
  tmp <- file.path(tempdir(), "blockcv_zip")
  unlink(tmp, recursive = TRUE)
  unzip(zip_file, exdir = tmp)
  src <- list.dirs(tmp, recursive = FALSE)[1]                 # blockCV-<commit>/
  install.packages(src, repos = NULL, type = "source", lib = lib)
  writeLines(commit, file.path(lib, "blockCV", "INSTALLED_FROM_GITHUB_COMMIT"))
} else if (requireNamespace("remotes", quietly = TRUE)) {
  remotes::install_github(paste0(repo, "@", commit), lib = lib, upgrade = "never", dependencies = FALSE)
} else {
  stop("\n\nNo blockCV ZIP of commit ", commit, " was found in data/raw, and R cannot download it. Do this once:\n",
       "  1. In your web browser open\n     https://github.com/", repo, "/archive/", commit, ".zip\n",
       "  2. Save the file (do NOT unzip it) in the folder\n     ", normalizePath(file.path("data", "raw"), mustWork = FALSE), "\n",
       "  3. Run this cell again.\n")
}

library(blockCV)
cat("blockCV version:", as.character(packageVersion("blockCV")), "\n")
cat("GitHub commit  :", installed_commit(), "\n")
stopifnot(as.character(packageVersion("blockCV")) == want_version, identical(installed_commit(), commit))
