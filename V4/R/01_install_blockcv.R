# Install blockCV exactly as published on GitHub (rvalavi/blockCV), pinned to one commit.
# Nothing in blockCV is changed or re-written.
#
# Needs internet. On Windows it also needs Rtools that matches your R version
# (blockCV contains C++ code). Notebook 02 runs this script for you.

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

test_source <- Sys.getenv("V4_BLOCKCV_SOURCE")   # software tests only (no internet): a git clone of the same commit

installed_ok <- function() {
  if (!requireNamespace("blockCV", quietly = TRUE)) return(FALSE)
  d <- packageDescription("blockCV")
  same_commit <- identical(d$RemoteSha, commit) || nzchar(test_source)
  identical(as.character(packageVersion("blockCV")), want_version) && same_commit
}

if (installed_ok()) {
  cat("blockCV", want_version, "from", repo, "commit", commit, "is already installed.\n")
} else if (nzchar(test_source)) {
  cat("TEST INSTALL from the local clone", test_source, "\n")
  install.packages(test_source, repos = NULL, type = "source", lib = lib)
} else {
  cran <- c("sf", "terra", "ggplot2", "cowplot", "automap", "Rcpp", "jsonlite", "remotes")
  missing <- setdiff(cran, rownames(installed.packages()))
  if (length(missing)) install.packages(missing, lib = lib, repos = "https://cloud.r-project.org")
  remotes::install_github(paste0(repo, "@", commit), lib = lib, upgrade = "never", dependencies = TRUE)
}

library(blockCV)
d <- packageDescription("blockCV")
cat("blockCV version:", as.character(packageVersion("blockCV")), "\n")
cat("GitHub commit  :", if (is.null(d$RemoteSha)) "(local test install)" else d$RemoteSha, "\n")
stopifnot(as.character(packageVersion("blockCV")) == want_version)
