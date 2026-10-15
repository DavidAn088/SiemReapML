# Install the R packages for the spatial analysis and the ORIGINAL blockCV package from the
# supplied repository (github.com/rvalavi/blockCV, version 4.0-1).
#
# Windows: install Rtools that matches your R version first (blockCV contains C++ code).
# Place the unzipped repository at data/raw/blockCV-master (or the zip at
# data/raw/blockCV-master.zip; it is unzipped automatically).

source(file.path(if (file.exists("R/_common.R")) "R" else ".", "_common.R"))
root <- find_root()

# Install into the first writable library; when R itself is read-only (common on Windows when
# run from Rscript or a notebook), create and use the personal library (R_LIBS_USER).
lib <- .libPaths()[1]
if (file.access(lib, 2) != 0) {
  lib <- strsplit(Sys.getenv("R_LIBS_USER"), .Platform$path.sep)[[1]][1]
  dir.create(lib, recursive = TRUE, showWarnings = FALSE)
  .libPaths(c(lib, .libPaths()))
}
cat("R library:", lib, "\n")

cran <- c("sf", "terra", "automap", "gstat", "ggplot2", "cowplot", "Rcpp", "jsonlite")
missing <- setdiff(cran, rownames(installed.packages()))
if (length(missing)) install.packages(missing, lib = lib, repos = "https://cloud.r-project.org")

if (requireNamespace("blockCV", quietly = TRUE) && as.character(packageVersion("blockCV")) == "4.0.1") {
  cat("blockCV 4.0.1 already installed.\n")
  quit(save = "no")
}

src <- file.path(root, "data", "raw", "blockCV-master")
zip <- paste0(src, ".zip")
if (!dir.exists(src) && file.exists(zip)) utils::unzip(zip, exdir = dirname(src))
if (!dir.exists(src)) stop("blockCV source not found at ", src)

desc <- read.dcf(file.path(src, "DESCRIPTION"))
cat("Installing blockCV", desc[, "Version"], "from", src, "\n")
status <- system2(file.path(R.home("bin"), "R"), c("CMD", "INSTALL", paste0("--library=", shQuote(lib)), shQuote(src)))
if (status != 0) stop("blockCV installation failed (on Windows check that Rtools is installed).")

library(blockCV)
stopifnot(as.character(packageVersion("blockCV")) == "4.0.1")
cat("blockCV", as.character(packageVersion("blockCV")), "installed.\n")
print(sapply(cran, function(p) as.character(packageVersion(p))))
