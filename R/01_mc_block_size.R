# Block size from spatial autocorrelation with the ORIGINAL blockCV package (Valavi et al. 2019),
# for the multiclass soil-class target (design report section 11).
#
# Input : <processed>/cells_development.csv   development samples (external boreholes removed)
# Output: <folds>/block_size_decision.json     the block size used by R/02_blockcv_folds.R
#         <outputs>/04_spatial_autocorrelation/aggregate/
#             blockcv_autocor_by_slice_class.csv   one blockCV range per depth slice and class
#             block_size_summary.json, fig_autocor_ranges.png
#
# Method. The data are 3D, so the horizontal autocorrelation is measured within depth slices:
# for every slice (`slice_m`, default 1.5 m = the SPT cycle) each Downtown development borehole
# gives one value per class, the share of its samples in that slice that belong to the class
# (one indicator per class, as blockCV treats a multi-class response). blockCV::cv_spatial_autocor
# fits a variogram to each indicator (automap) and returns its range. The block size is the
# median of all these ranges, which is how cv_spatial_autocor itself combines several layers
# ("the median of all calculated ranges"). Not used, and listed as such: slices with fewer than
# `min_boreholes_per_slice` boreholes, and class indicators present in fewer than
# `min_boreholes_with_class` boreholes of the slice (a variogram of a handful of presences has
# no defined range). Both limits are fixed in the configuration before any fit.
# Nothing here uses a model or a test score.

source(file.path(if (file.exists("R/_common.R")) "R" else ".", "_common.R"))
root <- find_root()
setwd(root)
suppressPackageStartupMessages({ library(sf); library(blockCV); library(ggplot2) })
st <- read_settings(root)
sp <- st$spatial
agg <- out_dir(root, st, "04_spatial_autocorrelation", "aggregate")
fdir <- file.path(root, st$files$folds_dir)
dir.create(fdir, recursive = TRUE, showWarnings = FALSE)
set.seed(st$seed)

cells <- read_cells(root, st)
dt <- cells[cells$is_downtown == 1, ]
classes <- st$classes                     # data frame: name, group (classes.json order = y)
K <- nrow(classes)
dt$slice <- floor(dt$depth_top_m / sp$slice_m + 1e-9)
slice_label <- function(s) sprintf("%.1f-%.1f m", s * sp$slice_m, (s + 1) * sp$slice_m)

rows <- list()
for (s in sort(unique(dt$slice))) {
  d <- dt[dt$slice == s, ]
  bh <- unique(d[, c("BH_Code", "UTM_Easting_m", "UTM_Northing_m")])
  for (k in seq_len(K) - 1) {
    share <- tapply(d$y == k, d$BH_Code, mean)
    bh[[paste0("c", k)]] <- as.numeric(share[bh$BH_Code])
  }
  pts <- st_as_sf(bh, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = st$crs_epsg)
  for (k in seq_len(K) - 1) {
    col <- paste0("c", k)
    v <- bh[[col]]
    row <- data.frame(slice = slice_label(s), slice_top_m = s * sp$slice_m, class = k, class_name = classes$name[k + 1],
                      n_boreholes = nrow(bh), boreholes_with_class = sum(v > 0), range_m = NA_real_, status = "ok")
    if (nrow(bh) < sp$min_boreholes_per_slice) {
      row$status <- "too few boreholes"
    } else if (length(unique(v)) < 2) {
      row$status <- "constant indicator"
    } else if (sum(v > 0) < sp$min_boreholes_with_class) {
      row$status <- "class in too few boreholes"
    } else {
      fit <- tryCatch(suppressMessages(cv_spatial_autocor(x = pts, column = col, plot = FALSE, progress = FALSE)),
                      error = function(e) conditionMessage(e))
      if (is.character(fit)) row$status <- paste("fit failed:", fit) else row$range_m <- as.numeric(fit$range)
    }
    rows[[length(rows) + 1]] <- row
  }
}
tab <- do.call(rbind, rows)
write.csv(tab, file.path(agg, "blockcv_autocor_by_slice_class.csv"), row.names = FALSE)
ok <- tab[!is.na(tab$range_m), ]
if (nrow(ok) == 0) stop("no autocorrelation range could be estimated; see blockcv_autocor_by_slice_class.csv")

size <- round(stats::median(ok$range_m))
by_class <- aggregate(range_m ~ class + class_name, data = ok, FUN = stats::median)
summary <- list(
  block_size_m = size,
  method = paste0("median of blockCV::cv_spatial_autocor ranges over depth slices (", sp$slice_m,
                  " m) x class indicators (Downtown development boreholes)"),
  ranges_used = nrow(ok), ranges_not_used = nrow(tab) - nrow(ok),
  range_quantiles_m = as.list(stats::quantile(ok$range_m, c(0.1, 0.25, 0.5, 0.75, 0.9))),
  median_range_by_class_m = setNames(as.list(round(by_class$range_m)), by_class$class_name),
  blockcv_version = as.character(utils::packageVersion("blockCV")),
  automap_version = as.character(utils::packageVersion("automap")),
  note = "Ranges of the class indicators (response), as recommended by blockCV when residuals are not yet available.")
jsonlite::write_json(summary, file.path(agg, "block_size_summary.json"), auto_unbox = TRUE, pretty = TRUE)
jsonlite::write_json(summary, file.path(fdir, "block_size_decision.json"), auto_unbox = TRUE, pretty = TRUE)

p <- ggplot(ok, aes(x = range_m, y = slice_top_m, colour = class_name)) +
  geom_point(size = 2, alpha = 0.8) +
  geom_vline(xintercept = size, linetype = 2) +
  annotate("text", x = size, y = 0, label = paste0(" block size ", size, " m"), hjust = 0, vjust = 1, size = 3) +
  scale_y_reverse() + labs(x = "blockCV autocorrelation range (m)", y = "Depth slice top (m)", colour = "Class",
                           title = "Autocorrelation range of each class indicator by depth slice") +
  theme_bw(9)
ggsave(file.path(agg, "fig_autocor_ranges.png"), p, width = 7, height = 5, dpi = 200)
write_session_info(agg, "block_size")
print(tab)
cat("\nBlock size (median of", nrow(ok), "ranges):", size, "m\n")
