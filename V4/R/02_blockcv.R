# Block size and spatial folds, using only blockCV's own functions.
#
# 1. Block size = cv_spatial_autocor()$range.
#    One point per borehole; one column per 3 m depth slice (the borehole's clay share in it).
#    With several columns, blockCV fits a variogram to each and returns the MEDIAN of their
#    ranges as $range. We use that number as it is.
# 2. Folds = cv_spatial() with k = 5 square blocks of that size (and 2x, 3x for sensitivity),
#    one point per borehole, balanced on the borehole clay share.
#
# Input  (written by notebook 01): <processed>/slices_<w>m.csv, <processed>/boreholes.csv
# Output: <folds>/block_size.json, <folds>/folds_x1.csv, folds_x2.csv, folds_x3.csv
#         <out>/... tables and figures; <out>/private/... maps with borehole locations

suppressPackageStartupMessages({ library(sf); library(blockCV); library(ggplot2) })
s <- jsonlite::fromJSON("data/r_settings.json")
b <- s$blockcv
dir.create(s$folds_dir, recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(s$out_dir, "private"), recursive = TRUE, showWarnings = FALSE)
cat("blockCV", as.character(packageVersion("blockCV")), "| area:", s$area, "\n\n")

autocorrelation <- function(slice_m) {
  t <- read.csv(file.path(s$processed_dir, sprintf("slices_%gm.csv", slice_m)), check.names = FALSE)
  cols <- grep("^clay_", names(t), value = TRUE)
  x <- st_as_sf(t, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = s$epsg)
  set.seed(b$seed)
  a <- cv_spatial_autocor(x = x, column = cols, plot = FALSE, progress = FALSE)
  list(result = a, boreholes = nrow(t), columns = cols)
}

# ---- 1. block size from the main slice thickness ----------------------------------------
main <- autocorrelation(b$slice_m)
print(main$result)
block_m <- main$result$range
cat(sprintf("\nBlock size = blockCV range = %.1f m (%d boreholes, %d slices of %g m)\n\n",
            block_m, main$boreholes, length(main$columns), b$slice_m))
write.csv(main$result$range_table, file.path(s$out_dir, "range_by_slice.csv"), row.names = FALSE)
ggsave(file.path(s$out_dir, "fig_range_by_slice.png"), main$result$plots$barchart, width = 6, height = 4, dpi = 200)

# ---- information only: other slice thicknesses ------------------------------------------
info <- do.call(rbind, lapply(b$info_slices_m, function(w) {
  r <- tryCatch(autocorrelation(w), error = function(e) NULL)
  if (is.null(r)) return(data.frame(slice_m = w, boreholes = NA, slices = NA, blockcv_range_m = NA))
  data.frame(slice_m = w, boreholes = r$boreholes, slices = length(r$columns), blockcv_range_m = r$result$range)
}))
info <- rbind(data.frame(slice_m = b$slice_m, boreholes = main$boreholes, slices = length(main$columns),
                         blockcv_range_m = block_m), info)
info$used_for_block_size <- info$slice_m == b$slice_m
cat("blockCV range for other slice thicknesses (information only):\n")
print(info, row.names = FALSE)
write.csv(info, file.path(s$out_dir, "range_other_slices_info_only.csv"), row.names = FALSE)

# ---- 2. folds ---------------------------------------------------------------------------
bh <- read.csv(file.path(s$processed_dir, "boreholes.csv"))
x <- st_as_sf(bh, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = s$epsg, remove = FALSE)
fold_rows <- list()
for (m in b$multiples) {
  size <- block_m * m
  sb <- cv_spatial(x = x, column = "clay_share", k = b$k, size = size, hexagon = b$hexagon,
                   selection = b$selection, iteration = b$iteration, seed = b$seed,
                   biomod2 = FALSE, plot = FALSE, report = FALSE, progress = FALSE)
  f <- data.frame(BH_Code = bh$BH_Code, fold = sb$folds_ids, block_m = size, multiple = m)
  write.csv(f, file.path(s$folds_dir, sprintf("folds_x%d.csv", m)), row.names = FALSE)
  for (k in seq_len(b$k)) {
    te <- bh[f$fold == k, ]
    fold_rows[[length(fold_rows) + 1]] <- data.frame(
      multiple = m, block_m = round(size, 1), fold = k, test_boreholes = nrow(te),
      test_cells = sum(te$cells), test_clay_share = round(weighted.mean(te$clay_share, te$cells), 3))
  }
  st_write(sb$blocks, file.path(s$out_dir, "private", sprintf("blocks_x%d.gpkg", m)), delete_dsn = TRUE, quiet = TRUE)
  p <- ggplot() + geom_sf(data = sb$blocks, aes(fill = factor(folds)), colour = "grey40", linewidth = 0.1, alpha = 0.5) +
    geom_sf(data = x, aes(colour = factor(f$fold)), size = 1.2) +
    labs(fill = "Fold", colour = "Fold", title = sprintf("blockCV folds, %.0f m square blocks", size)) + theme_bw(9)
  ggsave(file.path(s$out_dir, "private", sprintf("fig_folds_x%d.png", m)), p, width = 8, height = 6, dpi = 200)
}
folds <- do.call(rbind, fold_rows)
print(folds, row.names = FALSE)
write.csv(folds, file.path(s$out_dir, "fold_summary.csv"), row.names = FALSE)

jsonlite::write_json(list(
  block_m = block_m, slice_m = b$slice_m, boreholes_used = main$boreholes, slices = main$columns,
  range_by_slice_m = as.list(setNames(main$result$range_table$range, main$result$range_table$layers)),
  multiples = b$multiples,
  blockcv_version = as.character(packageVersion("blockCV")),
  blockcv_commit = if (is.null(packageDescription("blockCV")$RemoteSha)) "local test install" else packageDescription("blockCV")$RemoteSha,
  automap_version = as.character(packageVersion("automap"))),
  file.path(s$folds_dir, "block_size.json"), auto_unbox = TRUE, pretty = TRUE, digits = NA)
cat("\nDone: folds written to", s$folds_dir, "\n")
