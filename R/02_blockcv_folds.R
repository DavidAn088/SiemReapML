# Outer spatial folds with the ORIGINAL blockCV package (brief section 7).
#
# Input : data/processed/cells_development.csv   development cells (external boreholes removed)
#         data/processed/prediction_columns.csv  25 m grid columns inside the Downtown polygon
#         outputs/04_spatial_autocorrelation/aggregate/spatial_autocorrelation_summary.json (optional)
# Output: data/folds/outer_folds_<b>m.csv        BH_Code, fold, role, block_size_m   (read by Python)
#         data/folds/inner_fallback_<b>m.csv     BH_Code, outer_fold, inner_fold     (used only if the
#                                                re-used outer groups fail the inner adequacy check)
#         outputs/04_blockcv_folds/aggregate/    fold summaries, test-to-train distances, cv_distance,
#                                                block-size rule record
#         outputs/04_blockcv_folds/restricted/   fold maps and block polygons (show borehole locations)
#
# Design (design report section 4): square blocks (hexagon = FALSE) of 100, 200 and 300 m,
# k = 5, random block-to-fold assignment balanced on the cell-level Clay/Sand class, fixed seed.
# Folds are built on Downtown development cells; 5 km ring boreholes are training-only
# (fold 0): they lie >= 1.6 km from any Downtown borehole, so no block can place them in a test fold.

source(file.path(if (file.exists("R/_common.R")) "R" else ".", "_common.R"))
root <- find_root()
setwd(root)
suppressPackageStartupMessages({ library(sf); library(blockCV); library(ggplot2) })
st <- read_settings(root)
bc <- st$blockcv
agg <- out_dir(root, st, "04_blockcv_folds", "aggregate")
res <- out_dir(root, st, "04_blockcv_folds", "restricted")
fdir <- file.path(root, st$files$folds_dir)
dir.create(fdir, recursive = TRUE, showWarnings = FALSE)

cells <- read_cells(root, st)
dt <- cells[cells$is_downtown == 1, ]
ring <- cells[cells$is_downtown == 0, ]
x <- st_as_sf(dt, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = st$crs_epsg, remove = FALSE)
bh_dt <- unique(dt[, c("BH_Code", "UTM_Easting_m", "UTM_Northing_m")])
bh_ring <- unique(ring[, c("BH_Code", "UTM_Easting_m", "UTM_Northing_m")])
pred <- read.csv(file.path(root, st$files$pred_points))
pred_sf <- st_as_sf(pred, coords = c("X", "Y"), crs = st$crs_epsg)

nn_dist <- function(from, to) {  # horizontal distance from each `from` row to the nearest `to` row
  d <- st_distance(st_as_sf(from, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = st$crs_epsg),
                   st_as_sf(to, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = st$crs_epsg))
  apply(matrix(as.numeric(d), nrow = nrow(from)), 1, min)
}

split_boreholes <- function(code, fold) {
  n <- tapply(fold, code, function(f) length(unique(f)))
  names(n)[n != 1]
}

# cv_spatial resolves a point that lies exactly on a block edge record by record, which can
# split one borehole's cells between folds (many coordinates are whole metres, as are the
# block edges). If that happens the blocks are shifted by a tiny offset (0.01 % of the block
# size per attempt, i.e. 1 cm for 100 m blocks) and the call is repeated; the offset used is
# recorded with the folds.
run_cv_spatial <- function(xx, k, size, seed) {
  for (attempt in 0:20) {
    off <- c(1, 1) * attempt * 1e-4
    sb <- cv_spatial(x = xx, column = "cls", k = k, size = size, hexagon = isTRUE(bc$hexagon),
                     selection = bc$selection, iteration = bc$iteration, seed = seed, offset = off,
                     biomod2 = FALSE, plot = FALSE, report = FALSE, progress = FALSE)
    if (length(split_boreholes(xx$BH_Code, sb$folds_ids)) == 0) {
      sb$offset_used <- off
      return(sb)
    }
  }
  stop("boreholes remain split between folds after 20 offset attempts")
}

one_fold_per_borehole <- function(code, fold) {
  bad <- split_boreholes(code, fold)
  if (length(bad)) stop("blockCV assigned the cells of ", length(bad), " borehole(s) to more than one fold.")
  tapply(fold, code, function(f) f[1])
}

# block sizes: fixed in the configuration, or (auto_block_size) the blockCV autocorrelation
# range written by R/01_mc_block_size.R
decision <- NULL
if (isTRUE(bc$auto_block_size)) {
  dec_file <- file.path(fdir, "block_size_decision.json")
  if (!file.exists(dec_file)) stop("Run the autocorrelation script first (R/01_spatial_autocorrelation.R, or ",
                                    "R/01_mc_block_size.R for the 5-class model); it writes ", dec_file, ".")
  decision <- jsonlite::fromJSON(dec_file)
  block_sizes <- unique(c(decision$block_size_m, round(decision$block_size_m * unlist(bc$extra_block_multiples)),
                          unlist(bc$extra_block_sizes_m)))
  cat("Block size from the autocorrelation range:", decision$block_size_m, "m\n")
} else {
  block_sizes <- bc$block_sizes_m
}
clay_of <- function(d) if ("y_bin" %in% names(d)) d$y_bin else d$y

fold_summary <- list(); dist_summary <- list(); W_rows <- list(); offsets <- list()
for (b in block_sizes) {
  cat("Block size", b, "m\n")
  sb <- run_cv_spatial(x, bc$k, b, bc$seed)
  bh_fold <- one_fold_per_borehole(x$BH_Code, sb$folds_ids)
  cat("  block offset used (share of block size):", sb$offset_used, "\n")
  offsets[[as.character(b)]] <- sb$offset_used
  out <- data.frame(BH_Code = names(bh_fold), fold = as.integer(bh_fold), role = "cv", block_size_m = b)
  if (nrow(bh_ring)) out <- rbind(out, data.frame(BH_Code = bh_ring$BH_Code, fold = 0L, role = "train_only", block_size_m = b))
  write.csv(out, file.path(fdir, sprintf("outer_folds_%dm.csv", b)), row.names = FALSE)

  dt$fold <- bh_fold[dt$BH_Code]
  bh_dt$fold <- bh_fold[bh_dt$BH_Code]
  for (k in seq_len(bc$k)) {
    te <- bh_dt[bh_dt$fold == k, ]; tr <- bh_dt[bh_dt$fold != k, ]
    dA <- nn_dist(te, tr)
    dB <- nn_dist(te, rbind(tr[, 1:3], bh_ring))
    fc <- dt[dt$fold == k, ]
    fold_summary[[length(fold_summary) + 1]] <- data.frame(
      block_size_m = b, fold = k, test_boreholes = nrow(te), test_cells = nrow(fc), test_clay_share = mean(clay_of(fc)),
      test_classes = length(unique(fc$cls)),
      train_boreholes_A = nrow(tr), train_boreholes_B = nrow(tr) + nrow(bh_ring))
    for (sc in c("A", "B")) {
      d <- if (sc == "A") dA else dB
      dist_summary[[length(dist_summary) + 1]] <- data.frame(
        block_size_m = b, fold = k, scenario = sc, min_m = min(d), q25_m = unname(quantile(d, 0.25)),
        median_m = median(d), q75_m = unname(quantile(d, 0.75)), max_m = max(d),
        share_below_50m = mean(d < 50), share_below_100m = mean(d < 100), share_below_block = mean(d < b))
    }
  }

  # distance diagnostic against the prediction domain (blockCV cv_distance)
  cd <- tryCatch(cv_distance(cv = sb, x = x, pred_points = pred_sf, plot = FALSE, seed = bc$seed),
                 error = function(e) { message("cv_distance failed: ", conditionMessage(e)); NULL })
  if (!is.null(cd)) {
    W_rows[[length(W_rows) + 1]] <- data.frame(design = sprintf("blocks %d m", b), block_size_m = b,
                                               t(as.data.frame(cd$W)))
    write.csv(cbind(block_size_m = b, cd$distances), file.path(agg, sprintf("cv_distance_folds_%dm.csv", b)),
              row.names = FALSE)
    ggsave(file.path(agg, sprintf("fig_cv_distance_%dm.png", b)), cd$plot, width = 7, height = 4.5, dpi = 200)
  }

  # fold map and block polygons (restricted: they show borehole locations)
  blocks <- sb$blocks
  pmap <- ggplot() + geom_sf(data = blocks, aes(fill = factor(folds)), colour = "grey40", linewidth = 0.1, alpha = 0.6) +
    geom_sf(data = st_as_sf(bh_dt, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = st$crs_epsg),
            aes(colour = factor(fold)), size = 1.2) +
    labs(fill = "Fold (block)", colour = "Fold (borehole)", title = sprintf("blockCV folds, %d m square blocks", b)) +
    theme_bw(9)
  ggsave(file.path(res, sprintf("fig_fold_map_%dm.png", b)), pmap, width = 8, height = 6, dpi = 200)
  st_write(blocks, file.path(res, sprintf("blocks_%dm.gpkg", b)), delete_dsn = TRUE, quiet = TRUE)

  # R-generated fallback inner folds: blockCV on each outer-training set, k - 1 folds
  inner <- list()
  for (k in seq_len(bc$k)) {
    xk <- x[dt$fold != k, ]
    ib <- tryCatch(run_cv_spatial(xk, bc$k - 1, b, bc$seed + k), error = function(e) NULL)
    if (is.null(ib)) { message("fallback inner folds failed for outer fold ", k); next }
    f <- one_fold_per_borehole(xk$BH_Code, ib$folds_ids)
    inner[[length(inner) + 1]] <- data.frame(BH_Code = names(f), outer_fold = k, inner_fold = as.integer(f))
  }
  write.csv(do.call(rbind, inner), file.path(fdir, sprintf("inner_fallback_%dm.csv", b)), row.names = FALSE)
}

# optional: kNNDM (Linnenbrink et al. 2024) as a prediction-matched diagnostic (decision D15)
if (isTRUE(st$spatial$knndm_diagnostic)) {
  kn <- cv_knndm(x = x, column = "cls", pred_points = pred_sf, k = bc$k, seed = bc$seed, plot = FALSE, report = FALSE)
  cdk <- cv_distance(cv = kn, x = x, pred_points = pred_sf, plot = FALSE, seed = bc$seed)
  W_rows[[length(W_rows) + 1]] <- data.frame(design = "kNNDM", block_size_m = NA, t(as.data.frame(cdk$W)))
  f <- one_fold_per_borehole(x$BH_Code, kn$folds_ids)
  write.csv(data.frame(BH_Code = names(f), fold = as.integer(f), role = "cv", block_size_m = NA),
            file.path(fdir, "outer_folds_knndm.csv"), row.names = FALSE)
}

fs <- do.call(rbind, fold_summary); ds <- do.call(rbind, dist_summary); W <- do.call(rbind, W_rows)
write.csv(fs, file.path(agg, "fold_summary.csv"), row.names = FALSE)
write.csv(ds, file.path(agg, "test_to_train_distance_by_fold.csv"), row.names = FALSE)
if (!is.null(W)) write.csv(W, file.path(agg, "wasserstein_vs_prediction_domain.csv"), row.names = FALSE)

if (!is.null(decision)) {  # auto_block_size: the decision was made by the autocorrelation script
  rule <- list(auto_block_size = TRUE, block_size_m = decision$block_size_m, method = decision$method,
               block_sizes_run = block_sizes, block_offsets = offsets)
  if (!is.null(W)) rule$W_cv <- setNames(as.list(W$CV), W$block_size_m)
  jsonlite::write_json(rule, file.path(agg, "block_size_rule.json"), auto_unbox = TRUE, pretty = TRUE)
  write_session_info(agg, "blockcv_folds")
  print(fs); print(W); print(rule)
  cat("Fold files written to", fdir, "\n")
  quit(save = "no")
}

# block-size rule (design report section 4.3), recorded for review before any model is fitted
sac_file <- file.path(out_dir(root, st, "04_spatial_autocorrelation", "aggregate"), "spatial_autocorrelation_summary.json")
r_tilde <- if (file.exists(sac_file)) jsonlite::fromJSON(sac_file)$r_tilde_m else NA
rule <- list(r_tilde_m = r_tilde, planned_primary_m = bc$primary_block_m, block_offsets = offsets)
if (!is.null(W) && !is.na(r_tilde)) {
  Wb <- W[!is.na(W$block_size_m), ]
  cand <- Wb[Wb$block_size_m >= r_tilde, ]
  rule$W_cv <- setNames(as.list(Wb$CV), Wb$block_size_m)
  rule$rule_choice_m <- if (100 %in% cand$block_size_m && Wb$CV[Wb$block_size_m == 100] <= 1.1 * min(Wb$CV)) 100 else
    if (nrow(cand)) cand$block_size_m[which.min(cand$CV)] else max(Wb$block_size_m)
  rule$note <- if (nrow(cand) == 0) "no candidate reaches r_tilde; 300 m chosen and the limitation must be reported" else ""
} else {
  rule$rule_choice_m <- NA
  rule$note <- "r_tilde or cv_distance unavailable; run 01_spatial_autocorrelation.R first"
}
jsonlite::write_json(rule, file.path(agg, "block_size_rule.json"), auto_unbox = TRUE, pretty = TRUE)
write_session_info(agg, "blockcv_folds")
print(fs); print(W); print(rule)
cat("Fold files written to", fdir, "\n")
