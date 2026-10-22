# Spatial autocorrelation of the Clay/Sand indicator (brief section 7; design report section 3).
#
# Input : data/processed/cells_development.csv  (development cells only; external boreholes
#         were removed by notebooks/01_preprocessing.ipynb before this script can see them)
# Output: outputs/04_spatial_autocorrelation/aggregate/  (tables and figures, no coordinates)
#
# Analyses, Downtown development boreholes only:
#  1. horizontal indicator variograms per 1.5 m depth slice; each borehole contributes one
#     value per slice (its clay fraction in the slice), so no zero-distance within-borehole pairs;
#  2. a pooled within-slice variogram of the standardised indicator
#     (I - p_s) / sqrt(p_s (1 - p_s)), which removes the depth trend and puts every slice on
#     the same sill;
#  3. fitted models (spherical, exponential, Gaussian; best by weighted SSE) with nugget,
#     partial sill, range PARAMETER and PRACTICAL range (distance at 95 % of the total sill);
#  4. blockCV::cv_spatial_autocor per slice (automap fit; reports the range parameter);
#  5. Moran's I per slice on a k-nearest-borehole graph with a permutation test;
#  6. vertical indicator variogram from pairs within the same borehole;
#  7. uncertainty of the pooled practical range by site subsampling (80 % of 100 m sites).

source(file.path(if (file.exists("R/_common.R")) "R" else ".", "_common.R"))
root <- find_root()
setwd(root)
suppressPackageStartupMessages({ library(sf); library(gstat); library(blockCV); library(ggplot2) })
st <- read_settings(root)
sp <- st$spatial
agg <- out_dir(root, st, "04_spatial_autocorrelation", "aggregate")
set.seed(st$seed)

cells <- read_cells(root, st)
dt <- cells[cells$is_downtown == 1, ]
dt$slice <- floor(dt$depth_top_m / sp$slice_m + 1e-9)
bh_slice <- aggregate(y ~ BH_Code + slice + UTM_Easting_m + UTM_Northing_m, data = dt, FUN = mean)
names(bh_slice)[names(bh_slice) == "y"] <- "clay_frac"
slices <- sort(unique(bh_slice$slice))
slice_label <- function(s) sprintf("%.1f-%.1f m", s * sp$slice_m, (s + 1) * sp$slice_m)
bounds <- unlist(sp$variogram_boundaries_m)

practical_range <- function(model, share = 0.95) {
  if (is.null(model)) return(NA_real_)
  total <- sum(model$psill)
  h <- c(seq(1, 5000, by = 1), seq(5010, 50000, by = 10))
  g <- variogramLine(model, dist_vector = h)$gamma
  i <- which(g >= share * total)[1]
  if (is.na(i)) NA_real_ else h[i]
}

fit_best <- function(v) {
  if (is.null(v) || nrow(v) < 3) return(NULL)
  init <- vgm(psill = max(v$gamma) * 0.7, model = c("Sph", "Exp", "Gau"), range = median(v$dist),
              nugget = min(v$gamma) * 0.5)
  tryCatch(suppressWarnings(fit.variogram(v, init)), error = function(e) NULL)
}

model_row <- function(m, label, kind, n_bh, v) {
  if (is.null(m)) {
    return(data.frame(slice = label, variable = kind, n_boreholes = n_bh, model = NA, nugget = NA, psill = NA,
                      range_parameter_m = NA, practical_range_m = NA, nugget_to_sill = NA,
                      pairs_first_lag = if (is.null(v)) NA else v$np[1]))
  }
  s <- m[m$model != "Nug", ]
  nug <- if (any(m$model == "Nug")) m$psill[m$model == "Nug"] else 0
  data.frame(slice = label, variable = kind, n_boreholes = n_bh, model = as.character(s$model[1]),
             nugget = nug, psill = s$psill[1], range_parameter_m = s$range[1],
             practical_range_m = practical_range(m), nugget_to_sill = nug / sum(m$psill),
             pairs_first_lag = v$np[1])
}

# ---- 1-3: per-slice and pooled standardised variograms -------------------------------
mk_vgm <- function(tab) {  # gstat variogram object from a pooled table
  v <- data.frame(np = tab$np, dist = tab$dist, gamma = tab$gamma, dir.hor = 0, dir.ver = 0, id = factor("var1"))
  class(v) <- c("gstatVariogram", "data.frame")
  v
}
pool_slices <- function(d_all, keep = NULL) {
  # pairs are formed only within a slice; semivariances are pooled with pair-count weights
  long <- NULL
  for (s in slices) {
    d <- d_all[d_all$slice == s, ]
    if (!is.null(keep)) d <- d[d$BH_Code %in% keep, ]
    if (nrow(d) < 10 || length(unique(d$clay_frac)) < 2) next
    p <- mean(d$clay_frac)
    d$z <- (d$clay_frac - p) / sqrt(p * (1 - p))
    vz <- variogram(z ~ 1, st_as_sf(d, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = st$crs_epsg),
                    boundaries = bounds)
    long <- rbind(long, data.frame(bin = findInterval(vz$dist, bounds), np = vz$np, dist = vz$dist, gamma = vz$gamma))
  }
  if (is.null(long)) return(NULL)
  t <- do.call(rbind, lapply(split(long, long$bin), function(b)
    data.frame(np = sum(b$np), dist = sum(b$dist * b$np) / sum(b$np), gamma = sum(b$gamma * b$np) / sum(b$np))))
  t[order(t$dist), ]
}
rows <- list(); emp <- list()
for (s in slices) {
  d <- bh_slice[bh_slice$slice == s, ]
  if (nrow(d) < 10 || length(unique(d$clay_frac)) < 2) next
  pts <- st_as_sf(d, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = st$crs_epsg)
  v <- variogram(clay_frac ~ 1, pts, boundaries = bounds)
  rows[[length(rows) + 1]] <- model_row(fit_best(v), slice_label(s), "indicator", nrow(d), v)
  emp[[length(emp) + 1]] <- data.frame(slice = slice_label(s), dist = v$dist, gamma = v$gamma, np = v$np)
}
pool <- pool_slices(bh_slice)
fp <- list(v = mk_vgm(pool), m = fit_best(mk_vgm(pool)))
rows[[length(rows) + 1]] <- model_row(fp$m, "pooled 0-15 m", "standardised indicator",
                                      length(unique(bh_slice$BH_Code)), fp$v)
params <- do.call(rbind, rows)
# A fitted range shorter than the first lag class cannot be resolved by the data: the
# variogram is flat (pure nugget) at every lag the boreholes sample.
first_lag <- bounds[bounds > 0][1]
params$resolved <- !is.na(params$practical_range_m) & params$practical_range_m >= first_lag
write.csv(params, file.path(agg, "variogram_parameters.csv"), row.names = FALSE)
write.csv(do.call(rbind, emp), file.path(agg, "empirical_variograms_by_slice.csv"), row.names = FALSE)
write.csv(pool, file.path(agg, "pooled_standardised_variogram.csv"), row.names = FALSE)

# ---- 4: blockCV cv_spatial_autocor per slice --------------------------------------------
bcv <- list()
for (s in slices) {
  d <- bh_slice[bh_slice$slice == s, ]
  if (nrow(d) < 10) next
  pts <- st_as_sf(d, coords = c("UTM_Easting_m", "UTM_Northing_m"), crs = st$crs_epsg)
  r <- tryCatch({
    a <- cv_spatial_autocor(x = pts, column = "clay_frac", plot = FALSE, progress = FALSE)
    vm <- a$variograms[[1]]$var_model
    data.frame(slice = slice_label(s), model = as.character(vm$model[2]), range_parameter_m = a$range,
               practical_range_m = practical_range(vm), status = "ok")
  }, error = function(e) data.frame(slice = slice_label(s), model = NA, range_parameter_m = NA,
                                     practical_range_m = NA, status = conditionMessage(e)))
  bcv[[length(bcv) + 1]] <- r
}
bcv <- do.call(rbind, bcv)
write.csv(bcv, file.path(agg, "blockcv_spatial_autocor_by_slice.csv"), row.names = FALSE)

# ---- 5: Moran's I per slice (k-nearest-borehole weights, permutation test) -------------
moran_perm <- function(z, xy, k, nperm) {
  n <- length(z); d <- as.matrix(dist(xy)); diag(d) <- Inf
  W <- matrix(0, n, n)
  for (i in seq_len(n)) W[i, order(d[i, ])[seq_len(k)]] <- 1 / k
  stat <- function(v) { v <- v - mean(v); (n / sum(W)) * sum(W * outer(v, v)) / sum(v^2) }
  I <- stat(z); perm <- replicate(nperm, stat(sample(z)))
  c(I = I, expected = -1 / (n - 1), p_value = (sum(perm >= I) + 1) / (nperm + 1))
}
mor <- do.call(rbind, lapply(slices, function(s) {
  d <- bh_slice[bh_slice$slice == s, ]
  if (nrow(d) < sp$moran_k + 2 || length(unique(d$clay_frac)) < 2) return(NULL)
  r <- moran_perm(d$clay_frac, as.matrix(d[, c("UTM_Easting_m", "UTM_Northing_m")]), sp$moran_k, sp$moran_permutations)
  data.frame(slice = slice_label(s), n_boreholes = nrow(d), morans_I = r[["I"]], expected = r[["expected"]],
             p_value = r[["p_value"]])
}))
write.csv(mor, file.path(agg, "morans_I_by_slice.csv"), row.names = FALSE)

# ---- 6: vertical indicator variogram within boreholes ----------------------------------
vert <- do.call(rbind, lapply(split(dt, dt$BH_Code), function(b) {
  if (nrow(b) < 2) return(NULL)
  pr <- t(combn(nrow(b), 2))
  data.frame(lag = abs(b$depth_top_m[pr[, 1]] - b$depth_top_m[pr[, 2]]),
             sq = (b$y[pr[, 1]] - b$y[pr[, 2]])^2 / 2)
}))
vert <- vert[vert$lag <= sp$vertical_max_lag_m, ]
vv <- aggregate(sq ~ lag, data = vert, FUN = mean); vv$np <- aggregate(sq ~ lag, data = vert, FUN = length)$sq
names(vv)[2] <- "gamma"
write.csv(vv, file.path(agg, "vertical_variogram.csv"), row.names = FALSE)

# ---- 7: site subsampling for the pooled practical range ----------------------------------
bh_xy <- unique(dt[, c("BH_Code", "UTM_Easting_m", "UTM_Northing_m")])
site <- cutree(hclust(dist(bh_xy[, 2:3]), method = "single"), h = st$site_linkage_m)
names(site) <- bh_xy$BH_Code
reps <- if (identical(st$mode, "smoke")) 10 else sp$subsample_reps
sub_ranges <- replicate(reps, {
  keep <- names(site)[site %in% sample(unique(site), floor(sp$subsample_share * length(unique(site))))]
  t2 <- pool_slices(bh_slice, keep)
  if (is.null(t2)) NA_real_ else practical_range(fit_best(mk_vgm(t2)))
})
ok <- params$variable == "indicator" & !is.na(params$practical_range_m)
# unresolved slices enter the median at the first lag distance ("at most first_lag")
pr_slice <- ifelse(params$resolved[ok], params$practical_range_m[ok], first_lag)
summary <- list(
  n_downtown_development_boreholes = length(unique(dt$BH_Code)),
  slices_fitted = sum(ok),
  slices_resolved = sum(params$resolved[ok]),
  first_lag_m = first_lag,
  median_practical_range_by_slice_m = if (any(ok)) median(pr_slice) else NA,
  pooled_standardised = as.list(params[params$slice == "pooled 0-15 m", c("model", "nugget", "psill",
                                  "range_parameter_m", "practical_range_m", "nugget_to_sill")]),
  pooled_practical_range_subsample_m = as.list(stats::quantile(sub_ranges, c(0.05, 0.5, 0.95), na.rm = TRUE)),
  subsample_reps = reps, subsample_failed_fits = sum(is.na(sub_ranges)),
  blockcv_autocor_status = unique(bcv$status),
  note = paste("Response (indicator) variograms are exploratory proxies for residual autocorrelation;",
               "the block-size rule uses the median per-slice practical range (r_tilde).")
)
summary$r_tilde_m <- summary$median_practical_range_by_slice_m
ok_bcv <- bcv[!is.na(bcv$range_parameter_m), ]
summary$blockcv_suggested_block_size_m <- if (nrow(ok_bcv)) round(stats::median(ok_bcv$range_parameter_m)) else NA
jsonlite::write_json(summary, file.path(agg, "spatial_autocorrelation_summary.json"), auto_unbox = TRUE,
                     pretty = TRUE, digits = 6)

# ---- block size (auto_block_size): blockCV's own suggestion ----------------------------
# cv_spatial_autocor returns, for each slice, the range it recommends as block size; with
# several layers blockCV uses "the median of all calculated ranges" (Valavi et al. 2019).
if (isTRUE(st$blockcv$auto_block_size)) {
  if (nrow(ok_bcv) == 0) stop("blockCV cv_spatial_autocor failed for every slice; see blockcv_spatial_autocor_by_slice.csv")
  fdir <- file.path(root, st$files$folds_dir)
  dir.create(fdir, recursive = TRUE, showWarnings = FALSE)
  decision <- list(
    block_size_m = round(stats::median(ok_bcv$range_parameter_m)),
    method = "median of blockCV::cv_spatial_autocor ranges of the Clay indicator over 1.5 m depth slices",
    slices_used = nrow(ok_bcv), slices_failed = nrow(bcv) - nrow(ok_bcv),
    ranges_m = setNames(as.list(round(ok_bcv$range_parameter_m, 1)), ok_bcv$slice),
    blockcv_version = as.character(utils::packageVersion("blockCV")),
    automap_version = as.character(utils::packageVersion("automap")))
  jsonlite::write_json(decision, file.path(fdir, "block_size_decision.json"), auto_unbox = TRUE, pretty = TRUE)
  cat("\nblockCV block size (median of", nrow(ok_bcv), "slice ranges):", decision$block_size_m, "m\n")
}

# ---- figures --------------------------------------------------------------------------
e <- do.call(rbind, emp)
p1 <- ggplot(e, aes(dist, gamma)) + geom_point(aes(size = np), colour = "#2a78d6", alpha = 0.8) +
  facet_wrap(~slice) + scale_size_area(max_size = 3, name = "pairs") +
  labs(x = "Lag distance (m)", y = "Semivariance of clay fraction",
       title = "Indicator variograms by 1.5 m depth slice (Downtown development boreholes)") + theme_bw(9)
ggsave(file.path(agg, "fig_variograms_by_slice.png"), p1, width = 9, height = 6.5, dpi = 200)
lp <- if (!is.null(fp$m)) variogramLine(fp$m, maxdist = max(pool$dist)) else NULL
p2 <- ggplot(pool, aes(dist, gamma)) + geom_point(aes(size = np), colour = "#2a78d6") +
  { if (!is.null(lp)) geom_line(data = lp, aes(dist, gamma), colour = "#eb6834", linewidth = 0.8) } +
  geom_hline(yintercept = 1, linetype = 3) + scale_size_area(max_size = 4, name = "pairs") +
  labs(x = "Lag distance (m)", y = "Standardised semivariance",
       title = "Pooled within-slice standardised indicator variogram") + theme_bw(10)
ggsave(file.path(agg, "fig_pooled_standardised_variogram.png"), p2, width = 7, height = 4.2, dpi = 200)
p3 <- ggplot(vv, aes(lag, gamma)) + geom_point(aes(size = np), colour = "#2a78d6") + geom_line(colour = "#2a78d6") +
  scale_size_area(max_size = 3, name = "pairs") +
  labs(x = "Vertical lag (m)", y = "Semivariance", title = "Vertical indicator variogram (within boreholes)") +
  theme_bw(10)
ggsave(file.path(agg, "fig_vertical_variogram.png"), p3, width = 7, height = 4.2, dpi = 200)
write_session_info(agg, "spatial_autocorrelation")
cat("Spatial autocorrelation outputs written to", agg, "\n")
print(params)
