# Native reference agreement, including RNG consumption across all six calls.
arguments <- commandArgs(trailingOnly = TRUE)
source(file.path(arguments[[1]], "pcm_bridge.R"))
for (package in c("ranger", "jsonlite")) {
  if (!requireNamespace(package, quietly = TRUE)) stop(paste("Missing R package", package))
}
functions <- load_original_pcm(file.path(arguments[[1]], "vendor", "pcm_code", "test_functions.R"))
set.seed(103)
Z <- matrix(rnorm(180), ncol = 3)
X <- matrix(rbinom(60, 1, 0.5), ncol = 1)
Y <- sin(Z[, 1]) + X[, 1] * Z[, 2] + rnorm(60)

set.seed(907)
reference <- vapply(seq_len(6), function(b) {
  functions$pcm_test(Y, X, Z, functions$ranger_reg_method)
}, numeric(1))
reference_rng <- .Random.seed
result <- run_pcm_splits(Y, X, Z, 907L, 6L, functions)
wrapped_rng <- .Random.seed
stopifnot(all(vapply(result$splits, function(x) identical(x$status, "ok"), logical(1))))
observed <- vapply(result$splits, function(x) x$p_value, numeric(1))
stopifnot(identical(reference, observed), identical(reference_rng, wrapped_rng))
statistics <- vapply(result$splits, function(x) x$statistic, numeric(1))
stopifnot(
  identical(result$methods$pcm_single_split$p_value, reference[[1]]),
  identical(result$methods$pcm_derandomized$statistic, mean(statistics)),
  identical(result$methods$pcm_derandomized$p_value, 1 - pnorm(mean(statistics)))
)
for (split in result$splits) {
  stopifnot(
    length(intersect(split$direction_indices, split$main_indices)) == 0L,
    identical(sort(c(split$direction_indices, split$main_indices)), seq_len(length(Y)))
  )
}

# Preserve the upstream failure instead of inventing a non-rejection.
degenerate <- run_pcm_splits(rep(0, 60), X, Z, 907L, 1L, functions)
stopifnot(identical(degenerate$methods$pcm_single_split$status, "failed"))
cat(jsonlite::toJSON(list(
  reference_p_values = reference,
  maximum_absolute_difference = max(abs(reference - observed)),
  rng_states_identical = identical(reference_rng, wrapped_rng),
  runtime = pcm_runtime()
), auto_unbox = TRUE, digits = NA), "\n")
