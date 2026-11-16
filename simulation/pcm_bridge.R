#!/usr/bin/env Rscript
# Run the pinned PCM implementation with Ranger.

load_original_pcm <- function(path) {
  functions <- new.env(parent = globalenv())
  wanted <- c("pcm_test", "ranger_reg_method")
  for (expression in parse(path)) {
    if (is.call(expression) && identical(expression[[1]], as.name("<-")) &&
        is.symbol(expression[[2]]) && as.character(expression[[2]]) %in% wanted) {
      eval(expression, envir = functions)
    }
  }
  if (!all(vapply(wanted, exists, logical(1), envir = functions, inherits = FALSE))) {
    stop("The pinned source does not define the expected PCM functions.")
  }
  functions
}

pcm_with_details <- function(original) {
  # Expose the statistic and split diagnostics alongside the upstream p-value.
  detailed <- original
  expressions <- as.list(body(detailed))
  expected <- quote(return(1 - pnorm(test_statistic)))
  if (!identical(expressions[[length(expressions)]], expected)) {
    stop("Unexpected upstream PCM return expression.")
  }
  expressions[[length(expressions)]] <- quote(return(list(
    p_value = 1 - pnorm(test_statistic),
    statistic = test_statistic,
    direction_indices = direction_indices,
    main_indices = main_indices,
    n_direction = length(direction_indices),
    n_testing = length(main_indices),
    score_mean = mean(R),
    score_sd = stats::sd(R),
    variance_adjustment = chat,
    direction_sign = sgn
  )))
  body(detailed) <- as.call(expressions)
  detailed
}

pcm_runtime <- function() {
  list(
    R = R.version.string,
    ranger = as.character(utils::packageVersion("ranger")),
    jsonlite = as.character(utils::packageVersion("jsonlite")),
    RNG_kind = RNGkind()
  )
}

run_pcm_splits <- function(Y, X, Z, seed, splits, functions) {
  detailed <- pcm_with_details(functions$pcm_test)
  set.seed(seed)
  results <- lapply(seq_len(splits), function(b) {
    started <- proc.time()[["elapsed"]]
    value <- tryCatch({
      result <- detailed(Y, X, Z, functions$ranger_reg_method)
      if (!is.finite(result$statistic) || !is.finite(result$p_value)) {
        stop("Original PCM returned a non-finite statistic or p-value.")
      }
      result$status <- "ok"
      result
    }, error = function(e) list(status = "failed", error = conditionMessage(e)))
    value$elapsed_seconds <- proc.time()[["elapsed"]] - started
    value
  })
  single <- results[[1]]
  single$direction_indices <- NULL
  single$main_indices <- NULL
  single$elapsed_seconds <- NULL
  methods <- list(pcm_single_split = single)
  if (splits == 6L) {
    successful <- vapply(results, function(x) identical(x$status, "ok"), logical(1))
    if (all(successful)) {
      statistic <- mean(vapply(results, function(x) x$statistic, numeric(1)))
      methods$pcm_derandomized <- list(
        status = "ok", statistic = statistic, p_value = 1 - pnorm(statistic),
        n_testing = length(Y) / 2
      )
    } else {
      methods$pcm_derandomized <- list(
        status = "failed",
        error = paste("Original PCM failed in split(s)", paste(which(!successful), collapse = ","))
      )
    }
  }
  list(methods = methods, splits = results, runtime = pcm_runtime())
}

main <- function(arguments = commandArgs(trailingOnly = TRUE)) {
  for (package in c("ranger", "jsonlite")) {
    if (!requireNamespace(package, quietly = TRUE)) {
      stop(paste("Install the R package", package, "before running the original PCM comparator."))
    }
  }
  if (identical(arguments, "--metadata")) {
    cat(jsonlite::toJSON(pcm_runtime(), auto_unbox = TRUE, digits = NA), "\n")
    return(invisible(NULL))
  }
  if (length(arguments) != 5L) {
    stop("Usage: pcm_bridge.R source.R data.csv result.json seed splits")
  }
  seed <- as.integer(arguments[[4]])
  splits <- as.integer(arguments[[5]])
  if (is.na(seed) || seed < 0L || is.na(splits) || !splits %in% c(1L, 6L)) {
    stop("PCM needs a non-negative R seed and either one or six splits.")
  }
  data <- utils::read.csv(arguments[[2]], check.names = FALSE)
  if (ncol(data) < 3L || nrow(data) %% 2L != 0L || nrow(data) < 8L ||
      any(!is.finite(as.matrix(data)))) {
    stop("PCM input must be finite, with Y, D, covariates, and an even sample size >= 8.")
  }
  functions <- load_original_pcm(arguments[[1]])
  result <- run_pcm_splits(
    data[[1]], as.matrix(data[2]), as.matrix(data[-c(1, 2)]), seed, splits, functions
  )
  jsonlite::write_json(result, arguments[[3]], auto_unbox = TRUE, digits = NA, na = "null")
}

if (sys.nframe() == 0L) main()
