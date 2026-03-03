# =============================================================================
# HonestDiD R Package - Test Script for Python Comparison
# =============================================================================
# This script runs the main HonestDiD functions and exports results to JSON
# for comparison with the Python implementation.
#
# Run this script first, then run the Jupyter notebook to verify Python matches.
# =============================================================================

# Install packages if needed
# install.packages(c("remotes", "jsonlite", "here", "dplyr", "haven", "fixest"))
# Sys.setenv("R_REMOTES_NO_ERRORS_FROM_WARNINGS" = "true")
# remotes::install_github("asheshrambachan/HonestDiD")

library(HonestDiD)
library(jsonlite)
setwd('..')
cat("=" , rep("=", 59), "\n", sep="")
cat("HonestDiD R Package - Test Script\n")
cat("=" , rep("=", 59), "\n", sep="")

# =============================================================================
# PART 1: Using BCdata_EventStudy (built-in dataset)
# =============================================================================
cat("\n--- Part 1: BCdata_EventStudy Example ---\n")

# Load the built-in data
data(BCdata_EventStudy)
betahat <- BCdata_EventStudy$betahat
sigma <- BCdata_EventStudy$sigma

cat("betahat length:", length(betahat), "\n")
cat("sigma dimensions:", dim(sigma), "\n")
cat("betahat values:\n")
print(round(betahat, 6))

# Export data for Python
bc_data <- list(
  betahat = as.numeric(betahat),
  sigma = as.matrix(sigma)
)
write_json(bc_data, "_data/test_data_bcdata.json", digits = 16, pretty = TRUE)
cat("\nExported BCdata to test_data_bcdata.json\n")

# Parameters
numPrePeriods <- 4
numPostPeriods <- 4
l_vec <- basisVector(1, numPostPeriods)  # First post-period effect

# -----------------------------------------------------------------------------
# Test 1: Original Confidence Set
# -----------------------------------------------------------------------------
cat("\n--- Test 1: Original Confidence Set ---\n")
originalCS <- constructOriginalCS(
  betahat = betahat,
  sigma = sigma,
  numPrePeriods = numPrePeriods,
  numPostPeriods = numPostPeriods,
  l_vec = l_vec,
  alpha = 0.05
)
cat("Original CS:\n")
print(originalCS)

# -----------------------------------------------------------------------------
# Test 2: Sensitivity Results - DeltaSD (Smoothness)
# -----------------------------------------------------------------------------
cat("\n--- Test 2: Sensitivity Results (DeltaSD - Smoothness) ---\n")
Mvec <- c(0, 0.01, 0.02, 0.03)
delta_sd_results <- createSensitivityResults(
  betahat = betahat,
  sigma = sigma,
  numPrePeriods = numPrePeriods,
  numPostPeriods = numPostPeriods,
  l_vec = l_vec,
  Mvec = Mvec,
  alpha = 0.05
)
cat("DeltaSD Results:\n")
print(delta_sd_results)

# -----------------------------------------------------------------------------
# Test 3: Sensitivity Results - DeltaRM (Relative Magnitudes)
# -----------------------------------------------------------------------------
cat("\n--- Test 3: Sensitivity Results (DeltaRM - Relative Magnitudes) ---\n")
Mbarvec <- c(0.5, 1.0, 1.5, 2.0)
delta_rm_results <- createSensitivityResults_relativeMagnitudes(
  betahat = betahat,
  sigma = sigma,
  numPrePeriods = numPrePeriods,
  numPostPeriods = numPostPeriods,
  l_vec = l_vec,
  Mbarvec = Mbarvec,
  alpha = 0.05
)
cat("DeltaRM Results:\n")
print(delta_rm_results)

# -----------------------------------------------------------------------------
# Test 4: FLCI (Fixed-Length Confidence Interval)
# -----------------------------------------------------------------------------
cat("\n--- Test 4: FLCI ---\n")
flci_result <- findOptimalFLCI(
  betahat = betahat,
  sigma = sigma,
  numPrePeriods = numPrePeriods,
  numPostPeriods = numPostPeriods,
  l_vec = l_vec,
  M = 0,
  alpha = 0.05
)
cat("Optimal Half-Length:", flci_result$optimalHalfLength, "\n")
cat("FLCI: [", flci_result$FLCI[1], ", ", flci_result$FLCI[2], "]\n", sep="")

# -----------------------------------------------------------------------------
# Test 5: DeltaSD Upper Bound for M
# -----------------------------------------------------------------------------
cat("\n--- Test 5: DeltaSD Upper Bound for M ---\n")
M_ub <- DeltaSD_upperBound_Mpre(
  betahat = betahat,
  sigma = sigma,
  numPrePeriods = numPrePeriods,
  alpha = 0.05
)
cat("Upper bound M:", M_ub, "\n")

# =============================================================================
# PART 2: Medicaid Expansion Example (from README)
# =============================================================================
cat("\n\n")
cat("=" , rep("=", 59), "\n", sep="")
cat("--- Part 2: Medicaid Expansion Example ---\n")
cat("=" , rep("=", 59), "\n", sep="")

# Try to load the data, skip if packages not available
tryCatch({
  library(haven)
  library(dplyr)
  library(fixest)

  # Load Medicaid data
  df <- read_dta("https://raw.githubusercontent.com/Mixtape-Sessions/Advanced-DID/main/Exercises/Data/ehec_data.dta")

  # Keep years before 2016, drop 2015 cohort
  df_nonstaggered <- df %>%
    filter(year < 2016 & (is.na(yexp2) | yexp2 != 2015))

  # Create treatment dummy
  df_nonstaggered <- df_nonstaggered %>%
    mutate(D = case_when(yexp2 == 2014 ~ 1, TRUE ~ 0))

  # Run TWFE
  twfe_results <- fixest::feols(
    dins ~ i(year, D, ref = 2013) | stfips + year,
    cluster = "stfips",
    data = df_nonstaggered
  )

  betahat_med <- summary(twfe_results)$coefficients
  sigma_med <- summary(twfe_results)$cov.scaled

  cat("\nMedicaid betahat:\n")
  print(round(betahat_med, 6))

  # Export for Python
  medicaid_data <- list(
    betahat = as.numeric(betahat_med),
    sigma = as.matrix(sigma_med)
  )
  write_json(medicaid_data, "_data/test_data_medicaid.json", digits = 16, pretty = TRUE)
  cat("\nExported Medicaid data to test_data_medicaid.json\n")

  # Run sensitivity analysis
  numPrePeriods_med <- 5
  numPostPeriods_med <- 2

  cat("\n--- Medicaid: Original CS ---\n")
  originalCS_med <- constructOriginalCS(
    betahat = betahat_med,
    sigma = sigma_med,
    numPrePeriods = numPrePeriods_med,
    numPostPeriods = numPostPeriods_med
  )
  print(originalCS_med)

  cat("\n--- Medicaid: DeltaRM Results ---\n")
  delta_rm_med <- createSensitivityResults_relativeMagnitudes(
    betahat = betahat_med,
    sigma = sigma_med,
    numPrePeriods = numPrePeriods_med,
    numPostPeriods = numPostPeriods_med,
    Mbarvec = seq(0.5, 2, by = 0.5)
  )
  print(delta_rm_med)

  cat("\n--- Medicaid: DeltaSD Results ---\n")
  delta_sd_med <- createSensitivityResults(
    betahat = betahat_med,
    sigma = sigma_med,
    numPrePeriods = numPrePeriods_med,
    numPostPeriods = numPostPeriods_med,
    Mvec = seq(0, 0.05, by = 0.01)
  )
  print(delta_sd_med)

  cat("\n--- Medicaid: Average Effect (l_vec = c(0.5, 0.5)) ---\n")
  delta_rm_avg <- createSensitivityResults_relativeMagnitudes(
    betahat = betahat_med,
    sigma = sigma_med,
    numPrePeriods = numPrePeriods_med,
    numPostPeriods = numPostPeriods_med,
    Mbarvec = seq(0, 2, by = 0.5),
    l_vec = c(0.5, 0.5)
  )
  print(delta_rm_avg)

}, error = function(e) {
  cat("\nSkipping Medicaid example (packages not available):", conditionMessage(e), "\n")
})

# =============================================================================
# Export all results to JSON for comparison
# =============================================================================
cat("\n\n")
cat("=" , rep("=", 59), "\n", sep="")
cat("Exporting Results for Python Comparison\n")
cat("=" , rep("=", 59), "\n", sep="")

results <- list(
  # BCdata results
  bcdata = list(
    originalCS = list(
      lb = originalCS$lb,
      ub = originalCS$ub
    ),
    deltaSD = as.data.frame(delta_sd_results),
    deltaRM = as.data.frame(delta_rm_results),
    flci = list(
      optimalHalfLength = flci_result$optimalHalfLength,
      FLCI = flci_result$FLCI
    ),
    M_upperBound = M_ub
  )
)

write_json(results, "_data/r_test_results.json", digits = 16, pretty = TRUE)
cat("Exported results to r_test_results.json\n")

cat("\n")
cat("=" , rep("=", 59), "\n", sep="")
cat("All R tests completed!\n")
cat("=" , rep("=", 59), "\n", sep="")
