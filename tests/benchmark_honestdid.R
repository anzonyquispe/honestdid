# ============================================================
# Benchmark: HonestDiD::createSensitivityResults (Delta_SD)
# Mvec = seq(0, 0.1, by = 0.01) -> 11 valores
# Datos: Medicaid expansion (README oficial HonestDiD)
# ============================================================

# --- 1. Instalar paquetes si no están -----------------------
if (!requireNamespace("remotes",  quietly = TRUE)) install.packages("remotes")
if (!requireNamespace("HonestDiD",quietly = TRUE))
  remotes::install_github("asheshrambachan/HonestDiD")
if (!requireNamespace("fixest",   quietly = TRUE)) install.packages("fixest")
if (!requireNamespace("haven",    quietly = TRUE)) install.packages("haven")
if (!requireNamespace("dplyr",    quietly = TRUE)) install.packages("dplyr")

library(HonestDiD)
library(fixest)
library(haven)
library(dplyr)

# --- 2. Cargar datos y estimar TWFE -------------------------
cat("Cargando datos...\n")
df <- read_dta("https://raw.githubusercontent.com/Mixtape-Sessions/Advanced-DID/main/Exercises/Data/ehec_data.dta")

df_ns <- df %>%
  filter(year < 2016 & (is.na(yexp2) | yexp2 != 2015)) %>%
  mutate(D = if_else(yexp2 == 2014, 1L, 0L, missing = 0L))

twfe <- fixest::feols(dins ~ i(year, D, ref = 2013) | stfips + year,
                      cluster = "stfips", data = df_ns)

betahat <- summary(twfe)$coefficients
sigma   <- summary(twfe)$cov.scaled

cat("Estimación TWFE completada.\n")
cat(sprintf("numPrePeriods = 5 | numPostPeriods = 2 | Mvec: seq(0, 0.1, by=0.01) -> %d valores\n\n",
            length(seq(0, 0.1, by = 0.01))))

# --- 3. Benchmark -------------------------------------------
cat("Corriendo createSensitivityResults (Delta_SD)...\n")

t_start <- proc.time()

delta_sd_results <- HonestDiD::createSensitivityResults(
  betahat        = betahat,
  sigma          = sigma,
  numPrePeriods  = 5,
  numPostPeriods = 2,
  Mvec           = seq(from = 0, to = 0.1, by = 0.01)
)

t_end <- proc.time()
elapsed <- (t_end - t_start)[["elapsed"]]

# --- 4. Resultados ------------------------------------------
cat("\n--- Resultados Delta_SD ---\n")
print(delta_sd_results)

cat(sprintf("\n⏱  Tiempo de ejecución (R): %.4f segundos\n", elapsed))
