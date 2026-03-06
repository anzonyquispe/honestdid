# ============================================================
# Benchmark: honestdid (Python) - createSensitivityResults
# Delta_SD | Mvec = [0, 0.01, ..., 0.10] -> 11 valores
# Datos: Medicaid expansion (mismo que README de HonestDiD R)
# ============================================================

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"   # fix OpenMP conflict en Windows

import time
import numpy as np
import pandas as pd
import torch
import honestdid as hd

# --- Info del dispositivo --------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 55)
print(f"  honestdid (Python) - Benchmark Delta_SD")
print(f"  PyTorch version : {torch.__version__}")
print(f"  Dispositivo     : {device.upper()}")
if device == "cuda":
    print(f"  GPU             : {torch.cuda.get_device_name(0)}")
print("=" * 55)

# --- 1. Cargar betahat y sigma desde R ------------------
betahat = pd.read_csv("C:/Users/Usuario/Documents/betahat.csv")["betahat"].values
sigma   = pd.read_csv("C:/Users/Usuario/Documents/sigma.csv").values

print(f"\n  betahat shape : {betahat.shape}")
print(f"  sigma shape   : {sigma.shape}")

numPrePeriods  = 5
numPostPeriods = 2

l_vec = hd.basis_vector(index=1, size=numPostPeriods)
Mvec  = list(np.round(np.arange(0, 0.11, 0.01), 10))

print(f"  numPrePeriods  = {numPrePeriods}")
print(f"  numPostPeriods = {numPostPeriods}")
print(f"  Mvec           = {[round(m,2) for m in Mvec]}  ({len(Mvec)} valores)\n")

# --- 2. Benchmark ---------------------------------------
print("Corriendo hd.createSensitivityResults (Delta_SD)...\n")

t_start = time.perf_counter()

delta_sd_results = hd.createSensitivityResults(
    betahat        = betahat,
    sigma          = sigma,
    numPrePeriods  = numPrePeriods,
    numPostPeriods = numPostPeriods,
    l_vec          = l_vec,
    Mvec           = Mvec,
    alpha          = 0.05
)

t_end   = time.perf_counter()
elapsed = t_end - t_start

# --- 3. Resultados --------------------------------------
print("--- Resultados Delta_SD ---")
print(delta_sd_results)
print(f"\n⏱  Tiempo de ejecución (Python/{device.upper()}): {elapsed:.4f} segundos")
print("=" * 55)
