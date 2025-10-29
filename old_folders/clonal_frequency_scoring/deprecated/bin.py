import numpy as np
import pandas as pd

# Ensure numeric
csv_path = "/scratch/project/tcr_ml/gnn_release/clonal_frequency_scoring/20240530_WGS_20240530_sc_PICA0001-PICA0007_PMID_97-101_Pool_3_2_merged.csv"
df = pd.read_csv(csv_path)
mean_prob_unweighted = df["prob"].astype(float).mean()
w = df["CloneFreq"].astype(float).to_numpy()
p = df["prob"].astype(float).to_numpy()

df["CloneFreq"] = pd.to_numeric(df["CloneFreq"], errors="coerce")
df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
df = df.dropna(subset=["CloneFreq", "prob"]).copy()

# Log-frequency for stable binning on heavy-tailed counts
df["log_freq"] = np.log10(df["CloneFreq"] + 1.0)

# Create equal-count bins over log-frequency
# Adjust q if you have few clones
q = 5
df["freq_bin"] = pd.qcut(df["log_freq"], q=q, duplicates="drop")

# -------------------------------
# Method A: percentile-in-bin
# -------------------------------
df["prob_pct_in_bin"] = df.groupby("freq_bin")["prob"].rank(pct=True)

# Optional repertoire-level aggregates using normalized scores
n = df["CloneFreq"].to_numpy()
sA = df["prob_pct_in_bin"].to_numpy()

S_mean_A = np.average(sA, weights=n)  # frequency-weighted mean of percentiles
S_any_A = 1.0 - np.prod(
    (1.0 - sA) ** n
)  # independence, probability at least one cancer-like cell

# -------------------------------
# Method B: within-bin residuals and z-scores
# -------------------------------
g = df.groupby("freq_bin")["prob"]

mu = g.transform("mean")
sigma = g.transform("std")

df["prob_resid"] = df["prob"] - mu

# Guard against zero std
sigma_safe = sigma.replace(0, np.nan)
df["prob_z"] = (df["prob"] - mu) / sigma_safe

# You can map residuals or z to [0,1] if you need a probability-like score
# Here is an optional rank-based rescaling of residuals within each bin
df["prob_resid_pct_in_bin"] = df.groupby("freq_bin")["prob_resid"].rank(pct=True)

sB = df["prob_resid_pct_in_bin"].to_numpy()
S_mean_B = np.average(sB, weights=n)
S_any_B = 1.0 - np.prod((1.0 - sB) ** n)

# -------------------------------
# Quick summaries
# -------------------------------
print("Method A")
print("  Weighted mean of percentiles:", round(S_mean_A, 6))
print("  At least one (independence):", round(S_any_A, 6))

print("Method B")
print("  Weighted mean of resid percentiles:", round(S_mean_B, 6))
print("  At least one (independence):", round(S_any_B, 6))

print("Unwerighted mean prob:", round(mean_prob_unweighted, 6))
