# Ensure numeric
import pandas as pd
import numpy as np

csv_path = "/scratch/project/tcr_ml/gnn_release/icantcrscoring/model_2025_sc_curated/PICA/20240530_WGS_20240530_sc_PICA0001-PICA0007_PMID_97-101_Pool_4_1_merged.csv"
df = pd.read_csv(csv_path)
mean_prob_unweighted = df["prob"].astype(float).mean()
w = df["CloneFreq"].astype(float).to_numpy()
p = df["prob"].astype(float).to_numpy()

import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# Ensure numeric
df["CloneFreq"] = pd.to_numeric(df["CloneFreq"], errors="coerce")
df["prob"] = pd.to_numeric(df["prob"], errors="coerce")
df = df.dropna(subset=["CloneFreq", "prob"]).copy()

w = df["CloneFreq"].to_numpy()
p = df["prob"].to_numpy()

# Pearson correlation (linear)
pearson_r, pearson_pval = pearsonr(w, p)

# Spearman correlation (rank/monotonic)
spearman_rho, spearman_pval = spearmanr(w, p)

print("Pearson r:", pearson_r, " (p-value:", pearson_pval, ")")
print("Spearman rho:", spearman_rho, " (p-value:", spearman_pval, ")")
