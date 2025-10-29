import numpy as np
import pandas as pd

# p: df["prob"]
# w: df["CloneFreq"] with values like 0.003
csv_path = "/scratch/project/tcr_ml/gnn_release/clonal_frequency_scoring/01-0051_409375_T_R_22HM3TLT3_TCAGAAGGCG-GGCCATCATA_R1_merged.csv"
df = pd.read_csv(csv_path)
mean_prob_unweighted = df["prob"].astype(float).mean()

w = df["CloneFreq"].astype(float).to_numpy()
p = df["prob"].astype(float).to_numpy()


# Keep finite rows only
mask = np.isfinite(p) & np.isfinite(w)
p = p[mask]
w = w[mask]


# Decide whether to trust w as given frequencies
def looks_like_proportions(arr, tol_sum=1e-2):
    in_unit = np.all((arr >= 0) & (arr <= 1))
    s = arr.sum()
    close_to_one = np.isclose(s, 1.0, rtol=0, atol=tol_sum)
    return in_unit and close_to_one and s > 0


if looks_like_proportions(w):
    f = w.copy()
else:
    # Normalize to proportions
    w_sum = w.sum()
    if w_sum <= 0:
        raise ValueError("CloneFreq sum is nonpositive. Cannot make proportions.")
    f = w / w_sum
# Clip to [0, 1] for stability
f = np.clip(f, 0.0, 1.0)

# Probabilities should already be in [0, 1]. Clip just in case.
p = np.clip(p, 0.0, 1.0)

# Parameters
alpha_geo = 0.9  # weighted geometric mean
alpha_ari = 0.85  # weighted arithmetic
beta_pen = 0.5  # penalty exponent

# Small epsilon to avoid 0**0 in geometric form
eps = 1e-12
p_safe = np.clip(p, eps, 1.0)
f_safe = np.clip(f, eps, 1.0)

# Scores
S_wgeo = (p_safe**alpha_geo) * (f_safe ** (1.0 - alpha_geo))  # weighted geometric mean
S_wari = alpha_ari * p + (1.0 - alpha_ari) * f  # weighted arithmetic
S_pen = p * (f_safe**beta_pen)  # penalty variant

# Write back to DataFrame aligned to original indices
df.loc[mask, f"S_wgeo_a{alpha_geo:.2f}"] = S_wgeo
df.loc[mask, f"S_wari_a{alpha_ari:.2f}"] = S_wari
df.loc[mask, f"S_pen_b{beta_pen:.2f}"] = S_pen

# Optional summary checks
print("w sum used as f:", f.sum())

print(
    df.loc[
        mask,
        [
            "prob",
            "CloneFreq",
            f"S_wgeo_a{alpha_geo:.2f}",
            f"S_wari_a{alpha_ari:.2f}",
            f"S_pen_b{beta_pen:.2f}",
        ],
    ].head()
)
metrics = [
    f"S_wgeo_a{alpha_geo:.2f}",
    f"S_wari_a{alpha_ari:.2f}",
    f"S_pen_b{beta_pen:.2f}",
]

means = df[metrics].mean()

print(means)
print(mean_prob_unweighted)
