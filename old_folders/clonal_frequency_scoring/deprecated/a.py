import numpy as np
import pandas as pd


def adjust_prob(
    p,
    w,
    w_low_q=0.05,
    w_high_q=0.80,
    p_low_q=0.05,
    p_high_q=0.5,
    inc_big=0.50,
    inc_small=0.05,
    dec_big=0.05,
    dec_small=0.05,
    clip=True,
):
    """
    Returns adjusted probabilities per row.

    Rules
      High freq + high p -> increase p further (big)
      Low freq  + high p -> maintain or slightly decrease
      High freq + low  p -> maintain or slightly increase
      Low freq  + low  p -> decrease p

    p stays dominant because adjustments are small multiplicative nudges.
    """

    w = np.asarray(w, dtype=float)
    p = np.asarray(p, dtype=float)

    w_low = np.quantile(w, w_low_q)
    w_high = np.quantile(w, w_high_q)
    p_low = np.quantile(p, p_low_q)
    p_high = np.quantile(p, p_high_q)

    high_w = w >= w_high
    low_w = w <= w_low
    high_p = p >= p_high
    low_p = p <= p_low

    factor = np.ones_like(p, dtype=float)

    # High freq, high p -> increase more
    factor[high_w & high_p] += inc_big

    # Low freq, high p -> slight decrease
    factor[low_w & high_p] -= dec_small

    # High freq, low p -> slight increase
    factor[high_w & low_p] += inc_small

    # Low freq, low p -> decrease more
    factor[low_w & low_p] -= dec_big

    adj = p * factor
    if clip:
        adj = np.clip(adj, 0.0, 1.0)
    return adj


# Ensure numeric
csv_path = "/scratch/project/tcr_ml/gnn_release/clonal_frequency_scoring/20240530_WGS_20240530_sc_PICA0001-PICA0007_PMID_97-101_Pool_3_2_merged.csv"
df = pd.read_csv(csv_path)
mean_prob_unweighted = df["prob"].astype(float).mean()
print("Mean unweighted probability:", mean_prob_unweighted)
df["CloneFreq"] = df["CloneFreq"].astype(float).rank(pct=True, method="average")
w = df["CloneFreq"].astype(float).to_numpy()
p = df["prob"].astype(float).to_numpy()
df["adj_prob"] = adjust_prob(p, w)
mean_adj = df["adj_prob"].mean()
print("Mean adjusted probability:", mean_adj)
