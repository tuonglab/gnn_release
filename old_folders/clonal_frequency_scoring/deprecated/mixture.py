import numpy as np
import pandas as pd
eps = 1e-12

csv_path = "/scratch/project/tcr_ml/gnn_release/clonal_frequency_scoring/01-0051_409375_T_R_22HM3TLT3_TCAGAAGGCG-GGCCATCATA_R1_merged.csv"
df = pd.read_csv(csv_path)
mean_prob_unweighted = df["prob"].astype(float).mean()

w = df["CloneFreq"].astype(float).to_numpy()
p = df["prob"].astype(float).to_numpy()

def logit(p, eps=1e-12):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def first_method_score(w, p, beta=1.6, gamma=1.0, T=None, a_cap=None, shrink_lambda=0.0):
    # w: clone frequencies (>=0), p: per clone probabilities in [0,1]
    w = np.clip(np.asarray(w, float), 0.0, None)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)

    # normalize w, optional shrinkage toward uniform
    if np.all(w == 0):
        w = np.ones_like(w)
    w_norm = w / (w.sum() + eps)
    if shrink_lambda > 0:
        n = len(w_norm)
        w_norm = (1 - shrink_lambda) * w_norm + shrink_lambda * (1.0 / n)

    # optional temperature scaling of p before any exponent
    if T is not None and T > 1:
        p = sigmoid(logit(p, eps) / T)

    # weights a_i from frequency
    a = np.power(w_norm + eps, beta)

    # optional cap to prevent a single clone dominating
    if a_cap is not None:
        a = np.minimum(a, a_cap)

    # normalize a
    a = a / (a.sum() + eps)

    # convex emphasis of p via gamma (keep gamma near 1.0 for control down)
    p_gamma = np.power(p, gamma)

    # new sequence distribution and sample scores
    numer = a * p_gamma
    tilde = numer / (numer.sum() + eps)        # new per-sequence probability share
    S_old = float(np.sum(w_norm * p))          # old frequency-weighted mean
    S_new = float(np.sum(tilde * p))           # new mixture score
    S_unw = float(np.mean(p))                  # unweighted mean
    delta_vs_unw = S_new - S_unw
    delta_vs_old = S_new - S_old

    return {
        "w_norm": w_norm,
        "a": a,
        "tilde": tilde,
        "S_unweighted": S_unw,
        "S_old_weighted": S_old,
        "S_new_mixture": S_new,
        "delta_vs_unweighted": delta_vs_unw,
        "delta_vs_old": delta_vs_old,
    }

# method = first_method_score(w, p, beta=0.9, gamma=1.0, T=3, a_cap=0.01, shrink_lambda=0.0)
# print("Old prob: ", method["S_unweighted"])
# print("New prob: ", method["S_new_mixture"])


def mixture_below_mean_soft(
    w, p,
    beta=1.1,          # softer frequency priority
    eta=0.5,           # temper the deficit with a power 0<eta<=1
    hinge=0.0,         # move the cutoff: use deficit = max(mu - p - hinge, 0)
    lambda_uniform=0.02,  # add a small uniform mass to avoid collapse
    blend_old=0.20,    # blend back toward old weights
    a_cap=0.01,        # cap any one clone’s priority at 1%
    shrink_lambda=0.10,# shrink frequencies toward uniform
    clamp_abs=0.05,    # do not let New drop more than 0.05 absolute below mu
    clamp_rel=0.20     # or more than 20 percent relative
):
    """
    Returns:
      S_unweighted = mean(p)
      S_old_weighted = sum w_norm * p
      S_new_mixture_soft = softened and clamped mixture score
      diagnostics dict with w_norm, a, tilde, deficit
    Guarantee:
      After clamping, S_new_mixture_soft is in [mu - max(abs,rel), mu]
    """

    w = np.clip(np.asarray(w, float), 0.0, None)
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    n = len(p)

    if np.all(w == 0):
        w = np.ones_like(w)

    # normalize and shrink extreme clonality
    w_norm = w / (w.sum() + eps)
    if shrink_lambda > 0:
        w_norm = (1 - shrink_lambda) * w_norm + shrink_lambda * (1.0 / n)

    # priority from frequency
    a = np.power(w_norm + eps, beta)
    if a_cap is not None:
        a = np.minimum(a, a_cap)
    a = a / (a.sum() + eps)

    # soft deficit: emphasize below-mean but temper with eta
    mu = float(np.mean(p))
    deficit = np.maximum(mu - p - hinge, 0.0) ** eta

    # add a little uniform mass to avoid collapse
    numer = a * deficit + lambda_uniform * (1.0 / n)
    if np.all(numer == 0):
        numer = np.ones_like(p)
    tilde = numer / (numer.sum() + eps)

    # scores
    S_unweighted = float(np.mean(p))
    S_old_weighted = float(np.sum(w_norm * p))
    S_new_raw = float(np.sum(tilde * p))

    # clamp so the drop is not too large
    lo_abs = mu - clamp_abs
    lo_rel = mu * (1 - clamp_rel)
    lower_bound = max(lo_abs, lo_rel)
    S_new_soft = float(np.clip(S_new_raw, lower_bound, mu))

    diag = dict(w_norm=w_norm, a=a, tilde=tilde, deficit=deficit, mu=mu,
                S_new_raw=S_new_raw, lower_bound=lower_bound)
    return S_unweighted, S_old_weighted, S_new_soft, diag

S_unw, S_old, S_new_soft, diag = mixture_below_mean_soft(w, p,
    beta=1.1, eta=0.5, hinge=0.0,
    lambda_uniform=0.02, blend_old=0.20,
    a_cap=0.01, shrink_lambda=0.10,
    clamp_abs=0.05, clamp_rel=0.20)
print("Old prob: ", S_unw)
print("New prob: ", S_new_soft)