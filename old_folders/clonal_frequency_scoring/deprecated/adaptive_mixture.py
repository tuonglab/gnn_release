import numpy as np
import pandas as pd
eps = 1e-12


csv_path = "/scratch/project/tcr_ml/gnn_release/icantcrscoring/model_2025_sc_curated/PICA/20240918_WGS_20240924_sc_PICA0008-PICA0032_Pool_7_1_merged.csv"
df = pd.read_csv(csv_path)
mean_prob_unweighted = df["prob"].astype(float).mean()

w = df["CloneFreq"].astype(float).to_numpy()
p = df["prob"].astype(float).to_numpy()
def logit(p):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p) - np.log(1 - p)

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

def _normalize_w(w):
    w = np.clip(np.asarray(w, float), 0.0, None)
    if np.all(w == 0): w = np.ones_like(w)
    return w / (w.sum() + eps)

def _mixture_score(w_norm, p, beta=1.3, gamma=1.0, T=None, a_cap=None):
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    if T is not None and T > 1:
        p = sigmoid(logit(p) / T)  # compress highs gently
    a = np.power(w_norm + eps, beta)
    if a_cap is not None:
        a = np.minimum(a, a_cap)
    a = a / (a.sum() + eps)
    p_gamma = np.power(p, gamma)
    tilde = (a * p_gamma)
    tilde = tilde / (tilde.sum() + eps)
    return float(np.sum(tilde * p))

def _mixture_below_mean_soft(w_norm, p,
                             beta=1.1, eta=0.6, hinge=0.0,
                             lambda_uniform=0.02, a_cap=0.01):
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    n = len(p)
    a = np.power(w_norm + eps, beta)
    if a_cap is not None:
        a = np.minimum(a, a_cap)
    a = a / (a.sum() + eps)
    mu = float(np.mean(p))
    deficit = np.maximum(mu - p - hinge, 0.0) ** eta
    numer = a * deficit + lambda_uniform * (1.0 / n)
    if np.all(numer == 0): numer = np.ones_like(p)
    tilde = numer / (numer.sum() + eps)
    return float(np.sum(tilde * p))

import numpy as np

eps = 1e-12

import numpy as np

eps = 1e-12

def adaptive_mixture(
    w, p,
    tau_pos=0.02, tau_neg=0.02,    # switching thresholds
    shrink_lambda=0.10,            # reduce extreme clonality
    boost_beta=1.4, boost_T=2.0,   # gentle boost
    drop_beta=1.1, drop_eta=0.6,   # gentle drop
    a_cap=0.01,                    # max 1 percent priority per clone
    clamp_abs=0.08, clamp_rel=0.25,
    mu_high=0.60,                  # e.g. 0.60, force boost if mu >= mu_high
    mu_drop_max=0.55               # e.g. 0.55, only allow drop if mu <= mu_drop_max
):
    """
    Returns dict:
      S_unweighted, S_old_weighted, S_new_adaptive, regime, diagnostics
    Behavior:
      if gap >= tau_pos -> boost
      if gap <= -tau_neg -> drop
      else neutral
      Movement is clamped by absolute and relative caps around mu.
    """
    # normalize and gently shrink clonality
    w = np.clip(np.asarray(w, float), 0.0, None)
    if np.all(w == 0): w = np.ones_like(w)
    w_norm = w / (w.sum() + eps)
    if shrink_lambda > 0:
        n = len(w_norm)
        w_norm = (1 - shrink_lambda) * w_norm + shrink_lambda * (1.0 / n)

    # probabilities
    p = np.clip(np.asarray(p, float), 0.0, 1.0)
    mu = float(np.mean(p))                                  # unweighted mean
    S_old = float(np.sum(w_norm * p))                       # frequency-weighted mean
    gap = S_old - mu

    # clamp bounds around mu
    upper_bound = min(mu + clamp_abs, mu * (1 + clamp_rel))
    lower_bound = max(mu - clamp_abs, mu * (1 - clamp_rel))

    # decide regime with optional mu-based guards
    regime = None
    if mu_high is not None and mu >= mu_high:
        regime = "boost"
    if regime is None:
        if gap >= tau_pos:
            regime = "boost"
        elif gap <= -tau_neg:
            if (mu_drop_max is None) or (mu <= mu_drop_max):
                regime = "drop"
            else:
                regime = "neutral"
        else:
            regime = "neutral"

    # compute raw new score by branch
    if regime == "boost":
        S_new_raw = _mixture_score(w_norm, p, beta=boost_beta, gamma=1.0, T=boost_T, a_cap=a_cap)
        S_new = float(np.clip(S_new_raw, mu, upper_bound))
    elif regime == "drop":
        S_new_raw = _mixture_below_mean_soft(
            w_norm, p, beta=drop_beta, eta=drop_eta, hinge=0.0,
            lambda_uniform=0.02, a_cap=a_cap
        )
        S_new = float(np.clip(S_new_raw, lower_bound, mu))
    else:
        S_new_raw = mu
        S_new = mu

    return {
        "S_unweighted": mu,
        "S_old_weighted": S_old,
        "S_new_adaptive": S_new,
        "regime": regime,
        "diagnostics": {
            "gap": gap,
            "S_new_raw": S_new_raw,
            "upper_bound": upper_bound,
            "lower_bound": lower_bound,
            "tau_pos": tau_pos,
            "tau_neg": tau_neg,
            "mu_high": mu_high,
            "mu_drop_max": mu_drop_max
        }
    }


# w and p are your arrays
out = adaptive_mixture(w, p)
print("Regime:", out["regime"])
print("Unweighted mean:", out["S_unweighted"])
print("Old weighted:", out["S_old_weighted"])
print("New adaptive:", out["S_new_adaptive"])