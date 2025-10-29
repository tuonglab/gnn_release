import numpy as np

# ----------------------------- Utilities -------------------------------- #


def _apply_open_interval(arr, open_interval: bool, eps: float):
    if not open_interval:
        return arr
    out = arr.copy()
    out[out <= 0.0] = eps
    out[out >= 1.0] = 1.0 - eps
    return out


def _plotting_position(ranks, m, method: str):
    if method == "weibull":
        return ranks / (m + 1.0)
    if method == "hazen":
        return (ranks - 0.5) / m
    if method == "blom":
        return (ranks - 0.375) / (m + 0.25)
    if method == "bernard":
        return (ranks - 3.0 / 8.0) / (m + 0.25)
    if method == "rank":
        return np.full(m, 0.5) if m == 1 else (ranks - 1.0) / (m - 1.0)
    raise ValueError(f"Unknown method '{method}'")


def _midranks_for_ties(sorted_vals_len: int, diffs: np.ndarray):
    # diffs is np.diff of sorted values
    # build tie groups, assign midranks in 1..m
    m = sorted_vals_len
    if m == 0:
        return np.empty(0, dtype=float)
    boundaries = np.flatnonzero(diffs != 0.0)
    starts = np.r_[0, boundaries + 1]
    stops = np.r_[boundaries, m - 1]
    ranks = np.empty(m, dtype=float)
    for s, e in zip(starts, stops):  # noqa: B905
        mid = (s + e) / 2.0
        ranks[s : e + 1] = mid + 1.0
    return ranks


# ----------------------------- Main API ---------------------------------- #


def fraction_to_percentile(
    x,
    weights=None,
    method="hazen",  # "weibull", "hazen", "blom", "bernard", "rank"
    open_interval=False,  # map [0,1] to (eps,1-eps) if True
    eps=1e-9,
    nan_policy="omit",  # "omit", "propagate", "raise"
):
    """
    Percentile transform via ECDF with optional weights and plotting positions.

    Unweighted case uses chosen plotting position with midranks for ties.
    Weighted case uses midpoint-of-jump ECDF which is method-free.

    Returns array of percentiles aligned with x.
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n == 0:
        return x

    # NaN handling
    has_nan = np.isnan(x).any()
    if has_nan:
        if nan_policy == "raise":
            raise ValueError("NaNs present with nan_policy='raise'.")
        if nan_policy == "propagate":
            return np.full_like(x, np.nan, dtype=float)

    # Weights
    if weights is None:
        w = np.ones_like(x, dtype=float)
    else:
        w = np.asarray(weights, dtype=float)
        if w.shape != x.shape:
            raise ValueError("weights must have same shape as x")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")

    # Filter valid
    valid = ~np.isnan(x)
    xv = x[valid]
    wv = w[valid]
    m = xv.size

    out = np.full_like(x, np.nan, dtype=float)
    if m == 0:
        return out
    if m == 1:
        out[valid] = 0.5
        return _apply_open_interval(out, open_interval, eps)

    # Sort by value
    order = np.argsort(xv, kind="mergesort")
    xv_sorted = xv[order]
    wv_sorted = wv[order]

    # Unweighted branch if all weights are 1
    if weights is None or np.allclose(wv_sorted, 1.0):
        diffs = np.diff(xv_sorted)
        ranks = _midranks_for_ties(len(xv_sorted), diffs)
        p_sorted = _plotting_position(ranks, m, method)
    else:
        # Weighted midpoint-of-jump ECDF at unique values
        uniq_vals, idx_start, counts = np.unique(
            xv_sorted, return_index=True, return_counts=True
        )
        group_weights = np.add.reduceat(wv_sorted, idx_start)
        cw = np.cumsum(group_weights)
        total = cw[-1]
        cw_prev = cw - group_weights
        p_group = (cw_prev + 0.5 * group_weights) / total
        p_sorted = np.repeat(p_group, counts)

    # Unsort back to valid positions
    inv = np.empty(m, dtype=int)
    inv[order] = np.arange(m)
    out[valid] = p_sorted[inv]
    return _apply_open_interval(out, open_interval, eps)


def sample_skewness(x):
    """
    Compute the Fisher-Pearson sample skewness of a 1D array.
    Returns 0.0 if fewer than 3 elements or zero variance.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    if x.size < 3:
        return 0.0

    mu = x.mean()
    s = x.std(ddof=0)
    if s == 0:
        return 0.0

    return np.mean((x - mu) ** 3) / s**3


def combined_score_distribution_aware_simple(
    P, skew_strength=0.1, clip_strength=0.2, floor=0.0, ceil=1.0
):
    """Shift scores based on distribution skew (right-skew → down, left-skew → up)."""
    import numpy as np
    from scipy.stats import skew

    P = np.clip(P, 0, 1)
    adj = -np.tanh(skew(P) * skew_strength) * clip_strength
    return np.clip(P + adj, floor, ceil)


def combined_score_sample_blend(
    P, F_raw, high_P=0.9, high_F=0.9, alpha=0.6, beta=0.8, gamma=0.5
):
    """Blend scores P and frequencies F_raw using high-confidence weighting."""
    import numpy as np

    P = np.clip(P, 0, 1)
    R = fraction_to_percentile(F_raw)
    mask_high = (P > high_P) & (R > high_F)
    mask_low = (P < 1 - high_P) & (R < 1 - high_F)

    A_high = np.minimum(P, R) * mask_high
    A_low = np.minimum(1 - P, 1 - R) * mask_low

    S_adj = P + alpha * A_high - beta * A_low
    return np.clip((1 - gamma) * P + gamma * S_adj, 0, 1)
