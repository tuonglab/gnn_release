from __future__ import annotations

import numpy as np

# ----------------------------- Utilities -------------------------------- #


def _apply_open_interval(
    arr: np.ndarray, open_interval: bool, eps: float
) -> np.ndarray:
    """
    Optionally map values from the closed interval [0, 1] into the open interval (eps, 1 - eps).

    Args:
        arr: Array of values assumed to be within [0, 1].
        open_interval: If True, clamp endpoints away from 0 and 1 by eps.
        eps: Small positive value used for endpoint clamping.

    Returns:
        Array with endpoints adjusted when open_interval is True, else the original array.
    """
    if not open_interval:
        return arr
    out = arr.copy()
    out[out <= 0.0] = eps
    out[out >= 1.0] = 1.0 - eps
    return out


def _plotting_position(ranks: np.ndarray, m: int, method: str) -> np.ndarray:
    """
    Compute plotting positions from ranks using a chosen formula.

    Supported methods:
        - "weibull": ranks / (m + 1)
        - "hazen": (ranks - 0.5) / m
        - "blom": (ranks - 0.375) / (m + 0.25)
        - "bernard": (ranks - 3/8) / (m + 0.25)
        - "rank": 0.5 if m == 1 else (ranks - 1) / (m - 1)

    Args:
        ranks: Rank values in 1..m as floats.
        m: Sample size.
        method: Name of the plotting position method.

    Returns:
        Plotting positions in [0, 1].

    Raises:
        ValueError: If an unknown method is provided.
    """
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


def _midranks_for_ties(sorted_vals_len: int, diffs: np.ndarray) -> np.ndarray:
    """
    Assign midranks in 1..m for a sorted array with ties.

    Args:
        sorted_vals_len: Length m of the sorted array.
        diffs: np.diff of the sorted values, used to detect tie boundaries.

    Returns:
        Array of midranks as floats, length m.
    """
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


def _fraction_to_percentile(
    x: np.ndarray | list[float],
    weights: np.ndarray | list[float] | None = None,
    method: str = "hazen",  # "weibull", "hazen", "blom", "bernard", "rank"
    open_interval: bool = False,  # map [0, 1] to (eps, 1 - eps) if True
    eps: float = 1e-9,
    nan_policy: str = "omit",  # "omit", "propagate", "raise"
) -> np.ndarray:
    """
    Percentile transform via ECDF with optional weights and plotting positions.

    Unweighted case:
        Uses the chosen plotting position formula on midranks that account for ties.

    Weighted case:
        Uses midpoint-of-jump ECDF at unique values, which is method free.

    NaN handling:
        - "omit": drop NaNs then write NaN back to their original positions
        - "propagate": return an array of NaNs
        - "raise": raise ValueError when NaNs are present

    Args:
        x: Values to transform.
        weights: Optional nonnegative weights aligned to x.
        method: Plotting position formula for the unweighted case.
        open_interval: If True, map endpoints away from 0 and 1 by eps.
        eps: Small constant used when open_interval is True.
        nan_policy: Policy for handling NaNs.

    Returns:
        Array of percentiles in [0, 1] aligned with x.
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


def combined_score_distribution_aware_simple(
    P: np.ndarray | list[float],
    skew_strength: float = 0.1,
    clip_strength: float = 0.2,
    floor: float = 0.0,
    ceil: float = 1.0,
) -> np.ndarray:
    """
    Shift scores based on global distribution skew.

    Right skew decreases scores, left skew increases scores. The adjustment
    is bounded by clip_strength using a tanh squashing of the skew.

    Args:
        P: Scores in [0, 1] or array like convertible to float array.
        skew_strength: Sensitivity to skew magnitude.
        clip_strength: Maximum absolute adjustment size.
        floor: Lower bound of the returned scores.
        ceil: Upper bound of the returned scores.

    Returns:
        Adjusted scores clipped to [floor, ceil].
    """
    import numpy as np
    from scipy.stats import skew

    P = np.asarray(P, dtype=float)
    P = np.clip(P, 0.0, 1.0)
    adj = -np.tanh(skew(P) * skew_strength) * clip_strength
    return np.clip(P + adj, floor, ceil)


def combined_score_sample_blend(
    P: np.ndarray | list[float],
    F_raw: np.ndarray | list[float],
    high_P: float = 0.9,
    high_F: float = 0.9,
    alpha: float = 0.6,
    beta: float = 0.8,
    gamma: float = 0.5,
) -> np.ndarray:
    """
    Blend per item model scores with sample level frequency ranks.

    The frequency vector F_raw is turned into within sample percentiles R in [0, 1].
    Items that are simultaneously high in both P and R get boosted while items that
    are simultaneously low get penalized. The effect is mixed back with weight gamma.

    Args:
        P: Model scores in [0, 1] or array like.
        F_raw: Raw counts or frequencies aligned to P.
        high_P: Threshold for high model confidence.
        high_F: Threshold for high frequency percentile.
        alpha: Boost coefficient for high high items.
        beta: Penalty coefficient for low low items.
        gamma: Mixing weight between original P and adjusted S_adj.

    Returns:
        Blended scores in [0, 1].
    """
    import numpy as np

    P = np.asarray(P, dtype=float)
    F_raw = np.asarray(F_raw, dtype=float)

    P = np.clip(P, 0.0, 1.0)
    R = _fraction_to_percentile(F_raw)

    mask_high = (P > high_P) & (R > high_F)
    mask_low = (P < 1.0 - high_P) & (R < 1.0 - high_F)

    A_high = np.minimum(P, R) * mask_high
    A_low = np.minimum(1.0 - P, 1.0 - R) * mask_low

    S_adj = P + alpha * A_high - beta * A_low
    return np.clip((1.0 - gamma) * P + gamma * S_adj, 0.0, 1.0)
