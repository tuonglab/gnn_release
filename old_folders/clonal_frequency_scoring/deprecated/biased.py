#!/usr/bin/env python3
import argparse

import numpy as np
import pandas as pd

# ---------------- config defaults ----------------
DEF_CSV = "/scratch/project/tcr_ml/gnn_release/clonal_frequency_scoring/01-0051_409375_T_R_22HM3TLT3_TCAGAAGGCG-GGCCATCATA_R1_merged.csv"
# Strength of within-sample shape nudge from clonal frequency
DEF_TARGET_MEAN_ABS_DELTA = 0.08
# Max average shift magnitude driven by the sample pattern (positive moves mean up)
DEF_DIRECTION_STRENGTH = 0
# Gate on extremeness E in [0,1]
DEF_GAMMA = 1.0
DEF_E0 = 0.2
# Quantiles that define the top and bottom clonal frequency tails
DEF_Q_TOP = 0.98
DEF_Q_BOT = 0.02
# Dead zone on direction signal so tiny differences do not move the mean
DEF_DEAD = 0.02
# Bounds for autoscaled c_scale so things do not blow up
DEF_CSCALE_MIN = 0.02
DEF_CSCALE_MAX = 0.75
# -------------------------------------------------


def solve_shift_for_target(p, base_delta, target_shift, iters=40):
    """
    Find constant s so that mean(clip(p + base_delta + s, 0, 1) - p) ~= target_shift.
    Bisection over s in [-1, 1].
    """
    lo, hi = -1.0, 1.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        m = np.mean(np.clip(p + base_delta + mid, 0.0, 1.0) - p)
        if m < target_shift:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def weighted_mean(x, w):
    num = np.sum(w * x)
    den = np.sum(w) + 1e-12
    return float(num / den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--csv", default=DEF_CSV, help="Input CSV with columns prob and CloneFreq"
    )
    ap.add_argument(
        "--target_mean_abs_delta", type=float, default=DEF_TARGET_MEAN_ABS_DELTA
    )
    ap.add_argument("--direction_strength", type=float, default=DEF_DIRECTION_STRENGTH)
    ap.add_argument("--gamma", type=float, default=DEF_GAMMA)
    ap.add_argument("--e0", type=float, default=DEF_E0)
    ap.add_argument("--q_top", type=float, default=DEF_Q_TOP)
    ap.add_argument("--q_bot", type=float, default=DEF_Q_BOT)
    ap.add_argument("--dead", type=float, default=DEF_DEAD)
    ap.add_argument("--cscale_min", type=float, default=DEF_CSCALE_MIN)
    ap.add_argument("--cscale_max", type=float, default=DEF_CSCALE_MAX)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    p = df["prob"].astype(float).to_numpy()
    w = df["CloneFreq"].astype(float).to_numpy()

    # Optional safety: if p is not in [0,1], minmax it to that range
    pmin, pmax = np.nanmin(p), np.nanmax(p)
    if pmin < 0.0 or pmax > 1.0:
        p = (p - pmin) / (pmax - pmin + 1e-12)

    print("Mean unweighted probability:", float(np.nanmean(p)))

    # Normalize clonal frequency to [0,1] using log1p + minmax
    w_log = np.log1p(w)
    wlog_min = np.nanmin(w_log)
    wlog_max = np.nanmax(w_log)
    r_w = (w_log - wlog_min) / (wlog_max - wlog_min + 1e-12)

    # Centered variables
    rw_c = r_w - 0.5
    p_c = p - 0.5

    # Extremeness in [0,1]
    ext_w = 4.0 * rw_c**2
    ext_p = 4.0 * p_c**2
    E = np.minimum(ext_w, ext_p)

    # Logistic gate g in (0,1)
    g = 1.0 / (1.0 + np.exp(-args.gamma * (E - args.e0)))

    # Interaction strength in [0,1]
    strength = 2.0 * np.minimum(np.abs(rw_c), np.abs(p_c))

    # Shape term: high p nudged up, low p nudged down, scaled by extremeness and alignment
    base_shape = g * np.sign(p_c) * strength  # in [-1, 1]

    # Remove unintentional cohort drift from shape
    base_shape = base_shape - np.nanmean(base_shape)

    # Auto scale to hit target mean absolute delta for shape
    den = np.nanmean(np.abs(base_shape)) + 1e-12
    c_scale = float(
        np.clip(args.target_mean_abs_delta / den, args.cscale_min, args.cscale_max)
    )
    delta_shape = c_scale * base_shape  # zero mean before final shift

    # Label free direction based on top vs bottom clonal frequency tails
    # Masks for top and bottom r_w quantiles
    qt = float(np.clip(args.q_top, 0.5, 0.999))
    qb = float(np.clip(args.q_bot, 0.001, 0.5))
    qt = max(qt, 1.0 - qb)  # keep symmetric-ish tails
    thresh_top = np.quantile(r_w, qt)
    thresh_bot = np.quantile(r_w, qb)
    top_mask = r_w >= thresh_top
    bot_mask = r_w <= thresh_bot

    # Weighted means of p in the tails, focusing on confident extreme points
    wgt = g * strength
    top_mean_p = (
        weighted_mean(p[top_mask], wgt[top_mask])
        if np.any(top_mask)
        else float(np.nanmean(p))
    )
    bot_mean_p = (
        weighted_mean(p[bot_mask], wgt[bot_mask])
        if np.any(bot_mask)
        else float(np.nanmean(p))
    )

    # Direction signal: positive if top-w has higher p than bottom-w
    # Normalize by 0.5 so it stays in [-1, 1] when p is in [0,1]
    ds_raw = np.clip((top_mean_p - bot_mean_p) / 0.5, -1.0, 1.0)

    # Dead zone so tiny contrasts do not move the mean
    if abs(ds_raw) < args.dead:
        ds_eff = 0.0
    else:
        ds_eff = np.sign(ds_raw) * (abs(ds_raw) - args.dead) / (1.0 - args.dead)

    # Translate to target average shift
    target_shift = args.direction_strength * ds_eff

    # Solve constant shift so the mean moves by target_shift after clipping
    s = solve_shift_for_target(p, delta_shape, target_shift)

    # Final score
    raw = p + delta_shape + s
    score = np.clip(raw, 0.0, 1.0)

    # Metrics
    avg_transformed = float(np.nanmean(score))
    achieved_shift = avg_transformed - float(np.nanmean(p))
    g_min, g_med, g_max = (
        float(np.nanmin(g)),
        float(np.nanmedian(g)),
        float(np.nanmax(g)),
    )
    clip_up = int(np.sum(raw > 1.0))
    clip_dn = int(np.sum(raw < 0.0))

    print("New transformed score:", avg_transformed)
    print("c_scale used:", c_scale)
    print("mean |delta_shape|:", float(np.nanmean(np.abs(delta_shape))))
    print("gate g stats min/median/max:", g_min, g_med, g_max)
    print("top_mean_p:", top_mean_p, "bot_mean_p:", bot_mean_p)
    print("direction_signal raw:", float(ds_raw), "effective:", float(ds_eff))
    print(
        "target_shift:",
        float(target_shift),
        "achieved_shift:",
        achieved_shift,
        "constant s:",
        float(s),
    )
    print("clipped up:", clip_up, "clipped down:", clip_dn)


if __name__ == "__main__":
    main()
