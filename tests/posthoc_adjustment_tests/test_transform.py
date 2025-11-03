import numpy as np
import pytest

from tcrgnn.posthoc_adjustment._transform import (
    _fraction_to_percentile,
    _midranks_for_ties,
    _plotting_position,
    combined_score_distribution_aware_simple,
    combined_score_sample_blend,
)


def test_fraction_to_percentile_unweighted_with_ties():
    x = np.array([0.2, 0.5, 0.5, 0.9])
    result = _fraction_to_percentile(x, method="hazen")
    expected = np.array([0.125, 0.5, 0.5, 0.875])
    np.testing.assert_allclose(result, expected)


def test_fraction_to_percentile_weighted_midpoint():
    x = np.array([0.1, 0.2, 0.2, 0.9])
    weights = np.array([1.0, 2.0, 1.0, 1.0])
    result = _fraction_to_percentile(x, weights=weights)
    expected = np.array([0.1, 0.5, 0.5, 0.9])
    np.testing.assert_allclose(result, expected)


def test_fraction_to_percentile_open_interval_rank():
    x = np.array([0.0, 0.5, 1.0])
    result = _fraction_to_percentile(x, method="rank", open_interval=True, eps=1e-3)
    expected = np.array([1e-3, 0.5, 1 - 1e-3])
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "nan_policy, expected",
    [
        ("omit", np.array([0.25, np.nan, 0.75])),
        ("propagate", np.array([np.nan, np.nan, np.nan])),
    ],
)
def test_fraction_to_percentile_nan_policy(nan_policy, expected):
    x = np.array([0.1, np.nan, 0.3])
    result = _fraction_to_percentile(x, nan_policy=nan_policy)
    np.testing.assert_allclose(result, expected, equal_nan=True)


@pytest.mark.parametrize(
    "weights",
    [np.array([1.0, -0.2]), np.array([[1.0], [1.0]])],
)
def test_fraction_to_percentile_invalid_weights(weights):
    x = np.array([0.1, 0.2])
    with pytest.raises(ValueError):
        _fraction_to_percentile(x, weights=weights)


def test_fraction_to_percentile_unknown_method_raises():
    with pytest.raises(ValueError):
        _fraction_to_percentile([0.1, 0.2], method="unknown")


def test_combined_score_sample_blend_high_low_adjustments():
    P = np.array([0.95, 0.3, 0.05, 0.85])
    F = np.array([0.9, 0.2, 0.05, 0.8])
    result = combined_score_sample_blend(
        P, F, high_P=0.9, high_F=0.8, alpha=0.6, beta=0.8, gamma=0.5
    )
    expected = np.array([1.0, 0.3, 0.0, 0.85])
    np.testing.assert_allclose(result, expected)


def test_right_skew_decreases_mean():
    # many lows, few highs -> positive skew -> shift down
    P = np.array([0.05] * 90 + [0.9] * 10, dtype=float)
    out = combined_score_distribution_aware_simple(
        P, skew_strength=0.5, clip_strength=0.2
    )
    assert out.shape == P.shape
    assert out.mean() < P.mean()


def test_left_skew_increases_mean():
    # mirror the previous to make negative skew -> shift up
    P = 1.0 - np.array([0.05] * 90 + [0.9] * 10, dtype=float)
    out = combined_score_distribution_aware_simple(
        P, skew_strength=0.5, clip_strength=0.2
    )
    assert out.mean() > P.mean()


def test_near_zero_skew_no_change():
    # uniform linspace has skew ~ 0 -> adj ~ 0
    P = np.linspace(0.0, 1.0, 1000, dtype=float)
    out = combined_score_distribution_aware_simple(
        P, skew_strength=1.0, clip_strength=0.5
    )
    # added adjustment is a scalar; with skew ~0 it should be ~0
    assert np.allclose(out, P, atol=1e-12)


def test_input_clipping_and_output_floor_ceil():
    # values outside [0,1] are clipped first; final output obeys floor/ceil
    P = np.array([-1.5, -0.2, 0.2, 0.8, 1.1, 2.0], dtype=float)
    out = combined_score_distribution_aware_simple(
        P, skew_strength=1.0, clip_strength=0.5, floor=0.1, ceil=0.9
    )
    assert np.all(out >= 0.1) and np.all(out <= 0.9)


def test_zero_clip_strength_no_change_after_initial_clip():
    P = np.array([0.1, 0.3, 0.7, 0.95], dtype=float)
    out = combined_score_distribution_aware_simple(
        P, skew_strength=10.0, clip_strength=0.0
    )
    # with clip_strength==0, adj==0 regardless of skew
    assert np.allclose(out, np.clip(P, 0, 1))


def test_zero_skew_strength_no_change_after_initial_clip():
    P = np.array([0.1, 0.3, 0.7, 0.95], dtype=float)
    out = combined_score_distribution_aware_simple(
        P, skew_strength=0.0, clip_strength=1.0
    )
    # with skew_strength==0, adj==0
    assert np.allclose(out, np.clip(P, 0, 1))


def test_constant_array_produces_nan_output():
    # scipy.stats.skew returns nan for constant arrays
    P = np.array([0.4] * 10, dtype=float)
    out = combined_score_distribution_aware_simple(P)
    assert np.all(np.isnan(out))


def test_shapes_and_bounds_always_valid():
    rng = np.random.default_rng(0)
    P = rng.normal(loc=0.5, scale=0.3, size=1234)  # includes values outside [0,1]
    out = combined_score_distribution_aware_simple(P, floor=0.0, ceil=1.0)
    assert out.shape == P.shape
    assert np.min(out) >= 0.0 and np.max(out) <= 1.0


def test_fraction_to_percentile_single_value():
    x = np.array([0.42], dtype=float)
    out = _fraction_to_percentile(x, nan_policy="omit")
    # Must be 0.5 for m == 1
    assert out.shape == (1,)
    assert np.allclose(out, 0.5)


def test_fraction_to_percentile_empty_input_n_equals_zero():
    x = np.array([], dtype=float)
    out = _fraction_to_percentile(x, nan_policy="omit")
    # For n == 0, the function returns x directly
    assert out.size == 0
    assert out.dtype == float


def test_fraction_to_percentile_all_nan_triggers_m_equals_zero():
    # All NaN -> valid mask empty -> m == 0 block returns all NaN output
    x = np.array([np.nan, np.nan], dtype=float)
    out = _fraction_to_percentile(x, nan_policy="omit")
    # Expect same shape, all NaN
    assert out.shape == x.shape
    assert np.isnan(out).all()


def test_fraction_to_percentile_nan_policy_raise():
    x = np.array([0.2, np.nan, 0.7], dtype=float)
    with pytest.raises(ValueError):
        _fraction_to_percentile(x, nan_policy="raise")


def test_midranks_for_ties_m_equals_zero():
    # m == 0 → sorted_vals_len = 0
    sorted_vals_len = 0
    diffs = np.array([], dtype=float)  # np.diff of an empty array is empty

    out = _midranks_for_ties(sorted_vals_len, diffs)

    # Should return an empty float array
    assert isinstance(out, np.ndarray)
    assert out.size == 0
    assert out.dtype == float


def test_plotting_position_weibull():
    ranks = np.array([1.0, 2.0, 3.0])
    m = 3
    out = _plotting_position(ranks, m, method="weibull")
    expected = ranks / (m + 1.0)
    assert np.allclose(out, expected)


def test_plotting_position_blom():
    ranks = np.array([1.0, 2.0, 3.0])
    m = 3
    out = _plotting_position(ranks, m, method="blom")
    expected = (ranks - 0.375) / (m + 0.25)
    assert np.allclose(out, expected)


def test_plotting_position_bernard():
    ranks = np.array([1.0, 2.0, 3.0])
    m = 3
    out = _plotting_position(ranks, m, method="bernard")
    expected = (ranks - 3.0 / 8.0) / (m + 0.25)
    assert np.allclose(out, expected)
