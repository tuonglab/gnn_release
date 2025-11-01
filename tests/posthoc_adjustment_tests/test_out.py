from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

from tcrgnn.posthoc_adjustment.api import summary_scores, transform_scores


class DummyLoaderWithLen:
    def __init__(self, batch):
        self._batch = batch

    def __iter__(self):
        return iter([self._batch])

    def __len__(self):
        return 1


class EmptyLoader:
    def __iter__(self):
        return iter([])

    def __len__(self):
        return 0


def test_transform_scores_applies_all_adjustments(monkeypatch):
    df = pd.DataFrame({"score": [0.2, 0.8], "clonal_frequency": [0.1, 0.9]})
    add_mock = MagicMock(return_value=df)

    def fake_blend(scores, freqs):
        np.testing.assert_array_equal(scores, df["score"].values)
        np.testing.assert_array_equal(freqs, df["clonal_frequency"].values)
        return np.array([0.3, 0.7])

    def fake_distribution(blended):
        np.testing.assert_array_equal(blended, np.array([0.3, 0.7]))
        return np.array([0.31, 0.71])

    monkeypatch.setattr("tcrgnn.posthoc_adjustment.api.add_row_frequencies", add_mock)
    monkeypatch.setattr(
        "tcrgnn.posthoc_adjustment.api.combined_score_sample_blend",
        fake_blend,
    )
    monkeypatch.setattr(
        "tcrgnn.posthoc_adjustment.api.combined_score_distribution_aware_simple",
        fake_distribution,
    )

    result = transform_scores("model.txt", [1.0, 2.0])

    add_mock.assert_called_once_with("model.txt", [1.0, 2.0])
    np.testing.assert_array_equal(result, np.array([0.31, 0.71]))
    np.testing.assert_array_equal(df["blended_score"].values, np.array([0.3, 0.7]))
    np.testing.assert_array_equal(df["final_score"].values, np.array([0.31, 0.71]))


def test_summary_scores_returns_mean_inverse_logit():
    scores = np.array([-1.0, 0.0, 1.0])

    expected = float(expit(scores).mean())
    assert summary_scores(scores) == pytest.approx(expected)
