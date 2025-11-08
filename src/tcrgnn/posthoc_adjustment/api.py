from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tcrgnn.posthoc_adjustment._clonal_frequency import add_row_frequencies
from tcrgnn.posthoc_adjustment._transform import (
    combined_score_distribution_aware_simple,
    combined_score_sample_blend,
)


def transform_scores(
    model_output_txt: str | Path,
    counts: pd.DataFrame,
) -> np.ndarray:
    """
    Read model outputs, add clonal frequencies, and return the final adjusted scores.

    Pipeline:
      1) add_row_frequencies -> adds 'clonal_frequency'
      2) combined_score_sample_blend -> 'blended_score'
      3) combined_score_distribution_aware_simple -> final score array
    """
    df = add_row_frequencies(model_output_txt, counts)

    blended = combined_score_sample_blend(
        df["score"].to_numpy(), df["clonal_frequency"].to_numpy()
    )
    final_scores = combined_score_distribution_aware_simple(blended)

    # optional: keep these if other code inspects df, but still return the array
    df["blended_score"] = blended
    df["final_score"] = final_scores

    return final_scores


def summary_scores(scores: np.ndarray | list[float]) -> float:
    """
    Compute the mean of inverse logit transformed scores.

    Args:
        scores: Iterable of numeric scores.

    Returns:
        Mean inverse logit (scalar).
    """
    from scipy.special import expit  # inverse logit

    arr = np.asarray(scores, dtype=float)
    inv_logit = expit(arr)
    return float(inv_logit.mean())
