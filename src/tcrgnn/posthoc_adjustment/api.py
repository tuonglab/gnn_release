import numpy as np
import pandas as pd

from tcrgnn.posthoc_adjustment.clonal_frequency import add_row_frequencies
from tcrgnn.posthoc_adjustment.transform import (
    combined_score_distribution_aware_simple,
    combined_score_sample_blend,
)


def transform_scores(model_output_txt: str, counts: list[float]) -> pd.DataFrame:
    """
    Given a model output file and a list of counts, transform the scores by adding clonal frequencies.

    Args:
        model_output_txt (str): Path to the model output text file.
        counts (list[float]): List of counts corresponding to each sequence.

    Returns:
        pd.DataFrame: DataFrame with sequences, scores, counts, and clonal frequencies.
    """
    df_with_freq = add_row_frequencies(model_output_txt, counts)
    df_with_freq["blended_score"] = combined_score_sample_blend(
        df_with_freq["score"].values, df_with_freq["clonal_frequency"].values
    )
    S = combined_score_distribution_aware_simple(df_with_freq["blended_score"].values)
    df_with_freq["final_score"] = S
    return S


def summary_scores(scores: np.ndarray):
    from scipy.special import expit  # inverse-logit

    # inverse logit transform using scipy
    inv_logit = expit(scores)

    return float(inv_logit.mean())
