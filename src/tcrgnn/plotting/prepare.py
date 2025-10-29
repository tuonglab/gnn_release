from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple

def prepare_labels_and_scores(
    cancer_df: pd.DataFrame,
    control_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    cancer_scores = cancer_df["Mean Score"].to_numpy()
    control_scores = control_df["Mean Score"].to_numpy()

    cancer_labels = np.ones(len(cancer_scores), dtype=float)
    control_labels = np.zeros(len(control_scores), dtype=float)

    all_scores = np.concatenate([cancer_scores, control_scores])
    all_labels = np.concatenate([cancer_labels, control_labels])

    return all_scores, all_labels, cancer_scores, control_scores
