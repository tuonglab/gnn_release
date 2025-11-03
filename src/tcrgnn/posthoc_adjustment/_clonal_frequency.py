from __future__ import annotations

from pathlib import Path

import pandas as pd


def add_row_frequencies(
    model_output_txt: str | Path,
    counts: list[float],
) -> pd.DataFrame:
    """
    Read a model output file produced by write_scores_to_txt and attach
    per sequence clonal frequencies based on provided counts.

    Each row corresponds to a sequence–score pair. Counts are summed per
    unique sequence and normalized to produce clonal_frequency.

    The values in `counts` may represent:
        - UMI count
        - duplicate_count
        - consensus_count
        - or any other abundance proxy available from preprocessing

    These counts are aggregated across identical sequences, allowing
    clonal expansions to influence downstream post hoc score adjustments.

    Args:
        model_output_txt: Path to a CSV containing two columns: sequence, score.
        counts: Per row counts aligned to the input file.

    Returns:
        DataFrame containing:
            - sequence
            - score
            - count
            - clonal_frequency

    Raises:
        ValueError: When counts length does not match file rows or total count is zero.
    """
    df = pd.read_csv(
        model_output_txt,
        header=None,
        sep=",",
        names=["sequence", "score"],
        dtype={"sequence": str, "score": float},
    )

    if len(df) != len(counts):
        raise ValueError(
            "Length of counts does not match number of sequences in model output."
        )

    df["count"] = counts

    per_seq = df.groupby("sequence", dropna=False)["count"].sum()
    total = per_seq.sum()
    if total <= 0:
        raise ValueError("Total counts must be positive to compute frequencies.")

    freq_map = (per_seq / total).to_dict()
    df["clonal_frequency"] = df["sequence"].map(freq_map)

    return df
