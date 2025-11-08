from __future__ import annotations

from pathlib import Path

import pandas as pd


def add_row_frequencies(
    model_output_txt: str | Path,
    counts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach per-sequence clonal frequencies to a model output file.

    Args:
        model_output_txt: Path to a CSV (no header) with two columns: sequence, score.
        counts_df: DataFrame where:
            - first column is the sequence
            - second column is the clonal frequency

    Returns:
        DataFrame with columns:
            - sequence
            - score
            - clonal_frequency

    Raises:
        ValueError: if clonal frequencies are missing for any sequences.
    """
    # Read model output (no header)
    df = pd.read_csv(
        model_output_txt,
        header=None,
        sep=",",
        names=["sequence", "score"],
        dtype={0: str, 1: float},
    )

    # Normalize column names
    seq_col, freq_col = counts_df.columns[:2]
    counts_df = counts_df.loc[:, [seq_col, freq_col]].rename(
        columns={seq_col: "sequence", freq_col: "clonal_frequency"}
    )

    # Drop duplicate sequences, keep first
    counts_df = counts_df.drop_duplicates(subset=["sequence"], keep="first")

    # Merge on sequence (many-to-one now guaranteed)
    merged = df.merge(counts_df, on="sequence", how="left", validate="m:1")

    # Check for missing frequencies
    if merged["clonal_frequency"].isnull().any():
        missing = merged.loc[merged["clonal_frequency"].isnull(), "sequence"].unique()
        raise ValueError(f"Missing clonal frequencies for sequences: {missing}")

    return merged[["sequence", "score", "clonal_frequency"]]
