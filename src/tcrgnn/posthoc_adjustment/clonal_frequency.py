import pandas as pd


def add_row_frequencies(model_output_txt: str, counts: list[float]) -> pd.DataFrame:
    """
    Given model output file that is from `write_scores_to_txt` with sequences and scores,
    and a list of counts corresponding to each sequence, compute the frequency of each sequence
    based on the counts and add it as a new column to the DataFrame.

    The count can be from UMI count, duplicate_count, or consensus_count depending on the available data.
    """
    df = pd.read_csv(
        model_output_txt, header=None, sep=",", names=["sequence", "score"]
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
