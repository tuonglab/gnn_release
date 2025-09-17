import pandas as pd
import numpy as np
import math
from pathlib import Path
import re


def process_sequences(df_freq, df_prob, top_n=50, threshold=0.6):
    df = pd.merge(df_freq, df_prob, on="AA_seq", how="inner")
    df = df.sort_values("CloneFreq", ascending=False)
    df["high"] = df["prob"] > threshold
    high_idx = df.index[df["high"]].tolist()
    df_top = df.iloc[: min(top_n, len(df))]
    S = (df_top["prob"] * df_top["CloneFreq"]).sum()
    weight_score = math.sqrt(1 - math.exp(-S))
    return weight_score, high_idx, df


def batch_process(seq_dir, prob_dir, output_dir, top_n=50, threshold=0.6):
    """
    seq_dir: folder with CSVs containing AA_seq, CloneFreq (match up to '_cdr3', ignore leading 'TRUST_')
    prob_dir: folder with TXT files containing AA_seq, prob (match up to '_extracted')
    output_dir: folder to write per-sample merged CSVs and a summary
    """
    seq_dir = Path(seq_dir)
    prob_dir = Path(prob_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seq_pattern = re.compile(
        r"(?:TRUST_)?(.+?)(?:_(?:cdr3_trb_frequencies|airr_extracted_trb_frequencies_filtered))?\.csv$"
    )

    prob_pattern = re.compile(
        r"(?:TRUST_)?(.+?)(?:_extracted.*|_\d+)?_cancer_cdr3_scores\.txt$"
    )

    # Map prefix -> seq CSV path
    seq_files = {}
    for f in seq_dir.glob("*.csv"):
        m = seq_pattern.match(f.name)
        if m:
            prefix = m.group(1)
            seq_files[prefix] = f

    # Map prefix -> prob TXT path
    prob_files = {}
    for f in prob_dir.glob("*.txt"):
        m = prob_pattern.match(f.name)
        if m:
            prefix = m.group(1)
            prob_files[prefix] = f

    summary = []

    for prefix, seq_path in seq_files.items():
        prob_path = prob_files.get(prefix)
        if not prob_path:
            print(f"[WARNING] No probability file for prefix '{prefix}'")
            continue

        df_freq = pd.read_csv(seq_path)
        df_prob = pd.read_csv(prob_path, sep=",", names=["AA_seq", "prob"], header=None)

        weight_score, high_idx, df_merged = process_sequences(
            df_freq, df_prob, top_n, threshold
        )

        out_csv = output_dir / f"{prefix}_merged.csv"
        df_merged.to_csv(out_csv, index=False)

        summary.append(
            {
                "prefix": prefix,
                "weight_score": weight_score,
                "num_high": len(high_idx),
                "total_seqs": len(df_merged),
            }
        )
        # print(f"[INFO] {prefix}: score={weight_score:.4f}, high={len(high_idx)}/{len(df_merged)}")

    df_summary = pd.DataFrame(summary)
    summary_path = output_dir / "summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"[INFO] Summary saved to {summary_path}")


# Example usage:
batch_process(
    seq_dir="/scratch/project/tcr_ml/iCanTCR/gnn_benchmarking_data_clonal_freq/seekgene",
    prob_dir="/scratch/project/tcr_ml/gnn_release/model_2025_sc_curated/seekgene_scores",
    output_dir="model_2025_sc_curated/seekgene",
)
