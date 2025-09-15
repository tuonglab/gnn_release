from pathlib import Path
import pandas as pd
import numpy as np
import math
import re

def process_sequences(df_freq, df_prob, top_n=50, threshold=0.6):
    df = pd.merge(df_freq, df_prob, on='AA_seq', how='inner')
    df = df.sort_values('CloneFreq', ascending=False)
    df['high'] = df['prob'] > threshold
    high_idx = df.index[df['high']].tolist()
    df_top = df.iloc[:min(top_n, len(df))]
    S = (df_top['prob'] * df_top['CloneFreq']).sum()
    weight_score = math.sqrt(1 - math.exp(-S))
    return weight_score, high_idx, df

def batch_process(seq_dir, prob_dir, output_dir, top_n=50, threshold=0.6):
    seq_dir = Path(seq_dir)
    prob_dir = Path(prob_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Map keys "poolName_index" to seq CSVs
    seq_files = {}
    for seq_csv in seq_dir.rglob('*_filtered.csv'):
        pool_name = seq_csv.parent.name  # e.g., 20240530_..._Pool_1
        idx = seq_csv.stem.split('_')[0]  # "0", "1", "2"
        key = f"{pool_name}_{idx}"
        seq_files[key] = seq_csv

    prob_files = {}
    pattern = re.compile(r'(.+_Pool_\d+)_([0-9]+)_\d+_control_cdr3_scores\.txt$')
    for prob_txt in prob_dir.rglob('*.txt'):
        m = pattern.search(prob_txt.name)
        if m:
            pool_name, idx = m.group(1), m.group(2)
            key = f"{pool_name}_{idx}"
            prob_files[key] = prob_txt

    summary = []
    for key, seq_path in seq_files.items():
        prob_path = prob_files.get(key)
        if not prob_path:
            print(f"[WARNING] No prob file for {key}")
            continue

        df_freq = pd.read_csv(seq_path)
        df_prob = pd.read_csv(prob_path, sep=',', names=['AA_seq','prob'], header=None)

        weight_score, high_idx, df_merged = process_sequences(df_freq, df_prob, top_n, threshold)

        out_csv = output_dir / f"{key}_merged.csv"
        df_merged.to_csv(out_csv, index=False)
        summary.append({
            'key': key,
            'weight_score': weight_score,
            'num_high': len(high_idx),
            'total_seqs': len(df_merged)
        })
        print(f"[INFO] {key}: score={weight_score:.4f}, high={len(high_idx)}/{len(df_merged)}")

    pd.DataFrame(summary).to_csv(output_dir / "summary.csv", index=False)

# Example call:
batch_process(
   seq_dir="/scratch/project/tcr_ml/iCanTCR/gnn_benchmarking_data_clonal_freq/PICA",
   prob_dir="/scratch/project/tcr_ml/gnn_release/model_2025_bulk",
   output_dir="model_2025_bulk/PICA"
)
