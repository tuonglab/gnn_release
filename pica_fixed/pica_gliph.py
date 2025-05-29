import os
import glob
from collections import defaultdict

CLUSTER_DIR = "/scratch/project/tcr_ml/GLIPH2_PICA_cdr3_outputs"
SCORE_DIR = "/scratch/project/tcr_ml/gnn_release/model_2025_ccdi_only"
OUTPUT_DIR = os.path.join(SCORE_DIR, "pica_filtered_scores")
os.makedirs(OUTPUT_DIR, exist_ok=True)

cluster_cdr3s = defaultdict(set)

# Step 1: Read CDR3s from cluster files
for root, dirs, files in os.walk(CLUSTER_DIR):
    for fname in files:
        if fname.endswith("_cdr3_cluster.txt"):
            pool_id = fname.replace("_cdr3_cluster.txt", "")  # e.g., Pool_1_0
            full_path = os.path.join(root, fname)

            with open(full_path) as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    cdr3 = line.split()[0].strip().upper()
                    cluster_cdr3s[pool_id].add(cdr3)

# Step 2: Match each cluster to the correct score file(s)
for pool_id, cdr3s in cluster_cdr3s.items():
    pattern = os.path.join(SCORE_DIR, f"202*/**/{pool_id}*_cdr3_scores.txt")
    score_files = glob.glob(pattern, recursive=True)

    if not score_files:
        print(f"⚠️ No score file found for: {pool_id}")
        continue

    for score_file in score_files:
        basename = os.path.basename(score_file)
        print(f"🔍 Checking: {basename} starts with {pool_id}? -> {basename.startswith(pool_id)}")

        output_file = os.path.join(OUTPUT_DIR, basename.replace(".txt", "_filtered.txt"))

        matched = 0
        matched = 0
        with open(score_file) as fin, open(output_file, "w") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue

                # split on comma—first item is seq, second is score
                parts = line.split(",", 1)
                if len(parts) != 2:
                    continue

                seq, score = parts
                seq = seq.strip().upper()

                if seq in cdr3s:
                    # write back in the same comma-separated format
                    fout.write(f"{seq},{score}\n")
                    matched += 1

        print(f"✅ {matched} CDR3s written for: {pool_id} -> {output_file}")


        print(f"✅ {matched} CDR3s written for: {pool_id} -> {output_file}")
