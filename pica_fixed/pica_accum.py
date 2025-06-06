import os
import pandas as pd

# Root directory where subdirectories are located
root_dir = "/scratch/project/tcr_ml/gnn_release/model_2025_bulk"

# Initialize a list to collect dataframes
all_dfs = []

# List only subdirectories starting with '202'
for subdir_name in os.listdir(root_dir):
    subdir_path = os.path.join(root_dir, subdir_name)
    if os.path.isdir(subdir_path) and subdir_name.startswith("202"):
        metric_file = os.path.join(subdir_path, "metric_scores.csv")
        if os.path.isfile(metric_file):
            df = pd.read_csv(metric_file)
            all_dfs.append(df)

# Combine and save the result
if all_dfs:
    combined_df = pd.concat(all_dfs, ignore_index=True)

    # Create the PICA subdirectory if it doesn't exist
    output_dir = os.path.join(root_dir, "pica_filtered_scores")
    os.makedirs(output_dir, exist_ok=True)

    # Save the combined CSV inside the PICA directory
    output_file = os.path.join(output_dir, "metric_scores.csv")
    combined_df.to_csv(output_file, index=False)
    print(f"PICA_metric_scores.csv has been created at: {output_file}")
else:
    print("No matching metric_scores.csv files found.")
