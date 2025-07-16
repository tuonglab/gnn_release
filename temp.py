import os

import os

import os

import os

import os

def clean_filenames(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            original_file = file

            # Step 1: Remove 'boltz_results' prefix
            if file.startswith("boltz_results"):
                file = file.replace("boltz_results", "", 1).lstrip("_-")

            # Step 2: Insert '.tsv' before '_cdr3' (if not already there)
            if "_cdr3" in file and ".tsv_cdr3" not in file:
                file = file.replace("_cdr3", ".tsv_cdr3")

            # Step 3: Rename only if changed
            if file != original_file:
                old_path = os.path.join(subdir, original_file)
                new_path = os.path.join(subdir, file)
                os.rename(old_path, new_path)
                print(f"Renamed: {old_path} -> {new_path}")



# Example usage
clean_filenames("/scratch/project/tcr_ml/gnn_release/dataset_boltz/control_test/processed")




import os

def undo_renaming(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            original_file = file

            # Step 1: Remove '_cdr3' if it appears before '_edges'
            if "_cdr3_edges" in file:
                file = file.replace("_cdr3_edges", "_edges")

            # Step 2: Re-add 'boltz_results' prefix
            if not file.startswith("boltz_results"):
                file = "boltz_results_" + file

            # Rename only if filename has changed
            if file != original_file:
                old_path = os.path.join(subdir, original_file)
                new_path = os.path.join(subdir, file)
                os.rename(old_path, new_path)
                print(f"Reverted: {old_path} -> {new_path}")

# Example usage
# undo_renaming("/scratch/project/tcr_ml/gnn_release/dataset_boltz/control_training/processed")

import os

def fix_duplicate_tsv(root_dir):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if ".tsv.tsv" in file:
                fixed_name = file.replace(".tsv.tsv", ".tsv", 1)
                old_path = os.path.join(subdir, file)
                new_path = os.path.join(subdir, fixed_name)
                os.rename(old_path, new_path)
                print(f"Fixed: {old_path} -> {new_path}")

# Example usage
# fix_duplicate_tsv("/scratch/project/tcr_ml/gnn_release/dataset_boltz/control_training/processed")
