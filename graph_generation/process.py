import os

from graph import MultiGraphDataset

# Specify the root directory where the data should be stored
root_dir = "reference_data/control"
# Specify the directory
directory = f"/scratch/project/tcr_ml/GNN/{root_dir}/raw"

# Get a list of all files in the directory
all_files = os.listdir(directory)

# Filter the list to include only .tar.gz files
samples = [file for file in all_files if file.endswith(".tar.gz")]

# Filter the samples list to exclude files that have already been processed
samples = [
    sample
    for sample in samples
    if not os.path.exists(
        os.path.join(f"{root_dir}/processed", sample.replace(".tar.gz", ".tar.pt"))
    )
]
# Create an instance of the MultiGraphDataset class
new_root = f"../{root_dir}"
dataset = MultiGraphDataset(root=new_root, samples=samples, cancer=False)

# Process the data
dataset.process()

os.remove(os.path.join(new_root, "processed", "pre_filter.pt"))
os.remove(os.path.join(new_root, "processed", "pre_transform.pt"))
