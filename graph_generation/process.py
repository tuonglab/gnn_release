import os
import argparse
from graph import MultiGraphDataset

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process GNN dataset.")
    parser.add_argument("--root-dir", required=True, help="Root directory where the data should be stored.")
    parser.add_argument("--directory", required=True, help="Directory containing the .tar.gz files.")
    
    args = parser.parse_args()

    # Use the provided root directory and data directory from the arguments
    root_dir = args.root_dir
    directory = args.directory

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

    # Remove pre_filter.pt and pre_transform.pt files
    os.remove(os.path.join(new_root, "processed", "pre_filter.pt"))
    os.remove(os.path.join(new_root, "processed", "pre_transform.pt"))

if __name__ == "__main__":
    main()
