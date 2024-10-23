from Bio.PDB import PDBParser

import glob
import subprocess
import os
import tarfile
import shutil


def extract_tar_file(tar_file) -> str:
    """
    Extracts a tar file to a temporary directory.

    Parameters:
    tar_file (str): The path to the tar file.

    Returns:
    str: The path to the extracted directory.
    """
    # Rest of the code...
    # Get the TMPDIR environment variable, or use a default directory
    directory = os.getenv("TMPDIR", "/tmp/dm")
    base_name = os.path.splitext(os.path.splitext(os.path.basename(tar_file))[0])[0]
    tar_directory = os.path.join(directory, base_name)

    # Use Python's tarfile module to extract the tar file
    with tarfile.open(tar_file, "r:gz") as tar:
        tar.extractall(path=tar_directory)

    return tar_directory


def process_tar_file(tar_file, output_base_dir) -> None:
    """
    Process a tar file by extracting its contents, processing the extracted PDB files,
    and creating an output directory with the processed files.

    Args:
        tar_file (str): The path to the tar file to be processed.
        output_base_dir (str): The base directory where the output directory will be created.

    Returns:
        None
    """
    tar_directory = extract_tar_file(tar_file)
    # Get the list of extracted files
    extracted_files = os.listdir(tar_directory)
    output_dir = os.path.join(
        output_base_dir,
        os.path.basename(tar_directory),
    )
    output_dir += "_edges"  # Append 'edges' to the output directory name
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    from concurrent.futures import ThreadPoolExecutor

    def process_pdb_file(file):
        if "rank_001" in file and file.endswith(".pdb"):
            pdb_file = os.path.join(tar_directory, file)
            # Define the output directory based on the extracted file name
            check_all_distances(pdb_file, output_dir)

    # Process only the rank_001 PDB files
    print("Processing extracted files")
    with ThreadPoolExecutor() as executor:
        executor.map(process_pdb_file, extracted_files)
    print("Finish")

    tar_output_file = f"{output_dir}.tar.gz"
    subprocess.run(["tar", "-czf", tar_output_file, "-C", output_dir, "."])

    # Remove the output directory after tarring
    shutil.rmtree(output_dir)


def check_all_distances(pdb_file, output_dir):
    """
    Check all distances between residues in a protein structure and generate an edge list file.

    Parameters:
    - pdb_file (str): The path to the PDB file containing the protein structure.
    - output_dir (str): The directory where the output edge list file will be saved.

    Returns:
    None

    Raises:
    None

    """
    try:
        parser = PDBParser()
        structure = parser.get_structure("protein", pdb_file)

        # Generate the output file name based on the input PDB file name
        pdb_file_name = os.path.basename(pdb_file)
        output_file_name = pdb_file_name.replace(".pdb", "_edge.txt")
        output_file = os.path.join(output_dir, output_file_name)

        # Open the edge list file
        with open(output_file, "w") as f:
            for model in structure:
                for chain in model:
                    residues = [residue for residue in chain if residue.has_id("CA")]
                    for i in range(len(residues)):
                        for j in range(i + 1, len(residues)):
                            residue1 = residues[i]
                            residue2 = residues[j]
                            atom1 = (
                                residue1["CB"]
                                if residue1.get_resname() != "GLY"
                                else residue1["CA"]
                            )
                            atom2 = (
                                residue2["CB"]
                                if residue2.get_resname() != "GLY"
                                else residue2["CA"]
                            )
                            distance = atom1 - atom2
                            if distance <= 8.0:
                                # Write an edge to the file
                                f.write(
                                    f"{residue1.get_resname()} {residue1.id[1]} {residue2.get_resname()} {residue2.id[1]}\n"
                                )
    except:
        print(pdb_file)


def main():
    # Specify the directory containing the tar files
    # Replace with your directory path
    tar_dir = "/scratch/project/tcr_ml/GNN/raw"
    output_base_dir = "/scratch/project/tcr_ml/GNN/reference_data/control/raw"

    # Get a list of all tar files in the directory
    tar_files = glob.glob(os.path.join(tar_dir, "*.tar.gz"))

    # Process each tar file sequentially
    for tar_file in tar_files:
        process_tar_file(tar_file, output_base_dir)


if __name__ == "__main__":
    main()
