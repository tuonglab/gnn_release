import argparse
import glob
import os
import shutil
import subprocess
import tarfile
import traceback
from concurrent.futures import ThreadPoolExecutor

from alternative_parser import write_edges
from Bio.PDB import PDBParser


def extract_tar_file(tar_file) -> str:
    """
    Extracts a tar file to a temporary directory.

    Parameters:
    tar_file (str): The path to the tar file.

    Returns:
    str: The path to the extracted directory.
    """
    directory = os.getenv("TMPDIR", "/tmp/dm")
    base_name = os.path.splitext(os.path.splitext(os.path.basename(tar_file))[0])[0]
    tar_directory = os.path.join(directory, base_name)

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

    import os

    tar_directory = extract_tar_file(tar_file)

    # Recursively collect all files under tar_directory
    extracted_files = []
    for root, dirs, files in os.walk(tar_directory):
        for file in files:
            file_path = os.path.join(root, file)
            extracted_files.append(file_path)

    # Continue with your existing code
    output_dir = os.path.join(
        output_base_dir,
        os.path.basename(tar_directory),
    )
    output_dir += "_edges"
    os.makedirs(output_dir, exist_ok=True)

    def process_pdb_file(file):
        if ("rank_001" in file and file.endswith(".pdb")) or (
            "model_0" in file and file.endswith(".pdb")
        ):
            pdb_file = os.path.join(tar_directory, file)
            check_all_distances(pdb_file, output_dir)
        else:
            print(f"Error generating edgelist for file: {pdb_file}")

    with ThreadPoolExecutor() as executor:
        executor.map(process_pdb_file, extracted_files)
    print("Finish")

    tar_output_file = f"{output_dir}.tar.gz"
    subprocess.run(["tar", "-czf", tar_output_file, "-C", output_dir, "."])

    shutil.rmtree(output_dir)


def check_all_distances(pdb_file, output_dir):
    try:
        structure = load_structure_sanitized(pdb_file)

        pdb_file_name = os.path.basename(pdb_file)
        output_file_name = pdb_file_name.replace(".pdb", "_edge.txt")
        output_file = os.path.join(output_dir, output_file_name)

        with open(output_file, "w") as f:
            for model in structure:
                for chain in model:
                    residues = [r for r in chain if r.has_id("CA")]
                    n = len(residues)
                    for i in range(n):
                        for j in range(i + 1, n):
                            r1 = residues[i]
                            r2 = residues[j]
                            try:
                                a1 = atom_for_distance(r1)
                                a2 = atom_for_distance(r2)
                                d = a1 - a2
                                if d <= 8.0:
                                    f.write(
                                        f"{r1.get_resname()} {r1.id[1]} {r2.get_resname()} {r2.id[1]}\n"
                                    )
                            except KeyError:
                                # skip pairs with missing atoms
                                continue
    except Exception as e:
        # Last resort, your tolerant fallback
        try:
            write_edges(pdb_file, output_dir)
        except Exception as e2:
            print(f"Error processing {pdb_file}: {e} (fallback also failed: {e2})")
            traceback.print_exc()


from io import StringIO

from Bio.PDB.PDBExceptions import PDBConstructionException


def sanitize_pdb_text(pdb_text: str) -> str:
    """
    Fix common malformed PDB issues on the fly.
    1) Two digit insertion codes that shifted columns: insert a blank iCode at col 27.
    """
    out = []
    for line in pdb_text.splitlines(keepends=True):
        if line.startswith(("ATOM  ", "HETATM")) and len(line) > 27:
            # resSeq is cols 23-26 (0-based 22:26), iCode is col 27 (0-based 26)
            if line[22:26].strip().isdigit() and line[26].isdigit():
                # Two digits leaking into iCode, insert a space at col 27
                line = line[:26] + " " + line[26:]
        out.append(line)
    return "".join(out)


def load_structure_sanitized(pdb_path: str):
    """
    Try normal parse. If it fails, read the file, sanitize in memory,
    then parse from a StringIO handle.
    """
    parser = PDBParser(QUIET=True)
    try:
        return parser.get_structure("protein", pdb_path)
    except (ValueError, PDBConstructionException):
        with open(pdb_path) as fh:
            raw = fh.read()
        fixed = sanitize_pdb_text(raw)
        fh_like = StringIO(fixed)
        return parser.get_structure("protein", fh_like)


def atom_for_distance(res):
    # Prefer CB when present, fallback to CA
    return res["CB"] if res.has_id("CB") else res["CA"]


def main():
    parser = argparse.ArgumentParser(
        description="Process tar files and generate edge lists."
    )
    parser.add_argument(
        "--tar-dir", required=True, help="Directory containing the tar files."
    )
    parser.add_argument(
        "--output-base-dir",
        required=True,
        help="Base directory where the output will be stored.",
    )

    args = parser.parse_args()

    # Get the list of all tar files in the directory
    tar_files = glob.glob(os.path.join(args.tar_dir, "*.tar.gz"))

    # Process each tar file sequentially
    for tar_file in tar_files:
        process_tar_file(tar_file, args.output_base_dir)


main()
