import os
import numpy as np
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser

import os
import numpy as np
import matplotlib
matplotlib.use("pdf")  # Use PDF backend for vector output
import matplotlib.pyplot as plt
from Bio.PDB import PDBParser

# Set global font to be Illustrator-compatible
plt.rcParams["pdf.fonttype"] = 42  # Use TrueType fonts
plt.rcParams["ps.fonttype"] = 42

# Output settings
DISTANCE_THRESHOLD = 8.0  # Å
OUTPUT_FILE = "all_contact_maps.pdf"
FREQ_OUTPUT_FILE = "contact_frequency_heatmap.pdf"

# 3-letter to 1-letter amino acid mapping
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D',
    'CYS': 'C', 'GLN': 'Q', 'GLU': 'E', 'GLY': 'G',
    'HIS': 'H', 'ILE': 'I', 'LEU': 'L', 'LYS': 'K',
    'MET': 'M', 'PHE': 'F', 'PRO': 'P', 'SER': 'S',
    'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def get_cb_or_ca_coordinates(structure):
    coords = []
    labels = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if residue.has_id("CA"):
                    if residue.get_resname() == "GLY":
                        atom = residue["CA"]
                    elif residue.has_id("CB"):
                        atom = residue["CB"]
                    else:
                        continue
                    coords.append(atom.coord)
                    labels.append(three_to_one.get(residue.get_resname(), "X"))
    return np.array(coords), labels

def generate_contact_matrix(pdb_path):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("model", pdb_path)
    coords, labels = get_cb_or_ca_coordinates(structure)
    n = len(coords)

    contact_matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if i != j and np.linalg.norm(coords[i] - coords[j]) <= DISTANCE_THRESHOLD:
                contact_matrix[i, j] = 1

    return contact_matrix, labels

def compute_contact_frequency(pdb_paths):
    parser = PDBParser(QUIET=True)
    all_contact_matrices = []
    reference_labels = None

    for pdb_path in pdb_paths:
        structure = parser.get_structure("model", pdb_path)
        coords, labels = get_cb_or_ca_coordinates(structure)
        n = len(coords)
        contact_matrix = np.zeros((n, n), dtype=int)

        for i in range(n):
            for j in range(n):
                if i != j and np.linalg.norm(coords[i] - coords[j]) <= DISTANCE_THRESHOLD:
                    contact_matrix[i, j] = 1

        all_contact_matrices.append(contact_matrix)
        if reference_labels is None:
            reference_labels = labels

    contact_freq_matrix = np.sum(all_contact_matrices, axis=0)
    return contact_freq_matrix, reference_labels

# === Main: Process all PDB files ===
if __name__ == "__main__":
    input_folder = "/scratch/project/tcr_ml/boltz1/seed/seeds"
    pdb_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".pdb")])
    num_files = len(pdb_files)
    full_paths = [os.path.join(input_folder, f) for f in pdb_files]

    cols = 5
    rows = (num_files + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, pdb_file in enumerate(pdb_files):
        full_path = os.path.join(input_folder, pdb_file)
        try:
            matrix, labels = generate_contact_matrix(full_path)
            ax = axes[idx]
            ax.imshow(matrix, cmap="Greys", interpolation="none",rasterized=True)
            ax.set_title(pdb_file, fontsize=8)
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, fontsize=6, rotation=90)
            ax.set_yticklabels(labels, fontsize=6)
        except Exception as e:
            print(f"Error processing {pdb_file}: {e}")

    # Hide unused subplots
    for i in range(len(pdb_files), len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_FILE,dpi=500)
    print(f"\nCombined contact map saved to {OUTPUT_FILE}")

    # === Compute and plot contact frequency heatmap ===
    contact_freq_matrix, freq_labels = compute_contact_frequency(full_paths)

    plt.figure(figsize=(8, 6))
    plt.imshow(contact_freq_matrix, cmap="YlGnBu", interpolation="none",rasterized=True)
    plt.xticks(range(len(freq_labels)), freq_labels, fontsize=8, rotation=90)
    plt.yticks(range(len(freq_labels)), freq_labels, fontsize=8)
    plt.colorbar(label="Number of Models with Contact")
    plt.title("Residue-Residue Contact Frequency Across Models")
    plt.tight_layout()
    plt.savefig(FREQ_OUTPUT_FILE,dpi=1600,bbox_inches="tight")
    plt.show()

    print(f"Contact frequency heatmap saved to {FREQ_OUTPUT_FILE}")
