import logging
from collections.abc import Iterable
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException


def load_pdb_structure(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        return structure
    except PDBConstructionException as e:
        logging.error(f"Error parsing PDB file {pdb_path}: {e}")
        return None


def residue_pairs_within_cutoff(
    structure, cutoff: float
) -> Iterable[tuple[str, int, str, int]]:
    def atom_for_distance(res):
        return res["CB"] if res.has_id("CB") else res["CA"]

    for model in structure:
        for chain in model:
            residues = [r for r in chain if r.has_id("CA")]
            n = len(residues)
            for i in range(n):
                for j in range(i + 1, n):
                    r1, r2 = residues[i], residues[j]
                    try:
                        a1 = atom_for_distance(r1)
                        a2 = atom_for_distance(r2)
                        if (a1 - a2) <= cutoff:
                            yield (
                                r1.get_resname(),
                                r1.id[1],
                                r2.get_resname(),
                                r2.id[1],
                            )
                    except KeyError:
                        continue


def edges_text(structure, cutoff: float) -> str:
    lines = [
        f"{a} {i} {b} {j}\n"
        for a, i, b, j in residue_pairs_within_cutoff(structure, cutoff)
    ]
    return "".join(lines)
