# io.py
import logging
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

LOG = logging.getLogger("tcrgnn.edgegen.io")


def iter_target_pdbs(root: Path | str, patterns: tuple[str, ...]) -> list[Path | str]:
    """
    Recursively discover PDB files under a root directory matching patterns.

    Args:
        root: Directory to search.
        patterns: Tuple of substrings that must appear in the filename.

    Returns:
        List of matching PDB paths.

    """
    root = Path(root)
    return [p for p in root.rglob("*.pdb") if any(pat in p.name for pat in patterns)]


def load_pdb_structure(pdb_path: Path | str):
    """
    Parse a PDB file into a Biopython Structure.

    Args:
        pdb_path: Path to a PDB file.

    Returns:
        A Biopython Structure object if parsing succeeds, otherwise None.

    Logs:
        Errors encountered during parsing.

    Note:
        This function does not validate PDB extension or existence.
    """
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        return structure
    except PDBConstructionException as e:
        logging.error(f"Error parsing PDB file {pdb_path}: {e}")
        return None
