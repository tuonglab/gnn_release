import logging
from collections.abc import Iterable
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


def load_pdb_structure(pdb_path: Path) -> Structure | None:
    """
    Load a PDB structure from a file path using Bio.PDB.PDBParser.

    Parameters
    ----------
    pdb_path : pathlib.Path
        Filesystem path to the PDB file to parse. The parser will use
        pdb_path.stem as the structure id and str(pdb_path) as the file path.

    Returns
    -------
    Bio.PDB.Structure.Structure or None
        A Bio.PDB Structure object on successful parsing, or None if parsing
        fails (e.g., a PDBConstructionException is raised). Parsing errors are
        logged via the module's logging facility.

    Example
    -------
    structure = load_pdb_structure(Path("/path/to/file.pdb"))
    if structure is None:
        # handle parse failure
    """
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
    """
    Yield pairs of residues from the same chain whose representative atoms are within a given distance cutoff.

    This generator iterates models and chains in the provided structure, collects residues that contain a "CA"
    atom, and compares each residue pair within the same chain. For each residue it uses the "CB" atom if present;
    otherwise it falls back to the "CA" atom. If the Euclidean distance between the two chosen atoms is less than or
    equal to `cutoff`, a tuple describing the residue pair is yielded.

    Args:
        structure: An iterable structure object (e.g., a Bio.PDB Structure or Model) containing models -> chains -> residues -> atoms.
                   Each residue is expected to support .has_id(atom_name) and atom access like residue["CA"] or residue["CB"].
        cutoff (float): Distance threshold in the same units as the structure's atomic coordinates (typically Angstroms).

    Yields:
        tuple[str, int, str, int]: Tuples of the form (resname1, resid1, resname2, resid2) where:
            - resname1/resname2: three-letter residue names (e.g., "ALA")
            - resid1/resid2: integer residue sequence identifiers (value taken from residue.id[1])

    Example:
        >>> for res1_name, res1_id, res2_name, res2_id in residue_pairs_within_cutoff(
        ...     structure, 8.0
        ... ):
        ...     print(f"{res1_name}{res1_id} - {res2_name}{res2_id}")
    """

    def _atom_for_distance(res: Residue) -> Atom:
        return res["CB"] if res.has_id("CB") else res["CA"]

    for model in structure:
        for chain in model:
            residues = [r for r in chain if r.has_id("CA")]
            n = len(residues)
            for i in range(n):
                for j in range(i + 1, n):
                    r1, r2 = residues[i], residues[j]
                    try:
                        a1 = _atom_for_distance(r1)
                        a2 = _atom_for_distance(r2)
                        if (a1 - a2) <= cutoff:
                            yield (
                                r1.get_resname(),
                                r1.id[1],
                                r2.get_resname(),
                                r2.id[1],
                            )
                    except KeyError:
                        continue


def edges_text(structure: Structure, cutoff: float) -> str:
    """
    Return a textual representation of residue-residue edges found within a distance cutoff.

    This function iterates over residue pairs produced by residue_pairs_within_cutoff(structure, cutoff)
    and formats each pair as a single whitespace-separated line terminated with a newline.

    Parameters
    ----------
    structure
        An object describing the molecular structure to search. It must be acceptable to
        residue_pairs_within_cutoff, which yields tuples of four values for each edge.
    cutoff : float
        Distance threshold used to determine whether two residues form an edge.

    Returns
    -------
    str
        A multi-line string where each line corresponds to one residue pair and has the form:
        "<a> <i> <b> <j>\n"
        Here `a` and `b` are residue identifiers (e.g., chain ID or residue name) and `i` and `j`
        are residue indices. Lines appear in the same order as yielded by residue_pairs_within_cutoff.

    Raises
    ------
    Any exceptions raised by residue_pairs_within_cutoff are propagated to the caller.

    Examples
    --------
    # Example resulting string (two edges):
    # "A 12 B 45\nA 13 A 14\n"
    """
    lines = [
        f"{a} {i} {b} {j}\n"
        for a, i, b, j in residue_pairs_within_cutoff(structure, cutoff)
    ]
    return "".join(lines)
