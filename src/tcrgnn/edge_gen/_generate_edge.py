import logging
import math
import os
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.Atom import Atom
from Bio.PDB.PDBExceptions import PDBConstructionException
from Bio.PDB.Residue import Residue
from Bio.PDB.Structure import Structure


def load_pdb_structure(pdb_path: str | Path):
    """
    Load a PDB structure from a file path using Bio.PDB.PDBParser.

    Parameters
    ----------
    pdb_path : str or pathlib.Path
        Filesystem path to the PDB file to parse. The parser will use
        the stem (filename without suffix) of the path as the structure id.

    Returns
    -------
    Bio.PDB.Structure.Structure or None
        A Bio.PDB Structure object on successful parsing, or None if parsing
        fails. Parsing errors are logged.

    Example
    -------
    structure = load_pdb_structure("/path/to/file.pdb")
    if structure is None:
        # handle parse failure
    """
    if isinstance(pdb_path, Path):
        structure_id = pdb_path.stem
        pdb_path_str = str(pdb_path)
    else:
        pdb_path = Path(pdb_path)
        structure_id = pdb_path.stem
        pdb_path_str = str(pdb_path)

    parser = PDBParser(QUIET=True)

    try:
        structure = parser.get_structure(structure_id, pdb_path_str)
        return structure
    except PDBConstructionException as e:
        logging.error(f"Error parsing PDB file {pdb_path_str}: {e}")
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
                    a1 = _atom_for_distance(r1)
                    a2 = _atom_for_distance(r2)
                    if (a1 - a2) <= cutoff:
                        yield (
                            r1.get_resname(),
                            r1.id[1],
                            r2.get_resname(),
                            r2.id[1],
                        )


def alternate_parser(pdb_file, output_dir):
    """Generate a residue edge list from a PDB file by measuring distances between CB atoms.

    This function iterates over ATOM records in the provided PDB file, selects the CB atom for each residue
    (or the CA atom when the residue is glycine), and writes every pair of residues within 8 Å of each other
    to an edge list text file in the specified output directory.

    Args:
        pdb_file (str | os.PathLike): Path to the input PDB structure containing ATOM records.
        output_dir (str | os.PathLike): Directory where the generated edge list file will be created.

    Side Effects:
        Creates a ``*_edge.txt`` file alongside the input structure containing residue-residue contacts.
    """
    # residue_idx → { resname: str, coord: (x,y,z) }
    residues = OrderedDict()

    with open(pdb_file) as f:
        for line in f:
            if not line.startswith("ATOM"):
                continue
            cols = line.split()
            atom_name = cols[2]
            resname = cols[3]
            residx = int(cols[5])  # your “1…12” index
            x, y, z = map(float, cols[6:9])

            # pick CB if present, otherwise CA for glycine
            if atom_name == "CB" or (resname == "GLY" and atom_name == "CA"):
                residues.setdefault(residx, {})["coord"] = (x, y, z)
                residues[residx]["resname"] = resname

    # sort by residue index
    items = list(residues.items())  # [(1, {...}), (2, {...}), …]

    out_path = os.path.join(
        output_dir, os.path.basename(pdb_file).replace(".pdb", "_edge.txt")
    )
    with open(out_path, "w") as out:
        for i, (idx1, r1) in enumerate(items):
            for idx2, r2 in items[i + 1 :]:
                d = math.dist(r1["coord"], r2["coord"])
                if d <= 8.0:
                    out.write(f"{r1['resname']} {idx1} {r2['resname']} {idx2}\n")


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
