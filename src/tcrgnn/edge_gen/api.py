import _io as io
import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from .generate_edge import edges_text, load_pdb_structure

LOG = logging.getLogger("tcrgnn.edgegen")


@dataclass
class EdgeGenConfig:
    """Configuration options for generating graph edges.

    Attributes:
        cutoff: Maximum distance threshold used when determining whether to connect nodes.
        patterns: Tuple of pattern identifiers that select which edge-generation routines to execute.
        keep_expanded: Whether to retain expanded intermediate edge representations for downstream use.
    """

    cutoff: float = 8.0
    patterns: tuple[str, ...] = ("rank_001", "model_0")
    keep_expanded: bool = False


def write_edges_file_for_pdb(pdb_path: Path, out_dir: Path, cutoff: float) -> Path:
    """
    Generate an edge list text file for a given PDB structure using a distance cutoff.

    Parameters
    ----------
    pdb_path : pathlib.Path
        Path to the input PDB file; must exist and have a .pdb extension.
    out_dir : pathlib.Path
        Directory where the generated edge file will be written. Created if missing.
    cutoff : float
        Maximum distance used to determine whether an edge should be included.

    Returns
    -------
    pathlib.Path
        Path to the newly written edge file containing the serialized edge data.

    Raises
    ------
    FileNotFoundError
        If `pdb_path` does not point to an existing file.
    ValueError
        If `pdb_path` does not have a `.pdb` extension.
    """
    if not pdb_path.exists():
        raise FileNotFoundError(f"File not found: {pdb_path}")

    if pdb_path.suffix.lower() not in {".pdb"}:
        raise ValueError(f"Not a PDB file: {pdb_path}")

    structure = load_pdb_structure(pdb_path)
    text = edges_text(structure, cutoff)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (pdb_path.with_suffix("").name + "_edge.txt")
    out_file.write_text(text)
    return out_file


# New: handle a single PDB file as input
def generate_edges_from_pdb_file(
    pdb_file: Path, output_dir: Path, cfg: EdgeGenConfig
) -> Path:
    """
    Generate and write graph edges derived from a PDB structure to the given directory.

    Args:
        pdb_file (Path): Path to the input PDB file describing the molecular structure.
        output_dir (Path): Directory where the generated edge file will be stored.
        cfg (EdgeGenConfig): Configuration containing edge generation parameters such as distance cutoff.

    Returns:
        Path: Path to the generated edge file.

    Raises:
        FileNotFoundError: If the specified PDB file does not exist.
        ValueError: If the provided file is not a PDB file.
    """

    if not pdb_file.exists():
        raise FileNotFoundError(f"File not found: {pdb_file}")

    if pdb_file.suffix.lower() not in {".pdb"}:
        raise ValueError(f"Not a PDB file: {pdb_file}")

    output_dir.mkdir(parents=True, exist_ok=True)
    return write_edges_file_for_pdb(pdb_file, output_dir, cfg.cutoff)


def generate_edges_from_pdb_dir(
    pdb_dir: Path, output_dir: Path, cfg: EdgeGenConfig
) -> list[Path]:
    """Generate edge files for all PDBs within a directory.

    This function scans the provided PDB directory for files matching the patterns
    defined in the configuration, creates the output directory if necessary, and
    writes edge files in parallel using a process pool executor.

    Parameters
    ----------
    pdb_dir : Path
        Root directory containing input PDB files to process.
    output_dir : Path
        Directory where the generated edge files will be stored.
    cfg : EdgeGenConfig
        Configuration object specifying filename patterns and cutoff distance.

    Returns
    -------
    list[Path]
        A list of paths to the generated edge files.
    """
    pdbs = io.iter_target_pdbs(pdb_dir, cfg.patterns)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with ProcessPoolExecutor() as ex:
        for p in ex.map(
            lambda pth: write_edges_file_for_pdb(pth, output_dir, cfg.cutoff), pdbs
        ):
            paths.append(p)
    return paths


def generate_edges_from_tar(
    tar_file: Path, output_base_dir: Path, cfg: EdgeGenConfig
) -> Path:
    """Generate edge files from a compressed PDB tarball.

    Args:
        tar_file: Path to the input .tar.gz file containing PDB structures.
        output_base_dir: Directory where the generated edge data (tarball and optional expanded files) will be stored.
        cfg: Edge generator configuration specifying processing options, including whether to keep intermediate expanded data.

    Returns:
        Path to the resulting .tar.gz archive containing the generated edge data.
    """
    base = tar_file.with_suffix("").with_suffix("").name
    pdb_dir = io.safe_extract_tar_gz(tar_file, io.tmp_root() / base)
    out_dir = output_base_dir / f"{base}_edges"
    _ = generate_edges_from_pdb_dir(pdb_dir, out_dir, cfg)
    out_tar = out_dir.with_suffix(".tar.gz")
    io.make_archive(out_dir, out_tar)
    if not cfg.keep_expanded:
        io.cleanup(out_dir)
    io.cleanup(pdb_dir)
    return out_tar


def generate_edges_from_tar_dir(
    tar_dir: Path, output_base_dir: Path, cfg: EdgeGenConfig
) -> list[Path]:
    """Generate edge files for all `.tar.gz` archives within a directory.

    Args:
        tar_dir (Path): Directory containing the source archive files.
        output_base_dir (Path): Root directory where generated edge files are saved.
        cfg (EdgeGenConfig): Configuration options for the edge generation process.

    Returns:
        list[Path]: Paths to the generated edge files corresponding to each archive.
    """
    tars = sorted(tar_dir.glob("*.tar.gz"))
    return [generate_edges_from_tar(t, output_base_dir, cfg) for t in tars]
