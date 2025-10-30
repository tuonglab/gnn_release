# io.py
import logging
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

LOG = logging.getLogger("tcrgnn.edgegen.io")


def is_within_directory(base: Path, target: Path) -> bool:
    """
    Check whether a target path is within a base directory.

    Args:
        base: Base directory path.
        target: Path to test.

    Returns:
        True if target is inside base, otherwise False.
    """
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_tar_gz(tar_path: Path, dest: Path) -> Path:
    """
    Safely extract a tar.gz archive into a destination directory.

    Validates members to prevent path traversal attacks.

    Args:
        tar_path: Path to a .tar.gz archive.
        dest: Destination directory to extract into.

    Returns:
        The destination directory path.

    Raises:
        RuntimeError: If a path traversal attempt is detected.
        tarfile.TarError: If archive cannot be opened or read.
    """
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            if not is_within_directory(dest, dest / m.name):
                raise RuntimeError(f"Blocked path traversal: {m.name}")
        tar.extractall(path=dest)
    return dest


def make_archive(src_dir: Path, out_tar_gz: Path) -> None:
    """
    Create a tar.gz archive from a source directory using system tar.

    Args:
        src_dir: Directory to package.
        out_tar_gz: Output archive path.

    Raises:
        subprocess.CalledProcessError: If tar fails.
    """
    subprocess.run(
        ["tar", "-czf", str(out_tar_gz), "-C", str(src_dir), "."], check=True
    )


def tmp_root() -> Path:
    """
    Return a temporary workspace directory for intermediate files.

    Returns:
        Path object within system temp named dm.
    """
    return Path(tempfile.gettempdir()) / "dm"


def cleanup(path: Path) -> None:
    """
    Remove directories or files recursively, ignoring errors.

    Args:
        path: Directory or file to remove.
    """
    shutil.rmtree(path, ignore_errors=True)


def iter_target_pdbs(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    """
    Recursively discover PDB files under a root directory matching patterns.

    Args:
        root: Directory to search.
        patterns: Tuple of substrings that must appear in the filename.

    Returns:
        List of matching PDB paths.
    """
    return [p for p in root.rglob("*.pdb") if any(pat in p.name for pat in patterns)]


def load_pdb_structure(pdb_path: Path):
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
