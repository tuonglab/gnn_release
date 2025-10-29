# io.py
import logging
import os
import shutil
import subprocess
import tarfile
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

LOG = logging.getLogger("tcrgnn.edgegen.io")


def is_within_directory(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def safe_extract_tar_gz(tar_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            if not is_within_directory(dest, dest / m.name):
                raise RuntimeError(f"Blocked path traversal: {m.name}")
        tar.extractall(path=dest)
    return dest


def make_archive(src_dir: Path, out_tar_gz: Path) -> None:
    subprocess.run(
        ["tar", "-czf", str(out_tar_gz), "-C", str(src_dir), "."], check=True
    )


def tmp_root() -> Path:
    return Path(os.getenv("TMPDIR", "/tmp")) / "dm"


def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def iter_target_pdbs(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    return [p for p in root.rglob("*.pdb") if any(pat in p.name for pat in patterns)]


def load_pdb_structure(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        return structure
    except PDBConstructionException as e:
        logging.error(f"Error parsing PDB file {pdb_path}: {e}")
        return None
