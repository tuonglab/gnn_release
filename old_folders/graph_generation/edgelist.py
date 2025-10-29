from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tarfile
import traceback
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from Bio.PDB import PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionException

LOG = logging.getLogger("tcrgnn.edgegen")


@dataclass
class EdgeGenConfig:
    cutoff: float = 8.0
    tmp_root: Path | None = None  # default to $TMPDIR or /tmp
    keep_outputs_expanded: bool = False
    patterns: tuple[str, ...] = ("rank_001", "model_0")  # select PDB files


def load_pdb_structure(pdb_path: Path):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_path.stem, str(pdb_path))
        return structure
    except PDBConstructionException as e:
        logging.error(f"Error parsing PDB file {pdb_path}: {e}")
        return None


def atom_for_distance(res):
    return res["CB"] if res.has_id("CB") else res["CA"]


def _is_within_directory(base: Path, target: Path) -> bool:
    try:
        target.resolve().relative_to(base.resolve())
        return True
    except ValueError:
        return False


def _safe_extract_tar_gz(tar_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            if not _is_within_directory(dest, dest / m.name):
                raise RuntimeError(f"Blocked path traversal: {m.name}")
        tar.extractall(path=dest)


def write_edges_for_pdb(pdb_file: Path, output_dir: Path, cutoff: float) -> None:
    try:
        structure = load_pdb_structure(pdb_file)
        out_name = pdb_file.with_suffix("").name + "_edge.txt"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / out_name

        with out_file.open("w") as f:
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
                                d = a1 - a2
                                if d <= cutoff:
                                    f.write(
                                        f"{r1.get_resname()} {r1.id[1]} {r2.get_resname()} {r2.id[1]}\n"
                                    )
                            except KeyError:
                                continue
    except Exception as e:
        if write_edges is not None:
            try:
                write_edges(str(pdb_file), str(output_dir))
                return
            except Exception as e2:
                LOG.error("Fallback also failed for %s: %s", pdb_file, e2)
        LOG.error("Error processing %s: %s", pdb_file, e)
        traceback.print_exc()


def _iter_target_pdbs(root: Path, patterns: tuple[str, ...]) -> Iterable[Path]:
    for p in root.rglob("*.pdb"):
        name = p.name
        if any(pat in name for pat in patterns):
            yield p


def generate_edges_from_tar(
    tar_file: Path,
    output_base_dir: Path,
    config: EdgeGenConfig = EdgeGenConfig(),
) -> Path:
    tmp_root = config.tmp_root or Path(os.getenv("TMPDIR", "/tmp")) / "dm"
    base_name = tar_file.with_suffix("").with_suffix("").name
    extract_dir = tmp_root / base_name
    extract_dir.parent.mkdir(parents=True, exist_ok=True)

    _safe_extract_tar_gz(tar_file, extract_dir)
    pdbs = list(_iter_target_pdbs(extract_dir, config.patterns))
    if not pdbs:
        LOG.warning("No target PDBs found in %s", tar_file)
        shutil.rmtree(extract_dir, ignore_errors=True)
        return output_base_dir / f"{base_name}_edges.tar.gz"

    out_dir = output_base_dir / f"{base_name}_edges"
    out_dir.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor() as ex:
        for _ in ex.map(lambda p: write_edges_for_pdb(p, out_dir, config.cutoff), pdbs):
            pass

    tar_output = out_dir.with_suffix(".tar.gz")
    subprocess.run(
        ["tar", "-czf", str(tar_output), "-C", str(out_dir), "."], check=True
    )

    if not config.keep_outputs_expanded:
        shutil.rmtree(out_dir, ignore_errors=True)
    shutil.rmtree(extract_dir, ignore_errors=True)

    return tar_output
