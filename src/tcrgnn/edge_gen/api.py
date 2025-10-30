import logging
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from . import _io as fio
from . import processing as proc

LOG = logging.getLogger("tcrgnn.edgegen")


@dataclass
class EdgeGenConfig:
    cutoff: float = 8.0
    patterns: tuple[str, ...] = ("rank_001", "model_0")
    keep_expanded: bool = False


def write_edges_file_for_pdb(pdb_path: Path, out_dir: Path, cutoff: float) -> Path:
    structure = proc.load_structure_sanitized(pdb_path)
    text = proc.edges_text(structure, cutoff)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (pdb_path.with_suffix("").name + "_edge.txt")
    out_file.write_text(text)
    return out_file


# New: handle a single PDB file as input
def generate_edges_from_pdb_file(
    pdb_file: Path, output_dir: Path, cfg: EdgeGenConfig
) -> Path:
    fio.validate_file_exists(pdb_file)
    output_dir.mkdir(parents=True, exist_ok=True)
    return write_edges_file_for_pdb(pdb_file, output_dir, cfg.cutoff)


# New: handle a directory of PDB files
def generate_edges_from_pdb_dir(
    pdb_dir: Path, output_dir: Path, cfg: EdgeGenConfig
) -> list[Path]:
    pdbs = fio.find_target_pdbs_in_dir(pdb_dir, cfg.patterns)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    with ProcessPoolExecutor() as ex:
        for p in ex.map(
            lambda pth: write_edges_file_for_pdb(pth, output_dir, cfg.cutoff), pdbs
        ):
            paths.append(p)
    return paths


# Existing: tar inputs
def generate_edges_from_tar(
    tar_file: Path, output_base_dir: Path, cfg: EdgeGenConfig
) -> Path:
    base = tar_file.with_suffix("").with_suffix("").name
    extract_dir = fio.safe_extract_tar_gz(tar_file, fio.tmp_root() / base)
    out_dir = output_base_dir / f"{base}_edges"
    _ = generate_edges_from_pdb_dir(extract_dir, out_dir, cfg)
    out_tar = out_dir.with_suffix(".tar.gz")
    fio.make_archive(out_dir, out_tar)
    if not cfg.keep_expanded:
        fio.cleanup(out_dir)
    fio.cleanup(extract_dir)
    return out_tar


def generate_edges_from_tar_dir(
    tar_dir: Path, output_base_dir: Path, cfg: EdgeGenConfig
) -> list[Path]:
    tars = sorted(tar_dir.glob("*.tar.gz"))
    return [generate_edges_from_tar(t, output_base_dir, cfg) for t in tars]
