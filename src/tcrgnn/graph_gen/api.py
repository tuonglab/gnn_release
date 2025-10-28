from pathlib import Path
import os, tarfile, shutil, logging, tempfile

LOG = logging.getLogger(__name__)

def is_tar_gz(p: Path) -> bool:
    return p.suffixes[-2:] == [".tar", ".gz"]

def safe_extract_tar_gz(tar_path: Path, dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        for m in tar.getmembers():
            target = dest / m.name
            if not target.resolve().is_relative_to(dest.resolve()):
                raise RuntimeError(f"Blocked path traversal: {m.name}")
        tar.extractall(path=dest)
    return dest

def list_edge_txts(root: Path) -> list[Path]:
    return sorted([p for p in root.rglob("*.txt")])

def temp_workspace(prefix: str = "edge_") -> Path:
    return Path(tempfile.mkdtemp(prefix=prefix, dir=os.getenv("TMPDIR", None)))

def cleanup(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)
