import io as _io
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from tcrgnn.edge_gen import _io as io_mod


def test_is_within_directory_true_and_false(tmp_path):
    base = tmp_path / "base"
    child = base / "dir" / "file.txt"
    child.parent.mkdir(parents=True)
    child.write_text("x")
    outside = tmp_path.parent / "other" / "file.txt"
    outside.parent.mkdir(parents=True)
    outside.write_text("y")

    assert io_mod.is_within_directory(base, child)
    assert not io_mod.is_within_directory(base, outside)


def test_safe_extract_tar_gz_allows_safe_and_blocks_traversal(tmp_path):
    # safe tar
    src = tmp_path / "src"
    src.mkdir()
    (src / "safe.txt").write_text("safe")
    safe_tar = tmp_path / "safe.tar.gz"
    with tarfile.open(safe_tar, "w:gz") as t:
        t.add(src / "safe.txt", arcname="safe.txt")

    dest = tmp_path / "dest_safe"
    returned = io_mod.safe_extract_tar_gz(safe_tar, dest)
    assert returned == dest
    assert (dest / "safe.txt").read_text() == "safe"

    # traversal tar
    trav_tar = tmp_path / "trav.tar.gz"
    with tarfile.open(trav_tar, "w:gz") as t:
        ti = tarfile.TarInfo(name="../evil.txt")
        data = b"evil"
        ti.size = len(data)
        t.addfile(ti, _io.BytesIO(data))

    dest2 = tmp_path / "dest_trav"
    with pytest.raises(RuntimeError) as exc:
        io_mod.safe_extract_tar_gz(trav_tar, dest2)
    assert "Blocked path traversal" in str(exc.value)


def test_make_archive_calls_subprocess_run(tmp_path, monkeypatch):
    called = {}

    def fake_run(args, check):
        called["args"] = args
        called["check"] = check

    monkeypatch.setattr(io_mod, "subprocess", SimpleNamespace(run=fake_run))
    src = tmp_path / "sourcedir"
    src.mkdir()
    (src / "f.txt").write_text("x")
    out = tmp_path / "out.tar.gz"
    io_mod.make_archive(src, out)
    assert called["args"] == ["tar", "-czf", str(out), "-C", str(src), "."]
    assert called["check"] is True


def test_tmp_root_returns_subdirectory_of_system_temp():
    path = io_mod.tmp_root()

    # It should be a Path object
    assert isinstance(path, Path)

    # It should end with the directory name "dm"
    assert path.name == "dm"

    # Its parent should be the system temp directory
    import tempfile

    assert path.parent == Path(tempfile.gettempdir())


def test_cleanup_uses_shutil_rmtree(monkeypatch, tmp_path):
    called = {}

    def fake_rmtree(path, ignore_errors):
        called["path"] = path
        called["ignore_errors"] = ignore_errors

    monkeypatch.setattr(io_mod, "shutil", SimpleNamespace(rmtree=fake_rmtree))
    p = tmp_path / "to_remove"
    io_mod.cleanup(p)
    assert called["path"] == p
    assert called["ignore_errors"] is True


def test_iter_target_pdbs_filters_by_token_patterns(tmp_path):
    root = tmp_path / "root"
    root.mkdir()

    # Files that should match
    (root / "protein_rank_001.pdb").write_text("ok")
    (root / "model_0_complex.pdb").write_text("ok")
    (root / "foo_model_0_bar.pdb").write_text("ok")

    # Files that should not match
    (root / "nomodel.pdb").write_text("no")
    (root / "rankx001.pdb").write_text("no")
    (root / "random.pdb").write_text("no")

    patterns = ("rank_001", "model_0")

    results = io_mod.iter_target_pdbs(root, patterns)
    names = {p.name for p in results}

    assert names == {
        "protein_rank_001.pdb",
        "model_0_complex.pdb",
        "foo_model_0_bar.pdb",
    }


def test_load_pdb_structure_success_and_failure(monkeypatch, tmp_path, caplog):
    pdb_path = tmp_path / "1abc.pdb"
    pdb_path.write_text("dummy")

    # Success case: PDBParser returns object
    class DummyParserGood:
        def __init__(self, QUIET=True):
            self.quiet = QUIET

        def get_structure(self, id, path):
            return {"id": id, "path": path}

    monkeypatch.setattr(io_mod, "PDBParser", DummyParserGood)
    caplog.clear()
    struct = io_mod.load_pdb_structure(pdb_path)
    assert isinstance(struct, dict)
    assert struct["id"] == pdb_path.stem
    assert struct["path"] == str(pdb_path)

    # Failure case: parser raises PDBConstructionException
    class DummyParserBad:
        def __init__(self, QUIET=True):
            pass

        def get_structure(self, id, path):
            raise io_mod.PDBConstructionException("bad pdb")

    monkeypatch.setattr(io_mod, "PDBParser", DummyParserBad)
    caplog.clear()
    struct2 = io_mod.load_pdb_structure(pdb_path)
    assert struct2 is None
    # ensure error was logged
    assert any("Error parsing PDB file" in rec.getMessage() for rec in caplog.records)
