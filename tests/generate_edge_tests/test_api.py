import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from tcrgnn.edge_gen import api
from tcrgnn.edge_gen.api import generate_edges_from_pdb_file


def test_write_edges_file_for_pdb_success(tmp_path, monkeypatch):
    pdb_file = tmp_path / "sample.pdb"
    pdb_file.write_text("MODEL")
    output_dir = tmp_path / "out"
    monkeypatch.setattr(api, "load_pdb_structure", lambda path: "structure")
    monkeypatch.setattr(api, "edges_text", lambda structure, cutoff: f"text:{cutoff}")
    result = api.write_edges_file_for_pdb(pdb_file, output_dir, cutoff=5.0)
    assert result == output_dir / "sample_edge.txt"
    assert result.read_text() == "text:5.0"


def test_write_edges_file_for_pdb_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        api.write_edges_file_for_pdb(tmp_path / "missing.pdb", tmp_path / "out", 8.0)


def test_write_edges_file_for_pdb_invalid_extension(tmp_path):
    non_pdb = tmp_path / "sample.txt"
    non_pdb.write_text("data")
    with pytest.raises(ValueError):
        api.write_edges_file_for_pdb(non_pdb, tmp_path / "out", 8.0)


def test_generate_edges_from_pdb_file_invokes_writer(tmp_path, monkeypatch):
    pdb_file = tmp_path / "sample.pdb"
    pdb_file.write_text("MODEL")
    output_dir = tmp_path / "edges"
    captured = {}

    def fake_write(path, out_dir, cutoff):
        captured["args"] = (path, out_dir, cutoff)
        produced = out_dir / "out.txt"
        produced.write_text("edges")
        return produced

    monkeypatch.setattr(api, "write_edges_file_for_pdb", fake_write)
    cfg = api.EdgeGenConfig(cutoff=10.5)
    result = api.generate_edges_from_pdb_file(pdb_file, output_dir, cfg)
    assert result == output_dir / "out.txt"
    assert captured["args"] == (pdb_file, output_dir, 10.5)
    assert output_dir.exists()


def test_generate_edges_from_pdb_dir_uses_executor(tmp_path, monkeypatch):
    pdb_dir = tmp_path / "pdbs"
    pdb_dir.mkdir()
    output_dir = tmp_path / "edges"
    pdb_paths = [pdb_dir / "one.pdb", pdb_dir / "two.pdb"]
    fake_io = types.SimpleNamespace(iter_target_pdbs=lambda _, __: list(pdb_paths))
    monkeypatch.setattr(api, "io", fake_io)

    class DummyExecutor:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def map(self, func, iterable):
            for item in iterable:
                yield func(item)

    monkeypatch.setattr(api, "ProcessPoolExecutor", lambda: DummyExecutor())

    def fake_writer(pdb_path, out_dir, cutoff):
        produced = out_dir / f"{pdb_path.stem}_edges.txt"
        produced.write_text(pdb_path.stem)
        return produced

    monkeypatch.setattr(api, "write_edges_file_for_pdb", fake_writer)
    cfg = api.EdgeGenConfig(cutoff=9.0)
    result = api.generate_edges_from_pdb_dir(pdb_dir, output_dir, cfg)
    expected = [output_dir / "one_edges.txt", output_dir / "two_edges.txt"]
    assert result == expected
    assert output_dir.exists()


def test_generate_edges_from_tar_creates_archive(tmp_path, monkeypatch):
    tar_file = tmp_path / "bundle.tar.gz"
    tar_file.write_text("data")
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    tmp_root_dir = tmp_path / "root"
    tmp_root_dir.mkdir()
    out_base_dir = tmp_path / "out"
    record = {"archive": None, "cleanup": []}

    fake_io = types.SimpleNamespace(
        safe_extract_tar_gz=lambda tar, dest: extracted_dir,
        tmp_root=lambda: tmp_root_dir,
        make_archive=lambda src, dest: record.__setitem__("archive", (src, dest)),
        cleanup=lambda path: record["cleanup"].append(path),
    )
    monkeypatch.setattr(api, "io", fake_io)

    def fake_gen(pdb_dir, out_dir, cfg):
        out_dir.mkdir(parents=True, exist_ok=True)
        produced = out_dir / "edges.txt"
        produced.write_text("edges")
        return [produced]

    monkeypatch.setattr(api, "generate_edges_from_pdb_dir", fake_gen)
    cfg = api.EdgeGenConfig(keep_expanded=False)
    result = api.generate_edges_from_tar(tar_file, out_base_dir, cfg)
    expected_out_dir = out_base_dir / "bundle_edges"
    expected_tar = expected_out_dir.with_suffix(".tar.gz")
    assert result == expected_tar
    assert record["archive"] == (expected_out_dir, expected_tar)
    assert record["cleanup"] == [expected_out_dir, extracted_dir]


def test_generate_edges_from_tar_keep_expanded(tmp_path, monkeypatch):
    tar_file = tmp_path / "bundle.tar.gz"
    tar_file.write_text("data")
    extracted_dir = tmp_path / "extracted"
    extracted_dir.mkdir()
    tmp_root_dir = tmp_path / "root"
    tmp_root_dir.mkdir()
    out_base_dir = tmp_path / "out"
    record = {"cleanup": []}

    fake_io = types.SimpleNamespace(
        safe_extract_tar_gz=lambda tar, dest: extracted_dir,
        tmp_root=lambda: tmp_root_dir,
        make_archive=lambda src, dest: None,
        cleanup=lambda path: record["cleanup"].append(path),
    )
    monkeypatch.setattr(api, "io", fake_io)
    monkeypatch.setattr(
        api, "generate_edges_from_pdb_dir", lambda pdb_dir, out_dir, cfg: []
    )
    cfg = api.EdgeGenConfig(keep_expanded=True)
    api.generate_edges_from_tar(tar_file, out_base_dir, cfg)
    assert record["cleanup"] == [extracted_dir]


def test_generate_edges_from_tar_dir(tmp_path, monkeypatch):
    tar_dir = tmp_path / "tarballs"
    tar_dir.mkdir()
    output_base_dir = tmp_path / "out"
    files = []
    for name in ["b.tar.gz", "a.tar.gz"]:
        path = tar_dir / name
        path.write_text("data")
        files.append(path)
    cfg = api.EdgeGenConfig()
    calls = []

    def fake_generate(tar_path, out_dir, config):
        calls.append((tar_path, out_dir, config))
        result = out_dir / f"{tar_path.stem}_edges.tar.gz"
        result.parent.mkdir(parents=True, exist_ok=True)
        return result

    monkeypatch.setattr(api, "generate_edges_from_tar", fake_generate)
    results = api.generate_edges_from_tar_dir(tar_dir, output_base_dir, cfg)
    sorted_files = sorted(files, key=lambda p: p.name)
    assert [call[0] for call in calls] == sorted_files
    assert all(call[1] is output_base_dir for call in calls)
    assert all(call[2] is cfg for call in calls)
    expected_results = [
        output_base_dir / f"{p.stem}_edges.tar.gz" for p in sorted_files
    ]
    assert results == expected_results


def test_generate_edges_from_pdb_file_raises_file_not_found(tmp_path: Path):
    nonexistent = tmp_path / "missing.pdb"
    cfg = SimpleNamespace(cutoff=5.0)

    with pytest.raises(FileNotFoundError) as exc:
        generate_edges_from_pdb_file(nonexistent, tmp_path, cfg)

    assert "File not found" in str(exc.value)


def test_generate_edges_from_pdb_file_raises_value_error_for_non_pdb(tmp_path: Path):
    not_pdb = tmp_path / "structure.txt"
    not_pdb.write_text("dummy")
    cfg = SimpleNamespace(cutoff=5.0)

    with pytest.raises(ValueError) as exc:
        generate_edges_from_pdb_file(not_pdb, tmp_path, cfg)

    assert "Not a PDB file" in str(exc.value)
