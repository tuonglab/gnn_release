# test_io_edge_parsing.py
from pathlib import Path

import pytest

from tcrgnn.graph_gen._io import list_edge_txts, parse_edges


def write(p: Path, text: str = "") -> Path:
    p.write_text(text, encoding="utf-8")
    return p


def test_list_edge_txts_finds_only_txt_files_non_recursive(tmp_path: Path):
    # Files in root
    a = write(tmp_path / "a.txt", "A 1 B 2\n")
    b = write(tmp_path / "b.txt", "C 3 D 4\n")
    _ = write(tmp_path / "c.csv", "not used\n")
    _ = write(tmp_path / "d.txt.bak", "not used\n")  # suffix is .bak

    # Files in subdir should be ignored by non-recursive scan
    sub = tmp_path / "sub"
    sub.mkdir()
    _ = write(sub / "e.txt", "E 5 F 6\n")

    found = list_edge_txts(tmp_path)
    assert sorted(p.name for p in found) == sorted([a.name, b.name])
    # Ensure it is non-recursive
    assert "e.txt" not in {p.name for p in found}
    # Ensure non-.txt files are ignored
    assert "c.csv" not in {p.name for p in found}
    assert "d.txt.bak" not in {p.name for p in found}


def test_list_edge_txts_accepts_str_path(tmp_path: Path):
    write(tmp_path / "x.txt", "X 1 Y 2\n")
    # Pass str instead of Path
    found = list_edge_txts(str(tmp_path))
    assert [p.name for p in found] == ["x.txt"]


def test_list_edge_txts_is_case_sensitive_by_default(tmp_path: Path):
    # On most filesystems, suffix comparison is case sensitive
    write(tmp_path / "upper.TXT", "ignored\n")
    write(tmp_path / "lower.txt", "used\n")

    found = list_edge_txts(tmp_path)
    assert [p.name for p in found] == ["lower.txt"]


def test_parse_edges_basic_spaces_and_tabs(tmp_path: Path):
    content = "ALA 3 TYR 7\nGLY\t1\tSER\t2\n"
    f = write(tmp_path / "edges.txt", content)
    parsed = parse_edges(f)
    assert parsed == [
        ["ALA", "3", "TYR", "7"],
        ["GLY", "1", "SER", "2"],
    ]


def test_parse_edges_ignores_blank_and_whitespace_only_lines(tmp_path: Path):
    content = "\n  \t \nALA 3 TYR 7\n\nGLY 1 SER 2  \n"
    f = write(tmp_path / "edges.txt", content)
    parsed = parse_edges(f)
    assert parsed == [
        ["ALA", "3", "TYR", "7"],
        ["GLY", "1", "SER", "2"],
    ]


def test_parse_edges_trims_leading_and_trailing_whitespace(tmp_path: Path):
    content = "  ALA 3 TYR 7  \n\tGLY 1 SER 2\t\n"
    f = write(tmp_path / "edges.txt", content)
    parsed = parse_edges(f)
    assert parsed[0] == ["ALA", "3", "TYR", "7"]
    assert parsed[1] == ["GLY", "1", "SER", "2"]


def test_parse_edges_unicode_and_non_ascii_tokens(tmp_path: Path):
    # Ensure unicode is preserved
    content = "ÅLA 3 TYR 7\nGLY 1 SÉR 2\n"
    f = write(tmp_path / "edges.txt", content)
    parsed = parse_edges(f)
    assert parsed == [
        ["ÅLA", "3", "TYR", "7"],
        ["GLY", "1", "SÉR", "2"],
    ]


def test_parse_edges_empty_file_returns_empty_list(tmp_path: Path):
    f = write(tmp_path / "edges.txt", "")
    assert parse_edges(f) == []


def test_parse_edges_missing_file_raises(tmp_path: Path):
    missing = tmp_path / "nope.txt"
    with pytest.raises(FileNotFoundError):
        parse_edges(missing)
