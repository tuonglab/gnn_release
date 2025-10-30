from tcrgnn.edge_gen import _io as io_mod


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
