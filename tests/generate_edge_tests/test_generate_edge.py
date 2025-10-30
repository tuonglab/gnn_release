import logging
import math
from pathlib import Path
from unittest import mock

import tcrgnn.edge_gen.generate_edge as generate_edge


def test_load_pdb_structure_success(monkeypatch):
    # Arrange: create a fake parser instance whose get_structure returns a sentinel
    sentinel = object()
    fake_parser_instance = mock.Mock()
    fake_parser_instance.get_structure.return_value = sentinel

    fake_PDBParser = mock.Mock(return_value=fake_parser_instance)
    monkeypatch.setattr(generate_edge, "PDBParser", fake_PDBParser)

    pdb_path = Path("/tmp/example.pdb")

    # Act
    result = generate_edge.load_pdb_structure(pdb_path)

    # Assert
    assert result is sentinel
    fake_PDBParser.assert_called_once_with(QUIET=True)
    fake_parser_instance.get_structure.assert_called_once_with(
        pdb_path.stem, str(pdb_path)
    )


def test_load_pdb_structure_failure_logs_and_returns_none(monkeypatch, caplog):
    # Arrange: parser.get_structure raises PDBConstructionException
    fake_parser_instance = mock.Mock()
    fake_parser_instance.get_structure.side_effect = (
        generate_edge.PDBConstructionException("parse failure")
    )

    fake_PDBParser = mock.Mock(return_value=fake_parser_instance)
    monkeypatch.setattr(generate_edge, "PDBParser", fake_PDBParser)

    pdb_path = Path("/tmp/bad.pdb")
    caplog.set_level(logging.ERROR)

    # Act
    result = generate_edge.load_pdb_structure(pdb_path)

    # Assert
    assert result is None
    # Ensure an error was logged mentioning the pdb path
    assert any(
        rec.levelno == logging.ERROR and str(pdb_path) in rec.getMessage()
        for rec in caplog.records
    )
    fake_parser_instance.get_structure.assert_called_once_with(
        pdb_path.stem, str(pdb_path)
    )


# ----- Minimal fake Bio.PDB-like objects -----


class FakeAtom:
    def __init__(self, coord):
        self.coord = tuple(coord)

    # Bio.PDB Atom subtraction returns Euclidean distance
    def __sub__(self, other):
        x1, y1, z1 = self.coord
        x2, y2, z2 = other.coord
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


class FakeResidue:
    def __init__(self, resname, seq_id, ca=None, cb=None):
        self._resname = resname
        # Mimic Bio.PDB style: (' ', id, ' ')
        self.id = (" ", int(seq_id), " ")
        self._atoms = {}
        if ca is not None:
            self._atoms["CA"] = FakeAtom(ca)
        if cb is not None:
            self._atoms["CB"] = FakeAtom(cb)

    def get_resname(self):
        return self._resname

    def has_id(self, atom_name):
        return atom_name in self._atoms

    def __getitem__(self, atom_name):
        return self._atoms[atom_name]


class FakeChain:
    def __init__(self, residues):
        self._residues = list(residues)

    def __iter__(self):
        return iter(self._residues)


class FakeModel:
    def __init__(self, chains):
        self._chains = list(chains)

    def __iter__(self):
        return iter(self._chains)


class FakeStructure:
    def __init__(self, models):
        self._models = list(models)

    def __iter__(self):
        return iter(self._models)


# ----- Import the function under test -----
# Adjust the import path to where your function lives
# from your_module import residue_pairs_within_cutoff

# For illustration in this snippet, we'll assume it's already in scope.


def test_basic_same_chain_and_inclusive_cutoff():
    # Two residues in same chain: distance exactly at cutoff should be included
    r1 = FakeResidue("ALA", 1, ca=(0, 0, 0), cb=(1, 1, 1))
    r2 = FakeResidue("GLY", 2, ca=(0, 0, 8.0))  # no CB, will fallback to CA
    chain = FakeChain([r1, r2])
    model = FakeModel([chain])
    structure = FakeStructure([model])

    pairs = list(generate_edge.residue_pairs_within_cutoff(structure, cutoff=8.0))
    assert pairs == [("ALA", 1, "GLY", 2)]


def test_prefers_cb_over_ca():
    # If CB exists, it should be used, even if CA would have passed the cutoff
    # CA distance = 7.0 (within), CB distance = 9.0 (outside) -> should EXCLUDE
    r1 = FakeResidue("SER", 10, ca=(0, 0, 0), cb=(0, 0, 0))
    r2 = FakeResidue("LEU", 11, ca=(0, 0, 7.0), cb=(9.0, 0, 0))
    chain = FakeChain([r1, r2])
    model = FakeModel([chain])
    structure = FakeStructure([model])

    pairs = list(generate_edge.residue_pairs_within_cutoff(structure, cutoff=8.0))
    assert pairs == [], "CB must be preferred over CA when present"


def test_ca_fallback_when_no_cb():
    # Neither residue has CB, so CA is used and pair should be included
    r1 = FakeResidue("GLY", 5, ca=(0, 0, 0))
    r2 = FakeResidue("ALA", 6, ca=(0, 0, 6.5))
    chain = FakeChain([r1, r2])
    model = FakeModel([chain])
    structure = FakeStructure([model])

    pairs = list(generate_edge.residue_pairs_within_cutoff(structure, cutoff=8.0))
    assert pairs == [("GLY", 5, "ALA", 6)]


def test_residues_without_ca_are_ignored():
    # One residue lacks CA and should be filtered out from consideration
    r1 = FakeResidue("ALA", 1, ca=(0, 0, 0))
    r2 = FakeResidue("VAL", 2, ca=None, cb=(0, 0, 1))  # ignored entirely
    r3 = FakeResidue("GLY", 3, ca=(0, 0, 7.9))
    chain = FakeChain([r1, r2, r3])
    model = FakeModel([chain])
    structure = FakeStructure([model])

    pairs = list(generate_edge.residue_pairs_within_cutoff(structure, cutoff=8.0))
    # Only pairs formed from residues with CA: (1,3) is within cutoff
    assert pairs == [("ALA", 1, "GLY", 3)]


def test_different_chains_not_compared():
    # Same sequence positions but different chains should not be paired
    a1 = FakeResidue("ALA", 1, ca=(0, 0, 0))
    a2 = FakeResidue("GLY", 2, ca=(0, 0, 7.0))
    b1 = FakeResidue("ALA", 1, ca=(0, 0, 0))
    b2 = FakeResidue("GLY", 2, ca=(0, 0, 7.0))
    chain_a = FakeChain([a1, a2])
    chain_b = FakeChain([b1, b2])
    model = FakeModel([chain_a, chain_b])
    structure = FakeStructure([model])

    pairs = list(generate_edge.residue_pairs_within_cutoff(structure, cutoff=8.0))
    # Expect two pairs, one from each chain, but no cross-chain combos
    assert ("ALA", 1, "GLY", 2) in pairs and len(pairs) == 2


def test_multiple_models_are_independent():
    # Each model should be traversed independently
    r1 = FakeResidue("ALA", 1, ca=(0, 0, 0))
    r2 = FakeResidue("GLY", 2, ca=(0, 0, 7.5))
    chain_m1 = FakeChain([r1, r2])

    r3 = FakeResidue("THR", 3, ca=(10, 0, 0))
    r4 = FakeResidue("LYS", 4, ca=(10, 0, 7.5))
    chain_m2 = FakeChain([r3, r4])

    model1 = FakeModel([chain_m1])
    model2 = FakeModel([chain_m2])
    structure = FakeStructure([model1, model2])

    pairs = list(generate_edge.residue_pairs_within_cutoff(structure, cutoff=8.0))
    assert ("ALA", 1, "GLY", 2) in pairs
    assert ("THR", 3, "LYS", 4) in pairs
    assert len(pairs) == 2


def test_alternate_parser_basic(tmp_path):
    # Create a tiny PDB file locally
    pdb_text = """\
ATOM      1  CB  ALA A   1       0.000   0.000   0.000
ATOM      2  CA  GLY A   2       0.000   0.000   7.900
ATOM      3  CB  LYS A   3       0.000   0.000  20.000
ATOM      4  CB  THR A   4       0.000   0.000   8.100
"""

    pdb_file = tmp_path / "small.pdb"
    pdb_file.write_text(pdb_text)

    # Create an output directory
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    # Import here or adjust if already imported
    generate_edge.alternate_parser(str(pdb_file), str(out_dir))

    # Check output
    out_file = out_dir / "small_edge.txt"
    assert out_file.exists()

    # Read edges
    lines = [ln.strip() for ln in out_file.read_text().splitlines() if ln.strip()]

    assert lines == ["ALA 1 GLY 2", "GLY 2 THR 4"]


def test_edges_text_basic():
    # Build minimal fake structure
    r1 = FakeResidue("ALA", 1, ca=(0, 0, 0))
    r2 = FakeResidue("GLY", 2, ca=(0, 0, 7.5))
    r3 = FakeResidue("THR", 3, ca=(0, 0, 20.0))  # too far from both
    chain = FakeChain([r1, r2, r3])
    model = FakeModel([chain])
    structure = FakeStructure([model])

    # Call
    txt = generate_edge.edges_text(structure, cutoff=8.0)

    # Only ALA1 GLY2 should be listed
    assert txt == "ALA 1 GLY 2\n"


def test_edges_text_multiple_pairs_and_order():
    r1 = FakeResidue("ALA", 1, ca=(0, 0, 0))
    r2 = FakeResidue("GLY", 2, ca=(0, 0, 7.5))
    r3 = FakeResidue("SER", 3, ca=(0, 0, 7.9))
    chain = FakeChain([r1, r2, r3])
    model = FakeModel([chain])
    structure = FakeStructure([model])

    txt = generate_edge.edges_text(structure, cutoff=8.0)

    # Expect (1,2) then (1,3) then (2,3) based on pair order
    assert txt == ("ALA 1 GLY 2\nALA 1 SER 3\nGLY 2 SER 3\n")
