import pandas as pd
from pandas.testing import assert_frame_equal

from tcrgnn.graph_gen._encodings import load_pca_encoding


def test_load_pca_encoding_preserves_single_letter_index(tmp_path):
    df = pd.DataFrame(
        [[0.1, 0.2], [0.3, 0.4]], index=["A", "C"], columns=["PC1", "PC2"]
    )
    path = tmp_path / "pca.tsv"
    df.to_csv(path, sep="\t")
    loaded = load_pca_encoding(str(path))
    assert_frame_equal(loaded, df)


def test_load_pca_encoding_maps_three_letter_codes(tmp_path):
    df = pd.DataFrame(
        [[0.5, 0.6], [0.7, 0.8], [0.9, 1.0]],
        index=["ALA", "CYS", "UNK"],
        columns=["PC1", "PC2"],
    )
    path = tmp_path / "pca_three.tsv"
    df.to_csv(path, sep="\t")
    loaded = load_pca_encoding(path)
    expected = df.rename(index={"ALA": "A", "CYS": "C"})
    assert_frame_equal(loaded, expected)
