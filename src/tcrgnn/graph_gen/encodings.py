from pathlib import Path

import pandas as pd

AMINO_ACID_MAPPING = {
    "ALA": "A",
    "CYS": "C",
    "ASP": "D",
    "GLU": "E",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LYS": "K",
    "LEU": "L",
    "MET": "M",
    "ASN": "N",
    "PRO": "P",
    "GLN": "Q",
    "ARG": "R",
    "SER": "S",
    "THR": "T",
    "VAL": "V",
    "TRP": "W",
    "TYR": "Y",
}


def load_pca_encoding(path: str | Path) -> pd.DataFrame:
    """
    Load an amino acid PCA encoding table from a tab-separated file.

    This function reads a PCA table derived from AAindex data, where rows represent
    amino acids and columns represent principal components or feature dimensions.
    The index is expected to contain single-letter amino acid codes. If the file
    instead uses three-letter amino acid codes, they are automatically mapped to
    single letters once using the `AMINO_ACID_MAPPING` dictionary.

    Parameters
    ----------
    path : str or pathlib.Path
        The path to the tab-separated PCA table.

    Returns
    -------
    pandas.DataFrame
        A DataFrame where the index consists of single-letter amino acid codes and
        the columns contain PCA feature values.

    Notes
    -----
    This function assumes that the first column of the file can serve as the
    DataFrame index. Unknown three-letter codes are left unchanged.
    """
    df = pd.read_csv(path, sep="\t", index_col=0)
    # Expecting index to be single letter. If 3-letter, map once here.
    if len(df.index[0]) == 3:
        df = df.rename(index=lambda k: AMINO_ACID_MAPPING.get(k, k))
    return df
