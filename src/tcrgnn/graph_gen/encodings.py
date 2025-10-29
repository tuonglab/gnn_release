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
    """Load AAidx PCA table with single letter index."""
    df = pd.read_csv(path, sep="\t", index_col=0)
    # Expecting index to be single letter. If 3-letter, map once here.
    if len(df.index[0]) == 3:
        df = df.rename(index=lambda k: AMINO_ACID_MAPPING.get(k, k))
    return df
