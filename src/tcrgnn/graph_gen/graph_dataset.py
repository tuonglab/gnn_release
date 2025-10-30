from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from torch_geometric.data import Dataset

from tcrgnn.graph_gen._io import list_edge_txts
from tcrgnn.utils._common_utils import cleanup, safe_extract_tar_gz, tmp_root

from .build_graph import CANCEROUS, CONTROL, build_graph_from_edge_txt


class MultiGraphDataset(Dataset):
    """
    A dataset for loading and processing sets of graph samples from edge list files.

    This dataset consumes raw files that describe edges for one or more graphs.
    Raw inputs may be provided either as individual text files or as compressed
    `.tar.gz` archives containing multiple edge list files. Each raw file is
    converted into one or more PyTorch Geometric data objects using a helper
    function that constructs graph structures with PCA encoded amino acid
    features.

    Parameters
    ----------
    root : str or pathlib.Path
        Root directory used by PyTorch Geometric for raw and processed data
        management.
    samples : Sequence[str or pathlib.Path]
        Paths to raw edge text files or `.tar.gz` archives that contain edge
        text files.
    cancer : bool, optional
        If True, the generated graphs are labeled as cancerous. Defaults to False.
    pca_path : str or pathlib.Path
        Path to a PCA encoding table used to embed amino acids.
    aa_map : dict[str, str]
        Mapping from three letter amino acid codes to single letter codes.
    transform : callable, optional
        Optional transform applied on each sample when loaded.
    pre_transform : callable, optional
        Optional preprocessing transform applied before saving processed files.
    """

    def __init__(
        self,
        root: str | Path,
        samples: Sequence[str | Path],
        *,
        cancer: bool = False,
        pca_path: str | Path,
        aa_map: dict[str, str],
        transform=None,
        pre_transform=None,
    ):
        self.root = str(root)
        self.samples = [str(s) for s in samples]
        self.cancer = cancer
        self.pca_path = str(pca_path)
        self.aa_map = aa_map
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        """
        List of expected raw filenames.

        These correspond to the names of input edge text files or compressed
        archives without paths.

        Returns
        -------
        list[str]
            Raw filenames expected by PyTorch Geometric.
        """
        return [Path(s).name for s in self.samples]

    @property
    def processed_file_names(self):
        """
        List of expected processed filenames.

        Each raw file results in a corresponding `.pt` file. Compressed archives
        are normalized by removing both `.gz` and `.tar` suffixes.

        Returns
        -------
        list[str]
            Filenames of processed graph objects.
        """
        return [
            Path(s).with_suffix("").with_suffix("").name + ".pt"
            if str(s).endswith(".tar.gz")
            else Path(s).with_suffix(".pt").name
            for s in self.samples
        ]

    def process(self):
        """
        Process raw edge text files into PyTorch Geometric graph objects.

        For each input sample, this method:
        1. Extracts edge list text files if the sample is a `.tar.gz` archive.
        2. Loads PCA encoding features for amino acids.
        3. Builds graph objects by parsing edges and applying amino acid encodings.
        4. Saves the resulting graph objects to `.pt` files under the
           `processed` directory.

        Temporary directories created during extraction are cleaned up
        regardless of success or error.
        """
        from .encodings import load_pca_encoding

        processed_dir = Path(self.root) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        pca = load_pca_encoding(self.pca_path)
        label = CANCEROUS if self.cancer else CONTROL

        for raw_path in self.raw_paths:
            raw = Path(raw_path)
            out_pt = processed_dir / (
                raw.with_suffix("").with_suffix("").name + ".pt"
                if str(raw).endswith(".tar.gz")
                else raw.with_suffix(".pt").name
            )

            work = tmp_root() / "edge_"
            try:
                edge_files: list[Path] = []
                if str(raw).endswith(".tar.gz"):
                    extracted = safe_extract_tar_gz(raw, work / "extract")
                    edge_files = list_edge_txts(extracted)
                else:
                    edge_files = [raw]

                objs = [
                    build_graph_from_edge_txt(p, pca, self.aa_map, label=label)
                    for p in edge_files
                ]
                torch.save(objs, out_pt)
            finally:
                cleanup(work)

    def len(self):
        """
        Return the number of processed graph files in the dataset.

        Returns
        -------
        int
            Number of processed items.
        """
        return len(self.processed_paths)

    def get(self, idx):
        """
        Load and return the processed graph objects at the given index.

        Parameters
        ----------
        idx : int
            Index of the processed file to load.

        Returns
        -------
        list[torch_geometric.data.Data]
            A list of graph objects created from the original edge files.
        """
        return torch.load(self.processed_paths[idx])
