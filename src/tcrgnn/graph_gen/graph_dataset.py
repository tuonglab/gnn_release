from __future__ import annotations
from pathlib import Path
from typing import Sequence
import tarfile
import torch
from torch_geometric.data import Dataset
from .io import temp_workspace, safe_extract_tar_gz, list_edge_txts, cleanup
from .generate_graph import build_graph_from_edge_txt, CANCEROUS, CONTROL

class MultiGraphDataset(Dataset):
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
        return [Path(s).name for s in self.samples]

    @property
    def processed_file_names(self):
        # one .pt per raw tar
        return [Path(s).with_suffix("").with_suffix("").name + ".pt"
                if str(s).endswith(".tar.gz")
                else Path(s).with_suffix(".pt").name
                for s in self.samples]

    def process(self):
        import pandas as pd
        from .encodings import load_pca_encoding

        processed_dir = Path(self.root) / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        pca = load_pca_encoding(self.pca_path)
        label = CANCEROUS if self.cancer else CONTROL

        for raw_path in self.raw_paths:
            raw = Path(raw_path)
            out_pt = processed_dir / (raw.with_suffix("").with_suffix("").name + ".pt"
                                      if str(raw).endswith(".tar.gz")
                                      else raw.with_suffix(".pt").name)

            work = temp_workspace("edge_")
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
        return len(self.processed_paths)

    def get(self, idx):
        return torch.load(self.processed_paths[idx])
