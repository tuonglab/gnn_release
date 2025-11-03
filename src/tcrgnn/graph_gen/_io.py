from pathlib import Path


def list_edge_txts(root: Path | str) -> list[Path]:
    """
    Return all .txt edge list files inside the given directory (non-recursive).

    Parameters
    ----------
    root : str or pathlib.Path
        Directory to scan.

    Returns
    -------
    list[pathlib.Path]
        Paths of all files ending with .txt.
    """
    root = Path(root)
    return [p for p in root.iterdir() if p.suffix == ".txt" and p.is_file()]


def parse_edges(edge_file: Path) -> list[list[str]]:
    with edge_file.open() as f:
        return [line.strip().split() for line in f if line.strip()]


def save_graphs_to_disk(graphs, file: Path) -> None:
    """
    Save a list of PyTorch Geometric Data objects to disk.

    Parameters
    ----------
    graphs : list[torch_geometric.data.Data]
        List of graph objects to save.
    file : pathlib.Path
        Path to the output file.
    """
    import torch

    torch.save(graphs, str(file))
