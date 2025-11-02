from __future__ import annotations

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2(torch.nn.Module):
    """
    Two-layer GATv2 encoder with global mean pooling and a linear classifier.

    The model outputs unnormalized logits. A temperature scalar divides the
    logits for optional calibration.

    Args:
        nfeat: Input feature dimension per node.
        nhid: Hidden feature size for both GATv2 layers.
        nclass: Number of output classes.
        dropout: Dropout probability applied to node embeddings before pooling.
        temperature: Positive scalar to scale logits by 1 / temperature.
    """

    def __init__(
        self,
        nfeat: int,
        nhid: int,
        nclass: int,
        dropout: float,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")

        self.conv1 = GATv2Conv(
            in_channels=nfeat,
            out_channels=nhid,
            heads=16,
            dropout=dropout,
            concat=False,
        )
        self.conv2 = GATv2Conv(
            in_channels=nhid, out_channels=nhid, heads=16, dropout=dropout, concat=False
        )
        self.classifier = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.temperature = temperature

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features of shape [num_nodes, nfeat].
            edge_index: COO edge indices of shape [2, num_edges].
            batch: Graph assignment vector of shape [num_nodes], or None for a single graph.

        Returns:
            torch.Tensor: Logits of shape [num_graphs, nclass].
        """
        # Fallback to a single-graph batch if not provided
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x = x / self.temperature
        return x
