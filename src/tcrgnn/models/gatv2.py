from __future__ import annotations
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool

class GATv2(torch.nn.Module):
    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float, temperature: float = 1.0) -> None:
        super().__init__()
        self.conv1 = GATv2Conv(nfeat, nhid, heads=16, dropout=dropout, concat=False)
        self.conv2 = GATv2Conv(nhid, nhid, heads=16, dropout=dropout, concat=False)
        self.classifier = torch.nn.Linear(nhid, nclass)
        self.dropout = dropout
        self.temperature = temperature

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x, edge_index))
        x = F.leaky_relu(self.conv2(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x = x / self.temperature
        return x
