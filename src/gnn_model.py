"""Task 2 (Medium): GNN on music structure graphs (PDF Sec.4.2).

GraphSAGE: h_i^{l+1} = sigma( W^l . CONCAT(h_i^l, MEAN_{j in N(i)} h_j^l) )
Readout: g = 1/|V| sum_i h_i^L, y_hat = softmax(W g + b) (genre) / sigmoid (tags)
Also: GAT option + CNN mel baseline (B2) for comparison.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, global_mean_pool


class GraphSAGEGenre(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, num_layers: int = 2,
                 num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_dim, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, num_classes)

    def encode(self, x, edge_index, batch=None):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)
        g = global_mean_pool(x, batch)  # [B, hidden]
        return g

    def forward(self, x, edge_index, batch=None):
        return self.lin(self.encode(x, edge_index, batch))


class GATGenre(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64, heads: int = 4,
                 num_classes: int = 10, dropout: float = 0.3):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden // heads, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden, hidden // heads, heads=heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.lin = nn.Linear(hidden, num_classes)

    def encode(self, x, edge_index, batch=None):
        x = F.elu(self.gat1(x, edge_index))
        x = self.dropout(x)
        x = F.elu(self.gat2(x, edge_index))
        return global_mean_pool(x, batch)

    def forward(self, x, edge_index, batch=None):
        return self.lin(self.encode(x, edge_index, batch))


class SimpleCNNMel(nn.Module):
    """B2 baseline: CNN on log-mel [1, n_mels, T] -> genre. No graph, no text."""
    def __init__(self, n_mels: int = 128, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.fc = nn.Linear(32 * 8 * 8, num_classes)

    def forward(self, mel):
        # mel: [B, 1, M, T]
        x = F.relu(self.bn1(self.conv1(mel)))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        x = self.pool(x).flatten(1)
        return self.fc(x)
