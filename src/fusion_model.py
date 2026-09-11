"""Task 3 (Hard): GNN-BERT fusion (PDF Sec.4.3).

Cross-attention: A=softmax(QK^T/sqrt d), Q=gW_Q, K=HtextW_K,
  z=CONCAT(g, A Htext), y_hat=sigmoid(Wz+b)
Baselines: BERT-only, GNN-only, early-concat.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    def __init__(self, g_dim: int = 64, h_dim: int = 768, d: int = 128,
                 num_labels: int = 10, dropout: float = 0.2):
        super().__init__()
        self.wq = nn.Linear(g_dim, d)
        self.wk = nn.Linear(h_dim, d)
        self.wv = nn.Linear(h_dim, d)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(g_dim + d, num_labels)

    def forward(self, g, H):
        # g:[B,G] H:[B,L,H]
        Q = self.wq(g).unsqueeze(1)          # [B,1,D]
        K = self.wk(H)                        # [B,L,D]
        V = self.wv(H)                        # [B,L,D]
        scores = (Q @ K.transpose(1, 2)) / math.sqrt(K.size(-1))  # [B,1,L]
        A = scores.softmax(-1)
        ctx = (A @ V).squeeze(1)              # [B,D]
        z = torch.cat([g, self.dropout(ctx)], dim=1)
        return self.classifier(z), z, A.squeeze(1)


class EarlyConcatFusion(nn.Module):
    def __init__(self, g_dim: int = 64, h_dim: int = 768, num_labels: int = 10,
                 dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(g_dim + h_dim, num_labels)

    def forward(self, g, t_cls):
        z = torch.cat([g, self.dropout(t_cls)], dim=1)
        return self.classifier(z), z, None
