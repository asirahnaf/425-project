"""Task 4 (Advanced): Contrastive GNN-BERT (PDF Sec.4.4).

LNCE = -log exp(sim(gi,ti)/tau) / sum_j exp(sim(gi,tj)/tau), sim=cosine.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim),
                                 nn.ReLU(),
                                 nn.Linear(in_dim, out_dim))

    def forward(self, x):
        return F.normalize(self.net(x), dim=1)


def info_nce(g: torch.Tensor, t: torch.Tensor, tau: float = 0.07):
    # g,t: [B,D] normalized
    logits = (g @ t.T) / tau  # [B,B]
    labels = torch.arange(g.size(0), device=g.device)
    loss_g2t = F.cross_entropy(logits, labels)
    loss_t2g = F.cross_entropy(logits.T, labels)
    return (loss_g2t + loss_t2g) / 2, logits


@torch.no_grad()
def recall_at_k(sim: torch.Tensor, ks=(1, 5, 10)):
    # sim: [N,N] query x gallery, correct = diagonal
    N = sim.size(0)
    out = {}
    for k in ks:
        k = min(k, N)
        topk = sim.topk(k, dim=1).indices  # [N,k]
        hits = sum(1 for i in range(N) if i in topk[i].tolist())
        out[f"R@{k}"] = hits / N
    return out
