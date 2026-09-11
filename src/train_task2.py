"""Train Task 2: GNN (GraphSAGE/GAT) vs CNN-mel baseline on genre.

Outputs:
  results/metrics_task2.json
  results/plots/task2_curves.png (GNN vs CNN acc)
  results/plots/task2_confusion.png (optional)
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
from torch.utils.data import TensorDataset, DataLoader as TLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score

from src.gnn_model import GraphSAGEGenre, GATGenre, SimpleCNNMel
from src.task2_data import prepare, GENRES_10


def to_pyg_list(items):
    out = []
    for it in items:
        ei = torch.from_numpy(it["edge_index"]).long() if it["edge_index"].size else torch.empty((2, 0), dtype=torch.long)
        out.append(Data(x=torch.from_numpy(it["x"]), edge_index=ei,
                        y=torch.tensor([it["y"]]), track=it["track"]))
    return out

def split_items(items, seed=42):
    random.Random(seed).shuffle(items)
    n = len(items)
    ntr, nva = int(n * 0.7), int(n * 0.15)
    return items[:ntr], items[ntr:ntr + nva], items[ntr + nva:]

def train_gnn(train, val, in_dim, num_classes, model_type="sage", epochs=30, lr=1e-3, device="cpu"):
    tr_loader = PyGLoader(train, batch_size=32, shuffle=True)
    va_loader = PyGLoader(val, batch_size=32)
    model = (GraphSAGEGenre(in_dim, 64, 2, num_classes) if model_type == "sage"
             else GATGenre(in_dim, 64, 4, num_classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        for b in tr_loader:
            b = b.to(device)
            opt.zero_grad()
            loss = loss_fn(model(b.x, b.edge_index, b.batch), b.y)
            loss.backward()
            opt.step()
        # val acc
        model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for b in va_loader:
                b = b.to(device)
                preds += model(b.x, b.edge_index, b.batch).argmax(1).cpu().tolist()
                trues += b.y.cpu().tolist()
        acc = accuracy_score(trues, preds)
        hist["acc"].append(acc)
        if ep % 5 == 0 or ep == 1:
            print(f"GNN ep{ep}/{epochs} val_acc={acc:.3f}")
    return model, hist

def train_cnn(train_items, val_items, num_classes, epochs=10, device="cpu"):
    def stack(items):
        X = np.stack([it["mel"][:, :64] for it in items])[:, None, :, :].astype(np.float32)
        y = np.array([it["y"] for it in items], dtype=np.int64)
        return torch.from_numpy(X), torch.from_numpy(y)
    trX, try_ = stack(train_items)
    vaX, vay = stack(val_items)
    tr_loader = TLoader(TensorDataset(trX, try_), batch_size=32, shuffle=True)
    model = SimpleCNNMel(128, num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"acc": []}
    for ep in range(1, epochs + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            preds = model(vaX.to(device)).argmax(1).cpu().numpy()
        acc = accuracy_score(vay.numpy(), preds)
        hist["acc"].append(acc)
        print(f"CNN ep{ep}/{epochs} val_acc={acc:.3f}")
    return model, hist

@torch.no_grad()
def eval_all(model, items, kind="gnn", device="cpu"):
    if kind == "gnn":
        loader = PyGLoader(to_pyg_list(items), batch_size=32)
        model.eval()
        preds, trues = [], []
        for b in loader:
            b = b.to(device)
            preds += model(b.x, b.edge_index, b.batch).argmax(1).cpu().tolist()
            trues += b.y.cpu().tolist()
    else:
        X = torch.from_numpy(np.stack([it["mel"][:, :64] for it in items])[:, None]).float().to(device)
        preds = model(X).argmax(1).cpu().tolist()
        trues = [it["y"] for it in items]
    return {"acc": float(accuracy_score(trues, preds)),
            "macro_f1": float(f1_score(trues, preds, average="macro", zero_division=0))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_gnn", type=int, default=30)
    ap.add_argument("--epochs_cnn", type=int, default=10)
    ap.add_argument("--n_per_genre", type=int, default=30)
    ap.add_argument("--model", default="sage", choices=["sage", "gat"])
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    items, genres = prepare(root / "data" / "raw", args.n_per_genre)
    tr, va, te = split_items(items)
    print(f"split train{len(tr)} val{len(va)} test{len(te)} in_dim={tr[0]['x'].shape[1]}")
    gnn, h_gnn = train_gnn(to_pyg_list(tr), to_pyg_list(va), tr[0]["x"].shape[1],
                           len(genres), args.model, args.epochs_gnn, device=device)
    cnn, h_cnn = train_cnn(tr, va, len(genres), args.epochs_cnn, device=device)
    m_gnn = eval_all(gnn, te, "gnn", device)
    m_cnn = eval_all(cnn, te, "cnn", device)
    print("TEST GNN:", m_gnn, "CNN:", m_cnn)
    # coherence sample
    from src.graph_builder import graph_coherence_score
    coh = float(np.mean([graph_coherence_score(it["x"], it["edge_index"]) for it in te[:20]]))
    print(f"mean Sgraph (20 test): {coh:.3f}")
    res = root / "results"; (res / "plots").mkdir(parents=True, exist_ok=True)
    (res / "metrics_task2.json").write_text(json.dumps(
        {"gnn_test": m_gnn, "cnn_test": m_cnn, "gnn_val": h_gnn, "cnn_val": h_cnn,
         "genres": genres, "s_graph": coh, "model": args.model}, indent=2))
    fig, ax = plt.subplots()
    ax.plot(h_gnn["acc"], marker="o", label="GNN val-acc")
    ax.plot(h_cnn["acc"], marker="s", label="CNN val-acc")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy"); ax.legend(); fig.tight_layout()
    fig.savefig(res / "plots" / "task2_curves.png", dpi=150)
    torch.save(gnn.state_dict(), res / "gnn_task2.pt")
    torch.save(cnn.state_dict(), res / "cnn_mel.pt")
    print("saved metrics_task2.json + task2_curves.png")

if __name__ == "__main__":
    main()
