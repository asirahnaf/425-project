"""Task 2 real-100: train GNN vs CNN from prebuilt graphs+npz (no re-processing)."""
from __future__ import annotations
import json, random
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
from sklearn.metrics import accuracy_score, f1_score
from src.gnn_model import GraphSAGEGenre, SimpleCNNMel
from src.task2_data import GENRES_10
from src.graph_builder import graph_coherence_score

def load_real(root: Path):
    items = []
    for pt in sorted((root/"data"/"processed"/"graphs").glob("*.pt")):
        track = pt.stem  # blues.00000
        genre = track.split(".")[0].lower()
        if genre not in GENRES_10:
            continue
        g = torch.load(str(pt), weights_only=False)
        x = g.x.numpy() if hasattr(g, "x") else g["x"]
        ei = g.edge_index.numpy() if hasattr(g, "edge_index") else g["edge_index"]
        npz = root/"data"/"processed"/f"{track}.npz"
        mel = np.load(str(npz))["mel"].astype(np.float32) if npz.exists() else np.random.randn(128,64).astype(np.float32)
        # fix mel to 64 frames: center-crop or pad
        if mel.shape[1] >= 64:
            s = (mel.shape[1]-64)//2
            mel = mel[:, s:s+64]
        else:
            pad = np.zeros((128, 64-mel.shape[1]), dtype=np.float32)
            mel = np.concatenate([mel, pad], axis=1)
        items.append({"x": x.astype(np.float32), "edge_index": ei,
                      "mel": mel, "y": GENRES_10.index(genre), "track": track})
    return items

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs_gnn", type=int, default=30)
    ap.add_argument("--epochs_cnn", type=int, default=10)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    items = load_real(root)
    print(f"real items: {len(items)}")
    random.Random(42).shuffle(items)
    n = len(items); ntr, nva = int(n*0.7), int(n*0.15)
    tr, va, te = items[:ntr], items[ntr:ntr+nva], items[ntr+nva:]
    print(f"split {len(tr)}/{len(va)}/{len(te)}")
    # GNN
    def pyg(lst):
        out = []
        for it in lst:
            ei = torch.from_numpy(it["edge_index"]).long() if it["edge_index"].size else torch.empty((2,0), dtype=torch.long)
            out.append(Data(x=torch.from_numpy(it["x"]), edge_index=ei, y=torch.tensor([it["y"]])))
        return out
    tr_g, va_g, te_g = pyg(tr), pyg(va), pyg(te)
    gnn = GraphSAGEGenre(tr[0]["x"].shape[1], 64, 2, 10).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    lf = nn.CrossEntropyLoss()
    h_g = []
    for ep in range(1, args.epochs_gnn+1):
        gnn.train()
        for b in PyGLoader(tr_g, batch_size=16, shuffle=True):
            b = b.to(device); opt.zero_grad()
            loss = lf(gnn(b.x, b.edge_index, b.batch), b.y)
            loss.backward(); opt.step()
        gnn.eval()
        with torch.no_grad():
            pr, yu = [], []
            for b in PyGLoader(va_g, batch_size=16):
                b = b.to(device)
                pr += gnn(b.x, b.edge_index, b.batch).argmax(1).cpu().tolist()
                yu += b.y.cpu().tolist()
        acc = accuracy_score(yu, pr)
        h_g.append(acc)
        if ep % 5 == 0: print(f"GNN ep{ep} val={acc:.3f}")
    # CNN
    def stack(lst):
        X = np.stack([it["mel"] for it in lst])[:, None].astype(np.float32)
        return torch.from_numpy(X), torch.tensor([it["y"] for it in lst])
    trX, try_ = stack(tr); vaX, vay = stack(va); teX, tey = stack(te)
    cnn = SimpleCNNMel(128, 10).to(device)
    opt2 = torch.optim.Adam(cnn.parameters(), lr=1e-3)
    h_c = []
    for ep in range(1, args.epochs_cnn+1):
        cnn.train()
        for xb, yb in TLoader(TensorDataset(trX, try_), batch_size=16, shuffle=True):
            xb, yb = xb.to(device), yb.to(device)
            opt2.zero_grad()
            loss = lf(cnn(xb), yb)
            loss.backward(); opt2.step()
        cnn.eval()
        with torch.no_grad():
            acc = accuracy_score(vay.numpy(), cnn(vaX.to(device)).argmax(1).cpu().numpy())
        h_c.append(acc)
        print(f"CNN ep{ep} val={acc:.3f}")
    # test
    with torch.no_grad():
        pr = []
        for b in PyGLoader(te_g, batch_size=16):
            b = b.to(device)
            pr += gnn(b.x, b.edge_index, b.batch).argmax(1).cpu().tolist()
        yt = [it["y"] for it in te]
        m_g = {"acc": float(accuracy_score(yt, pr)), "macro_f1": float(f1_score(yt, pr, average="macro", zero_division=0))}
        pc = cnn(teX.to(device)).argmax(1).cpu().tolist()
        m_c = {"acc": float(accuracy_score(tey.tolist(), pc)), "macro_f1": float(f1_score(tey.tolist(), pc, average="macro", zero_division=0))}
    print("TEST GNN:", m_g, "CNN:", m_c)
    coh = float(np.mean([graph_coherence_score(it["x"], it["edge_index"]) for it in te]))
    print("Sgraph:", round(coh, 3))
    res = root/"results"
    (res/"metrics_task2_real100.json").write_text(json.dumps(
        {"gnn": m_g, "cnn": m_c, "val_gnn": h_g, "val_cnn": h_c, "s_graph": coh}, indent=2))
    fig, ax = plt.subplots()
    ax.plot(h_g, marker="o", label="GNN"); ax.plot(h_c, marker="s", label="CNN")
    ax.set_xlabel("epoch"); ax.set_ylabel("val acc"); ax.legend(); fig.tight_layout()
    fig.savefig(res/"plots"/"task2_real100_curves.png", dpi=150)
    torch.save(gnn.state_dict(), res/"gnn_task2_real100.pt")
    print("saved real100 metrics + curves")

if __name__ == "__main__":
    main()
