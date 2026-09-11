"""Train Task 3 fusion ablations (fast: BERT frozen + cached).

Ablations: bert_only, gnn_only, concat, cross-attention.
Outputs: metrics_task3.json, task3_tsne.png, task3_cases.json
"""
from __future__ import annotations
import argparse, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, average_precision_score
from sklearn.manifold import TSNE

from src.task3_data import prepare
from src.bert_encoder import get_tokenizer
from transformers import AutoModel
from src.gnn_model import GraphSAGEGenre
from src.fusion_model import CrossAttentionFusion, EarlyConcatFusion


def split_items(items, seed=42):
    random.Random(seed).shuffle(items)
    n = len(items)
    ntr, nva = int(n*0.7), int(n*0.15)
    return items[:ntr], items[ntr:ntr+nva], items[ntr+nva:]

@torch.no_grad()
def cache_bert(items, tok, bert, max_len=64, batch=16, device="cpu"):
    bert.eval()
    H_all, CLS_all = [], []
    for i in range(0, len(items), batch):
        batch_txt = [it["text"] for it in items[i:i+batch]]
        enc = tok(batch_txt, truncation=True, padding="max_length",
                  max_length=max_len, return_tensors="pt").to(device)
        out = bert(**enc).last_hidden_state.cpu()  # [B,L,H]
        H_all.append(out)
        CLS_all.append(out[:, 0, :])
    return torch.cat(H_all), torch.cat(CLS_all)

def to_pyg(items, offset_map):
    out = []
    for it in items:
        ei = torch.from_numpy(it["edge_index"]).long() if it["edge_index"].size else torch.empty((2,0), dtype=torch.long)
        out.append(Data(x=torch.from_numpy(it["x"]), edge_index=ei,
                        y=torch.tensor(it["tags"], dtype=torch.float).unsqueeze(0),
                        idx=torch.tensor(offset_map[it["track"]]),
                        genre=it["genre"], track=it["track"], text=it["text"]))
    return out

def metrics(y_true, y_prob):
    yp = (np.asarray(y_prob) >= 0.5).astype(int)
    yt = np.asarray(y_true)
    try:
        ap = float(average_precision_score(yt, y_prob, average="macro"))
    except Exception:
        ap = None
    return {"macro_f1": float(f1_score(yt, yp, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(yt, yp, average="micro", zero_division=0)),
            "auc_pr": ap}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--n_per_genre", type=int, default=12)
    ap.add_argument("--model_name", default="distilbert-base-uncased")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    items, labels = prepare(args.n_per_genre)
    tr_items, va_items, te_items = split_items(items)
    print(f"split {len(tr_items)}/{len(va_items)}/{len(te_items)} labels={labels}")
    # BERT cache (frozen)
    tok = get_tokenizer(args.model_name)
    bert = AutoModel.from_pretrained(args.model_name).to(device).eval()
    all_items = tr_items + va_items + te_items
    off = {it["track"]: i for i, it in enumerate(all_items)}
    H_all, CLS_all = cache_bert(all_items, tok, bert, 64, 16, device)
    print("cached BERT:", tuple(H_all.shape))
    tr, va, te = to_pyg(tr_items, off), to_pyg(va_items, off), to_pyg(te_items, off)
    in_dim = tr[0].x.shape[1]
    K = len(labels)
    H_all, CLS_all = H_all.to(device), CLS_all.to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    def get_batch_tensors(batch):
        idx = batch.idx.to(device)
        return H_all[idx], CLS_all[idx]

    results = {}
    # --- 1) BERT-only ---
    Wb = nn.Linear(768, K).to(device)
    opt = torch.optim.Adam(Wb.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        Wb.train()
        for b in PyGLoader(tr, batch_size=args.batch, shuffle=True):
            opt.zero_grad()
            _, cls = get_batch_tensors(b)
            loss = loss_fn(Wb(cls), b.y.to(device))
            loss.backward(); opt.step()
    results["bert_only"] = metrics(*eval_head(Wb, te, get_batch_tensors, "bert", device))
    print("bert_only:", results["bert_only"])

    # --- 2) GNN-only ---
    gnn = GraphSAGEGenre(in_dim, 64, 2, K).to(device)
    opt = torch.optim.Adam(gnn.parameters(), lr=1e-3)
    for ep in range(args.epochs):
        gnn.train()
        for b in PyGLoader(tr, batch_size=args.batch, shuffle=True):
            b = b.to(device); opt.zero_grad()
            loss = loss_fn(gnn(b.x, b.edge_index, b.batch), b.y)
            loss.backward(); opt.step()
    results["gnn_only"] = metrics(*eval_gnn(gnn, te, device))
    print("gnn_only:", results["gnn_only"])

    # --- 3/4) concat + cross (joint GNN+fusion, BERT frozen) ---
    for name, FCls in [("concat", EarlyConcatFusion), ("cross", CrossAttentionFusion)]:
        genc = GraphSAGEGenre(in_dim, 64, 2, K).to(device)  # use encode only
        fus = (FCls(64, 768, K) if name == "concat" else FCls(64, 768, 128, K)).to(device)
        opt = torch.optim.Adam(list(genc.parameters()) + list(fus.parameters()), lr=1e-3)
        for ep in range(args.epochs):
            genc.train(); fus.train()
            for b in PyGLoader(tr, batch_size=args.batch, shuffle=True):
                b = b.to(device); opt.zero_grad()
                H, CLS = get_batch_tensors(b)
                g = genc.encode(b.x, b.edge_index, b.batch)
                logits = fus(g, H if name == "cross" else CLS)[0]
                loss = loss_fn(logits, b.y)
                loss.backward(); opt.step()
        yt, yp, Z, genres = eval_fusion(genc, fus, te, get_batch_tensors, name, device)
        results[name] = metrics(yt, yp)
        print(name, results[name])
        if name == "cross":
            torch.save({"gnn": genc.state_dict(), "fus": fus.state_dict()}, root/"results"/"fusion_task3.pt")
            # t-SNE
            Z2 = TSNE(n_components=2, random_state=42, perplexity=min(15, len(Z)-1)).fit_transform(np.asarray(Z))
            fig, ax = plt.subplots()
            uniq = sorted(set(genres))
            for u in uniq:
                m = [i for i, g in enumerate(genres) if g == u]
                ax.scatter(Z2[m, 0], Z2[m, 1], label=u, s=20)
            ax.legend(fontsize=6, ncol=2); ax.set_title("t-SNE of z (cross-attention)")
            fig.tight_layout(); fig.savefig(root/"results"/"plots"/"task3_tsne.png", dpi=150)
            # 3 cases
            cases = []
            for i in range(min(3, len(te_items))):
                cases.append({"track": te[i].track, "genre": genres[i],
                              "text": te[i].text,
                              "true": [labels[k] for k in range(K) if yt[i][k] == 1],
                              "pred": [labels[k] for k in range(K) if yp[i][k] >= 0.5],
                              "num_nodes": int(te[i].x.shape[0]),
                              "num_edges": int(te[i].edge_index.shape[1])})
            (root/"results"/"task3_cases.json").write_text(json.dumps(cases, indent=2))
            print("cases:", cases)
    (root/"results"/"metrics_task3.json").write_text(json.dumps({"test": results, "labels": labels}, indent=2))
    print("saved metrics_task3.json + tsne + cases")

@torch.no_grad()
def eval_head(Wb, te, get_tensors, kind, device):
    yt, yp = [], []
    for b in PyGLoader(te, batch_size=32):
        b = b.to(device)
        _, cls = get_tensors(b)
        yp.append(torch.sigmoid(Wb(cls)).cpu().numpy())
        yt.append(b.y.cpu().numpy())
    import numpy as np
    return np.concatenate(yt), np.concatenate(yp)

@torch.no_grad()
def eval_gnn(gnn, te, device):
    import numpy as np
    gnn.eval(); yt, yp = [], []
    for b in PyGLoader(te, batch_size=32):
        b = b.to(device)
        yp.append(torch.sigmoid(gnn(b.x, b.edge_index, b.batch)).cpu().numpy())
        yt.append(b.y.cpu().numpy())
    return np.concatenate(yt), np.concatenate(yp)

@torch.no_grad()
def eval_fusion(genc, fus, te, get_tensors, name, device):
    import numpy as np
    genc.eval(); fus.eval(); yt, yp, Z, G = [], [], [], []
    for b in PyGLoader(te, batch_size=32):
        b = b.to(device)
        H, CLS = get_tensors(b)
        g = genc.encode(b.x, b.edge_index, b.batch)
        logits, z, _ = fus(g, H if name == "cross" else CLS)
        yp.append(torch.sigmoid(logits).cpu().numpy())
        yt.append(b.y.cpu().numpy())
        Z.append(z.cpu().numpy())
        G += list(b.genre)
    return np.concatenate(yt), np.concatenate(yp), np.concatenate(Z), G

if __name__ == "__main__":
    main()
