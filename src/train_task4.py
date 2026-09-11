"""Task 4 fast: 800 MusicCaps captions + tag-conditioned graphs, frozen BERT.

Outputs: metrics_task4.json (R@1/5/10), retrieval_examples_task4.json (10)
"""
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGLoader

from src.bert_encoder import get_tokenizer
from transformers import AutoModel
from src.gnn_model import GraphSAGEGenre
from src.contrastive import ProjectionHead, info_nce, recall_at_k
from src.graph_builder import build_segment_graph


def load_pairs(raw_csv: Path, max_n: int = 800, seed: int = 42):
    rows = list(csv.DictReader(open(raw_csv, encoding="utf-8")))
    random.Random(seed).shuffle(rows)
    rows = rows[:max_n]
    tag_cols = [c for c in rows[0].keys() if c.startswith("tag_")]
    tags = [c.replace("tag_", "") for c in tag_cols]
    pairs = []
    rng = np.random.default_rng(seed)
    for r in rows:
        vec = np.array([int(r[c]) for c in tag_cols], dtype=np.float32)
        # tag-conditioned graph
        proto = np.zeros(32, dtype=np.float32)
        for ti, v in enumerate(vec):
            if v:
                proto[(ti*3) % 32] += 1.2
                proto[(ti*3+1) % 32] += 0.8
        proto += rng.normal(0, 0.3, 32).astype(np.float32)
        X = np.stack([proto + rng.normal(0, 0.5, 32).astype(np.float32) for _ in range(10)])
        for i in range(1, 10):
            X[i] = 0.7*X[i] + 0.3*X[i-1]
        ei, _ = build_segment_graph(X, tau=0.6)
        pairs.append({"track": r["clip_id"], "text": r["text"],
                      "tags": vec.tolist(), "x": X, "edge_index": ei})
    return pairs, tags

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max_n", type=int, default=800)
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    pairs, tag_names = load_pairs(root/"data"/"raw"/"mtt_clips.csv", args.max_n)
    n = len(pairs); ntr, nva = int(n*0.7), int(n*0.15)
    tr, va, te = pairs[:ntr], pairs[ntr:ntr+nva], pairs[ntr+nva:]
    print(f"pairs {n} train{len(tr)} val{len(va)} test{len(te)} tags={tag_names}")

    # cache BERT CLS (frozen)
    tok = get_tokenizer("distilbert-base-uncased")
    bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()
    def encode_texts(texts):
        outs = []
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = tok(texts[i:i+32], truncation=True, padding="max_length",
                          max_length=64, return_tensors="pt").to(device)
                outs.append(bert(**enc).last_hidden_state[:, 0, :].cpu())
        return torch.cat(outs)
    all_txt = [p["text"] for p in tr+va+te]
    CLS_all = encode_texts(all_txt).to(device)  # [N,768]
    print("cached CLS:", tuple(CLS_all.shape))
    off = {p["track"]: i for i, p in enumerate(tr+va+te)}
    def to_pyg(items):
        out = []
        for p in items:
            ei = torch.from_numpy(p["edge_index"]).long() if p["edge_index"].size else torch.empty((2,0), dtype=torch.long)
            out.append(Data(x=torch.from_numpy(p["x"]), edge_index=ei,
                            idx=torch.tensor(off[p["track"]]), track=p["track"], text=p["text"]))
        return out
    tr_g, va_g, te_g = to_pyg(tr), to_pyg(va), to_pyg(te)

    genc = GraphSAGEGenre(32, 64, 2, 10).to(device)
    proj_g = ProjectionHead(64, 128).to(device)
    proj_t = ProjectionHead(768, 128).to(device)
    opt = torch.optim.Adam(list(genc.parameters())+list(proj_g.parameters())+list(proj_t.parameters()), lr=1e-3)
    for ep in range(1, args.epochs+1):
        genc.train(); proj_g.train(); proj_t.train()
        tot = 0.0; nb = 0
        for b in PyGLoader(tr_g, batch_size=args.batch, shuffle=True):
            b = b.to(device); opt.zero_grad()
            g = proj_g(genc.encode(b.x, b.edge_index, b.batch))
            t = proj_t(CLS_all[b.idx.to(device)])
            loss, _ = info_nce(g, t, 0.07)
            loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"ep{ep}/{args.epochs} loss={tot/max(1,nb):.4f}")

    # retrieval on test
    genc.eval(); proj_g.eval(); proj_t.eval()
    with torch.no_grad():
        Gs, Ts = [], []
        for b in PyGLoader(te_g, batch_size=64):
            b = b.to(device)
            Gs.append(proj_g(genc.encode(b.x, b.edge_index, b.batch)).cpu())
            Ts.append(proj_t(CLS_all[b.idx.to(device)]).cpu())
        G = torch.cat(Gs); T = torch.cat(Ts)
        sim = G @ T.T
        c2a = recall_at_k(sim); a2c = recall_at_k(sim.T)
    print("caption->audio:", c2a, "audio->caption:", a2c)
    # 10 examples: query caption -> top3
    ex = []
    sim_np = sim.numpy()
    for i in range(min(10, len(te))):
        top3 = sim_np[i].argsort()[::-1][:3].tolist()
        ex.append({"query": te[i]["text"][:200], "true_track": te[i]["track"],
                   "top3": [{"track": te[j]["track"], "text": te[j]["text"][:150],
                             "score": round(float(sim_np[i, j]), 3)} for j in top3]})
    res = root/"results"
    (res/"metrics_task4.json").write_text(json.dumps(
        {"caption_to_audio": c2a, "audio_to_caption": a2c, "n_test": len(te)}, indent=2))
    (res/"retrieval_examples_task4.json").write_text(json.dumps(ex, indent=2))
    torch.save({"gnn": genc.state_dict(), "pg": proj_g.state_dict(), "pt": proj_t.state_dict()},
               res/"contrastive_task4.pt")
    print("saved metrics_task4 + 10 examples")

if __name__ == "__main__":
    main()
