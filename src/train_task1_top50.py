"""Train Task 1 BERT tag classifier on TOP-50 MTT tags (spec Sec.4.1).

ADDITIVE track: does NOT touch the 20-tag pipeline (mtt_clips.csv / metrics_task1.json /
bert_task1.pt) that Task4 + human eval depend on.

Reads:  data/raw/mtt50_clips.csv (5521 MusicCaps captions x 50 tags)
Writes: data/splits/task1_top50_{train,val,test}.json, task1_top50_tags.json
        results/metrics_task1_top50.json, results/plots/task1_top50_f1_curves.png
        results/task1_top50_examples.json (5 examples per spec), results/bert_task1_top50.pt
"""
from __future__ import annotations
import argparse
import csv
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.bert_encoder import (get_tokenizer, MusicTagDataset, collate_fn,
                              BertTagClassifier, compute_metrics,
                              train_epoch, eval_logits)


def load_top50(raw_csv: Path):
    items, tag_names = [], []
    with open(raw_csv, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        tag_cols = [c for c in (r.fieldnames or []) if c.startswith("tag_")]
        tag_names = [c.replace("tag_", "") for c in tag_cols]
        for row in r:
            txt = row.get("text", "")
            if not txt:
                continue
            tags = [int(row.get(c, 0) or 0) for c in tag_cols]
            items.append({"id": row.get("clip_id", str(len(items))),
                          "text": txt, "tags": tags})
    return items, tag_names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    raw_csv = root / "data" / "raw" / "mtt50_clips.csv"
    assert raw_csv.exists(), f"missing {raw_csv} (run python -m src.make_top50_mtt first)"
    splits_dir, res_dir = root / "data" / "splits", root / "results"
    plot_dir = res_dir / "plots"
    splits_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(exist_ok=True); plot_dir.mkdir(parents=True, exist_ok=True)

    items, tag_names = load_top50(raw_csv)
    K = len(tag_names)
    assert K == 50, f"expected 50 tags, got {K}"
    print(f"clips={len(items)} tags={K}")
    random.Random(args.seed).shuffle(items)
    n = len(items); n_tr, n_va = int(n * 0.7), int(n * 0.15)
    splits = {"train": items[:n_tr], "val": items[n_tr:n_tr + n_va],
              "test": items[n_tr + n_va:]}
    for s, v in splits.items():
        (splits_dir / f"task1_top50_{s}.json").write_text(json.dumps(v))
    (splits_dir / "task1_top50_tags.json").write_text(json.dumps(tag_names, indent=2))
    print({s: len(v) for s, v in splits.items()}, "->", splits_dir)

    tok = get_tokenizer(args.model)
    tr = DataLoader(MusicTagDataset(splits["train"], tok, args.max_len),
                    batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    va = DataLoader(MusicTagDataset(splits["val"], tok, args.max_len),
                    batch_size=args.batch, collate_fn=collate_fn)
    te = DataLoader(MusicTagDataset(splits["test"], tok, args.max_len),
                    batch_size=args.batch, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "epochs:", args.epochs, "batch:", args.batch)
    model = BertTagClassifier(args.model, num_labels=K).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {"train_loss": [], "val_macro": [], "val_micro": []}
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, tr, optim, device, loss_fn)
        yt, yp = eval_logits(model, va, device)
        m = compute_metrics(yt, yp)
        hist["train_loss"].append(loss)
        hist["val_macro"].append(m["macro_f1"])
        hist["val_micro"].append(m["micro_f1"])
        print(f"ep{ep}/{args.epochs} loss={loss:.4f} val_macro={m['macro_f1']:.3f} micro={m['micro_f1']:.3f} AP={m['auc_pr_macro']:.3f}", flush=True)

    yt, yp = eval_logits(model, te, device)
    test_m = compute_metrics(yt, yp)
    print("TEST-50:", test_m, flush=True)

    fig, ax1 = plt.subplots()
    ax1.plot(hist["train_loss"], marker="o", label="train BCE")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(hist["val_macro"], marker="s", label="val macro-F1")
    ax2.plot(hist["val_micro"], marker="^", label="val micro-F1")
    ax2.set_ylabel("F1")
    fig.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(plot_dir / "task1_top50_f1_curves.png", dpi=150)
    print("plot ->", plot_dir / "task1_top50_f1_curves.png")

    import numpy as np
    idx = np.random.RandomState(0).choice(len(yp), size=min(5, len(yp)), replace=False)
    ex = []
    for i in map(int, idx):
        true = [tag_names[k] for k in range(K) if yt[i, k] == 1]
        pred = [tag_names[k] for k in range(K) if yp[i, k] >= 0.5]
        ex.append({"text": splits["test"][i]["text"], "true": true,
                   "pred": pred, "probs": [round(float(v), 3) for v in yp[i]]})
        print(f"- {ex[-1]['text'][:80]} | true={true} pred={pred}")
    (res_dir / "task1_top50_examples.json").write_text(json.dumps(ex, indent=2))
    (res_dir / "metrics_task1_top50.json").write_text(json.dumps(
        {"test": test_m, "history": hist, "tags": tag_names,
         "note": "Top-50 MTT tags via MusicCaps caption keyword proxy (5521 clips). 6/50 tags support<50 (quiet/woman/no vocal/sitar/new age/choral); low macro expected (sparsity). 20-tag pipeline untouched."}, indent=2))
    torch.save(model.state_dict(), res_dir / "bert_task1_top50.pt")
    print("saved metrics_task1_top50.json + bert_task1_top50.pt")


if __name__ == "__main__":
    main()
