"""Train Task 1 BERT tag classifier. Saves curves + metrics + examples.

Outputs:
  results/metrics_task1.json
  results/plots/task1_f1_curves.png
  results/task1_examples.json
"""
from __future__ import annotations
import argparse
import json
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
from src.task1_data import prepare, TAGS_10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max_len", type=int, default=128)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--n", type=int, default=240)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    splits_dir, raw_dir = root / "data" / "splits", root / "data" / "raw"
    res_dir, plot_dir = root / "results", root / "results" / "plots"
    res_dir.mkdir(exist_ok=True); plot_dir.mkdir(parents=True, exist_ok=True)

    splits, tag_names = prepare(splits_dir, raw_dir, n=args.n)
    K = len(tag_names)
    print(f"Tags({K}): {tag_names}")

    tok = get_tokenizer(args.model)
    tr = DataLoader(MusicTagDataset(splits["train"], tok, args.max_len),
                    batch_size=args.batch, shuffle=True, collate_fn=collate_fn)
    va = DataLoader(MusicTagDataset(splits["val"], tok, args.max_len),
                    batch_size=args.batch, collate_fn=collate_fn)
    te = DataLoader(MusicTagDataset(splits["test"], tok, args.max_len),
                    batch_size=args.batch, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    model = BertTagClassifier(args.model, num_labels=K, freeze_bert=args.freeze).to(device)
    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    hist = {"train_loss": [], "val_macro": [], "val_micro": []}
    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, tr, optim, device, loss_fn)
        yt, yp = eval_logits(model, va, device)
        m = compute_metrics(yt, yp)
        hist["train_loss"].append(loss)
        hist["val_macro"].append(m["macro_f1"])
        hist["val_micro"].append(m["micro_f1"])
        print(f"ep{ep}/{args.epochs} loss={loss:.4f} val_macro={m['macro_f1']:.3f} micro={m['micro_f1']:.3f} AP={m['auc_pr_macro']}")

    yt, yp = eval_logits(model, te, device)
    test_m = compute_metrics(yt, yp)
    print("TEST:", test_m)

    # plot
    fig, ax1 = plt.subplots()
    ax1.plot(hist["train_loss"], marker="o", label="train BCE")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("loss")
    ax2 = ax1.twinx()
    ax2.plot(hist["val_macro"], marker="s", label="val macro-F1")
    ax2.plot(hist["val_micro"], marker="^", label="val micro-F1")
    ax2.set_ylabel("F1")
    fig.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(plot_dir / "task1_f1_curves.png", dpi=150)
    print("plot ->", plot_dir / "task1_f1_curves.png")

    # 5 examples
    import numpy as np
    idx = np.random.RandomState(0).choice(len(yp), size=min(5, len(yp)), replace=False)
    ex = []
    for i in map(int, idx):
        true = [tag_names[k] for k in range(K) if yt[i, k] == 1]
        pred = [tag_names[k] for k in range(K) if yp[i, k] >= 0.5]
        ex.append({"text": splits["test"][i]["text"], "true": true,
                   "pred": pred, "probs": [round(float(v), 3) for v in yp[i]]})
        print(f"- {ex[-1]['text'][:80]} | true={true} pred={pred}")
    (res_dir / "task1_examples.json").write_text(json.dumps(ex, indent=2))
    (res_dir / "metrics_task1.json").write_text(json.dumps(
        {"test": test_m, "history": hist, "tags": tag_names}, indent=2))
    torch.save(model.state_dict(), res_dir / "bert_task1.pt")
    print("saved metrics + bert_task1.pt")


if __name__ == "__main__":
    main()
