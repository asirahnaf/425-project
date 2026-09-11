"""Task 1 (Easy): BERT Baseline for Music Tag Understanding (PDF Sec.4.1).

Model: t = BERT_CLS(Xtext), y_hat_k = sigmoid(w_k^T t + b_k)
Loss: LBERT = -1/K sum_k [ y_k log y_hat + (1-y_k) log(1-y_hat) ]  (BCE)
"""
from __future__ import annotations
import json
from pathlib import Path
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel


def get_tokenizer(name: str = "distilbert-base-uncased"):
    return AutoTokenizer.from_pretrained(name)


class MusicTagDataset(Dataset):
    """Expects list of {text:str, tags:list[int 0/1] of len K}."""
    def __init__(self, items: list[dict], tokenizer, max_length: int = 128):
        self.items = items
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        enc = self.tok(it["text"], truncation=True, padding="max_length",
                       max_length=self.max_length, return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(it["tags"], dtype=torch.float)}


def collate_fn(batch):
    return {"input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.stack([b["labels"] for b in batch])}


class BertTagClassifier(nn.Module):
    def __init__(self, model_name: str = "distilbert-base-uncased",
                 num_labels: int = 10, dropout: float = 0.2,
                 freeze_bert: bool = False):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False
        hidden = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT has no pooler; BERT has pooler. Use first token for both.
        cls = out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(cls))  # logits (BCEWithLogits)


def compute_metrics(y_true, y_prob, thresh: float = 0.5) -> dict:
    from sklearn.metrics import f1_score, average_precision_score
    import numpy as np
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= thresh).astype(int)
    macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    micro = f1_score(y_true, y_pred, average="micro", zero_division=0)
    try:
        ap = average_precision_score(y_true, y_prob, average="macro")
    except Exception:
        ap = float("nan")
    return {"macro_f1": float(macro), "micro_f1": float(micro),
            "auc_pr_macro": float(ap) if ap == ap else None}


def train_epoch(model, loader, optim, device, loss_fn):
    model.train()
    tot = 0.0
    for b in loader:
        optim.zero_grad()
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        loss = loss_fn(logits, b["labels"].to(device))
        loss.backward()
        optim.step()
        tot += loss.item()
    return tot / max(1, len(loader))


@torch.no_grad()
def eval_logits(model, loader, device):
    model.eval()
    ys, ps = [], []
    for b in loader:
        logits = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        ys.append(b["labels"].numpy())
        ps.append(torch.sigmoid(logits).cpu().numpy())
    import numpy as np
    return np.concatenate(ys), np.concatenate(ps)
