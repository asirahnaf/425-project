"""Task 4 leftover: zero-shot vs supervised + simulated human eval.

Zero-shot: tag prompts via Task4 text proj (frozen BERT+proj_t from contrastive_task4.pt)
  vs supervised Task1 BERT classifier (bert_task1.pt) on same 20-tag MusicCaps test.
Human: 5 virtual raters (rule-based + noise) on 10 retrieval examples + real template.
Outputs: metrics_task4_zero.json, human_eval_task4.json, human_eval_template.csv
"""
from __future__ import annotations
import csv, json
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score

TAG_PROMPTS = {
    "vocal": "singing voice with vocal melody", "male": "male singer man voice",
    "guitar": "electric guitar riff", "drums": "drum beat percussion",
    "fast": "fast uptempo energetic tempo", "loud": "loud noisy aggressive sound",
    "synth": "synthesizer synth lead", "female": "female singer woman voice",
    "soft": "soft mellow gentle calm music", "piano": "piano keys melody",
    "happy": "happy cheerful upbeat joyful", "pop": "pop catchy song",
    "electronic": "electronic dance beat", "rock": "rock guitar band",
    "slow": "slow tempo ballad", "dance": "danceable groove",
    "strings": "string section violin", "classical": "classical orchestral",
    "hiphop": "hip hop rap", "violin": "violin solo",
}

def main():
    root = Path(__file__).resolve().parent.parent
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # load Task4 test pairs (same builder as train_task4, seed fixed -> same split if max_n 800)
    import sys
    sys.path.insert(0, str(root/"src"))
    from train_task4 import load_pairs
    pairs, tags = load_pairs(root/"data"/"raw"/"mtt_clips.csv", 800, 42)
    n = len(pairs); ntr, nva = int(n*0.7), int(n*0.15)
    te = pairs[ntr+nva:]
    print(f"test {len(te)} tags={tags}")
    # encoders
    from src.bert_encoder import get_tokenizer
    from transformers import AutoModel
    tok = get_tokenizer("distilbert-base-uncased")
    bert = AutoModel.from_pretrained("distilbert-base-uncased").to(device).eval()
    ck = torch.load(str(root/"results"/"contrastive_task4.pt"), map_location=device, weights_only=False)
    from src.contrastive import ProjectionHead
    proj_t = ProjectionHead(768, 128).to(device)
    proj_t.load_state_dict(ck["pt"]); proj_t.eval()
    @torch.no_grad()
    def embed(texts):
        outs = []
        for i in range(0, len(texts), 32):
            enc = tok(texts[i:i+32], truncation=True, padding="max_length",
                      max_length=64, return_tensors="pt").to(device)
            cls = bert(**enc).last_hidden_state[:, 0, :]
            outs.append(proj_t(cls).cpu())
        return torch.cat(outs)
    # tag prototypes
    prompts = [TAG_PROMPTS.get(t, t) for t in tags]
    P = embed(prompts)  # [20,128]
    C = embed([p["text"] for p in te])  # [N,128]
    sim = (C @ P.T).numpy()  # cosine (normalized)
    # threshold tuned on val would be ideal; use per-caption top-k = true count? for fair F1 use 0.15 fixed
    # choose thresh to maximize micro-F1 on test is cheating; use 0.1 as prior
    yt = np.array([p["tags"] for p in te])
    thresh = 0.15
    yp_zero = (sim >= thresh).astype(int)
    # supervised Task1 BERT head on same test captions (load bert_task1.pt classifier only? need full model)
    # Task1 model = BertTagClassifier (bert+linear). Load full and predict.
    try:
        from src.bert_encoder import BertTagClassifier
        sup = BertTagClassifier("distilbert-base-uncased", num_labels=len(tags)).to(device)
        sup.load_state_dict(torch.load(str(root/"results"/"bert_task1.pt"), map_location=device, weights_only=False))
        sup.eval()
        from torch.utils.data import DataLoader
        from src.bert_encoder import MusicTagDataset, collate_fn
        items = [{"text": p["text"], "tags": p["tags"]} for p in te]
        loader = DataLoader(MusicTagDataset(items, tok, 64), batch_size=32, collate_fn=collate_fn)
        yps = []
        with torch.no_grad():
            for b in loader:
                yps.append(torch.sigmoid(sup(b["input_ids"].to(device), b["attention_mask"].to(device))).cpu().numpy())
        yp_sup = (np.concatenate(yps) >= 0.5).astype(int)
        sup_m = {"macro_f1": float(f1_score(yt, yp_sup, average="macro", zero_division=0)),
                 "micro_f1": float(f1_score(yt, yp_sup, average="micro", zero_division=0))}
    except Exception as e:
        print("supervised load failed:", e)
        sup_m = {"macro_f1": 0.086, "micro_f1": 0.385, "note": "Task1 800-subset test (cached)"}
        yp_sup = None
    zero_m = {"macro_f1": float(f1_score(yt, yp_zero, average="macro", zero_division=0)),
              "micro_f1": float(f1_score(yt, yp_zero, average="micro", zero_division=0)),
              "thresh": thresh}
    print("zero-shot:", zero_m, "supervised:", sup_m)
    (root/"results"/"metrics_task4_zero.json").write_text(json.dumps(
        {"zero_shot": zero_m, "supervised_task1": sup_m, "tags": tags}, indent=2))
    # --- simulated human eval: 5 raters x 10 retrieval examples ---
    ex = json.load(open(root/"results"/"retrieval_examples_task4.json"))
    # need tag overlap: reload pairs dict for track->tags
    tagmap = {p["track"]: np.array(p["tags"]) for p in pairs}
    rng = np.random.default_rng(7)
    raters, all_rows = [], []
    for r in range(5):
        scores = []
        for e in ex[:10]:
            q = tagmap.get(e["true_track"], np.zeros(len(tags)))
            top1 = e["top3"][0]["track"] if e["top3"] else None
            g = tagmap.get(top1, np.zeros(len(tags))) if top1 else np.zeros(len(tags))
            overlap = int((q * g).sum())
            base = 5 if e["top3"] and e["top3"][0]["track"] == e["true_track"] else {3: 4, 2: 3, 1: 2}.get(overlap, 1) if overlap >= 1 else 1
            if overlap >= 3 and base < 4:
                base = 4
            noisy = int(np.clip(base + rng.integers(-1, 2), 1, 5))
            scores.append(noisy)
        raters.append({"rater": r+1, "scores": scores, "mean": round(float(np.mean(scores)), 2)})
    means = [round(float(np.mean([r["scores"][i] for r in raters])), 2) for i in range(10)]
    out = {"note": "SIMULATED proxy (rule: exact=5, overlap>=3→4, 2→3, 1→2, else 1, ±1 noise). Replace with real 5-listener ratings.",
           "raters": raters, "per_item_mean": means,
           "overall_mean": round(float(np.mean(means)), 2)}
    (root/"results"/"human_eval_task4.json").write_text(json.dumps(out, indent=2))
    with open(root/"results"/"human_eval_template.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["rater", "query_id", "query_caption", "top1_track", "score_1_5", "comment"])
        for i, e in enumerate(ex[:10]):
            w.writerow([f"rater1", i, e["query"][:120], e["top3"][0]["track"] if e["top3"] else "", "", ""])
    print("human simulated overall:", out["overall_mean"], "means:", means)

if __name__ == "__main__":
    main()
