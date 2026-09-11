"""Task 3 paired data: (graph, caption, multi-labels).

Labels (10): rock,pop,jazz,classical,electronic,happy,sad,guitar,piano,vocal
- Synthetic: genre-conditioned graph (reuse Task2) + template caption.
- Real: 20 GTZAN graphs from data/processed/graphs/*.pt + genre-template caption.
"""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np

LABELS_10 = ["rock","pop","jazz","classical","electronic",
             "happy","sad","guitar","piano","vocal"]

_GENRE2TAGS = {
    "blues": ["sad","guitar","vocal"],
    "classical": ["classical","piano","sad"],
    "country": ["pop","guitar","vocal"],
    "disco": ["pop","happy","electronic"],
    "hiphop": ["pop","happy","vocal"],
    "jazz": ["jazz","piano","happy"],
    "metal": ["rock","loud","guitar"],
    "pop": ["pop","happy","vocal"],
    "reggae": ["happy","guitar","vocal"],
    "rock": ["rock","guitar","happy"],
}
_GENRE2CAP = {
    "blues": "slow blues track with expressive guitar and soulful vocal",
    "classical": "orchestral classical piece with piano and strings",
    "country": "country song with acoustic guitar and vocal storytelling",
    "disco": "upbeat disco track with electronic beat for dancing",
    "hiphop": "hiphop track with rap vocal over electronic drums",
    "jazz": "smooth jazz improvisation with piano solo",
    "metal": "heavy rock track with loud electric guitar",
    "pop": "catchy pop song with happy vocal melody",
    "reggae": "laid-back reggae groove with guitar offbeats",
    "rock": "energetic rock anthem with electric guitar",
}

def _tags_to_vec(tags: list[str]) -> list[int]:
    return [1 if t in tags else 0 for t in LABELS_10]

def make_synthetic_paired(n_per_genre: int = 12, seed: int = 42):
    from src.task2_data import GENRES_10
    rng = np.random.default_rng(seed)
    items = []
    for gi, genre in enumerate(GENRES_10):
        base_tags = _GENRE2TAGS.get(genre, [genre])
        for i in range(n_per_genre):
            # graph via Task2 synthetic single
            from src.task2_data import _synthetic_one
            X, ei, ew, mel, _ = _synthetic_one(gi, rng)
            mood_extra = rng.choice(["happy","sad"], p=[0.6,0.4])
            tags = sorted(set(base_tags + [mood_extra]))
            cap = _GENRE2CAP[genre] + f" (clip {i})"
            items.append({"x": X, "edge_index": ei, "text": cap,
                          "tags": _tags_to_vec(tags), "genre": genre,
                          "track": f"syn3_{genre}_{i}"})
    random.Random(seed).shuffle(items)
    return items

def load_real_gtzan_paired(graph_dir: Path, max_n: int = 150, seed: int = 42):
    import torch
    from collections import defaultdict
    items = []
    for pt in sorted(graph_dir.glob("*.pt")):
        try:
            g = torch.load(str(pt), weights_only=False)
            x = g.x.numpy() if hasattr(g, "x") else g["x"]
            ei = g.edge_index.numpy() if hasattr(g, "edge_index") else g["edge_index"]
            track = pt.stem  # e.g. blues.00000
            genre = track.split(".")[0].lower()
            tags = _GENRE2TAGS.get(genre, [genre])
            cap = _GENRE2CAP.get(genre, f"{genre} track")
            items.append({"x": x.astype(np.float32), "edge_index": ei,
                          "text": cap, "tags": _tags_to_vec(tags),
                          "genre": genre, "track": track})
        except Exception as e:
            print(f"[skip] {pt}: {e}")
    # balanced sample max_n across genres for speed
    if len(items) > max_n:
        by_g = defaultdict(list)
        for it in items:
            by_g[it["genre"]].append(it)
        per = max(1, max_n // max(1, len(by_g)))
        sampled = []
        rnd = random.Random(seed)
        for g, lst in by_g.items():
            rnd.shuffle(lst)
            sampled += lst[:per]
        items = sampled[:max_n]
        random.Random(seed).shuffle(items)
    return items

def prepare(n_per_genre: int = 12, seed: int = 42, max_real: int = 150):
    root = Path(__file__).resolve().parent.parent
    syn = make_synthetic_paired(n_per_genre, seed)
    real = load_real_gtzan_paired(root / "data" / "processed" / "graphs", max_n=max_real, seed=seed)
    items = syn + real
    random.Random(seed).shuffle(items)
    print(f"paired: synthetic {len(syn)} + real {len(real)} = {len(items)}")
    return items, LABELS_10
