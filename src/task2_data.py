"""Task 2 data: GTZAN/FMA genre graphs or synthetic fallback.

Real: data/raw/gtzan/<genre>/*.wav (10 GTZAN genres) or data/raw/fma_small/*/*.mp3
  -> process_track -> graph_from_npz + mel for CNN.
Fallback: synthetic genre-conditioned graphs (fast, CPU-friendly):
  10 genres, N=12 nodes, F=32, edges via segment-graph rule.
"""
from __future__ import annotations
import random
from pathlib import Path
import numpy as np

GENRES_10 = ["blues", "classical", "country", "disco", "hiphop",
             "jazz", "metal", "pop", "reggae", "rock"]

def _synthetic_one(genre_idx: int, rng: np.random.Generator,
                   n_nodes: int = 12, feat_dim: int = 32):
    proto = np.zeros(feat_dim, dtype=np.float32)
    proto[genre_idx * 3:(genre_idx * 3 + 6)] = 2.0  # genre signature block
    proto += rng.normal(0, 0.3, feat_dim).astype(np.float32)
    X = np.stack([proto + rng.normal(0, 0.5, feat_dim).astype(np.float32)
                  for _ in range(n_nodes)])
    # smooth temporally so temporal edges are meaningful
    for i in range(1, n_nodes):
        X[i] = 0.7 * X[i] + 0.3 * X[i - 1]
    from src.graph_builder import build_segment_graph
    ei, ew = build_segment_graph(X, tau=0.6, add_temporal=True)
    # synthetic mel: genre band energy + noise
    mel = rng.normal(0, 0.5, (128, 64)).astype(np.float32)
    mel[genre_idx * 12:(genre_idx + 1) * 12, :] += 1.5
    return X.astype(np.float32), ei, ew, mel, genre_idx

def make_synthetic(n_per_genre: int = 30, seed: int = 42):
    rng = np.random.default_rng(seed)
    items = []
    for g, _ in enumerate(GENRES_10):
        for i in range(n_per_genre):
            X, ei, ew, mel, y = _synthetic_one(g, rng)
            items.append({"x": X, "edge_index": ei, "edge_weight": ew,
                          "mel": mel, "y": y, "track": f"syn_{GENRES_10[g]}_{i}"})
    random.Random(seed).shuffle(items)
    return items

def try_load_gtzan(raw_dir: Path):
    cands = list(raw_dir.glob("gtzan/*/*.wav")) + list(raw_dir.glob("GTZAN/*/*.wav"))
    if not cands:
        cands = list(raw_dir.glob("*.wav"))[:20]
    if not cands:
        return None
    from src.audio_features import process_track
    from src.graph_builder import graph_from_npz
    import tempfile, os
    items = []
    for wav in cands[:200]:
        genre = wav.parent.name.lower()
        if genre not in GENRES_10:
            continue
        try:
            out = process_track(wav, window_sec=5.0, mode="fixed")
            # save temp npz for graph fn
            import numpy as _np
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tf:
                _np.savez_compressed(tf.name, mel=out["mel"], chroma=out["chroma"], mfcc=out["mfcc"])
                tmp = tf.name
            g, meta = graph_from_npz(tmp, tau=0.7)
            os.unlink(tmp)
            import torch as _t
            x = g.x.numpy() if hasattr(g, "x") else g["x"]
            ei = g.edge_index.numpy() if hasattr(g, "edge_index") else g["edge_index"]
            mel = out["mel"][:, :64] if out["mel"].shape[1] >= 64 else out["mel"]
            items.append({"x": x.astype(_np.float32), "edge_index": ei,
                          "edge_weight": _np.ones((ei.shape[1],), dtype=_np.float32),
                          "mel": mel.astype(_np.float32), "y": GENRES_10.index(genre),
                          "track": wav.stem})
        except Exception as e:
            print(f"[skip] {wav}: {e}")
    return items if items else None

def prepare(raw_dir: Path, n_per_genre: int = 30, seed: int = 42):
    real = try_load_gtzan(raw_dir)
    if real:
        print(f"Using REAL GTZAN: {len(real)} clips")
        return real, GENRES_10
    syn = make_synthetic(n_per_genre, seed)
    print(f"No GTZAN wav found -> synthetic: {len(syn)} graphs, 10 genres")
    return syn, GENRES_10
