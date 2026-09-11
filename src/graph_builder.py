"""Music structure graph construction (PDF Sec.3.3 + Task 2).

Two graph types:
 1. Segment graph: nodes=time segments, edges=temporal adjacency + cosine sim > tau.
 2. Chord-transition graph: nodes=24 (12 major + 12 minor), edges=observed transitions.

Node features init from audio segment embeddings (mel/chroma/MFCC pooled).
Output: torch_geometric.data.Data (x, edge_index, edge_attr) + .pt / .json samples.

Usage:
  python -m src.graph_builder --config config.yaml --input data/processed --output data/processed/graphs
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import torch
    from torch_geometric.data import Data
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False

CHORD_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
CHORDS_24 = [f"{n}" for n in CHORD_NAMES] + [f"{n}m" for n in CHORD_NAMES]

# 12-dim major/minor templates (Krumhansl-ish binary)
_MAJOR = np.array([1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)
_MINOR = np.array([1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0], dtype=np.float32)


def _roll(template: np.ndarray, k: int) -> np.ndarray:
    return np.roll(template, k)


TEMPLATES_24 = np.stack([_roll(_MAJOR, k) for k in range(12)] +
                        [_roll(_MINOR, k) for k in range(12)])  # [24,12]


def estimate_chords(chroma_seq: np.ndarray) -> list[int]:
    """Nearest-template chord per frame/segment. chroma_seq: [N,12] or [12,T]."""
    c = np.asarray(chroma_seq, dtype=np.float32)
    if c.ndim != 2:
        raise ValueError("chroma_seq must be 2D")
    if c.shape[0] == 12 and c.shape[1] != 12:
        c = c.T  # -> [N,12]
    # cosine to templates
    cn = c / (np.linalg.norm(c, axis=1, keepdims=True) + 1e-8)
    tn = TEMPLATES_24 / (np.linalg.norm(TEMPLATES_24, axis=1, keepdims=True) + 1e-8)
    sim = cn @ tn.T
    return sim.argmax(axis=1).tolist()


def build_chord_transition_graph(chord_ids: list[int], num_nodes: int = 24):
    """Count transitions i->j. Returns edge_index [2,E], edge_weight [E]."""
    counts: dict[tuple[int, int], int] = {}
    for a, b in zip(chord_ids[:-1], chord_ids[1:]):
        counts[(a, b)] = counts.get((a, b), 0) + 1
    if not counts:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    edges = sorted(counts)
    ei = np.array(edges, dtype=np.int64).T
    ew = np.array([counts[e] for e in edges], dtype=np.float32)
    ew = ew / ew.max()  # normalize
    return ei, ew


def _cosine_sim_matrix(X: np.ndarray) -> np.ndarray:
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-8)
    return (Xn @ Xn.T).astype(np.float32)


def build_segment_graph(segment_feats: np.ndarray, tau: float = 0.7,
                        add_temporal: bool = True):
    """segment_feats: [N,F] pooled per segment (mean over time/freq).

    Edges: (i,i+1) temporal + pairs with cosine > tau.
    Returns edge_index [2,E], edge_weight [E].
    """
    X = np.asarray(segment_feats, dtype=np.float32)
    N = X.shape[0]
    if N == 1:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    S = _cosine_sim_matrix(X)
    src, dst, w = [], [], []
    if add_temporal:
        for i in range(N - 1):
            src += [i, i + 1]; dst += [i + 1, i]; w += [1.0, 1.0]
    iu = np.triu_indices(N, k=2)
    for i, j in zip(iu[0], iu[1]):
        if S[i, j] > tau:
            src += [i, j]; dst += [j, i]; w += [float(S[i, j]), float(S[i, j])]
    # de-dup keeping max weight
    best: dict[tuple[int, int], float] = {}
    for s, d, vv in zip(src, dst, w):
        best[(s, d)] = max(best.get((s, d), 0.0), vv)
    if not best:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.float32)
    pairs = sorted(best)
    ei = np.array(pairs, dtype=np.int64).T
    ew = np.array([best[p] for p in pairs], dtype=np.float32)
    return ei, ew


def pooled_node_features(seg_dict: dict) -> np.ndarray:
    """Mean-pool each segment window to one vector. Accepts [N,F,W] or [N,F]."""
    feats = []
    for key in ("mfcc", "chroma", "mel"):
        if key not in seg_dict:
            continue
        v = np.asarray(seg_dict[key])
        if v.ndim == 3:
            v = v.mean(axis=2)  # [N,F]
        feats.append(v)
    if not feats:
        raise ValueError("seg_dict needs mfcc/chroma/mel")
    N = min(f.shape[0] for f in feats)
    return np.concatenate([f[:N] for f in feats], axis=1).astype(np.float32)  # [N, Fsum]


def to_pyg(x: np.ndarray, edge_index: np.ndarray,
           edge_weight: np.ndarray, y=None, track: str = ""):
    if not _HAS_PYG:
        return {"x": x, "edge_index": edge_index,
                "edge_weight": edge_weight, "y": y, "track": track}
    return Data(x=torch.from_numpy(x),
                edge_index=torch.from_numpy(edge_index).long() if edge_index.size else torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.from_numpy(edge_weight) if edge_weight.size else None,
                y=None if y is None else torch.tensor(y),
                track=track)


def graph_from_npz(npz_path: str | Path, tau: float = 0.7,
                   add_temporal: bool = True, chord_pool: int = 1):
    d = np.load(str(npz_path), allow_pickle=True)
    mel, chroma, mfcc = d["mel"], d["chroma"], d["mfcc"]
    # Segment-level pooling: split time axis into ~5s chunks for graph nodes
    # Simple: 43 frames/sec approx (22050/512); 5s ~= 215 frames
    def chunk_mean(F: np.ndarray, w: int = 215) -> np.ndarray:
        T = F.shape[1]
        outs = [F[:, i:i + w].mean(axis=1) for i in range(0, T, w)]
        return np.stack(outs).astype(np.float32)  # [N, Fdim]
    seg_m = chunk_mean(mfcc); seg_c = chunk_mean(chroma)
    X = np.concatenate([seg_m, seg_c], axis=1)
    ei, ew = build_segment_graph(X, tau=tau, add_temporal=add_temporal)
    chords = estimate_chords(seg_c)
    cei, cew = build_chord_transition_graph(chords)
    g = to_pyg(X, ei, ew, track=Path(npz_path).stem)
    meta = {"track": Path(npz_path).stem, "num_nodes": int(X.shape[0]),
            "num_edges": int(ei.shape[1]), "chords": chords,
            "chord_edges": cei.tolist(), "chord_weights": cew.tolist()}
    return g, meta


def graph_coherence_score(x: np.ndarray, edge_index: np.ndarray, tau: float = 0.7) -> float:
    """PDF Sec.6 Sgraph = fraction of edges with cos(hi,hj) > tau."""
    if edge_index.size == 0:
        return 0.0
    S = _cosine_sim_matrix(np.asarray(x, dtype=np.float32))
    hits = sum(1 for i, j in edge_index.T if S[i, j] > tau)
    return hits / max(1, edge_index.shape[1])


def batch_build(input_dir: str | Path, output_dir: str | Path,
                tau: float = 0.7, add_temporal: bool = True) -> list[dict]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metas = []
    for npz in sorted(input_dir.glob("*.npz")):
        g, meta = graph_from_npz(npz, tau=tau, add_temporal=add_temporal)
        if _HAS_PYG:
            import torch as _t
            _t.save(g, output_dir / f"{npz.stem}.pt")
        (output_dir / f"{npz.stem}.json").write_text(json.dumps(meta, indent=2))
        metas.append(meta)
    (output_dir / "manifest.json").write_text(json.dumps(metas, indent=2))
    print(f"Built {len(metas)} graphs -> {output_dir}")
    return metas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default="data/processed")
    ap.add_argument("--output", default="data/processed/graphs")
    ap.add_argument("--tau", type=float, default=0.7)
    args = ap.parse_args()
    try:
        import yaml
        if Path(args.config).exists():
            cfg = yaml.safe_load(open(args.config))
            args.tau = cfg.get("graph", {}).get("segment_sim_threshold", args.tau)
    except Exception:
        pass
    batch_build(args.input, args.output, tau=args.tau)


if __name__ == "__main__":
    main()
