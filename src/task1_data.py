"""Task 1 data prep: real CSV or synthetic MagnaTagATune-style fallback.

Real format (if you download MTT): data/raw/mtt_clips.csv with columns:
  clip_id, text (tags/caption joined), tag_0..tag_K-1  (0/1)
or MusicCaps: data/raw/musiccaps.csv with columns: caption, tags...

Fallback: generates demo_tags (200 clips, 10 tags) mimicking:
  tags = [rock,pop,jazz,classical,electronic,happy,sad,guitar,piano,vocal]
Splits 70/15/15 -> data/splits/task1_{train,val,test}.json
"""
from __future__ import annotations
import json
import random
from pathlib import Path

TAGS_10 = ["rock", "pop", "jazz", "classical", "electronic",
           "happy", "sad", "guitar", "piano", "vocal"]

_TEMPLATES = [
    ("upbeat {g} song with electric {i} and {m} vocals", ["rock", "happy", "guitar"]),
    ("mellow {g} ballad with soft {i} and {m} mood", ["pop", "sad", "piano"]),
    ("smooth {g} improvisation with {i} solo", ["jazz", "piano"]),
    ("orchestral {g} piece with {i} ensemble", ["classical", "piano"]),
    ("driving {g} track with synth and {i}", ["electronic", "happy"]),
    ("acoustic {g} tune featuring {i} and {m} singing", ["pop", "guitar", "vocal"]),
    ("melancholic {g} melody with {i}", ["sad", "piano", "classical"]),
    ("energetic {g} anthem with loud {i}", ["rock", "happy", "guitar", "vocal"]),
]

_FILL = {"g": ["rock", "pop", "jazz", "classical", "electronic"],
         "i": ["guitar", "piano", "drums", "synth", "strings"],
         "m": ["female", "male", "soft", "powerful"]}


def _synth_text() -> tuple[str, list[str]]:
    t, base = random.choice(_TEMPLATES)
    txt = t.format(g=random.choice(_FILL["g"]), i=random.choice(_FILL["i"]),
                   m=random.choice(_FILL["m"]))
    extra = random.choice([[], ["vocal"], ["happy"], ["sad"]])
    return txt, sorted(set(base + extra))


def make_synthetic(n: int = 240, seed: int = 42) -> list[dict]:
    random.seed(seed)
    items = []
    for i in range(n):
        txt, labs = _synth_text()
        # add variation so BERT must generalize
        txt = f"{txt} (clip {i % 50})"
        tags = [1 if t in labs else 0 for t in TAGS_10]
        # ensure at least 1 tag
        if sum(tags) == 0:
            tags[random.randrange(len(tags))] = 1
        items.append({"id": f"demo_{i:04d}", "text": txt, "tags": tags})
    return items


def try_load_real(raw_dir: Path):
    for name in ("mtt_clips.csv", "musiccaps.csv", "tags.csv"):
        p = raw_dir / name
        if p.exists():
            import csv
            items = []
            with open(p, newline="", encoding="utf-8") as f:
                r = csv.DictReader(f)
                tag_cols = [c for c in (r.fieldnames or []) if c.startswith("tag_")]
                for row in r:
                    txt = row.get("text") or row.get("caption") or ""
                    if not txt:
                        continue
                    tags = [int(row.get(c, 0) or 0) for c in tag_cols]
                    items.append({"id": row.get("clip_id", str(len(items))),
                                  "text": txt, "tags": tags})
            if items:
                tag_names = [c.replace("tag_", "") for c in tag_cols] or TAGS_10
                return items, tag_names
    return None, None


def prepare(task_split_dir: Path, raw_dir: Path, n: int = 240, seed: int = 42):
    task_split_dir.mkdir(parents=True, exist_ok=True)
    real, names = try_load_real(raw_dir)
    if real:
        items, tag_names = real, names
        print(f"Using REAL data: {len(items)} clips, {len(tag_names)} tags")
    else:
        items, tag_names = make_synthetic(n, seed), TAGS_10
        print(f"No real CSV found -> synthetic demo: {len(items)} clips, {len(tag_names)} tags")
    random.Random(seed).shuffle(items)
    k = len(items)
    n_tr, n_va = int(k * 0.7), int(k * 0.15)
    splits = {"train": items[:n_tr], "val": items[n_tr:n_tr + n_va],
              "test": items[n_tr + n_va:]}
    for s, v in splits.items():
        (task_split_dir / f"task1_{s}.json").write_text(json.dumps(v, indent=2))
    (task_split_dir / "task1_tags.json").write_text(json.dumps(tag_names, indent=2))
    print({s: len(v) for s, v in splits.items()}, "->", task_split_dir)
    return splits, tag_names


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--splits", default="data/splits")
    ap.add_argument("--n", type=int, default=240)
    args = ap.parse_args()
    prepare(Path(args.splits), Path(args.raw), n=args.n)
