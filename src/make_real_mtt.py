"""Build REAL mtt_clips.csv from MusicCaps captions -> MTT-style tags.

Sources (already downloaded):
  data/raw/musiccaps-public.csv  (5521 captions, aspect_list)
  data/raw/annotations_final.csv (MTT 188-tag vocab, for reference)

Strategy (PDF Task1: MusicCaps caption -> tag proxy):
  text = caption (real, written by musicians)
  tags = keyword match of caption+aspect_list against MTT-overlapping vocabulary.
  Keeps top tags with enough support for stable training.

Output: data/raw/mtt_clips.csv  (clip_id,text,tag_<name>...)
"""
from __future__ import annotations
import ast
import csv
from collections import Counter
from pathlib import Path

# Candidate tags overlapping MTT top tags + MusicCaps aspects. key -> variants to match.
TAG_VARIANTS = {
    "guitar": ["guitar"],
    "piano": ["piano"],
    "drums": ["drum", "drums"],
    "violin": ["violin"],
    "strings": ["strings", "string section"],
    "synth": ["synth"],
    "vocal": ["vocal", "singing", "sings", "singer"],
    "female": ["female", "woman", "girl"],
    "male": ["male", "man ", "men "],
    "rock": ["rock"],
    "pop": ["pop"],
    "jazz": ["jazz"],
    "classical": ["classical", "orchestral", "orchestra", "baroque"],
    "electronic": ["electronic", "electronica", "electro", "techno", "house music", "trance"],
    "hiphop": ["hip hop", "hip-hop", "rap"],
    "sad": ["sad", "melancholic", "mournful", "sorrow"],
    "happy": ["happy", "cheerful", "upbeat", "joyful"],
    "slow": ["slow"],
    "fast": ["fast", "uptempo", "up-tempo", "energetic"],
    "loud": ["loud", "noisy", "aggressive"],
    "soft": ["soft", "mellow", "gentle", "calm"],
    "dance": ["dance", "danceable"],
}

def parse_aspect(s: str) -> list[str]:
    try:
        v = ast.literal_eval(s)
        if isinstance(v, list):
            return [str(x).lower() for x in v]
    except Exception:
        pass
    return [s.lower()]

def main():
    root = Path(__file__).resolve().parent.parent
    src = root / "data" / "raw" / "musiccaps-public.csv"
    dst = root / "data" / "raw" / "mtt_clips.csv"
    assert src.exists(), f"missing {src}"
    rows = list(csv.DictReader(open(src, encoding="utf-8")))
    print(f"MusicCaps rows: {len(rows)}")
    # count support
    cnt = Counter()
    parsed = []
    for r in rows:
        blob = (r["caption"] + " " + " ".join(parse_aspect(r["aspect_list"]))).lower()
        hits = {t: any(v in blob for v in vs) for t, vs in TAG_VARIANTS.items()}
        parsed.append((r["ytid"], r["caption"], hits))
        for t, h in hits.items():
            if h:
                cnt[t] += 1
    print("support:", dict(cnt.most_common()))
    # keep tags with >=120 support, max 20
    kept = [t for t, c in cnt.most_common() if c >= 120][:20]
    print(f"kept {len(kept)} tags:", kept)
    with open(dst, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["clip_id", "text"] + [f"tag_{t}" for t in kept])
        w.writeheader()
        for ytid, cap, hits in parsed:
            w.writerow({"clip_id": ytid, "text": cap,
                        **{f"tag_{t}": int(bool(hits[t])) for t in kept}})
    print(f"wrote {dst} ({len(parsed)} clips, {len(kept)} tags)")
    # also save tag list
    (root / "data" / "raw" / "real_tags.txt").write_text("\n".join(kept))

if __name__ == "__main__":
    main()
