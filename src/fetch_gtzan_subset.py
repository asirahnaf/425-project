"""Fetch 30 GTZAN clips (3/genre) via HF streaming. ~40MB.
Saves to data/raw/gtzan/<genre>/*.wav
"""
from __future__ import annotations
from pathlib import Path
from collections import Counter

def main(n_per_genre: int = 3):
    from datasets import load_dataset
    import soundfile as sf
    root = Path(__file__).resolve().parent.parent
    out = root / "data" / "raw" / "gtzan"
    print("streaming marsyas/gtzan ...")
    try:
        ds = load_dataset("marsyas/gtzan", split="train", streaming=True)
    except Exception as e:
        print("marsyas failed, trying storylinez:", e)
        ds = load_dataset("storylinez/gtzan-music-genre-dataset", split="train", streaming=True)
    counts, saved = Counter(), []
    # genre names may be int or str; normalize
    for ex in ds:
        g = ex.get("genre", ex.get("label", "unknown"))
        # datasets ClassLabel may decode to str already or int
        if isinstance(g, int):
            # marsyas order
            names = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
            g = names[g] if 0 <= g < 10 else str(g)
        g = str(g).lower()
        if counts[g] >= n_per_genre:
            continue
        audio = ex.get("audio") or ex
        arr = audio.get("array") if isinstance(audio, dict) else None
        sr = audio.get("sampling_rate", 22050) if isinstance(audio, dict) else 22050
        if arr is None:
            continue
        d = out / g
        d.mkdir(parents=True, exist_ok=True)
        fp = d / f"{g}.{counts[g]:05d}.wav"
        sf.write(str(fp), arr, sr)
        counts[g] += 1
        saved.append(str(fp))
        print(f"saved {fp} ({len(saved)}/30)")
        if len(saved) >= n_per_genre * 10:
            break
    print("counts:", dict(counts), "total:", len(saved))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    main(ap.parse_args().n)
