"""Direct download 20 GTZAN wavs via hf_hub (no torchcodec)."""
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download
import shutil

REPO = "storylinez/gtzan-music-genre-dataset"
GENRES = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]

def main(n_per=2):
    root = Path(__file__).resolve().parent.parent
    out_base = root / "data" / "raw" / "gtzan"
    files = list(list_repo_files(REPO, repo_type="dataset"))
    print(f"repo files: {len(files)}")
    wavs = [f for f in files if f.lower().endswith(".wav")]
    print(f"wavs: {len(wavs)} sample: {wavs[:3]}")
    from collections import Counter
    cnt = Counter()
    saved = []
    for f in wavs:
        low = f.lower()
        g = next((x for x in GENRES if f"/{x}/" in low or f.startswith(x+"/")), None)
        if g is None:
            # try infer from filename like blues.00000.wav
            base = Path(f).name.lower()
            g = next((x for x in GENRES if base.startswith(x)), None)
        if g is None or cnt[g] >= n_per:
            continue
        local = hf_hub_download(REPO, f, repo_type="dataset")
        dest_dir = out_base / g
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / Path(f).name
        shutil.copy(local, dest)
        cnt[g] += 1
        saved.append(str(dest))
        print(f"saved {dest} ({len(saved)}/20)")
        if len(saved) >= n_per*10:
            break
    print("counts:", dict(cnt))

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=2)
    main(ap.parse_args().n)
