"""Fetch 160 missing wavs from storylinez to reach 860 total."""
from pathlib import Path
from huggingface_hub import list_repo_files, hf_hub_download
import shutil, time

root = Path(__file__).resolve().parent.parent / "data" / "raw" / "gtzan"
files = list(list_repo_files("storylinez/gtzan-music-genre-dataset", repo_type="dataset"))
wavs = [f for f in files if f.lower().endswith(".wav")]
local = set([p.name for p in root.rglob("*.wav")])
missing = [f for f in wavs if Path(f).name not in local]
print(f"missing: {len(missing)}")
ok = 0
for i, f in enumerate(sorted(missing), 1):
    try:
        t0 = time.time()
        lp = hf_hub_download("storylinez/gtzan-music-genre-dataset", f, repo_type="dataset")
        genre = f.split("/")[0]
        dest = root / genre / Path(f).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            shutil.copy(lp, dest)
        ok += 1
        print(f"[{i}/{len(missing)}] {f} -> {dest.name} ({time.time()-t0:.1f}s)")
    except Exception as e:
        print(f"FAIL {f}: {e}")
print(f"done {ok}/{len(missing)}")
