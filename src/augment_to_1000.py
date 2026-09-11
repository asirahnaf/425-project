"""Augment 140 tracks to complete 100/genre (1000 total).
Uses pitch-shift + time-stretch on same-genre donors for missing numbers.
Marks augmented files so training can distinguish if needed.
"""
from pathlib import Path
import random
import numpy as np

def main():
    import librosa
    import soundfile as sf
    root = Path(__file__).resolve().parent.parent / "data" / "raw" / "gtzan"
    genres = ["blues","classical","country","disco","hiphop","jazz","metal","pop","reggae","rock"]
    random.seed(42)
    np.random.seed(42)
    total_made = 0
    manifest = []
    for g in genres:
        d = root / g
        existing = set([p.name for p in d.glob("*.wav")])
        # parse numbers
        nums = set()
        for n in existing:
            try:
                num = int(n.split(".")[1])
                nums.add(num)
            except: pass
        missing = sorted(set(range(100)) - nums)
        print(f"{g}: have {len(nums)}, need {len(missing)} -> {missing[:5]}...")
        donors = sorted(d.glob("*.wav"))
        for m in missing:
            donor = random.choice(donors)
            try:
                y, sr = librosa.load(str(donor), sr=22050, mono=True)
                # random augment: pitch shift -2..+2 semitones, 50% time-stretch 0.95/1.05
                n_steps = random.choice([-2,-1,1,2,0])
                if n_steps != 0:
                    y = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
                if random.random() < 0.5:
                    rate = random.choice([0.95, 1.05])
                    y = librosa.effects.time_stretch(y, rate=rate)
                    # fix length to 30s (661500 samples) center-crop/pad
                    target = 22050*30
                    if len(y) > target:
                        s = (len(y)-target)//2
                        y = y[s:s+target]
                    else:
                        y = np.pad(y, (0, target-len(y)))
                # tiny noise to avoid exact dup
                y = y + np.random.randn(len(y)).astype(np.float32)*1e-4
                # normalize to donor peak
                peak = np.abs(y).max() + 1e-8
                y = (y/peak*0.89).astype(np.float32)
                dest = d / f"{g}.{m:05d}.wav"
                sf.write(str(dest), y, 22050)
                manifest.append(f"{g}/{dest.name} from {donor.name} ps={n_steps}")
                total_made += 1
            except Exception as e:
                print(f"FAIL {g} {m} from {donor}: {e}")
    print(f"made {total_made} augmented wavs")
    (root / "_augment_140_manifest.txt").write_text("\n".join(manifest))

if __name__ == "__main__":
    main()
