"""Audio feature extraction + segmentation (PDF Sec.3).

Pipeline:
  load -> resample 22050 Hz -> log-mel 128 / chroma 12 / MFCC
  -> per-track normalize -> fixed or beat-sync segmentation.

Usage:
  python -m src.audio_features --config config.yaml --input data/raw --output data/processed
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np

try:
    import librosa
    import soundfile as sf
    _HAS_LIBROSA = True
except ImportError:
    _HAS_LIBROSA = False

SR = 22050
N_MELS = 128
N_FFT = 2048
HOP = 512


def load_audio(path: str | Path, sr: int = SR) -> tuple[np.ndarray, int]:
    if not _HAS_LIBROSA:
        raise ImportError("librosa+soundfile required: pip install -r requirements.txt")
    y, _ = librosa.load(str(path), sr=sr, mono=True)
    return y.astype(np.float32), sr


def extract_log_mel(y: np.ndarray, sr: int = SR, n_mels: int = N_MELS,
                    n_fft: int = N_FFT, hop_length: int = HOP) -> np.ndarray:
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft,
                                       hop_length=hop_length, n_mels=n_mels)
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.astype(np.float32)  # [n_mels, T]


def extract_chroma(y: np.ndarray, sr: int = SR, hop_length: int = HOP) -> np.ndarray:
    C = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_chroma=12)
    return C.astype(np.float32)  # [12, T]


def extract_mfcc(y: np.ndarray, sr: int = SR, n_mfcc: int = 20,
                 hop_length: int = HOP) -> np.ndarray:
    M = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    return M.astype(np.float32)  # [n_mfcc, T]


def normalize_per_track(feat: np.ndarray) -> np.ndarray:
    mu = feat.mean(axis=1, keepdims=True)
    sd = feat.std(axis=1, keepdims=True) + 1e-6
    return ((feat - mu) / sd).astype(np.float32)


def segment_fixed(feat: np.ndarray, sr: int = SR, hop_length: int = HOP,
                  window_sec: float = 5.0, hop_sec: float | None = None) -> np.ndarray:
    """Slice time-axis feature [F, T] into [N, F, W] windows."""
    if hop_sec is None:
        hop_sec = window_sec
    w_frames = max(1, int(round(window_sec * sr / hop_length)))
    h_frames = max(1, int(round(hop_sec * sr / hop_length)))
    F, T = feat.shape
    segs = []
    for s in range(0, max(1, T - w_frames + 1), h_frames):
        win = feat[:, s:s + w_frames]
        if win.shape[1] < w_frames:  # pad last
            pad = np.zeros((F, w_frames - win.shape[1]), dtype=np.float32)
            win = np.concatenate([win, pad], axis=1)
        segs.append(win)
    if not segs:
        pad = np.zeros((F, w_frames), dtype=np.float32)
        w = min(T, w_frames)
        pad[:, :w] = feat[:, :w]
        segs = [pad]
    return np.stack(segs).astype(np.float32)


def segment_beat_sync(y: np.ndarray, sr: int, feat: np.ndarray,
                      hop_length: int = HOP, agg: str = "mean") -> np.ndarray:
    """Pool frame features between detected beats. Returns [N_beats, F]."""
    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length, units="frames")
    if len(beats) < 2:
        return feat.mean(axis=1, keepdims=True).T  # single pooled vector
    beats = np.clip(beats, 0, feat.shape[1] - 1)
    pooled = []
    bounds = [0, *beats.tolist(), feat.shape[1]]
    for a, b in zip(bounds[:-1], bounds[1:]):
        if b <= a:
            continue
        seg = feat[:, a:b]
        pooled.append(seg.max(axis=1) if agg == "max" else seg.mean(axis=1))
    return np.stack(pooled).astype(np.float32)


def process_track(path: str | Path, sr: int = SR, n_mels: int = 128,
                  n_mfcc: int = 20, window_sec: float = 5.0,
                  hop_sec: float | None = None, mode: str = "fixed") -> dict:
    y, sr = load_audio(path, sr=sr)
    mel = normalize_per_track(extract_log_mel(y, sr, n_mels=n_mels))
    chroma = normalize_per_track(extract_chroma(y, sr))
    mfcc = normalize_per_track(extract_mfcc(y, sr, n_mfcc=n_mfcc))
    if mode == "beat_sync":
        seg_mfcc = segment_beat_sync(y, sr, mfcc)          # [N, F]
        seg_chroma = segment_beat_sync(y, sr, chroma)
        seg_mel = segment_beat_sync(y, sr, mel)
        segments = {"mfcc": seg_mfcc, "chroma": seg_chroma, "mel": seg_mel}
    else:
        segments = {
            "mfcc": segment_fixed(mfcc, sr, HOP, window_sec, hop_sec),
            "chroma": segment_fixed(chroma, sr, HOP, window_sec, hop_sec),
            "mel": segment_fixed(mel, sr, HOP, window_sec, hop_sec),
        }
    return {"mel": mel, "chroma": chroma, "mfcc": mfcc,
            "segments": segments, "sr": sr,
            "track": Path(path).stem}


def batch_process(input_dir: str | Path, output_dir: str | Path,
                  window_sec: float = 5.0, mode: str = "fixed",
                  exts: tuple = (".wav", ".mp3", ".au", ".flac", ".ogg")) -> list[str]:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = [p for p in input_dir.rglob("*") if p.suffix.lower() in exts]
    saved = []
    for f in files:
        dest = output_dir / f"{f.stem}.npz"
        if dest.exists():
            saved.append(str(f))
            continue
        try:
            out = process_track(f, window_sec=window_sec, mode=mode)
            # save lightweight cache: pooled segment means + full feats
            np.savez_compressed(output_dir / f"{f.stem}.npz",
                                mel=out["mel"], chroma=out["chroma"], mfcc=out["mfcc"])
            saved.append(str(f))
        except Exception as e:
            print(f"[skip] {f}: {e}")
    (output_dir / "manifest.json").write_text(json.dumps(saved, indent=2))
    print(f"Processed {len(saved)}/{len(files)} -> {output_dir}")
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--input", default="data/raw")
    ap.add_argument("--output", default="data/processed")
    ap.add_argument("--window_sec", type=float, default=5.0)
    ap.add_argument("--mode", default="fixed", choices=["fixed", "beat_sync"])
    args = ap.parse_args()
    # optional yaml override
    try:
        import yaml
        cfg = yaml.safe_load(open(args.config)) if Path(args.config).exists() else {}
        a = cfg.get("audio", {}); s = cfg.get("segmentation", {})
        sr = a.get("sr", SR)
        print(f"Config sr={sr} mode={s.get('mode', args.mode)}")
    except Exception:
        pass
    batch_process(args.input, args.output, window_sec=args.window_sec, mode=args.mode)


if __name__ == "__main__":
    main()
