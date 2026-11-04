#!/usr/bin/env python3
"""End-to-end inference: raw audio (.wav) -> AU12 / AU25 / AU26 intensity curves (Korean-trained).

Usage:
    python infer.py --wav path/to/speech.wav --out predictions.npz

Pipeline: audio_feature_extractor.extract_wavlm() -> WavLM Base+ final hidden state (768-d, 25 fps),
then the mean of 10 TCN checkpoints (one per K-MIND cross-validation fold, `checkpoints/kor_tcn_fold*.pt`).
Each checkpoint carries its OWN feature mean/std (`mean`, `std`) and output scaling (`ymean`, `ystd`),
all fit on that fold's training set; they are applied per checkpoint, never shared.
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import audio_feature_extractor as afe  # noqa: E402
from model import SpeechToAU  # noqa: E402

CKPT_DIR = os.path.join(HERE, "checkpoints")
TARGET_FPS = 25
WIN = 400  # frames; the models were trained and evaluated on 400-frame windows

_CKPTS = {}


def load_checkpoints(device):
    if device not in _CKPTS:
        models = []
        for p in sorted(glob.glob(os.path.join(CKPT_DIR, "kor_tcn_fold*.pt"))):
            ck = torch.load(p, map_location=device, weights_only=False)
            m = SpeechToAU(input_dim=ck["input_dim"], num_aus=len(ck["target_aus"])).to(device)
            m.load_state_dict(ck["model_state"])
            m.eval()
            models.append((m, ck["mean"], ck["std"], ck["ymean"], ck["ystd"], ck["target_aus"]))
        if not models:
            raise FileNotFoundError(f"no kor_tcn_fold*.pt in {CKPT_DIR}")
        _CKPTS[device] = models
    return _CKPTS[device]


def predict(wav_path, device=None):
    """Returns dict: 'timestamp_sec' [T] plus float32[T] arrays 'AU12', 'AU25', 'AU26',
    averaged over the checkpoints and clipped to the OpenFace 0-5 intensity range."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    feats = afe.extract_wavlm(wav_path, device=device)  # [T, 768]
    T = feats.shape[0]
    models = load_checkpoints(device)
    aus = models[0][5]

    preds = []
    for m, mean, std, ymean, ystd, target_aus in models:
        assert target_aus == aus
        x = torch.from_numpy(((feats - mean) / std).astype(np.float32))
        out = []
        for s in range(0, T, WIN):
            with torch.no_grad():
                out.append(m(x[s:s + WIN].unsqueeze(0).to(device)).squeeze(0).cpu().numpy())
        preds.append(np.concatenate(out) * ystd + ymean)
    pred = np.mean(preds, axis=0)  # [T, 3]

    result = {"timestamp_sec": np.arange(T) / TARGET_FPS}
    for i, au in enumerate(aus):
        result[au] = np.clip(pred[:, i], 0, 5).astype(np.float32)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True, help="Path to a wav file (resampled to 16 kHz mono internally)")
    parser.add_argument("--out", default="predictions.npz", help="Output .npz path")
    args = parser.parse_args()

    result = predict(args.wav)
    np.savez(args.out, **result)
    print(f"wrote {args.out}: {len(result['timestamp_sec'])} frames at {TARGET_FPS} fps "
          f"({result['timestamp_sec'][-1]:.2f}s)")
    for au in ("AU12", "AU25", "AU26"):
        print(f"{au} range: [{result[au].min():.2f}, {result[au].max():.2f}]")


if __name__ == "__main__":
    main()
