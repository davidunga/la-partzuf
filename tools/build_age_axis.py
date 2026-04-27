"""Compute the age direction in InsightFace ArcFace embedding space.

Given the buffalo_s ArcFace ONNX model (the same one used at runtime) and
a directory of UTKFace-style images whose filenames start with the age
(e.g. ``25_0_1_<date>.jpg.chip.jpg``), this script:

1. Embeds each face with ArcFace (512-d, L2-normalized).
2. Fits a linear regression mapping embedding -> age via least squares.
3. Saves the unit-normalized weight vector as ``models/age_axis.json``.

At runtime ``index.html`` loads this vector and removes its component
from each face embedding before scoring, which weakens the age signal
without disturbing identity much.

Usage:
    python tools/build_age_axis.py --utkface-dir /path/to/UTKFace \
        --out models/age_axis.json --max-samples 4000
"""

import argparse
import json
import os
import random
import re
import sys

import numpy as np
import onnxruntime as ort
from PIL import Image

AGE_RE = re.compile(r"^(\d+)_")


def parse_age(name):
    m = AGE_RE.match(name)
    if not m:
        return None
    age = int(m.group(1))
    if age < 1 or age > 100:
        return None
    return age


def load_aligned(path):
    img = Image.open(path).convert("RGB").resize((112, 112), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)
    arr = (arr - 127.5) / 127.5
    arr = arr.transpose(2, 0, 1)[None]  # NCHW
    return arr


def stratified_sample(files, max_samples, seed=0):
    by_age = {}
    for f in files:
        age = parse_age(os.path.basename(f))
        if age is None:
            continue
        by_age.setdefault(age, []).append(f)
    rng = random.Random(seed)
    per_bucket = max(1, max_samples // max(1, len(by_age)))
    out = []
    for age, fs in by_age.items():
        rng.shuffle(fs)
        out.extend(fs[:per_bucket])
    rng.shuffle(out)
    return out[:max_samples]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--utkface-dir", required=True)
    ap.add_argument(
        "--rec-model",
        default=os.path.join(
            os.path.dirname(__file__), "..", "models", "recognition.onnx"
        ),
    )
    ap.add_argument(
        "--out",
        default=os.path.join(
            os.path.dirname(__file__), "..", "models", "age_axis.json"
        ),
    )
    ap.add_argument("--max-samples", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    files = [
        os.path.join(args.utkface_dir, f)
        for f in os.listdir(args.utkface_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    files = stratified_sample(files, args.max_samples)
    print(f"Embedding {len(files)} faces...", file=sys.stderr)

    sess = ort.InferenceSession(args.rec_model, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    embeddings = []
    ages = []
    for i in range(0, len(files), args.batch_size):
        batch_paths = files[i : i + args.batch_size]
        try:
            blobs = [load_aligned(p) for p in batch_paths]
        except Exception as e:
            print(f"  skip batch ({e})", file=sys.stderr)
            continue
        batch = np.concatenate(blobs, axis=0)
        out = sess.run(None, {input_name: batch})[0]
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        out = out / np.maximum(norms, 1e-12)
        embeddings.append(out)
        for p in batch_paths:
            ages.append(parse_age(os.path.basename(p)))
        if (i // args.batch_size) % 10 == 0:
            print(f"  {i + len(batch_paths)}/{len(files)}", file=sys.stderr)

    X = np.concatenate(embeddings, axis=0).astype(np.float64)
    y = np.array(ages, dtype=np.float64)
    y_centered = y - y.mean()

    # Linear least squares: w = argmin ||X w - y_centered||^2
    w, *_ = np.linalg.lstsq(X, y_centered, rcond=None)
    direction = w / np.linalg.norm(w)

    pred = X @ w
    r = np.corrcoef(pred, y_centered)[0, 1]
    print(f"in-sample Pearson r between predicted and true age: {r:.3f}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(
            {
                "direction": direction.astype(float).tolist(),
                "n_samples": int(len(y)),
                "in_sample_pearson_r": float(r),
            },
            f,
        )
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
