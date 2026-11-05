"""Measured ArcFace (DeepFace Keras model used by PVS) embedding cost on the crops of bench_detect.py, batched.

    ./run_with_cuda.sh python revision/runtime/bench_arcface.py --tag highjump_video1_s10 [--batch 256]
Preprocessing follows DeepFace for ArcFace (112x112 RGB, scaled to [-1, 1] as in DeepFace's 'ArcFace' normalisation).
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
import numpy as np
import tensorflow as tf
from deepface import DeepFace

CACHE = Path("/mnt/Data/mrt/PVS_revision_cache/runtime")
ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--batch", type=int, default=256)
args = ap.parse_args()

for g in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(g, True)
model = DeepFace.build_model("ArcFace").model
crops = np.load(CACHE / f"{args.tag}_crops.npy")


def embed(batch_bgr):
    x = (batch_bgr[..., ::-1].astype(np.float32) - 127.5) / 128.0
    e = model(x, training=False).numpy()
    return e / np.linalg.norm(e, axis=1, keepdims=True)


embed(crops[: min(len(crops), args.batch)] if len(crops) else np.zeros((1, 112, 112, 3), np.uint8))  # warm-up
t0 = time.perf_counter()
out = [embed(crops[i: i + args.batch]) for i in range(0, len(crops), args.batch)]
t = time.perf_counter() - t0
np.save(CACHE / f"{args.tag}_arcface.npy", np.concatenate(out).astype(np.float16) if out else np.zeros((0, 512), np.float16))
res = {"tag": args.tag, "faces": len(crops), "batch": args.batch, "embed_s": t, "ms_per_face": 1000 * t / max(len(crops), 1)}
json.dump(res, open(CACHE / f"{args.tag}_arcface.json", "w"), indent=2)
print(json.dumps(res))
