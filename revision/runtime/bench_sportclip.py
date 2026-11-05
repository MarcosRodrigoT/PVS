"""Measured per-frame cost of the SportCLIP stream (decoding + CLIP ViT-B/32 image embedding of every frame).

Run with the SportCLIP environment:
    /home/mrt/Projects/SportCLIP-official/venv/bin/python revision/runtime/bench_sportclip.py --video highjump_video1 --seconds 300
Frames are decoded sequentially, resized (shorter side 224, bicubic) and center-cropped on the GPU with
CLIP normalisation, and embedded in batches of 256. Scoring the 64 sentence pairs and post-processing
operate on 1-D score curves and are timed separately.
"""
import argparse
import json
import time
from pathlib import Path

import clip
import cv2
import numpy as np
import torch
import torch.nn.functional as F

CACHE = Path("/mnt/Data/mrt/PVS_revision_cache/runtime")
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True)
ap.add_argument("--seconds", type=float, default=300)
ap.add_argument("--batch", type=int, default=256)
args = ap.parse_args()

dev = "cuda"
model, _ = clip.load("ViT-B/32", device=dev)
model.eval()
mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=dev).view(1, 3, 1, 1)
std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=dev).view(1, 3, 1, 1)
text = clip.tokenize([f"sentence {i}" for i in range(16)]).to(dev)


@torch.no_grad()
def embed(frames):
    x = torch.from_numpy(np.stack(frames)).to(dev).permute(0, 3, 1, 2).float().div(255)[:, [2, 1, 0]]
    h, w = x.shape[-2:]
    s = 224 / min(h, w)
    x = F.interpolate(x, size=(round(h * s), round(w * s)), mode="bicubic", align_corners=False)
    H, W = x.shape[-2:]
    x = x[..., (H - 224) // 2:(H - 224) // 2 + 224, (W - 224) // 2:(W - 224) // 2 + 224]
    e = model.encode_image(((x - mean) / std).half())
    return F.normalize(e.float(), dim=1)


cap = cv2.VideoCapture(f"data/{args.video}.mp4")
fps = cap.get(cv2.CAP_PROP_FPS)
n = int(args.seconds * fps)
ok, f0 = cap.read()
embed([f0] * args.batch)  # warm-up with a full batch
torch.cuda.synchronize()
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
t_dec = t_emb = 0.0
buf, embs = [], []
for i in range(n):
    t0 = time.perf_counter()
    ok, fr = cap.read()
    t_dec += time.perf_counter() - t0
    if not ok:
        break
    buf.append(fr)
    if len(buf) == args.batch or i == n - 1:
        t0 = time.perf_counter()
        embs.append(embed(buf))
        torch.cuda.synchronize()
        t_emb += time.perf_counter() - t0
        buf = []
E_ = torch.cat(embs)
with torch.no_grad():
    T = F.normalize(model.encode_text(text).float(), dim=1)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(64 // 8):  # 64 sentence pairs (two classes each)
        _ = (100 * E_ @ T[:2].T).softmax(-1)
    torch.cuda.synchronize()
    t_pairs = time.perf_counter() - t0
res = {"video": args.video, "frames": int(E_.shape[0]), "decode_ms_per_frame": 1000 * t_dec / E_.shape[0],
       "embed_ms_per_frame": 1000 * t_emb / E_.shape[0], "pair_scoring_ms_per_frame": 1000 * t_pairs / E_.shape[0] * 8}
json.dump(res, open(CACHE / f"{args.video}_sportclip.json", "w"), indent=2)
print(json.dumps(res))
