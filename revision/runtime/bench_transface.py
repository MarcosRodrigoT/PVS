"""Measured TransFace-L embedding cost on the crops of bench_detect.py (batched GPU inference).

Run with the TransFace environment:
    /home/mrt/Projects/TransFace/venv/bin/python revision/runtime/bench_transface.py --tag highjump_video1_s10 [--batch 256]
Preprocessing is identical to the PVS TransFace wrapper (112x112, RGB, [-1, 1]).
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/mrt/Projects/TransFace")
from backbones import get_model  # noqa: E402

CACHE = Path("/mnt/Data/mrt/PVS_revision_cache/runtime")
WEIGHT = "/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt"

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--batch", type=int, default=256)
args = ap.parse_args()

dev = torch.device("cuda")
model = get_model("vit_l_dp005_mask_005", fp16=False)
model.load_state_dict(torch.load(WEIGHT, map_location=dev))
model = model.to(dev).eval()
crops = np.load(CACHE / f"{args.tag}_crops.npy")


def embed(batch_bgr):
    x = torch.from_numpy(batch_bgr[..., ::-1].copy()).to(dev).permute(0, 3, 1, 2).float().div(255).sub(0.5).div(0.5)
    e = model(x)
    e = e[0] if isinstance(e, (tuple, list)) else e  # TransFace returns (embedding, patch weights)
    return torch.nn.functional.normalize(e, dim=1).cpu().numpy()


with torch.no_grad():
    embed(crops[: min(len(crops), args.batch)] if len(crops) else np.zeros((1, 112, 112, 3), np.uint8))  # warm-up
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = [embed(crops[i: i + args.batch]) for i in range(0, len(crops), args.batch)]
    torch.cuda.synchronize()
    t = time.perf_counter() - t0
emb = np.concatenate(out) if out else np.zeros((0, 512), np.float32)
np.save(CACHE / f"{args.tag}_transface.npy", emb.astype(np.float16))
res = {"tag": args.tag, "faces": len(crops), "batch": args.batch, "embed_s": t, "ms_per_face": 1000 * t / max(len(crops), 1)}
json.dump(res, open(CACHE / f"{args.tag}_transface.json", "w"), indent=2)
print(json.dumps(res))
