"""Sparse PVS run on the crops/embeddings produced by the runtime benchmark: timing of the
resemblance curves, target update and clip assignment, and event-level accuracy of the
actually executed sparse configuration.

    python revision/runtime/bench_fusion.py --stride 10
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

CACHE = Path("/mnt/Data/mrt/PVS_revision_cache/runtime")
ap = argparse.ArgumentParser()
ap.add_argument("--stride", type=int, default=10)
args = ap.parse_args()

out = {}
for bb in ["transface", "arcface"]:
    for upd in [False, True]:
        for method in ["SEQ", "INS"]:
            for src in ["QD-DETR", "SportCLIP", "GT"]:
                pv, t_fuse = {}, 0.0
                for v in E.VIDEOS:
                    tag = f"{v}_s{args.stride}"
                    if not (CACHE / f"{tag}_{bb}.npy").exists():
                        continue
                    emb = np.load(CACHE / f"{tag}_{bb}.npy").astype(np.float32)
                    frames = np.load(CACHE / f"{tag}_frames.npy")
                    fps = E.video_meta(v)["fps"]
                    k = max(1, round(args.stride * fps / 25.0))  # frames are sampled every args.stride/25 s
                    window = max(1, round(E.DEFAULTS["window"] / k))  # same temporal extent as the dense setting
                    t0 = time.perf_counter()
                    uf, inv = np.unique(frames, return_inverse=True)
                    curves = {}
                    for tdir in sorted(d for d in (E.TARGETS_DIR / v).iterdir() if d.is_dir()):
                        ref = np.load(tdir / f"target_{bb}_embedding.npy").astype(np.float32).ravel()
                        ref /= np.linalg.norm(ref)
                        s = (emb @ ref + 1) / 2
                        if upd:
                            s = (emb @ emb[int(np.argmax(s))] + 1) / 2
                        c = np.full(len(uf), -np.inf)
                        np.maximum.at(c, inv, s)
                        curves[tdir.name] = {bb: c}
                    segs, labels = E.run_pvs(v, src, bb, method, False, window=window, curves_override=(uf / fps, curves))
                    t_fuse += time.perf_counter() - t0
                    pv[v] = E.video_scores(segs, labels, E.load_gt(v))
                key = f"{src}|{bb}|{method}|{'upd' if upd else 'orig'}"
                out[key] = {"videos": len(pv), "F": float(np.mean([x[2] for x in pv.values()])) if pv else None, "fusion_s_total": t_fuse,
                            "per_video_F": {v: pv[v][2] for v in pv}}
                if pv:
                    print(f"{key:32s} videos {len(pv)} F {100*out[key]['F']:.1f} fusion {t_fuse:.2f}s", flush=True)
json.dump(out, open(os.environ.get("PVS_FUSION_FILE", CACHE / f"fusion_s{args.stride}_bench.json"), "w"), indent=2)
