"""Measured QD-DETR inference cost on pre-extracted SlowFast+CLIP clip features (feature extraction excluded).

    cd /home/mrt/Projects/QD-DETR && venv/bin/python /home/mrt/Projects/PVS/revision/runtime/bench_qddetr.py
"""
import contextlib
import glob
import io
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/home/mrt/Projects/QD-DETR")
os.chdir("/home/mrt/Projects/QD-DETR")
from sports.process_saliency_curves import SaliencyProcessor  # noqa: E402

ANN = "data/OlympicHighlights_annotations/olympic_highlights_train_generic.jsonl"
FEATS = ["/mnt/Data/mrt/QD-DETR/olympic/slowfast_features", "/mnt/Data/mrt/QD-DETR/olympic/clip_features"]
TXT = "/mnt/Data/mrt/QD-DETR/olympic/clip_text_features_generic"
ckpt = sorted(glob.glob(f"results/hl-video_tef-{os.environ.get('PVS_QDDETR_EXP_PREFIX', 'olympic_bench_heldout_fold')}0-*"))[-1] + "/model_best.ckpt"
with contextlib.redirect_stdout(io.StringIO()):
    proc = SaliencyProcessor(ckpt_path=ckpt, device="cuda:0", use_preextracted_features=True)
    ds = proc.load_dataset(ANN, FEATS, TXT)
times, clips = [], 0
with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
    proc.get_saliency_scores(ds, 5)  # warm-up
    for i in range(5, len(ds)):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        r = proc.get_saliency_scores(ds, i)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
        clips += r["n_clips"]
res = {"videos": len(times), "clips": clips, "ms_per_clip": 1000 * sum(times) / clips, "ms_per_frame_25fps": 1000 * sum(times) / (clips * 50)}
print(json.dumps(res))
json.dump(res, open(os.environ.get("PVS_QDDETR_TIMING_FILE", "/mnt/Data/mrt/PVS_revision_cache/runtime/qddetr_bench.json"), "w"), indent=2)
