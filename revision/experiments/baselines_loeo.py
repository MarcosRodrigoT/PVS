"""LOEO threshold selection for the baselines (equal-tuning comparison with LOEO-selected PVS)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

base = json.load(open(E.OUT_DIR / "baselines.json"))
out = {}
for key, row in base.items():
    out[key] = {}
    for rule in E.BASELINES:
        sweep = {float(t): v["per_video_F"] for t, v in row[rule].items() if False} or None
    # recompute per-video F for every tau (baselines.json keeps per-video F only at 0.6 and at the oracle tau)
    src, model, upd = key.split("|")
    for rule in E.BASELINES:
        grid = {}
        for tau in np.round(np.arange(0.50, 0.801, 0.01), 2):
            grid[float(tau)] = {v: E.video_scores(*E.run_baseline(v, src, model, upd == "upd", rule, tau), E.load_gt(v))[2] for v in E.VIDEOS}
        pv, chosen = {}, {}
        for held in E.EVENTS:
            train = [v for v in E.VIDEOS if E.event_of(v) != held]
            t = max(grid, key=lambda t: np.mean([grid[t][v] for v in train]))
            chosen[held] = t
            pv.update({v: grid[t][v] for v in E.VIDEOS if E.event_of(v) == held})
        out[key][rule] = {"loeo_F": float(np.mean(list(pv.values()))), "loeo_per_video": pv, "chosen": chosen}
    print(key, " ".join(f"{r} {100*out[key][r]['loeo_F']:.1f}" for r in E.BASELINES), flush=True)
E.save_json(out, "baselines_loeo.json")
