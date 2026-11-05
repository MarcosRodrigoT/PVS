"""Corrected end-to-end grid (Tables 2-6 of the revision) and GT-boundary tables (Tables 7-9).

Protocol: strict one-to-one matching at IoU >= 0.3, per-video R/P/F averaged over
videos; QD-DETR proposals come from held-out fold models. An in-sample QD-DETR
model trained on all 20 benchmark broadcasts is also evaluated (optimism of in-sample
evaluation).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

CONFIGS = [(m, meth, upd) for m in ["arcface", "transface"] for meth in ["SEQ", "INS"] for upd in [False, True]]

results = {}
for src in ["SportCLIP", "QD-DETR", "QD-DETR-insample", "GT"]:
    results[src] = {}
    for model, method, upd in CONFIGS:
        key = f"{model}|{method}|{'upd' if upd else 'orig'}"
        pv = E.evaluate_config(src, model, method, upd)
        agg = E.aggregate(pv)
        f_per_video = {v: pv[v][2] for v in E.VIDEOS}
        ci = E.stratified_bootstrap(f_per_video)
        results[src][key] = {
            "per_video": {v: list(pv[v]) for v in E.VIDEOS},
            "events": {e: list(agg[e]) for e in E.EVENTS},
            "overall": list(agg["overall"]),
            "overall_F_ci95": [float(ci[0]), float(ci[1])],
        }
        o = agg["overall"]
        print(f"{src:17s} {key:22s} R {100*o[0]:5.1f} P {100*o[1]:5.1f} F {100*o[2]:5.1f} [{100*ci[0]:.1f}, {100*ci[1]:.1f}]  "
              + " ".join(f"{e[:3]} {100*agg[e][2]:.1f}" for e in E.EVENTS))
E.save_json(results, "main_grid.json")
