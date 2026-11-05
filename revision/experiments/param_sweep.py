"""Sensitivity of PVS to its fusion settings and leave-one-event-out (LOEO) selection (reviewer comment 3).

For every proposal source / recogniser / target / rule, F is computed per video on
a grid of (tau, smoothing window, lookback or expansion). Outputs:
  * the full grid (per-video F) for later analysis,
  * LOEO selection: settings chosen on three events, evaluated on the held-out one.
"""
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

TAUS = [round(x, 2) for x in np.arange(0.50, 0.801, 0.01)]
WINDOWS = [1, 25, 50, 100, 200, 400]
OFFSETS = {"SEQ": [1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0], "INS": [0.0, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]}


def run_cell(args):
    src, model, upd, method = args
    grid = {}
    for v in E.VIDEOS:
        segments = E.load_proposals(v, src)
        ts, curves = E.load_curves(v, upd)
        gt = E.load_gt(v)
        for w in WINDOWS:
            names, G = E.evidence(ts, curves, model, w)
            for off in OFFSETS[method]:
                for tau in TAUS:
                    if method == "SEQ":
                        labels = E.assign_sequential(ts, names, G, segments, tau, off)
                    else:
                        labels = E.assign_instant(ts, names, G, segments, tau, off)
                    grid.setdefault(f"{tau}|{w}|{off}", {})[v] = E.video_scores(segments, labels, gt)[2]
    return f"{src}|{model}|{'upd' if upd else 'orig'}|{method}", grid


def loeo(grid, default_key):
    """Choose settings on 15 videos of three events, evaluate on the 5 videos of the fourth."""
    per_video, chosen = {}, {}
    for held in E.EVENTS:
        train = [v for v in E.VIDEOS if E.event_of(v) != held]
        best = max(grid, key=lambda k: (np.mean([grid[k][v] for v in train]), k == default_key))
        chosen[held] = best
        for v in E.VIDEOS:
            if E.event_of(v) == held:
                per_video[v] = grid[best][v]
    return per_video, chosen


if __name__ == "__main__":
    cells = list(itertools.product(["GT", "SportCLIP", "QD-DETR"], ["arcface", "transface"], [False, True], ["SEQ", "INS"]))
    with Pool(24) as pool:
        grids = dict(pool.map(run_cell, cells))

    summary = {}
    for key, grid in grids.items():
        method = key.split("|")[-1]
        d = E.DEFAULTS
        default_key = f"{d['tau']}|{d['window']}|{d['lookback'] if method == 'SEQ' else d['expansion']}"
        default = np.mean(list(grid[default_key].values()))
        oracle_key = max(grid, key=lambda k: np.mean(list(grid[k].values())))
        pv, chosen = loeo(grid, default_key)
        summary[key] = {
            "default_key": default_key, "default_F": default,
            "oracle_key": oracle_key, "oracle_F": float(np.mean(list(grid[oracle_key].values()))),
            "loeo_F": float(np.mean(list(pv.values()))), "loeo_per_video": pv, "loeo_chosen": chosen,
            "default_per_video": grid[default_key],
        }
        print(f"{key:32s} default {100*default:.1f}  LOEO {100*summary[key]['loeo_F']:.1f}  oracle {100*summary[key]['oracle_F']:.1f} @ {oracle_key}  chosen {list(chosen.values())}", flush=True)
    E.save_json({k: {kk: {f"{t}": vv for t, vv in g.items()} for kk, g in [("grid", grids[k])]} for k in grids}, "param_grid.json")
    E.save_json(summary, "param_sweep_summary.json")
