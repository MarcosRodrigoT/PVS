"""Identity-aware baselines under the same proposals and recogniser (reviewer comment 2).

For every baseline the threshold is swept and the best value on the benchmark
(oracle, favourable to the baseline) is reported together with tau = 0.6. PVS
uses its fixed default settings.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

TAUS = np.round(np.arange(0.50, 0.901, 0.01), 2)
out = {}
for src in ["GT", "SportCLIP", "QD-DETR"]:
    for model in ["transface", "arcface"]:
        for upd in [False, True]:
            key = f"{src}|{model}|{'upd' if upd else 'orig'}"
            row = {}
            # chance level: uniform random target per proposal (mean of 50 draws)
            rng = np.random.default_rng(0)
            fs = []
            for _ in range(50):
                pv = {}
                for v in E.VIDEOS:
                    segs = E.load_proposals(v, src)
                    names = sorted(E.load_curves(v, upd)[1])
                    pv[v] = E.video_scores(segs, E.assign_random(names, segs, rng), E.load_gt(v))
                fs.append(E.aggregate(pv)["overall"])
            row["random"] = {"overall": list(np.mean(fs, axis=0))}
            for rule in E.BASELINES:
                sweep = {}
                for tau in TAUS:
                    pv = {v: E.video_scores(*E.run_baseline(v, src, model, upd, rule, tau), E.load_gt(v)) for v in E.VIDEOS}
                    sweep[float(tau)] = {"overall": list(E.aggregate(pv)["overall"]), "per_video_F": {v: pv[v][2] for v in E.VIDEOS}}
                best_tau = max(sweep, key=lambda t: sweep[t]["overall"][2])
                row[rule] = {"tau0.6": sweep[0.6], "best_tau": best_tau, "best": sweep[best_tau], "sweep_F": {t: sweep[t]["overall"][2] for t in sweep}}
            for method in ["SEQ", "INS"]:
                pv = E.evaluate_config(src, model, method, upd)
                row[f"PVS-{method}"] = {"overall": list(E.aggregate(pv)["overall"]), "per_video_F": {v: pv[v][2] for v in E.VIDEOS}}
            out[key] = row
            print(key, f"rand {100*row['random']['overall'][2]:.1f}",
                  " ".join(f"{r} {100*row[r]['tau0.6']['overall'][2]:.1f}/{100*row[r]['best']['overall'][2]:.1f}@{row[r]['best_tau']}" for r in E.BASELINES),
                  f"| SEQ {100*row['PVS-SEQ']['overall'][2]:.1f} INS {100*row['PVS-INS']['overall'][2]:.1f}", flush=True)
E.save_json(out, "baselines.json")
