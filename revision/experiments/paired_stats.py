"""Paired video-level comparisons (reviewer comment 6).

For each design factor, the per-video F-scores of two configurations that differ
only in that factor are compared over the 20 videos: mean difference, 95% CI from
a bootstrap that resamples videos within each event, Wilcoxon signed-rank p-value
and number of videos won/lost. The same is done for PVS vs the strongest baseline
under leave-one-event-out (LOEO) selection.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pvs_eval as E

grid = json.load(open(E.OUT_DIR / "main_grid.json"))
F = {src: {k: {v: x[2] for v, x in d["per_video"].items()} for k, d in cfgs.items()} for src, cfgs in grid.items()}

out = {}


def report(name, a, b):
    r = E.paired_comparison(a, b)
    out[name] = r
    print(f"{name:58s} {100*r['mean']:+5.1f} [{100*r['ci'][0]:+.1f}, {100*r['ci'][1]:+.1f}]  p={r['p_wilcoxon']:.3g}  W/L {r['wins']}/{r['losses']}")


for src in ["SportCLIP", "QD-DETR", "GT"]:
    for model in ["transface", "arcface"]:
        for upd in ["orig", "upd"]:
            report(f"{src} {model} {upd}: SEQ - INS", F[src][f"{model}|SEQ|{upd}"], F[src][f"{model}|INS|{upd}"])
        for method in ["SEQ", "INS"]:
            report(f"{src} {model} {method}: Upd - Orig", F[src][f"{model}|{method}|upd"], F[src][f"{model}|{method}|orig"])
    for method in ["SEQ", "INS"]:
        for upd in ["orig", "upd"]:
            report(f"{src} {method} {upd}: TransFace - ArcFace", F[src][f"transface|{method}|{upd}"], F[src][f"arcface|{method}|{upd}"])
for model in ["transface", "arcface"]:
    for method in ["SEQ", "INS"]:
        for upd in ["orig", "upd"]:
            k = f"{model}|{method}|{upd}"
            report(f"{k}: QD-DETR - SportCLIP", F["QD-DETR"][k], F["SportCLIP"][k])

# PVS vs best baseline, both with LOEO-selected settings
sweep = json.load(open(E.OUT_DIR / "param_sweep_summary.json"))
base = json.load(open(E.OUT_DIR / "baselines_loeo.json"))
for src in ["GT", "SportCLIP", "QD-DETR"]:
    for model in ["transface", "arcface"]:
        for upd in ["orig", "upd"]:
            pvs = max(["SEQ", "INS"], key=lambda m: sweep[f"{src}|{model}|{upd}|{m}"]["loeo_F"])
            bl = max(E.BASELINES, key=lambda r: base[f"{src}|{model}|{upd}"][r]["loeo_F"])
            report(f"LOEO {src} {model} {upd}: PVS-{pvs} - {bl}", sweep[f"{src}|{model}|{upd}|{pvs}"]["loeo_per_video"], base[f"{src}|{model}|{upd}"][bl]["loeo_per_video"])
E.save_json(out, "paired_stats.json")
