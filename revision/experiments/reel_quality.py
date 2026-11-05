"""What does the event-level score mean for the resulting reels? (reviewer comment 6)

For the main configurations this script reports
  * F as a function of the IoU threshold (0.1 ... 0.9),
  * boundary errors of matched predictions (|start offset|, |end offset|, seconds),
  * duration-based reel measures per target (aggregated over targets, then videos):
      purity        = time of the reel inside the target's own highlights / reel time
      contamination = time inside highlights of OTHER athletes / reel time
      background    = time inside non-highlight or uncertainty footage / reel time
      coverage      = time of the target's highlights covered by its reel / target highlight time
      redundancy    = fraction of reel clips whose main overlapping own highlight is
                      already the main overlapping highlight of an earlier clip of the reel,
  * robustness to simulated annotation-boundary disagreement (GT starts/ends jittered
    by U[-d, d] seconds, d in {0.5, 1, 2}, 20 draws).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

MAIN = [(src, model, method, upd) for src in ["QD-DETR", "SportCLIP", "GT"] for model, upd in [("transface", True), ("transface", False), ("arcface", False)] for method in ["SEQ", "INS"]]
IOUS = [round(x, 2) for x in np.arange(0.1, 0.91, 0.05)]


def overlap(a, b):
    return max(0.0, min(a[1], b[1]) - max(a[0], b[0]))


def union_length(intervals):
    total, cur = 0.0, None
    for s, e in sorted(intervals):
        if cur is None or s > cur[1]:
            if cur:
                total += cur[1] - cur[0]
            cur = [s, e]
        else:
            cur[1] = max(cur[1], e)
    return total + (cur[1] - cur[0] if cur else 0.0)


def reel_measures(video, segments, labels):
    gt = E.load_gt(video)
    timeline = E.load_timeline(video)
    acc = {"reel": 0.0, "own": 0.0, "other": 0.0, "bg": 0.0, "own_hl": 0.0, "covered": 0.0, "clips": 0, "redundant": 0}
    for athlete in sorted({g["athlete"] for g in gt}):
        clips = [s for s, lab in zip(segments, labels) if lab == athlete]
        own = [(g["start"], g["end"]) for g in gt if g["athlete"] == athlete]
        other = [(g["start"], g["end"]) for g in gt if g["athlete"] != athlete]
        bg = [(s, e) for kind, s, e in timeline if kind != "HL"]
        acc["own_hl"] += sum(e - s for s, e in own)
        seen = set()
        for c in clips:
            acc["reel"] += c[1] - c[0]
            acc["own"] += sum(overlap(c, g) for g in own)
            acc["other"] += sum(overlap(c, g) for g in other)
            acc["bg"] += sum(overlap(c, g) for g in bg)
            ov = [overlap(c, g) for g in own]
            if ov and max(ov) > 0:
                j = int(np.argmax(ov))
                acc["redundant"] += j in seen
                seen.add(j)
            acc["clips"] += 1
        # covered own-highlight time (union of clips intersected with own highlights)
        acc["covered"] += sum(union_length([(max(c[0], g[0]), min(c[1], g[1])) for c in clips if overlap(c, g) > 0]) for g in own)
    r = max(acc["reel"], 1e-9)
    return {"purity": acc["own"] / r, "contamination": acc["other"] / r, "background": acc["bg"] / r,
            "coverage": acc["covered"] / acc["own_hl"], "redundancy": acc["redundant"] / max(acc["clips"], 1)}


def jitter_gt(gt, d, rng):
    out = []
    for g in gt:
        s, e = g["start"] + rng.uniform(-d, d), g["end"] + rng.uniform(-d, d)
        out.append(dict(g, start=min(s, e - 0.1), end=e))
    return tuple(out)


if __name__ == "__main__":
    results = {}
    for src, model, method, upd in MAIN:
        key = f"{src}|{model}|{method}|{'upd' if upd else 'orig'}"
        runs = {v: E.run_pvs(v, src, model, method, upd) for v in E.VIDEOS}
        # IoU sweep
        sweep = {t: float(np.mean([E.video_scores(*runs[v], E.load_gt(v), t)[2] for v in E.VIDEOS])) for t in IOUS}
        # boundary errors of matched pairs at IoU 0.3
        ds, de = [], []
        for v in E.VIDEOS:
            segs, labels = runs[v]
            gt = E.load_gt(v)
            for i, j, _ in E.match(segs, labels, gt, 0.3)[3]:
                ds.append(abs(segs[i][0] - gt[j]["start"]))
                de.append(abs(segs[i][1] - gt[j]["end"]))
        # reel measures
        rm = {v: reel_measures(v, *runs[v]) for v in E.VIDEOS}
        reel = {k: float(np.mean([rm[v][k] for v in E.VIDEOS])) for k in rm[E.VIDEOS[0]]}
        # boundary jitter
        jit = {}
        for d in [0.5, 1.0, 2.0]:
            rng = np.random.default_rng(0)
            fs = [np.mean([E.video_scores(*runs[v], jitter_gt(E.load_gt(v), d, rng))[2] for v in E.VIDEOS]) for _ in range(20)]
            jit[d] = [float(np.mean(fs)), float(np.std(fs))]
        results[key] = {"iou_sweep": sweep, "start_err_median": float(np.median(ds)), "end_err_median": float(np.median(de)),
                        "start_err_p90": float(np.percentile(ds, 90)), "end_err_p90": float(np.percentile(de, 90)), "reel": reel, "jitter": jit}
        print(f"{key:32s} F@.3 {100*sweep[0.3]:.1f} @.5 {100*sweep[0.5]:.1f} @.7 {100*sweep[0.7]:.1f} | err s/e {np.median(ds):.1f}/{np.median(de):.1f}s | "
              + " ".join(f"{k[:5]} {100*v:.1f}" for k, v in reel.items()) + " | jit " + " ".join(f"{d}:{100*m:.1f}" for d, (m, s) in jit.items()), flush=True)
    E.save_json(results, "reel_quality.json")
