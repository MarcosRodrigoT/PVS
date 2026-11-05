"""Frame-level highlight detection of the summarisers when identity is ignored (Table 6).

Frames inside uncertainty (UN) bands are excluded from the counts ("don't care").
The lenient SportCLIP convention of the submission (UN frames counted as correct
whatever the prediction) is also reported for reference.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E


def frame_labels(video):
    meta = E.video_meta(video)
    fps, n = meta["fps"], meta["nb_frames"]
    lab = np.zeros(n, dtype=np.int8)  # 0 NHL, 1 HL, 2 UN
    for kind, s, e in E.load_timeline(video):
        a, b = int(round(s * fps)), int(round(e * fps))
        lab[a:b] = {"NHL": 0, "HL": 1, "UN": 2}[kind]
    return lab, fps


def frame_scores(video, source, lenient=False):
    lab, fps = frame_labels(video)
    pred = np.zeros(len(lab), dtype=bool)
    for s, e in E.load_proposals(video, source):
        pred[int(round(s * fps)): int(round(e * fps)) + 1] = True
    hl, nhl, un = lab == 1, lab == 0, lab == 2
    tp, fn, fp = np.sum(pred & hl), np.sum(~pred & hl), np.sum(pred & nhl)
    if lenient:
        tp += np.sum(pred & un)
    return E.prf(tp, fp, fn)


out = {}
for src in ["SportCLIP", "QD-DETR", "QD-DETR-insample", "QD-DETR-submitted"]:
    for lenient in [False, True]:
        pv = {v: frame_scores(v, src, lenient) for v in E.VIDEOS}
        agg = E.aggregate(pv)
        key = f"{src}|{'lenient' if lenient else 'UN-excluded'}"
        out[key] = {"events": {e: list(agg[e]) for e in E.EVENTS}, "overall": list(agg["overall"]), "per_video": {v: list(pv[v]) for v in E.VIDEOS}}
        print(f"{key:30s}", " | ".join(f"{e[:4]} {100*agg[e][0]:.1f}/{100*agg[e][1]:.1f}/{100*agg[e][2]:.1f}" for e in E.EVENTS + ["overall"]))
E.save_json(out, "frame_level_vs.json")
