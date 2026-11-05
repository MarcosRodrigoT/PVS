"""Check that pvs_eval reproduces every number of the original submission (Tables 6-7).

The submitted numbers were computed with an identity-label typo in javelin_video3
("Valdejch" instead of "Vadlejch", 2 events), corrected for the revision. This
script emulates the typo so that the original numbers are reproduced exactly:
end-to-end tables never matched those two events, and the GT-boundary script
only loaded targets whose name appeared in the ground truth (so "vadlejch" was
not a candidate there).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

_load_gt, _load_curves = E.load_gt, E.load_curves


def gt_with_typo(video):
    gt = _load_gt(video)
    if video != "javelin_video3":
        return gt
    return tuple(dict(g, athlete="valdejch") if g["athlete"] == "vadlejch" else g for g in gt)


def curves_without_vadlejch(video, updated):
    ts, curves = _load_curves(video, updated)
    if video != "javelin_video3":
        return ts, curves
    return ts, {k: v for k, v in curves.items() if k != "vadlejch"}


E.load_gt = gt_with_typo

# Table 6 (overall F, legacy protocol) and Table 7 (GT boundaries) of the submission
PAPER_T6 = [42.2, 39.5, 41.5, 41.7, 63.6, 68.5, 62.3, 66.0, 47.8, 44.8, 47.1, 47.9, 72.3, 76.8, 70.5, 75.6]
PAPER_T7 = {("arcface", "SEQ", False): (56.69, 56.99, 56.84), ("arcface", "SEQ", True): (52.63, 52.83, 52.73),
            ("arcface", "INS", False): (47.11, 63.61, 53.25), ("arcface", "INS", True): (54.06, 59.69, 56.54),
            ("transface", "SEQ", False): (84.43, 84.94, 84.68), ("transface", "SEQ", True): (89.74, 89.81, 89.78),
            ("transface", "INS", False): (70.89, 90.33, 78.97), ("transface", "INS", True): (82.02, 91.69, 86.46)}

i, ok = 0, True
for src in ["SportCLIP", "QD-DETR-submitted"]:
    for model in ["arcface", "transface"]:
        for method in ["SEQ", "INS"]:
            for upd in [False, True]:
                r = E.legacy_event_level(src, model, method, upd)
                got = 100 * r["overall_F"]
                ok &= abs(got - PAPER_T6[i]) < 0.06  # paper averaged the rounded per-event F-scores (e.g. 68.45 -> 68.5)
                print(f"T6 {src:17s} {model:9s} {method} {'Upd' if upd else 'Orig'}: {got:.3f} (paper {PAPER_T6[i]})")
                i += 1

E.load_curves = curves_without_vadlejch
for (model, method, upd), ref in PAPER_T7.items():
    pv = E.evaluate_config("GT", model, method, upd)
    got = tuple(round(100 * x, 2) for x in E.aggregate(pv)["overall"][:2]) + (round(100 * np.mean([pv[v][2] for v in E.VIDEOS]), 2),)
    ok &= all(abs(a - b) < 0.011 for a, b in zip(got, ref))
    print(f"T7 {model:9s} {method} {'Upd' if upd else 'Orig'}: {tuple(float(x) for x in got)} (paper {ref})")
print("ALL REPRODUCED" if ok else "MISMATCH")
