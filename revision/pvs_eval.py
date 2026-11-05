"""
Offline evaluation library used for the IMAVIS revision.

All experiments of the revision operate on cached intermediate outputs of the
PVS pipeline (highlight proposals in video_summaries/, resemblance curves in
resemblance_results/, face embeddings in cropped_faces_embeddings/), so that
assignment rules, parameters and evaluation protocols can be varied without
re-running face detection or embedding extraction.

Evaluation protocol (revised, used for every table of the revision)
-------------------------------------------------------------------
* A prediction is a proposal segment together with the athlete it was assigned
  to (unassigned proposals are discarded).
* Predictions are visited in temporal order. Each one is matched to the
  not-yet-matched ground-truth highlight of the SAME athlete with the highest
  temporal IoU, provided IoU >= iou_thr (strict one-to-one matching).
* Per video: TP = matched predictions, FP = unmatched predictions,
  FN = unmatched ground-truth highlights; R, P, F computed from these counts.
* Per event: mean of the per-video R, P, F of its five videos.
  Overall: mean of the per-video R, P, F of the 20 videos (equivalently, the
  mean of the four per-event means).

The original submission used `one_to_one=False` (a ground-truth event could be
matched by several predictions) and pooled TP/FP/GT counts over the five videos
of each event; `legacy_*` helpers reproduce those numbers for verification.
"""

import csv
import json
import os
import subprocess
from functools import lru_cache
from pathlib import Path

import numpy as np

PVS_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PVS_DIR / "data"
SUMMARIES_DIR = PVS_DIR / "video_summaries"
RESEMBLANCE_DIR = PVS_DIR / "resemblance_results"
EMBEDDINGS_DIR = PVS_DIR / "cropped_faces_embeddings"
TARGETS_DIR = PVS_DIR / "target_faces_updated"
GT3_DIR = Path("/mnt/Data/mrt/OlympicHighlights_dataset")
# PVS_OUT_DIR, PVS_QDDETR_CV and PVS_QDDETR_INSAMPLE allow re-running every experiment with
# other QD-DETR models (e.g., trained only on the benchmark videos) without touching the
# default outputs.
OUT_DIR = Path(os.environ.get("PVS_OUT_DIR", PVS_DIR / "revision" / "outputs"))

EVENTS = ["highjump", "javelin", "longjump", "polevault"]
EVENT_NAMES = {"highjump": "High jump", "javelin": "Javelin", "longjump": "Long jump", "polevault": "Pole vault"}
EVENT_DIRS = {"highjump": "High Jump", "javelin": "Javelin", "longjump": "Long Jump", "polevault": "Pole Vault"}
VIDEOS = [f"{e}_video{i}" for e in EVENTS for i in range(1, 6)]

# Settings of the submitted system (fixed throughout unless a sweep says otherwise)
DEFAULTS = {
    "tau": 0.6,  # threshold on the rescaled similarity (1 + cos) / 2, i.e. cos >= 0.2
    "window": 200,  # centred moving average over consecutive face-bearing frames
    "lookback": 5.0,  # seconds (SEQ)
    "expansion": 3.0,  # seconds on each side (INS)
}

PROPOSALS = {
    "SportCLIP": "SportCLIP",
    # held-out fold models, each trained on the 16 benchmark broadcasts not in its fold
    "QD-DETR": os.environ.get("PVS_QDDETR_CV", "QD-DETR-CV-bench"),
    # in-sample reference: one model trained on all 20 benchmark broadcasts
    "QD-DETR-insample": os.environ.get("PVS_QDDETR_INSAMPLE", "QD-DETR-insample-bench"),
    # checkpoint of the submitted version (used only to reproduce its numbers, verify_legacy.py)
    "QD-DETR-submitted": "QD-DETR",
}


def event_of(video):
    return video.split("_")[0]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def video_meta(video):
    """fps, number of frames and duration (s) of the local copy of a video."""
    cmd = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate,nb_frames", "-show_entries", "format=duration",
        "-of", "json", str(DATA_DIR / f"{video}.mp4"),
    ]
    info = json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)
    num, den = map(int, info["streams"][0]["r_frame_rate"].split("/"))
    return {"fps": num / den, "nb_frames": int(info["streams"][0]["nb_frames"]), "duration": float(info["format"]["duration"])}


@lru_cache(maxsize=None)
def load_gt(video):
    """Ground-truth highlights [{'start','end','athlete'}] in seconds."""
    fps = video_meta(video)["fps"]
    gt = []
    with open(DATA_DIR / f"{video}_identity.csv") as f:
        for row in csv.DictReader(f):
            if row["Event type"] == "Highlight":
                gt.append({
                    "start": int(row["First frame"]) / fps,
                    "end": int(row["Last frame"]) / fps,
                    "athlete": row["Identity"].strip().lower(),
                })
    return tuple(gt)


@lru_cache(maxsize=None)
def load_timeline(video):
    """Full three-class annotation [(label, start_s, end_s)] with label in {HL, NHL, UN}."""
    fps = video_meta(video)["fps"]
    event, idx = video.split("_video")
    path = GT3_DIR / EVENT_DIRS[event] / "ground_truth" / f"video{idx}_processed.csv"
    label = {"Highlight": "HL", "Not a highlight": "NHL", "Uncertainty": "UN"}
    out = []
    with open(path) as f:
        for row in csv.DictReader(f):
            out.append((label[row["Event type"]], int(row["First frame"]) / fps, (int(row["Last frame"]) + 1) / fps))
    return tuple(out)


@lru_cache(maxsize=None)
def load_proposals(video, source):
    """Candidate segments [(start_s, end_s)] in temporal order.

    source: 'SportCLIP', 'QD-DETR' (held-out), 'QD-DETR-insample', 'QD-DETR-submitted' or 'GT' (oracle boundaries).
    """
    if source == "GT":
        return tuple((g["start"], g["end"]) for g in load_gt(video))
    with open(SUMMARIES_DIR / PROPOSALS[source] / video / "segments.json") as f:
        data = json.load(f)
    segs = data["segments"]
    segs = segs["segments"] if isinstance(segs, dict) else segs
    out = [(s["start_time"], s["end_time"]) for s in segs]
    assert out == sorted(out)
    return tuple(out)


@lru_cache(maxsize=None)
def load_curves(video, updated):
    """Raw resemblance curves of the annotated targets.

    Returns (timestamps, {athlete: {'arcface': raw, 'transface': raw}}). Timestamps
    are those of frames with at least one detected face (frames without detections
    are absent from the curve, not set to zero).
    """
    suffix = "_updated" if updated else ""
    curves, ts = {}, None
    for athlete_dir in sorted((RESEMBLANCE_DIR / video).iterdir()):
        if not athlete_dir.is_dir():
            continue
        with open(athlete_dir / f"{athlete_dir.name}{suffix}.json") as f:
            d = json.load(f)
        t = np.asarray(d["timestamps_seconds"])
        if ts is None:
            ts = t
        assert len(t) == len(ts)
        curves[athlete_dir.name] = {"arcface": np.asarray(d["arcface_scores"]), "transface": np.asarray(d["transface_scores"])}
    return ts, curves


# ---------------------------------------------------------------------------
# Identity evidence and assignment rules
# ---------------------------------------------------------------------------


def smooth(scores, window):
    """Centred moving average, identical to the PVS implementation (edges use shrinking windows)."""
    scores = np.asarray(scores, dtype=float)
    if window <= 1 or len(scores) < window:
        return scores
    out = np.convolve(scores, np.ones(window) / window, mode="same")
    for i in range(window // 2):
        out[i] = np.mean(scores[: i + window // 2 + 1])
        out[-(i + 1)] = np.mean(scores[-(i + window // 2 + 1):])
    return out


def evidence(ts, curves, model, window):
    names = sorted(curves)
    return names, np.stack([smooth(curves[n][model], window) for n in names]) if names else np.zeros((0, len(ts)))


def assign_sequential(ts, names, G, segments, tau, lookback):
    """SEQ: most recent above-threshold target within [start - lookback, start]; otherwise propagate."""
    out, current = [], None
    for start, _ in segments:
        s_idx = np.searchsorted(ts, start)
        l_idx = np.searchsorted(ts, max(0.0, start - lookback))
        best, best_pos = None, -1
        for m, name in enumerate(names):
            w = G[m, l_idx: s_idx + 1]
            above = np.nonzero(w >= tau)[0]
            if len(above) and l_idx + above[-1] > best_pos:
                best, best_pos = name, l_idx + above[-1]
        if best is not None:
            current = best
        out.append(current)
    return out


def assign_instant(ts, names, G, segments, tau, expansion):
    """INS: among targets above threshold in [start - e, end + e], pick the highest mean above-threshold score."""
    out = []
    for start, end in segments:
        a = np.searchsorted(ts, max(0.0, start - expansion))
        b = np.searchsorted(ts, end + expansion)
        cands = []
        for m, name in enumerate(names):
            w = G[m, a:b]
            sel = w[w >= tau]
            if len(sel):
                cands.append((sel.mean(), w.max(), name))
        # stable sort as in the original code: ties go to the alphabetically first target
        out.append(sorted(cands, key=lambda c: (-c[0], -c[1]))[0][2] if cands else None)
    return out


def run_pvs(video, source, model, method, updated, tau=None, window=None, lookback=None, expansion=None, curves_override=None):
    """Assign the proposals of `video` with one of the two PVS rules. Returns (segments, labels)."""
    tau = DEFAULTS["tau"] if tau is None else tau
    window = DEFAULTS["window"] if window is None else window
    segments = load_proposals(video, source)
    ts, curves = curves_override if curves_override is not None else load_curves(video, updated)
    names, G = evidence(ts, curves, model, window)
    if method == "SEQ":
        labels = assign_sequential(ts, names, G, segments, tau, DEFAULTS["lookback"] if lookback is None else lookback)
    else:
        labels = assign_instant(ts, names, G, segments, tau, DEFAULTS["expansion"] if expansion is None else expansion)
    return segments, labels


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def match(segments, labels, gt, iou_thr=0.3, one_to_one=True):
    """Return (tp, fp, fn, pairs) with pairs = [(pred_idx, gt_idx, iou)]."""
    used = [False] * len(gt)
    pairs, n_pred = [], 0
    for i, (seg, lab) in enumerate(zip(segments, labels)):
        if lab is None:
            continue
        n_pred += 1
        best, best_j = 0.0, None
        for j, g in enumerate(gt):
            if g["athlete"] != lab or (one_to_one and used[j]):
                continue
            v = iou(seg, (g["start"], g["end"]))
            if v >= iou_thr and v > best:
                best, best_j = v, j
        if best_j is not None:
            used[best_j] = True
            pairs.append((i, best_j, best))
    tp = len(pairs)
    fn = len(gt) - sum(used)
    return tp, n_pred - tp, fn, pairs


def prf(tp, fp, fn):
    r = tp / (tp + fn) if tp + fn else 0.0
    p = tp / (tp + fp) if tp + fp else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return r, p, f


def video_scores(segments, labels, gt, iou_thr=0.3):
    tp, fp, fn, _ = match(segments, labels, gt, iou_thr)
    return prf(tp, fp, fn)


def aggregate(per_video):
    """per_video: {video: (R, P, F)} -> {'overall': (R,P,F), event: (R,P,F)} as macro averages over videos."""
    res = {}
    for e in EVENTS:
        vals = np.array([per_video[v] for v in VIDEOS if event_of(v) == e and v in per_video])
        res[e] = tuple(vals.mean(axis=0))
    vals = np.array([per_video[v] for v in VIDEOS if v in per_video])
    res["overall"] = tuple(vals.mean(axis=0))
    return res


def evaluate_config(source, model, method, updated, iou_thr=0.3, **params):
    per_video = {}
    for v in VIDEOS:
        segs, labels = run_pvs(v, source, model, method, updated, **params)
        per_video[v] = video_scores(segs, labels, load_gt(v), iou_thr)
    return per_video


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def stratified_bootstrap(per_video_values, n_boot=10000, seed=0):
    """95% CI of the mean over videos, resampling videos within each event."""
    rng = np.random.default_rng(seed)
    by_event = {e: np.array([per_video_values[v] for v in VIDEOS if event_of(v) == e]) for e in EVENTS}
    means = np.empty(n_boot)
    for b in range(n_boot):
        means[b] = np.mean(np.concatenate([x[rng.integers(0, len(x), len(x))] for x in by_event.values()]))
    return np.percentile(means, [2.5, 97.5])


def paired_comparison(a, b, n_boot=10000, seed=0):
    """Paired difference a - b over the 20 videos: mean, stratified bootstrap 95% CI, Wilcoxon p, #wins."""
    from scipy.stats import wilcoxon

    diff = {v: a[v] - b[v] for v in VIDEOS}
    d = np.array([diff[v] for v in VIDEOS])
    lo, hi = stratified_bootstrap(diff, n_boot, seed)
    nz = d[np.abs(d) > 1e-12]
    p = wilcoxon(nz).pvalue if len(nz) > 0 else 1.0
    return {"mean": float(d.mean()), "ci": [float(lo), float(hi)], "p_wilcoxon": float(p), "wins": int((d > 1e-12).sum()), "losses": int((d < -1e-12).sum())}


# ---------------------------------------------------------------------------
# Legacy protocol of the original submission (for verification only)
# ---------------------------------------------------------------------------


def legacy_event_level(source, model, method, updated):
    """Many-to-one matching; TP/FP/#GT pooled over the five videos of each event."""
    res = {}
    for e in EVENTS:
        tp = fp = ngt = 0
        for v in VIDEOS:
            if event_of(v) != e:
                continue
            segs, labels = run_pvs(v, source, model, method, updated)
            gt = load_gt(v)
            t, f_, _, _ = match(segs, labels, gt, 0.3, one_to_one=False)
            tp, fp, ngt = tp + t, fp + f_, ngt + len(gt)
        r, p = tp / ngt, tp / (tp + fp)
        res[e] = (r, p, 2 * p * r / (p + r))
    res["overall_F"] = np.mean([res[e][2] for e in EVENTS])
    return res


def save_json(obj, name):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / name, "w") as f:
        json.dump(obj, f, indent=2)


# ---------------------------------------------------------------------------
# Identity-aware baselines (same proposals, same recogniser, raw frame scores)
# ---------------------------------------------------------------------------


def raw_matrix(curves, model):
    names = sorted(curves)
    return names, np.stack([np.asarray(curves[n][model], dtype=float) for n in names])


def assign_inseg_max(ts, names, R, segments, tau):
    """In-segment evidence only: target with the highest single-frame score inside the segment."""
    out = []
    for start, end in segments:
        a, b = np.searchsorted(ts, start), np.searchsorted(ts, end, side="right")
        if b <= a:
            out.append(None)
            continue
        w = R[:, a:b].max(axis=1)
        m = int(np.argmax(w))
        out.append(names[m] if w[m] >= tau else None)
    return out


def assign_inseg_vote(ts, names, R, segments, tau):
    """In-segment majority vote of the per-frame best above-threshold identity (face-track style voting)."""
    out = []
    for start, end in segments:
        a, b = np.searchsorted(ts, start), np.searchsorted(ts, end, side="right")
        W = R[:, a:b]
        if W.shape[1] == 0:
            out.append(None)
            continue
        best = W.argmax(axis=0)
        ok = W.max(axis=0) >= tau
        if not ok.any():
            out.append(None)
            continue
        votes = np.bincount(best[ok], minlength=len(names))
        support = np.bincount(best[ok], weights=W.max(axis=0)[ok], minlength=len(names))
        m = max(range(len(names)), key=lambda k: (votes[k], support[k]))
        out.append(names[m])
    return out


def assign_adjacent(ts, names, R, segments, tau):
    """Adjacent-shot rule: identity of the temporally closest confident face (inside, before or after the segment)."""
    best = R.argmax(axis=0)
    conf = np.nonzero(R.max(axis=0) >= tau)[0]
    tconf = ts[conf]
    out = []
    for start, end in segments:
        if len(conf) == 0:
            out.append(None)
            continue
        a, b = np.searchsorted(tconf, start), np.searchsorted(tconf, end, side="right")
        if b > a:  # confident faces inside: take the most confident one
            k = conf[a + int(np.argmax(R.max(axis=0)[conf[a:b]]))]
        else:  # nearest confident face outside the segment (ties -> before)
            before = conf[a - 1] if a > 0 else None
            after = conf[a] if a < len(conf) else None
            if before is None:
                k = after
            elif after is None:
                k = before
            else:
                k = before if start - ts[before] <= ts[after] - end else after
        out.append(names[best[k]])
    return out


def assign_random(names, segments, rng):
    return [names[rng.integers(len(names))] for _ in segments]


BASELINES = {"inseg_max": assign_inseg_max, "inseg_vote": assign_inseg_vote, "adjacent": assign_adjacent}


def run_baseline(video, source, model, updated, rule, tau):
    segments = load_proposals(video, source)
    ts, curves = load_curves(video, updated)
    names, R = raw_matrix(curves, model)
    return segments, BASELINES[rule](ts, names, R, segments, tau)
