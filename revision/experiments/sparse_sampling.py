"""Accuracy of PVS when the face-analysis stream only processes a subset of frames (reviewer comment 7).

Detection and embedding are computed independently for every frame, so running
the face stream on one frame out of k is exactly equivalent to keeping the
cached detections/embeddings of those frames. For each stride the original and
updated resemblance curves are rebuilt from the sampled faces only (the update
exemplar is also searched among sampled faces) and the smoothing window is
scaled so that it spans the same time as in the dense pipeline. k = 1 is the
dense pipeline (every frame); for k > 1 one frame is processed every k/25 s
(stride k for 25 fps videos, 2k for 50 fps videos), i.e. at 25/k frames per second.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

CACHE = Path("/mnt/Data/mrt/PVS_revision_cache")
STRIDES = [1, 2, 5, 10, 25, 50]


def sampling(video, stride):
    """Frame stride and smoothing window (samples) giving the dense window duration."""
    fps = E.video_meta(video)["fps"]
    if stride == 1:
        return 1, E.DEFAULTS["window"]
    k = max(1, round(stride * fps / 25.0))
    return k, max(1, round(E.DEFAULTS["window"] / k))


def curves_for(video, stride):
    fps = E.video_meta(video)["fps"]
    k, _ = sampling(video, stride)
    out = {}
    for bb in ["arcface", "transface"]:
        c = np.load(CACHE / f"{video}_{bb}.npz")
        keep = c["frame"] % k == 0
        frames, emb = c["frame"][keep], c["emb"][keep].astype(np.float32)
        uf, inv = np.unique(frames, return_inverse=True)
        for tdir in sorted(d for d in (E.TARGETS_DIR / video).iterdir() if d.is_dir()):
            ref = np.load(tdir / f"target_{bb}_embedding.npy").astype(np.float32).ravel()
            ref /= np.linalg.norm(ref)
            s = (emb @ ref + 1) / 2
            for upd, sim in [(False, s), (True, (emb @ emb[int(np.argmax(s))] + 1) / 2)]:
                cur = np.full(len(uf), -np.inf)
                np.maximum.at(cur, inv, sim)
                out.setdefault(upd, {}).setdefault(tdir.name, {})[bb] = cur
        ts = uf / fps
    return ts, out


def process(video):
    res = {}
    for stride in STRIDES:
        ts, curves = curves_for(video, stride)
        _, window = sampling(video, stride)
        for upd in [False, True]:
            for bb in ["arcface", "transface"]:
                for method in ["SEQ", "INS"]:
                    for src in ["GT", "SportCLIP", "QD-DETR"]:
                        segs, labels = E.run_pvs(video, src, bb, method, False, window=window, curves_override=(ts, curves[upd]))
                        res[f"{stride}|{src}|{bb}|{method}|{'upd' if upd else 'orig'}"] = E.video_scores(segs, labels, E.load_gt(video))[2]
    return video, res


if __name__ == "__main__":
    with Pool(5) as p:
        per_video = dict(p.map(process, E.VIDEOS))
    keys = per_video[E.VIDEOS[0]].keys()
    out = {k: {"F": float(np.mean([per_video[v][k] for v in E.VIDEOS])), "per_video_F": {v: per_video[v][k] for v in E.VIDEOS}} for k in keys}
    for src in ["QD-DETR", "SportCLIP", "GT"]:
        for bb in ["transface", "arcface"]:
            for method in ["SEQ", "INS"]:
                for upd in ["upd", "orig"]:
                    print(f"{src:9s} {bb:9s} {method} {upd}: " + "  ".join(f"k={s}: {100*out[f'{s}|{src}|{bb}|{method}|{upd}']['F']:.1f}" for s in STRIDES))
    E.save_json(out, "sparse_sampling.json")
