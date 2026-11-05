"""Cache per-video face embeddings as dense arrays (frame index + float16 embeddings).

Reads cropped_faces_embeddings/{ArcFace,TransFace}/<video>.h5 (one dataset per
detected face, key frame_XXXXXX_face_K) and writes
<CACHE>/<video>_<backbone>.npz with arrays frame (int32), face (int16),
emb (float16, L2-normalised). Used by the absent-target and sparse-sampling experiments.
"""
import sys
from multiprocessing import Pool
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pvs_eval as E

CACHE = Path("/mnt/Data/mrt/PVS_revision_cache")


def cache(args):
    video, bb = args
    out = CACHE / f"{video}_{bb}.npz"
    if out.exists():
        return out.name
    with h5py.File(E.EMBEDDINGS_DIR / ("ArcFace" if bb == "arcface" else "TransFace") / f"{video}.h5", "r") as h:
        keys = sorted(h.keys())
        frame = np.array([int(k.split("_")[1]) for k in keys], dtype=np.int32)
        face = np.array([int(k.split("_")[3]) for k in keys], dtype=np.int16)
        emb = np.stack([h[k][:] for k in keys]).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    order = np.lexsort((face, frame))
    np.savez(out, frame=frame[order], face=face[order], emb=emb[order].astype(np.float16), keys=np.array(keys)[order])
    return out.name


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    jobs = [(v, bb) for v in E.VIDEOS for bb in ["arcface", "transface"]]
    with Pool(8) as p:
        for name in p.imap_unordered(cache, jobs):
            print(name, flush=True)
