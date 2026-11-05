"""Aggregate the runtime measurements into the numbers of the revised Tables 10-11."""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

R = Path("/mnt/Data/mrt/PVS_revision_cache/runtime")
load = lambda p: json.load(open(p))

sparse = {v: {"det": load(R / f"{v}_s10_detect.json"), "tf": load(R / f"{v}_s10_transface.json"), "af": load(R / f"{v}_s10_arcface.json")} for v in E.VIDEOS}
# dense = every native frame (stride 0); for 25 fps videos this is the same as stride 1
dense = {p.name.replace("_s1_detect.json", ""): load(p) for p in R.glob("*_s1_detect.json")}
dense.update({p.name.replace("_s0_detect.json", ""): load(p) for p in R.glob("*_s0_detect.json")})
assert all(d["stride_frames"] == 1 for d in dense.values()), "dense timing must process every native frame"
fusion = load(os.environ.get("PVS_FUSION_FILE", R / "fusion_s10_bench.json"))  # overridable, see pvs_eval.OUT_DIR

out = {}
# unit costs (sparse runs over all 20 videos)
proc = sum(s["det"]["frames_processed"] for s in sparse.values())
read = sum(s["det"]["frames_read"] for s in sparse.values())
faces = sum(s["det"]["faces"] for s in sparse.values())
out["detect_ms_per_frame"] = 1000 * sum(s["det"]["detect_s"] for s in sparse.values()) / proc
out["decode_read_ms_per_frame"] = 1000 * sum(s["det"]["decode_s"] for s in sparse.values()) / read  # sequential decoding, sparse retrieval
out["crop_ms_per_face"] = 1000 * sum(s["det"]["crop_s"] for s in sparse.values()) / faces
out["transface_ms_per_face"] = 1000 * sum(s["tf"]["embed_s"] for s in sparse.values()) / faces
out["arcface_ms_per_face"] = 1000 * sum(s["af"]["embed_s"] for s in sparse.values()) / faces
out["faces_per_frame_sparse"] = faces / proc
out["fusion_s_per_video"] = float(np.mean([fusion[k]["fusion_s_total"] / fusion[k]["videos"] for k in fusion]))
video_min = sum(s["det"]["video_seconds"] for s in sparse.values()) / 60

# sparse end-to-end face stream per minute of video (all 20 videos)
front = sum(s["det"]["wall_s"] for s in sparse.values())
for bb, key in [("transface", "tf"), ("arcface", "af")]:
    emb = sum(s[key]["embed_s"] for s in sparse.values())
    fus = np.mean([fusion[k]["fusion_s_total"] for k in fusion if f"|{bb}|" in k])
    out[f"sparse10_{bb}_s_per_video_min"] = (front + emb + fus) / video_min
out["sparse10_frontend_s_per_video_min"] = front / video_min

# dense (all 20 videos): measured decoding+detection, embeddings = measured faces x measured per-face cost
if dense:
    dmin = sum(d["video_seconds"] for d in dense.values()) / 60
    dfront = sum(d["wall_s"] for d in dense.values())
    dfaces = sum(d["faces"] for d in dense.values())
    out["dense_videos"] = sorted(dense)
    out["dense_detect_ms_per_frame"] = 1000 * sum(d["detect_s"] for d in dense.values()) / sum(d["frames_processed"] for d in dense.values())
    out["dense_decode_ms_per_frame"] = 1000 * sum(d["decode_s"] for d in dense.values()) / sum(d["frames_read"] for d in dense.values())
    out["dense_faces_per_frame"] = dfaces / sum(d["frames_processed"] for d in dense.values())
    out["dense_frontend_s_per_video_min"] = dfront / dmin
    assert sorted(dense) == sorted(E.VIDEOS), "dense timing must cover the 20 benchmark videos"
    for bb, key in [("transface", "tf"), ("arcface", "af")]:
        # fusion on dense curves: 10x the samples of the stride-10 run (negligible either way)
        out[f"dense_{bb}_s_per_video_min"] = (dfront + dfaces * out[f"{bb}_ms_per_face"] / 1000 + out["fusion_s_per_video"] * 10 * len(dense)) / dmin

# summarisers
for name in ["sportclip", "qddetr"]:
    files = list(R.glob(f"*{name}.json")) if name == "sportclip" else [Path(os.environ.get("PVS_QDDETR_TIMING_FILE", R / "qddetr_bench.json"))]
    files = [f for f in files if f.exists()]
    if files:
        out[name] = [load(f) for f in files]

# accuracy of the measured sparse run (TransFace) vs dense
grid = json.load(open(E.OUT_DIR / "main_grid.json"))
for k in ["QD-DETR|transface|SEQ|upd", "SportCLIP|transface|SEQ|upd", "QD-DETR|transface|INS|upd", "GT|transface|SEQ|upd"]:
    src, bb, meth, upd = k.split("|")
    out[f"acc|{k}"] = {"dense": grid[src][f"{bb}|{meth}|{upd}"]["overall"][2], "sparse10_measured": fusion[k]["F"]}
print(json.dumps(out, indent=1))
E.save_json(out, "runtime_summary.json")
