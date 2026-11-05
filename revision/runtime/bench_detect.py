"""Measured runtime of the face-analysis front end (decoding + SCRFD detection + cropping).

Run in the PVS environment through run_with_cuda.sh:
    ./run_with_cuda.sh python revision/runtime/bench_detect.py --video highjump_video1 --stride 10 [--seconds 300]

Only the SCRFD detector (det_10g of InsightFace buffalo_l, 640x640, threshold 0.5)
is executed, and faces smaller than 50x50 pixels are discarded as in the PVS
extraction (extract_all_faces.py --min-face-size 50). The original timing called FaceAnalysis.get(), which additionally
runs landmark, gender/age and recognition models on every face. Frames are read
sequentially; skipped frames are only grabbed (decoded but not converted).
Crops (112x112, BGR) are stored for the embedding benchmarks.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from insightface.model_zoo import get_model

CACHE = Path("/mnt/Data/mrt/PVS_revision_cache/runtime")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--stride", type=int, default=1, help="process one frame every `stride` frames (25 fps reference); 0 = every native frame")
    ap.add_argument("--seconds", type=float, default=None, help="only the first N seconds")
    ap.add_argument("--min-face-size", type=int, default=50)
    ap.add_argument("--no-save", action="store_true", help="do not keep crops (dense timing runs)")
    args = ap.parse_args()

    det = get_model(str(Path.home() / ".insightface/models/buffalo_l/det_10g.onnx"), providers=["CUDAExecutionProvider"])
    det.prepare(ctx_id=0, input_size=(640, 640), det_thresh=0.5)

    cap = cv2.VideoCapture(f"data/{args.video}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    # same temporal sampling for 25/50 fps videos; stride 0 = every native frame (dense pipeline)
    stride = 1 if args.stride == 0 else max(1, round(args.stride * fps / 25.0))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = n_total if args.seconds is None else min(n_total, int(args.seconds * fps))

    # warm-up
    ok, img = cap.read()
    for _ in range(5):
        det.detect(img, max_num=0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    t_decode = t_det = t_crop = 0.0
    crops, frame_idx, processed = [], [], 0
    wall0 = time.perf_counter()
    for f in range(n_frames):
        t0 = time.perf_counter()
        if not cap.grab():
            break
        if f % stride:
            t_decode += time.perf_counter() - t0
            continue
        ok, img = cap.retrieve()
        t1 = time.perf_counter()
        bboxes, _ = det.detect(img, max_num=0)
        t2 = time.perf_counter()
        h, w = img.shape[:2]
        for b in bboxes:
            x1, y1, x2, y2 = b[:4].astype(int)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 >= args.min_face_size and y2 - y1 >= args.min_face_size:
                crop = cv2.resize(img[y1:y2, x1:x2], (112, 112))
                frame_idx.append(f)
                if not args.no_save:
                    crops.append(crop)
        t3 = time.perf_counter()
        t_decode += t1 - t0
        t_det += t2 - t1
        t_crop += t3 - t2
        processed += 1
    wall = time.perf_counter() - wall0

    tag = f"{args.video}_s{args.stride}" + (f"_{int(args.seconds)}s" if args.seconds else "")
    CACHE.mkdir(parents=True, exist_ok=True)
    if not args.no_save:
        np.save(CACHE / f"{tag}_crops.npy", np.array(crops, dtype=np.uint8).reshape(-1, 112, 112, 3))
    np.save(CACHE / f"{tag}_frames.npy", np.array(frame_idx, dtype=np.int32))
    res = {"video": args.video, "stride_25fps": args.stride, "stride_frames": stride, "fps": fps, "video_seconds": n_frames / fps,
           "frames_read": n_frames, "frames_processed": processed, "faces": len(frame_idx), "faces_per_processed_frame": len(frame_idx) / max(processed, 1),
           "decode_s": t_decode, "detect_s": t_det, "crop_s": t_crop, "wall_s": wall,
           "detect_ms_per_processed_frame": 1000 * t_det / max(processed, 1)}
    with open(CACHE / f"{tag}_detect.json", "w") as fh:
        json.dump(res, fh, indent=2)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
