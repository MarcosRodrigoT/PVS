"""
Generate QD-DETR highlight proposals for each evaluation video with the fold
model that did not see that video during training.

Inference follows the original PVS pipeline exactly (video_summarization/
qddetr_wrapper.py): generic query "highlight moments in this sports video",
pre-extracted SlowFast+CLIP features, 2-s clips, min-max normalisation of the
saliency curve, expansion to frame level by duplication, and the TFM/SportCLIP
post-processing with context window 600 frames, instant window 60 frames,
closing kernel 61 frames, minimum duration 15 frames and minimum area 15.
Only the ffmpeg clip export of the wrapper is skipped (not needed for evaluation).

Must be run with the QD-DETR virtual environment from the QD-DETR directory:
    cd /home/mrt/Projects/QD-DETR && source venv/bin/activate
    python /home/mrt/Projects/PVS/revision/qddetr_cv/infer_heldout.py [--ckpt-override CKPT --out-subdir NAME]
The in-sample reference proposals are produced with
    --ckpt-override results/hl-video_tef-olympic_bench_all-*/model_best.ckpt --out-subdir QD-DETR-insample-bench
"""

import argparse
import contextlib
import glob
import io
import json
import os
import sys
import tempfile
from pathlib import Path

QDDETR_DIR = Path("/home/mrt/Projects/QD-DETR")
PVS_DIR = Path("/home/mrt/Projects/PVS")
sys.path.insert(0, str(QDDETR_DIR))
os.chdir(QDDETR_DIR)

from sports.process_saliency_curves import SaliencyProcessor, process_saliency_scores  # noqa: E402
from utils.basic_utils import load_jsonl  # noqa: E402

GENERIC_ANN = QDDETR_DIR / "data/OlympicHighlights_annotations/olympic_highlights_train_generic.jsonl"
FEAT_DIRS = ["/mnt/Data/mrt/QD-DETR/olympic/slowfast_features", "/mnt/Data/mrt/QD-DETR/olympic/clip_features"]
TEXT_FEAT_DIR = "/mnt/Data/mrt/QD-DETR/olympic/clip_text_features_generic"

EVENTS = {"highjump": "High Jump", "javelin": "Javelin", "longjump": "Long Jump", "polevault": "Pole Vault"}
CONTEXT_WINDOW = 600
INSTANT_DIVISOR = 10
CLOSING_DIVISOR = 10
MIN_DURATION = 15
MIN_AREA = 15.0


def fold_checkpoint(fold_idx, prefix="olympic_bench_heldout_fold"):
    dirs = sorted(glob.glob(str(QDDETR_DIR / f"results/hl-video_tef-{prefix}{fold_idx}-*")))
    if not dirs:
        raise FileNotFoundError(f"No trained model for fold {fold_idx}")
    return os.path.join(dirs[-1], "model_best.ckpt")


def run_video(processor, dataset, sample_idx, video_name, link_dir, device, ckpt):
    with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
        process_saliency_scores(
            processor=processor,
            dataset=dataset,
            sample_idx=sample_idx,
            output_dir=tmp,
            context_window=CONTEXT_WINDOW,
            instant_window=int(CONTEXT_WINDOW / INSTANT_DIVISOR),
            closing_kernel=int(CONTEXT_WINDOW / CLOSING_DIVISOR + 1),
            min_duration=MIN_DURATION,
            min_area=MIN_AREA,
            video_base_path=str(link_dir),
            gt_base_path="",
            expansion_mode="duplication",
            draw_plots=False,
            frames_to_plot=[0, 1000],
            clip_len=2,
            ckpt_path=ckpt,
            annotation_path=str(GENERIC_ANN),
            device=device,
            instant_divisor=INSTANT_DIVISOR,
            closing_divisor=CLOSING_DIVISOR,
            feat_dirs=FEAT_DIRS,
            text_feat_dir=TEXT_FEAT_DIR,
            use_preextracted_features=True,
        )
        res = json.load(open(next(Path(tmp).glob("*_results.json"))))
        sys.stdout.flush()

    fps = res["fps"]
    segments = []
    for event_id, ev in sorted(res["events_filtered"].items(), key=lambda kv: int(kv[0])):
        frames = ev["frames"]
        start_frame, end_frame = int(min(frames)), int(max(frames))
        segments.append(
            {
                "segment_id": int(event_id) + 1,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "start_time": start_frame / fps,
                "end_time": end_frame / fps,
                "duration_frames": len(frames),
                "duration_seconds": len(frames) / fps,
            }
        )
    return {"video_name": video_name, "model": "qddetr", "checkpoint": ckpt, "fps": fps, "query": res["query"], "total_segments": len(segments), "segments": segments}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-override", default=None, help="Use this checkpoint for every video (sanity check)")
    parser.add_argument("--out-subdir", default="QD-DETR-CV-bench", help="Output folder inside PVS/video_summaries")
    parser.add_argument("--exp-prefix", default="olympic_bench_heldout_fold", help="exp_id prefix of the fold models")
    parser.add_argument("--videos", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    annotations = load_jsonl(str(GENERIC_ANN))
    index_by_vid = {a["vid"]: i for i, a in enumerate(annotations)}

    # QD-DETR finds the video (for its FPS) by the annotation name, e.g. "High Jump_video1.mp4"
    link_dir = Path(tempfile.mkdtemp(prefix="qddetr_links_"))
    for event, pretty in EVENTS.items():
        for i in range(1, 6):
            (link_dir / f"{pretty}_video{i}.mp4").symlink_to(PVS_DIR / "data" / f"{event}_video{i}.mp4")

    videos = args.videos or [f"{e}_video{i}" for e in EVENTS for i in range(1, 6)]
    loaded = {}
    for video in videos:
        event, vid = video.split("_")
        fold_idx = int(vid.replace("video", "")) - 1
        ckpt = args.ckpt_override or fold_checkpoint(fold_idx, args.exp_prefix)
        if ckpt not in loaded:
            with contextlib.redirect_stdout(io.StringIO()):
                processor = SaliencyProcessor(ckpt_path=ckpt, device=args.device, use_preextracted_features=True)
                dataset = processor.load_dataset(str(GENERIC_ANN), FEAT_DIRS, TEXT_FEAT_DIR)
            loaded = {ckpt: (processor, dataset)}
        processor, dataset = loaded[ckpt]

        ann_name = f"{EVENTS[event]}_{vid}"
        out = run_video(processor, dataset, index_by_vid[ann_name], video, link_dir, args.device, ckpt)
        out_dir = PVS_DIR / "video_summaries" / args.out_subdir / video
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(out_dir / "segments.json", "w") as f:
            json.dump(out, f, indent=2)
        print(f"{video}: {out['total_segments']} segments (fold ckpt: {Path(ckpt).parent.name})", flush=True)


if __name__ == "__main__":
    main()
