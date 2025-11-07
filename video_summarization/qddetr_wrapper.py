"""
QD-DETR Wrapper Script

This script is called by the PVS pipeline to run QD-DETR processing.
It must be executed with the QD-DETR virtual environment.

Usage:
    python qddetr_wrapper.py --annotation-path ANNOTATION.jsonl --sample-idx 0 \
        --output-dir DIR --ckpt-path CKPT --feat-dirs DIR1 DIR2 --text-feat-dir DIR
"""

import argparse
import json
import os
import sys
import subprocess
from pathlib import Path

# Add QD-DETR project to path
QDDETR_PROJECT_PATH = "/home/mrt/Projects/QD-DETR"
if QDDETR_PROJECT_PATH not in sys.path:
    sys.path.insert(0, QDDETR_PROJECT_PATH)

# Import QD-DETR components
from sports.process_saliency_curves import SaliencyProcessor, process_saliency_scores
from utils.basic_utils import load_jsonl


def extract_highlight_videos(
    video_path: str,
    video_name: str,
    events: dict,
    fps: float,
    output_dir: str,
) -> list:
    """Extract individual highlight videos from detected events."""
    highlight_videos = []
    output_path = Path(output_dir)

    for event_id, event_data in events.items():
        frames = event_data["frames"]
        if not frames:
            continue

        # Get start and end frames
        start_frame = min(frames)
        end_frame = max(frames)

        # Convert to timestamps (seconds)
        start_time = start_frame / fps
        duration = (end_frame - start_frame + 1) / fps

        # Generate output filename
        output_filename = f"{video_name}_highlight_{event_id:03d}.mp4"
        output_file = output_path / output_filename

        # Extract video segment using ffmpeg
        try:
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file
                "-ss", str(start_time),  # Start time
                "-i", video_path,  # Input file
                "-t", str(duration),  # Duration
                "-c:v", "libx264",  # Video codec
                "-c:a", "aac",  # Audio codec
                "-strict", "experimental",
                "-preset", "fast",
                "-crf", "23",
                str(output_file),
            ]

            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )

            highlight_videos.append(str(output_file))
            print(f"  ✓ Extracted: {output_filename} (frames {start_frame}-{end_frame}, {duration:.2f}s)")

        except subprocess.CalledProcessError as e:
            print(f"  ✗ Failed to extract {output_filename}: {e.stderr}")
            continue

    return highlight_videos


def main():
    parser = argparse.ArgumentParser(description="QD-DETR Wrapper for PVS Pipeline")

    # Required arguments
    parser.add_argument("--annotation-path", required=True, help="Path to annotation JSONL file")
    parser.add_argument("--sample-idx", type=int, required=True, help="Sample index in the annotation file to process")
    parser.add_argument("--video-path", required=True, help="Full path to input video file")
    parser.add_argument("--output-dir", required=True, help="Output directory for highlight videos")
    parser.add_argument("--ckpt-path", required=True, help="Path to QD-DETR checkpoint")
    parser.add_argument("--feat-dirs", nargs="+", required=True, help="Video feature directories")
    parser.add_argument("--text-feat-dir", required=True, help="Text feature directory")

    # Optional arguments
    parser.add_argument("--temp-dir", default=None, help="Temporary directory (default: output-dir/temp)")
    parser.add_argument("--device", default="cuda:0", help="Device for inference")
    parser.add_argument("--context-window", type=int, default=600, help="Context window size in frames")
    parser.add_argument("--instant-divisor", type=int, default=10, help="Instant window divisor")
    parser.add_argument("--closing-divisor", type=int, default=10, help="Closing kernel divisor")
    parser.add_argument("--min-duration", type=int, default=15, help="Minimum event duration in frames")
    parser.add_argument("--min-area", type=float, default=15.0, help="Minimum event area threshold")
    parser.add_argument("--expansion-mode", default="duplication", choices=["duplication", "interpolation"], help="Clip-to-frame expansion mode")

    args = parser.parse_args()

    # Validate paths
    if not Path(args.annotation_path).exists():
        print(f"ERROR: Annotation file not found: {args.annotation_path}")
        sys.exit(1)
    if not Path(args.video_path).exists():
        print(f"ERROR: Video file not found: {args.video_path}")
        sys.exit(1)
    if not Path(args.ckpt_path).exists():
        print(f"ERROR: Checkpoint not found: {args.ckpt_path}")
        sys.exit(1)

    # Load annotation to get video name and query
    annotations = load_jsonl(args.annotation_path)
    if args.sample_idx >= len(annotations):
        print(f"ERROR: Sample index {args.sample_idx} out of range (file has {len(annotations)} samples)")
        sys.exit(1)

    annotation = annotations[args.sample_idx]
    video_name = annotation["vid"]
    query = annotation["query"]

    print(f"\nQD-DETR Wrapper: Processing sample {args.sample_idx}")
    print(f"  Video: {video_name}")
    print(f"  Query: {query}")

    # Setup directories
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if args.temp_dir is None:
        temp_dir = str(output_path / "temp")
    else:
        temp_dir = args.temp_dir
    temp_path = Path(temp_dir)
    temp_path.mkdir(parents=True, exist_ok=True)

    # Calculate derived parameters
    instant_window = int(args.context_window / args.instant_divisor)
    closing_kernel = int(args.context_window / args.closing_divisor + 1)

    # Step 1: Run QD-DETR saliency processing
    print("\n[1/3] Running QD-DETR saliency score processing...")
    print(f"  - Using pre-extracted features from: {args.feat_dirs}")

    # Initialize processor with pre-extracted features
    print(f"  - Loading QD-DETR model (checkpoint: {Path(args.ckpt_path).name})...")
    processor = SaliencyProcessor(
        ckpt_path=args.ckpt_path,
        device=args.device,
        use_preextracted_features=True,
        clip_model_name_or_path="ViT-B/32",  # Not used for pre-extracted
    )

    # Load dataset with pre-extracted features
    print("  - Loading dataset with pre-extracted features...")
    dataset = processor.load_dataset(
        annotation_path=args.annotation_path,
        feat_dirs=args.feat_dirs,
        text_feat_dir=args.text_feat_dir,
    )
    print(f"  - Dataset loaded with {len(dataset)} samples")

    # Process saliency scores
    print(f"  - Processing saliency scores for sample {args.sample_idx}...")
    results = process_saliency_scores(
        processor=processor,
        dataset=dataset,
        sample_idx=args.sample_idx,
        output_dir=str(temp_path),
        context_window=args.context_window,
        instant_window=instant_window,
        closing_kernel=closing_kernel,
        min_duration=args.min_duration,
        min_area=args.min_area,
        video_base_path=str(Path(args.video_path).parent),
        gt_base_path="",  # Not needed
        expansion_mode=args.expansion_mode,
        draw_plots=False,
        frames_to_plot=[0, 1000],
        clip_len=2,
        ckpt_path=args.ckpt_path,
        annotation_path=args.annotation_path,
        device=args.device,
        instant_divisor=args.instant_divisor,
        closing_divisor=args.closing_divisor,
        feat_dirs=args.feat_dirs,
        text_feat_dir=args.text_feat_dir,
        use_preextracted_features=True,
    )

    print(f"✓ QD-DETR processing completed")
    print(f"  - Detected {results['n_events_filtered']} highlight events")

    # Step 2: Extract highlight videos
    print(f"\n[2/3] Extracting {results['n_events_filtered']} highlight videos...")
    highlight_videos = extract_highlight_videos(
        video_path=args.video_path,
        video_name=video_name,
        events=results["events_filtered"],
        fps=results["fps"],
        output_dir=args.output_dir,
    )

    print(f"✓ Successfully extracted {len(highlight_videos)} highlight videos")

    # Step 3: Save results
    print("\n[3/3] Saving results...")

    # Save list of generated highlight videos
    results_file = output_path / f"{video_name}_highlights.json"
    with open(results_file, "w") as f:
        json.dump({"highlight_videos": highlight_videos}, f, indent=2)

    print(f"\n✓ QD-DETR wrapper completed successfully")
    print(f"  Results saved to: {results_file}")


if __name__ == "__main__":
    main()
