"""
Generate Video Summaries
Extract highlight segments from videos using SportCLIP or QD-DETR

This script:
1. Finds all videos in the data directory
2. Runs video summarization (SportCLIP or QD-DETR)
3. Saves highlight clips and segment timestamps
4. Organizes output by model and video name

Output structure:
    video_summaries/
    ├── SportCLIP/
    │   └── <video_name>/
    │       ├── clip_001.mp4
    │       ├── clip_002.mp4
    │       └── segments.json
    └── QD-DETR/
        └── <video_name>/
            ├── clip_001.mp4
            ├── clip_002.mp4
            └── segments.json
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Import video summarization modules
from video_summarization import generate_highlights_sportclip, generate_highlights_qddetr


def find_video_files(directory: str, video_name: Optional[str] = None) -> List[Path]:
    """
    Find video files in a directory

    Args:
        directory: Directory to search for videos
        video_name: Optional specific video name (without extension)

    Returns:
        List of video file paths
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]
    directory = Path(directory)

    if video_name:
        # Look for specific video with any supported extension
        for ext in video_extensions:
            video_path = directory / f"{video_name}{ext}"
            if video_path.exists():
                return [video_path]
        return []
    else:
        # Find all videos
        video_files = []
        for ext in video_extensions:
            video_files.extend(directory.glob(f"*{ext}"))
        return sorted(video_files)


def save_segments_metadata(segments: List[Dict], output_path: Path):
    """
    Save segment timestamps and metadata to JSON

    Args:
        segments: List of segment dictionaries with start/end times
        output_path: Path to save JSON file
    """
    data = {"total_segments": len(segments), "segments": segments}

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Segments metadata saved to: {output_path}")


def process_sportclip_summaries(video_path: Path, output_dir: Path, sportclip_dir: str, sportclip_dataset_dir: str, sentences_file: str) -> Optional[Dict]:
    """
    Process video using SportCLIP

    Args:
        video_path: Path to video file
        output_dir: Output directory for this video
        sportclip_dir: SportCLIP project directory
        sportclip_dataset_dir: SportCLIP dataset directory
        sentences_file: Path to sentences file

    Returns:
        Dictionary with clip paths and segment info, or None if failed
    """
    video_name = video_path.stem

    # Create output directory for this video
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    segments_file = video_output_dir / "segments.json"
    if segments_file.exists():
        print(f"\n  ✓ SportCLIP summaries already exist for {video_name}")
        print(f"    Skipping...")
        with open(segments_file, "r") as f:
            return json.load(f)

    print(f"\n  Processing {video_name} with SportCLIP...")

    try:
        # Generate highlights using SportCLIP
        clip_paths = generate_highlights_sportclip(
            video_name=video_name, sentences_file=sentences_file, output_dir=str(video_output_dir), sportclip_dir=sportclip_dir, sportclip_dataset_dir=sportclip_dataset_dir
        )

        if not clip_paths:
            print(f"  Warning: No highlights generated for {video_name}")
            return None

        # Extract segment information from clip filenames
        # SportCLIP clips are typically named with timestamps
        segments = []
        for i, clip_path in enumerate(clip_paths, 1):
            # Try to extract timestamps from filename if possible
            # Format varies, so we'll store basic info
            segments.append({"segment_id": i, "clip_path": str(Path(clip_path).relative_to(video_output_dir)), "clip_name": Path(clip_path).name})

        # Save segments metadata
        result = {"video_name": video_name, "total_segments": len(segments), "segments": segments}
        save_segments_metadata(result, segments_file)

        print(f"  ✓ Generated {len(clip_paths)} highlight clips")
        return result

    except Exception as e:
        print(f"  Error processing {video_name} with SportCLIP: {e}")
        return None


def process_qddetr_summaries(
    video_path: Path,
    output_dir: Path,
    qddetr_ckpt_path: str,
    qddetr_query: str,
    qddetr_device: str,
    qddetr_context_window: int,
    qddetr_min_duration: int,
    qddetr_min_area: float,
    qddetr_feat_dirs: Optional[List[str]],
    qddetr_text_feat_dir: Optional[str],
    qddetr_annotation_path: Optional[str],
) -> Optional[Dict]:
    """
    Process video using QD-DETR

    Args:
        video_path: Path to video file
        output_dir: Output directory for this video
        qddetr_ckpt_path: Path to QD-DETR checkpoint
        qddetr_query: Query string for QD-DETR
        (other QD-DETR parameters)

    Returns:
        Dictionary with clip paths and segment info, or None if failed
    """
    video_name = video_path.stem

    # Create output directory for this video
    video_output_dir = output_dir / video_name
    video_output_dir.mkdir(parents=True, exist_ok=True)

    # Check if already processed
    segments_file = video_output_dir / "segments.json"
    if segments_file.exists():
        print(f"\n  ✓ QD-DETR summaries already exist for {video_name}")
        print(f"    Skipping...")
        with open(segments_file, "r") as f:
            return json.load(f)

    print(f"\n  Processing {video_name} with QD-DETR...")

    try:
        # Generate highlights using QD-DETR
        clip_paths = generate_highlights_qddetr(
            video_name=video_name,
            query=qddetr_query,
            output_dir=str(video_output_dir),
            video_path=str(video_path),
            qddetr_ckpt_path=qddetr_ckpt_path,
            context_window=qddetr_context_window,
            min_duration_frames=qddetr_min_duration,
            min_area=qddetr_min_area,
            device=qddetr_device,
            feat_dirs=qddetr_feat_dirs,
            text_feat_dir=qddetr_text_feat_dir,
            annotation_path=qddetr_annotation_path,
        )

        if not clip_paths:
            print(f"  Warning: No highlights generated for {video_name}")
            return None

        # Extract segment information
        # QD-DETR typically provides temporal boundaries
        segments = []
        for i, clip_path in enumerate(clip_paths, 1):
            segments.append({"segment_id": i, "clip_path": str(Path(clip_path).relative_to(video_output_dir)), "clip_name": Path(clip_path).name})

        # Save segments metadata
        result = {"video_name": video_name, "total_segments": len(segments), "segments": segments}
        save_segments_metadata(result, segments_file)

        print(f"  ✓ Generated {len(clip_paths)} highlight clips")
        return result

    except Exception as e:
        print(f"  Error processing {video_name} with QD-DETR: {e}")
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Generate video summaries using SportCLIP or QD-DETR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all videos with SportCLIP
  python generate_video_summaries.py --model sportclip \\
      --sentences-file data/sentences/long_jump.json

  # Process specific video with SportCLIP
  python generate_video_summaries.py --model sportclip \\
      --video longjump_video1 \\
      --sentences-file data/sentences/long_jump.json

  # Process all videos with QD-DETR
  python generate_video_summaries.py --model qddetr \\
      --qddetr-query "highlight moments in this sports video"

  # Process specific video with QD-DETR
  python generate_video_summaries.py --model qddetr \\
      --video highjump_video1 \\
      --qddetr-query "highlight moments in this high jump competition"

  # Process with both models (run twice with different --model)
  python generate_video_summaries.py --model sportclip --sentences-file data/sentences/long_jump.json
  python generate_video_summaries.py --model qddetr --qddetr-query "highlights"
        """,
    )

    # Basic parameters
    parser.add_argument("--model", type=str, required=True, choices=["sportclip", "qddetr"], help="Video summarization model to use")
    parser.add_argument("--video", type=str, default=None, help="Specific video name to process (without extension). If not provided, processes all videos in data/")
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing video files (default: data)")
    parser.add_argument("--output-dir", type=str, default="video_summaries", help="Output directory for video summaries (default: video_summaries)")

    # SportCLIP parameters
    parser.add_argument("--sentences-file", type=str, default=None, help="Path to SportCLIP sentences file (required for SportCLIP)")
    parser.add_argument("--sportclip-dir", type=str, default="/home/mrt/Projects/SportCLIP-official", help="SportCLIP project directory")
    parser.add_argument("--sportclip-dataset-dir", type=str, default="/mnt/Data/mrt/SportCLIP-OlympicHighlights", help="SportCLIP dataset directory")

    # QD-DETR parameters
    parser.add_argument(
        "--qddetr-ckpt", type=str, default="/home/mrt/Projects/QD-DETR/results/OlympicHighlights-exp-2025_09_25_11_22_34/model_best.ckpt", help="Path to QD-DETR checkpoint"
    )
    parser.add_argument("--qddetr-query", type=str, default=None, help="Query for QD-DETR (required for QD-DETR)")
    parser.add_argument("--qddetr-device", type=str, default="cuda:0", help="Device for QD-DETR inference")
    parser.add_argument("--qddetr-context-window", type=int, default=600, help="Context window size in frames for QD-DETR postprocessing")
    parser.add_argument("--qddetr-min-duration", type=int, default=15, help="Minimum event duration in frames for QD-DETR")
    parser.add_argument("--qddetr-min-area", type=float, default=15.0, help="Minimum event area threshold for QD-DETR")
    parser.add_argument("--qddetr-feat-dirs", nargs="+", default=None, help="Video feature directories for QD-DETR")
    parser.add_argument("--qddetr-text-feat-dir", type=str, default=None, help="Text feature directory for QD-DETR")
    parser.add_argument("--qddetr-annotation-path", type=str, default=None, help="Path to QD-DETR annotation JSONL file")

    args = parser.parse_args()

    # Validate model-specific requirements
    if args.model == "sportclip" and not args.sentences_file:
        print("Error: --sentences-file is required when using SportCLIP")
        return

    if args.model == "qddetr" and not args.qddetr_query:
        print("Error: --qddetr-query is required when using QD-DETR")
        return

    print("=" * 80)
    print(f"GENERATE VIDEO SUMMARIES - {args.model.upper()}")
    print("=" * 80)

    # Find videos to process
    print(f"\nSearching for videos in: {args.data_dir}")
    video_files = find_video_files(args.data_dir, args.video)

    if not video_files:
        if args.video:
            print(f"\n⚠ Video not found: {args.video}")
        else:
            print(f"\n⚠ No video files found in {args.data_dir}")
        print("Supported formats: .mp4, .avi, .mov, .mkv")
        return

    print(f"✓ Found {len(video_files)} video(s) to process:")
    for video in video_files:
        print(f"  - {video.name}")

    # Create output directory structure
    output_base = Path(args.output_dir)
    if args.model == "sportclip":
        model_output_dir = output_base / "SportCLIP"
    else:
        model_output_dir = output_base / "QD-DETR"

    model_output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nOutput directory: {model_output_dir}")

    # Process each video
    successful = 0
    failed = 0

    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*80}")
        print(f"PROCESSING VIDEO {i}/{len(video_files)}: {video_path.name}")
        print(f"{'='*80}")

        if args.model == "sportclip":
            result = process_sportclip_summaries(
                video_path=video_path,
                output_dir=model_output_dir,
                sportclip_dir=args.sportclip_dir,
                sportclip_dataset_dir=args.sportclip_dataset_dir,
                sentences_file=args.sentences_file,
            )
        else:  # qddetr
            result = process_qddetr_summaries(
                video_path=video_path,
                output_dir=model_output_dir,
                qddetr_ckpt_path=args.qddetr_ckpt,
                qddetr_query=args.qddetr_query,
                qddetr_device=args.qddetr_device,
                qddetr_context_window=args.qddetr_context_window,
                qddetr_min_duration=args.qddetr_min_duration,
                qddetr_min_area=args.qddetr_min_area,
                qddetr_feat_dirs=args.qddetr_feat_dirs,
                qddetr_text_feat_dir=args.qddetr_text_feat_dir,
                qddetr_annotation_path=args.qddetr_annotation_path,
            )

        if result:
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Model: {args.model.upper()}")
    print(f"  Total videos: {len(video_files)}")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"\nResults saved to: {model_output_dir}/")


if __name__ == "__main__":
    main()
