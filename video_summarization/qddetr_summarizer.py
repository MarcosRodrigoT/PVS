"""
QD-DETR Video Summarizer

This module integrates with the QD-DETR framework to generate video highlights.
It calls the QD-DETR wrapper script using subprocess with the QD-DETR virtual environment.
"""

import os
import json
import subprocess
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import cv2


def generate_highlights(
    video_name: str,
    query: str,
    output_dir: str,
    video_path: str,
    qddetr_ckpt_path: str,
    context_window: int = 600,
    instant_divisor: int = 10,
    closing_divisor: int = 10,
    min_duration_frames: int = 15,
    min_area: float = 15.0,
    expansion_mode: str = "duplication",
    device: str = "cuda:0",
    clip_model_name: str = "ViT-B/32",
    temp_dir: Optional[str] = None,
    qddetr_project_dir: str = "/home/mrt/Projects/QD-DETR",
    feat_dirs: Optional[List[str]] = None,
    text_feat_dir: Optional[str] = None,
    annotation_path: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    """
    Generate video highlights using QD-DETR framework.

    This function:
    1. Calls the QD-DETR wrapper script with the QD-DETR virtual environment
    2. The wrapper runs saliency score extraction and postprocessing
    3. Extracts individual highlight videos for each detected event
    4. Returns paths to the generated highlight videos

    Args:
        video_name: Name of the video to process (without extension)
        query: Text query describing the desired highlights (e.g., "highlights of a long jump competition")
        output_dir: Directory to save generated highlight videos
        video_path: Full path to the input video file
        qddetr_ckpt_path: Path to trained QD-DETR model checkpoint
        context_window: Context window size in frames (default: 600)
        instant_divisor: Divisor for instant window calculation (default: 10)
        closing_divisor: Divisor for closing kernel calculation (default: 10)
        min_duration_frames: Minimum event duration in frames (default: 15)
        min_area: Minimum event area threshold (default: 15.0)
        expansion_mode: Method to expand clips to frames - "duplication" or "interpolation" (default: "duplication")
        device: Device to run inference on (default: "cuda:0")
        clip_model_name: CLIP model name for feature extraction (default: "ViT-B/32")
        temp_dir: Temporary directory for intermediate results (default: output_dir/temp)
        qddetr_project_dir: Path to QD-DETR project directory (default: /home/mrt/Projects/QD-DETR)
        feat_dirs: List of video feature directories for pre-extracted features (required for Olympic videos)
        text_feat_dir: Text feature directory for pre-extracted features (required for Olympic videos)
        annotation_path: Path to QD-DETR annotation JSONL file (required for Olympic videos)

    Returns:
        Tuple of (list of paths to highlight videos, list of segment metadata dicts)
        Each segment metadata dict contains: start_frame, end_frame, start_time, end_time, clip_path

    Raises:
        FileNotFoundError: If video file, checkpoint, or QD-DETR project not found
        subprocess.CalledProcessError: If QD-DETR wrapper script fails
    """
    print(f"\nRunning QD-DETR pipeline on: {video_name}")
    print(f"Query: {query}")

    # Validate paths
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    if not Path(qddetr_ckpt_path).exists():
        raise FileNotFoundError(f"QD-DETR checkpoint not found: {qddetr_ckpt_path}")

    qddetr_path = Path(qddetr_project_dir)
    if not qddetr_path.exists():
        raise FileNotFoundError(f"QD-DETR project directory not found: {qddetr_project_dir}")

    # For pre-extracted features, annotation path is required
    if feat_dirs and text_feat_dir:
        if not annotation_path:
            raise ValueError("annotation_path is required when using pre-extracted features (feat_dirs/text_feat_dir)")
        if not Path(annotation_path).exists():
            raise FileNotFoundError(f"Annotation file not found: {annotation_path}")

        # Find the sample index for this video/query in the annotation file
        import sys
        sys.path.insert(0, str(qddetr_path))
        from utils.basic_utils import load_jsonl

        annotations = load_jsonl(annotation_path)
        sample_idx = None
        for idx, ann in enumerate(annotations):
            if ann.get("vid") == video_name:
                sample_idx = idx
                break

        if sample_idx is None:
            raise ValueError(f"Could not find video '{video_name}' in annotation file: {annotation_path}")

        print(f"Found video '{video_name}' at sample index {sample_idx} in annotation file")

    # Get Python executable from QD-DETR virtual environment
    qddetr_python = qddetr_path / "venv" / "bin" / "python"
    if not qddetr_python.exists():
        raise FileNotFoundError(
            f"QD-DETR virtual environment not found: {qddetr_python}\n"
            f"Please ensure QD-DETR venv exists at {qddetr_path}/venv/"
        )

    # Get wrapper script path (in same directory as this file)
    wrapper_script = Path(__file__).parent / "qddetr_wrapper.py"
    if not wrapper_script.exists():
        raise FileNotFoundError(f"QD-DETR wrapper script not found: {wrapper_script}")

    # Setup output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if temp_dir is None:
        temp_dir = str(output_path / "temp")

    # Build command to run wrapper script with QD-DETR environment
    # IMPORTANT: Use absolute paths since subprocess runs from QD-DETR directory
    cmd = [
        str(qddetr_python),
        str(wrapper_script),
        "--annotation-path", str(Path(annotation_path).resolve()),
        "--sample-idx", str(sample_idx),
        "--video-path", str(Path(video_path).resolve()),
        "--output-dir", str(output_path.resolve()),
        "--ckpt-path", str(Path(qddetr_ckpt_path).resolve()),
        "--feat-dirs"] + [str(Path(d).resolve()) for d in feat_dirs] + [
        "--text-feat-dir", str(Path(text_feat_dir).resolve()),
        "--temp-dir", str(Path(temp_dir).resolve()),
        "--device", device,
        "--context-window", str(context_window),
        "--instant-divisor", str(instant_divisor),
        "--closing-divisor", str(closing_divisor),
        "--min-duration", str(min_duration_frames),
        "--min-area", str(min_area),
        "--expansion-mode", expansion_mode,
    ]

    # Run QD-DETR wrapper script
    print("\n[QD-DETR] Running wrapper script with QD-DETR environment...")
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(qddetr_path),  # Run from QD-DETR directory for proper imports
        )

        # Print wrapper output
        if result.stdout:
            print(result.stdout)

    except subprocess.CalledProcessError as e:
        print(f"✗ QD-DETR wrapper failed with error:")
        print(e.stderr if e.stderr else e.stdout)
        raise RuntimeError(f"QD-DETR processing failed: {e}")

    # Read generated highlight videos list
    results_file = output_path / f"{video_name}_highlights.json"
    if not results_file.exists():
        raise RuntimeError(f"QD-DETR results file not found: {results_file}")

    with open(results_file, "r") as f:
        results = json.load(f)

    highlight_videos = results.get("highlight_videos", [])

    if not highlight_videos:
        print("⚠ Warning: No highlight videos were generated")

    # Extract timing information from events_filtered.json
    events_file = output_path / "events_filtered.json"
    segments_metadata = []

    if events_file.exists():
        try:
            with open(events_file, "r") as f:
                events_data = json.load(f)

            for event_id_str, event_info in sorted(events_data.items(), key=lambda x: int(x[0])):
                event_id = int(event_id_str)
                # Find corresponding video path
                video_file = highlight_videos[event_id] if event_id < len(highlight_videos) else None
                clip_name = Path(video_file).name if video_file else f"highlight_{event_id:03d}.mp4"

                segments_metadata.append({
                    "segment_id": event_id + 1,
                    "clip_path": clip_name,
                    "clip_name": clip_name,
                    "start_frame": event_info.get("start_frame"),
                    "end_frame": event_info.get("end_frame"),
                    "start_time": event_info.get("start_time"),
                    "end_time": event_info.get("end_time"),
                    "duration_frames": event_info.get("duration_frames"),
                    "duration_seconds": event_info.get("duration_seconds")
                })

            print(f"✓ Extracted timing for {len(segments_metadata)} segments")
        except Exception as e:
            print(f"Warning: Could not load timing information: {e}")
            # Fallback without timing
            for i, video_file in enumerate(highlight_videos):
                segments_metadata.append({
                    "segment_id": i + 1,
                    "clip_path": str(Path(video_file).name),
                    "clip_name": str(Path(video_file).name),
                    "start_frame": None,
                    "end_frame": None,
                    "start_time": None,
                    "end_time": None
                })
    else:
        print(f"Warning: events_filtered.json not found")
        # Fallback without timing
        for i, video_file in enumerate(highlight_videos):
            segments_metadata.append({
                "segment_id": i + 1,
                "clip_path": str(Path(video_file).name),
                "clip_name": str(Path(video_file).name),
                "start_frame": None,
                "end_frame": None,
                "start_time": None,
                "end_time": None
            })

    return highlight_videos, segments_metadata
