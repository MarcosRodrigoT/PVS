"""
SportCLIP Video Summarizer

This module integrates with the SportCLIP framework to generate video highlights.
It handles running SportCLIP's multi_sentences.py and summarize.py scripts,
and manages the output highlight videos.
"""

import os
import subprocess
import shutil
import pickle
import cv2
from pathlib import Path
from typing import List, Optional, Dict, Tuple


def generate_highlights(
    video_name: str,
    sentences_file: str,
    output_dir: str,
    sportclip_dir: str = "/home/mrt/Projects/SportCLIP-official",
    sportclip_dataset_dir: str = "/mnt/Data/mrt/SportCLIP-OlympicHighlights",
    video_path: Optional[str] = None,
) -> Tuple[List[str], List[Dict]]:
    """
    Generate video highlights using SportCLIP framework.

    This function:
    1. Runs multi_sentences.py to analyze highlight/non-highlight sentence pairs
    2. Runs summarize.py to extract individual highlight videos
    3. Copies generated highlights to the specified output directory
    4. Extracts frame-level timing information for each highlight

    Args:
        video_name: Name of the video to process (without extension)
        sentences_file: Path to SportCLIP sentences JSON file (relative to sportclip_dir)
        output_dir: Directory to save generated highlight videos
        sportclip_dir: Path to SportCLIP project directory
        sportclip_dataset_dir: Path to SportCLIP dataset directory (with frames/embeddings)
        video_path: Optional path to original video file for FPS detection

    Returns:
        Tuple of (list of paths to highlight videos, list of segment metadata dicts)
        Each segment metadata dict contains: start_frame, end_frame, start_time, end_time, clip_path

    Raises:
        FileNotFoundError: If SportCLIP directory or sentences file not found
        subprocess.CalledProcessError: If SportCLIP scripts fail
    """
    print(f"\nRunning SportCLIP pipeline on: {video_name}")

    # Validate directories and files
    _validate_paths(sportclip_dir, sentences_file)

    # Get Python executable (use SportCLIP venv if available)
    python_exec = _get_python_executable(sportclip_dir)

    # Build paths
    sentences_file_path = Path(sportclip_dir) / sentences_file

    # Step 1: Run multi_sentences.py
    print("\n[1/2] Running multi_sentences.py to analyze highlight/non-highlight pairs...")
    _run_multi_sentences(
        python_exec=python_exec,
        sportclip_dir=sportclip_dir,
        dataset_dir=sportclip_dataset_dir,
        video_name=video_name,
        sentences_file=str(sentences_file_path),
    )
    print("✓ multi_sentences.py completed successfully")

    # Step 2: Run summarize.py
    print("\n[2/2] Running summarize.py to generate individual highlight videos...")
    _run_summarize(
        python_exec=python_exec,
        sportclip_dir=sportclip_dir,
        dataset_dir=sportclip_dataset_dir,
        video_name=video_name,
    )
    print("✓ summarize.py completed successfully")

    # Step 3: Copy highlight videos to output directory
    highlight_videos = _copy_highlights(
        sportclip_dir=sportclip_dir,
        video_name=video_name,
        output_dir=output_dir,
    )

    # Step 4: Extract frame-level timing information
    segments_metadata = _extract_segment_timing(
        sportclip_dir=sportclip_dir,
        video_name=video_name,
        highlight_videos=highlight_videos,
        video_path=video_path
    )

    print(f"\n✓ Successfully generated {len(highlight_videos)} video highlights with timing information")
    return highlight_videos, segments_metadata


def _validate_paths(sportclip_dir: str, sentences_file: str) -> None:
    """Validate that required directories and files exist."""
    sportclip_path = Path(sportclip_dir)
    if not sportclip_path.exists():
        raise FileNotFoundError(f"SportCLIP directory not found: {sportclip_dir}")

    sentences_file_path = sportclip_path / sentences_file
    if not sentences_file_path.exists():
        raise FileNotFoundError(f"Sentences file not found: {sentences_file_path}")


def _get_python_executable(sportclip_dir: str) -> str:
    """
    Determine which Python executable to use.
    Prefers SportCLIP's virtual environment if available.
    """
    sportclip_python = Path(sportclip_dir) / "venv" / "bin" / "python"
    if sportclip_python.exists():
        python_exec = str(sportclip_python)
        print(f"Using SportCLIP virtual environment: {python_exec}")
        return python_exec
    else:
        print("Warning: SportCLIP venv not found, using system Python")
        return "python"


def _run_multi_sentences(
    python_exec: str,
    sportclip_dir: str,
    dataset_dir: str,
    video_name: str,
    sentences_file: str,
) -> None:
    """Run SportCLIP's multi_sentences.py script."""
    cmd = [
        python_exec,
        "multi_sentences.py",
        "--root_dir",
        dataset_dir,
        "--video_name",
        video_name,
        "--sentences_file",
        sentences_file,
    ]

    try:
        subprocess.run(
            cmd,
            cwd=sportclip_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"✗ multi_sentences.py failed with error:\n{e.stderr}")
        raise


def _run_summarize(
    python_exec: str,
    sportclip_dir: str,
    dataset_dir: str,
    video_name: str,
) -> None:
    """Run SportCLIP's summarize.py script with individual export mode."""
    cmd = [
        python_exec,
        "summarize.py",
        "--dataset_dir",
        dataset_dir,
        "--video_name",
        video_name,
        "--export_highlight_reel",
        "--export_mode",
        "individual",
        "--frame_root",
        f"{dataset_dir}/imgs",
    ]

    try:
        subprocess.run(
            cmd,
            cwd=sportclip_dir,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"✗ summarize.py failed with error:\n{e.stderr}")
        raise


def _copy_highlights(
    sportclip_dir: str,
    video_name: str,
    output_dir: str,
) -> List[str]:
    """
    Copy generated highlight videos from SportCLIP results to output directory.

    Args:
        sportclip_dir: SportCLIP project directory
        video_name: Name of the processed video
        output_dir: Destination directory for highlights

    Returns:
        List of paths to copied highlight videos
    """
    # Find generated highlight videos
    sportclip_results_dir = Path(sportclip_dir) / "results" / video_name
    highlight_videos = sorted(sportclip_results_dir.glob("highlight_*.mp4"))

    if not highlight_videos:
        print(f"⚠ Warning: No highlight videos found in {sportclip_results_dir}")
        return []

    # Ensure output directory exists
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Copy each highlight video
    print(f"\n[3/3] Copying {len(highlight_videos)} highlight videos to {output_dir}...")
    copied_videos = []

    for highlight_video in highlight_videos:
        dest_filename = f"{video_name}_{highlight_video.name}"
        dest_path = Path(output_dir) / dest_filename
        shutil.copy2(highlight_video, dest_path)
        copied_videos.append(str(dest_path))
        print(f"  ✓ Copied: {highlight_video.name} → {dest_filename}")

    return copied_videos


def _extract_segment_timing(
    sportclip_dir: str,
    video_name: str,
    highlight_videos: List[str],
    video_path: Optional[str] = None
) -> List[Dict]:
    """
    Extract frame-level timing information for each highlight segment.

    Reads the events_filtered.json file that SportCLIP should have saved,
    or attempts to load from other available sources.

    Args:
        sportclip_dir: SportCLIP project directory
        video_name: Name of the processed video
        highlight_videos: List of generated highlight video paths
        video_path: Optional path to original video for FPS detection

    Returns:
        List of segment metadata dictionaries with start/end frames and times
    """
    sportclip_results_dir = Path(sportclip_dir) / "results" / video_name

    # Get video FPS
    fps = 25.0  # default
    if video_path and Path(video_path).exists():
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps == 0:
                fps = 25.0
        except:
            pass

    # Try to load events_filtered.json (created by our modified SportCLIP)
    events_file = sportclip_results_dir / "events_filtered.json"

    segments_metadata = []

    if events_file.exists():
        print(f"  ✓ Found events file: {events_file}")
        try:
            import json
            with open(events_file, 'r') as f:
                events_filtered = json.load(f)

            # Create a mapping from event numbers to video file paths
            # The event number corresponds to the highlight file number (e.g., event 7 -> highlight_007.mp4)
            video_map = {}
            for video_path in highlight_videos:
                video_name_only = Path(video_path).stem  # e.g., "highjump_video1_highlight_007"
                # Extract the highlight number from the filename
                parts = video_name_only.split('_')
                if len(parts) >= 3 and parts[-2] == 'highlight':
                    try:
                        highlight_num = int(parts[-1])
                        video_map[highlight_num] = video_path
                    except ValueError:
                        pass

            # events_filtered is a dict like {0: {"frames": [...], "length": X, "area": Y}, 1: {...}, ...}
            for event_num_str, event in sorted(events_filtered.items(), key=lambda x: int(x[0])):
                event_num = int(event_num_str)
                frames = event.get("frames", [])

                if not frames:
                    continue

                start_frame = frames[0]
                end_frame = frames[-1]

                # Convert frames to seconds
                start_time = start_frame / fps
                end_time = end_frame / fps

                # Find corresponding video by matching event number to highlight number
                # SportCLIP creates highlight files like highlight_007.mp4 for event 7
                # We need to find the video whose highlight number matches this event number
                video_file = None
                for vpath in highlight_videos:
                    # Check if this video corresponds to this event
                    # Format: {video_name}_highlight_{NNN}.mp4
                    if f"_highlight_{event_num:03d}.mp4" in vpath:
                        video_file = vpath
                        break

                clip_name = Path(video_file).name if video_file else f"{video_name}_highlight_{event_num:03d}.mp4"

                segments_metadata.append({
                    "segment_id": event_num + 1,  # 1-indexed for user-facing
                    "clip_path": clip_name,
                    "clip_name": clip_name,
                    "start_frame": int(start_frame),
                    "end_frame": int(end_frame),
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "duration_frames": int(end_frame - start_frame + 1),
                    "duration_seconds": float((end_frame - start_frame + 1) / fps)
                })

            print(f"  ✓ Extracted timing for {len(segments_metadata)} segments")
            return segments_metadata

        except Exception as e:
            print(f"  Warning: Error reading events file: {e}")

    # Fallback: create segments without frame info
    print(f"  Warning: events_filtered.json not found")
    print(f"  Generating segments without timing information")

    for i, video_file in enumerate(highlight_videos):
        segments_metadata.append({
            "segment_id": i + 1,
            "clip_path": str(Path(video_file).name),
            "clip_name": str(Path(video_file).name),
            "start_frame": None,
            "end_frame": None,
            "start_time": None,
            "end_time": None,
            "duration_frames": None,
            "duration_seconds": None
        })

    return segments_metadata
