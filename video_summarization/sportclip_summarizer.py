"""
SportCLIP Video Summarizer

This module integrates with the SportCLIP framework to generate video highlights.
It handles running SportCLIP's multi_sentences.py and summarize.py scripts,
and manages the output highlight videos.
"""

import os
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional


def generate_highlights(
    video_name: str,
    sentences_file: str,
    output_dir: str,
    sportclip_dir: str = "/home/mrt/Projects/SportCLIP-official",
    sportclip_dataset_dir: str = "/mnt/Data/mrt/SportCLIP-OlympicHighlights",
) -> List[str]:
    """
    Generate video highlights using SportCLIP framework.

    This function:
    1. Runs multi_sentences.py to analyze highlight/non-highlight sentence pairs
    2. Runs summarize.py to extract individual highlight videos
    3. Copies generated highlights to the specified output directory

    Args:
        video_name: Name of the video to process (without extension)
        sentences_file: Path to SportCLIP sentences JSON file (relative to sportclip_dir)
        output_dir: Directory to save generated highlight videos
        sportclip_dir: Path to SportCLIP project directory
        sportclip_dataset_dir: Path to SportCLIP dataset directory (with frames/embeddings)

    Returns:
        List of paths to generated highlight videos

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

    print(f"\n✓ Successfully generated {len(highlight_videos)} video highlights")
    return highlight_videos


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
