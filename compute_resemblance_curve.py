"""
Compute Resemblance Curve
Compares updated target face against all face crops in a video to create similarity-over-time plots

This script:
1. Loads updated target embeddings (ArcFace and TransFace)
2. Compares against all face embeddings in a video
3. Creates eye-catching plots showing resemblance score over time
"""

import os
import sys
import json
import numpy as np
import h5py
import cv2
import subprocess
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta

# Use Agg backend for non-interactive plotting
import matplotlib

matplotlib.use("Agg")


def get_video_fps(video_path: str) -> float:
    """
    Get FPS of a video using ffprobe

    Args:
        video_path: Path to video file

    Returns:
        FPS as float
    """
    try:
        cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=r_frame_rate", "-of", "csv=p=0", video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        fps_str = result.stdout.strip()

        # Parse fraction (e.g., "25/1" or "30000/1001")
        if "/" in fps_str:
            num, den = fps_str.split("/")
            fps = float(num) / float(den)
        else:
            fps = float(fps_str)

        return fps
    except Exception as e:
        print(f"Warning: Could not get FPS using ffprobe: {e}")
        print("Defaulting to 25 FPS")
        return 25.0


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings (normalized to [0, 1])

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Similarity score in [0, 1] range
    """
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    # Normalize to [0, 1]
    return (similarity + 1) / 2


def load_updated_target_embedding(target_dir: Path, model: str = "arcface", use_updated: bool = False) -> Optional[np.ndarray]:
    """
    Load updated target embedding

    Args:
        target_dir: Directory containing updated target files
        model: Model type ("arcface" or "transface")
        use_updated: If True, load embeddings from updated target face (second pass)

    Returns:
        Embedding as numpy array or None if not found
    """
    if model.lower() == "arcface":
        if use_updated:
            emb_path = target_dir / "updated_target_arcface_embedding.npy"
        else:
            emb_path = target_dir / "target_arcface_embedding.npy"
    elif model.lower() == "transface":
        if use_updated:
            emb_path = target_dir / "updated_target_transface_embedding.npy"
        else:
            emb_path = target_dir / "target_transface_embedding.npy"
    else:
        print(f"Unknown model: {model}")
        return None

    if not emb_path.exists():
        print(f"Error: Embedding not found: {emb_path}")
        return None

    return np.load(str(emb_path))


def compute_resemblance_scores(target_embedding: np.ndarray, embeddings_h5_path: Path, video_fps: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute resemblance scores for all frames in a video

    Args:
        target_embedding: Target face embedding
        embeddings_h5_path: Path to HDF5 file with video embeddings
        video_fps: Video FPS for time calculation

    Returns:
        Tuple of (timestamps in seconds, max similarity scores per frame)
    """
    if not embeddings_h5_path.exists():
        print(f"Error: Embeddings file not found: {embeddings_h5_path}")
        return np.array([]), np.array([])

    # Dictionary to store max similarity per frame
    frame_scores = {}

    with h5py.File(str(embeddings_h5_path), "r") as f:
        print(f"  Processing {len(f.keys())} face embeddings...")

        for face_name in tqdm(f.keys(), desc="  Computing similarities", unit="face"):
            embedding = f[face_name][:]
            similarity = compute_cosine_similarity(target_embedding, embedding)

            # Extract frame number from face_name (e.g., "frame_000241_face_0" -> 241)
            try:
                parts = face_name.split("_")
                frame_num = int(parts[1])

                # Keep max similarity per frame (in case multiple faces in same frame)
                if frame_num not in frame_scores or similarity > frame_scores[frame_num]:
                    frame_scores[frame_num] = similarity
            except (IndexError, ValueError) as e:
                print(f"  Warning: Could not parse frame number from {face_name}: {e}")
                continue

    if not frame_scores:
        print("  Warning: No valid frame scores computed")
        return np.array([]), np.array([])

    # Convert to sorted arrays
    frames = sorted(frame_scores.keys())
    scores = [frame_scores[f] for f in frames]

    # Convert frame numbers to timestamps (in seconds)
    timestamps = np.array(frames) / video_fps
    scores = np.array(scores)

    return timestamps, scores


def smooth_scores(scores: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Apply rolling average smoothing to scores

    Args:
        scores: Array of scores
        window_size: Window size for rolling average

    Returns:
        Smoothed scores
    """
    if len(scores) < window_size:
        return scores

    # Use convolution for rolling average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores, kernel, mode="same")

    # Fix edges (convolve has edge effects)
    for i in range(window_size // 2):
        smoothed[i] = np.mean(scores[: i + window_size // 2 + 1])
        smoothed[-(i + 1)] = np.mean(scores[-(i + window_size // 2 + 1) :])

    return smoothed


def create_resemblance_plot(
    timestamps: np.ndarray,
    arcface_scores: np.ndarray,
    transface_scores: np.ndarray,
    output_path: Path,
    target_name: str,
    video_name: str,
    threshold: float = 0.6,
    smooth_window: int = 5,
    use_updated: bool = False,
):
    """
    Create an eye-catching resemblance curve plot

    Args:
        timestamps: Timestamps in seconds
        arcface_scores: ArcFace similarity scores
        transface_scores: TransFace similarity scores
        output_path: Path to save plot
        target_name: Name of target person
        video_name: Name of video
        threshold: Similarity threshold to highlight
        smooth_window: Window size for rolling average smoothing
    """
    # Create figure with high DPI for quality
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("white")

    # Convert timestamps to timedelta for formatting
    time_deltas = [timedelta(seconds=t) for t in timestamps]

    # Helper function to format time as MM:SS
    def format_time(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    # Apply smoothing
    arcface_scores_smooth = smooth_scores(arcface_scores, smooth_window)
    transface_scores_smooth = smooth_scores(transface_scores, smooth_window)

    # Plot 1: ArcFace
    # Plot original scores (faded)
    ax1.plot(timestamps, arcface_scores, color="#2E86AB", linewidth=1.5, alpha=0.3, label="Raw Scores")
    ax1.fill_between(timestamps, arcface_scores, alpha=0.15, color="#A23B72")

    # Plot smoothed scores (prominent)
    ax1.plot(timestamps, arcface_scores_smooth, color="#2E86AB", linewidth=2.5, alpha=0.9, label=f"Smoothed (window={smooth_window})")

    ax1.axhline(y=threshold, color="#F18F01", linestyle="--", linewidth=2, label=f"Threshold ({threshold})", alpha=0.7)

    # Highlight regions above threshold (using smoothed scores)
    above_threshold = arcface_scores_smooth >= threshold
    if np.any(above_threshold):
        ax1.fill_between(timestamps, 0, 1, where=above_threshold, alpha=0.2, color="#06A77D", label="Above Threshold", transform=ax1.get_xaxis_transform())

    ax1.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    title_suffix = " (Updated Target)" if use_updated else ""
    ax1.set_title(f"ArcFace Resemblance Curve - {target_name} in {video_name}{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    ax1.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax1.set_ylim([0, 1])
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.set_xlim([timestamps[0], timestamps[-1]])

    # Format x-axis as MM:SS with more granularity
    video_duration = timestamps[-1]
    # Determine number of ticks based on video duration
    if video_duration <= 60:  # <= 1 minute
        tick_interval = 5  # Every 5 seconds
    elif video_duration <= 300:  # <= 5 minutes
        tick_interval = 15  # Every 15 seconds
    elif video_duration <= 600:  # <= 10 minutes
        tick_interval = 30  # Every 30 seconds
    else:
        tick_interval = 60  # Every 1 minute

    tick_times = np.arange(0, video_duration + tick_interval, tick_interval)
    tick_times = tick_times[tick_times <= video_duration]
    ax1.set_xticks(tick_times)
    ax1.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Plot 2: TransFace
    # Plot original scores (faded)
    ax2.plot(timestamps, transface_scores, color="#C73E1D", linewidth=1.5, alpha=0.3, label="Raw Scores")
    ax2.fill_between(timestamps, transface_scores, alpha=0.15, color="#6A4C93")

    # Plot smoothed scores (prominent)
    ax2.plot(timestamps, transface_scores_smooth, color="#C73E1D", linewidth=2.5, alpha=0.9, label=f"Smoothed (window={smooth_window})")

    ax2.axhline(y=threshold, color="#F18F01", linestyle="--", linewidth=2, label=f"Threshold ({threshold})", alpha=0.7)

    # Highlight regions above threshold (using smoothed scores)
    above_threshold = transface_scores_smooth >= threshold
    if np.any(above_threshold):
        ax2.fill_between(timestamps, 0, 1, where=above_threshold, alpha=0.2, color="#06A77D", label="Above Threshold", transform=ax2.get_xaxis_transform())

    ax2.set_xlabel("Video Time (MM:SS)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    title_suffix = " (Updated Target)" if use_updated else ""
    ax2.set_title(f"TransFace Resemblance Curve - {target_name} in {video_name}{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    ax2.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax2.set_ylim([0, 1])
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax2.set_xlim([timestamps[0], timestamps[-1]])

    # Format x-axis as MM:SS with more granularity (same as ax1)
    ax2.set_xticks(tick_times)
    ax2.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Add statistics text box
    arcface_mean = np.mean(arcface_scores)
    arcface_max = np.max(arcface_scores)
    arcface_above = np.sum(arcface_scores >= threshold)

    transface_mean = np.mean(transface_scores)
    transface_max = np.max(transface_scores)
    transface_above = np.sum(transface_scores >= threshold)

    stats_text = f"ArcFace - Mean: {arcface_mean:.3f} | Max: {arcface_max:.3f} | Above threshold: {arcface_above}\n"
    stats_text += f"TransFace - Mean: {transface_mean:.3f} | Max: {transface_max:.3f} | Above threshold: {transface_above}"

    fig.text(0.5, 0.02, stats_text, ha="center", fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"\n  Plot saved to: {output_path}")


def save_resemblance_data(timestamps: np.ndarray, arcface_scores: np.ndarray, transface_scores: np.ndarray, output_path: Path, target_name: str, video_name: str, video_fps: float):
    """
    Save resemblance scores to JSON file

    Args:
        timestamps: Timestamps in seconds
        arcface_scores: ArcFace similarity scores
        transface_scores: TransFace similarity scores
        output_path: Path to save JSON
        target_name: Name of target person
        video_name: Name of video
        video_fps: Video FPS
    """
    data = {
        "target": target_name,
        "video": video_name,
        "fps": video_fps,
        "total_frames_with_faces": len(timestamps),
        "timestamps_seconds": timestamps.tolist(),
        "arcface_scores": arcface_scores.tolist(),
        "transface_scores": transface_scores.tolist(),
        "statistics": {
            "arcface": {"mean": float(np.mean(arcface_scores)), "std": float(np.std(arcface_scores)), "max": float(np.max(arcface_scores)), "min": float(np.min(arcface_scores))},
            "transface": {
                "mean": float(np.mean(transface_scores)),
                "std": float(np.std(transface_scores)),
                "max": float(np.max(transface_scores)),
                "min": float(np.min(transface_scores)),
            },
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"  Data saved to: {output_path}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Compute resemblance curve for updated target face in a video",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all targets in target_faces_updated/ for highjump_video1 (default video)
  python compute_resemblance_curve.py

  # Process all targets for a different video
  python compute_resemblance_curve.py --video longjump_video1

  # Process specific target only
  python compute_resemblance_curve.py --target athlete1 --video highjump_video1

  # Use custom threshold and smoothing window
  python compute_resemblance_curve.py --threshold 0.6 --smooth-window 7

  # Custom directories
  python compute_resemblance_curve.py --video highjump_video1 \\
                                      --updated-targets custom/updated \\
                                      --embeddings custom/embeddings
        """,
    )

    parser.add_argument("--target", type=str, default=None, help="Target person name (must match directory name in target_faces_updated/). If not provided, processes all targets.")
    parser.add_argument("--video", type=str, default="highjump_video1", help="Video name to process (without extension, default: highjump_video1)")
    parser.add_argument("--video-path", type=str, default=None, help="Path to video file (for FPS detection). If not provided, assumes data/<video>.mp4")
    parser.add_argument("--updated-targets", type=str, default="target_faces_updated", help="Directory containing updated target faces (default: target_faces_updated)")
    parser.add_argument("--embeddings", type=str, default="cropped_faces_embeddings", help="Directory containing face embeddings (default: cropped_faces_embeddings)")
    parser.add_argument("--output", type=str, default="resemblance_results", help="Output directory for plots and data (default: resemblance_results)")
    parser.add_argument("--threshold", type=float, default=0.6, help="Similarity threshold for visualization (default: 0.6)")
    parser.add_argument("--smooth-window", type=int, default=5, help="Window size for rolling average smoothing (default: 5)")
    parser.add_argument(
        "--use-updated-target",
        action="store_true",
        help="Use embeddings from updated target faces (second pass, should achieve 100%% similarity). Files saved with '_updated' suffix.",
    )

    args = parser.parse_args()

    # Setup paths
    video_name = args.video
    updated_targets_dir = Path(args.updated_targets)
    embeddings_dir = Path(args.embeddings)
    output_dir = Path(args.output)

    # Check if updated targets directory exists
    if not updated_targets_dir.exists():
        print(f"Error: Updated targets directory not found: {updated_targets_dir}")
        print("Please run find_updated_target.py first.")
        return

    # Find all targets to process
    if args.target:
        # Process only specified target
        target_dirs = [updated_targets_dir / args.target]
        if not target_dirs[0].exists():
            print(f"Error: Target directory not found: {target_dirs[0]}")
            print("Please run find_updated_target.py first.")
            return
    else:
        # Process all targets in directory
        target_dirs = [d for d in updated_targets_dir.iterdir() if d.is_dir()]
        if not target_dirs:
            print(f"Error: No target directories found in {updated_targets_dir}")
            print("Please run find_updated_target.py first.")
            return

    print("=" * 80)
    print("COMPUTE RESEMBLANCE CURVE")
    print("=" * 80)
    print(f"\nVideo: {video_name}")
    print(f"Mode: {'Updated Target (Second Pass)' if args.use_updated_target else 'Original Target (First Pass)'}")
    print(f"Threshold: {args.threshold}")
    print(f"Smooth window: {args.smooth_window}")
    print(f"Targets to process: {len(target_dirs)}")
    for target_dir in target_dirs:
        print(f"  - {target_dir.name}")

    # Get video path for FPS detection
    if args.video_path:
        video_path = args.video_path
    else:
        video_path = f"data/{video_name}.mp4"

    if not os.path.exists(video_path):
        print(f"\nWarning: Video file not found: {video_path}")
        print("Will use default FPS of 25")
        video_fps = 25.0
    else:
        print(f"\nDetecting video FPS from: {video_path}")
        video_fps = get_video_fps(video_path)
        print(f"Video FPS: {video_fps:.2f}")

    # Process each target
    successful_targets = 0
    failed_targets = 0

    for i, target_dir in enumerate(target_dirs):
        target_name = target_dir.name

        print("\n" + "=" * 80)
        print(f"PROCESSING TARGET {i+1}/{len(target_dirs)}: {target_name}")
        print("=" * 80)

        # Load updated target embeddings
        print("\nLoading target embeddings...")
        arcface_emb = load_updated_target_embedding(target_dir, "arcface", use_updated=args.use_updated_target)
        transface_emb = load_updated_target_embedding(target_dir, "transface", use_updated=args.use_updated_target)

        if arcface_emb is None or transface_emb is None:
            print(f"\nError: Could not load target embeddings for {target_name}")
            print("Skipping this target...")
            failed_targets += 1
            continue

        print(f"  ArcFace embedding shape: {arcface_emb.shape}")
        print(f"  TransFace embedding shape: {transface_emb.shape}")

        # Compute resemblance scores
        print(f"\nComputing resemblance scores for {video_name}...")

        # ArcFace
        print("\n  ArcFace:")
        arcface_h5 = embeddings_dir / "ArcFace" / f"{video_name}.h5"
        timestamps, arcface_scores = compute_resemblance_scores(arcface_emb, arcface_h5, video_fps)

        if len(timestamps) == 0:
            print(f"Error: No ArcFace scores computed for {target_name}")
            print("Skipping this target...")
            failed_targets += 1
            continue

        # TransFace
        print("\n  TransFace:")
        transface_h5 = embeddings_dir / "TransFace" / f"{video_name}.h5"
        _, transface_scores = compute_resemblance_scores(transface_emb, transface_h5, video_fps)

        if len(transface_scores) == 0:
            print(f"Error: No TransFace scores computed for {target_name}")
            print("Skipping this target...")
            failed_targets += 1
            continue

        # Create output directory
        target_output_dir = output_dir / target_name
        target_output_dir.mkdir(parents=True, exist_ok=True)

        # Create plot with suffix if using updated target
        print("\nCreating resemblance curve plot...")
        suffix = "_updated" if args.use_updated_target else ""
        plot_path = target_output_dir / f"{target_name}{suffix}.png"
        create_resemblance_plot(
            timestamps=timestamps,
            arcface_scores=arcface_scores,
            transface_scores=transface_scores,
            output_path=plot_path,
            target_name=target_name,
            video_name=video_name,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            use_updated=args.use_updated_target,
        )

        # Save data with suffix if using updated target
        print("\nSaving resemblance data...")
        data_path = target_output_dir / f"{target_name}{suffix}.json"
        save_resemblance_data(
            timestamps=timestamps,
            arcface_scores=arcface_scores,
            transface_scores=transface_scores,
            output_path=data_path,
            target_name=target_name,
            video_name=video_name,
            video_fps=video_fps,
        )

        print(f"\n  ✓ Successfully processed {target_name}")
        print(f"    - Plot: {plot_path.relative_to(output_dir)}")
        print(f"    - Data: {data_path.relative_to(output_dir)}")
        print(f"    - Frames with faces: {len(timestamps)}")
        print(f"    - ArcFace max: {np.max(arcface_scores):.3f}")
        print(f"    - TransFace max: {np.max(transface_scores):.3f}")

        successful_targets += 1

    # Final summary
    print("\n" + "=" * 80)
    print("ALL PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"  Video: {video_name}")
    print(f"  Total targets: {len(target_dirs)}")
    print(f"  Successful: {successful_targets}")
    print(f"  Failed: {failed_targets}")
    print(f"\nResults saved to: {output_dir}/")


if __name__ == "__main__":
    main()
