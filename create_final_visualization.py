"""
Create Final Visualization with Highlight Overlays

This script:
1. Loads resemblance curve data for each target
2. Loads video summary segment data
3. Creates enhanced plots with highlight segments overlaid on timeline
4. Copies highlight clips where target appears to results directory
5. Generates a summary report
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse

# Use Agg backend for non-interactive plotting
import matplotlib

matplotlib.use("Agg")


def load_resemblance_data(resemblance_file: Path) -> Optional[Dict]:
    """
    Load resemblance curve data from JSON

    Args:
        resemblance_file: Path to resemblance JSON file

    Returns:
        Dictionary with resemblance data or None if failed
    """
    if not resemblance_file.exists():
        return None

    try:
        with open(resemblance_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading resemblance data: {e}")
        return None


def load_segments_data(segments_file: Path) -> Optional[Dict]:
    """
    Load video segments metadata from JSON

    Args:
        segments_file: Path to segments JSON file

    Returns:
        Dictionary with segments data or None if failed
    """
    if not segments_file.exists():
        return None

    try:
        with open(segments_file, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading segments data: {e}")
        return None


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

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores, kernel, mode="same")

    # Fix edges
    for i in range(window_size // 2):
        smoothed[i] = np.mean(scores[: i + window_size // 2 + 1])
        smoothed[-(i + 1)] = np.mean(scores[-(i + window_size // 2 + 1) :])

    return smoothed


def extract_clip_timestamps(clip_name: str) -> Optional[Tuple[float, float]]:
    """
    Extract start and end timestamps from clip filename

    Assumes format like: "clip_001_0025.00_0035.00.mp4" or similar

    Args:
        clip_name: Name of clip file

    Returns:
        Tuple of (start_time, end_time) in seconds or None if cannot parse
    """
    try:
        # Try to extract timestamps from filename
        # Common patterns: clip_XXX_START_END.mp4
        parts = clip_name.replace(".mp4", "").split("_")

        # Look for numeric parts that could be timestamps
        timestamps = []
        for part in parts:
            try:
                val = float(part)
                timestamps.append(val)
            except:
                continue

        if len(timestamps) >= 2:
            # Assume last two numeric values are start and end
            return (timestamps[-2], timestamps[-1])

        return None
    except Exception as e:
        print(f"  Warning: Could not parse timestamps from {clip_name}")
        return None


def create_enhanced_resemblance_plot(
    resemblance_data: Dict, segments_data: Dict, output_path: Path, threshold: float = 0.6, smooth_window: int = 5, relevant_segments_count: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create enhanced resemblance curve plot with highlight segments overlaid

    Args:
        resemblance_data: Resemblance curve data
        segments_data: Video segments metadata
        output_path: Path to save plot
        threshold: Similarity threshold
        smooth_window: Smoothing window size
        relevant_segments_count: Number of segments where target appears (optional)

    Returns:
        Tuple of (arcface_scores_smooth, transface_scores_smooth) for use in clip filtering
    """
    target_name = resemblance_data["target"]
    video_name = resemblance_data["video"]

    timestamps = np.array(resemblance_data["timestamps_seconds"])
    arcface_scores = np.array(resemblance_data["arcface_scores"])
    transface_scores = np.array(resemblance_data["transface_scores"])

    # Apply smoothing
    arcface_scores_smooth = smooth_scores(arcface_scores, smooth_window)
    transface_scores_smooth = smooth_scores(transface_scores, smooth_window)

    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("white")

    def format_time(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    video_duration = timestamps[-1]

    # Determine tick interval
    if video_duration <= 60:
        tick_interval = 5
    elif video_duration <= 300:
        tick_interval = 15
    elif video_duration <= 600:
        tick_interval = 30
    else:
        tick_interval = 60

    tick_times = np.arange(0, video_duration + tick_interval, tick_interval)
    tick_times = tick_times[tick_times <= video_duration]

    # Helper function to add highlight segments to plot
    def add_highlight_segments(ax, y_pos=0.95):
        """Add highlight segment markers to the plot"""
        # Handle nested structure
        segments_obj = segments_data.get("segments", [])
        if isinstance(segments_obj, dict):
            # Nested: segments_data['segments']['segments']
            segments = segments_obj.get("segments", [])
        else:
            # Flat: segments_data['segments']
            segments = segments_obj

        for i, segment in enumerate(segments, 1):
            # Use timing information from segment metadata
            start_time = segment.get("start_time")
            end_time = segment.get("end_time")

            if start_time is not None and end_time is not None:
                # Draw rectangle spanning the highlight duration
                width = end_time - start_time
                rect = patches.Rectangle(
                    (start_time, 0),
                    width,
                    1,
                    linewidth=2,
                    edgecolor="red",
                    facecolor="red",
                    alpha=0.15,
                    transform=ax.get_xaxis_transform(),
                    label="Highlight Segment" if i == 1 else "",
                )
                ax.add_patch(rect)

                # Add segment number label
                ax.text(
                    start_time + width / 2,
                    y_pos,
                    f"H{i}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7, edgecolor="darkred"),
                    color="white",
                )

    # Plot 1: ArcFace
    ax1.plot(timestamps, arcface_scores, color="#2E86AB", linewidth=1.5, alpha=0.3, label="Raw Scores")
    ax1.fill_between(timestamps, arcface_scores, alpha=0.15, color="#A23B72")
    ax1.plot(timestamps, arcface_scores_smooth, color="#2E86AB", linewidth=2.5, alpha=0.9, label=f"Smoothed (window={smooth_window})")
    ax1.axhline(y=threshold, color="#F18F01", linestyle="--", linewidth=2, label=f"Threshold ({threshold})", alpha=0.7)

    # Highlight regions above threshold
    above_threshold = arcface_scores_smooth >= threshold
    if np.any(above_threshold):
        ax1.fill_between(timestamps, 0, 1, where=above_threshold, alpha=0.2, color="#06A77D", label="Above Threshold", transform=ax1.get_xaxis_transform())

    # Add highlight segments
    add_highlight_segments(ax1)

    ax1.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    ax1.set_title(f"ArcFace Resemblance Curve with Highlights - {target_name} in {video_name}", fontsize=16, fontweight="bold", pad=20)
    ax1.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax1.set_ylim([0, 1])
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax1.set_xlim([timestamps[0], timestamps[-1]])
    ax1.set_xticks(tick_times)
    ax1.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Plot 2: TransFace
    ax2.plot(timestamps, transface_scores, color="#C73E1D", linewidth=1.5, alpha=0.3, label="Raw Scores")
    ax2.fill_between(timestamps, transface_scores, alpha=0.15, color="#6A4C93")
    ax2.plot(timestamps, transface_scores_smooth, color="#C73E1D", linewidth=2.5, alpha=0.9, label=f"Smoothed (window={smooth_window})")
    ax2.axhline(y=threshold, color="#F18F01", linestyle="--", linewidth=2, label=f"Threshold ({threshold})", alpha=0.7)

    # Highlight regions above threshold
    above_threshold = transface_scores_smooth >= threshold
    if np.any(above_threshold):
        ax2.fill_between(timestamps, 0, 1, where=above_threshold, alpha=0.2, color="#06A77D", label="Above Threshold", transform=ax2.get_xaxis_transform())

    # Add highlight segments
    add_highlight_segments(ax2)

    ax2.set_xlabel("Video Time (MM:SS)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    ax2.set_title(f"TransFace Resemblance Curve with Highlights - {target_name} in {video_name}", fontsize=16, fontweight="bold", pad=20)
    ax2.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax2.set_ylim([0, 1])
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9)
    ax2.set_xlim([timestamps[0], timestamps[-1]])
    ax2.set_xticks(tick_times)
    ax2.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Add statistics
    arcface_mean = np.mean(arcface_scores)
    arcface_max = np.max(arcface_scores)
    arcface_above = np.sum(arcface_scores >= threshold)
    transface_mean = np.mean(transface_scores)
    transface_max = np.max(transface_scores)
    transface_above = np.sum(transface_scores >= threshold)

    # Get segment count (handle nested structure)
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        num_segments = len(segments_obj.get("segments", []))
    else:
        num_segments = len(segments_obj)

    stats_text = f"ArcFace - Mean: {arcface_mean:.3f} | Max: {arcface_max:.3f} | Above threshold: {arcface_above}\n"
    stats_text += f"TransFace - Mean: {transface_mean:.3f} | Max: {transface_max:.3f} | Above threshold: {transface_above}\n"

    # Show relevant segments count if available
    if relevant_segments_count is not None:
        stats_text += f"Highlight Segments: {relevant_segments_count}/{num_segments} (with target)"
    else:
        stats_text += f"Highlight Segments: {num_segments}"

    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout(rect=[0, 0.06, 1, 1])

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  ✓ Created enhanced plot: {output_path.name}")

    # Return smoothed scores for use in clip filtering
    return arcface_scores_smooth, transface_scores_smooth


def check_segment_overlap(segment_times: Tuple[float, float], timestamps: np.ndarray, scores: np.ndarray, threshold: float) -> bool:
    """
    Check if a highlight segment overlaps with high resemblance scores

    Args:
        segment_times: Tuple of (start, end) times in seconds
        timestamps: Array of timestamps
        scores: Array of resemblance scores
        threshold: Score threshold

    Returns:
        True if segment has significant overlap with high scores
    """
    if segment_times is None:
        return False

    start, end = segment_times

    # Find scores within this segment
    mask = (timestamps >= start) & (timestamps <= end)
    segment_scores = scores[mask]

    if len(segment_scores) == 0:
        return False

    # Check if any scores in segment are above threshold
    return np.any(segment_scores >= threshold)


def copy_relevant_clips(
    video_name: str,
    segments_data: Dict,
    resemblance_data: Dict,
    video_summaries_dir: Path,
    output_dir: Path,
    threshold: float = 0.6,
    arcface_scores_smooth: np.ndarray = None,
    transface_scores_smooth: np.ndarray = None,
) -> List[str]:
    """
    Copy highlight clips where target person appears to results directory

    Args:
        video_name: Video name (without extension)
        segments_data: Video segments metadata
        resemblance_data: Resemblance curve data
        video_summaries_dir: Directory containing video summaries
        output_dir: Output directory for relevant clips
        threshold: Resemblance score threshold
        arcface_scores_smooth: Pre-smoothed ArcFace scores (if None, uses raw scores)
        transface_scores_smooth: Pre-smoothed TransFace scores (if None, uses raw scores)

    Returns:
        List of copied clip names
    """
    model = segments_data.get("model", "sportclip")
    target_name = resemblance_data["target"]

    timestamps = np.array(resemblance_data["timestamps_seconds"])

    # Use pre-smoothed scores if provided, otherwise use raw scores
    if arcface_scores_smooth is None or transface_scores_smooth is None:
        arcface_scores = np.array(resemblance_data["arcface_scores"])
        transface_scores = np.array(resemblance_data["transface_scores"])
        avg_scores = (arcface_scores + transface_scores) / 2
    else:
        avg_scores = (arcface_scores_smooth + transface_scores_smooth) / 2

    # Find source clips directory
    model_dir = "SportCLIP" if model == "sportclip" else "QD-DETR"
    clips_dir = video_summaries_dir / model_dir / video_name

    if not clips_dir.exists():
        print(f"  Warning: Clips directory not found: {clips_dir}")
        return []

    # Create output directory
    target_clips_dir = output_dir / "highlight_clips"
    target_clips_dir.mkdir(parents=True, exist_ok=True)

    copied_clips = []

    # Handle nested structure
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    for segment in segments:
        clip_name = segment["clip_name"]
        clip_path = clips_dir / clip_name

        if not clip_path.exists():
            print(f"  Warning: Clip not found: {clip_path}")
            continue

        # Use timing information from segment metadata
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")

        if start_time is not None and end_time is not None:
            times = (start_time, end_time)
            has_target = check_segment_overlap(times, timestamps, avg_scores, threshold)

            if has_target:
                # Copy clip to results
                dest_path = target_clips_dir / clip_name
                shutil.copy2(clip_path, dest_path)
                copied_clips.append(clip_name)
                print(f"  ✓ Copied clip: {clip_name}")

    return copied_clips


def create_summary_report(target_name: str, video_name: str, resemblance_data: Dict, segments_data: Dict, copied_clips: List[str], output_path: Path, threshold: float):
    """
    Create a summary report JSON file

    Args:
        target_name: Target person name
        video_name: Video name
        resemblance_data: Resemblance curve data
        segments_data: Video segments metadata
        copied_clips: List of copied clip names
        output_path: Path to save report
        threshold: Resemblance threshold used
    """
    arcface_scores = np.array(resemblance_data["arcface_scores"])
    transface_scores = np.array(resemblance_data["transface_scores"])

    # Get segment count (handle nested structure)
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        total_segments = len(segments_obj.get("segments", []))
    else:
        total_segments = len(segments_obj)

    report = {
        "target": target_name,
        "video": video_name,
        "model": segments_data.get("model", "unknown"),
        "threshold": threshold,
        "total_highlight_segments": total_segments,
        "relevant_highlight_segments": len(copied_clips),
        "relevant_clips": copied_clips,
        "resemblance_statistics": {
            "arcface": {"mean": float(np.mean(arcface_scores)), "max": float(np.max(arcface_scores)), "frames_above_threshold": int(np.sum(arcface_scores >= threshold))},
            "transface": {"mean": float(np.mean(transface_scores)), "max": float(np.max(transface_scores)), "frames_above_threshold": int(np.sum(transface_scores >= threshold))},
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Created summary report: {output_path.name}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Create final visualization with highlight overlays",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all targets for a video
  python create_final_visualization.py --video highjump_video1 --model sportclip

  # Process specific target
  python create_final_visualization.py --video highjump_video1 --model sportclip --target barshim

  # Custom threshold
  python create_final_visualization.py --video highjump_video1 --model sportclip --threshold 0.65
        """,
    )

    parser.add_argument("--video", type=str, required=True, help="Video name (without extension)")
    parser.add_argument("--model", type=str, required=True, choices=["sportclip", "qddetr"], help="Video summarization model used")
    parser.add_argument("--target", type=str, default=None, help="Specific target to process. If not provided, processes all targets.")
    parser.add_argument("--video-summaries", type=str, default="video_summaries", help="Directory containing video summaries")
    parser.add_argument("--resemblance-results", type=str, default="resemblance_results", help="Directory containing resemblance results")
    parser.add_argument("--output", type=str, default="results", help="Output directory for final results")
    parser.add_argument("--threshold", type=float, default=0.6, help="Resemblance score threshold (default: 0.6)")
    parser.add_argument("--smooth-window", type=int, default=5, help="Smoothing window size (default: 5)")
    parser.add_argument(
        "--use-updated-target",
        action="store_true",
        help="Use results from updated target embeddings (second pass). Reads files with '_updated' suffix.",
    )

    args = parser.parse_args()

    print("=" * 80)
    print("CREATE FINAL VISUALIZATION WITH HIGHLIGHT OVERLAYS")
    print("=" * 80)
    print(f"\nVideo: {args.video}")
    print(f"Model: {args.model.upper()}")
    print(f"Mode: {'Updated Target (Second Pass)' if args.use_updated_target else 'Original Target (First Pass)'}")
    print(f"Threshold: {args.threshold}")

    # Setup paths
    video_summaries_dir = Path(args.video_summaries)
    resemblance_results_dir = Path(args.resemblance_results)
    output_dir = Path(args.output)

    # Load segments data
    model_dir = "SportCLIP" if args.model == "sportclip" else "QD-DETR"
    segments_file = video_summaries_dir / model_dir / args.video / "segments.json"

    if not segments_file.exists():
        print(f"\nError: Segments file not found: {segments_file}")
        return

    segments_data = load_segments_data(segments_file)
    if not segments_data:
        print(f"\nError: Could not load segments data")
        return

    # Get segment count (handle nested structure)
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        num_segments = len(segments_obj.get("segments", []))
    else:
        num_segments = len(segments_obj)

    print(f"\nLoaded {num_segments} highlight segments")

    # Find targets to process
    if args.target:
        target_dirs = [resemblance_results_dir / args.target]
        if not target_dirs[0].exists():
            print(f"\nError: Target directory not found: {target_dirs[0]}")
            return
    else:
        target_dirs = [d for d in resemblance_results_dir.iterdir() if d.is_dir()]
        if not target_dirs:
            print(f"\nError: No target directories found in {resemblance_results_dir}")
            return

    print(f"\nProcessing {len(target_dirs)} target(s):")
    for td in target_dirs:
        print(f"  - {td.name}")

    # Process each target
    successful = 0

    for target_dir in target_dirs:
        target_name = target_dir.name

        print(f"\n{'='*80}")
        print(f"PROCESSING TARGET: {target_name}")
        print(f"{'='*80}")

        # Find resemblance data file (with suffix if using updated target)
        suffix = "_updated" if args.use_updated_target else ""
        resemblance_file = target_dir / args.video / f"{target_name}_{args.video}_resemblance_data{suffix}.json"

        if not resemblance_file.exists():
            print(f"  Warning: Resemblance data not found: {resemblance_file}")
            continue

        resemblance_data = load_resemblance_data(resemblance_file)
        if not resemblance_data:
            print(f"  Error: Could not load resemblance data")
            continue

        # Create output directory for this target
        target_output_dir = output_dir / args.video / target_name
        target_output_dir.mkdir(parents=True, exist_ok=True)

        # Apply smoothing once (to be used by both plot and clip filtering)
        arcface_scores = np.array(resemblance_data["arcface_scores"])
        transface_scores = np.array(resemblance_data["transface_scores"])
        arcface_scores_smooth = smooth_scores(arcface_scores, args.smooth_window)
        transface_scores_smooth = smooth_scores(transface_scores, args.smooth_window)

        # Copy relevant clips first (needed for plot statistics)
        # Use the same smoothed scores that will be plotted
        copied_clips = copy_relevant_clips(
            video_name=args.video,
            segments_data=segments_data,
            resemblance_data=resemblance_data,
            video_summaries_dir=video_summaries_dir,
            output_dir=target_output_dir,
            threshold=args.threshold,
            arcface_scores_smooth=arcface_scores_smooth,
            transface_scores_smooth=transface_scores_smooth,
        )

        # Create enhanced plot with relevant segments count (with suffix if using updated target)
        # Note: plot function will apply smoothing internally, but we'll ignore returned values
        plot_path = target_output_dir / f"{target_name}_{args.video}_enhanced_curve{suffix}.png"
        _, _ = create_enhanced_resemblance_plot(
            resemblance_data=resemblance_data,
            segments_data=segments_data,
            output_path=plot_path,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            relevant_segments_count=len(copied_clips),
        )

        # Create summary report (with suffix if using updated target)
        report_path = target_output_dir / f"{target_name}_{args.video}_summary{suffix}.json"
        create_summary_report(
            target_name=target_name,
            video_name=args.video,
            resemblance_data=resemblance_data,
            segments_data=segments_data,
            copied_clips=copied_clips,
            output_path=report_path,
            threshold=args.threshold,
        )

        # Get total segments count (handle nested structure)
        segments_obj = segments_data.get("segments", [])
        if isinstance(segments_obj, dict):
            total_segs = len(segments_obj.get("segments", []))
        else:
            total_segs = len(segments_obj)

        print(f"\n  ✓ Completed {target_name}")
        print(f"    - Enhanced plot: {plot_path.relative_to(output_dir)}")
        print(f"    - Relevant clips: {len(copied_clips)}/{total_segs}")
        print(f"    - Summary report: {report_path.relative_to(output_dir)}")

        successful += 1

    # Final summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"\nSuccessfully processed {successful}/{len(target_dirs)} targets")
    print(f"Results saved to: {output_dir}/{args.video}/")


if __name__ == "__main__":
    main()
