"""
Unified Visualization Script for Face Recognition Results

This script handles all visualization tasks in the project:
1. Resemblance Curves Mode: Create resemblance curves for face recognition results
2. Final with Highlights Mode: Create enhanced plots with highlight segments overlaid

Features:
- Unified interface for all visualization workflows
- Supports both original and updated target embeddings
- Creates summary reports and copies relevant clips
"""

import os
import sys
import json
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import argparse

# Use Agg backend for non-interactive plotting
import matplotlib
matplotlib.use("Agg")


# =============================================================================
# Common Functions
# =============================================================================


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


# =============================================================================
# Mode 1: Resemblance Curves (Basic visualization from face recognition results)
# =============================================================================


class ResemblanceVisualizer:
    """Visualizer for face recognition resemblance curves"""

    def __init__(self, recognition_results_path: str):
        """
        Initialize the visualizer

        Args:
            recognition_results_path: Path to face recognition results JSON
        """
        self.recognition_results_path = recognition_results_path
        self.results = self._load_results()

    def _load_results(self) -> Dict:
        """Load recognition results from JSON file"""
        with open(self.recognition_results_path, "r") as f:
            return json.load(f)

    def compute_frame_resemblance(self, clip_results: List[Dict]) -> Dict[int, float]:
        """
        Compute maximum resemblance per frame

        Args:
            clip_results: List of recognition results for a clip

        Returns:
            Dictionary mapping frame numbers to maximum similarity scores
        """
        frame_resemblance = defaultdict(float)

        for result in clip_results:
            frame_num = result["frame_number"]
            similarity = result["similarity"]

            # Keep the maximum similarity for each frame
            frame_resemblance[frame_num] = max(frame_resemblance[frame_num], similarity)

        return dict(frame_resemblance)

    def plot_resemblance_curves(self, output_dir: str = "results/visualizations", show_plot: bool = False):
        """
        Create resemblance curves for all clips and targets

        Args:
            output_dir: Directory to save the plots
            show_plot: Whether to display plots interactively
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.results:
            print("No recognition results to visualize")
            return

        # Get all targets
        all_targets = set()
        for clip_results in self.results.values():
            all_targets.update(clip_results.keys())

        # Create a plot for each target
        for target_name in all_targets:
            self._plot_target_resemblance(target_name, output_path, show_plot)

        print(f"\nVisualization complete. Plots saved to {output_dir}/")

    def _plot_target_resemblance(self, target_name: str, output_path: Path, show_plot: bool = False):
        """
        Create resemblance curve for a specific target across all clips

        Args:
            target_name: Name of the target person
            output_path: Directory to save the plot
            show_plot: Whether to display the plot interactively
        """
        num_clips = len(self.results)

        if num_clips == 0:
            return

        # Determine subplot layout
        cols = min(3, num_clips)
        rows = (num_clips + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        fig.suptitle(f"Resemblance Curves - Target: {target_name}", fontsize=16, fontweight="bold")

        # Handle single subplot case
        if num_clips == 1:
            axes = np.array([axes])

        axes = axes.flatten() if num_clips > 1 else axes

        clip_ids = sorted(self.results.keys())

        for idx, clip_id in enumerate(clip_ids):
            ax = axes[idx] if num_clips > 1 else axes[0]

            # Get results for this clip and target
            if target_name in self.results[clip_id]:
                clip_results = self.results[clip_id][target_name]
                frame_resemblance = self.compute_frame_resemblance(clip_results)

                if frame_resemblance:
                    frames = sorted(frame_resemblance.keys())
                    similarities = [frame_resemblance[f] for f in frames]

                    # Plot the curve (use horizontal line for single points, line for multiple)
                    if len(frames) == 1:
                        # Single point - draw horizontal line from frame 0 to detected frame
                        frame_num = frames[0]
                        ax.axhline(y=similarities[0], color="b", linewidth=2, alpha=0.7)
                        ax.fill_between([0, frame_num], similarities[0], alpha=0.3)
                        # Set x-axis limits to show from 0 to the detected frame
                        ax.set_xlim([0, frame_num])
                    else:
                        # Multiple points - use line plot
                        ax.plot(frames, similarities, "b-", linewidth=1.5, alpha=0.7)
                        ax.fill_between(frames, similarities, alpha=0.3)
                        # Set x-axis limits based on actual frame range
                        ax.set_xlim([min(frames), max(frames)])

                    # Add threshold line
                    ax.axhline(y=0.6, color="r", linestyle="--", linewidth=1, label="Match Threshold (0.6)", alpha=0.7)

                    # Styling
                    ax.set_xlabel("Frame Number", fontsize=10)
                    ax.set_ylabel("Resemblance Score", fontsize=10)
                    ax.set_title(f"{clip_id}", fontsize=11, fontweight="bold")
                    ax.set_ylim([0, 1])
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)

                    # Add statistics
                    avg_sim = np.mean(similarities)
                    max_sim = np.max(similarities)
                    matches = sum(1 for s in similarities if s >= 0.6)
                    stats_text = f"Avg: {avg_sim:.3f}\nMax: {max_sim:.3f}\nMatches: {matches}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
                else:
                    ax.text(0.5, 0.5, "No faces detected", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                    ax.set_title(f"{clip_id}", fontsize=11, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No data for target", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{clip_id}", fontsize=11, fontweight="bold")

        # Hide unused subplots
        for idx in range(num_clips, len(axes)):
            axes[idx].set_visible(False)

        # Adjust layout to make room for suptitle at the top
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save the plot
        # If multiple targets exist, include target name in filename
        all_targets = list(set(target for clip in self.results.values() for target in clip.keys()))
        if len(all_targets) > 1:
            plot_filename = output_path / f"resemblance_results_{target_name}.png"
        else:
            plot_filename = output_path / "resemblance_results.png"

        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {plot_filename}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def generate_summary_report(self, output_path: str = "results/summary_report.txt"):
        """
        Generate a text summary report of recognition results

        Args:
            output_path: Path to save the summary report
        """
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PERSONALIZED VIDEO SUMMARIZATION - RECOGNITION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Get all targets
            all_targets = set()
            for clip_results in self.results.values():
                all_targets.update(clip_results.keys())

            for target_name in all_targets:
                f.write(f"\nTarget Person: {target_name}\n")
                f.write("-" * 80 + "\n")

                for clip_id in sorted(self.results.keys()):
                    if target_name in self.results[clip_id]:
                        clip_results = self.results[clip_id][target_name]
                        frame_resemblance = self.compute_frame_resemblance(clip_results)

                        if frame_resemblance:
                            similarities = list(frame_resemblance.values())
                            avg_sim = np.mean(similarities)
                            max_sim = np.max(similarities)
                            min_sim = np.min(similarities)
                            matches = sum(1 for s in similarities if s >= 0.6)
                            match_rate = (matches / len(similarities)) * 100

                            f.write(f"\n  {clip_id}:\n")
                            f.write(f"    Frames analyzed: {len(frame_resemblance)}\n")
                            f.write(f"    Average resemblance: {avg_sim:.4f}\n")
                            f.write(f"    Max resemblance: {max_sim:.4f}\n")
                            f.write(f"    Min resemblance: {min_sim:.4f}\n")
                            f.write(f"    Matches (>0.6): {matches} ({match_rate:.1f}%)\n")
                        else:
                            f.write(f"\n  {clip_id}: No faces detected\n")

                f.write("\n")

        print(f"Summary report saved to {output_path}")


def run_resemblance_curves_mode(args):
    """Run resemblance curves mode"""
    print("=" * 80)
    print("MODE: RESEMBLANCE CURVES VISUALIZATION")
    print("=" * 80)

    visualizer = ResemblanceVisualizer(args.recognition_results)
    visualizer.plot_resemblance_curves(output_dir=args.output, show_plot=args.show)
    visualizer.generate_summary_report()


# =============================================================================
# Mode 2: Final with Highlights (Enhanced visualization with highlight overlays)
# =============================================================================


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
    use_updated_target: bool = False,
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
        use_updated_target: Whether using updated target (affects clips directory name)

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

    # Create output directory (separate for original vs updated target)
    clips_dirname = "highlight_clips_updated" if use_updated_target else "highlight_clips"
    target_clips_dir = output_dir / clips_dirname
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


def run_final_with_highlights_mode(args):
    """Run final with highlights mode"""
    print("=" * 80)
    print("MODE: FINAL VISUALIZATION WITH HIGHLIGHT OVERLAYS")
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
        resemblance_file = target_dir / f"{target_name}{suffix}.json"

        if not resemblance_file.exists():
            print(f"  Warning: Resemblance data not found: {resemblance_file}")
            continue

        resemblance_data = load_resemblance_data(resemblance_file)
        if not resemblance_data:
            print(f"  Error: Could not load resemblance data")
            continue

        # Create output directory for this target
        # New structure: results/video/model/target
        model_dir_name = args.model  # "sportclip" or "qddetr"
        target_output_dir = output_dir / args.video / model_dir_name / target_name
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
            use_updated_target=args.use_updated_target,
        )

        # Create enhanced plot with relevant segments count (with suffix if using updated target)
        # Note: plot function will apply smoothing internally, but we'll ignore returned values
        plot_path = target_output_dir / f"{target_name}{suffix}.png"
        _, _ = create_enhanced_resemblance_plot(
            resemblance_data=resemblance_data,
            segments_data=segments_data,
            output_path=plot_path,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            relevant_segments_count=len(copied_clips),
        )

        # Create summary report (with suffix if using updated target)
        report_path = target_output_dir / f"{target_name}{suffix}.json"
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


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main function to visualize results based on mode"""
    parser = argparse.ArgumentParser(
        description="Unified visualization for face recognition results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  resemblance_curves:   Create resemblance curves from face recognition results
  final_with_highlights: Create enhanced plots with highlight segments overlaid

Examples:
  # Create resemblance curves (simple mode)
  python visualize_results.py --mode resemblance_curves --recognition-results results/face_recognition_results.json

  # Create final visualization with highlights
  python visualize_results.py --mode final_with_highlights --video highjump_video1 --model sportclip

  # Create final visualization with updated target embeddings
  python visualize_results.py --mode final_with_highlights --video highjump_video1 --model sportclip --use-updated-target
        """,
    )

    parser.add_argument(
        "--mode",
        type=str,
        choices=["resemblance_curves", "final_with_highlights"],
        required=True,
        help="Visualization mode"
    )

    # Resemblance curves mode arguments
    parser.add_argument("--recognition-results", type=str, default="results/face_recognition_results.json", help="[resemblance_curves] Path to face recognition results JSON")
    parser.add_argument("--show", action="store_true", help="[resemblance_curves] Show plots interactively")

    # Final with highlights mode arguments
    parser.add_argument("--video", type=str, default=None, help="[final_with_highlights] Video name (without extension, required for this mode)")
    parser.add_argument("--model", type=str, choices=["sportclip", "qddetr"], default=None, help="[final_with_highlights] Video summarization model used (required for this mode)")
    parser.add_argument("--target", type=str, default=None, help="[final_with_highlights] Specific target to process. If not provided, processes all targets.")
    parser.add_argument("--video-summaries", type=str, default="video_summaries", help="[final_with_highlights] Directory containing video summaries")
    parser.add_argument("--resemblance-results", type=str, default="resemblance_results", help="[final_with_highlights] Directory containing resemblance results")
    parser.add_argument("--threshold", type=float, default=0.6, help="[final_with_highlights] Resemblance score threshold (default: 0.6)")
    parser.add_argument("--smooth-window", type=int, default=5, help="[final_with_highlights] Smoothing window size (default: 5)")
    parser.add_argument(
        "--use-updated-target",
        action="store_true",
        help="[final_with_highlights] Use results from updated target embeddings (second pass). Reads files with '_updated' suffix.",
    )

    # Common arguments
    parser.add_argument("--output", type=str, default="results", help="Output directory for visualizations")

    args = parser.parse_args()

    # Route to appropriate mode
    if args.mode == "resemblance_curves":
        run_resemblance_curves_mode(args)
    elif args.mode == "final_with_highlights":
        if not args.video or not args.model:
            print("Error: --video and --model are required for final_with_highlights mode")
            parser.print_help()
            return
        run_final_with_highlights_mode(args)


if __name__ == "__main__":
    main()
