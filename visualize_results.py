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
    face_recognition_models: List[str] = None,
) -> Dict[str, List[str]]:
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
        face_recognition_models: List of models to use (e.g., ["arcface", "transface"]).
                                 If None, defaults to both.

    Returns:
        Dictionary mapping model name to list of copied clip names
    """
    # Default to both models if not specified
    if face_recognition_models is None:
        face_recognition_models = ["arcface", "transface"]

    # Normalize model names to lowercase
    face_recognition_models = [m.lower() for m in face_recognition_models]

    model = segments_data.get("model", "sportclip")
    target_name = resemblance_data["target"]

    timestamps = np.array(resemblance_data["timestamps_seconds"])

    # Use pre-smoothed scores if provided, otherwise use raw scores
    if arcface_scores_smooth is None:
        arcface_scores = np.array(resemblance_data["arcface_scores"])
    else:
        arcface_scores = arcface_scores_smooth

    if transface_scores_smooth is None:
        transface_scores = np.array(resemblance_data["transface_scores"])
    else:
        transface_scores = transface_scores_smooth

    # Find source clips directory
    model_dir = "SportCLIP" if model == "sportclip" else "QD-DETR"
    clips_dir = video_summaries_dir / model_dir / video_name

    if not clips_dir.exists():
        print(f"  Warning: Clips directory not found: {clips_dir}")
        return {}

    # Dictionary to store copied clips per model
    all_copied_clips = {}

    # Handle nested structure
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    # Process each face recognition model independently
    for face_model in face_recognition_models:
        if face_model == "arcface":
            scores = arcface_scores
            model_name = "ArcFace"
        elif face_model == "transface":
            scores = transface_scores
            model_name = "TransFace"
        else:
            print(f"  Warning: Unknown face recognition model: {face_model}")
            continue

        # Create model-specific output directory
        suffix = "_updated" if use_updated_target else ""
        clips_dirname = f"{model_name}_highlight_clips{suffix}"
        target_clips_dir = output_dir / clips_dirname
        target_clips_dir.mkdir(parents=True, exist_ok=True)

        copied_clips = []

        print(f"\n  Processing {model_name} model:")

        for segment in segments:
            clip_name = segment["clip_name"]
            clip_path = clips_dir / clip_name

            if not clip_path.exists():
                print(f"    Warning: Clip not found: {clip_path}")
                continue

            # Use timing information from segment metadata
            start_time = segment.get("start_time")
            end_time = segment.get("end_time")

            if start_time is not None and end_time is not None:
                times = (start_time, end_time)
                has_target = check_segment_overlap(times, timestamps, scores, threshold)

                if has_target:
                    # Copy clip to results
                    dest_path = target_clips_dir / clip_name
                    shutil.copy2(clip_path, dest_path)
                    copied_clips.append(clip_name)
                    print(f"    ✓ Copied clip: {clip_name}")

        all_copied_clips[face_model] = copied_clips
        print(f"  {model_name}: Copied {len(copied_clips)} clip(s)")

    return all_copied_clips


def create_summary_report(target_name: str, video_name: str, resemblance_data: Dict, segments_data: Dict, copied_clips: Dict[str, List[str]], output_path: Path, threshold: float):
    """
    Create a summary report JSON file

    Args:
        target_name: Target person name
        video_name: Video name
        resemblance_data: Resemblance curve data
        segments_data: Video segments metadata
        copied_clips: Dictionary mapping model name to list of copied clip names
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
        "relevant_clips_per_model": {
            "arcface": {"count": len(copied_clips.get("arcface", [])), "clips": copied_clips.get("arcface", [])},
            "transface": {"count": len(copied_clips.get("transface", [])), "clips": copied_clips.get("transface", [])},
        },
        "resemblance_statistics": {
            "arcface": {"mean": float(np.mean(arcface_scores)), "max": float(np.max(arcface_scores)), "frames_above_threshold": int(np.sum(arcface_scores >= threshold))},
            "transface": {"mean": float(np.mean(transface_scores)), "max": float(np.max(transface_scores)), "frames_above_threshold": int(np.sum(transface_scores >= threshold))},
        },
    }

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    print(f"  ✓ Created summary report: {output_path.name}")


def assign_highlights_to_athletes(
    all_resemblance_data: Dict[str, Dict],
    segments_data: Dict,
    threshold: float,
    smooth_window: int,
    model: str = "arcface",
) -> Dict[int, Optional[str]]:
    """
    Assign each highlight segment to an athlete based on resemblance curves

    Algorithm: Scan timeline left-to-right. When an athlete's score crosses threshold,
    assign all subsequent highlights to that athlete until another athlete crosses threshold.

    Args:
        all_resemblance_data: Dictionary mapping target_name to resemblance data
        segments_data: Video segments metadata
        threshold: Similarity threshold
        smooth_window: Smoothing window size
        model: Which model to use for assignment ("arcface" or "transface")

    Returns:
        Dictionary mapping segment index to assigned athlete name (or None if unassigned)
    """
    if not all_resemblance_data:
        return {}

    # Get timestamps from first target
    first_target = next(iter(all_resemblance_data.values()))
    timestamps = np.array(first_target["timestamps_seconds"])

    # Get sorted athlete names (consistent ordering)
    athlete_names = sorted(all_resemblance_data.keys())

    # Compute smoothed scores for all athletes using the specified model
    athlete_scores = {}
    for athlete_name in athlete_names:
        data = all_resemblance_data[athlete_name]

        if model.lower() == "arcface":
            scores = np.array(data["arcface_scores"])
        elif model.lower() == "transface":
            scores = np.array(data["transface_scores"])
        else:
            raise ValueError(f"Unknown model: {model}. Must be 'arcface' or 'transface'")

        smoothed = smooth_scores(scores, smooth_window)
        athlete_scores[athlete_name] = smoothed

    # Get segments
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    # Assignment dictionary: segment_index -> athlete_name
    assignments = {}
    current_athlete = None

    # Scan timeline left-to-right
    for seg_idx, segment in enumerate(segments):
        start_time = segment.get("start_time")

        if start_time is None:
            assignments[seg_idx] = current_athlete
            continue

        # Check if any athlete crosses threshold at or before this segment start
        # Find the timestamp index closest to segment start
        start_idx = np.searchsorted(timestamps, start_time)

        # Look back a bit to catch threshold crossings just before the segment
        # (within 5 seconds before segment start)
        lookback_time = max(0, start_time - 5)
        lookback_idx = np.searchsorted(timestamps, lookback_time)

        # Check if any athlete crossed threshold in this window
        crossed_athletes = []
        for athlete_name in athlete_names:
            scores = athlete_scores[athlete_name]
            # Check scores in lookback window
            window_scores = scores[lookback_idx : start_idx + 1] if start_idx < len(scores) else scores[lookback_idx:]
            if len(window_scores) > 0 and np.any(window_scores >= threshold):
                # Find the last crossing point in this window
                crossing_indices = np.where(window_scores >= threshold)[0]
                last_crossing = lookback_idx + crossing_indices[-1]
                crossed_athletes.append((athlete_name, last_crossing))

        # If athletes crossed threshold, assign to the most recent one
        if crossed_athletes:
            # Sort by crossing time (most recent first)
            crossed_athletes.sort(key=lambda x: x[1], reverse=True)
            current_athlete = crossed_athletes[0][0]

        assignments[seg_idx] = current_athlete

    return assignments


def assign_highlights_to_athletes_instant(
    all_resemblance_data: Dict[str, Dict],
    segments_data: Dict,
    threshold: float,
    smooth_window: int,
    model: str = "arcface",
    temporal_expansion: float = 3.0,
) -> Dict[int, Optional[str]]:
    """
    Assign each highlight segment to an athlete based on instant overlap with resemblance curves

    Algorithm: For each highlight segment, check if any athlete's resemblance score exceeds
    the threshold during that segment's time window (with optional temporal expansion).
    If multiple athletes exceed threshold, assign to the one with highest average score.

    Args:
        all_resemblance_data: Dictionary mapping target_name to resemblance data
        segments_data: Video segments metadata
        threshold: Similarity threshold
        smooth_window: Smoothing window size
        model: Which model to use for assignment ("arcface" or "transface")
        temporal_expansion: Seconds to expand on each side of segment (default: 3.0)
                           This helps catch athletes even when face is occluded during performance

    Returns:
        Dictionary mapping segment index to assigned athlete name (or None if unassigned)
    """
    if not all_resemblance_data:
        return {}

    # Get timestamps from first target
    first_target = next(iter(all_resemblance_data.values()))
    timestamps = np.array(first_target["timestamps_seconds"])

    # Get sorted athlete names (consistent ordering)
    athlete_names = sorted(all_resemblance_data.keys())

    # Compute smoothed scores for all athletes using the specified model
    athlete_scores = {}
    for athlete_name in athlete_names:
        data = all_resemblance_data[athlete_name]

        if model.lower() == "arcface":
            scores = np.array(data["arcface_scores"])
        elif model.lower() == "transface":
            scores = np.array(data["transface_scores"])
        else:
            raise ValueError(f"Unknown model: {model}. Must be 'arcface' or 'transface'")

        smoothed = smooth_scores(scores, smooth_window)
        athlete_scores[athlete_name] = smoothed

    # Get segments
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    # Assignment dictionary: segment_index -> athlete_name
    assignments = {}

    # Process each segment independently
    for seg_idx, segment in enumerate(segments):
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")

        if start_time is None or end_time is None:
            assignments[seg_idx] = None
            continue

        # Expand temporal window
        expanded_start = max(0, start_time - temporal_expansion)
        expanded_end = end_time + temporal_expansion

        # Find timestamp indices for this segment window
        start_idx = np.searchsorted(timestamps, expanded_start)
        end_idx = np.searchsorted(timestamps, expanded_end)

        # Check each athlete's scores in this window
        candidate_athletes = []
        for athlete_name in athlete_names:
            scores = athlete_scores[athlete_name]

            # Get scores in this window
            window_scores = scores[start_idx:end_idx]

            if len(window_scores) > 0 and np.any(window_scores >= threshold):
                # Calculate average score above threshold
                above_threshold_scores = window_scores[window_scores >= threshold]
                avg_score = np.mean(above_threshold_scores)
                max_score = np.max(window_scores)

                candidate_athletes.append((athlete_name, avg_score, max_score))

        # Assign to athlete with highest average score above threshold
        if candidate_athletes:
            # Sort by average score (descending), then by max score
            candidate_athletes.sort(key=lambda x: (x[1], x[2]), reverse=True)
            assignments[seg_idx] = candidate_athletes[0][0]
        else:
            assignments[seg_idx] = None

    return assignments


def create_assigned_highlights_plot(
    all_resemblance_data: Dict[str, Dict],
    segments_data: Dict,
    assignments_arcface: Dict[int, Optional[str]],
    assignments_transface: Dict[int, Optional[str]],
    athlete_colors: Dict[str, str],
    output_path: Path,
    threshold: float,
    video_name_arg: str = None,
) -> None:
    """
    Create a plot showing highlight segments colored by their assigned athlete
    Shows both ArcFace and TransFace assignments in separate subplots, plus ground truth if available

    Args:
        all_resemblance_data: Dictionary mapping target_name to resemblance data
        segments_data: Video segments metadata
        assignments_arcface: Dictionary mapping segment index to athlete name (ArcFace)
        assignments_transface: Dictionary mapping segment index to athlete name (TransFace)
        athlete_colors: Color mapping for athletes
        output_path: Path to save plot
        threshold: Similarity threshold
        video_name_arg: Video name for finding ground truth CSV (optional)
    """
    if not all_resemblance_data:
        print("  Warning: No resemblance data to plot")
        return

    # Get video info
    first_target = next(iter(all_resemblance_data.values()))
    video_name = first_target["video"]
    timestamps = np.array(first_target["timestamps_seconds"])
    video_duration = timestamps[-1]

    # Get segments
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    # Try to load ground truth annotations
    ground_truth_data = None
    if video_name_arg:
        import csv
        import cv2

        gt_csv_path = Path("data") / f"{video_name_arg}_identity.csv"
        if gt_csv_path.exists():
            try:
                # Get video FPS to convert frames to seconds
                video_path = None
                data_dir = Path("data")
                video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
                for ext in video_extensions:
                    candidate = data_dir / f"{video_name_arg}{ext}"
                    if candidate.exists():
                        video_path = str(candidate)
                        break

                if video_path:
                    cap = cv2.VideoCapture(video_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    cap.release()

                    # Parse CSV
                    ground_truth_data = []
                    with open(gt_csv_path, "r") as f:
                        reader = csv.DictReader(f)
                        for row in reader:
                            if row["Event type"] == "Highlight":
                                first_frame = int(row["First frame"])
                                last_frame = int(row["Last frame"])
                                identity = row["Identity"].lower()  # Convert to lowercase

                                # Convert frames to seconds
                                start_time = first_frame / fps
                                end_time = last_frame / fps

                                ground_truth_data.append({"start_time": start_time, "end_time": end_time, "athlete": identity})

                    print(f"  ✓ Loaded ground truth with {len(ground_truth_data)} segments")
            except Exception as e:
                print(f"  Warning: Could not load ground truth: {e}")
                ground_truth_data = None

    # Create figure with two or three subplots depending on ground truth availability
    # Slightly taller figure to accommodate evaluation table
    if ground_truth_data:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(16, 12), dpi=150)
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("white")

    def format_time(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

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

    # Helper function to plot assignments on an axis
    def plot_assignments_on_axis(ax, assignments, model_name):
        """Plot assigned segments on a single axis"""
        # Draw assigned highlight segments
        unassigned_segments = []
        assigned_count = {}

        for seg_idx, segment in enumerate(segments):
            start_time = segment.get("start_time")
            end_time = segment.get("end_time")

            if start_time is None or end_time is None:
                continue

            athlete = assignments.get(seg_idx)

            if athlete:
                # Draw colored segment
                color = athlete_colors.get(athlete, "#CCCCCC")
                width = end_time - start_time

                rect = patches.Rectangle(
                    (start_time, 0),
                    width,
                    1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.6,
                    transform=ax.get_xaxis_transform(),
                )
                ax.add_patch(rect)

                # Add segment number label
                ax.text(
                    start_time + width / 2,
                    0.95,
                    f"H{seg_idx + 1}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8, edgecolor="black"),
                    color="white",
                )

                # Track assignment counts
                assigned_count[athlete] = assigned_count.get(athlete, 0) + 1
            else:
                # Draw unassigned segment in gray
                width = end_time - start_time

                rect = patches.Rectangle(
                    (start_time, 0),
                    width,
                    1,
                    linewidth=2,
                    edgecolor="#999999",
                    facecolor="#CCCCCC",
                    alpha=0.3,
                    transform=ax.get_xaxis_transform(),
                )
                ax.add_patch(rect)

                # Add segment number label
                ax.text(
                    start_time + width / 2,
                    0.95,
                    f"H{seg_idx + 1}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#CCCCCC", alpha=0.6, edgecolor="gray"),
                    color="black",
                )

                unassigned_segments.append(seg_idx + 1)

        # Axis formatting
        ax.set_ylabel("", fontsize=14)
        ax.set_title(f"Highlight Segments Assigned to Athletes ({model_name}) - {video_name}", fontsize=14, fontweight="bold", pad=15)
        ax.set_ylim([0, 1])
        ax.set_xlim([0, video_duration])
        ax.set_xticks(tick_times)
        ax.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")
        ax.set_yticks([])
        ax.grid(True, axis="x", alpha=0.3, linestyle=":", linewidth=1)

        # Create legend with athlete assignments
        legend_elements = []
        athlete_names = sorted(assigned_count.keys())
        for athlete in athlete_names:
            color = athlete_colors.get(athlete, "#CCCCCC")
            count = assigned_count[athlete]
            legend_elements.append(patches.Patch(facecolor=color, edgecolor="black", alpha=0.6, label=f"{athlete} ({count})"))

        if unassigned_segments:
            legend_elements.append(patches.Patch(facecolor="#CCCCCC", edgecolor="gray", alpha=0.3, label=f"Unassigned ({len(unassigned_segments)})"))

        ax.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, framealpha=0.9, title="Athlete (Segments)", title_fontsize=10)

        return len(unassigned_segments), sum(assigned_count.values())

    # Evaluate assignments against ground truth if available
    def evaluate_assignments(assignments, ground_truth_data, segments):
        """
        Evaluate predicted assignments against ground truth using IoU-based matching

        Matching approach:
        - For each predicted segment, find all overlapping ground truth segments
        - Use Intersection over Union (IoU) >= 0.3 as matching criterion
        - A match is correct if IoU >= 0.3 AND athlete names match
        - Handle many-to-many relationships by tracking best matches

        Returns:
            Dict mapping athlete name to (gt_count, pred_count, correct_count)
        """
        if not ground_truth_data:
            return {}

        athlete_stats = {}

        # Convert assignments to time-based segments
        pred_segments = []
        for seg_idx, segment in enumerate(segments):
            athlete = assignments.get(seg_idx)
            if athlete:
                pred_segments.append({"start": segment.get("start_time"), "end": segment.get("end_time"), "athlete": athlete, "matched": False})

        # Track ground truth segments
        gt_segments_by_athlete = {}
        for gt_seg in ground_truth_data:
            athlete = gt_seg["athlete"]
            if athlete not in gt_segments_by_athlete:
                gt_segments_by_athlete[athlete] = []
            gt_segments_by_athlete[athlete].append({"start": gt_seg["start_time"], "end": gt_seg["end_time"], "matched": False})

        # Compute IoU and match segments
        def compute_iou(seg1_start, seg1_end, seg2_start, seg2_end):
            """Compute Intersection over Union for two time segments"""
            intersection_start = max(seg1_start, seg2_start)
            intersection_end = min(seg1_end, seg2_end)
            intersection = max(0, intersection_end - intersection_start)

            union = (seg1_end - seg1_start) + (seg2_end - seg2_start) - intersection
            return intersection / union if union > 0 else 0

        # Match predicted segments to ground truth
        iou_threshold = 0.3
        matched_pairs = []

        for pred_seg in pred_segments:
            if pred_seg["athlete"] not in gt_segments_by_athlete:
                continue

            best_iou = 0
            best_gt_idx = None

            # Find best matching ground truth segment for this athlete
            for gt_idx, gt_seg in enumerate(gt_segments_by_athlete[pred_seg["athlete"]]):
                iou = compute_iou(pred_seg["start"], pred_seg["end"], gt_seg["start"], gt_seg["end"])
                if iou >= iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx

            if best_gt_idx is not None:
                matched_pairs.append((pred_seg, gt_segments_by_athlete[pred_seg["athlete"]][best_gt_idx]))
                pred_seg["matched"] = True
                gt_segments_by_athlete[pred_seg["athlete"]][best_gt_idx]["matched"] = True

        # Compute stats per athlete
        all_athletes = set(gt_segments_by_athlete.keys()) | set(assignments.values())
        for athlete in all_athletes:
            if athlete is None:
                continue

            gt_count = len(gt_segments_by_athlete.get(athlete, []))
            pred_count = sum(1 for seg in pred_segments if seg["athlete"] == athlete)
            correct_count = sum(1 for seg in pred_segments if seg["athlete"] == athlete and seg["matched"])

            athlete_stats[athlete] = {"gt": gt_count, "pred": pred_count, "correct": correct_count}

        return athlete_stats

    # Evaluate both models
    eval_arcface = evaluate_assignments(assignments_arcface, ground_truth_data, segments) if ground_truth_data else {}
    eval_transface = evaluate_assignments(assignments_transface, ground_truth_data, segments) if ground_truth_data else {}

    # Plot ArcFace assignments
    unassigned_arc, assigned_arc = plot_assignments_on_axis(ax1, assignments_arcface, "ArcFace")

    # Plot TransFace assignments
    unassigned_trans, assigned_trans = plot_assignments_on_axis(ax2, assignments_transface, "TransFace")

    # Plot ground truth if available
    if ground_truth_data:
        # Count ground truth segments per athlete
        gt_count = {}
        for gt_segment in ground_truth_data:
            start_time = gt_segment["start_time"]
            end_time = gt_segment["end_time"]
            athlete = gt_segment["athlete"]

            if start_time is not None and end_time is not None:
                # Draw colored segment
                color = athlete_colors.get(athlete, "#CCCCCC")
                width = end_time - start_time

                rect = patches.Rectangle(
                    (start_time, 0),
                    width,
                    1,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.6,
                    transform=ax3.get_xaxis_transform(),
                )
                ax3.add_patch(rect)

                # Track counts
                gt_count[athlete] = gt_count.get(athlete, 0) + 1

        # Axis formatting
        ax3.set_xlabel("Video Time (MM:SS)", fontsize=14, fontweight="bold")
        ax3.set_ylabel("", fontsize=14)
        ax3.set_title(f"Ground Truth - {video_name}", fontsize=14, fontweight="bold", pad=15)
        ax3.set_ylim([0, 1])
        ax3.set_xlim([0, video_duration])
        ax3.set_xticks(tick_times)
        ax3.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")
        ax3.set_yticks([])
        ax3.grid(True, axis="x", alpha=0.3, linestyle=":", linewidth=1)

        # Create legend with ground truth counts
        legend_elements = []
        athlete_names_gt = sorted(gt_count.keys())
        for athlete in athlete_names_gt:
            color = athlete_colors.get(athlete, "#CCCCCC")
            count = gt_count[athlete]
            legend_elements.append(patches.Patch(facecolor=color, edgecolor="black", alpha=0.6, label=f"{athlete} ({count})"))

        ax3.legend(handles=legend_elements, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9, framealpha=0.9, title="Athlete (Segments)", title_fontsize=10)

        # Add statistics including ground truth and evaluation table
        total_segments = len(segments)
        gt_segments = len(ground_truth_data)

        # Create evaluation table if we have ground truth
        if eval_arcface or eval_transface:
            # Get all athletes
            all_athletes = sorted(set(list(eval_arcface.keys()) + list(eval_transface.keys())))

            # Build table header
            table_text = "Evaluation Results:\n"
            table_text += "─" * 90 + "\n"
            table_text += f"{'Athlete':<15} │ {'GT':<4} │ ArcFace: {'Pred':<4} {'Corr':<4} {'Prec':<6} │ TransFace: {'Pred':<4} {'Corr':<4} {'Prec':<6}\n"
            table_text += "─" * 90 + "\n"

            # Build table rows and track totals
            total_gt = 0
            total_arc_pred = 0
            total_arc_corr = 0
            total_trans_pred = 0
            total_trans_corr = 0
            arc_prec_values = []
            trans_prec_values = []

            for athlete in all_athletes:
                arc_stats = eval_arcface.get(athlete, {"gt": 0, "pred": 0, "correct": 0})
                trans_stats = eval_transface.get(athlete, {"gt": 0, "pred": 0, "correct": 0})

                gt_count = arc_stats["gt"]  # Same for both models

                # ArcFace stats
                arc_pred = arc_stats["pred"]
                arc_corr = arc_stats["correct"]
                # Show 0% if athlete has ground truth but no predictions, "-" if no ground truth
                if arc_pred > 0:
                    arc_prec = f"{(arc_corr/arc_pred*100):.1f}%"
                    arc_prec_values.append(arc_corr / arc_pred * 100)
                elif gt_count > 0:
                    arc_prec = " 0.0%"
                    arc_prec_values.append(0.0)
                else:
                    arc_prec = "  -  "

                # TransFace stats
                trans_pred = trans_stats["pred"]
                trans_corr = trans_stats["correct"]
                # Show 0% if athlete has ground truth but no predictions, "-" if no ground truth
                if trans_pred > 0:
                    trans_prec = f"{(trans_corr/trans_pred*100):.1f}%"
                    trans_prec_values.append(trans_corr / trans_pred * 100)
                elif gt_count > 0:
                    trans_prec = " 0.0%"
                    trans_prec_values.append(0.0)
                else:
                    trans_prec = "  -  "

                # Update totals
                total_gt += gt_count
                total_arc_pred += arc_pred
                total_arc_corr += arc_corr
                total_trans_pred += trans_pred
                total_trans_corr += trans_corr

                # Format row
                table_text += f"{athlete:<15} │ {gt_count:<4} │          {arc_pred:<4} {arc_corr:<4} {arc_prec:<6} │           {trans_pred:<4} {trans_corr:<4} {trans_prec:<6}\n"

            # Add summary row
            table_text += "─" * 90 + "\n"
            avg_arc_prec = f"{(sum(arc_prec_values)/len(arc_prec_values)):.1f}%" if arc_prec_values else "  -  "
            avg_trans_prec = f"{(sum(trans_prec_values)/len(trans_prec_values)):.1f}%" if trans_prec_values else "  -  "
            table_text += (
                f"{'AVERAGE/TOTAL':<15} │ {total_gt:<4} │          {total_arc_pred:<4} {total_arc_corr:<4} {avg_arc_prec:<6} │          "
                f" {total_trans_pred:<4} {total_trans_corr:<4} {avg_trans_prec:<6}\n"
            )
            table_text += "─" * 90 + "\n"
            table_text += f"Total: {total_segments} segments | Ground Truth: {gt_segments} segments | IoU threshold: 0.3"

            fig.text(0.5, 0.01, table_text, ha="center", fontsize=8, family="monospace", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5), verticalalignment="bottom")
        else:
            # Fallback to simple stats if no evaluation data
            stats_text = f"Total Segments: {total_segments} | Ground Truth Segments: {gt_segments}\n"
            stats_text += f"ArcFace: {assigned_arc} assigned, {unassigned_arc} unassigned | "
            stats_text += f"TransFace: {assigned_trans} assigned, {unassigned_trans} unassigned"
            fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
    else:
        # Set xlabel only on bottom plot (ax2)
        ax2.set_xlabel("Video Time (MM:SS)", fontsize=14, fontweight="bold")

        # Add statistics without ground truth
        total_segments = len(segments)
        stats_text = f"Total Segments: {total_segments}\n"
        stats_text += f"ArcFace: {assigned_arc} assigned, {unassigned_arc} unassigned | "
        stats_text += f"TransFace: {assigned_trans} assigned, {unassigned_trans} unassigned"
        fig.text(0.5, 0.01, stats_text, ha="center", fontsize=10, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Adjust layout - more bottom space for evaluation table
    # Calculate dynamic bottom margin based on number of athletes
    if ground_truth_data and (eval_arcface or eval_transface):
        num_athletes = len(set(list(eval_arcface.keys()) + list(eval_transface.keys())))
        # Base margin + additional space per athlete + space for summary row (approximately 0.012 per row)
        bottom_margin = 0.12 + ((num_athletes + 1) * 0.012)
        plt.tight_layout(rect=[0, bottom_margin, 0.85, 1])
    else:
        plt.tight_layout(rect=[0, 0.06, 0.85, 1])

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight", facecolor="white")
    plt.close()

    print(f"  ✓ Created assigned highlights plot: {output_path.name}")
    print(f"    ArcFace: {assigned_arc}/{total_segments} assigned | TransFace: {assigned_trans}/{total_segments} assigned")

    # Copy assigned clips to organized directory structure
    # Structure: results/<video>/<model>/highlights_assigned/<athlete>/{ArcFace,TransFace}/
    import shutil

    highlights_assigned_dir = output_path.parent / "highlights_assigned"

    # Get all unique athletes from both assignments
    all_assigned_athletes = set()
    for athlete in assignments_arcface.values():
        if athlete:
            all_assigned_athletes.add(athlete)
    for athlete in assignments_transface.values():
        if athlete:
            all_assigned_athletes.add(athlete)

    # Copy clips for each athlete
    clips_copied = {"ArcFace": 0, "TransFace": 0}

    for athlete in sorted(all_assigned_athletes):
        # Create athlete directories
        athlete_arcface_dir = highlights_assigned_dir / athlete / "ArcFace"
        athlete_transface_dir = highlights_assigned_dir / athlete / "TransFace"
        athlete_arcface_dir.mkdir(parents=True, exist_ok=True)
        athlete_transface_dir.mkdir(parents=True, exist_ok=True)

        # Copy ArcFace clips
        for seg_idx, assigned_athlete in assignments_arcface.items():
            if assigned_athlete == athlete and seg_idx < len(segments):
                segment = segments[seg_idx]
                clip_path = segment.get("clip_path")
                if clip_path:
                    source = Path("video_summaries") / clip_path
                    if source.exists():
                        dest = athlete_arcface_dir / source.name
                        shutil.copy2(source, dest)
                        clips_copied["ArcFace"] += 1

        # Copy TransFace clips
        for seg_idx, assigned_athlete in assignments_transface.items():
            if assigned_athlete == athlete and seg_idx < len(segments):
                segment = segments[seg_idx]
                clip_path = segment.get("clip_path")
                if clip_path:
                    source = Path("video_summaries") / clip_path
                    if source.exists():
                        dest = athlete_transface_dir / source.name
                        shutil.copy2(source, dest)
                        clips_copied["TransFace"] += 1

    if clips_copied["ArcFace"] > 0 or clips_copied["TransFace"] > 0:
        print(f"  ✓ Copied assigned clips to: {highlights_assigned_dir.relative_to(Path.cwd())}")
        print(f"    ArcFace: {clips_copied['ArcFace']} clips | TransFace: {clips_copied['TransFace']} clips")


def create_combined_athletes_plot(
    all_resemblance_data: Dict[str, Dict],
    segments_data: Dict,
    output_path: Path,
    threshold: float = 0.6,
    smooth_window: int = 5,
    zoom: bool = False,
) -> None:
    """
    Create combined resemblance curve plot showing all athletes together

    Args:
        all_resemblance_data: Dictionary mapping target_name to resemblance data
        segments_data: Video segments metadata
        output_path: Path to save plot
        threshold: Similarity threshold
        smooth_window: Smoothing window size
        zoom: If True, auto-scale y-axis to data range; if False, use [0, 1]
    """
    if not all_resemblance_data:
        print("  Warning: No resemblance data to plot")
        return

    # Get video name from first target
    first_target = next(iter(all_resemblance_data.values()))
    video_name = first_target["video"]

    # Generate distinct colors for athletes using Kelly's maximal contrast palette
    import hashlib
    import colorsys

    def get_distinct_colors(names: List[str]) -> Dict[str, str]:
        """Generate visually distinct colors for athletes"""
        sorted_names = sorted(names)
        n = len(sorted_names)

        # Use Kelly's 22 colors of maximum contrast for small n
        if n <= 12:
            distinct_palette = [
                "#F3C300",  # vivid yellow
                "#875692",  # strong purple
                "#F38400",  # vivid orange
                "#A1CAF1",  # very light blue
                "#BE0032",  # vivid red
                "#C2B280",  # grayish yellow
                "#008856",  # vivid green
                "#E68FAC",  # strong purplish pink
                "#0067A5",  # strong blue
                "#F99379",  # strong yellowish pink
                "#604E97",  # strong violet
                "#F6A600",  # vivid orange yellow
                "#B3446C",  # strong purplish red
                "#DCD300",  # vivid greenish yellow
                "#882D17",  # strong reddish brown
                "#8DB600",  # vivid yellowish green
                "#654522",  # deep yellowish brown
                "#E25822",  # vivid reddish orange
                "#2B3D26",  # dark olive green
            ]

            # Use hash to deterministically assign colors
            color_mapping = {}
            used_indices = set()
            for name in sorted_names:
                hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
                for attempt in range(len(distinct_palette)):
                    color_idx = (hash_val + attempt) % len(distinct_palette)
                    if color_idx not in used_indices:
                        color_mapping[name] = distinct_palette[color_idx]
                        used_indices.add(color_idx)
                        break
            return color_mapping
        else:
            # For larger n, use evenly distributed hues
            colors = {}
            for i, name in enumerate(sorted_names):
                hue = i / n
                rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                colors[name] = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
            return colors

    # Create color mapping for all athletes
    athlete_names = sorted(all_resemblance_data.keys())
    athlete_colors = get_distinct_colors(athlete_names)

    # Collect all smoothed scores for y-axis limits
    all_arcface_smooth = []
    all_transface_smooth = []
    athlete_data = {}

    for athlete_name in athlete_names:
        data = all_resemblance_data[athlete_name]
        arcface_scores = np.array(data["arcface_scores"])
        transface_scores = np.array(data["transface_scores"])
        arcface_smooth = smooth_scores(arcface_scores, smooth_window)
        transface_smooth = smooth_scores(transface_scores, smooth_window)

        athlete_data[athlete_name] = {"arcface_smooth": arcface_smooth, "transface_smooth": transface_smooth}

        all_arcface_smooth.extend(arcface_smooth)
        all_transface_smooth.extend(transface_smooth)

    # Determine y-axis limits
    if zoom:
        arcface_min = min(all_arcface_smooth)
        arcface_max = max(all_arcface_smooth)
        transface_min = min(all_transface_smooth)
        transface_max = max(all_transface_smooth)

        # Add 5% padding
        arcface_padding = (arcface_max - arcface_min) * 0.05
        transface_padding = (transface_max - transface_min) * 0.05
        arcface_ylim = [max(0, arcface_min - arcface_padding), min(1, arcface_max + arcface_padding)]
        transface_ylim = [max(0, transface_min - transface_padding), min(1, transface_max + transface_padding)]
    else:
        arcface_ylim = [0, 1]
        transface_ylim = [0, 1]

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("white")

    def format_time(seconds):
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    # Get timestamps from first target (all should have same timestamps)
    timestamps = np.array(first_target["timestamps_seconds"])
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
        segments_obj = segments_data.get("segments", [])
        if isinstance(segments_obj, dict):
            segments = segments_obj.get("segments", [])
        else:
            segments = segments_obj

        for i, segment in enumerate(segments, 1):
            start_time = segment.get("start_time")
            end_time = segment.get("end_time")

            if start_time is not None and end_time is not None:
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

    # Plot ArcFace curves for all athletes
    athlete_lines = []
    for athlete_name in athlete_names:
        arcface_smooth = athlete_data[athlete_name]["arcface_smooth"]
        color = athlete_colors[athlete_name]
        (line,) = ax1.plot(timestamps, arcface_smooth, color=color, linewidth=2.0, alpha=0.8)
        athlete_lines.append((line, athlete_name))

    # Add threshold line and highlight segments to ArcFace plot
    threshold_line = ax1.axhline(y=threshold, color="#F18F01", linestyle="--", linewidth=2, alpha=0.7)
    add_highlight_segments(ax1)

    ax1.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    title_suffix = " (Zoomed)" if zoom else ""
    ax1.set_title(f"ArcFace Resemblance Curves - All Athletes in {video_name}{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    ax1.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax1.set_ylim(arcface_ylim)
    ax1.set_xlim([timestamps[0], timestamps[-1]])
    ax1.set_xticks(tick_times)
    ax1.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Create internal legend for threshold and highlights
    highlight_patch = patches.Patch(facecolor="red", edgecolor="red", alpha=0.15, linewidth=2)
    ax1.legend(handles=[threshold_line, highlight_patch], labels=[f"Threshold ({threshold})", "Highlight Segment"], loc="upper right", fontsize=10, framealpha=0.9)

    # Plot TransFace curves for all athletes
    for athlete_name in athlete_names:
        transface_smooth = athlete_data[athlete_name]["transface_smooth"]
        color = athlete_colors[athlete_name]
        ax2.plot(timestamps, transface_smooth, color=color, linewidth=2.0, alpha=0.8)

    # Add threshold line and highlight segments to TransFace plot
    threshold_line_ax2 = ax2.axhline(y=threshold, color="#F18F01", linestyle="--", linewidth=2, alpha=0.7)
    add_highlight_segments(ax2)

    ax2.set_xlabel("Video Time (MM:SS)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    ax2.set_title(f"TransFace Resemblance Curves - All Athletes in {video_name}{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    ax2.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax2.set_ylim(transface_ylim)
    ax2.set_xlim([timestamps[0], timestamps[-1]])
    ax2.set_xticks(tick_times)
    ax2.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Create internal legend for threshold and highlights
    highlight_patch_ax2 = patches.Patch(facecolor="red", edgecolor="red", alpha=0.15, linewidth=2)
    ax2.legend(handles=[threshold_line_ax2, highlight_patch_ax2], labels=[f"Threshold ({threshold})", "Highlight Segment"], loc="upper right", fontsize=10, framealpha=0.9)

    # Create external legend for athletes (shared for both plots)
    fig.legend(
        [line for line, _ in athlete_lines],
        [name for _, name in athlete_lines],
        loc="center left",
        bbox_to_anchor=(0.88, 0.5),
        fontsize=10,
        framealpha=0.9,
        ncol=1,
        title="Athletes",
        title_fontsize=11,
    )

    # Get segment count
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        num_segments = len(segments_obj.get("segments", []))
    else:
        num_segments = len(segments_obj)

    stats_text = f"Athletes: {len(athlete_names)} | Highlight Segments: {num_segments} | Threshold: {threshold}"
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Adjust layout to make room for legend
    plt.tight_layout(rect=[0, 0.04, 0.85, 1])

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight", facecolor="white")
    plt.close()

    zoom_suffix = " (zoomed)" if zoom else ""
    print(f"  ✓ Created combined athletes plot{zoom_suffix}: {output_path.name}")


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
    all_resemblance_data = {}  # Collect all resemblance data for combined plot

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

        # Store resemblance data for combined plot
        all_resemblance_data[target_name] = resemblance_data

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
        # Parse face recognition models from command line
        face_models = None
        if hasattr(args, "face_recognition_models") and args.face_recognition_models:
            face_models = args.face_recognition_models

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
            face_recognition_models=face_models,
        )

        # Calculate total relevant clips across all models
        total_relevant_clips = sum(len(clips) for clips in copied_clips.values())

        # Create enhanced plot with relevant segments count (with suffix if using updated target)
        # Note: plot function will apply smoothing internally, but we'll ignore returned values
        plot_path = target_output_dir / f"{target_name}{suffix}.png"
        _, _ = create_enhanced_resemblance_plot(
            resemblance_data=resemblance_data,
            segments_data=segments_data,
            output_path=plot_path,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            relevant_segments_count=total_relevant_clips,
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

    # Create combined athletes plots (normal and zoomed versions)
    if all_resemblance_data:
        print("\n" + "=" * 80)
        print("CREATING COMBINED ATHLETES PLOTS")
        print("=" * 80)

        # Create combined plot in video/model directory
        model_dir_name = args.model
        combined_output_dir = output_dir / args.video / model_dir_name
        combined_output_dir.mkdir(parents=True, exist_ok=True)

        suffix = "_updated" if args.use_updated_target else ""

        # Create normal version (full y-axis range [0, 1])
        combined_plot_path = combined_output_dir / f"all_athletes_combined{suffix}.png"
        create_combined_athletes_plot(
            all_resemblance_data=all_resemblance_data,
            segments_data=segments_data,
            output_path=combined_plot_path,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            zoom=False,
        )

        # Create zoomed version (auto-scaled y-axis)
        combined_plot_zoom_path = combined_output_dir / f"all_athletes_combined{suffix}_zoom.png"
        create_combined_athletes_plot(
            all_resemblance_data=all_resemblance_data,
            segments_data=segments_data,
            output_path=combined_plot_zoom_path,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            zoom=True,
        )

        # Get athlete colors (need to use same color mapping as in combined plots)
        import hashlib
        import colorsys

        def get_distinct_colors(names: List[str]) -> Dict[str, str]:
            """Generate visually distinct colors for athletes"""
            sorted_names = sorted(names)
            n = len(sorted_names)

            if n <= 12:
                distinct_palette = [
                    "#F3C300",
                    "#875692",
                    "#F38400",
                    "#A1CAF1",
                    "#BE0032",
                    "#C2B280",
                    "#008856",
                    "#E68FAC",
                    "#0067A5",
                    "#F99379",
                    "#604E97",
                    "#F6A600",
                    "#B3446C",
                    "#DCD300",
                    "#882D17",
                    "#8DB600",
                    "#654522",
                    "#E25822",
                    "#2B3D26",
                ]

                color_mapping = {}
                used_indices = set()
                for name in sorted_names:
                    hash_val = int(hashlib.md5(name.encode()).hexdigest(), 16)
                    for attempt in range(len(distinct_palette)):
                        color_idx = (hash_val + attempt) % len(distinct_palette)
                        if color_idx not in used_indices:
                            color_mapping[name] = distinct_palette[color_idx]
                            used_indices.add(color_idx)
                            break
                return color_mapping
            else:
                colors = {}
                for i, name in enumerate(sorted_names):
                    hue = i / n
                    rgb = colorsys.hsv_to_rgb(hue, 0.8, 0.9)
                    colors[name] = f"#{int(rgb[0]*255):02x}{int(rgb[1]*255):02x}{int(rgb[2]*255):02x}"
                return colors

        athlete_names = sorted(all_resemblance_data.keys())
        athlete_colors = get_distinct_colors(athlete_names)

        # Assign highlights to athletes for both ArcFace and TransFace
        print("\n  Assigning highlight segments to athletes...")
        assignments_arcface = assign_highlights_to_athletes(
            all_resemblance_data=all_resemblance_data,
            segments_data=segments_data,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            model="arcface",
        )

        assignments_transface = assign_highlights_to_athletes(
            all_resemblance_data=all_resemblance_data,
            segments_data=segments_data,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            model="transface",
        )

        # Create assigned highlights visualization
        assigned_plot_path = combined_output_dir / f"highlights_assigned{suffix}.png"
        create_assigned_highlights_plot(
            all_resemblance_data=all_resemblance_data,
            segments_data=segments_data,
            assignments_arcface=assignments_arcface,
            assignments_transface=assignments_transface,
            athlete_colors=athlete_colors,
            output_path=assigned_plot_path,
            threshold=args.threshold,
            video_name_arg=args.video,
        )

        # Assign highlights using instant assignment method (with temporal expansion)
        print("\n  Assigning highlight segments to athletes (instant method with temporal expansion)...")
        assignments_arcface_instant = assign_highlights_to_athletes_instant(
            all_resemblance_data=all_resemblance_data,
            segments_data=segments_data,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            model="arcface",
            temporal_expansion=3.0,  # Expand 3 seconds on each side
        )

        assignments_transface_instant = assign_highlights_to_athletes_instant(
            all_resemblance_data=all_resemblance_data,
            segments_data=segments_data,
            threshold=args.threshold,
            smooth_window=args.smooth_window,
            model="transface",
            temporal_expansion=3.0,  # Expand 3 seconds on each side
        )

        # Create instant assigned highlights visualization
        assigned_instant_plot_path = combined_output_dir / f"highlights_instant_assigned{suffix}.png"
        create_assigned_highlights_plot(
            all_resemblance_data=all_resemblance_data,
            segments_data=segments_data,
            assignments_arcface=assignments_arcface_instant,
            assignments_transface=assignments_transface_instant,
            athlete_colors=athlete_colors,
            output_path=assigned_instant_plot_path,
            threshold=args.threshold,
            video_name_arg=args.video,
        )

        print(f"\nCombined plots saved to:")
        print(f"  - {combined_plot_path.relative_to(output_dir)}")
        print(f"  - {combined_plot_zoom_path.relative_to(output_dir)}")
        print(f"  - {assigned_plot_path.relative_to(output_dir)}")
        print(f"  - {assigned_instant_plot_path.relative_to(output_dir)}")

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

    parser.add_argument("--mode", type=str, choices=["resemblance_curves", "final_with_highlights"], required=True, help="Visualization mode")

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
    parser.add_argument(
        "--face-recognition-models",
        nargs="+",
        choices=["arcface", "transface"],
        default=None,
        help="[final_with_highlights] Face recognition models to use for clip selection. Options: arcface, transface. Default: both models (arcface transface).",
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
