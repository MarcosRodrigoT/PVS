#!/usr/bin/env python3
"""
Temporary script to generate resemblance curve plots without highlight segments overlay
and with separate legends for each subplot.

This script will be deleted after use.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List
import hashlib
import colorsys


def smooth_scores(scores: np.ndarray, window: int = 5) -> np.ndarray:
    """Apply moving average smoothing to scores"""
    if len(scores) < window:
        return scores

    kernel = np.ones(window) / window
    smoothed = np.convolve(scores, kernel, mode='same')

    # Fix edges
    for i in range(window // 2):
        smoothed[i] = np.mean(scores[:i + window // 2 + 1])
        smoothed[-(i + 1)] = np.mean(scores[-(i + window // 2 + 1):])

    return smoothed


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


def load_resemblance_data(video_name: str, resemblance_dir: Path) -> Dict[str, Dict]:
    """Load resemblance data for all athletes"""
    all_resemblance_data = {}

    video_results_dir = resemblance_dir / video_name
    if not video_results_dir.exists():
        print(f"Error: Resemblance results directory not found: {video_results_dir}")
        return {}

    # Find all resemblance JSON files in subdirectories (look for *_updated.json)
    json_files = list(video_results_dir.glob("*/*_updated.json"))

    if not json_files:
        print(f"Error: No resemblance JSON files found in {video_results_dir}")
        return {}

    for json_file in json_files:
        target_name = json_file.stem.replace("_updated", "")

        with open(json_file, "r") as f:
            data = json.load(f)
            all_resemblance_data[target_name] = data

    return all_resemblance_data


def create_clean_resemblance_plot(
    all_resemblance_data: Dict[str, Dict],
    output_path: Path,
    threshold: float = 0.6,
    smooth_window: int = 5,
    zoom: bool = False,
) -> None:
    """
    Create resemblance curve plot without highlight segments overlay
    and with one legend per subplot
    """
    if not all_resemblance_data:
        print("Warning: No resemblance data to plot")
        return

    # Get video name from first target
    first_target = next(iter(all_resemblance_data.values()))
    video_name = first_target["video"]

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

        athlete_data[athlete_name] = {
            "arcface_smooth": arcface_smooth,
            "transface_smooth": transface_smooth
        }

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

    # Get timestamps from first target
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

    # Plot ArcFace curves for all athletes
    for athlete_name in athlete_names:
        arcface_smooth = athlete_data[athlete_name]["arcface_smooth"]
        color = athlete_colors[athlete_name]
        ax1.plot(timestamps, arcface_smooth, color=color, linewidth=2.0, alpha=0.8, label=athlete_name)

    # Add threshold line to ArcFace plot
    ax1.axhline(y=threshold, color="#F18F01", linestyle="--", linewidth=2, alpha=0.7, label=f"Threshold ({threshold})")

    ax1.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    title_suffix = " (Zoomed)" if zoom else ""
    ax1.set_title(f"ArcFace Resemblance Curves - All Athletes in {video_name}{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    ax1.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax1.set_ylim(arcface_ylim)
    ax1.set_xlim([timestamps[0], timestamps[-1]])
    ax1.set_xticks(tick_times)
    ax1.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Add legend for ArcFace plot
    ax1.legend(loc="upper right", fontsize=10, framealpha=0.9, ncol=1)

    # Plot TransFace curves for all athletes
    for athlete_name in athlete_names:
        transface_smooth = athlete_data[athlete_name]["transface_smooth"]
        color = athlete_colors[athlete_name]
        ax2.plot(timestamps, transface_smooth, color=color, linewidth=2.0, alpha=0.8, label=athlete_name)

    # Add threshold line to TransFace plot
    ax2.axhline(y=threshold, color="#F18F01", linestyle="--", linewidth=2, alpha=0.7, label=f"Threshold ({threshold})")

    ax2.set_xlabel("Video Time (MM:SS)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    ax2.set_title(f"TransFace Resemblance Curves - All Athletes in {video_name}{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    ax2.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax2.set_ylim(transface_ylim)
    ax2.set_xlim([timestamps[0], timestamps[-1]])
    ax2.set_xticks(tick_times)
    ax2.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Add legend for TransFace plot
    ax2.legend(loc="upper right", fontsize=10, framealpha=0.9, ncol=1)

    # Add stats text at bottom
    stats_text = f"Athletes: {len(athlete_names)} | Threshold: {threshold}"
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Adjust layout
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight", facecolor="white")
    plt.close()

    zoom_suffix = " (zoomed)" if zoom else ""
    print(f"✓ Created clean resemblance plot{zoom_suffix}: {output_path.name}")


def main():
    """Main function"""
    print("=" * 80)
    print("GENERATING CLEAN RESEMBLANCE PLOTS (WITHOUT HIGHLIGHT OVERLAYS)")
    print("=" * 80)

    # Configuration
    video_name = "highjump_video1"
    resemblance_dir = Path("resemblance_results")
    threshold = 0.6
    smooth_window = 5

    # Load resemblance data
    print(f"\nLoading resemblance data for {video_name}...")
    all_resemblance_data = load_resemblance_data(video_name, resemblance_dir)

    if not all_resemblance_data:
        print("Error: Could not load resemblance data")
        return

    print(f"Found {len(all_resemblance_data)} athletes: {', '.join(sorted(all_resemblance_data.keys()))}")

    # Create normal version (full y-axis range [0, 1])
    print("\nCreating normal plot...")
    output_normal = Path("all_athletes_combined_clean.png")
    create_clean_resemblance_plot(
        all_resemblance_data=all_resemblance_data,
        output_path=output_normal,
        threshold=threshold,
        smooth_window=smooth_window,
        zoom=False,
    )

    # Create zoomed version (auto-scaled y-axis)
    print("Creating zoomed plot...")
    output_zoom = Path("all_athletes_combined_clean_zoom.png")
    create_clean_resemblance_plot(
        all_resemblance_data=all_resemblance_data,
        output_path=output_zoom,
        threshold=threshold,
        smooth_window=smooth_window,
        zoom=True,
    )

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nOutput files:")
    print(f"  - {output_normal}")
    print(f"  - {output_zoom}")
    print("\nThese files are in the project root directory.")
    print("You can now copy them to your thesis document and delete this script.")


if __name__ == "__main__":
    main()
