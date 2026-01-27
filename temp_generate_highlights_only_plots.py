#!/usr/bin/env python3
"""
Temporary script to generate plots showing only highlight segments without resemblance curves.

This script will be deleted after use.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path


def format_time(seconds):
    """Format seconds to MM:SS"""
    minutes = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{minutes:02d}:{secs:02d}"


def load_segments_data(segments_file: Path) -> dict:
    """Load segments data from JSON file"""
    with open(segments_file, "r") as f:
        return json.load(f)


def get_video_duration(resemblance_file: Path) -> float:
    """Get video duration from resemblance data"""
    with open(resemblance_file, "r") as f:
        data = json.load(f)
        return data["timestamps_seconds"][-1]


def create_highlights_only_plot(
    segments_data: dict,
    video_duration: float,
    video_name: str,
    model_name: str,
    output_path: Path,
    zoom: bool = False,
) -> None:
    """
    Create plot showing only highlight segments without resemblance curves

    Args:
        segments_data: Video segments metadata
        video_duration: Total duration of video in seconds
        video_name: Name of the video
        model_name: Name of the model (e.g., "SportCLIP", "QD-DETR")
        output_path: Path to save plot
        zoom: If True, auto-scale y-axis to data range; if False, use [0, 1]
    """
    # Create figure with two subplots (keeping same structure as original)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), dpi=150)
    fig.patch.set_facecolor("white")

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

    # Extract segments from nested structure
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    # Determine y-axis limits based on zoom
    if zoom:
        # For zoomed version, use a smaller range to make segments more visible
        y_min, y_max = 0.4, 1.0
    else:
        # For normal version, use full range
        y_min, y_max = 0.0, 1.0

    # Helper function to add highlight segments to plot
    def add_highlight_segments(ax, subplot_title):
        """Add highlight segment markers to the plot"""
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

                # Add segment number label at the top
                ax.text(
                    start_time + width / 2,
                    0.95,
                    f"H{i}",
                    transform=ax.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=9,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.7, edgecolor="darkred"),
                    color="white",
                )

    # Configure ArcFace subplot
    add_highlight_segments(ax1, "ArcFace")

    ax1.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    title_suffix = " (Zoomed)" if zoom else ""
    ax1.set_title(f"ArcFace - {model_name} Highlight Segments in {video_name}{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    ax1.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax1.set_ylim([y_min, y_max])
    ax1.set_xlim([0, video_duration])
    ax1.set_xticks(tick_times)
    ax1.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Add legend for highlight segments
    highlight_patch = patches.Patch(facecolor="red", edgecolor="red", alpha=0.15, linewidth=2)
    ax1.legend(handles=[highlight_patch], labels=["Highlight Segment"], loc="upper right", fontsize=10, framealpha=0.9)

    # Configure TransFace subplot
    add_highlight_segments(ax2, "TransFace")

    ax2.set_xlabel("Video Time (MM:SS)", fontsize=14, fontweight="bold")
    ax2.set_ylabel("Resemblance Score", fontsize=14, fontweight="bold")
    ax2.set_title(f"TransFace - {model_name} Highlight Segments in {video_name}{title_suffix}", fontsize=16, fontweight="bold", pad=20)
    ax2.grid(True, alpha=0.3, linestyle=":", linewidth=1)
    ax2.set_ylim([y_min, y_max])
    ax2.set_xlim([0, video_duration])
    ax2.set_xticks(tick_times)
    ax2.set_xticklabels([format_time(t) for t in tick_times], rotation=45, ha="right")

    # Add legend for highlight segments
    highlight_patch_ax2 = patches.Patch(facecolor="red", edgecolor="red", alpha=0.15, linewidth=2)
    ax2.legend(handles=[highlight_patch_ax2], labels=["Highlight Segment"], loc="upper right", fontsize=10, framealpha=0.9)

    # Add stats text at bottom
    stats_text = f"Model: {model_name} | Highlight Segments: {len(segments)}"
    fig.text(0.5, 0.01, stats_text, ha="center", fontsize=11, bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Adjust layout
    plt.tight_layout(rect=[0, 0.04, 1, 1])

    # Save plot
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), bbox_inches="tight", facecolor="white")
    plt.close()

    zoom_suffix = " (zoomed)" if zoom else ""
    print(f"✓ Created highlights-only plot{zoom_suffix}: {output_path.name}")


def main():
    """Main function"""
    print("=" * 80)
    print("GENERATING HIGHLIGHTS-ONLY PLOTS (NO RESEMBLANCE CURVES)")
    print("=" * 80)

    # Configuration
    video_name = "highjump_video1"
    models = ["SportCLIP", "QD-DETR"]

    # Get video duration from any resemblance file
    resemblance_file = Path("resemblance_results/highjump_video1/baba/baba_updated.json")

    if not resemblance_file.exists():
        print(f"Error: Resemblance file not found: {resemblance_file}")
        return

    video_duration = get_video_duration(resemblance_file)
    print(f"\nVideo: {video_name}")
    print(f"Video duration: {format_time(video_duration)} ({video_duration:.2f} seconds)")

    output_files = []

    # Process each model
    for model in models:
        print("\n" + "=" * 80)
        print(f"Processing {model} model")
        print("=" * 80)

        segments_file = Path(f"video_summaries/{model}/highjump_video1/segments.json")

        # Load data
        print(f"\nLoading segments data for {model}...")
        if not segments_file.exists():
            print(f"Error: Segments file not found: {segments_file}")
            continue

        segments_data = load_segments_data(segments_file)

        # Extract segments from nested structure
        segments_obj = segments_data.get("segments", [])
        if isinstance(segments_obj, dict):
            segments = segments_obj.get("segments", [])
        else:
            segments = segments_obj

        print(f"Found {len(segments)} highlight segments")

        # Create normal version (full y-axis range [0, 1])
        print(f"\nCreating normal plot for {model}...")
        output_normal = Path(f"all_athletes_highlights_only_{model.lower()}.png")
        create_highlights_only_plot(
            segments_data=segments_data,
            video_duration=video_duration,
            video_name=video_name,
            model_name=model,
            output_path=output_normal,
            zoom=False,
        )
        output_files.append(output_normal)

        # Create zoomed version (reduced y-axis range)
        print(f"Creating zoomed plot for {model}...")
        output_zoom = Path(f"all_athletes_highlights_only_{model.lower()}_zoom.png")
        create_highlights_only_plot(
            segments_data=segments_data,
            video_duration=video_duration,
            video_name=video_name,
            model_name=model,
            output_path=output_zoom,
            zoom=True,
        )
        output_files.append(output_zoom)

    print("\n" + "=" * 80)
    print("DONE!")
    print("=" * 80)
    print(f"\nOutput files:")
    for output_file in output_files:
        print(f"  - {output_file}")
    print("\nThese files are in the project root directory.")
    print("You can now copy them to your thesis document and delete this script.")


if __name__ == "__main__":
    main()
