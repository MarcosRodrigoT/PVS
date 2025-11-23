#!/usr/bin/env python3
"""
Comprehensive timing tracking and analysis for the PVS pipeline.

This module tracks fine-grained timing data and normalizes it by natural units of work,
similar to the SportCLIP-official project approach.

Normalization strategy:
- Video Summarization: per video (one-time cost)
- Face Detection: per frame processed
- Face Embedding (ArcFace/TransFace): per face crop
- Target Finding: per target × per frame
- Resemblance Computation: per target × per face
- Visualization: per target (one-time cost per target)
"""

import time
import json
import psutil
import os
import argparse
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager
from typing import Dict, List, Optional
from collections import defaultdict, OrderedDict
from pathlib import Path


# =============================================================================
# TIMING TRACKER (for data collection during pipeline execution)
# =============================================================================


class TimingTracker:
    """Track timing and resource usage for pipeline components."""

    def __init__(self, log_file: str = "timing_log_aggregate.json"):
        self.log_file = log_file
        self.timings = []
        self.current_context = []
        self.process = psutil.Process(os.getpid())

    @contextmanager
    def track(self, component_name: str, metadata: Optional[Dict] = None):
        """Context manager to track timing and resources for a component."""
        # Record start state
        start_time = time.time()
        start_cpu = self.process.cpu_percent()
        start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

        # Small delay to get accurate CPU reading
        time.sleep(0.01)

        try:
            yield
        finally:
            # Record end state
            end_time = time.time()
            end_cpu = self.process.cpu_percent()
            end_memory = self.process.memory_info().rss / (1024 * 1024)  # MB

            duration = end_time - start_time
            avg_cpu = (start_cpu + end_cpu) / 2
            memory_delta = end_memory - start_memory

            # Store timing data
            timing_entry = {
                "component": component_name,
                "duration_seconds": round(duration, 4),
                "cpu_percent": round(avg_cpu, 2),
                "memory_mb": round(end_memory, 2),
                "memory_delta_mb": round(memory_delta, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }

            # Add metadata if provided
            if metadata:
                timing_entry.update(metadata)

            self.timings.append(timing_entry)

    def save(self, output_dir: str = None):
        """Save timing data to both per-run and aggregate log files."""
        if not self.timings:
            return

        # Determine output directory
        if output_dir is None:
            output_dir = os.path.dirname(self.log_file) or "results"

        os.makedirs(output_dir, exist_ok=True)

        # Save to aggregate log (all runs combined)
        aggregate_log = os.path.join(output_dir, "timing_log_aggregate.json")
        existing_data = []
        if os.path.exists(aggregate_log):
            try:
                with open(aggregate_log, "r") as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []

        # Append new timings
        existing_data.extend(self.timings)

        # Save aggregate log
        with open(aggregate_log, "w") as f:
            json.dump(existing_data, f, indent=2)

        # Also save per-run log with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        video_name = self.timings[0].get("video_name", "unknown") if self.timings else "unknown"
        per_run_log = os.path.join(output_dir, f"timing_log_{video_name}_{timestamp}.json")
        with open(per_run_log, "w") as f:
            json.dump(self.timings, f, indent=2)

        print(f"✓ Timing data saved to: {aggregate_log}")
        print(f"  Per-run log: {per_run_log}")

        # Clear current timings
        self.timings = []

        return aggregate_log

    def print_summary(self, title: str = "Timing Summary"):
        """Print a summary of timings for the current run."""
        if not self.timings:
            return

        print(f"\n{'='*80}")
        print(f"{title:^80}")
        print(f"{'='*80}")

        # Group by component
        component_times = defaultdict(list)
        for entry in self.timings:
            component_times[entry["component"]].append(entry["duration_seconds"])

        # Print summary table
        print(f"\n{'Component':<50} {'Time (s)':>12} {'Avg (s)':>12}")
        print(f"{'-'*80}")

        total_time = 0
        for component, times in sorted(component_times.items()):
            total = sum(times)
            avg = total / len(times)
            count = len(times)
            total_time += total

            if count > 1:
                print(f"{component:<50} {total:>12.4f} {avg:>12.4f} (n={count})")
            else:
                print(f"{component:<50} {total:>12.4f} {'-':>12}")

        print(f"{'-'*80}")
        print(f"{'TOTAL':<50} {total_time:>12.4f}")
        print(f"{'='*80}\n")


# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================

# Component grouping and normalization units
COMPONENT_CONFIG = OrderedDict(
    [
        (
            "Video Summarization",
            {
                "components": ["Video Summarization"],
                "unit": "per_video",
                "unit_label": "s/video",
            },
        ),
        (
            "Face Detection",
            {
                "components": ["Face Detection"],
                "unit": "per_frame",
                "unit_label": "ms/frame",
            },
        ),
        (
            "Face Embedding (ArcFace)",
            {
                "components": ["Face Embedding (ArcFace)"],
                "unit": "per_face",
                "unit_label": "ms/face",
            },
        ),
        (
            "Face Embedding (TransFace)",
            {
                "components": ["Face Embedding (TransFace)"],
                "unit": "per_face",
                "unit_label": "ms/face",
            },
        ),
        (
            "Target Finding",
            {
                "components": ["Target Finding"],
                "unit": "per_target_per_frame",
                "unit_label": "μs/(target×frame)",
            },
        ),
        (
            "Target Embedding (Original)",
            {
                "components": ["Target Embedding (Original)"],
                "unit": "per_target",
                "unit_label": "ms/target",
            },
        ),
        (
            "Target Embedding (Updated)",
            {
                "components": ["Target Embedding (Updated)"],
                "unit": "per_target",
                "unit_label": "ms/target",
            },
        ),
        (
            "Resemblance Computation",
            {
                "components": ["Resemblance Computation"],
                "unit": "per_target_per_face",
                "unit_label": "μs/(target×face)",
            },
        ),
        (
            "Visualization",
            {
                "components": ["Visualization", "Copy Highlight Clips"],
                "unit": "per_target",
                "unit_label": "ms/target",
            },
        ),
    ]
)


# =============================================================================
# DATA LOADING & PROCESSING
# =============================================================================


def load_timing_log(log_file: str) -> pd.DataFrame:
    """Load timing log JSON and convert to DataFrame."""
    with open(log_file, "r") as f:
        data = json.load(f)
    return pd.DataFrame(data)


def normalize_timing_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize timing data by natural units of work.

    Similar to SportCLIP approach:
    - per_video: total time in seconds
    - per_frame: time / num_frames * 1000 (ms/frame)
    - per_face: time / num_faces * 1000 (ms/face)
    - per_target: time / num_targets * 1000 (ms/target)
    - per_target_per_frame: time / (num_targets * num_frames) * 1e6 (μs/(target×frame))
    - per_target_per_face: time / (num_targets * num_faces) * 1e6 (μs/(target×face))
    """
    normalized_records = []

    for _, row in df.iterrows():
        component = row["component"]
        duration = row["duration_seconds"]

        # Find the paper component this belongs to
        paper_component = None
        unit_type = None
        for pc, config in COMPONENT_CONFIG.items():
            if component in config["components"]:
                paper_component = pc
                unit_type = config["unit"]
                break

        if paper_component is None:
            continue  # Skip unknown components

        # Get normalization factors from metadata
        video_name = row.get("video_name", "unknown")
        num_frames = row.get("num_frames", 0)
        num_faces = row.get("num_faces", 0)
        num_targets = row.get("num_targets", 1)

        # Normalize based on unit type
        if unit_type == "per_video":
            normalized_time = duration  # seconds
        elif unit_type == "per_frame":
            normalized_time = (duration / num_frames * 1000) if num_frames > 0 else 0  # ms/frame
        elif unit_type == "per_face":
            normalized_time = (duration / num_faces * 1000) if num_faces > 0 else 0  # ms/face
        elif unit_type == "per_target":
            normalized_time = (duration / num_targets * 1000) if num_targets > 0 else 0  # ms/target
        elif unit_type == "per_target_per_frame":
            denom = num_targets * num_frames
            normalized_time = (duration / denom * 1e6) if denom > 0 else 0  # μs/(target×frame)
        elif unit_type == "per_target_per_face":
            denom = num_targets * num_faces
            normalized_time = (duration / denom * 1e6) if denom > 0 else 0  # μs/(target×face)
        else:
            normalized_time = duration

        normalized_records.append(
            {
                "paper_component": paper_component,
                "component": component,
                "video_name": video_name,
                "duration_seconds": duration,
                "normalized_time": normalized_time,
                "unit": unit_type,
                "unit_label": COMPONENT_CONFIG[paper_component]["unit_label"],
                "num_frames": num_frames,
                "num_faces": num_faces,
                "num_targets": num_targets,
            }
        )

    return pd.DataFrame(normalized_records)


def compute_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, std for each paper component."""
    stats = df.groupby(["paper_component", "unit_label"]).agg({"normalized_time": ["mean", "std", "count"]}).reset_index()

    # Flatten column names
    stats.columns = ["paper_component", "unit_label", "mean", "std", "count"]

    # Fill NaN std with 0
    stats["std"] = stats["std"].fillna(0)

    # Sort by order in COMPONENT_CONFIG
    component_order = {comp: idx for idx, comp in enumerate(COMPONENT_CONFIG.keys())}
    stats["order"] = stats["paper_component"].map(component_order)
    stats = stats.sort_values("order").drop(columns=["order"])

    return stats


# =============================================================================
# OUTPUT GENERATION
# =============================================================================


def generate_markdown_table(stats: pd.DataFrame, output_file: str):
    """Generate a markdown file with normalized timing table."""
    lines = []

    # Title
    lines.append("# Computational Cost Analysis - PVS Pipeline")
    lines.append("")
    lines.append("## Normalized Processing Times")
    lines.append("")
    lines.append("All times normalized by their natural unit of work for fair comparison.")
    lines.append("")

    # Main table
    lines.append("| Component | Unit of Work | Time (Mean ± Std) |")
    lines.append("|-----------|--------------|-------------------|")

    for _, row in stats.iterrows():
        component = row["paper_component"]
        unit = row["unit_label"]
        mean = row["mean"]
        std = row["std"]

        # Format based on magnitude
        if mean >= 1.0:
            time_str = f"{mean:.3f} ± {std:.3f}"
        else:
            time_str = f"{mean:.4f} ± {std:.4f}"

        lines.append(f"| {component} | {unit} | {time_str} |")

    lines.append("")
    lines.append(f"*Note: Based on {stats['count'].sum():,} timing measurements*")
    lines.append("")

    # Methodology notes
    lines.append("---")
    lines.append("")
    lines.append("## Methodology")
    lines.append("")
    lines.append("### Normalization Strategy")
    lines.append("")
    lines.append("Each component is normalized by its natural unit of work:")
    lines.append("")
    lines.append("- **Video Summarization**: per video (one-time cost)")
    lines.append("- **Face Detection**: per frame processed")
    lines.append("- **Face Embedding**: per face crop extracted")
    lines.append("- **Target Finding**: per target per frame (search cost)")
    lines.append("- **Resemblance Computation**: per target per face (comparison cost)")
    lines.append("- **Visualization**: per target (one-time cost per target)")
    lines.append("")
    lines.append("### Data Collection")
    lines.append("")
    lines.append("- Timing measurements collected using Python's `time.time()` for wall-clock time")
    lines.append("- Each component timed individually using context managers")
    lines.append("- Measurements include CPU usage and memory consumption")
    lines.append("- Statistics aggregated across multiple videos and runs")
    lines.append("")

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"✓ Markdown table written to: {output_file}")


def create_bar_chart(stats: pd.DataFrame, output_file: str):
    """Create a bar chart showing normalized times with error bars."""
    # Set up style
    sns.set_style("whitegrid")
    plt.rcParams["figure.facecolor"] = "white"

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 7))

    # Prepare data
    x = range(len(stats))
    colors = sns.color_palette("husl", n_colors=len(stats))

    # Create bar chart
    bars = ax.bar(
        x,
        stats["mean"],
        yerr=stats["std"],
        color=colors,
        edgecolor="black",
        linewidth=1.5,
        capsize=5,
        alpha=0.8,
    )

    # Styling
    ax.set_xticks(x)
    labels = [f"{row['paper_component']}\n({row['unit_label']})" for _, row in stats.iterrows()]
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Normalized Time", fontsize=12, fontweight="bold")
    ax.set_title("Normalized Processing Time by Component\n(with standard deviation)", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, (bar, row) in enumerate(zip(bars, stats.iterrows())):
        _, data = row
        height = bar.get_height()
        if data["mean"] >= 1.0:
            label = f'{data["mean"]:.2f}'
        else:
            label = f'{data["mean"]:.3f}'
        ax.text(bar.get_x() + bar.get_width() / 2.0, height, label, ha="center", va="bottom", fontsize=8, fontweight="bold")

    plt.tight_layout()

    # Save
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Bar chart saved to: {output_file}")
    plt.close()


def create_comparison_table(stats: pd.DataFrame, output_file: str, fps: int = 30, num_targets: int = 1, avg_faces_per_frame: int = 5):
    """
    Create a comparison table showing total time for different video lengths.
    Similar to Table in SportCLIP paper.
    """
    lines = []

    lines.append("# Processing Time Estimates")
    lines.append("")
    lines.append(f"## Configuration")
    lines.append("")
    lines.append(f"- Frame rate: {fps} fps")
    lines.append(f"- Number of targets: {num_targets}")
    lines.append(f"- Average faces per frame: {avg_faces_per_frame}")
    lines.append("")

    # Calculate for different video durations
    durations = [1, 5, 10, 60]  # minutes

    lines.append("| Video Duration | Total Frames | Total Faces | Est. Total Time |")
    lines.append("|----------------|--------------|-------------|-----------------|")

    for duration_min in durations:
        total_frames = duration_min * 60 * fps
        total_faces = total_frames * avg_faces_per_frame

        total_time = 0.0

        for _, row in stats.iterrows():
            unit = row["unit_label"]
            mean_time = row["mean"]  # already in correct unit

            if "s/video" in unit:
                total_time += mean_time  # seconds, one-time
            elif "ms/frame" in unit:
                total_time += (mean_time / 1000) * total_frames  # convert ms to s
            elif "ms/face" in unit:
                total_time += (mean_time / 1000) * total_faces
            elif "ms/target" in unit:
                total_time += (mean_time / 1000) * num_targets
            elif "μs/(target×frame)" in unit:
                total_time += (mean_time / 1e6) * total_frames * num_targets
            elif "μs/(target×face)" in unit:
                total_time += (mean_time / 1e6) * total_faces * num_targets

        # Format output
        if duration_min == 60:
            duration_str = "1 hour"
        else:
            duration_str = f"{duration_min} min"

        if total_time < 60:
            time_str = f"{total_time:.1f} s"
        else:
            time_str = f"{total_time / 60:.1f} min"

        lines.append(f"| {duration_str} | {total_frames:,} | {total_faces:,} | {time_str} |")

    lines.append("")
    lines.append("*Note: These are estimates based on average processing times. Actual times may vary.*")
    lines.append("")

    # Write to file
    with open(output_file, "w") as f:
        f.write("\n".join(lines))

    print(f"✓ Comparison table written to: {output_file}")


# =============================================================================
# MAIN FUNCTION
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate timing analysis for PVS pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--timing-log", type=str, default="results/timing_log_aggregate.json", help="Path to aggregate timing log (default: results/timing_log_aggregate.json)")
    parser.add_argument("--output-dir", type=str, default="results/timing_analysis", help="Output directory for results (default: results/timing_analysis)")

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading timing data...")
    df = load_timing_log(args.timing_log)
    print(f"✓ Loaded {len(df)} timing entries")

    # Normalize data
    print("\nNormalizing timing data...")
    normalized_df = normalize_timing_data(df)

    if normalized_df.empty:
        print("⚠ No valid timing data found. Make sure components are named correctly.")
        return

    print(f"✓ Normalized {len(normalized_df)} entries")

    # Compute statistics
    print("\nComputing statistics...")
    stats = compute_statistics(normalized_df)

    print("\n" + "=" * 90)
    print("NORMALIZED TIMING STATISTICS")
    print("=" * 90)
    print(stats[["paper_component", "unit_label", "mean", "std", "count"]].to_string(index=False))
    print("=" * 90)

    # Generate outputs
    print("\nGenerating outputs...")

    # Markdown table
    markdown_file = output_dir / "timing_summary.md"
    generate_markdown_table(stats, str(markdown_file))

    # Bar chart
    bar_chart_file = output_dir / "timing_bar_chart.png"
    create_bar_chart(stats, str(bar_chart_file))

    # Comparison table
    comparison_file = output_dir / "timing_comparison.md"
    create_comparison_table(stats, str(comparison_file))

    # Summary
    print("\n" + "=" * 90)
    print("OUTPUTS GENERATED")
    print("=" * 90)
    print(f"1. Summary table: {markdown_file}")
    print(f"2. Bar chart: {bar_chart_file}")
    print(f"3. Comparison table: {comparison_file}")
    print("=" * 90)

    print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
