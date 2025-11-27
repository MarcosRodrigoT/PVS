#!/usr/bin/env python3
"""
Face Recognition Evaluation Script

This script evaluates face recognition models (ArcFace and TransFace) independently
from video summarization by using ground truth highlight segments.

It produces similar visualization plots as the main pipeline but uses ground truth
highlights instead of predicted highlights, allowing fair comparison of face recognition
models without video summarization affecting the results.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict


# Kelly's 22 colors for maximum contrast
KELLY_COLORS = [
    '#F2F3F4', '#222222', '#F3C300', '#875692', '#F38400', '#A1CAF1',
    '#BE0032', '#C2B280', '#848482', '#008856', '#E68FAC', '#0067A5',
    '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17',
    '#8DB600', '#654522', '#E25822', '#2B3D26'
]


def load_ground_truth_highlights(csv_path: str) -> List[Dict]:
    """
    Load ground truth highlight segments with identity information

    Args:
        csv_path: Path to ground truth CSV file with identity column

    Returns:
        List of dictionaries with segment information
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ground truth file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    segments = []
    for idx, row in df.iterrows():
        if row["Event type"] == "Highlight":
            segment = {
                "segment_id": idx,
                "start_frame": int(row["First frame"]),
                "end_frame": int(row["Last frame"]),
                "identity": row["Identity"] if "Identity" in row else None
            }
            segments.append(segment)

    return segments


def load_resemblance_data(resemblance_results_dir: str,
                           video_name: str,
                           target_names: List[str],
                           use_updated_target: bool = False) -> Dict[str, Dict]:
    """
    Load resemblance data for all targets

    Args:
        resemblance_results_dir: Directory containing resemblance results
        video_name: Video name
        target_names: List of target names to load
        use_updated_target: Whether to use updated target results

    Returns:
        Dictionary mapping target names to resemblance data
    """
    suffix = "_updated" if use_updated_target else ""
    all_resemblance_data = {}

    video_results_dir = Path(resemblance_results_dir) / video_name

    for target_name in target_names:
        target_dir = video_results_dir / target_name
        json_file = target_dir / f"{target_name}{suffix}.json"

        if not json_file.exists():
            print(f"Warning: Resemblance data not found for {target_name}: {json_file}")
            continue

        with open(json_file, 'r') as f:
            data = json.load(f)
            all_resemblance_data[target_name] = data

    return all_resemblance_data


def smooth_scores(scores: np.ndarray, window_size: int) -> np.ndarray:
    """Apply moving average smoothing"""
    if window_size <= 1:
        return scores

    # Use convolution for moving average
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores, kernel, mode='same')

    # Handle edges
    for i in range(min(window_size // 2, len(scores))):
        smoothed[i] = np.mean(scores[:i + window_size // 2 + 1])
        smoothed[-(i + 1)] = np.mean(scores[-(i + window_size // 2 + 1):])

    return smoothed


def assign_highlights_to_athletes_with_gt_segments(
    all_resemblance_data: Dict[str, Dict],
    gt_segments: List[Dict],
    fps: float,
    threshold: float,
    smooth_window: int,
    model: str = "arcface",
    method: str = "sequential"
) -> Dict[int, Optional[str]]:
    """
    Assign each ground truth highlight segment to an athlete based on resemblance curves

    Args:
        all_resemblance_data: Dictionary mapping target_name to resemblance data
        gt_segments: List of ground truth segment dictionaries
        fps: Video frames per second
        threshold: Similarity threshold
        smooth_window: Smoothing window size
        model: Which model to use for assignment ("arcface" or "transface")
        method: Assignment method ("sequential" or "instant")

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

    # Assignment dictionary: segment_index -> athlete_name
    assignments = {}

    if method == "sequential":
        # Sequential assignment (same as assign_highlights_to_athletes)
        current_athlete = None

        for seg_idx, segment in enumerate(gt_segments):
            start_frame = segment["start_frame"]
            start_time = start_frame / fps

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

    elif method == "instant":
        # Instant assignment with temporal expansion
        for seg_idx, segment in enumerate(gt_segments):
            start_frame = segment["start_frame"]
            end_frame = segment["end_frame"]
            start_time = start_frame / fps
            end_time = end_frame / fps

            # Extend search window by 3 seconds on each side
            search_start = max(0, start_time - 3)
            search_end = end_time + 3

            # Find timestamp indices for this window
            start_idx = np.searchsorted(timestamps, search_start)
            end_idx = np.searchsorted(timestamps, search_end)

            # Check which athletes exceed threshold in this window
            candidate_athletes = []
            for athlete_name in athlete_names:
                scores = athlete_scores[athlete_name]
                window_scores = scores[start_idx:end_idx] if end_idx <= len(scores) else scores[start_idx:]

                if len(window_scores) > 0 and np.any(window_scores >= threshold):
                    # Calculate average score above threshold
                    above_threshold = window_scores[window_scores >= threshold]
                    avg_score = np.mean(above_threshold)
                    candidate_athletes.append((athlete_name, avg_score))

            # Assign to athlete with highest average score
            if candidate_athletes:
                candidate_athletes.sort(key=lambda x: x[1], reverse=True)
                assignments[seg_idx] = candidate_athletes[0][0]
            else:
                assignments[seg_idx] = None

    else:
        raise ValueError(f"Unknown method: {method}. Must be 'sequential' or 'instant'")

    return assignments


def evaluate_assignments(
    assignments: Dict[int, Optional[str]],
    gt_segments: List[Dict],
    iou_threshold: float = 0.3
) -> Dict:
    """
    Evaluate assignment accuracy against ground truth identities

    Args:
        assignments: Dictionary mapping segment index to assigned athlete name
        gt_segments: List of ground truth segments with identity information
        iou_threshold: IoU threshold for matching (not used here since we use GT segments)

    Returns:
        Dictionary with evaluation metrics per athlete
    """
    # Count predictions and ground truth per athlete
    athlete_stats = defaultdict(lambda: {"gt": 0, "pred": 0, "correct": 0})

    for seg_idx, segment in enumerate(gt_segments):
        gt_identity = segment.get("identity")
        pred_identity = assignments.get(seg_idx)

        if gt_identity:
            gt_identity_lower = gt_identity.lower()
            athlete_stats[gt_identity_lower]["gt"] += 1

            if pred_identity:
                pred_identity_lower = pred_identity.lower()
                athlete_stats[pred_identity_lower]["pred"] += 1

                # Check if correct
                if gt_identity_lower == pred_identity_lower:
                    athlete_stats[gt_identity_lower]["correct"] += 1

    # Compute precision for each athlete
    results = {}
    for athlete, stats in athlete_stats.items():
        precision = stats["correct"] / stats["pred"] if stats["pred"] > 0 else 0
        recall = stats["correct"] / stats["gt"] if stats["gt"] > 0 else 0
        fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results[athlete] = {
            "gt": stats["gt"],
            "pred": stats["pred"],
            "correct": stats["correct"],
            "precision": precision,
            "recall": recall,
            "fscore": fscore
        }

    return results


def create_evaluation_plot(
    all_resemblance_data: Dict[str, Dict],
    gt_segments: List[Dict],
    assignments_arcface: Dict[int, Optional[str]],
    assignments_transface: Dict[int, Optional[str]],
    fps: float,
    video_duration: float,
    threshold: float,
    video_name: str,
    output_path: str,
    method_name: str = "sequential"
):
    """
    Create visualization plot showing assignments for both models

    Similar to highlights_assigned.png but with ground truth segments
    """
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))

    # Get all athlete names
    all_athletes = sorted(set(all_resemblance_data.keys()))

    # Assign colors to athletes
    athlete_colors = {}
    for idx, athlete in enumerate(all_athletes):
        athlete_colors[athlete] = KELLY_COLORS[idx % len(KELLY_COLORS)]

    # Color for unassigned segments
    unassigned_color = '#CCCCCC'

    # Plot for ArcFace
    ax = axes[0]
    ax.set_title(f'Ground Truth Highlight Segments Assigned to Athletes (ArcFace) - {video_name}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, video_duration)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Video Time (MM:SS)', fontsize=12)
    ax.set_yticks([])

    # Count segments per athlete for ArcFace
    arcface_counts = defaultdict(int)
    for seg_idx, segment in enumerate(gt_segments):
        assigned_athlete = assignments_arcface.get(seg_idx)
        if assigned_athlete:
            arcface_counts[assigned_athlete] += 1
        else:
            arcface_counts["Unassigned"] += 1

        # Draw segment
        start_time = segment["start_frame"] / fps
        end_time = segment["end_frame"] / fps
        color = athlete_colors.get(assigned_athlete, unassigned_color) if assigned_athlete else unassigned_color

        rect = mpatches.Rectangle((start_time, 0), end_time - start_time, 1,
                                   linewidth=1, edgecolor='black', facecolor=color, alpha=0.7,
                                   transform=ax.get_xaxis_transform())
        ax.add_patch(rect)

        # Add segment label in colored box above the bar
        width = end_time - start_time
        label_color = color if assigned_athlete else unassigned_color
        text_color = 'white' if assigned_athlete else 'black'
        ax.text(start_time + width / 2, 0.95, f'H{seg_idx+1}',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=label_color, alpha=0.8, edgecolor='black'),
                color=text_color)

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x//60):02d}:{int(x%60):02d}'))

    # Legend for ArcFace
    legend_elements = []
    for athlete in sorted(arcface_counts.keys()):
        if athlete == "Unassigned":
            color = unassigned_color
        else:
            color = athlete_colors.get(athlete, unassigned_color)
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black',
                                              label=f'{athlete} ({arcface_counts[athlete]})'))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # Plot for TransFace
    ax = axes[1]
    ax.set_title(f'Ground Truth Highlight Segments Assigned to Athletes (TransFace) - {video_name}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, video_duration)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Video Time (MM:SS)', fontsize=12)
    ax.set_yticks([])

    # Count segments per athlete for TransFace
    transface_counts = defaultdict(int)
    for seg_idx, segment in enumerate(gt_segments):
        assigned_athlete = assignments_transface.get(seg_idx)
        if assigned_athlete:
            transface_counts[assigned_athlete] += 1
        else:
            transface_counts["Unassigned"] += 1

        # Draw segment
        start_time = segment["start_frame"] / fps
        end_time = segment["end_frame"] / fps
        color = athlete_colors.get(assigned_athlete, unassigned_color) if assigned_athlete else unassigned_color

        rect = mpatches.Rectangle((start_time, 0), end_time - start_time, 1,
                                   linewidth=1, edgecolor='black', facecolor=color, alpha=0.7,
                                   transform=ax.get_xaxis_transform())
        ax.add_patch(rect)

        # Add segment label in colored box above the bar
        width = end_time - start_time
        label_color = color if assigned_athlete else unassigned_color
        text_color = 'white' if assigned_athlete else 'black'
        ax.text(start_time + width / 2, 0.95, f'H{seg_idx+1}',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=label_color, alpha=0.8, edgecolor='black'),
                color=text_color)

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x//60):02d}:{int(x%60):02d}'))

    # Legend for TransFace
    legend_elements = []
    for athlete in sorted(transface_counts.keys()):
        if athlete == "Unassigned":
            color = unassigned_color
        else:
            color = athlete_colors.get(athlete, unassigned_color)
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black',
                                              label=f'{athlete} ({transface_counts[athlete]})'))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # Plot for Ground Truth
    ax = axes[2]
    ax.set_title(f'Ground Truth - {video_name}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, video_duration)
    ax.set_ylim(0, 1)
    ax.set_xlabel('Video Time (MM:SS)', fontsize=12)
    ax.set_yticks([])

    # Count segments per athlete for ground truth
    gt_counts = defaultdict(int)
    for seg_idx, segment in enumerate(gt_segments):
        gt_identity = segment.get("identity")
        if gt_identity:
            gt_counts[gt_identity.lower()] += 1

        # Draw segment
        start_time = segment["start_frame"] / fps
        end_time = segment["end_frame"] / fps
        color = athlete_colors.get(gt_identity.lower(), unassigned_color) if gt_identity else unassigned_color

        rect = mpatches.Rectangle((start_time, 0), end_time - start_time, 1,
                                   linewidth=1, edgecolor='black', facecolor=color, alpha=0.7,
                                   transform=ax.get_xaxis_transform())
        ax.add_patch(rect)

        # Add segment label in colored box above the bar
        width = end_time - start_time
        label_color = color
        text_color = 'white' if gt_identity else 'black'
        ax.text(start_time + width / 2, 0.95, f'H{seg_idx+1}',
                transform=ax.get_xaxis_transform(),
                ha='center', va='top',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor=label_color, alpha=0.8, edgecolor='black'),
                color=text_color)

    # Format x-axis
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{int(x//60):02d}:{int(x%60):02d}'))

    # Legend for Ground Truth
    legend_elements = []
    for athlete in sorted(gt_counts.keys()):
        color = athlete_colors.get(athlete, unassigned_color)
        legend_elements.append(mpatches.Patch(facecolor=color, edgecolor='black',
                                              label=f'{athlete} ({gt_counts[athlete]})'))

    ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    # Evaluate assignments
    results_arcface = evaluate_assignments(assignments_arcface, gt_segments)
    results_transface = evaluate_assignments(assignments_transface, gt_segments)

    # Add evaluation table with proper alignment (fine-tuned spacing)
    fig.text(0.5, 0.10, 'Evaluation Results:', ha='center', fontsize=14, fontweight='bold')

    # Create table data with proper formatting
    all_athletes_in_gt = sorted(set([s["identity"].lower() for s in gt_segments if s.get("identity")]))

    # Calculate dynamic column width for athlete names
    max_athlete_len = max(len(athlete) for athlete in all_athletes_in_gt) if all_athletes_in_gt else 7
    max_athlete_len = max(max_athlete_len, len("TOTAL"), len("Athlete"))
    athlete_col_width = max_athlete_len + 1

    # Build table rows with dynamic width formatting for alignment
    table_lines = []

    # Header
    header = f"{'Athlete':<{athlete_col_width}} │ {'GT':<4} │ ArcFace: {'Pred':<4} {'Corr':<4} {'R':<6} {'P':<6} {'F':<6} │ TransFace: {'Pred':<4} {'Corr':<4} {'R':<6} {'P':<6} {'F':<6}"
    table_lines.append(header)
    table_lines.append("─" * len(header))

    # Data rows
    for athlete in all_athletes_in_gt:
        arcface_stats = results_arcface.get(athlete, {"gt": 0, "pred": 0, "correct": 0, "recall": 0, "precision": 0, "fscore": 0})
        transface_stats = results_transface.get(athlete, {"gt": 0, "pred": 0, "correct": 0, "recall": 0, "precision": 0, "fscore": 0})

        # Format percentages with consistent width
        arc_rec = f"{arcface_stats['recall']*100:.1f}%"
        arc_prec = f"{arcface_stats['precision']*100:.1f}%" if arcface_stats['pred'] > 0 else " 0.0%"
        arc_f1 = f"{arcface_stats['fscore']*100:.1f}%"

        trans_rec = f"{transface_stats['recall']*100:.1f}%"
        trans_prec = f"{transface_stats['precision']*100:.1f}%" if transface_stats['pred'] > 0 else " 0.0%"
        trans_f1 = f"{transface_stats['fscore']*100:.1f}%"

        row = (f"{athlete.capitalize():<{athlete_col_width}} │ {arcface_stats['gt']:<4} │          "
               f"{arcface_stats['pred']:<4} {arcface_stats['correct']:<4} {arc_rec:<6} {arc_prec:<6} {arc_f1:<6} │           "
               f"{transface_stats['pred']:<4} {transface_stats['correct']:<4} {trans_rec:<6} {trans_prec:<6} {trans_f1:<6}")
        table_lines.append(row)

    # Compute totals
    total_gt = sum([s["gt"] for s in results_arcface.values()])
    total_pred_arcface = sum([s["pred"] for s in results_arcface.values()])
    total_correct_arcface = sum([s["correct"] for s in results_arcface.values()])
    total_pred_transface = sum([s["pred"] for s in results_transface.values()])
    total_correct_transface = sum([s["correct"] for s in results_transface.values()])

    total_recall_arcface = total_correct_arcface / total_gt if total_gt > 0 else 0
    total_precision_arcface = total_correct_arcface / total_pred_arcface if total_pred_arcface > 0 else 0
    total_fscore_arcface = 2 * total_precision_arcface * total_recall_arcface / (total_precision_arcface + total_recall_arcface) if (total_precision_arcface + total_recall_arcface) > 0 else 0

    total_recall_transface = total_correct_transface / total_gt if total_gt > 0 else 0
    total_precision_transface = total_correct_transface / total_pred_transface if total_pred_transface > 0 else 0
    total_fscore_transface = 2 * total_precision_transface * total_recall_transface / (total_precision_transface + total_recall_transface) if (total_precision_transface + total_recall_transface) > 0 else 0

    # Format total percentages with consistent width
    total_arc_rec = f"{total_recall_arcface*100:.1f}%"
    total_arc_prec = f"{total_precision_arcface*100:.1f}%" if total_pred_arcface > 0 else " 0.0%"
    total_arc_f1 = f"{total_fscore_arcface*100:.1f}%"

    total_trans_rec = f"{total_recall_transface*100:.1f}%"
    total_trans_prec = f"{total_precision_transface*100:.1f}%" if total_pred_transface > 0 else " 0.0%"
    total_trans_f1 = f"{total_fscore_transface*100:.1f}%"

    # Totals row
    table_lines.append("─" * len(header))
    totals_row = (f"{'TOTAL':<{athlete_col_width}} │ {total_gt:<4} │          "
                  f"{total_pred_arcface:<4} {total_correct_arcface:<4} {total_arc_rec:<6} {total_arc_prec:<6} {total_arc_f1:<6} │           "
                  f"{total_pred_transface:<4} {total_correct_transface:<4} {total_trans_rec:<6} {total_trans_prec:<6} {total_trans_f1:<6}")
    table_lines.append(totals_row)

    # Draw table as text with larger font size (positioned close to title with minimal gap)
    table_text = "\n".join(table_lines)
    y_pos = 0.07
    fig.text(0.5, y_pos, table_text, ha='center', va='top', fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    # Save figure
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {output_path}")

    return results_arcface, results_transface


def get_video_info(video_path: str) -> Tuple[float, int, float]:
    """
    Get video FPS and duration using ffprobe

    Returns:
        Tuple of (fps, total_frames, duration_seconds)
    """
    import subprocess

    # Get FPS
    cmd_fps = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
        video_path
    ]

    result = subprocess.run(cmd_fps, capture_output=True, text=True)
    fps_str = result.stdout.strip()

    if '/' in fps_str:
        num, den = map(int, fps_str.split('/'))
        fps = num / den
    else:
        fps = float(fps_str)

    # Get duration
    cmd_duration = [
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", video_path
    ]

    result = subprocess.run(cmd_duration, capture_output=True, text=True)
    duration = float(result.stdout.strip())

    total_frames = int(fps * duration)

    return fps, total_frames, duration


def evaluate_video(
    video_name: str,
    data_dir: str = "data",
    resemblance_results_dir: str = "resemblance_results",
    output_dir: str = "evaluation_results/face_recognition",
    threshold: float = 0.6,
    smooth_window: int = 200,
    use_updated_target: bool = False,
    verbose: bool = True
) -> Dict:
    """
    Evaluate face recognition for a single video using ground truth highlights

    Args:
        video_name: Video name (without extension)
        data_dir: Directory containing videos and ground truth
        resemblance_results_dir: Directory containing resemblance results
        output_dir: Output directory for evaluation results
        threshold: Similarity threshold
        smooth_window: Smoothing window size
        use_updated_target: Whether to use updated target results
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Evaluating Face Recognition: {video_name}")
        print(f"{'='*80}")

    # Load ground truth highlights with identity
    gt_csv_path = os.path.join(data_dir, f"{video_name}_identity.csv")

    if not os.path.exists(gt_csv_path):
        print(f"  Warning: No identity ground truth found for {video_name}")
        print(f"  Expected: {gt_csv_path}")
        return None

    gt_segments = load_ground_truth_highlights(gt_csv_path)

    if not gt_segments:
        print(f"  Warning: No highlight segments found in {gt_csv_path}")
        return None

    if verbose:
        print(f"  Loaded {len(gt_segments)} ground truth highlight segments")

    # Get video info
    video_path = None
    for ext in [".mp4", ".avi", ".mov", ".mkv"]:
        candidate = os.path.join(data_dir, f"{video_name}{ext}")
        if os.path.exists(candidate):
            video_path = candidate
            break

    if not video_path:
        print(f"  Error: Video file not found for {video_name}")
        return None

    fps, total_frames, duration = get_video_info(video_path)

    if verbose:
        print(f"  Video FPS: {fps:.2f}")
        print(f"  Video duration: {duration:.2f} seconds")

    # Get list of target names (athletes in ground truth)
    target_names = sorted(set([s["identity"].lower() for s in gt_segments if s.get("identity")]))

    if verbose:
        print(f"  Athletes: {', '.join(target_names)}")

    # Load resemblance data
    all_resemblance_data = load_resemblance_data(
        resemblance_results_dir, video_name, target_names, use_updated_target
    )

    if not all_resemblance_data:
        print(f"  Error: No resemblance data found for any targets")
        return None

    # Create output directory
    output_path = Path(output_dir) / video_name
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "video_name": video_name,
        "num_gt_segments": len(gt_segments),
        "athletes": target_names,
        "methods": {}
    }

    # Evaluate with both assignment methods
    for method in ["sequential", "instant"]:
        if verbose:
            print(f"\n  Evaluating with {method} assignment method...")

        # Assign highlights for ArcFace
        assignments_arcface = assign_highlights_to_athletes_with_gt_segments(
            all_resemblance_data, gt_segments, fps, threshold, smooth_window,
            model="arcface", method=method
        )

        # Assign highlights for TransFace
        assignments_transface = assign_highlights_to_athletes_with_gt_segments(
            all_resemblance_data, gt_segments, fps, threshold, smooth_window,
            model="transface", method=method
        )

        # Create visualization plot
        suffix = "_updated" if use_updated_target else ""
        method_suffix = "_instant" if method == "instant" else ""
        plot_path = output_path / f"gt_highlights_assigned{method_suffix}{suffix}.png"

        results_arcface, results_transface = create_evaluation_plot(
            all_resemblance_data, gt_segments,
            assignments_arcface, assignments_transface,
            fps, duration, threshold, video_name,
            str(plot_path), method
        )

        # Store results
        results["methods"][method] = {
            "arcface": results_arcface,
            "transface": results_transface
        }

    # Save JSON results
    json_path = output_path / f"gt_highlights_results{suffix}.json"
    with open(json_path, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        json_safe_results = json.loads(json.dumps(results, default=lambda x: float(x) if isinstance(x, np.floating) else x))
        json.dump(json_safe_results, f, indent=2)

    if verbose:
        print(f"\n  Results saved to: {output_path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate face recognition models using ground truth highlight segments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single video
  python evaluate_face_recognition.py --video highjump_video1

  # Evaluate all videos of a sport
  python evaluate_face_recognition.py --sport highjump

  # Use updated target results
  python evaluate_face_recognition.py --video highjump_video1 --use-updated-target
        """
    )

    # Video selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Single video name to evaluate")
    group.add_argument("--sport", type=str, help="Sport name to evaluate all videos")

    # Directories
    parser.add_argument("--data-dir", type=str, default="data",
                       help="Directory containing videos and ground truth (default: data)")
    parser.add_argument("--resemblance-results-dir", type=str, default="resemblance_results",
                       help="Directory containing resemblance results (default: resemblance_results)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results/face_recognition",
                       help="Output directory (default: evaluation_results/face_recognition)")

    # Parameters
    parser.add_argument("--threshold", type=float, default=0.6,
                       help="Similarity threshold (default: 0.6)")
    parser.add_argument("--smooth-window", type=int, default=200,
                       help="Smoothing window size (default: 200)")
    parser.add_argument("--use-updated-target", action="store_true",
                       help="Use updated target results")

    # Options
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")

    args = parser.parse_args()

    # Determine which videos to evaluate
    videos_to_evaluate = []

    if args.video:
        videos_to_evaluate = [args.video]
    elif args.sport:
        # Find all videos for this sport that have identity annotations
        data_dir = Path(args.data_dir)
        sport_prefix = args.sport.lower()

        # Find all identity CSV files matching the sport
        identity_files = sorted(data_dir.glob(f"{sport_prefix}_video*_identity.csv"))

        videos_to_evaluate = [f.stem.replace("_identity", "") for f in identity_files]

        if not videos_to_evaluate:
            print(f"Error: No videos with identity annotations found for sport '{args.sport}'")
            print(f"Searched in: {data_dir}")
            return

        print(f"Found {len(videos_to_evaluate)} videos for {args.sport}:")
        for v in videos_to_evaluate:
            print(f"  - {v}")

    # Evaluate each video
    all_results = []
    verbose = not args.quiet

    for video_name in videos_to_evaluate:
        try:
            result = evaluate_video(
                video_name=video_name,
                data_dir=args.data_dir,
                resemblance_results_dir=args.resemblance_results_dir,
                output_dir=args.output_dir,
                threshold=args.threshold,
                smooth_window=args.smooth_window,
                use_updated_target=args.use_updated_target,
                verbose=verbose
            )

            if result:
                all_results.append(result)

        except Exception as e:
            print(f"\nError evaluating {video_name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    if all_results:
        print(f"\n{'='*80}")
        print(f"SUMMARY - Face Recognition Evaluation (Ground Truth Highlights)")
        print(f"{'='*80}")
        print(f"\nEvaluated {len(all_results)} videos")

        # Compute average metrics across all videos for each method and model
        for method in ["sequential", "instant"]:
            print(f"\n{method.upper()} Assignment Method:")

            # ArcFace
            arcface_metrics = []
            for result in all_results:
                if method in result.get("methods", {}):
                    method_results = result["methods"][method]["arcface"]
                    # Compute overall precision, recall, fscore
                    total_gt = sum([s["gt"] for s in method_results.values()])
                    total_pred = sum([s["pred"] for s in method_results.values()])
                    total_correct = sum([s["correct"] for s in method_results.values()])

                    recall = total_correct / total_gt if total_gt > 0 else 0
                    precision = total_correct / total_pred if total_pred > 0 else 0
                    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    arcface_metrics.append({"recall": recall, "precision": precision, "fscore": fscore})

            if arcface_metrics:
                avg_recall = np.mean([m["recall"] for m in arcface_metrics])
                avg_precision = np.mean([m["precision"] for m in arcface_metrics])
                avg_fscore = np.mean([m["fscore"] for m in arcface_metrics])

                print(f"  ArcFace:")
                print(f"    Avg Recall:    {avg_recall*100:.2f}%")
                print(f"    Avg Precision: {avg_precision*100:.2f}%")
                print(f"    Avg F-score:   {avg_fscore*100:.2f}%")

            # TransFace
            transface_metrics = []
            for result in all_results:
                if method in result.get("methods", {}):
                    method_results = result["methods"][method]["transface"]
                    # Compute overall precision, recall, fscore
                    total_gt = sum([s["gt"] for s in method_results.values()])
                    total_pred = sum([s["pred"] for s in method_results.values()])
                    total_correct = sum([s["correct"] for s in method_results.values()])

                    recall = total_correct / total_gt if total_gt > 0 else 0
                    precision = total_correct / total_pred if total_pred > 0 else 0
                    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                    transface_metrics.append({"recall": recall, "precision": precision, "fscore": fscore})

            if transface_metrics:
                avg_recall = np.mean([m["recall"] for m in transface_metrics])
                avg_precision = np.mean([m["precision"] for m in transface_metrics])
                avg_fscore = np.mean([m["fscore"] for m in transface_metrics])

                print(f"  TransFace:")
                print(f"    Avg Recall:    {avg_recall*100:.2f}%")
                print(f"    Avg Precision: {avg_precision*100:.2f}%")
                print(f"    Avg F-score:   {avg_fscore*100:.2f}%")

        print(f"\nResults saved to: {args.output_dir}")
    else:
        print("\nNo results to display.")


if __name__ == "__main__":
    main()
