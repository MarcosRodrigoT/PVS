"""
F-Score vs IoU Threshold Plot for PVS System

This script generates a plot showing how F-scores vary across different IoU thresholds
for all 16 PVS configurations. This helps justify the 0.3 IoU threshold choice.

The 16 configurations are:
- Video Summarizer: SportCLIP, QD-DETR
- Face Recognition: ArcFace, TransFace
- Assignment Method: Sequential (SEQ), Instant (INS)
- Target Embedding: Original, Updated
"""

import os
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from dataclasses import dataclass


# =============================================================================
# Configuration
# =============================================================================

DATA_DIR = Path("data")
VIDEO_SUMMARIES_DIR = Path("video_summaries")
RESEMBLANCE_DIR = Path("resemblance_results")
OUTPUT_DIR = Path("figures")

# Video list (all 20 videos)
VIDEOS = [
    "highjump_video1", "highjump_video2", "highjump_video3", "highjump_video4", "highjump_video5",
    "javelin_video1", "javelin_video2", "javelin_video3", "javelin_video4", "javelin_video5",
    "longjump_video1", "longjump_video2", "longjump_video3", "longjump_video4", "longjump_video5",
    "polevault_video1", "polevault_video2", "polevault_video3", "polevault_video4", "polevault_video5",
]

# Configuration options
SUMMARIZERS = ["SportCLIP", "QD-DETR"]
FR_MODELS = ["arcface", "transface"]
ASSIGNMENT_METHODS = ["sequential", "instant"]
TARGET_EMBEDDINGS = ["original", "updated"]

# Assignment parameters (matching evaluate_face_recognition.py)
THRESHOLD = 0.6
SMOOTH_WINDOW = 200
TEMPORAL_EXPANSION = 3.0

FPS = 25  # Frames per second for all videos


@dataclass
class Configuration:
    """Represents a single PVS configuration"""
    summarizer: str
    fr_model: str
    assignment_method: str
    target_embedding: str

    @property
    def name(self) -> str:
        """Short name for the configuration"""
        summ = "SC" if self.summarizer == "SportCLIP" else "QD"
        fr = "AF" if self.fr_model == "arcface" else "TF"
        method = "SEQ" if self.assignment_method == "sequential" else "INS"
        target = "UPD" if self.target_embedding == "updated" else "ORI"
        return f"{summ}-{fr}-{method}-{target}"

    @property
    def legend_name(self) -> str:
        """Descriptive name for legend"""
        summ = "SportCLIP" if self.summarizer == "SportCLIP" else "QD-DETR"
        fr = "ArcFace" if self.fr_model == "arcface" else "TransFace"
        method = "Seq" if self.assignment_method == "sequential" else "Ins"
        target = "Upd" if self.target_embedding == "updated" else "Ori"
        return f"{summ}/{fr}/{method}/{target}"


def get_all_configurations() -> List[Configuration]:
    """Generate all 16 configurations"""
    configs = []
    for summarizer in SUMMARIZERS:
        for fr_model in FR_MODELS:
            for method in ASSIGNMENT_METHODS:
                for target in TARGET_EMBEDDINGS:
                    configs.append(Configuration(summarizer, fr_model, method, target))
    return configs


# =============================================================================
# Utility Functions (from visualize_results.py)
# =============================================================================

def smooth_scores(scores: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply rolling average smoothing to scores"""
    if len(scores) < window_size:
        return scores

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores, kernel, mode="same")

    # Fix edges
    for i in range(window_size // 2):
        smoothed[i] = np.mean(scores[: i + window_size // 2 + 1])
        smoothed[-(i + 1)] = np.mean(scores[-(i + window_size // 2 + 1) :])

    return smoothed


def compute_iou(seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float) -> float:
    """Compute Intersection over Union for two time segments"""
    intersection_start = max(seg1_start, seg2_start)
    intersection_end = min(seg1_end, seg2_end)
    intersection = max(0, intersection_end - intersection_start)

    union = (seg1_end - seg1_start) + (seg2_end - seg2_start) - intersection
    return intersection / union if union > 0 else 0


# =============================================================================
# Data Loading Functions
# =============================================================================

def get_video_fps(video_name: str) -> float:
    """Get video FPS using ffprobe"""
    import subprocess
    video_path = DATA_DIR / f"{video_name}.mp4"
    if not video_path.exists():
        return 25.0  # Default fallback

    try:
        cmd = [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate", "-of", "default=noprint_wrappers=1:nokey=1",
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        fps_str = result.stdout.strip()
        if '/' in fps_str:
            num, den = map(int, fps_str.split('/'))
            return num / den
        return float(fps_str)
    except Exception:
        return 25.0  # Default fallback


def load_ground_truth(video_name: str, fps: float = None) -> List[Dict]:
    """
    Load ground truth from identity CSV file.
    Converts frame numbers to seconds using video FPS.
    """
    csv_path = DATA_DIR / f"{video_name}_identity.csv"
    if not csv_path.exists():
        print(f"Warning: Ground truth file not found: {csv_path}")
        return []

    # Get video FPS if not provided
    if fps is None:
        fps = get_video_fps(video_name)

    ground_truth = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Event type') == 'Highlight':
                try:
                    first_frame = int(row['First frame'])
                    last_frame = int(row['Last frame'])
                    identity = row['Identity'].strip()

                    # Convert frames to seconds using correct FPS
                    start_time = first_frame / fps
                    end_time = last_frame / fps

                    ground_truth.append({
                        'start_time': start_time,
                        'end_time': end_time,
                        'athlete': identity.lower()  # Convert to lowercase to match resemblance data
                    })
                except (ValueError, KeyError) as e:
                    continue

    return ground_truth


def load_segments(video_name: str, summarizer: str) -> Dict:
    """Load predicted segments for a video from a summarizer"""
    segments_path = VIDEO_SUMMARIES_DIR / summarizer / video_name / "segments.json"
    if not segments_path.exists():
        return {}

    with open(segments_path, 'r') as f:
        return json.load(f)


def load_resemblance_data(video_name: str, use_updated: bool) -> Dict[str, Dict]:
    """
    Load resemblance data for all athletes in a video.

    Args:
        video_name: Name of the video
        use_updated: Whether to use updated target embeddings

    Returns:
        Dictionary mapping athlete name to resemblance data
    """
    video_dir = RESEMBLANCE_DIR / video_name
    if not video_dir.exists():
        return {}

    all_resemblance = {}

    for athlete_dir in video_dir.iterdir():
        if not athlete_dir.is_dir():
            continue

        athlete_name = athlete_dir.name

        # Choose original or updated file
        if use_updated:
            json_file = athlete_dir / f"{athlete_name}_updated.json"
        else:
            json_file = athlete_dir / f"{athlete_name}.json"

        if not json_file.exists():
            continue

        try:
            with open(json_file, 'r') as f:
                data = json.load(f)
            all_resemblance[athlete_name] = data
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
            continue

    return all_resemblance


# =============================================================================
# Assignment Functions (from visualize_results.py)
# =============================================================================

def assign_highlights_sequential(
    all_resemblance_data: Dict[str, Dict],
    segments_data: Dict,
    threshold: float,
    smooth_window: int,
    model: str = "arcface",
) -> Dict[int, Optional[str]]:
    """
    Assign each highlight segment to an athlete based on resemblance curves.

    Algorithm: Scan timeline left-to-right. When an athlete's score crosses threshold,
    assign all subsequent highlights to that athlete until another athlete crosses threshold.
    """
    if not all_resemblance_data:
        return {}

    # Get timestamps from first target
    first_target = next(iter(all_resemblance_data.values()))
    timestamps = np.array(first_target["timestamps_seconds"])

    # Get sorted athlete names
    athlete_names = sorted(all_resemblance_data.keys())

    # Compute smoothed scores for all athletes
    athlete_scores = {}
    for athlete_name in athlete_names:
        data = all_resemblance_data[athlete_name]

        if model.lower() == "arcface":
            scores = np.array(data["arcface_scores"])
        else:
            scores = np.array(data["transface_scores"])

        smoothed = smooth_scores(scores, smooth_window)
        athlete_scores[athlete_name] = smoothed

    # Get segments
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    # Assignment dictionary
    assignments = {}
    current_athlete = None

    # Scan timeline left-to-right
    for seg_idx, segment in enumerate(segments):
        start_time = segment.get("start_time")

        if start_time is None:
            assignments[seg_idx] = current_athlete
            continue

        # Find timestamp index closest to segment start
        start_idx = np.searchsorted(timestamps, start_time)

        # Look back to catch threshold crossings
        lookback_time = max(0, start_time - 5)
        lookback_idx = np.searchsorted(timestamps, lookback_time)

        # Check if any athlete crossed threshold
        crossed_athletes = []
        for athlete_name in athlete_names:
            scores = athlete_scores[athlete_name]
            window_scores = scores[lookback_idx : start_idx + 1] if start_idx < len(scores) else scores[lookback_idx:]
            if len(window_scores) > 0 and np.any(window_scores >= threshold):
                crossing_indices = np.where(window_scores >= threshold)[0]
                last_crossing = lookback_idx + crossing_indices[-1]
                crossed_athletes.append((athlete_name, last_crossing))

        if crossed_athletes:
            crossed_athletes.sort(key=lambda x: x[1], reverse=True)
            current_athlete = crossed_athletes[0][0]

        assignments[seg_idx] = current_athlete

    return assignments


def assign_highlights_instant(
    all_resemblance_data: Dict[str, Dict],
    segments_data: Dict,
    threshold: float,
    smooth_window: int,
    model: str = "arcface",
    temporal_expansion: float = 3.0,
) -> Dict[int, Optional[str]]:
    """
    Assign each highlight segment to an athlete based on instant overlap with resemblance curves.

    Algorithm: For each highlight segment, check if any athlete's resemblance score exceeds
    the threshold during that segment's time window. Assign to highest average score.
    """
    if not all_resemblance_data:
        return {}

    # Get timestamps from first target
    first_target = next(iter(all_resemblance_data.values()))
    timestamps = np.array(first_target["timestamps_seconds"])

    # Get sorted athlete names
    athlete_names = sorted(all_resemblance_data.keys())

    # Compute smoothed scores for all athletes
    athlete_scores = {}
    for athlete_name in athlete_names:
        data = all_resemblance_data[athlete_name]

        if model.lower() == "arcface":
            scores = np.array(data["arcface_scores"])
        else:
            scores = np.array(data["transface_scores"])

        smoothed = smooth_scores(scores, smooth_window)
        athlete_scores[athlete_name] = smoothed

    # Get segments
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    # Assignment dictionary
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

        # Find timestamp indices
        start_idx = np.searchsorted(timestamps, expanded_start)
        end_idx = np.searchsorted(timestamps, expanded_end)

        # Check each athlete's scores
        candidate_athletes = []
        for athlete_name in athlete_names:
            scores = athlete_scores[athlete_name]
            window_scores = scores[start_idx:end_idx]

            if len(window_scores) > 0 and np.any(window_scores >= threshold):
                above_threshold_scores = window_scores[window_scores >= threshold]
                avg_score = np.mean(above_threshold_scores)
                max_score = np.max(window_scores)
                candidate_athletes.append((athlete_name, avg_score, max_score))

        if candidate_athletes:
            candidate_athletes.sort(key=lambda x: (x[1], x[2]), reverse=True)
            assignments[seg_idx] = candidate_athletes[0][0]
        else:
            assignments[seg_idx] = None

    return assignments


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_at_iou_threshold(
    segments_data: Dict,
    assignments: Dict[int, Optional[str]],
    ground_truth: List[Dict],
    iou_threshold: float
) -> Tuple[int, int, int]:
    """
    Evaluate predictions against ground truth at a specific IoU threshold.

    Returns:
        Tuple of (TP, FP, FN)
    """
    # Get segments
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    # Build predicted segments with assignments
    pred_segments = []
    for seg_idx, segment in enumerate(segments):
        athlete = assignments.get(seg_idx)
        if athlete is not None:
            pred_segments.append({
                "start": segment.get("start_time"),
                "end": segment.get("end_time"),
                "athlete": athlete,
                "matched": False
            })

    # Build ground truth segments by athlete
    gt_segments_by_athlete = {}
    for gt_seg in ground_truth:
        athlete = gt_seg["athlete"]
        if athlete not in gt_segments_by_athlete:
            gt_segments_by_athlete[athlete] = []
        gt_segments_by_athlete[athlete].append({
            "start": gt_seg["start_time"],
            "end": gt_seg["end_time"],
            "matched": False
        })

    # Match predicted to ground truth
    for pred_seg in pred_segments:
        if pred_seg["athlete"] not in gt_segments_by_athlete:
            continue

        best_iou = 0
        best_gt_idx = None

        for gt_idx, gt_seg in enumerate(gt_segments_by_athlete[pred_seg["athlete"]]):
            if gt_seg["matched"]:
                continue

            iou = compute_iou(pred_seg["start"], pred_seg["end"], gt_seg["start"], gt_seg["end"])
            if iou >= iou_threshold and iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        if best_gt_idx is not None:
            pred_seg["matched"] = True
            gt_segments_by_athlete[pred_seg["athlete"]][best_gt_idx]["matched"] = True

    # Count TP, FP, FN
    tp = sum(1 for seg in pred_segments if seg["matched"])
    fp = sum(1 for seg in pred_segments if not seg["matched"])

    fn = 0
    for athlete, gt_segs in gt_segments_by_athlete.items():
        fn += sum(1 for seg in gt_segs if not seg["matched"])

    return tp, fp, fn


def compute_fscore(tp: int, fp: int, fn: int) -> float:
    """Compute F1-score from TP, FP, FN"""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# =============================================================================
# Main IoU Sweep Function
# =============================================================================

# Sports and their videos
SPORTS = {
    "highjump": ["highjump_video1", "highjump_video2", "highjump_video3", "highjump_video4", "highjump_video5"],
    "javelin": ["javelin_video1", "javelin_video2", "javelin_video3", "javelin_video4", "javelin_video5"],
    "longjump": ["longjump_video1", "longjump_video2", "longjump_video3", "longjump_video4", "longjump_video5"],
    "polevault": ["polevault_video1", "polevault_video2", "polevault_video3", "polevault_video4", "polevault_video5"],
}


def sweep_iou_for_config(config: Configuration, iou_thresholds: np.ndarray) -> np.ndarray:
    """
    Sweep IoU thresholds for a single configuration across all videos.

    Uses the same aggregation method as the thesis:
    1. Aggregate TP/FP/FN across all 5 videos within each sport
    2. Compute F-score per sport
    3. Average F-scores across the 4 sports

    Returns:
        Array of F-scores for each IoU threshold
    """
    use_updated = config.target_embedding == "updated"

    # Store per-sport F-scores for each IoU threshold
    per_sport_fscores = {iou: [] for iou in iou_thresholds}

    for sport_name, sport_videos in SPORTS.items():
        # Aggregate TP/FP/FN across all videos in this sport
        sport_stats = {iou: {"tp": 0, "fp": 0, "fn": 0} for iou in iou_thresholds}

        for video_name in sport_videos:
            # Load data for this video
            ground_truth = load_ground_truth(video_name)
            if not ground_truth:
                continue

            segments_data = load_segments(video_name, config.summarizer)
            if not segments_data:
                continue

            resemblance_data = load_resemblance_data(video_name, use_updated)
            if not resemblance_data:
                continue

            # Generate assignments
            if config.assignment_method == "sequential":
                assignments = assign_highlights_sequential(
                    resemblance_data, segments_data, THRESHOLD, SMOOTH_WINDOW, config.fr_model
                )
            else:
                assignments = assign_highlights_instant(
                    resemblance_data, segments_data, THRESHOLD, SMOOTH_WINDOW,
                    config.fr_model, TEMPORAL_EXPANSION
                )

            # Evaluate at each IoU threshold
            for iou_thresh in iou_thresholds:
                tp, fp, fn = evaluate_at_iou_threshold(
                    segments_data, assignments, ground_truth, iou_thresh
                )
                # Accumulate stats for this sport
                sport_stats[iou_thresh]["tp"] += tp
                sport_stats[iou_thresh]["fp"] += fp
                sport_stats[iou_thresh]["fn"] += fn

        # Compute F-score for this sport at each IoU threshold
        for iou_thresh in iou_thresholds:
            stats = sport_stats[iou_thresh]
            fscore = compute_fscore(stats["tp"], stats["fp"], stats["fn"])
            per_sport_fscores[iou_thresh].append(fscore)

    # Average F-scores across sports (matching thesis methodology)
    fscores = np.array([
        np.mean(per_sport_fscores[iou]) if per_sport_fscores[iou] else 0.0
        for iou in iou_thresholds
    ])

    return fscores


# =============================================================================
# Plotting Function
# =============================================================================

def create_fscore_vs_iou_plot(
    results: Dict[str, np.ndarray],
    iou_thresholds: np.ndarray,
    output_prefix: str = "fscore_vs_iou_threshold"
):
    """
    Create publication-quality plot of F-score vs IoU threshold.

    Visual encoding:
    - Color family: Blue tones = ArcFace, Red tones = TransFace
    - Color intensity: Darker = Updated target, Lighter = Original target
    - Line style: Solid = Sequential, Dashed = Instant
    - Markers: Circle = SportCLIP, Square = QD-DETR
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    # Color definitions
    colors = {
        ("arcface", "original"): "#6baed6",      # Light blue
        ("arcface", "updated"): "#08519c",       # Dark blue
        ("transface", "original"): "#fb6a4a",    # Light red
        ("transface", "updated"): "#a50f15",     # Dark red
    }

    # Line styles
    linestyles = {
        "sequential": "-",
        "instant": "--"
    }

    # Markers
    markers = {
        "SportCLIP": "o",
        "QD-DETR": "s"
    }

    # Plot each configuration
    lines = []
    labels = []

    for config in get_all_configurations():
        config_key = config.name
        if config_key not in results:
            continue

        fscores = results[config_key]

        color = colors[(config.fr_model, config.target_embedding)]
        linestyle = linestyles[config.assignment_method]
        marker = markers[config.summarizer]

        # Plot with sparse markers (every 4 points)
        line, = ax.plot(
            iou_thresholds, fscores,
            color=color,
            linestyle=linestyle,
            marker=marker,
            markevery=4,
            markersize=6,
            linewidth=1.5,
            alpha=0.9
        )

        lines.append(line)
        labels.append(config.legend_name)

    # Add vertical reference line at IoU = 0.3
    ax.axvline(x=0.3, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='IoU = 0.3')

    # Formatting
    ax.set_xlabel("IoU Threshold", fontsize=12, fontweight='bold')
    ax.set_ylabel("F-Score", fontsize=12, fontweight='bold')
    ax.set_title("F-Score vs IoU Threshold for All PVS Configurations", fontsize=14, fontweight='bold')

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Create custom legend with grouping
    # First, create legend entries for summarizers
    from matplotlib.lines import Line2D

    legend_elements = [
        # Summarizers
        Line2D([0], [0], color='gray', marker='o', linestyle='none', markersize=8, label='SportCLIP'),
        Line2D([0], [0], color='gray', marker='s', linestyle='none', markersize=8, label='QD-DETR'),
        Line2D([0], [0], color='none', marker='none', linestyle='none', label=''),  # Spacer
        # FR Models + Target Embeddings
        Line2D([0], [0], color='#6baed6', marker='none', linestyle='-', linewidth=2, label='ArcFace (Ori)'),
        Line2D([0], [0], color='#08519c', marker='none', linestyle='-', linewidth=2, label='ArcFace (Upd)'),
        Line2D([0], [0], color='#fb6a4a', marker='none', linestyle='-', linewidth=2, label='TransFace (Ori)'),
        Line2D([0], [0], color='#a50f15', marker='none', linestyle='-', linewidth=2, label='TransFace (Upd)'),
        Line2D([0], [0], color='none', marker='none', linestyle='none', label=''),  # Spacer
        # Assignment methods
        Line2D([0], [0], color='gray', marker='none', linestyle='-', linewidth=2, label='Sequential'),
        Line2D([0], [0], color='gray', marker='none', linestyle='--', linewidth=2, label='Instant'),
        Line2D([0], [0], color='none', marker='none', linestyle='none', label=''),  # Spacer
        # Reference line
        Line2D([0], [0], color='gray', marker='none', linestyle=':', linewidth=2, label='Current Threshold (0.3)'),
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=9,
              framealpha=0.95, ncol=1)

    plt.tight_layout()

    # Save figures
    OUTPUT_DIR.mkdir(exist_ok=True)

    pdf_path = OUTPUT_DIR / f"{output_prefix}.pdf"
    png_path = OUTPUT_DIR / f"{output_prefix}.png"

    plt.savefig(pdf_path, format='pdf', dpi=300, bbox_inches='tight')
    plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")

    plt.close()


def save_results_json(
    results: Dict[str, np.ndarray],
    iou_thresholds: np.ndarray,
    output_path: Path
):
    """Save raw numerical results to JSON"""
    output_data = {
        "iou_thresholds": iou_thresholds.tolist(),
        "configurations": {}
    }

    for config_name, fscores in results.items():
        output_data["configurations"][config_name] = {
            "fscores": fscores.tolist()
        }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    print("=" * 60)
    print("F-Score vs IoU Threshold Analysis")
    print("=" * 60)

    # IoU thresholds to evaluate
    iou_thresholds = np.arange(0.0, 1.05, 0.05)

    print(f"\nEvaluating {len(iou_thresholds)} IoU thresholds: {iou_thresholds[0]:.2f} to {iou_thresholds[-1]:.2f}")
    print(f"Across {len(VIDEOS)} videos and 16 configurations\n")

    # Get all configurations
    configurations = get_all_configurations()

    # Store results
    results = {}

    # Process each configuration
    for i, config in enumerate(configurations):
        print(f"[{i+1:2d}/16] Processing {config.name}...")

        fscores = sweep_iou_for_config(config, iou_thresholds)
        results[config.name] = fscores

        # Print F-score at IoU=0.3 for validation
        idx_03 = np.argmin(np.abs(iou_thresholds - 0.3))
        print(f"        F-score at IoU=0.3: {fscores[idx_03]:.4f}")

    print("\n" + "=" * 60)
    print("Generating plots...")
    print("=" * 60)

    # Create plot
    create_fscore_vs_iou_plot(results, iou_thresholds)

    # Save raw data
    DATA_DIR.mkdir(exist_ok=True)
    save_results_json(results, iou_thresholds, DATA_DIR / "fscore_iou_sweep_results.json")

    print("\nDone!")

    # Print summary at IoU=0.3
    print("\n" + "=" * 60)
    print("F-Scores at IoU=0.3 (Current Threshold)")
    print("=" * 60)
    idx_03 = np.argmin(np.abs(iou_thresholds - 0.3))

    for config in configurations:
        fscore = results[config.name][idx_03]
        print(f"{config.name:20s}: {fscore:.4f}")


if __name__ == "__main__":
    main()
