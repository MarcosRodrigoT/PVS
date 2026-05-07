"""
F-Score vs Frames Skipped Plot for PVS System

Generates a plot showing how F-scores (at IoU=0.3) vary as we skip more frames
during the face-analysis stage. Visualises the speed/quality trade-off discussed
in the thesis (Section 4.7, Table 4.24).

X-axis: processing stride (0=every frame, 10=every 10th frame, ...)
Y-axis: F-score at IoU=0.3
"""

import json
import csv
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


# =============================================================================
# Configuration (mirrors plot_fscore_vs_iou.py)
# =============================================================================

DATA_DIR = Path("data")
VIDEO_SUMMARIES_DIR = Path("video_summaries")
RESEMBLANCE_DIR = Path("resemblance_results")
OUTPUT_DIR = Path("figures")

SUMMARIZERS = ["SportCLIP", "QD-DETR"]
FR_MODELS = ["arcface", "transface"]
ASSIGNMENT_METHODS = ["sequential", "instant"]
TARGET_EMBEDDINGS = ["original", "updated"]

THRESHOLD = 0.6
SMOOTH_WINDOW = 200
TEMPORAL_EXPANSION = 3.0
FPS = 25

IOU_THRESHOLD = 0.3
SKIP_VALUES = list(range(0, 325, 25))  # 0, 25, 50, …, 300

SPORTS = {
    "highjump":  ["highjump_video1",  "highjump_video2",  "highjump_video3",  "highjump_video4",  "highjump_video5"],
    "javelin":   ["javelin_video1",   "javelin_video2",   "javelin_video3",   "javelin_video4",   "javelin_video5"],
    "longjump":  ["longjump_video1",  "longjump_video2",  "longjump_video3",  "longjump_video4",  "longjump_video5"],
    "polevault": ["polevault_video1", "polevault_video2", "polevault_video3", "polevault_video4", "polevault_video5"],
}


@dataclass
class Configuration:
    summarizer: str
    fr_model: str
    assignment_method: str
    target_embedding: str

    @property
    def name(self) -> str:
        summ = "SC" if self.summarizer == "SportCLIP" else "QD"
        fr = "AF" if self.fr_model == "arcface" else "TF"
        method = "SEQ" if self.assignment_method == "sequential" else "INS"
        target = "UPD" if self.target_embedding == "updated" else "ORI"
        return f"{summ}-{fr}-{method}-{target}"


def get_all_configurations() -> List[Configuration]:
    configs = []
    for summarizer in SUMMARIZERS:
        for fr_model in FR_MODELS:
            for method in ASSIGNMENT_METHODS:
                for target in TARGET_EMBEDDINGS:
                    configs.append(Configuration(summarizer, fr_model, method, target))
    return configs


# =============================================================================
# Utility / Data Loading helpers (verbatim from plot_fscore_vs_iou.py)
# =============================================================================

def get_video_fps(video_name: str) -> float:
    video_path = DATA_DIR / f"{video_name}.mp4"
    if not video_path.exists():
        return 25.0
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
        return 25.0

def smooth_scores(scores: np.ndarray, window_size: int = 5) -> np.ndarray:
    if len(scores) < window_size:
        return scores
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(scores, kernel, mode="same")
    for i in range(window_size // 2):
        smoothed[i] = np.mean(scores[: i + window_size // 2 + 1])
        smoothed[-(i + 1)] = np.mean(scores[-(i + window_size // 2 + 1):])
    return smoothed


def compute_iou(seg1_start: float, seg1_end: float, seg2_start: float, seg2_end: float) -> float:
    intersection_start = max(seg1_start, seg2_start)
    intersection_end = min(seg1_end, seg2_end)
    intersection = max(0, intersection_end - intersection_start)
    union = (seg1_end - seg1_start) + (seg2_end - seg2_start) - intersection
    return intersection / union if union > 0 else 0


# =============================================================================
# Data Loading (verbatim from plot_fscore_vs_iou.py)
# =============================================================================

def load_ground_truth(video_name: str, fps: float = None) -> List[Dict]:
    csv_path = DATA_DIR / f"{video_name}_identity.csv"
    if not csv_path.exists():
        print(f"Warning: Ground truth file not found: {csv_path}")
        return []
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
                    ground_truth.append({
                        'start_time': first_frame / fps,
                        'end_time': last_frame / fps,
                        'athlete': identity.lower(),
                    })
                except (ValueError, KeyError):
                    continue
    return ground_truth


def load_segments(video_name: str, summarizer: str) -> Dict:
    segments_path = VIDEO_SUMMARIES_DIR / summarizer / video_name / "segments.json"
    if not segments_path.exists():
        return {}
    with open(segments_path, 'r') as f:
        return json.load(f)


def load_resemblance_data(video_name: str, use_updated: bool) -> Dict[str, Dict]:
    video_dir = RESEMBLANCE_DIR / video_name
    if not video_dir.exists():
        return {}
    all_resemblance = {}
    for athlete_dir in video_dir.iterdir():
        if not athlete_dir.is_dir():
            continue
        athlete_name = athlete_dir.name
        suffix = "_updated" if use_updated else ""
        json_file = athlete_dir / f"{athlete_name}{suffix}.json"
        if not json_file.exists():
            continue
        try:
            with open(json_file, 'r') as f:
                all_resemblance[athlete_name] = json.load(f)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")
    return all_resemblance


# =============================================================================
# Subsampling
# =============================================================================

def subsample_resemblance(resemblance_data: Dict[str, Dict], stride: int) -> Dict[str, Dict]:
    """Keep only entries whose frame index is a multiple of stride.

    Uses the fps stored in each athlete's JSON so that non-25-fps videos are
    handled correctly (e.g. highjump_video3 is 50 fps).
    """
    if stride <= 1:
        return resemblance_data
    subsampled = {}
    for athlete, data in resemblance_data.items():
        fps = data.get("fps", 25.0)
        timestamps = np.array(data["timestamps_seconds"])
        frame_indices = np.round(timestamps * fps).astype(int)
        mask = (frame_indices % stride) == 0
        subsampled[athlete] = {
            **data,
            "timestamps_seconds": timestamps[mask].tolist(),
            "arcface_scores": np.array(data["arcface_scores"])[mask].tolist(),
            "transface_scores": np.array(data["transface_scores"])[mask].tolist(),
        }
    return subsampled


# =============================================================================
# Assignment (verbatim from plot_fscore_vs_iou.py)
# =============================================================================

def assign_highlights_sequential(
    all_resemblance_data: Dict[str, Dict],
    segments_data: Dict,
    threshold: float,
    smooth_window: int,
    model: str = "arcface",
) -> Dict[int, Optional[str]]:
    if not all_resemblance_data:
        return {}
    first_target = next(iter(all_resemblance_data.values()))
    timestamps = np.array(first_target["timestamps_seconds"])
    athlete_names = sorted(all_resemblance_data.keys())
    athlete_scores = {}
    for athlete_name in athlete_names:
        data = all_resemblance_data[athlete_name]
        scores = np.array(data["arcface_scores"] if model.lower() == "arcface" else data["transface_scores"])
        athlete_scores[athlete_name] = smooth_scores(scores, smooth_window)
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj
    assignments = {}
    current_athlete = None
    for seg_idx, segment in enumerate(segments):
        start_time = segment.get("start_time")
        if start_time is None:
            assignments[seg_idx] = current_athlete
            continue
        start_idx = np.searchsorted(timestamps, start_time)
        lookback_time = max(0, start_time - 5)
        lookback_idx = np.searchsorted(timestamps, lookback_time)
        crossed_athletes = []
        for athlete_name in athlete_names:
            scores = athlete_scores[athlete_name]
            window_scores = scores[lookback_idx: start_idx + 1] if start_idx < len(scores) else scores[lookback_idx:]
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
    if not all_resemblance_data:
        return {}
    first_target = next(iter(all_resemblance_data.values()))
    timestamps = np.array(first_target["timestamps_seconds"])
    athlete_names = sorted(all_resemblance_data.keys())
    athlete_scores = {}
    for athlete_name in athlete_names:
        data = all_resemblance_data[athlete_name]
        scores = np.array(data["arcface_scores"] if model.lower() == "arcface" else data["transface_scores"])
        athlete_scores[athlete_name] = smooth_scores(scores, smooth_window)
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj
    assignments = {}
    for seg_idx, segment in enumerate(segments):
        start_time = segment.get("start_time")
        end_time = segment.get("end_time")
        if start_time is None or end_time is None:
            assignments[seg_idx] = None
            continue
        expanded_start = max(0, start_time - temporal_expansion)
        expanded_end = end_time + temporal_expansion
        start_idx = np.searchsorted(timestamps, expanded_start)
        end_idx = np.searchsorted(timestamps, expanded_end)
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
# Evaluation (verbatim from plot_fscore_vs_iou.py)
# =============================================================================

def evaluate_at_iou_threshold(
    segments_data: Dict,
    assignments: Dict[int, Optional[str]],
    ground_truth: List[Dict],
    iou_threshold: float,
) -> Tuple[int, int, int]:
    segments_obj = segments_data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj
    pred_segments = []
    for seg_idx, segment in enumerate(segments):
        athlete = assignments.get(seg_idx)
        if athlete is not None:
            pred_segments.append({
                "start": segment.get("start_time"),
                "end": segment.get("end_time"),
                "athlete": athlete,
                "matched": False,
            })
    gt_segments_by_athlete: Dict[str, List[Dict]] = {}
    for gt_seg in ground_truth:
        athlete = gt_seg["athlete"]
        if athlete not in gt_segments_by_athlete:
            gt_segments_by_athlete[athlete] = []
        gt_segments_by_athlete[athlete].append({
            "start": gt_seg["start_time"],
            "end": gt_seg["end_time"],
            "matched": False,
        })
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
    tp = sum(1 for seg in pred_segments if seg["matched"])
    fp = sum(1 for seg in pred_segments if not seg["matched"])
    fn = sum(
        1 for segs in gt_segments_by_athlete.values() for seg in segs if not seg["matched"]
    )
    return tp, fp, fn


def compute_fscore(tp: int, fp: int, fn: int) -> float:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return f1


# =============================================================================
# Frame-skip sweep
# =============================================================================

def evaluate_config_at_stride(config: Configuration, stride: int) -> float:
    """
    Evaluate a single configuration at a given processing stride.
    Returns the mean F-score averaged across 4 sports.
    """
    use_updated = config.target_embedding == "updated"
    per_sport_fscores = []

    for sport_name, sport_videos in SPORTS.items():
        tp_sport, fp_sport, fn_sport = 0, 0, 0
        sport_has_data = False

        for video_name in sport_videos:
            ground_truth = load_ground_truth(video_name)
            if not ground_truth:
                continue
            segments_data = load_segments(video_name, config.summarizer)
            if not segments_data:
                continue
            resemblance_data = load_resemblance_data(video_name, use_updated)
            if not resemblance_data:
                continue

            subsampled = subsample_resemblance(resemblance_data, stride)

            # Scale smooth_window so it covers the same temporal span regardless of stride.
            # SMOOTH_WINDOW=200 was calibrated on dense 25-fps data (~10s window).
            # At stride=10 the same 200 samples would span ~100s — catastrophic over-smoothing.
            effective_smooth = max(1, SMOOTH_WINDOW // stride)

            if config.assignment_method == "sequential":
                assignments = assign_highlights_sequential(
                    subsampled, segments_data, THRESHOLD, effective_smooth, config.fr_model
                )
            else:
                assignments = assign_highlights_instant(
                    subsampled, segments_data, THRESHOLD, effective_smooth,
                    config.fr_model, TEMPORAL_EXPANSION
                )

            tp, fp, fn = evaluate_at_iou_threshold(
                segments_data, assignments, ground_truth, IOU_THRESHOLD
            )
            tp_sport += tp
            fp_sport += fp
            fn_sport += fn
            sport_has_data = True

        if sport_has_data:
            per_sport_fscores.append(compute_fscore(tp_sport, fp_sport, fn_sport))

    return float(np.mean(per_sport_fscores)) if per_sport_fscores else 0.0


def run_sweep() -> Dict[int, Dict[str, float]]:
    """Run the full skip×config sweep and return results[skip][config_name] = fscore."""
    configurations = get_all_configurations()
    results: Dict[int, Dict[str, float]] = {}

    for skip in SKIP_VALUES:
        stride = max(1, skip)
        print(f"\n--- Stride {stride:3d} (skip={skip}) ---")
        results[skip] = {}

        for config in configurations:
            fscore = evaluate_config_at_stride(config, stride)
            results[skip][config.name] = fscore
            print(f"  {config.name}: {fscore:.4f}")

    return results


def load_baseline() -> Dict[str, float]:
    """Load F-scores at IoU=0.3 from the existing sweep results."""
    path = DATA_DIR / "fscore_iou_sweep_results.json"
    if not path.exists():
        return {}
    with open(path) as f:
        d = json.load(f)
    thresholds = d["iou_thresholds"]
    idx = min(range(len(thresholds)), key=lambda i: abs(thresholds[i] - IOU_THRESHOLD))
    return {k: v["fscores"][idx] for k, v in d["configurations"].items()}


def sanity_check(results: Dict[int, Dict[str, float]]) -> None:
    baseline = load_baseline()
    if not baseline:
        print("Warning: baseline file not found; skipping sanity check.")
        return
    print("\n=== Sanity check (skip=0 vs baseline at IoU=0.3) ===")
    skip0 = results.get(0, {})
    any_mismatch = False
    for config_name, expected in baseline.items():
        got = skip0.get(config_name)
        if got is None:
            print(f"  MISSING  {config_name}")
            any_mismatch = True
        elif abs(got - expected) > 0.001:
            print(f"  MISMATCH {config_name}: got {got:.4f}, expected {expected:.4f} (diff {got-expected:+.4f})")
            any_mismatch = True
    if not any_mismatch:
        print("  All configurations match baseline within 0.001 ✓")


# =============================================================================
# Plotting
# =============================================================================

def create_fscore_vs_frames_skipped_plot(
    results: Dict[int, Dict[str, float]],
    output_prefix: str = "fscore_vs_frames_skipped",
) -> None:
    """
    Line plot with embedded markers — same visual encoding as fscore_vs_iou_threshold.png:
    - Color: blue tones = ArcFace, red tones = TransFace; dark = Updated, light = Original
    - Marker shape on line: circle = SportCLIP, square = QD-DETR
    - Line style: solid = Sequential, dashed = Instant
    """
    from matplotlib.lines import Line2D

    colors = {
        ("arcface",   "original"): "#6baed6",
        ("arcface",   "updated"):  "#08519c",
        ("transface", "original"): "#fb6a4a",
        ("transface", "updated"):  "#a50f15",
    }
    markers   = {"SportCLIP": "o", "QD-DETR": "s"}
    linestyle = {"sequential": "-", "instant": "--"}

    x_values = SKIP_VALUES  # 0 = "All frames", 10, 20, …, 120

    fig, ax = plt.subplots(figsize=(9, 7))

    for config in get_all_configurations():
        color  = colors[(config.fr_model, config.target_embedding)]
        marker = markers[config.summarizer]
        ls     = linestyle[config.assignment_method]
        y_values = [results[skip].get(config.name, np.nan) for skip in x_values]

        ax.plot(
            x_values, y_values,
            color=color, linestyle=ls,
            marker=marker, markevery=2,
            markersize=6, linewidth=1.5, alpha=0.9,
        )

    ax.set_xlabel("Processing stride (frames)", fontsize=12, fontweight="bold")
    ax.set_ylabel("F-Score", fontsize=12, fontweight="bold")
    ax.set_title("F-Score vs Frame-Skipping for All PVS Configurations", fontsize=14, fontweight="bold")

    ax.set_xticks(x_values)
    ax.set_xticklabels(["All\nframes" if s == 0 else str(s) for s in x_values])
    ax.set_xlim(-10, 310)
    ax.set_ylim(0, 1)
    ax.set_yticks(np.arange(0, 1.1, 0.1))

    ax.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    legend_elements = [
        Line2D([0], [0], color="gray", marker="o", linestyle="none", markersize=8, label="SportCLIP"),
        Line2D([0], [0], color="gray", marker="s", linestyle="none", markersize=8, label="QD-DETR"),
        Line2D([0], [0], color="none", marker="none", linestyle="none", label=""),
        Line2D([0], [0], color="#6baed6", linestyle="-", linewidth=2, label="ArcFace (Ori)"),
        Line2D([0], [0], color="#08519c", linestyle="-", linewidth=2, label="ArcFace (Upd)"),
        Line2D([0], [0], color="#fb6a4a", linestyle="-", linewidth=2, label="TransFace (Ori)"),
        Line2D([0], [0], color="#a50f15", linestyle="-", linewidth=2, label="TransFace (Upd)"),
        Line2D([0], [0], color="none", marker="none", linestyle="none", label=""),
        Line2D([0], [0], color="gray", linestyle="-",  linewidth=2, label="Sequential"),
        Line2D([0], [0], color="gray", linestyle="--", linewidth=2, label="Instant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9, framealpha=0.95)

    plt.tight_layout()

    OUTPUT_DIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        path = OUTPUT_DIR / f"{output_prefix}.{ext}"
        plt.savefig(path, format=ext, dpi=300, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close()


def save_results_json(results: Dict[int, Dict[str, float]], output_path: Path) -> None:
    serialisable = {str(skip): configs for skip, configs in results.items()}
    with open(output_path, "w") as f:
        json.dump({"iou_threshold": IOU_THRESHOLD, "results_by_skip": serialisable}, f, indent=2)
    print(f"Saved: {output_path}")


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    print("=" * 60)
    print("F-Score vs Frames Skipped Analysis")
    print(f"IoU threshold: {IOU_THRESHOLD} | Skip values: {SKIP_VALUES}")
    print("=" * 60)

    results = run_sweep()

    sanity_check(results)

    print("\n=== Generating plot ===")
    create_fscore_vs_frames_skipped_plot(results)

    DATA_DIR.mkdir(exist_ok=True)
    save_results_json(results, DATA_DIR / "fscore_vs_frames_skipped_results.json")

    print("\nDone!")


if __name__ == "__main__":
    main()
