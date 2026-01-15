#!/usr/bin/env python3
"""
Video Summarization Evaluation Script

This script evaluates video summarization models (SportCLIP and QD-DETR)
using frame-level recall, precision, and F-score metrics.

It compares predicted highlight segments against ground truth annotations
to assess the quality of video summarization independently from face recognition.
"""

import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def load_ground_truth(csv_path: str) -> Dict[int, str]:
    """
    Load ground truth annotations from CSV file

    Args:
        csv_path: Path to ground truth CSV file

    Returns:
        Dictionary mapping frame numbers to event types ("Highlight" or "Not a highlight")
    """
    ground_truth = {}

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Ground truth file not found: {csv_path}")

    df = pd.read_csv(csv_path)

    for _, row in df.iterrows():
        event_type = row["Event type"]
        first_frame = int(row["First frame"])
        last_frame = int(row["Last frame"])

        # Mark all frames in the range with the event type
        for frame_num in range(first_frame, last_frame + 1):
            ground_truth[frame_num] = event_type

    return ground_truth


def load_predicted_segments(segments_json_path: str) -> Dict[int, Dict]:
    """
    Load predicted highlight segments from segments.json file

    Args:
        segments_json_path: Path to segments.json file

    Returns:
        Dictionary of predicted events with frame ranges
    """
    if not os.path.exists(segments_json_path):
        raise FileNotFoundError(f"Segments file not found: {segments_json_path}")

    with open(segments_json_path, 'r') as f:
        data = json.load(f)

    # Handle nested structure
    segments_obj = data.get("segments", [])
    if isinstance(segments_obj, dict):
        segments = segments_obj.get("segments", [])
    else:
        segments = segments_obj

    # Convert to event dictionary format (similar to SportCLIP's events_detected)
    events_detected = {}
    for idx, segment in enumerate(segments):
        start_frame = segment["start_frame"]
        end_frame = segment["end_frame"]
        frames = list(range(start_frame, end_frame + 1))

        events_detected[idx] = {
            "frames": frames,
            "start": start_frame,
            "end": end_frame,
            "length": len(frames)
        }

    return events_detected


def ground_truth_dict_to_list(ground_truth_dict: Dict[int, str],
                               skip_uncertainty: bool = True) -> List[int]:
    """
    Convert ground truth dictionary to list format

    Args:
        ground_truth_dict: Dictionary mapping frame numbers to event types
        skip_uncertainty: Whether to skip uncertainty frames

    Returns:
        List of binary labels (0 = not highlight, 1 = highlight)
    """
    ground_truth_list = []

    for frame_num in sorted(ground_truth_dict.keys()):
        val = ground_truth_dict[frame_num]

        if val == "Uncertainty":
            if skip_uncertainty:
                continue
            else:
                ground_truth_list.append(0.5)
        elif val == "Not a highlight":
            ground_truth_list.append(0)
        elif val == "Highlight":
            ground_truth_list.append(1)

    return ground_truth_list


def compute_frame_level_results(ground_truth: List[int],
                                 events_detected: Dict[int, Dict]) -> Tuple[float, float, float]:
    """
    Compute frame-level recall, precision, and F-score

    This follows the same methodology as SportCLIP's computeFrameLevelResults function.

    Args:
        ground_truth: List of ground truth labels (0 or 1)
        events_detected: Dictionary of detected events with frame ranges

    Returns:
        Tuple of (recall, precision, fscore)
    """
    # Initialize variables
    tp = 0  # True positives
    tn = 0  # True negatives
    fp = 0  # False positives
    fn = 0  # False negatives
    tu = 0  # True uncertainty
    fu = 0  # False uncertainty

    # Convert detected events to a list
    events_frames = []
    for event_number, event_content in events_detected.items():
        events_frames.extend(event_content["frames"])

    # Create binary prediction list (same length as ground truth)
    events_list = [0 for _ in range(len(ground_truth))]
    for event_frame_idx in range(len(events_list)):
        # We need to map back to actual frame numbers
        # Since ground_truth is already a list, we just check if this index corresponds to a predicted frame
        pass

    # Actually, we need to handle this differently
    # Let's get the frame number mapping first
    # We'll create a set of predicted frame numbers for faster lookup
    predicted_frames_set = set(events_frames)

    # Now we need to know which frame numbers correspond to which indices in ground_truth
    # We'll need to pass frame numbers explicitly
    # Let's rewrite this function to handle it properly

    # For now, let's assume events_frames contains absolute frame numbers
    # and ground_truth list corresponds to consecutive frames starting from the first GT frame

    # Get the range of frames in ground truth
    # We'll need to modify this - let's use a different approach

    # Create a mapping: index -> frame_number for ground truth
    # We'll need the original ground_truth_dict for this
    raise NotImplementedError("This function needs ground_truth_dict to map indices to frames")


def compute_frame_level_results_v2(ground_truth_dict: Dict[int, str],
                                    events_detected: Dict[int, Dict]) -> Tuple[float, float, float]:
    """
    Compute frame-level recall, precision, and F-score

    This follows the same methodology as SportCLIP's computeFrameLevelResults function.

    Args:
        ground_truth_dict: Dictionary mapping frame numbers to event types
        events_detected: Dictionary of detected events with frame ranges

    Returns:
        Tuple of (recall, precision, fscore)
    """
    # Initialize variables
    tp = 0  # True positives
    tn = 0  # True negatives
    fp = 0  # False positives
    fn = 0  # False negatives
    tu = 0  # True uncertainty
    fu = 0  # False uncertainty

    # Convert detected events to a set of frame numbers
    predicted_frames_set = set()
    for event_number, event_content in events_detected.items():
        predicted_frames_set.update(event_content["frames"])

    # Get all frames from ground truth
    all_frames = sorted(ground_truth_dict.keys())

    # Compare ground truth vs predictions for each frame
    for frame_num in all_frames:
        gt_label = ground_truth_dict[frame_num]
        pred_label = 1 if frame_num in predicted_frames_set else 0

        # If ground truth is "Highlight"
        if gt_label == "Highlight":
            if pred_label == 1:
                tp += 1
            else:
                fn += 1
        # If ground truth is "Not a highlight"
        elif gt_label == "Not a highlight":
            if pred_label == 1:
                fp += 1
            else:
                tn += 1
        # If ground truth is "Uncertainty"
        elif gt_label == "Uncertainty":
            if pred_label == 1:
                tu += 1
            else:
                fu += 1

    # Convert uncertainty to whatever increases results (following SportCLIP's approach)
    tp += tu
    tn += fu

    # Compute metrics
    recall = tp / (tp + fn + 1e-7)
    precision = tp / (tp + fp + 1e-7)
    fscore = (2 * tp) / (2 * tp + fp + fn + 1e-7)

    return recall, precision, fscore


def evaluate_video(video_name: str,
                    model: str,
                    video_summaries_dir: str = "video_summaries",
                    ground_truth_dir: str = "/mnt/Data/mrt/OlympicHighlights_dataset",
                    verbose: bool = True) -> Dict:
    """
    Evaluate a single video's summarization results

    Args:
        video_name: Name of the video (without extension, e.g., 'highjump_video1')
        model: Model name ("sportclip" or "qddetr")
        video_summaries_dir: Directory containing video summaries
        ground_truth_dir: Directory containing ground truth with processed annotations
        verbose: Whether to print results

    Returns:
        Dictionary with evaluation metrics
    """
    # Construct paths
    model_dir = "SportCLIP" if model == "sportclip" else "QD-DETR"
    segments_path = os.path.join(video_summaries_dir, model_dir, video_name, "segments.json")

    # Parse sport and video number from video_name (e.g., "highjump_video1" -> "High Jump", "video1")
    sport_map = {
        "highjump": "High Jump",
        "longjump": "Long Jump",
        "polevault": "Pole Vault",
        "javelin": "Javelin",
        "triplejump": "Triple Jump",
        "100m": "100m"
    }

    # Extract sport prefix
    sport_prefix = video_name.rsplit('_', 1)[0].lower()
    video_id = video_name.rsplit('_', 1)[1]  # e.g., "video1"

    sport_name = sport_map.get(sport_prefix, sport_prefix.replace('_', ' ').title())

    # Look for ground truth file with processed annotations (includes uncertainty)
    gt_path = os.path.join(ground_truth_dir, sport_name, "ground_truth", f"{video_id}_processed.csv")

    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    if not os.path.exists(segments_path):
        raise FileNotFoundError(f"Segments file not found: {segments_path}")

    # Load data
    if verbose:
        print(f"\nEvaluating {video_name} with {model.upper()}...")
        print(f"  Ground truth: {gt_path}")
        print(f"  Predictions:  {segments_path}")

    ground_truth_dict = load_ground_truth(gt_path)
    events_detected = load_predicted_segments(segments_path)

    # Compute metrics
    recall, precision, fscore = compute_frame_level_results_v2(ground_truth_dict, events_detected)

    if verbose:
        print(f"\n  Results:")
        print(f"    Recall:    {recall*100:.2f}%")
        print(f"    Precision: {precision*100:.2f}%")
        print(f"    F-score:   {fscore*100:.2f}%")

    return {
        "video_name": video_name,
        "model": model,
        "recall": float(recall),
        "precision": float(precision),
        "fscore": float(fscore),
        "recall_pct": float(recall * 100),
        "precision_pct": float(precision * 100),
        "fscore_pct": float(fscore * 100)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video summarization models (SportCLIP and QD-DETR) using frame-level metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single video with SportCLIP
  python evaluate_video_summarization.py --video highjump_video1 --model sportclip

  # Evaluate a single video with QD-DETR
  python evaluate_video_summarization.py --video highjump_video1 --model qddetr

  # Evaluate all videos of a sport
  python evaluate_video_summarization.py --sport highjump --model sportclip
        """
    )

    # Video selection
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", type=str, help="Single video name to evaluate (e.g., highjump_video1)")
    group.add_argument("--sport", type=str, help="Sport name to evaluate all videos (e.g., highjump, longjump)")

    # Model selection
    parser.add_argument("--model", type=str, required=True, choices=["sportclip", "qddetr"],
                       help="Model to evaluate")

    # Directories
    parser.add_argument("--video-summaries-dir", type=str, default="video_summaries",
                       help="Directory containing video summaries (default: video_summaries)")
    parser.add_argument("--ground-truth-dir", type=str, default="/mnt/Data/mrt/OlympicHighlights_dataset",
                       help="Directory containing ground truth with processed CSV files (default: /mnt/Data/mrt/OlympicHighlights_dataset)")
    parser.add_argument("--output-dir", type=str, default="evaluation_results",
                       help="Directory to save evaluation results (default: evaluation_results)")

    # Options
    parser.add_argument("--quiet", action="store_true", help="Suppress verbose output")
    parser.add_argument("--append", action="store_true",
                       help="Append to existing results file instead of overwriting")

    args = parser.parse_args()

    # Determine which videos to evaluate
    videos_to_evaluate = []

    if args.video:
        videos_to_evaluate = [args.video]
    elif args.sport:
        # Map sport name to directory name
        sport_map = {
            "highjump": "High Jump",
            "longjump": "Long Jump",
            "polevault": "Pole Vault",
            "javelin": "Javelin",
            "triplejump": "Triple Jump",
            "100m": "100m"
        }

        sport_prefix = args.sport.lower()
        sport_name = sport_map.get(sport_prefix, sport_prefix.replace('_', ' ').title())

        # Find all processed CSV files for this sport
        gt_dir = Path(args.ground_truth_dir) / sport_name / "ground_truth"

        if not gt_dir.exists():
            print(f"Error: Ground truth directory not found: {gt_dir}")
            return

        # Find all processed CSV files (video1_processed.csv, video2_processed.csv, etc.)
        csv_files = sorted(gt_dir.glob("video*_processed.csv"))

        if not csv_files:
            print(f"Error: No processed CSV files found for sport '{args.sport}'")
            print(f"Searched in: {gt_dir}")
            return

        # Convert to video names (e.g., "highjump_video1")
        videos_to_evaluate = [f"{sport_prefix}_{f.stem.replace('_processed', '')}" for f in csv_files]

        print(f"Found {len(videos_to_evaluate)} videos for {args.sport}:")
        for v in videos_to_evaluate:
            print(f"  - {v}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each video
    results = []
    verbose = not args.quiet

    for video_name in videos_to_evaluate:
        try:
            result = evaluate_video(
                video_name=video_name,
                model=args.model,
                video_summaries_dir=args.video_summaries_dir,
                ground_truth_dir=args.ground_truth_dir,
                verbose=verbose
            )
            results.append(result)
        except Exception as e:
            print(f"\nError evaluating {video_name}: {e}")
            continue

    # Save results
    if results:
        # Save as JSON
        output_file = output_dir / f"{args.model}_evaluation_results.json"

        # If append mode and file exists, load existing results and append
        if args.append and output_file.exists():
            try:
                with open(output_file, 'r') as f:
                    existing_results = json.load(f)
                # Append new results
                existing_results.extend(results)
                results = existing_results
            except Exception as e:
                print(f"Warning: Could not load existing results: {e}")
                print("Overwriting file instead...")

        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n{'='*80}")
        print(f"SUMMARY - {args.model.upper()}")
        print(f"{'='*80}")

        # Compute averages
        avg_recall = np.mean([r["recall"] for r in results])
        avg_precision = np.mean([r["precision"] for r in results])
        avg_fscore = np.mean([r["fscore"] for r in results])

        print(f"\nEvaluated {len(results)} videos")
        print(f"\nAverage Metrics:")
        print(f"  Recall:    {avg_recall*100:.2f}%")
        print(f"  Precision: {avg_precision*100:.2f}%")
        print(f"  F-score:   {avg_fscore*100:.2f}%")

        print(f"\nPer-video results:")
        print(f"{'Video':<25} {'Recall':>10} {'Precision':>10} {'F-score':>10}")
        print(f"{'-'*25} {'-'*10} {'-'*10} {'-'*10}")
        for r in results:
            print(f"{r['video_name']:<25} {r['recall_pct']:>9.2f}% {r['precision_pct']:>9.2f}% {r['fscore_pct']:>9.2f}%")

        print(f"\nResults saved to: {output_file}")

        # Also save as CSV for easy analysis
        csv_file = output_dir / f"{args.model}_evaluation_results.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        print(f"CSV saved to: {csv_file}")

        # Generate detailed statistics summary
        print(f"\n{'='*80}")
        print(f"DETAILED STATISTICS")
        print(f"{'='*80}")

        # Add sport column to dataframe
        df['sport'] = df['video_name'].apply(lambda x: x.rsplit('_', 1)[0])

        # Compute per-sport statistics
        print(f"\nPER-SPORT STATISTICS:")
        print(f"{'-'*80}")
        print(f"{'Sport':<15} {'Metric':<12} {'Mean':>12} {'Std Dev':>12}")
        print(f"{'-'*15} {'-'*12} {'-'*12} {'-'*12}")

        sport_stats = {}
        for sport in sorted(df['sport'].unique()):
            sport_df = df[df['sport'] == sport]
            sport_display = sport.replace('_', ' ').title()

            # Store stats for later saving
            sport_stats[sport] = {
                'num_videos': len(sport_df),
                'recall': {
                    'mean': float(sport_df['recall'].mean()),
                    'std': float(sport_df['recall'].std()),
                    'mean_pct': float(sport_df['recall_pct'].mean()),
                    'std_pct': float(sport_df['recall_pct'].std())
                },
                'precision': {
                    'mean': float(sport_df['precision'].mean()),
                    'std': float(sport_df['precision'].std()),
                    'mean_pct': float(sport_df['precision_pct'].mean()),
                    'std_pct': float(sport_df['precision_pct'].std())
                },
                'fscore': {
                    'mean': float(sport_df['fscore'].mean()),
                    'std': float(sport_df['fscore'].std()),
                    'mean_pct': float(sport_df['fscore_pct'].mean()),
                    'std_pct': float(sport_df['fscore_pct'].std())
                }
            }

            # Print
            print(f"{sport_display:<15} {'Recall':<12} {sport_stats[sport]['recall']['mean_pct']:>11.2f}% {sport_stats[sport]['recall']['std_pct']:>11.2f}%")
            print(f"{'':15} {'Precision':<12} {sport_stats[sport]['precision']['mean_pct']:>11.2f}% {sport_stats[sport]['precision']['std_pct']:>11.2f}%")
            print(f"{'':15} {'F-score':<12} {sport_stats[sport]['fscore']['mean_pct']:>11.2f}% {sport_stats[sport]['fscore']['std_pct']:>11.2f}%")
            print()

        # Compute overall statistics with std dev
        print(f"OVERALL STATISTICS (All {len(df)} videos, {len(df['sport'].unique())} sports):")
        print(f"{'-'*80}")
        print(f"{'Metric':<15} {'Mean':>12} {'Std Dev':>12}")
        print(f"{'-'*15} {'-'*12} {'-'*12}")

        std_recall = np.std([r["recall_pct"] for r in results])
        std_precision = np.std([r["precision_pct"] for r in results])
        std_fscore = np.std([r["fscore_pct"] for r in results])

        print(f"{'Recall':<15} {avg_recall*100:>11.2f}% {std_recall:>11.2f}%")
        print(f"{'Precision':<15} {avg_precision*100:>11.2f}% {std_precision:>11.2f}%")
        print(f"{'F-score':<15} {avg_fscore*100:>11.2f}% {std_fscore:>11.2f}%")
        print()

        # Save summary statistics to JSON
        summary_stats = {
            'model': args.model,
            'num_videos': len(results),
            'num_sports': len(df['sport'].unique()),
            'by_sport': sport_stats,
            'overall': {
                'recall': {
                    'mean': float(avg_recall),
                    'std': float(std_recall / 100),
                    'mean_pct': float(avg_recall * 100),
                    'std_pct': float(std_recall)
                },
                'precision': {
                    'mean': float(avg_precision),
                    'std': float(std_precision / 100),
                    'mean_pct': float(avg_precision * 100),
                    'std_pct': float(std_precision)
                },
                'fscore': {
                    'mean': float(avg_fscore),
                    'std': float(std_fscore / 100),
                    'mean_pct': float(avg_fscore * 100),
                    'std_pct': float(std_fscore)
                }
            }
        }

        summary_file = output_dir / f"{args.model}_summary_statistics.json"
        with open(summary_file, 'w') as f:
            json.dump(summary_stats, f, indent=2)

        print(f"Summary statistics saved to: {summary_file}")

        # Save per-sport statistics to CSV
        sport_csv_data = []
        for sport, stats in sport_stats.items():
            sport_csv_data.append({
                'model': args.model,
                'sport': sport,
                'recall_mean': stats['recall']['mean_pct'],
                'recall_std': stats['recall']['std_pct'],
                'precision_mean': stats['precision']['mean_pct'],
                'precision_std': stats['precision']['std_pct'],
                'fscore_mean': stats['fscore']['mean_pct'],
                'fscore_std': stats['fscore']['std_pct'],
                'num_videos': stats['num_videos']
            })

        # Add overall row
        sport_csv_data.append({
            'model': args.model,
            'sport': 'OVERALL',
            'recall_mean': avg_recall * 100,
            'recall_std': std_recall,
            'precision_mean': avg_precision * 100,
            'precision_std': std_precision,
            'fscore_mean': avg_fscore * 100,
            'fscore_std': std_fscore,
            'num_videos': len(results)
        })

        sport_csv_file = output_dir / f"{args.model}_summary_statistics.csv"
        sport_df = pd.DataFrame(sport_csv_data)
        sport_df.to_csv(sport_csv_file, index=False)
        print(f"Per-sport statistics CSV saved to: {sport_csv_file}")

    else:
        print("\nNo results to save.")


if __name__ == "__main__":
    main()
