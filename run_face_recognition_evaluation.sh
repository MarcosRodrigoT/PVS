#!/bin/bash

# ============================================================================
# Face Recognition Evaluation Script
# ============================================================================
# This script evaluates ArcFace and TransFace face recognition models
# using ground truth highlight segments (independent of video summarization).
#
# Usage: ./run_face_recognition_evaluation.sh
# ============================================================================

set -e  # Exit on error

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Configuration
OUTPUT_DIR="evaluation_results/face_recognition"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/evaluation_log_${TIMESTAMP}.txt"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Face Recognition Evaluation (Ground Truth Highlights)${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "This script will evaluate ArcFace and TransFace on ground truth highlights"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of sports to evaluate (excluding 100m)
SPORTS=(
    "highjump"
    "javelin"
    "longjump"
    "polevault"
)

# Evaluate each sport
for sport in "${SPORTS[@]}"; do
    echo -e "${YELLOW}============================================================================${NC}"
    echo -e "${YELLOW}Sport: ${sport}${NC}"
    echo -e "${YELLOW}============================================================================${NC}"

    # Evaluate with original target
    echo -e "${GREEN}>>> Evaluating with original target${NC}"
    python evaluate_face_recognition.py \
        --sport "$sport" \
        --output-dir "$OUTPUT_DIR" \
        --threshold 0.6 \
        --smooth-window 200 \
        2>&1 | tee -a "$LOG_FILE"

    echo ""

    # Evaluate with updated target
    echo -e "${GREEN}>>> Evaluating with updated target${NC}"
    python evaluate_face_recognition.py \
        --sport "$sport" \
        --output-dir "$OUTPUT_DIR" \
        --threshold 0.6 \
        --smooth-window 200 \
        --use-updated-target \
        2>&1 | tee -a "$LOG_FILE"

    echo ""
done

# ============================================================================
# Aggregate results across all sports
# ============================================================================

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Aggregating Results Across All Sports${NC}"
echo -e "${BLUE}============================================================================${NC}"

# Create Python script to aggregate results
python - <<'EOF'
import json
from pathlib import Path
import numpy as np
from collections import defaultdict

output_dir = Path("evaluation_results/face_recognition")

# Find all result JSON files
result_files = list(output_dir.glob("*/gt_highlights_results*.json"))

if not result_files:
    print("No result files found!")
    exit(1)

print(f"Found {len(result_files)} result files\n")

# Separate by updated/original
original_results = []
updated_results = []

for result_file in result_files:
    with open(result_file) as f:
        data = json.load(f)

    if "_updated" in result_file.name:
        updated_results.append(data)
    else:
        original_results.append(data)

# Function to compute aggregate metrics
def compute_aggregate_metrics(results_list, label):
    print(f"\n{'='*80}")
    print(f"{label}")
    print(f"{'='*80}")

    for method in ["sequential", "instant"]:
        print(f"\n{method.upper()} Assignment Method:")

        # Aggregate ArcFace metrics
        arcface_metrics = []
        for result in results_list:
            if method in result.get("methods", {}):
                method_results = result["methods"][method]["arcface"]

                # Compute overall metrics for this video
                total_gt = sum([s["gt"] for s in method_results.values()])
                total_pred = sum([s["pred"] for s in method_results.values()])
                total_correct = sum([s["correct"] for s in method_results.values()])

                recall = total_correct / total_gt if total_gt > 0 else 0
                precision = total_correct / total_pred if total_pred > 0 else 0
                fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                arcface_metrics.append({
                    "video": result["video_name"],
                    "recall": recall,
                    "precision": precision,
                    "fscore": fscore
                })

        if arcface_metrics:
            avg_recall = np.mean([m["recall"] for m in arcface_metrics])
            avg_precision = np.mean([m["precision"] for m in arcface_metrics])
            avg_fscore = np.mean([m["fscore"] for m in arcface_metrics])

            print(f"\n  ArcFace (averaged across {len(arcface_metrics)} videos):")
            print(f"    Recall:    {avg_recall*100:.2f}%")
            print(f"    Precision: {avg_precision*100:.2f}%")
            print(f"    F-score:   {avg_fscore*100:.2f}%")

        # Aggregate TransFace metrics
        transface_metrics = []
        for result in results_list:
            if method in result.get("methods", {}):
                method_results = result["methods"][method]["transface"]

                # Compute overall metrics for this video
                total_gt = sum([s["gt"] for s in method_results.values()])
                total_pred = sum([s["pred"] for s in method_results.values()])
                total_correct = sum([s["correct"] for s in method_results.values()])

                recall = total_correct / total_gt if total_gt > 0 else 0
                precision = total_correct / total_pred if total_pred > 0 else 0
                fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                transface_metrics.append({
                    "video": result["video_name"],
                    "recall": recall,
                    "precision": precision,
                    "fscore": fscore
                })

        if transface_metrics:
            avg_recall = np.mean([m["recall"] for m in transface_metrics])
            avg_precision = np.mean([m["precision"] for m in transface_metrics])
            avg_fscore = np.mean([m["fscore"] for m in transface_metrics])

            print(f"\n  TransFace (averaged across {len(transface_metrics)} videos):")
            print(f"    Recall:    {avg_recall*100:.2f}%")
            print(f"    Precision: {avg_precision*100:.2f}%")
            print(f"    F-score:   {avg_fscore*100:.2f}%")

        # Comparison
        if arcface_metrics and transface_metrics:
            print(f"\n  Comparison (TransFace - ArcFace):")
            diff_recall = np.mean([m["recall"] for m in transface_metrics]) - np.mean([m["recall"] for m in arcface_metrics])
            diff_precision = np.mean([m["precision"] for m in transface_metrics]) - np.mean([m["precision"] for m in arcface_metrics])
            diff_fscore = np.mean([m["fscore"] for m in transface_metrics]) - np.mean([m["fscore"] for m in arcface_metrics])

            print(f"    Recall:    {diff_recall*100:+.2f}%")
            print(f"    Precision: {diff_precision*100:+.2f}%")
            print(f"    F-score:   {diff_fscore*100:+.2f}%")

# Compute aggregates
if original_results:
    compute_aggregate_metrics(original_results, "ORIGINAL TARGET RESULTS")

if updated_results:
    compute_aggregate_metrics(updated_results, "UPDATED TARGET RESULTS")

# Per-video breakdown
print(f"\n{'='*80}")
print(f"PER-VIDEO BREAKDOWN")
print(f"{'='*80}")

print(f"\n{'Video':<25} {'Method':<12} {'Model':<10} {'Recall':>10} {'Precision':>10} {'F-score':>10}")
print("-"*90)

for result in sorted(original_results + updated_results, key=lambda x: x["video_name"]):
    video_name = result["video_name"]
    use_updated = "Updated" if any(video_name in r["video_name"] for r in updated_results) else "Original"

    for method in ["sequential", "instant"]:
        if method in result.get("methods", {}):
            for model in ["arcface", "transface"]:
                method_results = result["methods"][method][model]

                # Compute overall metrics
                total_gt = sum([s["gt"] for s in method_results.values()])
                total_pred = sum([s["pred"] for s in method_results.values()])
                total_correct = sum([s["correct"] for s in method_results.values()])

                recall = total_correct / total_gt if total_gt > 0 else 0
                precision = total_correct / total_pred if total_pred > 0 else 0
                fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

                method_label = method[:3].upper()
                print(f"{video_name:<25} {method_label:<12} {model.upper():<10} {recall*100:>9.2f}% {precision*100:>9.2f}% {fscore*100:>9.2f}%")

print(f"\n")

# Save summary
summary = {
    "original": {},
    "updated": {}
}

for method in ["sequential", "instant"]:
    # Original
    if original_results:
        arcface_metrics = []
        transface_metrics = []

        for result in original_results:
            if method in result.get("methods", {}):
                # ArcFace
                method_results = result["methods"][method]["arcface"]
                total_gt = sum([s["gt"] for s in method_results.values()])
                total_pred = sum([s["pred"] for s in method_results.values()])
                total_correct = sum([s["correct"] for s in method_results.values()])
                recall = total_correct / total_gt if total_gt > 0 else 0
                precision = total_correct / total_pred if total_pred > 0 else 0
                fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                arcface_metrics.append({"recall": recall, "precision": precision, "fscore": fscore})

                # TransFace
                method_results = result["methods"][method]["transface"]
                total_gt = sum([s["gt"] for s in method_results.values()])
                total_pred = sum([s["pred"] for s in method_results.values()])
                total_correct = sum([s["correct"] for s in method_results.values()])
                recall = total_correct / total_gt if total_gt > 0 else 0
                precision = total_correct / total_pred if total_pred > 0 else 0
                fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                transface_metrics.append({"recall": recall, "precision": precision, "fscore": fscore})

        if arcface_metrics:
            summary["original"][method] = {
                "arcface": {
                    "recall": float(np.mean([m["recall"] for m in arcface_metrics])),
                    "precision": float(np.mean([m["precision"] for m in arcface_metrics])),
                    "fscore": float(np.mean([m["fscore"] for m in arcface_metrics]))
                },
                "transface": {
                    "recall": float(np.mean([m["recall"] for m in transface_metrics])),
                    "precision": float(np.mean([m["precision"] for m in transface_metrics])),
                    "fscore": float(np.mean([m["fscore"] for m in transface_metrics]))
                }
            }

    # Updated
    if updated_results:
        arcface_metrics = []
        transface_metrics = []

        for result in updated_results:
            if method in result.get("methods", {}):
                # ArcFace
                method_results = result["methods"][method]["arcface"]
                total_gt = sum([s["gt"] for s in method_results.values()])
                total_pred = sum([s["pred"] for s in method_results.values()])
                total_correct = sum([s["correct"] for s in method_results.values()])
                recall = total_correct / total_gt if total_gt > 0 else 0
                precision = total_correct / total_pred if total_pred > 0 else 0
                fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                arcface_metrics.append({"recall": recall, "precision": precision, "fscore": fscore})

                # TransFace
                method_results = result["methods"][method]["transface"]
                total_gt = sum([s["gt"] for s in method_results.values()])
                total_pred = sum([s["pred"] for s in method_results.values()])
                total_correct = sum([s["correct"] for s in method_results.values()])
                recall = total_correct / total_gt if total_gt > 0 else 0
                precision = total_correct / total_pred if total_pred > 0 else 0
                fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                transface_metrics.append({"recall": recall, "precision": precision, "fscore": fscore})

        if arcface_metrics:
            summary["updated"][method] = {
                "arcface": {
                    "recall": float(np.mean([m["recall"] for m in arcface_metrics])),
                    "precision": float(np.mean([m["precision"] for m in arcface_metrics])),
                    "fscore": float(np.mean([m["fscore"] for m in arcface_metrics]))
                },
                "transface": {
                    "recall": float(np.mean([m["recall"] for m in transface_metrics])),
                    "precision": float(np.mean([m["precision"] for m in transface_metrics])),
                    "fscore": float(np.mean([m["fscore"] for m in transface_metrics]))
                }
            }

# Save summary to file
summary_file = output_dir / "summary_all_sports.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print(f"Summary saved to: {summary_file}")

EOF

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}EVALUATION COMPLETE${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
echo "Generated plots:"
echo "  - gt_highlights_assigned.png (sequential method, original target)"
echo "  - gt_highlights_instant_assigned.png (instant method, original target)"
echo "  - gt_highlights_assigned_updated.png (sequential method, updated target)"
echo "  - gt_highlights_instant_assigned_updated.png (instant method, updated target)"
echo ""
