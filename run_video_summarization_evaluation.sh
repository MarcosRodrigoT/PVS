#!/bin/bash

# ============================================================================
# Video Summarization Evaluation Script
# ============================================================================
# This script evaluates both SportCLIP and QD-DETR video summarization models
# on all Olympic Highlights videos (excluding 100m) using frame-level metrics.
#
# Usage: ./run_video_summarization_evaluation.sh
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
OUTPUT_DIR="evaluation_results/video_summarization"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${OUTPUT_DIR}/evaluation_log_${TIMESTAMP}.txt"

echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Video Summarization Evaluation${NC}"
echo -e "${BLUE}============================================================================${NC}"
echo ""
echo "This script will evaluate SportCLIP and QD-DETR on all Olympic Highlights videos"
echo "Output directory: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# List of sports to evaluate (excluding 100m and triple jump)
SPORTS=(
    "highjump"
    "javelin"
    "longjump"
    "polevault"
)

# Collect all videos from all sports
echo "Collecting videos from all sports..."
ALL_VIDEOS=()
for sport in "${SPORTS[@]}"; do
    # Get videos for this sport
    videos=$(python -c "
from pathlib import Path
import sys

sport_map = {
    'highjump': 'High Jump',
    'longjump': 'Long Jump',
    'polevault': 'Pole Vault',
    'javelin': 'Javelin',
}

sport_prefix = '$sport'
sport_name = sport_map.get(sport_prefix, sport_prefix.replace('_', ' ').title())
gt_dir = Path('/mnt/Data/mrt/OlympicHighlights_dataset') / sport_name / 'ground_truth'

if gt_dir.exists():
    csv_files = sorted(gt_dir.glob('video*_processed.csv'))
    for f in csv_files:
        video_name = f'{sport_prefix}_{f.stem.replace(\"_processed\", \"\")}'
        print(video_name)
" 2>/dev/null)

    for video in $videos; do
        ALL_VIDEOS+=("$video")
    done
done

echo "Found ${#ALL_VIDEOS[@]} videos total:"
for v in "${ALL_VIDEOS[@]}"; do
    echo "  - $v"
done
echo ""

# Evaluate all videos with SportCLIP
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}Evaluating SportCLIP on all videos${NC}"
echo -e "${GREEN}============================================================================${NC}"

# Remove old results file to start fresh
rm -f "$OUTPUT_DIR/sportclip_evaluation_results.json"
rm -f "$OUTPUT_DIR/sportclip_evaluation_results.csv"

first_video=true
for video in "${ALL_VIDEOS[@]}"; do
    if [ "$first_video" = true ]; then
        # First video - create new file
        python evaluate_video_summarization.py \
            --video "$video" \
            --model sportclip \
            --output-dir "$OUTPUT_DIR" \
            --quiet \
            2>&1 | tee -a "$LOG_FILE"
        first_video=false
    else
        # Subsequent videos - append to file
        python evaluate_video_summarization.py \
            --video "$video" \
            --model sportclip \
            --output-dir "$OUTPUT_DIR" \
            --quiet \
            --append \
            2>&1 | tee -a "$LOG_FILE"
    fi
done

# Generate full summary with statistics for SportCLIP (re-run to show detailed stats)
echo ""
echo -e "${GREEN}Generating SportCLIP Summary Statistics...${NC}"
python evaluate_video_summarization.py \
    --sport highjump \
    --model sportclip \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# Append other sports quietly to complete the dataset
for sport in "javelin" "longjump" "polevault"; do
    python evaluate_video_summarization.py \
        --sport "$sport" \
        --model sportclip \
        --output-dir "$OUTPUT_DIR" \
        --quiet \
        --append \
        2>&1 | tee -a "$LOG_FILE"
done

echo ""

# Evaluate all videos with QD-DETR
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}Evaluating QD-DETR on all videos${NC}"
echo -e "${GREEN}============================================================================${NC}"

# Remove old results file to start fresh
rm -f "$OUTPUT_DIR/qddetr_evaluation_results.json"
rm -f "$OUTPUT_DIR/qddetr_evaluation_results.csv"
rm -f "$OUTPUT_DIR/qddetr_summary_statistics.json"
rm -f "$OUTPUT_DIR/qddetr_summary_statistics.csv"

first_video=true
for video in "${ALL_VIDEOS[@]}"; do
    if [ "$first_video" = true ]; then
        # First video - create new file
        python evaluate_video_summarization.py \
            --video "$video" \
            --model qddetr \
            --output-dir "$OUTPUT_DIR" \
            --quiet \
            2>&1 | tee -a "$LOG_FILE"
        first_video=false
    else
        # Subsequent videos - append to file
        python evaluate_video_summarization.py \
            --video "$video" \
            --model qddetr \
            --output-dir "$OUTPUT_DIR" \
            --quiet \
            --append \
            2>&1 | tee -a "$LOG_FILE"
    fi
done

# Generate full summary with statistics for QD-DETR (re-run to show detailed stats)
echo ""
echo -e "${GREEN}Generating QD-DETR Summary Statistics...${NC}"
python evaluate_video_summarization.py \
    --sport highjump \
    --model qddetr \
    --output-dir "$OUTPUT_DIR" \
    2>&1 | tee -a "$LOG_FILE"

# Append other sports quietly to complete the dataset
for sport in "javelin" "longjump" "polevault"; do
    python evaluate_video_summarization.py \
        --sport "$sport" \
        --model qddetr \
        --output-dir "$OUTPUT_DIR" \
        --quiet \
        --append \
        2>&1 | tee -a "$LOG_FILE"
done

# ============================================================================
# Cross-model comparison summary
# ============================================================================

echo ""
echo -e "${BLUE}============================================================================${NC}"
echo -e "${BLUE}Cross-Model Comparison${NC}"
echo -e "${BLUE}============================================================================${NC}"

python - <<'EOF'
import json
import pandas as pd
from pathlib import Path
import numpy as np

output_dir = Path("evaluation_results/video_summarization")

# Load results for both models
sportclip_file = output_dir / "sportclip_evaluation_results.json"
qddetr_file = output_dir / "qddetr_evaluation_results.json"

if sportclip_file.exists() and qddetr_file.exists():
    with open(sportclip_file) as f:
        sportclip_results = json.load(f)

    with open(qddetr_file) as f:
        qddetr_results = json.load(f)

    # Extract sport name from video name
    for r in sportclip_results:
        r['sport'] = r['video_name'].rsplit('_', 1)[0]

    for r in qddetr_results:
        r['sport'] = r['video_name'].rsplit('_', 1)[0]

    # Convert to DataFrames
    df_sportclip = pd.DataFrame(sportclip_results)
    df_qddetr = pd.DataFrame(qddetr_results)

    # Compute per-sport averages
    print("\n" + "="*80)
    print("PER-SPORT AVERAGES")
    print("="*80)

    print("\nSportCLIP:")
    print("-"*80)
    sportclip_by_sport = df_sportclip.groupby('sport')[['recall_pct', 'precision_pct', 'fscore_pct']].mean()
    print(sportclip_by_sport.to_string())

    print("\n\nQD-DETR:")
    print("-"*80)
    qddetr_by_sport = df_qddetr.groupby('sport')[['recall_pct', 'precision_pct', 'fscore_pct']].mean()
    print(qddetr_by_sport.to_string())

    # Overall averages
    print("\n" + "="*80)
    print("OVERALL AVERAGES (All Sports)")
    print("="*80)

    print(f"\nSportCLIP:")
    print(f"  Recall:    {df_sportclip['recall_pct'].mean():.2f}%")
    print(f"  Precision: {df_sportclip['precision_pct'].mean():.2f}%")
    print(f"  F-score:   {df_sportclip['fscore_pct'].mean():.2f}%")

    print(f"\nQD-DETR:")
    print(f"  Recall:    {df_qddetr['recall_pct'].mean():.2f}%")
    print(f"  Precision: {df_qddetr['precision_pct'].mean():.2f}%")
    print(f"  F-score:   {df_qddetr['fscore_pct'].mean():.2f}%")

    # Save aggregated results
    summary = {
        "sportclip": {
            "overall": {
                "recall": float(df_sportclip['recall'].mean()),
                "precision": float(df_sportclip['precision'].mean()),
                "fscore": float(df_sportclip['fscore'].mean())
            },
            "by_sport": sportclip_by_sport.to_dict()
        },
        "qddetr": {
            "overall": {
                "recall": float(df_qddetr['recall'].mean()),
                "precision": float(df_qddetr['precision'].mean()),
                "fscore": float(df_qddetr['fscore'].mean())
            },
            "by_sport": qddetr_by_sport.to_dict()
        }
    }

    summary_file = output_dir / "summary_all_sports.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n\nSummary saved to: {summary_file}")

    # Create comparison table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(f"\n{'Metric':<15} {'SportCLIP':>12} {'QD-DETR':>12} {'Difference':>12}")
    print("-"*80)

    sportclip_avg = df_sportclip[['recall_pct', 'precision_pct', 'fscore_pct']].mean()
    qddetr_avg = df_qddetr[['recall_pct', 'precision_pct', 'fscore_pct']].mean()

    metrics = [('Recall', 'recall_pct'), ('Precision', 'precision_pct'), ('F-score', 'fscore_pct')]
    for name, col in metrics:
        sc_val = sportclip_avg[col]
        qd_val = qddetr_avg[col]
        diff = sc_val - qd_val
        sign = "+" if diff > 0 else ""
        print(f"{name:<15} {sc_val:>11.2f}% {qd_val:>11.2f}% {sign}{diff:>11.2f}%")

    print("\n")
else:
    print("Error: Could not find result files")
    print(f"  SportCLIP: {sportclip_file}")
    print(f"  QD-DETR: {qddetr_file}")
EOF

echo ""
echo -e "${GREEN}============================================================================${NC}"
echo -e "${GREEN}EVALUATION COMPLETE${NC}"
echo -e "${GREEN}============================================================================${NC}"
echo ""
echo "Results saved to: $OUTPUT_DIR"
echo "Log file: $LOG_FILE"
echo ""
