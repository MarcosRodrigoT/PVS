#!/bin/bash

# ============================================================================
# Computational Cost Measurement Script
# ============================================================================
# This script runs individual component measurements for the PVS pipeline.
# Simply comment out (with #) the components you don't want to measure.
# ============================================================================

set -e

# Activate virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# ============================================================================
# COMPONENT SELECTION - Comment out lines you don't want to run
# ============================================================================

# Video summarization components
RUN_SPORTCLIP=true         # SportCLIP
RUN_QDDETR=true            # QD-DETR

# Face recognition components
RUN_FACE_DETECTION=true    # Face detection
RUN_ARCFACE=true           # ArcFace embeddings
RUN_TRANSFACE=true         # TransFace embeddings

# Shared components
RUN_RESEMBLANCE=true       # Resemblance computation
RUN_CLIP_ASSIGNMENT=true   # Clip assignment

# ============================================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================================

VIDEO_PATH="data/highjump_video1.mp4"

# Convert bash booleans to Python
[ "$RUN_SPORTCLIP" = "true" ] && PY_RUN_SPORTCLIP="True" || PY_RUN_SPORTCLIP="False"
[ "$RUN_QDDETR" = "true" ] && PY_RUN_QDDETR="True" || PY_RUN_QDDETR="False"
[ "$RUN_FACE_DETECTION" = "true" ] && PY_RUN_FACE_DETECTION="True" || PY_RUN_FACE_DETECTION="False"
[ "$RUN_ARCFACE" = "true" ] && PY_RUN_ARCFACE="True" || PY_RUN_ARCFACE="False"
[ "$RUN_TRANSFACE" = "true" ] && PY_RUN_TRANSFACE="True" || PY_RUN_TRANSFACE="False"
[ "$RUN_RESEMBLANCE" = "true" ] && PY_RUN_RESEMBLANCE="True" || PY_RUN_RESEMBLANCE="False"
[ "$RUN_CLIP_ASSIGNMENT" = "true" ] && PY_RUN_CLIP_ASSIGNMENT="True" || PY_RUN_CLIP_ASSIGNMENT="False"

# Build the measurement script with selected components
PYTHON_SCRIPT="
import sys
sys.path.insert(0, '.')

from measure_all_components import *
import shutil
import tempfile

video_path = '$VIDEO_PATH'
video_name = 'highjump_video1'
temp_dir = Path(tempfile.mkdtemp(prefix='pvs_measurement_'))

try:
    results = {}

    if $PY_RUN_SPORTCLIP:
        results['sportclip_per_frame_ms'] = measure_sportclip(video_path, video_name, temp_dir)

    if $PY_RUN_QDDETR:
        results['qddetr_per_frame_ms'] = measure_qddetr(video_path, video_name, temp_dir)

    if $PY_RUN_FACE_DETECTION:
        face_det_ms, avg_faces = measure_face_detection(video_path)
        results['face_detection_per_frame_ms'] = face_det_ms
        results['avg_faces_per_frame'] = avg_faces

    if $PY_RUN_ARCFACE:
        results['arcface_per_face_ms'] = measure_arcface(video_path, temp_dir)

    if $PY_RUN_TRANSFACE:
        results['transface_per_face_ms'] = measure_transface(video_path, temp_dir)

    if $PY_RUN_RESEMBLANCE:
        results['resemblance_per_comparison_ms'] = measure_resemblance()

    if $PY_RUN_CLIP_ASSIGNMENT:
        results['clip_assignment_per_segment_ms'] = measure_clip_assignment()

    # Display results
    print(f\"\n\n{'='*80}\")
    print('RESULTS')
    print(f\"{'='*80}\")
    for key, value in results.items():
        print(f'{key}: {value:.4f}')
    print(f\"{'='*80}\")

    # Save results to JSON
    import json
    results_file = 'measurement_results.json'
    if Path(results_file).exists():
        with open(results_file, 'r') as f:
            existing = json.load(f)
        existing.update(results)
        results = existing

    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f\"\n✓ Results saved to: {results_file}\")

    # Generate outputs if all components are measured
    if all([
        'sportclip_per_frame_ms' in results,
        'qddetr_per_frame_ms' in results,
        'face_detection_per_frame_ms' in results,
        'arcface_per_face_ms' in results,
        'transface_per_face_ms' in results,
        'resemblance_per_comparison_ms' in results,
        'clip_assignment_per_segment_ms' in results,
        'avg_faces_per_frame' in results
    ]):
        generate_measurement_report(results, 'measurements_report.md')
        generate_latex_table(results, 'table.tex')
    else:
        print(\"\nNote: Not all components measured yet. Run all components to generate outputs\")

finally:
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
"

echo "================================================================================"
echo "COMPUTATIONAL COST MEASUREMENT"
echo "================================================================================"
echo ""
echo "Components to measure:"
[ "$RUN_SPORTCLIP" = "true" ] && echo "  ✓ SportCLIP"
[ "$RUN_QDDETR" = "true" ] && echo "  ✓ QD-DETR"
[ "$RUN_FACE_DETECTION" = "true" ] && echo "  ✓ Face Detection"
[ "$RUN_ARCFACE" = "true" ] && echo "  ✓ ArcFace"
[ "$RUN_TRANSFACE" = "true" ] && echo "  ✓ TransFace"
[ "$RUN_RESEMBLANCE" = "true" ] && echo "  ✓ Resemblance"
[ "$RUN_CLIP_ASSIGNMENT" = "true" ] && echo "  ✓ Clip Assignment"
echo ""

# Run measurements with CUDA support
./run_with_cuda.sh python -c "$PYTHON_SCRIPT"

echo ""
echo "================================================================================"
echo "MEASUREMENT COMPLETE"
echo "================================================================================"
