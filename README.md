# Personalized Video Summarization (PVS)

A pipeline for generating personalized video summaries by detecting and recognizing specific people in video content.

## Overview

The PVS pipeline has been redesigned to run components more independently to better handle sports videos where athletes may not be facing the camera during highlight moments.

### New Approach

1. **Video Summarization** - Generate highlight clips using SportCLIP or QD-DETR (runs independently)
2. **Full Video Face Detection** - Detect and extract faces from ALL frames of original videos (not just summaries)
3. **Face Embedding Extraction** - Extract embeddings from all face crops using ArcFace and TransFace
4. **Two-Pass Face Recognition**:
   - First pass: Compare target face against all crops to find best match (updated target)
   - Second pass: Compare updated target against all crops to build resemblance curves
5. **Score-Based Filtering** - Keep video summaries that show high resemblance scores

This approach allows construction of resemblance score curves across entire videos, enabling identification of when and where the target person appears.

### Supported Video Summarization Methods

- **SportCLIP**: Sports highlight detection using text-video matching with CLIP embeddings
- **QD-DETR**: Query-based moment retrieval using transformer-based temporal localization

### Supported Face Recognition Models

- **ArcFace**: State-of-the-art CNN-based face recognition (via DeepFace)
- **TransFace**: Transformer-based face recognition model

## Project Structure

```
PVS/
├── main.py                       # Complete pipeline orchestrator
├── generate_video_summaries.py   # Standalone video summarization script
├── extract_all_faces.py          # Extract faces from all video frames (saves metadata)
├── extract_embeddings.py         # Extract embeddings from face crops
├── find_updated_target.py        # Find best matching face as updated target
├── compute_resemblance_curve.py  # Compute and visualize resemblance scores
├── create_final_visualization.py # Create enhanced plots with highlight overlays
├── config.py                     # Configuration settings
├── visualizer.py                 # Visualization component
├── requirements.txt              # Python dependencies
├── face_detection/
│   └── detector.py               # Face detection using InsightFace SCRFD
├── face_recognition/
│   └── recognizer.py             # Face recognition using ArcFace/TransFace
├── video_summarization/
│   ├── sportclip_summarizer.py   # SportCLIP integration
│   ├── qddetr_summarizer.py      # QD-DETR integration
│   └── qddetr_wrapper.py         # QD-DETR wrapper script
├── data/                         # Input videos
├── target_faces/                 # Place target person face images here
├── video_summaries/              # NEW: Organized video summaries (cached)
│   ├── SportCLIP/                # SportCLIP summaries
│   │   └── <video_name>/
│   │       ├── clip_001.mp4
│   │       ├── clip_002.mp4
│   │       └── segments.json     # Segment metadata
│   └── QD-DETR/                  # QD-DETR summaries
│       └── <video_name>/
│           ├── clip_001.mp4
│           └── segments.json
├── cropped_faces/                # Extracted face crops (cached, shared)
│   ├── <video_name>/
│   │   ├── frame_000001_face_0.jpg
│   │   └── ...
│   └── face_detections.json      # Detection metadata (coordinates, landmarks, confidence)
├── cropped_faces_embeddings/     # Face embeddings in HDF5 format (cached, shared)
│   ├── ArcFace/                  # ArcFace embeddings (<video>.h5)
│   └── TransFace/                # TransFace embeddings (<video>.h5)
├── target_faces_updated/         # Updated target faces (best matches)
│   └── <target_name>/            # One directory per target
│       ├── target_face.jpg       # Updated target face image
│       ├── target_arcface_embedding.npy
│       └── target_transface_embedding.npy
├── resemblance_results/          # Resemblance curve analysis
│   └── <target_name>/            # One directory per target
│       └── <video_name>/         # One directory per video
│           ├── *_resemblance_curve.png
│           └── *_resemblance_data.json
└── results/                      # Final results directory
    └── <video_name>/             # Per-video results
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Place target face images in the `target_faces/` directory:
   - Supported formats: .jpg, .png, .jpeg
   - One or more face images per target person
   - Clear, frontal face images work best

3. Place input videos in `data/`:
   - Supported formats: .mp4, .avi, .mov, .mkv

## Quick Start - Complete Pipeline

### Option 1: Run Complete Pipeline (Recommended)

Run the entire pipeline on a single video using `main.py`:

```bash
# With SportCLIP
python main.py --video longjump_video1 \
    --video-summarizer sportclip \
    --sentences-file data/sentences/long_jump.json

# With QD-DETR
python main.py --video highjump_video1 \
    --video-summarizer qddetr \
    --qddetr-query "highlight moments in this high jump competition"
```

This single command will:
1. Generate video summaries (saved to `video_summaries/`)
2. Extract faces from entire video (saved to `cropped_faces/`)
3. Extract embeddings (saved to `cropped_faces_embeddings/`)
4. Find updated targets (saved to `target_faces_updated/`)
5. Compute resemblance curves (saved to `resemblance_results/`)
6. Create final visualizations with highlight overlays (saved to `results/<video>/<target>/`)

**Key Features:**
- **Smart caching**: Automatically skips steps that have already been completed
- **Fast iteration**: Re-run on same video to skip expensive face extraction
- **Shared resources**: Face crops and embeddings are cached at project root for reuse across experiments

### Option 2: Run Steps Individually

For more control, run each step separately:

## Individual Workflow Steps

### Step 1: Generate Video Summaries (Optional First)

Generate highlight clips before or after face extraction:

```bash
# Process all videos with SportCLIP
python generate_video_summaries.py --model sportclip \
    --sentences-file data/sentences/long_jump.json

# Process specific video with QD-DETR
python generate_video_summaries.py --model qddetr \
    --video highjump_video1 \
    --qddetr-query "highlight moments in this sports video"
```

Output: `video_summaries/<model>/<video_name>/`

### Step 2: Extract Faces from All Videos

Extract face crops from every frame of all videos in the data directory:

```bash
python extract_all_faces.py
```

Options:
```bash
# Filter out faces smaller than 50x50 pixels
python extract_all_faces.py --min-face-size 50

# Custom directories
python extract_all_faces.py --data-dir /path/to/videos --output-dir /path/to/output
```

Output structure:
```
cropped_faces/
├── 100m_video1/
│   ├── frame_000241_face_0.jpg
│   ├── frame_000241_face_1.jpg
│   └── ...
├── 100m_video2/
├── face_detections.json  # NEW: Metadata with landmarks, coordinates, confidence
└── ...
```

The `face_detections.json` file contains detailed metadata for each detected face:
- Frame number and video name
- Bounding box coordinates (x1, y1, x2, y2)
- Detection confidence score
- **Facial landmarks** (5 keypoints: left_eye, right_eye, nose, left_mouth, right_mouth)
- Path to the cropped face image

This metadata is useful for face alignment, quality assessment, and advanced analysis.

### Step 2: Extract Embeddings

Extract face embeddings from all crops using ArcFace and/or TransFace:

```bash
python extract_embeddings.py
```

Options:
```bash
# Extract only ArcFace embeddings
python extract_embeddings.py --models ArcFace

# Extract only TransFace embeddings
python extract_embeddings.py --models TransFace

# Use specific GPU for ArcFace
python extract_embeddings.py --models ArcFace --gpu-id 1

# Custom TransFace batch size
python extract_embeddings.py --models TransFace --batch-size 2000
```

Features:
- **Progressive saving**: Saves embeddings incrementally (every 1000) to avoid memory issues
- **Resume support**: If stopped, just run again and it skips already processed faces
- **HDF5 format**: Efficient storage with one .h5 file per video

Output structure:
```
cropped_faces_embeddings/
├── ArcFace/
│   ├── 100m_video1.h5
│   ├── 100m_video2.h5
│   └── ...
└── TransFace/
    ├── 100m_video1.h5
    └── ...
```

Loading embeddings:
```python
import h5py

with h5py.File('cropped_faces_embeddings/ArcFace/100m_video1.h5', 'r') as f:
    # Get a specific embedding
    embedding = f['frame_000241_face_0'][:]

    # List all available embeddings
    all_keys = list(f.keys())
```

### Step 3: Find Updated Target Face

Compare your target face against all face crops to find the best matching face (updated target):

```bash
python find_updated_target.py --target athlete1 --video highjump_video1
```

Options:
```bash
# Process different video
python find_updated_target.py --target athlete1 --video longjump_video1

# Adjust similarity threshold (default: 0.5)
python find_updated_target.py --target athlete1 --video highjump_video1 --threshold 0.6

# Show top 20 matches in log
python find_updated_target.py --target athlete1 --video highjump_video1 --top-k 20
```

This creates an "updated target" by finding the face crop with the highest similarity to your original target. The updated target is used for more accurate resemblance curve computation.

Output structure:
```
target_faces_updated/
└── athlete1/
    ├── target_face.jpg                    # Best matching face image
    ├── target_arcface_embedding.npy       # ArcFace embedding
    ├── target_transface_embedding.npy     # TransFace embedding
    └── athlete1_highjump_video1_matching_log.txt  # Match details
```

### Step 4: Compute Resemblance Curves

Generate resemblance score curves showing when and where the target person appears in the video:

```bash
# Process all targets for default video (highjump_video1)
python compute_resemblance_curve.py

# Process all targets for a different video
python compute_resemblance_curve.py --video longjump_video1

# Process specific target only
python compute_resemblance_curve.py --target athlete1 --video highjump_video1
```

Options:
```bash
# Adjust similarity threshold for visualization (default: 0.6)
python compute_resemblance_curve.py --threshold 0.65

# Adjust smoothing window for curves (default: 5)
python compute_resemblance_curve.py --smooth-window 7

# Custom directories
python compute_resemblance_curve.py --video highjump_video1 \
                                    --updated-targets custom/updated \
                                    --embeddings custom/embeddings
```

Features:
- **Automatic multi-target processing**: Processes all targets in `target_faces_updated/` by default
- **Dual model comparison**: Shows both ArcFace and TransFace similarity curves
- **Smoothed visualization**: Displays raw scores (faded) with smoothed overlay for clarity
- **Accurate timestamps**: Uses ffprobe to detect video FPS for precise timing
- **Adaptive x-axis**: Automatically adjusts tick granularity based on video duration

Output structure:
```
resemblance_results/
└── athlete1/
    └── highjump_video1/
        ├── athlete1_highjump_video1_resemblance_curve.png  # Visualization
        └── athlete1_highjump_video1_resemblance_data.json  # Raw data
```

The plot shows:
- Raw resemblance scores (faded lines)
- Smoothed scores using rolling average (prominent lines)
- Threshold line (default: 0.6)
- Highlighted regions above threshold
- Statistics (mean, max, frames above threshold)

### Step 5: Create Final Visualizations with Highlights

Generate enhanced resemblance curves with highlight segments overlaid on the timeline, and copy relevant clips:

```bash
# Process all targets for a video
python create_final_visualization.py --video highjump_video1 --model sportclip

# Process specific target only
python create_final_visualization.py --video highjump_video1 --model sportclip --target barshim

# Custom threshold
python create_final_visualization.py --video highjump_video1 --model sportclip --threshold 0.65
```

This script:
- Creates enhanced resemblance curve plots with highlight segments marked on the timeline (red boxes labeled "H1", "H2", etc.)
- Analyzes which highlight clips contain the target person (based on **raw** resemblance scores, not smoothed)
- Copies only relevant highlight clips to `results/<video>/<target>/highlight_clips/`
- Generates a summary report with statistics

**Important**: The script uses **raw (unsmoothed) scores** for filtering clips to avoid false negatives. Smoothing reduces peak values and can cause the system to miss clips where the target appears.

Output structure:
```
results/
└── highjump_video1/
    └── barshim/
        ├── barshim_highjump_video1_enhanced_curve.png  # Plot with highlights
        ├── barshim_highjump_video1_summary.json        # Summary report
        └── highlight_clips/                            # Relevant clips only
            ├── highjump_video1_highlight_007.mp4
            └── highjump_video1_highlight_021.mp4
```

The enhanced plot shows:
- Raw and smoothed resemblance scores (both displayed for visualization)
- **Red rectangular overlays** marking highlight segment positions on the timeline
- **Red labels** ("H1", "H2", etc.) identifying each highlight
- Threshold line and above-threshold regions
- Summary statistics including number of highlight segments

**Note**: Highlight segment timing is extracted directly from SportCLIP's or QD-DETR's event detection, ensuring accurate frame-level alignment with the original video.

## Models Used

### Face Detection: InsightFace SCRFD
- State-of-the-art single-stage face detector
- High accuracy in challenging conditions
- Provides facial landmarks and confidence scores

### Face Recognition

#### ArcFace (via DeepFace)
- State-of-the-art CNN-based face recognition
- 99.63% accuracy on LFW dataset
- Robust to variations in lighting and pose

#### TransFace
- Transformer-based face recognition model
- Better handling of partial occlusions
- Requires separate installation at `/home/mrt/Projects/TransFace`

## Command-line Arguments Reference

### main.py (Complete Pipeline)

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | (required) | Video name to process (without extension) |
| `--video-summarizer` | `sportclip` | Video summarization method (sportclip or qddetr) |
| `--data-dir` | `data` | Directory containing video files |
| `--target-faces` | `target_faces` | Directory containing target face images |
| `--output` | `results` | Output directory for final results |
| `--sentences-file` | (required for sportclip) | Path to SportCLIP sentences file |
| `--sportclip-dir` | `/home/mrt/Projects/SportCLIP-official` | SportCLIP project directory |
| `--sportclip-dataset-dir` | `/mnt/Data/mrt/SportCLIP-OlympicHighlights` | SportCLIP dataset directory |
| `--qddetr-ckpt` | (default path) | Path to QD-DETR checkpoint |
| `--qddetr-query` | (required for qddetr) | Query for QD-DETR |
| `--qddetr-device` | `cuda:0` | Device for QD-DETR inference |
| `--qddetr-context-window` | `600` | Context window size in frames |
| `--qddetr-min-duration` | `15` | Minimum event duration in frames |
| `--qddetr-min-area` | `15.0` | Minimum event area threshold |
| `--transface-weight` | (default path) | Path to TransFace model weights |
| `--transface-network` | `vit_l_dp005_mask_005` | TransFace network architecture |
| `--similarity-threshold` | `0.5` | Similarity threshold for finding updated targets |
| `--resemblance-threshold` | `0.6` | Resemblance threshold for visualization |
| `--smooth-window` | `5` | Smoothing window for resemblance curves |

### generate_video_summaries.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--model` | (required) | Video summarization model (sportclip or qddetr) |
| `--video` | (optional) | Specific video name to process. If not provided, processes all videos |
| `--data-dir` | `data` | Directory containing video files |
| `--output-dir` | `video_summaries` | Output directory for video summaries |
| `--sentences-file` | (required for sportclip) | Path to SportCLIP sentences file |
| `--sportclip-dir` | `/home/mrt/Projects/SportCLIP-official` | SportCLIP project directory |
| `--sportclip-dataset-dir` | `/mnt/Data/mrt/SportCLIP-OlympicHighlights` | SportCLIP dataset directory |
| `--qddetr-ckpt` | (default path) | Path to QD-DETR checkpoint |
| `--qddetr-query` | (required for qddetr) | Query for QD-DETR |
| `--qddetr-device` | `cuda:0` | Device for QD-DETR inference |
| `--qddetr-context-window` | `600` | Context window size in frames |
| `--qddetr-min-duration` | `15` | Minimum event duration in frames |
| `--qddetr-min-area` | `15.0` | Minimum event area threshold |

### extract_all_faces.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data` | Directory containing video files |
| `--output-dir` | `cropped_faces` | Directory to save cropped faces |
| `--min-face-size` | `0` | Minimum face width/height in pixels |
| `--det-size` | `640 640` | Detection size for the model |

### extract_embeddings.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--input-dir` | `cropped_faces` | Directory containing cropped faces |
| `--output-dir` | `cropped_faces_embeddings` | Directory to save embeddings |
| `--models` | `ArcFace TransFace` | Models to use for extraction |
| `--gpu-id` | `0` | GPU ID for ArcFace |
| `--batch-size` | `1000` | Batch size for TransFace |
| `--transface-weight` | (default path) | Path to TransFace weights |
| `--transface-network` | `vit_l_dp005_mask_005` | TransFace architecture |

### find_updated_target.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | (required) | Target person name (matches directory in target_faces/) |
| `--video` | `highjump_video1` | Video name to process (without extension) |
| `--cropped-faces` | `cropped_faces` | Directory containing face crops |
| `--embeddings` | `cropped_faces_embeddings` | Directory containing embeddings |
| `--output` | `target_faces_updated` | Output directory for updated targets |
| `--threshold` | `0.5` | Minimum similarity threshold |
| `--top-k` | `10` | Number of top matches to log |
| `--transface-weight` | (default path) | Path to TransFace weights |
| `--transface-network` | `vit_l_dp005_mask_005` | TransFace architecture |

### compute_resemblance_curve.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--target` | (optional) | Target person name. If not provided, processes all targets |
| `--video` | `highjump_video1` | Video name to process (without extension) |
| `--video-path` | `data/<video>.mp4` | Path to video file (for FPS detection) |
| `--updated-targets` | `target_faces_updated` | Directory containing updated target faces |
| `--embeddings` | `cropped_faces_embeddings` | Directory containing embeddings |
| `--output` | `resemblance_results` | Output directory for plots and data |
| `--threshold` | `0.6` | Similarity threshold for visualization |
| `--smooth-window` | `5` | Window size for rolling average smoothing |

### create_final_visualization.py

| Argument | Default | Description |
|----------|---------|-------------|
| `--video` | (required) | Video name (without extension) |
| `--model` | (required) | Video summarization model used (sportclip or qddetr) |
| `--target` | (optional) | Specific target to process. If not provided, processes all targets |
| `--video-summaries` | `video_summaries` | Directory containing video summaries |
| `--resemblance-results` | `resemblance_results` | Directory containing resemblance results |
| `--output` | `results` | Output directory for final results |
| `--threshold` | `0.6` | Resemblance score threshold |
| `--smooth-window` | `5` | Smoothing window size |

## Video Summarization Integration

### SportCLIP Integration

SportCLIP uses text-video matching with CLIP embeddings to identify sports highlights.

**Required setup:**
1. SportCLIP project installed at `/home/mrt/Projects/SportCLIP-official`
2. SportCLIP virtual environment at `<sportclip_dir>/venv/`
3. Dataset available at `/mnt/Data/mrt/SportCLIP-OlympicHighlights`
4. Sentences file for your sport

### QD-DETR Integration

QD-DETR uses transformer-based query-conditioned moment retrieval.

**Required setup:**
1. QD-DETR project installed at `/home/mrt/Projects/QD-DETR`
2. QD-DETR virtual environment at `<qddetr_dir>/venv/`
3. Trained checkpoint and pre-extracted features

## Face Detection Metadata

Detected faces are tracked with:
- **clip_id**: Identifier for the video clip
- **frame_number**: Frame where face was detected
- **coordinates**: Bounding box (x1, y1, x2, y2)
- **confidence**: Detection confidence score
- **landmarks**: Facial landmark coordinates
- **cropped_face_path**: Path to saved face image

## Requirements

- Python 3.8+
- OpenCV
- InsightFace
- DeepFace
- TensorFlow
- h5py
- Matplotlib
- NumPy
- tqdm

See `requirements.txt` for specific versions.

## Performance Notes

- Face extraction processes ~45 frames/second on CPU
- GPU acceleration available for embedding extraction
- HDF5 storage format handles 400k+ embeddings efficiently
- Progressive saving prevents memory issues with large videos

## Troubleshooting

### Highlight Segment Timing Issues

**Problem**: Highlight clips don't align with resemblance curves, or wrong clips are being selected.

**Common causes and fixes**:

1. **Event-to-video matching bug** (Fixed in v1.1):
   - **Issue**: SportCLIP creates non-consecutive highlight files (e.g., `highlight_001.mp4`, `highlight_007.mp4`, `highlight_021.mp4`), but the code was using array indices instead of matching by filename.
   - **Result**: Event 7 was incorrectly matched to the 8th video file instead of `highlight_007.mp4`.
   - **Fix**: Modified `sportclip_summarizer.py` to match events to videos by searching for the highlight number in the filename: `f"_highlight_{event_num:03d}.mp4"`.

2. **Smoothing reduces peak scores**:
   - **Issue**: Applying smoothing to resemblance scores before filtering clips reduces peak values below the threshold.
   - **Result**: Clips where the target appears are incorrectly filtered out (false negatives).
   - **Fix**: `create_final_visualization.py` now uses **raw (unsmoothed) scores** for clip filtering, while still displaying smoothed curves for visualization.

3. **Missing JSON import in SportCLIP**:
   - **Issue**: SportCLIP's `summarize.py` uses `json.dump()` but was missing `import json`.
   - **Fix**: Added `import json` to line 7 of `/home/mrt/Projects/SportCLIP-official/summarize.py`.

4. **segments.json structure**:
   - The segments.json file should contain:
     - `video_name`: Name of the source video
     - `model`: Which summarization model was used ("sportclip" or "qddetr")
     - `total_segments`: Number of highlight segments
     - `segments`: Array of segment objects with:
       - `segment_id`: Segment identifier
       - `clip_path`: Relative path to the clip file
       - `clip_name`: Clip filename
       - `start_frame`: Starting frame number in original video
       - `end_frame`: Ending frame number in original video
       - `start_time`: Starting timestamp in seconds
       - `end_time`: Ending timestamp in seconds
       - `duration_frames`: Duration in frames
       - `duration_seconds`: Duration in seconds

### SportCLIP Integration Setup

**Required modifications to SportCLIP** (already applied):

1. Add JSON export to `summarize.py` (around line 520, before `if args.export_highlight_reel:`):
   ```python
   # Export events_filtered to JSON for downstream processing
   events_json_path = os.path.join("results", VIDEO, "events_filtered.json")
   with open(events_json_path, 'w') as f:
       events_json = {str(k): {"frames": v["frames"], "length": v["length"], "area": float(v["area"])}
                     for k, v in events_filtered.items()}
       json.dump(events_json, f, indent=2)
   print(f"Events exported to: {events_json_path}")
   ```

2. Add `import json` to the imports section of `summarize.py` (line 7).

### Regenerating Segments After Fixes

If you already generated video summaries before these fixes were applied, regenerate the segments.json files:

```bash
# Delete old segments.json
rm video_summaries/SportCLIP/*/segments.json
rm video_summaries/QD-DETR/*/segments.json

# Regenerate with fixed code
python generate_video_summaries.py --model sportclip --video highjump_video1 --sentences-file data/sentences/high_jump.json
python generate_video_summaries.py --model qddetr --video highjump_video1 --qddetr-query "your query here"
```

The video clips don't need to be regenerated - only the segments.json metadata files need updating.

## License

TBD
