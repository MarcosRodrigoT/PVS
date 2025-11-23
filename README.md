# Personalized Video Summarization (PVS)

A pipeline for generating personalized video summaries by detecting and recognizing specific people in video content.

## Overview

The PVS pipeline has been redesigned to run components more independently to better handle sports videos where athletes may not be facing the camera during highlight moments.

### Pipeline Workflow

The pipeline always runs these steps:

#### Steps 1-3: Video Summarization and Face Data Extraction
1. **Video Summarization** - Generate highlight clips using SportCLIP or QD-DETR
2. **Full Video Face Detection** - Detect and extract faces from ALL frames of original videos
3. **Face Embedding Extraction** - Extract embeddings from all face crops using ArcFace and TransFace

#### Steps 4-6: Face Recognition and Resemblance Analysis
The behavior of Step 4 depends on the `--use-updated-target` flag:

**Without `--use-updated-target` flag (default - uses original target):**
- **Step 4**: Extract embeddings from **original target images** in `target_faces/`
- **Step 5**: Compute resemblance curves using original target embeddings
- **Step 6**: Create final visualizations
- **Result**: Max similarity ~80-90% (comparing original target vs. video faces)

**With `--use-updated-target` flag (finds best match first):**
- **Step 4**: Find best matching face in video → save as updated target
- **Step 4b**: Extract embeddings from **updated target** (the best match)
- **Step 5**: Compute resemblance curves using updated target embeddings
- **Step 6**: Create final visualizations
- **Result**: Max similarity ~100% (comparing updated target vs. itself in video)

### Supported Video Summarization Methods

- **SportCLIP**: Sports highlight detection using text-video matching with CLIP embeddings
- **QD-DETR**: Query-based moment retrieval using transformer-based temporal localization

### Supported Face Recognition Models

- **ArcFace**: State-of-the-art CNN-based face recognition (via DeepFace)
- **TransFace**: Transformer-based face recognition model

**Independent Model Processing**: By default, ArcFace and TransFace models process highlight clips independently, creating separate folders for each model's results (`ArcFace_highlight_clips/` and `TransFace_highlight_clips/`). This allows you to compare which clips each model considers relevant to your target person. You can also choose to run only one model using the `--face-recognition-models` flag.

### Highlight Assignment and Evaluation

The pipeline includes two automated highlight-to-athlete assignment methods that determine which athlete appears in each highlight segment:

**Method 1: Sequential Assignment (Default - `highlights_assigned.png`)**
- Scans the video timeline from left to right
- For each highlight segment, looks back 5 seconds before the segment starts
- Assigns the segment to the first athlete whose resemblance score crosses the threshold
- All subsequent highlights are assigned to that athlete until another athlete is identified
- **Use case**: Works well when athletes are shown in close-up shots before their performance
- Processes ArcFace and TransFace independently (no averaging of scores)
- Creates visualization showing assigned highlights color-coded by athlete

**Method 2: Instant Assignment with Temporal Expansion (`highlights_instant_assigned.png`)**
- Evaluates each highlight segment independently based on direct temporal overlap
- For each highlight, checks if any athlete's resemblance score exceeds the threshold during that segment's time window
- **Temporal expansion**: Extends the search window by 3 seconds on each side to account for face occlusion during performance
- Assigns to the athlete with the highest average score above threshold when multiple candidates exist
- **Use case**: Better handles cases where athletes are not identified before their performance, or when recognition fails
- Processes ArcFace and TransFace independently
- Creates a separate visualization for comparison with Method 1

**Evaluation Against Ground Truth:**

When ground truth annotations are available (CSV files in `data/<video>_identity.csv`), the system evaluates assignment accuracy using IoU-based matching:

1. **Intersection over Union (IoU) Matching:**
   - For each predicted segment, computes IoU with all ground truth segments of the same athlete
   - IoU = (Overlap Duration) / (Total Duration Covered by Both Segments)
   - Uses IoU threshold of 0.3 to determine if segments match

2. **Many-to-Many Matching:**
   - Handles cases where one predicted segment overlaps multiple ground truth segments
   - Handles cases where one ground truth segment overlaps multiple predicted segments
   - Uses best-match strategy: each predicted segment matches to the ground truth segment with highest IoU

3. **Correctness Criteria:**
   - A prediction is considered **correct** if:
     - IoU ≥ 0.3 with a ground truth segment
     - The athlete names match (case-insensitive)

4. **Evaluation Metrics:**
   - **GT**: Number of ground truth segments for each athlete
   - **Pred**: Number of predicted segments assigned to each athlete
   - **Corr**: Number of correctly matched predictions (IoU ≥ 0.3 and same athlete)
   - **Prec**: Precision = Correct / Predicted (percentage of predictions that are correct)

The evaluation results are displayed in a table at the bottom of the `highlights_assigned.png` plot, showing per-athlete metrics for both ArcFace and TransFace models.

**Organized Clip Output:**

Assigned clips are automatically copied to an organized directory structure:
```
results/<video>/<model>/highlights_assigned/
├── <athlete1>/
│   ├── ArcFace/
│   │   ├── clip_001.mp4
│   │   └── clip_007.mp4
│   └── TransFace/
│       ├── clip_001.mp4
│       └── clip_005.mp4
├── <athlete2>/
│   ├── ArcFace/
│   └── TransFace/
└── ...
```

This structure makes it easy to review which clips were assigned to each athlete by each face recognition model.

## Project Structure

```
PVS/
├── main.py                            # Complete pipeline orchestrator
├── generate_video_summaries.py        # Standalone video summarization script
├── extract_all_faces.py               # Extract faces from all video frames (saves metadata)
├── extract_embeddings.py              # UNIFIED: Extract embeddings (video faces, original targets, updated targets)
├── find_updated_target.py             # Find best matching face as updated target
├── compute_resemblance_curve.py       # Compute and visualize resemblance scores
├── visualize_results.py               # UNIFIED: Create resemblance curves and enhanced visualizations
├── config.py                          # Configuration settings
├── requirements.txt                   # Python dependencies
├── face_detection/
│   └── detector.py               # Face detection using InsightFace SCRFD
├── face_recognition/
│   └── recognizer.py             # Face recognition using ArcFace/TransFace
├── video_summarization/
│   ├── sportclip_summarizer.py   # SportCLIP integration
│   ├── qddetr_summarizer.py      # QD-DETR integration
│   └── qddetr_wrapper.py         # QD-DETR wrapper script
├── data/                         # Input videos and ground truth CSV files
├── target_faces/                 # Target person face images (organized by video)
│   ├── <video_name>/             # One directory per video
│   │   ├── athlete1.jpg
│   │   └── athlete2.jpg
│   └── ...
├── video_summaries/              # Organized video summaries (cached)
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
├── target_faces_updated/         # Updated target faces (best matches, organized by video)
│   └── <video_name>/             # One directory per video
│       └── <target_name>/        # One directory per target
│           ├── face_crop.jpg                         # Cropped face from original target
│           ├── target_arcface_embedding.npy          # Embedding from original target
│           ├── target_transface_embedding.npy        # Embedding from original target
│           ├── face_crop_arcface_updated.jpg         # Best matching face from video (ArcFace)
│           ├── face_crop_transface_updated.jpg       # Best matching face from video (TransFace)
│           ├── updated_target_arcface_embedding.npy  # Embedding from updated target (with --use-updated-target)
│           ├── updated_target_transface_embedding.npy # Embedding from updated target (with --use-updated-target)
│           └── <target>_log.txt                      # Match details
├── resemblance_results/          # Resemblance curve analysis (organized by video)
│   └── <video_name>/             # One directory per video
│       └── <target_name>/        # One directory per target
│           ├── <target>.png          # Resemblance curve plot
│           ├── <target>_updated.png  # Resemblance curve plot (with --use-updated-target)
│           ├── <target>.json         # Resemblance data
│           └── <target>_updated.json # Resemblance data (with --use-updated-target)
└── results/                      # Final results directory (organized by video → model → target)
    └── <video_name>/
        └── <model>/              # "sportclip" or "qddetr"
            └── <target_name>/
                ├── <target>.png                # Enhanced curve plot
                ├── <target>_updated.png        # Enhanced curve plot (with --use-updated-target)
                ├── <target>.json               # Summary report
                ├── <target>_updated.json       # Summary report (with --use-updated-target)
                ├── config_log.txt                      # Configuration parameters for this run
                ├── ArcFace_highlight_clips/            # Clips selected by ArcFace model
                │   └── clip_*.mp4
                ├── TransFace_highlight_clips/          # Clips selected by TransFace model
                │   └── clip_*.mp4
                ├── ArcFace_highlight_clips_updated/    # ArcFace clips (with --use-updated-target)
                │   └── clip_*.mp4
                └── TransFace_highlight_clips_updated/  # TransFace clips (with --use-updated-target)
                    └── clip_*.mp4
```

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

2. Organize target face images in video-specific subdirectories within `target_faces/`:
   - Create a subdirectory for each video: `target_faces/<video_name>/`
   - Place athlete images in the corresponding video directory
   - Supported formats: .jpg, .png, .jpeg
   - Clear, frontal face images work best
   - Example structure:
     ```
     target_faces/
     ├── highjump_video1/
     │   ├── barshim.jpg
     │   └── starc.jpg
     └── javelin_video1/
         ├── athlete1.jpg
         └── athlete2.jpg
     ```

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

**Without `--use-updated-target` flag (uses original target face):**
```bash
python main.py --video highjump_video1 \
    --video-summarizer qddetr \
    --qddetr-query "high jump moments"
```

This performs:
1. Generate video summaries (saved to `video_summaries/`)
2. Extract faces from entire video (saved to `cropped_faces/`)
3. Extract embeddings from all video faces (saved to `cropped_faces_embeddings/`)
4. Extract embeddings from **original target images** in `target_faces/`
5. Compute resemblance curves using original target embeddings
6. Create final visualizations with highlight overlays

**Result**: Max similarity ~80-90% (comparing original target vs. video faces)
**Output**: `resemblance_results/` and `results/<video>/<target>/`
- Both ArcFace and TransFace create separate highlight clip folders

**Selecting Specific Face Recognition Models:**
```bash
# Use only ArcFace model
python main.py --video highjump_video1 \
    --video-summarizer qddetr \
    --qddetr-query "high jump moments" \
    --face-recognition-models arcface

# Use only TransFace model
python main.py --video highjump_video1 \
    --video-summarizer qddetr \
    --qddetr-query "high jump moments" \
    --face-recognition-models transface

# Use both models (default behavior - can be omitted)
python main.py --video highjump_video1 \
    --video-summarizer qddetr \
    --qddetr-query "high jump moments" \
    --face-recognition-models arcface transface
```

**With `--use-updated-target` flag (finds best match first):**
```bash
python main.py --video highjump_video1 \
    --video-summarizer qddetr \
    --qddetr-query "high jump moments" \
    --use-updated-target
```

This performs:
1. Generate video summaries (saved to `video_summaries/`)
2. Extract faces from entire video (saved to `cropped_faces/`)
3. Extract embeddings from all video faces (saved to `cropped_faces_embeddings/`)
4. **Find best matching face in video** → save as updated target (saved to `target_faces_updated/`)
5. Extract embeddings from **updated target** (second pass)
6. Compute resemblance curves using updated target embeddings
7. Create final visualizations with highlight overlays

**Result**: Max similarity ~100% (comparing updated target vs. itself in video)
**Output**: Files saved with `_updated` suffix to avoid overwriting the original results

**Key Features:**
- **Smart caching**: Automatically skips steps that have already been completed
- **Fast iteration**: Re-run on same video to skip expensive face extraction
- **Shared resources**: Face crops and embeddings are cached at project root for reuse across experiments

### Option 2: Run Steps Individually

For more control, run each step separately.

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

**IMPORTANT**: This script requires GPU acceleration. Use the `run_with_cuda.sh` wrapper to enable CUDA libraries:

```bash
./run_with_cuda.sh python extract_all_faces.py
```

Options:
```bash
# Filter out faces smaller than 50x50 pixels
./run_with_cuda.sh python extract_all_faces.py --min-face-size 50

# Custom directories
./run_with_cuda.sh python extract_all_faces.py --data-dir /path/to/videos --output-dir /path/to/output
```

**Why the wrapper?** The system has CUDA 13.0, but `onnxruntime-gpu` needs CUDA 12.x libraries. The wrapper script automatically sets up the library path to use CUDA 12 libraries from pip packages.

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

### Step 3: Extract Embeddings

The unified `extract_embeddings.py` script handles all embedding extraction workflows with different modes.

#### Mode 1: Extract embeddings from video faces (default)

```bash
python extract_embeddings.py --mode video_faces
```

Options:
```bash
# Extract only ArcFace embeddings
python extract_embeddings.py --mode video_faces --models ArcFace

# Extract only TransFace embeddings
python extract_embeddings.py --mode video_faces --models TransFace

# Use specific GPU for ArcFace
python extract_embeddings.py --mode video_faces --models ArcFace --gpu-id 1

# Custom TransFace batch size
python extract_embeddings.py --mode video_faces --models TransFace --batch-size 2000
```

#### Mode 2: Extract embeddings from original target faces

```bash
python extract_embeddings.py --mode original_targets
```

#### Mode 3: Extract embeddings from updated target faces

```bash
python extract_embeddings.py --mode updated_targets --video highjump_video1
```

**Note**: This script uses TensorFlow for face recognition, which works fine with the system's CUDA 13.0 installation, so no wrapper is needed.

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

### Step 4 (Optional): Find Updated Target Face

**This step is only needed for face recognition and resemblance analysis.**

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
    ├── face_crop.jpg                      # Cropped face from original target
    ├── target_arcface_embedding.npy       # ArcFace embedding (from original)
    ├── target_transface_embedding.npy     # TransFace embedding (from original)
    ├── face_crop_arcface_updated.jpg      # Best matching face from video (ArcFace)
    ├── face_crop_transface_updated.jpg    # Best matching face from video (TransFace)
    └── athlete1_log.txt                   # Match details
```

#### Step 4b: Extract Updated Target Embeddings - Second Pass

**This step is automatically performed when using `--use-updated-target` flag with main.py.**

When running the standalone scripts, resemblance curves are computed using embeddings from your **original target image** by default. This is why you may see maximum similarity scores of ~85% instead of 100%.

To achieve 100% similarity at the point where the updated target appears, use the `--use-updated-target` flag. This performs a second pass that:

1. Extracts embeddings from the **updated target face** (the best match found in step 4)
2. Compares these new embeddings against all video faces
3. Results in 100% similarity when comparing the updated target against itself in the video

**Usage with main.py:**
```bash
# Run complete pipeline with second pass
python main.py --video highjump_video1 \
    --video-summarizer qddetr \
    --qddetr-query "high jump moments" \
    --use-updated-target
```

**Standalone usage:**
```bash
# Extract updated target embeddings manually
python extract_embeddings.py --mode updated_targets --video highjump_video1

# Compute resemblance curves with updated embeddings
python compute_resemblance_curve.py --video highjump_video1 --use-updated-target

# Create visualizations with updated embeddings
python visualize_results.py --mode final_with_highlights --video highjump_video1 --model qddetr --use-updated-target
```

**What gets generated:**
When using `--use-updated-target`, additional files are created with `_updated` suffix to avoid overwriting:
```
target_faces_updated/athlete1/
    ├── updated_target_arcface_embedding.npy    # NEW: Embedding from best match
    └── updated_target_transface_embedding.npy  # NEW: Embedding from best match

resemblance_results/athlete1/
    ├── athlete1_updated.png
    └── athlete1_updated.json

results/highjump_video1/qddetr/athlete1/
    ├── athlete1_updated.png
    ├── athlete1_updated.json
    ├── config_log.txt  # Configuration parameters
    └── highlight_clips_updated/
        └── clip_*.mp4
```

**Comparison:**
- **Without flag**: Max similarity ~80-90% (comparing original target vs. video faces)
- **With flag**: Max similarity ~100% (comparing updated target vs. itself in video)

This allows you to compare both approaches side-by-side to see the difference!

### Step 5 (Optional): Compute Resemblance Curves

**This step is only needed for face recognition and resemblance analysis.**

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
    ├── athlete1.png                    # Resemblance curve visualization
    ├── athlete1_updated.png            # Resemblance curve (with --use-updated-target)
    ├── athlete1.json                   # Raw resemblance data
    └── athlete1_updated.json           # Raw data (with --use-updated-target)
```

The plot shows:
- Raw resemblance scores (faded lines)
- Smoothed scores using rolling average (prominent lines)
- Threshold line (default: 0.6)
- Highlighted regions above threshold
- Statistics (mean, max, frames above threshold)

### Step 6 (Optional): Create Final Visualizations with Highlights

**This step is only needed for face recognition and resemblance analysis.**

The unified `visualize_results.py` script handles all visualization needs with different modes.

#### Mode 1: Create basic resemblance curves

```bash
python visualize_results.py --mode resemblance_curves --recognition-results results/face_recognition_results.json
```

#### Mode 2: Create enhanced visualizations with highlight overlays

Generate enhanced resemblance curves with highlight segments overlaid on the timeline, and copy relevant clips:

```bash
# Process all targets for a video
python visualize_results.py --mode final_with_highlights --video highjump_video1 --model sportclip

# Process specific target only
python visualize_results.py --mode final_with_highlights --video highjump_video1 --model sportclip --target barshim

# Custom threshold
python visualize_results.py --mode final_with_highlights --video highjump_video1 --model sportclip --threshold 0.65

# With updated target embeddings
python visualize_results.py --mode final_with_highlights --video highjump_video1 --model sportclip --use-updated-target

# Use only ArcFace model
python visualize_results.py --mode final_with_highlights --video highjump_video1 --model sportclip --face-recognition-models arcface

# Use only TransFace model
python visualize_results.py --mode final_with_highlights --video highjump_video1 --model sportclip --face-recognition-models transface
```

This script:
- Creates enhanced resemblance curve plots with highlight segments marked on the timeline (red boxes labeled "H1", "H2", etc.)
- Creates **combined athletes plots** showing all athletes' resemblance curves together (both normal and zoomed versions)
- Creates **assigned highlights plot** (`highlights_assigned.png`) with color-coded segments by athlete, ground truth comparison, and evaluation metrics
- **Independently analyzes** which highlight clips contain the target person using ArcFace and TransFace models
- Copies relevant clips to separate model-specific folders:
  - `ArcFace_highlight_clips/` - Clips selected by ArcFace
  - `TransFace_highlight_clips/` - Clips selected by TransFace
  - With `--use-updated-target`: `ArcFace_highlight_clips_updated/` and `TransFace_highlight_clips_updated/`
  - `highlights_assigned/<athlete>/{ArcFace,TransFace}/` - Clips organized by assigned athlete
- Generates a summary report with statistics for both models
- Creates a `config_log.txt` file documenting all configuration parameters

**Important**:
- Each face recognition model processes clips independently based on its own resemblance scores
- The script uses **raw (unsmoothed) scores** for filtering clips to avoid false negatives
- You can compare which clips each model considers relevant

Output structure:
```
results/
└── highjump_video1/
    └── sportclip/                                  # Organized by video summarization model
        ├── all_athletes_combined.png               # Combined plot of all athletes (normal scale)
        ├── all_athletes_combined_zoom.png          # Combined plot of all athletes (zoomed scale)
        ├── all_athletes_combined_updated.png       # Combined plot (with --use-updated-target)
        ├── all_athletes_combined_updated_zoom.png  # Combined plot zoomed (with --use-updated-target)
        ├── highlights_assigned.png                 # Highlight assignments (sequential method) with evaluation metrics
        ├── highlights_assigned_updated.png         # Highlight assignments (with --use-updated-target)
        ├── highlights_instant_assigned.png         # Highlight assignments (instant method with temporal expansion)
        ├── highlights_instant_assigned_updated.png # Instant assignments (with --use-updated-target)
        ├── highlights_assigned/                    # Organized clips by assigned athlete
        │   ├── athlete1/
        │   │   ├── ArcFace/
        │   │   │   └── clip_*.mp4
        │   │   └── TransFace/
        │   │       └── clip_*.mp4
        │   └── athlete2/
        │       ├── ArcFace/
        │       └── TransFace/
        └── barshim/
            ├── barshim.png                         # Enhanced curve plot
            ├── barshim_updated.png                 # Enhanced curve plot (with --use-updated-target)
            ├── barshim.json                        # Summary report
            ├── barshim_updated.json                # Summary report (with --use-updated-target)
            ├── config_log.txt                      # Configuration parameters
            ├── ArcFace_highlight_clips/            # Clips selected by ArcFace
            │   ├── clip_007.mp4
            │   └── clip_021.mp4
            ├── TransFace_highlight_clips/          # Clips selected by TransFace
            │   ├── clip_005.mp4
            │   └── clip_021.mp4
            ├── ArcFace_highlight_clips_updated/    # ArcFace clips (with --use-updated-target)
            │   └── clip_*.mp4
            └── TransFace_highlight_clips_updated/  # TransFace clips (with --use-updated-target)
                └── clip_*.mp4
```

**Visualization outputs:**

1. **Per-athlete enhanced plots** (`<athlete>.png`):
   - Raw and smoothed resemblance scores (both displayed for visualization)
   - **Red rectangular overlays** marking highlight segment positions on the timeline
   - **Red labels** ("H1", "H2", etc.) identifying each highlight
   - Threshold line and above-threshold regions
   - Summary statistics including number of highlight segments

2. **Combined athletes plots** (`all_athletes_combined.png` and `_zoom.png`):
   - All athletes' resemblance curves displayed together in a single figure
   - Two subplots: ArcFace (top) and TransFace (bottom)
   - Color-coded by athlete using Kelly's 22 colors for maximum contrast
   - Normal version: y-axis 0-1, Zoom version: auto-scaled to data range
   - Separate legends: athletes (external) and threshold/highlights (internal)

3. **Assigned highlights plots** (`highlights_assigned.png` and `highlights_instant_assigned.png`):
   - Timeline view showing which athlete was assigned to each highlight segment
   - Three subplots: ArcFace assignments, TransFace assignments, Ground Truth (if available)
   - Color-coded segments by athlete
   - Evaluation table showing per-athlete metrics (GT, Pred, Corr, Precision)
   - Uses IoU-based matching (threshold 0.3) to evaluate predictions against ground truth
   - **Two assignment methods generated**:
     - `highlights_assigned.png`: Sequential assignment (assigns subsequent highlights to identified athlete)
     - `highlights_instant_assigned.png`: Instant assignment with 3-second temporal expansion (independent segment evaluation)

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
| `--use-updated-target` | `False` | **Enable face recognition workflow** (Steps 4-7). Without this flag, only video summarization and face data extraction (Steps 1-3) are performed. |
| `--face-recognition-models` | `arcface transface` | Face recognition models to use for clip selection. Options: `arcface`, `transface`, or both. Each model creates separate highlight clip folders. |

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

**Usage**: `./run_with_cuda.sh python extract_all_faces.py [options]`

| Argument | Default | Description |
|----------|---------|-------------|
| `--data-dir` | `data` | Directory containing video files |
| `--output-dir` | `cropped_faces` | Directory to save cropped faces |
| `--min-face-size` | `0` | Minimum face width/height in pixels |
| `--det-size` | `640 640` | Detection size for the model |

**Note**: Requires the `run_with_cuda.sh` wrapper for GPU acceleration. See [Troubleshooting](#cuda-library-compatibility-onnxruntime-gpu) section.

### extract_embeddings.py (Unified Script)

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | `video_faces` | Extraction mode: `video_faces`, `original_targets`, or `updated_targets` |
| `--video` | (required for updated_targets) | Video name (without extension) |
| `--input-dir` | `cropped_faces` | Directory containing cropped faces (video_faces mode) |
| `--output-dir` | `cropped_faces_embeddings` | Directory to save embeddings (video_faces mode) |
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
| `--use-updated-target` | `False` | Use embeddings from updated target (second pass). Files saved with '_updated' suffix |

### visualize_results.py (Unified Script)

| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | (required) | Visualization mode: `resemblance_curves` or `final_with_highlights` |
| `--video` | (required for final_with_highlights) | Video name (without extension) |
| `--model` | (required for final_with_highlights) | Video summarization model used (sportclip or qddetr) |
| `--target` | (optional) | Specific target to process. If not provided, processes all targets |
| `--recognition-results` | (required for resemblance_curves) | Path to face recognition results JSON |
| `--video-summaries` | `video_summaries` | Directory containing video summaries |
| `--resemblance-results` | `resemblance_results` | Directory containing resemblance results |
| `--output` | `results` | Output directory for final results |
| `--threshold` | `0.6` | Resemblance score threshold |
| `--smooth-window` | `5` | Smoothing window size |
| `--use-updated-target` | `False` | Use results from updated target embeddings (second pass). Reads files with '_updated' suffix |
| `--face-recognition-models` | `arcface transface` | Face recognition models to use for clip selection. Options: `arcface`, `transface`, or both. Each model creates separate highlight clip folders. |

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

## Reorganization Notes

The project has been reorganized for better maintainability:

### Unified Scripts

- **`extract_embeddings.py`**: Combines three separate embedding extraction scripts into one with `--mode` parameter
  - `video_faces`: Extract embeddings from video face crops
  - `original_targets`: Extract embeddings from original target faces
  - `updated_targets`: Extract embeddings from updated target faces

- **`visualize_results.py`**: Combines visualization scripts into one with `--mode` parameter
  - `resemblance_curves`: Create basic resemblance curves
  - `final_with_highlights`: Create enhanced visualizations with highlight overlays

### Simplified Filenames

Filenames have been simplified to remove redundant video names:
- `target_faces_updated/{target}/{target}_log.txt` (was: `{target}_{video}_matching_log.txt`)
- `resemblance_results/{target}/{target}.png` (was: `{target}_{video}_resemblance_curve.png`)
- `results/{video}/{model}/{target}/{target}.png` (was: `{target}_{video}_enhanced_curve.png`)

### Improved Directory Structure

- Results are now organized as: `results/{video}/{model}/{target}/`
- Separate `highlight_clips/` and `highlight_clips_updated/` directories prevent overwrites
- Each target directory includes a `config_log.txt` with all configuration parameters

## Performance Notes

- Face extraction with GPU: ~40-100 frames/second (using run_with_cuda.sh wrapper)
- Face extraction on CPU only: ~2-5 frames/second
- GPU acceleration available for embedding extraction
- HDF5 storage format handles 400k+ embeddings efficiently
- Progressive saving prevents memory issues with large videos

### Computational Cost Tracking

The pipeline automatically tracks timing information for all major components to help analyze computational costs for research papers. When you run `main.py`, timing data is collected and analyzed automatically.

**Components Tracked:**

1. **Video Summarization** - Time to generate highlight clips (normalized per video)
2. **Face Detection** - Time to detect faces in video frames (normalized per frame)
3. **Face Embedding (ArcFace)** - Time to extract ArcFace embeddings (normalized per face)
4. **Face Embedding (TransFace)** - Time to extract TransFace embeddings (normalized per face)
5. **Target Finding** - Time to find best matching face in video (normalized per target×frame)
6. **Target Embedding (Original)** - Time to extract embeddings from original target images (normalized per target)
7. **Target Embedding (Updated)** - Time to extract embeddings from updated target faces (normalized per target)
8. **Resemblance Computation** - Time to compute resemblance curves (normalized per target×face)
9. **Visualization** - Time to create plots and visualizations (normalized per target)
10. **Copy Highlight Clips** - Time to copy relevant video clips (normalized per target)

**Normalization by Natural Units:**

Timing measurements are normalized by the natural unit of work for each component:

- **s/video**: One-time costs (Video Summarization)
- **ms/frame**: Per-frame operations (Face Detection)
- **ms/face**: Per-face operations (Face Embedding)
- **ms/target**: Per-target operations (Target Embedding)
- **μs/(target×frame)**: Per-target per-frame operations (Target Finding)
- **μs/(target×face)**: Per-target per-face operations (Resemblance Computation)

This normalization allows for fair comparison across videos of different lengths and with different numbers of faces/targets.

**Output Files:**

Timing data is automatically saved to:

```
results/
├── timing_log_aggregate.json              # Combined log from all runs
├── timing_log_<video>_<timestamp>.json   # Per-run logs
└── timing_analysis/                       # Generated automatically after each run
    ├── timing_summary.md                  # Markdown table with statistics
    ├── timing_bar_chart.png               # Bar chart visualization (300 DPI)
    └── timing_comparison.md               # Estimated times for different video lengths
```

**Timing Analysis Outputs:**

1. **Summary Table** (`timing_summary.md`):
   - Mean, median, std dev, min, max for each component
   - Times normalized by appropriate units (ms/frame, ms/face, etc.)
   - Ready for inclusion in research papers

2. **Bar Chart** (`timing_bar_chart.png`):
   - Visual comparison of component timings
   - Error bars showing standard deviation
   - Publication-quality 300 DPI resolution

3. **Comparison Table** (`timing_comparison.md`):
   - Estimated processing times for different video lengths
   - Projections for 1min, 5min, 10min, 30min, 60min videos
   - Helps plan computational requirements

**Example Usage:**

```bash
# Run pipeline - timing tracked automatically
python main.py --video highjump_video1 \
    --video-summarizer qddetr \
    --qddetr-query "high jump moments"

# Timing analysis generated automatically in results/timing_analysis/
# - No additional flags needed
# - Works across multiple runs and videos
# - Aggregates statistics from all executions
```

**Multi-Run Aggregation:**

The system automatically aggregates timing data across multiple runs:
- Run the pipeline on different videos multiple times
- Statistics are computed across all runs
- Provides robust timing estimates with variance information
- Each run is logged separately with timestamp for debugging

**Implementation Details:**

- Timing tracking is integrated directly into each pipeline component
- Uses high-precision `time.time()` for measurements
- Saves timing data progressively (doesn't require pipeline completion)
- Timing analysis is generated automatically at the end of each `main.py` run
- Uses pandas for statistical analysis and matplotlib for visualization

## Troubleshooting

### CUDA Library Compatibility (onnxruntime-gpu)

**Problem**: Error message "libcublasLt.so.12: cannot open shared object file" when running `extract_all_faces.py`.

**Cause**: System has CUDA 13.0, but `onnxruntime-gpu 1.20.2` requires CUDA 12.x libraries.

**Solution**: Use the `run_with_cuda.sh` wrapper script:

```bash
./run_with_cuda.sh python extract_all_faces.py --min-face-size 50
```

The wrapper automatically adds CUDA 12 libraries (installed via pip) to the library search path.

**Verification**: You should see `Applied providers: ['CUDAExecutionProvider', 'CPUExecutionProvider']` in the output instead of falling back to CPU only.

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
