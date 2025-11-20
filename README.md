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
├── main.py                       # Main pipeline orchestrator (original sequential approach)
├── extract_all_faces.py          # Extract faces from all video frames
├── extract_embeddings.py         # Extract embeddings from face crops
├── find_updated_target.py        # Find best matching face as updated target
├── compute_resemblance_curve.py  # Compute and visualize resemblance scores
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
├── cropped_faces/                # Extracted face crops (one folder per video)
├── cropped_faces_embeddings/     # Face embeddings in HDF5 format
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
└── results/                      # Results directory
    ├── video_summaries/          # Video summary clips
    ├── visualizations/           # Resemblance curve plots
    └── *.json                    # Detection and recognition results
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

## New Workflow

### Step 1: Extract Faces from All Videos

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
└── ...
```

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

### Step 3: Generate Video Summaries (Independent)

Generate highlight clips using SportCLIP or QD-DETR. This step runs independently and can be done before or after face extraction.

#### Using SportCLIP:
```bash
python main.py --video-summarizer sportclip \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --sentences-file data/sentences/long_jump.json
```

#### Using QD-DETR:
```bash
python main.py --video-summarizer qddetr \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --video-path /path/to/longjump_video1.mp4 \
               --qddetr-query "highlights of the long jump"
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

### Step 5: Filter and Generate Video Summaries

Use resemblance scores to filter video summaries and keep only segments where the target person appears.

**(Coming soon)** Filter summaries based on resemblance curve data.

### Optional: Generate Video Summaries (Independent)

Generate highlight clips using SportCLIP or QD-DETR. This step runs independently and can be done before or after face extraction.

#### Using SportCLIP:
```bash
python main.py --video-summarizer sportclip \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --sentences-file data/sentences/long_jump.json
```

#### Using QD-DETR:
```bash
python main.py --video-summarizer qddetr \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --video-path /path/to/longjump_video1.mp4 \
               --qddetr-query "highlights of the long jump"
```

## Original Sequential Pipeline

The original sequential pipeline is still available via `main.py`:

```bash
# Run on existing video clips
python main.py

# With TransFace model
python main.py --model TransFace
```

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

## Command-line Arguments

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

### main.py

See full documentation in the original usage section below.

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

## License

TBD
