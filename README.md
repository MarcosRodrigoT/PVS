# Personalized Video Summarization (PVS)

A pipeline for generating personalized video summaries by detecting and recognizing specific people in video content.

## Overview

The PVS pipeline performs the following operations:

1. **Video Summarization** - Generate highlight clips using SportCLIP or QD-DETR
2. **Face Detection** - Detects faces in every frame using state-of-the-art InsightFace SCRFD model
3. **Face Cropping** - Crops and saves detected faces with tracking metadata
4. **Face Recognition** - Compares detected faces against target faces using ArcFace
5. **Visualization** - Generates resemblance curves showing target person appearance over time

### Supported Video Summarization Methods

- **SportCLIP**: Sports highlight detection using text-video matching with CLIP embeddings
- **QD-DETR**: Query-based moment retrieval using transformer-based temporal localization

## Project Structure

```
PVS/
├──  main.py                      # Main pipeline orchestrator
├──  config.py                    # Configuration settings
├──  visualizer.py                # Visualization component
├──  requirements.txt             # Python dependencies
├──  face_detection/
├──  video_summarization/
│   ├── sportclip_summarizer.py  # SportCLIP integration
│   ├── qddetr_summarizer.py     # QD-DETR integration
│   ├── qddetr_wrapper.py        # QD-DETR wrapper script
│   ├── detector.py             # Face detection using InsightFace SCRFD
├── face_recognition/
│   ├── recognizer.py           # Face recognition using ArcFace
├── target_faces/               # Place target person face images here
├── results/                     # Results directory (auto-created)
│   ├── video_summaries/        # Video summary clips (from VS project)
│   ├── cropped_faces/          # Cropped face images
│   ├── visualizations/         # Resemblance curve plots
│   ├── face_detections.json    # Face detection metadata
│   ├── face_recognition_results.json  # Recognition results
│   ├── summary_report.txt      # Text summary report
└── README.md
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

3. Place video summary clips in `data/video_summaries/`:
   - Or use SportCLIP/QD-DETR integration to generate highlights automatically

## Usage

### Basic Usage

Run the pipeline on existing video clips:
```bash
python main.py
```

### Video Summarization Options

The pipeline supports two video summarization methods: **SportCLIP** (default) and **QD-DETR**.

#### Option 1: SportCLIP (Text-Video Matching)

Generate video highlights using SportCLIP and run the full pipeline:

```bash
# Example: Long jump video
python main.py --video-summarizer sportclip \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --sentences-file data/sentences/long_jump.json

# Example: 100m sprint video
python main.py --video-summarizer sportclip \
               --input 100m_video1 \
               --video-name 100m_video1 \
               --sentences-file data/sentences/100_meters.json
```

**How it works:**
1. Runs SportCLIP's `multi_sentences.py` to analyze highlight/non-highlight pairs
2. Runs SportCLIP's `summarize.py` to generate individual highlight videos
3. Copies the generated highlights to `data/video_summaries/`
4. Runs face detection, recognition, and visualization on the highlights

**Required SportCLIP setup:**
- SportCLIP project installed at configured location (default: `/home/mrt/Projects/SportCLIP-official`)
- SportCLIP virtual environment at `<sportclip_dir>/venv/`
- Dataset (videos + frames + embeddings) available (default: `/mnt/Data/mrt/SportCLIP-OlympicHighlights`)
- Sentences file for your sport (e.g., `data/sentences/long_jump.json`)

#### Option 2: QD-DETR (Query-Based Moment Retrieval)

Generate video highlights using QD-DETR with pre-extracted features:

```bash
# Example: Long jump video with QD-DETR
python main.py --video-summarizer qddetr \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --video-path /mnt/Data/mrt/SportCLIP-OlympicHighlights/longjump_video1.mp4 \
               --qddetr-query "highlights of the long jump, including the approach, take-off, flight, landing in the sand pit, and measurement of the jump" \
               --qddetr-feat-dirs /mnt/Data/mrt/QD-DETR/olympic/clip_features /mnt/Data/mrt/QD-DETR/olympic/slowfast_features \
               --qddetr-text-feat-dir /mnt/Data/mrt/QD-DETR/olympic/clip_text_features_specific \
               --qddetr-annotation-path /home/mrt/Projects/QD-DETR/data/OlympicHighlights_annotations/olympic_highlights_train_specific.jsonl
```

**How it works:**
1. Loads pre-extracted CLIP + SlowFast features for the video
2. Runs QD-DETR model to predict saliency scores for each video segment
3. Applies postprocessing (rolling averages, morphological closing) to detect highlight events
4. Extracts individual highlight videos based on detected events
5. Saves highlights to `data/video_summaries/`
6. Runs face detection, recognition, and visualization on the highlights

**Required QD-DETR setup:**
- QD-DETR project installed at `/home/mrt/Projects/QD-DETR`
- QD-DETR virtual environment at `<qddetr_dir>/venv/`
- Trained QD-DETR checkpoint (default: `results/OlympicHighlights-exp-2025_09_25_11_22_34/model_best.ckpt`)
- Pre-extracted video features (CLIP + SlowFast)
- Pre-extracted text features (CLIP text embeddings)
- Annotation JSONL file with video metadata

### Advanced Options

```bash
# Show plots interactively
python main.py --show-plots

# Use custom directories
python main.py --video-summaries custom/summaries --target-faces custom/targets

# Use different face recognition model
python main.py --model Facenet

# Custom SportCLIP directories
python main.py --video-summarizer sportclip \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --sentences-file data/sentences/long_jump.json \
               --sportclip-dir /path/to/SportCLIP \
               --sportclip-dataset-dir /path/to/dataset

# Custom QD-DETR parameters
python main.py --video-summarizer qddetr \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --video-path /path/to/video.mp4 \
               --qddetr-query "your query here" \
               --qddetr-ckpt /path/to/checkpoint.ckpt \
               --qddetr-context-window 600 \
               --qddetr-min-duration 15 \
               --qddetr-min-area 15.0
```

### Command-line Arguments

#### Video Input
- `--input`: Video name for processing (without extension)
- `--video-name`: Video name for processing (without extension)
- `--video-path`: Full path to input video file (required for QD-DETR)

#### Video Summarizer Selection
- `--video-summarizer`: Choose video summarization method: `sportclip` or `qddetr` (default: `sportclip`)

#### SportCLIP Configuration
- `--sentences-file`: Path to SportCLIP sentences file (e.g., `data/sentences/long_jump.json`)
- `--sportclip-dir`: SportCLIP project directory (default: `/home/mrt/Projects/SportCLIP-official`)
- `--sportclip-dataset-dir`: SportCLIP dataset directory (default: `/mnt/Data/mrt/SportCLIP-OlympicHighlights`)

#### QD-DETR Configuration
- `--qddetr-ckpt`: Path to QD-DETR checkpoint (default: `/home/mrt/Projects/QD-DETR/results/OlympicHighlights-exp-2025_09_25_11_22_34/model_best.ckpt`)
- `--qddetr-query`: Query for QD-DETR (e.g., `"highlight moments in this sports video"`)
- `--qddetr-annotation-path`: Path to QD-DETR annotation JSONL file (required when using pre-extracted features)
- `--qddetr-feat-dirs`: Video feature directories for QD-DETR (space-separated list)
- `--qddetr-text-feat-dir`: Text feature directory for QD-DETR
- `--qddetr-device`: Device for QD-DETR inference (default: `cuda:0`)
- `--qddetr-context-window`: Context window size in frames (default: 600)
- `--qddetr-min-duration`: Minimum event duration in frames (default: 15)
- `--qddetr-min-area`: Minimum event area threshold (default: 15.0)

#### PVS Directories
- `--video-summaries`: Directory containing video summary clips (default: `results/video_summaries`)
- `--target-faces`: Directory containing target face images (default: `target_faces`)
- `--cropped-faces`: Directory to save cropped faces (default: `results/cropped_faces`)
- `--output`: Output directory for results (default: `results`)

#### Model & Display
- `--model`: Face recognition model (ArcFace, Facenet, VGG-Face, DeepFace)
- `--show-plots`: Display plots interactively

## Models Used

### Face Detection: InsightFace SCRFD
- State-of-the-art single-stage face detector
- High accuracy in challenging conditions
- Provides facial landmarks and confidence scores

### Face Recognition: ArcFace
- State-of-the-art face recognition model
- 99.63% accuracy on LFW dataset
- Robust to variations in lighting and pose
- Alternative models available: Facenet, VGG-Face, DeepFace

## Output

The pipeline generates:

1. **face_detections.json** - Metadata for all detected faces including:
   - Frame number and coordinates
   - Confidence scores
   - Facial landmarks
   - Paths to cropped face images

2. **face_recognition_results.json** - Recognition results including:
   - Similarity scores per face
   - Target person matches
   - Frame-level resemblance data

3. **Resemblance curve plots** - Visual plots showing:
   - Resemblance score over time per clip
   - Match threshold indicators
   - Statistics (avg, max, match count)

4. **summary_report.txt** - Text summary with:
   - Per-clip statistics
   - Match rates and resemblance scores
   - Overall pipeline results

## Video Summarization Integration

The video summarization step (Step 1 in `main.py`) supports two frameworks: **SportCLIP** and **QD-DETR**. Both are fully integrated and can be selected via command-line arguments.

### SportCLIP Integration

SportCLIP uses text-video matching with CLIP embeddings to identify sports highlights.

**How it works:**
1. **Multi-sentence analysis**: Calls `multi_sentences.py` to test all combinations of highlight/non-highlight sentence pairs and compute similarity scores
2. **Highlight extraction**: Calls `summarize.py` to filter the best pairs and extract individual highlight videos
3. **Video copying**: Copies the generated highlight videos to the PVS `data/video_summaries/` directory
4. **Face processing**: Proceeds with face detection and recognition on the extracted highlights

**Required setup:**
1. SportCLIP project installed at the configured location (default: `/home/mrt/Projects/SportCLIP-official`)
2. SportCLIP's virtual environment set up at `<sportclip_dir>/venv/` with all dependencies installed
3. Dataset (videos + frames + embeddings) available (default: `/mnt/Data/mrt/SportCLIP-OlympicHighlights`)
4. Sentences file for your sport (e.g., `data/sentences/long_jump.json`)

**Example usage:**

```bash
python main.py --video-summarizer sportclip \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --sentences-file data/sentences/long_jump.json
```

This automatically:
- Generates individual highlight videos using SportCLIP
- Copies them to `data/video_summaries/longjump_video1_highlight_000.mp4`, etc.
- Runs face detection and recognition on each highlight
- Creates visualizations showing when target faces appear in the highlights

**Note**: The integration automatically uses SportCLIP's virtual environment if available, ensuring all dependencies are accessible.

### QD-DETR Integration

QD-DETR uses transformer-based query-conditioned moment retrieval to detect highlight segments.

**How it works:**
1. **Feature loading**: Loads pre-extracted CLIP and SlowFast features for the video
2. **Saliency prediction**: Runs QD-DETR model to predict saliency scores for each 2-second video segment
3. **Postprocessing**: Applies rolling averages and morphological closing to detect continuous highlight events
4. **Video extraction**: Extracts individual highlight videos using ffmpeg based on detected frame ranges
5. **Face processing**: Proceeds with face detection and recognition on the extracted highlights

**Required setup:**
1. QD-DETR project installed at `/home/mrt/Projects/QD-DETR`
2. QD-DETR virtual environment at `<qddetr_dir>/venv/` with PyTorch and dependencies
3. Trained QD-DETR checkpoint (e.g., `model_best.ckpt`)
4. Pre-extracted video features in separate directories:
   - CLIP features (512-dim per 2s clip)
   - SlowFast features (2304-dim per 2s clip)
   - CLIP text features for queries
5. Annotation JSONL file containing video metadata (qid, vid, query, duration, etc.)

**Example usage:**

```bash
python main.py --video-summarizer qddetr \
               --input longjump_video1 \
               --video-name longjump_video1 \
               --video-path /mnt/Data/mrt/SportCLIP-OlympicHighlights/longjump_video1.mp4 \
               --qddetr-query "highlights of the long jump" \
               --qddetr-feat-dirs /mnt/Data/mrt/QD-DETR/olympic/clip_features /mnt/Data/mrt/QD-DETR/olympic/slowfast_features \
               --qddetr-text-feat-dir /mnt/Data/mrt/QD-DETR/olympic/clip_text_features_specific \
               --qddetr-annotation-path /home/mrt/Projects/QD-DETR/data/OlympicHighlights_annotations/olympic_highlights_train_specific.jsonl
```

This automatically:
- Processes the video using the trained QD-DETR model
- Detects highlight events using saliency score postprocessing
- Extracts individual highlight videos to `data/video_summaries/`
- Runs face detection and recognition on each highlight
- Creates visualizations showing when target faces appear in the highlights

**Note**: The integration runs QD-DETR code in its own virtual environment via subprocess to avoid dependency conflicts with PVS.

## Face Detection Metadata

Detected faces are tracked with:
- **clip_id**: Identifier for the video clip
- **frame_number**: Frame where face was detected
- **coordinates**: Bounding box (x1, y1, x2, y2)
- **confidence**: Detection confidence score
- **landmarks**: Facial landmark coordinates
- **cropped_face_path**: Path to saved face image

This enables future tracking algorithms to follow faces across frames.

## Configuration

Edit `config.py` to customize:
- Directory paths
- Model selections
- Similarity thresholds
- Frame processing intervals

## Requirements

- Python 3.8+
- OpenCV
- InsightFace SCRFD
- DeepFace
- TensorFlow
- Matplotlib
- NumPy

See `requirements.txt` for specific versions.

## License

TBD
