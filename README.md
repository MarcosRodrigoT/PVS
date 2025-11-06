# Personalized Video Summarization (PVS)

A pipeline for generating personalized video summaries by detecting and recognizing specific people in video content.

## Overview

The PVS pipeline performs the following operations:

1. **Video Summarization (VS)** - Template call to external video summarization project
2. **Face Detection** - Detects faces in every frame using state-of-the-art InsightFace SCRFD model
3. **Face Cropping** - Crops and saves detected faces with tracking metadata
4. **Face Recognition** - Compares detected faces against target faces using ArcFace
5. **Visualization** - Generates resemblance curves showing target person appearance over time

## Project Structure

```
PVS/
├──  main.py                      # Main pipeline orchestrator
├──  config.py                    # Configuration settings
├──  visualizer.py                # Visualization component
├──  requirements.txt             # Python dependencies
├──  face_detection/
│   ├── detector.py             # Face detection using InsightFace SCRFD
├── face_recognition/
│   ├── recognizer.py           # Face recognition using ArcFace
├── target_faces/               # Place target person face images here
├── data/
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
   - Or integrate with the VS (Video Summarization) project when available

## Usage

### Basic Usage

Run the pipeline on existing video clips:
```bash
python main.py
```

### Using SportCLIP Integration

Generate video highlights using SportCLIP and run the full pipeline:

```bash
# Example: Long jump video
python main.py --input longjump_video1 \
               --video-name longjump_video1 \
               --sentences-file data/sentences/long_jump.json

# Example: 100m sprint video
python main.py --input 100m_video1 \
               --video-name 100m_video1 \
               --sentences-file data/sentences/100_meters.json
```

This will:
1. Run SportCLIP's `multi_sentences.py` to analyze highlight/non-highlight pairs
2. Run SportCLIP's `summarize.py` to generate individual highlight videos
3. Copy the generated highlights to `data/video_summaries/`
4. Run face detection, recognition, and visualization on the highlights

### Advanced Options

```bash
# Show plots interactively
python main.py --show-plots

# Use custom directories
python main.py --video-summaries custom/summaries --target-faces custom/targets

# Use different face recognition model
python main.py --model Facenet

# Custom SportCLIP directories
python main.py --input longjump_video1 \
               --video-name longjump_video1 \
               --sentences-file data/sentences/long_jump.json \
               --sportclip-dir /path/to/SportCLIP \
               --sportclip-dataset-dir /path/to/dataset
```

### Command-line Arguments

#### Video Input
- `--input`: Video name for SportCLIP processing (without extension)
- `--video-name`: Video name for processing (without extension)

#### SportCLIP Configuration
- `--sentences-file`: Path to SportCLIP sentences file (e.g., `data/sentences/long_jump.json`)
- `--sportclip-dir`: SportCLIP project directory (default: `/home/mrt/Projects/SportCLIP-official`)
- `--sportclip-dataset-dir`: SportCLIP dataset directory (default: `/mnt/Data/mrt/SportCLIP-OlympicHighlights`)

#### PVS Directories
- `--video-summaries`: Directory containing video summary clips (default: `data/video_summaries`)
- `--target-faces`: Directory containing target face images (default: `target_faces`)
- `--cropped-faces`: Directory to save cropped faces (default: `data/cropped_faces`)
- `--output`: Output directory for results (default: `data`)

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

## Integration with SportCLIP Project

The video summarization step (Step 1 in `main.py`) is now fully integrated with the SportCLIP framework. The integration works as follows:

### How It Works

1. **Multi-sentence analysis**: Calls `multi_sentences.py` to test all combinations of highlight/non-highlight sentence pairs and compute similarity scores
2. **Highlight extraction**: Calls `summarize.py` to filter the best pairs and extract individual highlight videos
3. **Video copying**: Copies the generated highlight videos to the PVS `data/video_summaries/` directory
4. **Face processing**: Proceeds with face detection and recognition on the extracted highlights

### Required Setup

1. Ensure SportCLIP project is installed at the configured location (default: `/home/mrt/Projects/SportCLIP-official`)
2. Ensure SportCLIP's virtual environment is set up at `<sportclip_dir>/venv/` with all dependencies installed
3. Ensure your dataset (videos + frames + embeddings) is available at the configured location (default: `/mnt/Data/mrt/SportCLIP-OlympicHighlights`)
4. Prepare a sentences file for your sport (e.g., `data/sentences/long_jump.json`)

**Note**: The integration automatically uses SportCLIP's virtual environment if available, ensuring all dependencies are accessible.

### Example Usage

```bash
python main.py --input longjump_video1 \
               --video-name longjump_video1 \
               --sentences-file data/sentences/long_jump.json
```

This automatically:
- Generates individual highlight videos using SportCLIP
- Copies them to `data/video_summaries/longjump_video1_highlight_000.mp4`, etc.
- Runs face detection and recognition on each highlight
- Creates visualizations showing when target faces appear in the highlights

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
