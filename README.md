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

### Advanced Options

```bash
# Specify input video for summarization (when VS is integrated)
python main.py --input my_video.mp4

# Show plots interactively
python main.py --show-plots

# Use custom directories
python main.py --video-summaries custom/summaries --target-faces custom/targets

# Use different face recognition model
python main.py --model Facenet
```

### Command-line Arguments

- `--input`: Input video file for summarization
- `--video-summaries`: Directory containing video summary clips (default: `data/video_summaries`)
- `--target-faces`: Directory containing target face images (default: `target_faces`)
- `--cropped-faces`: Directory to save cropped faces (default: `data/cropped_faces`)
- `--output`: Output directory for results (default: `data`)
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

## Integration with VS Project

The video summarization step (Step 1 in `main.py`) is currently a template. To integrate with your VS project:

1. Import your VS module in `main.py`
2. Replace the template call in `step1_video_summarization()` with your VS API
3. Ensure VS outputs video clips to the specified directory

Example integration:
```python
def step1_video_summarization(self, input_video: str) -> List[str]:
    from vs_project import VideoSummarizer

    summarizer = VideoSummarizer()
    video_clips = summarizer.summarize(
        input_video=input_video,
        output_dir=self.video_summaries_dir
    )

    return video_clips
```

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
