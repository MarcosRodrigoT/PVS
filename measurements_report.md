# PVS Computational Cost Measurements

**Video:** highjump_video1.mp4
**Measurement Date:** 2025-12-07 12:34:26

---

## Raw Measurements (Minimal Units)

All times are measured at their **minimal unit of work** to allow flexible optimization calculations.

### Video Summarization

| Component | Unit | Time (ms) | Notes |
|-----------|------|-----------|-------|
| **SportCLIP** | per frame | 19.875 | Zero-shot, processes every frame |
| **QD-DETR** | per frame | 1.787 | Supervised, processes every frame |

### Face Detection & Recognition

| Component | Unit | Time (ms) | Notes |
|-----------|------|-----------|-------|
| **Face Detection** | per frame | 83.893 | InsightFace on GPU |
| **ArcFace Embedding** | per detected face | 35.951 | DeepFace with ArcFace model |
| **TransFace Embedding** | per detected face | 10.957 | ViT-based model |

**Average faces detected per frame:** 16.88

### Shared Components

| Component | Unit | Time (ms) | Notes |
|-----------|------|-----------|-------|
| **Resemblance Computation** | per face comparison | 0.0004 | Cosine similarity (np.dot) |
| **Clip Assignment** | per video segment | 0.0018 | Threshold checking |

---

## Optimization Considerations

### Parallelization Opportunities

- **Video Summarization + Face Detection:** Can run in parallel
  - Video summarization: processes frames independently
  - Face detection: processes frames independently
  - Both can read the video simultaneously

### Frame Sampling Optimizations

- **Face Detection:** Process every Nth frame instead of all frames
  - Current: 83.893 ms/frame
  - Every 5 frames: divide by 5
  - Every 10 frames: divide by 10
  - Trade-off: May miss faces in skipped frames

### Face Filtering Optimizations

- **Current:** 16.88 faces detected per frame on average
- **Size filtering:** Filter out faces smaller than minimum size
  - Reduces face embedding computation
  - Reduces resemblance computation
  - Each face avoided saves: 35.951 ms (ArcFace) or 10.957 ms (TransFace)

### Total Pipeline Cost Examples

For a 10-minute video at 30 fps with 1 target athlete:

#### Scenario 1: Non-optimized (Sequential, All Frames)
```
SportCLIP:              357.76s
Face Detection:        1510.08s (18000 frames)
ArcFace Embeddings:   10923.50s (303840 faces)
Resemblance:              0.11s (303840 comparisons)
Clip Assignment:          0.00s (12 segments)
---
TOTAL:                12791.45s (213.2 minutes)
```

#### Scenario 2: Optimized (Parallel Video+Face, Every 10th Frame)
```
Video (SportCLIP):      357.76s  (parallel)
Face Detection:         151.01s  (1800 frames, parallel)
ArcFace Embeddings:    1092.35s  (30384 faces, parallel)
Resemblance:              0.01s  (30384 comparisons, parallel)
Clip Assignment:          0.00s  (12 segments, after parallel)
---
TOTAL:                 1243.37s (20.7 minutes)
Speedup:              10.29x faster
```

---

## Custom Calculation Formula

Use these base measurements to calculate your own optimized timings:

```python
# Base measurements (ms)
sportclip_per_frame = 19.8754
qddetr_per_frame = 1.7874
face_detection_per_frame = 83.8932
arcface_per_face = 35.9515
transface_per_face = 10.9570
resemblance_per_comparison = 0.000369
clip_assignment_per_segment = 0.001845
avg_faces_per_frame = 16.8800

# Example: 10-minute video, 30 fps, every 10th frame for face detection
duration_min = 10
fps = 30
face_frame_skip = 10  # Process every 10th frame
n_targets = 1
n_segments = 12

total_frames = duration_min * 60 * fps
face_frames = total_frames // face_frame_skip
total_faces = face_frames * avg_faces_per_frame

# Sequential calculation
video_summarization_time = sportclip_per_frame * total_frames / 1000  # seconds
face_detection_time = face_detection_per_frame * face_frames / 1000
face_embedding_time = arcface_per_face * total_faces / 1000
resemblance_time = resemblance_per_comparison * total_faces * n_targets / 1000
clip_assignment_time = clip_assignment_per_segment * n_segments / 1000

# Parallel calculation (if video summarization runs parallel to face pipeline)
face_pipeline_time = face_detection_time + face_embedding_time + resemblance_time
parallel_time = max(video_summarization_time, face_pipeline_time) + clip_assignment_time
```
