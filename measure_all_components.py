#!/usr/bin/env python3
"""
Comprehensive computational cost measurement script.
Measures all components from scratch for highjump_video1 only.
Times ONLY the actual computational components, excluding I/O and overhead.
"""

import os
import sys
import time
import cv2
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Tuple

# Configure environment
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_video_info(video_path: str) -> Dict:
    """Get video FPS and frame count"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return {"fps": fps, "frame_count": frame_count}


def measure_sportclip(video_path: str, video_name: str, temp_dir: Path) -> float:
    """Measure SportCLIP per-frame inference time"""
    print(f"\n{'='*80}")
    print(f"Measuring SportCLIP: {video_name}")
    print(f"{'='*80}")

    info = get_video_info(video_path)
    total_frames = info["frame_count"]

    output_dir = temp_dir / "sportclip" / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    sportclip_dir = "/home/mrt/Projects/SportCLIP-official"
    dataset_dir = "/mnt/Data/mrt/SportCLIP-OlympicHighlights"
    sentences_file = "data/sentences/high_jump.json"

    # Temporarily add to path, then remove to avoid conflicts
    sys.path.insert(0, sportclip_dir)
    try:
        from video_summarization import generate_highlights_sportclip

        # Time only the actual generation
        start_time = time.perf_counter()
        _, _ = generate_highlights_sportclip(
            video_name=video_name,
            sentences_file=sentences_file,
            output_dir=str(output_dir),
            sportclip_dir=sportclip_dir,
            sportclip_dataset_dir=dataset_dir,
            video_path=video_path,
        )
        duration = time.perf_counter() - start_time

        per_frame_ms = (duration / total_frames) * 1000
        print(f"  Per frame: {per_frame_ms:.3f} ms")
        return per_frame_ms
    finally:
        # Remove SportCLIP from path to avoid import conflicts
        if sportclip_dir in sys.path:
            sys.path.remove(sportclip_dir)


def measure_qddetr(video_path: str, video_name: str, temp_dir: Path) -> float:
    """Measure QD-DETR per-frame inference time"""
    print(f"\n{'='*80}")
    print(f"Measuring QD-DETR: {video_name}")
    print(f"{'='*80}")

    info = get_video_info(video_path)
    total_frames = info["frame_count"]

    output_dir = temp_dir / "qddetr" / video_name
    output_dir.mkdir(parents=True, exist_ok=True)

    from video_summarization import generate_highlights_qddetr

    # Use correct QD-DETR lookup name (e.g., "highjump_video1" -> "High Jump_video1")
    qddetr_lookup_name = "High Jump_video1"

    # Time only the actual generation
    start_time = time.perf_counter()
    _, _ = generate_highlights_qddetr(
        video_name=qddetr_lookup_name,
        query="highlight moments in this high jump competition",
        output_dir=str(output_dir),
        video_path=video_path,
        qddetr_ckpt_path="/home/mrt/Projects/QD-DETR/results/OlympicHighlights-exp-2025_09_25_11_22_34/model_best.ckpt",
        context_window=600,
        min_duration_frames=15,
        min_area=15.0,
        device="cuda:0",
        feat_dirs=["/mnt/Data/mrt/QD-DETR/olympic/slowfast_features", "/mnt/Data/mrt/QD-DETR/olympic/clip_features"],
        text_feat_dir="/mnt/Data/mrt/QD-DETR/olympic/clip_text_features_generic",
        annotation_path="/home/mrt/Projects/QD-DETR/data/OlympicHighlights_annotations/olympic_highlights_train_generic.jsonl",
    )
    duration = time.perf_counter() - start_time

    per_frame_ms = (duration / total_frames) * 1000
    print(f"  Per frame: {per_frame_ms:.3f} ms")
    return per_frame_ms


def measure_face_detection(video_path: str) -> Tuple[float, float]:
    """Measure face detection per-frame time - ONLY the detection call"""
    print(f"\n{'='*80}")
    print(f"Measuring Face Detection")
    print(f"{'='*80}")

    from insightface.app import FaceAnalysis

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample 100 frames evenly
    sample_stride = max(1, total_frames // 100)
    frame_indices = list(range(0, total_frames, sample_stride))[:100]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
    cap.release()

    # Time ONLY the app.get() calls
    timings = []
    total_faces = 0
    for frame in frames:
        start = time.perf_counter()
        faces = app.get(frame)
        timings.append((time.perf_counter() - start) * 1000)
        total_faces += len(faces)

    avg_time_ms = np.mean(timings)
    avg_faces = total_faces / len(frames)
    print(f"  Per frame: {avg_time_ms:.3f} ms")
    print(f"  Avg faces: {avg_faces:.2f}")
    return avg_time_ms, avg_faces


def measure_arcface(video_path: str, temp_dir: Path) -> float:
    """Measure ArcFace per-face embedding time - ONLY the represent() call"""
    print(f"\n{'='*80}")
    print(f"Measuring ArcFace")
    print(f"{'='*80}")

    from deepface import DeepFace
    from insightface.app import FaceAnalysis

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Extract face crops
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_stride = max(1, total_frames // 20)

    face_crops = []
    for idx in range(0, total_frames, sample_stride):
        if len(face_crops) >= 30:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        faces = app.get(frame)
        for face in faces[:2]:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                face_crops.append(crop)
    cap.release()

    # Save crops
    temp_crop_dir = temp_dir / "arcface_crops"
    temp_crop_dir.mkdir(parents=True, exist_ok=True)
    crop_paths = []
    for i, crop in enumerate(face_crops[:30]):
        crop_path = temp_crop_dir / f"face_{i}.jpg"
        cv2.imwrite(str(crop_path), crop)
        crop_paths.append(str(crop_path))

    # Warm up
    _ = DeepFace.represent(img_path=crop_paths[0], model_name="ArcFace", enforce_detection=False)

    # Time ONLY DeepFace.represent() calls
    timings = []
    for crop_path in crop_paths:
        start = time.perf_counter()
        _ = DeepFace.represent(img_path=crop_path, model_name="ArcFace", enforce_detection=False)
        timings.append((time.perf_counter() - start) * 1000)

    avg_time_ms = np.mean(timings)
    print(f"  Per face: {avg_time_ms:.3f} ms")
    return avg_time_ms


def measure_transface(video_path: str, temp_dir: Path) -> float:
    """Measure TransFace per-face embedding time - tries direct import, falls back to subprocess"""
    print(f"\n{'='*80}")
    print(f"Measuring TransFace")
    print(f"{'='*80}")

    # Try subprocess approach first (works without torch in current venv)
    try:
        return measure_transface_subprocess(video_path, temp_dir)
    except Exception as e:
        print(f"  Subprocess approach failed: {e}")
        print(f"  Trying direct import approach...")

    # Fallback to direct import (requires torch)
    TRANSFACE_DIR = "/home/mrt/Projects/TransFace"
    sys.path.insert(0, TRANSFACE_DIR)

    try:
        import torch
        from recognition.arcface_torch.configs.config import get_config
        from recognition.arcface_torch.backbones import get_model
        from torchvision import transforms
        from insightface.app import FaceAnalysis

        transface_weight = "/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt"
        transface_network = "vit_l_dp005_mask_005"

        cfg = get_config(transface_network)
        model = get_model(transface_network, dropout=0.0, fp16=False, num_features=cfg.embedding_size)
        state_dict = torch.load(transface_weight, map_location='cpu')
        model.load_state_dict(state_dict)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        app.prepare(ctx_id=0, det_size=(640, 640))

        # Extract face crops
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_stride = max(1, total_frames // 20)

        face_crops = []
        for idx in range(0, total_frames, sample_stride):
            if len(face_crops) >= 30:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            faces = app.get(frame)
            for face in faces[:2]:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0:
                    face_crops.append(crop)
        cap.release()

        # Prepare tensors (do this OUTSIDE timing)
        tensors = []
        for crop in face_crops[:30]:
            img = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            img_tensor = transform(img).unsqueeze(0).to(device)
            tensors.append(img_tensor)

        # Warm up
        with torch.no_grad():
            _ = model(tensors[0])

        # Time ONLY model inference
        timings = []
        for img_tensor in tensors:
            start = time.perf_counter()
            with torch.no_grad():
                _ = model(img_tensor)
            timings.append((time.perf_counter() - start) * 1000)

        avg_time_ms = np.mean(timings)
        print(f"  Per face: {avg_time_ms:.3f} ms")
        return avg_time_ms
    finally:
        # Remove TransFace from path to avoid import conflicts
        if TRANSFACE_DIR in sys.path:
            sys.path.remove(TRANSFACE_DIR)


def measure_transface_subprocess(video_path: str, temp_dir: Path) -> float:
    """Measure TransFace per-face embedding time using subprocess (fallback)"""
    print(f"\n{'='*80}")
    print(f"Measuring TransFace (subprocess mode)")
    print(f"{'='*80}")

    from insightface.app import FaceAnalysis
    import subprocess

    TRANSFACE_DIR = "/home/mrt/Projects/TransFace"
    TRANSFACE_VENV_PYTHON = os.path.join(TRANSFACE_DIR, "venv", "bin", "python")
    TRANSFACE_BATCH_SCRIPT = os.path.join(TRANSFACE_DIR, "batch_recognize.py")
    transface_weight = "/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt"
    transface_network = "vit_l_dp005_mask_005"

    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))

    # Extract and save 30 face crops
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    sample_stride = max(1, total_frames // 20)

    temp_crop_dir = temp_dir / "transface_crops"
    temp_crop_dir.mkdir(parents=True, exist_ok=True)

    crop_paths = []
    for idx in range(0, total_frames, sample_stride):
        if len(crop_paths) >= 30:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue

        faces = app.get(frame)
        for face in faces[:2]:
            if len(crop_paths) >= 30:
                break
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = max(0, bbox[0]), max(0, bbox[1]), min(frame.shape[1], bbox[2]), min(frame.shape[0], bbox[3])
            crop = frame[y1:y2, x1:x2]
            if crop.size > 0:
                crop_path = temp_crop_dir / f"face_{len(crop_paths)}.jpg"
                cv2.imwrite(str(crop_path), crop)
                crop_paths.append(str(crop_path))
    cap.release()

    # Create input list file (JSON format required by batch_recognize.py)
    import json
    input_list_file = temp_crop_dir / "input_list.json"
    with open(input_list_file, 'w') as f:
        json.dump({"images": crop_paths}, f)

    # Run TransFace batch extraction with timing
    output_file = temp_crop_dir / "embeddings.json"

    start_time = time.perf_counter()
    cmd = [
        TRANSFACE_VENV_PYTHON,
        TRANSFACE_BATCH_SCRIPT,
        "--mode", "extract-batch",
        "--input-list", str(input_list_file),
        "--output", str(output_file),
        "--weight", transface_weight,
        "--network", transface_network,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    total_duration = time.perf_counter() - start_time

    # Read timing information from output JSON
    import json
    with open(output_file, 'r') as f:
        output_data = json.load(f)

    # Extract inference-only timing (excludes model loading)
    if 'timing' in output_data:
        per_face_ms = output_data['timing']['avg_inference_time_ms']
        model_load_time = output_data['timing']['model_load_time_s']
        print(f"  Model load time: {model_load_time:.2f}s")
        print(f"  Per face (inference only): {per_face_ms:.3f} ms")
    else:
        # Fallback: use total time (includes model loading - less accurate)
        per_face_ms = (total_duration / len(crop_paths)) * 1000
        print(f"  Per face (total time/faces): {per_face_ms:.3f} ms")
        print(f"  WARNING: Using total time including model loading")

    return per_face_ms


def measure_resemblance() -> float:
    """Measure resemblance computation - ONLY the cosine similarity"""
    print(f"\n{'='*80}")
    print(f"Measuring Resemblance Computation")
    print(f"{'='*80}")

    # Create normalized embeddings
    n_samples = 500
    embedding_dim = 512
    face_embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    target_embedding = np.random.randn(embedding_dim).astype(np.float32)
    face_embeddings = face_embeddings / np.linalg.norm(face_embeddings, axis=1, keepdims=True)
    target_embedding = target_embedding / np.linalg.norm(target_embedding)

    # Time ONLY the dot product
    timings = []
    for face_emb in face_embeddings:
        start = time.perf_counter()
        _ = np.dot(target_embedding, face_emb)
        timings.append((time.perf_counter() - start) * 1e6)

    avg_time_us = np.mean(timings)
    avg_time_ms = avg_time_us / 1000
    print(f"  Per comparison: {avg_time_ms:.4f} ms")
    return avg_time_ms


def measure_clip_assignment() -> float:
    """Measure clip assignment - ONLY the segment checking logic"""
    print(f"\n{'='*80}")
    print(f"Measuring Clip Assignment")
    print(f"{'='*80}")

    # Pre-create data
    n_frames = 10000
    resemblance_curve = np.random.rand(n_frames)
    threshold = 0.6

    segments = []
    for i in range(50):
        start = np.random.randint(0, n_frames - 100)
        end = start + np.random.randint(30, 100)
        segments.append((start, min(end, n_frames)))

    # Time ONLY the checking logic
    timings = []
    for start_frame, end_frame in segments:
        start = time.perf_counter()
        segment_scores = resemblance_curve[start_frame:end_frame]
        _ = np.any(segment_scores > threshold)
        timings.append((time.perf_counter() - start) * 1000)

    avg_time_ms = np.mean(timings)
    print(f"  Per segment: {avg_time_ms:.4f} ms")
    return avg_time_ms


def generate_measurement_report(results: Dict, output_file: str):
    """Generate detailed markdown report with raw measurements"""
    lines = []
    lines.append("# PVS Computational Cost Measurements")
    lines.append("")
    lines.append("**Video:** highjump_video1.mp4")
    lines.append(f"**Measurement Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    lines.append("## Raw Measurements (Minimal Units)")
    lines.append("")
    lines.append("All times are measured at their **minimal unit of work** to allow flexible optimization calculations.")
    lines.append("")

    # Video Summarization
    lines.append("### Video Summarization")
    lines.append("")
    lines.append("| Component | Unit | Time (ms) | Notes |")
    lines.append("|-----------|------|-----------|-------|")
    lines.append(f"| **SportCLIP** | per frame | {results['sportclip_per_frame_ms']:.3f} | Zero-shot, processes every frame |")
    lines.append(f"| **QD-DETR** | per frame | {results['qddetr_per_frame_ms']:.3f} | Supervised, processes every frame |")
    lines.append("")

    # Face Detection & Recognition
    lines.append("### Face Detection & Recognition")
    lines.append("")
    lines.append("| Component | Unit | Time (ms) | Notes |")
    lines.append("|-----------|------|-----------|-------|")
    lines.append(f"| **Face Detection** | per frame | {results['face_detection_per_frame_ms']:.3f} | InsightFace on GPU |")
    lines.append(f"| **ArcFace Embedding** | per detected face | {results['arcface_per_face_ms']:.3f} | DeepFace with ArcFace model |")
    lines.append(f"| **TransFace Embedding** | per detected face | {results['transface_per_face_ms']:.3f} | ViT-based model |")
    lines.append("")
    lines.append(f"**Average faces detected per frame:** {results['avg_faces_per_frame']:.2f}")
    lines.append("")

    # Shared Components
    lines.append("### Shared Components")
    lines.append("")
    lines.append("| Component | Unit | Time (ms) | Notes |")
    lines.append("|-----------|------|-----------|-------|")
    lines.append(f"| **Resemblance Computation** | per face comparison | {results['resemblance_per_comparison_ms']:.4f} | Cosine similarity (np.dot) |")
    lines.append(f"| **Clip Assignment** | per video segment | {results['clip_assignment_per_segment_ms']:.4f} | Threshold checking |")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Optimization Considerations")
    lines.append("")
    lines.append("### Parallelization Opportunities")
    lines.append("")
    lines.append("- **Video Summarization + Face Detection:** Can run in parallel")
    lines.append("  - Video summarization: processes frames independently")
    lines.append("  - Face detection: processes frames independently")
    lines.append("  - Both can read the video simultaneously")
    lines.append("")

    lines.append("### Frame Sampling Optimizations")
    lines.append("")
    lines.append("- **Face Detection:** Process every Nth frame instead of all frames")
    lines.append(f"  - Current: {results['face_detection_per_frame_ms']:.3f} ms/frame")
    lines.append("  - Every 5 frames: divide by 5")
    lines.append("  - Every 10 frames: divide by 10")
    lines.append("  - Trade-off: May miss faces in skipped frames")
    lines.append("")

    lines.append("### Face Filtering Optimizations")
    lines.append("")
    lines.append(f"- **Current:** {results['avg_faces_per_frame']:.2f} faces detected per frame on average")
    lines.append("- **Size filtering:** Filter out faces smaller than minimum size")
    lines.append("  - Reduces face embedding computation")
    lines.append("  - Reduces resemblance computation")
    lines.append(f"  - Each face avoided saves: {results['arcface_per_face_ms']:.3f} ms (ArcFace) or {results['transface_per_face_ms']:.3f} ms (TransFace)")
    lines.append("")

    lines.append("### Total Pipeline Cost Examples")
    lines.append("")
    lines.append("For a 10-minute video at 30 fps with 1 target athlete:")
    lines.append("")

    # Calculate different scenarios
    total_frames = 10 * 60 * 30  # 10 min * 60 sec * 30 fps
    avg_faces = results['avg_faces_per_frame']
    total_faces = total_frames * avg_faces
    n_segments = 12  # assumed 12 segments per 10 min

    # Scenario 1: Non-optimized SportCLIP + ArcFace
    video_sum_s1 = results['sportclip_per_frame_ms'] * total_frames / 1000
    face_det_s1 = results['face_detection_per_frame_ms'] * total_frames / 1000
    face_emb_s1 = results['arcface_per_face_ms'] * total_faces / 1000
    resemblance_s1 = results['resemblance_per_comparison_ms'] * total_faces / 1000
    clip_assign_s1 = results['clip_assignment_per_segment_ms'] * n_segments / 1000
    total_s1 = video_sum_s1 + face_det_s1 + face_emb_s1 + resemblance_s1 + clip_assign_s1

    # Scenario 2: Optimized (parallel video+face, every 10 frames face detection)
    video_sum_s2 = results['sportclip_per_frame_ms'] * total_frames / 1000
    face_det_s2 = results['face_detection_per_frame_ms'] * (total_frames / 10) / 1000
    total_faces_s2 = (total_frames / 10) * avg_faces
    face_emb_s2 = results['arcface_per_face_ms'] * total_faces_s2 / 1000
    resemblance_s2 = results['resemblance_per_comparison_ms'] * total_faces_s2 / 1000
    clip_assign_s2 = results['clip_assignment_per_segment_ms'] * n_segments / 1000
    # Parallel: max of (video_sum, face_det + face_emb + resemblance)
    total_s2 = max(video_sum_s2, face_det_s2 + face_emb_s2 + resemblance_s2) + clip_assign_s2

    lines.append("#### Scenario 1: Non-optimized (Sequential, All Frames)")
    lines.append("```")
    lines.append(f"SportCLIP:            {video_sum_s1:8.2f}s")
    lines.append(f"Face Detection:       {face_det_s1:8.2f}s ({total_frames} frames)")
    lines.append(f"ArcFace Embeddings:   {face_emb_s1:8.2f}s ({total_faces:.0f} faces)")
    lines.append(f"Resemblance:          {resemblance_s1:8.2f}s ({total_faces:.0f} comparisons)")
    lines.append(f"Clip Assignment:      {clip_assign_s1:8.2f}s ({n_segments} segments)")
    lines.append(f"---")
    lines.append(f"TOTAL:                {total_s1:8.2f}s ({total_s1/60:.1f} minutes)")
    lines.append("```")
    lines.append("")

    lines.append("#### Scenario 2: Optimized (Parallel Video+Face, Every 10th Frame)")
    lines.append("```")
    lines.append(f"Video (SportCLIP):    {video_sum_s2:8.2f}s  (parallel)")
    lines.append(f"Face Detection:       {face_det_s2:8.2f}s  ({total_frames//10} frames, parallel)")
    lines.append(f"ArcFace Embeddings:   {face_emb_s2:8.2f}s  ({total_faces_s2:.0f} faces, parallel)")
    lines.append(f"Resemblance:          {resemblance_s2:8.2f}s  ({total_faces_s2:.0f} comparisons, parallel)")
    lines.append(f"Clip Assignment:      {clip_assign_s2:8.2f}s  ({n_segments} segments, after parallel)")
    lines.append(f"---")
    lines.append(f"TOTAL:                {total_s2:8.2f}s ({total_s2/60:.1f} minutes)")
    lines.append(f"Speedup:              {total_s1/total_s2:.2f}x faster")
    lines.append("```")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("## Custom Calculation Formula")
    lines.append("")
    lines.append("Use these base measurements to calculate your own optimized timings:")
    lines.append("")
    lines.append("```python")
    lines.append("# Base measurements (ms)")
    lines.append(f"sportclip_per_frame = {results['sportclip_per_frame_ms']:.4f}")
    lines.append(f"qddetr_per_frame = {results['qddetr_per_frame_ms']:.4f}")
    lines.append(f"face_detection_per_frame = {results['face_detection_per_frame_ms']:.4f}")
    lines.append(f"arcface_per_face = {results['arcface_per_face_ms']:.4f}")
    lines.append(f"transface_per_face = {results['transface_per_face_ms']:.4f}")
    lines.append(f"resemblance_per_comparison = {results['resemblance_per_comparison_ms']:.6f}")
    lines.append(f"clip_assignment_per_segment = {results['clip_assignment_per_segment_ms']:.6f}")
    lines.append(f"avg_faces_per_frame = {results['avg_faces_per_frame']:.4f}")
    lines.append("")
    lines.append("# Example: 10-minute video, 30 fps, every 10th frame for face detection")
    lines.append("duration_min = 10")
    lines.append("fps = 30")
    lines.append("face_frame_skip = 10  # Process every 10th frame")
    lines.append("n_targets = 1")
    lines.append("n_segments = 12")
    lines.append("")
    lines.append("total_frames = duration_min * 60 * fps")
    lines.append("face_frames = total_frames // face_frame_skip")
    lines.append("total_faces = face_frames * avg_faces_per_frame")
    lines.append("")
    lines.append("# Sequential calculation")
    lines.append("video_summarization_time = sportclip_per_frame * total_frames / 1000  # seconds")
    lines.append("face_detection_time = face_detection_per_frame * face_frames / 1000")
    lines.append("face_embedding_time = arcface_per_face * total_faces / 1000")
    lines.append("resemblance_time = resemblance_per_comparison * total_faces * n_targets / 1000")
    lines.append("clip_assignment_time = clip_assignment_per_segment * n_segments / 1000")
    lines.append("")
    lines.append("# Parallel calculation (if video summarization runs parallel to face pipeline)")
    lines.append("face_pipeline_time = face_detection_time + face_embedding_time + resemblance_time")
    lines.append("parallel_time = max(video_summarization_time, face_pipeline_time) + clip_assignment_time")
    lines.append("```")
    lines.append("")

    # Write to file
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ Detailed measurement report saved to: {output_file}")


def generate_latex_table(results: Dict, output_file: str):
    """Generate LaTeX table"""
    sportclip_ms = results["sportclip_per_frame_ms"]
    qddetr_ms = results["qddetr_per_frame_ms"]
    face_det_ms = results["face_detection_per_frame_ms"]
    arcface_ms = results["arcface_per_face_ms"]
    transface_ms = results["transface_per_face_ms"]
    resemblance_ms = results["resemblance_per_comparison_ms"]
    clip_ms = results["clip_assignment_per_segment_ms"]
    faces_per_frame = results["avg_faces_per_frame"]

    fps = 30
    n_targets = 1
    segments_per_10min = 12

    def compute_total(duration_min, use_qddetr=False, use_transface=False):
        total_frames = duration_min * 60 * fps
        total_faces = total_frames * faces_per_frame
        n_segments = int(segments_per_10min * (duration_min / 10))

        video_sum = (qddetr_ms if use_qddetr else sportclip_ms) / 1000 * total_frames
        face_det = face_det_ms / 1000 * total_frames
        face_emb = (transface_ms if use_transface else arcface_ms) / 1000 * total_faces
        resemblance = resemblance_ms / 1000 * total_faces * n_targets
        clip_assign = clip_ms / 1000 * n_segments

        return video_sum + face_det + face_emb + resemblance + clip_assign

    lines = []
    lines.append("% Computational cost table - Measured from highjump_video1")
    lines.append("")
    lines.append("\\begin{table}[htbp]")
    lines.append("    \\centering")
    lines.append("    \\footnotesize")
    lines.append("    \\caption{Computational cost of the main components of the PVS pipeline.}")
    lines.append("    \\label{tab: PVS - computational cost}")
    lines.append("    \\begin{tabular}{l l c}")
    lines.append("        \\hline")
    lines.append("        \\textbf{Component}                          & \\textbf{Unit of work}         & \\textbf{Time} \\\\")
    lines.append("        \\hline")
    lines.append("        \\multicolumn{3}{l}{\\textbf{Video summarization}} \\\\")
    lines.append(f"        \\quad SportCLIP                             & per frame                     & {sportclip_ms:.2f} ms \\\\")
    lines.append(f"        \\quad QD-DETR                               & per frame                     & {qddetr_ms:.2f} ms \\\\")
    lines.append("        \\hline")
    lines.append("        \\multicolumn{3}{l}{\\textbf{Face recognition}} \\\\")
    lines.append(f"        \\quad ArcFace (face embeddings)             & per detected face             & {arcface_ms:.2f} ms \\\\")
    lines.append(f"        \\quad TransFace (face embeddings)           & per detected face             & {transface_ms:.2f} ms \\\\")
    lines.append("        \\hline")
    lines.append("        \\multicolumn{3}{l}{\\textbf{Shared components}} \\\\")
    lines.append(f"        \\quad Face detection                        & per frame                     & {face_det_ms:.2f} ms \\\\")
    lines.append(f"        \\quad Resemblance-curve computation         & per frame $\\times$ athlete    & {resemblance_ms:.3f} ms \\\\")
    lines.append("        \\quad Target update (optional)              & per target athlete            & --- \\\\")
    lines.append(f"        \\quad Clip assignment and selection         & per highlight segment         & {clip_ms:.2f} ms \\\\")
    lines.append("        \\hline")

    for duration_min in [1, 5, 10, 60]:
        if duration_min == 1:
            lines.append("        \\multicolumn{3}{l}{\\textbf{Total PVS runtime (per 1-min video)}} \\\\")
            unit = "per 1-min video"
        elif duration_min == 5:
            lines.append("        \\multicolumn{3}{l}{\\textbf{Total PVS runtime (per 5-min video)}} \\\\")
            unit = "per 5-min video"
        elif duration_min == 10:
            lines.append("        \\multicolumn{3}{l}{\\textbf{Total PVS runtime (per 10-min video)}} \\\\")
            unit = "per 10-min video"
        else:
            lines.append("        \\multicolumn{3}{l}{\\textbf{Total PVS runtime (per 1-hour video)}} \\\\")
            unit = "per 1-hour video"

        configs = [
            ("SportCLIP + ArcFace", False, False),
            ("SportCLIP + TransFace", False, True),
            ("QD-DETR + ArcFace", True, False),
            ("QD-DETR + TransFace", True, True),
        ]

        for name, use_qddetr, use_transface in configs:
            total = compute_total(duration_min, use_qddetr, use_transface)
            if duration_min == 60:
                time_str = f"{total/60:.2f} min"
            else:
                time_str = f"{total:.2f} s"
            lines.append(f"        {name:<40} & {unit:<30} & {time_str} \\\\")

        lines.append("        \\hline")

    lines.append("    \\end{tabular}")
    lines.append("\\end{table}")

    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))

    print(f"\n✓ Table saved to: {output_file}")


def main():
    print("="*80)
    print("COMPUTATIONAL COST MEASUREMENT")
    print("="*80)

    video_path = "data/highjump_video1.mp4"
    video_name = "highjump_video1"

    if not Path(video_path).exists():
        print(f"Error: Video not found: {video_path}")
        return

    temp_dir = Path(tempfile.mkdtemp(prefix="pvs_measurement_"))

    try:
        results = {}

        results["sportclip_per_frame_ms"] = measure_sportclip(video_path, video_name, temp_dir)
        results["qddetr_per_frame_ms"] = measure_qddetr(video_path, video_name, temp_dir)

        face_det_ms, avg_faces = measure_face_detection(video_path)
        results["face_detection_per_frame_ms"] = face_det_ms
        results["avg_faces_per_frame"] = avg_faces

        results["arcface_per_face_ms"] = measure_arcface(video_path, temp_dir)
        results["transface_per_face_ms"] = measure_transface(video_path, temp_dir)
        results["resemblance_per_comparison_ms"] = measure_resemblance()
        results["clip_assignment_per_segment_ms"] = measure_clip_assignment()

        print(f"\n\n{'='*80}")
        print("RESULTS")
        print(f"{'='*80}")
        print(f"SportCLIP:       {results['sportclip_per_frame_ms']:.2f} ms/frame")
        print(f"QD-DETR:         {results['qddetr_per_frame_ms']:.2f} ms/frame")
        print(f"Face Detection:  {results['face_detection_per_frame_ms']:.2f} ms/frame")
        print(f"ArcFace:         {results['arcface_per_face_ms']:.2f} ms/face")
        print(f"TransFace:       {results['transface_per_face_ms']:.2f} ms/face")
        print(f"Resemblance:     {results['resemblance_per_comparison_ms']:.4f} ms/comparison")
        print(f"Clip Assignment: {results['clip_assignment_per_segment_ms']:.4f} ms/segment")
        print(f"Avg faces:       {results['avg_faces_per_frame']:.2f}/frame")
        print(f"{'='*80}")

        generate_measurement_report(results, "measurements_report.md")
        generate_latex_table(results, "table.tex")

    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)


if __name__ == "__main__":
    main()
