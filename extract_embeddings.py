"""
Unified Embedding Extraction Script

This script handles all embedding extraction tasks in the project:
1. Video Faces Mode: Extract embeddings from cropped video faces
2. Original Targets Mode: Extract embeddings from original target face images
3. Updated Targets Mode: Extract embeddings from updated target faces (best matches from video)

Features:
- Progressive saving: saves embeddings incrementally to avoid memory issues
- Resume support: skips already processed embeddings if script is restarted
- Unified interface for all embedding extraction workflows
"""

import os
import sys
import json
import numpy as np
import h5py
import cv2
from pathlib import Path
from typing import List, Dict, Set, Optional
from tqdm import tqdm
import argparse

# Use tf-keras for compatibility with DeepFace (Keras 2 API)
os.environ["TF_USE_LEGACY_KERAS"] = "1"

# Configure TensorFlow for ArcFace (via DeepFace)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
try:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception as e:
    print(f"GPU configuration warning: {e}")

from deepface import DeepFace
from insightface.app import FaceAnalysis

# TransFace configuration
TRANSFACE_DIR = "/home/mrt/Projects/TransFace"
TRANSFACE_BATCH_SCRIPT = os.path.join(TRANSFACE_DIR, "batch_recognize.py")
TRANSFACE_VENV_PYTHON = os.path.join(TRANSFACE_DIR, "venv", "bin", "python")

# Batch size for progressive saving (to limit memory usage)
SAVE_BATCH_SIZE = 1000


# =============================================================================
# Common Functions
# =============================================================================


def detect_and_crop_face(image_path: str, face_detector: FaceAnalysis) -> Optional[np.ndarray]:
    """
    Detect and crop the largest face from an image

    Args:
        image_path: Path to the image
        face_detector: InsightFace detector

    Returns:
        Cropped face image or None if no face detected
    """
    img = cv2.imread(image_path)
    if img is None:
        print(f"  Error: Could not read image {image_path}")
        return None

    faces = face_detector.get(img)
    if not faces:
        print(f"  Warning: No face detected in {image_path}")
        return None

    # Get the largest face
    largest_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    bbox = largest_face.bbox.astype(int)
    x1, y1, x2, y2 = bbox

    # Ensure coordinates are within bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img.shape[1], x2)
    y2 = min(img.shape[0], y2)

    face_crop = img[y1:y2, x1:x2]
    return face_crop


def extract_arcface_embedding(image_path: str) -> Optional[np.ndarray]:
    """
    Extract ArcFace embedding from an image

    Args:
        image_path: Path to the image

    Returns:
        Embedding as numpy array or None if extraction failed
    """
    try:
        embedding = DeepFace.represent(img_path=image_path, model_name="ArcFace", enforce_detection=False)

        if isinstance(embedding, list) and len(embedding) > 0:
            return np.array(embedding[0]["embedding"], dtype=np.float32)
    except Exception as e:
        print(f"  Error extracting ArcFace embedding: {e}")
    return None


def extract_transface_embedding(image_path: str, transface_weight: str, transface_network: str) -> Optional[np.ndarray]:
    """
    Extract TransFace embedding from an image

    Args:
        image_path: Path to the image
        transface_weight: Path to TransFace weights
        transface_network: TransFace network architecture

    Returns:
        Embedding as numpy array or None if extraction failed
    """
    import subprocess
    import tempfile

    try:
        # Create temp files
        with tempfile.NamedTemporaryFile(mode="w", suffix="_input.json", delete=False) as f:
            input_file = f.name
            json.dump({"images": [image_path]}, f)

        with tempfile.NamedTemporaryFile(mode="w", suffix="_output.json", delete=False) as f:
            output_file = f.name

        # Call TransFace
        cmd = [
            TRANSFACE_VENV_PYTHON,
            TRANSFACE_BATCH_SCRIPT,
            "--mode",
            "extract-batch",
            "--input-list",
            input_file,
            "--output",
            output_file,
            "--weight",
            transface_weight,
            "--network",
            transface_network,
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Load embedding
        with open(output_file, "r") as f:
            data = json.load(f)

        results = data.get("embeddings", {})
        if image_path in results:
            result_data = results[image_path]
            if result_data["status"] == "success" and result_data["embedding"] is not None:
                return np.array(result_data["embedding"], dtype=np.float32)

    except Exception as e:
        print(f"  Error extracting TransFace embedding: {e}")
    finally:
        # Clean up
        if "input_file" in locals() and os.path.exists(input_file):
            os.remove(input_file)
        if "output_file" in locals() and os.path.exists(output_file):
            os.remove(output_file)

    return None


def get_existing_embeddings(h5_path: Path) -> Set[str]:
    """
    Get set of embedding names already in an HDF5 file

    Args:
        h5_path: Path to the HDF5 file

    Returns:
        Set of embedding names (e.g., {"frame_000241_face_0", ...})
    """
    if not h5_path.exists():
        return set()

    try:
        with h5py.File(str(h5_path), "r") as f:
            return set(f.keys())
    except Exception:
        return set()


def save_embeddings_to_h5(embeddings: Dict[str, np.ndarray], output_path: Path):
    """
    Save embeddings to an HDF5 file (appends if file exists)

    Args:
        embeddings: Dictionary mapping face_name -> embedding
        output_path: Path to save the HDF5 file (e.g., video_name.h5)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if output_path.exists() else "w"
    with h5py.File(str(output_path), mode) as f:
        for face_name, embedding in embeddings.items():
            if face_name not in f:
                f.create_dataset(face_name, data=embedding, compression="gzip", compression_opts=4)


# =============================================================================
# Mode 1: Video Faces (Extract from cropped video faces)
# =============================================================================


def extract_arcface_embeddings_progressive(image_paths: List[str], output_path: Path, gpu_id: int = 0, save_interval: int = SAVE_BATCH_SIZE) -> int:
    """
    Extract ArcFace embeddings and save progressively to HDF5

    Args:
        image_paths: List of paths to face images
        output_path: Path to HDF5 file to save embeddings
        gpu_id: GPU ID to use
        save_interval: Number of embeddings to accumulate before saving

    Returns:
        Number of embeddings successfully extracted
    """
    # Set GPU device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    # Get already processed embeddings for resume support
    existing = get_existing_embeddings(output_path)

    # Filter out already processed images
    remaining_paths = []
    for img_path in image_paths:
        face_name = Path(img_path).stem
        if face_name not in existing:
            remaining_paths.append(img_path)

    if not remaining_paths:
        print(f"  All {len(image_paths)} embeddings already exist, skipping...")
        return len(existing)

    if existing:
        print(f"  Resuming: {len(existing)} already processed, {len(remaining_paths)} remaining")

    embeddings_buffer = {}
    total_extracted = len(existing)

    for img_path in tqdm(remaining_paths, desc=f"  [GPU {gpu_id}] ArcFace", unit="img"):
        if not os.path.exists(img_path):
            continue

        try:
            embedding = DeepFace.represent(img_path=img_path, model_name="ArcFace", enforce_detection=False)

            if isinstance(embedding, list) and len(embedding) > 0:
                face_name = Path(img_path).stem
                embeddings_buffer[face_name] = np.array(embedding[0]["embedding"], dtype=np.float32)
                total_extracted += 1

                # Save periodically to avoid memory issues
                if len(embeddings_buffer) >= save_interval:
                    save_embeddings_to_h5(embeddings_buffer, output_path)
                    embeddings_buffer = {}

        except Exception as e:
            print(f"  [GPU {gpu_id}] Error processing {img_path}: {e}")

    # Save remaining embeddings
    if embeddings_buffer:
        save_embeddings_to_h5(embeddings_buffer, output_path)

    return total_extracted


def extract_transface_embeddings_batch(image_paths: List[str], transface_weight: str, transface_network: str, temp_dir: str = "/tmp") -> Dict[str, np.ndarray]:
    """
    Extract TransFace embeddings using batch processing

    Args:
        image_paths: List of paths to face images
        transface_weight: Path to TransFace model weights
        transface_network: TransFace network architecture
        temp_dir: Temporary directory for intermediate JSON files

    Returns:
        Dictionary mapping face_name -> embedding (numpy array)
    """
    import subprocess
    import tempfile

    if not image_paths:
        return {}

    # Create temp files for input list and output
    with tempfile.NamedTemporaryFile(mode="w", suffix="_input.json", delete=False, dir=temp_dir) as f:
        input_file = f.name
        json.dump({"images": image_paths}, f)

    with tempfile.NamedTemporaryFile(mode="w", suffix="_output.json", delete=False, dir=temp_dir) as f:
        output_file = f.name

    try:
        # Call TransFace batch subprocess
        cmd = [
            TRANSFACE_VENV_PYTHON,
            TRANSFACE_BATCH_SCRIPT,
            "--mode",
            "extract-batch",
            "--input-list",
            input_file,
            "--output",
            output_file,
            "--weight",
            transface_weight,
            "--network",
            transface_network,
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"\n  ERROR: TransFace batch processing failed!")
            print(f"  Command: {' '.join(cmd)}")
            if e.stdout:
                print(f"  STDOUT:\n{e.stdout}")
            if e.stderr:
                print(f"  STDERR:\n{e.stderr}")
            raise

        # Load all embeddings from JSON
        with open(output_file, "r") as f:
            data = json.load(f)

        results = data.get("embeddings", {})

        # Convert to numpy arrays with face_name as key
        embeddings = {}
        failed_count = 0
        for img_path, result_data in results.items():
            if result_data["status"] == "success" and result_data["embedding"] is not None:
                face_name = Path(img_path).stem
                embeddings[face_name] = np.array(result_data["embedding"], dtype=np.float32)
            else:
                failed_count += 1

        if failed_count > 0:
            print(f"  Warning: Failed to extract {failed_count} embeddings")

        return embeddings

    finally:
        # Clean up temp files
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)


def extract_transface_embeddings_progressive(image_paths: List[str], output_path: Path, transface_weight: str, transface_network: str, batch_size: int = 1000) -> int:
    """
    Extract TransFace embeddings and save progressively to HDF5

    Args:
        image_paths: List of paths to face images
        output_path: Path to HDF5 file to save embeddings
        transface_weight: Path to TransFace model weights
        transface_network: TransFace network architecture
        batch_size: Number of images to process in each batch

    Returns:
        Number of embeddings successfully extracted
    """
    # Get already processed embeddings for resume support
    existing = get_existing_embeddings(output_path)

    # Filter out already processed images
    remaining_paths = []
    for img_path in image_paths:
        face_name = Path(img_path).stem
        if face_name not in existing:
            remaining_paths.append(img_path)

    if not remaining_paths:
        print(f"  All {len(image_paths)} embeddings already exist, skipping...")
        return len(existing)

    if existing:
        print(f"  Resuming: {len(existing)} already processed, {len(remaining_paths)} remaining")

    total_extracted = len(existing)
    num_batches = (len(remaining_paths) - 1) // batch_size + 1

    for i in range(0, len(remaining_paths), batch_size):
        batch_paths = remaining_paths[i : i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Batch {batch_num}/{num_batches}: {len(batch_paths)} images")

        # Extract embeddings for this batch
        batch_embeddings = extract_transface_embeddings_batch(image_paths=batch_paths, transface_weight=transface_weight, transface_network=transface_network)

        # Save immediately after each batch
        if batch_embeddings:
            save_embeddings_to_h5(batch_embeddings, output_path)
            total_extracted += len(batch_embeddings)
            print(f"    Saved {len(batch_embeddings)} embeddings")

    return total_extracted


def find_all_face_crops(cropped_faces_dir: str) -> Dict[str, List[str]]:
    """
    Find all face crop images organized by video

    Args:
        cropped_faces_dir: Directory containing cropped faces

    Returns:
        Dictionary mapping video_name -> list of face image paths
    """
    cropped_faces_dir = Path(cropped_faces_dir)

    if not cropped_faces_dir.exists():
        print(f"Error: Directory {cropped_faces_dir} does not exist")
        return {}

    video_faces = {}

    # Each video has its own subdirectory
    for video_dir in sorted(cropped_faces_dir.iterdir()):
        if not video_dir.is_dir():
            continue

        video_name = video_dir.name

        # Find all face images in this video directory
        face_images = list(video_dir.glob("*.jpg")) + list(video_dir.glob("*.png"))

        if face_images:
            video_faces[video_name] = sorted([str(img) for img in face_images])

    return video_faces


def process_arcface_embeddings(video_faces: Dict[str, List[str]], output_dir: Path, gpu_id: int = 0):
    """
    Extract and save ArcFace embeddings for all videos

    Args:
        video_faces: Dictionary mapping video_name -> list of face image paths
        output_dir: Output directory for embeddings (cropped_faces_embeddings/ArcFace/)
        gpu_id: GPU ID to use
    """
    print("\n" + "=" * 80)
    print("EXTRACTING ARCFACE EMBEDDINGS")
    print("=" * 80)

    total_faces = sum(len(faces) for faces in video_faces.values())
    print(f"Total videos: {len(video_faces)}")
    print(f"Total faces: {total_faces}")
    print(f"Using GPU: {gpu_id}")
    print(f"Output format: HDF5 (one file per video)")
    print(f"Progressive save interval: {SAVE_BATCH_SIZE} embeddings")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_faces = 0

    for video_name, face_paths in video_faces.items():
        print(f"\nProcessing video: {video_name} ({len(face_paths)} faces)")

        output_path = output_dir / f"{video_name}.h5"

        # Extract and save progressively
        num_extracted = extract_arcface_embeddings_progressive(image_paths=face_paths, output_path=output_path, gpu_id=gpu_id)

        processed_faces += num_extracted
        print(f"  Total embeddings in {output_path.name}: {num_extracted}")

    print(f"\n✓ ArcFace: Processed {processed_faces}/{total_faces} faces")


def process_transface_embeddings(video_faces: Dict[str, List[str]], output_dir: Path, transface_weight: str, transface_network: str, batch_size: int = 1000):
    """
    Extract and save TransFace embeddings for all videos

    Args:
        video_faces: Dictionary mapping video_name -> list of face image paths
        output_dir: Output directory for embeddings (cropped_faces_embeddings/TransFace/)
        transface_weight: Path to TransFace model weights
        transface_network: TransFace network architecture
        batch_size: Number of images to process in each batch
    """
    print("\n" + "=" * 80)
    print("EXTRACTING TRANSFACE EMBEDDINGS")
    print("=" * 80)

    total_faces = sum(len(faces) for faces in video_faces.values())
    print(f"Total videos: {len(video_faces)}")
    print(f"Total faces: {total_faces}")
    print(f"Batch size: {batch_size}")
    print(f"Weight: {transface_weight}")
    print(f"Network: {transface_network}")
    print(f"Output format: HDF5 (one file per video)")

    output_dir.mkdir(parents=True, exist_ok=True)
    processed_faces = 0

    for video_name, face_paths in video_faces.items():
        print(f"\nProcessing video: {video_name} ({len(face_paths)} faces)")

        output_path = output_dir / f"{video_name}.h5"

        # Extract and save progressively
        num_extracted = extract_transface_embeddings_progressive(
            image_paths=face_paths, output_path=output_path, transface_weight=transface_weight, transface_network=transface_network, batch_size=batch_size
        )

        processed_faces += num_extracted
        print(f"  Total embeddings in {output_path.name}: {num_extracted}")

    print(f"\n✓ TransFace: Processed {processed_faces}/{total_faces} faces")


def run_video_faces_mode(args):
    """Run video faces extraction mode"""
    print("=" * 80)
    print("MODE: VIDEO FACES EMBEDDING EXTRACTION")
    print("=" * 80)
    print(f"\nSearching for face crops in: {args.input_dir}")

    video_faces = find_all_face_crops(args.input_dir)

    if not video_faces:
        print(f"\n⚠ No face crops found in {args.input_dir}")
        print("Please run extract_all_faces.py first to extract face crops from videos.")
        return

    total_videos = len(video_faces)
    total_faces = sum(len(faces) for faces in video_faces.values())

    print(f"✓ Found {total_videos} video(s) with {total_faces} total face crops")
    for video_name, faces in list(video_faces.items())[:5]:
        print(f"  - {video_name}: {len(faces)} faces")
    if total_videos > 5:
        print(f"  ... and {total_videos - 5} more videos")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each model
    if "ArcFace" in args.models:
        arcface_output_dir = output_dir / "ArcFace"
        process_arcface_embeddings(video_faces=video_faces, output_dir=arcface_output_dir, gpu_id=args.gpu_id)

    if "TransFace" in args.models:
        # Verify TransFace is available
        if not os.path.exists(TRANSFACE_BATCH_SCRIPT):
            print(f"\n⚠ TransFace batch script not found: {TRANSFACE_BATCH_SCRIPT}")
            print("Skipping TransFace extraction.")
        elif not os.path.exists(TRANSFACE_VENV_PYTHON):
            print(f"\n⚠ TransFace Python interpreter not found: {TRANSFACE_VENV_PYTHON}")
            print("Skipping TransFace extraction.")
        elif not os.path.exists(args.transface_weight):
            print(f"\n⚠ TransFace weights not found: {args.transface_weight}")
            print("Skipping TransFace extraction.")
        else:
            transface_output_dir = output_dir / "TransFace"
            process_transface_embeddings(
                video_faces=video_faces, output_dir=transface_output_dir, transface_weight=args.transface_weight, transface_network=args.transface_network, batch_size=args.batch_size
            )

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Embeddings saved to: {args.output_dir}/")

    if "ArcFace" in args.models:
        print(f"  - ArcFace embeddings: {args.output_dir}/ArcFace/<video_name>.h5")
    if "TransFace" in args.models:
        print(f"  - TransFace embeddings: {args.output_dir}/TransFace/<video_name>.h5")


# =============================================================================
# Mode 2: Original Targets (Extract from original target images)
# =============================================================================


def process_original_target_face(target_image_path: Path, output_dir: Path, face_detector: FaceAnalysis, transface_weight: str, transface_network: str) -> bool:
    """
    Extract embeddings from original target face image

    Args:
        target_image_path: Path to target face image
        output_dir: Output directory for embeddings
        face_detector: InsightFace detector
        transface_weight: Path to TransFace weights
        transface_network: TransFace network

    Returns:
        True if successful, False otherwise
    """
    target_name = target_image_path.stem
    print(f"\nProcessing target: {target_name}")

    # Create output directory for this target
    target_output_dir = output_dir / target_name
    target_output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Detect and crop face from target image
    print("  Step 1: Detecting and cropping face from target image...")
    face_crop = detect_and_crop_face(str(target_image_path), face_detector)
    if face_crop is None:
        return False

    # Save cropped target face
    target_crop_path = target_output_dir / "face_crop.jpg"
    cv2.imwrite(str(target_crop_path), face_crop)
    print(f"  Saved cropped face to {target_crop_path}")

    # Step 2: Extract embeddings for target face
    print("  Step 2: Extracting embeddings from target face...")

    # ArcFace
    print("    Extracting ArcFace embedding...")
    arcface_emb = extract_arcface_embedding(str(target_crop_path))
    if arcface_emb is None:
        print("    Failed to extract ArcFace embedding")
        return False

    # Save ArcFace embedding
    arcface_emb_path = target_output_dir / "target_arcface_embedding.npy"
    np.save(str(arcface_emb_path), arcface_emb)
    print(f"    ✓ Saved ArcFace embedding")

    # TransFace
    print("    Extracting TransFace embedding...")
    transface_emb = extract_transface_embedding(str(target_crop_path), transface_weight, transface_network)
    if transface_emb is None:
        print("    Failed to extract TransFace embedding")
        return False

    # Save TransFace embedding
    transface_emb_path = target_output_dir / "target_transface_embedding.npy"
    np.save(str(transface_emb_path), transface_emb)
    print(f"    ✓ Saved TransFace embedding")

    print(f"  ✓ Successfully processed {target_name}")
    return True


def run_original_targets_mode(args):
    """Run original targets extraction mode"""
    target_faces_dir = Path(args.target_faces)
    output_dir = Path(args.output_dir)

    print("=" * 80)
    print("MODE: ORIGINAL TARGET FACES EMBEDDING EXTRACTION")
    print("=" * 80)

    # Validate directories
    if not target_faces_dir.exists():
        print(f"\nError: Target faces directory not found: {target_faces_dir}")
        return

    # Find target images
    target_images = list(target_faces_dir.glob("*.jpg")) + list(target_faces_dir.glob("*.png")) + list(target_faces_dir.glob("*.jpeg"))

    if not target_images:
        print(f"\nError: No target images found in {target_faces_dir}")
        return

    print(f"\nFound {len(target_images)} target image(s):")
    for img in target_images:
        print(f"  - {img.name}")

    print(f"\nOutput directory: {output_dir}")

    # Initialize face detector
    print("\nInitializing face detector...")
    try:
        face_detector = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        face_detector.prepare(ctx_id=0, det_size=(640, 640))
        print("Face detector initialized (GPU enabled)")
    except Exception as e:
        print(f"GPU initialization failed, using CPU: {e}")
        face_detector = FaceAnalysis(providers=["CPUExecutionProvider"])
        face_detector.prepare(ctx_id=-1, det_size=(640, 640))
        print("Face detector initialized (CPU mode)")

    # Process each target
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = 0
    failed = 0

    for target_img in target_images:
        if process_original_target_face(target_img, output_dir, face_detector, args.transface_weight, args.transface_network):
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nEmbeddings saved in: {output_dir}/")
    print("\nNote: These are embeddings from ORIGINAL target images.")


# =============================================================================
# Mode 3: Updated Targets (Extract from updated target images)
# =============================================================================


def process_updated_target(target_dir: Path, target_name: str, video_name: str, transface_weight: str, transface_network: str) -> bool:
    """
    Extract embeddings from updated target face images

    Args:
        target_dir: Directory for this target (e.g., target_faces_updated/starc/)
        target_name: Name of target person
        video_name: Video name
        transface_weight: Path to TransFace weights
        transface_network: TransFace network

    Returns:
        True if successful, False otherwise
    """
    print(f"\nProcessing updated target: {target_name}")

    # Look for updated target images
    arcface_img = target_dir / "face_crop_arcface_updated.jpg"
    transface_img = target_dir / "face_crop_transface_updated.jpg"

    if not arcface_img.exists() and not transface_img.exists():
        print(f"  Warning: No updated target images found for {target_name}")
        return False

    # Extract ArcFace embedding from updated target
    if arcface_img.exists():
        print(f"  Extracting ArcFace embedding from updated target...")
        arcface_emb = extract_arcface_embedding(str(arcface_img))
        if arcface_emb is not None:
            # Save with suffix to distinguish from original
            emb_path = target_dir / "updated_target_arcface_embedding.npy"
            np.save(str(emb_path), arcface_emb)
            print(f"    ✓ Saved to: {emb_path}")
        else:
            print(f"    ✗ Failed to extract ArcFace embedding")
            return False
    else:
        print(f"  Warning: ArcFace updated target image not found")

    # Extract TransFace embedding from updated target
    if transface_img.exists():
        print(f"  Extracting TransFace embedding from updated target...")
        transface_emb = extract_transface_embedding(str(transface_img), transface_weight, transface_network)
        if transface_emb is not None:
            # Save with suffix to distinguish from original
            emb_path = target_dir / "updated_target_transface_embedding.npy"
            np.save(str(emb_path), transface_emb)
            print(f"    ✓ Saved to: {emb_path}")
        else:
            print(f"    ✗ Failed to extract TransFace embedding")
            return False
    else:
        print(f"  Warning: TransFace updated target image not found")

    print(f"  ✓ Successfully processed {target_name}")
    return True


def run_updated_targets_mode(args):
    """Run updated targets extraction mode"""
    updated_targets_dir = Path(args.updated_targets)
    video_name = args.video

    print("=" * 80)
    print("MODE: UPDATED TARGET FACES EMBEDDING EXTRACTION (SECOND PASS)")
    print("=" * 80)
    print(f"\nVideo: {video_name}")

    # Validate directories
    if not updated_targets_dir.exists():
        print(f"\nError: Updated targets directory not found: {updated_targets_dir}")
        print("Please run find_updated_target.py first.")
        return

    # Find all target directories
    target_dirs = [d for d in updated_targets_dir.iterdir() if d.is_dir()]

    if not target_dirs:
        print(f"\nError: No target directories found in {updated_targets_dir}")
        return

    print(f"\nFound {len(target_dirs)} target(s):")
    for target_dir in target_dirs:
        print(f"  - {target_dir.name}")

    # Process each target
    successful = 0
    failed = 0

    for target_dir in target_dirs:
        target_name = target_dir.name

        # Check if this target has a matching log for the video
        log_file = target_dir / f"{target_name}_log.txt"
        if not log_file.exists():
            print(f"\nSkipping {target_name} (no matching log for {video_name})")
            continue

        if process_updated_target(target_dir, target_name, video_name, args.transface_weight, args.transface_network):
            successful += 1
        else:
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"\nUpdated embeddings saved in: {updated_targets_dir}/")


# =============================================================================
# Main Entry Point
# =============================================================================


def main():
    """Main function to extract embeddings based on mode"""
    parser = argparse.ArgumentParser(
        description="Unified embedding extraction for video faces, original targets, and updated targets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  video_faces:      Extract embeddings from cropped video faces (default)
  original_targets: Extract embeddings from original target face images
  updated_targets:  Extract embeddings from updated target faces (best matches from video)

Examples:
  # Extract embeddings from video faces (default mode)
  python extract_embeddings.py --mode video_faces

  # Extract embeddings from original target faces
  python extract_embeddings.py --mode original_targets

  # Extract embeddings from updated target faces
  python extract_embeddings.py --mode updated_targets --video highjump_video1

  # Video faces mode: Extract only ArcFace embeddings
  python extract_embeddings.py --mode video_faces --models ArcFace

  # Video faces mode: Use custom directories
  python extract_embeddings.py --mode video_faces --input-dir cropped_faces --output-dir my_embeddings
        """,
    )

    parser.add_argument("--mode", type=str, choices=["video_faces", "original_targets", "updated_targets"], default="video_faces", help="Extraction mode (default: video_faces)")

    # Common arguments
    parser.add_argument(
        "--transface-weight", type=str, default="/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt", help="Path to TransFace model weights"
    )
    parser.add_argument("--transface-network", type=str, default="vit_l_dp005_mask_005", help="TransFace network architecture (default: vit_l_dp005_mask_005)")

    # Video faces mode arguments
    parser.add_argument("--input-dir", type=str, default="cropped_faces", help="[video_faces] Directory containing cropped face images (default: cropped_faces)")
    parser.add_argument("--output-dir", type=str, default="cropped_faces_embeddings", help="[video_faces] Directory to save embeddings (default: cropped_faces_embeddings)")
    parser.add_argument(
        "--models", nargs="+", choices=["ArcFace", "TransFace"], default=["ArcFace", "TransFace"], help="[video_faces] Which models to use for embedding extraction (default: both)"
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="[video_faces] GPU ID to use for ArcFace (0 or 1, default: 0)")
    parser.add_argument("--batch-size", type=int, default=1000, help="[video_faces] Batch size for TransFace processing (default: 1000)")

    # Original targets mode arguments
    parser.add_argument("--target-faces", type=str, default="target_faces", help="[original_targets] Directory containing target face images (default: target_faces)")

    # Updated targets mode arguments
    parser.add_argument(
        "--updated-targets", type=str, default="target_faces_updated", help="[updated_targets] Directory containing updated target faces (default: target_faces_updated)"
    )
    parser.add_argument("--video", type=str, default=None, help="[updated_targets] Video name to process (without extension, required for updated_targets mode)")

    args = parser.parse_args()

    # Route to appropriate mode
    if args.mode == "video_faces":
        run_video_faces_mode(args)
    elif args.mode == "original_targets":
        # Override output_dir for original_targets to save in target_faces_updated
        args.output_dir = "target_faces_updated"
        run_original_targets_mode(args)
    elif args.mode == "updated_targets":
        if not args.video:
            print("Error: --video is required for updated_targets mode")
            parser.print_help()
            return
        run_updated_targets_mode(args)


if __name__ == "__main__":
    main()
