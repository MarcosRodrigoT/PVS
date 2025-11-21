"""
Find Updated Target Face
Compares target face against all face crops to find the best match (updated target)

This script:
1. Detects and crops faces from target images in target_faces/
2. Extracts embeddings (ArcFace and TransFace) for target faces
3. Compares against all face crops in videos to find best matches
4. Saves updated targets with their embeddings and similarity logs
"""

import os
import sys
import json
import numpy as np
import h5py
import cv2
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse
from datetime import datetime

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
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)

    return None


def compute_cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings (normalized to [0, 1])

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Similarity score in [0, 1] range
    """
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    # Normalize to [0, 1]
    return (similarity + 1) / 2


def find_best_match(target_embedding: np.ndarray, embeddings_h5_path: Path, cropped_faces_dir: Path, top_k: int = 10) -> List[Tuple[str, float, str]]:
    """
    Find best matching faces from embeddings

    Args:
        target_embedding: Target face embedding
        embeddings_h5_path: Path to HDF5 file with embeddings
        cropped_faces_dir: Directory with cropped face images
        top_k: Number of top matches to return

    Returns:
        List of (face_name, similarity, image_path) tuples, sorted by similarity (highest first)
    """
    if not embeddings_h5_path.exists():
        return []

    matches = []

    with h5py.File(str(embeddings_h5_path), "r") as f:
        for face_name in tqdm(f.keys(), desc="  Comparing embeddings", unit="face"):
            embedding = f[face_name][:]
            similarity = compute_cosine_similarity(target_embedding, embedding)

            # Construct image path
            image_path = cropped_faces_dir / f"{face_name}.jpg"

            matches.append((face_name, similarity, str(image_path)))

    # Sort by similarity (highest first) and return top k
    matches.sort(key=lambda x: x[1], reverse=True)
    return matches[:top_k]


def process_target_face(
    target_image_path: Path,
    cropped_faces_dir: Path,
    embeddings_dir: Path,
    output_dir: Path,
    face_detector: FaceAnalysis,
    transface_weight: str,
    transface_network: str,
    video_name: str,
    similarity_threshold: float = 0.5,
    top_k_matches: int = 10,
) -> Optional[str]:
    """
    Process a single target face to find its best match

    Args:
        target_image_path: Path to target face image
        cropped_faces_dir: Directory with cropped faces
        embeddings_dir: Directory with embeddings (ArcFace/ and TransFace/ subdirs)
        output_dir: Output directory for updated target
        face_detector: InsightFace detector
        transface_weight: Path to TransFace weights
        transface_network: TransFace network
        video_name: Name of video to process
        similarity_threshold: Minimum similarity to accept a match
        top_k_matches: Number of top matches to log

    Returns:
        Path to log file or None if failed
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
        return None

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
        return None

    # Save ArcFace embedding
    arcface_emb_path = target_output_dir / "target_arcface_embedding.npy"
    np.save(str(arcface_emb_path), arcface_emb)

    # TransFace
    print("    Extracting TransFace embedding...")
    transface_emb = extract_transface_embedding(str(target_crop_path), transface_weight, transface_network)
    if transface_emb is None:
        print("    Failed to extract TransFace embedding")
        return None

    # Save TransFace embedding
    transface_emb_path = target_output_dir / "target_transface_embedding.npy"
    np.save(str(transface_emb_path), transface_emb)

    print(f"  Saved embeddings to {target_output_dir}")

    # Step 3: Find best matches in specified video
    print(f"  Step 3: Finding best matches in video: {video_name}")

    arcface_dir = embeddings_dir / "ArcFace"
    transface_dir = embeddings_dir / "TransFace"

    # Verify video files exist
    arcface_h5 = arcface_dir / f"{video_name}.h5"
    transface_h5 = transface_dir / f"{video_name}.h5"
    cropped_video_dir = cropped_faces_dir / video_name

    if not arcface_h5.exists():
        print(f"  Error: ArcFace embeddings not found: {arcface_h5}")
        return None

    if not transface_h5.exists():
        print(f"  Error: TransFace embeddings not found: {transface_h5}")
        return None

    if not cropped_video_dir.exists():
        print(f"  Error: Cropped faces directory not found: {cropped_video_dir}")
        return None

    # ArcFace matches
    print("    Finding ArcFace matches...")
    arcface_matches = find_best_match(arcface_emb, arcface_h5, cropped_video_dir, top_k=top_k_matches)
    all_arcface_matches = [(video_name, *match) for match in arcface_matches]

    # TransFace matches
    print("    Finding TransFace matches...")
    transface_matches = find_best_match(transface_emb, transface_h5, cropped_video_dir, top_k=top_k_matches)
    all_transface_matches = [(video_name, *match) for match in transface_matches]

    # Matches are already sorted by find_best_match

    # Step 4: Find best overall match (updated target)
    print("\n  Step 4: Selecting best match as updated target...")

    best_arcface = all_arcface_matches[0] if all_arcface_matches else None
    best_transface = all_transface_matches[0] if all_transface_matches else None

    # Create log file
    log_path = target_output_dir / f"{target_name}_log.txt"
    with open(log_path, "w") as log:
        log.write(f"Updated Target Matching Log\n")
        log.write(f"=" * 80 + "\n")
        log.write(f"Target: {target_name}\n")
        log.write(f"Video: {video_name}\n")
        log.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log.write(f"Similarity threshold: {similarity_threshold}\n")
        log.write(f"\n")

        # ArcFace results
        log.write(f"ArcFace Best Matches (Top {top_k_matches} overall)\n")
        log.write(f"-" * 80 + "\n")
        if best_arcface:
            log.write(f"BEST MATCH:\n")
            log.write(f"  Video: {best_arcface[0]}\n")
            log.write(f"  Face: {best_arcface[1]}\n")
            log.write(f"  Similarity: {best_arcface[2]:.6f}\n")
            log.write(f"  Image: {best_arcface[3]}\n")
            log.write(f"\n")

            if best_arcface[2] >= similarity_threshold:
                # Copy best match as updated target
                updated_path = target_output_dir / "face_crop_arcface_updated.jpg"
                if os.path.exists(best_arcface[3]):
                    import shutil

                    shutil.copy2(best_arcface[3], updated_path)
                    log.write(f"  Updated target saved to: {updated_path}\n")
                    print(f"  ArcFace: Best match similarity = {best_arcface[2]:.6f} (ACCEPTED)")
                else:
                    log.write(f"  Warning: Image file not found\n")
                    print(f"  ArcFace: Best match similarity = {best_arcface[2]:.6f} (image not found)")
            else:
                log.write(f"  Below threshold ({similarity_threshold}), no updated target saved\n")
                print(f"  ArcFace: Best match similarity = {best_arcface[2]:.6f} (BELOW THRESHOLD)")

        log.write(f"\nTop {min(top_k_matches, len(all_arcface_matches))} ArcFace matches:\n")
        for i, (video, face, sim, img_path) in enumerate(all_arcface_matches[:top_k_matches], 1):
            log.write(f"  {i}. Video: {video}, Face: {face}, Similarity: {sim:.6f}\n")

        # TransFace results
        log.write(f"\n\nTransFace Best Matches (Top {top_k_matches} overall)\n")
        log.write(f"-" * 80 + "\n")
        if best_transface:
            log.write(f"BEST MATCH:\n")
            log.write(f"  Video: {best_transface[0]}\n")
            log.write(f"  Face: {best_transface[1]}\n")
            log.write(f"  Similarity: {best_transface[2]:.6f}\n")
            log.write(f"  Image: {best_transface[3]}\n")
            log.write(f"\n")

            if best_transface[2] >= similarity_threshold:
                # Copy best match as updated target
                updated_path = target_output_dir / "face_crop_transface_updated.jpg"
                if os.path.exists(best_transface[3]):
                    import shutil

                    shutil.copy2(best_transface[3], updated_path)
                    log.write(f"  Updated target saved to: {updated_path}\n")
                    print(f"  TransFace: Best match similarity = {best_transface[2]:.6f} (ACCEPTED)")
                else:
                    log.write(f"  Warning: Image file not found\n")
                    print(f"  TransFace: Best match similarity = {best_transface[2]:.6f} (image not found)")
            else:
                log.write(f"  Below threshold ({similarity_threshold}), no updated target saved\n")
                print(f"  TransFace: Best match similarity = {best_transface[2]:.6f} (BELOW THRESHOLD)")

        log.write(f"\nTop {min(top_k_matches, len(all_transface_matches))} TransFace matches:\n")
        for i, (video, face, sim, img_path) in enumerate(all_transface_matches[:top_k_matches], 1):
            log.write(f"  {i}. Video: {video}, Face: {face}, Similarity: {sim:.6f}\n")

    print(f"\n  Log saved to {log_path}")
    return str(log_path)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Find updated target face by comparing against all video face crops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process with default video (highjump_video1)
  python find_updated_target.py

  # Process a different video
  python find_updated_target.py --video longjump_video1

  # Use custom similarity threshold
  python find_updated_target.py --video highjump_video1 --threshold 0.6

  # Log more top matches
  python find_updated_target.py --video highjump_video1 --top-k 20

  # Use custom directories
  python find_updated_target.py --video highjump_video1 \\
                                --target-faces custom/targets \\
                                --cropped-faces custom/crops \\
                                --embeddings custom/embeddings
        """,
    )

    parser.add_argument("--target-faces", type=str, default="target_faces", help="Directory containing target face images (default: target_faces)")
    parser.add_argument("--cropped-faces", type=str, default="cropped_faces", help="Directory containing cropped face images (default: cropped_faces)")
    parser.add_argument("--embeddings", type=str, default="cropped_faces_embeddings", help="Directory containing face embeddings (default: cropped_faces_embeddings)")
    parser.add_argument("--output", type=str, default="target_faces_updated", help="Output directory for updated targets (default: target_faces_updated)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Minimum similarity threshold to accept a match (default: 0.5)")
    parser.add_argument("--top-k", type=int, default=10, help="Number of top matches to log (default: 10)")
    parser.add_argument(
        "--transface-weight", type=str, default="/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt", help="Path to TransFace model weights"
    )
    parser.add_argument("--transface-network", type=str, default="vit_l_dp005_mask_005", help="TransFace network architecture (default: vit_l_dp005_mask_005)")
    parser.add_argument("--video", type=str, default="highjump_video1", help="Video name to process (without extension, default: highjump_video1)")

    args = parser.parse_args()

    # Setup paths
    target_faces_dir = Path(args.target_faces)
    cropped_faces_dir = Path(args.cropped_faces)
    embeddings_dir = Path(args.embeddings)
    output_dir = Path(args.output)
    video_name = args.video

    print("=" * 80)
    print("FIND UPDATED TARGET FACE")
    print("=" * 80)

    # Validate directories
    if not target_faces_dir.exists():
        print(f"\nError: Target faces directory not found: {target_faces_dir}")
        return

    if not cropped_faces_dir.exists():
        print(f"\nError: Cropped faces directory not found: {cropped_faces_dir}")
        return

    if not embeddings_dir.exists():
        print(f"\nError: Embeddings directory not found: {embeddings_dir}")
        return

    # Find target images
    target_images = list(target_faces_dir.glob("*.jpg")) + list(target_faces_dir.glob("*.png")) + list(target_faces_dir.glob("*.jpeg"))

    if not target_images:
        print(f"\nError: No target images found in {target_faces_dir}")
        return

    print(f"\nFound {len(target_images)} target image(s):")
    for img in target_images:
        print(f"  - {img.name}")

    print(f"\nVideo to process: {video_name}")
    print(f"Similarity threshold: {args.threshold}")
    print(f"Top-K matches to log: {args.top_k}")
    print(f"Output directory: {output_dir}")

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
    log_files = []

    # Verify video files exist before processing
    arcface_h5 = embeddings_dir / "ArcFace" / f"{video_name}.h5"
    transface_h5 = embeddings_dir / "TransFace" / f"{video_name}.h5"
    cropped_video_dir = cropped_faces_dir / video_name

    if not arcface_h5.exists():
        print(f"\nError: ArcFace embeddings not found: {arcface_h5}")
        print("Please run extract_embeddings.py first.")
        return

    if not transface_h5.exists():
        print(f"\nError: TransFace embeddings not found: {transface_h5}")
        print("Please run extract_embeddings.py first.")
        return

    if not cropped_video_dir.exists():
        print(f"\nError: Cropped faces directory not found: {cropped_video_dir}")
        print("Please run extract_all_faces.py first.")
        return

    for target_img in target_images:
        log_file = process_target_face(
            target_image_path=target_img,
            cropped_faces_dir=cropped_faces_dir,
            embeddings_dir=embeddings_dir,
            output_dir=output_dir,
            face_detector=face_detector,
            transface_weight=args.transface_weight,
            transface_network=args.transface_network,
            video_name=video_name,
            similarity_threshold=args.threshold,
            top_k_matches=args.top_k,
        )

        if log_file:
            log_files.append(log_file)

    # Summary
    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Processed {len(target_images)} target image(s)")
    print(f"Results saved to: {output_dir}/")
    print(f"\nLog files:")
    for log in log_files:
        print(f"  - {log}")


if __name__ == "__main__":
    main()
