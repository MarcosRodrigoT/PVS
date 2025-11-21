"""
Extract Embeddings from Original Target Faces

This script extracts embeddings directly from the original target face images in target_faces/.
Used when user wants face recognition without finding updated targets first (first pass only).

The difference from other scripts:
- find_updated_target.py: Extracts from original target → finds best match → saves updated target image
- extract_updated_target_embeddings.py: Extracts from updated target image (best match from video)
- This script: Extracts from original target only (no searching for best match)
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Optional
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


def process_target_face(
    target_image_path: Path, output_dir: Path, face_detector: FaceAnalysis, transface_weight: str, transface_network: str
) -> bool:
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
    target_crop_path = target_output_dir / "target_face_crop.jpg"
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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from original target faces (no video matching)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all target faces
  python extract_original_target_embeddings.py

  # Use custom directories and TransFace config
  python extract_original_target_embeddings.py --target-faces custom/targets \\
                                                --output custom/output \\
                                                --transface-weight path/to/weights.pt
        """,
    )

    parser.add_argument("--target-faces", type=str, default="target_faces", help="Directory containing target face images (default: target_faces)")
    parser.add_argument("--output", type=str, default="target_faces_updated", help="Output directory for embeddings (default: target_faces_updated)")
    parser.add_argument(
        "--transface-weight", type=str, default="/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt", help="Path to TransFace model weights"
    )
    parser.add_argument("--transface-network", type=str, default="vit_l_dp005_mask_005", help="TransFace network architecture (default: vit_l_dp005_mask_005)")

    args = parser.parse_args()

    # Setup paths
    target_faces_dir = Path(args.target_faces)
    output_dir = Path(args.output)

    print("=" * 80)
    print("EXTRACT EMBEDDINGS FROM ORIGINAL TARGET FACES")
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
        if process_target_face(target_img, output_dir, face_detector, args.transface_weight, args.transface_network):
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
    print("To find updated targets from video, use --use-updated-target flag with main.py")


if __name__ == "__main__":
    main()
