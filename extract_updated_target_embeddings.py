"""
Extract Embeddings from Updated Target Faces (Second Pass)

This script extracts embeddings from the updated target face crops (best matches from video).
These embeddings will be used to compute resemblance curves that should achieve 100% similarity
for the exact frame where the updated target came from.

The difference from find_updated_target.py:
- find_updated_target.py: Extracts embeddings from ORIGINAL target → finds best match in video
- This script: Extracts embeddings from UPDATED target (best match) → to be used in resemblance curves
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Optional
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

# TransFace configuration
TRANSFACE_DIR = "/home/mrt/Projects/TransFace"
TRANSFACE_BATCH_SCRIPT = os.path.join(TRANSFACE_DIR, "batch_recognize.py")
TRANSFACE_VENV_PYTHON = os.path.join(TRANSFACE_DIR, "venv", "bin", "python")


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


def process_updated_target(
    target_dir: Path, target_name: str, video_name: str, transface_weight: str, transface_network: str
) -> bool:
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
    arcface_img = target_dir / "updated_target_arcface.jpg"
    transface_img = target_dir / "updated_target_transface.jpg"

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


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from updated target faces (second pass)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all updated targets for a video
  python extract_updated_target_embeddings.py --video highjump_video1

  # Use custom directories and TransFace config
  python extract_updated_target_embeddings.py --video highjump_video1 \\
                                               --updated-targets custom/updated \\
                                               --transface-weight path/to/weights.pt
        """,
    )

    parser.add_argument("--updated-targets", type=str, default="target_faces_updated", help="Directory containing updated target faces (default: target_faces_updated)")
    parser.add_argument("--video", type=str, required=True, help="Video name to process (without extension)")
    parser.add_argument(
        "--transface-weight", type=str, default="/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt", help="Path to TransFace model weights"
    )
    parser.add_argument("--transface-network", type=str, default="vit_l_dp005_mask_005", help="TransFace network architecture (default: vit_l_dp005_mask_005)")

    args = parser.parse_args()

    # Setup paths
    updated_targets_dir = Path(args.updated_targets)
    video_name = args.video

    print("=" * 80)
    print("EXTRACT EMBEDDINGS FROM UPDATED TARGETS (SECOND PASS)")
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
        log_file = target_dir / f"{target_name}_{video_name}_matching_log.txt"
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
    print("These embeddings will be used for resemblance curve computation with --use-updated-target flag.")


if __name__ == "__main__":
    main()
