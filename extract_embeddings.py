"""
Extract embeddings from all face crops using ArcFace and TransFace
Processes all face crops in cropped_faces/ and saves embeddings to cropped_faces_embeddings/
Embeddings are stored in HDF5 format (one file per video) for efficient storage

Features:
- Progressive saving: saves embeddings incrementally to avoid memory issues
- Resume support: skips already processed embeddings if script is restarted
"""

import os
import sys
import json
import numpy as np
import h5py
from pathlib import Path
from typing import List, Dict, Set
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

# TransFace configuration
TRANSFACE_DIR = "/home/mrt/Projects/TransFace"
TRANSFACE_BATCH_SCRIPT = os.path.join(TRANSFACE_DIR, "batch_recognize.py")
TRANSFACE_VENV_PYTHON = os.path.join(TRANSFACE_DIR, "venv", "bin", "python")

# Batch size for progressive saving (to limit memory usage)
SAVE_BATCH_SIZE = 1000


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


def load_embedding_from_h5(h5_path: Path, face_name: str) -> np.ndarray:
    """
    Load a single embedding from an HDF5 file

    Args:
        h5_path: Path to the HDF5 file
        face_name: Name of the face (e.g., "frame_000241_face_0")

    Returns:
        Embedding as numpy array
    """
    with h5py.File(str(h5_path), "r") as f:
        return f[face_name][:]


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


def main():
    """Main function to extract embeddings from all face crops"""
    parser = argparse.ArgumentParser(
        description="Extract embeddings from all face crops using ArcFace and TransFace",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract embeddings using both models
  python extract_embeddings.py --models ArcFace TransFace

  # Extract only ArcFace embeddings
  python extract_embeddings.py --models ArcFace

  # Extract only TransFace embeddings
  python extract_embeddings.py --models TransFace

  # Use custom directories
  python extract_embeddings.py --input-dir cropped_faces --output-dir my_embeddings

  # Use custom TransFace weights
  python extract_embeddings.py --models TransFace --transface-weight /path/to/weights.pt

  # Use specific GPU for ArcFace
  python extract_embeddings.py --models ArcFace --gpu-id 1

  # Custom TransFace batch size
  python extract_embeddings.py --models TransFace --batch-size 2000

Features:
  - Progressive saving: saves embeddings incrementally (every 1000) to avoid memory issues
  - Resume support: if you stop the script, just run it again and it will skip already processed faces

Output Format:
  Embeddings are saved in HDF5 format (one .h5 file per video).

  To load an embedding:
    import h5py
    with h5py.File('video_name.h5', 'r') as f:
        embedding = f['frame_000241_face_0'][:]

  Or to list all embeddings in a file:
    with h5py.File('video_name.h5', 'r') as f:
        print(list(f.keys()))
        """,
    )

    parser.add_argument("--input-dir", type=str, default="cropped_faces", help="Directory containing cropped face images (default: cropped_faces)")
    parser.add_argument("--output-dir", type=str, default="cropped_faces_embeddings", help="Directory to save embeddings (default: cropped_faces_embeddings)")
    parser.add_argument("--models", nargs="+", choices=["ArcFace", "TransFace"], default=["ArcFace", "TransFace"], help="Which models to use for embedding extraction (default: both)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use for ArcFace (0 or 1, default: 0)")
    parser.add_argument(
        "--transface-weight", type=str, default="/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt", help="Path to TransFace model weights"
    )
    parser.add_argument("--transface-network", type=str, default="vit_l_dp005_mask_005", help="TransFace network architecture (default: vit_l_dp005_mask_005)")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for TransFace processing (default: 1000)")

    args = parser.parse_args()

    # Find all face crops
    print("=" * 80)
    print("FACE EMBEDDING EXTRACTION")
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

    print("\nTo load an embedding:")
    print("  import h5py")
    print("  with h5py.File('video_name.h5', 'r') as f:")
    print("      embedding = f['frame_000241_face_0'][:]")


if __name__ == "__main__":
    main()
