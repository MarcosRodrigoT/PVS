"""
Face Recognition Module using DeepFace with ArcFace model or TransFace transformer model
Compares detected faces against target faces
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from deepface import DeepFace
from collections import defaultdict

# Configure TensorFlow to use GPU if available (for DeepFace models)
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
try:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {len(gpus)} GPU(s) available for face recognition")
    else:
        print("No GPU detected, using CPU for face recognition")
except Exception as e:
    print(f"GPU configuration warning: {e}")

# TransFace configuration (called via subprocess)
TRANSFACE_DIR = "/home/mrt/Projects/TransFace"
TRANSFACE_SCRIPT = os.path.join(TRANSFACE_DIR, "recognize_face.py")
TRANSFACE_BATCH_SCRIPT = os.path.join(TRANSFACE_DIR, "batch_recognize.py")
TRANSFACE_VENV_PYTHON = os.path.join(TRANSFACE_DIR, "venv", "bin", "python")

# Check if TransFace is available
TRANSFACE_AVAILABLE = os.path.exists(TRANSFACE_BATCH_SCRIPT) and os.path.exists(TRANSFACE_VENV_PYTHON)


def _call_transface_extract_embeddings_batch(image_paths: List[str], transface_weight: str, transface_network: str, temp_dir: str = "/tmp") -> Dict[str, np.ndarray]:
    """
    Call TransFace via subprocess to extract embeddings from multiple images in batch.
    This is MUCH faster than calling per-image as it loads the model only once.

    Args:
        image_paths: List of paths to input images
        transface_weight: Path to TransFace model weights
        transface_network: TransFace network architecture
        temp_dir: Temporary directory for intermediate JSON files

    Returns:
        Dictionary mapping image_path -> embedding (numpy array)
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

        # The batch script returns: {"total_images": N, "successful": M, "failed": K, "embeddings": {...}}
        results = data.get("embeddings", {})

        # Convert to numpy arrays
        embeddings = {}
        for img_path, result_data in results.items():
            if result_data["status"] == "success" and result_data["embedding"] is not None:
                embeddings[img_path] = np.array(result_data["embedding"], dtype=np.float32)
            else:
                print(f"  Warning: Failed to extract embedding for {img_path}: {result_data.get('error_message', 'Unknown error')}")

        print(f"  Successfully loaded {len(embeddings)} embeddings from batch results")
        return embeddings

    finally:
        # Clean up temp files
        if os.path.exists(input_file):
            os.remove(input_file)
        if os.path.exists(output_file):
            os.remove(output_file)


def _call_transface_compute_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings (local computation, no subprocess needed).

    Args:
        emb1: First embedding
        emb2: Second embedding

    Returns:
        Normalized similarity score [0, 1]
    """
    # Cosine similarity
    similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

    # Normalize to [0, 1] range
    similarity = (similarity + 1) / 2

    return float(similarity)


class FaceRecognizer:
    """Face recognizer using DeepFace (ArcFace, Facenet, etc.) or TransFace transformer model"""

    def __init__(self, target_faces_dir: str = "target_faces", model_name: str = "ArcFace", transface_weight: str = None, transface_network: str = "vit_l_dp005_mask_005"):
        """
        Initialize the face recognizer

        Args:
            target_faces_dir: Directory containing target face images
            model_name: Face recognition model to use (ArcFace, Facenet, VGG-Face, DeepFace, TransFace)
            transface_weight: Path to TransFace model weights (only used if model_name="TransFace")
            transface_network: TransFace network architecture (only used if model_name="TransFace")
        """
        self.target_faces_dir = Path(target_faces_dir)
        self.model_name = model_name
        self.target_embeddings = {}
        self.recognition_results = []

        # TransFace configuration (for subprocess calls)
        self.transface_weight = transface_weight
        self.transface_network = transface_network

        # Validate TransFace if requested
        if self.model_name == "TransFace":
            if not TRANSFACE_AVAILABLE:
                raise ImportError(
                    "TransFace not available. Missing files:\n"
                    f"  Script: {TRANSFACE_SCRIPT} (exists: {os.path.exists(TRANSFACE_SCRIPT)})\n"
                    f"  Python: {TRANSFACE_VENV_PYTHON} (exists: {os.path.exists(TRANSFACE_VENV_PYTHON)})\n"
                    f"Please ensure TransFace is installed at {TRANSFACE_DIR}/ with venv activated."
                )

            # Default weight path if not provided
            if self.transface_weight is None:
                self.transface_weight = "/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt"

            # Verify weight file exists
            if not os.path.exists(self.transface_weight):
                raise FileNotFoundError(f"TransFace weights not found: {self.transface_weight}")

            print(f"Using TransFace model (via subprocess)...")
            print(f"  Network: {self.transface_network}")
            print(f"  Weights: {self.transface_weight}")
            print(f"  Python: {TRANSFACE_VENV_PYTHON}")

        # Load target face embeddings
        self._load_target_faces()

    def _load_target_faces(self):
        """Load and compute embeddings for all target faces"""
        if not self.target_faces_dir.exists():
            print(f"Warning: Target faces directory {self.target_faces_dir} does not exist")
            return

        target_images = list(self.target_faces_dir.glob("*.jpg")) + list(self.target_faces_dir.glob("*.png")) + list(self.target_faces_dir.glob("*.jpeg"))

        if not target_images:
            print(f"Warning: No target face images found in {self.target_faces_dir}")
            return

        print(f"Loading {len(target_images)} target face(s)...")

        if self.model_name == "TransFace":
            # Use TransFace batch processing (loads model once for all targets)
            image_paths = [str(img_path) for img_path in target_images]
            embeddings = _call_transface_extract_embeddings_batch(image_paths=image_paths, transface_weight=self.transface_weight, transface_network=self.transface_network)

            for img_path in target_images:
                if str(img_path) in embeddings:
                    self.target_embeddings[img_path.stem] = embeddings[str(img_path)]
                    print(f"  Loaded target: {img_path.name}")
                else:
                    print(f"  Error loading {img_path.name}")
        else:
            # Use DeepFace (ArcFace, Facenet, etc.) - process individually
            for img_path in target_images:
                try:
                    embedding = DeepFace.represent(img_path=str(img_path), model_name=self.model_name, enforce_detection=False)
                    # DeepFace.represent returns a list of dictionaries
                    if isinstance(embedding, list) and len(embedding) > 0:
                        self.target_embeddings[img_path.stem] = embedding[0]["embedding"]
                        print(f"  Loaded target: {img_path.name}")
                except Exception as e:
                    print(f"  Error loading {img_path.name}: {e}")

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """
        Compute cosine similarity between two embeddings

        Args:
            embedding1: First face embedding
            embedding2: Second face embedding

        Returns:
            Similarity score (0-1, higher is more similar)

        Note:
            Both DeepFace and TransFace implementations return similarity in [0,1] range:
            - Raw cosine similarity is in [-1, 1] range
            - Normalized to [0, 1] using: (similarity + 1) / 2
            - This ensures consistency across all models
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Cosine similarity (raw range: [-1, 1])
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Convert to [0, 1] range for consistency
        # Note: TransFace wrapper already returns normalized similarity,
        # but this ensures consistency if called directly
        similarity = (similarity + 1) / 2

        return float(similarity)

    def recognize_faces(self, detection_metadata_path: str) -> Dict:
        """
        Compare all detected faces against target faces

        Args:
            detection_metadata_path: Path to face detection metadata JSON

        Returns:
            Dictionary containing recognition results per clip
        """
        # Load detection metadata
        with open(detection_metadata_path, "r") as f:
            detections = json.load(f)

        if not self.target_embeddings:
            print("Warning: No target faces loaded. Cannot perform recognition.")
            return {}

        print(f"Recognizing {len(detections)} detected faces against {len(self.target_embeddings)} target(s)...")

        # Group results by clip
        clip_results = defaultdict(lambda: defaultdict(list))

        if self.model_name == "TransFace":
            # BATCH PROCESSING: Extract all embeddings at once (loads model only once!)
            print("  Extracting embeddings for all detected faces (batch mode)...")

            # Collect all valid image paths
            valid_detections = []
            image_paths = []
            for detection in detections:
                cropped_face_path = detection["cropped_face_path"]
                if os.path.exists(cropped_face_path):
                    valid_detections.append(detection)
                    image_paths.append(cropped_face_path)

            print(f"  Processing {len(image_paths)} valid face images...")

            # Extract all embeddings in one batch call
            detected_embeddings = _call_transface_extract_embeddings_batch(image_paths=image_paths, transface_weight=self.transface_weight, transface_network=self.transface_network)

            print(f"  Successfully extracted {len(detected_embeddings)} embeddings")

            # Now compute similarities (local computation, no subprocess)
            print("  Computing similarities against target faces...")
            for i, detection in enumerate(valid_detections):
                if i % 1000 == 0 and i > 0:
                    print(f"    Processed {i}/{len(valid_detections)} faces...")

                cropped_face_path = detection["cropped_face_path"]
                clip_id = detection["clip_id"]
                frame_number = detection["frame_number"]

                if cropped_face_path not in detected_embeddings:
                    continue

                detected_emb = detected_embeddings[cropped_face_path]

                # Compare against all target faces
                for target_name, target_emb in self.target_embeddings.items():
                    similarity = _call_transface_compute_similarity(detected_emb, target_emb)

                    result = {
                        "frame_number": frame_number,
                        "face_id": detection["face_id"],
                        "similarity": similarity,
                        "coordinates": detection["coordinates"],
                        "cropped_face_path": cropped_face_path,
                    }

                    clip_results[clip_id][target_name].append(result)

        else:
            # DeepFace models: process individually (no batch API available)
            for i, detection in enumerate(detections):
                if i % 100 == 0:
                    print(f"  Processing detection {i+1}/{len(detections)}...")

                cropped_face_path = detection["cropped_face_path"]
                clip_id = detection["clip_id"]
                frame_number = detection["frame_number"]

                if not os.path.exists(cropped_face_path):
                    continue

                try:
                    detected_embedding = DeepFace.represent(img_path=cropped_face_path, model_name=self.model_name, enforce_detection=False)

                    if isinstance(detected_embedding, list) and len(detected_embedding) > 0:
                        detected_emb = detected_embedding[0]["embedding"]
                    else:
                        continue

                    # Compare against all target faces
                    for target_name, target_emb in self.target_embeddings.items():
                        similarity = self.compute_similarity(detected_emb, target_emb)

                        result = {
                            "frame_number": frame_number,
                            "face_id": detection["face_id"],
                            "similarity": similarity,
                            "coordinates": detection["coordinates"],
                            "cropped_face_path": cropped_face_path,
                        }

                        clip_results[clip_id][target_name].append(result)

                except Exception as e:
                    print(f"  Error processing {cropped_face_path}: {e}")

        self.recognition_results = dict(clip_results)
        return self.recognition_results

    def save_recognition_results(self, output_path: str = "results/face_recognition_results.json"):
        """
        Save recognition results to a JSON file

        Args:
            output_path: Path to save the recognition results
        """
        with open(output_path, "w") as f:
            json.dump(self.recognition_results, f, indent=2)

        print(f"Saved recognition results to {output_path}")

    def get_recognition_stats(self, threshold: float = 0.6) -> Dict:
        """
        Get statistics about recognition results

        Args:
            threshold: Similarity threshold to consider a match

        Returns:
            Dictionary containing recognition statistics
        """
        stats = {}

        for clip_id, targets in self.recognition_results.items():
            for target_name, results in targets.items():
                matches = [r for r in results if r["similarity"] >= threshold]

                if target_name not in stats:
                    stats[target_name] = {"total_detections": 0, "total_matches": 0, "clips_with_matches": 0, "max_similarity_per_clip": {}}

                stats[target_name]["total_detections"] += len(results)
                stats[target_name]["total_matches"] += len(matches)

                if matches:
                    stats[target_name]["clips_with_matches"] += 1
                    max_sim = max(r["similarity"] for r in results)
                    stats[target_name]["max_similarity_per_clip"][clip_id] = max_sim

        return stats


def recognize_faces(
    detection_metadata_path: str, target_faces_dir: str = "target_faces", model_name: str = "ArcFace", transface_weight: str = None, transface_network: str = "vit_l_dp005_mask_005"
) -> str:
    """
    Perform face recognition on detected faces

    Args:
        detection_metadata_path: Path to face detection metadata JSON
        target_faces_dir: Directory containing target face images
        model_name: Face recognition model to use (ArcFace, Facenet, VGG-Face, DeepFace, TransFace)
        transface_weight: Path to TransFace model weights (only used if model_name="TransFace")
        transface_network: TransFace network architecture (only used if model_name="TransFace")

    Returns:
        Path to the saved recognition results file
    """
    recognizer = FaceRecognizer(target_faces_dir=target_faces_dir, model_name=model_name, transface_weight=transface_weight, transface_network=transface_network)
    recognizer.recognize_faces(detection_metadata_path)

    # Save results
    results_path = "results/face_recognition_results.json"
    recognizer.save_recognition_results(results_path)

    # Print statistics
    stats = recognizer.get_recognition_stats()
    print(f"\nRecognition Summary:")
    for target_name, target_stats in stats.items():
        print(f"  Target: {target_name}")
        print(f"    Total detections compared: {target_stats['total_detections']}")
        print(f"    Matches (>0.6 similarity): {target_stats['total_matches']}")
        print(f"    Clips with matches: {target_stats['clips_with_matches']}")

    return results_path
