"""
Face Recognition Module using DeepFace with ArcFace model
Compares detected faces against target faces
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from deepface import DeepFace
from collections import defaultdict

# Configure TensorFlow to use GPU if available
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
try:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU detected: {len(gpus)} GPU(s) available for face recognition")
    else:
        print("No GPU detected, using CPU for face recognition")
except Exception as e:
    print(f"GPU configuration warning: {e}")


class FaceRecognizer:
    """Face recognizer using DeepFace with ArcFace for state-of-the-art performance"""

    def __init__(self, target_faces_dir: str = "target_faces", model_name: str = "ArcFace"):
        """
        Initialize the face recognizer

        Args:
            target_faces_dir: Directory containing target face images
            model_name: Face recognition model to use (ArcFace, Facenet, VGG-Face, etc.)
        """
        self.target_faces_dir = Path(target_faces_dir)
        self.model_name = model_name
        self.target_embeddings = {}
        self.recognition_results = []

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

        for img_path in target_images:
            try:
                # Generate embedding for target face
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
        """
        emb1 = np.array(embedding1)
        emb2 = np.array(embedding2)

        # Cosine similarity
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

        # Convert to 0-1 range
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

        for i, detection in enumerate(detections):
            if i % 100 == 0:
                print(f"  Processing detection {i+1}/{len(detections)}...")

            cropped_face_path = detection["cropped_face_path"]
            clip_id = detection["clip_id"]
            frame_number = detection["frame_number"]

            if not os.path.exists(cropped_face_path):
                continue

            try:
                # Generate embedding for detected face
                detected_embedding = DeepFace.represent(img_path=cropped_face_path, model_name=self.model_name, enforce_detection=False)

                if isinstance(detected_embedding, list) and len(detected_embedding) > 0:
                    detected_emb = detected_embedding[0]["embedding"]

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

    def save_recognition_results(self, output_path: str = "data/face_recognition_results.json"):
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


def recognize_faces(detection_metadata_path: str, target_faces_dir: str = "target_faces", model_name: str = "ArcFace") -> str:
    """
    Perform face recognition on detected faces

    Args:
        detection_metadata_path: Path to face detection metadata JSON
        target_faces_dir: Directory containing target face images
        model_name: Face recognition model to use

    Returns:
        Path to the saved recognition results file
    """
    recognizer = FaceRecognizer(target_faces_dir=target_faces_dir, model_name=model_name)
    recognizer.recognize_faces(detection_metadata_path)

    # Save results
    results_path = "data/face_recognition_results.json"
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
