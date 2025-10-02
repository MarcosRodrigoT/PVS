"""
Face Detection Module using InsightFace (SCRFD)
Detects faces in video frames and saves detection metadata
"""

import cv2
import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np
from insightface.app import FaceAnalysis


class FaceDetector:
    """Face detector using InsightFace SCRFD for state-of-the-art performance"""

    def __init__(self, output_dir: str = "data/cropped_faces", det_size=(640, 640)):
        """
        Initialize the face detector

        Args:
            output_dir: Directory to save cropped face images
            det_size: Detection size for the model (width, height)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.detection_results = []

        # Initialize InsightFace detector
        print("Initializing face detector...")
        # Try to use GPU (CUDA), fall back to CPU if not available
        try:
            self.app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.app.prepare(ctx_id=0, det_size=det_size)
            print("Face detector initialized (GPU enabled)")
        except Exception as e:
            print(f"GPU initialization failed, using CPU: {e}")
            self.app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self.app.prepare(ctx_id=-1, det_size=det_size)
            print("Face detector initialized (CPU mode)")

    def detect_faces_in_video(self, video_path: str, clip_id: str) -> List[Dict]:
        """
        Detect faces in all frames of a video clip

        Args:
            video_path: Path to the video clip
            clip_id: Identifier for this video clip

        Returns:
            List of detection results containing face coordinates and metadata
        """
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detections = []

        # Create clip-specific directory for cropped faces
        clip_dir = self.output_dir / clip_id
        clip_dir.mkdir(parents=True, exist_ok=True)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect faces in the frame
            faces = self.app.get(frame)

            for face_idx, face in enumerate(faces):
                # Extract face coordinates (bbox is in format [x1, y1, x2, y2])
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox

                # Ensure coordinates are within frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)

                # Crop and save face
                face_crop = frame[y1:y2, x1:x2]
                face_filename = f"frame_{frame_count:06d}_face_{face_idx}.jpg"
                face_path = clip_dir / face_filename
                cv2.imwrite(str(face_path), face_crop)

                # Extract landmarks (5 points: left_eye, right_eye, nose, left_mouth, right_mouth)
                landmarks = {}
                if hasattr(face, "kps") and face.kps is not None:
                    kps = face.kps.astype(int)
                    landmarks = {
                        "left_eye": [int(kps[0][0]), int(kps[0][1])],
                        "right_eye": [int(kps[1][0]), int(kps[1][1])],
                        "nose": [int(kps[2][0]), int(kps[2][1])],
                        "left_mouth": [int(kps[3][0]), int(kps[3][1])],
                        "right_mouth": [int(kps[4][0]), int(kps[4][1])],
                    }

                # Store detection metadata
                detection_info = {
                    "clip_id": clip_id,
                    "frame_number": frame_count,
                    "face_id": f"face_{face_idx}",
                    "coordinates": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                    "confidence": float(face.det_score) if hasattr(face, "det_score") else 1.0,
                    "landmarks": landmarks,
                    "cropped_face_path": str(face_path),
                }
                detections.append(detection_info)

            frame_count += 1

        cap.release()
        self.detection_results.extend(detections)

        return detections

    def save_detection_metadata(self, output_path: str = "data/face_detections.json"):
        """
        Save all detection results to a JSON file

        Args:
            output_path: Path to save the detection metadata
        """
        with open(output_path, "w") as f:
            json.dump(self.detection_results, f, indent=2)

        print(f"Saved {len(self.detection_results)} face detections to {output_path}")

    def get_detection_stats(self) -> Dict:
        """
        Get statistics about the detections

        Returns:
            Dictionary containing detection statistics
        """
        if not self.detection_results:
            return {"total_detections": 0}

        clips = set(d["clip_id"] for d in self.detection_results)

        return {
            "total_detections": len(self.detection_results),
            "total_clips_processed": len(clips),
            "average_faces_per_clip": len(self.detection_results) / len(clips) if clips else 0,
        }


def process_video_clips(video_clips: List[str], output_dir: str = "data/cropped_faces") -> str:
    """
    Process multiple video clips and detect faces in each

    Args:
        video_clips: List of paths to video clips
        output_dir: Directory to save cropped faces

    Returns:
        Path to the saved detection metadata file
    """
    detector = FaceDetector(output_dir=output_dir)

    for i, video_path in enumerate(video_clips):
        clip_id = f"clip_{i:04d}"
        print(f"Processing {video_path} (clip {i+1}/{len(video_clips)})...")
        detections = detector.detect_faces_in_video(video_path, clip_id)
        print(f"  Detected {len(detections)} faces")

    # Save all detection metadata
    metadata_path = "data/face_detections.json"
    detector.save_detection_metadata(metadata_path)

    # Print statistics
    stats = detector.get_detection_stats()
    print(f"\nDetection Summary:")
    print(f"  Total detections: {stats['total_detections']}")
    print(f"  Clips processed: {stats['total_clips_processed']}")
    print(f"  Avg faces per clip: {stats['average_faces_per_clip']:.2f}")

    return metadata_path
