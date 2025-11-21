"""
Extract all faces from all videos in the data directory
Processes every frame of every video and saves:
- Cropped face images
- Detection metadata (coordinates, confidence, facial landmarks)
"""

import cv2
import os
from pathlib import Path
from typing import List
from insightface.app import FaceAnalysis
from tqdm import tqdm


class FullVideoFaceExtractor:
    """Extract faces from all frames of videos"""

    def __init__(self, output_dir: str = "cropped_faces", det_size=(640, 640), min_face_size: int = 0):
        """
        Initialize the face extractor

        Args:
            output_dir: Directory to save cropped face images
            det_size: Detection size for the model (width, height)
            min_face_size: Minimum face width/height in pixels (faces smaller than this are filtered out)
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.min_face_size = min_face_size

        # Initialize InsightFace detector (same as in detector.py)
        print("Initializing face detector...")
        try:
            self.app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.app.prepare(ctx_id=0, det_size=det_size)
            print("Face detector initialized (GPU enabled)")
        except Exception as e:
            print(f"GPU initialization failed, using CPU: {e}")
            self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
            self.app.prepare(ctx_id=-1, det_size=det_size)
            print("Face detector initialized (CPU mode)")

    def extract_faces_from_video(self, video_path: str) -> int:
        """
        Extract faces from all frames of a video

        Args:
            video_path: Path to the video file

        Returns:
            Number of faces extracted
        """
        import json

        video_path = Path(video_path)
        video_name = video_path.stem  # Filename without extension

        # Create video-specific directory
        video_dir = self.output_dir / video_name
        video_dir.mkdir(parents=True, exist_ok=True)

        # Path for video-specific metadata file
        metadata_path = video_dir / "detections.json"

        # Load existing detections if resuming
        video_detections = []
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                video_detections = json.load(f)

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return 0

        # Get total frame count for progress bar
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Check for resume: find the last processed frame
        existing_files = list(video_dir.glob("frame_*_face_*.jpg"))
        start_frame = 0
        if existing_files:
            # Extract frame numbers from existing files
            frame_numbers = set()
            for f in existing_files:
                try:
                    # Parse frame number from filename like "frame_000001_face_0.jpg"
                    frame_num = int(f.stem.split("_")[1])
                    frame_numbers.add(frame_num)
                except (IndexError, ValueError):
                    continue

            if frame_numbers:
                # Find the last consecutive frame (to handle partial frames)
                max_frame = max(frame_numbers)
                # Start from the next frame after the last one
                start_frame = max_frame + 1

        print(f"\nProcessing: {video_name}")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Output directory: {video_dir}")

        # Check if already fully processed
        if start_frame >= total_frames:
            existing_face_count = len(existing_files)
            print(f"  ⏭ Already fully processed ({existing_face_count} faces extracted previously)")
            return existing_face_count

        # Show resume info if applicable
        if start_frame > 0:
            print(f"  ⏩ Resuming from frame {start_frame} (skipping {start_frame} already processed frames)")
            # Seek to the start frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_number = start_frame
        total_faces_extracted = 0

        # Progress bar for frames
        remaining_frames = total_frames - start_frame
        with tqdm(total=remaining_frames, desc=f"  Extracting faces", unit="frame", initial=0) as pbar:
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

                    # Calculate face size
                    face_width = x2 - x1
                    face_height = y2 - y1

                    # Filter out faces that are too small
                    if self.min_face_size > 0 and (face_width < self.min_face_size or face_height < self.min_face_size):
                        continue

                    # Crop and save face
                    face_crop = frame[y1:y2, x1:x2]

                    # Use naming pattern: frame_000001_face_0.jpg
                    face_filename = f"frame_{frame_number:06d}_face_{face_idx}.jpg"
                    face_path = video_dir / face_filename

                    # Save the cropped face
                    cv2.imwrite(str(face_path), face_crop)
                    total_faces_extracted += 1

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
                        "video_name": video_name,
                        "frame_number": frame_number,
                        "face_id": face_idx,
                        "coordinates": {"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)},
                        "confidence": float(face.det_score) if hasattr(face, "det_score") else 1.0,
                        "landmarks": landmarks,
                        "cropped_face_path": str(face_path),
                    }
                    video_detections.append(detection_info)

                frame_number += 1
                pbar.update(1)

        cap.release()

        # Save metadata for this video immediately
        with open(metadata_path, "w") as f:
            json.dump(video_detections, f, indent=2)
        print(f"  ✓ Saved detection metadata to {metadata_path}")

        # Include previously extracted faces in the count
        if start_frame > 0:
            previous_faces = len(existing_files)
            print(f"  ✓ Extracted {total_faces_extracted} new faces from {frame_number - start_frame} frames")
            print(f"  ✓ Total: {previous_faces + total_faces_extracted} faces from {frame_number} frames")
            return previous_faces + total_faces_extracted
        else:
            print(f"  ✓ Extracted {total_faces_extracted} faces from {frame_number} frames")
            return total_faces_extracted

    def get_all_detection_stats(self):
        """
        Get statistics about all detections across all processed videos

        Returns:
            Dictionary containing detection statistics
        """
        import json

        all_detections = []
        video_dirs = [d for d in self.output_dir.iterdir() if d.is_dir()]

        for video_dir in video_dirs:
            metadata_path = video_dir / "detections.json"
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    all_detections.extend(json.load(f))

        if not all_detections:
            return {"total_detections": 0, "total_videos_processed": 0}

        videos = set(d["video_name"] for d in all_detections)

        return {
            "total_detections": len(all_detections),
            "total_videos_processed": len(videos),
            "average_faces_per_video": len(all_detections) / len(videos) if videos else 0,
        }


def find_video_files(directory: str) -> List[Path]:
    """
    Find all video files in a directory

    Args:
        directory: Directory to search for videos

    Returns:
        List of video file paths
    """
    video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV"]
    directory = Path(directory)

    video_files = []
    for ext in video_extensions:
        video_files.extend(directory.glob(f"*{ext}"))

    return sorted(video_files)


def main():
    """Main function to process all videos in the data directory"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract all faces from all videos in the data directory",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract all faces from all videos (no size filter)
  python extract_all_faces.py

  # Filter out faces smaller than 50x50 pixels
  python extract_all_faces.py --min-face-size 50

  # Use custom data directory and output directory
  python extract_all_faces.py --data-dir /path/to/videos --output-dir /path/to/output

  # Custom detection size (may affect detection accuracy)
  python extract_all_faces.py --det-size 1024 1024
        """,
    )

    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing video files (default: data)")
    parser.add_argument("--output-dir", type=str, default="cropped_faces", help="Directory to save cropped faces (default: cropped_faces)")
    parser.add_argument("--min-face-size", type=int, default=0, help="Minimum face width/height in pixels (default: 0 = no filter)")
    parser.add_argument("--det-size", type=int, nargs=2, default=[640, 640], metavar=("WIDTH", "HEIGHT"), help="Detection size for the model (default: 640 640)")

    args = parser.parse_args()

    # Find all video files
    print("=" * 80)
    print("FULL VIDEO FACE EXTRACTION")
    print("=" * 80)
    print(f"\nSearching for videos in: {args.data_dir}")

    video_files = find_video_files(args.data_dir)

    if not video_files:
        print(f"\n⚠ No video files found in {args.data_dir}")
        print("Supported formats: .mp4, .avi, .mov, .mkv")
        return

    print(f"✓ Found {len(video_files)} video(s):")
    for video in video_files:
        print(f"  - {video.name}")

    # Initialize face extractor
    print(f"\nOutput directory: {args.output_dir}")
    if args.min_face_size > 0:
        print(f"Minimum face size filter: {args.min_face_size}x{args.min_face_size} pixels")
    else:
        print("No face size filter (all detected faces will be saved)")

    extractor = FullVideoFaceExtractor(output_dir=args.output_dir, det_size=tuple(args.det_size), min_face_size=args.min_face_size)

    # Process each video
    total_faces = 0
    for video_path in video_files:
        faces_extracted = extractor.extract_faces_from_video(video_path)
        total_faces += faces_extracted

    # Get and print statistics
    stats = extractor.get_all_detection_stats()

    # Summary
    print("\n" + "=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print(f"Total videos processed: {len(video_files)}")
    print(f"Total faces extracted: {total_faces}")
    print(f"Average faces per video: {total_faces / len(video_files):.1f}")
    print(f"\nAll cropped faces saved to: {args.output_dir}/")
    print(f"Detection metadata saved per video in: {args.output_dir}/<video_name>/detections.json")


if __name__ == "__main__":
    main()
