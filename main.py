"""
Main Pipeline for Personalized Video Summarization (PVS)

This script orchestrates the complete workflow:
1. Video summarization (VS) - template call to external project
2. Face detection on video summary clips
3. Face cropping and metadata storage
4. Face recognition against target faces
5. Visualization of resemblance curves
"""

import os
import sys
from pathlib import Path
from typing import List
import argparse

# Import pipeline components
from face_detection.detector import process_video_clips
from face_recognition.recognizer import recognize_faces
from visualizer import visualize_results


class PVSPipeline:
    """Main pipeline orchestrator for Personalized Video Summarization"""

    def __init__(self, config: dict = None):
        """
        Initialize the PVS pipeline

        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config or {}

        # Default paths
        self.video_summaries_dir = self.config.get("video_summaries_dir", "data/video_summaries")
        self.target_faces_dir = self.config.get("target_faces_dir", "target_faces")
        self.cropped_faces_dir = self.config.get("cropped_faces_dir", "data/cropped_faces")
        self.output_dir = self.config.get("output_dir", "data")

        # Model configurations
        self.face_recognition_model = self.config.get("face_recognition_model", "ArcFace")

        # Ensure directories exist
        Path(self.video_summaries_dir).mkdir(parents=True, exist_ok=True)
        Path(self.target_faces_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cropped_faces_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def step1_video_summarization(self, input_video: str) -> List[str]:
        """
        Step 1: Video Summarization (VS)

        This is a template call to the external video summarization project.
        Replace this with actual integration when VS project is complete.

        Args:
            input_video: Path to the input video file

        Returns:
            List of paths to video summary clips
        """
        print("\n" + "=" * 80)
        print("STEP 1: VIDEO SUMMARIZATION")
        print("=" * 80)

        # TODO: Replace with actual VS project call
        # Example: video_clips = vs_project.summarize(input_video, output_dir=self.video_summaries_dir)

        print(f"[TEMPLATE] Calling VS project for video summarization...")
        print(f"[TEMPLATE] Input video: {input_video}")
        print(f"[TEMPLATE] Output directory: {self.video_summaries_dir}")
        print(f"[TEMPLATE] This will be replaced with actual VS project integration")

        # For now, look for existing video clips in the summaries directory
        video_clips = list(Path(self.video_summaries_dir).glob("*.mp4")) + list(Path(self.video_summaries_dir).glob("*.avi")) + list(Path(self.video_summaries_dir).glob("*.mov"))

        # Sort video clips by filename to ensure consistent ordering
        video_clips = sorted([str(clip) for clip in video_clips])

        if video_clips:
            print(f"\nFound {len(video_clips)} existing video clips in {self.video_summaries_dir}")
        else:
            print(f"\nNo video clips found in {self.video_summaries_dir}")
            print("Please run the VS project separately and place clips in this directory, or")
            print("integrate the VS project API here.")

        return video_clips

    def step2_face_detection(self, video_clips: List[str]) -> str:
        """
        Step 2: Face Detection

        Detect faces in all frames of video summary clips using InsightFace SCRFD

        Args:
            video_clips: List of paths to video summary clips

        Returns:
            Path to face detection metadata JSON
        """
        print("\n" + "=" * 80)
        print("STEP 2: FACE DETECTION")
        print("=" * 80)

        if not video_clips:
            print("No video clips to process. Skipping face detection.")
            return None

        detection_metadata = process_video_clips(video_clips=video_clips, output_dir=self.cropped_faces_dir)

        return detection_metadata

    def step3_face_recognition(self, detection_metadata: str) -> str:
        """
        Step 3: Face Recognition

        Compare detected faces against target faces using ArcFace

        Args:
            detection_metadata: Path to face detection metadata JSON

        Returns:
            Path to face recognition results JSON
        """
        print("\n" + "=" * 80)
        print("STEP 3: FACE RECOGNITION")
        print("=" * 80)

        if not detection_metadata or not os.path.exists(detection_metadata):
            print("No detection metadata found. Skipping face recognition.")
            return None

        recognition_results = recognize_faces(detection_metadata_path=detection_metadata, target_faces_dir=self.target_faces_dir, model_name=self.face_recognition_model)

        return recognition_results

    def step4_visualization(self, recognition_results: str, show_plots: bool = False):
        """
        Step 4: Visualization

        Generate resemblance curves for each video clip

        Args:
            recognition_results: Path to face recognition results JSON
            show_plots: Whether to display plots interactively
        """
        print("\n" + "=" * 80)
        print("STEP 4: VISUALIZATION")
        print("=" * 80)

        if not recognition_results or not os.path.exists(recognition_results):
            print("No recognition results found. Skipping visualization.")
            return

        visualize_results(recognition_results_path=recognition_results, output_dir=self.output_dir, show_plots=show_plots)

    def run(self, input_video: str = None, show_plots: bool = False):
        """
        Run the complete PVS pipeline

        Args:
            input_video: Path to input video (for VS step)
            show_plots: Whether to display plots interactively
        """
        print("\n" + "#" * 80)
        print("# PERSONALIZED VIDEO SUMMARIZATION PIPELINE")
        print("#" * 80)

        # Step 1: Video Summarization
        video_clips = self.step1_video_summarization(input_video)

        # Step 2: Face Detection
        detection_metadata = self.step2_face_detection(video_clips)

        # Step 3: Face Recognition
        recognition_results = self.step3_face_recognition(detection_metadata)

        # Step 4: Visualization
        self.step4_visualization(recognition_results, show_plots=show_plots)

        print("\n" + "#" * 80)
        print("# PIPELINE COMPLETE")
        print("#" * 80)
        print(f"\nResults saved to: {self.output_dir}/")
        print(f"  - Face detections: {self.output_dir}/face_detections.json")
        print(f"  - Recognition results: {self.output_dir}/face_recognition_results.json")
        print(f"  - Visualizations: {self.output_dir}/visualizations/")
        print(f"  - Summary report: {self.output_dir}/summary_report.txt")


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Personalized Video Summarization Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline on existing video clips in data/video_summaries/
  python main.py

  # Specify input video for summarization
  python main.py --input my_video.mp4

  # Show plots interactively
  python main.py --show-plots

  # Use custom directories
  python main.py --video-summaries custom/summaries --target-faces custom/targets
        """,
    )

    parser.add_argument("--input", type=str, default=None, help="Input video file for summarization")
    parser.add_argument("--video-summaries", type=str, default="data/video_summaries", help="Directory containing video summary clips")
    parser.add_argument("--target-faces", type=str, default="target_faces", help="Directory containing target face images")
    parser.add_argument("--cropped-faces", type=str, default="data/cropped_faces", help="Directory to save cropped face images")
    parser.add_argument("--output", type=str, default="data", help="Output directory for results")
    parser.add_argument("--model", type=str, default="ArcFace", choices=["ArcFace", "Facenet", "VGG-Face", "DeepFace"], help="Face recognition model to use")
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")

    args = parser.parse_args()

    # Create configuration from arguments
    config = {
        "video_summaries_dir": args.video_summaries,
        "target_faces_dir": args.target_faces,
        "cropped_faces_dir": args.cropped_faces,
        "output_dir": args.output,
        "face_recognition_model": args.model,
    }

    # Initialize and run pipeline
    pipeline = PVSPipeline(config=config)
    pipeline.run(input_video=args.input, show_plots=args.show_plots)


if __name__ == "__main__":
    main()
