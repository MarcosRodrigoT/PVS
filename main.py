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
from video_summarization import generate_highlights_sportclip, generate_highlights_qddetr
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

        # Video summarization method selection
        self.video_summarizer = self.config.get("video_summarizer", "sportclip")  # "sportclip" or "qddetr"

        # SportCLIP configuration
        self.sportclip_dir = self.config.get("sportclip_dir", "/home/mrt/Projects/SportCLIP-official")
        self.sportclip_dataset_dir = self.config.get("sportclip_dataset_dir", "/mnt/Data/mrt/SportCLIP-OlympicHighlights")
        self.sportclip_sentences_file = self.config.get("sportclip_sentences_file", None)  # e.g., "data/sentences/long_jump.json"

        # QD-DETR configuration
        self.qddetr_ckpt_path = self.config.get("qddetr_ckpt_path", "/home/mrt/Projects/QD-DETR/results/OlympicHighlights-exp-2025_09_25_11_22_34/model_best.ckpt")
        self.qddetr_query = self.config.get("qddetr_query", None)  # e.g., "highlight moments in this sports video"
        self.qddetr_device = self.config.get("qddetr_device", "cuda:0")
        self.qddetr_context_window = self.config.get("qddetr_context_window", 600)
        self.qddetr_min_duration = self.config.get("qddetr_min_duration", 15)
        self.qddetr_min_area = self.config.get("qddetr_min_area", 15.0)
        self.qddetr_feat_dirs = self.config.get("qddetr_feat_dirs", None)  # For pre-extracted features
        self.qddetr_text_feat_dir = self.config.get("qddetr_text_feat_dir", None)  # For pre-extracted text features
        self.qddetr_annotation_path = self.config.get("qddetr_annotation_path", None)  # Annotation JSONL file path

        # Model configurations
        self.face_recognition_model = self.config.get("face_recognition_model", "ArcFace")

        # Ensure directories exist
        Path(self.video_summaries_dir).mkdir(parents=True, exist_ok=True)
        Path(self.target_faces_dir).mkdir(parents=True, exist_ok=True)
        Path(self.cropped_faces_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def step1_video_summarization(self, input_video: str = None, video_name: str = None, video_path: str = None) -> List[str]:
        """
        Step 1: Video Summarization

        Generates video highlights using the selected video summarization framework (SportCLIP or QD-DETR).
        If input_video is provided, runs the selected summarization pipeline.
        Otherwise, looks for existing video clips in the summaries directory.

        Args:
            input_video: Video name (without extension) for processing
            video_name: Video name for processing (without extension)
            video_path: Full path to the input video file (required for QD-DETR)

        Returns:
            List of paths to video summary clips
        """
        print("\n" + "=" * 80)
        print(f"STEP 1: VIDEO SUMMARIZATION ({self.video_summarizer.upper()})")
        print("=" * 80)

        if input_video:
            # Validate input parameters
            if not video_name:
                raise ValueError("video_name must be provided when input_video is specified")

            # Select and run video summarizer
            if self.video_summarizer == "sportclip":
                video_clips = self._run_sportclip_summarizer(video_name)
            elif self.video_summarizer == "qddetr":
                video_clips = self._run_qddetr_summarizer(video_name, video_path)
            else:
                raise ValueError(f"Unknown video summarizer: {self.video_summarizer}. Choose 'sportclip' or 'qddetr'")

            return video_clips

        else:
            # Look for existing video clips in the summaries directory
            print("\nNo input video specified. Looking for existing video clips...")
            video_clips = list(Path(self.video_summaries_dir).glob("*.mp4")) + list(Path(self.video_summaries_dir).glob("*.avi")) + list(Path(self.video_summaries_dir).glob("*.mov"))

            # Sort video clips by filename to ensure consistent ordering
            video_clips = sorted([str(clip) for clip in video_clips])

            if video_clips:
                print(f"✓ Found {len(video_clips)} existing video clips in {self.video_summaries_dir}")
            else:
                print(f"⚠ No video clips found in {self.video_summaries_dir}")
                print("\nTo generate highlights, run with:")
                if self.video_summarizer == "sportclip":
                    print("  --input <video_name> --video-name <video_name> --sentences-file <path>")
                elif self.video_summarizer == "qddetr":
                    print("  --input <video_name> --video-name <video_name> --video-path <path> --qddetr-query <query>")

            return video_clips

    def _run_sportclip_summarizer(self, video_name: str) -> List[str]:
        """Run SportCLIP video summarizer."""
        if not self.sportclip_sentences_file:
            raise ValueError("sportclip_sentences_file must be provided in config. Example: 'data/sentences/long_jump.json'")

        # Generate highlights using SportCLIP
        video_clips = generate_highlights_sportclip(
            video_name=video_name,
            sentences_file=self.sportclip_sentences_file,
            output_dir=self.video_summaries_dir,
            sportclip_dir=self.sportclip_dir,
            sportclip_dataset_dir=self.sportclip_dataset_dir,
        )

        return video_clips

    def _run_qddetr_summarizer(self, video_name: str, video_path: str) -> List[str]:
        """Run QD-DETR video summarizer."""
        if not video_path:
            raise ValueError("video_path must be provided when using QD-DETR summarizer")

        if not self.qddetr_query:
            raise ValueError("qddetr_query must be provided in config. Example: 'highlight moments in this sports video'")

        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        # Generate highlights using QD-DETR
        video_clips = generate_highlights_qddetr(
            video_name=video_name,
            query=self.qddetr_query,
            output_dir=self.video_summaries_dir,
            video_path=video_path,
            qddetr_ckpt_path=self.qddetr_ckpt_path,
            context_window=self.qddetr_context_window,
            min_duration_frames=self.qddetr_min_duration,
            min_area=self.qddetr_min_area,
            device=self.qddetr_device,
            feat_dirs=self.qddetr_feat_dirs,
            text_feat_dir=self.qddetr_text_feat_dir,
            annotation_path=self.qddetr_annotation_path,
        )

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

    def run(self, input_video: str = None, video_name: str = None, video_path: str = None, show_plots: bool = False):
        """
        Run the complete PVS pipeline

        Args:
            input_video: Video name for processing (without extension)
            video_name: Video name for processing (without extension)
            video_path: Full path to input video file (required for QD-DETR)
            show_plots: Whether to display plots interactively
        """
        print("\n" + "#" * 80)
        print("# PERSONALIZED VIDEO SUMMARIZATION PIPELINE")
        print("#" * 80)

        # Step 1: Video Summarization
        video_clips = self.step1_video_summarization(input_video, video_name, video_path)

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

  # Generate highlights using SportCLIP and run full pipeline
  python main.py --input longjump_video1 --video-name longjump_video1 \\
      --sentences-file data/sentences/long_jump.json

  # Generate highlights using QD-DETR and run full pipeline
  python main.py --video-summarizer qddetr --input longjump_video1 \\
      --video-name longjump_video1 --video-path /path/to/longjump_video1.mp4 \\
      --qddetr-query "highlight moments in this long jump competition"

  # Show plots interactively
  python main.py --show-plots

  # Use custom directories
  python main.py --video-summaries custom/summaries --target-faces custom/targets
        """,
    )

    # Video input parameters
    parser.add_argument("--input", type=str, default=None, help="Video name for processing (without extension)")
    parser.add_argument("--video-name", type=str, default=None, help="Video name for processing (without extension)")
    parser.add_argument("--video-path", type=str, default=None, help="Full path to input video file (required for QD-DETR)")

    # Video summarizer selection
    parser.add_argument("--video-summarizer", type=str, default="sportclip", choices=["sportclip", "qddetr"], help="Video summarization method (default: sportclip)")

    # SportCLIP parameters
    parser.add_argument("--sentences-file", type=str, default=None, help="Path to SportCLIP sentences file (e.g., data/sentences/long_jump.json)")
    parser.add_argument("--sportclip-dir", type=str, default="/home/mrt/Projects/SportCLIP-official", help="SportCLIP project directory")
    parser.add_argument("--sportclip-dataset-dir", type=str, default="/mnt/Data/mrt/SportCLIP-OlympicHighlights", help="SportCLIP dataset directory")

    # QD-DETR parameters
    parser.add_argument(
        "--qddetr-ckpt", type=str, default="/home/mrt/Projects/QD-DETR/results/OlympicHighlights-exp-2025_09_25_11_22_34/model_best.ckpt", help="Path to QD-DETR checkpoint"
    )
    parser.add_argument("--qddetr-query", type=str, default=None, help="Query for QD-DETR (e.g., 'highlight moments in this sports video')")
    parser.add_argument("--qddetr-device", type=str, default="cuda:0", help="Device for QD-DETR inference")
    parser.add_argument("--qddetr-context-window", type=int, default=600, help="Context window size in frames for QD-DETR postprocessing")
    parser.add_argument("--qddetr-min-duration", type=int, default=15, help="Minimum event duration in frames for QD-DETR")
    parser.add_argument("--qddetr-min-area", type=float, default=15.0, help="Minimum event area threshold for QD-DETR")
    parser.add_argument("--qddetr-feat-dirs", nargs="+", default=None, help="Video feature directories for QD-DETR (e.g., /path/to/clip_features /path/to/slowfast_features)")
    parser.add_argument("--qddetr-text-feat-dir", type=str, default=None, help="Text feature directory for QD-DETR (e.g., /path/to/clip_text_features_specific)")
    parser.add_argument("--qddetr-annotation-path", type=str, default=None, help="Path to QD-DETR annotation JSONL file (required when using pre-extracted features)")

    # PVS directories
    parser.add_argument("--video-summaries", type=str, default="data/video_summaries", help="Directory containing video summary clips")
    parser.add_argument("--target-faces", type=str, default="target_faces", help="Directory containing target face images")
    parser.add_argument("--cropped-faces", type=str, default="data/cropped_faces", help="Directory to save cropped face images")
    parser.add_argument("--output", type=str, default="data", help="Output directory for results")

    # Model parameters
    parser.add_argument("--model", type=str, default="ArcFace", choices=["ArcFace", "Facenet", "VGG-Face", "DeepFace"], help="Face recognition model to use")

    # Display parameters
    parser.add_argument("--show-plots", action="store_true", help="Display plots interactively")

    args = parser.parse_args()

    # Create configuration from arguments
    config = {
        "video_summaries_dir": args.video_summaries,
        "target_faces_dir": args.target_faces,
        "cropped_faces_dir": args.cropped_faces,
        "output_dir": args.output,
        "face_recognition_model": args.model,
        "video_summarizer": args.video_summarizer,
        # SportCLIP config
        "sportclip_dir": args.sportclip_dir,
        "sportclip_dataset_dir": args.sportclip_dataset_dir,
        "sportclip_sentences_file": args.sentences_file,
        # QD-DETR config
        "qddetr_ckpt_path": args.qddetr_ckpt,
        "qddetr_query": args.qddetr_query,
        "qddetr_device": args.qddetr_device,
        "qddetr_context_window": args.qddetr_context_window,
        "qddetr_min_duration": args.qddetr_min_duration,
        "qddetr_min_area": args.qddetr_min_area,
        "qddetr_feat_dirs": args.qddetr_feat_dirs,
        "qddetr_text_feat_dir": args.qddetr_text_feat_dir,
        "qddetr_annotation_path": args.qddetr_annotation_path,
    }

    # Initialize and run pipeline
    pipeline = PVSPipeline(config=config)
    pipeline.run(input_video=args.input, video_name=args.video_name, video_path=args.video_path, show_plots=args.show_plots)


if __name__ == "__main__":
    main()
