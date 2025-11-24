"""
Main Pipeline for Personalized Video Summarization (PVS) - New Approach

This script orchestrates the complete workflow with the new independent component approach:
1. Video summarization (SportCLIP or QD-DETR) - saves to video_summaries/
2. Face detection on entire videos - uses/creates cropped_faces/
3. Face embedding extraction - uses/creates cropped_faces_embeddings/
4. Find updated target faces - creates target_faces_updated/
5. Compute resemblance curves - creates resemblance_results/
6. Generate final visualizations with highlight segments overlaid

Key differences from old approach:
- Face detection runs on ENTIRE videos, not just summaries
- Face crops and embeddings are cached at project root for reuse
- Video summaries are organized by model (SportCLIP/QD-DETR) and video
- Resemblance curves show where highlights occur in the timeline
"""

import os
import sys
import json
import subprocess
import numpy as np
import h5py
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import argparse

# Import existing modules
from video_summarization import generate_highlights_sportclip, generate_highlights_qddetr
from timing import TimingTracker


def get_cuda_library_path() -> str:
    """
    Get the CUDA library path for onnxruntime-gpu compatibility

    Returns the colon-separated library paths from nvidia pip packages,
    or empty string if not found.
    """
    try:
        # Base path for nvidia packages
        base_path = os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages", "nvidia")

        # List of library directories to add
        lib_dirs = [
            "cublas/lib",
            "cudnn/lib",
            "cuda_runtime/lib",
            "cuda_nvrtc/lib",
            "curand/lib",
            "cufft/lib",
            "cusparse/lib",
            "cusolver/lib",
            "nvjitlink/lib",
        ]

        # Collect existing directories
        paths = []
        for lib_dir in lib_dirs:
            full_path = os.path.join(base_path, lib_dir)
            if os.path.exists(full_path):
                paths.append(full_path)

        return ":".join(paths)
    except Exception as e:
        print(f"Warning: Could not determine CUDA library path: {e}")
        return ""


class PVSPipelineNew:
    """Main pipeline orchestrator for Personalized Video Summarization (new approach)"""

    def __init__(self, config: dict = None):
        """
        Initialize the PVS pipeline

        Args:
            config: Configuration dictionary with pipeline parameters
        """
        self.config = config or {}

        # Directory paths (new structure)
        self.data_dir = self.config.get("data_dir", "data")
        self.target_faces_dir = self.config.get("target_faces_dir", "target_faces")
        self.cropped_faces_dir = self.config.get("cropped_faces_dir", "cropped_faces")
        self.embeddings_dir = self.config.get("embeddings_dir", "cropped_faces_embeddings")
        self.video_summaries_dir = self.config.get("video_summaries_dir", "video_summaries")
        self.target_faces_updated_dir = self.config.get("target_faces_updated_dir", "target_faces_updated")
        self.resemblance_results_dir = self.config.get("resemblance_results_dir", "resemblance_results")
        self.output_dir = self.config.get("output_dir", "results")

        # Video summarization config
        self.video_summarizer = self.config.get("video_summarizer", "sportclip")
        self.use_updated_target = self.config.get("use_updated_target", False)

        # SportCLIP config
        self.sportclip_dir = self.config.get("sportclip_dir", "/home/mrt/Projects/SportCLIP-official")
        self.sportclip_dataset_dir = self.config.get("sportclip_dataset_dir", "/mnt/Data/mrt/SportCLIP-OlympicHighlights")
        self.sportclip_sentences_file = self.config.get("sportclip_sentences_file", None)

        # QD-DETR config
        self.qddetr_ckpt_path = self.config.get("qddetr_ckpt_path", "/home/mrt/Projects/QD-DETR/results/OlympicHighlights-exp-2025_09_25_11_22_34/model_best.ckpt")
        self.qddetr_query = self.config.get("qddetr_query", None)
        self.qddetr_device = self.config.get("qddetr_device", "cuda:0")
        self.qddetr_context_window = self.config.get("qddetr_context_window", 600)
        self.qddetr_min_duration = self.config.get("qddetr_min_duration", 15)
        self.qddetr_min_area = self.config.get("qddetr_min_area", 15.0)
        self.qddetr_feat_dirs = self.config.get("qddetr_feat_dirs", None)
        self.qddetr_text_feat_dir = self.config.get("qddetr_text_feat_dir", None)
        self.qddetr_annotation_path = self.config.get("qddetr_annotation_path", None)
        self.qddetr_vid_name = self.config.get("qddetr_vid_name", None)

        # Face recognition config
        self.transface_weight = self.config.get("transface_weight", "/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt")
        self.transface_network = self.config.get("transface_network", "vit_l_dp005_mask_005")
        self.similarity_threshold = self.config.get("similarity_threshold", 0.5)
        self.resemblance_threshold = self.config.get("resemblance_threshold", 0.6)
        self.smooth_window = self.config.get("smooth_window", 5)
        self.face_recognition_models = self.config.get("face_recognition_models", None)

        # Ensure key directories exist
        Path(self.target_faces_dir).mkdir(parents=True, exist_ok=True)
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Setup CUDA library path for onnxruntime-gpu
        self.cuda_lib_path = get_cuda_library_path()

        # Initialize timing tracker (aggregate log for all runs)
        self.timing_tracker = TimingTracker(log_file=str(Path(self.output_dir) / "timing_log_aggregate.json"))

    def _get_subprocess_env(self) -> dict:
        """
        Get environment variables for subprocess calls with CUDA library path

        Returns:
            Dictionary of environment variables with LD_LIBRARY_PATH configured
        """
        env = os.environ.copy()
        if self.cuda_lib_path:
            # Prepend CUDA library path to LD_LIBRARY_PATH
            current_ld_path = env.get("LD_LIBRARY_PATH", "")
            if current_ld_path:
                env["LD_LIBRARY_PATH"] = f"{self.cuda_lib_path}:{current_ld_path}"
            else:
                env["LD_LIBRARY_PATH"] = self.cuda_lib_path
        return env

    def step1_video_summarization(self, video_name: str, video_path: str) -> Optional[Dict]:
        """
        Step 1: Video Summarization

        Generate highlight clips using SportCLIP or QD-DETR.
        Saves clips to video_summaries/<model>/<video_name>/

        Args:
            video_name: Video name (without extension)
            video_path: Full path to video file

        Returns:
            Dictionary with segment info or None if failed
        """
        print("\n" + "=" * 80)
        print(f"STEP 1: VIDEO SUMMARIZATION ({self.video_summarizer.upper()})")
        print("=" * 80)

        # Track timing for video summarization
        import time

        step_start_time = time.time()

        # Determine output directory
        model_dir = "SportCLIP" if self.video_summarizer == "sportclip" else "QD-DETR"
        video_output_dir = Path(self.video_summaries_dir) / model_dir / video_name
        segments_file = video_output_dir / "segments.json"

        # Check if already processed
        if segments_file.exists():
            print(f"\n✓ Video summaries already exist for {video_name}")
            print(f"  Location: {video_output_dir}")
            print("  Skipping video summarization step...")

            with open(segments_file, "r") as f:
                segments_data = json.load(f)

            # Get segment count (handle nested structure)
            segments_obj = segments_data.get("segments", [])
            if isinstance(segments_obj, dict):
                num_clips = len(segments_obj.get("segments", []))
            else:
                num_clips = len(segments_obj)

            print(f"  Found {num_clips} highlight clips")
            return segments_data

        # Create output directory
        video_output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nProcessing {video_name} with {self.video_summarizer.upper()}...")

        try:
            if self.video_summarizer == "sportclip":
                if not self.sportclip_sentences_file:
                    raise ValueError("sportclip_sentences_file must be provided")

                clip_paths, segments_metadata = generate_highlights_sportclip(
                    video_name=video_name,
                    sentences_file=self.sportclip_sentences_file,
                    output_dir=str(video_output_dir),
                    sportclip_dir=self.sportclip_dir,
                    sportclip_dataset_dir=self.sportclip_dataset_dir,
                    video_path=video_path,
                )
            else:  # qddetr
                if not self.qddetr_query:
                    raise ValueError("qddetr_query must be provided")

                # Use qddetr_vid_name if provided, otherwise use video_name
                qddetr_lookup_name = self.qddetr_vid_name if self.qddetr_vid_name else video_name

                clip_paths, segments_metadata = generate_highlights_qddetr(
                    video_name=qddetr_lookup_name,
                    query=self.qddetr_query,
                    output_dir=str(video_output_dir),
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

            if not clip_paths:
                print(f"Warning: No highlights generated for {video_name}")
                return None

            # Save segments metadata (with timing information)
            segments_data = {"video_name": video_name, "model": self.video_summarizer, "total_segments": len(segments_metadata), "segments": segments_metadata}

            # Save metadata
            with open(segments_file, "w") as f:
                json.dump(segments_data, f, indent=2)

            print(f"\n✓ Generated {len(clip_paths)} highlight clips with timing information")
            print(f"  Saved to: {video_output_dir}")

            # Save timing data for video summarization
            step_duration = time.time() - step_start_time
            tracker = TimingTracker(log_file=str(Path(self.output_dir) / "timing_log_aggregate.json"))
            tracker.timings.append(
                {
                    "component": "Video Summarization",
                    "duration_seconds": round(step_duration, 4),
                    "cpu_percent": 0,
                    "memory_mb": 0,
                    "memory_delta_mb": 0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "video_name": video_name,
                    "method": self.video_summarizer,
                }
            )
            tracker.save(output_dir=self.output_dir)

            return segments_data

        except Exception as e:
            print(f"\nError in video summarization: {e}")
            return None

    def step2_face_detection(self, video_name: str) -> bool:
        """
        Step 2: Face Detection on Entire Video

        Extract faces from all frames using extract_all_faces.py script.
        Checks if faces already extracted to avoid reprocessing.

        Args:
            video_name: Video name (without extension)

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("STEP 2: FACE DETECTION ON ENTIRE VIDEO")
        print("=" * 80)

        video_faces_dir = Path(self.cropped_faces_dir) / video_name

        # Check if already processed
        if video_faces_dir.exists():
            face_images = list(video_faces_dir.glob("*.jpg")) + list(video_faces_dir.glob("*.png"))
            if face_images:
                print(f"\n✓ Face crops already exist for {video_name}")
                print(f"  Location: {video_faces_dir}")
                print(f"  Found {len(face_images)} face crops")
                print("  Skipping face detection step...")
                return True

        print(f"\nExtracting faces from {video_name}...")
        print(f"This may take a while as it processes every frame...")

        try:
            # Call extract_all_faces.py script
            cmd = [sys.executable, "extract_all_faces.py", "--data-dir", self.data_dir, "--output-dir", self.cropped_faces_dir]

            # Run with real-time output streaming to show progress bars
            # Use environment with CUDA library path for GPU support
            result = subprocess.run(cmd, check=True, text=True, env=self._get_subprocess_env())

            # Verify faces were extracted
            if video_faces_dir.exists():
                face_images = list(video_faces_dir.glob("*.jpg"))
                print(f"\n✓ Extracted {len(face_images)} faces from {video_name}")
                return True
            else:
                print(f"\nError: Face extraction failed - no output directory created")
                return False

        except subprocess.CalledProcessError as e:
            print(f"\nError running face detection: {e}")
            return False

    def step3_embedding_extraction(self, video_name: str) -> bool:
        """
        Step 3: Extract Face Embeddings

        Extract ArcFace and TransFace embeddings using extract_embeddings.py script.
        Checks if embeddings already exist to avoid reprocessing.

        Args:
            video_name: Video name (without extension)

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("STEP 3: FACE EMBEDDING EXTRACTION")
        print("=" * 80)

        arcface_h5 = Path(self.embeddings_dir) / "ArcFace" / f"{video_name}.h5"
        transface_h5 = Path(self.embeddings_dir) / "TransFace" / f"{video_name}.h5"

        # Check if already processed
        if arcface_h5.exists() and transface_h5.exists():
            print(f"\n✓ Embeddings already exist for {video_name}")
            print(f"  ArcFace: {arcface_h5}")
            print(f"  TransFace: {transface_h5}")

            # Show count
            try:
                with h5py.File(str(arcface_h5), "r") as f:
                    arcface_count = len(f.keys())
                with h5py.File(str(transface_h5), "r") as f:
                    transface_count = len(f.keys())
                print(f"  ArcFace embeddings: {arcface_count}")
                print(f"  TransFace embeddings: {transface_count}")
            except:
                pass

            print("  Skipping embedding extraction step...")
            return True

        print(f"\nExtracting embeddings for {video_name}...")
        print(f"This may take a while...")

        try:
            # Call extract_embeddings.py script
            cmd = [
                sys.executable,
                "extract_embeddings.py",
                "--input-dir",
                self.cropped_faces_dir,
                "--output-dir",
                self.embeddings_dir,
                "--models",
                "ArcFace",
                "TransFace",
                "--transface-weight",
                self.transface_weight,
                "--transface-network",
                self.transface_network,
            ]

            # Run with real-time output streaming to show progress bars
            result = subprocess.run(cmd, check=True, text=True)

            # Verify embeddings were created
            if arcface_h5.exists() and transface_h5.exists():
                print(f"\n✓ Successfully extracted embeddings for {video_name}")
                return True
            else:
                print(f"\nError: Embedding extraction failed - output files not created")
                return False

        except subprocess.CalledProcessError as e:
            print(f"\nError running embedding extraction: {e}")
            return False

    def step4_find_updated_targets(self, video_name: str) -> List[str]:
        """
        Step 4: Find Updated Target Faces

        For each target in target_faces/, find the best matching face in the video.
        Uses find_updated_target.py script.

        Args:
            video_name: Video name (without extension)

        Returns:
            List of target names that were processed successfully
        """
        print("\n" + "=" * 80)
        print("STEP 4: FIND UPDATED TARGET FACES")
        print("=" * 80)

        # Find all targets for this specific video
        video_target_faces_dir = Path(self.target_faces_dir) / video_name

        if not video_target_faces_dir.exists():
            print(f"\nWarning: No target faces directory found for {video_name}")
            print(f"  Expected location: {video_target_faces_dir}")
            print(f"  Please create the directory and add target face images for this video.")
            return []

        target_images = list(video_target_faces_dir.glob("*.jpg")) + list(video_target_faces_dir.glob("*.png")) + list(video_target_faces_dir.glob("*.jpeg"))

        if not target_images:
            print(f"\nWarning: No target faces found in {video_target_faces_dir}")
            return []

        print(f"\nFound {len(target_images)} target(s) for {video_name}:")
        for img in target_images:
            print(f"  - {img.stem}")

        successful_targets = []

        for target_img in target_images:
            target_name = target_img.stem
            # Updated targets are organized by video name and then by target name
            target_output_dir = Path(self.target_faces_updated_dir) / video_name / target_name
            log_file = target_output_dir / f"{target_name}_log.txt"

            # Check if already processed
            if log_file.exists():
                print(f"\n✓ Updated target already exists for {target_name}")
                print(f"  Skipping...")
                successful_targets.append(target_name)
                continue

            print(f"\nProcessing target: {target_name}")

            try:
                # Call find_updated_target.py script
                # Pass video-specific output directory
                video_updated_output_dir = Path(self.target_faces_updated_dir) / video_name

                cmd = [
                    sys.executable,
                    "find_updated_target.py",
                    "--target-faces",
                    str(video_target_faces_dir),
                    "--cropped-faces",
                    self.cropped_faces_dir,
                    "--embeddings",
                    self.embeddings_dir,
                    "--output",
                    str(video_updated_output_dir),
                    "--video",
                    video_name,
                    "--threshold",
                    str(self.similarity_threshold),
                    "--transface-weight",
                    self.transface_weight,
                    "--transface-network",
                    self.transface_network,
                ]

                # Run with real-time output streaming to show progress bars
                # Use environment with CUDA library path for GPU support
                result = subprocess.run(cmd, check=True, text=True, env=self._get_subprocess_env())

                if log_file.exists():
                    print(f"  ✓ Successfully processed {target_name}")
                    successful_targets.append(target_name)
                else:
                    print(f"  Warning: Log file not created for {target_name}")

            except subprocess.CalledProcessError as e:
                print(f"  Error processing {target_name}: {e}")
                continue

        print(f"\n✓ Processed {len(successful_targets)}/{len(target_images)} targets")
        return successful_targets

    def step4_extract_original_target_embeddings(self, video_name: str, target_names: List[str]) -> bool:
        """
        Step 4 (Alternative): Extract Embeddings from Original Target Faces

        This step extracts embeddings directly from the original target images in target_faces/<video_name>/.
        Used when --use-updated-target is NOT set (first pass only).

        Args:
            video_name: Video name (without extension)
            target_names: List of target names to process

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("STEP 4: EXTRACT EMBEDDINGS FROM ORIGINAL TARGET FACES")
        print("=" * 80)

        if not target_names:
            print("\nNo targets to process")
            return False

        print(f"\nExtracting embeddings from original target faces for {len(target_names)} target(s)...")
        print("Note: Using original target images (not updated from video)")

        # Use video-specific target faces directory
        video_target_faces_dir = Path(self.target_faces_dir) / video_name
        # Use video-specific output directory for updated targets
        video_updated_targets_dir = Path(self.target_faces_updated_dir) / video_name

        try:
            # Call unified extract_embeddings.py script with original_targets mode
            cmd = [
                sys.executable,
                "extract_embeddings.py",
                "--mode",
                "original_targets",
                "--target-faces",
                str(video_target_faces_dir),
                "--output-dir",
                str(video_updated_targets_dir),
                "--transface-weight",
                self.transface_weight,
                "--transface-network",
                self.transface_network,
            ]

            # Run with real-time output streaming to show progress bars
            # Use environment with CUDA library path for GPU support
            result = subprocess.run(cmd, check=True, text=True, env=self._get_subprocess_env())

            print(f"\n✓ Successfully extracted original target embeddings")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\nError extracting original target embeddings: {e}")
            return False

    def step4b_extract_updated_target_embeddings(self, video_name: str, target_names: List[str]) -> bool:
        """
        Step 4b: Extract Embeddings from Updated Targets (Second Pass)

        This step extracts embeddings from the updated target faces (best matches from video).
        When these embeddings are compared against the video, we should get 100% similarity
        for the exact frame where the updated target came from.

        Args:
            video_name: Video name (without extension)
            target_names: List of target names to process

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("STEP 4b: EXTRACT EMBEDDINGS FROM UPDATED TARGETS (SECOND PASS)")
        print("=" * 80)

        if not target_names:
            print("\nNo targets to process")
            return False

        print(f"\nExtracting embeddings from updated targets for {len(target_names)} target(s)...")
        print("This will enable 100% similarity match in resemblance curves")

        try:
            # Call unified extract_embeddings.py script with updated_targets mode
            # Pass video-specific updated targets directory
            video_updated_targets_dir = Path(self.target_faces_updated_dir) / video_name

            cmd = [
                sys.executable,
                "extract_embeddings.py",
                "--mode",
                "updated_targets",
                "--updated-targets",
                str(video_updated_targets_dir),
                "--video",
                video_name,
                "--transface-weight",
                self.transface_weight,
                "--transface-network",
                self.transface_network,
            ]

            # Run with real-time output streaming to show progress bars
            result = subprocess.run(cmd, check=True, text=True)

            print(f"\n✓ Successfully extracted updated target embeddings")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\nError extracting updated target embeddings: {e}")
            return False

    def step5_compute_resemblance_curves(self, video_name: str, target_names: List[str]) -> bool:
        """
        Step 5: Compute Resemblance Curves

        Generate resemblance score curves for all targets.
        Uses compute_resemblance_curve.py script.

        Args:
            video_name: Video name (without extension)
            target_names: List of target names to process

        Returns:
            True if successful, False otherwise
        """
        print("\n" + "=" * 80)
        print("STEP 5: COMPUTE RESEMBLANCE CURVES")
        print("=" * 80)

        if not target_names:
            print("\nNo targets to process")
            return False

        print(f"\nComputing resemblance curves for {len(target_names)} target(s) in {video_name}...")

        try:
            # Call compute_resemblance_curve.py script
            # This will process all targets automatically
            # Pass video-specific updated targets path
            video_updated_targets_dir = Path(self.target_faces_updated_dir) / video_name
            video_resemblance_results_dir = Path(self.resemblance_results_dir) / video_name

            cmd = [
                sys.executable,
                "compute_resemblance_curve.py",
                "--video",
                video_name,
                "--updated-targets",
                str(video_updated_targets_dir),
                "--embeddings",
                self.embeddings_dir,
                "--output",
                str(video_resemblance_results_dir),
                "--threshold",
                str(self.resemblance_threshold),
                "--smooth-window",
                str(self.smooth_window),
            ]

            # Add flag if using updated target embeddings
            if self.use_updated_target:
                cmd.append("--use-updated-target")

            # Run with real-time output streaming to show progress bars
            result = subprocess.run(cmd, check=True, text=True)

            print(f"\n✓ Successfully computed resemblance curves")
            print(f"  Results saved to: {self.resemblance_results_dir}")
            return True

        except subprocess.CalledProcessError as e:
            print(f"\nError computing resemblance curves: {e}")
            return False

    def _save_config_log(self, video_name: str, target_name: str, output_dir: Path):
        """
        Save configuration parameters to config_log.txt

        Args:
            video_name: Video name
            target_name: Target person name
            output_dir: Directory to save config file
        """
        from datetime import datetime

        config_file = output_dir / "config_log.txt"

        with open(config_file, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PERSONALIZED VIDEO SUMMARIZATION - CONFIGURATION LOG\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Video: {video_name}\n")
            f.write(f"Target: {target_name}\n\n")

            f.write("VIDEO SUMMARIZATION\n")
            f.write("-" * 80 + "\n")
            f.write(f"Method: {self.video_summarizer.upper()}\n")

            if self.video_summarizer == "sportclip":
                f.write(f"SportCLIP Directory: {self.sportclip_dir}\n")
                f.write(f"Dataset Directory: {self.sportclip_dataset_dir}\n")
                f.write(f"Sentences File: {self.sportclip_sentences_file}\n")
            else:  # qddetr
                f.write(f"Checkpoint: {self.qddetr_ckpt_path}\n")
                f.write(f"Query: {self.qddetr_query}\n")
                f.write(f"Device: {self.qddetr_device}\n")
                f.write(f"Context Window: {self.qddetr_context_window} frames\n")
                f.write(f"Min Duration: {self.qddetr_min_duration} frames\n")
                f.write(f"Min Area: {self.qddetr_min_area}\n")

            f.write("\nFACE RECOGNITION\n")
            f.write("-" * 80 + "\n")
            f.write("Models: ArcFace + TransFace\n")
            f.write(f"TransFace Weight: {self.transface_weight}\n")
            f.write(f"TransFace Network: {self.transface_network}\n")
            f.write(f"Similarity Threshold (target matching): {self.similarity_threshold}\n")
            f.write(f"Resemblance Threshold (visualization): {self.resemblance_threshold}\n")
            f.write(f"Smoothing Window: {self.smooth_window}\n")
            f.write(f"Use Updated Target: {self.use_updated_target}\n")

            f.write("\nDIRECTORY PATHS\n")
            f.write("-" * 80 + "\n")
            f.write(f"Data Directory: {self.data_dir}\n")
            f.write(f"Target Faces: {self.target_faces_dir}\n")
            f.write(f"Cropped Faces: {self.cropped_faces_dir}\n")
            f.write(f"Embeddings: {self.embeddings_dir}\n")
            f.write(f"Video Summaries: {self.video_summaries_dir}\n")
            f.write(f"Updated Targets: {self.target_faces_updated_dir}\n")
            f.write(f"Resemblance Results: {self.resemblance_results_dir}\n")
            f.write(f"Final Output: {self.output_dir}\n")

        print(f"    ✓ Saved config log: {config_file.name}")

    def step6_create_final_visualization(self, video_name: str, segments_data: Dict, target_names: List[str]):
        """
        Step 6: Create Final Visualization

        Create enhanced resemblance curve plots with highlight segments overlaid.
        Shows where highlights occur on the timeline.
        Copies relevant highlight clips to results directory.

        Args:
            video_name: Video name (without extension)
            segments_data: Video summary segments metadata
            target_names: List of target names
        """
        print("\n" + "=" * 80)
        print("STEP 6: CREATE FINAL VISUALIZATION WITH HIGHLIGHTS")
        print("=" * 80)

        if not target_names:
            print("\nNo targets to visualize")
            return

        # Get segment count (handle nested structure)
        segments_obj = segments_data.get("segments", [])
        if isinstance(segments_obj, dict):
            num_segments = len(segments_obj.get("segments", []))
        else:
            num_segments = len(segments_obj)

        print(f"\nCreating enhanced visualizations for {len(target_names)} target(s)...")
        print(f"Highlight segments: {num_segments}")

        try:
            # Call unified visualize_results.py script with final_with_highlights mode
            # Pass video-specific resemblance results path
            video_resemblance_results_dir = Path(self.resemblance_results_dir) / video_name

            cmd = [
                sys.executable,
                "visualize_results.py",
                "--mode",
                "final_with_highlights",
                "--video",
                video_name,
                "--model",
                self.video_summarizer,
                "--video-summaries",
                self.video_summaries_dir,
                "--resemblance-results",
                str(video_resemblance_results_dir),
                "--output",
                self.output_dir,
                "--threshold",
                str(self.resemblance_threshold),
                "--smooth-window",
                str(self.smooth_window),
            ]

            # Add flag if using updated target embeddings
            if self.use_updated_target:
                cmd.append("--use-updated-target")

            # Add face recognition models if specified
            if self.face_recognition_models:
                cmd.append("--face-recognition-models")
                cmd.extend(self.face_recognition_models)

            # Run with real-time output streaming to show progress bars
            result = subprocess.run(cmd, check=True, text=True)

            print(f"\n✓ Successfully created final visualizations")
            print(f"  Results saved to: {self.output_dir}/{video_name}/{self.video_summarizer}/")

            # Save config log for each target
            print(f"\nSaving configuration logs...")
            for target_name in target_names:
                target_output_dir = Path(self.output_dir) / video_name / self.video_summarizer / target_name
                if target_output_dir.exists():
                    self._save_config_log(video_name, target_name, target_output_dir)

        except subprocess.CalledProcessError as e:
            print(f"\nError creating final visualizations: {e}")

    def run(self, video_name: str, video_path: str):
        """
        Run the complete PVS pipeline on a single video

        Args:
            video_name: Video name (without extension)
            video_path: Full path to video file
        """
        print("\n" + "#" * 80)
        print("# PERSONALIZED VIDEO SUMMARIZATION PIPELINE - NEW APPROACH")
        print("#" * 80)
        print(f"\nProcessing: {video_name}")
        print(f"Video path: {video_path}")

        # Step 1: Video Summarization
        segments_data = self.step1_video_summarization(video_name, video_path)
        if segments_data is None:
            print("\nError: Video summarization failed. Aborting pipeline.")
            return

        # Step 2: Face Detection
        success = self.step2_face_detection(video_name)
        if not success:
            print("\nError: Face detection failed. Aborting pipeline.")
            return

        # Step 3: Embedding Extraction
        success = self.step3_embedding_extraction(video_name)
        if not success:
            print("\nError: Embedding extraction failed. Aborting pipeline.")
            return

        # Step 4 (conditional): Find Updated Targets - only if --use-updated-target is enabled
        target_names = []
        if self.use_updated_target:
            target_names = self.step4_find_updated_targets(video_name)
            if not target_names:
                print("\nWarning: No targets processed. Aborting face recognition workflow...")
                return

            # Step 4b: Extract Updated Target Embeddings (second pass)
            success = self.step4b_extract_updated_target_embeddings(video_name, target_names)
            if not success:
                print("\nWarning: Updated target embedding extraction failed.")
        else:
            # Without --use-updated-target, we'll use original target faces directly
            # Get target names from video-specific target_faces/<video_name>/ directory
            video_target_faces_dir = Path(self.target_faces_dir) / video_name

            if not video_target_faces_dir.exists():
                print(f"\nWarning: No target faces directory found for {video_name}")
                print(f"  Expected location: {video_target_faces_dir}")
                print(f"  Please create the directory and add target face images for this video.")
                return

            target_images = list(video_target_faces_dir.glob("*.jpg")) + list(video_target_faces_dir.glob("*.png")) + list(video_target_faces_dir.glob("*.jpeg"))
            target_names = [img.stem for img in target_images]

            if not target_names:
                print(f"\nWarning: No target faces found in {video_target_faces_dir}")
                return

            # Extract embeddings from original target faces
            success = self.step4_extract_original_target_embeddings(video_name, target_names)
            if not success:
                print("\nWarning: Original target embedding extraction failed.")
                return

        # Step 5: Compute Resemblance Curves (always runs)
        success = self.step5_compute_resemblance_curves(video_name, target_names)
        if not success:
            print("\nWarning: Resemblance curve computation failed.")

        # Step 6: Create Final Visualization (always runs)
        self.step6_create_final_visualization(video_name, segments_data, target_names)

        # Save timing data and print summary
        self.timing_tracker.save(output_dir=self.output_dir)
        self.timing_tracker.print_summary(title=f"Timing Summary - {video_name}")

        # ALWAYS generate timing analysis (automatic)
        print("\nGenerating timing analysis...")
        try:
            import timing

            timing_output_dir = Path(self.output_dir) / "timing_analysis"
            timing_output_dir.mkdir(parents=True, exist_ok=True)

            aggregate_log = Path(self.output_dir) / "timing_log_aggregate.json"
            if aggregate_log.exists():
                df = timing.load_timing_log(str(aggregate_log))
                normalized_df = timing.normalize_timing_data(df)

                if not normalized_df.empty:
                    stats = timing.compute_statistics(normalized_df)

                    # Generate outputs
                    timing.generate_markdown_table(stats, str(timing_output_dir / "timing_summary.md"))
                    timing.create_bar_chart(stats, str(timing_output_dir / "timing_bar_chart.png"))
                    timing.create_comparison_table(stats, str(timing_output_dir / "timing_comparison.md"))

                    print(f"✓ Timing analysis saved to: {timing_output_dir}")
                else:
                    print("⚠ No valid timing data for analysis yet")
        except Exception as e:
            print(f"⚠ Warning: Could not generate timing analysis: {e}")

        print("\n" + "#" * 80)
        print("# PIPELINE COMPLETE")
        print("#" * 80)
        print(f"\nResults summary:")
        print(f"  - Video summaries: {self.video_summaries_dir}/{self.video_summarizer.upper()}/{video_name}/")
        print(f"  - Face crops: {self.cropped_faces_dir}/{video_name}/")
        print(f"  - Embeddings: {self.embeddings_dir}/")
        if self.use_updated_target:
            print(f"  - Updated targets (from video): {self.target_faces_updated_dir}/")
        else:
            print(f"  - Target embeddings (from original): {self.target_faces_updated_dir}/")
        print(f"  - Resemblance curves: {self.resemblance_results_dir}/")
        print(f"  - Final output: {self.output_dir}/{video_name}/{self.video_summarizer}/")
        print(f"  - Timing log: {self.timing_tracker.log_file}")


def main():
    """Main entry point with command-line argument parsing"""
    parser = argparse.ArgumentParser(
        description="Personalized Video Summarization Pipeline (New Approach)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a video with SportCLIP
  python main_new.py --video longjump_video1 \\
      --video-summarizer sportclip \\
      --sentences-file data/sentences/long_jump.json

  # Process a video with QD-DETR
  python main_new.py --video highjump_video1 \\
      --video-summarizer qddetr \\
      --qddetr-query "highlight moments in this high jump competition"

  # Use custom similarity thresholds
  python main_new.py --video longjump_video1 \\
      --video-summarizer sportclip \\
      --sentences-file data/sentences/long_jump.json \\
      --similarity-threshold 0.6 \\
      --resemblance-threshold 0.65
        """,
    )

    # Required parameters
    parser.add_argument("--video", type=str, required=True, help="Video name to process (without extension)")

    # Video summarizer selection
    parser.add_argument("--video-summarizer", type=str, default="sportclip", choices=["sportclip", "qddetr"], help="Video summarization method (default: sportclip)")

    # Directory parameters
    parser.add_argument("--data-dir", type=str, default="data", help="Directory containing video files (default: data)")
    parser.add_argument("--target-faces", type=str, default="target_faces", help="Directory containing target face images (default: target_faces)")
    parser.add_argument("--output", type=str, default="results", help="Output directory for final results (default: results)")

    # SportCLIP parameters
    parser.add_argument("--sentences-file", type=str, default=None, help="Path to SportCLIP sentences file (required for SportCLIP)")
    parser.add_argument("--sportclip-dir", type=str, default="/home/mrt/Projects/SportCLIP-official", help="SportCLIP project directory")
    parser.add_argument("--sportclip-dataset-dir", type=str, default="/mnt/Data/mrt/SportCLIP-OlympicHighlights", help="SportCLIP dataset directory")

    # QD-DETR parameters
    parser.add_argument(
        "--qddetr-ckpt", type=str, default="/home/mrt/Projects/QD-DETR/results/OlympicHighlights-exp-2025_09_25_11_22_34/model_best.ckpt", help="Path to QD-DETR checkpoint"
    )
    parser.add_argument("--qddetr-query", type=str, default=None, help="Query for QD-DETR (required for QD-DETR)")
    parser.add_argument("--qddetr-device", type=str, default="cuda:0", help="Device for QD-DETR inference")
    parser.add_argument("--qddetr-context-window", type=int, default=600, help="Context window size in frames for QD-DETR")
    parser.add_argument("--qddetr-min-duration", type=int, default=15, help="Minimum event duration in frames for QD-DETR")
    parser.add_argument("--qddetr-min-area", type=float, default=15.0, help="Minimum event area threshold for QD-DETR")
    parser.add_argument("--qddetr-feat-dirs", nargs="+", default=None, help="Video feature directories for QD-DETR")
    parser.add_argument("--qddetr-text-feat-dir", type=str, default=None, help="Text feature directory for QD-DETR")
    parser.add_argument("--qddetr-annotation-path", type=str, default=None, help="Path to QD-DETR annotation JSONL file")
    parser.add_argument("--qddetr-vid-name", type=str, default=None, help="Video name as it appears in QD-DETR annotation file (if different from --video)")

    # Face recognition parameters
    parser.add_argument(
        "--transface-weight", type=str, default="/home/mrt/Projects/TransFace/work_dirs/glint360k_vit_l/glint360k_model_TransFace_L.pt", help="Path to TransFace model weights"
    )
    parser.add_argument("--transface-network", type=str, default="vit_l_dp005_mask_005", help="TransFace network architecture")
    parser.add_argument("--similarity-threshold", type=float, default=0.5, help="Similarity threshold for finding updated targets (default: 0.5)")
    parser.add_argument("--resemblance-threshold", type=float, default=0.6, help="Resemblance threshold for visualization (default: 0.6)")
    parser.add_argument("--smooth-window", type=int, default=5, help="Smoothing window for resemblance curves (default: 5)")
    parser.add_argument(
        "--use-updated-target",
        action="store_true",
        help="Perform second pass using updated target embeddings (should achieve 100%% similarity). Results saved with '_updated' suffix to avoid overwriting.",
    )
    parser.add_argument(
        "--face-recognition-models",
        nargs="+",
        choices=["arcface", "transface"],
        default=None,
        help=(
            "Face recognition models to use for highlight clip selection. Options: arcface, transface. Default: both models. "
            "Results will be saved in separate folders (e.g., ArcFace_highlight_clips, TransFace_highlight_clips)."
        ),
    )

    args = parser.parse_args()

    # Validate required parameters
    if args.video_summarizer == "sportclip" and not args.sentences_file:
        print("Error: --sentences-file is required when using SportCLIP")
        return

    if args.video_summarizer == "qddetr" and not args.qddetr_query:
        print("Error: --qddetr-query is required when using QD-DETR")
        return

    # Find video file
    data_dir = Path(args.data_dir)
    video_extensions = [".mp4", ".avi", ".mov", ".mkv"]
    video_path = None

    for ext in video_extensions:
        candidate = data_dir / f"{args.video}{ext}"
        if candidate.exists():
            video_path = str(candidate)
            break

    if not video_path:
        print(f"Error: Video not found: {args.video}")
        print(f"Searched in: {data_dir}")
        return

    # Create configuration
    config = {
        "data_dir": args.data_dir,
        "target_faces_dir": args.target_faces,
        "output_dir": args.output,
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
        "qddetr_vid_name": args.qddetr_vid_name,
        # Face recognition config
        "transface_weight": args.transface_weight,
        "transface_network": args.transface_network,
        "similarity_threshold": args.similarity_threshold,
        "resemblance_threshold": args.resemblance_threshold,
        "smooth_window": args.smooth_window,
        "use_updated_target": args.use_updated_target,
        "face_recognition_models": args.face_recognition_models,
    }

    # Initialize and run pipeline
    pipeline = PVSPipelineNew(config=config)
    pipeline.run(video_name=args.video, video_path=video_path)


if __name__ == "__main__":
    main()
