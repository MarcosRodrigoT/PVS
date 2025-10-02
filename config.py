"""
Configuration settings for the PVS pipeline
"""

import os
from pathlib import Path


class Config:
    """Configuration class for PVS pipeline"""

    # Project root directory
    ROOT_DIR = Path(__file__).parent.absolute()

    # Data directories
    DATA_DIR = ROOT_DIR / "data"
    VIDEO_SUMMARIES_DIR = DATA_DIR / "video_summaries"
    CROPPED_FACES_DIR = DATA_DIR / "cropped_faces"
    VISUALIZATIONS_DIR = DATA_DIR / "visualizations"

    # Input directories
    TARGET_FACES_DIR = ROOT_DIR / "target_faces"

    # Output files
    FACE_DETECTIONS_JSON = DATA_DIR / "face_detections.json"
    FACE_RECOGNITION_JSON = DATA_DIR / "face_recognition_results.json"
    SUMMARY_REPORT_TXT = DATA_DIR / "summary_report.txt"

    # Model configurations
    FACE_DETECTION_MODEL = "InsightFace SCRFD"  # InsightFace SCRFD for state-of-the-art accuracy
    FACE_RECOGNITION_MODEL = "ArcFace"  # ArcFace, Facenet, VGG-Face, DeepFace, etc.

    # Recognition parameters
    SIMILARITY_THRESHOLD = 0.6  # Threshold for face match (0-1)

    # Video processing parameters
    PROCESS_EVERY_N_FRAMES = 1  # Process every Nth frame (1 = all frames)

    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [cls.DATA_DIR, cls.VIDEO_SUMMARIES_DIR, cls.CROPPED_FACES_DIR, cls.VISUALIZATIONS_DIR, cls.TARGET_FACES_DIR]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)

    @classmethod
    def get_config_dict(cls) -> dict:
        """
        Get configuration as a dictionary

        Returns:
            Dictionary containing all configuration parameters
        """
        return {
            "root_dir": str(cls.ROOT_DIR),
            "data_dir": str(cls.DATA_DIR),
            "video_summaries_dir": str(cls.VIDEO_SUMMARIES_DIR),
            "cropped_faces_dir": str(cls.CROPPED_FACES_DIR),
            "visualizations_dir": str(cls.VISUALIZATIONS_DIR),
            "target_faces_dir": str(cls.TARGET_FACES_DIR),
            "face_detection_model": cls.FACE_DETECTION_MODEL,
            "face_recognition_model": cls.FACE_RECOGNITION_MODEL,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "process_every_n_frames": cls.PROCESS_EVERY_N_FRAMES,
        }

    @classmethod
    def print_config(cls):
        """Print current configuration"""
        print("\n" + "=" * 80)
        print("PVS PIPELINE CONFIGURATION")
        print("=" * 80)

        config = cls.get_config_dict()
        for key, value in config.items():
            print(f"{key:.<40} {value}")

        print("=" * 80 + "\n")


# Create directories on import
Config.create_directories()
