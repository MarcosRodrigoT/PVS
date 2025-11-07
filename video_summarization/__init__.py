"""
Video Summarization Module

This module handles video summarization using multiple frameworks:
- SportCLIP: Sports highlight detection using text-video matching
- QD-DETR: Query-based moment retrieval and highlight detection
"""

from .sportclip_summarizer import generate_highlights as generate_highlights_sportclip
from .qddetr_summarizer import generate_highlights as generate_highlights_qddetr

# For backward compatibility, default to SportCLIP
generate_highlights = generate_highlights_sportclip

__all__ = [
    "generate_highlights",
    "generate_highlights_sportclip",
    "generate_highlights_qddetr",
]
