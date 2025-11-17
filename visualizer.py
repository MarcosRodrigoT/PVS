"""
Visualization Module for Face Recognition Results
Creates resemblance curves for each video clip
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


class ResemblanceVisualizer:
    """Visualizer for face recognition resemblance curves"""

    def __init__(self, recognition_results_path: str):
        """
        Initialize the visualizer

        Args:
            recognition_results_path: Path to face recognition results JSON
        """
        self.recognition_results_path = recognition_results_path
        self.results = self._load_results()

    def _load_results(self) -> Dict:
        """Load recognition results from JSON file"""
        with open(self.recognition_results_path, "r") as f:
            return json.load(f)

    def compute_frame_resemblance(self, clip_results: List[Dict]) -> Dict[int, float]:
        """
        Compute maximum resemblance per frame

        Args:
            clip_results: List of recognition results for a clip

        Returns:
            Dictionary mapping frame numbers to maximum similarity scores
        """
        frame_resemblance = defaultdict(float)

        for result in clip_results:
            frame_num = result["frame_number"]
            similarity = result["similarity"]

            # Keep the maximum similarity for each frame
            frame_resemblance[frame_num] = max(frame_resemblance[frame_num], similarity)

        return dict(frame_resemblance)

    def plot_resemblance_curves(self, output_dir: str = "results/visualizations", show_plot: bool = False):
        """
        Create resemblance curves for all clips and targets

        Args:
            output_dir: Directory to save the plots
            show_plot: Whether to display plots interactively
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        if not self.results:
            print("No recognition results to visualize")
            return

        # Get all targets
        all_targets = set()
        for clip_results in self.results.values():
            all_targets.update(clip_results.keys())

        # Create a plot for each target
        for target_name in all_targets:
            self._plot_target_resemblance(target_name, output_path, show_plot)

        print(f"\nVisualization complete. Plots saved to {output_dir}/")

    def _plot_target_resemblance(self, target_name: str, output_path: Path, show_plot: bool = False):
        """
        Create resemblance curve for a specific target across all clips

        Args:
            target_name: Name of the target person
            output_path: Directory to save the plot
            show_plot: Whether to display the plot interactively
        """
        num_clips = len(self.results)

        if num_clips == 0:
            return

        # Determine subplot layout
        cols = min(3, num_clips)
        rows = (num_clips + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows))
        fig.suptitle(f"Resemblance Curves - Target: {target_name}", fontsize=16, fontweight="bold")

        # Handle single subplot case
        if num_clips == 1:
            axes = np.array([axes])

        axes = axes.flatten() if num_clips > 1 else axes

        clip_ids = sorted(self.results.keys())

        for idx, clip_id in enumerate(clip_ids):
            ax = axes[idx] if num_clips > 1 else axes[0]

            # Get results for this clip and target
            if target_name in self.results[clip_id]:
                clip_results = self.results[clip_id][target_name]
                frame_resemblance = self.compute_frame_resemblance(clip_results)

                if frame_resemblance:
                    frames = sorted(frame_resemblance.keys())
                    similarities = [frame_resemblance[f] for f in frames]

                    # Plot the curve (use horizontal line for single points, line for multiple)
                    if len(frames) == 1:
                        # Single point - draw horizontal line from frame 0 to detected frame
                        frame_num = frames[0]
                        ax.axhline(y=similarities[0], color="b", linewidth=2, alpha=0.7)
                        ax.fill_between([0, frame_num], similarities[0], alpha=0.3)
                        # Set x-axis limits to show from 0 to the detected frame
                        ax.set_xlim([0, frame_num])
                    else:
                        # Multiple points - use line plot
                        ax.plot(frames, similarities, "b-", linewidth=1.5, alpha=0.7)
                        ax.fill_between(frames, similarities, alpha=0.3)
                        # Set x-axis limits based on actual frame range
                        ax.set_xlim([min(frames), max(frames)])

                    # Add threshold line
                    ax.axhline(y=0.6, color="r", linestyle="--", linewidth=1, label="Match Threshold (0.6)", alpha=0.7)

                    # Styling
                    ax.set_xlabel("Frame Number", fontsize=10)
                    ax.set_ylabel("Resemblance Score", fontsize=10)
                    ax.set_title(f"{clip_id}", fontsize=11, fontweight="bold")
                    ax.set_ylim([0, 1])
                    ax.grid(True, alpha=0.3)
                    ax.legend(fontsize=8)

                    # Add statistics
                    avg_sim = np.mean(similarities)
                    max_sim = np.max(similarities)
                    matches = sum(1 for s in similarities if s >= 0.6)
                    stats_text = f"Avg: {avg_sim:.3f}\nMax: {max_sim:.3f}\nMatches: {matches}"
                    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=8, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
                else:
                    ax.text(0.5, 0.5, "No faces detected", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                    ax.set_title(f"{clip_id}", fontsize=11, fontweight="bold")
            else:
                ax.text(0.5, 0.5, "No data for target", ha="center", va="center", transform=ax.transAxes, fontsize=10)
                ax.set_title(f"{clip_id}", fontsize=11, fontweight="bold")

        # Hide unused subplots
        for idx in range(num_clips, len(axes)):
            axes[idx].set_visible(False)

        # Adjust layout to make room for suptitle at the top
        plt.tight_layout(rect=[0, 0, 1, 0.97])

        # Save the plot
        # If multiple targets exist, include target name in filename
        all_targets = list(set(target for clip in self.results.values() for target in clip.keys()))
        if len(all_targets) > 1:
            plot_filename = output_path / f"resemblance_results_{target_name}.png"
        else:
            plot_filename = output_path / "resemblance_results.png"

        plt.savefig(plot_filename, dpi=150, bbox_inches="tight")
        print(f"Saved plot: {plot_filename}")

        if show_plot:
            plt.show()
        else:
            plt.close()

    def generate_summary_report(self, output_path: str = "results/summary_report.txt"):
        """
        Generate a text summary report of recognition results

        Args:
            output_path: Path to save the summary report
        """
        with open(output_path, "w") as f:
            f.write("=" * 80 + "\n")
            f.write("PERSONALIZED VIDEO SUMMARIZATION - RECOGNITION SUMMARY\n")
            f.write("=" * 80 + "\n\n")

            # Get all targets
            all_targets = set()
            for clip_results in self.results.values():
                all_targets.update(clip_results.keys())

            for target_name in all_targets:
                f.write(f"\nTarget Person: {target_name}\n")
                f.write("-" * 80 + "\n")

                for clip_id in sorted(self.results.keys()):
                    if target_name in self.results[clip_id]:
                        clip_results = self.results[clip_id][target_name]
                        frame_resemblance = self.compute_frame_resemblance(clip_results)

                        if frame_resemblance:
                            similarities = list(frame_resemblance.values())
                            avg_sim = np.mean(similarities)
                            max_sim = np.max(similarities)
                            min_sim = np.min(similarities)
                            matches = sum(1 for s in similarities if s >= 0.6)
                            match_rate = (matches / len(similarities)) * 100

                            f.write(f"\n  {clip_id}:\n")
                            f.write(f"    Frames analyzed: {len(frame_resemblance)}\n")
                            f.write(f"    Average resemblance: {avg_sim:.4f}\n")
                            f.write(f"    Max resemblance: {max_sim:.4f}\n")
                            f.write(f"    Min resemblance: {min_sim:.4f}\n")
                            f.write(f"    Matches (>0.6): {matches} ({match_rate:.1f}%)\n")
                        else:
                            f.write(f"\n  {clip_id}: No faces detected\n")

                f.write("\n")

        print(f"Summary report saved to {output_path}")


def visualize_results(recognition_results_path: str = "results/face_recognition_results.json", output_dir: str = "results", show_plots: bool = False):
    """
    Generate visualizations and summary report

    Args:
        recognition_results_path: Path to face recognition results JSON
        output_dir: Directory to save visualizations
        show_plots: Whether to display plots interactively
    """
    visualizer = ResemblanceVisualizer(recognition_results_path)
    visualizer.plot_resemblance_curves(output_dir=output_dir, show_plot=show_plots)
    visualizer.generate_summary_report()
