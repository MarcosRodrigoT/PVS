# Plan: F-score vs Frame-Skip Plot for PVS Thesis Presentation

## Context

For the PhD thesis defense, we need a plot showing how PVS system quality (F-score) degrades as we skip more frames during the face-analysis stage. The x-axis represents the "skip interval" (0 = process every frame; 10 = process every 10th frame; up to 60), and the y-axis shows the F-score. At each x position, 16 scatter points are plotted — one per PVS configuration.

This directly visualises the speed/quality trade-off discussed in the thesis (Section 4.7, Table 4.24), where frame subsampling is proposed as a simple optimisation.

---

## Disk Space Assessment

**Current state:** disk 94% full, 117 GB free.

The new experiment requires **zero new large files** — it reads from existing cached data and produces only:
- `data/fscore_vs_frames_skipped_results.json` (~50 KB)
- `figures/fscore_vs_frames_skipped.png` + `.pdf` (~2 MB total)

**Verdict: No storage concerns whatsoever.**

The large directories (cropped_faces: 19 GB, cropped_faces_embeddings: 21 GB, video_summaries: 30 GB, resemblance_results: ~1 GB of JSON) are **not touched** and **do not grow**.

---

## Thesis vs. Implementation: Key Discrepancies to Be Aware Of

**1. Projected vs. measured runtimes (minor — already transparent in thesis)**
Table 4.23 contains *measured* per-unit timings (from `measurement_results.json`). Table 4.24 contains *projected* end-to-end runtimes under an optimised scenario (parallel branches + every-10th-frame). The thesis explicitly labels these as projections. No actual end-to-end timing was measured; the numbers are extrapolations. This is academically fine — just be prepared to say so in the defense.

**2. FPS assumption in timing projections (subtle mismatch)**
`measurements_report.md` projects timing for a hypothetical "10-minute video at **30 fps**." However, the actual videos in the project are **25 fps** (confirmed by `fps: 25.0` in the resemblance curve JSONs and `FPS = 25` constant in `plot_fscore_vs_iou.py`). The timing projections therefore *overstate* computational cost by a factor of 30/25 = 1.2×. For a 10-minute video at the actual 25 fps:
- Frames: 15,000 (not 18,000)
- Face detection: ~1,258 s (not ~1,510 s)
- This is ~83% of the projected cost

**3. Avg faces per frame is video-specific**
The 16.88 faces/frame figure is measured on `highjump_video1.mp4` only. Different sports (javelin, pole vault, etc.) will have different values. This affects the accuracy of end-to-end timing projections. For the plot experiment this doesn't matter, but worth noting for the defense Q&A.

---

## The 16 Configurations (Recap)

| Dimension | Options |
|-----------|---------|
| Video Summarizer | SportCLIP (SC), QD-DETR (QD) |
| Face Recognition | ArcFace (AF), TransFace (TF) |
| Assignment Method | Sequential (SEQ), Instant (INS) |
| Target Embedding | Original (ORI), Updated (UPD) |

Named as e.g. `SC-AF-SEQ-ORI`, `QD-TF-INS-UPD`, etc.

---

## Implementation Plan

### New script: `plot_fscore_vs_frames_skipped.py`

Model it closely on `plot_fscore_vs_iou.py` — reuse all its data-loading and assignment functions verbatim.

### What is reused (100% of intermediate results)

| Data | Location | Status |
|------|----------|--------|
| Video summary segments | `video_summaries/{SportCLIP,QD-DETR}/{video}/segments.json` | ✅ reuse as-is |
| Resemblance curves (original) | `resemblance_results/{video}/{athlete}/{athlete}.json` | ✅ subsample in memory |
| Resemblance curves (updated) | `resemblance_results/{video}/{athlete}/{athlete}_updated.json` | ✅ subsample in memory |
| Ground truth | `data/{video}_identity.csv` | ✅ reuse as-is |
| Baseline F-scores (skip=0) | `data/fscore_iou_sweep_results.json` at IoU=0.3 | ✅ can load directly |

**No re-running of face detection, embedding extraction, or resemblance computation is needed.**

### Algorithm

```
for skip in [0, 10, 20, 30, 40, 50, 60]:
    stride = max(1, skip)   # skip=0 → stride=1 (every frame)
                             # skip=10 → stride=10 (every 10th frame)

    for each of 16 configurations:
        per_sport_stats = {}

        for each of 4 sports:
            tp, fp, fn = 0, 0, 0

            for each of 5 videos in sport:
                resemblance_data = load_resemblance_data(video, use_updated)

                # --- Subsampling ---
                subsampled = subsample_resemblance(resemblance_data, stride, fps=25)

                segments  = load_segments(video, summarizer)
                gt        = load_ground_truth(video)

                assignments = assign_highlights_{method}(
                    subsampled, segments, threshold=0.6, smooth_window=200, model=fr_model
                )

                tp_, fp_, fn_ = evaluate_at_iou_threshold(
                    segments, assignments, gt, iou_threshold=0.3
                )
                tp += tp_; fp += fp_; fn += fn_

            per_sport_stats[sport] = compute_fscore(tp, fp, fn)

        config_fscore = mean(per_sport_stats.values())
        results[skip][config.name] = config_fscore
```

### Subsampling logic

The resemblance JSON stores `timestamps_seconds`, `arcface_scores`, `transface_scores` only for frames where faces were detected.

```python
def subsample_resemblance(resemblance_data: dict, stride: int, fps: float = 25.0) -> dict:
    """Keep only entries whose frame index is a multiple of stride."""
    if stride <= 1:
        return resemblance_data   # no-op for skip=0

    subsampled = {}
    for athlete, data in resemblance_data.items():
        timestamps = np.array(data["timestamps_seconds"])
        frame_indices = np.round(timestamps * fps).astype(int)
        mask = (frame_indices % stride) == 0

        subsampled[athlete] = {
            **data,                             # preserve metadata
            "timestamps_seconds": timestamps[mask].tolist(),
            "arcface_scores": np.array(data["arcface_scores"])[mask].tolist(),
            "transface_scores": np.array(data["transface_scores"])[mask].tolist(),
        }
    return subsampled
```

No interpolation is used — the assignment algorithms (`assign_highlights_sequential` and `assign_highlights_instant`) both use `np.searchsorted` on timestamps and handle sparse arrays naturally. Using the raw subsampled scores is more honest about what a real frame-skipping implementation would do.

### IoU threshold

Use **IoU = 0.3** throughout — this is the canonical threshold already used in the thesis and in `plot_fscore_vs_iou.py` ("This helps justify the 0.3 IoU threshold choice").

### Evaluation aggregation

Identical to `sweep_iou_for_config` in `plot_fscore_vs_iou.py`:
1. Accumulate TP/FP/FN over all 5 videos within each sport
2. Compute F-score per sport
3. Average the 4 sport-level F-scores → single number per (config, skip) pair

### Baseline sanity check

For skip=0, results should match `data/fscore_iou_sweep_results.json` at IoU=0.3. Load that file and compare; log a warning if any config differs by > 0.001.

---

## Plot Design

### X-axis
Values: 0, 10, 20, 30, 40, 50, 60
Label: **"Frames skipped (processing stride)"**
Tick labels: 0 (all frames), 10, 20, 30, 40, 50, 60
Optionally add a secondary axis label showing the corresponding speedup (1×, 10×, 20×, ...).

### Y-axis
Label: **"F-score (IoU = 0.3)"**
Range: [0, 1] or auto-scaled to data.

### 16 scatter points per x position
Reuse the same visual encoding from `plot_fscore_vs_iou.py`:
- **Color**: blue tones = ArcFace, red tones = TransFace; dark = Updated, light = Original
- **Marker shape**: circle = SportCLIP, square = QD-DETR
- Add slight horizontal jitter (±0.3) to avoid overlapping points

Optionally overlay thin trend lines (one per configuration, connecting their 7 dots) to make the degradation trajectory visible.

---

## Critical Files

| File | Role |
|------|------|
| `plot_fscore_vs_iou.py` | **Template** — copy data-loading, assignment, evaluation functions verbatim |
| `resemblance_results/` | Source for resemblance curves (subsampled in memory) |
| `video_summaries/` | Source for segment predictions |
| `data/*_identity.csv` | Ground truth |
| `data/fscore_iou_sweep_results.json` | Baseline F-scores for sanity check |
| `figures/` | Output directory |

New file to create: **`plot_fscore_vs_frames_skipped.py`**
New outputs: `data/fscore_vs_frames_skipped_results.json`, `figures/fscore_vs_frames_skipped.{png,pdf}`

---

## Verification

1. Run with skip=0 only; confirm F-scores match `fscore_iou_sweep_results.json` at IoU=0.3.
2. Run full sweep. Expect F-scores to be monotonically non-increasing (or nearly so) as skip grows.
3. Visually inspect the plot: 16 scatter points clearly visible at each x mark; no overlapping; legend readable.
4. Cross-check one configuration manually: load its resemblance JSON, subsample by stride=10, re-run assignment by hand for one video, confirm TP/FP/FN match.
