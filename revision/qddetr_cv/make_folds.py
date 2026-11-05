"""
Create held-out training splits for QD-DETR on Olympic Highlights.

The QD-DETR checkpoint used in the original submission was trained on the evaluation
videos. For the revision we train five fold-specific models instead. Fold k holds
out the (k+1)-th broadcast of each event (high jump, javelin, long jump, pole vault),
and each fold model is trained on the 16 remaining benchmark broadcasts, so every
evaluation video is processed by a model that never saw it. A split with all 20
benchmark broadcasts is also written for the in-sample reference model.

Usage (any Python 3 interpreter):
    python revision/qddetr_cv/make_folds.py
"""

import json
from pathlib import Path

QDDETR_DIR = Path("/home/mrt/Projects/QD-DETR")
SPECIFIC_ANN = QDDETR_DIR / "data/OlympicHighlights_annotations/olympic_highlights_train_specific.jsonl"
OUT_DIR = Path(__file__).parent / "folds"

EVAL_EVENTS = ["highjump", "javelin", "longjump", "polevault"]
N_FOLDS = 5


def heldout_videos(fold_idx):
    return [f"{event}_video{fold_idx + 1}" for event in EVAL_EVENTS]


def main():
    entries = [json.loads(line) for line in SPECIFIC_ANN.read_text().splitlines() if line.strip()]
    entries = [e for e in entries if e["vid"].split("_")[0] in EVAL_EVENTS]  # the 20 benchmark broadcasts
    assert len(entries) == 20
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "all_benchmark_train_specific.jsonl", "w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    split_summary = {}
    for fold_idx in range(N_FOLDS):
        test_vids = set(heldout_videos(fold_idx))
        train = [e for e in entries if e["vid"] not in test_vids]
        assert len(train) == len(entries) - len(test_vids)

        out_path = OUT_DIR / f"fold{fold_idx}_train_specific.jsonl"
        with open(out_path, "w") as f:
            for e in train:
                f.write(json.dumps(e) + "\n")

        split_summary[f"fold{fold_idx}"] = {
            "train": sorted(e["vid"] for e in train),
            "test": sorted(test_vids),
        }
        print(f"fold {fold_idx}: {len(train)} training broadcasts, held out {sorted(test_vids)}")

    with open(OUT_DIR / "splits.json", "w") as f:
        json.dump(split_summary, f, indent=2)


if __name__ == "__main__":
    main()
