"""Build release v1.1 of the Olympic Highlights annotations.

Changes with respect to v1.0 (OlympicHighlights_release/):
  * every highlight row carries the identity (surname) of the athlete;
  * corrected YouTube links for Javelin 2, Long Jump 3, Pole Vault 2 and Pole Vault 5;
  * videos.csv lists native frame rate, frame count, duration, competition, title and channel;
  * corrected identity label "Valdejch" -> "Vadlejch" (Javelin 3, 2 events).
The reproduction package (proposals, resemblance curves, reference embeddings,
held-out QD-DETR splits) is written next to the annotations as a separate folder.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import pvs_eval as E

OUT = E.PVS_DIR / "dataset" / "OlympicHighlights_v1.1"
EXTRA = Path("/mnt/Data/mrt/OlympicHighlights_v1.1_reproduction")
NAMES = {"highjump": ("High Jump", "high_jump"), "javelin": ("Javelin", "javelin"), "longjump": ("Long Jump", "long_jump"), "polevault": ("Pole Vault", "pole_vault")}
stats = json.load(open(E.OUT_DIR / "dataset_stats.json"))
meta = {r["video"]: r for r in stats["videos"]}

OUT.mkdir(parents=True, exist_ok=True)
rows = []
for v in E.VIDEOS:
    ev, idx = v.split("_video")
    folder, prefix = NAMES[ev]
    (OUT / folder).mkdir(exist_ok=True)
    fname = f"{prefix}_{idx}.csv"
    gt3 = list(csv.DictReader(open(E.GT3_DIR / E.EVENT_DIRS[ev] / "ground_truth" / f"video{idx}_processed.csv")))
    ids = [r for r in csv.DictReader(open(E.DATA_DIR / f"{v}_identity.csv")) if r["Event type"] == "Highlight"]
    hl = [r for r in gt3 if r["Event type"] == "Highlight"]
    assert len(hl) == len(ids) and all(a["First frame"] == b["First frame"] and a["Last frame"] == b["Last frame"] for a, b in zip(hl, ids)), v
    it = iter(ids)
    with open(OUT / folder / fname, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Event type", "First frame", "Last frame", "Num. frames", "Identity"])
        for r in gt3:
            ident = next(it)["Identity"].strip() if r["Event type"] == "Highlight" else ""
            w.writerow([r["Event type"], r["First frame"], r["Last frame"], r["Num. frames"], ident])
    m = meta[v]
    rows.append({"sport": folder, "ground_truth_file": fname, "url": m["url"], "competition": m["competition"], "sex": m["sex"], "venue": m["venue"],
                 "youtube_title": m["youtube_title"], "channel": m["channel"], "fps": f"{m['fps']:.6g}", "frames": m["frames"], "duration_s": f"{m['duration_s']:.2f}",
                 "athletes": m["athletes"], "highlights": m["events"]})
with open(OUT / "videos.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# reproduction package (large; distributed separately)
EXTRA.mkdir(parents=True, exist_ok=True)
for v in E.VIDEOS:
    d = EXTRA / v
    (d / "proposals").mkdir(parents=True, exist_ok=True)
    for src, folder in [("SportCLIP", E.PROPOSALS["SportCLIP"]), ("QD-DETR", E.PROPOSALS["QD-DETR"])]:
        shutil.copy(E.SUMMARIES_DIR / folder / v / "segments.json", d / "proposals" / f"{src}.json")
    for tdir in sorted(x for x in (E.RESEMBLANCE_DIR / v).iterdir() if x.is_dir()):
        (d / "resemblance" / tdir.name).mkdir(parents=True, exist_ok=True)
        for suffix in ["", "_updated"]:
            shutil.copy(tdir / f"{tdir.name}{suffix}.json", d / "resemblance" / tdir.name / f"{tdir.name}{suffix}.json")
        for emb in (E.TARGETS_DIR / v / tdir.name).glob("*.npy"):
            shutil.copy(emb, d / "resemblance" / tdir.name / emb.name)
shutil.copytree(E.PVS_DIR / "revision" / "qddetr_cv" / "folds", EXTRA / "qddetr_folds", dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.log"))
print("annotations:", OUT, "| reproduction package:", EXTRA)
