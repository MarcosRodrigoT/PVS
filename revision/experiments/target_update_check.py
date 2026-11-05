"""Which in-video face did the target update select, and is it plausibly the target? (reviewer comment 4)

For each annotated target and backbone, the selected exemplar (maximum similarity
over all detected faces of the video) is parsed from the update logs. As an
automatic plausibility check we test whether the exemplar's frame lies within
+-W seconds of a ground-truth highlight of the same athlete and the nearest
highlight belongs to that athlete. Contact sheets (reference crop | ArcFace
exemplar | TransFace exemplar) are written for visual verification.
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from PIL import Image, ImageDraw
import pvs_eval as E

NEAR = 20.0  # seconds

rows = []
for v in E.VIDEOS:
    fps = E.video_meta(v)["fps"]
    gt = E.load_gt(v)
    for tdir in sorted((E.TARGETS_DIR / v).iterdir()):
        if not tdir.is_dir():
            continue
        log = (tdir / f"{tdir.name}_log.txt").read_text()
        rec = {"video": v, "athlete": tdir.name}
        for bb, tag in [("arcface", "ArcFace"), ("transface", "TransFace")]:
            m = re.search(rf"{tag} Best Matches.*?BEST MATCH:.*?Face: (frame_(\d+)_face_\d+)\s+Similarity: ([0-9.]+)", log, re.S)
            frame = int(m.group(2))
            t = frame / fps
            dist = [(max(0.0, g["start"] - t, t - g["end"]), g["athlete"]) for g in gt]
            dmin, nearest = min(dist)
            own = min(d for d, a in dist if a == tdir.name)
            rec[bb] = {"face": m.group(1), "time": t, "similarity": float(m.group(3)), "nearest_athlete": nearest,
                       "dist_nearest": dmin, "dist_own": own, "plausible": bool(nearest == tdir.name and own <= NEAR)}
        rows.append(rec)

for bb in ["arcface", "transface"]:
    ok = np.mean([r[bb]["plausible"] for r in rows])
    sims = np.array([r[bb]["similarity"] for r in rows])
    print(f"{bb}: plausible exemplar for {100*ok:.1f}% of {len(rows)} targets; exemplar similarity median {np.median(sims):.3f} min {sims.min():.3f}")
agree = np.mean([r["arcface"]["face"].split("_face")[0] == r["transface"]["face"].split("_face")[0] or abs(r["arcface"]["time"] - r["transface"]["time"]) < 2 for r in rows])
print(f"ArcFace and TransFace exemplars within 2 s of each other: {100*agree:.1f}%")
E.save_json(rows, "target_update_check.json")

# contact sheets
sheet_dir = E.OUT_DIR / "target_update_sheets"
sheet_dir.mkdir(parents=True, exist_ok=True)
S, per_sheet = 120, 24
for k in range(0, len(rows), per_sheet):
    chunk = rows[k:k + per_sheet]
    img = Image.new("RGB", (3 * S + 260, len(chunk) * S), "white")
    d = ImageDraw.Draw(img)
    for i, r in enumerate(chunk):
        tdir = E.TARGETS_DIR / r["video"] / r["athlete"]
        for j, name in enumerate(["face_crop.jpg", "face_crop_arcface_updated.jpg", "face_crop_transface_updated.jpg"]):
            p = tdir / name
            if p.exists():
                img.paste(Image.open(p).convert("RGB").resize((S, S)), (j * S, i * S))
        d.text((3 * S + 5, i * S + 5), f"#{k+i} {r['video']}\n{r['athlete']}\nAF {r['arcface']['similarity']:.2f} {'ok' if r['arcface']['plausible'] else 'CHECK'}\nTF {r['transface']['similarity']:.2f} {'ok' if r['transface']['plausible'] else 'CHECK'}", fill="black")
    img.save(sheet_dir / f"sheet_{k//per_sheet:02d}.jpg", quality=85)
