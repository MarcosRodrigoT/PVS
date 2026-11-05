"""Benchmark documentation for the revision (reviewer comment 5).

Produces per-broadcast metadata (source competition, sex, venue type, URL, fps,
frames, duration, number of athletes and events), identity statistics (athletes
per video, events per athlete, athletes recurring across broadcasts), the
structure of the uncertainty bands, and verifies that every YouTube URL resolves
to a video whose duration matches the annotated copy (|diff| < 1.5 s).

Four URLs of the first public release pointed to the wrong video (Javelin 2,
Long Jump 3, Pole Vault 2, Pole Vault 5); the corrected ones below were
identified by exact duration and content match.
"""
import collections
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

SOURCES = {
    "highjump_video1": ("zN-ANaOyveE", "World Championships Beijing 2015", "M", "outdoor"),
    "highjump_video2": ("RHvoWHs2MFI", "World Championships London 2017", "M", "outdoor"),
    "highjump_video3": ("UZhe_SVdXRc", "World Championships Oregon 2022", "M", "outdoor"),
    "highjump_video4": ("zC_qvKupyVs", "World Indoor Championships Glasgow 2024", "M", "indoor"),
    "highjump_video5": ("WtIWGLLpP9M", "World Indoor Championships Glasgow 2024", "W", "indoor"),
    "javelin_video1": ("Huw8C64ZGeA", "European Championships Berlin 2018", "M", "outdoor"),
    "javelin_video2": ("e1L93ySj38s", "World Championships London 2017", "M", "outdoor"),
    "javelin_video3": ("Hz1jQGYJqK4", "World Championships Doha 2019", "M", "outdoor"),
    "javelin_video4": ("oKXHkxbjMPg", "World Championships Doha 2019", "W", "outdoor"),
    "javelin_video5": ("aw2u7qP_a8A", "World Championships Oregon 2022", "W", "outdoor"),
    "longjump_video1": ("bZE4faCuw_I", "World Indoor Championships Birmingham 2018", "M", "indoor"),
    "longjump_video2": ("1i1crQyV3vM", "European Indoor Championships Apeldoorn 2025", "M", "indoor"),
    "longjump_video3": ("wNpA02SNgUg", "World Championships London 2017", "M", "outdoor"),
    "longjump_video4": ("zoYtaNBXjKc", "World Championships Doha 2019", "M", "outdoor"),
    "longjump_video5": ("5Zs1QpOiM0w", "World Championships Oregon 2022", "M", "outdoor"),
    "polevault_video1": ("dDPAwOmxF7Y", "European Championships Berlin 2018", "M", "outdoor"),
    "polevault_video2": ("s2nMjCKPE2M", "World Championships London 2017", "M", "outdoor"),
    "polevault_video3": ("GpdNcHdPEpg", "World Championships Oregon 2022", "M", "outdoor"),
    "polevault_video4": ("qeefjE74SXI", "World Indoor Championships Belgrade 2022", "M", "indoor"),
    "polevault_video5": ("Lbi-LLiAso8", "World Championships Oregon 2022", "W", "outdoor"),
}
# The surname key "barber" denotes two different athletes (javelin: K.-L. Barber; pole vault: S. Barber)
HOMONYMS = {"barber"}


def youtube_info(vid):
    cache = E.OUT_DIR / "youtube_metadata.json"
    data = json.load(open(cache)) if cache.exists() else {}
    if vid not in data:
        out = subprocess.run(["yt-dlp", "--skip-download", "--no-warnings", "--print", "%(title)s\t%(channel)s\t%(duration)s",
                              f"https://www.youtube.com/watch?v={vid}"], capture_output=True, text=True)
        title, channel, dur = out.stdout.strip().split("\t")
        data[vid] = {"title": title, "channel": channel, "duration": float(dur)}
        E.OUT_DIR.mkdir(parents=True, exist_ok=True)
        json.dump(data, open(cache, "w"), indent=2)
    return data[vid]


rows, events_per_athlete, athlete_videos = [], [], collections.defaultdict(set)
un_before, un_after = [], []
for v in E.VIDEOS:
    meta = E.video_meta(v)
    gt = E.load_gt(v)
    cnt = collections.Counter(g["athlete"] for g in gt)
    events_per_athlete += list(cnt.values())
    for a in cnt:
        athlete_videos[(a, E.event_of(v)) if a in HOMONYMS else a].add(v)
    tl = E.load_timeline(v)
    for i, (kind, s, e) in enumerate(tl):
        if kind == "HL":
            if i > 0 and tl[i - 1][0] == "UN":
                un_before.append(round((tl[i - 1][2] - tl[i - 1][1]) * meta["fps"]))
            if i + 1 < len(tl) and tl[i + 1][0] == "UN":
                un_after.append(round((tl[i + 1][2] - tl[i + 1][1]) * meta["fps"]))
    yid, comp, sex, venue = SOURCES[v]
    yt = youtube_info(yid)
    assert abs(yt["duration"] - meta["duration"]) < 1.5, (v, yt, meta)
    hl = [g["end"] - g["start"] for g in gt]
    rows.append({"video": v, "youtube_id": yid, "url": f"https://www.youtube.com/watch?v={yid}", "youtube_title": yt["title"],
                 "channel": yt["channel"], "competition": comp, "sex": sex, "venue": venue, "fps": meta["fps"],
                 "frames": meta["nb_frames"], "duration_s": meta["duration"], "athletes": len(cnt), "events": len(gt),
                 "hl_mean_s": float(np.mean(hl))})

recurring = {a: sorted(vs) for a, vs in athlete_videos.items() if len(vs) > 1}
summary = {
    "videos": len(rows), "competitions": len({r["competition"] for r in rows}),
    "years": sorted({int(r["competition"][-4:]) for r in rows}),
    "women_finals": sum(r["sex"] == "W" for r in rows), "indoor": sum(r["venue"] == "indoor" for r in rows),
    "total_hours": sum(r["duration_s"] for r in rows) / 3600, "events": sum(r["events"] for r in rows),
    "athlete_video_pairs": sum(r["athletes"] for r in rows), "unique_athletes": len(athlete_videos),
    "athletes_per_video": [min(r["athletes"] for r in rows), float(np.median([r["athletes"] for r in rows])), max(r["athletes"] for r in rows)],
    "events_per_athlete_video": [int(min(events_per_athlete)), float(np.median(events_per_athlete)), int(max(events_per_athlete))],
    "recurring_athletes": len(recurring), "max_videos_per_athlete": max(len(x) for x in recurring.values()),
    "fps_values": sorted({round(r["fps"], 3) for r in rows}),
    "un_band_frames_before": dict(collections.Counter(un_before)), "un_band_frames_after": dict(collections.Counter(un_after)),
}
print(json.dumps(summary, indent=1))
for r in rows:
    print(f"{r['video']:17s} {r['competition']:45s} {r['sex']} {r['venue']:7s} {r['fps']:6.3f} {r['duration_s']/60:5.1f} min  ath {r['athletes']:2d} ev {r['events']:2d}  {r['youtube_title'][:60]}")
E.save_json({"summary": summary, "videos": rows, "recurring_athletes": {str(k): v for k, v in recurring.items()}}, "dataset_stats.json")
