"""Target update: exemplar verification and behaviour for absent targets (reviewer comment 4).

For every video, the queried targets are the annotated athletes (present) plus
all athletes annotated in the other four broadcasts of the same event who do not
compete in this one (absent distractors, queried with their own reference image).

Per target and backbone we compute the original resemblance curve, the in-video
exemplar selected by the update (maximum-similarity face) and the updated curve.

Reported:
  * exemplar similarity of present vs absent targets (the update always returns an
    exemplar, also for absent targets) and ROC-AUC of this score as a presence test;
  * cross-model check of the exemplars of present targets: is the exemplar face,
    re-embedded with the OTHER backbone, closest to the same target's reference
    among all targets annotated in the video?;
  * end-to-end effect of absent queries with the default PVS settings: share of
    proposals attributed to absent targets, share of absent targets that receive
    a non-empty reel, and F on the present athletes (absent attributions are FPs);
  * a verified update (update only if exemplar similarity >= tau_v; otherwise
    keep the original reference) and a presence test (no reel if exemplar
    similarity < tau_v), with tau_v chosen by leave-one-event-out.
"""
import collections
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

CACHE = Path("/mnt/Data/mrt/PVS_revision_cache")
BACKBONES = ["arcface", "transface"]


def present_targets(video):
    return sorted(d.name for d in (E.TARGETS_DIR / video).iterdir() if d.is_dir())


def absent_targets(video):
    ev, present = E.event_of(video), set(present_targets(video))
    out = {}
    for other in E.VIDEOS:
        if E.event_of(other) == ev and other != video:
            for a in present_targets(other):
                if a not in present and a not in out:
                    out[a] = other
    return out  # athlete -> video whose reference image is used


def ref_embedding(video, athlete, bb):
    e = np.load(E.TARGETS_DIR / video / athlete / f"target_{bb}_embedding.npy").astype(np.float32).ravel()
    return e / np.linalg.norm(e)


def per_frame_max(sim, inv, n):
    out = np.full(n, -np.inf)
    np.maximum.at(out, inv, sim)
    return out


def process(video):
    fps = E.video_meta(video)["fps"]
    data = {bb: np.load(CACHE / f"{video}_{bb}.npz") for bb in BACKBONES}
    frames = data["arcface"]["frame"]
    assert np.array_equal(frames, data["transface"]["frame"]) and np.array_equal(data["arcface"]["face"], data["transface"]["face"])
    emb = {bb: data[bb]["emb"].astype(np.float32) for bb in BACKBONES}
    uf, inv = np.unique(frames, return_inverse=True)
    ts = uf / fps

    present = {a: video for a in present_targets(video)}
    absent = absent_targets(video)
    targets = {**present, **absent}
    refs = {bb: {a: ref_embedding(src, a, bb) for a, src in targets.items()} for bb in BACKBONES}

    curves = {"orig": collections.defaultdict(dict), "upd": collections.defaultdict(dict)}
    exemplar = {bb: {} for bb in BACKBONES}
    for bb in BACKBONES:
        for a in targets:
            s = (emb[bb] @ refs[bb][a] + 1) / 2
            k = int(np.argmax(s))
            exemplar[bb][a] = {"index": k, "similarity": float(s[k]), "present": a in present}
            curves["orig"][a][bb] = per_frame_max(s, inv, len(uf))
            curves["upd"][a][bb] = per_frame_max((emb[bb] @ emb[bb][k] + 1) / 2, inv, len(uf))

    # sanity: present-target curves equal the stored ones
    ts_json, stored = E.load_curves(video, True)
    a0 = next(iter(present))
    assert np.allclose(ts_json, ts) and np.abs(stored[a0]["transface"] - curves["upd"][a0]["transface"]).max() < 1e-3

    # cross-model verification of present exemplars
    xcheck = {}
    for bb in BACKBONES:
        other = "transface" if bb == "arcface" else "arcface"
        for a in present:
            k = exemplar[bb][a]["index"]
            sims = {b: float(emb[other][k] @ refs[other][b]) for b in present}
            same = {b: float(emb[bb][k] @ refs[bb][b]) for b in present}
            xcheck[f"{bb}|{a}"] = {"other_model_top1": max(sims, key=sims.get) == a, "same_model_top1": max(same, key=same.get) == a}
    return video, {"ts": ts, "curves": curves, "exemplar": exemplar, "present": sorted(present), "absent": sorted(absent), "xcheck": xcheck}


def evaluate(results, src, model, method, mode, tau_v=None, with_absent=True):
    """mode: 'orig' | 'upd' | 'verified' (update only if exemplar sim >= tau_v) | 'presence' (verified + drop targets below tau_v)."""
    pv, share_absent, nonempty_absent, n_absent = {}, [], 0, 0
    for v in E.VIDEOS:
        r = results[v]
        names = r["present"] + (r["absent"] if with_absent else [])
        curves = {}
        for a in names:
            ok = r["exemplar"][model][a]["similarity"] >= (tau_v if tau_v is not None else -1)
            if mode == "presence" and not ok:
                continue
            use_upd = mode == "upd" or (mode in ("verified", "presence") and ok)
            curves[a] = {model: r["curves"]["upd" if use_upd else "orig"][a][model]}
        segs, labels = E.run_pvs(v, src, model, method, False, curves_override=(r["ts"], curves)) if curves else (E.load_proposals(v, src), [None] * len(E.load_proposals(v, src)))
        pv[v] = E.video_scores(segs, labels, E.load_gt(v))
        absent = set(r["absent"]) if with_absent else set()
        share_absent.append(np.mean([lab in absent for lab in labels]) if labels else 0.0)
        assigned = set(labels)
        nonempty_absent += sum(a in assigned for a in absent)
        n_absent += len(absent)
    return {"F": float(np.mean([x[2] for x in pv.values()])), "P": float(np.mean([x[1] for x in pv.values()])),
            "per_video_F": {v: pv[v][2] for v in pv}, "share_to_absent": float(np.mean(share_absent)),
            "absent_with_reel": nonempty_absent / max(n_absent, 1)}


if __name__ == "__main__":
    from sklearn.metrics import roc_auc_score

    with Pool(4) as p:
        results = dict(p.map(process, E.VIDEOS))

    out = {"presence_auc": {}, "exemplar_sim": {}, "xcheck": {}, "assignment": {}}
    for bb in BACKBONES:
        sims = [(x["similarity"], x["present"]) for r in results.values() for x in r["exemplar"][bb].values()]
        y, s = np.array([p for _, p in sims]), np.array([v for v, _ in sims])
        out["presence_auc"][bb] = float(roc_auc_score(y, s))
        out["exemplar_sim"][bb] = {"present": np.percentile(s[y], [5, 25, 50, 75, 95]).tolist(), "absent": np.percentile(s[~y], [5, 25, 50, 75, 95]).tolist(),
                                   "n_present": int(y.sum()), "n_absent": int((~y).sum())}
        xc = [x for r in results.values() for k, x in r["xcheck"].items() if k.startswith(bb)]
        out["xcheck"][bb] = {"other_model_top1": float(np.mean([x["other_model_top1"] for x in xc])), "same_model_top1": float(np.mean([x["same_model_top1"] for x in xc]))}
        print(bb, "AUC", round(out["presence_auc"][bb], 3), "exemplar sim present", np.round(out["exemplar_sim"][bb]["present"], 3), "absent", np.round(out["exemplar_sim"][bb]["absent"], 3), out["xcheck"][bb])

    # LOEO choice of tau_v: maximise mean F (with absent queries) on three events, apply to the fourth
    taus = np.round(np.arange(0.60, 0.951, 0.01), 2)
    for src in ["GT", "QD-DETR", "SportCLIP"]:
        for model in BACKBONES:
            for method in ["SEQ", "INS"]:
                key = f"{src}|{model}|{method}"
                row = {}
                for mode in ["orig", "upd"]:
                    row[f"{mode}|present_only"] = evaluate(results, src, model, method, mode, with_absent=False)
                    row[f"{mode}|with_absent"] = evaluate(results, src, model, method, mode)
                for mode in ["verified", "presence"]:
                    grid = {t: evaluate(results, src, model, method, mode, t) for t in taus}
                    pv, share, chosen = {}, [], {}
                    for held in E.EVENTS:
                        train = [v for v in E.VIDEOS if E.event_of(v) != held]
                        t = max(taus, key=lambda t: np.mean([grid[t]["per_video_F"][v] for v in train]))
                        chosen[held] = float(t)
                        pv.update({v: grid[t]["per_video_F"][v] for v in E.VIDEOS if E.event_of(v) == held})
                    t_med = float(np.median(list(chosen.values())))
                    row[f"{mode}|with_absent|loeo"] = {"F": float(np.mean(list(pv.values()))), "chosen": chosen,
                                                       "share_to_absent@median_tau": grid[min(taus, key=lambda t: abs(t - t_med))]["share_to_absent"],
                                                       "absent_with_reel@median_tau": grid[min(taus, key=lambda t: abs(t - t_med))]["absent_with_reel"],
                                                       "per_video_F": pv}
                    row[f"{mode}|present_only|tau_med"] = evaluate(results, src, model, method, mode, t_med, with_absent=False)
                out["assignment"][key] = row
                print(key, " | ".join(f"{k}: F {100*v['F']:.1f}" + (f" abs {100*v['share_to_absent']:.1f}% reel {100*v['absent_with_reel']:.0f}%" if "share_to_absent" in v else
                                                                       (f" abs {100*v['share_to_absent@median_tau']:.1f}% reel {100*v['absent_with_reel@median_tau']:.0f}% tau {v['chosen']}" if "chosen" in v else ""))
                                       for k, v in row.items()), flush=True)
    E.save_json(out, "absent_targets.json")
