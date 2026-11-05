"""Generate the LaTeX table bodies of the revised manuscript from revision/outputs/*.json."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import pvs_eval as E

OUT = E.OUT_DIR / "latex"
OUT.mkdir(parents=True, exist_ok=True)
load = lambda n: json.load(open(E.OUT_DIR / n))
pct = lambda x, d=1: f"{100 * x:.{d}f}"
FR = {"arcface": "ArcFace", "transface": "TransFace"}
CFGS = [(m, meth, u) for m in ["arcface", "transface"] for meth in ["SEQ", "INS"] for u in ["orig", "upd"]]
TGT = {"orig": "Orig", "upd": "Upd"}


def write(name, lines):
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")


def per_event_tables():
    grid = load("main_grid.json")
    for e in E.EVENTS:
        rows = []
        for src, name in [("SportCLIP", "SportCLIP"), ("QD-DETR", "QD-DETR")]:
            for m, meth, u in CFGS:
                r, p, f = grid[src][f"{m}|{meth}|{u}"]["events"][e]
                rows.append(f"{name:9s} & {FR[m]:9s} & {meth} & {TGT[u]:4s} & {pct(r)} & {pct(p)} & {pct(f)}\\\\")
        write(f"e2e_{e}", rows)
    rows = []
    for i, (src, name) in enumerate([("SportCLIP", "SportCLIP"), ("QD-DETR", "QD-DETR")]):
        for j, (m, meth, u) in enumerate(CFGS):
            d = grid[src][f"{m}|{meth}|{u}"]
            lo, hi = d["overall_F_ci95"]
            rows.append(f"{8 * i + j + 1:2d} & {name:9s} & {FR[m]:9s} & {meth} & {TGT[u]:4s} & {pct(d['overall'][2])} & [{pct(lo)}, {pct(hi)}]\\\\")
    write("e2e_overall", rows)
    # in-sample QD-DETR (for the response letter)
    write("e2e_overall_insample", [f"{FR[m]} {meth} {TGT[u]} & {pct(grid['QD-DETR-insample'][f'{m}|{meth}|{u}']['overall'][2])} & {pct(grid['QD-DETR'][f'{m}|{meth}|{u}']['overall'][2])}\\\\" for m, meth, u in CFGS])


def gt_tables():
    grid = load("main_grid.json")["GT"]
    rows = []
    for m, meth, u in CFGS:
        r, p, f = grid[f"{m}|{meth}|{u}"]["overall"]
        rows.append(f"{FR[m]:9s} & {meth} & {TGT[u]:4s} & {pct(r)} & {pct(p)} & {pct(f)}\\\\")
    write("gt_all", rows)
    rows = []
    for e in E.EVENTS:
        for m, meth, u in CFGS:
            r, p, f = grid[f"{m}|{meth}|{u}"]["events"][e]
            rows.append(f"{E.EVENT_NAMES[e]:10s} & {FR[m]:9s} & {meth} & {TGT[u]:4s} & {pct(r)} & {pct(p)} & {pct(f)}\\\\")
        rows.append("\\midrule")
    write("gt_by_event", rows[:-1])


def frame_level():
    d = load("frame_level_vs.json")
    rows = []
    for src in ["SportCLIP", "QD-DETR"]:
        k = f"{src}|UN-excluded"
        for e in E.EVENTS + ["overall"]:
            r, p, f = d[k]["events"][e] if e != "overall" else d[k]["overall"]
            name = E.EVENT_NAMES.get(e, "All events")
            rows.append(f"{src:9s} & {name:10s} & {pct(r)} & {pct(p)} & {pct(f)}\\\\")
        rows.append("\\midrule")
    write("frame_level", rows[:-1])


def dataset_table():
    d = load("dataset_stats.json")
    short = {"World Championships": "WCH", "World Indoor Championships": "WICH", "European Championships": "ECH", "European Indoor Championships": "EICH"}
    rows = []
    prev = None
    for r in d["videos"]:
        comp = r["competition"]
        for k, v in sorted(short.items(), key=lambda kv: -len(kv[0])):
            comp = comp.replace(k, v)
        gt = E.load_gt(r["video"])
        tl = E.load_timeline(r["video"])
        dur = r["duration_s"]
        hl = sum(e - s for k, s, e in tl if k == "HL") / dur
        un = sum(e - s for k, s, e in tl if k == "UN") / dur
        ev = E.event_of(r["video"])
        if prev and ev != prev:
            rows.append("\\addlinespace")
        prev = ev
        name = f"{E.EVENT_NAMES[ev]} V{r['video'][-1]}"
        rows.append(f"\\textbf{{{name}}} & {comp} & {r['sex']} & {'I' if r['venue'] == 'indoor' else 'O'} & {r['fps']:.2f} & {dur / 60:.1f} & {r['athletes']} & {r['events']} & {np.mean([g['end'] - g['start'] for g in gt]):.1f} & {100 * hl:.1f} & {100 * un:.1f}\\\\")
    write("dataset", rows)


def baselines_table():
    sweep, base, b0 = load("param_sweep_summary.json"), load("baselines_loeo.json"), load("baselines.json")
    cols = [(m, s) for m in ["transface", "arcface"] for s in ["SportCLIP", "QD-DETR", "GT"]]
    rows = []
    rows.append("Random target & " + " & ".join(pct(b0[f"{s}|{m}|orig"]["random"]["overall"][2]) for m, s in cols) + "\\\\")
    for rule, name in [("inseg_max", "In-segment max"), ("inseg_vote", "In-segment vote"), ("adjacent", "Nearest confident face")]:
        rows.append(f"{name} & " + " & ".join(pct(base[f"{s}|{m}|orig"][rule]["loeo_F"]) for m, s in cols) + "\\\\")
    rows.append("\\midrule")
    for meth in ["SEQ", "INS"]:
        rows.append(f"PVS-{meth} & " + " & ".join(pct(sweep[f"{s}|{m}|orig|{meth}"]["loeo_F"]) for m, s in cols) + "\\\\")
    # PVS with its fixed default settings is reported in Tables 3 and 5, not here (equal tuning)
    write("baselines", rows)
    # updated-target variant for the text
    lines = []
    for m, s in cols:
        best_b = max(E.BASELINES, key=lambda r: base[f"{s}|{m}|upd"][r]["loeo_F"])
        best_p = max(["SEQ", "INS"], key=lambda me: sweep[f"{s}|{m}|upd|{me}"]["loeo_F"])
        lines.append(f"% upd {m} {s}: best baseline {best_b} {pct(base[f'{s}|{m}|upd'][best_b]['loeo_F'])} vs PVS-{best_p} {pct(sweep[f'{s}|{m}|upd|{best_p}']['loeo_F'])}")
    write("baselines_upd_note", lines)


def loeo_table():
    sweep = load("param_sweep_summary.json")
    rows = []
    for m, meth, u in CFGS:
        cells = []
        for s in ["SportCLIP", "QD-DETR", "GT"]:
            d = sweep[f"{s}|{m}|{u}|{meth}"]
            cells += [pct(d["default_F"]), pct(d["loeo_F"]), pct(d["oracle_F"])]
        rows.append(f"{FR[m]:9s} & {meth} & {TGT[u]:4s} & " + " & ".join(cells) + "\\\\")
    write("loeo", rows)


def paired_table():
    st = load("paired_stats.json")
    items = [
        ("TransFace $-$ ArcFace", "SportCLIP SEQ upd: TransFace - ArcFace", "QD-DETR SEQ upd: TransFace - ArcFace", "GT SEQ upd: TransFace - ArcFace"),
        ("Upd $-$ Orig (TransFace, SEQ)", "SportCLIP transface SEQ: Upd - Orig", "QD-DETR transface SEQ: Upd - Orig", "GT transface SEQ: Upd - Orig"),
        ("Upd $-$ Orig (TransFace, INS)", "SportCLIP transface INS: Upd - Orig", "QD-DETR transface INS: Upd - Orig", "GT transface INS: Upd - Orig"),
        ("Upd $-$ Orig (ArcFace, SEQ)", "SportCLIP arcface SEQ: Upd - Orig", "QD-DETR arcface SEQ: Upd - Orig", "GT arcface SEQ: Upd - Orig"),
        ("SEQ $-$ INS (TransFace, Upd)", "SportCLIP transface upd: SEQ - INS", "QD-DETR transface upd: SEQ - INS", "GT transface upd: SEQ - INS"),
        ("SEQ $-$ INS (TransFace, Orig)", "SportCLIP transface orig: SEQ - INS", "QD-DETR transface orig: SEQ - INS", "GT transface orig: SEQ - INS"),
    ]
    fmt = lambda r: f"{100 * r['mean']:+.1f} [{100 * r['ci'][0]:+.1f}, {100 * r['ci'][1]:+.1f}] & {r['p_wilcoxon']:.3f} & {r['wins']}/{r['losses']}".replace("+", "$+$").replace("-", "$-$")
    rows = [f"{name} & " + " & ".join(fmt(st[k]) for k in keys) + "\\\\" for name, *keys in items]
    rows.append("\\midrule")
    rows.append("QD-DETR $-$ SportCLIP (TransFace, SEQ, Upd) & \\multicolumn{3}{c}{--} & " + fmt(st["transface|SEQ|upd: QD-DETR - SportCLIP"]) + " & \\multicolumn{3}{c}{--}\\\\")
    write("paired", [r.replace("0.000", "<0.001") for r in rows])


def reel_table():
    rq = load("reel_quality.json")
    rows = []
    for key in ["SportCLIP|transface|SEQ|upd", "SportCLIP|transface|INS|upd", "SportCLIP|arcface|SEQ|orig",
                "QD-DETR|transface|SEQ|upd", "QD-DETR|transface|INS|upd", "QD-DETR|transface|SEQ|orig", "QD-DETR|arcface|SEQ|orig"]:
        d = rq[key]
        s, m, meth, u = key.split("|")
        sw = d["iou_sweep"]
        rows.append(f"{s} & {FR[m]} & {meth} & {TGT[u]} & {pct(sw['0.3'])} & {pct(sw['0.5'])} & {pct(sw['0.7'])} & {d['start_err_median']:.1f}/{d['end_err_median']:.1f} & "
                    f"{pct(d['reel']['purity'])} & {pct(d['reel']['contamination'])} & {pct(d['reel']['background'])} & {pct(d['reel']['coverage'])} & {pct(d['reel']['redundancy'])} & {pct(d['jitter']['2.0'][0])}\\\\")
    write("reel", rows)


def update_table():
    ab = load("absent_targets.json")
    rows = []
    for src in ["SportCLIP", "QD-DETR", "GT"]:
        a = ab["assignment"][f"{src}|transface|SEQ"]
        for label, k_present, k_absent in [("Original reference", "orig|present_only", "orig|with_absent"), ("Update (unconditional)", "upd|present_only", "upd|with_absent"),
                                           ("Verified update", "verified|present_only|tau_med", "verified|with_absent|loeo"), ("Verified update + presence test", "presence|present_only|tau_med", "presence|with_absent|loeo")]:
            p, w = a[k_present], a[k_absent]
            share = w.get("share_to_absent", w.get("share_to_absent@median_tau"))
            reel = w.get("absent_with_reel", w.get("absent_with_reel@median_tau"))
            rows.append(f"{src if label.startswith('Original') else '':9s} & {label} & {pct(p['F'])} & {pct(w['F'])} & {pct(share)} & {pct(reel)}\\\\")
        rows.append("\\midrule")
    write("update", rows[:-1])


if __name__ == "__main__":
    per_event_tables()
    gt_tables()
    frame_level()
    dataset_table()
    baselines_table()
    loeo_table()
    paired_table()
    reel_table()
    update_table()
    for f in sorted(OUT.glob("*.tex")):
        print(f"--- {f.name}\n{f.read_text()}")
