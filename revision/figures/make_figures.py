"""Figures of the revised manuscript (written to the paper's Images/ folder)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pvs_eval as E

OUT = Path("/home/mrt/Projects/PVS-paper/paper-IVC/Images")
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
                     "axes.axisbelow": True, "legend.frameon": False, "pdf.fonttype": 42})
load = lambda n: json.load(open(E.OUT_DIR / n))


def fig_e2e():
    grid = load("main_grid.json")
    cfgs = [(m, meth, u) for m in ["arcface", "transface"] for meth in ["SEQ", "INS"] for u in ["orig", "upd"]]
    labels = [f"{'ArcFace' if m == 'arcface' else 'TransFace'}\n{meth} / {'Upd' if u == 'upd' else 'Orig'}" for m, meth, u in cfgs]
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    x = np.arange(len(cfgs))
    for k, (src, col, name) in enumerate([("SportCLIP", BLUE, "SportCLIP (zero-shot)"), ("QD-DETR", ORANGE, "QD-DETR (held-out folds)"), ("GT", AQUA, "Ground-truth segments")]):
        vals = [100 * grid[src][f"{m}|{meth}|{u}"]["overall"][2] for m, meth, u in cfgs]
        ci = np.array([[100 * c for c in grid[src][f"{m}|{meth}|{u}"]["overall_F_ci95"]] for m, meth, u in cfgs])
        pos = x + (k - 1) * 0.27
        ax.bar(pos, vals, width=0.25, color=col, label=name, zorder=2)
        ax.errorbar(pos, vals, yerr=[np.array(vals) - ci[:, 0], ci[:, 1] - np.array(vals)], fmt="none", ecolor=INK, elinewidth=0.7, capsize=1.5, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, 2, f"{v:.1f}", rotation=90, ha="center", va="bottom", fontsize=5.5, color="white")
    ax.set_xticks(x, labels, fontsize=6.5)
    ax.set_ylabel("Overall F-score (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, loc="upper left", fontsize=7, bbox_to_anchor=(0, 1.13))
    fig.tight_layout()
    fig.savefig(OUT / "PVS - e2e results.pdf")
    plt.close(fig)


def fig_sensitivity():
    grid = load("param_grid.json")
    d = E.DEFAULTS
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.2), sharey=True)
    series = [("transface", "SEQ", BLUE, "-", "TransFace, SEQ"), ("transface", "INS", BLUE, "--", "TransFace, INS"),
              ("arcface", "SEQ", ORANGE, "-", "ArcFace, SEQ"), ("arcface", "INS", ORANGE, "--", "ArcFace, INS")]
    for model, method, col, ls, name in series:
        g = grid[f"QD-DETR|{model}|orig|{method}"]["grid"]
        off = d["lookback"] if method == "SEQ" else d["expansion"]
        F = lambda t, w, o: 100 * np.mean(list(g[f"{t}|{w}|{o}"].values()))
        taus = sorted({float(k.split("|")[0]) for k in g})
        axes[0].plot(taus, [F(t, d["window"], off) for t in taus], color=col, ls=ls, lw=1.4, label=name)
        ws = sorted({int(k.split("|")[1]) for k in g})
        axes[1].plot(range(len(ws)), [F(d["tau"], w, off) for w in ws], color=col, ls=ls, lw=1.4, marker="o", ms=3)
        offs = sorted({float(k.split("|")[2]) for k in g})
        axes[2].plot(offs, [F(d["tau"], d["window"], o) for o in offs], color=col, ls=ls, lw=1.4, marker="o", ms=3)
    axes[0].axvline(d["tau"], color=MUTED, lw=0.8, ls=":")
    axes[1].axvline(4, color=MUTED, lw=0.8, ls=":")  # index of the default window (200)
    axes[2].axvline(d["lookback"], color=MUTED, lw=0.8, ls=":")
    axes[2].axvline(d["expansion"], color=MUTED, lw=0.8, ls=(0, (1, 3)))
    axes[0].set_xlabel(r"(a) threshold $\tau_{\mathrm{sim}}$")
    axes[1].set_xlabel("(b) smoothing window $|W|$ (samples)")
    axes[1].set_xticks(range(6), ["1", "25", "50", "100", "200", "400"])
    axes[2].set_xlabel(r"(c) $\Delta L_{\mathrm{look}}$ (SEQ), $\Delta L_{\mathrm{exp}}$ (INS) [s]")
    axes[0].set_ylabel("Overall F-score (%)")
    axes[0].set_ylim(20, 80)
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="upper center", fontsize=7, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "PVS - sensitivity.pdf")
    plt.close(fig)


def fig_robustness():
    reel = load("reel_quality.json")
    sparse = load("sparse_sampling.json")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for src, col, mk in [("SportCLIP", BLUE, "o"), ("QD-DETR", ORANGE, "s")]:
        for model, upd, ls in [("transface", "upd", "-"), ("arcface", "orig", "--")]:
            sw = reel[f"{src}|{model}|SEQ|{upd}"]["iou_sweep"]
            th = sorted(float(t) for t in sw)
            axes[0].plot(th, [100 * sw[str(t)] for t in th], color=col, ls=ls, lw=1.4, marker=mk, ms=3, markevery=2,
                         label=f"{src}, {'TransFace Upd' if model == 'transface' else 'ArcFace Orig'}")
            ks = [1, 2, 5, 10, 25, 50]
            axes[1].plot(ks, [100 * sparse[f"{k}|{src}|{model}|SEQ|{upd}"]["F"] for k in ks], color=col, ls=ls, lw=1.4, marker=mk, ms=3)
    axes[0].axvline(0.3, color=MUTED, lw=0.8, ls=":")
    axes[0].set_xlabel("Temporal IoU threshold")
    axes[0].set_ylabel("Overall F-score (%) — SEQ")
    axes[0].set_ylim(0, 85)
    axes[1].set_xscale("log")
    axes[1].set_xticks([1, 2, 5, 10, 25, 50], ["1", "2", "5", "10", "25", "50"])
    axes[1].set_xlabel("Face analysis every $k/25$ s ($k=1$: every frame)")
    axes[1].set_ylim(0, 85)
    axes[0].set_title("(a) Matching threshold", fontsize=8, color=INK, loc="left")
    axes[1].set_title("(b) Sparse face analysis", fontsize=8, color=INK, loc="left")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=4, loc="upper center", fontsize=6.5, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(OUT / "PVS - robustness.pdf")
    plt.close(fig)


if __name__ == "__main__":
    fig_e2e()
    fig_sensitivity()
    fig_robustness()
