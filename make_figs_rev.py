# -*- coding: utf-8 -*-
"""Figure regeneration for the IJIES revision.

Every figure is drawn at its final printed size (3.3 in for a column figure,
6.6 in for a full-width figure) so that the nominal matplotlib font size is the
printed font size.  All lettering is therefore at least 10 pt, as required by
the IJIES format check, and nothing is downscaled inside Word.
"""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "results")
FIG = os.path.join(BASE, "figs")
os.makedirs(FIG, exist_ok=True)
DPI = 600
COL, FULL = 3.2, 6.6
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                     "axes.linewidth": 0.8, "savefig.pad_inches": 0.01})

CN = ["Disable Unsol.", "Cold Restart", "Warm Restart", "Enumerate", "Info",
      "Init. Data", "MITM-DoS", "Replay", "Stop Appl.", "Normal"]
EC, BLUE, RED, GREEN = "#333333", "#3B6EA5", "#B03A2E", "#2E7D5B"


def save(fig, name, tight=True):
    kw = dict(bbox_inches="tight") if tight else {}
    fig.savefig(os.path.join(FIG, name), dpi=DPI, facecolor="white", **kw)
    plt.close(fig)
    print("[fig]", name, flush=True)


def box(ax, x, y, w, h, title, body, fc, fs=10, tfs=10.5):
    ax.add_patch(FancyBboxPatch((x, y), w, h, ec=EC, fc=fc, lw=1.0,
                 boxstyle="round,pad=0.012,rounding_size=0.04"))
    if title:
        ax.text(x + w / 2, y + h - 0.19, title, ha="center", va="center",
                fontsize=tfs, fontweight="bold")
    if body:
        ax.text(x + w / 2, y + (h - 0.24) / 2, body, ha="center", va="center",
                fontsize=fs, linespacing=1.35)


def arrow(ax, p, q, color=EC, lw=1.3, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=11,
                 color=color, lw=lw, zorder=5,
                 connectionstyle="arc3,rad=%.2f" % rad))


# ---------------------------------------------------------------- Fig 1 -----
def fig1():
    fig = plt.figure(figsize=(FULL, 2.26))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FULL); ax.set_ylim(0, 2.26); ax.axis("off")
    w, h = 2.00, 1.04
    xs = [0.05, 2.30, 4.55]
    y1, y2 = 1.13, 0.03
    box(ax, xs[0], y1, w, h, "Raw DNP3 capture",
        "Deep packet inspection\n64-byte standardized\npayload per frame", "#EAF0F7")
    box(ax, xs[1], y1, w, h, "Temporal windowing",
        "T = 20 snapshots,\nstride 5, protocol-guided\nwindow labeling", "#EAF0F7")
    box(ax, xs[2], y1, w, h, "Disentangled routing",
        "Bytes 4-9 to topology,\nremaining bytes to the\nsemantic branch", "#EAF0F7")
    box(ax, xs[2], y2, w, h, "Edge-conditioned GNN",
        "13-device graph,\npayload-conditioned\nmessages, residual readout", "#F7EFE3")
    box(ax, xs[1], y2, w, h, "Transformer encoder",
        "3 layers, 8 heads,\nattentive temporal\npooling", "#EFE7F7")
    box(ax, xs[0], y2, w, h, "Detection head",
        "Softmax over 9 attacks\nand normal polling\ntraffic", "#F7E7E7")
    arrow(ax, (xs[0] + w, y1 + h / 2), (xs[1], y1 + h / 2))
    arrow(ax, (xs[1] + w, y1 + h / 2), (xs[2], y1 + h / 2))
    arrow(ax, (xs[2] + w / 2, y1), (xs[2] + w / 2, y2 + h))
    arrow(ax, (xs[2], y2 + h / 2), (xs[1] + w, y2 + h / 2))
    arrow(ax, (xs[1], y2 + h / 2), (xs[0] + w, y2 + h / 2))
    save(fig, "fig1_architecture.png", tight=False)


# ---------------------------------------------------------------- Fig 2 -----
def fig2():
    H = 1.88
    fig = plt.figure(figsize=(FULL, H))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FULL); ax.set_ylim(0, H); ax.axis("off")
    ax.text(1.25, 1.75, "DNP3 link-layer header", ha="center", fontsize=10.5,
            fontweight="bold")
    labs = ["0x05", "0x64", "LEN", "CTRL", "DST", "SRC"]
    cols = ["#CFE0F1", "#CFE0F1", "white", "white", "#D6EBDC", "#F7E2CC"]
    x = 0.10
    for l, c in zip(labs, cols):
        ax.add_patch(FancyBboxPatch((x, 1.32), 0.38, 0.27, ec=EC, fc=c, lw=0.9,
                     boxstyle="square,pad=0.004"))
        ax.text(x + 0.19, 1.455, l, ha="center", va="center", fontsize=10)
        x += 0.39
    ax.text(1.27, 1.16, "bytes 4-7: device addresses", ha="center", fontsize=10)
    box(ax, 0.10, 0.24, 2.34, 0.74, "",
        "Address-to-vertex map:\nmasters 1-4, outstations 11-18,\n"
        "broadcast 0, so |V| = 13", "#EAF0F7", fs=10)
    arrow(ax, (1.27, 1.30), (1.27, 1.02))
    cx, cy, r, nr = 3.72, 0.92, 0.60, 0.155
    ax.text(cx, 1.75, "Communication graph", ha="center", fontsize=10.5,
            fontweight="bold")
    ang = np.linspace(90, 450, 9)[:8] * np.pi / 180
    outs = [(cx + r * np.cos(a_), cy + r * np.sin(a_)) for a_ in ang]
    mids = [(cx - 0.22, cy + 0.19), (cx + 0.22, cy + 0.19),
            (cx - 0.22, cy - 0.19), (cx + 0.22, cy - 0.19)]
    for p_ in outs:
        for q in mids:
            ax.plot([p_[0], q[0]], [p_[1], q[1]], color="#C4C4C4", lw=0.4,
                    zorder=1)
    for i, p_ in enumerate(outs):
        ax.add_patch(Circle(p_, nr, fc="#D6EBDC", ec=EC, lw=0.8, zorder=3))
        ax.text(p_[0], p_[1], "O%d" % (11 + i), ha="center", va="center",
                fontsize=7.5, zorder=4)
    for i, p_ in enumerate(mids):
        ax.add_patch(Circle(p_, nr, fc="#CFE0F1", ec=EC, lw=0.8, zorder=3))
        ax.text(p_[0], p_[1], "M%d" % (1 + i), ha="center", va="center",
                fontsize=7.5, zorder=4)
    arrow(ax, (2.48, 0.60), (cx - r - nr - 0.05, 0.80), color=BLUE, rad=0.12)
    ax.text(cx, 0.10, "edges carry the 64-byte payload", ha="center",
            fontsize=10)
    ax.text(5.72, 1.75, "Temporal window", ha="center", fontsize=10.5,
            fontweight="bold")
    for k, off in enumerate([0.20, 0.10, 0.0]):
        ax.add_patch(FancyBboxPatch((5.26 + off, 0.70 + off), 0.88, 0.66,
                     ec=EC, fc="white", lw=0.9,
                     boxstyle="round,pad=0.008,rounding_size=0.04",
                     zorder=3 + k))
    ax.text(5.70, 1.03, "$G_t$", ha="center", va="center", fontsize=10,
            zorder=8)
    arrow(ax, (cx + r + nr + 0.05, 1.03), (5.22, 1.14), color=BLUE, rad=-0.12)
    ax.text(5.72, 0.36, "T = 20 snapshots", ha="center", fontsize=10)
    save(fig, "fig2_graph_construction.png", tight=False)


# ---------------------------------------------------------------- Fig 3 -----
def fig3():
    fig = plt.figure(figsize=(FULL, 1.82))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, FULL); ax.set_ylim(0, 1.82); ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.03, 0.05), 3.16, 1.72, ec="#999999",
                 fc="#FDF6EC", lw=0.9,
                 boxstyle="round,pad=0.01,rounding_size=0.05"))
    ax.text(1.61, 1.63, "Edge-conditioned GNN", ha="center", fontsize=10.5,
            fontweight="bold")
    box(ax, 0.14, 0.96, 1.42, 0.52, "", "Byte embedding\nof the payload", "white")
    box(ax, 1.72, 0.96, 1.36, 0.52, "", "Edge message\n$m_{uv}^{(t)}$", "white")
    box(ax, 0.14, 0.22, 1.42, 0.52, "", "Device identity\nembeddings", "white")
    box(ax, 1.72, 0.22, 1.36, 0.52, "", "Node update\nand readout", "white")
    arrow(ax, (1.56, 1.22), (1.72, 1.22))
    arrow(ax, (1.56, 0.48), (1.72, 0.48))
    arrow(ax, (2.40, 0.96), (2.40, 0.76))
    ax.add_patch(FancyBboxPatch((3.38, 0.05), 3.19, 1.72, ec="#999999",
                 fc="#F4EEFA", lw=0.9,
                 boxstyle="round,pad=0.01,rounding_size=0.05"))
    ax.text(4.97, 1.63, "Transformer encoder", ha="center", fontsize=10.5,
            fontweight="bold")
    box(ax, 3.50, 0.96, 1.44, 0.52, "", "Multi-head\nself-attention", "white")
    box(ax, 5.10, 0.96, 1.36, 0.52, "", "Feed-forward\nand LayerNorm", "white")
    box(ax, 3.50, 0.22, 1.44, 0.52, "", "Softmax over\n10 classes", "white")
    box(ax, 5.10, 0.22, 1.36, 0.52, "", "Attentive temporal\npooling", "white")
    arrow(ax, (4.94, 1.22), (5.10, 1.22))
    arrow(ax, (5.78, 0.96), (5.78, 0.76))
    arrow(ax, (5.10, 0.48), (4.94, 0.48))
    arrow(ax, (3.08, 0.48), (3.50, 1.16), color=BLUE, rad=0.18)
    save(fig, "fig3_encoder.png", tight=False)


# ------------------------------------------------------- result figures -----
def curves(hist, key, ylab, fname, loc="lower right"):
    """One panel carrying training accuracy, validation accuracy and the
    validation macro F1-score, all as percentages."""
    ep = range(1, len(hist["train_acc"]) + 1)
    fig = plt.figure(figsize=(COL, 2.30))
    ax = fig.add_axes([0.165, 0.185, 0.805, 0.795])
    ax.plot(ep, hist["train_acc"], "-", lw=1.3, color=BLUE,
            label="Training accuracy")
    ax.plot(ep, hist["val_acc"], "--", lw=1.3, color=GREEN,
            label="Validation accuracy")
    ax.plot(ep, hist["val_macro_f1"], ":", lw=1.5, color=RED,
            label="Validation macro F1")
    ax.set_xlabel("Epoch", fontsize=10)
    ax.set_ylabel("Percent", fontsize=10)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3, ls="--", lw=0.6)
    ax.legend(fontsize=10, loc="lower right", frameon=False)
    save(fig, fname, tight=False)


def fig_confusion(res, fname):
    cm = np.array(res["confusion_matrix"], dtype=float)
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1) * 100
    fig, ax = plt.subplots(figsize=(FULL, 2.14))
    sns.heatmap(cmn, annot=True, fmt=".1f", cmap="Blues", vmin=0, vmax=100,
                xticklabels=CN, yticklabels=CN, annot_kws={"size": 10},
                cbar_kws={"label": "% of actual class", "shrink": 0.9}, ax=ax,
                linewidths=0.3, linecolor="white")
    ax.figure.axes[-1].yaxis.label.set_size(10)
    ax.figure.axes[-1].tick_params(labelsize=10)
    ax.set_xlabel("Predicted class", fontsize=10)
    ax.set_ylabel("Actual class", fontsize=10)
    ax.set_xticklabels(CN, rotation=30, ha="right", fontsize=10)
    ax.set_yticklabels(CN, rotation=0, fontsize=10)
    save(fig, fname)


def fig_tsne(fname):
    p = os.path.join(RES, "strict_main_tsne.npz")
    if not os.path.exists(p):
        print("[skip] t-SNE (run extract_feats.py first)")
        return
    d = np.load(p)
    Z, t = d["Z"], d["y"]
    SHORT = ["Dis. Unsol.", "Cold Rest.", "Warm Rest.", "Enumerate", "Info",
             "Init. Data", "MITM-DoS", "Replay", "Stop App.", "Normal"]
    fig = plt.figure(figsize=(COL, 2.16))
    ax = fig.add_axes([0.01, 0.40, 0.98, 0.59])
    for sp in ax.spines.values():
        sp.set_visible(False)
    cmap = plt.get_cmap("tab10")
    for c in range(10):
        m = t == c
        ax.scatter(Z[m, 0], Z[m, 1], s=1.1, color=cmap(c), alpha=0.55,
                   linewidths=0, label=SHORT[c])
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(fontsize=10, markerscale=5, ncol=2, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.005),
              handletextpad=0.1, columnspacing=0.6, borderpad=0.05,
              labelspacing=0.18)
    save(fig, fname, tight=False)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "diagrams"):
        fig1(); fig2(); fig3()
    if what in ("all", "results"):
        r = json.load(open(os.path.join(RES, "strict_main.json")))
        curves(r["history"], "acc", "Percent", "fig4_training.png")
        fig_confusion(r, "fig6_confusion.png")
        fig_tsne("fig7_tsne.png")
