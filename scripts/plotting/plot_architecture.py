#!/usr/bin/env python3
"""Four ways to compare a query with documents, and what each one costs.

The progression is the paper's: how many documents the model is allowed to see
jointly with the query, and how often it has to look.

  Bi-encoder            never jointly. Documents encoded once, offline.
  Cross-encoder         one document at a time, k times per query.
  LLM listwise          k documents at once, one pass per query.
  LLM corpus-in-context all N documents at once, one pass per query.

Cost follows the highlighted (query-document) attention: it is the only work
that cannot be amortised across queries. Reranking costs scale with the
shortlist k; corpus-in-context scales with the corpus N.

Output: visualizations/architecture_schematic.{png,pdf}
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Patch
from matplotlib.colors import ListedColormap

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))
from pub_style import save_fig  # noqa: E402

INK, GRAY, RULE = "#1A1A1A", "#5C5C5C", "#C9C9C9"
FILL, SELF, NONE_C = "#F4F4F4", "#DCDCDC", "#FFFFFF"
ACCENT = "#B03A2E"

FS_T, FS_S, FS_B, FS_R = 19, 15.5, 16.5, 14.5
Y_IN, Y_MID, Y_OUT, H = 0.72, 0.38, 0.04, 0.22


def box(ax, x, y, w, h, t, *, fill=FILL, fs=FS_B):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=INK, lw=1.0,
                           zorder=3))
    ax.text(x + w / 2, y + h / 2, t, ha="center", va="center", fontsize=fs,
            color=INK, zorder=4)


def arr(ax, p0, p1, color=GRAY, lw=1.0, ms=9):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=ms,
                                 color=color, lw=lw, shrinkA=1, shrinkB=1,
                                 zorder=5))


def arch_bi(ax):
    for cx, lab in ((0.25, "Query"), (0.75, "Doc")):
        box(ax, cx - 0.22, Y_IN, 0.44, H, lab)
        arr(ax, (cx, Y_IN), (cx, Y_MID + H))
        box(ax, cx - 0.22, Y_MID, 0.44, H, "Encoder")
        arr(ax, (cx, Y_MID), (cx, Y_OUT + H))
        box(ax, cx - 0.18, Y_OUT, 0.36, H, "vec", fill="white")
    ax.plot([0.43, 0.57], [Y_OUT + H / 2] * 2, color=GRAY, lw=1.0, zorder=2)
    ax.text(0.5, Y_OUT + H / 2, "cos", ha="center", va="center", fontsize=FS_B,
            color=INK, zorder=6,
            bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none"))
    ax.plot([0.5, 0.5], [Y_MID, Y_IN + H], color=RULE, lw=1.0, ls=(0, (4, 4)),
            zorder=1)


def arch_joint(ax, a, b, c):
    box(ax, 0.01, Y_IN, 0.98, H, a)
    arr(ax, (0.5, Y_IN), (0.5, Y_MID + H))
    box(ax, 0.16, Y_MID, 0.68, H, b)
    arr(ax, (0.5, Y_MID), (0.5, Y_OUT + H))
    box(ax, 0.26, Y_OUT, 0.48, H, c, fill="white")


def build_mask(kind, nq, nd, read, dlabel="docs"):
    """0 no attention, 1 self-attention, 2 query-document attention.

    Every panel spans the same candidate set and uses the same cell size, so the
    red region is comparable across panels. `read` is how many of those document
    positions enter a single forward pass: 1 for a cross-encoder, k for listwise,
    all N for corpus-in-context. Drawing each panel at equal size with unequal
    token counts made the red AREA shrink as documents grew, which inverted the
    figure's own argument.
    """
    n = nq + nd
    a = np.zeros((n, n))
    a[:nq, :nq] = 1                      # the query always attends to itself
    if kind == "bi":
        a[nq:, nq:] = 1                  # each document encoded on its own
        return a, nq, [((nq - 1) / 2, "Q"), (nq + (nd - 1) / 2, dlabel)]
    e = nq + read
    a[nq:e, nq:e] = 1                    # documents in context attend to each other
    a[:nq, nq:e] = 2                     # query reads them
    a[nq:e, :nq] = 2
    return a, nq, [((nq - 1) / 2, "Q"), (nq + (nd - 1) / 2, dlabel)]


def attn(ax, kind, nq, nd, read, dlabel="docs"):
    a, split, ticks = build_mask(kind, nq, nd, read, dlabel)
    ax.imshow(a, cmap=ListedColormap([NONE_C, SELF, ACCENT]), vmin=0, vmax=2,
              interpolation="nearest")
    ax.axhline(split - 0.5, color=INK, lw=0.9)
    ax.axvline(split - 0.5, color=INK, lw=0.9)
    for sp in ax.spines.values():
        sp.set_edgecolor(INK)
        sp.set_linewidth(0.9)
    ax.set_xticks([t for t, _ in ticks])
    ax.set_xticklabels([l for _, l in ticks], fontsize=FS_R, color=INK)
    ax.set_yticks([t for t, _ in ticks])
    ax.set_yticklabels([l for _, l in ticks], fontsize=FS_R, color=INK,
                       rotation=90, va="center")
    ax.tick_params(length=0, pad=2)


COLS = [
    dict(t="Bi-encoder", s="embedding pipeline", arch=arch_bi,
         attn=dict(kind="bi", nq=3, nd=18, read=0, dlabel="the $N$ documents"),
         note="never jointly",
         per="$N$ vector comparisons\nper query",),
    dict(t="Cross-encoder", s="pairwise reranker",
         arch=lambda ax: arch_joint(ax, "Query $+$ 1 doc", "Encoder", "score"),
         attn=dict(kind="cross", nq=3, nd=18, read=2, dlabel="the $N$ documents"),
         note="one document at a time",
         per="$k$ pairwise passes\nper query",),
    dict(t="LLM listwise", s="reranker over top-$k$",
         arch=lambda ax: arch_joint(ax, "Query $+$ $k$ docs", "LLM", "ranking"),
         attn=dict(kind="cross", nq=3, nd=18, read=8, dlabel="the $N$ documents"),
         note="$k$ documents at once",
         per="one pass over $k$ docs\nper query",),
    dict(t="LLM corpus-in-context", s="generative retrieval",
         arch=lambda ax: arch_joint(ax, "Query $+$ all $N$ docs", "LLM",
                                    "ranking"),
         attn=dict(kind="cross", nq=3, nd=18, read=18, dlabel="the $N$ documents"),
         note="the whole corpus at once",
         per="one pass over $N$ docs\nper query",),
]



def main():
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["DejaVu Sans", "Arial"]})
    FW, FH = 11.6, 7.5
    fig = plt.figure(figsize=(FW, FH), facecolor="white")

    GUT, GAP = 0.020, 0.008
    W = (0.992 - GUT - 3 * GAP) / 4
    xs = [GUT + i * (W + GAP) for i in range(4)]

    for x0, c in zip(xs, COLS):
        cx = x0 + W / 2
        fig.text(cx, 0.990, c["t"], ha="center", va="top", fontsize=FS_T,
                 fontweight="bold", color=INK)
        fig.text(cx, 0.940, c["s"], ha="center", va="top", fontsize=FS_S,
                 color=GRAY)

        ax = fig.add_axes([x0, 0.615, W, 0.280])
        ax.set_axis_off()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        c["arch"](ax)

        sh = 0.300
        axm = fig.add_axes([cx - sh * FH / FW / 2, 0.250, sh * FH / FW, sh])
        attn(axm, **c["attn"])

        fig.text(cx, 0.192, c["note"], ha="center", va="center", fontsize=FS_S,
                 style="italic", color=INK)
        fig.text(cx, 0.126, c["per"], ha="center", va="center", fontsize=FS_S,
                 color=INK, linespacing=1.45)
        # compact two-line key/value; the keys live once, in the gutter



    handles = [Patch(facecolor=fc, edgecolor=INK, lw=0.7, label=lab)
               for fc, lab in ((SELF, "self-attention"),
                               (ACCENT, "query\u2013document"),
                               (NONE_C, "no attention"))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=FS_S, bbox_to_anchor=(0.5, 0.012), handlelength=1.15,
               handleheight=1.15, handletextpad=0.7, columnspacing=3.0,
               labelcolor=GRAY)

    save_fig(fig, "architecture_schematic")


if __name__ == "__main__":
    main()
