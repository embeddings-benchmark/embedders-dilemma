#!/usr/bin/env python3
"""The thinking-token tax, in one two-panel figure.

Replaces the standalone cost-breakdown figure plus the cross-family ablation
table, which together took a full-width figure and a table to make one point.

  (a) What reasoning costs: API cost per benchmark pass, split by token type,
      for all ten LLMs.
  (b) What it buys on retrieval: the change in retrieval score when reasoning
      is disabled at the serving layer, for the models that have an ablation.

Both panels use the leaderboard visual language of the other figures in the
paper: a soft panel behind the bar field, a halo under each bar, brand logos
and rotated names beneath. The panels are independent; they share a style, not
an axis.

Ablation numbers come from generate_tables (_ablation_scores, _sum_gen_tokens),
so the figure cannot drift from what the table reported.

Output: visualizations/thinking_tax.{png,pdf}
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches as mpatches
from matplotlib.offsetbox import AnnotationBbox

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry  # noqa: E402
import generate_tables as gt  # noqa: E402
from plot_category_leaderboards import load_logo  # noqa: E402

C_INPUT, C_CACHED, C_OUTPUT, C_THINKING = "#94A3B8", "#CBD5E1", "#3B82F6", "#DC2626"
C_TEXT, C_GRID, C_AXIS = "#1F2937", "#F1F5F9", "#CBD5E1"
C_BETTER, C_WORSE = "#1D4ED8", "#B45309"   # blue/amber: safe under deuteranopia
C_OFFBAR = "#64748B"

_STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "sans-serif"],
    "text.color": C_TEXT, "axes.labelcolor": C_TEXT,
    "xtick.color": C_TEXT, "ytick.color": C_TEXT,
    "figure.facecolor": "white", "axes.facecolor": "white",
}

# sized so nothing lands under ~6pt once the 26in canvas is placed at
# \textwidth (5.5in), i.e. a 0.21x reduction
FS_VAL, FS_NAME, FS_TICK, FS_TITLE, FS_LEG = 32, 28, 28, 36, 29


def soft_panel(ax, x0, x1, floor, height):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x0, floor), x1 - x0, height,
        boxstyle="round,pad=0.02,rounding_size=6", mutation_aspect=0.02,
        facecolor="#FBFCFE", edgecolor="#EEF2F7", lw=1.2, zorder=0))


def logo_row(ax, models, xs, y_logo, y_text):
    for m, x in zip(models, xs):
        key = registry.LLM_META.get(m, {}).get("logo", "")
        img = load_logo(key, target_h=42) if key else None
        if img is not None:
            ax.add_artist(AnnotationBbox(img, (x, y_logo), frameon=False, pad=0.0,
                                         clip_on=False, annotation_clip=False,
                                         zorder=6))
        ax.text(x, y_text, registry.display_name(m), ha="right", va="top",
                rotation=38, rotation_mode="anchor", fontsize=FS_NAME,
                fontweight="bold", color=C_TEXT, clip_on=False, zorder=6)


def cost_frame():
    df = pd.read_csv(ROOT / "data" / "llm_token_usage.csv")
    df["c_in"] = (df.input_tokens - df.cached_tokens) * df.price_input_per_mtok / 1e6
    df["c_ca"] = df.cached_tokens * df.price_cached_per_mtok / 1e6
    df["c_out"] = df.output_tokens * df.price_output_per_mtok / 1e6
    df["c_th"] = df.thinking_tokens * df.price_output_per_mtok / 1e6
    df["total"] = df.c_in + df.c_ca + df.c_out + df.c_th
    return df.sort_values("total", ascending=False).reset_index(drop=True)


def ablation_rows(cs):
    off = gt._ablation_scores("off")
    rows = []
    for m in [m for m in cs[cs.model_type == "llm"].index if m in off]:
        vals = [v for v in off[m]["score"].values() if v is not None]
        if not vals:                       # a run exists but scored no tasks
            continue
        gd = gt._sum_gen_tokens(ROOT / "llm_results" / m, gt._RETR_STEMS)
        go = gt._sum_gen_tokens(ROOT / "ablation_results" / "thinking" / "off" / m,
                                gt._RETR_STEMS)
        default = cs.loc[m, "Retrieval"] * 100
        offv = sum(vals) / len(vals) * 100
        rows.append(dict(model=m, default=default, off=offv, delta=offv - default,
                         gen_on=gd / 1e6, gen_off=go / 1e6,
                         cut=(1 - go / gd) * 100 if gd else np.nan))
    rows.sort(key=lambda r: -r["delta"])
    return rows


def panel_cost(ax, df):
    n = len(df)
    xs = np.arange(n, dtype=float)
    tot, mx = df["total"], df["total"].max()
    Y_TOP, Y_LOGO, Y_TEXT, Y_BOT = mx * 1.30, -mx * 0.085, -mx * 0.165, -mx * 0.46

    ax.set_ylim(Y_BOT, Y_TOP)
    ax.set_xlim(xs[0] - 0.9, xs[-1] + 0.9)
    soft_panel(ax, xs[0] - 0.62, xs[-1] + 0.62, 0.0, mx * 1.06)
    for g in range(0, int(mx) + 30, 50):
        ax.axhline(g, color=C_GRID, lw=1.2, zorder=1)
    ax.axhline(0, color=C_AXIS, lw=1.6, zorder=2)

    bottoms = np.zeros(n)
    for v, c, lab in ((df.c_in, C_INPUT, "input"), (df.c_ca, C_CACHED, "cached"),
                      (df.c_out, C_OUTPUT, "output"),
                      (df.c_th, C_THINKING, "reasoning")):
        ax.bar(xs, v.values, 0.64, bottom=bottoms, color=c, edgecolor="white",
               lw=1.6, label=lab, zorder=3)
        bottoms += v.values
    for i, h in enumerate(tot):
        ax.text(xs[i], h + mx * 0.04, f"${h:,.0f}", ha="center", va="bottom",
                fontsize=FS_VAL, fontweight="bold", color=C_TEXT, zorder=6)
    logo_row(ax, df["model"], xs, Y_LOGO, Y_TEXT)

    ax.set_xticks([])
    ax.set_yticks([0, 50, 100, 150])
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f"${y:,.0f}"))
    ax.tick_params(axis="y", labelsize=FS_TICK, length=0)
    ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.legend(fontsize=FS_LEG, loc="upper right", frameon=False, ncol=2,
              handletextpad=0.6, labelspacing=0.45, columnspacing=1.4,
              borderaxespad=0.6)
    ax.set_title("(a)  what reasoning costs", fontsize=FS_TITLE,
                 fontweight="bold", loc="left", pad=16)


def panel_pair(ax, rows, key_on, key_off, *, ylabel, yticks, fmt, title,
               annot=None, annot_label=None, show_models=True,
               legend=False, inside=True):
    """One paired-bar panel: reasoning on vs off, in the leaderboard style."""
    xs = np.arange(len(rows), dtype=float)
    on = np.array([r[key_on] for r in rows])
    off = np.array([r[key_off] for r in rows])
    mx = max(on.max(), off.max())
    head = 1.94 if annot else 1.20
    Y_TOP = mx * head
    Y_LOGO, Y_TEXT, Y_BOT = -mx * 0.17, -mx * 0.30, -mx * 0.72
    w = 0.44          # pair stays tight; the gap between models opens up

    ax.set_ylim(Y_BOT if show_models else -mx * 0.06, Y_TOP)
    ax.set_xlim(xs[0] - 0.78, xs[-1] + 0.78)
    soft_panel(ax, xs[0] - 0.56, xs[-1] + 0.56, 0.0, mx * 1.06)
    for g in yticks[1:]:
        ax.axhline(g, color=C_GRID, lw=1.2, zorder=1)
    ax.axhline(0, color=C_AXIS, lw=1.6, zorder=2)

    ax.bar(xs - w / 2 * 1.02, on, w, color=C_THINKING, edgecolor="white", lw=1.6,
           zorder=3)
    ax.bar(xs + w / 2 * 1.02, off, w, color=C_OFFBAR, edgecolor="white", lw=1.6,
           zorder=3)
    for i, r in enumerate(rows):
        for sgn, v in ((-1.02, on[i]), (1.02, off[i])):
            # labels sit inside only when the bar is both tall and wide enough
            # for them; token strings like "3.3M" overflow a 0.4-wide bar
            put_in = inside and v > mx * 0.30
            ax.text(xs[i] + sgn * w / 2, v - mx * 0.05 if put_in else v + mx * 0.03,
                    fmt(v), ha="center", va="top" if put_in else "bottom",
                    fontsize=FS_VAL + 4, fontweight="bold",
                    color="white" if put_in else "#64748B", zorder=6)
        if annot:
            ax.text(xs[i], mx * 1.26, annot(r), ha="center", va="center",
                    fontsize=FS_VAL, fontweight="bold", color=C_TEXT, zorder=6)
    if annot_label:
        ax.text((xs[0] + xs[-1]) / 2, mx * 1.50, annot_label, ha="center",
                va="center", fontsize=FS_TICK + 6, color="#5B6673", zorder=6)
    if show_models:
        logo_row(ax, [r["model"] for r in rows], xs, Y_LOGO, Y_TEXT)

    ax.set_xticks([])
    ax.set_yticks(yticks)
    ax.set_ylabel(ylabel, fontsize=FS_TICK, color="#64748B", labelpad=8)
    ax.tick_params(axis="y", labelsize=FS_TICK, length=0)
    ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    if legend:
        ax.legend(handles=[mpatches.Patch(color=C_THINKING, label="reasoning on"),
                           mpatches.Patch(color=C_OFFBAR, label="reasoning off")],
                  fontsize=FS_LEG, loc="upper center", frameon=False, ncol=2,
                  handletextpad=0.6, columnspacing=1.6, borderaxespad=0.1)
    ax.set_title(title, fontsize=FS_TITLE, fontweight="bold", loc="left",
                 pad=14)


def main():
    plt.rcParams.update(_STYLE)
    cs = registry.category_scores(registry.load_scores(), complete_only=True)
    df, rows = cost_frame(), ablation_rows(cs)

    fig, (axa, axb) = plt.subplots(
        1, 2, figsize=(26, 9.4), facecolor="white",
        gridspec_kw=dict(width_ratios=[len(df) * 0.86, len(rows) + 2.4],
                         wspace=0.10))
    panel_cost(axa, df)
    panel_pair(axb, rows, "default", "off",
               ylabel="mean retrieval score", yticks=[0, 20, 40, 60],
               fmt=lambda v: f"{v:.0f}",
               title="(b)  what reasoning buys",
               annot=lambda r: f"{chr(0x2212)}{r['cut']:.0f}%",
               annot_label="reduction in generated tokens", legend=True)
    fig.subplots_adjust(left=0.045, right=0.99, top=0.93, bottom=0.055)

    out = ROOT / "visualizations"
    out.mkdir(exist_ok=True, parents=True)
    for ext in ("png", "pdf"):
        fig.savefig(out / f"thinking_tax.{ext}", dpi=200, bbox_inches="tight")
    print(f"Saved: thinking_tax.png + .pdf  ({len(df)} models, {len(rows)} ablations)")


if __name__ == "__main__":
    main()
