"""Generate the manuscript figures, in English and Turkish, from the result files.

Every value is loaded through figstyle.py rather than typed here, so a figure
cannot disagree with the text that cites it. Run after any change to the result
files:

    python scripts/paper_figures.py            # all figures, both languages
    python scripts/paper_figures.py --only 3   # one figure
    python scripts/paper_figures.py --lang tr

Output goes to reports/figures/fig<N>_<name>_<lang>.{pdf,png}. PDF is the
manuscript copy with real embedded glyphs; PNG at 400 dpi is for everything
else.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, Wedge  # noqa: E402
import matplotlib.patheffects as pe                                # noqa: E402

# A value printed over a filled or hatched bar needs separating from it; a
# thin white stroke does that without a label box eating the bar.
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import figstyle as F                     # noqa: E402


def _box(ax, x, y, w, h, label, fc="none", ec=F.MUTED, lw=0.8, fs=7.5,
         tc=None, ls="-", z=2):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec,
                           linewidth=lw, linestyle=ls, zorder=z))
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fs, color=tc or F.INK, zorder=z + 1, linespacing=1.35)


def _arrow(ax, xy0, xy1, color=F.FAINT, lw=0.8, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(xy0, xy1, arrowstyle=style, mutation_scale=8,
                                 linewidth=lw, color=color, linestyle=ls,
                                 shrinkA=1, shrinkB=1, zorder=1))


# ===================================================================== fig 1
def fig1(lang: str):
    T = F.L[lang]
    d = F.wt_decomposition()
    fig = plt.figure(figsize=(7.2, 4.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.0], hspace=0.42)

    # --- panel a: the three arms ------------------------------------------
    ax = fig.add_subplot(gs[0]); ax.set_axis_off()
    ax.set_xlim(0, 100); ax.set_ylim(0, 34)
    _box(ax, 1, 12, 13, 10, "MRI\n(FLAIR/T1ce/T2)", fs=7)

    rows = [(25.5, T["oracle_s"], F.ORACLE, T["gtbox"]),
            (14.0, T["pipeline_s"], F.LOC, T["detbox"]),
            (2.5, T["direct_s"], F.SEG_C, None)]
    for y, name, col, src in rows:
        _arrow(ax, (14.5, 17), (23, y + 4))
        if src:
            _box(ax, 23, y, 15, 8, src, ec=col, fs=7, tc=col)
            _arrow(ax, (38.5, y + 4), (46, y + 4))
            _box(ax, 46, y, 17, 8, "SAM 2.1-L", fs=7)
            _arrow(ax, (63.5, y + 4), (71, y + 4))
        else:
            _box(ax, 23, y, 40, 8, "Medical SAM 3 + LoRA\n" + T["direct_s"], ec=col,
                 fs=7, tc=col)
            _arrow(ax, (63.5, y + 4), (72, y + 4))
        _box(ax, 72, y, 12, 8, T["dice"].split()[-1], ec=F.RULE, fs=7)
        ax.text(86, y + 4, name, ha="left", va="center", fontsize=8.5,
                color=col, fontweight="bold")

    ax.text(23, -1.5,
            ("detector stage = oracle-box − pipeline        "
             "residual = detector-free − oracle-box"
             if lang == "en" else
             "dedektör aşaması = oracle-kutu − boru hattı        "
             "artık = dedektörsüz − oracle-kutu"),
            fontsize=8.5, color=F.MUTED, va="top")
    ax.text(0, 31.5, "a", fontsize=10, fontweight="bold")

    # --- panel b: the identity, measured ----------------------------------
    ax = fig.add_subplot(gs[1])
    names = [T[c].replace("\n", " ") for c in F.COHORTS]
    loc = np.array([(d[c]["oracle"] - d[c]["pipeline"]).mean() for c in F.COHORTS])
    seg = np.array([(d[c]["direct"] - d[c]["oracle"]).mean() for c in F.COHORTS])
    tot = loc + seg
    y = np.arange(len(names))[::-1]

    ax.barh(y, loc, height=0.42, color=F.LOC,
            label=T["loc_term"].replace(chr(10), " "))
    # A waterfall: the segmentation term continues from the end of the
    # localisation bar, forward when positive and back when negative. A
    # negative step drawn solid reads as a positive one, so it is hatched
    # and its signed value is printed under it.
    for i, (l, sg) in enumerate(zip(loc, seg)):
        left = l if sg > 0 else l + sg
        ax.barh(y[i], abs(sg), height=0.42, left=left,
                color=F.SEG_C if sg > 0 else "white",
                edgecolor=F.SEG_C, linewidth=0.9,
                hatch=None if sg > 0 else "/////",
                label=T["seg_term"].replace(chr(10), " ") if i == 0 else None)
    # The arithmetic goes above the bar rather than inside it. A negative
    # segmentation step is drawn white over the localisation bar, so any
    # label placed inside disappears under it; and the row is the identity,
    # which reads better written out than split across three fills.
    for i, (l, sg, t) in enumerate(zip(loc, seg, tot)):
        ax.plot([t, t], [y[i] - 0.31, y[i] + 0.31], color=F.INK, lw=1.5, zorder=7)
        ax.text(0.001, y[i] + 0.30, f"{l:+.4f}", ha="left", va="bottom",
                fontsize=8.5, color=F.LOC)
        ax.text(0.026, y[i] + 0.30, f"{sg:+.4f}", ha="left", va="bottom",
                fontsize=8.5, color=F.SEG_C)
        ax.text(0.052, y[i] + 0.30, f"=  {t:+.3f}", ha="left", va="bottom",
                fontsize=7.5, color=F.INK, fontweight="bold")
    ax.axvline(0, color=F.MUTED, lw=0.7)
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=8.5)
    ax.set_xlabel(T["gap"] + "  —  " + T["total_gap"], fontsize=8.5)
    ax.set_xlim(-0.012, 0.152)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28),
              fontsize=7.4, ncol=2)
    ax.text(-0.20, 1.14, "b", fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    ax.set_title(r"(detector-free $-$ pipeline) = (oracle-box $-$ pipeline) + (detector-free $-$ oracle-box)"
                 if lang == "en" else
                 r"(dedektörsüz $-$ boru hattı) = (oracle-kutu $-$ boru hattı) + (dedektörsüz $-$ oracle-kutu)",
                 fontsize=8.2, color=F.MUTED, pad=7)
    return fig, "fig2_decomposition"


# ===================================================================== fig 2
def fig2(lang: str):
    T = F.L[lang]
    a = F.alpha_sweep()
    fig = plt.figure(figsize=(7.2, 4.9))
    gs = fig.add_gridspec(2, 5, height_ratios=[0.88, 1.0],
                          hspace=0.30, wspace=0.55)

    # --- panel a: three configurations, one shared coordinate system so the
    #     squares stay the same size whatever the subplot aspect turns out to be
    ax = fig.add_subplot(gs[0, :]); ax.set_axis_off()
    ax.set_xlim(-0.30, 5.10); ax.set_ylim(-1.02, 1.62); ax.set_aspect("equal")
    cases = [(T["tight"], 0.578, 1.000, (0.18, 0.18, 0.64, 0.64)),
             (T["exact"], 1.000, 1.000, (0.0, 0.0, 1.0, 1.0)),
             (T["inflated"], 0.510, 0.510, (-0.21, -0.21, 1.42, 1.42))]
    for k, (name, iou, cp, (px, py, pw, ph)) in enumerate(cases):
        ox = k * 1.72
        ax.add_patch(Rectangle((ox, 0), 1, 1, facecolor=F.SEG_C, alpha=0.14,
                               edgecolor=F.SEG_C, lw=1.0))
        ax.add_patch(Rectangle((ox + px, py), pw, ph, facecolor="none",
                               edgecolor=F.LOC, lw=1.2, linestyle=(0, (3.5, 2.2))))
        ax.text(ox + 0.5, -0.34, name, ha="center", va="top", fontsize=7.5)
        ax.text(ox + 0.5, -0.58, f"IoU {iou:.3f}", ha="center", va="top",
                fontsize=8.5, color=F.MUTED, family="monospace")
        ax.text(ox + 0.5, -0.80, f"$C_p$ {cp:.3f}", ha="center", va="top",
                fontsize=8.5, color=F.LOC, family="monospace")
    ax.text(-0.30, 1.60, "a", fontsize=10, fontweight="bold", va="top")
    ax.plot([1.30, 1.60], [1.44, 1.44], color=F.SEG_C, lw=1.4)
    ax.text(1.67, 1.44, T["gt"], fontsize=8.5, color=F.MUTED, va="center")
    ax.plot([2.75, 3.05], [1.44, 1.44], color=F.LOC, lw=1.2,
            linestyle=(0, (3.5, 2.2)))
    ax.text(3.12, 1.44, T["pred"], fontsize=8.5, color=F.MUTED, va="center")

    # --- panel b: the derivative, one relation per line -------------------
    ax = fig.add_subplot(gs[1, :2]); ax.set_axis_off()
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.text(-0.09, 1.14, "b", fontsize=10, fontweight="bold", va="top")
    for y, expr in ((0.93, r"$\dfrac{\partial\,\mathrm{IoU}}{\partial |P|}"
                           r"\;=\;-\dfrac{I}{U^{2}}$"),
                    (0.66, r"$\dfrac{\partial C_p}{\partial |P|}"
                           r"\;=\;-\dfrac{I}{|P|^{2}}$"),
                    (0.36, r"$U \geq |P|\;\;\Rightarrow\;\;"
                           r"\left|\dfrac{\partial C_p}{\partial |P|}\right|"
                           r"\geq\left|\dfrac{\partial\,\mathrm{IoU}}"
                           r"{\partial |P|}\right|$")):
        ax.text(0.03, y, expr, fontsize=9.0, va="center")
    ax.text(0.03, 0.0,
            ("Lowering α should tighten boxes:\n$C_p$ up and $C_t$ down, together."
             if lang == "en" else
             "α düşünce kutular daralmalı:\n$C_p$ yukarı, $C_t$ aşağı, birlikte."),
            fontsize=7.6, color=F.MUTED, va="bottom", linespacing=1.5)

    # --- panel c: the measured sweep --------------------------------------
    ax = fig.add_subplot(gs[1, 2:])
    al = np.array([r["alpha"] for r in a["runs"]])
    cp = np.array([r["cp"] for r in a["runs"]])
    ct = np.array([r["ct"] for r in a["runs"]])
    ax.plot(al, cp, "-o", color=F.LOC, ms=3.0, label=T["cp"])
    ax.plot(al, ct, "-o", color=F.SEG_C, ms=3.0, label=T["ct"])
    for x, sp in a["spread"].items():
        for key, col in (("cp", F.LOC), ("ct", F.SEG_C)):
            lo, hi = sp[key]
            ax.plot([x, x], [lo, hi], color=col, lw=2.2, solid_capstyle="butt")
            for v in (lo, hi):
                ax.plot([x - 0.022, x + 0.022], [v, v], color=col, lw=1.4)
    ax.set_xlabel(T["alpha"], fontsize=8.5)
    ax.set_xlim(-0.07, 1.07)
    ax.set_ylim(0.872, 0.976)
    ax.grid(axis="y", alpha=0.5); ax.set_axisbelow(True)
    # the two curves converge on the right, so the key goes in the gap between
    # them on the left rather than over either one
    ax.legend(loc="center left", bbox_to_anchor=(0.06, 0.52), fontsize=7.4)
    n = list(a["spread"].values())[0]["n"] if a["spread"] else 0
    ax.text(0.99, 0.02, f"{T['seedbar']} (n={n})", transform=ax.transAxes,
            ha="right", fontsize=6.8, color=F.FAINT)
    ax.text(-0.155, 1.05, "c", fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    return fig, "fig2_loss_geometry"


# ===================================================================== fig 3
def fig3(lang: str):
    T = F.L[lang]
    d = F.wt_decomposition()
    f1 = F.detection_f1()
    fig, ax = plt.subplots(figsize=(6.0, 3.5))

    x = np.arange(3)
    loc = np.array([(d[c]["oracle"] - d[c]["pipeline"]).mean() for c in F.COHORTS])
    seg = np.array([(d[c]["direct"] - d[c]["oracle"]).mean() for c in F.COHORTS])
    w = 0.34
    ax.bar(x - w / 2, loc, w, color=F.LOC, label=T["loc_term"].replace("\n", " "))
    ax.bar(x + w / 2, seg, w, color=F.SEG_C, label=T["seg_term"].replace("\n", " "))
    for xi, v in zip(x - w / 2, loc):
        ax.text(xi, v + 0.004, f"{v:+.4f}", ha="center", fontsize=8.5, color=F.LOC)
    for xi, v in zip(x + w / 2, seg):
        off = 0.004 if v >= 0 else -0.010
        ax.text(xi, v + off, f"{v:+.4f}", ha="center", fontsize=8.5, color=F.SEG_C)
    ax.axhline(0, color=F.MUTED, lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([T[c] + chr(10) + T["f1"] + f" {f1[c]:.3f}"
                        for c in F.COHORTS], fontsize=8.5)
    ax.set_ylabel(T["gap"], fontsize=8.5)
    ax.set_ylim(-0.088, 0.118)
    ax.grid(axis="y", alpha=0.5); ax.set_axisbelow(True)
    # below the negative bar, not over its label
    ax.legend(loc="lower center", fontsize=7.2, ncols=2,
              bbox_to_anchor=(0.5, -0.02), frameon=False)

    ax.text(0.5, 1.045,
            ("only the detector-stage term responds to external shift"
             if lang == "en" else
             "dışsal kaymaya yalnızca dedektör aşaması terimi tepki veriyor"),
            transform=ax.transAxes, ha="center", va="bottom", fontsize=8.2,
            color=F.MUTED)
    return fig, "fig3_domain_shift"


# ===================================================================== fig 4
def fig4(lang: str):
    """Why a box cannot ask for a rim, and what it costs."""
    T = F.L[lang]
    reg = F.oracle_regions()
    fig = plt.figure(figsize=(7.2, 3.5))
    gs = fig.add_gridspec(1, 5, wspace=0.9)

    # --- panel a: the geometry --------------------------------------------
    ax = fig.add_subplot(gs[0, :2]); ax.set_axis_off()
    # side by side, so the two identical boxes can be compared directly;
    # stacked vertically the point of the panel is much harder to see
    ax.set_xlim(-1.35, 1.35); ax.set_ylim(-1.55, 1.15); ax.set_aspect("equal")
    for cx, notch, col, name in ((-0.62, True, F.LOC, T["ring"]),
                                 (0.62, False, F.SEG_C, T["disc"])):
        ax.add_patch(Circle((cx, 0.10), 0.50, facecolor=col, alpha=0.28,
                            edgecolor=col, lw=1.1))
        if notch:
            # the non-enhancing part, drawn open to the lower edge rather than
            # enclosed: enclosure is the annulus configuration the text tests
            # and rejects. The enhancing tissue still reaches all four sides of
            # the box, which is the property that makes the two boxes equal.
            ax.add_patch(Wedge((cx, 0.10), 0.52, 205, 250, facecolor="white",
                               edgecolor=col, lw=0.9))
        ax.add_patch(Rectangle((cx - 0.50, -0.40), 1.0, 1.0,
                               facecolor="none", edgecolor=F.INK, lw=1.1,
                               linestyle=(0, (3.5, 2.2))))
        ax.text(cx, -0.56, name.replace(" (", chr(10) + "("),
                fontsize=7.2, color=col, va="top", ha="center",
                linespacing=1.4)
    ax.text(0, 0.74, T["samebox"], ha="center", va="bottom", fontsize=8.5,
            color=F.INK, fontweight="bold")
    ax.text(-1.35, 1.12, "a", fontsize=10, fontweight="bold", va="top")

    # --- panel b: what it costs -------------------------------------------
    ax = fig.add_subplot(gs[0, 2:])
    regions = ["WT", "TC", "ET"]
    cols = {"WT": F.ORACLE, "TC": F.SEG_C, "ET": F.LOC}
    xs = np.arange(len(F.COHORTS))
    w = 0.26
    for k, r in enumerate(regions):
        vals = [reg[c][r]["rel_vol_diff"] * 100 for c in F.COHORTS]
        ax.bar(xs + (k - 1) * w, vals, w, color=cols[r], label=T[r.lower()])
        for xi, v in zip(xs + (k - 1) * w, vals):
            ax.text(xi, v + 1.4, f"{v:+.0f}%", ha="center", fontsize=6.8,
                    color=cols[r])
    ax.axhline(0, color=F.MUTED, lw=0.7)
    ax.set_xticks(xs)
    ax.set_xticklabels([T[c + "_s"] for c in F.COHORTS], fontsize=8.5)
    ax.set_ylabel(T["volerr"], fontsize=8.5)
    ax.set_ylim(-2, 58)
    ax.grid(axis="y", alpha=0.5); ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=7.4, ncol=3, columnspacing=1.1)
    ax.text(-0.18, 1.06, "b", fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    ax.set_title(("the oracle box over-fills ET, and only ET"
                  if lang == "en" else
                  "oracle kutusu yalnızca ET'yi taşırıyor"),
                 fontsize=8.5, color=F.MUTED, pad=6)
    return fig, "fig4_prompt_ceiling"


# ===================================================================== fig 5
def fig5(lang: str):
    """Seed instability, and where it comes from."""
    T = F.L[lang]
    rep = F.tc_summary()
    seeds = [str(s) for s in rep.get("detector_seeds", [])]
    have = [c for c in F.COHORTS if rep["cohorts"].get(c, {}).get("seeds")]
    fig = plt.figure(figsize=(7.2, 3.6))
    gs = fig.add_gridspec(1, 5, wspace=1.15)

    # --- panel a: the spread widens ---------------------------------------
    ax = fig.add_subplot(gs[0, :2])
    for i, c in enumerate(have):
        vals = [rep["cohorts"][c]["seeds"][s][F.SEG]["pipeline"] for s in seeds]
        ax.plot([i] * len(vals), vals, "o", color=F.LOC, ms=4.5, zorder=3)
        ax.plot([i, i], [min(vals), max(vals)], color=F.LOC, lw=1.2, zorder=2)
        # the rightmost label has no room to its right: it ran outside
        # the axes and printed over panel b's cohort names
        last = i == len(have) - 1
        ax.text(i + (-0.14 if last else 0.14), np.mean(vals),
                f"Δ {max(vals) - min(vals):.3f}", fontsize=8.5, color=F.LOC,
                va="center", ha="right" if last else "left")
    ax.set_xticks(range(len(have)))
    ax.set_xticklabels([T[c + "_s"] for c in have], fontsize=7.6,
                       rotation=16, ha="right")
    ax.set_ylabel(T["dice_tc"], fontsize=8.5)
    ax.set_xlim(-0.45, len(have) - 0.25)
    ax.grid(axis="y", alpha=0.5); ax.set_axisbelow(True)
    ax.text(-0.34, 1.14, "a", fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    ax.set_title(f"{T['spread']} ({len(seeds)} {T['seeds_n']})", fontsize=8.5,
                 color=F.MUTED, pad=6)

    # --- panel b: it is carried by a tail, not a shift ---------------------
    ax = fig.add_subplot(gs[0, 2:])
    rng = np.random.default_rng(0)
    for i, c in enumerate(have):
        pp = rep["cohorts"][c]["seeds"][seeds[0]][F.SEG]["per_patient"]
        v = np.array([x["pipeline"] for x in pp.values()])
        jit = rng.uniform(-0.16, 0.16, size=len(v))
        low = v < 0.5
        ax.scatter(v[~low], i + jit[~low], s=7, color=F.SEG_C, alpha=0.55,
                   linewidths=0)
        ax.scatter(v[low], i + jit[low], s=9, color=F.WARN, alpha=0.85,
                   linewidths=0)
        ax.plot([np.median(v)], [i], "|", color=F.INK, ms=14, mew=1.6, zorder=4)
        ax.text(1.02, i, f"n={len(v)}", fontsize=6.8, color=F.FAINT,
                va="center")
    ax.set_yticks(range(len(have)))
    ax.set_yticklabels([T[c + "_s"] for c in have], fontsize=8.5)
    ax.set_xlabel(T["dice_tc"], fontsize=8.5)
    ax.set_xlim(-0.03, 1.09)
    ax.set_ylim(-0.5, len(have) - 0.5)
    ax.grid(axis="x", alpha=0.5); ax.set_axisbelow(True)
    ax.text(-0.30, 1.14, "b", fontsize=10, fontweight="bold",
            transform=ax.transAxes)
    ax.plot([], [], "|", color=F.INK, ms=10, mew=1.6, label=T["median"])
    ax.scatter([], [], s=9, color=F.WARN, label="< 0.50")
    ax.legend(loc="upper left", fontsize=8.5, ncol=2, handletextpad=0.4)
    return fig, "fig5_seed_instability"


# ===================================================================== fig 6
def fig6(lang: str):
    """Both systems, with the two places this study intervened."""
    T = F.L[lang]
    en = lang == "en"
    fig, ax = plt.subplots(figsize=(7.2, 3.9))
    ax.set_axis_off(); ax.set_xlim(0, 100); ax.set_ylim(-6, 62)

    def stage(x, y, w, h, txt, touched=False, fs=6.8):
        # Turkish runs longer than English in every one of these captions, so
        # the size is fitted to the box rather than set per language by hand.
        longest = max(len(t) for t in txt.split(chr(10)))
        fits = w / (0.145 * fs)
        if longest > fits:
            fs *= fits / longest
        _box(ax, x, y, w, h, txt, fc="#FBF3E7" if touched else "none",
             ec=F.LOC if touched else F.MUTED, lw=1.2 if touched else 0.8,
             fs=fs, tc=F.LOC if touched else F.INK)

    # ---- RT-DETR -> SAM ---------------------------------------------------
    ax.text(0, 59, "RT-DETR → SAM 2.1-L", fontsize=9, fontweight="bold")
    y = 42
    xs = [(0, 13, T["s_bb"]), (15, 15, T["s_enc"]),
          (32, 15, T["s_qs"]), (49, 15, T["s_dec"]),
          (66, 13, T["s_box"]), (81, 15, T["s_mask"])]
    for i, (x, w, t) in enumerate(xs):
        stage(x, y, w, 11, t)
        if i:
            _arrow(ax, (xs[i - 1][0] + xs[i - 1][1], y + 5.5), (x, y + 5.5))
    stage(49, y - 15, 15, 10,
          ("overlap loss\n" r"$\mathcal{L}_{ov}(\alpha)$") if en else
          ("örtüşme kaybı\n" r"$\mathcal{L}_{ov}(\alpha)$"), touched=True, fs=7)
    _arrow(ax, (56.5, y - 5), (56.5, y), color=F.LOC, lw=1.1)
    ax.text(66, y - 10,
            ("replaces GIoU in the main, auxiliary and denoising heads;\n"
             "Hungarian matching and the L1 term untouched" if en else
             "ana, yardımcı ve gürültü-giderme başlıklarında GIoU'nun yerine;\n"
             "Hungarian eşleştirme ve L1 terimi değiştirilmedi"),
            fontsize=7.6, color=F.MUTED, va="center", linespacing=1.5)

    # ---- Medical SAM 3 + LoRA ---------------------------------------------------
    ax.text(0, 24, "Medical SAM 3 + LoRA", fontsize=9, fontweight="bold")
    y = 9
    # LoRA is not a stage: it is attached to the frozen encoder. Drawing it
    # inline made it look like one and overflowed the row.
    row2 = [(0, 17, ("volume slab" + chr(10) + "+ text prompt") if en else
             ("hacim dilimi" + chr(10) + "+ metin istemi")),
            (20, 29, ("image encoder" + chr(10) + "32 blocks × width 1024, frozen")
             if en else ("görüntü kodlayıcı" + chr(10) +
                         "32 blok × genişlik 1024, dondurulmuş")),
            (53, 19, "mask decoder" if en else "maske kod çözücü"),
            (76, 16, "WT / TC / ET")]
    for k, (x, w, t) in enumerate(row2):
        stage(x, y, w, 11, t)
        if k:
            _arrow(ax, (row2[k - 1][0] + row2[k - 1][1], y + 5.5), (x, y + 5.5))
    stage(19, y - 10.5, 31, 9,
          ("LoRA r=8 on attn.qkv + attn.proj" + chr(10) +
           "64 modules · 1.57 M params · 19 MB") if en else
          ("attn.qkv + attn.proj üzerinde LoRA r=8" + chr(10) +
           "64 modül · 1.57 M parametre · 19 MB"), touched=True, fs=6.5)
    _arrow(ax, (34.5, y - 1.5), (34.5, y), color=F.LOC, lw=1.1)
    ax.text(53, y - 6.0,
            ("only the adapter is trained;" + chr(10) +
             "the 10.0 GB base stays frozen") if en else
            ("yalnızca adaptör eğitiliyor;" + chr(10) +
             "10.0 GB temel model donduruldu"),
            fontsize=7.6, color=F.MUTED, va="center", linespacing=1.5)

    ax.text(0, -5.0,
            ("grey: used as published        amber: changed in this study"
             if en else
             "gri: yayınlandığı hâliyle        amber: bu çalışmada değiştirildi"),
            fontsize=8.5, color=F.MUTED)
    return fig, "fig1_architectures"


# fig2 drew the loss geometry, which went with the ablation
FIGS = {1: fig1, 3: fig3, 4: fig4, 5: fig5, 6: fig6}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", type=int, default=None)
    ap.add_argument("--lang", choices=("en", "tr", "both"), default="both")
    a = ap.parse_args()

    langs = ["en", "tr"] if a.lang == "both" else [a.lang]
    todo = [a.only] if a.only else sorted(FIGS)
    plt.rcParams.update(F.rc())
    for n in todo:
        for lang in langs:
            fig, name = FIGS[n](lang)
            paths = F.save(fig, name, lang)
            plt.close(fig)
            print(f"  {name}_{lang}: " + ", ".join(p.name for p in paths))


if __name__ == "__main__":
    main()
