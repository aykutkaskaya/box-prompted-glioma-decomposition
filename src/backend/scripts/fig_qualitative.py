"""Figure 7: ten BraTS-Africa patients, one slice each, through both box arms.

The other figures are aggregates. This one shows what a localisation failure
actually looks like, because the claim is hard to believe from a bar chart:
00129, 00180 and 00215 score 0.000 Dice over the whole volume through the
pipeline and 0.86-0.91 through the oracle, with the same segmenter on the same
slices. The detector proposes nothing; the segmenter is never given a chance.
The other seven span the rest of the outcome range, so the failures are not
cherry-picked. Panels are one axial slice each and the Dice printed is that
slice's, so a patient whose volume Dice is non-zero can still show an empty
slice.

Masks are recomputed here with the same functions full_validation.py uses, so
the picture is of the run that produced the numbers rather than a re-enactment.

    RAW_DIR=<brats_africa dir> python scripts/fig_qualitative.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import figstyle as F                     # noqa: E402
from core import models, segmenters      # noqa: E402
from core.brats import MODALITY_ORDER, build_rgb_slice  # noqa: E402
from core.metrics import tight_bbox      # noqa: E402
from full_validation import (DEFAULT_CONF, SCORE_FLOOR, load_patient,  # noqa: E402
                             region_mask, slice_sets)

SEG_ID = "sam2.1_l"
RUN = 37

# Ordered by pipeline Dice on the full volume: the three total failures first,
# then the range up to the cases where the detector costs nothing.
CASES = ["BraTS_SSA_00129_000", "BraTS_SSA_00180_000", "BraTS_SSA_00215_000",
         "BraTS_SSA_00158_000", "BraTS_SSA_00213_000", "BraTS_SSA_00206_000",
         "BraTS_SSA_00113_000", "BraTS_SSA_00150_000", "BraTS_SSA_00008_000",
         "BraTS_SSA_00046_000"]


def grey(rgb):
    """FLAIR alone. The RGB composite stacks FLAIR/T1ce/T2 and reads as false
    colour, which is not how anyone looks at these scans."""
    f = rgb[:, :, MODALITY_ORDER.index("flair")].astype(float)
    lo, hi = np.percentile(f, (1, 99))
    return np.clip((f - lo) / (hi - lo + 1e-6), 0, 1)


def panel(ax, bg, title, colour=None):
    ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(F.RULE)
    ax.set_title(title, fontsize=7.6, color=colour or F.INK, pad=2.5)


def overlay(ax, mask, colour, alpha=0.42):
    if mask is None or not mask.any():
        return
    rgba = np.zeros(mask.shape + (4,))
    rgba[mask] = list(matplotlib.colors.to_rgb(colour)) + [alpha]
    ax.imshow(rgba)


def contour(ax, mask, colour, lw=0.7):
    if mask is not None and mask.any():
        ax.contour(mask.astype(float), levels=[0.5], colors=[colour],
                   linewidths=lw)


def case(pid):
    vols, seg_raw = load_patient(pid)
    gt = region_mask(seg_raw, "WT")
    usable, _ = slice_sets(vols, gt, 1)
    z = max(usable, key=lambda k: gt[:, :, k].sum())
    rgb = build_rgb_slice({m: vols[m][:, :, z] for m in MODALITY_ORDER})
    g = gt[:, :, z]

    gt_box = tight_bbox(g)
    om = segmenters.segment(SEG_ID, rgb, [gt_box])[0] if gt_box else None

    raw = models.detect(RUN, rgb, score_floor=SCORE_FLOOR)
    keep = [i for i, s in enumerate(raw["scores"]) if s >= DEFAULT_CONF]
    boxes = [raw["boxes"][i] for i in keep]
    pm = segmenters.segment(SEG_ID, rgb, boxes).any(axis=0) if boxes else None

    def dice(m):
        if m is None:
            return 0.0
        inter = 2 * (m & g).sum()
        tot = m.sum() + g.sum()
        return inter / tot if tot else 1.0

    return grey(rgb), g, gt_box, om, dice(om), boxes, pm, dice(pm), z


def main() -> None:
    # The supplement carries all ten patients; the main text carries six of
    # them, spanning the same range in half the height. Both are drawn from
    # one list, so the two figures cannot disagree about a case.
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", type=int, default=len(CASES))
    ap.add_argument("--slug", default="figS1_qualitative")
    args = ap.parse_args()
    # The main text shows the three patients the detector misses entirely and
    # the one where it costs nothing, which is the whole range in four panels.
    # The supplement keeps all ten, drawn from this same list.
    SHORT = ["BraTS_SSA_00129_000", "BraTS_SSA_00180_000",
             "BraTS_SSA_00215_000", "BraTS_SSA_00046_000"]
    cases = CASES if args.cases >= len(CASES) else SHORT[:args.cases]

    plt.rcParams.update(F.rc())
    data = []
    for pid in cases:
        print("  ", pid, flush=True)
        data.append((pid,) + case(pid))

    for lang in ("en", "tr"):
        T = F.L[lang]
        rows = (len(data) + 1) // 2
        fig, axes = plt.subplots(rows, 4, figsize=(7.2, 2.0 * rows))
        for k, (pid, bg, g, gt_box, om, od, boxes, pm, pd, z) in enumerate(data):
            a = axes[k // 2]
            c0, c1 = (k % 2) * 2, (k % 2) * 2 + 1
            tag = pid.replace("BraTS_SSA_", "SSA ").replace("_000", "")

            panel(a[c0], bg, f"{tag}  z={z}\n{T['oracle_s']}  {od:.3f}")
            overlay(a[c0], om, F.SEG_C)
            if gt_box:
                x0, y0, x1, y1 = gt_box
                a[c0].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                          edgecolor=F.ORACLE, lw=0.8,
                                          linestyle=(0, (3, 2))))
            contour(a[c0], g, F.WARN, lw=0.7)

            failed = not boxes
            unit = T["box_1"] if len(boxes) == 1 else T["boxes_n"]
            panel(a[c1], bg, f"{len(boxes)} {unit}\n{T['pipeline_s']}  {pd:.3f}",
                  colour=F.WARN if failed else None)
            overlay(a[c1], pm, F.LOC)
            for x0, y0, x1, y1 in boxes:
                a[c1].add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False,
                                          edgecolor=F.LOC, lw=0.5, alpha=0.75))
            contour(a[c1], g, F.WARN, lw=0.7)
            if failed:
                a[c1].text(0.5, 0.05, T["nobox"], transform=a[c1].transAxes,
                           ha="center", fontsize=8.4, color=F.WARN,
                           fontweight="bold")

        # the note sits above the first row of panel titles, and the space it
        # needs is a fixed number of inches rather than a fixed fraction, so a
        # six-panel figure has to give it proportionally more
        head = 0.42 / (2.0 * rows)
        fig.subplots_adjust(wspace=0.04, hspace=0.20, top=1 - head,
                            bottom=0.012, left=0.01, right=0.99)
        fig.text(0.5, 1 - head * 0.30, T["fig7_note"], ha="center",
                 fontsize=8.4, color=F.MUTED)
        for p in F.save(fig, args.slug, lang):
            print("  ", p.name)
        plt.close(fig)


if __name__ == "__main__":
    models.init(os.environ.get("FORCE_DEVICE"))
    segmenters.set_device(models.device())
    main()
