"""Does the loss ranking survive a change of segmenter?

The 37-run study ranked losses by SAM overall Dice, measured with SAM 1 ViT-B.
If a newer segmenter reorders them, the study's conclusions are a property of
that segmenter rather than of the losses — a direct threat to validity worth
testing before publication.

Protocol is the study's, so the numbers are comparable with what it reports:
    conf = 0.25, detector box -> segmenter, Dice averaged over every positive
    test slice with undetected positives counted as 0 (`overall_dice`).

Only the segmenter changes. The SAM 1 numbers are already in each run's
`summary/run_summary.json`, so this re-scores with the new segmenter and
compares the two orderings (Spearman + the actual swaps).

    python scripts/segmenter_ranking.py --segmenter sam2.1_l
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DRIVE_ROOT, SCORE_FLOOR  # noqa: E402
from core import models, segmenters  # noqa: E402
from core.metrics import mask_dice  # noqa: E402
from core.runs import get_run, list_runs  # noqa: E402

PROC = DRIVE_ROOT / "data" / "processed"
OUT = DRIVE_ROOT / "reports" / "segmenter_ranking.json"

# Spread across the study's groups: two IoU baselines (one of them the precision
# outlier), the three IC-Arb alpha positions that anchor the trend, the best
# segmentation loss and the best cls-ablation run.
DEFAULT_RUNS = [1, 6, 9, 16, 19, 20, 34, 37]
STUDY_CONF = 0.25


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def positives(stride: int):
    out = []
    for f in sorted(glob.glob(str(PROC / "images" / "test" / "*.png")))[::stride]:
        stem = os.path.splitext(os.path.basename(f))[0]
        mp = PROC / "masks" / "test" / f"{stem}_mask.png"
        if not mp.exists():
            continue
        m = np.array(Image.open(mp).convert("L")) > 0
        if m.any():
            out.append((f, str(mp)))
    return out


def spearman(a: List[float], b: List[float]) -> float:
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for pos, i in enumerate(order):
            r[i] = pos
        return r
    ra, rb = rank(a), rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = sum((x - ma) ** 2 for x in ra) ** 0.5
    db = sum((y - mb) ** 2 for y in rb) ** 0.5
    return num / (da * db) if da and db else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--segmenter", default="sam2.1_l")
    ap.add_argument("--runs", default=",".join(str(r) for r in DEFAULT_RUNS))
    ap.add_argument("--conf", type=float, default=STUDY_CONF)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    run_ids = [int(x) for x in a.runs.split(",") if x.strip()]
    models.init(os.environ.get("FORCE_DEVICE"))
    segmenters.set_device(models.device())

    items = positives(a.stride)
    log(f"{len(items)} positive test slices | segmenter={a.segmenter} conf={a.conf} "
        f"runs={run_ids} device={models.device()}")

    slices = [(np.array(Image.open(f).convert("RGB")),
               np.array(Image.open(m).convert("L")) > 0) for f, m in items]

    results = {}
    for rid in run_ids:
        run = get_run(rid)
        if run is None:
            log(f"  run {rid} not found, skipped")
            continue
        t0 = time.time()
        dices, detected = [], 0
        for rgb, gt in slices:
            raw = models.detect(rid, rgb, score_floor=SCORE_FLOOR)
            boxes = [b for b, s in zip(raw["boxes"], raw["scores"]) if s >= a.conf]
            if not boxes:
                dices.append(0.0)      # the study counts a miss as Dice 0
                continue
            detected += 1
            masks = segmenters.segment(a.segmenter, rgb, boxes)
            dices.append(mask_dice(masks.any(axis=0), gt))
        el = time.time() - t0
        results[rid] = {
            "run_id": rid, "label": run["label"], "group": run["group"],
            "overall_dice_new": round(float(np.mean(dices)), 6),
            "conditional_dice_new": round(float(np.mean([d for d in dices if d > 0])), 6) if detected else 0.0,
            "detection_rate_pos": round(detected / len(slices), 6),
            "overall_dice_study_sam1": run["metrics"]["sam_overall_dice"],
            "elapsed_s": round(el, 1),
        }
        log(f"  run {rid:2d} {run['label']:<26} new {results[rid]['overall_dice_new']:.4f} "
            f"| study(SAM1) {run['metrics']['sam_overall_dice']:.4f} | {el/60:.1f} min")

    ids = list(results.keys())
    new = [results[i]["overall_dice_new"] for i in ids]
    old = [results[i]["overall_dice_study_sam1"] for i in ids]

    order_new = [i for _, i in sorted(zip(new, ids), reverse=True)]
    order_old = [i for _, i in sorted(zip(old, ids), reverse=True)]
    rho = spearman(new, old)

    payload = {
        "segmenter": a.segmenter, "conf": a.conf, "stride": a.stride,
        "n_slices": len(slices), "runs": results,
        "ranking_new": order_new, "ranking_study_sam1": order_old,
        "spearman": round(rho, 4),
        "ranking_preserved": order_new == order_old,
        "mean_shift": round(float(np.mean(new) - np.mean(old)), 6),
        "note": ("Study protocol: conf=0.25, overall Dice over positive test slices with "
                 "undetected positives scored 0. Only the segmenter differs."),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(payload, indent=1), encoding="utf-8")

    log("")
    log(f"Spearman(new, study) = {rho:+.4f} | identical order: {payload['ranking_preserved']}")
    log(f"mean Dice shift      = {payload['mean_shift']:+.4f}")
    log(f"order with {a.segmenter}: {order_new}")
    log(f"order in the study   : {order_old}")
    log(f"wrote {a.out}")


if __name__ == "__main__":
    main()
