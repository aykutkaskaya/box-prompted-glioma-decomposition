"""Pick the confidence threshold on the VALIDATION split, not the test split.

The 0.55 default shipped with the console was measured on the test slices. That
is fine for a demo default but it is test-set tuning, and a threshold quoted in
a paper has to come from data the reported numbers are not measured on. This
re-derives it on the held-out validation split and prints both, so the size of
the mistake is visible rather than assumed away.

Runs on CPU by default so it can share a machine with a GPU job.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import DRIVE_ROOT, MATCH_IOU, SCORE_FLOOR  # noqa: E402
from core.metrics import sweep_thresholds  # noqa: E402
from core.runs import get_run  # noqa: E402

PROC = DRIVE_ROOT / "data" / "processed"


def gt_boxes(stem: str, split: str):
    p = PROC / "labels" / split / f"{stem}.txt"
    out = []
    if p.exists():
        for ln in p.read_text().splitlines():
            q = ln.split()
            if len(q) == 5:
                cx, cy, bw, bh = [float(v) for v in q[1:]]
                out.append([(cx - bw / 2) * 240, (cy - bh / 2) * 240,
                            (cx + bw / 2) * 240, (cy + bh / 2) * 240])
    return out


def collect(split: str, run_id: int, device: str, stride: int):
    from ultralytics import RTDETR
    run = get_run(run_id)
    if run is None:
        raise SystemExit(f"unknown run {run_id}")
    model = RTDETR(run["checkpoint"])

    files = sorted(glob.glob(str(PROC / "images" / split / "*.png")))[::stride]
    print(f"{split}: {len(files)} slices (stride {stride}) on {device}", flush=True)

    dets, gts = [], {}
    t0 = time.time()
    for i, f in enumerate(files):
        stem = os.path.splitext(os.path.basename(f))[0]
        bgr = np.array(Image.open(f).convert("RGB"))[:, :, ::-1].copy()
        r = model.predict(source=bgr, conf=SCORE_FLOOR, iou=0.45, device=device, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy().tolist() if r.boxes is not None else []
        scores = r.boxes.conf.cpu().numpy().tolist() if r.boxes is not None else []
        dets.append({"z": i, "boxes": boxes, "scores": scores})
        gts[i] = gt_boxes(stem, split)
        if (i + 1) % 500 == 0:
            el = time.time() - t0
            print(f"   {i+1}/{len(files)}  ({el/60:.1f} min, ~{el/(i+1)*(len(files)-i-1)/60:.1f} left)", flush=True)
    return dets, gts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, default=37)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=str(DRIVE_ROOT / "reports" / "threshold_on_val.json"))
    a = ap.parse_args()

    thresholds = [round(0.05 + 0.01 * i, 2) for i in range(91)]
    result = {"run_id": a.run_id, "match_iou": MATCH_IOU, "stride": a.stride, "splits": {}}

    for split in ("val", "test"):
        dets, gts = collect(split, a.run_id, a.device, a.stride)
        curve = sweep_thresholds(dets, gts, thresholds, MATCH_IOU)
        best = max(curve, key=lambda r: r["f1"])
        result["splits"][split] = {"n_slices": len(dets), "best": best, "curve": curve}
        print(f"  {split}: best F1 {best['f1']:.4f} at conf {best['threshold']}", flush=True)

    val_best = result["splits"]["val"]["best"]["threshold"]
    test_curve = result["splits"]["test"]["curve"]
    at_val = next(r for r in test_curve if abs(r["threshold"] - val_best) < 1e-9)
    test_best = result["splits"]["test"]["best"]

    result["honest_operating_point"] = {
        "threshold_chosen_on_val": val_best,
        "test_f1_at_that_threshold": at_val["f1"],
        "test_f1_at_test_optimum": test_best["f1"],
        "optimism_from_tuning_on_test": round(test_best["f1"] - at_val["f1"], 6),
        "test_optimum_threshold": test_best["threshold"],
    }

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=1), encoding="utf-8")

    h = result["honest_operating_point"]
    print("\n=== honest operating point ===")
    print(f"  threshold picked on val      : {h['threshold_chosen_on_val']}")
    print(f"  test F1 there                : {h['test_f1_at_that_threshold']:.4f}")
    print(f"  test F1 at the test optimum  : {h['test_f1_at_test_optimum']:.4f} "
          f"(conf {h['test_optimum_threshold']})")
    print(f"  optimism from tuning on test : {h['optimism_from_tuning_on_test']:+.4f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
