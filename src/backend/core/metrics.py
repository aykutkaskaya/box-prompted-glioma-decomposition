"""Metric definitions, matched to the ones the 37-run study reports.

Box geometry (from the study's geometry_metrics):
    IoU              |P n G| / |P u G|
    C_p              |P n G| / |P|      prediction coverage - the IC-Arb term
    C_t              |P n G| / |G|      target coverage

Segmentation:
    Dice             2|A n B| / (|A| + |B|)
    IoU              |A n B| / |A u B|

Detection matching uses greedy highest-confidence-first assignment at
IoU >= 0.5, the same rule as the study's detection evaluation.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

Box = Sequence[float]


def box_iou(a: Box, b: Box) -> float:
    inter = _intersection(a, b)
    union = _area(a) + _area(b) - inter
    return float(inter / union) if union > 0 else 0.0


def box_coverages(pred: Box, gt: Box) -> Tuple[float, float]:
    """(C_p, C_t) - how much of the prediction is on target, and vice versa."""
    inter = _intersection(pred, gt)
    ap, ag = _area(pred), _area(gt)
    return (float(inter / ap) if ap > 0 else 0.0,
            float(inter / ag) if ag > 0 else 0.0)


def _area(b: Box) -> float:
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def _intersection(a: Box, b: Box) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    s = int(a.sum()) + int(b.sum())
    if s == 0:
        return 1.0          # both empty: a correct negative, not a failure
    return float(2 * int((a & b).sum()) / s)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(bool); b = b.astype(bool)
    u = int((a | b).sum())
    if u == 0:
        return 1.0
    return float(int((a & b).sum()) / u)


def match_boxes(pred_boxes: List[Box], scores: Sequence[float], gt_boxes: List[Box],
                iou_thr: float = 0.5) -> Dict[str, object]:
    """Greedy match, highest confidence first. Returns per-prediction outcomes."""
    order = sorted(range(len(pred_boxes)), key=lambda i: -scores[i]) if len(scores) else []
    used: set = set()
    outcome: List[Optional[int]] = [None] * len(pred_boxes)
    ious: List[float] = [0.0] * len(pred_boxes)

    for i in order:
        best_iou, best_j = 0.0, -1
        for j, g in enumerate(gt_boxes):
            if j in used:
                continue
            v = box_iou(pred_boxes[i], g)
            if v > best_iou:
                best_iou, best_j = v, j
        ious[i] = best_iou
        if best_iou >= iou_thr and best_j >= 0:
            outcome[i] = best_j
            used.add(best_j)

    tp = sum(1 for o in outcome if o is not None)
    return {
        "matched_gt": outcome,                      # index into gt_boxes, or None for a false positive
        "ious": ious,
        "tp": tp,
        "fp": len(pred_boxes) - tp,
        "fn": len(gt_boxes) - len(used),
        "unmatched_gt": [j for j in range(len(gt_boxes)) if j not in used],
    }


def prf(tp: int, fp: int, fn: int) -> Dict[str, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1}


def sweep_thresholds(detections: List[Dict], gt_by_slice: Dict[int, List[Box]],
                     thresholds: Sequence[float], iou_thr: float = 0.5) -> List[Dict]:
    """Re-threshold one stored forward pass into a full P/R/F1 curve.

    `detections` is a list of {"z", "boxes", "scores"} captured at the score
    floor, so the curve costs no extra GPU work.
    """
    rows = []
    for th in thresholds:
        tp = fp = fn = 0
        for det in detections:
            keep = [i for i, s in enumerate(det["scores"]) if s >= th]
            boxes = [det["boxes"][i] for i in keep]
            scores = [det["scores"][i] for i in keep]
            m = match_boxes(boxes, scores, gt_by_slice.get(det["z"], []), iou_thr)
            tp += m["tp"]; fp += m["fp"]; fn += m["fn"]
        rows.append({"threshold": round(float(th), 4), "tp": tp, "fp": fp, "fn": fn, **prf(tp, fp, fn)})
    return rows


def tight_bbox(mask: np.ndarray) -> Optional[List[float]]:
    """Tight xyxy box around a binary mask, in the study's inclusive-pixel convention."""
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]
