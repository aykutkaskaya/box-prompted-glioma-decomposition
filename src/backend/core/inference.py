"""One place where a slice becomes a scored, measured result.

Everything the API returns for a slice goes through `analyze_slice`, so the
single-slice view, the run comparison and the whole-study sweep can never drift
apart in how they threshold, match or measure.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from config import DEFAULT_SEGMENTER, MATCH_IOU, SCORE_FLOOR
from core import models, segmenters, study
from core.metrics import box_coverages, box_iou, mask_dice, mask_iou, match_boxes, prf


def analyze_slice(study_id: str, z: int, run_key: str, conf: float,
                  want_masks: bool = True, segmenter: str = DEFAULT_SEGMENTER,
                  want_oracle: bool = False, region: str = "WT") -> Dict:
    """Detect, segment and (when ground truth exists) score one slice.

    With want_oracle the same segmenter is also prompted with the ground-truth
    box. That arm is not deployable -- it needs the answer to produce the
    prompt -- but it is the ceiling the pipeline is measured against, and the
    difference between the two is the localisation term. It does not depend on
    the confidence threshold, so the sweep leaves it off.
    """
    rgb = study.slice_rgb(study_id, z)

    raw = models.detect(run_key, rgb, score_floor=SCORE_FLOOR)
    keep = [i for i, s in enumerate(raw["scores"]) if s >= conf]
    boxes = [raw["boxes"][i] for i in keep]
    scores = [raw["scores"][i] for i in keep]

    masks = (segmenters.segment(segmenter, rgb, boxes)
             if (want_masks and boxes) else np.zeros((0,) + rgb.shape[:2], bool))
    union = masks.any(axis=0) if len(masks) else np.zeros(rgb.shape[:2], dtype=bool)

    # a tumour-core detector is scored against tumour core, not whole tumour
    gt_m = study.gt_mask(study_id, z, region)
    gt_b = study.gt_box(study_id, z, region)
    gt_boxes = [gt_b] if gt_b else []

    detections: List[Dict] = []
    match = match_boxes(boxes, scores, gt_boxes, MATCH_IOU)
    for i, (b, s) in enumerate(zip(boxes, scores)):
        entry: Dict = {
            "bbox": [round(float(v), 2) for v in b],
            "score": round(float(s), 4),
            "area_px": int(masks[i].sum()) if i < len(masks) else 0,
            "matched": match["matched_gt"][i] is not None,
            "gt_iou": round(float(match["ious"][i]), 4) if gt_boxes else None,
        }
        if gt_boxes:
            cp, ct = box_coverages(b, gt_boxes[0])
            entry["pred_coverage"] = round(cp, 4)
            entry["target_coverage"] = round(ct, 4)
        detections.append(entry)

    result: Dict = {
        "study_id": study_id, "slice_index": z, "run_key": run_key,
        "conf_threshold": conf, "segmenter": segmenter, "region": region,
        "detections": detections,
        "n_detections": len(detections),
        "raw_scores": [round(float(s), 4) for s in raw["scores"]],
        "brain_px": None, "tumor_px": None,
        "ground_truth": None,
    }

    oracle = None
    if want_oracle and gt_boxes and want_masks:
        om = segmenters.segment(segmenter, rgb, gt_boxes)
        oracle = om.any(axis=0) if len(om) else None

    if gt_m is not None:
        gt_present = bool(gt_m.any())
        seg_dice = mask_dice(union, gt_m) if (gt_present or union.any()) else 1.0
        result["ground_truth"] = {
            "present": gt_present,
            "box": [round(float(v), 2) for v in gt_b] if gt_b else None,
            "area_px": int(gt_m.sum()),
            "box_iou": round(float(box_iou(boxes[0], gt_b)), 4) if (boxes and gt_b) else None,
            "seg_dice": round(float(seg_dice), 4),
            "seg_iou": round(float(mask_iou(union, gt_m)), 4),
            "detected": bool(match["tp"] > 0),
            **{k: round(v, 4) for k, v in prf(match["tp"], match["fp"], match["fn"]).items()},
            "tp": match["tp"], "fp": match["fp"], "fn": match["fn"],
        }
        if oracle is not None:
            od = float(mask_dice(oracle, gt_m))
            result["ground_truth"]["oracle_dice"] = round(od, 4)
            # what the detector costs on this slice, in Dice
            result["ground_truth"]["localisation"] = round(od - float(seg_dice), 4)

    return {"result": result, "rgb": rgb, "masks": masks, "union": union,
            "gt_mask": gt_m, "gt_box": gt_b, "raw": raw, "oracle": oracle}


def detect_only(study_id: str, z: int, run_key: str) -> Dict[str, list]:
    """Boxes at the score floor - the input to the operating-point sweep."""
    rgb = study.slice_rgb(study_id, z)
    return models.detect(run_key, rgb, score_floor=SCORE_FLOOR)
