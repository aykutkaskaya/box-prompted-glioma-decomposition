"""Overlay rendering.

Colour carries meaning and nothing else:
    cyan     prediction mask / box that matched ground truth
    red      prediction that did not match (false positive)
    amber    ground truth outline
Everything is drawn on the training-parity RGB slice so what the reader sees is
what the model saw.
"""
from __future__ import annotations

import base64
import io
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw

PRED_OK = (34, 211, 238)      # cyan
PRED_BAD = (248, 113, 113)    # red
GT = (251, 191, 36)           # amber
MASK_ALPHA = 0.38


def png_data_url(rgb: np.ndarray, scale: int = 2) -> str:
    img = Image.fromarray(rgb)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def mask_png_data_url(mask: np.ndarray, scale: int = 2) -> str:
    img = Image.fromarray((mask.astype(np.uint8) * 255))
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")


def _blend(base: np.ndarray, mask: np.ndarray, colour) -> np.ndarray:
    out = base.astype(np.float32)
    sel = mask.astype(bool)
    if sel.any():
        out[sel] = out[sel] * (1 - MASK_ALPHA) + np.array(colour, dtype=np.float32) * MASK_ALPHA
    return np.clip(out, 0, 255).astype(np.uint8)


def overlay(rgb: np.ndarray,
            masks: np.ndarray,
            boxes: List[List[float]],
            matched: List[bool],
            scores: List[float],
            gt_box: Optional[List[float]] = None,
            gt_mask: Optional[np.ndarray] = None,
            show_gt: bool = True,
            scale: int = 2) -> str:
    """Composite prediction masks, boxes and ground truth into one PNG data URL."""
    canvas = rgb.copy()

    if show_gt and gt_mask is not None and gt_mask.any():
        canvas = _blend(canvas, gt_mask, GT)
    for i in range(len(masks)):
        canvas = _blend(canvas, masks[i], PRED_OK if (i < len(matched) and matched[i]) else PRED_BAD)

    img = Image.fromarray(canvas)
    if scale > 1:
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)

    if show_gt and gt_box:
        draw.rectangle([gt_box[0] * scale, gt_box[1] * scale, gt_box[2] * scale, gt_box[3] * scale],
                       outline=GT, width=max(1, scale))

    for i, b in enumerate(boxes):
        colour = PRED_OK if (i < len(matched) and matched[i]) else PRED_BAD
        draw.rectangle([b[0] * scale, b[1] * scale, b[2] * scale, b[3] * scale],
                       outline=colour, width=max(1, scale))
        if i < len(scores):
            label = f"{scores[i]:.2f}"
            ty = max(0, b[1] * scale - 12)
            draw.rectangle([b[0] * scale, ty, b[0] * scale + 7 * len(label) + 4, ty + 12], fill=(10, 10, 15))
            draw.text((b[0] * scale + 2, ty + 1), label, fill=colour)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")
