"""The segmenters the console can put behind a detector box.

Three run in-process (SAM1 ViT-B, SAM 2.1-B, SAM 2.1-L); MedSAM3 and its LoRA
adapters live in a sidecar because their dependency set conflicts with this
one, and are reached over HTTP. A sidecar that is not running simply reports as
unavailable rather than breaking the console.

The in-process ones are all *box-prompted* and interchangeable, so switching is
a fair comparison: same slice, same box, same Dice. MedSAM3's LoRA adapters were
trained for a detector-free text-prompted pipeline instead, which is a different
question and is exposed separately.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Dict, List, Optional

import numpy as np

from config import SAM2_DIR, SAM_BOX_PADDING, SAM_CHECKPOINT, SAM_MODEL_TYPE
from logger import get_logger

log = get_logger("segmenters")

# id -> (label, kind, note). `kind` decides how it is loaded and called.
CATALOG: "OrderedDict[str, dict]" = OrderedDict([
    ("sam1_vit_b", {
        "label": "SAM 1 ViT-B",
        "kind": "sam1",
        "note": "The 37-run ablation was scored with this. The three-cohort "
                "comparison was not - that ran on SAM 2.1 Large.",
    }),
    ("sam2.1_b", {
        "label": "SAM 2.1 Base",
        "kind": "ultralytics",
        "weights": "sam2.1_b.pt",
        "note": "Newer and faster than ViT-B. Not part of any reported "
                "measurement here.",
    }),
    ("sam2.1_l", {
        "label": "SAM 2.1 Large",
        "kind": "ultralytics",
        "weights": "sam2.1_l.pt",
        "note": "The segmenter behind every three-cohort number, and the "
                "oracle ceiling. +0.032 overall Dice over ViT-B across 8 runs "
                "and 3768 slices at conf 0.25 (reports/segmenter_ranking.json).",
        "reference": True,
    }),
])

_lock = threading.Lock()
_loaded: Dict[str, object] = {}
_device = "cpu"


def set_device(dev: str) -> None:
    global _device
    _device = dev


def catalog(sidecar_state: Optional[dict] = None) -> List[dict]:
    """What the UI should offer, including whatever the sidecar reports."""
    out = []
    for sid, spec in CATALOG.items():
        out.append({
            "id": sid, "label": spec["label"], "kind": spec["kind"],
            "note": spec["note"], "available": True,
            "prompt": "box", "reference": bool(spec.get("reference")),
            "loaded": sid in _loaded,
        })
    if sidecar_state:
        out.extend(sidecar_state.get("entries", []))
    return out


def _load(sid: str):
    spec = CATALOG.get(sid)
    if spec is None:
        raise ValueError(f"Unknown segmenter '{sid}'")
    with _lock:
        if sid in _loaded:
            return _loaded[sid]

    if spec["kind"] == "sam1":
        from segment_anything import SamPredictor, sam_model_registry
        if not SAM_CHECKPOINT.exists():
            raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CHECKPOINT}")
        log.info(f"loading {spec['label']} on {_device}")
        sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
        sam.to(device=_device)
        obj = SamPredictor(sam)
    else:
        from ultralytics import SAM
        local = SAM2_DIR / spec["weights"]
        target = str(local) if local.exists() else spec["weights"]
        if not local.exists():
            log.warning(f"{spec['weights']} not in {SAM2_DIR}; Ultralytics will download it")
        log.info(f"loading {spec['label']} from {target} on {_device}")
        obj = SAM(target)

    with _lock:
        _loaded[sid] = obj
    return obj


def segment(sid: str, rgb: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
    """Box-prompted masks from whichever segmenter is selected."""
    h, w = rgb.shape[:2]
    if not boxes:
        return np.zeros((0, h, w), dtype=bool)

    spec = CATALOG[sid]
    model = _load(sid)
    out = []

    if spec["kind"] == "sam1":
        model.set_image(np.ascontiguousarray(rgb))
        for b in boxes:
            box = np.array(_pad(b, w, h), dtype=np.float32)[None, :]
            masks, _, _ = model.predict(box=box, multimask_output=False)
            out.append(masks[0].astype(bool))
    else:
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        for b in boxes:
            r = model.predict(bgr, bboxes=[_pad(b, w, h)], device=_device, verbose=False)[0]
            if r.masks is None or len(r.masks.data) == 0:
                out.append(np.zeros((h, w), dtype=bool))
                continue
            m = r.masks.data[0].cpu().numpy().astype(bool)
            if m.shape != (h, w):
                from PIL import Image
                m = np.array(Image.fromarray(m.astype(np.uint8) * 255)
                             .resize((w, h), Image.NEAREST)) > 127
            out.append(m)

    return np.stack(out, axis=0).astype(bool)


def _pad(b: List[float], w: int, h: int) -> List[float]:
    if SAM_BOX_PADDING <= 0:
        return [float(v) for v in b]
    bw, bh = b[2] - b[0], b[3] - b[1]
    return [max(0.0, b[0] - bw * SAM_BOX_PADDING), max(0.0, b[1] - bh * SAM_BOX_PADDING),
            min(float(w), b[2] + bw * SAM_BOX_PADDING), min(float(h), b[3] + bh * SAM_BOX_PADDING)]


def loaded_ids() -> List[str]:
    with _lock:
        return list(_loaded.keys())
