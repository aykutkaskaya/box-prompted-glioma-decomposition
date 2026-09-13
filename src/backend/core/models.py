"""Model loading: one shared SAM, RT-DETR detectors cached per run."""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import (MAX_CACHED_DETECTORS, SAM_BOX_PADDING, SAM_CHECKPOINT,
                    SAM_MODEL_TYPE, SAM_MULTIMASK, SCORE_FLOOR)
from core.runs import get_run
from logger import get_logger

log = get_logger("models")

_lock = threading.Lock()
_detectors: "OrderedDict[str, object]" = OrderedDict()
_sam = None
_device = "cpu"


def device() -> str:
    return _device


def init(force_device: Optional[str] = None) -> str:
    """Resolve the device and load SAM once. Detectors load lazily per run."""
    global _sam, _device
    import torch
    if force_device:
        _device = force_device
    elif torch.cuda.is_available():
        _device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        _device = "mps"
    else:
        _device = "cpu"

    from segment_anything import SamPredictor, sam_model_registry
    if not SAM_CHECKPOINT.exists():
        raise FileNotFoundError(f"SAM checkpoint not found: {SAM_CHECKPOINT}")
    log.info(f"Loading SAM {SAM_MODEL_TYPE} from {SAM_CHECKPOINT.name} on {_device}")
    sam = sam_model_registry[SAM_MODEL_TYPE](checkpoint=str(SAM_CHECKPOINT))
    sam.to(device=_device)
    _sam = SamPredictor(sam)
    log.info(f"SAM ready on {_device}")
    return _device


def get_detector(key: str):
    """RT-DETR for one run, loaded on first use and kept in a small LRU.

    Keyed on the registry key, not the run id: run 37 exists as a whole-tumour
    detector and twice more as a tumour-core one, and caching by id would serve
    whichever was loaded first.
    """
    key = str(key)
    with _lock:
        if key in _detectors:
            _detectors.move_to_end(key)
            return _detectors[key]

    run = get_run(key)
    if run is None:
        raise ValueError(f"Unknown run {key}")

    from ultralytics import RTDETR
    log.info(f"Loading detector {run['key']} ({run['label']}) from {run['checkpoint']}")
    model = RTDETR(run["checkpoint"])

    with _lock:
        _detectors[run["key"]] = model
        _detectors.move_to_end(run["key"])
        while len(_detectors) > MAX_CACHED_DETECTORS:
            evicted, _ = _detectors.popitem(last=False)
            log.info(f"Evicted detector for run {evicted} from cache")
    return model


def register_detector(key: str, checkpoint: str) -> None:
    """Bind a key to a checkpoint the registry does not know about.

    The registry now covers experiments_repeated as well, so the scripts that
    used this to reach the tumour-core weights no longer have to. It stays for
    checkpoints living outside both trees.
    """
    path = Path(checkpoint)
    if not path.exists():
        raise FileNotFoundError(f"detector checkpoint not found: {path}")

    from ultralytics import RTDETR
    log.info(f"Registering detector {key} from {path}")
    model = RTDETR(str(path))
    with _lock:
        _detectors[str(key)] = model
        _detectors.move_to_end(str(key))


def cached_run_keys() -> List[str]:
    with _lock:
        return list(_detectors.keys())


def detect(key: str, rgb: np.ndarray, score_floor: float = SCORE_FLOOR) -> Dict[str, list]:
    """Run detection on a training-parity RGB slice.

    Ultralytics expects a BGR array and flips it internally, so the RGB slice is
    reversed here to arrive at the model in the order it was trained on.
    """
    model = get_detector(key)
    bgr = np.ascontiguousarray(rgb[:, :, ::-1])
    res = model.predict(source=bgr, conf=score_floor, iou=0.45, device=_device, verbose=False)[0]
    if res.boxes is None or len(res.boxes) == 0:
        return {"boxes": [], "scores": []}
    return {
        "boxes": res.boxes.xyxy.detach().cpu().numpy().astype(float).tolist(),
        "scores": res.boxes.conf.detach().cpu().numpy().astype(float).tolist(),
    }


def segment(rgb: np.ndarray, boxes: List[List[float]]) -> np.ndarray:
    """Box-prompted SAM masks, matching the study's prompting exactly."""
    if _sam is None:
        raise RuntimeError("SAM is not loaded")
    h, w = rgb.shape[:2]
    if not boxes:
        return np.zeros((0, h, w), dtype=bool)

    _sam.set_image(np.ascontiguousarray(rgb))
    out = []
    for b in boxes:
        box = np.array(_pad(b, w, h), dtype=np.float32)[None, :]
        masks, scores, _ = _sam.predict(box=box, multimask_output=SAM_MULTIMASK)
        out.append(masks[int(np.argmax(scores))] if SAM_MULTIMASK else masks[0])
    return np.stack(out, axis=0).astype(bool)


def _pad(b: List[float], w: int, h: int) -> List[float]:
    if SAM_BOX_PADDING <= 0:
        return list(b)
    bw, bh = b[2] - b[0], b[3] - b[1]
    return [max(0, b[0] - bw * SAM_BOX_PADDING), max(0, b[1] - bh * SAM_BOX_PADDING),
            min(w, b[2] + bw * SAM_BOX_PADDING), min(h, b[3] + bh * SAM_BOX_PADDING)]
