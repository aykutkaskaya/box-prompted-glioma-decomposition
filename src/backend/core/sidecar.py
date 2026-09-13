"""Thin client for the MedSAM3 sidecar.

Everything here degrades to "unavailable" rather than raising, so the console
works unchanged when the sidecar is not running.
"""
from __future__ import annotations

import base64
import io
import os
from typing import Dict, List, Optional

import numpy as np

from logger import get_logger

log = get_logger("sidecar")

MEDSAM3_URL = os.environ.get("MEDSAM3_URL", "http://127.0.0.1:8020")
TIMEOUT_HEALTH = 2.0
TIMEOUT_INFER = float(os.environ.get("MEDSAM3_TIMEOUT", 900))


def _get(path: str, timeout: float):
    import urllib.request, json
    with urllib.request.urlopen(f"{MEDSAM3_URL}{path}", timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def _post(path: str, payload: dict, timeout: float):
    import urllib.request, json
    req = urllib.request.Request(
        f"{MEDSAM3_URL}{path}",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def health() -> Optional[dict]:
    try:
        return _get("/health", TIMEOUT_HEALTH)
    except Exception:
        return None


def catalog_entries() -> dict:
    """UI entries for whatever the sidecar can currently serve."""
    h = health()
    if not h:
        return {"available": False, "entries": [{
            "id": "medsam3", "label": "MedSAM3 (+LoRA)", "kind": "sidecar",
            "prompt": "text", "available": False,
            "note": "Sidecar not running — start segmenters/medsam3_service.py to enable.",
        }]}

    entries = [{
        "id": "medsam3", "label": "MedSAM3 (no adapter)", "kind": "sidecar",
        "prompt": "text", "available": True, "adapter": None,
        "note": "Foundation model, detector-free text prompt. Over-segments badly without the adapter.",
    }]
    for a in h.get("adapters", []):
        entries.append({
            "id": f"medsam3+{a['id']}", "label": f"MedSAM3 + LoRA {a['id'].replace('seed_', 'seed ')}",
            "kind": "sidecar", "prompt": "text", "available": True, "adapter": a["id"],
            "experimental": bool(a.get("experimental")),
            "note": "Detector-free. BraTS2020 LoRA fine-tune, verified against its stored hash."
                    + (" Experimental seed." if a.get("experimental") else ""),
        })
    return {"available": True, "entries": entries, "device": h.get("device"),
            "active_adapter": h.get("active_adapter")}


def segment_volume(prgb: np.ndarray, adapter: Optional[str], regions: List[str],
                   batch_size: int = 2, threshold: Optional[float] = None) -> Dict:
    """Run the research's detector-free pipeline on a pseudo-RGB volume."""
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(prgb, dtype=np.float32), allow_pickle=False)
    payload = {
        "prgb_npy_b64": base64.b64encode(buf.getvalue()).decode("utf-8"),
        "adapter": adapter, "regions": regions, "batch_size": batch_size,
    }
    if threshold is not None:
        payload["threshold"] = threshold
    res = _post("/segment/volume", payload, TIMEOUT_INFER)

    out = {}
    for region, blob in res.get("regions", {}).items():
        packed = np.load(io.BytesIO(base64.b64decode(blob["packed_npy_b64"])), allow_pickle=False)
        shape = tuple(blob["shape"])
        n = int(np.prod(shape))
        out[region] = np.unpackbits(packed)[:n].reshape(shape).astype(bool)
    return {"masks": out, "elapsed_ms": res.get("elapsed_ms"),
            "adapter": res.get("adapter"), "threshold": res.get("threshold"),
            "mode": res.get("mode"), "note": res.get("note"),
            "all_logits_finite": res.get("all_logits_finite")}
