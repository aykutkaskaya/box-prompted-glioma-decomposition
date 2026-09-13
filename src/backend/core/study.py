"""A loaded BraTS study: pre-rendered slices, optional ground truth, metadata.

Uploading three ~50 MB volumes per slice request would be absurd, so a study is
ingested once. Every axial slice is rendered to the exact uint8 RGB the detector
was trained on and memory-mapped from disk afterwards (~27 MB for 240x240x155),
which makes scrubbing a read rather than a decode.
"""
from __future__ import annotations

import json
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from config import MIN_BRAIN_PIXELS_FALLBACK, STUDIES_DIR, STUDY_TTL_SECONDS
from core.brats import MODALITY_ORDER, build_rgb_slice, load_volume
from core.metrics import tight_bbox
from logger import get_logger

log = get_logger("study")


@dataclass
class StudyMeta:
    study_id: str
    name: str
    total_slices: int
    height: int
    width: int
    default_slice: int
    brain_px: List[int]
    tumor_px: List[int]
    usable_slices: List[int]
    has_ground_truth: bool

    def to_dict(self) -> dict:
        return {
            "study_id": self.study_id, "name": self.name,
            "total_slices": self.total_slices,
            "shape": [self.height, self.width],
            "default_slice": self.default_slice,
            "brain_px": self.brain_px, "tumor_px": self.tumor_px,
            "usable_slices": self.usable_slices,
            "has_ground_truth": self.has_ground_truth,
            "positive_slices": [i for i, t in enumerate(self.tumor_px) if t > 0],
        }


CLIP_SIGMA = 3.0


def _zscore_clip(vol: np.ndarray) -> np.ndarray:
    """Per-volume z-score over non-zero voxels, clipped to +/-3, background left at 0.

    Mirrors `normalize_volume` in the MedSAM3 research project so the sidecar
    receives exactly the representation it was trained on.
    """
    v = np.asarray(vol, dtype=np.float32)
    brain = v > 0
    out = np.zeros_like(v)
    if brain.any():
        vals = v[brain]
        mu, sd = float(vals.mean()), float(vals.std()) + 1e-8
        out[brain] = np.clip((vals - mu) / sd, -CLIP_SIGMA, CLIP_SIGMA)
    return out


def _zscore_clip_bgfix(vol: np.ndarray) -> np.ndarray:
    """`_zscore_clip` with the brain mask taken from the volume, not from zero.

    Identical to `_zscore_clip` wherever background is already zero, which is
    every BraTS2020 and BraTS-Africa volume. RHUH-GBM ships z-score normalised
    with a negative constant background, where `v > 0` is a threshold at the
    volume mean rather than a brain mask. Used only for the sensitivity arm of
    section 4.4; the default path stays the verbatim mirror of the adapter's
    own preprocessing.
    """
    v = np.asarray(vol, dtype=np.float32)
    corners = np.array([v[i, j, k] for i in (0, -1) for j in (0, -1)
                        for k in (0, -1)], dtype=np.float64)
    vals, counts = np.unique(corners, return_counts=True)
    # min(.., 0) makes this provably identical to `v > 0` wherever the
    # background is already zero or positive
    bg = min(float(vals[counts.argmax()]), 0.0)
    brain = v > bg
    out = np.zeros_like(v)
    if brain.any():
        vals_b = v[brain]
        mu, sd = float(vals_b.mean()), float(vals_b.std()) + 1e-8
        out[brain] = np.clip((vals_b - mu) / sd, -CLIP_SIGMA, CLIP_SIGMA)
    return out


def prgb_volume(study_id: str) -> np.ndarray:
    """(3, H, W, Z) pseudo-RGB for the detector-free MedSAM3 path."""
    p = _dir(study_id) / "prgb.npy"
    if not p.exists():
        raise KeyError("This study was ingested before pseudo-RGB caching; re-upload it.")
    return np.load(p, mmap_mode="r")


def _dir(study_id: str) -> Path:
    if not study_id or len(study_id) > 40 or any(c not in "0123456789abcdef-" for c in study_id):
        raise ValueError("Invalid study id")
    d = STUDIES_DIR / study_id
    if not d.is_dir():
        raise KeyError("Study not found or expired")
    return d


def purge_expired() -> None:
    if not STUDIES_DIR.is_dir():
        return
    now = time.time()
    for d in STUDIES_DIR.iterdir():
        try:
            if d.is_dir() and now - d.stat().st_mtime > STUDY_TTL_SECONDS:
                shutil.rmtree(d, ignore_errors=True)
                log.info(f"Purged expired study {d.name}")
        except OSError:
            pass


def ingest(volume_paths: Dict[str, Path], seg_path: Optional[Path], name: str) -> StudyMeta:
    """Render every slice once and cache it alongside the ground truth, if given."""
    purge_expired()
    study_id = str(uuid.uuid4())
    d = STUDIES_DIR / study_id
    d.mkdir(parents=True, exist_ok=True)

    try:
        vols = {m: load_volume(volume_paths[m]) for m in MODALITY_ORDER}
        shapes = {v.shape for v in vols.values()}
        if len(shapes) != 1:
            raise ValueError(f"Modality volumes differ in shape: {sorted(str(s) for s in shapes)}")
        H, W, Z = next(iter(vols.values())).shape

        seg = None
        if seg_path is not None:
            seg = load_volume(seg_path)
            if seg.shape != (H, W, Z):
                raise ValueError(f"Segmentation shape {seg.shape} does not match the volumes {(H, W, Z)}")

        rendered = np.zeros((Z, H, W, 3), dtype=np.uint8)
        brain_px, tumor_px = [], []
        gt_masks = np.zeros((Z, H, W), dtype=bool) if seg is not None else None
        gt_boxes: Dict[str, Optional[list]] = {}

        for z in range(Z):
            rendered[z] = build_rgb_slice({m: vols[m][:, :, z] for m in MODALITY_ORDER})
            nz = np.zeros((H, W), dtype=bool)
            for m in MODALITY_ORDER:
                nz |= vols[m][:, :, z] > 0
            brain_px.append(int(nz.sum()))
            if seg is not None:
                # whole tumour, matching the study's WT target
                wt = seg[:, :, z] > 0
                gt_masks[z] = wt
                tumor_px.append(int(wt.sum()))
                gt_boxes[str(z)] = tight_bbox(wt)
            else:
                tumor_px.append(0)

        np.save(d / "slices.npy", rendered)

        # The detector pipeline wants the uint8 RGB above; MedSAM3 wants the
        # research project's pseudo-RGB, which is the same three modalities
        # z-scored over non-zero voxels and clipped to +/-3. Cache both so a
        # segmenter switch never needs the NIfTI files again.
        prgb = np.stack([_zscore_clip(vols[m]) for m in MODALITY_ORDER], axis=0)
        np.save(d / "prgb.npy", prgb.astype(np.float32))

        if gt_masks is not None:
            np.save(d / "gt_masks.npy", gt_masks)
            (d / "gt_boxes.json").write_text(json.dumps(gt_boxes), encoding="utf-8")
            # the raw labels too, so a tumour-core detector can be scored
            # against tumour core rather than against whole tumour
            np.save(d / "seg.npy", np.transpose(seg, (2, 0, 1)).astype(np.uint8))

        usable = [z for z in range(Z) if brain_px[z] >= MIN_BRAIN_PIXELS_FALLBACK]
        default_slice = int(np.argmax(tumor_px)) if (seg is not None and max(tumor_px) > 0) \
            else int(np.argmax(brain_px))

        meta = StudyMeta(study_id=study_id, name=name, total_slices=int(Z), height=int(H), width=int(W),
                         default_slice=default_slice, brain_px=brain_px, tumor_px=tumor_px,
                         usable_slices=usable, has_ground_truth=seg is not None)
        (d / "meta.json").write_text(json.dumps(meta.to_dict()), encoding="utf-8")
        log.info(f"Ingested {name} as {study_id}: {H}x{W}x{Z}, GT={'yes' if seg is not None else 'no'}")
        return meta
    except Exception:
        shutil.rmtree(d, ignore_errors=True)
        raise


def load_meta(study_id: str) -> dict:
    return json.loads((_dir(study_id) / "meta.json").read_text(encoding="utf-8"))


def slice_rgb(study_id: str, z: int) -> np.ndarray:
    arr = np.load(_dir(study_id) / "slices.npy", mmap_mode="r")
    return np.ascontiguousarray(arr[z])


# BraTS label values, same mapping the evaluation scripts use.
REGION_LABELS = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}


def _region_slice(study_id: str, z: int, region: str) -> Optional[np.ndarray]:
    p = _dir(study_id) / "seg.npy"
    if not p.exists():
        return None
    arr = np.load(p, mmap_mode="r")
    lab = np.ascontiguousarray(arr[z])
    return np.isin(lab, REGION_LABELS[region])


def gt_mask(study_id: str, z: int, region: str = "WT") -> Optional[np.ndarray]:
    if region != "WT":
        return _region_slice(study_id, z, region)
    p = _dir(study_id) / "gt_masks.npy"
    if not p.exists():
        return None
    arr = np.load(p, mmap_mode="r")
    return np.ascontiguousarray(arr[z])


def gt_box(study_id: str, z: int, region: str = "WT") -> Optional[List[float]]:
    if region != "WT":
        m = _region_slice(study_id, z, region)
        return tight_bbox(m) if m is not None else None
    p = _dir(study_id) / "gt_boxes.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8")).get(str(z))


def all_gt_boxes(study_id: str, region: str = "WT") -> Dict[int, List[List[float]]]:
    if region != "WT":
        p = _dir(study_id) / "seg.npy"
        if not p.exists():
            return {}
        arr = np.load(p, mmap_mode="r")
        out = {}
        for z in range(arr.shape[0]):
            b = tight_bbox(np.isin(np.ascontiguousarray(arr[z]), REGION_LABELS[region]))
            out[z] = [b] if b else []
        return out
    p = _dir(study_id) / "gt_boxes.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text(encoding="utf-8"))
    return {int(k): ([v] if v else []) for k, v in raw.items()}


def release(study_id: str) -> None:
    shutil.rmtree(_dir(study_id), ignore_errors=True)


def clamp_slice(meta: dict, z: Optional[int]) -> int:
    total = meta["total_slices"]
    if z is None or not (0 <= int(z) < total):
        return int(meta["default_slice"])
    return int(z)
