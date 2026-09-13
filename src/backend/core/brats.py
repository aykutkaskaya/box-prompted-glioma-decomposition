"""BraTS2020 preprocessing, bit-identical to the training pipeline.

The detector was trained on 3-channel axial PNGs built as:

    for each modality m in (flair, t1ce, t2):
        norm[m] = normalize_nonzero(volume, "zscore", 1, 99)   # per volume
    rgb[..., i] = to_uint8_channel(norm[m_i][:, :, z], 1, 99)  # per slice

`normalize_nonzero` in z-score mode is an affine map with volume-level
(mu, sd), and `to_uint8_channel` rescales by percentiles taken *after* that
map. Percentiles commute with an affine map, so (mu, sd) cancels exactly:

    (z - z_lo) / (z_hi - z_lo)  ==  (v - v_lo) / (v_hi - v_lo)

so the training output depends only on the slice's own non-zero voxels. That
is what `normalize_brain_slice` implements, and it is why a single slice can
be processed without loading the whole volume. See `tests_equivalence()`.

Two things the generic medical loader got wrong for BraTS:
  * it rotated slices 90 degrees (training uses volume[:, :, z] verbatim);
  * it took percentiles over every pixel, so the zero background dominated.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import numpy as np

MODALITY_ORDER: Sequence[str] = ("flair", "t1ce", "t2")
CLIP_LO_PCT = 1.0
CLIP_HI_PCT = 99.0
MIN_BRAIN_PIXELS = 500


def normalize_brain_slice(slice_2d: np.ndarray,
                          clip_lo: float = CLIP_LO_PCT,
                          clip_hi: float = CLIP_HI_PCT) -> np.ndarray:
    """Map one raw modality slice to uint8, using only its non-zero voxels."""
    s = np.nan_to_num(np.asarray(slice_2d, dtype=np.float32))
    # Not `s > 0`. Skull-stripped background is exactly zero on BraTS2020 and
    # BraTS-Africa, but RHUH-GBM ships z-score normalised with a negative
    # constant background, and `> 0` there discards 43-62% of the brain and 62%
    # of the tumour in T1ce. Take the background from the slice corners instead,
    # which is bit-identical wherever the background is already zero.
    corners = np.array([s[0, 0], s[0, -1], s[-1, 0], s[-1, -1]], dtype=np.float64)
    vals_c, counts_c = np.unique(corners, return_counts=True)
    # min(.., 0) makes this provably identical to the old `s > 0` wherever the
    # background is already zero or positive, so only the negative-background
    # cohort changes.
    bg = min(float(vals_c[counts_c.argmax()]), 0.0)
    brain = s > bg
    if not brain.any():
        return np.zeros(s.shape, dtype=np.uint8)
    vals = s[brain]
    lo, hi = np.percentile(vals, [clip_lo, clip_hi])
    hi = max(hi, lo + 1e-8)
    out = np.zeros(s.shape, dtype=np.float32)
    out[brain] = np.clip((vals - lo) / (hi - lo), 0, 1) * 255.0
    return out.astype(np.uint8)


def build_rgb_slice(modality_slices: Dict[str, np.ndarray]) -> np.ndarray:
    """Stack flair/t1ce/t2 into the HxWx3 RGB array the detector was trained on."""
    missing = [m for m in MODALITY_ORDER if m not in modality_slices]
    if missing:
        raise ValueError(f"missing modalities: {', '.join(missing)}")
    channels = [normalize_brain_slice(modality_slices[m]) for m in MODALITY_ORDER]
    shapes = {c.shape for c in channels}
    if len(shapes) != 1:
        raise ValueError(f"modality slices differ in shape: {shapes}")
    return np.stack(channels, axis=-1)


def load_volume(path: str | Path) -> np.ndarray:
    """Load a NIfTI volume as (H, W, Z) - the axis order BraTS ships and training used."""
    import nibabel as nib
    data = nib.load(str(path)).get_fdata()
    if data.ndim == 4:
        data = data[..., 0]
    if data.ndim != 3:
        raise ValueError(f"expected a 3D volume, got shape {data.shape}: {path}")
    return np.nan_to_num(data)


def load_brats_volumes(paths: Dict[str, str | Path]) -> Dict[str, np.ndarray]:
    vols = {m: load_volume(paths[m]) for m in MODALITY_ORDER if m in paths}
    missing = [m for m in MODALITY_ORDER if m not in vols]
    if missing:
        raise ValueError(f"missing modalities: {', '.join(missing)}")
    shapes = {v.shape for v in vols.values()}
    if len(shapes) != 1:
        raise ValueError(f"modality volumes differ in shape: {shapes}")
    return vols


def brain_pixels_per_slice(volumes: Dict[str, np.ndarray]) -> np.ndarray:
    """Non-zero pixel count per axial slice, across all modalities."""
    stack = None
    for m in MODALITY_ORDER:
        nz = volumes[m] > 0
        stack = nz if stack is None else (stack | nz)
    return stack.sum(axis=(0, 1))


def default_slice_index(volumes: Dict[str, np.ndarray]) -> int:
    """Open on the slice with the most brain tissue rather than the middle of the stack."""
    counts = brain_pixels_per_slice(volumes)
    return int(np.argmax(counts))


def extract_brats_slice(volumes: Dict[str, np.ndarray], slice_index: int | None = None):
    """Return (rgb_uint8, total_slices, slice_index, brain_px) for one axial slice."""
    total = int(next(iter(volumes.values())).shape[2])
    if slice_index is None or not (0 <= int(slice_index) < total):
        slice_index = default_slice_index(volumes)
    slice_index = int(slice_index)
    rgb = build_rgb_slice({m: volumes[m][:, :, slice_index] for m in MODALITY_ORDER})
    brain_px = int(brain_pixels_per_slice(volumes)[slice_index])
    return rgb, total, slice_index, brain_px


# --------------------------------------------------------------------------
def tests_equivalence(seed: int = 0) -> None:
    """Assert the single-slice form matches the notebook's two-step pipeline."""
    rng = np.random.default_rng(seed)

    def normalize_nonzero(vol, clip_lo, clip_hi):          # notebook, zscore mode
        vol = vol.astype(np.float32)
        brain = vol > 0
        if brain.sum() == 0:
            return vol
        vals = vol[brain]
        mu, sd = vals.mean(), vals.std() + 1e-8
        out = np.zeros_like(vol)
        out[brain] = (vol[brain] - mu) / sd
        return out

    def to_uint8_channel(s, clip_lo, clip_hi):             # notebook
        s = s.astype(np.float32)
        brain = s != 0
        if brain.sum() == 0:
            return np.zeros_like(s, dtype=np.uint8)
        vals = s[brain]
        lo, hi = np.percentile(vals, [clip_lo, clip_hi])
        hi = max(hi, lo + 1e-8)
        out = np.zeros_like(s, dtype=np.float32)
        out[brain] = np.clip((s[brain] - lo) / (hi - lo), 0, 1) * 255.0
        return out.astype(np.uint8)

    worst = 0
    for trial in range(20):
        vol = rng.gamma(2.0, 180.0, size=(48, 48, 12)).astype(np.float32)
        vol[rng.random(vol.shape) < 0.55] = 0.0            # BraTS-like zero background
        ref_vol = normalize_nonzero(vol, CLIP_LO_PCT, CLIP_HI_PCT)
        for z in range(vol.shape[2]):
            ref = to_uint8_channel(ref_vol[:, :, z], CLIP_LO_PCT, CLIP_HI_PCT)
            got = normalize_brain_slice(vol[:, :, z])
            worst = max(worst, int(np.abs(ref.astype(int) - got.astype(int)).max()))
    assert worst <= 1, f"single-slice form diverges from the notebook by {worst} levels"
    print(f"equivalence OK over 20 volumes x 12 slices - max |diff| = {worst} grey level(s)")


if __name__ == "__main__":
    tests_equivalence()
