"""Paths and tunables for the src backend.

Everything is resolved relative to the drive root so the app runs from a
checkout without editing code; each path can still be overridden by an
environment variable.
"""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
DEMOV2_DIR = BACKEND_DIR.parent
DRIVE_ROOT = Path(os.environ.get("DRIVE_ROOT", DEMOV2_DIR.parent))

EXPERIMENTS_DIR = Path(os.environ.get("EXPERIMENTS_DIR", DRIVE_ROOT / "experiments"))
# Seed replicates and the tumour-core detectors. Same run ids as the originals,
# so the registry has to key on more than the id.
REPEATED_DIR = Path(os.environ.get("REPEATED_DIR", DRIVE_ROOT / "experiments_repeated"))
SHARED_ARTIFACTS = Path(os.environ.get("SHARED_ARTIFACTS", DRIVE_ROOT / "shared_artifacts"))
SAM_CHECKPOINT = Path(os.environ.get("SAM_CHECKPOINT", SHARED_ARTIFACTS / "sam" / "sam_vit_b_01ec64.pth"))
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_b")
# SAM 2.x weights live here rather than the working directory: Ultralytics
# resolves bare filenames against cwd and will happily re-download 428 MB
# every time the process is started from somewhere new.
SAM2_DIR = Path(os.environ.get("SAM2_DIR", SHARED_ARTIFACTS / "sam2"))

CACHE_DIR = Path(os.environ.get("DEMOV2_CACHE", BACKEND_DIR / "cache"))
STUDIES_DIR = CACHE_DIR / "studies"
EXPORTS_DIR = CACHE_DIR / "exports"

STUDY_TTL_SECONDS = int(os.environ.get("STUDY_TTL_SECONDS", 6 * 60 * 60))

# How many RT-DETR checkpoints to hold on the GPU at once. Each is ~66 MB on
# disk; the cap exists so a run-comparison sweep cannot exhaust an 8 GB card.
MAX_CACHED_DETECTORS = int(os.environ.get("MAX_CACHED_DETECTORS", 4))

# Detection is always run at this floor so a single forward pass can be
# re-thresholded offline for the operating-point explorer.
SCORE_FLOOR = 0.01
# Brain-coverage floor the training slice generator used (min_brain_pixels).
MIN_BRAIN_PIXELS_FALLBACK = 500
# Frozen protocol value, not an optimum. Derived on the validation split,
# where F1 peaks at 0.46 (0.8626 val / 0.8537 test) on a flat 0.45-0.65 band;
# 0.55 sits inside it and is what all three cohorts were evaluated at.
DEFAULT_CONF = 0.55
MATCH_IOU = 0.5              # the IoU used for TP/FP matching in the paper
# The three-cohort comparison ran on SAM 2.1 Large, so that is what the demo
# offers first; picking SAM 1 ViT-B gives the 37-run ablation's numbers instead.
DEFAULT_SEGMENTER = os.environ.get("DEFAULT_SEGMENTER", "sam2.1_l")
# The notebook's SAM evaluation (cell 37, `sam_mask_from_box`) prompts with the
# raw box and takes the single mask: no padding, multimask_output=False. Demo v1
# diverged on both counts, which is why its masks never matched the reported
# numbers. Keep parity here.
SAM_BOX_PADDING = 0.0
SAM_MULTIMASK = False

for _d in (CACHE_DIR, STUDIES_DIR, EXPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
