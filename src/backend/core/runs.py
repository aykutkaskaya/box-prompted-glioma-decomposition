"""Registry over the trained runs in drive/experiments and experiments_repeated.

Read straight from the run directories rather than a copied summary table, so
what the demo offers can never drift from what was actually trained.

Run ids are not unique. The sub-region detectors were trained by the same
notebook and carry the same ids as their whole-tumour counterparts -- run 37
exists three times over, once as WT and twice more as TC at different seeds,
with identical run_id, run_name, experiment_group and registry_hash. The only
field that separates them is the dataset path in resolved_config.json
(media/ vs media_tc/). So every run gets a `key`, unique across both trees,
and that is what callers address it by; `run_id` is kept for display.
"""
from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from config import EXPERIMENTS_DIR, REPEATED_DIR


def _read_json(path: Path) -> Optional[dict]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _target(resolved: Optional[dict]) -> str:
    """WT or TC, from the dataset the run was trained on."""
    yaml = ((resolved or {}).get("dataset_yaml") or "").replace("\\", "/")
    return "TC" if "/media_tc/" in yaml else "WT"


def _entry(d: Path, variant: str) -> Optional[Dict[str, Any]]:
    ckpt = d / "checkpoints" / "best.pt"
    cfg = _read_json(d / "config" / "requested_config.json")
    if not ckpt.exists() or cfg is None:
        return None
    resolved = _read_json(d / "config" / "resolved_config.json")
    summary = _read_json(d / "summary" / "run_summary.json")

    test = (summary or {}).get("test", {})
    geom = (summary or {}).get("geometry", {})
    sam = (summary or {}).get("sam", {})
    params = cfg.get("loss_params") or {}
    run_id = cfg["run_id"]
    target = _target(resolved)
    seed = (resolved or {}).get("seed", cfg.get("seed"))

    label = _label(cfg, params)
    if variant:
        label = f"{label} · {target} seed {seed}"

    return {
        "key": f"{variant}/{run_id}" if variant else str(run_id),
        "run_id": run_id,
        "variant": variant or None,
        "target": target,
        "seed": seed,
        "slug": d.name,
        "label": label,
        "group": cfg.get("experiment_group"),
        "loss": cfg.get("loss_name"),
        "alpha": params.get("alpha"),
        "overlap_weight": cfg.get("overlap_loss_weight"),
        "cls_loss": cfg.get("cls_loss"),
        "notes": cfg.get("notes"),
        "checkpoint": str(ckpt),
        "metrics": {
            "precision": test.get("precision"),
            "recall": test.get("recall"),
            "f1": test.get("f1"),
            "map50": test.get("map50"),
            "map50_95": test.get("map50_95"),
            "box_iou": geom.get("iou"),
            "pred_coverage": geom.get("pred_coverage"),
            "target_coverage": geom.get("target_coverage"),
            "sam_overall_dice": sam.get("overall_dice"),
            "sam_conditional_dice": sam.get("conditional_dice"),
            "sam_gtbox_dice": sam.get("gtbox_dice"),
        },
    }


@lru_cache(maxsize=1)
def list_runs() -> List[Dict[str, Any]]:
    """Every completed run that still has a checkpoint on disk.

    The original tree first, then the replicates grouped by their directory.
    """
    runs: List[Dict[str, Any]] = []

    if EXPERIMENTS_DIR.is_dir():
        for d in sorted(EXPERIMENTS_DIR.iterdir()):
            if d.is_dir() and d.name.startswith("run_"):
                e = _entry(d, "")
                if e:
                    runs.append(e)

    if REPEATED_DIR.is_dir():
        for variant in sorted(REPEATED_DIR.iterdir()):
            if not variant.is_dir():
                continue
            for d in sorted(variant.iterdir()):
                if d.is_dir() and d.name.startswith("run_"):
                    e = _entry(d, variant.name)
                    if e:
                        runs.append(e)
    return runs


def _label(cfg: dict, params: dict) -> str:
    """Short human label, e.g. 'IC-Arb a=0.50 w3.0 (fl)'."""
    names = {
        "giou": "GIoU", "diou": "DIoU", "ciou": "CIoU", "eiou": "EIoU", "siou": "SIoU",
        "alpha_iou": "Alpha-IoU", "wiou": "WIoU", "piou": "PIoU", "icarb": "IC-Arb",
        "dice": "Dice", "focal_tversky": "Focal-Tversky", "boundary": "Boundary",
        "gsl": "GSL", "lovasz_hinge": "Lovasz-Hinge",
    }
    base = names.get(cfg.get("loss_name"), cfg.get("loss_name", "?"))
    bits = [base]
    if "alpha" in params:
        bits.append(f"a={params['alpha']:.2f}")
    w = cfg.get("overlap_loss_weight")
    if w is not None:
        bits.append(f"w{w:g}")
    cls = cfg.get("cls_loss")
    if cls and cls != "vfl":
        bits.append(f"({cls})")
    return " ".join(bits)


def get_run(key: str) -> Optional[Dict[str, Any]]:
    """Look up by key. A bare id still works when it is unambiguous."""
    key = str(key)
    for r in list_runs():
        if r["key"] == key:
            return r
    if re.fullmatch(r"\d+", key):
        hits = [r for r in list_runs() if not r["variant"] and str(r["run_id"]) == key]
        if len(hits) == 1:
            return hits[0]
    return None


def default_run_key() -> str:
    """Best F1 among the original runs; falls back to the first one.

    Deliberately restricted to the original tree. The replicates exist to show
    seed spread, and picking whichever of them happened to score highest would
    be exactly the single-seed selection the paper argues against.
    """
    runs = [r for r in list_runs() if not r["variant"]]
    if not runs:
        raise RuntimeError(f"No runs with checkpoints found under {EXPERIMENTS_DIR}")
    scored = [r for r in runs if r["metrics"].get("f1") is not None]
    if not scored:
        return runs[0]["key"]
    return max(scored, key=lambda r: r["metrics"]["f1"])["key"]
