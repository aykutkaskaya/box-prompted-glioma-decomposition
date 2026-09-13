"""Measure what a deployment of each arm has to hold, in model tensors.

Not file sizes. Two of the five checkpoints are training checkpoints: the
released Medical SAM 3 and LoRA files carry optimizer state that a deployment
never holds, and are 9 555 and 18 MiB against 3 215 and 6 MiB of weights. The
box arms' files carry weights only. Taking file sizes throughout therefore
counted two different things and overstated the memory ratio threefold, which
is what the cost table reported until this script replaced it.

Writes reports/weight_sizes.json, which audit_numbers.py reads -- the audit
runs without torch, so the measurement lives here.

    python scripts/weight_sizes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

import os  # noqa: E402

MEDSAM3_DIR = Path(os.environ.get("MEDSAM3_DIR", "medsam3-not-set"))
OUT = DRIVE_ROOT / "reports" / "weight_sizes.json"

FILES = {
    "detector": DRIVE_ROOT / "experiments" /
                "run_037_icarb_alpha_050_w30_clsfl" / "checkpoints" / "best.pt",
    "sam1": DRIVE_ROOT / "demo" / "rt-detr-sam-pipeline" / "models" /
            "sam_vit_b_01ec64.pth",
    "sam2": DRIVE_ROOT / "shared_artifacts" / "sam2" / "sam2.1_l.pt",
    "lora": DRIVE_ROOT / "models" / "lora" / "seed_42" / "ckpt_epoch3_final.pt",
    "medsam": MEDSAM3_DIR / "models/medical_sam3/checkpoint_2D.pt",
}
# where the weights sit when the checkpoint is a dict
MODEL_KEYS = ("model", "model_state_dict", "state_dict", "lora_state_dict")


def tensor_bytes(obj) -> int:
    if hasattr(obj, "numel"):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def main() -> int:
    import torch

    missing = [k for k, p in FILES.items() if not p.exists()]
    if missing:
        print(f"  missing: {missing}; nothing written")
        return 1

    out = {}
    for name, path in FILES.items():
        obj = torch.load(path, map_location="cpu", weights_only=False)
        weights = 0
        if isinstance(obj, dict):
            for k in MODEL_KEYS:
                if k in obj:
                    weights = tensor_bytes(obj[k])
                    break
        if not weights:
            # an ultralytics checkpoint keeps its weights in a module object,
            # and its file carries no optimizer state, so the file stands
            weights = tensor_bytes(obj) or path.stat().st_size
        out[name] = {"weights_bytes": weights,
                     "file_bytes": path.stat().st_size}
        mib = 1024 ** 2
        print(f"  {name:9s} weights {weights / mib:8.0f} MiB   "
              f"file {path.stat().st_size / mib:8.0f} MiB")

    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {OUT.relative_to(DRIVE_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
