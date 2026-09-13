"""The fourth arm: the reference box handed to the detector-free model.

The paper's residual term compares a text-prompted foundation model against a
box-prompted segmenter, so it confounds prompt type with architecture. Giving
the same reference box to MedSAM3 separates them: against the detector-free arm
it isolates prompt type (same model, different prompt), and against the
oracle-box arm it isolates architecture (same prompt, different model).

The box mask is a padding mask -- 0 means the box is valid. Setting it to 1
silently drops the box and returns the text-only result, which is a null this
script was written to avoid producing by accident.

    RAW_DIR=<cohort dir> python scripts/fourth_arm.py --patients-file <list> --out <jsonl>
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.brats import MODALITY_ORDER                              # noqa: E402
from core.metrics import tight_bbox                                # noqa: E402
from core.study import _zscore_clip_bgfix                          # noqa: E402
from full_validation import (load_patient, log, region_mask,       # noqa: E402
                             slice_sets, volumetric, _zscore_clip)

BASE = "http://127.0.0.1:8020"


def box_arm(prgb, boxes, adapter):
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(prgb, dtype=np.float32), allow_pickle=False)
    r = requests.post(f"{BASE}/segment/volume_box", json={
        "prgb_npy_b64": base64.b64encode(buf.getvalue()).decode(),
        "boxes": boxes, "adapter": adapter}, timeout=3600)
    r.raise_for_status()
    res = r.json()
    p = np.load(io.BytesIO(base64.b64decode(res["packed_npy_b64"])), allow_pickle=False)
    sh = tuple(res["shape"])
    return np.unpackbits(p)[:int(np.prod(sh))].reshape(sh).astype(bool), res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients-file", required=True)
    ap.add_argument("--adapter", default="seed_42")
    ap.add_argument("--region", default="WT")
    ap.add_argument("--bgfix", action="store_true",
                    help="take the background from the volume rather than "
                         "assuming zero. Required on RHUH-GBM, which ships "
                         "z-score normalised with a negative constant "
                         "background: there the default rule thresholds at the "
                         "volume mean and discards half the brain (see the "
                         "preprocessing fault reported in the manuscript). "
                         "Bit-identical wherever background is already zero.")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                done.add(json.loads(line)["patient"])
        log(f"resuming: {len(done)} already done")

    pats = [x.strip() for x in io.open(a.patients_file, encoding="utf-8") if x.strip()]
    todo = [p for p in pats if p not in done]
    log(f"{len(todo)} patients (of {len(pats)}), adapter {a.adapter}, region {a.region}")

    t0 = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, pid in enumerate(todo, 1):
            vols, seg = load_patient(pid)
            gt = region_mask(seg, a.region)
            usable, _ = slice_sets(vols, seg, 1)
            z0, z1 = min(usable), max(usable) + 1
            norm = _zscore_clip_bgfix if a.bgfix else _zscore_clip
            prgb = np.stack([norm(vols[m]) for m in MODALITY_ORDER],
                            axis=0)[:, :, :, z0:z1]
            H, W = gt.shape[0], gt.shape[1]
            boxes = {}
            for z in usable:
                g = gt[:, :, z]
                if not g.any():
                    continue
                x0, y0, x1, y1 = tight_bbox(g)
                boxes[str(z - z0)] = [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H,
                                      (x1 - x0) / W, (y1 - y0) / H]
            if not boxes:
                continue
            mask, res = box_arm(prgb, boxes, a.adapter)
            preds = {z: mask[:, :, z - z0] for z in usable}
            rec = {"patient": pid, "region": a.region, "adapter": a.adapter,
                   "n_slices": len(usable), "n_boxes": len(boxes),
                   "arms": {f"oraclebox_medsam3:{a.adapter}":
                            dict(volumetric(preds, gt, usable),
                                 elapsed_ms=res.get("elapsed_ms"))}}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            el = (time.time() - t0) / 60
            log(f"  [{i}/{len(todo)}] {pid} "
                f"Dice {rec['arms'][f'oraclebox_medsam3:{a.adapter}']['vol_dice']:.4f} "
                f"({el:.0f} min, ~{el / i * (len(todo) - i):.0f} left)")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
