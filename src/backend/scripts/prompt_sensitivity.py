"""Does the detector-free arm's prompt protocol matter?

The deployed arm asks for the three regions by name -- "Whole tumor",
"Tumor core", "Enhancing tissue" -- and two of those three are absent from the
model's training query vocabulary. The adapter's own source project instead
recommends prompting for the three raw BraTS classes, whose names are all in
vocabulary, and composing the regions by union. That recommendation was never
run on a cohort, so the manuscript reports one protocol and cannot say what the
other would have given. This runs both on the same patients, same adapter, same
threshold, and writes the comparison.

    python scripts/prompt_sensitivity.py --cohort clean
    python scripts/prompt_sensitivity.py --cohort rhuh --patients-file ...
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
from core import sidecar                                   # noqa: E402
from core.brats import MODALITY_ORDER                      # noqa: E402
from full_validation import (load_patient, log, region_mask,  # noqa: E402
                             slice_sets, volumetric, _zscore_clip)

BASE = "http://127.0.0.1:8020"
REGIONS = ["WT", "TC", "ET"]


def composed(prgb, adapter, batch_size=2):
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(prgb, dtype=np.float32), allow_pickle=False)
    r = requests.post(f"{BASE}/segment/volume_composed", json={
        "prgb_npy_b64": base64.b64encode(buf.getvalue()).decode(),
        "adapter": adapter, "batch_size": batch_size}, timeout=3600)
    r.raise_for_status()
    res = r.json()
    out = {}
    for region, blob in res["regions"].items():
        packed = np.load(io.BytesIO(base64.b64decode(blob["packed_npy_b64"])),
                         allow_pickle=False)
        shape = tuple(blob["shape"])
        out[region] = np.unpackbits(packed)[:int(np.prod(shape))].reshape(shape).astype(bool)
    return out, res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients-file", required=True)
    ap.add_argument("--adapter", default="seed_42")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
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
    if a.limit:
        pats = pats[:a.limit]
    todo = [p for p in pats if p not in done]
    log(f"{len(todo)} patients (of {len(pats)}), adapter {a.adapter}")

    t_start = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, pid in enumerate(todo, 1):
            vols, seg = load_patient(pid)
            usable, _ = slice_sets(vols, seg, 1)
            z0, z1 = min(usable), max(usable) + 1
            prgb = np.stack([_zscore_clip(vols[m]) for m in MODALITY_ORDER],
                            axis=0)[:, :, :, z0:z1]
            rec = {"patient": pid, "adapter": a.adapter, "n_slices": len(usable),
                   "arms": {}}
            dep = sidecar.segment_volume(prgb, a.adapter, REGIONS, batch_size=2)
            cmp_masks, meta = composed(prgb, a.adapter)
            for lbl, masks in (("deployed", dep["masks"]), ("composed", cmp_masks)):
                d = {}
                for region in REGIONS:
                    vol = masks.get(region)
                    if vol is None:
                        continue
                    gt = region_mask(seg, region)
                    d[region] = volumetric({z: vol[:, :, z - z0] for z in usable},
                                           gt, usable)
                rec["arms"][lbl] = d
            rec["queries"] = meta.get("queries")
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            el = (time.time() - t_start) / 60
            dw = rec["arms"]["deployed"].get("WT", {}).get("vol_dice")
            cw = rec["arms"]["composed"].get("WT", {}).get("vol_dice")
            log(f"  [{i}/{len(todo)}] {pid}  WT deployed {dw}  composed {cw}  "
                f"({el:.0f} min, ~{el / i * (len(todo) - i):.0f} left)")
    log(f"done -> {out}")


if __name__ == "__main__":
    main()
