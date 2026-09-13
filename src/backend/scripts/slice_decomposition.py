"""Split the detector-stage term into what it is made of.

The term is oracle-box minus pipeline, and the paper is careful to say it
aggregates several things: slices where the detector emitted nothing, boxes on
the wrong structure, and boxes that are correct but shaped differently. That
caveat can be turned into a measurement.

Define a hybrid arm that keeps the pipeline's own mask wherever the detector
fired and substitutes the oracle-box mask wherever it did not. Then

    oracle - pipeline  =  (oracle - hybrid)  +  (hybrid - pipeline)
                          box geometry          missed detection

and the two sum exactly, because the hybrid is the pipeline on one part of the
slice set and the oracle on the other. Neither term is a cause on its own --
"box geometry" still contains boxes placed on the wrong structure -- but the
split separates what a better recall threshold could fix from what it could
not, and the paper currently asserts the balance rather than showing it.

Both arms are computed in one pass so the slice, the RGB and the detection are
shared. Per-slice counts are written out, so any later question about the
partition can be answered without running the segmenter again.

    RAW_DIR=<cohort dir> python scripts/slice_decomposition.py \
        --patients-file <list> --out reports/validation/<cohort>_slices.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import models, segmenters                      # noqa: E402
from core.brats import MODALITY_ORDER, build_rgb_slice   # noqa: E402
from core.metrics import tight_bbox                      # noqa: E402
from full_validation import (DEFAULT_CONF, DEFAULT_RUN, SCORE_FLOOR,  # noqa: E402
                             load_patient, log, region_mask, slice_sets,
                             test_patients)

SEG_ID = "sam2.1_l"


def counts(mask, g):
    """The three sums a pooled Dice needs, for one slice."""
    if mask is None:
        return 0, 0
    return int((mask & g).sum()), int(mask.sum())


def one_patient(pid: str, run_id: int, conf: float, region: str) -> dict:
    vols, seg_raw = load_patient(pid)
    gt = region_mask(seg_raw, region)
    usable, _ = slice_sets(vols, gt, 1)

    rows = []
    t0 = time.time()
    for z in usable:
        g = gt[:, :, z]
        rgb = build_rgb_slice({m: vols[m][:, :, z] for m in MODALITY_ORDER})

        # oracle: the tight ground-truth box, only where the region exists
        gt_box = tight_bbox(g) if g.any() else None
        if gt_box:
            om = segmenters.segment(SEG_ID, rgb, [gt_box])
            om = om[0] if len(om) else None
        else:
            om = None

        # pipeline: whatever the detector proposes, on every usable slice
        raw = models.detect(run_id, rgb, score_floor=SCORE_FLOOR)
        boxes = [raw["boxes"][i] for i, s in enumerate(raw["scores"]) if s >= conf]
        pm = segmenters.segment(SEG_ID, rgb, boxes).any(axis=0) if boxes else None

        oi, op = counts(om, g)
        pi, pp = counts(pm, g)
        rows.append({"z": int(z), "gt": int(g.sum()), "boxes": len(boxes),
                     "oracle_i": oi, "oracle_p": op,
                     "pipe_i": pi, "pipe_p": pp})

    return {"patient": pid, "region": region, "run_id": run_id, "conf": conf,
            "segmenter": SEG_ID, "n_slices": len(usable),
            "elapsed_s": round(time.time() - t0, 1), "slices": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", type=int, default=DEFAULT_RUN)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--region", default="WT")
    ap.add_argument("--patients-file", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # resume: a patient already written is not recomputed
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    done.add(json.loads(line)["patient"])
                except Exception:
                    pass
        log(f"resuming: {len(done)} patients already in {out.name}")

    patients = ([x.strip() for x in Path(a.patients_file).read_text().splitlines()
                 if x.strip()] if a.patients_file else test_patients())
    if a.limit:
        patients = patients[:a.limit]
    todo = [p for p in patients if p not in done]

    models.init(os.environ.get("FORCE_DEVICE"))
    segmenters.set_device(models.device())
    log(f"device={models.device()} run={a.run_id} conf={a.conf} "
        f"region={a.region} segmenter={SEG_ID}")
    log(f"{len(todo)} patients to process (of {len(patients)})")

    t_start = time.time()
    with out.open("a", encoding="utf-8") as fh:
        for i, pid in enumerate(todo, 1):
            try:
                rec = one_patient(pid, a.run_id, a.conf, a.region)
            except Exception as e:
                log(f"  [{i}/{len(todo)}] {pid} FAILED: {e}")
                fh.write(json.dumps({"patient": pid, "error": str(e)}) + "\n")
                fh.flush()
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            el = (time.time() - t_start) / 60
            left = el / i * (len(todo) - i)
            log(f"  [{i}/{len(todo)}] {pid} "
                f"{rec['n_slices']} slices  ({el:.0f} min elapsed, ~{left:.0f} min left)")
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
