"""HD95 and NSD for the two box arms, over a cohort.

The manuscript reports overlap only, cites Metrics Reloaded for the advice that
a distance metric should accompany it, and then shows in its own limitations
that Dice missed an 89 mm HD95 defect in one adapter. A referee will ask for
this, and the ask is right: two masks with the same Dice can differ by a
component sitting far from the lesion, which is what matters clinically.

Both metrics are computed on the reconstructed volume, not per slice, because a
distance in the through-plane direction is a real distance and a per-slice
figure cannot see it. BraTS volumes are 1 mm isotropic, so voxel units are
millimetres.

HD95 is the symmetric 95th percentile: the 95th percentile of the distances
from each surface voxel of one mask to the nearest surface voxel of the other,
taken in both directions, and the larger of the two reported. NSD is the
normalised surface Dice at a 2 mm tolerance -- the fraction of surface voxels,
over both surfaces, that lie within tolerance of the other surface.

An empty prediction has no surface. Rather than score it 0 or drop it, those
patients are counted and reported separately, since a pipeline that predicts
nothing is exactly the failure mode this paper is about and averaging it away
would hide it.

    RAW_DIR=<cohort dir> python scripts/boundary_metrics.py \
        --patients-file <list> --out reports/validation/<cohort>_boundary.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import models, segmenters                      # noqa: E402
from core.brats import MODALITY_ORDER, build_rgb_slice   # noqa: E402
from core.metrics import tight_bbox                      # noqa: E402
from full_validation import (DEFAULT_CONF, DEFAULT_RUN, SCORE_FLOOR,  # noqa: E402
                             load_patient, log, region_mask, slice_sets,
                             test_patients)

SEG_ID = "sam2.1_l"
TOL_MM = 2.0


def surface(mask: np.ndarray) -> np.ndarray:
    """Voxels of the mask that touch the outside, in 6-connectivity."""
    if not mask.any():
        return mask
    er = ndimage.binary_erosion(mask, ndimage.generate_binary_structure(3, 1),
                                border_value=0)
    return mask & ~er


def boundary(pred: np.ndarray, gt: np.ndarray) -> dict:
    """HD95 in mm and NSD at TOL_MM, or None where a surface is missing."""
    sp, sg = surface(pred), surface(gt)
    if not sp.any() or not sg.any():
        return {"hd95": None, "nsd": None,
                "empty": "prediction" if not sp.any() else "reference"}
    # distance from every voxel to the nearest surface voxel of the other mask
    dp = ndimage.distance_transform_edt(~sg)[sp]   # pred surface -> gt surface
    dg = ndimage.distance_transform_edt(~sp)[sg]   # gt surface -> pred surface
    hd95 = float(max(np.percentile(dp, 95), np.percentile(dg, 95)))
    nsd = float(((dp <= TOL_MM).sum() + (dg <= TOL_MM).sum())
                / (dp.size + dg.size))
    return {"hd95": round(hd95, 2), "nsd": round(nsd, 4), "empty": None}


def one_patient(pid: str, run_id: int, conf: float, region: str) -> dict:
    vols, seg_raw = load_patient(pid)
    gt = region_mask(seg_raw, region)
    usable, _ = slice_sets(vols, gt, 1)

    shape = gt.shape
    oracle = np.zeros(shape, bool)
    pipeline = np.zeros(shape, bool)
    t0 = time.time()
    for z in usable:
        g = gt[:, :, z]
        rgb = build_rgb_slice({m: vols[m][:, :, z] for m in MODALITY_ORDER})

        box = tight_bbox(g) if g.any() else None
        if box:
            m = segmenters.segment(SEG_ID, rgb, [box])
            if len(m):
                oracle[:, :, z] = m[0]

        raw = models.detect(run_id, rgb, score_floor=SCORE_FLOOR)
        boxes = [raw["boxes"][i] for i, s in enumerate(raw["scores"]) if s >= conf]
        if boxes:
            pipeline[:, :, z] = segmenters.segment(SEG_ID, rgb, boxes).any(axis=0)

    return {"patient": pid, "region": region, "run_id": run_id, "conf": conf,
            "segmenter": SEG_ID, "tolerance_mm": TOL_MM,
            "gt_voxels": int(gt.sum()),
            "oracle": boundary(oracle, gt),
            "pipeline": boundary(pipeline, gt),
            "elapsed_s": round(time.time() - t0, 1)}


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
        f"region={a.region} segmenter={SEG_ID} tolerance={TOL_MM}mm")
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
            log(f"  [{i}/{len(todo)}] {pid} "
                f"oracle HD95 {rec['oracle']['hd95']}  "
                f"pipeline HD95 {rec['pipeline']['hd95']}  "
                f"({el:.0f} min elapsed, ~{el / i * (len(todo) - i):.0f} min left)")
    log(f"done in {(time.time() - t_start) / 60:.1f} min -> {out}")


if __name__ == "__main__":
    main()
