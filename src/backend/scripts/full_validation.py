"""Full held-out validation: does the detector earn its place, and does a newer
segmenter raise the ceiling?

Three arms, all scored the same way on the same 55 test patients:

  oracle    ground-truth box -> segmenter. Not deployable; it is the ceiling a
            perfect detector could reach. --oracle-regions extends it to TC and
            ET, which is how the box approach's sub-region ceiling gets measured
            without a sub-region detector.
  pipeline  RT-DETR box -> segmenter. The real system.
  direct    MedSAM3 (+LoRA) with no detector and no box at all, via the sidecar.

Per patient it reports volumetric Dice/IoU over the whole tumour, plus detection
counts for the arms that have boxes. Results stream to a JSONL as they finish,
so the run is resumable: re-running skips patients already recorded.

    python scripts/full_validation.py --arms oracle,pipeline,direct
    python scripts/full_validation.py --stride 2          # every other slice
    python scripts/full_validation.py --arms oracle --oracle-regions WT,TC,ET
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from config import DEFAULT_CONF, DRIVE_ROOT, MATCH_IOU, SCORE_FLOOR  # noqa: E402
from core import models, segmenters, sidecar  # noqa: E402
from core.brats import MODALITY_ORDER, build_rgb_slice, load_volume  # noqa: E402
from core.metrics import mask_dice, mask_iou, match_boxes, prf, tight_bbox  # noqa: E402
from core.study import _zscore_clip, _zscore_clip_bgfix  # noqa: E402

RAW = Path(os.environ.get("RAW_DIR", DRIVE_ROOT / "data" / "raw"))
SPLIT = DRIVE_ROOT / "data" / "processed" / "metadata" / "patient_split.csv"
OUT_DIR = Path(os.environ.get("VALIDATION_OUT", DRIVE_ROOT / "reports" / "validation"))
MIN_BRAIN = 500
# smallest connected component that earns its own oracle box
MIN_COMPONENT = 10
DEFAULT_RUN = 37
DEFAULT_ADAPTER = "seed_42"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def test_patients() -> List[str]:
    import csv
    return [r["patient_id"] for r in csv.DictReader(open(SPLIT)) if r["split"] == "test"]


REGION_LABELS = {"WT": (1, 2, 4), "TC": (1, 4), "ET": (4,)}


def region_mask(seg_raw: np.ndarray, region: str) -> np.ndarray:
    """Binary mask for a BraTS sub-region from the raw label volume."""
    vals = REGION_LABELS[region]
    out = np.zeros(seg_raw.shape, dtype=bool)
    for v in vals:
        out |= (np.rint(seg_raw) == v)
    return out


def load_patient(pid: str):
    d = RAW / pid
    paths = {}
    for m in list(MODALITY_ORDER) + ["seg"]:
        hit = next((p for p in (d / f"{pid}_{m}.nii", d / f"{pid}_{m}.nii.gz") if p.exists()), None)
        if hit is None:
            return None
        paths[m] = hit
    vols = {m: load_volume(paths[m]) for m in MODALITY_ORDER}
    seg_raw = load_volume(paths["seg"])
    return vols, seg_raw


def background_value(vol: np.ndarray) -> float:
    """The value skull-stripped background carries in this volume.

    BraTS2020 and BraTS-Africa ship raw intensities and their background is
    exactly 0, so `vol > 0` is a brain mask. RHUH-GBM ships z-score normalised
    and its background is a negative constant, where the same test instead
    selects voxels above the volume mean -- a subset of the brain. Taking the
    modal value of the eight corners recovers the right constant in both cases
    and costs nothing on the volumes where it is already zero.
    """
    c = np.array([vol[i, j, k]
                  for i in (0, -1) for j in (0, -1) for k in (0, -1)],
                 dtype=np.float64)
    vals, counts = np.unique(c, return_counts=True)
    return float(vals[counts.argmax()])


def slice_sets(vols, seg, stride: int):
    Z = seg.shape[2]
    brain = np.zeros(Z, dtype=np.int64)
    nz = None
    for m in MODALITY_ORDER:
        cur = vols[m] != background_value(vols[m])
        nz = cur if nz is None else (nz | cur)
    brain = nz.sum(axis=(0, 1))
    usable = [z for z in range(0, Z, stride) if brain[z] >= MIN_BRAIN]
    positive = [z for z in usable if seg[:, :, z].any()]
    return usable, positive


def volumetric(pred_by_z: Dict[int, np.ndarray], seg, zs: List[int]):
    inter = p_vol = g_vol = 0
    for z in zs:
        p = pred_by_z.get(z)
        g = seg[:, :, z]
        if p is None:
            p = np.zeros_like(g)
        inter += int((p & g).sum())
        p_vol += int(p.sum())
        g_vol += int(g.sum())
    denom = p_vol + g_vol
    union = p_vol + g_vol - inter
    return {
        "vol_dice": round(2 * inter / denom, 6) if denom else 1.0,
        "vol_iou": round(inter / union, 6) if union else 1.0,
        "pred_voxels": p_vol, "gt_voxels": g_vol,
        "rel_vol_diff": round((p_vol - g_vol) / g_vol, 6) if g_vol else None,
    }


# --------------------------------------------------------------------- arms
def component_boxes(g, min_voxels=MIN_COMPONENT):
    """A tight box per connected component, largest first.

    8-connectivity, matching the label geometry reported in the manuscript.
    Components below `min_voxels` are dropped: a single stray voxel is a valid
    component and would otherwise add a 1x1 prompt the detector would never
    emit, which would flatter the multi-box arm rather than test it.
    """
    lab, n = ndimage.label(g, structure=np.ones((3, 3), int))
    out = []
    for i in range(1, n + 1):
        c = lab == i
        if c.sum() < min_voxels:
            continue
        b = tight_bbox(c)
        if b is not None:
            out.append((int(c.sum()), b))
    out.sort(reverse=True, key=lambda t: t[0])
    return [b for _n, b in out]


def arm_oracle(pid, vols, seg_raw, usable, seg_id, regions,
               per_component=False) -> dict:
    """GT box -> segmenter, one region at a time. The ceiling; uses ground truth,
    so not deployable.

    Run on TC and ET this answers the reviewer's question about sub-regions
    without a sub-region detector existing: it is the best any detector could do
    with this segmenter. Each region brings its own box and its own slice set --
    the tight box of TC is not the box of WT, and a slice carrying WT but no ET
    is no evidence about ET. Scoring over the region's positive slices rather
    than over `usable` gives the identical number, since a slice with neither a
    prediction nor ground truth adds nothing to any term of Dice, so this stays
    comparable to the other arms.
    """
    per = {}
    for region in regions:
        gt = region_mask(seg_raw, region)
        positive = [z for z in usable if gt[:, :, z].any()]
        preds, t0 = {}, time.time()
        n_prompts = 0
        for z in positive:
            g = gt[:, :, z]
            box = tight_bbox(g)
            if per_component:
                # falling back to the whole-region box keeps the arm from being
                # penalised on slices whose components are all below the size
                # floor: the comparison is one box against several, never one
                # box against none.
                boxes = component_boxes(g) or ([box] if box is not None else [])
            else:
                boxes = [box] if box is not None else []
            if not boxes:
                continue
            n_prompts += len(boxes)
            rgb = build_rgb_slice({m: vols[m][:, :, z] for m in MODALITY_ORDER})
            masks = segmenters.segment(seg_id, rgb, boxes)
            # union, exactly as arm_pipeline combines its own boxes
            preds[z] = (masks.any(axis=0) if len(masks)
                        else np.zeros(g.shape, bool))
        r = volumetric(preds, gt, positive)
        # gt_voxels comes from volumetric and is what the analysis filters on: a
        # patient with no ET at all scores an empty prediction against an empty
        # ground truth as Dice 1.0, which would inflate the ET mean silently.
        r.update({"n_slices": len(positive), "n_prompts": n_prompts,
                  "per_component": bool(per_component),
                  "elapsed_s": round(time.time() - t0, 1)})
        per[region] = r
    # The default WT-only call writes exactly the record earlier runs wrote, so
    # old and new JSONLs stay comparable and every existing reader is untouched.
    if list(per) == ["WT"]:
        return per["WT"]
    out = {"regions": per}
    if "WT" in per:
        out.update(per["WT"])
    return out


def arm_pipeline(pid, vols, seg_raw, usable, seg_id, run_id, conf, region="WT") -> dict:
    """RT-DETR box -> segmenter, over every in-distribution slice.

    `region` is what the detector was trained to find and what the result is
    scored against; the two must agree. A whole-tumour detector scored against
    TC would look catastrophic for a reason that has nothing to do with either
    model, so the region is written into the record rather than inferred later
    from which file the record happens to sit in.
    """
    seg = region_mask(seg_raw, region)
    preds, t0 = {}, time.time()
    tp = fp = fn = 0
    n_boxes = 0
    for z in usable:
        rgb = build_rgb_slice({m: vols[m][:, :, z] for m in MODALITY_ORDER})
        raw = models.detect(run_id, rgb, score_floor=SCORE_FLOOR)
        keep = [i for i, s in enumerate(raw["scores"]) if s >= conf]
        boxes = [raw["boxes"][i] for i in keep]
        scores = [raw["scores"][i] for i in keep]
        n_boxes += len(boxes)

        g = seg[:, :, z]
        gt_box = tight_bbox(g)
        m = match_boxes(boxes, scores, [gt_box] if gt_box else [], MATCH_IOU)
        tp += m["tp"]; fp += m["fp"]; fn += m["fn"]

        if boxes:
            masks = segmenters.segment(seg_id, rgb, boxes)
            preds[z] = masks.any(axis=0)
    out = volumetric(preds, seg, usable)
    out.update({"n_slices": len(usable), "n_boxes": n_boxes, "region": region,
                "tp": tp, "fp": fp, "fn": fn,
                **{k: round(v, 6) for k, v in prf(tp, fp, fn).items()},
                "elapsed_s": round(time.time() - t0, 1)})
    return out


def arm_direct(pid, vols, seg_raw, usable, adapter, regions,
               bgfix=False) -> dict:
    """MedSAM3 (+LoRA) with no detector: the sidecar's text-prompted pipeline.

    One forward pass returns every requested region, so scoring TC and ET on top
    of WT costs no extra inference.
    """
    z0, z1 = min(usable), max(usable) + 1
    norm = _zscore_clip_bgfix if bgfix else _zscore_clip
    prgb = np.stack([norm(vols[m]) for m in MODALITY_ORDER], axis=0)[:, :, :, z0:z1]
    t0 = time.time()
    res = sidecar.segment_volume(prgb, adapter, regions, batch_size=2)
    elapsed = round(time.time() - t0, 1)

    out = {"n_slices": len(usable), "slab": [z0, z1], "bgfix": bool(bgfix),
           "adapter": res.get("adapter"), "elapsed_s": elapsed, "regions": {}}
    for region in regions:
        vol = res["masks"].get(region)
        if vol is None:
            continue
        gt = region_mask(seg_raw, region)
        preds = {z: vol[:, :, z - z0] for z in usable}
        out["regions"][region] = volumetric(preds, gt, usable)
    # keep the WT numbers at the top level so older readers still work
    if "WT" in out["regions"]:
        out.update({k: v for k, v in out["regions"]["WT"].items()})
    return out


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="oracle,pipeline,direct")
    ap.add_argument("--segmenters", default="sam1_vit_b,sam2.1_l",
                    help="box-prompted segmenters for the oracle and pipeline arms")
    ap.add_argument("--run-id", type=int, default=DEFAULT_RUN)
    ap.add_argument("--conf", type=float, default=DEFAULT_CONF)
    ap.add_argument("--detector-ckpt", default=None,
                    help="load the detector from this checkpoint instead of the registry; "
                         "needed for the sub-region detectors, which reuse run ids")
    ap.add_argument("--adapter", default=DEFAULT_ADAPTER)
    ap.add_argument("--direct-bgfix", action="store_true",
                    help="normalise the detector-free arm's input with the "
                         "corrected brain mask, so that on RHUH-GBM all three "
                         "arms see the same volumes")
    ap.add_argument("--regions", default="WT", help="detector-free arm: WT[,TC][,ET]")
    ap.add_argument("--oracle-regions", default="WT", help="oracle arm: WT[,TC][,ET]")
    ap.add_argument("--oracle-per-component", action="store_true",
                    help="one tight box per connected component, unioned, "
                         "which is the prompt protocol the pipeline arm uses")
    ap.add_argument("--pipeline-region", default="WT",
                    help="what the detector was trained to find, and what its output is "
                         "scored against; must match the run-id's training target")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth slice")
    ap.add_argument("--limit", type=int, default=None, help="first N patients only")
    ap.add_argument("--patients-file", default=None,
                    help="evaluate this explicit patient list instead of the study's test split")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    regions = [x.strip().upper() for x in a.regions.split(",") if x.strip()]
    oracle_regions = [x.strip().upper() for x in a.oracle_regions.split(",") if x.strip()]
    pipeline_region = a.pipeline_region.strip().upper()
    assert all(r in REGION_LABELS for r in regions + oracle_regions + [pipeline_region]), \
        "regions must be WT, TC or ET"
    seg_ids = [x.strip() for x in a.segmenters.split(",") if x.strip()]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out) if a.out else OUT_DIR / f"validation_run{a.run_id}_stride{a.stride}.jsonl"

    expected_arms = []
    for sid in seg_ids:
        if "oracle" in arms:
            expected_arms.append(f"oracle:{sid}")
        if "pipeline" in arms:
            expected_arms.append(f"pipeline:{sid}")
    if "direct" in arms:
        expected_arms.append(f"direct:{a.adapter}")

    # A record only counts as done if it carries every arm this invocation asks
    # for and recorded no error. Anything else is dropped so it gets redone --
    # the previous behaviour skipped precisely the patients that had failed.
    done, keep = set(), []
    if out_path.exists():
        seen = set()
        total = 0
        for line in out_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            total += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            pid = rec.get("patient")
            complete = ("error" not in rec) and all(k in rec.get("arms", {}) for k in expected_arms)
            if complete and pid not in seen:
                seen.add(pid)
                done.add(pid)
                keep.append(line)
        if len(keep) != total:
            out_path.write_text("\n".join(keep) + ("\n" if keep else ""), encoding="utf-8")
            log(f"resume: kept {len(keep)} complete records, dropped {total - len(keep)} "
                f"failed/duplicate lines from {out_path.name}")
        else:
            log(f"resuming: {len(done)} complete patients in {out_path.name}")

    models.init(os.environ.get("FORCE_DEVICE"))
    if a.detector_ckpt:
        models.register_detector(a.run_id, a.detector_ckpt)
    segmenters.set_device(models.device())
    log(f"device={models.device()} arms={arms} segmenters={seg_ids} "
        f"run={a.run_id} conf={a.conf} adapter={a.adapter} regions={regions} "
        f"oracle_regions={oracle_regions} pipeline_region={pipeline_region} "
        f"stride={a.stride}"
        + (f" detector_ckpt={a.detector_ckpt}" if a.detector_ckpt else ""))

    if "direct" in arms and sidecar.health() is None:
        log("WARNING: MedSAM3 sidecar is not up — dropping the 'direct' arm")
        arms = [x for x in arms if x != "direct"]

    if a.patients_file:
        patients = [x.strip() for x in Path(a.patients_file).read_text().splitlines() if x.strip()]
        log(f"patient list from {Path(a.patients_file).name}: {len(patients)} patients")
    else:
        patients = test_patients()
    if a.limit:
        patients = patients[:a.limit]
    todo = [p for p in patients if p not in done]
    log(f"{len(todo)} patients to process (of {len(patients)})")

    t_start = time.time()
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3
    for i, pid in enumerate(todo, 1):
        loaded = load_patient(pid)
        if loaded is None:
            log(f"  {pid}: volumes missing, skipped")
            continue
        vols, seg_raw = loaded
        seg = region_mask(seg_raw, "WT")
        usable, positive = slice_sets(vols, seg, a.stride)
        if not positive:
            log(f"  {pid}: no tumour slices, skipped")
            continue

        rec: dict = {"patient": pid, "n_usable": len(usable), "n_positive": len(positive),
                     "gt_voxels": int(seg.sum()), "run_id": a.run_id, "conf": a.conf,
                     "stride": a.stride, "arms": {}}
        try:
            for sid in seg_ids:
                if "oracle" in arms:
                    rec["arms"][f"oracle:{sid}"] = arm_oracle(
                        pid, vols, seg_raw, usable, sid, oracle_regions,
                        per_component=a.oracle_per_component)
                if "pipeline" in arms:
                    rec["arms"][f"pipeline:{sid}"] = arm_pipeline(
                        pid, vols, seg_raw, usable, sid, a.run_id, a.conf, pipeline_region)
            if "direct" in arms:
                rec["arms"][f"direct:{a.adapter}"] = arm_direct(
                    pid, vols, seg_raw, usable, a.adapter, regions,
                    bgfix=a.direct_bgfix)
            consecutive_failures = 0
        except Exception as e:
            rec["error"] = f"{type(e).__name__}: {e}"
            consecutive_failures += 1
            log(f"  {pid}: FAILED ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES}) {rec['error']}")

        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec) + "\n")

        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            log(f"ABORTING: {consecutive_failures} patients failed in a row. "
                f"Something is broken (GPU out of memory, sidecar down); "
                f"grinding through the rest would only produce more error records.")
            raise SystemExit(2)

        el = time.time() - t_start
        eta = el / i * (len(todo) - i)
        summary = " | ".join(f"{k.split(':')[0][:4]}:{v.get('vol_dice', 0):.3f}"
                             for k, v in rec["arms"].items())
        log(f"  [{i}/{len(todo)}] {pid} {summary}  ({el/60:.0f} min elapsed, ~{eta/60:.0f} min left)")

    log(f"done in {(time.time()-t_start)/60:.1f} min -> {out_path}")


if __name__ == "__main__":
    main()
