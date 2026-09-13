"""Consolidate every validation pass into one table.

Reads the per-patient JSONL files and reports paired comparisons: the same 55
held-out patients go through every arm, so differences can be tested patient by
patient rather than compared as unpaired averages.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "validation_summary.json"


def load(name: str) -> Dict[str, dict]:
    p = V / name
    if not p.exists():
        return {}
    out = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or not r.get("arms"):
            continue
        out[r["patient"]] = r
    return out


def wilcoxon(a: List[float], b: List[float]):
    try:
        from scipy.stats import wilcoxon as w
        stat, p = w(a, b)
        return float(p)
    except Exception:
        return None


def describe(vals: List[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"n": int(v.size), "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4),
            "sd": round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}


box = load("box_arms.jsonl")
d42 = load("direct_seed_42.jsonl")
d62 = load("direct_seed_62.jsonl")
d52 = load("direct_seed_52.jsonl")

common = sorted(set(box) & set(d42) & set(d62) & set(d52))
print(f"patients: box {len(box)}, seed42 {len(d42)}, seed62 {len(d62)}, seed52 {len(d52)} "
      f"-> {len(common)} with every arm\n")

BOX_ARMS = ["oracle:sam1_vit_b", "oracle:sam2.1_l", "pipeline:sam1_vit_b", "pipeline:sam2.1_l"]
series: Dict[str, List[float]] = {}
for arm in BOX_ARMS:
    series[arm] = [box[p]["arms"][arm]["vol_dice"] for p in common]
for tag, src in (("direct:seed_42", d42), ("direct:seed_62", d62), ("direct:seed_52", d52)):
    series[tag] = [src[p]["arms"][list(src[p]["arms"])[0]]["vol_dice"] for p in common]

LABEL = {
    "oracle:sam1_vit_b":  "oracle GT box -> SAM1 ViT-B",
    "oracle:sam2.1_l":    "oracle GT box -> SAM 2.1-L",
    "pipeline:sam1_vit_b": "RT-DETR -> SAM1 ViT-B",
    "pipeline:sam2.1_l":  "RT-DETR -> SAM 2.1-L",
    "direct:seed_42":     "MedSAM3+LoRA seed 42 (no detector)",
    "direct:seed_62":     "MedSAM3+LoRA seed 62 (no detector)",
    "direct:seed_52":     "MedSAM3+LoRA seed 52 (no detector)",
}

print("=== volumetric Dice, whole tumour, %d paired patients ===" % len(common))
print(f"{'arm':<38}{'mean':>8}{'median':>9}{'sd':>8}{'min':>8}{'max':>8}")
stats = {}
for k in LABEL:
    s = describe(series[k])
    stats[k] = s
    print(f"{LABEL[k]:<38}{s['mean']:>8.4f}{s['median']:>9.4f}{s['sd']:>8.4f}{s['min']:>8.4f}{s['max']:>8.4f}")

print("\n=== paired comparisons (Wilcoxon signed-rank) ===")
PAIRS = [
    ("pipeline:sam1_vit_b", "pipeline:sam2.1_l", "does SAM 2.1-L beat ViT-B in the real pipeline?"),
    ("oracle:sam1_vit_b", "oracle:sam2.1_l", "does SAM 2.1-L raise the oracle ceiling?"),
    ("pipeline:sam2.1_l", "direct:seed_42", "detector-free vs the best box pipeline"),
    ("oracle:sam2.1_l", "direct:seed_42", "detector-free vs the oracle ceiling"),
    ("pipeline:sam1_vit_b", "direct:seed_42", "detector-free vs the study's pipeline"),
]
comparisons = []
for a, b, q in PAIRS:
    va, vb = np.array(series[a]), np.array(series[b])
    diff = vb - va
    p = wilcoxon(list(va), list(vb))
    wins = int((diff > 0).sum())
    ci = (float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5)))
    comparisons.append({"a": a, "b": b, "question": q,
                        "mean_diff": round(float(diff.mean()), 4),
                        "median_diff": round(float(np.median(diff)), 4),
                        "b_wins": wins, "n": len(diff),
                        "diff_pct2.5": round(ci[0], 4), "diff_pct97.5": round(ci[1], 4),
                        "wilcoxon_p": p})
    star = "" if p is None else ("  p<1e-9" if p < 1e-9 else f"  p={p:.2e}")
    print(f"  {q}")
    print(f"    {LABEL[b]} minus {LABEL[a]}: mean {diff.mean():+.4f}, "
          f"median {np.median(diff):+.4f}, wins {wins}/{len(diff)}{star}")

print("\n=== LoRA seed variance (per patient, then averaged) ===")
seeds = np.vstack([series["direct:seed_42"], series["direct:seed_62"], series["direct:seed_52"]])
per_patient_sd = seeds.std(axis=0, ddof=1)
print(f"  mean Dice by seed: 42 {seeds[0].mean():.4f} | 62 {seeds[1].mean():.4f} | 52 {seeds[2].mean():.4f}")
print(f"  spread of the three means: {seeds.mean(axis=1).max() - seeds.mean(axis=1).min():.4f}")
print(f"  within-patient sd across seeds: mean {per_patient_sd.mean():.4f}, max {per_patient_sd.max():.4f}")

print("\n=== sub-regions, detector-free seed 42 ===")
regions = {}
for region in ("WT", "TC", "ET"):
    vals = []
    for p in common:
        arm = d42[p]["arms"][list(d42[p]["arms"])[0]]
        r = arm.get("regions", {}).get(region)
        if r:
            vals.append(r["vol_dice"])
    if vals:
        regions[region] = describe(vals)
        s = regions[region]
        print(f"  {region}: mean {s['mean']:.4f}  median {s['median']:.4f}  sd {s['sd']:.4f}  (n={s['n']})")
if not regions:
    print("  (no per-region breakdown recorded)")

det = [box[p]["arms"]["pipeline:sam1_vit_b"] for p in common]
tp = sum(x["tp"] for x in det); fp = sum(x["fp"] for x in det); fn = sum(x["fn"] for x in det)
pr, rc = tp / (tp + fp), tp / (tp + fn)
detection = {"tp": tp, "fp": fp, "fn": fn, "precision": round(pr, 4), "recall": round(rc, 4),
             "f1": round(2 * pr * rc / (pr + rc), 4)}
print(f"\n=== detection (RT-DETR run 37, conf 0.55) ===")
print(f"  TP {tp}  FP {fp}  FN {fn}  |  P {pr:.4f}  R {rc:.4f}  F1 {detection['f1']:.4f}")

OUT.write_text(json.dumps({
    "n_patients": len(common), "patients": common,
    "arms": {k: stats[k] for k in LABEL},
    "comparisons": comparisons,
    "seed_variance": {
        "means": {s: round(float(seeds[i].mean()), 4) for i, s in enumerate(["seed_42", "seed_62", "seed_52"])},
        "within_patient_sd_mean": round(float(per_patient_sd.mean()), 4),
        "within_patient_sd_max": round(float(per_patient_sd.max()), 4),
    },
    "regions": regions, "detection": detection,
}, indent=1), encoding="utf-8")
print(f"\nwrote {OUT}")
