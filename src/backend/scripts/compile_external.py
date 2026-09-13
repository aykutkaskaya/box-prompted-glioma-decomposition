"""Paired analysis for an external cohort.

compile_results.py cannot be pointed at one. It reads four fixed filenames, it
expects three LoRA seeds so it can report seed variance, and it ends with the
RT-DETR detection counts for run 37 -- none of which an external pass produces.
It also asks a different question. Internally the interesting number was the
ranking of six arms; here the ranking is already known and what is being tested
is whether it survives a change of hospital, so this reports the paired
contrasts and, more importantly, how wide their intervals are.

The cohort name is an argument rather than a constant. RHUH-GBM is the first
external set, not the last, and the next one should not need a second copy of
this file.

    python scripts/compile_external.py                              # RHUH-GBM
    python scripts/compile_external.py other_box.jsonl other_direct.jsonl \
        --out reports/other_summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

# Same override full_validation.py writes through, so a dry run on fixtures can
# be pointed somewhere harmless without editing anything.
V = Path(os.environ.get("VALIDATION_OUT", DRIVE_ROOT / "reports" / "validation"))
OUT = DRIVE_ROOT / "reports" / "rhuh_summary.json"

RNG = np.random.default_rng(20260824)
BOOT = 20000
ALPHA = 0.05

BOX_ARMS = ["oracle:sam1_vit_b", "oracle:sam2.1_l", "pipeline:sam1_vit_b", "pipeline:sam2.1_l"]
LABEL = {
    "oracle:sam1_vit_b": "oracle GT box -> SAM 1 ViT-B",
    "oracle:sam2.1_l": "oracle GT box -> SAM 2.1-L",
    "pipeline:sam1_vit_b": "RT-DETR -> SAM 1 ViT-B",
    "pipeline:sam2.1_l": "RT-DETR -> SAM 2.1-L",
}
CONTRASTS = [
    ("pipeline:sam2.1_l", "detector-free minus the best box pipeline"),
    ("pipeline:sam1_vit_b", "detector-free minus the study's pipeline"),
    ("oracle:sam2.1_l", "detector-free minus the oracle ceiling"),
]


def load(name: str) -> Dict[str, dict]:
    p = V / name
    if not p.exists():
        raise SystemExit(f"missing {p} -- run scripts/rhuh_cohort.sh first")
    out: Dict[str, dict] = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or not r.get("arms"):
            continue
        out[r["patient"]] = r
    return out


def describe(vals: List[float]) -> dict:
    v = np.asarray(vals, dtype=float)
    return {"n": int(v.size), "mean": round(float(v.mean()), 4),
            "median": round(float(np.median(v)), 4),
            "sd": round(float(v.std(ddof=1)), 4) if v.size > 1 else 0.0,
            "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}


def boot_ci(d: np.ndarray, stat_fn, n_boot: int = BOOT) -> Tuple[float, float]:
    """Percentile bootstrap interval for a statistic of the paired differences.

    Same convention as power_analysis.py, and the same warning: this is an
    interval for the *statistic*, not the 2.5/97.5 percentiles of the
    differences themselves. The latter describes how much patients differ from
    each other and is much wider; quoting it as a confidence interval would
    understate the result rather than hedge it.
    """
    idx = RNG.integers(0, d.size, size=(n_boot, d.size))
    draws = stat_fn(d[idx])
    return float(np.percentile(draws, 100 * ALPHA / 2)), float(np.percentile(draws, 100 * (1 - ALPHA / 2)))


def wilcoxon(a: np.ndarray, b: np.ndarray):
    try:
        return float(stats.wilcoxon(a, b)[1])
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("box", nargs="?", default="rhuh_box.jsonl",
                    help="box-arm JSONL under reports/validation")
    ap.add_argument("direct", nargs="?", default="rhuh_direct.jsonl",
                    help="detector-free JSONL under reports/validation")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    box, dir_recs = load(a.box), load(a.direct)
    common = sorted(set(box) & set(dir_recs))
    if not common:
        raise SystemExit("no patient carries both the box and the detector-free arm")
    print(f"{a.box}: {len(box)} patients | {a.direct}: {len(dir_recs)} "
          f"-> {len(common)} with every arm\n")

    # The detector-free arm is keyed by adapter, which the caller may vary, so
    # take the key from the data rather than assuming seed_42.
    dkey = list(dir_recs[common[0]]["arms"])[0]
    series = {k: np.array([box[p]["arms"][k]["vol_dice"] for p in common])
              for k in BOX_ARMS if all(k in box[p]["arms"] for p in common)}
    series[dkey] = np.array([dir_recs[p]["arms"][dkey]["vol_dice"] for p in common])
    labels = dict(LABEL)
    labels[dkey] = f"MedSAM3+LoRA {dkey.split(':', 1)[1]} (no detector)"

    print(f"=== volumetric Dice, whole tumour, {len(common)} paired patients ===")
    print(f"{'arm':<38}{'mean':>8}{'median':>9}{'sd':>8}{'min':>8}{'max':>8}")
    arm_stats = {}
    for k in series:
        s = describe(list(series[k]))
        arm_stats[k] = s
        print(f"{labels[k]:<38}{s['mean']:>8.4f}{s['median']:>9.4f}{s['sd']:>8.4f}"
              f"{s['min']:>8.4f}{s['max']:>8.4f}")

    print("\n=== paired contrasts (detector-free minus each box arm) ===")
    contrasts = []
    for key, question in CONTRASTS:
        if key not in series:
            continue
        d = series[dkey] - series[key]
        p = wilcoxon(series[key], series[dkey])
        ci = boot_ci(d, lambda m: m.mean(axis=1))
        contrasts.append({
            "contrast": question, "base": key, "test": dkey, "n": int(d.size),
            "mean_diff": round(float(d.mean()), 4),
            "median_diff": round(float(np.median(d)), 4),
            "sd_diff": round(float(d.std(ddof=1)), 4),
            "wins": int((d > 0).sum()),
            "ci95": [round(ci[0], 4), round(ci[1], 4)],
            "ci95_halfwidth": round((ci[1] - ci[0]) / 2, 4),
            "wilcoxon_p": p,
        })
        r = contrasts[-1]
        star = "" if p is None else (" p<1e-9" if p < 1e-9 else f" p={p:.2e}")
        print(f"  {question}")
        print(f"    vs {labels[key]}: mean {r['mean_diff']:+.4f}, median {r['median_diff']:+.4f}, "
              f"wins {r['wins']}/{r['n']}{star}")
        print(f"    95% CI of the mean difference [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
              f"  -> +/-{r['ci95_halfwidth']:.4f}")

    # A mean over 39 patients hides a bimodal failure. Externally both deployable
    # arms drop to near zero on a couple of patients while the oracle never does,
    # which says the failures are localisation, not segmentation -- and it raises
    # the obvious objection that the whole gap is a handful of collapses. Report
    # the primary contrast with them removed so that objection can be settled
    # rather than argued.
    COLLAPSE = 0.5
    print(f"\n=== collapses (volumetric Dice below {COLLAPSE}) ===")
    collapse = {k: [p for p, v in zip(common, series[k]) if v < COLLAPSE] for k in series}
    for k in series:
        who = collapse[k]
        print(f"  {labels[k]:<38}{len(who)}/{len(common)}"
              + (f"  {', '.join(who)}" if who else ""))
    base = CONTRASTS[0][0]
    robustness = None
    if base in series:
        hit = sorted(set(collapse[base]) | set(collapse[dkey]))
        keep = [i for i, p in enumerate(common) if p not in hit]
        if hit and len(keep) > 2:
            b, t = series[base][keep], series[dkey][keep]
            d = t - b
            robustness = {"threshold": COLLAPSE, "excluded": hit, "n": len(keep),
                          "mean_diff": round(float(d.mean()), 4),
                          "median_diff": round(float(np.median(d)), 4),
                          "wins": int((d > 0).sum()),
                          "wilcoxon_p": wilcoxon(b, t)}
            print(f"  primary contrast with the {len(hit)} collapsed patients removed "
                  f"({robustness['n']} left): mean {robustness['mean_diff']:+.4f}, "
                  f"median {robustness['median_diff']:+.4f}, "
                  f"wins {robustness['wins']}/{robustness['n']}"
                  + (f", p={robustness['wilcoxon_p']:.2e}" if robustness["wilcoxon_p"] else ""))

    print(f"\n=== sub-regions, {labels[dkey]} ===")
    # full_validation.volumetric scores an empty prediction against an empty
    # ground truth as Dice 1.0, which is defensible per patient and ruinous in a
    # mean. It never mattered internally: a BraTS HGG case with no enhancing
    # component barely exists. It matters here, where the cohort is GBM from
    # another hospital under another annotation protocol -- one patient with no
    # ET would post a perfect score and drag the ET mean up. Score only the
    # patients that have something to segment, and report how many were left out
    # instead of quietly averaging them in.
    regions, no_gt = {}, {}
    for region in ("WT", "TC", "ET"):
        got = [r for r in (dir_recs[p]["arms"][dkey].get("regions", {}).get(region)
                           for p in common) if r]
        scored = [r["vol_dice"] for r in got if r.get("gt_voxels", 1) > 0]
        no_gt[region] = len(got) - len(scored)
        if scored:
            regions[region] = describe(scored)
            s = regions[region]
            skipped = f"  [{no_gt[region]} with no {region} in the ground truth, excluded]" \
                if no_gt[region] else ""
            print(f"  {region}: mean {s['mean']:.4f}  median {s['median']:.4f}  "
                  f"sd {s['sd']:.4f}  (n={s['n']}){skipped}")
    if not regions:
        print("  (no per-region breakdown recorded)")

    out_path = Path(a.out)
    out_path.write_text(json.dumps({
        "box_file": a.box, "direct_file": a.direct,
        "n_patients": len(common), "patients": common,
        "alpha": ALPHA, "bootstrap_draws": BOOT,
        "arms": arm_stats, "contrasts": contrasts, "regions": regions,
        "patients_without_region_gt": no_gt,
        "collapses": {k: v for k, v in collapse.items()}, "robustness": robustness,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
