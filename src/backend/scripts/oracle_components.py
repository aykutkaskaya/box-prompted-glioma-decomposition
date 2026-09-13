"""What changes when the oracle arm is prompted the way the pipeline is.

The oracle arm passed one tight box over the whole region; the pipeline passes
every box the detector emits and unions the masks. On a multi-component slice
those are different protocols, so two readings depended on which one the oracle
was given: the size of the detector-stage term, and whether the enhancing-tumour
cap is a property of box prompting or only of a one-box oracle.

This compares the two oracle protocols arm for arm, on whatever cohorts the
per-component rerun has finished. It prints nothing it cannot source.

    python scripts/oracle_components.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "oracle_components.json"
SEG = "sam2.1_l"
DRAWS = 20000
COHORTS = [("clean", "BraTS2020 (shared held-out set)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]


def load(name: str) -> dict:
    p = V / name
    p = corrected(p)
    if not p.exists():
        return {}
    out = {}
    for line in io.open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[r["patient"]] = r
    return out


def region(rec: dict, arm: str, reg: str, field: str = "vol_dice"):
    a = rec["arms"].get(arm)
    if a is None:
        return None
    if "regions" in a:
        return a["regions"].get(reg, {}).get(field)
    return a.get(field) if reg == "WT" else None


def ci(d, seed: int = 1337):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(DRAWS)])
    return (float(d.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


def main() -> None:
    result = {"draws": DRAWS, "cohorts": {}}
    for c, label in COHORTS:
        comp = load(f"{c}_oracle_components.jsonl")
        if not comp:
            print(f"{label}: per-component rerun not present")
            continue
        box = load(f"{c}_box.jsonl")
        one = load(f"{c}_oracle_regions.jsonl")
        ks = sorted(set(comp) & set(box) & set(one))
        if not ks:
            print(f"{label}: no overlapping patients")
            continue
        row = {"n": len(ks), "complete": len(comp) == len(box), "regions": {}}
        print(f"\n{label}  n = {len(ks)}"
              f"{'' if row['complete'] else '  (rerun still in progress)'}")
        print(f"  {'region':<5}{'1 box':>9}{'per comp':>10}{'delta':>10}"
              f"{'95% CI':>22}{'prompts/slice':>15}")
        for reg in ("WT", "TC", "ET"):
            # Dice against an empty reference is not a measurement, and two of
            # the 24 held-out patients carry no enhancing tumour. The
            # sub-region table already averages over the patients that carry
            # the region; averaging over the whole cohort here gave the same
            # cohort and region two different oracle scores.
            rk = [k for k in ks
                  if region(one[k], f"oracle:{SEG}", reg, "gt_voxels")]
            a = np.array([region(one[k], f"oracle:{SEG}", reg) for k in rk],
                         dtype=float)
            b = np.array([region(comp[k], f"oracle:{SEG}", reg) for k in rk],
                         dtype=float)
            if np.isnan(a).any() or np.isnan(b).any():
                continue
            m, lo, hi = ci(b - a)
            npr = float(np.mean([
                comp[k]["arms"][f"oracle:{SEG}"]["regions"][reg]["n_prompts"]
                / max(comp[k]["arms"][f"oracle:{SEG}"]["regions"][reg]["n_slices"], 1)
                for k in rk]))
            entry = {"one_box": round(float(a.mean()), 4),
                     "per_component": round(float(b.mean()), 4),
                     "delta": round(m, 4), "ci": [round(lo, 4), round(hi, 4)],
                     "prompts_per_slice": round(npr, 2)}
            print(f"  {reg:<5}{a.mean():>9.4f}{b.mean():>10.4f}{m:>+10.4f}"
                  f"   [{lo:+.4f}, {hi:+.4f}]{npr:>13.2f}")

            # the volume error is what the enhancing-tumour cap is argued from
            va = [region(one[k], f"oracle:{SEG}", reg, "rel_vol_diff") for k in ks]
            vb = [region(comp[k], f"oracle:{SEG}", reg, "rel_vol_diff") for k in ks]
            pairs = [(x, y) for x, y in zip(va, vb)
                     if x is not None and y is not None]
            if pairs:
                entry["vol_err_median_one_box"] = round(
                    float(np.median([p[0] for p in pairs])) * 100, 1)
                entry["vol_err_median_per_component"] = round(
                    float(np.median([p[1] for p in pairs])) * 100, 1)
                print(f"        median volume error "
                      f"{entry['vol_err_median_one_box']:+.1f}% -> "
                      f"{entry['vol_err_median_per_component']:+.1f}%")
            row["regions"][reg] = entry

        # the detector-stage term is the reason the protocol matters for the
        # headline, so it is recomputed under both oracle prompts
        pipe = np.array([region(box[k], f"pipeline:{SEG}", "WT") for k in ks],
                        dtype=float)
        o1 = np.array([region(one[k], f"oracle:{SEG}", "WT") for k in ks],
                      dtype=float)
        o2 = np.array([region(comp[k], f"oracle:{SEG}", "WT") for k in ks],
                      dtype=float)
        t1, l1, h1 = ci(o1 - pipe)
        t2, l2, h2 = ci(o2 - pipe)
        row["detector_stage"] = {
            "one_box": [round(t1, 4), round(l1, 4), round(h1, 4)],
            "per_component": [round(t2, 4), round(l2, 4), round(h2, 4)],
        }
        print(f"  detector-stage term, WT: {t1:+.4f} [{l1:+.4f}, {h1:+.4f}]"
              f"  ->  {t2:+.4f} [{l2:+.4f}, {h2:+.4f}]")
        result["cohorts"][c] = row

    # ------------------------------------------------ the between-cohort contrast
    # This is the quantity the paper's claim rests on, so it is recomputed under
    # both prompt protocols rather than inferred from the two point estimates.
    # Unpaired: two independent cohorts, difference of two means.
    if "clean" in result["cohorts"]:
        base = {}
        for c, _ in COHORTS:
            comp = load(f"{c}_oracle_components.jsonl")
            box = load(f"{c}_box.jsonl")
            one = load(f"{c}_oracle_regions.jsonl")
            ks = sorted(set(comp) & set(box) & set(one))
            if not ks:
                continue
            pipe = np.array([region(box[k], f"pipeline:{SEG}", "WT") for k in ks])
            base[c] = {
                "one_box": np.array([region(one[k], f"oracle:{SEG}", "WT")
                                     for k in ks]) - pipe,
                "per_component": np.array([region(comp[k], f"oracle:{SEG}", "WT")
                                           for k in ks]) - pipe,
            }
        result["contrast_vs_held_out"] = {}
        print(f"\n{'contrast against the held-out cohort':<40}"
              f"{'one box':>26}{'per component':>26}")
        for c, label in COHORTS:
            if c == "clean" or c not in base:
                continue
            row = {}
            for proto in ("one_box", "per_component"):
                a, b = base[c][proto], base["clean"][proto]
                bs = np.sort([
                    np.random.default_rng([1337, i]).choice(a, len(a)).mean()
                    - np.random.default_rng([7331, i]).choice(b, len(b)).mean()
                    for i in range(DRAWS)])
                row[proto] = [round(float(a.mean() - b.mean()), 4),
                              round(float(np.percentile(bs, 2.5)), 4),
                              round(float(np.percentile(bs, 97.5)), 4)]
            result["contrast_vs_held_out"][c] = row
            o, p = row["one_box"], row["per_component"]
            print(f"  {label:<38}{o[0]:+8.4f} [{o[1]:+.4f}, {o[2]:+.4f}]"
                  f"{p[0]:+8.4f} [{p[1]:+.4f}, {p[2]:+.4f}]")

    if result["cohorts"]:
        OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
