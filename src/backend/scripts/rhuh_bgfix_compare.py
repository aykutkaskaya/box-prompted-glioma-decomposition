"""The detector-free arm on RHUH-GBM under both normalisations.

The box arms were re-run when the `> 0` brain mask was found to be wrong on this
cohort. The detector-free arm was not, because its input mirrors the MedSAM3
project's own preprocessing, so on RHUH-GBM the three arms stopped seeing the
same volumes. This compares the arm under the verbatim mirror (the primary arm)
with the same arm under the corrected mask, on the patients both cover, and
recomputes the two quantities that depend on it: the total gap and the residual.

    python scripts/rhuh_bgfix_compare.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "rhuh_bgfix.json"
SEG = "sam2.1_l"
DRAWS = 20000


def load(name: str) -> dict:
    p = V / name
    if not p.exists():
        return {}
    out = {}
    for line in io.open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[r["patient"]] = r
    return out


def wt(rec: dict, arm: str) -> float:
    a = rec["arms"][arm]
    return a["regions"]["WT"]["vol_dice"] if "regions" in a else a["vol_dice"]


def ci(d, seed: int = 1337):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(DRAWS)])
    return (float(d.mean()), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


SEEDS = (42, 52, 62)


def seed_files(sd: int) -> tuple:
    return (f"rhuh_direct_seed{sd}_normfix.jsonl",
            f"rhuh_direct_seed{sd}_bgfix.jsonl")


def main() -> None:
    box = load("rhuh_box_normfix.jsonl")

    # every seed that has been run both ways, so the table grows as they land
    per_seed = {}
    for sd in SEEDS:
        nf, bf = seed_files(sd)
        a, b = load(nf), load(bf)
        if a and b:
            per_seed[sd] = (a, b)
    if not per_seed:
        print("no corrected-normalisation run is present")
        return

    print(f"{'seed':<6}{'published':>11}{'corrected':>11}{'rise':>10}"
          f"{'gap pub':>10}{'gap cor':>10}{'res pub':>10}{'res cor':>10}")
    rows = {}
    for sd, (a, b) in per_seed.items():
        kk = sorted(set(box) & set(a) & set(b))
        p_ = np.array([wt(box[k], f"pipeline:{SEG}") for k in kk])
        o_ = np.array([wt(box[k], f"oracle:{SEG}") for k in kk])
        na = np.array([wt(a[k], f"direct:seed_{sd}") for k in kk])
        fx = np.array([wt(b[k], f"direct:seed_{sd}") for k in kk])
        rows[sd] = {
            "n": len(kk),
            "published": round(float(na.mean()), 4),
            "corrected": round(float(fx.mean()), 4),
            "rise": [round(v, 4) for v in ci(fx - na)],
            "gap_published": [round(v, 4) for v in ci(na - p_)],
            "gap_corrected": [round(v, 4) for v in ci(fx - p_)],
            "residual_published": [round(v, 4) for v in ci(na - o_)],
            "residual_corrected": [round(v, 4) for v in ci(fx - o_)],
        }
        r = rows[sd]
        print(f"{sd:<6}{r['published']:>11.4f}{r['corrected']:>11.4f}"
              f"{r['rise'][0]:>+10.4f}{r['gap_published'][0]:>+10.4f}"
              f"{r['gap_corrected'][0]:>+10.4f}"
              f"{r['residual_published'][0]:>+10.4f}"
              f"{r['residual_corrected'][0]:>+10.4f}")

    neg = [sd for sd, r in rows.items() if r["residual_corrected"][0] < 0]
    print(f"\n  residual is positive at "
          f"{len(rows) - len(neg)}/{len(rows)} seeds under the corrected mask, "
          f"and at {sum(1 for r in rows.values() if r['residual_published'][0] > 0)}"
          f"/{len(rows)} under the published one")
    (D := OUT).write_text(json.dumps({"seeds": rows}, indent=1), encoding="utf-8")
    print(f"  -> {D.name}\n")

    native = per_seed[42][0]
    fixed = per_seed[42][1]
    ks = sorted(set(box) & set(native) & set(fixed))
    complete = True
    print(f"RHUH-GBM, adapter seed 42, n = {len(ks)}")

    pipe = np.array([wt(box[k], f"pipeline:{SEG}") for k in ks])
    orac = np.array([wt(box[k], f"oracle:{SEG}") for k in ks])
    nat = np.array([wt(native[k], "direct:seed_42") for k in ks])
    fix = np.array([wt(fixed[k], "direct:seed_42") for k in ks])

    print(f"\n  {'arm':<34}{'Dice':>8}")
    print(f"  {'oracle-box':<34}{orac.mean():>8.4f}")
    print(f"  {'pipeline':<34}{pipe.mean():>8.4f}")
    print(f"  {'detector-free, published preproc':<34}{nat.mean():>8.4f}")
    print(f"  {'detector-free, corrected mask':<34}{fix.mean():>8.4f}")

    m, lo, hi = ci(fix - nat)
    p = stats.wilcoxon(fix - nat).pvalue
    print(f"\n  corrected minus published: {m:+.4f} [{lo:+.4f}, {hi:+.4f}] "
          f"p={p:.1e}, better on {(fix > nat).sum()}/{len(ks)}")

    res = json.loads(OUT.read_text(encoding="utf-8"))
    res.update({"n": len(ks), "complete": complete,
           "oracle": round(float(orac.mean()), 4),
           "pipeline": round(float(pipe.mean()), 4),
           "direct_published": round(float(nat.mean()), 4),
           "direct_corrected": round(float(fix.mean()), 4),
           "corrected_minus_published": [round(m, 4), round(lo, 4),
                                        round(hi, 4)]})

    print(f"\n  {'quantity':<24}{'published':>26}{'corrected':>26}")
    for label, a, b in (("total gap", nat - pipe, fix - pipe),
                        ("residual", nat - orac, fix - orac)):
        (ma, la, ha), (mb, lb, hb) = ci(a), ci(b)
        res[label.replace(" ", "_")] = {
            "published": [round(ma, 4), round(la, 4), round(ha, 4)],
            "corrected": [round(mb, 4), round(lb, 4), round(hb, 4)]}
        print(f"  {label:<24}{ma:+9.4f} [{la:+.4f}, {ha:+.4f}]"
              f"{mb:+9.4f} [{lb:+.4f}, {hb:+.4f}]")
    print("\n  the detector-stage term is unchanged by this: it does not involve "
          "the detector-free arm")

    OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
