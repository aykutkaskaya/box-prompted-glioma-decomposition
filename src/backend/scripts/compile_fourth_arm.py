"""Table 2's residual intervals and Table 3, from the per-patient files.

fourth_arm.py scores the arm and writes one record per patient; nothing turned
those into the intervals the two tables carry, so twelve numbers in the paper's
two central tables had no released computation behind them while Section 2.6
said every number in Methods and Results did. This closes that: it writes the
residual and its between-cohort contrasts, and the prompt and model terms of
the fourth-arm split, to a report file audit_numbers.py can check the text
against.

Paired within a cohort and unpaired between, 20 000 draws at generator seed
1337, matching Section 2.5.

    python scripts/compile_fourth_arm.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "fourth_arm_summary.json"
SEG = "sam2.1_l"
ADAPTER = 42
DRAWS = 20000
SEED = 1337
COHORTS = [("clean", "BraTS2020 (shared held-out set)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]


def read(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if "error" not in r:
            out[r["patient"]] = r.get("arms", {})
    return out


def arms(cohort: str) -> dict:
    """Per-patient Dice for the four arms, on the patients all four cover.

    RHUH-GBM's corrected runs are the primary ones and live under different
    names; corrected() is the only correct way to reach them.
    """
    box = read(corrected(V / f"{cohort}_box.jsonl", SEG))
    free = read(corrected(V / f"{cohort}_direct.jsonl", SEG))
    fourth = read(V / f"{cohort}_fourth_arm.jsonl")
    ok, pk = f"oracle:{SEG}", f"pipeline:{SEG}"
    fk, bk = f"direct:seed_{ADAPTER}", f"oraclebox_medsam3:seed_{ADAPTER}"
    ids = sorted(p for p in box
                 if ok in box.get(p, {}) and pk in box.get(p, {})
                 and fk in free.get(p, {}) and bk in fourth.get(p, {}))
    if not ids:
        return {}
    g = lambda src, p, k: src[p][k]["vol_dice"]  # noqa: E731
    return {
        "ids": ids,
        "orc": np.array([g(box, p, ok) for p in ids]),
        "pipe": np.array([g(box, p, pk) for p in ids]),
        "free": np.array([g(free, p, fk) for p in ids]),
        "box": np.array([g(fourth, p, bk) for p in ids]),
    }


def ci(rng, *samples):
    """Percentile interval on the mean, or on a difference of two means."""
    stat = []
    for _ in range(DRAWS):
        means = [s[rng.integers(0, len(s), len(s))].mean() for s in samples]
        stat.append(means[0] if len(means) == 1 else means[0] - means[1])
    stat = np.sort(stat)
    return (round(float(np.percentile(stat, 2.5)), 4),
            round(float(np.percentile(stat, 97.5)), 4))


# the three quantities the two tables carry, each a per-patient difference
TERMS = [
    ("residual", lambda a: a["free"] - a["orc"]),
    ("prompt", lambda a: a["box"] - a["free"]),
    ("model", lambda a: a["box"] - a["orc"]),
]


def main() -> None:
    rng = np.random.default_rng(SEED)
    result = {"draws": DRAWS, "seed": SEED, "segmenter": SEG,
              "adapter_seed": ADAPTER, "within": {}, "between": {}}
    data = {}

    print(f"fourth-arm terms, {DRAWS} bootstrap draws\n")
    print(f"{'cohort':<32}{'term':>10}{'n':>5}{'mean':>10}{'95% CI':>24}")
    for c, label in COHORTS:
        a = arms(c)
        if not a:
            print(f"{label:<32}   no patient carries all four arms")
            continue
        for name, fn in TERMS:
            d = fn(a)
            data[(c, name)] = d
            lo, hi = ci(rng, d)
            result["within"][f"{c}/{name}"] = {
                "n": len(d), "mean": round(float(d.mean()), 4), "ci": [lo, hi],
                "excludes_zero": bool(lo > 0 or hi < 0)}
            print(f"{label:<32}{name:>10}{len(d):>5}{d.mean():>+10.4f}"
                  f"   [{lo:+.4f}, {hi:+.4f}]")

    print("\nexternal minus held-out\n")
    print(f"{'contrast':<32}{'term':>10}{'difference':>12}{'95% CI':>24}")
    for c, label in COHORTS[1:]:
        for name, _ in TERMS:
            if (c, name) not in data or ("clean", name) not in data:
                continue
            a, b = data[(c, name)], data[("clean", name)]
            diff = round(float(a.mean() - b.mean()), 4)
            lo, hi = ci(rng, a, b)
            result["between"][f"{c}-clean/{name}"] = {
                "difference": diff, "ci": [lo, hi],
                "excludes_zero": bool(lo > 0 or hi < 0)}
            print(f"{label + ' - held out':<32}{name:>10}{diff:>+12.4f}"
                  f"   [{lo:+.4f}, {hi:+.4f}]")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
