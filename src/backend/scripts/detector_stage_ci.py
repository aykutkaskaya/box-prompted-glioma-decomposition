"""Uncertainty on the detector-stage term and on its between-cohort contrasts.

The paper's headline is that this term is an order of magnitude larger on the
external cohorts than on the held-out one. Table 2 carries intervals for the
end-to-end gap, which is a different quantity; the claim the title rests on had
none. This supplies them.

Two kinds of interval, because the two comparisons are not the same shape.

Within a cohort the two arms see the same patients, so oracle-box minus pipeline
is a paired difference and the interval is a percentile bootstrap over patients
of its mean, matching §3.5.

Between cohorts the patients are different people, so the contrast is a
difference of two independent means. Each cohort is resampled on its own and
the two means differenced, which is the two-sample analogue and does not
pretend to a pairing that does not exist.

    python scripts/detector_stage_ci.py [--segmenter sam1_vit_b]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import FIX, corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
# the second segmenter is a robustness check on the same stored predictions, so
# it is the same computation with one name changed rather than a second script
SEG = sys.argv[sys.argv.index("--segmenter") + 1] if "--segmenter" in sys.argv     else "sam2.1_l"
OUT = DRIVE_ROOT / "reports" / (
    "detector_stage_ci.json" if SEG == "sam2.1_l"
    else f"detector_stage_ci_{SEG.replace('.', '')}.json")
DRAWS = 20000
SEED = 1337
COHORTS = [("clean", "BraTS2020 (mutually held out)"),
           ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]
SEEDS = [1337, 42]





def _fix(p):
    """The corrected file for `p`, for whichever segmenter is being scored.

    The corrected box run exists only for the primary segmenter; the second
    segmenter's corrected arms live in the corrected oracle-regions file, so
    SEG has to reach corrected(). Dropping it silently returned RHUH-GBM's
    primary-segmenter file, which carries no SAM 1 arms -- the cohort then
    scored n=0 and fell out of the table without an error.
    """
    return corrected(p, SEG)


def read(path: Path) -> dict:
    out = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:
                out[r["patient"]] = r.get("arms", {})
    return out


def terms(cohort: str, seed: int) -> tuple:
    """Per-patient oracle-box minus pipeline, on the patients both arms cover."""
    box = read(_fix(V / f"{cohort}_box.jsonl"))
    pipe = (box if seed == 1337
            else read(_fix(V / f"{cohort}_wt_pipeline_seed{seed}.jsonl")))
    ok, pk = f"oracle:{SEG}", f"pipeline:{SEG}"
    ids = sorted(p for p in box if ok in box[p] and p in pipe and pk in pipe[p])
    # A seed this segmenter was never run at leaves its pipeline file without
    # the arm, and the caller reports the cohort as not run. Reading a box
    # file that carries none of this segmenter's oracle arms is the different
    # failure: the wrong file for the segmenter. That one scored n=0 and left
    # the table silently, which is how the SAM 1 row lost RHUH-GBM entirely.
    if box and not any(ok in arms for arms in box.values()):
        raise SystemExit(f"{cohort}: {_fix(V / f'{cohort}_box.jsonl').name} "
                         f"carries no '{ok}' arm -- wrong file for segmenter "
                         f"{SEG}, not an empty cohort")
    d = np.array([box[p][ok]["vol_dice"] - pipe[p][pk]["vol_dice"] for p in ids])
    return ids, d


def bca(d, draws: int = DRAWS, seed: int = SEED):
    """Bias-corrected and accelerated interval, beside the percentile one.

    The per-patient term is right-skewed, and a plain percentile interval
    corrects neither median bias nor acceleration in that regime. Reported so a
    reader can see the reading does not depend on the choice.
    """
    from scipy import stats
    d = np.asarray(d)
    rng = np.random.default_rng(seed)
    bs = np.sort([rng.choice(d, len(d)).mean() for _ in range(draws)])
    p0 = (bs < d.mean()).mean()
    z0 = stats.norm.ppf(min(max(p0, 1e-9), 1 - 1e-9))
    jk = np.array([np.delete(d, i).mean() for i in range(len(d))])
    den = 6 * (((jk.mean() - jk) ** 2).sum()) ** 1.5
    a = (((jk.mean() - jk) ** 3).sum() / den) if den else 0.0

    def q(p):
        z = stats.norm.ppf(p)
        return stats.norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z))) * 100

    return (round(float(np.percentile(bs, q(0.025))), 4),
            round(float(np.percentile(bs, q(0.975))), 4))


def ci(rng, draws: int, *samples) -> tuple:
    """Percentile interval on the mean, or on a difference of two means."""
    stat = []
    for _ in range(draws):
        means = [s[rng.integers(0, len(s), len(s))].mean() for s in samples]
        stat.append(means[0] if len(means) == 1 else means[0] - means[1])
    stat = np.sort(stat)
    return float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5))


def main() -> None:
    rng = np.random.default_rng(SEED)
    result = {"draws": DRAWS, "seed": SEED, "segmenter": SEG,
              "within": {}, "between": {}}

    print(f"detector-stage term, {DRAWS} bootstrap draws\n")
    print(f"{'cohort':<30}{'seed':>6}{'n':>5}{'mean':>10}"
          f"{'95% CI':>24}")
    data = {}
    for c, label in COHORTS:
        for sd in SEEDS:
            ids, d = terms(c, sd)
            if not len(d):
                print(f"{label:<30}{sd:>6}{'-':>5}   not run under {SEG}")
                continue
            data[(c, sd)] = d
            lo, hi = ci(rng, DRAWS, d)
            result["within"][f"{c}/{sd}"] = {
                "n": len(d), "mean": round(float(d.mean()), 4),
                "ci": [round(lo, 4), round(hi, 4)],
                "bca": list(bca(d))}
            print(f"{label:<30}{sd:>6}{len(d):>5}{d.mean():>+10.4f}"
                  f"   [{lo:+.4f}, {hi:+.4f}]")

    print(f"\nexternal minus held-out, same seed\n")
    print(f"{'contrast':<30}{'seed':>6}{'difference':>12}{'95% CI':>24}")
    base = "clean"
    for c, label in COHORTS[1:]:
        for sd in SEEDS:
            if (c, sd) not in data or (base, sd) not in data:
                continue
            a, b = data[(c, sd)], data[(base, sd)]
            diff = float(a.mean() - b.mean())
            lo, hi = ci(rng, DRAWS, a, b)
            result["between"][f"{c}-clean/{sd}"] = {
                "difference": round(diff, 4), "ci": [round(lo, 4), round(hi, 4)],
                "excludes_zero": bool(lo > 0)}
            print(f"{label + ' - held out':<30}{sd:>6}{diff:>+12.4f}"
                  f"   [{lo:+.4f}, {hi:+.4f}]")

    all_pos = all(v["excludes_zero"] for v in result["between"].values())
    result["all_contrasts_exclude_zero"] = bool(all_pos)
    print("\n  every external-minus-held-out interval excludes zero"
          if all_pos else
          "\n  at least one external-minus-held-out interval includes zero")

    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"\n-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
