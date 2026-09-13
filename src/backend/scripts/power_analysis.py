"""How much of the n=24 result is signal, and what would n=64 buy?

The mutually held-out cohort answers "which direction" convincingly and "by how
much" only loosely. This quantifies both halves of that sentence so the report
can say it with numbers instead of a hedge:

  * a bootstrap confidence interval on the paired mean difference -- the honest
    width of the "+0.0576" claim;
  * observed power, by resampling patients and re-running the same Wilcoxon
    test that produced the headline p-value;
  * the minimum detectable effect at n=24 versus n=64, which is what adding the
    40 preoperative RHUH-GBM patients would give.

One caveat governs every projection below. Resampling the BraTS differences to
simulate n=64 assumes the external cohort behaves like the internal one -- which
is the very thing an external cohort is run to test. These are planning numbers
for deciding whether the run is worth it, not predictions of its outcome. If
RHUH's effect is smaller (different scanner, different annotation protocol,
GBM-only rather than HGG+LGG), the realised interval will be wider than the
projection here.

    python scripts/power_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "power_analysis.json"

RNG = np.random.default_rng(20260824)
BOOT = 20000
ALPHA = 0.05
TARGET_POWER = 0.80
RHUH_N = 40          # preoperative studies in RHUH-GBM
BOOT_POWER = 4000    # resamples per power estimate; Wilcoxon per draw is the cost


def load(name: str) -> Dict[str, dict]:
    p = V / name
    out: Dict[str, dict] = {}
    if not p.exists():
        raise SystemExit(f"missing {p}")
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "error" in r or not r.get("arms"):
            continue
        out[r["patient"]] = r
    return out


def boot_ci(d: np.ndarray, stat_fn, n_boot: int = BOOT) -> Tuple[float, float]:
    """Percentile bootstrap interval for a statistic of the paired differences.

    Note this is an interval for the *statistic*, not the 2.5/97.5 percentiles of
    the differences themselves -- the latter describes how much patients vary,
    which is a much wider number and not a confidence interval for anything.
    """
    idx = RNG.integers(0, d.size, size=(n_boot, d.size))
    draws = stat_fn(d[idx])
    return float(np.percentile(draws, 100 * ALPHA / 2)), float(np.percentile(draws, 100 * (1 - ALPHA / 2)))


def observed_power(d: np.ndarray, n: int, n_boot: int = BOOT_POWER) -> float:
    """Fraction of resampled cohorts of size n where Wilcoxon still rejects.

    Resampling from the observed differences treats the sample as the population,
    so this is power against the *observed* effect, not against a hypothesised one.
    """
    hits = 0
    for _ in range(n_boot):
        s = RNG.choice(d, size=n, replace=True)
        if np.all(s == 0):
            continue
        try:
            if stats.wilcoxon(s)[1] < ALPHA:
                hits += 1
        except ValueError:
            continue
    return hits / n_boot


def mde(sd: float, n: int) -> float:
    """Smallest paired mean difference a two-sided t-test detects at 80% power."""
    from scipy.stats import nct, t as tdist

    crit = tdist.ppf(1 - ALPHA / 2, n - 1)
    lo, hi = 0.0, 10 * sd
    for _ in range(80):
        mid = (lo + hi) / 2
        ncp = mid / (sd / np.sqrt(n))
        power = 1 - nct.cdf(crit, n - 1, ncp) + nct.cdf(-crit, n - 1, ncp)
        if power < TARGET_POWER:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def analyse(name: str, base: np.ndarray, test: np.ndarray, base_key: str = "") -> dict:
    d = test - base
    n = d.size
    sd = float(d.std(ddof=1))
    mean = float(d.mean())
    ci = boot_ci(d, lambda m: m.mean(axis=1))
    p = float(stats.wilcoxon(base, test)[1])
    dz = mean / sd if sd else float("inf")

    proj_n = n + RHUH_N
    idx = RNG.integers(0, n, size=(BOOT, proj_n))
    proj_draws = d[idx].mean(axis=1)
    proj_ci = (float(np.percentile(proj_draws, 2.5)), float(np.percentile(proj_draws, 97.5)))

    res = {
        "contrast": name, "base": base_key, "n": n,
        "mean_diff": round(mean, 4), "median_diff": round(float(np.median(d)), 4),
        "sd_diff": round(sd, 4), "cohens_dz": round(dz, 3),
        "wins": int((d > 0).sum()),
        "ci95": [round(ci[0], 4), round(ci[1], 4)],
        "ci95_halfwidth": round((ci[1] - ci[0]) / 2, 4),
        "wilcoxon_p": p,
        "power_at_n": round(observed_power(d, n), 3),
        "mde_at_n": round(mde(sd, n), 4),
        "projected_n": proj_n,
        "projected_ci95": [round(proj_ci[0], 4), round(proj_ci[1], 4)],
        "projected_ci95_halfwidth": round((proj_ci[1] - proj_ci[0]) / 2, 4),
        "projected_power": round(observed_power(d, proj_n), 3),
        "mde_at_projected_n": round(mde(sd, proj_n), 4),
    }
    return res


def main() -> None:
    cb, cd = load("clean_box.jsonl"), load("clean_direct.jsonl")
    common = sorted(set(cb) & set(cd))
    if not common:
        raise SystemExit("no patients carry both the box and the detector-free arm")

    direct = np.array([cd[p]["arms"][list(cd[p]["arms"])[0]]["vol_dice"] for p in common])
    arms = {k: np.array([cb[p]["arms"][k]["vol_dice"] for p in common])
            for k in ("oracle:sam1_vit_b", "oracle:sam2.1_l",
                      "pipeline:sam1_vit_b", "pipeline:sam2.1_l")
            if all(k in cb[p]["arms"] for p in common)}

    print(f"{len(common)} patients with every arm\n")
    contrasts = [
        ("detector-free minus RT-DETR -> SAM 2.1-L", "pipeline:sam2.1_l"),
        ("detector-free minus RT-DETR -> SAM 1 ViT-B", "pipeline:sam1_vit_b"),
        ("detector-free minus oracle GT box -> SAM 2.1-L", "oracle:sam2.1_l"),
    ]
    results = []
    for label, key in contrasts:
        if key not in arms:
            continue
        r = analyse(label, arms[key], direct, key)
        results.append(r)
        print(f"=== {label} ===")
        print(f"  n {r['n']}   mean {r['mean_diff']:+.4f}   median {r['median_diff']:+.4f}"
              f"   sd {r['sd_diff']:.4f}   dz {r['cohens_dz']:.2f}   wins {r['wins']}/{r['n']}")
        print(f"  95% CI (bootstrap) [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}]"
              f"  -> +/-{r['ci95_halfwidth']:.4f}     Wilcoxon p {r['wilcoxon_p']:.2e}")
        print(f"  power against the observed effect: {r['power_at_n']:.3f}")
        print(f"  smallest effect detectable at 80% power: {r['mde_at_n']:+.4f}")
        print(f"  with RHUH-GBM (n={r['projected_n']}): CI +/-{r['projected_ci95_halfwidth']:.4f}"
              f"  (from +/-{r['ci95_halfwidth']:.4f}),"
              f"  power {r['projected_power']:.3f},"
              f"  detectable effect {r['mde_at_projected_n']:+.4f}")
        print()

    primary = results[0]
    print("--- reading ---")
    shrink = 1 - primary["projected_ci95_halfwidth"] / primary["ci95_halfwidth"]
    print(f"The direction is not in question: power {primary['power_at_n']:.2f} at n={primary['n']}, "
          f"{primary['wins']}/{primary['n']} patients.")
    print(f"The magnitude is: the interval is +/-{primary['ci95_halfwidth']:.4f} wide, "
          f"{primary['ci95_halfwidth'] / abs(primary['mean_diff']) * 100:.0f}% of the effect itself.")
    print(f"Adding {RHUH_N} external patients narrows it by {shrink * 100:.0f}% "
          f"to +/-{primary['projected_ci95_halfwidth']:.4f}, assuming they behave like these -- "
          f"which is the assumption the run exists to test.")

    OUT.write_text(json.dumps({
        "n_patients": len(common), "patients": common,
        "rhuh_n_assumed": RHUH_N, "alpha": ALPHA, "target_power": TARGET_POWER,
        "bootstrap_draws": BOOT, "power_draws": BOOT_POWER,
        "contrasts": results,
    }, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
