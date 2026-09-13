"""Every manuscript quantity that moves when the corrected mask becomes primary.

The box arms were re-run under a corrected brain mask; the detector-free arm was
not, so on RHUH-GBM the three arms saw different volumes. Making the corrected
run primary puts them back on the same inputs. This recomputes each quantity the
change touches so the edit is made from numbers rather than by hand.

    python scripts/rhuh_bgfix_primary.py
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
SEG = "sam2.1_l"
DRAWS = 20000


def load(name: str) -> dict:
    p = V / name
    out = {}
    for line in io.open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            if "error" not in r:
                out[r["patient"]] = r
    return out


def reg(rec, arm, region="WT", field="vol_dice"):
    a = rec["arms"][arm]
    if "regions" in a:
        return a["regions"].get(region, {}).get(field)
    return a.get(field) if region == "WT" else None


def ci(d, seed=1337):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    bs = np.sort([d[rng.integers(0, len(d), len(d))].mean() for _ in range(DRAWS)])
    return d.mean(), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def unpaired(a, b):
    bs = np.sort([np.random.default_rng([1337, i]).choice(a, len(a)).mean()
                  - np.random.default_rng([7331, i]).choice(b, len(b)).mean()
                  for i in range(DRAWS)])
    return (a.mean() - b.mean(), float(np.percentile(bs, 2.5)),
            float(np.percentile(bs, 97.5)))


def main() -> None:
    box = load("rhuh_box_normfix.jsonl")
    orr = load("rhuh_oracle_regions_normfix.jsonl")
    fix = load("rhuh_direct_seed42_bgfix.jsonl")
    ks = sorted(set(box) & set(fix) & set(orr))

    pipe = np.array([reg(box[k], f"pipeline:{SEG}") for k in ks])
    orac = np.array([reg(box[k], f"oracle:{SEG}") for k in ks])
    free = np.array([reg(fix[k], "direct:seed_42") for k in ks])

    print(f"RHUH-GBM under the corrected mask, n = {len(ks)}\n")
    print("Table 2 row")
    g = free - pipe
    m, lo, hi = ci(g)
    print(f"  oracle {orac.mean():.4f}  pipeline {pipe.mean():.4f}  "
          f"detector-free {free.mean():.4f}")
    print(f"  gap {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  p={stats.wilcoxon(g).pvalue:.1e}  "
          f"wins {(g > 0).sum()}/{len(g)}")

    print("\nTables 3 and 4")
    r = free - orac
    rm, rlo, rhi = ci(r)
    print(f"  residual {rm:+.4f} [{rlo:+.4f}, {rhi:+.4f}]")

    # the contrast against the in-domain arm, which is unchanged
    cb = load("clean_box.jsonl")
    cd = load("clean_direct.jsonl")
    cks = sorted(set(cb) & set(cd))
    c_res = (np.array([reg(cd[k], "direct:seed_42") for k in cks])
             - np.array([reg(cb[k], f"oracle:{SEG}") for k in cks]))
    cm, clo, chi = unpaired(r, c_res)
    print(f"  residual contrast vs in-domain {cm:+.4f} [{clo:+.4f}, {chi:+.4f}]")

    print("\nTable 10, three adapter seeds")
    gaps = {}
    for sd in (42, 52, 62):
        f2 = load(f"rhuh_direct_seed{sd}_bgfix.jsonl")
        kk = sorted(set(box) & set(f2))
        d = (np.array([reg(f2[k], f"direct:seed_{sd}") for k in kk])
             - np.array([reg(box[k], f"pipeline:{SEG}") for k in kk]))
        m2, l2, h2 = ci(d)
        gaps[sd] = m2
        print(f"  seed {sd}: {m2:+.4f} [{l2:+.4f}, {h2:+.4f}]")
    print(f"  spread {max(gaps.values()) - min(gaps.values()):.4f}")

    print("\nCollapses below 0.5")
    print(f"  pipeline {(pipe < 0.5).sum()}/{len(ks)}  "
          f"oracle {(orac < 0.5).sum()}/{len(ks)}  "
          f"detector-free {(free < 0.5).sum()}/{len(ks)}")
    keep = (pipe >= 0.5) & (free >= 0.5)
    ge = g[keep]
    print(f"  excluding collapses {ge.mean():+.4f} ({(ge > 0).sum()}/{len(ge)}, "
          f"p={stats.wilcoxon(ge).pvalue:.3f})")

    print("\nTable 14, sub-regions")
    for region in ("TC", "ET"):
        o = np.array([reg(orr[k], f"oracle:{SEG}", region) for k in ks], float)
        f3 = np.array([reg(fix[k], "direct:seed_42", region) for k in ks], float)
        print(f"  {region}: oracle {o.mean():.4f}  detector-free {f3.mean():.4f}  "
              f"headroom {o.mean() - f3.mean():+.4f}")

    print("\n  the detector-stage term is unchanged: "
          f"{ci(orac - pipe)[0]:+.4f}")


if __name__ == "__main__":
    main()
