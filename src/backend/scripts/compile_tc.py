"""Read the tumour-core pipeline arm against the ceiling it was meant to test.

Section 18 could say what a perfect TC detector would reach, because the oracle
arm was given ground-truth sub-region boxes, but not whether one could be
trained. A TC detector now exists, so the same decomposition the whole tumour
got is available for tumour core:

    direct - pipeline  =  (oracle - pipeline)  +  (direct - oracle)
       total gap          localisation           segmentation

The first term is what the detector loses by finding the box itself instead of
being handed it; the second is what the box-prompted segmenter loses against a
detector-free model. Reading them separately is the point: a large total says
nothing about which half to fix.

Every detector seed is compiled separately and nothing is pooled. The first
version of this script read one seed, and the conclusion it supported did not
survive the second: on BraTS-Africa the retained headroom was -25% at seed 1337
and +1% at seed 42, from detectors whose detection F1 differs by 0.0031. A
handful of catastrophic localisation failures moves the cohort mean far more
than any detection metric anticipates, so a single-seed reading of an
end-to-end pipeline out of domain is not trustworthy and the spread has to be
carried into the report rather than averaged away.

Only patients present in all three arms are used, so every number in a row is
computed over the same set. Dice is pooled over voxels within a patient and
averaged across patients, which is what the box arms already report.

    bash scripts/tc_pipeline.sh all          # seed 1337
    bash scripts/tc_pipeline.sh all 42       # replicates
    python scripts/compile_tc.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import FIX, corrected  # noqa: E402

V = DRIVE_ROOT / "reports" / "validation"
OUT = DRIVE_ROOT / "reports" / "tc_detector_summary.json"
COHORTS = [("clean", "BraTS2020 held-out"), ("rhuh", "RHUH-GBM"),
           ("brats_africa", "BraTS-Africa")]
SEGS = ["sam1_vit_b", "sam2.1_l"]
SEEDS = [1337, 42, 62]           # detector seeds; 1337 wrote the untagged files
ADAPTERS = [42, 52, 62]          # LoRA seeds; 42 wrote the untagged files
BOOT = 10000
RNG = 0





def _fix(p):
    from pathlib import Path
    p = Path(p)
    return corrected(p)


def suffix(seed: int) -> str:
    return "" if seed == 1337 else f"_seed{seed}"


def adapter_suffix(seed: int) -> str:
    return "" if seed == 42 else f"_seed{seed}"


def read(path: Path) -> dict:
    """patient -> arms, keeping the last record for a patient if any repeat."""
    if not path.exists():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue
        if "error" not in rec:
            out[rec["patient"]] = rec.get("arms", {})
    return out


def boot_ci(d: np.ndarray) -> tuple:
    """Percentile bootstrap CI of the mean difference.

    Of the mean, not of the differences: the interval has to describe where the
    mean sits, and the spread of the raw differences answers a question nobody
    asked.
    """
    rng = np.random.default_rng(RNG)
    idx = rng.integers(0, len(d), size=(BOOT, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    """b - a, with the test the study uses elsewhere."""
    d = b - a
    lo, hi = boot_ci(d)
    p = float(stats.wilcoxon(a, b).pvalue) if np.any(d) else 1.0
    return {"mean_a": float(a.mean()), "mean_b": float(b.mean()),
            "delta": float(d.mean()), "median_delta": float(np.median(d)),
            "ci95": [lo, hi], "p_wilcoxon": p,
            "n": int(len(d)), "n_better": int((d > 0).sum())}


def arm(pipe: dict, orac: dict, dire: dict, sid: str) -> dict | None:
    pk, ok = f"pipeline:{sid}", f"oracle:{sid}"
    shared = sorted(p for p in pipe
                    if pk in pipe.get(p, {})
                    and ok in orac.get(p, {})
                    and "direct:seed_42" in dire.get(p, {}))
    if not shared:
        return None

    P = np.array([pipe[p][pk]["vol_dice"] for p in shared])
    O = np.array([orac[p][ok]["regions"]["TC"]["vol_dice"] for p in shared])
    D = np.array([dire[p]["direct:seed_42"]["regions"]["TC"]["vol_dice"] for p in shared])
    boxes = np.array([pipe[p][pk]["n_boxes"] for p in shared], dtype=float)
    head = O.mean() - D.mean()

    return {
        "n": len(shared),
        "pipeline": float(P.mean()), "pipeline_median": float(np.median(P)),
        "oracle": float(O.mean()), "direct": float(D.mean()),
        "localisation_gap": float((O - P).mean()),
        "segmentation_gap": float((D - O).mean()),
        "total_gap": float((D - P).mean()),
        "retained": float((P.mean() - D.mean()) / head) if head else None,
        "mean_boxes": float(boxes.mean()),
        "patients_with_no_box": [p for p, n in zip(shared, boxes) if n == 0],
        "patients_near_zero": [p for p, x in zip(shared, P) if x < 0.05],
        "patients_below_half": int((P < 0.5).sum()),
        "pipeline_vs_direct": paired(P, D),
        "pipeline_vs_oracle": paired(P, O),
        "per_patient": {p: {"pipeline": float(x), "oracle": float(y),
                            "direct": float(z), "n_boxes": int(b)}
                        for p, x, y, z, b in zip(shared, P, O, D, boxes)},
    }


def main() -> None:
    report = {"cohorts": {}, "bootstrap": BOOT, "seed": RNG, "detector_seeds": []}
    lines = ["# Tumour core: a trained detector against the oracle ceiling", ""]

    found_seeds: list[int] = []
    for name, label in COHORTS:
        orac = read(_fix(V / f"{name}_oracle_regions.jsonl"))
        dire = read(_fix(V / f"{name}_direct.jsonl"))
        entry = {"label": label, "seeds": {}}
        n_full = len(dire)   # the completed adapter-42 run defines the cohort

        for seed in SEEDS:
            pipe = read(_fix(V / f"{name}_tc_pipeline{suffix(seed)}.jsonl"))
            if not pipe:
                continue
            per_seg = {sid: a for sid in SEGS
                       if (a := arm(pipe, orac, dire, sid)) is not None}
            if per_seg:
                entry["seeds"][str(seed)] = per_seg
                if seed not in found_seeds:
                    found_seeds.append(seed)

        if not entry["seeds"]:
            lines += [f"## {label}", "", "_pipeline arm not run yet._", ""]
            continue

        # The detector-free arm at each LoRA seed. The primary tables stay on
        # adapter 42, matching the rest of the paper; this is reported beside
        # them so the two arms' replicate spreads can be compared at all.
        free, partial = {}, {}
        for aseed in ADAPTERS:
            recs = read(_fix(V / f"{name}_direct{adapter_suffix(aseed)}.jsonl"))
            key = f"direct:seed_{aseed}"
            rows = [r[key]["regions"] for r in recs.values() if key in r]
            if not rows:
                continue
            vals = {r: float(np.mean([x[r]["vol_dice"] for x in rows if r in x]))
                    for r in ("WT", "TC", "ET") if any(r in x for x in rows)}
            vals["n"] = len(rows)
            # A run still in progress averages over a different subset of
            # patients. That is not comparable with the completed seeds but
            # looks exactly like one in a table, so it is held aside.
            (partial if len(rows) < n_full else free)[str(aseed)] = vals
        if free:
            entry["detector_free_seeds"] = free
        if partial:
            entry["detector_free_incomplete"] = partial

        report["cohorts"][name] = entry
        lines += [f"## {label}", ""]

        for sid in SEGS:
            rows = [(s, entry["seeds"][s][sid]) for s in entry["seeds"]
                    if sid in entry["seeds"][s]]
            if not rows:
                continue
            a0 = rows[0][1]
            lines += [
                f"### {sid} (n = {a0['n']})", "",
                "| detector seed | pipeline | median | retained | total gap | 95% CI | p | <0.05 | no box |",
                "|---|---|---|---|---|---|---|---|---|"]
            for s, a in rows:
                pv = a["pipeline_vs_direct"]
                lines.append(
                    f"| {s} | **{a['pipeline']:.4f}** | {a['pipeline_median']:.4f} | "
                    f"{a['retained'] * 100:.0f}% | {a['total_gap']:+.4f} | "
                    f"[{pv['ci95'][0]:+.4f}, {pv['ci95'][1]:+.4f}] | "
                    f"{pv['p_wilcoxon']:.2g} | {len(a['patients_near_zero'])} | "
                    f"{len(a['patients_with_no_box'])} |")
            lines += ["",
                      f"Fixed for this cohort and segmenter: oracle {a0['oracle']:.4f}, "
                      f"detector-free {a0['direct']:.4f}. A negative total gap means "
                      f"the pipeline is ahead.", ""]
            if len(rows) > 1:
                pipes = [a["pipeline"] for _, a in rows]
                rets = [a["retained"] for _, a in rows]
                signs = {a["total_gap"] < 0 for _, a in rows}
                lines += [
                    f"Across {len(rows)} detector seeds the pipeline mean spans "
                    f"{min(pipes):.4f}-{max(pipes):.4f} "
                    f"(spread {max(pipes) - min(pipes):.4f}) and retained headroom "
                    f"{min(rets) * 100:.0f}% to {max(rets) * 100:.0f}%. "
                    + ("The sign of the total gap is consistent across seeds."
                       if len(signs) == 1 else
                       "**The sign of the total gap is not consistent across seeds**, "
                       "so which model leads on this cohort is not resolved by "
                       "these runs."), ""]

    report["detector_seeds"] = found_seeds
    OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = OUT.with_suffix(".md")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {OUT.name} and {md.name} (detector seeds: {found_seeds})")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
