"""Check every number the manuscript states against the files that produced it.

Spot checks were done as sections were written, but a paper is read as a whole
and the failure that embarrasses an author is the same quantity written two
different ways in two sections. This recomputes the values that matter from
reports/ and reports/validation/ and looks for each in the draft, so a stale
number surfaces as a missing string rather than as a reviewer's question.

Absence of a value in the text is reported, never repaired: the fix depends on
which of the two is right, and that is a judgement.

    python scripts/audit_numbers.py
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402
from core.rhuh import corrected  # noqa: E402

DOCS = {"paper": DRIVE_ROOT / "PAPER_DRAFT.md",
        "tr": DRIVE_ROOT / "PAPER_DRAFT_TR.md",
        "runs": DRIVE_ROOT / "RUNS_ANALYSIS.md"}
# abridged documents legitimately omit values, so only the reverse direction --
# is every number in the text still derivable -- is enforced on them
ABRIDGED = {"tr"}
V = DRIVE_ROOT / "reports" / "validation"
R = DRIVE_ROOT / "reports"
SEG = "sam2.1_l"
COHORTS = ["clean", "rhuh", "brats_africa"]

# The LoRA adapter lives in a separate checkout, so it has no sensible
# default. Set MEDSAM3_DIR to point at it; the checks that need it are
# skipped when it is absent.
MEDSAM3_DIR = Path(os.environ.get("MEDSAM3_DIR", "medsam3-not-set"))


def rhuh(p: Path) -> Path:
    """Redirect a RHUH-GBM artefact to its corrected-normalisation rerun."""
    return corrected(p)


def read(p: Path) -> dict:
    p = rhuh(p)
    out = {}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except Exception:
                continue
            if "error" not in r:
                out[r["patient"]] = r.get("arms", {})
    return out


def interval_count(draft: str, supp: str) -> int:
    """Distinct confidence intervals in the prose, under Section 2.5's rule.

    A bracketed pair of signed decimals or percentages is an interval; a
    citation bracket holds bare integers and does not match. Bounds are
    compared at three decimals so that a lower-precision restatement of an
    interval -- the abstract quotes several -- counts once with its source.
    """
    body = (draft.split("## References")[0] + chr(10)
            + supp.split("## References")[0]).replace(chr(160), " ")
    # a bound may be a whole percentage: S6's ratio interval reads
    # "[68%, 93%]", which a decimal-only pattern does not see
    pat = re.compile(r"\[([+\u2212-]?\d+(?:\.\d+)?%?),\s*([+\u2212-]?\d+(?:\.\d+)?%?)\]")
    seen = set()
    for lo, hi in pat.findall(body):
        # "[10, 11]" is a citation, not an interval; an interval carries a
        # decimal point or a percent sign on at least one of its bounds
        if not any(c in lo + hi for c in ".%"):
            continue
        f = lambda x: round(float(x.replace("\u2212", "-").rstrip("%")), 3)
        seen.add((f(lo), f(hi)))
    return len(seen)


def main() -> None:
    ap = argparse.ArgumentParser()
    # Both documents are checked against the same recomputed values, so
    # agreeing with the files is also how they are shown to agree with
    # each other -- a direct document-to-document diff would only find
    # the quantities that happen to be worded identically.
    ap.add_argument("--doc", choices=sorted(DOCS), default="paper")
    a = ap.parse_args()
    target = DOCS[a.doc]
    # `a` is rebound later in this function, so capture the flag while it is
    # still the parsed arguments
    abridged = a.doc in ABRIDGED
    # The manuscript is not redistributed with the code. Without it the
    # recomputation still runs and every value is printed for comparison
    # against the published article; only the string matching is unavailable.
    have_doc = target.exists()
    text = io.open(target, encoding="utf-8").read() if have_doc else ""
    # analyses that moved to the supporting information are still published, so
    # their figures are checked against it rather than reported as missing
    supp = DRIVE_ROOT / "SUPPLEMENTARY.md"
    if a.doc == "paper" and supp.exists():
        text += "\n" + supp.read_text(encoding="utf-8")
    # The manuscript sets minus signs typographically (U+2212) while
    # format() writes ASCII hyphens. Comparing them raw reported every
    # negative quantity as missing, which would have buried a real
    # mismatch among false alarms.
    # The manuscript also sets a non-breaking space inside intervals and
    # thousands separators, so that "[-0.0078, +0.0219]" and "3 215" are
    # not split across lines. format() writes an ordinary space.
    # the thousands separator is typographic in the documents and absent from
    # format(), so it is removed on both sides rather than matched
    _thou = re.compile(r"(?<=\d)[" + chr(160) + r" ](?=\d{3}(?!\d))")
    norm = lambda s: _thou.sub("", s.replace(chr(8722), "-")
                                     .replace(chr(8211), "-")
                                     .replace(chr(160), " "))
    text = norm(text)
    # The reverse direction reads the prose without the bibliographies. Each
    # document is trimmed on its own: splitting the joined text on the first
    # reference heading discarded the supplement along with the manuscript's
    # reference list, leaving that direction blind to half the corpus.
    def _drop_refs(doc: str) -> str:
        for head in ("## References", "## Kaynaklar"):
            doc = doc.split(head)[0]
        return doc

    _body_parts = [io.open(target, encoding="utf-8").read()] if have_doc else []
    if a.doc == "paper" and supp.exists():
        _body_parts.append(supp.read_text(encoding="utf-8"))
    prose = norm("\n".join(_drop_refs(d) for d in _body_parts))
    checks: list = []
    unverifiable: list = []
    # optional inputs that were not present, so their checks did not run
    skipped: list = []

    def want(label: str, value: str) -> None:
        v = norm(value)
        found = v in text
        if not found and abridged:
            # Turkish writes the sign before the per-cent sign and the per-cent
            # sign before the number: +32.0% is written +%32.0
            m = re.fullmatch(r"([+-]?)([\d.]+)%?", v)
            if m:
                sign, num = m.groups()
                found = (sign + "%" + num) in text
        checks.append((label, value, found))

    def want_flat(label: str, value: str) -> None:
        """For a value the text prints as a phrase, which the hard wrap splits."""
        checks.append((label, value, norm(value) in " ".join(text.split())))

    def want_ci(label: str, lo: float, hi: float) -> None:
        """An interval is one fact.

        Checking the bounds separately lets each of them match some unrelated
        number elsewhere in the manuscript while the interval as written is a
        pair that was never computed. That is how four stale RHUH-GBM bounds
        survived a clean run of this audit.
        """
        v = f"[{lo:+.4f}, {hi:+.4f}]"
        # an interval can be split across the hard wrap, so the pair is looked
        # for in a whitespace-flattened copy rather than line by line
        flat = " ".join(text.split())
        checks.append((label, v, norm(v) in flat))

    # The manuscript and its supplementary carry different numbers, so each is
    # checked against only the values it is meant to contain.
    # Appendix A carries the replication noise floor
    rep = json.loads((R / "seed_replicates.json").read_text(encoding="utf-8"))
    noise = rep["noise"]
    want("seed-to-seed F1 noise", f"{noise['test.f1']['mean']:.4f}")
    # the coverage figures were quoted only by the overlap-loss ablation, which
    # is provenance for the detector rather than a result of this paper and is
    # no longer reported here

    # Run 37 retrained at a second seed, which is what makes the
    # whole-tumour/tumour-core detection comparison of Section 3.4 seed-paired
    # rather than a comparison against the ablation's spread.
    # Only seed 42 retrains run 37; adding 62 put a permanent entry in
    # `skipped`, and `(loose and not skipped)` then disarmed the
    # superseded-value gate on every run. The value is read from the per-run
    # summary rather than the ablation CSV, which the released archive does
    # not carry -- a reader downloading it could not otherwise re-derive the
    # one number Section 3.4's seed-paired comparison rests on.
    for _sd in (42,):
        _hits = sorted((DRIVE_ROOT / "experiments_repeated" / f"seed_{_sd}")
                       .glob("run_037_*/summary/run_summary.json"))
        if not _hits:
            skipped.append(f"run 37 at seed {_sd}: run summary not present")
            continue
        _r = json.loads(_hits[0].read_text(encoding="utf-8"))
        want(f"run 37 test F1 at seed {_sd}", f"{_r['test']['f1']:.4f}")

    # ---------------- whole-tumour three-arm decomposition, from the JSONL
    for c in COHORTS:
        box, direct = read(V / f"{c}_box.jsonl"), read(V / f"{c}_direct.jsonl")
        ps = sorted(p for p in box
                    if f"oracle:{SEG}" in box[p] and f"pipeline:{SEG}" in box[p]
                    and "direct:seed_42" in direct.get(p, {}))
        P = np.array([box[p][f"pipeline:{SEG}"]["vol_dice"] for p in ps])
        O = np.array([box[p][f"oracle:{SEG}"]["vol_dice"] for p in ps])
        D = np.array([direct[p]["direct:seed_42"]["regions"]["WT"]["vol_dice"]
                      for p in ps])
        want(f"{c} WT pipeline", f"{P.mean():.4f}")
        want(f"{c} WT oracle-box", f"{O.mean():.4f}")
        want(f"{c} WT detector-free", f"{D.mean():.4f}")
        want(f"{c} WT localisation", f"{(O - P).mean():+.4f}")
        want(f"{c} WT residual", f"{(D - O).mean():+.4f}")
        want(f"{c} n", str(len(ps)))

    # ---------------------------------- tumour core, per detector seed
    tc = json.loads((R / "tc_detector_summary.json").read_text(encoding="utf-8"))
    spreads = {}
    for c, e in tc["cohorts"].items():
        vals = sorted(per[SEG]["pipeline"] for per in e["seeds"].values())
        want(f"{c} TC pipeline, lowest seed", f"{vals[0]:.4f}")
        want(f"{c} TC pipeline, highest seed", f"{vals[-1]:.4f}")
        a0 = next(iter(e["seeds"].values()))[SEG]
        want(f"{c} TC oracle-box", f"{a0['oracle']:.4f}")
        want(f"{c} TC detector-free", f"{a0['direct']:.4f}")
        fr = [v["TC"] for v in e.get("detector_free_seeds", {}).values()]
        if len(fr) > 1:
            spreads[c] = max(fr) - min(fr)

    if len(spreads) == 3:
        want_flat("detector-free TC spreads, as the text lists them",
             ", ".join(f"{spreads[c]:.4f}" for c in
                       ("clean", "rhuh", "brats_africa"))
             .replace(", " + f"{spreads['brats_africa']:.4f}",
                      " and " + f"{spreads['brats_africa']:.4f}"))

    # ------------------------------- the cohort detection F1 of Table 2
    # Nothing checked this column, and RHUH-GBM's entry stayed at the
    # pre-correction 0.844 while every other RHUH figure moved to the
    # corrected run, where it is 0.909.
    for c in ("clean", "rhuh", "brats_africa"):
        p = rhuh(V / f"{c}_box.jsonl")
        if not p.exists():
            continue
        tp = fp = fn = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            a = json.loads(line).get("arms", {}).get(f"pipeline:{SEG}", {})
            tp += a.get("tp", 0)
            fp += a.get("fp", 0)
            fn += a.get("fn", 0)
        if tp:
            f1 = 2 * tp / (2 * tp + fp + fn)
            term = {"clean": "+0.0059", "rhuh": "+0.0180",
                    "brats_africa": "+0.0914"}[c]
            # as a row, not as a bare number: 0.867 and +0.0059 each appear
            # elsewhere, so checking them separately can never fail. The
            # column between them names the term the row reports.
            want_flat(f"{c} detection F1, in its Table 2 row",
                      f"| {f1:.3f} | detector-stage | {term} |")

    # ------------------------------------------ the detectors themselves
    slug = "run_037_icarb_alpha_050_w30_clsfl"
    wt = json.loads((DRIVE_ROOT / "experiments" / slug / "summary" /
                     "run_summary.json").read_text(encoding="utf-8"))
    want("WT detector F1", f"{wt['test']['f1']:.4f}")
    want("WT detector box IoU", f"{wt['geometry']['iou']:.4f}")
    for sd in (1337, 42, 62):
        p = (DRIVE_ROOT / "experiments_repeated" / f"tc_seed_{sd}" / slug /
             "summary" / "run_summary.json")
        if p.exists():
            d = json.loads(p.read_text(encoding="utf-8"))
            want(f"TC detector F1 seed {sd}", f"{d['test']['f1']:.4f}")

    # the per-seed adapter HD95 figures were quoted only by the seed_52 case
    # study, which was a digression into another project's test split and is
    # no longer reported; the limitation it illustrated remains in §6

    # ------------------------------------------- ET volume error (median)
    for c in COHORTS:
        recs = read(V / f"{c}_oracle_regions.jsonl")
        key = f"oracle:{SEG}"
        rows = [r[key]["regions"] for r in recs.values() if key in r]
        for reg in ("TC", "ET"):
            vals = [x[reg]["rel_vol_diff"] for x in rows
                    if reg in x and x[reg].get("rel_vol_diff") is not None]
            if vals:
                want(f"{c} {reg} volume error", f"{np.median(vals) * 100:+.1f}%")

    # the pipeline scores above the oracle-box arm on a quarter of the
    # patients, which is why the arm is not called a ceiling
    n_over = {}
    for c in COHORTS:
        b = read(V / f"{c}_box.jsonl")
        n_over[c] = sum(1 for p in b
                        if f"oracle:{SEG}" in b[p] and f"pipeline:{SEG}" in b[p]
                        and b[p][f"pipeline:{SEG}"]["vol_dice"]
                            > b[p][f"oracle:{SEG}"]["vol_dice"])
    want("patients where pipeline beats oracle-box", str(sum(n_over.values())))
    for c in COHORTS:
        want(f"{c} pipeline above oracle-box", str(n_over[c]))

    # ------------------------------- the detector-stage term at two seeds
    ws = R / "wt_seed_check.json"
    if ws.exists():
        w = json.loads(ws.read_text(encoding="utf-8"))
        for c, row in w["cohorts"].items():
            want(f"{c} n, seed check", str(row["n"]))
            for sd, v in row["seeds"].items():
                want(f"{c} detector-stage seed {sd}", f"{v['detector_stage']:+.4f}")
            # the spread is the difference between the two seed columns of
            # Table 10, which the reader can take from the table; the prose
            # that quoted it separately went with the compression
            if c == "clean":
                want(f"{c} seed spread", f"{row['spread']:.4f}")
        for sd, o in w["ordering"].items():
            for r in o["external_over_internal"]:
                want(f"external/internal ratio seed {sd}", f"{r:.1f}")
        want("smallest external/internal ratio",
             f"{w['categorical_min_ratio']:.1f}")

    # ------------------------------- what the detector-stage term is made of
    sp = R / "slice_split.json"
    if sp.exists():
        w = json.loads(sp.read_text(encoding="utf-8"))
        for c, row in w["cohorts"].items():
            want(f"{c} missed detection", f"{row['missed_detection']:+.4f}")
            want(f"{c} box geometry", f"{row['box_geometry']:+.4f}")
            want(f"{c} missed positive slices",
                 f"{row['missed_positive_slices']}")
            want(f"{c} positive slices", f"{row['positive_slices']}")
            want(f"{c} tumour voxels on missed slices",
                 f"{row['tumour_voxels_on_missed_slices'] * 100:.1f}%")
            want(f"{c} oracle median on missed slices",
                 f"{row['oracle_median_dice_on_missed']:.3f}")
            if row["detector_stage"] > 0.01:      # the share of a term that
                want(f"{c} missed share",          # small is not worth quoting
                     # 12.5% printed as "12%" matched an unrelated
                     # headroom cell, so the check passed without
                     # verifying the value it names
                     f"{row['missed_share'] * 100:.1f}%")

    # ------------------------------- intervals on the detector-stage term
    for ci, seg in ((R / "detector_stage_ci.json", "sam2.1_l"),
                    (R / "detector_stage_ci_sam1_vit_b.json", "sam1_vit_b")):
        if not ci.exists():
            continue
        w = json.loads(ci.read_text(encoding="utf-8"))
        for k, v in w["within"].items():
            c, sd = k.split("/")
            want_ci(f"{seg} {c} detector-stage CI, seed {sd}", *v["ci"])
            if v.get("bca") and sd == "1337" and seg == "sam2.1_l":
                want_ci(f"{c} detector-stage BCa", *v["bca"])
        for k, v in w["between"].items():
            c, sd = k.split("/")
            want(f"{seg} {c} contrast, seed {sd}", f"{v['difference']:+.4f}")
            want_ci(f"{seg} {c} contrast CI, seed {sd}", *v["ci"])
        # the two external cohorts against each other, which the abstract
        # and Section 3.1 quote and nothing here used to re-derive. The
        # second segmenter's version of it is computed but not quoted, so
        # requiring it here would fail the run over an unclaimed value.
        for k, v in ((w.get("external") or {}) if seg == "sam2.1_l" else {}).items():
            c, sd = k.split("/")
            want(f"{seg} {c} contrast, seed {sd}", f"{v['difference']:+.4f}")
            want_ci(f"{seg} {c} contrast CI, seed {sd}", *v["ci"])
        # the same contrast with the nine collapses dropped, which Section 3.2
        # asserts and used to assert without an interval
        for k, v in ((w.get("no_collapse") or {}) if seg == "sam2.1_l" else {}).items():
            c, sd = k.split("/")
            want(f"{c} contrast without collapses, seed {sd}",
                 f"{v['difference']:+.4f}")
            want_ci(f"{c} contrast CI without collapses, seed {sd}", *v["ci"])
            want(f"{c} term without collapses, seed {sd}", f"{v['term']:+.4f}")

    # ---------------------------------------------------- boundary metrics
    bs_ = R / "boundary_summary.json"
    if bs_.exists():
        w = json.loads(bs_.read_text(encoding="utf-8"))
        for c, row in w["cohorts"].items():
            for arm, v in row["arms"].items():
                want(f"{c} {arm} HD95 median", f"{v['hd95_median']:.2f}")
                want(f"{c} {arm} HD95 mean", f"{v['hd95_mean']:.2f}")
                want(f"{c} {arm} NSD median", f"{v['nsd_median']:.4f}")
            d = row.get("pipeline_minus_oracle_hd95")
            if d:
                want(f"{c} HD95 pipeline minus oracle", f"{d['mean']:+.2f}")
                want(f"{c} HD95 diff CI low", f"{d['ci'][0]:+.2f}")
                want(f"{c} HD95 diff CI high", f"{d['ci'][1]:+.2f}")
        tot = sum(v["pipeline_minus_oracle_hd95"]["pipeline_better"]
                  for v in w["cohorts"].values()
                  if "pipeline_minus_oracle_hd95" in v)
        n_ = sum(v["pipeline_minus_oracle_hd95"]["n"]
                 for v in w["cohorts"].values()
                 if "pipeline_minus_oracle_hd95" in v)
        want("patients where pipeline beats oracle-box on HD95", str(tot))
        want("HD95 comparison n", str(n_))

    # ------------------------------------------ the threshold sweep, all of it
    O = {p: a[f"oracle:{SEG}"]["vol_dice"]
         for p, a in read(V / "brats_africa_box.jsonl").items()
         if f"oracle:{SEG}" in a}
    dr = read(V / "brats_africa_direct.jsonl")
    for conf, fn in (("0.55", "brats_africa_box.jsonl"),
                     ("0.45", "brats_africa_conf045.jsonl"),
                     ("0.35", "brats_africa_conf035.jsonl"),
                     ("0.25", "brats_africa_conf025.jsonl"),
                     ("0.15", "brats_africa_conf015.jsonl")):
        arms = read(V / fn)
        k = f"pipeline:{SEG}"
        ids = sorted(p for p in arms if k in arms[p] and p in O)
        if not ids:
            continue
        Pv = np.array([arms[p][k]["vol_dice"] for p in ids])
        Ov = np.array([O[p] for p in ids])
        want(f"sweep {conf} pipeline", f"{Pv.mean():.4f}")
        want(f"sweep {conf} detector-stage", f"{(Ov - Pv).mean():+.4f}")
        want(f"sweep {conf} collapses", str(int((Pv < 0.5).sum())))
        di = [p for p in ids if "direct:seed_42" in dr.get(p, {})]
        if di:
            Dv = np.array([dr[p]["direct:seed_42"]["regions"]["WT"]["vol_dice"]
                           for p in di])
            Pd = np.array([arms[p][k]["vol_dice"] for p in di])
            want(f"sweep {conf} gap", f"{(Dv - Pd).mean():+.4f}")

    # ------------------------------------- tumour volume as a rival explanation
    cf = R / "confounder_check.json"
    if cf.exists():
        w = json.loads(cf.read_text(encoding="utf-8"))
        want("pooled term-vs-volume Spearman", f"{w['pooled']['spearman']:.3f}")
        for c, v in w["between"].items():
            want(f"{c} median volume ratio", f"{v['median_ratio']:.2f}")

    # ---------------------------------- the gap at three LoRA adapter seeds
    asc = R / "adapter_seed_check.json"
    if asc.exists():
        w = json.loads(asc.read_text(encoding="utf-8"))
        for c, row in w["cohorts"].items():
            want(f"{c} adapter n", str(row["n"]))
            for sd, v in row["seeds"].items():
                want(f"{c} gap, adapter {sd}", f"{v['gap']:+.4f}")
                want(f"{c} gap CI low, adapter {sd}", f"{v['ci'][0]:+.4f}")
                want(f"{c} gap CI high, adapter {sd}", f"{v['ci'][1]:+.4f}")
            want(f"{c} gap spread over adapters", f"{row['gap_spread']:.4f}")
        # S12 says "No ratio between the two external cohorts is reported",
        # so there is nothing to check. These three ran anyway and passed on
        # "1.5 T" and "p = 1.4", certifying a claim the paper withdrew.

    # ---------------------- the in-domain term on the detector's own test split
    dt = V / "dettest_slices.jsonl"
    if dt.exists():
        rows = []
        for line in dt.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "slices" in r:
                    S = r["slices"]
                    g = sum(x["gt"] for x in S)
                    f = lambda i, p: (2 * sum(x[i] for x in S) /
                                      (sum(x[p] for x in S) + g)) if g else 1.0
                    rows.append((f("oracle_i", "oracle_p"), f("pipe_i", "pipe_p")))
        if rows:
            o = np.array([a for a, _ in rows]); p_ = np.array([b for _, b in rows])
            want("detector test split n", str(len(rows)))
            want("detector test split oracle-box", f"{o.mean():.4f}")
            want("detector test split pipeline", f"{p_.mean():.4f}")
            want("detector test split detector-stage", f"{(o - p_).mean():+.4f}")
            rng = np.random.default_rng(1337)
            d = o - p_
            bs = np.sort([d[rng.integers(0, len(d), len(d))].mean()
                          for _ in range(20000)])
            want("detector test split CI low", f"{np.percentile(bs, 2.5):+.4f}")
            want("detector test split CI high", f"{np.percentile(bs, 97.5):+.4f}")

    # ------------------------------- what the collapses are, from the slices
    sl = V / "brats_africa_slices.jsonl"
    if sl.exists():
        rows = {}
        for line in sl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "slices" in r:
                    rows[r["patient"]] = r["slices"]
        b = read(V / "brats_africa_box.jsonl")
        coll = [p for p in b if f"pipeline:{SEG}" in b[p]
                and b[p][f"pipeline:{SEG}"]["vol_dice"] < 0.5]
        maj = 0
        for p in coll:
            pos = [x for x in rows.get(p, []) if x["gt"]]
            if pos and len([x for x in pos if not x["boxes"]]) > len(pos) / 2:
                maj += 1
        want("Africa collapses with no box on most positive slices", str(maj))
        want("Africa collapses", str(len(coll)))

    # ------------------------------- second segmenter, and the clean baseline
    # Two claims a referee found unsupported: the contrast reproducing under an
    # architecturally independent segmenter, and what happens when the
    # in-domain baseline is restricted to the patients the threshold never saw.
    def dterm(c, seg):
        # the corrected RHUH box run carries SAM 2.1-L only; its SAM 1 arms live
        # in the per-region rerun, which was scored with both segmenters
        d = read(V / f"{c}_box.jsonl")
        if not all(f"oracle:{seg}" in v for v in d.values()):
            d = read(V / f"{c}_oracle_regions.jsonl")
        out = []
        for v in d.values():
            o, p = v[f"oracle:{seg}"], v[f"pipeline:{seg}"]
            out.append((o["regions"]["WT"]["vol_dice"] if "regions" in o else o["vol_dice"])
                       - (p["regions"]["WT"]["vol_dice"] if "regions" in p else p["vol_dice"]))
        return np.array(out)
    for seg, lbl in (("sam1_vit_b", "SAM 1"), ("sam2.1_l", "SAM 2.1-L")):
        base = dterm("clean", seg)
        want(f"{lbl} held-out term", f"{base.mean():+.4f}")
        for c, cl in (("rhuh", "RHUH"), ("brats_africa", "Africa")):
            t = dterm(c, seg)
            want(f"{lbl} {cl} term", f"{t.mean():+.4f}")
            # the interval on this contrast is published by
            # detector_stage_ci.py for both segmenters and checked above;
            # resampling it a second time here produced a second answer
            want(f"{lbl} {cl} contrast", f"{t.mean() - base.mean():+.4f}")

    # the three adapter seeds' tumour-core scores on the held-out cohort. Only
    # the first is quoted in Results; the other two appear in Limitations, and
    # a value quoted nowhere else has to be checked where it is used.
    for sd in (42, 52, 62):
        fn = "clean_direct.jsonl" if sd == 42 else f"clean_direct_seed{sd}.jsonl"
        rr = [json.loads(x) for x in (V / fn).read_text(
            encoding="utf-8").splitlines() if x.strip()]
        rr = [x for x in rr if "error" not in x]
        v = np.mean([x["arms"][f"direct:seed_{sd}"]["regions"]["TC"]["vol_dice"]
                     for x in rr])
        if sd == 42:   # the other two seeds were quoted only by that case
            want(f"held-out TC adapter seed {sd}", f"{v:.4f}")

    # ------------------------------ the oracle arm's prompt protocol (4.11)
    # The oracle arm was prompted with one box per region and the pipeline arm
    # with one per detection, so on a multi-component slice the two arms were
    # not under the same protocol. These are the per-component reruns.
    oc = R / "oracle_components.json"
    if oc.exists():
        ocj = json.loads(oc.read_text(encoding="utf-8"))
        for c, e in ocj["cohorts"].items():
            for reg, r in e["regions"].items():
                want(f"{c} {reg} oracle one box", f"{r['one_box']:.4f}")
                want(f"{c} {reg} oracle per component", f"{r['per_component']:.4f}")
                want(f"{c} {reg} per-component gain", f"{r['delta']:+.4f}")
                want(f"{c} {reg} per-component low", f"{r['ci'][0]:+.4f}")
                want(f"{c} {reg} per-component high", f"{r['ci'][1]:+.4f}")
                want(f"{c} {reg} prompts per slice", f"{r['prompts_per_slice']:.2f}")
                if "vol_err_median_per_component" in r:
                    want(f"{c} {reg} per-component volume error",
                         f"{r['vol_err_median_per_component']:+.1f}")
            d = e["detector_stage"]
            for lbl, v in (("one box", d["one_box"]),
                           ("per component", d["per_component"])):
                want(f"{c} detector-stage {lbl}", f"{v[0]:+.4f}")
                if lbl == "per component":   # the one-box interval belongs to
                    want_ci(f"{c} {lbl} CI", v[1], v[2])   # Table 4
        for c, k in ocj.get("contrast_vs_held_out", {}).items():
            for lbl, v in (("one box", k["one_box"]),
                           ("per component", k["per_component"])):
                want(f"{c} contrast {lbl}", f"{v[0]:+.4f}")
                if lbl == "per component":
                    want_ci(f"{c} contrast {lbl} CI", v[1], v[2])

    # ---------------- the detector-free arm under the corrected brain mask
    # Correcting the fault on the box arms only would leave the three arms on
    # RHUH-GBM seeing different volumes, so the arm was run both ways.
    bg = R / "rhuh_bgfix.json"
    if bg.exists():
        bj = json.loads(bg.read_text(encoding="utf-8"))
        for sd, r in bj.get("seeds", {}).items():
            want(f"RHUH bgfix seed {sd} published", f"{r['published']:.4f}")
            want(f"RHUH bgfix seed {sd} corrected", f"{r['corrected']:.4f}")
            for k in ("gap_published", "gap_corrected",
                      "residual_published", "residual_corrected"):
                want(f"RHUH bgfix seed {sd} {k}", f"{r[k][0]:+.4f}")
        seedj = json.loads((R / "adapter_seed_check.json").read_text(
            encoding="utf-8"))
        if bj.get("seeds"):
            g = [r["gap_published"][0] for r in bj["seeds"].values()]
            c = [r["gap_corrected"][0] for r in bj["seeds"].values()]
            want("RHUH bgfix gap spread published", f"{max(g) - min(g):.4f}")
            want("RHUH bgfix gap spread corrected", f"{max(c) - min(c):.4f}")
        want("RHUH detector-free, published", f"{bj['direct_published']:.4f}")
        want("RHUH detector-free, corrected", f"{bj['direct_corrected']:.4f}")
        for lbl, v in (("bgfix rise", bj["corrected_minus_published"]),
                       ("bgfix gap published", bj["total_gap"]["published"]),
                       ("bgfix gap corrected", bj["total_gap"]["corrected"]),
                       ("bgfix residual published", bj["residual"]["published"]),
                       ("bgfix residual corrected", bj["residual"]["corrected"])):
            want(lbl, f"{v[0]:+.4f}")
            # The corrected gap is also bootstrapped by adapter_seed_check.py,
            # which is what the per-seed table quotes. Two scripts checking the
            # same interval against the text is how the manuscript came to
            # print two of them; the per-seed file is canonical here and the
            # two are asserted to agree instead.
            if lbl == "bgfix gap corrected":
                ref = seedj["cohorts"]["rhuh"]["seeds"]["42"]["ci"]
                assert abs(v[1] - ref[0]) < 5e-4 and abs(v[2] - ref[1]) < 5e-4, (
                    f"the two bootstraps of the corrected RHUH gap disagree: "
                    f"{v[1:]} against {ref}")
                continue
            want(lbl + " low", f"{v[1]:+.4f}")
            want(lbl + " high", f"{v[2]:+.4f}")

    split = {r["patient_id"]: r["split"] for r in csv.DictReader(
        io.open(DRIVE_ROOT / "data" / "processed" / "metadata" /
                "patient_split.csv", encoding="utf-8"))}
    cl24 = read(V / f"clean_box.jsonl")
    clean10 = np.array([v[f"oracle:{SEG}"]["vol_dice"] - v[f"pipeline:{SEG}"]["vol_dice"]
                        for p, v in cl24.items() if split.get(p) == "test"])
    want("uncontaminated in-domain term", f"{clean10.mean():+.4f}")
    for c, cl in (("rhuh", "RHUH"), ("brats_africa", "Africa")):
        t = dterm(c, SEG)
        bs = np.sort([np.random.default_rng([1337, i]).choice(t, len(t)).mean()
                      - np.random.default_rng([7331, i]).choice(clean10, len(clean10)).mean()
                      for i in range(20000)])
        want(f"{cl} vs clean baseline", f"{t.mean() - clean10.mean():+.4f}")
        want(f"{cl} vs clean low", f"{np.percentile(bs, 2.5):+.4f}")
        want(f"{cl} vs clean high", f"{np.percentile(bs, 97.5):+.4f}")
        want(f"{cl} clean fold", f"{t.mean() / clean10.mean():.1f}")

    # ------------------------------- prompt protocol sensitivity (4.9)
    # The detector-free arm is prompted by region name, and two of the three
    # names are out of the model's training vocabulary. Its source project
    # recommends an in-vocabulary composed protocol that was never run; this is
    # that run.
    ps = V / "prompt_sens_clean.jsonl"
    if ps.exists():
        recs = [json.loads(x) for x in ps.read_text(encoding="utf-8").splitlines()
                if x.strip()]
        g = np.random.default_rng(1337)
        for reg in ("WT", "TC", "ET"):
            pair = [(r["arms"]["deployed"][reg]["vol_dice"],
                     r["arms"]["composed"][reg]["vol_dice"])
                    for r in recs
                    if reg in r["arms"]["deployed"]
                    and r["arms"]["deployed"][reg]["gt_voxels"]]
            if not pair:
                continue
            a = np.array([x[0] for x in pair])
            b = np.array([x[1] for x in pair])
            want(f"prompt {reg} deployed", f"{a.mean():.4f}")
            want(f"prompt {reg} composed", f"{b.mean():.4f}")
            if reg == "ET":
                assert np.array_equal(a, b), "ET should be identical across protocols"
                continue
            d = a - b
            want(f"prompt {reg} difference", f"{d.mean():+.4f}")
            bs = np.sort([g.choice(d, len(d)).mean() for _ in range(20000)])
            want(f"prompt {reg} CI low", f"{np.percentile(bs, 2.5):+.4f}")
            want(f"prompt {reg} CI high", f"{np.percentile(bs, 97.5):+.4f}")
            want(f"prompt {reg} deployed ahead", str(int((d > 0).sum())))
    else:
        print("  (prompt sensitivity run absent)")

    # ------------------------------- both sides of the in-domain cohort
    # The paper discloses that 14 of the 24 sit in the detector's validation
    # split; a referee pointed out the symmetric fact on the adapter side was
    # missing. Both counts are checked so neither can drift.
    lo_p = MEDSAM3_DIR / "training/data/splits/brats2020_fixed_split.json"
    if lo_p.exists():
        lo = json.loads(lo_p.read_text(encoding="utf-8"))
        clean = {x.strip() for x in
                 (R / "clean_cohort.txt").read_text(encoding="utf-8").splitlines()
                 if x.strip()}
        det = {r["patient_id"]: r["split"] for r in csv.DictReader(
            io.open(DRIVE_ROOT / "data" / "processed" / "metadata" /
                    "patient_split.csv", encoding="utf-8"))}
        want("held-out in detector validation",
             str(sum(1 for p in clean if det.get(p) == "val")))
        want("held-out in adapter validation",
             str(len(clean & set(lo["validation"]))))

    # ------------------------------- the fourth arm
    try:
        FOURTH_CI = json.loads(io.open(
            DRIVE_ROOT / "reports" / "fourth_arm_summary.json",
            encoding="utf-8").read())
    except (OSError, ValueError):
        FOURTH_CI = {}
    # The reference box handed to the detector-free model, which splits the
    # residual into a prompt-type and an architecture component. RHUH-GBM uses
    # the corrected-mask files, so that all three arms there see the same
    # volumes; the box arms on that cohort live in the oracle-regions file.
    FOURTH = {
        "clean": ("clean_fourth_arm.jsonl", "clean_box.jsonl",
                  "clean_direct.jsonl", "direct:seed_42"),
        "rhuh": ("rhuh_fourth_arm.jsonl", "rhuh_oracle_regions_normfix.jsonl",
                 "rhuh_direct_seed42_bgfix.jsonl", "direct:seed_42"),
        "brats_africa": ("brats_africa_fourth_arm.jsonl",
                         "brats_africa_box.jsonl",
                         "brats_africa_direct.jsonl", "direct:seed_42"),
    }
    for cohort, (fa_n, box_n, dir_n, dkey) in FOURTH.items():
        fa_p = V / fa_n
        if not fa_p.exists():
            continue
        fa, bx4, dr4 = read(fa_p), read(V / box_n), read(V / dir_n)
        kk = sorted(set(fa) & set(bx4) & set(dr4))
        # a run still in flight would otherwise be summarised as though it had
        # finished, and the partial mean quoted as the cohort's
        expected = {"clean": 24, "rhuh": 39, "brats_africa": 95}[cohort]
        if len(kk) < expected:
            print(f"  ({cohort} fourth arm: {len(kk)} of {expected} patients; "
                  f"not summarised)")
            continue

        def dice(rec, key):
            a = rec[key]
            return a.get("vol_dice") or a["regions"]["WT"]["vol_dice"]

        o4 = np.array([dice(bx4[k], f"oracle:{SEG}") for k in kk])
        f4 = np.array([dice(dr4[k], dkey) for k in kk])
        q4 = np.array([dice(fa[k], "oraclebox_medsam3:seed_42") for k in kk])
        tag = "" if cohort == "clean" else f" {cohort}"
        want(f"fourth arm{tag} Dice", f"{q4.mean():.4f}")
        for lab, d in (("prompt type", q4 - f4), ("architecture", q4 - o4),
                       ("residual", f4 - o4)):
            want(f"fourth arm{tag} {lab}", f"{d.mean():+.4f}")
            # the residual is Table 3's quantity, bootstrapped by the script
            # that builds it; a second interval for it here would be the
            # duplicate the manuscript was cleared of. The identity below
            # already checks that this arm reproduces it.
            if lab == "residual":
                continue
            # The interval comes from compile_fourth_arm.py, which is what the
            # paper quotes and what the archive carries. Drawing a second
            # bootstrap here would compare two correct answers that differ in
            # the fourth decimal, which is a sequence difference and not a
            # discrepancy.
            term = {"prompt type": "prompt", "architecture": "model"}[lab]
            rec = (FOURTH_CI.get("within", {}) or {}).get(f"{cohort}/{term}")
            if rec:
                want_ci(f"fourth arm{tag} {lab} CI", *rec["ci"])
            else:
                skipped.append(f"fourth arm{tag} {lab} CI: "
                               "run compile_fourth_arm.py")
        # Table 2's residual intervals and Table 3's contrast column come from
        # the same file and were the twelve values with no released check
        for term in ("residual", "prompt", "model"):
            rec = (FOURTH_CI.get("within", {}) or {}).get(f"{cohort}/{term}")
            if rec and term == "residual":
                want_ci(f"fourth arm{tag} residual CI", *rec["ci"])
            btw = (FOURTH_CI.get("between", {}) or {}).get(f"{cohort}-clean/{term}")
            if btw:
                want(f"fourth arm{tag} {term} contrast",
                     f"{btw['difference']:+.4f}")
                want_ci(f"fourth arm{tag} {term} contrast CI", *btw["ci"])
        assert abs(((q4 - o4) - (q4 - f4)).mean() - (f4 - o4).mean()) < 1e-9,             f"the fourth-arm identity does not close on {cohort}"

    # ------------------------------- how widely each reference is drawn
    # Section 4.1 names annotation extent as a rival explanation the design
    # cannot separate from image shift, and quotes two ratios for it.
    ext = R / "reference_extent.json"
    if ext.exists():
        w = json.loads(ext.read_text(encoding="utf-8"))
        for c_, row in (w.get("cohorts") or {}).items():
            for field, label in (("volume_ratio_to_held_out", "volume ratio"),
                                 ("area_ratio_to_held_out", "area ratio")):
                if field in row:
                    want(f"{c_} reference {label}", f"{row[field]:.2f}")
    else:
        skipped.append("reference extent: run reference_extent.py")

    # ------------------------------- the decomposition on tumour core
    # Table S14 is the newest claim in the paper and was outside the gate.
    # tc_detector_summary.json computes the same nine means independently as
    # `localisation_gap`, so the two are also checked against each other.
    tcci = R / "tc_detector_stage_ci.json"
    if tcci.exists():
        w = json.loads(tcci.read_text(encoding="utf-8"))
        tcs, SEG_TC = {}, "sam2.1_l"
        p_ = R / "tc_detector_summary.json"
        if p_.exists():
            raw = json.loads(p_.read_text(encoding="utf-8"))
            # cohorts[c]["seeds"][seed][segmenter]["localisation_gap"]; the
            # first version of this walked cohorts[c] directly, found the
            # keys "label", "seeds" and "detector_free_seeds" instead of
            # seeds, and so compared nothing at all
            for c_, rows in (raw.get("cohorts") or {}).items():
                for sd_, row in ((rows or {}).get("seeds") or {}).items():
                    g = ((row or {}).get(SEG_TC) or {}).get("localisation_gap")
                    if g is not None:
                        tcs[f"{c_}/{sd_}"] = g
            if not tcs:
                raise SystemExit(
                    "tc_detector_summary.json carries no localisation_gap "
                    "under cohorts[c]['seeds'][seed]['" + SEG_TC + "']; the "
                    "cross-check would pass without comparing anything")
        for k, v in w["within"].items():
            want(f"tumour-core term {k}", f"{v['mean']:+.4f}")
            want_ci(f"tumour-core term CI {k}", *v["ci"])
            other = tcs.get(k)
            if other is not None:
                assert abs(other - v["mean"]) < 5e-4, (
                    f"the two tumour-core computations disagree on {k}: "
                    f"{other} against {v['mean']}")
        for k, v in w["between"].items():
            want(f"tumour-core contrast {k}", f"{v['difference']:+.4f}")
            want_ci(f"tumour-core contrast CI {k}", *v["ci"])
    else:
        skipped.append("tumour-core decomposition: run tc_detector_stage_ci.py")

    # ------------------------------- the slice rule on RHUH-GBM
    # The intensity fault and the slice-selection fault are the same `> 0`
    # assumption in two places. The paper reports how many slices the corrected
    # slice rule adds, over the patients it actually changes.
    fx = V / "rhuh_box_maskfix.jsonl"
    if fx.exists():
        def by_patient(p):
            out = {}
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if "error" not in r:
                        out[r["patient"]] = r
            return out
        o = by_patient(V / "rhuh_box.jsonl")
        n = by_patient(fx)
        ks = [k for k in sorted(set(o) & set(n))
              if n[k]["n_usable"] != o[k]["n_usable"]]
        want("slice rule patients changed", str(len(ks)))
        want("slice rule slices added",
             f"{np.mean([n[k]['n_usable'] - o[k]['n_usable'] for k in ks]):.2f}")

    # ------------------------------- selection bias, measured (section 4.4)
    # The oracle arm never runs the detector, so it is identical across the 37
    # ablation runs and cancels in any between-run difference of detector-stage
    # terms. That makes the volumetric selection bias measurable from the
    # per-run pipeline arm alone, which an earlier draft called unmeasurable.
    per_run = sorted(DRIVE_ROOT.glob(
        "experiments/run_*/ground_truth_evaluation/per_patient_metrics.csv"))
    if len(per_run) == 37:
        tbl = {}
        for p in per_run:
            rid = int(p.parts[-3].split("_")[1])
            tbl[rid] = {r["patient_id"]: float(r["vol_dice"])
                        for r in csv.DictReader(io.open(p, encoding="utf-8"))}
        pats = sorted(tbl[37])
        assert len({tuple(sorted(v)) for v in tbl.values()}) == 1, "patient sets differ"
        M = np.array([[tbl[r][q] for q in pats] for r in sorted(tbl)])
        i = sorted(tbl).index(37)
        d = M[i] - np.delete(M, i, axis=0).mean(axis=0)
        want("run 37 volumetric advantage", f"{d.mean():+.4f}")
        g = np.random.default_rng(1337)
        bs = np.sort([g.choice(d, len(d)).mean() for _ in range(20000)])
        want("selection bias CI low", f"{np.percentile(bs, 2.5):+.4f}")
        want("selection bias CI high", f"{np.percentile(bs, 97.5):+.4f}")
        want("run 37 patients ahead", str(int((d > 0).sum())))
        want("run 37 volumetric rank",
             str(list(np.argsort(-M.mean(axis=1))).index(i) + 1))
    else:
        print(f"  ({len(per_run)} of 37 per-run metric files; selection bias unchecked)")

    # ------------------------------- weights (Table S3)
    # The ratios were computed from the table's own printed figures while the
    # column mixed MiB and decimal MB, so all three were wrong. They are now
    # derived from the files.
    MiB = 1024 ** 2
    files = {
        "detector": DRIVE_ROOT / "experiments" /
                    "run_037_icarb_alpha_050_w30_clsfl" / "checkpoints" / "best.pt",
        "sam1": DRIVE_ROOT / "demo" / "rt-detr-sam-pipeline" / "models" /
                "sam_vit_b_01ec64.pth",
        "sam2": DRIVE_ROOT / "shared_artifacts" / "sam2" / "sam2.1_l.pt",
        "lora": DRIVE_ROOT / "models" / "lora" / "seed_42" / "ckpt_epoch3_final.pt",
        "medsam": MEDSAM3_DIR / ("models/"
                       "medical_sam3/checkpoint_2D.pt"),
    }
    # measured by weight_sizes.py, which needs torch; this script does not
    ws = R / "weight_sizes.json"
    if ws.exists():
        b = {k: v["weights_bytes"]
             for k, v in json.loads(ws.read_text(encoding="utf-8")).items()}
        want("detector weights MiB", f"{b['detector'] / MiB:.0f}")
        want("SAM 1 weights MiB", f"{b['sam1'] / MiB:.0f}")
        want("SAM 2.1-L weights MiB", f"{b['sam2'] / MiB:.0f}")
        want("MedSAM3 weights MiB", f"{b['medsam'] / MiB:,.0f}".replace(",", " "))
        want("detector-free total MiB",
             f"{(b['medsam'] + b['lora']) / MiB:,.0f}".replace(",", " "))
        free = b["medsam"] + b["lora"]
        want("weights ratio vs SAM 2.1-L",
             f"{free / (b['detector'] + b['sam2']):.1f}")
        want("weights ratio vs SAM 1",
             f"{free / (b['detector'] + b['sam1']):.1f}")
        want("MedSAM3 over SAM 2.1-L", f"{b['medsam'] / b['sam2']:.0f}")
    else:
        print("  (weight_sizes.json absent; weight ratios not re-derived)")

    # ------------------------------- cost at more than one region
    # Table 13's headline ratio compared a whole-tumour box pipeline against a
    # detector-free arm that returns three regions in one pass. These are the
    # like-for-like figures the text now gives instead.
    def mean_elapsed(fn, key):
        v = []
        for c in ("clean", "rhuh", "brats_africa"):
            for line in rhuh(V / f"{c}_{fn}.jsonl").read_text(
                    encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)
                    if "error" in r or key not in r["arms"]:
                        continue
                    v.append(r["arms"][key]["elapsed_s"])
        return sum(v) / len(v)
    tc_pipe = mean_elapsed("tc_pipeline", f"pipeline:{SEG}")
    wt_pipe = mean_elapsed("box", f"pipeline:{SEG}")
    free = mean_elapsed("direct", "direct:seed_42")
    want("TC pipeline seconds", f"{tc_pipe:.1f}")
    want("two-region box seconds", f"{wt_pipe + tc_pipe:.1f}")
    want("two-region ratio", f"{free / (wt_pipe + tc_pipe):.1f}")
    # the oracle arm gives the per-region increment directly
    per = {}
    for reg in ("WT", "TC", "ET"):
        v = []
        for c in ("clean", "rhuh", "brats_africa"):
            for line in rhuh(V / f"{c}_oracle_regions.jsonl").read_text(
                    encoding="utf-8").splitlines():
                if line.strip():
                    r = json.loads(line)["arms"][f"oracle:{SEG}"]["regions"]
                    v.append(r[reg]["elapsed_s"])
        per[reg] = sum(v) / len(v)
    want("oracle WT seconds", f"{per['WT']:.1f}")
    want("oracle two-region seconds", f"{per['WT'] + per['TC']:.1f}")
    want("oracle three-region seconds",
         f"{per['WT'] + per['TC'] + per['ET']:.1f}")
    factor = (per["WT"] + per["TC"] + per["ET"]) / per["WT"]
    want("three-region factor", f"{factor:.2f}")
    want("three-region box seconds", f"{wt_pipe * factor:.1f}")

    # ------------------------------- what the ET prompt actually returns
    # The paper used to argue the enhancing-tumour cap from a volume error
    # against ET's own reference. The sharper statement is that the ET-prompted
    # mask is the TC-prompted mask, and it holds in all three cohorts.
    for c, lab in (("clean", "held-out"), ("rhuh", "RHUH"),
                   ("brats_africa", "Africa")):
        recs = [json.loads(x) for x in
                rhuh(V / f"{c}_oracle_regions.jsonl").read_text(
                    encoding="utf-8").splitlines() if x.strip()]
        vs_tc_gt, vs_tc_pred = [], []
        for r in recs:
            reg = r["arms"][f"oracle:{SEG}"]["regions"]
            et, tc = reg.get("ET"), reg.get("TC")
            if not (et and tc and et.get("gt_voxels")):
                continue
            vs_tc_gt.append(abs(et["pred_voxels"] / tc["gt_voxels"] - 1))
            if tc["pred_voxels"]:
                vs_tc_pred.append(et["pred_voxels"] / tc["pred_voxels"])
        want(f"{lab} ET vs TC reference", f"{np.median(vs_tc_gt) * 100:.1f}")
        want(f"{lab} ET over TC prediction", f"{np.median(vs_tc_pred):.3f}")

    # ------------------------------- ET and TC boxes coincide (section 4.9)
    # Measured on the reference segmentations by scripts/et_tc_box_identity.py.
    # The paper used to argue this from equality of predicted mask volume, a
    # per-patient criterion strict enough to miss the mechanism entirely in the
    # held-out cohort.
    bx = R / "et_tc_box_identity.json"
    if bx.exists():
        for lab, row in json.loads(bx.read_text(encoding="utf-8")).items():
            want(f"{lab} slices with both", f"{row['slices']:,}".replace(",", " "))
            want(f"{lab} box-equal pct", f"{row['pct']:.1f}")
            want(f"{lab} mean box IoU", f"{row['mean_box_iou']:.3f}")
    else:
        print("  (run scripts/et_tc_box_identity.py; box-identity figures unchecked)")

    # ------------------------------- claims that live outside this repository
    # The adapter was trained in a separate project. Two numbers in the paper
    # come from it: the leakage count of section 4.3 and the adapter HD95
    # figures. Both are checked here when that project is present, and skipped
    # with a note when it is not, rather than being trusted silently.
    MEDSAM = MEDSAM3_DIR
    lora_split = MEDSAM / "training" / "data" / "splits" / "brats2020_fixed_split.json"
    if lora_split.exists():
        lo = json.loads(lora_split.read_text(encoding="utf-8"))
        det = {r["patient_id"]: r["split"] for r in csv.DictReader(
            io.open(DRIVE_ROOT / "data" / "processed" / "metadata" /
                    "patient_split.csv", encoding="utf-8"))}
        dt = {p for p, v in det.items() if v == "test"}
        for k, lbl in (("train", "training"), ("validation", "validation"),
                       ("test", "test")):
            want(f"detector test in adapter {lbl}", str(len(dt & set(lo[k]))))
        assert lo["excluded"] == ["BraTS20_Training_355"], lo["excluded"]
    else:
        print("  (adapter project not mounted; leakage count not re-derived)")
        skipped.append("leakage count")

    hd = MEDSAM / "research_evidence" / "internal_test_one_time" / "patient_metrics.jsonl"
    if hd.exists():
        recs = [json.loads(x) for x in hd.read_text(encoding="utf-8").splitlines()
                if x.strip()]
        sel = [r for r in recs if r.get("condition") == "clean"
               and r.get("method") == "no_tta" and r.get("split") == "test"]
        for sd in (42, 52, 62):
            g = [r for r in sel if r["seed"] == sd]

    else:
        print("  (adapter evidence not mounted; HD95 figures not re-derived)")
        skipped.append("adapter HD95 and Dice")

    # the threshold section quotes the validation optimum, not the deployed value
    tv = json.loads((R / "threshold_on_val.json").read_text(encoding="utf-8"))
    want("validation optimum threshold", str(tv["honest_operating_point"]["threshold_chosen_on_val"]))
    want("optimism from tuning on test",
         f"{tv['honest_operating_point']['optimism_from_tuning_on_test']:.4f}")

    # ------------------------------- sub-region claims a referee checked
    # Three claims here were stated more strongly than the data supported: the
    # ET mechanism was observed in two cohorts of three, the tumour-core
    # headroom is negative on BraTS-Africa, and the two detector F1 figures
    # come from test splits of different size.
    for c, lab in (("clean", "held-out"), ("rhuh", "RHUH"),
                   ("brats_africa", "Africa")):
        recs = [json.loads(x) for x in
                rhuh(V / f"{c}_oracle_regions.jsonl").read_text(
                    encoding="utf-8").splitlines() if x.strip()]
        same = tot = 0
        for r in recs:
            reg = r["arms"][f"oracle:{SEG}"]["regions"]
            et, tc = reg.get("ET"), reg.get("TC")
            if not (et and tc and et.get("gt_voxels")):
                continue
            tot += 1
            same += et.get("pred_voxels") == tc.get("pred_voxels")
        want(f"{lab} ET==TC volume count", str(same))
        want(f"{lab} ET-bearing patients", str(tot))

    tc_n = json.loads((DRIVE_ROOT / "experiments_repeated" / "tc_seed_1337" /
                       "run_037_icarb_alpha_050_w30_clsfl" / "summary" /
                       "run_summary.json").read_text(encoding="utf-8")
                      )["test"]["n_images"]
    # the draft writes thousands with a plain space, not a comma
    want("TC detector test slices", f"{tc_n:,}".replace(",", " "))

    # ------------------------------- size as a rival explanation
    # The pooled Spearman mixes within-cohort and between-cohort variation and
    # a referee was right that the between-cohort part is the one the rival
    # explanation is about. The per-cohort figures are what the text now quotes.
    for c, lab in (("clean", "held-out"), ("rhuh", "RHUH"),
                   ("brats_africa", "Africa")):
        recs = [json.loads(x) for x in rhuh(V / f"{c}_box.jsonl").read_text(
            encoding="utf-8").splitlines() if x.strip()]
        vol = np.array([r["gt_voxels"] for r in recs], float)
        trm = np.array([r["arms"][f"oracle:{SEG}"]["vol_dice"]
                        - r["arms"][f"pipeline:{SEG}"]["vol_dice"]
                        for r in recs])
        rho, pv = stats.spearmanr(vol, trm)
        want(f"{lab} volume-term rho", f"{rho:+.3f}")
        want(f"{lab} volume-term p", f"{pv:.3f}")

    # ------------------------------- where the term grows (Table S6)
    # The detector-stage term is a difference of two arms and either can move
    # it. A referee showed that on RHUH-GBM most of the growth is the oracle
    # arm rising, not the pipeline falling, so both columns are checked here.
    def arm_means(c, key):
        return np.array([v[key]["vol_dice"]
                         for v in read(V / f"{c}_box.jsonl").values()])
    base_o, base_p = arm_means("clean", f"oracle:{SEG}"),         arm_means("clean", f"pipeline:{SEG}")
    for c in ("rhuh", "brats_africa"):
        o, p = arm_means(c, f"oracle:{SEG}"), arm_means(c, f"pipeline:{SEG}")
        want(f"{c} change in term",
             f"{(o - p).mean() - (base_o - base_p).mean():+.4f}")
        for lab, a, b in (("oracle shift", o, base_o),
                          ("pipeline shift", p, base_p)):
            want(f"{c} {lab}", f"{a.mean() - b.mean():+.4f}")
            g = np.random.default_rng(1337)
            bs = np.sort([g.choice(a, len(a)).mean() - g.choice(b, len(b)).mean()
                          for _ in range(20000)])
            want(f"{c} {lab} CI low", f"{np.percentile(bs, 2.5):+.4f}")
            want(f"{c} {lab} CI high", f"{np.percentile(bs, 97.5):+.4f}")

    # ------------------------------- inference cost (Table S3)
    # This table was quoted from an unrecorded benchmark session and none of
    # its three rows matched the timings stored with the results. It is now
    # derived from those timings, pooled over all three cohorts, and checked.
    cost = {}
    # the deployable pipeline, not the oracle arm: the oracle segments only the
    # slices that carry tumour, so it is faster than anything deployable and an
    # earlier draft of this table quoted it by mistake
    for arm, key, fn in (("sam1", "pipeline:sam1_vit_b", "box"),
                         ("sam2", "pipeline:sam2.1_l", "box"),
                         ("direct", "direct:seed_42", "direct")):
        t, n = [], []
        for c in ("clean", "rhuh", "brats_africa"):
            # read() keeps only the arms; the slice count lives on the record
            for line in rhuh(V / f"{c}_{fn}.jsonl").read_text(
                    encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                # SAM 1 was not re-run on RHUH-GBM under the corrected
                # normalisation, so its row covers two cohorts. The count is
                # checked against the caption rather than left implicit.
                if "error" in r or key not in r["arms"]:
                    continue
                t.append(r["arms"][key]["elapsed_s"])
                n.append(r["n_usable"])
        want(f"Table 15 {arm} patients", str(len(t)))
        cost[arm] = (sum(t) / len(t),
                     sum(a / b for a, b in zip(t, n)) / len(t) * 1000)
        want(f"Table 12 {arm} seconds", f"{cost[arm][0]:.1f}")
        want(f"Table 12 {arm} ms/slice", f"{cost[arm][1]:.0f}")
    want("cost ratio vs SAM 2.1-L", f"{cost['direct'][0] / cost['sam2'][0]:.2f}")
    # SAM 1 covers two cohorts, so the ratio is taken on those patients only
    d1 = []
    for c in ("clean", "brats_africa"):
        for line in rhuh(V / f"{c}_direct.jsonl").read_text(
                encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                if "error" not in r:
                    d1.append(r["arms"]["direct:seed_42"]["elapsed_s"])
    want("detector-free on the SAM 1 patients", f"{sum(d1) / len(d1):.1f}")
    want("cost ratio vs SAM 1", f"{sum(d1) / len(d1) / cost['sam1'][0]:.1f}")

    # ------------------------------- the case excluded before the split
    # The paper names a specific case and a specific reason. Both come from the
    # preprocessing metadata, not from memory: an earlier draft guessed at the
    # reason and the guess had to be withdrawn.
    meta = DRIVE_ROOT / "data" / "processed" / "metadata"
    split = [r for r in io.open(meta / "patient_split.csv",
                                encoding="utf-8").read().splitlines()[1:] if r]
    want("cases in the split", str(len(split)))
    ids = {r.split(",")[0] for r in split}
    missing = sorted({f"BraTS20_Training_{i:03d}" for i in range(1, 370)} - ids)
    if len(missing) == 1:
        want("the excluded case", missing[0])
    excl = io.open(meta / "excluded_patients.csv", encoding="utf-8").read()
    # the recorded reason must be the missing segmentation, not something else
    row = [l for l in excl.splitlines()[1:] if l.startswith(missing[0])]
    assert row and "'seg'" in row[0], f"exclusion reason changed: {row}"

    # ------------------------------- the ten cases drawn in Figure S1
    box = read(V / "brats_africa_box.jsonl")
    for pid in ("00129", "00180", "00215", "00046"):
        k = f"BraTS_SSA_{pid}_000"
        want(f"{pid} oracle-box", f"{box[k][f'oracle:{SEG}']['vol_dice']:.3f}")
        want(f"{pid} pipeline", f"{box[k][f'pipeline:{SEG}']['vol_dice']:.3f}")

    bad = [(l, v) for l, v, ok in checks if not ok]
    if not have_doc:
        print(f"{target.name} is not in this checkout, so values are recomputed "
              f"and listed rather than matched against it.\n")
        w = max(len(l) for l, _v, _o in checks)
        for label, value, _ok in checks:
            print(f"  {label:<{w}}  {value}")
        print(f"\n{len(checks)} values recomputed. To check them automatically, "
              f"save the article text as {target.name} at the repository root.")
        return 0
    if abridged:
        print(f"{target.name}: abridged, so the forward direction is reported "
              f"but not enforced -- {len(checks) - len(bad)}/{len(checks)} "
              f"recomputed values appear")
    else:
        print(f"{target.name}: {len(checks) - len(bad)}/{len(checks)} "
              f"values found verbatim")

    # ---------------------------------------------------- the other direction
    # The check above asks whether a recomputed value appears in the text. It
    # never asks whether a value in the text can be recomputed, so a figure that
    # a rerun has superseded survives silently -- which is exactly how a batch of
    # RHUH-GBM numbers outlived their own correction. This lists every decimal
    # in the body that no check produced, so a human can see what is unsourced.
    # the Turkish version heads its reference list differently, and a DOI is
    # not a measurement
    # Splitting the joined text on the first reference heading threw the
    # supplement away together with the manuscript's bibliography, so this
    # direction read half the corpus and called the other half clean. Each
    # document is trimmed on its own and the two are rejoined.
    body = prose
    for _pat, _rep in ((chr(8722), "-"),):
        body = body.replace(_pat, _rep)
    quoted = set()
    for tok in re.findall(r"-?\d+[.]\d+", body):
        quoted.add(tok)
    produced = set()
    for _l, v, _ok in checks:
        produced.add(str(v).replace(chr(8722), "-").lstrip("+"))
        produced.add(str(v).replace(chr(8722), "-"))
    # section numbers, years and table indices are not measurements
    ignore = re.compile(r"^-?(?:[0-9]|1[0-9]|20[0-9][0-9])[.][0-9]{1,2}$")
    orphan = sorted(t for t in quoted
                    if t.lstrip("+-") not in {p.lstrip("+-") for p in produced}
                    and not ignore.match(t)
                    and len(t.split(".")[1]) >= 3)
    print(f"\n{len(orphan)} decimals in the body match no recomputed value")
    if orphan:
        print("  (not all are errors -- literals, thresholds and cited figures "
              "belong here too; scan for ones that should have moved)")
        for i in range(0, len(orphan), 10):
            print("   " + "  ".join(orphan[i:i + 10]))

    # ------------------------------------------------ the restating sections
    # A flat list of orphans is unreadable at this length, which is how a batch
    # of superseded RHUH-GBM figures sat inside it unnoticed. The abstract, the
    # introduction, the discussion and the limitations state no measurement of
    # their own: every number in them restates one computed in Methods or
    # Results. A value there matching nothing in those sections, at its own
    # precision, is stale until shown otherwise -- and that is a short list.
    def _sections(txt, wanted):
        """Lines under a top-level section, subsections included.

        A heading is `## 4. Results` or `### 4.6 Seeds`, so the section a line
        belongs to is the leading integer of the nearest heading -- matching on
        the string "4. " silently drops every subsection, and with it most of
        the Results.
        """
        out, cur, head = [], None, ""
        for ln, line in enumerate(txt.split("\n"), 1):
            m = re.match(r"^#{1,3} +(?:(\d+)[.]?)?\s*(.*)", line)
            if m and line.lstrip().startswith("#"):
                head = (m.group(1) or "") + " " + m.group(2).strip()
                cur = m.group(1) or m.group(2).strip().split()[0:1]
                cur = m.group(1) if m.group(1) else m.group(2).strip()
            elif cur is not None and cur in wanted:
                out.append((ln, head.strip(), line))
        return out

    if abridged:
        # nothing in it is a primary measurement, so every decimal is restated
        restating = [(i + 1, "", l) for i, l in enumerate(body.split("\n"))]
        # what a translation restates is the English draft, so that is what its
        # numbers get checked against
        sourced = DOCS["paper"].read_text(encoding="utf-8").split(
            "## References")[0].replace("−", "-")
    else:
        restating = _sections(body, {"Abstract", "1", "5", "6"})
        sourced = "\n".join(l for _n, _h, l in
                            _sections(body, {"2", "3", "4", "7"}))
    DEC = r"-?\d+[.]\d{3,}"
    src_vals = [abs(float(o.lstrip("+-")))
                for o in re.findall(DEC, sourced.replace("−", "-"))]
    numeric = []
    for _l, v, _ok in checks:
        try:
            numeric.append(abs(float(str(v).replace("−", "-"))))
        except ValueError:
            pass

    stale = []
    for lineno, head, line in restating:
        for tok in re.findall(DEC, line.replace("−", "-")):
            bare = tok.lstrip("+-")
            # 658.094 is a thousands separator in Turkish, not a decimal
            if abridged and re.fullmatch(r"\d{1,3}[.]\d{3}", bare):
                continue
            if ignore.match(tok) or bare in sourced:
                continue
            val = abs(float(tok))
            tol = 0.5 * 10 ** -len(bare.split(".")[1]) + 1e-12
            if any(abs(val - q) <= tol for q in numeric + src_vals):
                continue
            stale.append((lineno, head, tok, line.strip()))

    where = ("in the text" if abridged else
             "in the abstract, introduction, discussion or limitations")
    if skipped:
        print(f"\n  note: {', '.join(skipped)} not re-derived -- set MEDSAM3_DIR "
              f"to include the adapter-side checks")
    # ------------------------------- values the corrected mask superseded
    # These belong to the model's published preprocessing on RHUH-GBM, which is
    # the sensitivity and not the primary analysis. Each is a real number from a
    # real run, so nothing above rejects them -- but used as though they were
    # primary they say the opposite of what the tables say, which is how the
    # residual came to "change sign" in six places after the correction.
    NL = chr(10)
    # +0.0045 and 0.6569 are deliberately not listed: each also names a
    # different quantity elsewhere (a median estimator, a BraTS-Africa pipeline
    # score), so flagging them costs more in false alarms than it catches.
    SUPERSEDED = {
        "-0.0135": "the RHUH-GBM residual under the published preprocessing",
        "-0.0652": "its contrast against the held-out cohort",
        "0.0279": "the adapter-seed spread under the published preprocessing",
        "28.3": "the cohort ratio taken against that gap",
        "+0.2505": "the headroom taken against the superseded score",
    }
    # both languages, so the Turkish reading version does not have to carry
    # an English word to satisfy this check
    ALLOWED = ("published preprocessing", "published", "Table 5",
               "yayımlanmış", "Tablo 5")
    loose = []
    for para in text.split(chr(10) * 2):
        if any(a in para for a in ALLOWED):
            continue
        for v, what in SUPERSEDED.items():
            if norm(v) in para:
                loose.append((v, what, " ".join(para.split())[:70]))
    if loose:
        print(NL + "SUPERSEDED -- a published-preprocessing figure used "
              "outside its sensitivity context:")
        for v, what, ctx in loose:
            print(f"  {v:<9} {what}" + NL + f"    {ctx}")

    if unverifiable:
        print(NL + "TOO SHORT TO VERIFY -- a one- or two-character value "
              "matches somewhere in any manuscript, so these were not checked:")
        for lbl, v in unverifiable:
            print(f"  {v:<9} {lbl}")
    print(f"\n{len(stale)} restated values {where} are sourced nowhere else")
    for lineno, head, tok, line in stale:
        row = f"  line {lineno:>5}  {tok:<10} {head[:24]:<26}{line[:76]}"
        enc = sys.stdout.encoding or "utf-8"
        print(row.encode(enc, "replace").decode(enc))
    if bad:
        print("\nNOT FOUND — either the draft is stale or the value is simply not quoted:")
        for l, v in bad:
            print(f"  {l:<44} {v}")
    else:
        print("every recomputed value appears in the text")
    # both directions gate: a recomputed value the draft is missing, and a
    # restated value in the prose sections that nothing sources.
    # A skipped optional check used to suppress the exit code entirely, so the
    # gate never fired for anyone without the adapter project -- that is, for
    # every reader of the released archive. Skipped checks are reported and a
    # loose value one of them would have sourced is still forgiven, but a stale
    # value or a missing recomputed one now fails the run.
    # Section 2.5 counts the intervals its multiplicity argument covers; every
    # analysis added since made that number wrong without anything noticing
    miscount = None
    try:
        dtext = io.open(DRIVE_ROOT / "PAPER_DRAFT.md", encoding="utf-8").read()
        stext = io.open(DRIVE_ROOT / "SUPPLEMENTARY.md", encoding="utf-8").read()
    except OSError:
        dtext = stext = ""
    if dtext and stext:
        n = interval_count(dtext, stext)
        m = re.search(r"multiplicity is applied to the (\d+) distinct confidence",
                      " ".join(dtext.split()))
        said = int(m.group(1)) if m else None
        print(f"\nINTERVALS: {n} distinct; Section 2.5 says "
              f"{said if said is not None else 'nothing'}")
        if said is not None and said != n:
            miscount = (said, n)
            print(f"  Section 2.5 is stale: {said} against {n}   ** check **")

    if skipped:
        print(f"  note: {len(skipped)} optional checks were skipped; "
              f"they do not gate this exit code")
    return 1 if (stale or miscount or (loose and not skipped)
                 or (bad and not abridged)) else 0


if __name__ == "__main__":
    sys.exit(main())
