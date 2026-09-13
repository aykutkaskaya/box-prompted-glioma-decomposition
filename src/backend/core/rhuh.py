"""Where RHUH-GBM's corrected result files live, in one place.

Two preprocessing rules took skull-stripped background to be exactly zero. That
is true of BraTS2020 and BraTS-Africa and false of RHUH-GBM, which ships z-score
normalised with a negative constant background, so `> 0` there thresholds at the
volume mean. Every RHUH-GBM arm was re-run under a corrected rule and written to
a `_normfix` or `_bgfix` file; every figure in the paper is from those.

This map used to be copied into each script that read a result file, and the
copies drifted: six carried all eleven entries and three carried four, so the
figures were drawn from the superseded run while the tables were not, and one
summary file reported an oracle score the manuscript identifies as an artefact.
Import `corrected` instead of writing the map again.

    from core.rhuh import corrected
    path = corrected(V / "rhuh_box.jsonl")
"""
from __future__ import annotations

from pathlib import Path

#: uncorrected file name -> the corrected run to read instead
FIX = {
    "rhuh_box.jsonl": "rhuh_box_normfix.jsonl",
    "rhuh_boundary.jsonl": "rhuh_boundary_normfix.jsonl",
    "rhuh_direct.jsonl": "rhuh_direct_seed42_bgfix.jsonl",
    "rhuh_direct_seed52.jsonl": "rhuh_direct_seed52_bgfix.jsonl",
    "rhuh_direct_seed62.jsonl": "rhuh_direct_seed62_bgfix.jsonl",
    "rhuh_oracle_regions.jsonl": "rhuh_oracle_regions_normfix.jsonl",
    "rhuh_slices.jsonl": "rhuh_slices_normfix.jsonl",
    "rhuh_tc_pipeline.jsonl": "rhuh_tc_pipeline_s1337_normfix.jsonl",
    "rhuh_tc_pipeline_seed42.jsonl": "rhuh_tc_pipeline_s42_normfix.jsonl",
    "rhuh_tc_pipeline_seed62.jsonl": "rhuh_tc_pipeline_s62_normfix.jsonl",
    "rhuh_wt_pipeline_seed42.jsonl": "rhuh_wt_pipeline_seed42_normfix.jsonl",
}

#: the corrected box run was only ever produced for the primary segmenter; the
#: corrected oracle-regions file carries both, so that is where the second
#: segmenter's corrected box arms live
_SECOND_SEGMENTER = {"rhuh_box.jsonl": "rhuh_oracle_regions_normfix.jsonl"}


def corrected(path, segmenter: str | None = None) -> Path:
    """The corrected file for `path`, or `path` where no correction applies."""
    p = Path(path)
    if segmenter and segmenter != "sam2.1_l" and p.name in _SECOND_SEGMENTER:
        return p.with_name(_SECOND_SEGMENTER[p.name])
    return p.with_name(FIX.get(p.name, p.name))
