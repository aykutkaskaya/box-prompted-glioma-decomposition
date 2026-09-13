"""Recompute the two patient-split hashes the manuscript identifies.

Neither is a sha256 of the released file, so `sha256sum` on either gives a
different value and a reader cannot check them. Both are digests of a canonical
JSON dump, and the two recipes differ:

  detector  sha256(json.dumps(split, sort_keys=True)) over the train/val/test
            dict, with the patient lists in FILE ORDER -- sorting them gives a
            different digest.
  adapter   the same dump with the `split_sha256` key REMOVED first, since the
            field cannot contain a digest of itself.

Prints both, with the expected values, and says whether each matches.

    python scripts/verify_split_hashes.py
"""
from __future__ import annotations

import hashlib
import io
import os
import json
import sys
from pathlib import Path

# The LoRA adapter lives in a separate checkout, so it has no sensible
# default. Set MEDSAM3_DIR to point at it; the checks that need it are
# skipped when it is absent.
MEDSAM3_DIR = Path(os.environ.get("MEDSAM3_DIR", "medsam3-not-set"))


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

ADAPTER = MEDSAM3_DIR / ("training/data/splits/"
               "brats2020_fixed_split.json")
EXPECT_DET = "fae05e5d48dfa31e5d867360d602a04c3e4be61055cbe9fe4d2920c7b3970bfc"
EXPECT_ADP = "f4c2efe198b76a5e4fe10584c7ba9f4eb33538c5463af6caf3f4f9547987ff80"


def digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def main() -> None:
    ok = True

    p = DRIVE_ROOT / "data" / "processed" / "metadata" / "patient_split.json"
    if p.exists():
        got = digest(json.loads(io.open(p, encoding="utf-8").read()))
        ok &= got == EXPECT_DET
        print(f"detector split  {got}  {'MATCH' if got == EXPECT_DET else 'MISMATCH'}")
    else:
        print(f"detector split  missing: {p}")
        ok = False

    if ADAPTER.exists():
        d = json.loads(io.open(ADAPTER, encoding="utf-8").read())
        d.pop("split_sha256", None)
        got = digest(d)
        ok &= got == EXPECT_ADP
        print(f"adapter split   {got}  {'MATCH' if got == EXPECT_ADP else 'MISMATCH'}")
    else:
        print(f"adapter split   not mounted: {ADAPTER}")

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
