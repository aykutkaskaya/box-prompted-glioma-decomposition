"""Carry the usable half of the failed overnight run into the new box-arm pass.

That run wrote one record per patient covering all five arms. When the sidecar
died, the record was stamped with an error even though the four box arms had
already completed successfully. Those box measurements are valid — the failure
was in a later, unrelated arm — so they are salvaged here rather than recomputed.

The error field and the detector-free arm are dropped, because only the box arms
are being carried over.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

SRC = DRIVE_ROOT / "reports" / "validation" / "validation_run37_stride1.jsonl"
DST = DRIVE_ROOT / "reports" / "validation" / "box_arms.jsonl"
NEEDED = ["oracle:sam1_vit_b", "oracle:sam2.1_l", "pipeline:sam1_vit_b", "pipeline:sam2.1_l"]

if not SRC.exists():
    print(f"nothing to salvage: {SRC} not found")
    raise SystemExit(0)

kept, seen = [], set()
total = 0
for line in SRC.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    total += 1
    rec = json.loads(line)
    pid = rec.get("patient")
    arms = rec.get("arms", {})
    if pid in seen or not all(k in arms for k in NEEDED):
        continue
    seen.add(pid)
    rec.pop("error", None)
    rec["arms"] = {k: v for k, v in arms.items() if k in NEEDED}
    rec["salvaged_from"] = SRC.name
    kept.append(json.dumps(rec))

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
print(f"read {total} records, salvaged {len(kept)} patients with all four box arms -> {DST.name}")
print(f"the box pass will resume at patient {len(kept) + 1} of 55")
