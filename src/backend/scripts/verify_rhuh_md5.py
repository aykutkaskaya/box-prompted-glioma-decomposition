"""Check the RHUH-GBM payload against the MD5 manifest shipped with it.

The manuscript states that the package was verified, 720 of 720. That was true
but unrecorded: nothing on disk held the result, so the claim could not be
checked by a reader. This writes reports/rhuh_md5_verify.json.

    python scripts/verify_rhuh_md5.py
"""
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import DRIVE_ROOT  # noqa: E402

PKG = DRIVE_ROOT / "data" / "external" / "rhuh_raw" / "PKG - RHUH-GBM-nii-v1"
OUT = DRIVE_ROOT / "reports" / "rhuh_md5_verify.json"


def main() -> None:
    sums = next(PKG.glob("*.sums"), None)
    if sums is None:
        print(f"no .sums manifest under {PKG}")
        return
    entries = []
    for line in io.open(sums, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition(" ")
        entries.append((digest.strip(), rel.strip().lstrip("*")))

    ok = missing = mismatch = 0
    bad = []
    for digest, rel in entries:
        p = PKG / rel
        if not p.exists():
            missing += 1
            bad.append({"file": rel, "problem": "missing"})
            continue
        h = hashlib.md5()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        if h.hexdigest() == digest:
            ok += 1
        else:
            mismatch += 1
            bad.append({"file": rel, "problem": "mismatch"})

    result = {"manifest": sums.name, "entries": len(entries), "ok": ok,
              "missing": missing, "mismatch": mismatch, "failures": bad[:20]}
    OUT.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"{ok}/{len(entries)} verified, {missing} missing, {mismatch} mismatched")
    print(f"-> {OUT.relative_to(DRIVE_ROOT)}")


if __name__ == "__main__":
    main()
