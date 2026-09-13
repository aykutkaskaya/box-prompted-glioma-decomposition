#!/usr/bin/env bash
# The detector-free arm on RHUH-GBM under the corrected brain mask.
#
# The box arms were re-run when the `> 0` brain mask was found to be wrong on
# this cohort; the detector-free arm was not, because its input mirrors the
# MedSAM3 project's own preprocessing verbatim so the model sees what it was
# trained on. The cost of that choice is that on RHUH-GBM the three arms stop
# seeing identical volumes, which is a fair objection to a three-arm
# comparison. This runs the arm both ways so the two can be reported side by
# side; the verbatim mirror remains the primary arm.
#
# Writes rhuh_direct_seed{42,52,62}_bgfix.jsonl. Nothing is overwritten.
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
MPY="${MPY:-$DRIVE/models/medical_sam3/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
V="$DRIVE/reports/validation"
SIDECAR="$DRIVE/src/segmenters/medsam3_service.py"
PATIENTS="$DRIVE/data/external/rhuh/patients.txt"

export YOLO_AUTOINSTALL=false
export RAW_DIR="$(cygpath -w "$DRIVE/data/external/rhuh")"
cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [rhuh-direct-bgfix] $*"; }

if ! curl -s -m 5 http://127.0.0.1:8020/health >/dev/null 2>&1; then
  say "starting sidecar"
  "$MPY" "$SIDECAR" > "$V/rhuh_direct_bgfix_sidecar.log" 2>&1 &
  for _ in $(seq 1 30); do
    curl -s -m 5 http://127.0.0.1:8020/health >/dev/null 2>&1 && break
    sleep 5
  done
  curl -s -m 5 http://127.0.0.1:8020/health >/dev/null || {
    say "sidecar did not come up"; exit 1; }
fi

for s in 42 52 62; do
  say "detector-free arm, adapter seed $s, corrected normalisation"
  "$PY" -u scripts/full_validation.py \
      --arms direct --adapter "seed_$s" --regions WT,TC,ET --stride 1 \
      --direct-bgfix --patients-file "$PATIENTS" \
      --out "$V/rhuh_direct_seed${s}_bgfix.jsonl" \
      > "$V/rhuh_direct_seed${s}_bgfix.log" 2>&1 || { say "seed $s FAILED"; exit 1; }
  n=$(grep -c '"patient"' "$V/rhuh_direct_seed${s}_bgfix.jsonl")
  [ "$n" -ge 39 ] || { say "ERROR: seed $s wrote $n/39"; exit 1; }
  say "seed $s done ($n patients)"
done

say "ALL DONE"
