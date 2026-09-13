#!/usr/bin/env bash
# Pass 3: the detector-free arms, run against an already-running sidecar.
#
# Split out of run_queue.sh because that script's attempt to launch the sidecar
# from bash through PowerShell silently failed on quoting -- it reported "did
# not come up" while nothing had ever been started. This one refuses to run
# unless the sidecar answers first, so the failure mode is a clear message
# rather than an hour of error records.
#
# Nothing else may hold the GPU: MedSAM3 needs about 5 GB of the 8 GB card.
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
REPORTS="$DRIVE/reports"

export YOLO_AUTOINSTALL=false
cd "$BACKEND" || exit 1
mkdir -p "$REPORTS/validation"

say() { echo "[$(date '+%H:%M:%S')] [pass3] $*"; }

if ! curl -s --max-time 5 http://127.0.0.1:8020/health >/dev/null 2>&1; then
  say "ERROR: the sidecar is not answering on 8020. Start it first:"
  say "  drive/models/medical_sam3/venv/Scripts/python.exe drive/src/segmenters/medsam3_service.py"
  exit 1
fi
say "sidecar is up (GPU free: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) MiB)"

run_one() {   # label, extra args...
  local label="$1"; shift
  local logf="$REPORTS/validation/${label}.log"
  say "START $label"
  "$PY" -u scripts/full_validation.py --arms direct --stride 1 "$@" \
      --out "$REPORTS/validation/${label}.jsonl" > "$logf" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    say "FAILED $label (exit $rc) — see validation/${label}.log; stopping"
    return 1
  fi
  say "OK $label"
}

# seed_42 carries the extra regions: one forward pass yields WT, TC and ET, so
# scoring the sub-regions costs nothing beyond the WT run.
run_one "direct_seed_42" --adapter seed_42 --regions WT,TC,ET || exit 1
run_one "direct_seed_62" --adapter seed_62 --regions WT        || exit 1
run_one "direct_seed_52" --adapter seed_52 --regions WT        || exit 1

say "pass 3 complete."
