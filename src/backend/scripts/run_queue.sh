#!/usr/bin/env bash
# Sequential validation passes, one model family on the card at a time.
#
# The first attempt ran the box segmenters and MedSAM3 concurrently on an 8 GB
# card. MedSAM3 alone wants ~5 GB and SAM1 + SAM2.1-L + RT-DETR together want
# ~3 GB, so the sidecar OOM'd, the CUDA context was poisoned, and 30 of 55
# patients produced nothing. The fix is not more retries: it is never having
# both resident.
#
#   pass 1  box arms  (oracle + pipeline, SAM1 and SAM2.1-L)   sidecar stopped
#   pass 2  B: does SAM 2.1-L reorder the 37 losses?           sidecar stopped
#   pass 3  direct arms (MedSAM3 +LoRA seeds, then TC/ET)      sidecar only
#
# Every pass is resumable and stops the queue if it fails, rather than letting
# the next one run against a broken GPU.
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
MPY="${MPY:-$DRIVE/models/medical_sam3/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
SIDECAR="$DRIVE/src/segmenters/medsam3_service.py"
REPORTS="$DRIVE/reports"

export YOLO_AUTOINSTALL=false
cd "$BACKEND" || exit 1
mkdir -p "$REPORTS/validation"

say() { echo "[$(date '+%H:%M:%S')] [queue] $*"; }

gpu_free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1; }

stop_sidecar() {
  local pid
  pid=$(netstat -ano 2>/dev/null | grep -E "LISTENING" | grep ":8020" | awk '{print $5}' | head -1)
  if [ -n "${pid:-}" ]; then
    powershell.exe -NoProfile -Command "Stop-Process -Id $pid -Force -ErrorAction SilentlyContinue" >/dev/null 2>&1
    say "sidecar (pid $pid) stopped"
    sleep 5
  fi
}

start_sidecar() {
  say "starting sidecar…"
  powershell.exe -NoProfile -Command \
    "Start-Process -FilePath '$(cygpath -w "$MPY")' -ArgumentList '$(cygpath -w "$SIDECAR")' -WorkingDirectory '$(cygpath -w "$DRIVE")' -RedirectStandardOutput '$REPORTS\\sidecar.log' -RedirectStandardError '$REPORTS\\sidecar.err.log' -WindowStyle Hidden" >/dev/null 2>&1
  for _ in $(seq 1 30); do
    if curl -s --max-time 3 http://127.0.0.1:8020/health >/dev/null 2>&1; then
      say "sidecar up (GPU free: $(gpu_free_mib) MiB)"
      return 0
    fi
    sleep 3
  done
  say "ERROR: sidecar did not come up"
  return 1
}

run_pass() {   # name, logfile, command...
  local name="$1"; shift
  local logf="$1"; shift
  say "START $name  (GPU free: $(gpu_free_mib) MiB)"
  "$@" > "$logf" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    say "FAILED $name (exit $rc) — see $(basename "$logf"); stopping the queue"
    return 1
  fi
  say "OK $name"
  return 0
}

# ---------------------------------------------------------------- pass 1
stop_sidecar
run_pass "pass1 box arms" "$REPORTS/validation_box.log" \
  "$PY" -u scripts/full_validation.py \
    --arms oracle,pipeline --segmenters sam1_vit_b,sam2.1_l \
    --run-id 37 --conf 0.55 --stride 1 \
    --out "$REPORTS/validation/box_arms.jsonl" || exit 1

# ---------------------------------------------------------------- pass 2
run_pass "pass2 segmenter ranking" "$REPORTS/segmenter_ranking.log" \
  "$PY" -u scripts/segmenter_ranking.py \
    --segmenter sam2.1_l --conf 0.25 --stride 1 \
    --out "$REPORTS/segmenter_ranking.json" || exit 1

# ---------------------------------------------------------------- pass 3
start_sidecar || exit 1

run_pass "pass3a direct seed_42 (WT,TC,ET)" "$REPORTS/validation/direct_seed_42.log" \
  "$PY" -u scripts/full_validation.py \
    --arms direct --adapter seed_42 --regions WT,TC,ET --stride 1 \
    --out "$REPORTS/validation/direct_seed_42.jsonl" || exit 1

for SEED in seed_62 seed_52; do
  run_pass "pass3 direct $SEED" "$REPORTS/validation/direct_${SEED}.log" \
    "$PY" -u scripts/full_validation.py \
      --arms direct --adapter "$SEED" --regions WT --stride 1 \
      --out "$REPORTS/validation/direct_${SEED}.jsonl" || exit 1
done

stop_sidecar
say "all passes complete."
