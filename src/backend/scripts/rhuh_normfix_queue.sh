#!/usr/bin/env bash
# Re-run every RHUH-GBM artefact under the corrected intensity normalisation.
#
# `normalize_brain_slice` took the brain mask as `s > 0`, which is right where
# skull-stripped background is zero and wrong on RHUH-GBM, which ships z-score
# normalised with a negative constant background. On that cohort the old rule
# discarded 43-62% of the brain and 62% of the tumour in T1ce, so every RHUH
# number was produced from a differently preprocessed image than the other two
# cohorts. The fix is bit-identical on BraTS2020 and BraTS-Africa, so only this
# cohort needs re-running.
#
# Writes *_normfix.jsonl beside the originals; nothing is overwritten.
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
MPY="${MPY:-$DRIVE/models/medical_sam3/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
V="$DRIVE/reports/validation"
PATIENTS="$DRIVE/data/external/rhuh/patients.txt"
SIDECAR="$DRIVE/src/segmenters/medsam3_service.py"

export YOLO_AUTOINSTALL=false
export RAW_DIR="$(cygpath -w "$DRIVE/data/external/rhuh")"
cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [rhuh-normfix] $*"; }

check() {
  local n; n=$([ -f "$1" ] && grep -c '"patient"' "$1" || echo 0)
  [ "$n" -ge 39 ] || { say "ERROR: $2 wrote $n/39; see ${1%.jsonl}.log"; exit 1; }
  say "$2 done ($n patients)"
}

stop_sidecar() {
  local pid
  pid=$(netstat -ano 2>/dev/null | grep LISTENING | grep ":8020" | awk '{print $5}' | head -1)
  [ -n "${pid:-}" ] && { taskkill //PID "$pid" //F >/dev/null 2>&1; sleep 4; say "sidecar stopped"; }
}

# ---------------------------------------------------------------- GPU phase
stop_sidecar
say "phase 1: detector and SAM arms (sidecar down so the card is free)"

say "box arms, both segmenters, per region"
"$PY" -u scripts/full_validation.py \
    --arms oracle,pipeline --segmenters sam1_vit_b,sam2.1_l \
    --run-id 37 --conf 0.55 --stride 1 --oracle-regions WT,TC,ET \
    --patients-file "$PATIENTS" \
    --out "$V/rhuh_oracle_regions_normfix.jsonl" \
    > "$V/rhuh_oracle_regions_normfix.log" 2>&1 || { say "box arms FAILED"; exit 1; }
check "$V/rhuh_oracle_regions_normfix.jsonl" "box arms + regions"

say "second detector seed"
"$PY" -u scripts/full_validation.py \
    --arms pipeline --segmenters sam2.1_l \
    --run-id 37 --conf 0.55 --stride 1 \
    --detector-ckpt "$(cygpath -m "$DRIVE/experiments_repeated/seed_42/run_037_icarb_alpha_050_w30_clsfl/checkpoints/best.pt")" \
    --patients-file "$PATIENTS" \
    --out "$V/rhuh_wt_pipeline_seed42_normfix.jsonl" \
    > "$V/rhuh_wt_pipeline_seed42_normfix.log" 2>&1 || { say "seed42 FAILED"; exit 1; }
check "$V/rhuh_wt_pipeline_seed42_normfix.jsonl" "detector seed 42"

for s in 1337 42 62; do
  say "tumour-core pipeline, detector seed $s"
  "$PY" -u scripts/full_validation.py \
      --arms pipeline --segmenters sam1_vit_b,sam2.1_l \
      --run-id 37 --conf 0.55 --stride 1 \
      --pipeline-region TC \
      --detector-ckpt "$(cygpath -m "$DRIVE/experiments_repeated/tc_seed_$s/run_037_icarb_alpha_050_w30_clsfl/checkpoints/best.pt")" \
      --patients-file "$PATIENTS" \
      --out "$V/rhuh_tc_pipeline_s${s}_normfix.jsonl" \
      > "$V/rhuh_tc_pipeline_s${s}_normfix.log" 2>&1 || { say "TC $s FAILED"; exit 1; }
  check "$V/rhuh_tc_pipeline_s${s}_normfix.jsonl" "TC pipeline seed $s"
done

say "slice decomposition"
"$PY" -u scripts/slice_decomposition.py \
    --run-id 37 --conf 0.55 --region WT --patients-file "$PATIENTS" \
    --out "$V/rhuh_slices_normfix.jsonl" \
    > "$V/rhuh_slices_normfix.log" 2>&1 || { say "slices FAILED"; exit 1; }
check "$V/rhuh_slices_normfix.jsonl" "slice decomposition"

say "boundary metrics"
"$PY" -u scripts/boundary_metrics.py \
    --run-id 37 --conf 0.55 --region WT --patients-file "$PATIENTS" \
    --out "$V/rhuh_boundary_normfix.jsonl" \
    > "$V/rhuh_boundary_normfix.log" 2>&1 || { say "boundary FAILED"; exit 1; }
check "$V/rhuh_boundary_normfix.jsonl" "boundary metrics"

# ------------------------------------------------------------ sidecar phase
say "phase 2: detector-free arm (sidecar up)"
"$MPY" "$SIDECAR" > "$DRIVE/reports/validation/rhuh_normfix_sidecar.log" 2>&1 &
sleep 50
curl -s -m 5 http://127.0.0.1:8020/health >/dev/null || { say "sidecar did not come up"; exit 1; }

for s in 42 52 62; do
  say "detector-free arm, adapter seed $s"
  "$PY" -u scripts/full_validation.py \
      --arms direct --adapter "seed_$s" --regions WT,TC,ET --stride 1 \
      --patients-file "$PATIENTS" \
      --out "$V/rhuh_direct_seed${s}_normfix.jsonl" \
      > "$V/rhuh_direct_seed${s}_normfix.log" 2>&1 || { say "direct $s FAILED"; exit 1; }
  check "$V/rhuh_direct_seed${s}_normfix.jsonl" "detector-free seed $s"
done

stop_sidecar
say "ALL RHUH ARTEFACTS REBUILT"
