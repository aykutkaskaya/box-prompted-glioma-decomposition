#!/usr/bin/env bash
# The mutually held-out cohort: 24 patients that are in neither model's
# training set.
#
# The 55-patient comparison is not usable as it stands. The RT-DETR study and
# the MedSAM3 LoRA project each split BraTS2020 into 258/55/55, but with
# different assignments, so 45 of the RT-DETR test patients sit in the LoRA
# training set. Measuring one model on data the other memorised is not a
# comparison.
#
# This runs both pipelines on the intersection of the two held-out pools. The
# cohort is stratified (see reports/clean_cohort.txt and the compile step):
# 4 patients are test-set for both, the rest are test for one and validation
# for the other, so the residual advantage can be checked for direction.
#
# The box arms already cover the 10 cohort members that are in the RT-DETR
# test split; only the 14 from its validation split need computing.
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
REPORTS="$DRIVE/reports"
COHORT="$REPORTS/clean_cohort.txt"

export YOLO_AUTOINSTALL=false
cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [clean] $*"; }

[ -f "$COHORT" ] || { say "ERROR: $COHORT missing"; exit 1; }
say "$(wc -l < "$COHORT") patients in the mutually held-out cohort"

if ! curl -s --max-time 5 http://127.0.0.1:8020/health >/dev/null 2>&1; then
  say "ERROR: sidecar not answering on 8020; start it before running this"
  exit 1
fi

say "box arms (oracle + pipeline, SAM1 and SAM 2.1-L)"
"$PY" -u scripts/full_validation.py \
    --arms oracle,pipeline --segmenters sam1_vit_b,sam2.1_l \
    --run-id 37 --conf 0.55 --stride 1 --patients-file "$COHORT" \
    --out "$REPORTS/validation/clean_box.jsonl" \
    > "$REPORTS/validation/clean_box.log" 2>&1 || { say "box arms FAILED"; exit 1; }
say "box arms done"

say "detector-free arm (MedSAM3 + LoRA seed_42, WT/TC/ET)"
"$PY" -u scripts/full_validation.py \
    --arms direct --adapter seed_42 --regions WT,TC,ET --stride 1 \
    --patients-file "$COHORT" \
    --out "$REPORTS/validation/clean_direct.jsonl" \
    > "$REPORTS/validation/clean_direct.log" 2>&1 || { say "direct arm FAILED"; exit 1; }
say "detector-free arm done"

say "clean cohort complete."
