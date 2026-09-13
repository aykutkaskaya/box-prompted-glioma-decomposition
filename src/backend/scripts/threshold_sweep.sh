#!/usr/bin/env bash
# Re-run the detector arm at several confidence thresholds on one cohort.
#
# The 0.55 threshold was chosen on the BraTS2020 validation split and never
# revisited. Internally that was defensible -- section 16.2 measured the
# optimism from tuning it on test at +0.0003, and the optimum was broad. On the
# external cohorts it is an open wound: RT-DETR's recall falls from 0.811
# internally to 0.763 on RHUH and 0.720 on BraTS-Africa while its precision
# stays high, which is the signature of a threshold set too conservatively for
# the data in front of it.
#
# That matters because the paper's strongest claim rests on the gap between the
# oracle ceiling and the real pipeline, and reads that gap as the detector's
# cost. If part of it is a mis-set threshold rather than the detector's ability,
# the claim is overstated. This measures which.
#
# Only the detector arm and only one segmenter: the oracle arm does not use a
# threshold, and the second segmenter would double the cost to re-confirm a
# ranking already established three times.
#
#   bash scripts/threshold_sweep.sh brats_africa "0.35 0.45"
set -u

NAME="${1:-}"
CONFS="${2:-0.35 0.45}"
SEGMENTER="${3:-sam2.1_l}"
[ -n "$NAME" ] || { echo "usage: threshold_sweep.sh <cohort> [\"conf conf ...\"] [segmenter]"; exit 1; }

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
REPORTS="$DRIVE/reports"
COHORT_DIR="$DRIVE/data/external/$NAME"
COHORT="$COHORT_DIR/patients.txt"

export YOLO_AUTOINSTALL=false
# see external_cohort.sh: Git Bash does not convert POSIX paths in the environment
export RAW_DIR="$(cygpath -w "$COHORT_DIR")"

cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [sweep:$NAME] $*"; }
records() { [ -f "$1" ] && grep -c '"patient"' "$1" || echo 0; }

[ -f "$COHORT" ] || { say "ERROR: $COHORT missing"; exit 1; }
say "$(wc -l < "$COHORT") patients, segmenter $SEGMENTER, thresholds: $CONFS"

for CONF in $CONFS; do
  TAG="${CONF//./}"
  OUT="$REPORTS/validation/${NAME}_conf${TAG}.jsonl"
  say "conf $CONF -> $(basename "$OUT")"
  "$PY" -u scripts/full_validation.py \
      --arms pipeline --segmenters "$SEGMENTER" \
      --run-id 37 --conf "$CONF" --stride 1 --patients-file "$COHORT" \
      --out "$OUT" \
      > "${OUT%.jsonl}.log" 2>&1 || { say "conf $CONF FAILED"; exit 1; }
  n=$(records "$OUT")
  [ "$n" -gt 0 ] || { say "ERROR: conf $CONF wrote no records; see ${OUT%.jsonl}.log"; exit 1; }
  say "conf $CONF done ($n patients)"
done

say "sweep complete. The 0.55 baseline is already in ${NAME}_box.jsonl."
