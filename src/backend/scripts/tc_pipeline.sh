#!/usr/bin/env bash
# The tumour-core pipeline arm, over all three cohorts.
#
# Section 18 measured the oracle ceiling for TC and ET by feeding the segmenter
# ground-truth sub-region boxes: it said what a perfect sub-region detector
# could reach, and left open whether one can be trained at all. A TC detector
# now exists (experiments_repeated/tc_seed_1337), so the gap between that
# ceiling and a real detector can be closed by measurement instead of argument.
#
# Only the pipeline arm runs here. The oracle and detector-free arms already
# cover TC on all three cohorts and must not be recomputed: they are the fixed
# reference this arm is read against.
#
# The protocol is the frozen one -- run 37, conf 0.55, stride 1, both box
# segmenters -- with two deliberate changes:
#
#   --pipeline-region TC   scores the arm against tumour core, not whole tumour
#   --detector-ckpt ...    loads the TC weights directly
#
# The second is necessary because the sub-region detector is also run 37: the
# notebook names runs by configuration, not by target, so the registry over
# drive/experiments would hand back the whole-tumour weights without error.
#
#   bash scripts/tc_pipeline.sh            # all three cohorts
#   bash scripts/tc_pipeline.sh rhuh       # one of clean|rhuh|brats_africa
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
REPORTS="$DRIVE/reports"
SEED="${2:-1337}"
CKPT="$DRIVE/experiments_repeated/tc_seed_$SEED/run_037_icarb_alpha_050_w30_clsfl/checkpoints/best.pt"
TAG=""
[ "$SEED" = "1337" ] || TAG="_seed$SEED"

export YOLO_AUTOINSTALL=false
cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [tc] $*"; }

[ -f "$CKPT" ] || { say "ERROR: TC checkpoint for seed $SEED missing at $CKPT"; exit 1; }
say "detector: seed $SEED"

# full_validation.py logs a patient it cannot load and still exits 0, so an arm
# that loaded nothing reports success. A wrong RAW_DIR fails exactly that way.
check() {
  local n; n=$([ -f "$1" ] && grep -c '"patient"' "$1" || echo 0)
  [ "$n" -gt 0 ] || { say "ERROR: $2 wrote no records; see ${1%.jsonl}.log"; exit 1; }
  say "$2 done ($n patients)"
}

run_cohort() {
  local name="$1" cohort="$2" raw="$3"
  local out="$REPORTS/validation/${name}_tc_pipeline${TAG}.jsonl"
  [ -f "$cohort" ] || { say "ERROR: $cohort missing"; exit 1; }

  # Git Bash rewrites POSIX paths in arguments but never in the environment, so
  # RAW_DIR is handed to the Windows interpreter in its own form.
  if [ -n "$raw" ]; then export RAW_DIR="$(cygpath -w "$raw")"; else unset RAW_DIR; fi

  say "$name: $(wc -l < "$cohort") patients${raw:+, reading from $RAW_DIR}"
  "$PY" -u scripts/full_validation.py \
      --arms pipeline --segmenters sam1_vit_b,sam2.1_l \
      --run-id 37 --conf 0.55 --stride 1 \
      --pipeline-region TC --detector-ckpt "$(cygpath -m "$CKPT")" \
      --patients-file "$cohort" \
      --out "$out" > "${out%.jsonl}.log" 2>&1 \
    || { say "$name FAILED -- see ${out%.jsonl}.log"; exit 1; }
  check "$out" "$name"
}

WHICH="${1:-all}"
case "$WHICH" in
  clean|all)         run_cohort clean "$REPORTS/clean_cohort.txt" "" ;;&
  rhuh|all)          run_cohort rhuh "$DRIVE/data/external/rhuh/patients.txt" \
                                     "$DRIVE/data/external/rhuh" ;;&
  brats_africa|all)  run_cohort brats_africa "$DRIVE/data/external/brats_africa/patients.txt" \
                                             "$DRIVE/data/external/brats_africa" ;;&
  clean|rhuh|brats_africa|all) ;;
  *) echo "usage: tc_pipeline.sh [clean|rhuh|brats_africa|all]"; exit 1 ;;
esac

say "complete. next: python scripts/compile_tc.py"
