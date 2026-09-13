#!/usr/bin/env bash
# The whole-tumour pipeline arm at a second detector seed, over all three cohorts.
#
# The headline result of the paper is a decomposition measured from one
# detector. The paper also argues that out-of-domain comparisons should not be
# believed from a single training run, which leaves an obvious question about
# its own headline. This answers it for the term the title rests on: whether
# the detector-stage term still grows across the three cohorts when the
# detector is trained from a different seed.
#
# Only the pipeline arm runs. The oracle arm is prompted with the ground-truth
# box and the detector-free arm uses no box at all, so neither depends on the
# detector seed and both are reused from the frozen protocol run. One
# segmenter, SAM 2.1-L, because that is the one the three-cohort tables report.
#
# Everything else is the frozen protocol: conf 0.55, stride 1, whole tumour.
#
#   bash scripts/wt_pipeline_seed.sh 42          # all three cohorts
#   bash scripts/wt_pipeline_seed.sh 42 rhuh     # one of clean|rhuh|brats_africa
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
REPORTS="$DRIVE/reports"
SEED="${1:-42}"
CKPT="$DRIVE/experiments_repeated/seed_$SEED/run_037_icarb_alpha_050_w30_clsfl/checkpoints/best.pt"

export YOLO_AUTOINSTALL=false
cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [wt seed $SEED] $*"; }

[ -f "$CKPT" ] || { say "ERROR: no run 37 checkpoint for WT seed $SEED at $CKPT"; exit 1; }
say "detector: $CKPT"

# full_validation.py logs a patient it cannot load and still exits 0, so an arm
# that loaded nothing reports success. A wrong RAW_DIR fails exactly that way.
check() {
  local n; n=$([ -f "$1" ] && grep -c '"patient"' "$1" || echo 0)
  [ "$n" -gt 0 ] || { say "ERROR: $2 wrote no records; see ${1%.jsonl}.log"; exit 1; }
  say "$2 done ($n patients)"
}

run_cohort() {
  local name="$1" cohort="$2" raw="$3"
  local out="$REPORTS/validation/${name}_wt_pipeline_seed${SEED}.jsonl"
  [ -f "$cohort" ] || { say "ERROR: $cohort missing"; exit 1; }

  # Git Bash rewrites POSIX paths in arguments but never in the environment, so
  # RAW_DIR is handed to the Windows interpreter in its own form.
  if [ -n "$raw" ]; then export RAW_DIR="$(cygpath -w "$raw")"; else unset RAW_DIR; fi

  say "$name: $(wc -l < "$cohort") patients${raw:+, reading from $RAW_DIR}"
  "$PY" -u scripts/full_validation.py \
      --arms pipeline --segmenters sam2.1_l \
      --run-id 37 --conf 0.55 --stride 1 \
      --pipeline-region WT --detector-ckpt "$(cygpath -m "$CKPT")" \
      --patients-file "$cohort" \
      --out "$out" > "${out%.jsonl}.log" 2>&1 \
    || { say "$name FAILED -- see ${out%.jsonl}.log"; exit 1; }
  check "$out" "$name"
}

WHICH="${2:-all}"
case "$WHICH" in
  clean|all)         run_cohort clean "$REPORTS/clean_cohort.txt" "" ;;&
  rhuh|all)          run_cohort rhuh "$DRIVE/data/external/rhuh/patients.txt" \
                                     "$DRIVE/data/external/rhuh" ;;&
  brats_africa|all)  run_cohort brats_africa "$DRIVE/data/external/brats_africa/patients.txt" \
                                             "$DRIVE/data/external/brats_africa" ;;&
  clean|rhuh|brats_africa|all) ;;
  *) echo "usage: wt_pipeline_seed.sh <seed> [clean|rhuh|brats_africa|all]"; exit 1 ;;
esac

say "complete. next: python scripts/compile_wt_seeds.py"
