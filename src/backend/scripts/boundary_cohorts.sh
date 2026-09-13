#!/usr/bin/env bash
# HD95 and NSD for the oracle-box and pipeline arms, over all three cohorts.
#
# Answers the reviewer question the manuscript invites by citing Metrics
# Reloaded and then reporting overlap only. Resumable: a patient already in the
# output is skipped, so this can be interrupted and restarted.
#
# The detector-free arm is not covered here -- it needs the MedSAM3 sidecar on
# port 8020, and its masks would have to come from that service rather than
# from the segmenter this script loads. Start the sidecar and extend the script
# if that arm is wanted too.
#
#   bash scripts/boundary_cohorts.sh            # all three
#   bash scripts/boundary_cohorts.sh rhuh       # one of clean|rhuh|brats_africa
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
REPORTS="$DRIVE/reports"

export YOLO_AUTOINSTALL=false
cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [boundary] $*"; }

check() {
  local n; n=$([ -f "$1" ] && grep -c '"patient"' "$1" || echo 0)
  [ "$n" -gt 0 ] || { say "ERROR: $2 wrote no records; see ${1%.jsonl}.log"; exit 1; }
  say "$2 done ($n patients)"
}

run_cohort() {
  local name="$1" cohort="$2" raw="$3"
  local out="$REPORTS/validation/${name}_boundary.jsonl"
  [ -f "$cohort" ] || { say "ERROR: $cohort missing"; exit 1; }

  # Git Bash rewrites POSIX paths in arguments but never in the environment.
  if [ -n "$raw" ]; then export RAW_DIR="$(cygpath -w "$raw")"; else unset RAW_DIR; fi

  say "$name: $(wc -l < "$cohort") patients${raw:+, reading from $RAW_DIR}"
  "$PY" -u scripts/boundary_metrics.py \
      --run-id 37 --conf 0.55 --region WT \
      --patients-file "$cohort" \
      --out "$out" >> "${out%.jsonl}.log" 2>&1 \
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
  *) echo "usage: boundary_cohorts.sh [clean|rhuh|brats_africa|all]"; exit 1 ;;
esac

say "complete. next: python scripts/compile_boundary.py"
