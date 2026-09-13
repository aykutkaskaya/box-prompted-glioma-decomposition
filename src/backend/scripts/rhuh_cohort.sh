#!/usr/bin/env bash
# The external cohort: RHUH-GBM preoperative studies, from a hospital neither
# model has ever seen.
#
# The mutually held-out BraTS cohort settled the direction -- detector-free
# ahead on 22 of 24 patients -- but n=24 is too small to pin the size of the
# gap, and both models still met that cohort inside BraTS2020: same
# preprocessing, same annotation protocol, same scanners. A margin that holds
# on the data both models grew up on may be a property of that data.
#
# RHUH-GBM is outside all of it. RT-DETR was trained on BraTS2020 and the
# MedSAM3 LoRA adapter was fine-tuned on BraTS2020, so every patient here is
# held out for both by construction -- no split intersection to compute.
#
# Everything else is deliberately identical to clean_cohort.sh: same run 37
# detector at conf 0.55, same segmenters, same adapter, stride 1. If the
# protocol moved as well as the cohort, a change in the result would have two
# explanations and neither could be ruled out.
#
# Run scripts/prep_rhuh.py first; it produces the layout and the patient list.
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
REPORTS="$DRIVE/reports"
RHUH="$DRIVE/data/external/rhuh"
COHORT="$RHUH/patients.txt"

export YOLO_AUTOINSTALL=false
# RAW_DIR is read by the Windows interpreter, and Git Bash rewrites POSIX paths
# only in command-line arguments, never in the environment. Passed as
# /c/Users/... it would be resolved against C:\c\Users\... and every patient
# would come back "volumes missing, skipped".
export RAW_DIR="$(cygpath -w "$RHUH")"

cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [rhuh] $*"; }

# full_validation.py skips a patient it cannot load with a log line and still
# exits 0, so an arm that loaded nothing at all reports success. A wrong RAW_DIR
# and a label remap that produced no tumour voxels both fail exactly that way,
# and the next sign of trouble would otherwise be compile_external.py refusing
# to run -- after both arms have burned their wall-clock.
records() { [ -f "$1" ] && grep -c '"patient"' "$1" || echo 0; }
check() {
  local n; n=$(records "$1")
  [ "$n" -gt 0 ] || { say "ERROR: $2 wrote no records; see ${1%.jsonl}.log"; exit 1; }
  say "$2 done ($n patients)"
}

[ -f "$COHORT" ] || { say "ERROR: $COHORT missing -- run scripts/prep_rhuh.py first"; exit 1; }
say "$(wc -l < "$COHORT") patients converted from RHUH-GBM, reading from $RAW_DIR"

if ! curl -s --max-time 5 http://127.0.0.1:8020/health >/dev/null 2>&1; then
  say "ERROR: sidecar not answering on 8020; start it before running this"
  exit 1
fi

say "box arms (oracle + pipeline, SAM1 and SAM 2.1-L)"
"$PY" -u scripts/full_validation.py \
    --arms oracle,pipeline --segmenters sam1_vit_b,sam2.1_l \
    --run-id 37 --conf 0.55 --stride 1 --patients-file "$COHORT" \
    --out "$REPORTS/validation/rhuh_box.jsonl" \
    > "$REPORTS/validation/rhuh_box.log" 2>&1 || { say "box arms FAILED"; exit 1; }
check "$REPORTS/validation/rhuh_box.jsonl" "box arms"

say "detector-free arm (MedSAM3 + LoRA seed_42, WT/TC/ET)"
"$PY" -u scripts/full_validation.py \
    --arms direct --adapter seed_42 --regions WT,TC,ET --stride 1 \
    --patients-file "$COHORT" \
    --out "$REPORTS/validation/rhuh_direct.jsonl" \
    > "$REPORTS/validation/rhuh_direct.log" 2>&1 || { say "direct arm FAILED"; exit 1; }
check "$REPORTS/validation/rhuh_direct.jsonl" "detector-free arm"

say "external cohort complete. next: python scripts/compile_external.py"
