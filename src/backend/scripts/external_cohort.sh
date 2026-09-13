#!/usr/bin/env bash
# Run both arms over any prepared external cohort.
#
# rhuh_cohort.sh hardwires the RHUH paths; it stays as the record of what was
# actually run for that cohort. This is the same two-stage recipe with the
# cohort as an argument, because RHUH-GBM was the first external set and is
# not the last.
#
# The protocol is deliberately frozen: run 37 at conf 0.55, the same two
# segmenters, the same seed_42 adapter, stride 1 -- identical to the internal
# held-out run. If the protocol moved along with the cohort, a change in the
# result would have two explanations and neither could be ruled out.
#
#   bash scripts/external_cohort.sh brats_africa
#   bash scripts/external_cohort.sh <name> [/path/to/prepared/cohort]
#
# The cohort must already be converted by prep_rhuh.py: one directory per
# patient holding <pid>_{flair,t1ce,t2,seg}.nii.gz on the BraTS grid with
# BraTS 2020 label values, plus patients.txt.
set -u

NAME="${1:-}"
[ -n "$NAME" ] || { echo "usage: external_cohort.sh <name> [cohort_dir]"; exit 1; }

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
REPORTS="$DRIVE/reports"
COHORT_DIR="${2:-$DRIVE/data/external/$NAME}"
COHORT="$COHORT_DIR/patients.txt"

export YOLO_AUTOINSTALL=false
# RAW_DIR is read by the Windows interpreter, and Git Bash rewrites POSIX paths
# only in command-line arguments, never in the environment. Passed as
# /c/Users/... it would resolve against C:\c\Users\... and every patient would
# come back "volumes missing, skipped" while the run still exited 0.
export RAW_DIR="$(cygpath -w "$COHORT_DIR")"

cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [$NAME] $*"; }

# full_validation.py skips a patient it cannot load with a log line and still
# exits 0, so an arm that loaded nothing at all reports success. A wrong RAW_DIR
# and a label remap that produced no tumour voxels both fail exactly that way.
records() { [ -f "$1" ] && grep -c '"patient"' "$1" || echo 0; }
check() {
  local n; n=$(records "$1")
  [ "$n" -gt 0 ] || { say "ERROR: $2 wrote no records; see ${1%.jsonl}.log"; exit 1; }
  say "$2 done ($n patients)"
}

[ -f "$COHORT" ] || { say "ERROR: $COHORT missing -- run scripts/prep_rhuh.py first"; exit 1; }
say "$(wc -l < "$COHORT") patients, reading from $RAW_DIR"

if ! curl -s --max-time 5 http://127.0.0.1:8020/health >/dev/null 2>&1; then
  say "ERROR: sidecar not answering on 8020; start it before running this"
  exit 1
fi

say "box arms (oracle + pipeline, SAM1 and SAM 2.1-L)"
"$PY" -u scripts/full_validation.py \
    --arms oracle,pipeline --segmenters sam1_vit_b,sam2.1_l \
    --run-id 37 --conf 0.55 --stride 1 --patients-file "$COHORT" \
    --out "$REPORTS/validation/${NAME}_box.jsonl" \
    > "$REPORTS/validation/${NAME}_box.log" 2>&1 || { say "box arms FAILED"; exit 1; }
check "$REPORTS/validation/${NAME}_box.jsonl" "box arms"

say "detector-free arm (MedSAM3 + LoRA seed_42, WT/TC/ET)"
"$PY" -u scripts/full_validation.py \
    --arms direct --adapter seed_42 --regions WT,TC,ET --stride 1 \
    --patients-file "$COHORT" \
    --out "$REPORTS/validation/${NAME}_direct.jsonl" \
    > "$REPORTS/validation/${NAME}_direct.log" 2>&1 || { say "direct arm FAILED"; exit 1; }
check "$REPORTS/validation/${NAME}_direct.jsonl" "detector-free arm"

say "complete. next: python scripts/compile_external.py ${NAME}_box.jsonl ${NAME}_direct.jsonl --out $REPORTS/${NAME}_summary.json"
