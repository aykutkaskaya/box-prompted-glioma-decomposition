#!/usr/bin/env bash
# Re-run the oracle-box arm with one tight box per connected component.
#
# The oracle arm passed a single tight box over the whole region; the pipeline
# arm passes every surviving detector box and unions the masks. On a slice
# whose target is multi-component those are not the same protocol, and whole
# tumour is multi-component on roughly half of its positive slices, so the two
# arms were not comparable there. Two claims rest on the difference:
#
#   - the detector-stage term (oracle minus pipeline) is measured between a
#     one-box arm and a many-box arm, which understates it;
#   - "no prompt of this shape can express a multi-component target" is a
#     statement about the oracle instrument, not about box prompting, since
#     this project's own pipeline already unions several boxes per slice.
#
# This measures both. Writes *_components.jsonl beside the originals; nothing
# is overwritten. SAM 2.1-L only, which is the configuration every three-cohort
# result in the manuscript uses.
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
V="$DRIVE/reports/validation"

export YOLO_AUTOINSTALL=false
cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [oracle-components] $*"; }

check() {
  local n; n=$([ -f "$1" ] && grep -c '"patient"' "$1" || echo 0)
  [ "$n" -ge "$3" ] || { say "ERROR: $2 wrote $n/$3; see ${1%.jsonl}.log"; exit 1; }
  say "$2 done ($n patients)"
}

# the sidecar holds most of the card; the oracle arm needs none of it
pid=$(netstat -ano 2>/dev/null | grep LISTENING | grep ":8020" | awk '{print $5}' | head -1)
[ -n "${pid:-}" ] && { taskkill //PID "$pid" //F >/dev/null 2>&1; sleep 4; say "sidecar stopped"; }

run() {  # cohort  raw_dir  patients_file  n
  local c="$1" raw="$2" pats="$3" n="$4"
  say "$c: oracle arm, per-component boxes, WT/TC/ET"
  if [ -n "$raw" ]; then export RAW_DIR="$(cygpath -w "$raw")"; else unset RAW_DIR; fi
  "$PY" -u scripts/full_validation.py \
      --arms oracle --segmenters sam2.1_l \
      --run-id 37 --conf 0.55 --stride 1 \
      --oracle-regions WT,TC,ET --oracle-per-component \
      --patients-file "$pats" \
      --out "$V/${c}_oracle_components.jsonl" \
      > "$V/${c}_oracle_components.log" 2>&1 || { say "$c FAILED"; exit 1; }
  check "$V/${c}_oracle_components.jsonl" "$c" "$n"
}

run clean        ""                            "$DRIVE/reports/clean_cohort.txt"                24
run rhuh         "$DRIVE/data/external/rhuh"   "$DRIVE/data/external/rhuh/patients.txt"         39
run brats_africa "$DRIVE/data/external/brats_africa" \
                 "$DRIVE/data/external/brats_africa/patients.txt"                               95

say "ALL PER-COMPONENT ORACLE ARMS DONE"
