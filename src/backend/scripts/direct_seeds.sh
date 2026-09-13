#!/usr/bin/env bash
# The detector-free arm at every LoRA seed, over all three cohorts.
#
# The study replicated the detector three times but evaluated the detector-free
# arm externally at one adapter seed only, on the grounds that its internal
# three-seed spread was 0.0023 WT Dice. Section 20.4 then established that seed
# spread grows with distance from the training distribution -- which makes the
# internal figure the wrong basis for that decision. This closes the asymmetry:
# both arms get the same replication.
#
# The adapters were checked before running: config, dataset, split and notebook
# hashes are identical across seeds 42, 52 and 62, and all three stopped at step
# 25788. Only the seed differs. seed_52 carries an "experimental" label in the
# research service, but nothing in the checkpoint metadata distinguishes it, and
# its internal WT Dice sits mid-pack, so it is included. That decision was taken
# before any external number for it existed.
#
#   bash scripts/direct_seeds.sh              # seeds 52 and 62
#   bash scripts/direct_seeds.sh 52           # one seed
set -u

# resolve the project root from this script's own location so the file
# runs from a clone; override with DRIVE=/path/to/root
DRIVE="${DRIVE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)}"
PY="${PY:-$DRIVE/demo/rt-detr-sam-pipeline/backend/venv/Scripts/python.exe}"
BACKEND="$DRIVE/src/backend"
REPORTS="$DRIVE/reports"

export YOLO_AUTOINSTALL=false
cd "$BACKEND" || exit 1
say() { echo "[$(date '+%H:%M:%S')] [direct] $*"; }

curl -s --max-time 5 http://127.0.0.1:8020/health >/dev/null 2>&1 \
  || { say "ERROR: sidecar not answering on 8020"; exit 1; }

check() {
  local n; n=$([ -f "$1" ] && grep -c '"patient"' "$1" || echo 0)
  [ "$n" -gt 0 ] || { say "ERROR: $2 wrote no records; see ${1%.jsonl}.log"; exit 1; }
  say "$2 done ($n patients)"
}

run_one() {
  local seed="$1" name="$2" cohort="$3" raw="$4"
  local out="$REPORTS/validation/${name}_direct_seed${seed}.jsonl"
  [ -f "$cohort" ] || { say "ERROR: $cohort missing"; exit 1; }

  # Git Bash rewrites POSIX paths in arguments but never in the environment.
  if [ -n "$raw" ]; then export RAW_DIR="$(cygpath -w "$raw")"; else unset RAW_DIR; fi

  say "seed_${seed} / ${name}: $(wc -l < "$cohort") patients"
  "$PY" -u scripts/full_validation.py \
      --arms direct --adapter "seed_${seed}" --regions WT,TC,ET --stride 1 \
      --patients-file "$cohort" \
      --out "$out" > "${out%.jsonl}.log" 2>&1 \
    || { say "seed_${seed}/${name} FAILED -- see ${out%.jsonl}.log"; exit 1; }
  check "$out" "seed_${seed}/${name}"
}

for seed in ${@:-52 62}; do
  run_one "$seed" clean        "$REPORTS/clean_cohort.txt" ""
  run_one "$seed" rhuh         "$DRIVE/data/external/rhuh/patients.txt"         "$DRIVE/data/external/rhuh"
  run_one "$seed" brats_africa "$DRIVE/data/external/brats_africa/patients.txt" "$DRIVE/data/external/brats_africa"
done

say "complete."
