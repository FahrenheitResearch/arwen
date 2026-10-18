#!/usr/bin/env bash
# Bounded holds for the full recording, shared warm code and isolated outputs.
set -u
L=${1:?lane directory}; E=${2:?box environment}; Q=${3:?queue helper}; R=${4:?recording root}
T=$L/src
source "$E"
export GPUWM_WRF_EXACT=1
workers=${COUPLED_WORKERS:-4}
case "$workers" in 1|2|3|4) ;; *) echo 'COUPLED_WORKERS must be 1..4' >&2; exit 2;; esac
mkdir -p "$L/replay" "$L/receipts"
printf '%s\n' "$L/replay" "$L/receipts" "$L/.prune.lock" >> "$L/MANIFEST.txt"
worker() {
  local worker=$1 shard rc
  for ((shard=worker; shard<=48; shard+=workers)); do
    # Resume the same frozen baseline after queue failures. A complete CLI
    # selection has one valid verdict per selected case/combo, including
    # natural refusals and crashes. Partial files are replayed afresh.
    if [ "${COUPLED_RESUME:-0}" = 1 ] && python - "$L" "$R" "$shard" <<'PY'
import csv, json, sys
from pathlib import Path
root, recordings, shard = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
with (recordings / 'runs.csv').open(newline='') as f:
    keys = {(r['case'], r['combo']) for r in csv.DictReader(f)}
ordered = sorted(keys, key=lambda k: (k[1], k[0]))[shard-1::48]
p = root / f'replay/shard-{shard}/results/{shard}of48.jsonl'
try:
    actual = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    found = [(v['case'], v['combo']) for v in actual]
except (OSError, ValueError, KeyError):
    raise SystemExit(1)
raise SystemExit(0 if found == ordered else 1)
PY
    then
      printf '%s,resumed-complete,%s\n' "$shard" "$(date -u +%FT%TZ)" >> "$L/receipts/shard-exits.csv"
      continue
    fi
    "$Q" pl-coupled-exact-chain --cards 1 --min-cards 1 --est 600 --wait 3600 -- \
      python -m gpuwm.cli verify-exact --recordings "$R" --run-summary "$R/per-run-summary.csv" \
      --shard "$shard/48" --out "$L/replay/shard-$shard" \
      > "$L/receipts/shard-$shard.log" 2>&1
    rc=$?
    printf '%s,%s,%s\n' "$shard" "$rc" "$(date -u +%FT%TZ)" >> "$L/receipts/shard-exits.csv"
    flock "$L/.prune.lock" python tools/coupled_prune.py "$L" >> "$L/receipts/prune.log" 2>&1
  done
}
cd "$T"
for ((worker=1; worker<=workers; worker++)); do worker "$worker" & done
wait
python tools/coupled_exact_chain.py --recordings "$R" --results "$L"/replay/shard-*/results/*of48.jsonl \
  --out "$L/receipts/table"
