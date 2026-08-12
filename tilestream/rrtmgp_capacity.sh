#!/usr/bin/env bash
# What the RRTMGP reclamations buy, in domain terms, on a 12 GB and a 16 GB
# card.  One fresh subprocess per bisection trial (tilestream.vram_probe
# ceiling), pool-capped so every refusal is a budget refusal and not a
# device-level OOM caused by a co-tenant.
#
# The cap is CARD TOTAL minus the measured non-pool footprint -- the CUDA
# context plus the NVRTC module images, which no pool counter sees and which
# a small card pays first.  Measure it for the rung you are bisecting;
# full+MYNN+Noah-MP compiles more than 'full' does.
#
# usage:  rrtmgp_capacity.sh <nonpool_gib> [rung]
set -euo pipefail
NONPOOL_GIB="${1:?usage: rrtmgp_capacity.sh <nonpool_gib> [rung]}"
RUNG="${2:-full+MYNN+Noah-MP}"
NZ=49
STEPS=2

cap () {  # $1 = card total GiB -> pool cap in bytes
  python3 -c "print(int((${1} - ${NONPOOL_GIB}) * 2**30))"
}

run () {  # $1 = label, $2 = cap bytes, rest = extra flags
  local label="$1"; shift
  local limit="$1"; shift
  echo "=== ${label}  (pool cap $(python3 -c "print(f'{${limit}/2**30:.3f}')") GiB)"
  taskset -c 64-127 python -u -m tilestream.vram_probe ceiling \
      --rung "${RUNG}" --nz "${NZ}" --nbuffers 2 --steps "${STEPS}" \
      --low 96 --high 640 --step 16 \
      --pool-limit-bytes "${limit}" "$@" 2>&1 | tail -3
  echo
}

for TOTAL in 11.940 15.920; do
  LIMIT="$(cap "${TOTAL}")"
  echo "############ ${TOTAL} GiB CARD ############"
  run "as shipped (private workspaces)"        "${LIMIT}"
  run "shared, chunk 3125"                     "${LIMIT}" --share
  run "shared, chunk 3125, TIGHT"              "${LIMIT}" --share --rrtmgp-tight
  run "shared, chunk 1024, MYNN 4096"          "${LIMIT}" --share \
      --rrtmgp-column-chunk 1024 --mynn-column-chunk 4096
  run "shared, chunk 1024, MYNN 4096, TIGHT"   "${LIMIT}" --share \
      --rrtmgp-column-chunk 1024 --mynn-column-chunk 4096 --rrtmgp-tight
done
