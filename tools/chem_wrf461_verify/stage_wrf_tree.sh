#!/usr/bin/env bash
# Stage a plain WRF checkout as the chem oracles' pinned reference layout.
#
#   bash tools/chem_wrf461_verify/stage_wrf_tree.sh WRF_CHECKOUT OUT_ROOT REPORT
#
# The chem oracle builders (tools/chem_wrf471_oracle/**/build*.sh) refuse any
# source whose bytes differ from its pinned SHA-256.  Those pins were taken
# from a Windows checkout of WRF v4.7.1, so most are the CRLF form of the file
# and a few are the LF form; a plain Linux clone of v4.7.1 fails them all on
# line endings alone.
#
# For every path pinned in any SOURCES*.sha256 (GSL and UPP paths excluded,
# they are not WRF), this copies WRF_CHECKOUT/<path> to OUT_ROOT/<path> in
# whichever line-ending form (as-is, or LF->CRLF) reproduces the pin.  When
# neither form does, the file's CONTENT differs from the pinned v4.7.1 file:
# it is staged in CRLF form anyway (so the builders' own check reports it)
# and listed as DIFFERS in REPORT.  Line endings are the only bytes this
# script ever changes; Fortran reads both forms identically.
#
# The harness's own reference-pm/registry-pm.F90 is staged at oracle/ as the
# PM builder expects.
set -euo pipefail
src=$(realpath "${1:?usage: stage_wrf_tree.sh WRF_CHECKOUT OUT_ROOT REPORT}")
out=$(realpath -m "${2:?}")
report=$(realpath -m "${3:?}")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
oracle=$(cd "$here/../chem_wrf471_oracle" && pwd)
mkdir -p "$out"
: > "$report"
cat "$oracle"/SOURCES*.sha256 "$oracle"/gocart/SOURCES*.sha256 \
    "$oracle"/wesely/SOURCES.sha256 "$oracle"/gocart/source-sha256sums.txt |
    grep -v '^#' | sed 's/ \*/  /' | awk 'NF==2' | sort -u -k2,2 |
while read -r sum rel; do
    case "$rel" in gsl-ccpp-physics/*|upp/*|phys/MYNN-EDMF/*|SOURCES.txt|oracle/*) continue ;; esac
    if [[ ! -f "$src/$rel" ]]; then
        echo "MISSING  $rel" >> "$report"; continue
    fi
    mkdir -p "$(dirname "$out/$rel")"
    if [[ "$(sha256sum < "$src/$rel" | cut -c1-64)" == "$sum" ]]; then
        cp "$src/$rel" "$out/$rel"; echo "SAME-LF   $rel" >> "$report"
    elif [[ "$(sed 's/$/\r/' "$src/$rel" | sha256sum | cut -c1-64)" == "$sum" ]]; then
        sed 's/$/\r/' "$src/$rel" > "$out/$rel"; echo "SAME-CRLF $rel" >> "$report"
    else
        sed 's/$/\r/' "$src/$rel" > "$out/$rel"; echo "DIFFERS   $rel" >> "$report"
    fi
done
mkdir -p "$out/oracle"
cp "$oracle/reference-pm/registry-pm.F90" "$out/oracle/registry-pm.F90"
sort -k2 -o "$report" "$report"
awk '{print $1}' "$report" | sort | uniq -c
