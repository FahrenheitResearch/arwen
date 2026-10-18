#!/usr/bin/env bash
# Rebuild the WRF-side chem column oracles from a staged WRF tree and compare
# every word with the fixtures the parity tests use.
#
#   bash tools/chem_wrf461_verify/rebuild_oracles.sh STAGED_ROOT OUT [REL ...]
#
# STAGED_ROOT comes from stage_wrf_tree.sh.  The builders run from a scratch
# copy of tools/chem_wrf471_oracle under OUT (the repository copy is never
# touched).  Changes made to that scratch copy, and nothing else:
#   * gocart/run_gocart_optics.F90 is left out: the optics oracle line-range
#     extracts from module_optical_averaging.F, which WRF changed between
#     4.6.1 and 4.7.1; build_optics461.sh handles it.
#   * gocart/run_gocart_niwfa.F90 is left out when STAGED_ROOT has no
#     gsl-ccpp-physics/ tree: it is NOAA GSL's Thompson code, not WRF's.
#   * each REL named on the command line has its pin replaced by the staged
#     file's own hash.  Use it only for a file whose difference has been read
#     and shown not to reach compiled code (written to OUT/pin-overrides.txt).
# Then:
#   * gocart/build_sulfur.sh: pins chemics_init.F, checks the committed solar
#     include against it, then runs the shared gocart/build.sh;
#   * build.sh once per lane, each from its own staged harness folder holding
#     only that lane's lists, stubs and drivers.  build.sh compiles every list
#     it can see and the lanes' stubs define the same Registry modules
#     differently, so the lanes' own scripts (build_fire_oracles.sh,
#     build_plume_oracles.sh) stage it the same way;
#   * wesely/build.sh;
#   * compare_fixtures.py against tests/data/oracles/chem.
set -euo pipefail
root=$(realpath "${1:?usage: rebuild_oracles.sh STAGED_ROOT OUT [REL ...]}")
out=$(realpath -m "${2:?}")
shift 2
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
tree=$(cd "$here/../.." && pwd)
mkdir -p "$out"
work="$out/harness"
rm -rf "$work"
cp -a "$tree/tools/chem_wrf471_oracle" "$work"
rm -f "$work/gocart/run_gocart_optics.F90"
[[ -d "$root/gsl-ccpp-physics" ]] || rm -f "$work/gocart/run_gocart_niwfa.F90"
: > "$out/pin-overrides.txt"
for rel in "$@"; do
    new=$(sha256sum "$root/$rel" | cut -c1-64)
    for pin in "$work"/SOURCES*.sha256 "$work"/gocart/SOURCES*.sha256 "$work"/wesely/SOURCES.sha256; do
        if grep -qE "[ *]${rel//./\\.}\$" "$pin"; then
            old=$(grep -E "[ *]${rel//./\\.}\$" "$pin" | cut -c1-64)
            sed -i "s#^${old}\([ *]\+${rel//./\\.}\)\$#${new}\1#" "$pin"
            echo "$rel $(basename "$pin") $old -> $new" >> "$out/pin-overrides.txt"
        fi
    done
done
{
    echo "gfortran: $(gfortran --version | sed -n 1p)"
    echo "glibc:    $(ldd --version | sed -n 1p)"
    echo "root:     $root"
} > "$out/toolchain.txt"
rm -rf "$out/gocart-build" "$out/inventory-build" "$out/wesely-build" "$out/fixtures-gocart" "$out"/lane-*
: > "$out/failures.txt"

bash "$work/gocart/build_sulfur.sh" "$root" "$out/gocart-build" "$out/fixtures-gocart" \
    > "$out/gocart-build.log" 2>&1 || echo "gocart: build failed (gocart-build.log)" >> "$out/failures.txt"

# lane | lists | extra stubs | drivers
lanes=(
  "mono|extract-mono.list||run_advect_mono"
  "vertmx|extract-vertmx.list sources-vertmx.list||run_chem_prep run_vertmx"
  "bdy|extract-bdy.list||run_flowdep_bdy"
  "pm|extract-pm.list sources-pm.list||run_sum_pm"
  "smoke|extract-smoke.list sources-smoke.list|stub_smoke.F90|run_wetdep_ls run_plumerise_wrfchem"
)
for spec in "${lanes[@]}"; do
    IFS='|' read -r lane lists stubs drivers <<< "$spec"
    h="$out/lane-$lane/harness"
    mkdir -p "$h"
    cp "$work"/build.sh "$work"/stub_wrf.F90 "$work"/oracle_io.F90 "$work"/libmvec_positive_control.F90 \
       "$work"/SOURCES*.sha256 "$work"/*.inc "$h/"
    for f in $lists $stubs; do cp "$work/$f" "$h/"; done
    for d in $drivers; do cp "$work/$d.F90" "$h/"; done
    # shellcheck disable=SC2086
    bash "$h/build.sh" "$root" "$out/lane-$lane/build" $drivers > "$out/lane-$lane.log" 2>&1 ||
        echo "lane $lane: build failed (lane-$lane.log)" >> "$out/failures.txt"
done
# The published WRF-Chem plume fixture is the observed build's output: the
# original solver's words plus write-only trace arrays (build_plume_oracles.sh).
h="$out/lane-smoke/harness"
cp "$work"/plume_trace.F90 "$work"/plume_poison_wrf.F90 "$work"/build_plume_observed.py "$h/"
python3 "$work/build_plume_observed.py" wrf "$root" "$out/lane-smoke/build" "$h"     >> "$out/lane-smoke.log" 2>&1 || echo "plume observed build failed (lane-smoke.log)" >> "$out/failures.txt"
# The emission inventory oracle has its own builder (no run_gocart_ driver).
bash "$work/gocart/build_inventory.sh" "$root" "$out/inventory-build" "$out/fixtures-gocart/inventory"     > "$out/inventory-build.log" 2>&1 || echo "inventory: build failed (inventory-build.log)" >> "$out/failures.txt"

bash "$work/wesely/build.sh" "$root" "$out/wesely-build" > "$out/wesely-build.log" 2>&1 ||
    echo "wesely: build failed (wesely-build.log)" >> "$out/failures.txt"

cat "$out/failures.txt"
python3 "$here/compare_fixtures.py" "$out" "$tree/tests/data/oracles/chem" > "$out/compare.json"
python3 - "$out/compare.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
for k, v in d["components"].items():
    print(f"{k:26s} cases {v['cases']:4d} files {v['files']:5d} identical {v['identical']:5d} "
          f"differ {v['differ']:3d} missing {v['missing']:3d} max_ulp {v['max_ulp']}")
print("TOTAL", d["total"])
PY
