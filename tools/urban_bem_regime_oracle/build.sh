#!/usr/bin/env bash
# Build and run the BEP+BEM regime column oracle (sf_urban_physics = 3) on
# byte-unmodified WRF v4.6.1.
#
#   bash tools/urban_bem_regime_oracle/build.sh WRF_SRC_DIR BUILD_DIR OUT_DIR
#
# WRF_SRC_DIR is what fetch_sources.sh writes.  Every file compiled or read is
# checked against SOURCES.sha256 (and the v4.7.1 control file against
# SOURCES-v471-control.sha256) before anything is built.
#
# Builds, all gfortran -O0 with WRF's own phys/ defines (no contraction, no
# libmvec; the script fails on any _ZGV symbol in a WRF object):
#   ref   THE FIXTURE.
#   snan  -finit-real=snan -finit-integer=-2147483647: a variant whose dump
#         differs from ref read an uninitialised local somewhere it matters.
#   v471  ref with v4.7.1's module_sf_bep_bem.F, whose only change from
#         v4.6.1 is the loop bound of the outdoor-temperature loop
#         (nz_um -> nzu, line 1600).  A control, never a fixture.
#
# Optional environment:
#   BEM_LIBM_SHIM=1  link libm_shim.c ahead of libm: atanf, asinf, acosf,
#                    tanf and log10f become the double routine rounded once
#                    to float.  Used only to attribute a difference between a
#                    C library whose float routines for those five are not
#                    correctly rounded and one whose are.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: build.sh WRF_SRC_DIR BUILD_DIR OUT_DIR" >&2
    exit 2
fi
src=$(realpath "$1")
build=$(realpath -m "$2")
out=$(realpath -m "$3")
here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
stub=$here/../urban_wrf471_oracle/stub_wrf.F90

(cd "$src" && sha256sum --quiet -c "$here/SOURCES.sha256")
have_v471=0
if [[ -f "$src/v471/phys/module_sf_bep_bem.F" ]]; then
    (cd "$src" && sha256sum --quiet -c "$here/SOURCES-v471-control.sha256")
    have_v471=1
fi

mkdir -p "$build" "$out"
defines="-Dwrfmodel -DEM_CORE=1 -DNMM_CORE=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4"
free="-ffree-form -ffree-line-length-none"
shim_obj=()
if [[ "${BEM_LIBM_SHIM:-0}" == 1 ]]; then
    gcc -O2 -fno-builtin -c "$here/libm_shim.c" -o "$build/libm_shim.o"
    shim_obj=("$build/libm_shim.o")
fi

build_one() {   # $1 = tag, $2 = module_sf_bep_bem.F path, rest = extra flags
    local tag=$1 bepbem=$2; shift 2
    mkdir -p "$build/$tag"
    ( cd "$build/$tag"
      fc="gfortran -c -O0 $* -cpp ${defines} ${free} -fallow-argument-mismatch -I."
      gfortran -c -O0 "$@" -cpp ${free} -fallow-argument-mismatch "$stub"
      $fc "$src/share/module_model_constants.F"
      $fc "$src/frame/module_wrf_error.F"
      $fc "$src/phys/module_sf_urban.F"
      $fc "$src/phys/module_bep_bem_helper.F"
      $fc "$src/phys/module_sf_bem.F"
      $fc "$bepbem"
      for o in module_sf_urban module_sf_bem module_sf_bep_bem; do
          nm -u "$o.o" | sed 's/^ *//' | sort > "undefined-O0-$o.txt"
          if grep -q '_ZGV' "undefined-O0-$o.txt"; then
              echo "$tag: -O0 $o.o pulled in a libmvec vector symbol" >&2
              exit 4
          fi
      done
      nm module_sf_bep_bem.o > symbols-module_sf_bep_bem.txt
      grep -q '__module_sf_bep_bem_MOD_bep_bem$' symbols-module_sf_bep_bem.txt || {
          echo "$tag: module_sf_bep_bem.o does not export bep_bem" >&2; exit 6; }
      $fc "$here/run_bep_bem_regime.F90"
      gfortran -o run_bep_bem_regime "${shim_obj[@]}" stub_wrf.o \
          module_model_constants.o module_wrf_error.o module_sf_urban.o \
          module_bep_bem_helper.o module_sf_bem.o module_sf_bep_bem.o \
          run_bep_bem_regime.o )
}
build_one ref "$src/phys/module_sf_bep_bem.F"
build_one snan "$src/phys/module_sf_bep_bem.F" -finit-real=snan -finit-integer=-2147483647
if [[ $have_v471 == 1 ]]; then
    build_one v471 "$src/v471/phys/module_sf_bep_bem.F"
fi

# The two derived tables switch on the arms WRF ships switched off (green
# roof, photovoltaics, irrigation, air-conditioning schedule); the same sed
# lines as tools/urban_wrf471_oracle/run_bep_bem.sh.
sed -e 's/^GR_FLAG:0/GR_FLAG:1/' -e 's/^GR_TYPE: 2/GR_TYPE: 1/' \
    -e 's/^GR_FRAC_ROOF:0,0,0/GR_FRAC_ROOF:0.5,0.3,0.6/' \
    -e 's/^PV_FRAC_ROOF: 0,0,0/PV_FRAC_ROOF: 0.3,0.2,0.4/' \
    -e 's/^IRHO:.*/IRHO:0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,1,1/' \
    -e 's/^TIME_ON: .*/TIME_ON: 8., 0., 6./' -e 's/^TIME_OFF: .*/TIME_OFF: 18., 24., 20./' \
    -e 's/^SW_COND: .*/SW_COND: 1, 0, 1/' \
    "$src/run/URBPARM.TBL" > "$build/URBPARM.gr1pv.TBL"
sed -e 's/^GR_FLAG:0/GR_FLAG:1/' \
    -e 's/^GR_FRAC_ROOF:0,0,0/GR_FRAC_ROOF:0.4,0.8,0.2/' \
    -e 's/^IRHO:.*/IRHO:1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1/' \
    "$src/run/URBPARM.TBL" > "$build/URBPARM.gr2.TBL"
for v in gr1pv gr2; do
    if cmp -s "$src/run/URBPARM.TBL" "$build/URBPARM.$v.TBL"; then
        echo "URBPARM.$v.TBL is the stock table: a sed edit matched nothing" >&2
        exit 3
    fi
done

# NAME  USE_WUDAPT_LCZ  TABLE  SEASON  NSTEPS  DT
variants=(
    "nlcd-summer  0 $src/run/URBPARM.TBL      S  6 60"
    "nlcd-winter  0 $src/run/URBPARM.TBL      W  6 30"
    "lcz-summer   1 $src/run/URBPARM_LCZ.TBL  S  6 60"
    "lcz-winter   1 $src/run/URBPARM_LCZ.TBL  W  6 30"
    "gr1pv-summer 0 $build/URBPARM.gr1pv.TBL  S  6 60"
    "gr2-winter   0 $build/URBPARM.gr2.TBL    W  6 60"
    "long-summer  0 $src/run/URBPARM.TBL      S 90 20"
    "stop-summer  0 $src/run/URBPARM.TBL      S 40 60"
)
run_one() {   # $1 = build tag, $2 = variant line, $3 = destination root
    set -- "$1" $2 "$3"
    local tag=$1 name=$2 lcz=$3 table=$4 season=$5 nsteps=$6 dt=$7 dest=$8
    local tbl=URBPARM.TBL
    [[ $lcz == 1 ]] && tbl=URBPARM_LCZ.TBL
    local d="$dest/$name"
    rm -rf "$d"
    mkdir -p "$d/run" "$d/dump"
    cp "$table" "$d/run/$tbl"
    (cd "$d/run" && "$build/$tag/run_bep_bem_regime" "$d/dump" "$lcz" "$season" \
        "$nsteps" "$dt" "$here/REGIMES.csv" > "$d/stdout.txt" 2>&1) || {
        echo "$tag $name failed" >&2; tail -20 "$d/stdout.txt" >&2; exit 1; }
    rm -rf "$d/run"
    # A Fortran STOP inside BEP_BEM exits 0; the dump then ends with a
    # call's inputs and no outputs, and the run is recorded as stopped.
    if ! grep -q '^final/' "$d/dump/manifest.txt"; then
        echo "$tag $name: WRF stopped: $(grep -m1 -E 'in upward_rad|too small|STOP' "$d/stdout.txt" | xargs)"             >> "$dest/stops.txt"
    fi
}
rm -f "$out/stops.txt" "$build/out-snan/stops.txt" "$build/out-v471/stops.txt"
for line in "${variants[@]}"; do
    run_one ref "$line" "$out"
    run_one snan "$line" "$build/out-snan"
    [[ $have_v471 == 1 ]] && run_one v471 "$line" "$build/out-v471"
done

{
    echo "# gfortran: $(gfortran --version | sed -n 1p)"
    echo "# glibc:    $(ldd --version | sed -n 1p)"
    echo "# libm shim (atanf/asinf/acosf/tanf/log10f double-rounded-once): ${BEM_LIBM_SHIM:-0}"
    echo "# -O0 libm symbols per WRF object:"
    for o in module_sf_urban module_sf_bem module_sf_bep_bem; do
        echo "## $o: $(grep -oE '(expf|powf|logf|log10f|atanf|atan2f|sqrtf|sinf|cosf|tanf|asinf|acosf|sincosf|__powisf2|_ZGV.*)$' \
            "$build/ref/undefined-O0-$o.txt" | xargs)"
    done
    for line in "${variants[@]}"; do
        set -- $line
        if cmp -s "$out/$1/dump/data.bin" "$build/out-snan/$1/dump/data.bin"; then
            echo "$1: snan build identical to ref"
        else
            echo "$1: snan build DIFFERS from ref (an uninitialised read reaches the dump)"
        fi
        if [[ $have_v471 == 1 ]]; then
            if cmp -s "$out/$1/dump/data.bin" "$build/out-v471/$1/dump/data.bin"; then
                echo "$1: v4.7.1 module_sf_bep_bem.F identical to v4.6.1"
            else
                echo "$1: v4.7.1 module_sf_bep_bem.F DIFFERS from v4.6.1"
            fi
        fi
    done
    echo "# WRF stops (Fortran STOP inside BEP_BEM; the engine must stop in the same call):"
    cat "$out/stops.txt" 2>/dev/null || echo "none"
} > "$out/build-report.txt"
gfortran --version | sed -n 1p > "$out/compiler.txt"
ldd --version | sed -n 1p >> "$out/compiler.txt"
( cd "$out" && sha256sum */dump/data.bin > data-sha256sums.txt )
cat "$out/build-report.txt"
echo "BEP+BEM regime oracle written to $out"
