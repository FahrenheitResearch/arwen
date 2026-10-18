#!/usr/bin/env bash
# Build and run the topo_wind chain oracle against WRF v4.6.1.
#
#   bash tools/topo_wind_wrf461_oracle/build.sh WRF_SOURCE_ROOT BUILD_DIR [OUT_DIR]
#   bash tools/topo_wind_wrf461_oracle/build.sh --sweep NPATCH SEED WRF_SOURCE_ROOT BUILD_DIR OUT_DIR
#
# WRF_SOURCE_ROOT is a WRF v4.6.1 checkout (commit d66e442f); every compiled
# file is checked against SOURCES.sha256 (LF checkout) first.  Compiled, all
# byte-unmodified:
#
#   phys/ccpp_kind_types.F         -DRWORDSIZE=4, WRF's default (real(4))
#   phys/physics_mmm/bl_ysu.F90    YSU with ctopo/ctopo2 present
#   phys/module_bl_shinhong.F      Shin-Hong, which reads them too
#   dyn_em/start_em.F:1539-1626    LAP_HGT, CTOPO, CTOPO2 (cut by extract.py)
#
# YSU and the static block, two builds, one driver (run_topo_wind_chain.F90,
# -O0):
#   wrfflags  WRF's own FCOPTIM from configure.wrf's GNU entry,
#             -O2 -ftree-vectorize -funroll-loops: THE reference, the
#             arithmetic a stock wrf.exe performs (GCC folds x**2. to x*x
#             there, as the YSU wide oracle records);
#   O0        the same at -O0, which calls powf(x, 2.0).
# Shin-Hong (run_shinhong_topo_chain.F90), three builds, as
# tools/shinhong_column_oracle/build.sh builds it:
#   O2s       WRF's flags with the vectoriser off, every libm call scalar:
#             THE Shin-Hong reference;
#   wrfflags  WRF's flags exactly; the vectoriser calls libmvec's
#             _ZGVbN4vv_powf there (recorded beside it, not the reference);
#   O0        -O0.
# The script fails if an object carries an FMA instruction or if a
# reference object imports a libmvec (_ZGV) symbol.
# OUT_DIR (default tests/data/oracles/topo_wind) receives the inputs, every
# build's outputs (gzip -n), and the receipts.
set -euo pipefail

sweep=()
if [[ "${1:-}" == "--sweep" ]]; then
    sweep=(--sweep "$2" "$3"); shift 3
fi
if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: build.sh [--sweep NPATCH SEED] WRF_SOURCE_ROOT BUILD_DIR [OUT_DIR]" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd "${script_dir}/../.." && pwd)
out_dir=$(realpath -m "${3:-${repo_root}/tests/data/oracles/topo_wind}")

( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - ) || {
    echo "a WRF source differs from SOURCES.sha256; refusing to build" >&2
    exit 3
}

mkdir -p "${build_dir}" "${out_dir}"
cd "${build_dir}"
python3 "${script_dir}/make_cases.py" "${sweep[@]}" "${out_dir}"
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}"

ff=(-ffree-form -ffree-line-length-none)
no_fma() {   # no_fma <object>
    if objdump -d "$1" | grep -Eq '(vfmadd|vfmsub|vfnmadd|vfnmsub)'; then
        echo "$1 contains FMA instructions" >&2
        exit 4
    fi
}
build() {   # build <tag> <flags...>
    local tag=$1; shift
    rm -rf "${tag}"; mkdir -p "${tag}"
    (cd "${tag}" &&
     gfortran -c "$@" -cpp -DRWORDSIZE=4 "${ff[@]}" \
         "${source_root}/phys/ccpp_kind_types.F" &&
     gfortran -c "$@" -cpp "${ff[@]}" -I . -o bl_ysu.o \
         "${source_root}/phys/physics_mmm/bl_ysu.F90" &&
     gfortran -c "$@" "${ff[@]}" -I . -o wrf_topo_wind_inline.o \
         "${build_dir}/wrf_topo_wind_inline.F90" &&
     gfortran -c -O0 "${ff[@]}" -I . \
         "${script_dir}/run_topo_wind_chain.F90" &&
     gfortran -o run_topo_wind_chain ccpp_kind_types.o bl_ysu.o \
         wrf_topo_wind_inline.o run_topo_wind_chain.o)
    for obj in bl_ysu.o wrf_topo_wind_inline.o; do
        nm -u "${tag}/${obj}" | sed 's/^ *//' | sort > "${tag}/${obj%.o}.undefined.txt"
        no_fma "${tag}/${obj}"
    done
    # The topo_wind outputs read logf (CTOPO), expf/powf/tanhf (YSU).  A
    # vector form of any of them would move the reference off scalar libm.
    # (The slope loop's (hx**2+hy**2)**.5 may take _ZGVbN4vv_powf at -O2, as
    # in the stock start_em.o; topo_wind never reads the slope.)
    if grep -E '_ZGV' "${tag}/bl_ysu.undefined.txt"; then
        echo "${tag}/bl_ysu.o imports a libmvec symbol" >&2; exit 4
    fi
    if grep -E '_ZGV.*(logf|sqrtf)' "${tag}/wrf_topo_wind_inline.undefined.txt"; then
        echo "${tag}/wrf_topo_wind_inline.o vectorised the CTOPO logf" >&2; exit 4
    fi
}
build_shinhong() {   # build_shinhong <tag> <flags...>
    local tag=$1; shift
    rm -rf "sh-${tag}"; mkdir -p "sh-${tag}"
    (cd "sh-${tag}" &&
     gfortran -c "$@" -cpp "${ff[@]}" -I . -o module_bl_shinhong.o \
         "${source_root}/phys/module_bl_shinhong.F" &&
     gfortran -c -O0 "${ff[@]}" -I . -I ../wrfflags \
         "${script_dir}/run_shinhong_topo_chain.F90" &&
     gfortran -o run_shinhong_topo_chain module_bl_shinhong.o \
         ../wrfflags/wrf_topo_wind_inline.o run_shinhong_topo_chain.o)
    nm -u "sh-${tag}/module_bl_shinhong.o" | sed 's/^ *//' | sort \
        > "sh-${tag}/module_bl_shinhong.undefined.txt"
    no_fma "sh-${tag}/module_bl_shinhong.o"
}
build wrfflags -O2 -ftree-vectorize -funroll-loops
build O0 -O0
build_shinhong O2s -O2 -funroll-loops -fno-tree-vectorize
build_shinhong wrfflags -O2 -ftree-vectorize -funroll-loops
build_shinhong O0 -O0
if grep -q '_ZGV' sh-O2s/module_bl_shinhong.undefined.txt; then
    echo "the Shin-Hong reference object imports a libmvec symbol" >&2; exit 4
fi

patches="${out_dir}/topo-wind-patches.bin"
columns="${out_dir}/topo-wind-columns.bin"
extras="${out_dir}/topo-wind-shinhong.bin"
for tag in wrfflags O0; do
    "./${tag}/run_topo_wind_chain" "${patches}" "${columns}" \
        "${tag}-static.bin" "${tag}-ysu.bin"
done
for tag in O2s wrfflags O0; do
    "./sh-${tag}/run_shinhong_topo_chain" "${patches}" "${columns}" \
        "${extras}" "${tag}-shinhong.bin"
done
static_diff=$(cmp -l wrfflags-static.bin O0-static.bin | wc -l || true)
ysu_diff=$(cmp -l wrfflags-ysu.bin O0-ysu.bin | wc -l || true)
sh_vec_diff=$(cmp -l O2s-shinhong.bin wrfflags-shinhong.bin | wc -l || true)
sh_o0_diff=$(cmp -l O2s-shinhong.bin O0-shinhong.bin | wc -l || true)
for tag in wrfflags O0; do
    for part in static ysu; do
        gzip -n -9 -c "${tag}-${part}.bin" > "${out_dir}/wrf-${tag}-${part}.bin.gz"
    done
done
for tag in O2s wrfflags O0; do
    gzip -n -9 -c "${tag}-shinhong.bin" > "${out_dir}/wrf-${tag}-shinhong.bin.gz"
done
gzip -n -9 -f "${columns}"
gzip -n -9 -f "${extras}"

{
    echo "# gfortran: $(gfortran --version | head -1)"
    echo "# glibc:    $(ldd --version | head -1)"
    echo "# WRF:      v4.6.1 d66e442fccc04111067e29274c9f9eaccc3cef28 (SOURCES.sha256)"
    for tag in wrfflags O0; do
        for obj in bl_ysu wrf_topo_wind_inline; do
            echo "# ${tag} ${obj}.o undefined libm symbols:"
            grep -E 'f$|_ZGV' "${tag}/${obj}.undefined.txt" | grep -Ev '^_gfortran' \
                | sed 's/^/#   /' || true
        done
    done
    for tag in O2s wrfflags O0; do
        echo "# Shin-Hong ${tag} module_bl_shinhong.o undefined libm symbols:"
        grep -E 'f$|_ZGV' "sh-${tag}/module_bl_shinhong.undefined.txt" \
            | grep -Ev '^_gfortran' | sed 's/^/#   /' || true
    done
    echo "# FMA instructions in any object: 0"
    echo "# bytes differing wrfflags vs O0: static ${static_diff}, ysu ${ysu_diff}"
    echo "# Shin-Hong bytes differing from O2s: wrfflags ${sh_vec_diff}, O0 ${sh_o0_diff}"
} > "${out_dir}/build-receipt.txt"
cat "${out_dir}/build-receipt.txt"

(cd "${out_dir}" && sha256sum topo-wind-patches.bin topo-wind-columns.bin.gz \
     topo-wind-shinhong.bin.gz \
     wrf-wrfflags-static.bin.gz wrf-wrfflags-ysu.bin.gz wrf-O0-static.bin.gz \
     wrf-O0-ysu.bin.gz wrf-O2s-shinhong.bin.gz wrf-wrfflags-shinhong.bin.gz \
     wrf-O0-shinhong.bin.gz > oracle-sha256sums.txt)
(cd "${script_dir}" && sha256sum build.sh extract.py make_cases.py \
     run_topo_wind_chain.F90 run_shinhong_topo_chain.F90 SOURCES.sha256 \
     ../ysu_wrf461_oracle/make_wide_columns.py \
     ../terrain_drag_wrf471_oracle/extract.py) \
    | sed "s#  #  tools/topo_wind_wrf461_oracle/#" > "${out_dir}/harness-sha256sums.txt"
echo "topo_wind oracle written to ${out_dir}"
