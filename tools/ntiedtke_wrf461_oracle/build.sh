#!/usr/bin/env bash
# Build and run the New Tiedtke (cu_physics = 16) cumulus oracle against the
# byte-unmodified WRF v4.6.1 sources.
#
# THE FILE SET IS THREE FILES.  New Tiedtke has no WRF framework dependency
# at all -- no wrf_error_fatal, no wrf_debug, no registry, no
# module_state_description.  `grep -E '^\s*use '` over the two scheme files
# lists ccpp_kind_types, cu_ntiedtke and cu_ntiedtke_common and nothing else,
# and cu_ntiedtke_common lives inside cu_ntiedtke.F90.  So unlike the
# Grell-Freitas oracle -- which had to pull in module_gfs_machine and
# module_gfs_physcons for its constants -- this harness links the scheme and
# stops.  `nm -u` on the objects is the receipt: nothing but libm and
# libgfortran.
#
# WHY -DRWORDSIZE=4 IS NOT OPTIONAL, and is the single most dangerous line in
# this script.  v4.6.1's phys/ccpp_kind_types.F reads
#
#     #if ( RWORDSIZE == 4 )
#        integer, parameter :: kind_phys = selected_real_kind(6)
#     #else
#        integer, parameter :: kind_phys = selected_real_kind(12)
#     #endif
#
# cpp evaluates an UNDEFINED identifier as 0, and 0 == 4 is false, so a build
# that forgets the define takes the DOUBLE branch.  It compiles clean, it
# links clean, and it writes a double-precision oracle that looks entirely
# plausible -- against which a correct float32 port would fail every bitwise
# gate, and the failure would read as a porting bug.  WRF itself sets this
# from configure.wrf (RWORDSIZE = 4 on any default, non-`-r8` build), which
# is why the file never looks dangerous in situ.
#
# Note this file is one of the few that is NOT byte-identical between v4.6.1
# and v4.8.0: 4.8.0 respells the guard as `#ifndef DOUBLE_PRECISION`.  Both
# resolve to single on a default build, so the ANSWER is the same, but the
# digest is not and the define that reaches it is not.  Pin the 4.6.1 bytes.
#
# run_cu_ntiedtke.F90 refuses to run if kind(1.0_kind_phys) /= 4, so the
# define is belt AND braces.
#
# libmvec: same guard as the Grell-Freitas, YSU and Shin-Hong oracles.  The
# reference is built at -O0 and this script FAILS if any _ZGV* symbol appears
# in an -O0 object.
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: build.sh WRF_SOURCE_ROOT BUILD_DIR" >&2
    exit 2
fi

source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# --- the pinned WRF v4.6.1 New Tiedtke file set -----------------------------
# Paths are relative to WRF_SOURCE_ROOT.  Note the scheme itself lives under
# phys/physics_mmm/ -- the CCPP/MPAS-shared physics directory -- and NOT in
# phys/ beside module_cu_ntiedtke.F.  That split is present in v4.6.1; it is
# not a later reorganisation, and the 533-line phys/module_cu_ntiedtke.F is a
# thin driver, not the scheme.  Do not size this port from the wrapper.
#
# Digests verified 2026-08-28 two ways: read from this workstation's WRF tree
# and independently fetched from wrf-model/WRF at tag v4.6.1.  The two scheme
# files are byte-identical between v4.6.1 and v4.8.0, so a screening run on a
# 4.8.0 build exercises the same physics as this parity target.
declare -A pinned=(
  ["phys/physics_mmm/cu_ntiedtke.F90"]="e762101f04d4acd2d19047a92d0b7cd4e244df930f9f9ef7aabae54bfe9a9fd1"
  ["phys/module_cu_ntiedtke.F"]="447406d1550d4095e4e6f129ee74a7ec0ccdebd21383ba0cc6fe3d282ac58d2f"
  ["phys/ccpp_kind_types.F"]="a76e1e5b52fc7cd40be9ccb506fde28b1ef2486b812d518f7f1c882766d484db"
  # PINNED 2026-08-29, when it became load-bearing.  The adapter
  # (gpuwm/core/ntiedtke.py NewTiedtke) transcribes this file's forcing
  # fold -- RTHFTEN = (RTHFTEN + RTHRATEN + RTHBLTEN) * pi, RQVFTEN =
  # RQVFTEN + RQVBLTEN, :879-880 -- and an unpinned source for a
  # transcribed line is exactly what the other three pins exist to
  # prevent (review).
  #
  # It is NOT byte-identical between v4.6.1 and v4.8.0, unlike the two
  # scheme files: 4.8.0 renames GFSCHEME to GFLSCHEME and swaps
  # module_cu_gf_wrfdrv for module_cu_gfl.  The FOLD BLOCK is identical
  # in both -- diffed, not assumed -- but the file is not, so reading the
  # fold off this workstation's 4.8.0 tree was luck rather than method.
  #
  # QUOTED 2026-08-29 from the file that verifies against this digest, so
  # the transcription rests on a quotation rather than on an inference
  # about which version was read (asked for by review, whose point was
  # that the conclusion had survived while the reasoning under it moved):
  #
  #   :867  if(cu_physics == G3SCHEME .or. cu_physics == NTIEDTKESCHEME) then
  #   ...
  #   :879        RTHFTEN(i,k,j)=(RTHFTEN(i,k,j)+RTHRATEN(i,k,j) &
  #   :880                       +RTHBLTEN(i,k,j))*pi(i,k,j)
  #   :881        RQVFTEN(i,k,j)=RQVFTEN(i,k,j)+RQVBLTEN(i,k,j)
  #
  # The `* pi` is there, on the theta lane only.  The guard names exactly
  # two schemes and Kain-Fritsch is not one of them, which is the whole
  # reason CUMULUS_ADVECTIVE_FORCING_SCHEMES is {3, 16} and excludes 1.
  # Note v4.6.1 spells it G3SCHEME where 4.8.0 says GFLSCHEME -- the
  # rename above -- so this quotation could not have been taken off the
  # 4.8.0 tree without the same version question reopening.
  #
  # PROVENANCE, corrected: this digest is not an upstream fetch. It is
  # ~/WRF+L on this box, which is a v4.6.1 tree and matches three of the
  # four pins outright. Earlier searches looked only at ~/WRF (4.8.0) and
  # concluded no v4.6.1 source was present.
  ["phys/module_cumulus_driver.F"]="d73eb4b670599e0bbf98a3661b5dbf17413aed01139d18fa1a84161578dc31e3"
)

for rel in "${!pinned[@]}"; do
    src="${source_root}/${rel}"
    if [[ ! -f "${src}" ]]; then
        echo "missing ${rel} under ${source_root}" >&2
        exit 3
    fi
    actual=$(sha256sum "${src}" | awk '{print $1}')
    if [[ "${actual}" != "${pinned[$rel]}" ]]; then
        echo "${rel} is ${actual}," >&2
        echo "   not the pinned ${pinned[$rel]}" >&2
        if [[ "${rel}" == "phys/ccpp_kind_types.F" ]]; then
            echo "   (v4.8.0 spells this file differently -- it gates on" >&2
            echo "    DOUBLE_PRECISION, not RWORDSIZE.  This oracle pins" >&2
            echo "    the v4.6.1 bytes; point it at a v4.6.1 tree.)" >&2
        fi
        exit 3
    fi
done

mkdir -p "${build_dir}"
cd "${build_dir}"

# --- the reference build: -O0, no vectoriser --------------------------------
# Compile order is the USE order:
#   ccpp_kind_types -> cu_ntiedtke (carries cu_ntiedtke_common in the same
#   file) -> module_cu_ntiedtke -> nt_cases -> the program.
#
# All three WRF files are FREE FORM despite two of them carrying the .F
# extension gfortran reads as fixed form, hence -ffree-form throughout.
gfortran -c -O0 -cpp -DRWORDSIZE=4 -ffree-form -ffree-line-length-none \
    -o ccpp_kind_types_O0.o "${source_root}/phys/ccpp_kind_types.F"

gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" \
    -o cu_ntiedtke_O0.o "${source_root}/phys/physics_mmm/cu_ntiedtke.F90"

gfortran -c -O0 -cpp -ffree-form -ffree-line-length-none \
    -I "${build_dir}" \
    -o module_cu_ntiedtke_O0.o "${source_root}/phys/module_cu_ntiedtke.F"

gfortran -c -O0 -ffree-form -ffree-line-length-none \
    -I "${build_dir}" "${script_dir}/nt_cases.F90"

# --- reach the private routines WITHOUT editing pinned source ---------------
# cumastrn / cuinin / cutypen / cuadjtqn are PRIVATE to module cu_ntiedtke,
# and gfortran gives private module procedures LOCAL symbol binding, so a
# plain external declaration will not link.  Making them public would mean
# editing cu_ntiedtke.F90 and breaking the sha256 pin this oracle rests on.
#
# objcopy --globalize-symbol flips a binding bit in the ELF symbol table and
# touches no instruction.  ASSERTED here every run, not trusted: .text must
# be byte-identical across the operation or the build stops.
# DRIVEN BY THE SET, NOT BY NAMING ONE MEMBER.  This was two hand-written
# copies -- one per object -- and two copies is how a set goes stale: the
# SECOND object was globalized before it had an assertion, because the
# assertion named the first object explicitly (caught by review, review).
# A third object later inherits the assertion instead of needing someone to
# remember it.
globalize_objects=(
  # object                      symbols to globalize
  "cu_ntiedtke|__cu_ntiedtke_MOD_cumastrn __cu_ntiedtke_MOD_cuinin __cu_ntiedtke_MOD_cutypen __cu_ntiedtke_MOD_cuascn __cu_ntiedtke_MOD_cuadjtqn __cu_ntiedtke_MOD_cubasmcn __cu_ntiedtke_MOD_cuentrn __cu_ntiedtke_MOD_cudlfsn __cu_ntiedtke_MOD_cuddrafn __cu_ntiedtke_MOD_cuflxn __cu_ntiedtke_MOD_cudtdqn __cu_ntiedtke_MOD_cududvn"
  # cu_ntiedtke_post_run forms rthcuten/rqvcuten/rqccuten/rqicuten/
  # rucuten/rvcuten/raincv/pratec -- every graded field of nt-levels.csv
  # that cu_ntiedtke_run does not produce.
  # cu_ntiedtke_pre_run does the WRF -> scheme flip and the _hv
  # slicing; the post_run harness needs it to reach cu_ntiedtke_run
  # with the same inputs the driver would hand it.  Adding it here is
  # one line BECAUSE the loop exists -- which is the loop paying for
  # itself one commit after it was written.
  "module_cu_ntiedtke|__module_cu_ntiedtke_MOD_cu_ntiedtke_post_run __module_cu_ntiedtke_MOD_cu_ntiedtke_pre_run"
)

# The objects everything downstream must use, accumulated by the loop.
# It was a hand-written list in THREE places -- the link line, the libmvec
# guard, and the undefined-symbol receipt -- and all three were wrong the
# moment a second object joined the set: they still named
# module_cu_ntiedtke_O0.o, so the newly globalized symbols were not
# reachable at all and the two guards were inspecting an object the
# fixtures do not link.  Same fix as the .text assertion above, and this is
# the third instance of it in this one block: derive from the set, never
# restate it.
globalized_link_objs=()
: > nt-globalize-receipt.txt
echo "# objcopy --globalize-symbol receipt" >> nt-globalize-receipt.txt
for entry in "${globalize_objects[@]}"; do
    obj="${entry%%|*}"
    syms="${entry#*|}"
    flags=()
    for sym in ${syms}; do flags+=(--globalize-symbol="${sym}"); done

    objcopy --dump-section ".text=${obj}-text-before.bin"         "${obj}_O0.o" /dev/null
    objcopy "${flags[@]}" "${obj}_O0.o" "${obj}_glob.o"
    objcopy --dump-section ".text=${obj}-text-after.bin"         "${obj}_glob.o" /dev/null

    if ! cmp -s "${obj}-text-before.bin" "${obj}-text-after.bin"; then
        echo "FATAL: globalize changed ${obj}'s .text; the object is no" >&2
        echo "       longer the compilation of the pinned source." >&2
        exit 8
    fi
    {
      echo "${obj}:"
      echo "  text sha256 before: $(sha256sum "${obj}-text-before.bin" | awk '{print $1}')"
      echo "  text sha256 after : $(sha256sum "${obj}-text-after.bin"  | awk '{print $1}')"
      echo -n "  differing bytes in whole object (symbol table only): "
      # cmp -l exits 1 when the files differ, which is exactly what is
      # expected here (the symbol table changed).  Guard it or set -e
      # with pipefail kills the build at the receipt.
      { cmp -l "${obj}_O0.o" "${obj}_glob.o" || true; } | wc -l
    } >> nt-globalize-receipt.txt
    globalized_link_objs+=("${obj}_glob.o")
done


for prog in run_cu_ntiedtke run_nt_prep run_nt_cuinin run_nt_cumastrn; do
    gfortran -c -O0 -ffree-form -ffree-line-length-none \
        -I "${build_dir}" "${script_dir}/${prog}.F90"
    gfortran -o "${prog}" \
        ccpp_kind_types_O0.o "${globalized_link_objs[@]}" \
        nt_cases.o "${prog}.o"
done

# --- receipts: the precision claim, checked on the built artefact -----------
# Not on the source.  The claim is about what the compiler DID, and the only
# honest place to read that is the object.  kind_phys = selected_real_kind(6)
# makes every real in the scheme 4 bytes; if the double branch had been taken
# the module file would say 8 and this grep would fail.
if ! gfortran -c -O0 -cpp -DRWORDSIZE=4 -ffree-form \
        -o /dev/null "${source_root}/phys/ccpp_kind_types.F" 2>/dev/null; then
    echo "ccpp_kind_types.F will not compile with -DRWORDSIZE=4" >&2
    exit 4
fi

# --- libmvec guard ----------------------------------------------------------
for obj in ccpp_kind_types_O0.o "${globalized_link_objs[@]}"; do
    if nm -u "${obj}" 2>/dev/null | grep -q '_ZGV'; then
        echo "FATAL: ${obj} carries a libmvec vector-math symbol at -O0" >&2
        nm -u "${obj}" | grep '_ZGV' >&2
        exit 5
    fi
done

# What the objects actually need.  New Tiedtke should reference nothing but
# libm and libgfortran; anything else means the file set above is incomplete.
{
  echo "# undefined symbols in the -O0 reference objects"
  for obj in ccpp_kind_types_O0.o "${globalized_link_objs[@]}"; do
      echo "## ${obj}"
      nm -u "${obj}" 2>/dev/null || true
  done
} > nt-undefined-symbols.txt

gfortran --version | head -1 > compiler.txt

# --- run --------------------------------------------------------------------
./run_cu_ntiedtke
# The prep replication proves itself: it reruns the same columns through the
# public cu_ntiedtke_run and compares its own post_run against the real
# driver bitwise.  It exits nonzero on any differing word.
./run_nt_prep
./run_nt_cuinin
./run_nt_cumastrn

# --- the isolation claim ----------------------------------------------------
# cu_ntiedtke.F90 has no horizontal coupling: no (jl+/-1) access anywhere and
# no reduction over the jl dimension.  So a column packed with 17 others must
# answer bitwise identically to the same column run alone, on every row.  The
# fixture measures it rather than asserting it, and the build fails if the
# claim is ever false -- which is what would happen if a future WRF made any
# part of this scheme horizontally aware.
bad=$(awk -F, 'NR > 1 && $3 != 0' nt-isolation.csv | wc -l)
if [[ "${bad}" -ne 0 ]]; then
    echo "FATAL: ${bad} isolation rows differ; New Tiedtke is not" >&2
    echo "       column-independent on this tree." >&2
    awk -F, 'NR > 1 && $3 != 0' nt-isolation.csv >&2
    exit 6
fi

sha256sum nt-*.csv compiler.txt nt-undefined-symbols.txt \
    > oracle-sha256sums.txt

echo
echo "built and ran in ${build_dir}"
echo "  nt-levels.csv     $(wc -l < nt-levels.csv) rows"
echo "  nt-surface.csv    $(wc -l < nt-surface.csv) rows"
echo "  nt-isolation.csv  $(wc -l < nt-isolation.csv) rows, all zero"
echo "  nt-prep-levels.csv       $(wc -l < nt-prep-levels.csv) rows"
echo "  nt-prep-surface.csv      $(wc -l < nt-prep-surface.csv) rows"
echo "  nt-prep-consistency.csv  $(wc -l < nt-prep-consistency.csv) rows, all zero"
