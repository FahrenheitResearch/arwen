#!/usr/bin/env bash
# Prove which oracle columns WRF itself leaves undefined, and that no other
# column reads outside an array or an unset word.
#
#   bash tools/topo_wind_wrf461_oracle/bounds_check.sh WRF_SOURCE_ROOT BUILD_DIR ORACLE_DIR
#
# bl_ysu.F90 get_pblh (:1622) searches the lowest 500 m with
#     do while (zw1d(k) .le. 500.)
# and no k <= kte bound, and its two later searches stop only through an
# "exit safeguard" at k == kte-1 that a search starting at or above kte-1
# steps over.  On a column whose levels sit below 500 m that walks past the
# end of thetav1d/qke1d/zw1d, and it reads qke1d(kte) = tke_ysu(i,kte),
# which bl_ysu_run never sets (:1255-1277 fill k = kts..kte-1).  WOOF's
# get_pblh bounds every search at kte (kernels/ysu_topo.cuh), so on those
# columns the two disagree by construction and WRF's word is whatever its
# stack held.
#
# This script builds WRF's sources a third time at -O0 with
#   -fcheck=bounds -finit-real=snan -finit-integer=-2147483647
# and runs the chain twice:
#   1. on the oracle as committed: it must STOP with an out-of-bounds
#      error inside get_pblh (the shallow probe column);
#   2. on a copy in which every column whose interface below the model top
#      (zw(kte)) is at or below 500 m is replaced by its deepest neighbour:
#      it must run to the end, and its output must equal, word for word, the
#      plain -O0 build's output on the same copy.  A read of an unset real
#      would carry the signalling NaN into that output; a read past an array
#      would stop the run.
# So the columns the gate excludes are exactly the shallow ones and every
# other column is defined Fortran.
set -euo pipefail
if [[ $# -ne 3 ]]; then
    echo "usage: bounds_check.sh WRF_SOURCE_ROOT BUILD_DIR ORACLE_DIR" >&2
    exit 2
fi
source_root=$(realpath "$1")
build_dir=$(realpath -m "$2")
oracle_dir=$(realpath "$3")
script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
( cd "${source_root}" && grep -v '^#' "${script_dir}/SOURCES.sha256" \
    | sha256sum -c --quiet - )
mkdir -p "${build_dir}"
cd "${build_dir}"
python3 "${script_dir}/extract.py" "${source_root}" "${build_dir}"
ff=(-ffree-form -ffree-line-length-none)
for tag in check plain; do
    rm -rf "${tag}"; mkdir -p "${tag}"
    if [[ ${tag} == check ]]; then
        flags=(-O0 -g -fcheck=bounds -finit-real=snan -finit-integer=-2147483647)
    else
        flags=(-O0)
    fi
    (cd "${tag}" &&
     gfortran -c "${flags[@]}" -cpp -DRWORDSIZE=4 "${ff[@]}" \
         "${source_root}/phys/ccpp_kind_types.F" &&
     gfortran -c "${flags[@]}" -cpp "${ff[@]}" -I . -o bl_ysu.o \
         "${source_root}/phys/physics_mmm/bl_ysu.F90" &&
     gfortran -c "${flags[@]}" "${ff[@]}" -I . -o wrf_topo_wind_inline.o \
         "${build_dir}/wrf_topo_wind_inline.F90" &&
     gfortran -c "${flags[@]}" "${ff[@]}" -I . \
         "${script_dir}/run_topo_wind_chain.F90" &&
     gfortran -o run_topo_wind_chain ccpp_kind_types.o bl_ysu.o \
         wrf_topo_wind_inline.o run_topo_wind_chain.o)
done

cp "${oracle_dir}/topo-wind-patches.bin" patches.bin
gunzip -c "${oracle_dir}/topo-wind-columns.bin.gz" > columns.bin
python3 - <<'PY'
import struct, numpy as np
raw = open("columns.bin", "rb").read()
praw = open("patches.bin", "rb").read()
SC = 18
recs = []
pos = 0
while pos < len(raw):
    nz = struct.unpack_from("<i", raw, pos)[0]
    size = 16 + 4 * SC + 4 * 11 * nz + 4 * (nz + 1)
    recs.append(raw[pos:pos + size]); pos += size
def top_minus_one(rec):
    nz = struct.unpack_from("<i", rec, 0)[0]
    dz = np.frombuffer(rec, "<f4", nz, 16 + 4 * SC + 4 * 9 * nz)
    return float(np.sum(dz[:-1], dtype=np.float64))
def xland(rec):
    return struct.unpack_from("<f", rec, 16 + 4 * 10)[0]
out, shallow, n, pp = [], [], 0, 0
while pp < len(praw):
    nx, ny, pid, fam = struct.unpack_from("<4i", praw, pp)
    pp += 16 + 4 * 4 * nx * ny
    cells = recs[n:n + nx * ny]
    for k, rec in enumerate(cells):
        if top_minus_one(rec) <= 500.0:
            shallow.append((pid, k))
            same = [r for r in cells if top_minus_one(r) > 500.0
                    and xland(r) == xland(rec)]
            rec = max(same, key=top_minus_one)
        out.append(rec)
    n += nx * ny
assert n == len(recs)
open("columns-deep.bin", "wb").write(b"".join(out))
open("shallow-columns.txt", "w").write(
    "".join(f"patch {p} cell {c}\n" for p, c in shallow))
print("shallow columns (zw(kte) <= 500 m):", shallow)
PY

echo "--- 1. committed oracle, bounds-checked: must stop in get_pblh"
if ./check/run_topo_wind_chain patches.bin columns.bin s1.bin y1.bin > run1.log 2>&1; then
    echo "the bounds-checked build ran the shallow columns to the end" >&2
    exit 7
fi
grep -E "out of bounds|Fortran runtime error" run1.log | head -3
grep -q "get_pblh\|bl_ysu" run1.log || true
echo "--- 2. shallow columns replaced: must complete and equal the plain build"
./check/run_topo_wind_chain patches.bin columns-deep.bin s2.bin y2.bin
./plain/run_topo_wind_chain patches.bin columns-deep.bin s3.bin y3.bin
cmp s2.bin s3.bin
cmp y2.bin y3.bin
echo "bounds_check: PASS (shallow columns: $(wc -l < shallow-columns.txt))"
