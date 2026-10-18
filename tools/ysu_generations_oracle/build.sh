#!/usr/bin/env bash
# Compile each unmodified release separately, with checked and ordinary flags.
#
#   build.sh SOURCES BUILD [TAG ...]     (default: all four tags)
#
# SOURCES holds one checkout per tag with its MMM physics in place, as
# fetch_sources.sh writes them (v4.7.x take theirs from arch/Externals.cfg,
# not from a submodule).  gate.py needs v4.8.0 and v4.6.1 only.
set -euo pipefail
T=$(cd "$(dirname "$0")/../.." && pwd)
P=$(realpath "$1")
B=$(realpath -m "$2")
shift 2
wanted=" $* "
mkdir -p "$B"
for spec in 'v4.6.1:d66e442fccc04111067e29274c9f9eaccc3cef28:a695daa712591b30fa72b8f24eac003e8e7016531283d29443bcc04ce3ea77d8' \
            'v4.7.0:eeacb821626feb34c62c3c1e169594d886d13adb:b2c9f11f92f967b053948b42a7c5dbaf1b7f72ddb9d35948e96c32a5d91092d2' \
            'v4.7.1:f52c197ed39d12e087d02c50f412d90d418f6186:b2c9f11f92f967b053948b42a7c5dbaf1b7f72ddb9d35948e96c32a5d91092d2' \
            'v4.8.0:06d4240ae989cc3e50af412bb472df3d9048783c:15d76892fc6ffa5c9f8f7ebecfeb6eaa5c23d71b44e8c53a8832f49ff711ea7e'; do
 IFS=: read -r tag pin digest <<< "$spec"
 [[ "$wanted" == "  " || "$wanted" == *" $tag "* ]] || continue
 root=$P/$tag
 test "$(git -C "$root" rev-parse HEAD)" = "$pin"
 source=$root/phys/physics_mmm/bl_ysu.F90
 test "$(sha256sum "$source" | cut -d' ' -f1)" = "$digest"
 if [[ "$tag" == v4.8.0 ]]; then
   test "$(git -C "$root/phys/physics_mmm" rev-parse HEAD)" = 550b5b494758ad2756394e93cb08b5709b8feb83
 elif [[ "$tag" != v4.6.1 ]]; then
   test "$(git -C "$root/phys/physics_mmm" rev-parse HEAD)" = 0ea59b1cd673006ee7a9a9958c533a6a0e354243
 fi
 for mode in scalar checked diagnostics bounded bep_corrected; do
   out=$B/$tag/$mode
   mkdir -p "$out"
   cd "$out"
   flags=(-O2 -fno-tree-vectorize -ffp-contract=off -cpp -ffree-form -ffree-line-length-none)
   if [[ "$mode" == checked || "$mode" == bounded || "$mode" == bep_corrected ]]; then flags+=(-fcheck=all -fbacktrace); fi
   gfortran "${flags[@]}" -DRWORDSIZE=4 -c "$root/phys/ccpp_kind_types.F"
   if [[ "$mode" == bounded || "$mode" == bep_corrected ]]; then
     # Only the explicitly declared undefined shallow-domain path uses this.
     "${PYTHON:-python3}" - "$source" "$mode" <<'PY'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text()
needle = '      do while (zw1d(k) .le. 500.)'
assert s.count(needle) == 1
s = s.replace(needle, '      do while (k .le. kte)\n        if (zw1d(k) .gt. 500.) exit')
for needle in ('do while (zi .eq. 0.)', 'do while (pblh_tke .eq. 0.)'):
    assert s.count(needle) == 1
    s = s.replace(needle, needle[:-1] + ' .and. k .le. kte)')
needle = '      do k = kts,kte\n         thvx_1d(k)'
assert s.count(needle) == 1
s = s.replace(needle, '      tke_ysu(i,kte) = 0.\n' + needle)
if sys.argv[2] == 'bep_corrected':
    needle = 'ad(i,1) = ad(i,1) - bepswitch*frc_urb1d(i)*'
    assert s.count(needle) == 1
    s = s.replace(needle, 'ad(i,1) = ad(i,1) - bepswitch*')
Path('bounded.F90').write_text(s)
PY
     gfortran "${flags[@]}" -c bounded.F90 -o bl_ysu.o
   elif [[ "$mode" == diagnostics ]]; then
     # A separate traced copy, never the numerical referee. Verify its output
     # bytes equal the unmodified referee before accepting any private local.
     "${PYTHON:-python3}" - "$source" <<'PY'
from pathlib import Path
import sys
s = Path(sys.argv[1]).read_text()
needle = '                hgamv(i) = brint*vx(i,1)'
assert s.count(needle) == 1
s = s.replace(needle, needle + '''
                write(*,'(A,5(1X,Z8.8))') 'YSUTD', transfer(tmp1,0), &
                  transfer(ent_eff,0), transfer(radsum,0), &
                  transfer(wstar3_2(i),0), merge(1,0,cloudflg(i))''')
Path('diagnostics.F90').write_text(s)
PY
     gfortran "${flags[@]}" -c diagnostics.F90 -o bl_ysu.o
   else
     gfortran "${flags[@]}" -c "$source" -o bl_ysu.o
   fi
   gfortran "${flags[@]}" -c "$T/tools/ysu_wrf461_oracle/run_bl_ysu_wide.F90"
   gfortran -o oracle ccpp_kind_types.o bl_ysu.o run_bl_ysu_wide.o
   {
     gfortran --version | head -1
     printf 'release=%s commit=%s kind_phys=4 flags=' "$tag" "$pin"
     printf '%q ' "${flags[@]}"
     printf '\n'
     sha256sum "$source" "$root/phys/ccpp_kind_types.F" "$root/phys/module_bl_ysu.F" "$T/tools/ysu_wrf461_oracle/run_bl_ysu_wide.F90"
     nm -u bl_ysu.o
   } > receipt.txt
 done
done
