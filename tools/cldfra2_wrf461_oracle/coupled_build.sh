#!/usr/bin/env bash
# Reuse the established RRTMG oracle harness; add the public cloud driver
# before preparation. Generated drivers stay in the lane's build folder.
set -euo pipefail
root=$(realpath "$1"); out=$(realpath -m "$2")
here=$(cd "$(dirname "$0")" && pwd); old=$here/../rrtmg_wrf461_oracle
bash "$here/build.sh" "$root" "$out/cloud"
(cd "$out/cloud" && gfortran -O0 -ffree-form -ffree-line-length-none -c state_stub.F90 cal_cldfra2.F90)
mkdir -p "$out/lw" "$out/sw" "$out/lw-results"
cp "$old"/lw_{stub_wrf,binio,extract}.F90 "$out/lw/"
cp "$old"/lw_gen_dumpers.py "$out/lw/"
cp "$old"/sw_{stub_wrf,dump_tables,fixture_driver}.F90 "$out/sw/"
python - "$out" <<'PY'
from pathlib import Path
import sys
out=Path(sys.argv[1])
p=out/'lw/lw_extract.F90'; s=p.read_text()
needle='     f_qi_l = (ifqi /= 0)'
assert needle in s
s=s.replace(needle,needle+'''
     if (icloud==2) then
        cldfra_in=0.
        if (ifqc/=0.or.ifqi/=0) call cal_cldfra2(cldfra_in,qc_in,qi_in,ifqc/=0,ifqi/=0, &
           1,2,1,2,1,nz+1,1,1,1,1,1,nz,1,1,1,1,1,nz)
     end if
'''); p.write_text(s)
p=out/'sw/sw_fixture_driver.F90'; s=p.read_text()
s=s.replace('  integer :: iu, ou', '  integer :: iu, ou\n  logical :: driver_fqc, driver_fqi')
s=s.replace('     call read_case()', '     call read_case()')
s=s.replace('          f_qv=.true., f_qc=.true., f_qr=.true.,',
            '          f_qv=.true., f_qc=driver_fqc, f_qr=.true.,')
s=s.replace('          f_qi=.true., f_qs=.true., f_qg=.true.,',
            '          f_qi=driver_fqi, f_qs=.true., f_qg=.true.,')
# The existing replicated-prep harness pins all flags true. False-flag
# cases use the untouched public wrapper directly as their authority.
s=s.replace('     call dump_wrf_outputs()', '     call dump_wrf_outputs()\n     if (.not.driver_fqc.or..not.driver_fqi) cycle')
s=s.replace("    call wi0(trim(cpre)//'in/yr', yr)",
            "    call wi0(trim(cpre)//'in/f_qc', merge(1,0,driver_fqc))\n"
            "    call wi0(trim(cpre)//'in/f_qi', merge(1,0,driver_fqi))\n"
            "    call wi0(trim(cpre)//'in/yr', yr)")

needle='    xlat = xlat0; xlong = xlong0; albedo = albedo0; tsk = tsk0'
assert needle in s
s=s.replace(needle,'''    driver_fqc=btest(mod((caseid-1)/8,4),0)
    driver_fqi=btest(mod((caseid-1)/8,4),1)
    if (icloud_in==2) then
       cldfra3d=0.
       if (driver_fqc.or.driver_fqi) call cal_cldfra2(cldfra3d,qc3d,qi3d,driver_fqc,driver_fqi, &
          ids,ide,jds,jde,kds,kde,ims,ime,jms,jme,kms,kme,its,ite,jts,jte,kts,kte)
    end if
'''+needle); p.write_text(s)
PY
flags='-O0 -cpp -ffree-form -ffree-line-length-none -fconvert=big-endian -fallow-argument-mismatch -DEM_CORE=1 -DWRF_CHEM=0 -DHWRF=0 -DRWORDSIZE=4 -DIWORDSIZE=4 -DDWORDSIZE=8 -DLWORDSIZE=4 -ffp-contract=off'
cd "$out/lw"
python lw_gen_dumpers.py "$root/phys/module_ra_rrtmg_lw.F" lw_dump_state.F90
gfortran -c $flags lw_stub_wrf.F90
gfortran -c $flags -I . "$root/phys/module_ra_rrtmg_lw.F"
gfortran -c $flags lw_binio.F90 lw_dump_state.F90 lw_extract.F90
gfortran -o lw_extract lw_stub_wrf.o module_ra_rrtmg_lw.o ../cloud/cal_cldfra2.o lw_binio.o lw_dump_state.o lw_extract.o
cp "$root/run/RRTMG_LW_DATA" .
cd "$out/sw"
gfortran -c $flags sw_stub_wrf.F90
gfortran -c $flags "$root/phys/module_ra_rrtmg_lw.F" "$root/phys/module_ra_rrtmg_sw.F" sw_fixture_driver.F90
gfortran -o sw_fixture_driver sw_stub_wrf.o module_ra_rrtmg_lw.o module_ra_rrtmg_sw.o ../cloud/cal_cldfra2.o sw_fixture_driver.o
cp "$root/run/RRTMG_SW_DATA" "$root/run/RRTMG_LW_DATA" .
python "$here/make_coupled_inputs.py" "$out"
cd "$out/lw"; ./lw_extract "$out/lw-input.txt" "$out/lw-results"
cd "$out/sw"; ./sw_fixture_driver "$out/sw-input.txt" "$out/sw-results.swd"
sha256sum "$root/phys/module_ra_rrtmg_lw.F" "$root/phys/module_ra_rrtmg_sw.F" "$out/lw-input.txt" "$out/sw-input.txt" > "$out/coupled-sha256sums.txt"
