"""Extract unmodified WRF solar routines and build the scalar referee.

Run on a sprint box: python tools/solar_geometry_oracle/build.py WRF_ROOT OUT.
"""
from pathlib import Path
import hashlib
import json
import re
import subprocess
import sys


def main():
    source, out = map(Path, sys.argv[1:])
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    from tools.solar_geometry_oracle.prepare_versions import PINS
    assert revision in PINS.values(), revision
    out.mkdir(parents=True, exist_ok=True)
    text = (source / 'phys/module_radiation_driver.F').read_text()
    blocks = []
    for name in ('radconst', 'calc_coszen'):
        match = re.search(r'^\s*SUBROUTINE ' + name + r'\s*\(.*?^\s*END SUBROUTINE ' + name + r'\s*$', text, re.M | re.S | re.I)
        if match is None:
            raise RuntimeError(f'WRF routine missing: {name}')
        blocks.append(match.group(0))
    solar = 'MODULE module_wrf_error\nEND MODULE\nMODULE solar_blocks\nCONTAINS\n' + '\n'.join(blocks) + '\nEND MODULE\n'
    (out / 'solar.f90').write_text(solar)
    (out / 'driver.f90').write_text('''program solar_driver
use solar_blocks
implicit none
integer :: u,v,nx,ny
real :: julian,xtime,gmt,declin,solcon,degrad,dpd
real,allocatable :: xlat(:,:),xlon(:,:),coszen(:,:),hrang(:,:)
open(newunit=u,file='in.bin',access='stream',form='unformatted')
read(u) nx,ny,julian,xtime,gmt,degrad,dpd
allocate(xlat(nx,ny),xlon(nx,ny),coszen(nx,ny),hrang(nx,ny))
read(u) xlat,xlon
call radconst(xtime,declin,solcon,julian,degrad,dpd)
call calc_coszen(1,nx,1,ny,1,nx,1,ny,julian,xtime,gmt,declin,degrad,xlon,xlat,coszen,hrang)
open(newunit=v,file='out.bin',access='stream',form='unformatted',status='replace')
write(v) declin,solcon,coszen,hrang
close(u)
close(v)
end program
''')
    cmd = ['gfortran', '-O0', '-ffp-contract=off', '-fcheck=all', '-finit-real=snan', '-ffree-line-length-none', 'solar.f90', 'driver.f90', '-o', 'solar-oracle']
    subprocess.run(cmd, cwd=out, check=True)
    receipt = {'argv': cmd, 'source': str(source), 'source_revision':revision, 'routines_sha256': hashlib.sha256('\n'.join(blocks).encode()).hexdigest(), 'compiler': subprocess.check_output(['gfortran', '--version'], text=True).splitlines()[0], 'scalar_kind':'REAL(4)', 'module_dependencies':['module_wrf_error (empty stub, neither routine calls it)'], 'tables':{}, 'source_ranges':[{'routine':name,'start': text[:re.search(r'^\s*SUBROUTINE '+name+r'\s*\(',text,re.M|re.I).start()].count('\n')+1} for name in ('radconst','calc_coszen')]}
    (out / 'build.json').write_text(json.dumps(receipt, indent=2) + '\n')
    # Small synthetic column fixture, not model history or prepared inputs.
    import numpy as np
    from gpuwm.core.rrtmg_legacy import _DEGRAD, _DPD
    from tempfile import TemporaryDirectory
    f = np.float32
    y, x = np.mgrid[:190, :23]
    lat = (y / 189 * 179.8 - 89.9).astype(f)
    lon = (x / 22 * 360 - 180).astype(f)
    scalars, refs = [], []
    for k, day in enumerate((0.,1.,79.99,80.,80.01,172.4,240.7,355.,365.,365.99)):
        j, xt, g = f(day), f(k*177%1440), f(k*.37)
        payload = np.array([23,190],np.int32).tobytes()+np.array([j,xt,g,_DEGRAD,_DPD],f).tobytes()+lat.tobytes()+lon.tobytes()
        with TemporaryDirectory(dir=out) as wd:
            Path(wd,'in.bin').write_bytes(payload)
            subprocess.run([str((out/'solar-oracle').resolve())],cwd=wd,check=True)
            refs.append(np.frombuffer(Path(wd,'out.bin').read_bytes(),f).copy())
        scalars.append([j,xt,g])
    polar_lat = np.broadcast_to(np.asarray([-90.,90.],f)[:,None],(2,23)).copy()
    polar_lon = np.broadcast_to(lon[:1],(2,23)).copy()
    polar_scalars, polar_refs = [], []
    midnight = (-.001,0.,1439.999,1440.,1440.001,2880.,2880.001,0.,720.,1440.)
    for k,(j,_xt,g) in enumerate(scalars):
        xt = f(midnight[k])
        payload = np.array([23,2],np.int32).tobytes()+np.array([j,xt,g,_DEGRAD,_DPD],f).tobytes()+polar_lat.tobytes()+polar_lon.tobytes()
        with TemporaryDirectory(dir=out) as wd:
            Path(wd,'in.bin').write_bytes(payload)
            subprocess.run([str((out/'solar-oracle').resolve())],cwd=wd,check=True)
            polar_refs.append(np.frombuffer(Path(wd,'out.bin').read_bytes(),f).copy())
        polar_scalars.append([j,xt,g])
    np.savez_compressed(out/'solar_refs.npz',lat=lat,lon=lon,scalars=np.array(scalars,f),ref=np.stack(refs),
                        polar_lat=polar_lat,polar_lon=polar_lon,polar_scalars=np.array(polar_scalars,f),polar_ref=np.stack(polar_refs))


if __name__ == '__main__':
    main()
