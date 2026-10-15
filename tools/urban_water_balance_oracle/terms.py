"""Compile source-extracted release water updates and grade production terms."""
from pathlib import Path
import ctypes
import hashlib
import itertools
import json
import re
import subprocess
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
host_path = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else root / "ucm.so"
host = ctypes.CDLL(str(host_path))
production = host.ucm_host_water_update
production.argtypes = [ctypes.c_float]*6 + [ctypes.c_int]
production.restype = ctypes.c_float
records = {}
for tag,generation in (("v4.7.1",0),("v4.8.0",1)):
    src = root / "sources" / tag / "module_sf_urban.F"
    lines = src.read_text().splitlines()
    active = [(i+1,l.split("!")[0].strip()) for i,l in enumerate(lines) if l.split("!")[0].strip()]
    bodies = []
    anchors = []
    for surface in "RBG":
        name = "DREL" + surface
        start = next(j for j,(i,l) in enumerate(active) if re.match(name+r"\s*=\s*"+name+"P",l,re.I))
        lineno,statement = active[start]
        while statement.rstrip().endswith("&"):
            start += 1
            statement += "\n" + active[start][1]
        nextline = active[start+1][1]
        clamp = nextline if re.match(r"IF.*RAIN.*"+name+"P",nextline,re.I) else ""
        anchors.append({"surface":surface,"line":lineno,"statement":statement,"clamp":clamp})
        bodies.append(f"{statement}\n{clamp}\n")
    fortran = """subroutine update(prior, flux, rain, rhoo, delt, porimp, result) bind(C)
use iso_c_binding
implicit none
integer, parameter :: IMPR=1, IMPB=2, IMPG=3
real(c_float), value :: prior,flux,rain,rhoo,delt
real(c_float), intent(in) :: porimp(3)
real(c_float), intent(out) :: result(3)
real :: rain1,drelr,drelb,drelg,drelrp,drelbp,drelgp,flxhumrp,flxhumbp,flxhumgp
drelrp=prior; drelbp=prior; drelgp=prior
flxhumrp=max(flux,0.); flxhumbp=max(flux,0.); flxhumgp=max(flux,0.)
RAIN1=RAIN*0.001/3600.
""" + "".join(bodies) + "result=[drelr,drelb,drelg]\nend subroutine\n"
    f = root / f"terms-{tag}.F90"
    f.write_text(fortran)
    library = root / f"terms-{tag}.so"
    command = ["gfortran","-O0","-g","-ffp-contract=off","-fcheck=all","-fPIC","-shared","-ffree-line-length-none",str(f),"-o",str(library)]
    subprocess.run(command,check=True)
    oracle = ctypes.CDLL(str(library)).update
    oracle.argtypes = [ctypes.c_float]*5 + [ctypes.c_void_p]*2
    por = np.asarray([0.05,0.1,0.2],np.float32)
    cases, misses = [],[]
    for values in itertools.product((0.,1.e-7,1.e-4,0.01),(-3.e-5,0.,3.e-5),
                 (0.,np.uint32(1).view(np.float32),1.e-8,0.5,4.,100.),(0.7,1.2),(0.,1.,60.,3600.)):
        prior,flux,rain,rho,dt = map(np.float32,values)
        want = np.zeros(3,np.float32)
        oracle(prior,flux,rain,rho,dt,por.ctypes.data,want.ctypes.data)
        got = np.asarray([production(prior,flux,rain,rho,dt,p,generation) for p in por],np.float32)
        index = len(cases)
        bad = np.flatnonzero(got.view(np.uint32)!=want.view(np.uint32))
        for cell in bad:
            misses.append({"case":index,"surface":int(cell),"got":int(got.view(np.uint32)[cell]),"want":int(want.view(np.uint32)[cell])})
        cases.append({"inputs":list(map(float,(prior,flux,rain,rho,dt))),"words":want.view(np.uint32).tolist()})
    records[tag] = {"command":command,"source_sha256":hashlib.sha256(src.read_bytes()).hexdigest(),
                    "anchors":anchors,"cases":len(cases),"words":len(cases)*3,"differing_words":len(misses),
                    "misses":misses,"full_field_sha256":hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest()}
    print(tag,len(cases),"cases",len(misses),"differing words")
(root/("terms-mutant.json" if len(sys.argv) > 2 else "terms.json")).write_text(json.dumps(records,indent=2)+"\n")
raise SystemExit(int(any(r["differing_words"] for r in records.values())))
