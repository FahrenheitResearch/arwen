"""Compare the diagnostics substage with unchanged compiled WRF 4.7.1.

CPU-only by default. The optional GPU replay uses the production kernel
loader and requires both exact-mode environment selectors before import.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np


WRF_SOURCE_SHA256 = "bd177b6b5ba7949cf9e694d7ad654fd9ae2f07d39d85802f0716c5318889a815"
FIELDS = ("T", "QVAPOR", "PH", "PHB", "MU", "MUB", "PB", "ALB",
          "C1H", "C2H", "C3H", "C4H", "C3F", "C4F", "ZNU", "ZNW",
          "DNW", "RDNW", "RDN", "P_TOP")


def build_reference(wrf: Path, out: Path) -> tuple[Path, dict]:
    path = wrf / "dyn_em/module_big_step_utilities_em.F"
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != WRF_SOURCE_SHA256:
        raise ValueError("WRF diagnostics source differs from pinned v4.7.1")
    match = re.search(
        r"(?ims)^SUBROUTINE calc_p_rho_phi\s*\(.*?^END SUBROUTINE calc_p_rho_phi\s*$",
        raw.decode())
    if match is None:
        raise ValueError("unchanged WRF calc_p_rho_phi not found")
    body = match.group()
    mass_raw = (wrf / "frame/libmassv.F").read_text()
    mass_match = re.search(r"(?ims)^\s*subroutine vspow\(.*?^\s*end\s*$", mass_raw)
    if mass_match is None:
        raise ValueError("unchanged WRF REAL vspow not found")
    mass_body = mass_match.group()
    constants = (wrf / "share/module_model_constants.F").read_bytes()
    (out / "module_model_constants.F90").write_bytes(constants)
    source = """#define VPOW vspow
module exact_diagnostics
use module_model_constants
implicit none
integer, parameter :: p_qv=2, param_first_scalar=2
contains
""" + body + """
subroutine diagnostics_c(nx,ny,nz,hypso,wet,mu,mub,t,qv,ph,phb,pb,alb, &
c1,c2,c3h,c4h,c3f,c4f,znu,znw,dnw,rdnw,rdn,ptop,al,p) bind(C)
use iso_c_binding
integer(c_int), value :: nx,ny,nz,hypso,wet
real(c_float) :: mu(nx,ny),mub(nx,ny),t(nx,nz+1,ny),qv(nx,nz+1,ny)
real(c_float) :: ph(nx,nz+1,ny),phb(nx,nz+1,ny),pb(nx,nz+1,ny),alb(nx,nz+1,ny)
real(c_float) :: c1(nz+1),c2(nz+1),c3h(nz+1),c4h(nz+1),c3f(nz+1),c4f(nz+1)
real(c_float) :: znu(nz+1),znw(nz+1),dnw(nz+1),rdnw(nz+1),rdn(nz+1)
real(c_float), value :: ptop
real(c_float) :: al(nx,nz+1,ny),p(nx,nz+1,ny)
real :: moist(nx,nz+1,ny,2),muts(nx,ny)
integer :: n_moist
moist=0.; moist(:,:,:,2)=qv; muts=mub+mu
n_moist=1
if(wet==1) n_moist=2
call calc_p_rho_phi(moist,n_moist,hypso,al,alb,mu,muts,c1,c2,c3h,c4h,c3f,c4f, &
ph,phb,p,pb,t,p0,t0,ptop,znu,znw,dnw,rdnw,rdn,.true.,0, &
1,nx+1,1,ny+1,1,nz+1,1,nx,1,ny,1,nz+1,1,nx,1,ny,1,nz+1)
end subroutine
end module
subroutine wrf_error_fatal(message)
character(*) :: message
error stop 'WRF fatal branch in nonhydrostatic diagnostic oracle'
end subroutine
""" + mass_body + "\n"
    (out / "diagnostics_exact.F90").write_text(source)
    library = out / "diagnostics.so"
    command = ["gfortran", "-cpp", "-O2", "-ffp-contract=off", "-fno-fast-math",
               "-fno-tree-vectorize", "-fcheck=bounds", "-ffree-line-length-none",
               "-fPIC", "-shared", "module_model_constants.F90",
               "diagnostics_exact.F90", "-o", library.name]
    done = subprocess.run(command, cwd=out, text=True, capture_output=True)
    (out / "build.log").write_text(done.stdout + done.stderr)
    if done.returncode:
        raise RuntimeError(f"Fortran build failed, see {out / 'build.log'}")
    return library, {
        "wrf_source_sha256": hashlib.sha256(raw).hexdigest(),
        "routine_sha256": hashlib.sha256(body.encode()).hexdigest(),
        "constants_sha256": hashlib.sha256(constants).hexdigest(),
        "vspow_sha256": hashlib.sha256(mass_body.encode()).hexdigest(),
        "command": command,
        "compiler": subprocess.check_output(["gfortran", "--version"], text=True).splitlines()[0],
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }


def load_input(path: Path, bridge: Path | None) -> dict[str, np.ndarray]:
    from gpuwm.netcdf_bridge import open_dataset
    result = {}
    with open_dataset(path, executable=bridge) as ds:
        for name in FIELDS:
            v = ds.variables[name]
            a = np.asarray(v[...], np.float32)
            if v.dimensions and v.dimensions[0] == "Time":
                a = a[0]
            if a.ndim == 3:
                a = a[:, 60:70, 60:72]
            elif a.ndim == 2:
                a = a[60:70, 60:72]
            result[name] = np.ascontiguousarray(a)
    return result


def native_array(a: np.ndarray, nz: int) -> np.ndarray:
    if a.ndim == 3:
        padded = np.zeros((nz + 1, a.shape[1], a.shape[2]), np.float32)
        padded[:a.shape[0]] = a
        return np.asfortranarray(padded.transpose(2, 0, 1))
    if a.ndim == 2:
        return np.asfortranarray(a.T)
    padded = np.zeros(nz + 1, np.float32)
    padded[:a.size] = a.reshape(-1)
    return padded


def reference(fn, raw: dict, hypso: int, wet: int) -> dict:
    nz, ny, nx = raw["T"].shape
    names = ("MU", "MUB", "T", "QVAPOR", "PH", "PHB", "PB", "ALB",
             "C1H", "C2H", "C3H", "C4H", "C3F", "C4F", "ZNU", "ZNW",
             "DNW", "RDNW", "RDN")
    arrays = [native_array(raw[name], nz) for name in names]
    al = np.full((nx, nz + 1, ny), -987654.0, np.float32, order="F")
    p = al.copy(order="F")
    pointer = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    fn(nx, ny, nz, hypso, wet, *[pointer(a) for a in arrays],
       float(raw["P_TOP"].reshape(-1)[0]), pointer(al), pointer(p))
    al = np.ascontiguousarray(al.transpose(1, 2, 0)[:nz])
    p = np.ascontiguousarray(p.transpose(1, 2, 0)[:nz])
    return {"al": al, "alt": np.add(al, raw["ALB"], dtype=np.float32),
            "p": np.add(p, raw["PB"], dtype=np.float32), "p_perturbation": p}


def gpu(raw: dict, hypso: int, wet: int) -> dict:
    from gpuwm.wrf_exact import DIAGNOSTICS_ENABLED
    if not DIAGNOSTICS_ENABLED:
        raise ValueError("GPU replay requires exact mode with the diagnostics control on, before import")
    import cupy as cp
    from gpuwm.core.diagnostics import update_diagnostics
    nz, ny, nx = raw["T"].shape
    g = lambda a: cp.asarray(a, dtype=cp.float32)
    residual = (np.diff(raw["PHB"].astype(np.float64), axis=0)
                - np.diff(raw["PHB"], axis=0).astype(np.float64)).astype(np.float32)
    dc3 = np.subtract(raw["C3F"][:-1].astype(np.float64), raw["C3F"][1:].astype(np.float64))
    dc4 = np.subtract(raw["C4F"][:-1].astype(np.float64), raw["C4F"][1:].astype(np.float64))
    out = {name: cp.full((nz, ny, nx), -987654.0, np.float32)
           for name in ("p", "al", "alt", "p_perturbation")}
    values = {name.lower(): g(raw[name]) for name in
              ("PHB", "ALB", "PB", "RDNW", "C1H", "C2H", "C3H", "C4H", "C3F", "C4F")}
    values.update(thp=g(raw["T"]), php=g(raw["PH"]), mup=g(raw["MU"]),
                  thb=g(np.full((nz, ny, nx), 300.0, np.float32)),
                  dphb_resid=g(residual), dc3f=g(dc3), dc4f=g(dc4),
                  mub2d=g(raw["MUB"]), qv=g(raw["QVAPOR"]) if wet else None,
                  p_top=np.float32(raw["P_TOP"].reshape(-1)[0]))
    state = SimpleNamespace(**values, **out)
    update_diagnostics(state, hypsometric_opt=hypso)
    cp.cuda.Device().synchronize()
    return {name: a.get() for name, a in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrf-root", type=Path, required=True)
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--bridge", type=Path)
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    library, build = build_reference(args.wrf_root, args.out)
    dll = ctypes.CDLL(str(library.resolve()))
    fn = dll.diagnostics_c
    fn.argtypes = ([ctypes.c_int] * 5 + [ctypes.POINTER(ctypes.c_float)] * 19
                   + [ctypes.c_float] + [ctypes.POINTER(ctypes.c_float)] * 2)
    fn.restype = None
    initial = load_input(args.input, args.bridge)
    results = {}
    all_equal = True
    for probe in ("initial", "nonzero_phi_mass"):
        raw = {n: a.copy() for n, a in initial.items()}
        if probe != "initial":
            nz, ny, nx = raw["T"].shape
            k, j, i = np.indices((nz + 1, ny, nx), dtype=np.float32)
            raw["PH"] = np.asarray(raw["PH"] + (k * 0.003 + j * 0.007 - i * 0.011), np.float32)
            raw["MU"] = np.asarray(raw["MU"] + 3.25, np.float32)
        for hypso in (1, 2):
            for wet in (0, 1):
                key = f"{probe}-hypso{hypso}-moist{wet}"
                expected = reference(fn, raw, hypso, wet)
                np.savez(args.out / f"{key}-reference.npz", **expected)
                row = {"finite": all(bool(np.isfinite(a).all()) for a in expected.values()),
                       "reference_sha256": {n: hashlib.sha256(a.tobytes()).hexdigest() for n, a in expected.items()}}
                if args.gpu:
                    actual = gpu(raw, hypso, wet)
                    row["gpu"] = {}
                    for name, a in actual.items():
                        e = expected[name]
                        unequal = a.view(np.uint32) != e.view(np.uint32)
                        count = int(np.count_nonzero(unequal))
                        all_equal &= count == 0
                        row["gpu"][name] = {"words": int(a.size), "unequal": count,
                            "max_abs": float(np.max(np.abs(a.astype(np.float64) - e.astype(np.float64)))),
                            "sha256": hashlib.sha256(a.tobytes()).hexdigest()}
                results[key] = row
    receipt = {"schema": "wrf-exact-diagnostics-oracle-v1", "build": build,
               "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "fixture": "12x10 full-column native crop plus explicit phi/mass probe",
               "gpu_exercised": args.gpu, "all_gpu_words_equal": all_equal if args.gpu else None,
               "cases": results}
    (args.out / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"cases": len(results), "gpu_exercised": args.gpu,
                      "all_gpu_words_equal": receipt["all_gpu_words_equal"]}))
    if args.gpu and not all_equal:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
