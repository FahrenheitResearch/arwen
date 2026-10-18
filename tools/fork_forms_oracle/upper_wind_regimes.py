"""upper_wind_limiter_form = noaa_wrf39 on regime columns.

Under damp_opt = 3 the operational RAP/HRRR branch's advance_w (NOAA-EMC/
HRRR v4.1.21 WRFV3.9 dyn_em/module_small_step_em.F:1705-1736) damps the
saved horizontal winds of the top layers when the top three levels exceed
110 m/s: qq = max(|V|^2)/110^2, then down the zdamp layer qq =
sqrt((qq-1)*qq) and u, v times 1 - (dts/86400)*2*qq*2**(0.4*(k-k_end)).
The block is compiled verbatim (comments dropped, as
tools/wrf_diffusion_oracle/upper_wind_fork_fixture.py does) inside a
driver with the grid read from the stream, and WOOF's production launcher
``gpuwm.core.acoustic.prepare_upper_wind_limiter`` runs the same substeps
on the device.  Every u_save and v_save word is compared.

Grid: 16 x 10 columns, 50 levels to 20 km (HRRR's vertical extent),
zdamp 5000 m (HRRR), dts 5 s.  Regimes per column: jet cores well above
110 m/s, the threshold itself and its float32 neighbours, sub-threshold
jets, calm stable columns, high terrain (3-D base geopotential), ocean;
specified (one-point frame) and unforced domains; 1 to 4 substeps.

    python upper_wind_regimes.py build FORK_SRC BUILD_DIR   (CPU)
    python upper_wind_regimes.py run BUILD_DIR OUT_DIR       (GPU)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
F = np.float32
NX, NY, NZ = 16, 10, 50
SOURCE = "dyn_em/module_small_step_em.F"

HEAD = """program oracle
implicit none
integer :: nx, ny, nz, hdr(4)
real, allocatable :: u_save(:,:,:), v_save(:,:,:), ph_1(:,:,:), phb(:,:,:)
real, allocatable :: w(:,:,:), w_save(:,:,:), mut(:,:), qq(:,:), dampwt(:)
real :: rlthresh, factor, htop, hk, hbot, hdepth, dampmag, pi, dts, rhdr(2)
real, parameter :: g=9.81
integer :: i, j, k, i_start, i_end, k_end, zone, substeps, step
character(512) :: a, b
call get_command_argument(1, a)
call get_command_argument(2, b)
open(10, file=trim(a), access='stream', form='unformatted', status='old')
read(10) hdr, rhdr
nx = hdr(1); ny = hdr(2); nz = hdr(3); zone = hdr(4)
dts = rhdr(1); hdepth = rhdr(2)
allocate(u_save(0:nx,1:nz,0:ny-1), v_save(0:nx-1,1:nz,0:ny))
allocate(ph_1(0:nx-1,1:nz+1,0:ny-1), phb(0:nx-1,1:nz+1,0:ny-1))
allocate(w(0:nx-1,1:nz+1,0:ny-1), w_save(0:nx-1,1:nz+1,0:ny-1))
allocate(mut(0:nx-1,0:ny-1), qq(0:nx-1,0:ny-1), dampwt(1:nz+1))
read(10) substeps
read(10) u_save, v_save, ph_1, phb
close(10)
i_start = zone; i_end = nx-1-zone; k_end = nz
pi = 4.*atan(1.); dampmag = dts*0.2
w = 0.; w_save = 0.; mut = 90000.
do step = 1, substeps
do j = zone, ny-1-zone
"""
FOOT = """enddo
enddo
open(11, file=trim(b), access='stream', form='unformatted', status='replace')
write(11) u_save, v_save
close(11)
end program
"""


def build(fork_src: Path, build_dir: Path) -> dict:
    import hashlib
    fork_src, build_dir = fork_src.resolve(), build_dir.resolve()
    sys.path.insert(0, str(HERE))
    from fetch_fork import fetch
    hashes = fetch(fork_src, [SOURCE])
    lines = (fork_src / SOURCE).read_text(encoding="utf-8").splitlines()
    block = "\n".join(s for s in lines[1704:1736] if not s.lstrip().startswith("!"))
    assert "qq(i,j)=sqrt((qq(i,j)-1.)*qq(i,j))" in block
    assert "rlthresh = 110." in block
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "oracle.f90").write_text(HEAD + block + "\n" + FOOT, encoding="utf-8")
    flags = ["-O0", "-ffp-contract=off", "-fcheck=bounds", "-ffree-line-length-none"]
    subprocess.run(["gfortran", *flags, "oracle.f90", "-o", "oracle"],
                   cwd=build_dir, check=True)
    meta = {"sources": hashes, "source_lines": [1705, 1736],
            "source_slice_sha256": hashlib.sha256(
                ("\n".join(lines[1704:1736]) + "\n").encode()).hexdigest(),
            "flags": flags,
            "compiler": subprocess.check_output(["gfortran", "--version"],
                                                text=True).splitlines()[0]}
    (build_dir / "build.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def _heights():
    """Interface heights (m), stretched to 20 km over 50 layers."""
    eta = np.linspace(0.0, 1.0, NZ + 1)
    return 20000.0 * (0.6 * eta + 0.4 * eta ** 2)


def _regime(i, j):
    return ("jet_core", "threshold", "subthreshold", "calm_stable",
            "high_terrain", "ocean", "jet_core_extreme", "threshold")[(i + 3 * j) % 8]


def make_case(seed, specified, substeps, terrain3d):
    rng = np.random.default_rng(seed)
    z = _heights()
    zm = 0.5 * (z[1:] + z[:-1]) / 20000.0                  # mass levels, 0..1
    u = np.zeros((NZ, NY, NX + 1), F)
    v = np.zeros((NZ, NY + 1, NX), F)
    regimes = []
    for j in range(NY):
        for i in range(NX):
            reg = _regime(i, j)
            regimes.append(reg)
            shape = np.exp(-((zm - 0.93) / 0.12) ** 2)
            if reg == "jet_core":
                peak = 115.0 + 15.0 * rng.random()
            elif reg == "jet_core_extreme":
                peak = 160.0 + 60.0 * rng.random()
            elif reg == "subthreshold":
                peak = 80.0 + 25.0 * rng.random()
            elif reg == "calm_stable":
                peak = 5.0 + 5.0 * rng.random()
            elif reg == "threshold":
                peak = 110.0
            else:
                peak = 90.0 + 40.0 * rng.random()
            spd = (peak * shape + 3.0).astype(F)
            ang = 0.3 + 0.2 * rng.random()
            uu = (spd * np.cos(ang)).astype(F)
            vv = (spd * np.sin(ang)).astype(F)
            if reg == "threshold":
                # |V| exactly 110 at the top level, and its two float32
                # neighbours on alternate columns (qq = 1 and just above).
                top = F(110.0)
                if (i + j) % 3 == 1:
                    top = np.nextafter(top, F(200))
                elif (i + j) % 3 == 2:
                    top = np.nextafter(top, F(0))
                uu[-1], vv[-1] = top, F(0.0)
            u[:, j, i] = uu
            v[:, j, i] = vv
    # Faces not owned by a mass column keep plausible values too.
    u[:, :, NX] = u[:, :, NX - 1]
    v[:, NY, :] = v[:, NY - 1, :]
    php = (rng.standard_normal((NZ + 1, NY, NX)) * 30.0).astype(F)
    php[0] = 0.0
    base = (9.81 * z).astype(F)
    if terrain3d:
        hill = 2500.0 * np.exp(-((np.arange(NX)[None, :] - NX / 2) ** 2) / 10.0) \
            * np.ones((NY, 1))
        frac = (1.0 - z / z[-1])[:, None, None]
        phb = (9.81 * (z[:, None, None] + hill[None] * frac)).astype(F)
    else:
        phb = np.broadcast_to(base[:, None, None], (NZ + 1, NY, NX)).copy()
    return dict(u=u, v=v, php=php, phb=phb, phb1d=base, zone=int(specified),
                substeps=substeps, dts=F(5.0), zdamp=F(5000.0), regimes=regimes,
                terrain3d=terrain3d)


def run_fortran(build_dir, case, work):
    fk = lambda a: np.ascontiguousarray(a.transpose(1, 0, 2))
    inp = work / "case.bin"
    with inp.open("wb") as fh:
        np.array([NX, NY, NZ, case["zone"]], "<i4").tofile(fh)
        np.array([case["dts"], case["zdamp"]], "<f4").tofile(fh)
        np.array([case["substeps"]], "<i4").tofile(fh)
        for a in (case["u"], case["v"], case["php"], case["phb"]):
            fk(a).astype("<f4").tofile(fh)
    out = work / "out.bin"
    subprocess.run([str(build_dir / "oracle"), str(inp), str(out)], check=True)
    data = np.fromfile(out, "<f4")
    nu = case["u"].size
    uo = data[:nu].reshape(NY, NZ, NX + 1).transpose(1, 0, 2)
    vo = data[nu:].reshape(NY + 1, NZ, NX).transpose(1, 0, 2)
    return np.ascontiguousarray(uo), np.ascontiguousarray(vo)


def run_woof(case):
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core.acoustic import prepare_upper_wind_limiter
    cfg = RunConfig(nx=NX, ny=NY, nz=NZ, dx=3000., dy=3000., ztop=20000.,
                    dt=20., run_seconds=60., damp_opt=3, zdamp=float(case["zdamp"]),
                    specified=bool(case["zone"]), upper_wind_limiter_form="noaa_wrf39")
    state = SimpleNamespace(u=cp.asarray(case["u"]), v=cp.asarray(case["v"]),
                            php=cp.asarray(case["php"]))
    if case["terrain3d"]:
        state.phb = cp.asarray(case["phb"])
        state.thb = state.phb
    else:
        state.phb = cp.asarray(case["phb1d"])
        state.thb = cp.zeros(NZ, cp.float32)
    launch = prepare_upper_wind_limiter(state, cfg, float(case["dts"]))
    for _ in range(case["substeps"]):
        launch()
    cp.cuda.Device().synchronize()
    return state.u.get(), state.v.get()


def run(build_dir: Path, out_dir: Path) -> dict:
    import cupy as cp
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    build_dir, out_dir = build_dir.resolve(), out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default"
    rows = []
    seed = 0
    for specified in (True, False):
        for terrain3d in (False, True):
            for substeps in (1, 2, 4):
                seed += 1
                case = make_case(seed, specified, substeps, terrain3d)
                uw, vw = run_fortran(build_dir, case, out_dir)
                ug, vg = run_woof(case)
                changed = int((uw.view(np.uint32) != case["u"].view(np.uint32)).sum()
                              + (vw.view(np.uint32) != case["v"].view(np.uint32)).sum())
                du = ug.view(np.uint32) != uw.view(np.uint32)
                dv = vg.view(np.uint32) != vw.view(np.uint32)
                rows.append({"specified": specified, "terrain3d": terrain3d,
                             "substeps": substeps, "columns": NX * NY,
                             "words": int(ug.size + vg.size),
                             "words_the_limiter_changed": changed,
                             "different_words": int(du.sum() + dv.sum()),
                             "max_ulp": int(max(fp32_ulp_distance(ug, uw).max(initial=0),
                                                fp32_ulp_distance(vg, vw).max(initial=0)))})
    summary = {"mode": mode, "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
               "grid": [NX, NY, NZ], "cases": len(rows),
               "columns": sum(r["columns"] for r in rows),
               "words": sum(r["words"] for r in rows),
               "words_the_limiter_changed": sum(r["words_the_limiter_changed"] for r in rows),
               "different_words": sum(r["different_words"] for r in rows),
               "max_ulp": max(r["max_ulp"] for r in rows),
               "regimes": sorted(set(make_case(1, True, 1, False)["regimes"])),
               "rows": rows, "build": json.loads((build_dir / "build.json").read_text())}
    (out_dir / f"upper_wind-{mode}.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(f"upper_wind {mode}: cases={summary['cases']} words={summary['words']} "
          f"changed_by_limiter={summary['words_the_limiter_changed']} "
          f"different={summary['different_words']} max_ulp={summary['max_ulp']}")
    return summary


if __name__ == "__main__":
    if sys.argv[1] == "build":
        print(json.dumps(build(Path(sys.argv[2]), Path(sys.argv[3])), indent=1))
    else:
        r = run(Path(sys.argv[2]), Path(sys.argv[3]))
        sys.exit(1 if r["different_words"] else 0)
