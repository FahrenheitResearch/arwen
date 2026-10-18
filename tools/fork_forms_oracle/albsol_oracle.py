"""alb_sol = 1: WOOF's radiation-call ALBSOL update against the branch's Fortran.

The operational RAP/HRRR branch (NOAA-EMC/HRRR v4.1.21, WRFV3.9
phys/module_radiation_driver.F) normalises the land albedo for the sun angle
at every radiation call: radconst (:1013-1014), calc_coszen at the interval
midpoint (:1032-1034, no [-1, 1] clamp in this generation), then the ALBSOL
block (:1038-1063) with the MODIS istwe table and d = (0.1, 0.25)
(:781-789).  This oracle extracts those lines and the two routines verbatim
from the pinned file (fetch_fork.py), wraps them in one column subroutine,
and compiles it with gfortran.  The WOOF side is the production function
gpuwm.core.solar_albedo.solar_albedo_radiation_call, the one
physics.PhysicsDriver._run_radiation calls, on the device.

Columns: 128, spanning convective afternoon land in all 21 MODIS classes,
stable night land, snow and sea ice, open ocean, high-latitude and polar
points, the subsolar point (COSZEN at or above 1 in float32), grazing sun on
both sides of the terminator, the 0.9 cap, and the mask edges (XLAND at 1.5,
trace SNOW and XICE).  Four start times, three radiation calls each (the
first copies ALBEDO/ALBBCK, as itimestep == 1 does), state carried between
calls.  Outputs compared word for word: ALBSOL, ALBBCKSOL, COSZEN.

    python albsol_oracle.py build FORK_SRC BUILD_DIR       (CPU)
    python albsol_oracle.py run BUILD_DIR OUT_DIR [--before] (GPU, via the mutex)

``--before`` grades the pre-fix path (host NumPy radconst/calc_coszen fed
to the same update kernel) for the record.  Analysis tooling only.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
F = np.float32
NCOL = 128
RADT = 15.0
STARTS = (datetime(2026, 7, 15, 18, 0), datetime(2026, 1, 10, 6, 0),
          datetime(2026, 3, 20, 12, 0), datetime(2026, 12, 31, 23, 50))
NCALLS = 3
ARMS = {"o0": ["-O0", "-ffp-contract=off"],
        "wrf_o2": ["-O2", "-ftree-vectorize", "-funroll-loops"]}
COMMON = ["-ffree-form", "-ffree-line-length-none"]
SOURCE = "phys/module_radiation_driver.F"


def _lines(text, first, last):
    return "\n".join(text.splitlines()[first - 1:last]) + "\n"


def _routine(text, name):
    m = re.search(rf"(?ims)^ *SUBROUTINE {name}\b.*?^ *END SUBROUTINE {name}[^\n]*\n",
                  text)
    assert m, name
    return m.group(0)


def extract(fork_src: Path) -> str:
    text = (fork_src / SOURCE).read_text(encoding="utf-8")
    declarations = "\n".join(
        line for line in _lines(text, 781, 789).splitlines()
        if not line.lstrip().startswith("!"))
    call_radconst = _lines(text, 1013, 1014)
    call_coszen = _lines(text, 1032, 1034)
    block = _lines(text, 1038, 1063)
    assert "CALL radconst(XTIME,DECLIN,SOLCON,JULIAN" in call_radconst
    assert "call calc_coszen(" in call_coszen and "xtime+radt*0.5" in call_coszen
    assert "albsol(i,j) = albbck(i,j)*dm" in block
    assert "albsol(i,j) = MIN(albsol(i,j),0.9)" in block
    # USE module_wrf_error in radconst reads nothing; the stub module supplies
    # the name.
    return f"""module module_wrf_error
end module module_wrf_error
module fork_albsol_extract
use module_model_constants, only: DEGRAD, DPD
implicit none
contains
{_routine(text, "radconst")}
{_routine(text, "calc_coszen")}
subroutine albsol_call(n, itimestep, alb_sol, julian, xtime, gmt, radt, &
                       xlat, xlong, albedo, albbck, xland, snow, xice, ivgtyp, &
                       albsol, albbcksol, coszen)
integer, intent(in) :: n, itimestep, alb_sol
real, intent(in) :: julian, xtime, gmt, radt
real, intent(in) :: xlat(n,1), xlong(n,1), albedo(n,1), albbck(n,1)
real, intent(in) :: xland(n,1), snow(n,1), xice(n,1)
integer, intent(in) :: ivgtyp(n,1)
real, intent(inout) :: albsol(n,1), albbcksol(n,1)
real, intent(out) :: coszen(n,1)
real :: hrang(n,1), declin, solcon
integer :: i, j, ims, ime, jms, jme, its, ite, jts, jte
{declarations}
ims = 1; ime = n; jms = 1; jme = 1
its = 1; ite = n; jts = 1; jte = 1
{call_radconst}{call_coszen}{block}end subroutine albsol_call
end module fork_albsol_extract
"""


DRIVER = """program albsol_driver
use fork_albsol_extract
implicit none
integer :: hdr(4), n, ncase, ncall, c, k, itimestep
real :: s(3)
real, allocatable :: xlat(:,:), xlong(:,:), albedo(:,:), albbck(:,:), xland(:,:)
real, allocatable :: snow(:,:), xice(:,:), albsol(:,:), albbcksol(:,:), coszen(:,:)
integer, allocatable :: ivgtyp(:,:)
character(len=1024) :: a, b
call get_command_argument(1, a)
call get_command_argument(2, b)
open(11, file=trim(a), access='stream', form='unformatted', status='old')
open(12, file=trim(b), access='stream', form='unformatted', status='replace')
read(11) hdr
n = hdr(1); ncase = hdr(2); ncall = hdr(3)
allocate(xlat(n,1), xlong(n,1), albedo(n,1), albbck(n,1), xland(n,1))
allocate(snow(n,1), xice(n,1), albsol(n,1), albbcksol(n,1), coszen(n,1), ivgtyp(n,1))
read(11) xlat, xlong, albedo, albbck, xland, snow, xice, ivgtyp
do c = 1, ncase
  albsol = -1.0
  albbcksol = -1.0
  do k = 1, ncall
    read(11) s
    itimestep = 1
    if (k > 1) itimestep = 2
    call albsol_call(n, itimestep, 1, s(1), s(2), s(3), %RADT%, xlat, xlong, &
                     albedo, albbck, xland, snow, xice, ivgtyp, albsol, albbcksol, coszen)
    write(12) albsol, albbcksol, coszen
  end do
end do
close(11)
close(12)
end program albsol_driver
"""


def build(fork_src: Path, build_dir: Path) -> dict:
    fork_src, build_dir = fork_src.resolve(), build_dir.resolve()
    sys.path.insert(0, str(HERE))
    from fetch_fork import fetch
    hashes = fetch(fork_src, [SOURCE, "share/module_model_constants.F"])
    build_dir.mkdir(parents=True, exist_ok=True)
    (build_dir / "extract.f90").write_text(extract(fork_src), encoding="utf-8")
    (build_dir / "driver.f90").write_text(
        DRIVER.replace("%RADT%", f"{RADT:.1f}"), encoding="utf-8")
    for arm, flags in ARMS.items():
        b = build_dir / arm
        b.mkdir(exist_ok=True)
        run = lambda *a: subprocess.run(["gfortran", *flags, *COMMON, *a],
                                        cwd=b, check=True)
        run("-c", "-cpp", "-DNMM_CORE=0",
            str(fork_src / "share/module_model_constants.F"))
        run("-c", str(build_dir / "extract.f90"))
        run("-c", str(build_dir / "driver.f90"))
        run("-o", "albsol_driver", "module_model_constants.o", "extract.o",
            "driver.o")
    version = subprocess.run(["gfortran", "--version"], capture_output=True,
                             text=True, check=True).stdout.splitlines()[0]
    receipt = {"sources": hashes, "gfortran": version,
               "lines": {"declarations": [781, 789], "radconst_call": [1013, 1014],
                         "calc_coszen_call": [1032, 1034],
                         "albsol_block": [1038, 1063]},
               "routines": ["radconst", "calc_coszen"],
               "arms": {a: f + COMMON for a, f in ARMS.items()}}
    (build_dir / "receipt.json").write_text(json.dumps(receipt, indent=1) + "\n")
    return receipt


def make_columns(seed: int = 20261007):
    """128 columns: per-column inputs and a regime label."""
    rng = np.random.default_rng(seed)
    cols = []

    def add(regime, lat, lon, veg, albbck, albedo=None, xland=1.0, snow=0.0,
            xice=0.0):
        cols.append(dict(regime=regime, lat=lat, lon=lon, veg=veg,
                         albbck=albbck, albedo=albbck if albedo is None else albedo,
                         xland=xland, snow=snow, xice=xice))

    # Convective afternoon land over CONUS in every MODIS class, twice.
    for k in range(42):
        veg = k % 21 + 1
        add("convective_land", 30.0 + 0.37 * k, -100.0 + 0.9 * k, veg,
            0.08 + 0.015 * (k % 21))
    # Stable night land (same longitudes half a world away).
    for k in range(12):
        add("stable_night", 35.0 + k, 80.0 + 2.0 * k, k % 21 + 1, 0.15)
    # Snow-covered land and trace snow (mask edge).
    for k in range(6):
        add("snow", 45.0 + k, -95.0 - k, 10 + k % 3, 0.2, albedo=0.75, snow=5.0 + k)
    add("edge_trace_snow", 40.0, -90.0, 12, 0.18, snow=1.0e-6)
    add("edge_trace_snow", 41.0, -91.0, 12, 0.18, snow=np.nextafter(F(0), F(1)))
    # Sea ice, full and fractional, and trace ice.
    for k, xi in enumerate((1.0, 0.8, 0.5, 0.2, 0.05, 1.0e-6)):
        add("sea_ice", 70.0 + k, -150.0 + 5 * k, 15, 0.55, albedo=0.65,
            xland=1.0, xice=xi)
    # Open ocean.
    for k in range(10):
        add("ocean", -30.0 + 6 * k, -170.0 + 30 * k, 17, 0.08, xland=2.0)
    # High latitude and the poles.
    for lat in (85.0, 89.9, -85.0, -89.9, 66.5, -66.5):
        add("polar", lat, 10.0 * lat, 16, 0.3)
    # High terrain (Rockies, Tibet, Andes): land; lat/lon only.
    for k, (lat, lon) in enumerate(((39.5, -106.0), (40.6, -111.8), (30.0, 90.0),
                                    (-15.0, -70.0), (46.5, 9.0), (37.0, -118.5))):
        add("high_terrain", lat, lon, (7, 1, 16, 10, 19, 20)[k], 0.2 + 0.03 * k)
    # The 0.9 cap: high background albedo at low sun; ALBEDO above 0.9.
    for k in range(6):
        add("cap", 50.0 + 2 * k, -60.0 - 15 * k, 15 if k % 2 else 16,
            0.7 + 0.04 * k, albedo=0.92 + 0.01 * k)
    # Mask edges on XLAND.
    for x in (np.nextafter(F(1.5), F(0)), F(1.5), np.nextafter(F(1.5), F(2))):
        add("edge_xland", 33.0, -97.0, 12, 0.17, xland=float(x))
    # Fill the remainder with subsolar and terminator columns, whose
    # coordinates depend on the start time and are set per case below.
    while len(cols) < NCOL:
        k = len(cols)
        add("subsolar" if k % 2 else "terminator", 0.0, 0.0, k % 21 + 1,
            0.1 + 0.6 * rng.random())
    return cols


def case_columns(cols, start: datetime):
    """Per-case latitude/longitude: the subsolar and terminator columns are
    placed from the case's own sun (float64 here; the oracle grades what
    both sides compute from these float32 inputs)."""
    import math
    lat = np.array([c["lat"] for c in cols], np.float64)
    lon = np.array([c["lon"] for c in cols], np.float64)
    t = start + timedelta(minutes=RADT * 0.5)
    doy = t.timetuple().tm_yday - 1 + (t.hour + t.minute / 60.0) / 24.0
    decl = math.degrees(math.asin(math.sin(math.radians(23.5)) * math.sin(
        math.radians((360.0 / 365.0) * ((doy - 80.0) if doy >= 80 else (doy + 285.0))))))
    utc = t.hour + t.minute / 60.0
    sublon = (12.0 - utc) * 15.0
    k_sub = k_term = 0
    for i, c in enumerate(cols):
        if c["regime"] == "subsolar":
            lat[i] = decl + 0.02 * (k_sub - 4)
            lon[i] = sublon + 0.05 * (k_sub - 4)
            k_sub += 1
        elif c["regime"] == "terminator":
            lat[i] = 20.0 + 3.0 * k_term
            lon[i] = sublon + (90.0 if k_term % 2 else -90.0) + 0.4 * (k_term - 4)
            k_term += 1
    lon = (lon + 180.0) % 360.0 - 180.0
    return lat.astype(F), lon.astype(F)


def scalars(start: datetime, call: int):
    sys.path.insert(0, str(HERE.parents[1]))
    from gpuwm.core.swint import calendar_scalars
    return calendar_scalars(start, call * RADT * 60.0)


def run_woof(inputs, before=False):
    import cupy as cp
    from gpuwm.core.solar_albedo import (solar_albedo_radiation_call,
                                         update_solar_albedo_cuda)
    out = np.zeros((len(STARTS), NCALLS, 3, NCOL), F)
    for c in range(len(STARTS)):
        lat = cp.asarray(inputs["lat_case"][c].reshape(1, NCOL))
        lon = cp.asarray(inputs["lon_case"][c].reshape(1, NCOL))
        fields = {n: cp.asarray(np.asarray(inputs[n], F).reshape(1, NCOL))
                  for n in ("albedo", "albbck", "xland", "snow", "xice")}
        fields["ivgtyp"] = cp.asarray(np.asarray(inputs["ivgtyp"], np.int32).reshape(1, NCOL))
        fields["albsol"] = cp.full((1, NCOL), -1.0, cp.float32)
        fields["albbcksol"] = cp.full((1, NCOL), -1.0, cp.float32)
        for k in range(NCALLS):
            julian, xtime, gmt = inputs["scalars"][c][k]
            if before:
                # The pre-fix path: NumPy radconst/calc_coszen on the host.
                from gpuwm.core.rrtmg_legacy import radconst, calc_coszen
                declin, _ = radconst(F(julian))
                cosine = calc_coszen(F(julian), F(xtime) + F(RADT) * F(0.5), F(gmt),
                                     inputs["lat_case"][c], inputs["lon_case"][c],
                                     declin)
                cosine = cp.asarray(np.asarray(cosine, F)).reshape(1, NCOL)
                update_solar_albedo_cuda(fields, cosine, initialize=(k == 0))
            else:
                cosine = solar_albedo_radiation_call(
                    fields, lat, lon, F(julian), F(xtime), F(gmt), F(RADT),
                    initialize=(k == 0))
            out[c, k, 0] = cp.asnumpy(fields["albsol"]).reshape(-1)
            out[c, k, 1] = cp.asnumpy(fields["albbcksol"]).reshape(-1)
            out[c, k, 2] = cp.asnumpy(cosine).reshape(-1)
    return out


def inputs_for():
    cols = make_columns()
    inputs = {"regime": [c["regime"] for c in cols],
              "albedo": [c["albedo"] for c in cols],
              "albbck": [c["albbck"] for c in cols],
              "xland": [c["xland"] for c in cols],
              "snow": [c["snow"] for c in cols],
              "xice": [c["xice"] for c in cols],
              "ivgtyp": [c["veg"] for c in cols]}
    lats, lons, sc = [], [], []
    for start in STARTS:
        lat, lon = case_columns(cols, start)
        lats.append(lat)
        lons.append(lon)
        sc.append([tuple(F(v) for v in scalars(start, k)) for k in range(NCALLS)])
    inputs["lat_case"], inputs["lon_case"], inputs["scalars"] = lats, lons, sc
    return inputs


def ulp(a, b):
    from gpuwm.core.fp32_ulp import monotone_fp32_key
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))


def run(build_dir: Path, out_dir: Path, before=False) -> dict:
    import cupy as cp
    build_dir, out_dir = build_dir.resolve(), out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "strict" if os.environ.get("GPUWM_WRF_EXACT") == "1" else "default"
    tag = mode + ("-before" if before else "")
    inputs = inputs_for()
    woof = run_woof(inputs, before)
    report = {"mode": mode, "path": "before_fix_numpy_coszen" if before else "production",
              "columns": NCOL, "cases": [s.isoformat() for s in STARTS],
              "calls_per_case": NCALLS, "radt_minutes": RADT,
              "device": cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
              "regimes": sorted(set(inputs["regime"])), "arms": {}}
    regime = np.array(inputs["regime"])
    for arm in ARMS:
        ref = np.zeros_like(woof)
        for c in range(len(STARTS)):
            sub = dict(inputs, lat=inputs["lat_case"][c], lon=inputs["lon_case"][c])
            sub["scalars"] = [inputs["scalars"][c]]
            ref[c] = _one_case(build_dir, arm, out_dir, sub)
        fields = {}
        for f, name in enumerate(("albsol", "albbcksol", "coszen")):
            a, b = woof[:, :, f], ref[:, :, f]
            d = ulp(a, b)
            bad = d > 0
            entry = {"words": int(a.size), "differing": int(bad.sum()),
                     "max_ulp": int(d.max()),
                     "max_abs": float(np.abs(a.astype(np.float64) - b).max())}
            if bad.any():
                entry["regimes"] = sorted(set(regime[np.nonzero(bad.any(axis=(0, 1)))[0]]))
            fields[name] = entry
        sub_one = woof[:, :, 2] > 1.0
        report["arms"][arm] = {"fields": fields,
                               "coszen_above_one_words": int((ref[:, :, 2] > 1.0).sum()),
                               "woof_coszen_above_one_words": int(sub_one.sum()),
                               "active_words": int(((ref[:, :, 2] > 0)).sum())}
    (out_dir / f"albsol-{tag}.json").write_text(json.dumps(report, indent=1) + "\n")
    o0 = report["arms"]["o0"]["fields"]
    print(f"albsol {tag}: " + ", ".join(
        f"{n} {v['differing']}/{v['words']} max_ulp {v['max_ulp']}" for n, v in o0.items()))
    return report


def _one_case(build_dir, arm, out_dir, sub):
    """Run the Fortran driver on one start time (NCALLS calls)."""
    work = out_dir / "work"
    work.mkdir(exist_ok=True)
    inp = work / "in.bin"
    with inp.open("wb") as fh:
        np.array([NCOL, 1, NCALLS, 0], "<i4").tofile(fh)
        for name in ("lat", "lon", "albedo", "albbck", "xland", "snow", "xice"):
            np.asarray(sub[name], "<f4").tofile(fh)
        np.asarray(sub["ivgtyp"], "<i4").tofile(fh)
        for k in range(NCALLS):
            np.asarray(sub["scalars"][0][k], "<f4").tofile(fh)
    out = work / f"out-{arm}.bin"
    subprocess.run([str(build_dir / arm / "albsol_driver"), str(inp), str(out)],
                   check=True)
    return np.fromfile(out, "<f4").reshape(NCALLS, 3, NCOL)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[0] == "build":
        print(json.dumps(build(Path(argv[1]), Path(argv[2])), indent=1))
        return 0
    if argv[0] == "run":
        rep = run(Path(argv[1]), Path(argv[2]), before="--before" in argv)
        return 1 if any(v["differing"] for v in rep["arms"]["o0"]["fields"].values()) else 0
    raise SystemExit(__doc__)


if __name__ == "__main__":
    sys.exit(main())
