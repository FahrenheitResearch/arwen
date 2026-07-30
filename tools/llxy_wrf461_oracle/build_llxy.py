#!/usr/bin/env python3
"""Build and run the WRF v4.6.1 module_llxy projection oracle (WSL).

Pattern follows tools/noahmp_wrf461_oracle/build_snow.py in the gpuwm
research repo: the build is driven from Python because shell scripts in
a Windows checkout arrive with CRLF terminators and die under WSL bash.
The oracle compiles the PRISTINE pinned source -- no patching -- with
gfortran under WSL Ubuntu (glibc libm), promoted to binary64 with
-fdefault-real-8 so the fixture gates the product's float64 NumPy
transcription directly.

Authorities:
  * WRF v4.6.1 share/module_llxy.F at the pinned gate tree
    (a pinned WRF v4.6.1 checkout; set WRF_TREE) (commit
    d66e442fccc04111067e29274c9f9eaccc3cef28); llij/ijll/set_* for
    Lambert (both hemispheres), Mercator, and polar stereographic.
  * WPS v4.6.0 geogrid/src/process_tile_module.F get_map_factor and
    get_rotang (the code that populates geo_em MAPFAC_* and
    SINALPHA/COSALPHA), transcribed per point inside run_llxy.F90 with
    the source sha pinned below.

Outputs (committed):
  tests/data/llxy_oracle/llxy_wrf461_fixture.csv  -- hex binary64 rows
  tests/data/llxy_oracle/llxy_deck.json           -- the query deck

Usage:  python tools/llxy_wrf461_oracle/build_llxy.py
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FIXTURE_DIR = REPO / "tests" / "data" / "llxy_oracle"

WRF_TREE = os.environ.get("WRF_TREE", "<wrf-v4.6.1>")
WSL_SOURCE = f"{WRF_TREE}/share/module_llxy.F"
#: What the fixture header records: the tree identity is the commit
#: and the sha256 below, never one machine's directory layout.
SOURCE_LABEL = "<wrf-v4.6.1>/share/module_llxy.F"
SOURCE_SHA256 = ("0c777e66ffbb0602479cadcc0ab484769a140c4cb2b3e9a8f757597"
                 "086f0c76b")
WRF_COMMIT = "d66e442fccc04111067e29274c9f9eaccc3cef28"
WPS_PROCESS_TILE = os.environ.get(
    "WPS_TREE", "<wps-v4.6.0>") + "/geogrid/src/process_tile_module.F"
WPS_PROCESS_TILE_SHA256 = ("ef546e2747987948f1aa681566936713fffa6c2cf2542a80"
                           "4540033bf9b479c2")

FFLAGS = ("-cpp -ffree-form -ffree-line-length-none -fdefault-real-8 "
          "-O0 -ffp-contract=off -fno-fast-math -g0")

STUBS = """\
module module_wrf_error
contains
   subroutine wrf_error_fatal(msg)
      character(len=*), intent(in) :: msg
      write (0, '(A,A)') 'wrf_error_fatal: ', trim(msg)
      stop 9
   end subroutine wrf_error_fatal
   subroutine wrf_debug(level, msg)
      integer, intent(in) :: level
      character(len=*), intent(in) :: msg
      if (level < -huge(level)) write (0, '(A)') trim(msg)
   end subroutine wrf_debug
   subroutine wrf_message(msg)
      character(len=*), intent(in) :: msg
      if (len_trim(msg) < 0) write (0, '(A)') trim(msg)
   end subroutine wrf_message
end module module_wrf_error
"""

# --- deck ------------------------------------------------------------
# One entry per projection configuration.  knowni/knownj pin the
# reference point at the WPS default ref_x/ref_y for e_we=111, e_sn=89
# (mass centre), matching the product grid classes' default.
# code: 1 = PROJ_LC, 2 = PROJ_PS, 3 = PROJ_MERC (module_llxy values).
CASES = [
    # id           code ref_lat  ref_lon  stdlon   tl1    tl2
    ("lc_nh_sec",   1,  39.7,    -84.0,   -84.5,   30.0,  60.0),
    ("lc_sh_sec",   1,  -27.5,   153.0,   153.5,   -17.5, -37.5),
    ("lc_sh_tan",   1,  -35.0,   145.0,   145.0,   -35.0, -35.0),
    ("merc_trop",   3,  1.3,     103.8,   103.8,   1.3,   1.3),
    ("merc_fiji",   3,  -17.8,   178.5,   178.5,   -17.8, -17.8),
    ("merc_nh",     3,  23.1,    -82.4,   -82.4,   23.1,  23.1),
    ("ps_nh",       2,  64.8,    -147.7,  -147.7,  64.8,  64.8),
    ("ps_sh",       2,  -77.85,  166.7,   166.7,   -71.0, -71.0),
    ("ps_pole",     2,  90.0,    0.0,     -147.7,  60.0,  60.0),
]
KNOWN_I, KNOWN_J = 55.5, 44.5   # e_we=111, e_sn=89 mass-centre ref
DX = 12000.0

I_QUERIES = [-9.5, 1.0, 27.75, 55.5, 56.0, 83.25, 111.0, 121.5]
J_QUERIES = [-7.5, 1.0, 22.25, 44.5, 45.0, 66.75, 89.0, 97.5]
LAT_OFFSETS = [-12.0, -7.5, -3.0, 0.0, 3.0, 7.5, 12.0]
LON_OFFSETS = [-15.0, -9.0, -4.0, 0.0, 4.0, 9.0, 15.0]
MAPF_OFFSETS = [-12.0, -6.0, 0.0, 6.0, 12.0]
ROTA_OFFSETS = [-179.5, -120.0, -30.0, 0.0, 30.0, 120.0, 179.5]


def _wrap180(lon: float) -> float:
    while lon > 180.0:
        lon -= 360.0
    while lon < -180.0:
        lon += 360.0
    return lon


def build_deck() -> tuple[str, dict]:
    lines: list[str] = []
    deck: dict = {"known_i": KNOWN_I, "known_j": KNOWN_J, "dx": DX,
                  "cases": {}}
    for cid, code, rlat, rlon, stdlon, tl1, tl2 in CASES:
        lines.append(f"CASE {cid} {code} {rlat!r} {rlon!r} {KNOWN_I!r} "
                     f"{KNOWN_J!r} {DX!r} {stdlon!r} {tl1!r} {tl2!r}")
        llij: list[list[float]] = []
        for dlat in LAT_OFFSETS:
            lat = rlat + dlat
            if not -89.5 <= lat <= 89.5:
                continue
            for dlon in LON_OFFSETS:
                lon = _wrap180(rlon + dlon)
                llij.append([lat, lon])
        if code == 2:  # polar stereographic: exact pole query
            llij.append([90.0 if tl1 >= 0.0 else -90.0,
                         _wrap180(stdlon + 90.0)])
        if cid == "merc_fiji":  # antimeridian specials
            for lon in (180.0, -180.0, 179.9, -179.9):
                llij.append([-17.8, lon])
        ijll = [[i, j] for i in I_QUERIES for j in J_QUERIES]
        mapf = []
        for dlat in MAPF_OFFSETS:
            lat = rlat + dlat
            if -89.5 <= lat <= 89.5:
                mapf.append(lat)
        for t in (tl1, tl2):
            if t not in mapf and -89.5 <= t <= 89.5:
                mapf.append(t)
        rota = [_wrap180(stdlon + d) for d in ROTA_OFFSETS]
        for lat, lon in llij:
            lines.append(f"LLIJ {cid} {lat!r} {lon!r}")
        for i, j in ijll:
            lines.append(f"IJLL {cid} {i!r} {j!r}")
        for lat in mapf:
            lines.append(f"MAPF {cid} {lat!r}")
        for lon in rota:
            lines.append(f"ROTA {cid} {lon!r}")
        deck["cases"][cid] = {
            "code": code, "ref_lat": rlat, "ref_lon": rlon,
            "stand_lon": stdlon, "truelat1": tl1, "truelat2": tl2,
            "llij": llij, "ijll": ijll, "mapf": mapf, "rota": rota,
        }
    return "\n".join(lines) + "\n", deck


def wsl(cmd: str) -> str:
    result = subprocess.run(["wsl", "-e", "bash", "-lc", cmd],
                            capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stdout)
        sys.stderr.write(result.stderr)
        raise RuntimeError(f"WSL command failed ({result.returncode}): "
                           f"{cmd[:200]}")
    return result.stdout


def to_wsl(path: Path) -> str:
    text = str(path).replace("\\", "/")
    drive, rest = text.split(":", 1)
    return f"/mnt/{drive.lower()}{rest}"


def main() -> int:
    got = wsl(f"sha256sum {WSL_SOURCE}").split()[0]
    if got != SOURCE_SHA256:
        raise RuntimeError(
            f"pinned module_llxy.F sha mismatch: {got} != {SOURCE_SHA256}")
    got = wsl(f"sha256sum {WPS_PROCESS_TILE}").split()[0]
    if got != WPS_PROCESS_TILE_SHA256:
        raise RuntimeError("pinned process_tile_module.F sha mismatch: "
                           f"{got} != {WPS_PROCESS_TILE_SHA256}")

    deck_text, deck = build_deck()
    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    build = wsl("mktemp -d").strip()
    driver = to_wsl(HERE / "run_llxy.F90")
    stubs_path = HERE / "wrf_stubs.f90.gen"
    deck_path = HERE / "deck.txt.gen"
    stubs_path.write_text(STUBS, newline="\n", encoding="ascii")
    deck_path.write_text(deck_text, newline="\n", encoding="ascii")
    try:
        wsl(f"cp {WSL_SOURCE} {build}/module_llxy.F")
        # CRLF-strip Windows-checkout inputs (build_snow.py pattern).
        wsl(f"tr -d '\\r' < {driver} > {build}/run_llxy.F90")
        wsl(f"tr -d '\\r' < {to_wsl(stubs_path)} > {build}/wrf_stubs.f90")
        wsl(f"tr -d '\\r' < {to_wsl(deck_path)} > {build}/deck.txt")
        wsl(f"cd {build} && gfortran -c {FFLAGS} wrf_stubs.f90 && "
            f"gfortran -c {FFLAGS} module_llxy.F && "
            f"gfortran -c {FFLAGS} run_llxy.F90 && "
            f"gfortran -o run_llxy run_llxy.o module_llxy.o wrf_stubs.o -lm")
        out = wsl(f"cd {build} && ./run_llxy < deck.txt")
        compiler = wsl("gfortran --version | head -1").strip()
        glibc = wsl("ldd --version | head -1").strip()
    finally:
        wsl(f"rm -rf {build}")
        stubs_path.unlink(missing_ok=True)
        deck_path.unlink(missing_ok=True)

    header = "\n".join([
        "# llxy_wrf461_fixture.csv -- WRF v4.6.1 map projection oracle",
        f"# source: {SOURCE_LABEL}",
        f"# source_sha256: {SOURCE_SHA256}",
        f"# wrf_commit: {WRF_COMMIT} (tag v4.6.1)",
        "# map_factor/rotang authority: WPS v4.6.0 "
        "geogrid/src/process_tile_module.F get_map_factor:1735-1876 "
        "get_rotang:1920-2067",
        f"# wps_process_tile_sha256: {WPS_PROCESS_TILE_SHA256}",
        f"# compiler: {compiler}",
        f"# glibc: {glibc}",
        f"# flags: {FFLAGS}",
        "# precision: -fdefault-real-8 promotes the pristine REAL source "
        "to binary64; values are IEEE-754 binary64 hex words",
        "# deck: llxy_deck.json (regenerate with "
        "tools/llxy_wrf461_oracle/build_llxy.py)",
        "# rows: SETUP id code hemi cone rebydx rsw polei polej dlon | "
        "LLIJ id lat lon i j | IJLL id i j lat lon | "
        "MAPF id lat m | ROTA id lon sina cosa",
    ]) + "\n"

    fixture = FIXTURE_DIR / "llxy_wrf461_fixture.csv"
    with open(fixture, "w", newline="\n", encoding="ascii") as stream:
        stream.write(header)
        stream.write(out)
    with open(FIXTURE_DIR / "llxy_deck.json", "w", newline="\n",
              encoding="ascii") as stream:
        json.dump(deck, stream, indent=1)
        stream.write("\n")
    n_rows = sum(1 for line in out.splitlines() if line.strip())
    print(f"wrote {fixture} ({n_rows} rows)")
    print(f"wrote {FIXTURE_DIR / 'llxy_deck.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def fixture_sha256() -> str:
    """Committed-fixture digest helper for provenance tests."""
    return hashlib.sha256(
        (FIXTURE_DIR / "llxy_wrf461_fixture.csv").read_bytes()).hexdigest()
