"""Generate the input deck for the o3input=2 ozone oracle (ozn_extract).

Writes ozn_inputs.txt (see ozn_extract.F90's header for the format).  All
real values are emitted as float32 rendered with 10 significant digits, so
gfortran's list-directed READ (correctly-rounded strtof) and numpy's
float32 parse recover the identical bit pattern on both sides.

Coverage intent:
* latitudes ON the 64-point climatology grid (<= comparisons), between
  grid points, and OUTSIDE the grid in both hemispheres (edge-slope
  extrapolation);
* julian dates on both sides of every date_oz month boundary, mid-month,
  Dec->Jan year wrap, day-366 wrap (MOD(IJUL,365)), and fractional days;
* pressure columns of several shapes/depths (a 49-level column with a
  100 hPa top is one shape among several), columns hitting the
  top-of-data scaling branch, the below-data constant branch, exact
  pin boundary values (including the pmid == pin(1) stale-kupper path),
  and mixed found/not-found column batches.

usage: python ozn_make_fixtures.py OUT_TXT
       (paths on any drive; run from the repo root or via absolute paths)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from gpuwm.ingest.wrf_ozone import load_ozone_climatology  # noqa: E402


def _fmt(v) -> str:
    return f"{float(np.float32(v)):.10g}"


def test_latitudes(lat: np.ndarray) -> list[np.float32]:
    lats: list[np.float32] = []
    # Outside the grid, southern side (edge-slope extrapolation).
    lats += [np.float32(-90.0), np.float32(-89.25)]
    # Exact grid points (y == x(k) resolves to the k-1..k interval).
    for idx in (0, 1, 31, 32, 62, 63):
        lats.append(np.float32(lat[idx]))
    # Between grid points (midpoints in float64, rounded to float32).
    for a, b in ((0, 1), (15, 16), (31, 32), (47, 48), (62, 63)):
        lats.append(np.float32(0.5 * (float(lat[a]) + float(lat[b]))))
    # Generic off-grid values, both hemispheres.
    lats += [np.float32(v) for v in
             (-45.5, -10.0, 0.0, 10.0, 33.3333321, 71.271)]
    # Outside the grid, northern side.
    lats += [np.float32(88.5), np.float32(90.0)]
    return lats


def julian_cases() -> list[np.float32]:
    return [np.float32(v) for v in (
        0.0,        # Jan 1 00Z: intjulian=1.0, Dec-Jan January branch
        0.25,
        14.999,     # just below the np1=2 boundary (intjulian < 16)
        15.0,       # intjulian == 16 == date_oz(1): strict > sends np1=2
        15.5,
        30.5,
        44.0,       # intjulian == 45 == date_oz(2)
        59.7,
        100.25,
        166.0,
        197.5,
        200.123456,  # fine fractional day
        227.9,
        258.3,
        289.1,
        335.5,
        349.0,      # intjulian == 350 == date_oz(12): np1 stays 1 (December)
        355.75,     # December branch
        364.9999,   # last moments of a 365-day year
        365.0,      # intjulian=366 -> IJUL wraps to 1 (day 366, leap year)
        365.25,     # day 366 with fraction
    )]


def _col(bottom: float, top: float, nz: int) -> list[np.float32]:
    """Bottom-up pressure column (Pa), log-spaced, as float32."""
    vals = np.geomspace(bottom, top, nz)
    return [np.float32(v) for v in vals]


def pressure_cases(pin: np.ndarray) -> list[tuple[int, list]]:
    """[(ijul, [column, ...]), ...]; columns all same nz within a case."""
    cases: list[tuple[int, list]] = []
    # Interior-only towers, 49 levels, ~100 hPa top (one shape among
    # several).
    cases.append((1, [
        _col(101325.0, 10000.0, 49),
        _col(98000.0, 10000.0, 49),
        _col(100200.0, 10050.0, 49),
        _col(96500.0, 9990.0, 49),
    ]))
    # One deep column crossing BOTH data bounds: bottom 102000 Pa >
    # pin(59), top 10 Pa < pin(1).
    cases.append((3, [_col(102000.0, 10.0, 64)]))
    # Coarse 20-level towers.
    cases.append((8, [
        _col(100000.0, 5000.0, 20),
        _col(99000.0, 4000.0, 20),
        _col(85000.0, 20000.0, 20),
    ]))
    # Exact pin boundary values.  Column 1 (bottom-up): a level exactly
    # at pin(59), one exactly at pin(31), one exactly at pin(2), one
    # exactly at pin(1) (the stale-kupper else path), one above the data
    # top.  Column 2 is an ordinary tower, so every level of this case
    # runs the mixed found/not-found fall-through.
    exact = [np.float32(101000.0), np.float32(pin[58]), np.float32(pin[30]),
             np.float32(pin[1]), np.float32(pin[0]), np.float32(14.0)]
    plain = [np.float32(v) for v in
             (101000.0, 80000.0, 50000.0, 20000.0, 5000.0, 1000.0)]
    cases.append((12, [exact, plain]))
    # One column entirely above the data top (every level extrapolated)
    # beside a normal one.
    cases.append((17, [
        [np.float32(v) for v in (27.0, 24.0, 20.0, 16.0, 12.0, 8.0, 4.0,
                                 1.0)],
        _col(95000.0, 3000.0, 8),
    ]))
    # Deeper 49-level towers with a 50 hPa top, wider batch.
    cases.append((20, [
        _col(101300.0, 5000.0, 49),
        _col(99500.0, 5000.0, 49),
        _col(97250.0, 5010.0, 49),
        _col(100800.0, 4995.0, 49),
        _col(93000.0, 5002.0, 49),
    ]))
    # Year-wrap julian paired with a 60-level tower to 2000 Pa.
    cases.append((21, [_col(101325.0, 2000.0, 60),
                       _col(98750.0, 2000.0, 60)]))
    return cases


def main() -> None:
    out_txt = Path(sys.argv[1]) if len(sys.argv) > 1 else \
        Path("ozn_inputs.txt")
    climo = load_ozone_climatology()
    lats = test_latitudes(climo.lat)
    juls = julian_cases()
    pcs = pressure_cases(climo.plev)

    lines: list[str] = [str(len(lats))]
    lines += [_fmt(v) for v in lats]
    lines.append(str(len(juls)))
    for j in juls:
        julday = int(np.float32(j)) + 1
        lines.append(f"{julday} {_fmt(j)}")
    lines.append(str(len(pcs)))
    for ijul, cols in pcs:
        ncol = len(cols)
        nz = len(cols[0])
        assert all(len(c) == nz for c in cols)
        assert ncol <= len(lats) and 1 <= ijul <= len(juls)
        lines.append(f"{ncol} {nz} {ijul}")
        for col in cols:
            lines += [_fmt(v) for v in col]
    out_txt.write_text("\n".join(lines) + "\n", encoding="ascii")
    print(f"wrote {out_txt}: {len(lats)} lats, {len(juls)} julian cases, "
          f"{len(pcs)} pressure cases")


if __name__ == "__main__":
    main()
