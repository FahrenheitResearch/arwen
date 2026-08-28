#!/usr/bin/env python
"""Equivalence gate: the Rust WIF ingest path against the NumPy oracle.

The NumPy implementations in ``gpuwm/ingest/wif_climatology.py`` are the
expressions that were measured against WRF-4.7.1 ``real.exe`` on node-4
(commit 94260bf44).  They are the ORACLE-OF-RECORD, not a fallback: this
harness is the only thing that can say the Rust engine still holds the
numbers that oracle bound.

It runs both engines on the SAME inputs -- a synthetic global dataset
written in the real dataset's own WPS intermediate layout, or a real
``.dat`` when one is given -- and reports max/RMS differences per field.

usage:
  python tools/wif_climatology_equivalence.py [--dat PATH] [--date D]
         [--nx N --ny N --nz N]
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpuwm.ingest import wif_climatology as wif  # noqa: E402

_MONTHS = ("JAN", "FEB", "MAR", "APR", "MAY", "JUN",
           "JUL", "AUG", "SEP", "OCT", "NOV", "DEC")


def _record(payload: bytes) -> bytes:
    head = struct.pack(">i", len(payload))
    return head + payload + head


def _fixed(text: str, width: int) -> bytes:
    return text.encode("ascii")[:width].ljust(width, b" ")


def write_synthetic_dataset(path: Path, nlat: int, nlon: int,
                            nlev: int, noise: float = 0.2) -> None:
    """A global IFV=5 file with the real dataset's geometry and layout.

    The projection records carry the same cylindrical-equidistant
    description the shipped ``QNWFA_QNIFA_SIGMA_MONTHLY.dat`` carries
    (SWCORNER -90/-180, spanning the full globe), so the seam, the poles
    and the 12x nlev record math are all exercised.  Values are smooth
    analytic functions -- the gate measures the ENGINES, and two engines
    disagree on smooth data exactly where they disagree on real data.
    """

    deltalat = 180.0 / (nlat - 1)
    deltalon = 360.0 / nlon
    lat = -90.0 + deltalat * np.arange(nlat)
    lon = -180.0 + deltalon * np.arange(nlon)
    lon2d, lat2d = np.meshgrid(lon, lat)
    rng = np.random.default_rng(20260827)
    with path.open("wb") as handle:
        for month_index, month in enumerate(_MONTHS):
            for level in range(nlev):
                # xlvl descends from 1.0 (surface sigma) so the decoded
                # level order is the file's own, as in the real dataset.
                sigma = 1.0 - level * (1.0 - 0.01) / (nlev - 1)
                pressure = (101325.0 * sigma
                            + 500.0 * np.cos(np.deg2rad(lat2d))).astype(
                                np.float32)
                phase = np.deg2rad(lon2d + 30.0 * month_index)
                shape = (1.0 + 0.5 * np.cos(phase)) * np.exp(-3.0 * (1.0 - sigma))
                qnwfa = (3.0e9 * shape
                         * (1.0 + 0.3 * np.cos(np.deg2rad(lat2d)))
                         ).astype(np.float32)
                qnifa = (5.0e5 * shape
                         * (1.0 + noise * rng.standard_normal(lat2d.shape))
                         ).astype(np.float32)
                for name, values in (("QNWFA", qnwfa), ("QNIFA", qnifa),
                                     ("P_WIF", pressure)):
                    handle.write(_record(struct.pack(">i", 5)))
                    header = b"".join((
                        _fixed("0000-00-00_00:00:00", 24),
                        struct.pack(">f", 0.0),
                        _fixed("synthetic-global", 32),
                        _fixed(f"{name}_{month}", 9),
                        _fixed("kg-1", 25),
                        _fixed("synthetic equivalence fixture", 46),
                        struct.pack(">fiii", float(sigma), nlon, nlat, 0),
                    ))
                    handle.write(_record(header))
                    proj = b"".join((
                        _fixed("SWCORNER", 8),
                        struct.pack(">fffff", -90.0, -180.0,
                                    deltalat, deltalon, 6371.229),
                    ))
                    handle.write(_record(proj))
                    handle.write(_record(struct.pack(">i", 0)))
                    handle.write(_record(
                        values.astype(">f4").tobytes(order="C")))


def model_grid(ny: int, nx: int, nz: int):
    """A regional target grid that crosses the source seam on purpose.

    The longitude band runs 170 E .. 190 E, i.e. straight across the
    dateline, which is exactly where the pre-existing bounded bilinear
    would clamp instead of wrapping.  If the Rust seam handling were
    wrong this is the grid that says so.
    """

    lat = np.linspace(20.0, 55.0, ny)
    lon = np.linspace(170.0, 190.0, nx)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    lon2d, lat2d = np.meshgrid(lon, lat)
    # Dry half-level eta pressure, the target WRF's WIF call receives.
    znw = np.linspace(1.0, 0.0, nz + 1)
    znu = 0.5 * (znw[:-1] + znw[1:])
    p_top = 5000.0
    mu0 = (99000.0 - p_top) + 1500.0 * np.cos(np.deg2rad(lat2d))
    pb = (znu[:, None, None] * mu0[None, :, :] + p_top).astype(np.float32)
    phb = np.empty((nz + 1, ny, nx), dtype=np.float32)
    for k in range(nz + 1):
        phb[k] = np.float32(9.81 * 30.0 * k * (1.0 + 0.02 * k))
    return (lat2d.astype(np.float64), lon2d.astype(np.float64),
            pb, phb)


def report(name: str, rust, numpy_ref) -> dict:
    rust = np.asarray(rust, dtype=np.float64)
    numpy_ref = np.asarray(numpy_ref, dtype=np.float64)
    diff = rust - numpy_ref
    scale = np.maximum(np.abs(numpy_ref), np.finfo(np.float32).tiny)
    ulp = np.abs(diff) / np.maximum(np.spacing(np.abs(numpy_ref).astype(
        np.float32)).astype(np.float64), np.finfo(np.float64).tiny)
    return {
        "field": name,
        "shape": tuple(int(v) for v in rust.shape),
        "ref_max": float(np.max(np.abs(numpy_ref))),
        "maxabs": float(np.max(np.abs(diff))),
        "rms": float(np.sqrt(np.mean(diff ** 2))),
        "rel_max": float(np.max(np.abs(diff) / scale)),
        "max_ulp": float(np.max(ulp)),
        "bit_identical": bool(np.array_equal(
            np.asarray(rust, dtype=np.float32),
            np.asarray(numpy_ref, dtype=np.float32))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dat", type=Path, default=None,
                        help="a real QNWFA_QNIFA_SIGMA_MONTHLY.dat")
    parser.add_argument("--date", default="2026-08-25_12:00:00")
    parser.add_argument("--source-nlat", type=int, default=181)
    parser.add_argument("--source-nlon", type=int, default=288)
    parser.add_argument("--source-nlev", type=int, default=30)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--nx", type=int, default=41)
    parser.add_argument("--nz", type=int, default=49)
    parser.add_argument("--keep", type=Path, default=None)
    # QNIFA carries per-level noise by default so the gate sees a
    # column that is NOT vertically smooth.  --noise 0 isolates how
    # much of the vertical residual is the fixture roughness rather
    # than the two engines.
    parser.add_argument("--noise", type=float, default=0.2)
    args = parser.parse_args()

    if args.dat is not None:
        path = args.dat
        cleanup = None
    else:
        path = args.keep or Path("wif_equivalence_fixture.dat")
        write_synthetic_dataset(path, args.source_nlat, args.source_nlon,
                                args.source_nlev, noise=args.noise)
        cleanup = None if args.keep else path

    try:
        rust_clim = wif.load_wif_climatology(path, backend="rust")
        numpy_clim = wif.load_wif_climatology(path, backend="numpy")

        # Stage 0 -- the DECODE gate.  Two decoders that disagree make
        # every later number meaningless, so this one is bit-exact or the
        # gate stops here.
        decode = []
        for name in ("qnwfa", "qnifa", "pressure"):
            decode.append(report(f"decode:{name}",
                                 getattr(rust_clim, name),
                                 getattr(numpy_clim, name)))
        decode.append(report("decode:latitude", rust_clim.latitude,
                             numpy_clim.latitude))
        decode.append(report("decode:longitude", rust_clim.longitude,
                             numpy_clim.longitude))

        lat2d, lon2d, pb, phb = model_grid(args.ny, args.nx, args.nz)
        rust_fields, rust_receipt = wif.wif_fields_for_grid(
            rust_clim, lat2d, lon2d, args.date, pb, phb, backend="rust")
        numpy_fields, numpy_receipt = wif.wif_fields_for_grid(
            numpy_clim, lat2d, lon2d, args.date, pb, phb, backend="numpy")

        rows = decode + [report(name, rust_fields[name], numpy_fields[name])
                         for name in ("nwfa", "nifa", "nwfa2d", "nifa2d")]

        print(f"source              : {path}")
        print(f"source grid         : "
              f"{numpy_clim.latitude.size}x{numpy_clim.longitude.size}, "
              f"{numpy_clim.pressure.shape[1]} levels x 12 months")
        print(f"target grid         : {args.nz}x{args.ny}x{args.nx}, "
              f"lon band crosses the dateline")
        print(f"valid date          : {args.date}")
        print(f"rust engine         : {rust_receipt['engine_detail']}")
        print()
        header = (f"{'field':18s} {'ref |max|':>13s} {'maxabs':>13s} "
                  f"{'rms':>13s} {'rel max':>11s} {'max ulp':>9s}  bits")
        print(header)
        print("-" * len(header))
        for row in rows:
            print(f"{row['field']:18s} {row['ref_max']:13.6e} "
                  f"{row['maxabs']:13.6e} {row['rms']:13.6e} "
                  f"{row['rel_max']:11.3e} {row['max_ulp']:9.1f}  "
                  f"{'EXACT' if row['bit_identical'] else 'differ'}")
        return 0
    finally:
        if cleanup is not None and cleanup.exists():
            cleanup.unlink()


if __name__ == "__main__":
    raise SystemExit(main())
