"""Column set for the ra=90 (analytic clear-sky) oracle.

Writes two files into the output directory:

  columns.npz   the WOOF-side inputs (geography, lowest-level theta,
                pressure, qv, surface albedo, region labels) and the
                calendar cases
  cz_in.bin     the same geography and calendar cases in the stream layout
                run_coszen.F90 reads

usage: python make_columns.py OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

F = np.float32

# (label, regime, lat, lon, theta K, pressure Pa, qv kg/kg, albedo)
_NAMED = [
    # convective: hot, humid boundary layers
    ("southern-plains", "convective", 35.5, -97.5, 305.0, 97000.0, 0.016, 0.18),
    ("florida", "convective", 27.8, -81.7, 303.0, 101200.0, 0.019, 0.15),
    ("amazon", "convective", -3.1, -60.0, 301.0, 100800.0, 0.020, 0.13),
    ("central-india", "convective", 22.0, 79.0, 304.0, 99500.0, 0.021, 0.17),
    ("sahel", "convective", 13.5, 2.1, 312.0, 97500.0, 0.012, 0.30),
    ("ganges-delta", "convective", 23.7, 90.4, 302.0, 100700.0, 0.022, 0.16),
    ("congo-basin", "convective", -0.5, 22.0, 300.0, 96000.0, 0.019, 0.12),
    ("yangtze-delta", "convective", 31.2, 121.5, 303.0, 100600.0, 0.020, 0.16),
    # stable night: cold, dry continental air
    ("great-plains-winter", "stable", 38.5, -98.0, 268.0, 96500.0, 0.0018, 0.20),
    ("mongolian-steppe", "stable", 47.9, 106.9, 255.0, 86000.0, 0.0006, 0.25),
    ("po-valley-fog", "stable", 45.0, 10.0, 274.0, 101800.0, 0.0040, 0.18),
    ("great-basin", "stable", 40.5, -111.5, 265.0, 85000.0, 0.0015, 0.55),
    ("canadian-prairie", "stable", 52.0, -106.0, 250.0, 94000.0, 0.0005, 0.70),
    ("ohio-valley", "stable", 39.5, -84.0, 271.0, 99000.0, 0.0030, 0.18),
    ("scandinavia", "stable", 60.0, 15.0, 262.0, 99000.0, 0.0012, 0.60),
    ("pannonian-basin", "stable", 47.0, 19.5, 266.0, 100500.0, 0.0020, 0.22),
    # snow and ice
    ("greenland-summit", "snow-ice", 72.6, -38.5, 240.0, 68000.0, 0.0003, 0.85),
    ("dome-c", "snow-ice", -75.1, 123.3, 220.0, 64500.0, 0.00005, 0.82),
    ("south-pole", "snow-ice", -90.0, 0.0, 225.0, 68000.0, 0.00004, 0.83),
    ("arctic-pack-ice", "snow-ice", 85.0, -150.0, 248.0, 101000.0, 0.0004, 0.80),
    ("east-siberia", "snow-ice", 62.0, 129.7, 233.0, 101500.0, 0.0002, 0.78),
    ("alpine-glacier", "snow-ice", 46.5, 8.0, 268.0, 70000.0, 0.0025, 0.65),
    ("hudson-bay-ice", "snow-ice", 60.0, -85.0, 245.0, 102000.0, 0.0003, 0.75),
    ("ross-ice-shelf", "snow-ice", -80.0, -170.0, 240.0, 99000.0, 0.0002, 0.80),
    # ocean
    ("equatorial-pacific", "ocean", 0.0, -140.0, 300.0, 101000.0, 0.018, 0.06),
    ("gulf-stream", "ocean", 38.0, -70.0, 295.0, 101500.0, 0.013, 0.07),
    ("north-atlantic", "ocean", 55.0, -30.0, 283.0, 98000.0, 0.007, 0.08),
    ("southern-ocean", "ocean", -55.0, 60.0, 278.0, 98500.0, 0.004, 0.08),
    ("arabian-sea", "ocean", 15.0, 65.0, 302.0, 100800.0, 0.021, 0.06),
    ("south-indian-ocean", "ocean", -20.0, 80.0, 298.0, 101800.0, 0.014, 0.06),
    ("north-pacific", "ocean", 40.0, 170.0, 288.0, 101200.0, 0.009, 0.07),
    ("caribbean", "ocean", 15.0, -75.0, 301.0, 101300.0, 0.018, 0.06),
    ("mediterranean", "ocean", 35.0, 18.0, 296.0, 101600.0, 0.012, 0.07),
    ("bay-of-bengal", "ocean", 15.0, 88.0, 302.0, 100500.0, 0.021, 0.06),
    # high terrain (low surface pressure)
    ("tibetan-plateau", "high-terrain", 29.65, 91.1, 290.0, 65000.0, 0.006, 0.22),
    ("bolivian-andes", "high-terrain", -16.5, -68.15, 285.0, 64000.0, 0.005, 0.18),
    ("colorado-rockies", "high-terrain", 39.7, -105.9, 280.0, 70000.0, 0.004, 0.20),
    ("ethiopian-highlands", "high-terrain", 9.0, 38.7, 295.0, 77000.0, 0.010, 0.18),
    ("mexican-plateau", "high-terrain", 19.4, -99.1, 292.0, 78000.0, 0.008, 0.17),
    ("himalaya-high", "high-terrain", 28.0, 86.9, 255.0, 50000.0, 0.0015, 0.60),
    ("altiplano", "high-terrain", -19.0, -67.0, 283.0, 62000.0, 0.003, 0.25),
    ("northern-rockies", "high-terrain", 45.0, -110.5, 276.0, 72000.0, 0.003, 0.30),
    # edge cases
    ("north-pole", "edge", 90.0, 0.0, 250.0, 101000.0, 0.0005, 0.80),
    ("south-pole-dateline", "edge", -90.0, -180.0, 228.0, 68500.0, 0.00004, 0.83),
    ("near-north-pole", "edge", 89.999, 45.0, 252.0, 101000.0, 0.0005, 0.80),
    ("dateline-east", "edge", 0.0, 180.0, 300.0, 101000.0, 0.018, 0.06),
    ("dateline-west", "edge", 0.0, -180.0, 300.0, 101000.0, 0.018, 0.06),
    ("near-dateline", "edge", -10.0, -179.999, 299.0, 101000.0, 0.017, 0.06),
    ("prime-meridian-equator", "edge", 0.0, 0.0, 300.0, 101000.0, 0.018, 0.06),
    ("qv-zero", "edge", 25.0, 45.0, 315.0, 95000.0, 0.0, 0.35),
    ("qv-negative", "edge", 25.0, 46.0, 315.0, 95000.0, -1.0e-4, 0.35),
    ("qv-tiny", "edge", 25.0, 47.0, 315.0, 95000.0, 1.0e-7, 0.35),
    ("qv-very-moist", "edge", 5.0, 100.0, 306.0, 101000.0, 0.035, 0.10),
    ("very-hot", "edge", 23.0, 10.0, 335.0, 100000.0, 0.004, 0.40),
    ("very-cold", "edge", -78.0, 106.0, 185.0, 61000.0, 0.00001, 0.85),
    ("albedo-zero", "edge", 10.0, -30.0, 299.0, 101000.0, 0.016, 0.0),
    ("albedo-one", "edge", -72.0, 0.0, 245.0, 70000.0, 0.0002, 1.0),
    ("tropic-of-cancer", "edge", 23.44, -100.0, 300.0, 90000.0, 0.010, 0.20),
]


def _rings():
    """Two latitude rings that put columns on both sides of every terminator."""
    rows = []
    for k in range(12):
        lon = -180.0 + 30.0 * k
        rows.append((f"ring-30N-{k:02d}", "ring", 30.0, lon,
                     296.0, 100000.0, 0.012, 0.15))
    for k in range(12):
        lon = -165.0 + 30.0 * k
        rows.append((f"ring-45S-{k:02d}", "ring", -45.0, lon,
                     285.0, 100500.0, 0.008, 0.10))
    return rows


COLUMNS = _NAMED + _rings()

# (label, start (Y,M,D,h,m,s), elapsed seconds, radt minutes, dt seconds)
CASES = [
    ("leap-new-year-t0", (2024, 1, 1, 0, 0, 0), 0, 30.0, 60.0),
    ("day1-odd-minute", (2024, 1, 1, 0, 0, 0), 1260, 10.0, 60.0),
    ("equinox-julian-below-80", (2023, 3, 21, 0, 0, 0), 21600, 30.0, 90.0),
    ("equinox-julian-above-80", (2023, 3, 21, 12, 0, 0), 44400, 15.0, 60.0),
    ("radt-not-multiple-of-dt", (2026, 7, 4, 18, 0, 0), 5400, 15.0, 72.0),
    ("multi-day-xtime-wrap", (2026, 6, 21, 12, 0, 0), 2 * 86400 + 3 * 3600 + 600,
     10.0, 18.0),
    ("year-rollover", (2025, 12, 31, 23, 0, 0), 7200, 30.0, 60.0),
    ("leap-day-366", (2024, 12, 31, 6, 0, 0), 8 * 3600 + 20 * 60, 20.0, 40.0),
    ("radt-zero", (2026, 10, 7, 12, 0, 0), 600, 0.0, 60.0),
    ("odd-start-time", (2026, 10, 7, 14, 23, 37), 12345, 7.5, 45.0),
    ("julian-127-binade", (2026, 5, 8, 0, 0, 0), 30000, 30.0, 60.0),
    ("julian-255-binade", (2026, 9, 13, 5, 0, 0), 4000, 12.0, 30.0),
    ("december-solstice", (2026, 12, 21, 18, 0, 0), 3 * 3600, 30.0, 60.0),
    ("long-run-day-9", (2026, 8, 1, 0, 0, 0), 9 * 86400 + 4321, 15.0, 9.0),
]


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = [row[0] for row in COLUMNS]
    if len(set(labels)) != len(labels):
        raise SystemExit("duplicate column label")
    arr = np.array([row[2:] for row in COLUMNS], dtype=np.float64)
    lat, lon, theta, pres, qv, alb = (arr[:, k].astype(F) for k in range(6))
    ncol = lat.size
    if ncol < 64:
        raise SystemExit("the oracle needs at least 64 columns")
    starts = np.array([c[1] for c in CASES], dtype=np.int32)
    elapsed = np.array([c[2] for c in CASES], dtype=np.int32)
    radt = np.array([c[3] for c in CASES], dtype=F)
    dt = np.array([c[4] for c in CASES], dtype=np.float64)
    np.savez(out / "columns.npz", lat=lat, lon=lon, theta=theta,
             pressure=pres, qv=qv, albedo=alb,
             labels=np.array(labels), regimes=np.array([r[1] for r in COLUMNS]),
             case_labels=np.array([c[0] for c in CASES]),
             starts=starts, elapsed=elapsed, radt=radt, dt=dt)
    with open(out / "cz_in.bin", "wb") as fh:
        np.array([ncol, len(CASES)], dtype="<i4").tofile(fh)
        lat.astype("<f4").tofile(fh)
        lon.astype("<f4").tofile(fh)
        for k in range(len(CASES)):
            np.array(list(starts[k]) + [elapsed[k]], dtype="<i4").tofile(fh)
            np.array([radt[k]], dtype="<f4").tofile(fh)
    summary = {"ncol": int(ncol), "ncase": len(CASES),
               "regimes": {r: sum(1 for c in COLUMNS if c[1] == r)
                           for r in dict.fromkeys(c[1] for c in COLUMNS)}}
    (out / "columns.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python make_columns.py OUT_DIR")
    main(sys.argv[1])
