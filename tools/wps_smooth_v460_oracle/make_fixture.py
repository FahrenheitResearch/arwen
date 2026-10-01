"""Write the WPS terrain-smoother oracle fixture.

Runs the driver build.sh produced over a fixed set of float32 planes and
every (smoother, pass count) case, and saves inputs, cases and WPS's own
outputs to one compressed .npz that tests/test_terrain_smoothing.py and the
static-fields crate replay bit for bit.

    python make_fixture.py BUILD_DIR OUT.npz [--real NAME=PATH.npy ...]

``--real`` adds planes read from .npy files (float32, 2-D), for example an
unsmoothed HGT_M written by WPS's own geogrid.  The synthetic planes are
seeded, so the fixture is reproducible from this script and the pinned WPS
tree alone.
"""
from __future__ import annotations

import argparse
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

#: (code, WPS routine) -- the driver's option codes.
ROUTINES = {1: "1-2-1", 2: "smth-desmth", 3: "smth-desmth_special"}
PASSES = {1: (0, 1, 2, 3, 5), 2: (0, 1, 2, 3), 3: (0, 1, 2, 3)}


def synthetic_planes() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(20260930)
    ny, nx = 41, 53
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)
    ridges = (1800.0 * np.exp(-((x - 20.0) ** 2) / 18.0)
              + 1200.0 * np.exp(-((y - 12.0) ** 2) / 6.0)
              + 400.0 * np.sin(x / 3.0) * np.cos(y / 4.0))
    noisy = ridges + rng.normal(0.0, 150.0, (ny, nx))
    # A coast: sea at exactly 0 and a narrow valley, so the desmoothing
    # overshoot drives land and sea points below zero and the special
    # smoother's restore has work to do.
    coast = np.where(x < 14.0, 0.0, noisy)
    coast[:, 14:17] = 3.0
    spikes = np.zeros((ny, nx))
    spikes[5::7, 3::9] = 2500.0
    spikes[20, :] = -30.0
    # Values near the float32 subnormal boundary: 0.25*(a+b) and 0.26*(a+b)
    # of these land below 1.17549435e-38, so a flush-to-zero anywhere in
    # the chain would show.
    tiny = (rng.integers(1, 64, (ny, nx)).astype(np.float64)
            * np.float64(np.finfo(np.float32).tiny) / 16.0)
    tiny[::3, ::2] *= -1.0
    return {
        "ridges_noise": noisy.astype(np.float32),
        "coast_valley": coast.astype(np.float32),
        "spikes": spikes.astype(np.float32),
        "near_subnormal": tiny.astype(np.float32),
    }


def run_driver(driver: Path, plane: np.ndarray, cases) -> np.ndarray:
    ny, nx = plane.shape
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, dst = tmp / "in.bin", tmp / "out.bin"
        with open(src, "wb") as handle:
            handle.write(struct.pack("<3i", ny, nx, len(cases)))
            for code, npass in cases:
                handle.write(struct.pack("<2i", code, npass))
            handle.write(np.ascontiguousarray(plane, dtype="<f4").tobytes())
        subprocess.run([str(driver), str(src), str(dst)], check=True)
        raw = np.frombuffer(dst.read_bytes(), dtype="<f4")
    return raw.reshape(len(cases), ny, nx).copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--real", action="append", default=[],
                        metavar="NAME=PATH")
    args = parser.parse_args()
    driver = args.build_dir / "wps_smooth_driver"
    provenance = (args.build_dir / "oracle-provenance.txt").read_text()

    planes = synthetic_planes()
    for item in args.real:
        name, path = item.split("=", 1)
        plane = np.load(path)
        if plane.dtype != np.float32 or plane.ndim != 2:
            raise SystemExit(f"{path}: need a 2-D float32 plane")
        planes[name] = plane

    cases = [(code, npass) for code in sorted(ROUTINES)
             for npass in PASSES[code]]
    payload: dict[str, np.ndarray] = {
        "cases": np.array(cases, dtype=np.int32),
        "case_names": np.array([ROUTINES[c] for c, _ in cases]),
        "plane_names": np.array(sorted(planes)),
        "provenance": np.array(provenance),
    }
    tiny = np.finfo(np.float32).tiny
    report = []
    for name in sorted(planes):
        plane = planes[name]
        out = run_driver(driver, plane, cases)
        payload[f"in_{name}"] = plane
        payload[f"out_{name}"] = out
        subnormal = int(np.count_nonzero(
            (out != 0.0) & (np.abs(out) < tiny)))
        report.append(f"{name}: {plane.shape}, {subnormal} subnormal "
                      "output values across the cases")
    np.savez_compressed(args.out, **payload)
    print("\n".join(report))
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
