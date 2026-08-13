"""Attribute the dealias cost inside ``vad.band_fit`` to named functions.

``tools/perf_obs_timing.py`` prices the observation front door by stage and
finds one stage carrying almost all of it (``obs.dealias.vad.band_fit``).
That is as far as a stage clock can see: ``band_fit`` is a loop over seeds
around an alternating solve, and "the fit is slow" does not say what to
port.  This puts a counting wrapper on the two functions inside it that a
port would target and reports what each actually costs, so the target is a
measurement rather than a reading of the code.

    python tools/perf_dealias_hotspot.py --pack <sweep pack> --out receipt.json

Both wrappers are installed by assignment onto the module under test and
removed afterwards, so nothing about the shipped path changes: the arm this
measures is the same arm ``perf_obs_timing`` measures, and the receipt
records the volume's dealias totals so the two can be checked against each
other gate for gate.

The candidate-grid arithmetic is read from the module's own constants, not
copied here, so a receipt cannot go on claiming a grid shape that the
search no longer uses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
# One definition of the target box, shared with the stage-timing tool it
# extends rather than re-derived beside it.
from perf_obs_timing import _grid_around  # noqa: E402

from gpuwm import perf_timing  # noqa: E402
from gpuwm.obs import dealias as dealias_module  # noqa: E402
from gpuwm.obs.dealias import DealiasParams  # noqa: E402
from gpuwm.obs.superob import SuperobParams, superob_volume  # noqa: E402
from gpuwm.obs.sweeps import read_sweep_pack  # noqa: E402


class _Counter:
    """Calls, wall seconds and one summed size, around any callable."""

    def __init__(self, wrapped, size_of=None) -> None:
        self.wrapped = wrapped
        self.size_of = size_of
        self.calls = 0
        self.seconds = 0.0
        self.size = 0

    def __call__(self, *args, **kwargs):
        if self.size_of is not None:
            self.size += self.size_of(*args, **kwargs)
        started = time.perf_counter()
        try:
            return self.wrapped(*args, **kwargs)
        finally:
            self.seconds += time.perf_counter() - started
            self.calls += 1


def _search_grid() -> dict:
    """The exhaustive search's shape, read from the module's constants."""
    speeds = int(dealias_module._COARSE_SPEEDS.size)
    directions = int(dealias_module._COARSE_DIRECTIONS.size)
    samples = int(dealias_module._COARSE_SAMPLES)
    candidates = speeds * directions
    return {
        "speeds": speeds,
        "directions": directions,
        "candidates": candidates,
        "samples_per_band": samples,
        "cost_matrix_elements": candidates * samples,
        "dtype": str(dealias_module._COARSE_SPEEDS.dtype),
        "note": ("one cosine per cost-matrix element; the model matrix is "
                 "formed by numpy broadcasting on one core"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True,
                        help="a decoded sweep pack (see perf_obs_timing.py)")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--nx", type=int, default=200)
    parser.add_argument("--ny", type=int, default=200)
    parser.add_argument("--nz", type=int, default=40)
    parser.add_argument("--dx-m", type=float, default=3000.0)
    parser.add_argument("--top-m", type=float, default=16000.0)
    args = parser.parse_args(argv)

    volume = read_sweep_pack(args.pack)
    site = volume.site
    grid = _grid_around(site.lat_deg, site.lon_deg, nx=args.nx, ny=args.ny,
                        nz=args.nz, dx_m=args.dx_m, top_m=args.top_m)

    seeds = _Counter(dealias_module._coarse_seeds)
    # np.linalg.lstsq(design, values, ...): the design's row count is the
    # number of samples the solve is over, which is the size that decides
    # whether an SVD or a 2x2 normal-equation solve is the right shape.
    lstsq = _Counter(np.linalg.lstsq,
                     size_of=lambda design, *rest, **kwargs: int(
                         np.asarray(design).shape[0]))

    dealias_module._coarse_seeds = seeds
    original_lstsq = np.linalg.lstsq
    np.linalg.lstsq = lstsq
    try:
        perf_timing.reset(on=True)
        params = SuperobParams(dealias=DealiasParams())
        started = time.perf_counter()
        contribution = superob_volume(volume, grid, params=params)
        wall = time.perf_counter() - started
        receipt = perf_timing.snapshot()
    finally:
        dealias_module._coarse_seeds = seeds.wrapped
        np.linalg.lstsq = original_lstsq

    stages = {stage["stage"]: stage for stage in receipt["stages"]}
    band_fit = stages.get("obs.dealias.vad.band_fit", {})
    band_fit_seconds = float(band_fit.get("self_seconds", 0.0))

    report = {
        "schema": "gpuwm-perf.dealias-hotspot.v1",
        "pack": str(args.pack),
        "site": site.id,
        "valid_time": str(volume.valid_time),
        "grid": {"nx": grid.nx, "ny": grid.ny, "nz": grid.nz,
                 "dx_m": grid.dx_m},
        "superob_volume_seconds": round(wall, 6),
        "band_fit_self_seconds": round(band_fit_seconds, 6),
        "coarse_seeds": {
            "function": "gpuwm.obs.dealias._coarse_seeds",
            "calls": seeds.calls,
            "seconds": round(seeds.seconds, 6),
            "seconds_per_call": round(seeds.seconds / seeds.calls, 6)
            if seeds.calls else None,
            "share_of_band_fit": round(seeds.seconds / band_fit_seconds, 4)
            if band_fit_seconds else None,
            "search_grid": _search_grid(),
            "transcendentals_per_volume":
                seeds.calls * _search_grid()["cost_matrix_elements"],
        },
        "lstsq": {
            "function": "numpy.linalg.lstsq",
            "calls": lstsq.calls,
            "seconds": round(lstsq.seconds, 6),
            "design_rows": lstsq.size,
            "share_of_band_fit": round(lstsq.seconds / band_fit_seconds, 4)
            if band_fit_seconds else None,
        },
        "dealias_totals": (contribution.dealias or {}).get("totals"),
        "stages": receipt["stages"],
        "note": ("the wrappers add their own per-call overhead, so the two "
                 "attributed seconds are upper bounds and their SHARES are "
                 "what the ranking rests on"),
    }
    text = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
