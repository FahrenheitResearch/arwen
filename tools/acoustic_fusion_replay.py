"""Replay every acoustic substep of a real forecast through both launch paths.

What it answers: do the joined acoustic launches (gpuwm/core/acoustic_fusion.py)
write exactly the words the separate kernels write, on the inputs a real
forecast hands them?

How.  The forecast runs in this process (the door's worker branch) with
``gpuwm.core.dycore.prepare_acoustic_substep_launch`` wrapped.  For each RK
stage the wrapper prepares the joined launcher and, with the joining switched
off, the separate one.  At every compared substep it

1. snapshots every array the joined launches may write,
2. runs the reference: ``apply_emdiv`` (when the joined u/v launch absorbed
   it), the separate substep kernels, ``zero_sumflux`` on the stage's first
   substep and ``accumulate_sumflux`` (when the joined column absorbed them),
3. keeps the reference words, restores the snapshot,
4. runs the joined launches on the same inputs and counts, per array, the
   32-bit words that differ from the reference.

The forecast continues from the joined words.  Counting stays on the device;
the receipt is written when the run ends.

Usage (forecast arguments after ``--``)::

    python tools/acoustic_fusion_replay.py --receipt R.json [--every N] -- \\
        --wrfinput RUN --run-seconds 3600 --outdir OUT --products none
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--every", type=int, default=1,
                        help="compare every Nth substep (default 1: all)")
    args, rest = parser.parse_known_args(argv)
    forecast = rest[1:] if rest[:1] == ["--"] else rest

    import cupy as cp
    from gpuwm.core import acoustic, acoustic_fusion, dycore
    from gpuwm.core.kernels import get_kernel

    original = acoustic.prepare_acoustic_substep_launch
    names: list[str] = []
    totals = {}
    per_substep = cp.zeros(1 << 16, dtype=cp.int64)
    stats = {"stages": 0, "stages_joined_uv": 0, "stages_joined_column": 0,
             "substeps": 0, "substeps_compared": 0, "words_compared": 0,
             "absorbs": set(), "arrays": set()}
    started = time.time()

    def wrapper(state, cfg, dtau, coefficients, *, mudf=None, emdiv_args=None,
                sumflux=None):
        acoustic_fusion.ENABLED = True
        joined = original(state, cfg, dtau, coefficients, mudf=mudf,
                          emdiv_args=emdiv_args, sumflux=sumflux)
        acoustic_fusion.ENABLED = False
        try:
            separate = original(state, cfg, dtau, coefficients, mudf=mudf)
        finally:
            acoustic_fusion.ENABLED = True
        absorbs = joined.absorbs
        written = dict(joined.joined_written)
        stats["stages"] += 1
        stats["stages_joined_uv"] += int(joined.joined[0])
        stats["stages_joined_column"] += int(joined.joined[1])
        stats["absorbs"] |= set(absorbs)
        stats["arrays"] |= set(written)
        if "emdiv" in absorbs:
            nz, ny, nx = state.p.shape
            emdiv_kernel = get_kernel("acoustic", "apply_emdiv")
            emdiv_grid = ((nz * (ny + 1) * (nx + 1) + 255) // 256,)
        if "sumflux" in absorbs:
            zero = dycore._prepare_sumflux_launch("zero_sumflux", tuple(sumflux))
            accumulate = dycore._prepare_sumflux_launch(
                "accumulate_sumflux", tuple(sumflux),
                (state.u_pp, state.v_pp, state.ww_pp))
        # One array object per device buffer, compared once.
        unique = {}
        for name, array in written.items():
            key = (array.data.ptr, array.shape, array.dtype.str)
            unique.setdefault(key, (name, array))
        arrays = list(unique.values())
        for name, _ in arrays:
            if name not in totals:
                names.append(name)
                totals[name] = cp.zeros((), dtype=cp.int64)

        def launch(*, first: bool) -> None:
            index = stats["substeps"]
            stats["substeps"] += 1
            if not written or index % args.every:
                joined(first=first)
                return
            snap = [array.copy() for _, array in arrays]
            if "emdiv" in absorbs:
                emdiv_kernel(emdiv_grid, (256,), emdiv_args)
            if "sumflux" in absorbs and first:
                zero()
            separate(first=first)
            if "sumflux" in absorbs:
                accumulate()
            reference = [array.copy() for _, array in arrays]
            for (_, array), saved in zip(arrays, snap):
                cp.copyto(array, saved)
            joined(first=first)
            for (name, array), ref in zip(arrays, reference):
                differ = cp.count_nonzero(array.view(cp.uint32) != ref.view(cp.uint32))
                totals[name] += differ
                if index < per_substep.size:
                    per_substep[index] += differ
                stats["words_compared"] += int(array.size)
            stats["substeps_compared"] += 1

        launch.absorbs = absorbs
        launch.joined = joined.joined
        launch.joined_written = written
        return launch

    dycore.prepare_acoustic_substep_launch = wrapper
    from gpuwm.wrfinput_forecast import main as forecast_main
    # The door's own in-process body (the supervised worker's branch), so the
    # wrapper above is the one the stepping loop calls.
    if "--_worker" not in forecast:
        forecast = list(forecast) + ["--_worker"]
    status = None
    try:
        status = forecast_main(forecast)
    except SystemExit as exc:
        status = exc.code
    finally:
        cp.cuda.runtime.deviceSynchronize()
        differing = {name: int(totals[name]) for name in names}
        steps = per_substep.get()
        first_bad = next((int(i) for i, value in enumerate(steps) if value), None)
        device = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)
        receipt = {
            "schema": "acoustic-fusion-replay/1",
            "forecast_arguments": forecast,
            "forecast_status": status,
            "device": {"name": device["name"].decode() if isinstance(device["name"], bytes) else device["name"],
                       "compute_capability": f"{device['major']}.{device['minor']}"},
            "cupy": cp.__version__,
            "nvrtc": list(cp.cuda.nvrtc.getVersion()),
            "every": args.every,
            "stages": stats["stages"],
            "stages_joined_uv": stats["stages_joined_uv"],
            "stages_joined_column": stats["stages_joined_column"],
            "substeps": stats["substeps"],
            "substeps_compared": stats["substeps_compared"],
            "words_compared": stats["words_compared"],
            "absorbed": sorted(stats["absorbs"]),
            "arrays_compared": sorted(stats["arrays"]),
            "differing_words_by_array": differing,
            "differing_words": sum(differing.values()),
            "first_differing_substep": first_bad,
            "wall_s": round(time.time() - started, 1),
        }
        with open(args.receipt, "w", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=1)
        print(json.dumps({key: receipt[key] for key in (
            "substeps_compared", "words_compared", "differing_words",
            "first_differing_substep", "absorbed")}))
    return 0 if status in (0, None) and receipt["differing_words"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
