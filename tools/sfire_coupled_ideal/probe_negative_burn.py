"""Capture unchanged public forecast state when its smoke source refuses.

Diagnostic orchestration only. Arrays are copied from the device; model
inputs, kernels, source guards, and forecast state are unchanged.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys


def main():
    import cupy as cp
    import numpy as np
    from gpuwm.core import chem_sfire
    from gpuwm.core.sfire import FireState

    output = Path(os.environ["SFIRE_NEGATIVE_BURN_CAPTURE"])
    original_step = FireState._step
    original_source = chem_sfire.step
    prior = {}

    def fire_step(self, dt, exchange):
        prior[id(self)] = {
            "arrays": {name: self.data[name].copy() for name in
                       ("lfn", "tign", "fuel_frac", "fire_area")},
            "metadata": self.metadata(), "dt": float(dt),
        }
        return original_step(self, dt, exchange)

    def source_step(ctx, dt, ktau):
        try:
            return original_source(ctx, dt, ktau)
        except ValueError as exc:
            if "consumed fraction" not in str(exc):
                raise
            fire = ctx.state.physics.fire
            grid = fire.grid
            output.mkdir(parents=True, exist_ok=False)
            before = prior[id(grid)]
            arrays = {"before_" + name: cp.asnumpy(array)
                      for name, array in before["arrays"].items()}
            arrays.update({"after_" + name: cp.asnumpy(array)
                           for name, array in grid.data.items()})
            np.savez(output / "state.npz", **arrays)
            interior = grid.interior
            burnt = grid.data["burnt_area_dt"][interior]
            fuel = grid.data["fgip"][interior]
            receipt = {
                "schema": "sfire-negative-burn-capture-v1",
                "error": str(exc), "before": before["metadata"],
                "after": grid.metadata(), "dt": before["dt"],
                "burnt_min": float(burnt.min().item()),
                "burnt_max": float(burnt.max().item()),
                "negative_burn_count": int(cp.count_nonzero(burnt < 0).item()),
                "fuel_min": float(fuel.min().item()),
                "negative_fuel_count": int(cp.count_nonzero(fuel < 0).item()),
                "burnt_argmin_flat_interior": int(burnt.argmin().item()),
                "capture_changes_model_state": False,
            }
            (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
            print(json.dumps(receipt), flush=True)
            raise

    FireState._step = fire_step
    chem_sfire.step = source_step
    from gpuwm.cli import main as cli_main
    return cli_main(sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
