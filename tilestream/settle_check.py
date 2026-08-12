"""Does the monolithic settling step change WHICH carriers exist?

``build_stores`` runs one full-domain ``dycore.step`` before it takes the
carrier manifest, because two carriers are allocated lazily on first use --
Kain-Fritsch's ``cumulus/w0avg`` above all.  Past the resident ceiling that
step is exactly what does not fit, so ``settle=False`` skips it.  This is the
control that says what skipping it costs, on a domain small enough that both
answers can be had:

    python -m tilestream.settle_check --config CFG

It prints the carrier-name set and the wrfout frame-plan field set BEFORE and
AFTER the step and diffs them.  An empty diff means the manifest a streamed
run builds without the step is the manifest it would have built with it.  A
non-empty diff names the carriers a ``settle=False`` run of THAT
configuration would be missing -- and is the reason ``run_tiled`` compares
inventories and refuses rather than trusting this result to generalise.
"""

from __future__ import annotations

import argparse
import json


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prepare-rows", type=int, default=64)
    args = ap.parse_args()

    import cupy as cp

    from tilestream import driver, output as tsout, physics_inventory as physinv
    from tilestream import realcase

    prep, exp, _peak = realcase.prepare_slabbed(
        args.config, rows_per_slab=args.prepare_rows)
    cfg = prep.cfg
    state = prep.initial_result.state

    before = set(physinv.carrier_inventory(state))
    before_geo = set(driver.geography_inventory(state))
    before_plan = set(tsout.frame_plan(state).order)

    from gpuwm.core.dycore import step
    step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()

    after = set(physinv.carrier_inventory(state))
    after_geo = set(driver.geography_inventory(state))
    after_plan = set(tsout.frame_plan(state).order)

    record = {
        "config": args.config,
        "nx": int(cfg.nx), "ny": int(cfg.ny), "nz": int(cfg.nz),
        "mp_physics": int(cfg.mp_physics),
        "bl_pbl_physics": int(cfg.bl_pbl_physics),
        "sf_surface_physics": int(cfg.sf_surface_physics),
        "cu_physics": int(getattr(cfg, "cu_physics", 0)),
        "carriers_before": len(before), "carriers_after": len(after),
        "carriers_only_after": sorted(after - before),
        "carriers_only_before": sorted(before - after),
        "geography_only_after": sorted(after_geo - before_geo),
        "plan_before": len(before_plan), "plan_after": len(after_plan),
        "plan_only_after": sorted(after_plan - before_plan),
        "plan_only_before": sorted(before_plan - after_plan),
    }
    record["identical"] = not (record["carriers_only_after"]
                               or record["carriers_only_before"]
                               or record["geography_only_after"]
                               or record["plan_only_after"]
                               or record["plan_only_before"])
    print("SETTLE_CHECK " + json.dumps(record))


if __name__ == "__main__":
    main()
