"""Isolated column-fusion timing/counter target after independent exact-word gates.

This is a component probe, not full-step throughput or forecast qualification.
Run only on an OWNER-held node. No hardware counter is inferred from timing.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('separate', 'fused', 'shared'), required=True)
    parser.add_argument('--nx', type=int, default=400)
    parser.add_argument('--ny', type=int, default=400)
    parser.add_argument('--nz', type=int, default=50)
    parser.add_argument('--members', type=int, default=20)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--receipt', type=Path, required=True)
    parser.add_argument('--estimate-only', action='store_true')
    args = parser.parse_args()
    if min(args.nx, args.ny, args.nz, args.members, args.repeats) < 1:
        parser.error('grid, members and repeats must be positive')
    from tools import ensemble_batch_step_probe as probe
    cfg = probe.configuration(SimpleNamespace(nx=args.nx, ny=args.ny, nz=args.nz,
        dx=3000.0, dt=6.0, warmup=2, steps=args.repeats, km_opt=1, diff6=0, terrain_height=0.0))
    plan, shared, extras, slots = probe.allocation_plan(cfg, 512 * 1024**2)
    receipt = {'scope': 'isolated three acoustic substeps; no full forecast throughput',
               'identity': 'not checked by this tool; require independent all-state/scratch gate',
               'mode': args.mode, 'members': args.members, 'shape': (args.nz, args.ny, args.nx),
               'dtau': cfg.dt / cfg.time_step_sound, 'repeats': args.repeats,
               'counter_status': 'not measured by this script; profile actual launched kernels externally',
               'additional_candidate_device_workspace_bytes': 0}
    if args.estimate_only:
        args.receipt.write_text(json.dumps(receipt, indent=2) + '\n')
        return
    import cupy as cp
    import numpy as np
    from gpuwm.ensemble.batch_state import BatchedDomainState
    from gpuwm.ensemble.batch_bookkeeping import prepare_bookkeeping
    from gpuwm.ensemble.batch_diagnostics import prepare_update_diagnostics
    from gpuwm.ensemble.batch_bigstep import prepare_small_step_init
    from gpuwm.ensemble import batch_acoustic as separate, batch_acoustic_fusion as fused
    from gpuwm.core.dycore import _PROGNOSTICS
    host = probe.host_members(cfg, args.members, extras)
    free, _ = cp.cuda.runtime.memGetInfo()
    state = BatchedDomainState.from_prepared(host, array_module=cp, available_bytes=free,
        shared_fields=shared, extra_specs=extras, scratch_slots={name: np.float32 for name in slots},
        reserved_bytes=512 * 1024**2)
    del host
    prepare_bookkeeping(tuple((getattr(state, name), getattr(state, name + '0'))
                               for name in _PROGNOSTICS), members=state.members)()
    prepare_update_diagnostics(state, cfg.hypsometric_opt)()
    initialize = prepare_small_step_init(state)
    coefficients = separate.prepare_acoustic_coefficients(state, cfg, receipt['dtau'])
    factory = separate.prepare_acoustic_substep_launch if args.mode == 'separate' else fused.prepare_acoustic_substep_launch
    options = {} if args.mode == 'separate' else {'shared_intermediates': args.mode == 'shared'}
    launch = factory(state, cfg, receipt['dtau'], coefficients, **options)
    initialize()
    launch(first=True)
    cp.cuda.get_current_stream().synchronize()
    receipt['fusion_receipt'] = getattr(launch, 'fusion_receipt', {'effective': 'original_split_columns'})
    receipt['compiled_attributes'] = getattr(launch, 'compiled_attributes', {})
    timings = []
    for _ in range(args.repeats):
        initialize()
        cp.cuda.get_current_stream().synchronize()
        begin, end = cp.cuda.Event(), cp.cuda.Event()
        started = time.perf_counter()
        cp.cuda.nvtx.RangePush('acoustic_mu_w/' + args.mode)
        begin.record()
        for first in (True, False, False):
            launch(first=first)
        end.record()
        cp.cuda.nvtx.RangePop()
        end.synchronize()
        timings.append({'gpu_ms_three_substeps': cp.cuda.get_elapsed_time(begin, end),
                        'wall_seconds_three_substeps': time.perf_counter() - started})
    receipt['timings'] = timings
    receipt['device'] = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode()
    receipt['runtime_version'] = cp.cuda.runtime.runtimeGetVersion()
    receipt['driver_version'] = cp.cuda.runtime.driverGetVersion()
    receipt['cupy_version'] = cp.__version__
    args.receipt.write_text(json.dumps(receipt, indent=2, default=str) + '\n')
    print(json.dumps(receipt, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
