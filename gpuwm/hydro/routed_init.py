"""Route the complete sealed spin-up on the admitted forecast card at startup.

Rust writes/validates the catchment-major runoff packet; CUDA converts it to
network lateral flow and the ordinary MC/reservoir launches advance the state.
There is no second soil pass and no simulation output or graph capture here.
"""
from __future__ import annotations

import ctypes
import math
from pathlib import Path
import time


def replay(*, router, network, runoff, reservoirs, manifest):
    import cupy as cp
    import numpy as np
    from gpuwm.hydrofabric import RawPayloadReader
    from gpuwm.core.ieee_module import compile_module
    from gpuwm.numerics import is_strict

    declaration = network.manifest['routing']['initial_replay']
    field = declaration['payload']
    hours = declaration['hours']
    shape = tuple(field['shape'])
    if (type(hours) is not int or hours <= 0 or shape != (len(network.catchment_ids), hours)
            or field['dtype'] != '<f8' or field['bytes'] != math.prod(shape) * 8
            or 3600 % router.dt_seconds):
        raise ValueError('continuous routing needs sealed whole-hour catchment runoff and whole routing substeps')
    if reservoirs is not None:
        # Assimilation series are forecast-relative. A historical replay needs
        # its own historical observations for any assimilating reservoir type.
        if bool(cp.any(reservoirs.arrays['types'] != 1).item()):
            raise ValueError('continuous routing initialization lacks historical observations for assimilating reservoirs')
        reservoirs.arrays['clock'].fill(-hours * 3600.)
    path = (Path(manifest).parent / field['path']).resolve()
    if not path.is_relative_to(Path(manifest).parent.resolve()):
        raise ValueError('routing replay payload escapes its prepared directory')
    start = time.perf_counter()
    size = field['bytes']
    pinned = cp.cuda.alloc_pinned_memory(max(1, size))
    pointer = ctypes.addressof(ctypes.c_char.from_buffer(pinned))
    RawPayloadReader().read_into(path, field['sha256'], pointer, size)
    depth = cp.empty(shape, cp.float64)
    cp.cuda.runtime.memcpy(depth.data.ptr, pointer, size, cp.cuda.runtime.memcpyHostToDevice)
    del pinned
    source = Path(__file__).with_name('kernels') / 'runoff_replay.cu'
    flags = ['--std=c++17', '--ftz=false']
    if is_strict():
        flags += ['--fmad=false', '--prec-div=true', '--prec-sqrt=true']
    module = compile_module(source.read_text(), flags, 'nextgen.runoff.replay')
    kernel = module.get_function('replay_lateral')
    a = runoff.arrays
    nodes = len(network.node_ids)
    fixed = (depth, a['area'], a['offsets'], a['indices'], a['routing_lateral'], a['status'],
             np.int32(nodes), np.int32(hours))
    scale = np.float64(1e6 if runoff.restart_identity['area_units'] == 'km2' else 1.)
    substeps = int(3600 / router.dt_seconds)
    stream = cp.cuda.get_current_stream()
    for hour in range(hours):
        kernel(((nodes + 127) // 128,), (128,), (*fixed, np.int32(hour), scale), stream=stream)
        for _ in range(substeps):
            router.launch(router.dt_seconds, stream=stream)
    stream.synchronize()
    if (not bool(cp.all(cp.isfinite(router.arrays['state'])).item())
            or bool(cp.any(a['status']).item())
            or (reservoirs is not None and (not bool(cp.all(cp.isfinite(reservoirs.arrays['height'])).item())
                                          or bool(cp.any(reservoirs.arrays['fault']).item())))):
        raise RuntimeError('continuous routed initialization produced nonfinite channel/reservoir state or a numerical fault')
    # Producer sums begin with the first forecast LSM call. MC previous/current
    # state, reservoir height/release/inflow and clocks remain the replay's
    # continuous final carriers, with clock zero at the forecast origin.
    a['routing_lateral'].fill(0)
    a['routing_discharge_m3_s'].fill(0)
    a['routing_volume_m3'].fill(0)
    record = dict(declaration, method='continuous MC and level-pool routing of every replay hour',
                  routing_steps=hours * substeps, routing_interval_seconds=router.dt_seconds,
                  seconds=time.perf_counter() - start,
                  parallelism='CUDA receiver gather; normal level-ordered MC and reservoir hooks; one host launch thread',
                  temporary_device_bytes=size, temporary_pinned_bytes=size)
    router.initialization_receipt = record
    router.restart_identity['initial_routing_replay'] = declaration
    if reservoirs is not None:
        reservoirs.restart_identity['initial_routing_replay'] = declaration
    return record
