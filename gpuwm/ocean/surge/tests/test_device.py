"""Exercise the real shared ABI, CUDA library and non-default stream."""
import ctypes
import json
import math
import os
from pathlib import Path
import time

import pytest


def test_device_contract(monkeypatch):
    cp = pytest.importorskip('cupy')
    from gpuwm.ocean.interface import (OceanGrid, Bathymetry, AtmosphereForcing,
                                       WaveOutputs, SurgeSolver, SurgeOutputs)
    from gpuwm.ocean.surge import create_solver
    started = time.monotonic()
    stream = cp.cuda.Stream(non_blocking=True)
    shape = (192, 256)
    ny, nx = shape
    def plane(v):
        return cp.full(shape, v, dtype=cp.float32)
    with stream:
        zero, one = plane(0), plane(1)
        grid = OceanGrid(nx, ny, 100., 100., zero, zero, one, zero)
        bed = plane(-40)
        bath = Bathymetry(bed, plane(40), one, zero, 'test datum', 'analytic test')
        config = dict(boundary_mask=0, manning=0.)
        calm = create_solver(grid, bath, config=config)
        wind = create_solver(grid, bath, config=config)
        pressure = create_solver(grid, bath, config=config)
        waves = create_solver(grid, bath, config=config)
        # Rotation by 90 degrees sends eastward wind stress into -grid y.
        rotated_grid = OceanGrid(nx, ny, 100., 100., zero, zero, zero, one)
        rotated = create_solver(rotated_grid, bath, config=config)
        # A sloping beach exercises initially dry cells and wet/dry outputs.
        beach_bed = cp.ascontiguousarray(cp.broadcast_to(
            cp.linspace(-2, .2, nx, dtype=cp.float32), shape))
        wet = cp.ascontiguousarray((beach_bed < 0).astype(cp.float32))
        beach = create_solver(grid, Bathymetry(beach_bed, cp.maximum(-beach_bed, 0),
                    wet, zero, 'test datum', 'analytic beach'), config=dict(
                        config, initial_level=.15))
        p = plane(101325)
        gradient = cp.ascontiguousarray(p + cp.arange(nx, dtype=cp.float32)[None, :]*2)
        hs = cp.ascontiguousarray(plane(1) + cp.arange(nx, dtype=cp.float32)[None, :]*.002)
        still = AtmosphereForcing(zero, zero, p, zero, zero, zero, one, 0.)
        stress = AtmosphereForcing(zero, zero, p, zero, one, zero, one, 0.)
        pf = AtmosphereForcing(zero, zero, gradient, zero, zero, zero, one, 0.)
        wave = WaveOutputs(hs, plane(8), plane(90))
        for solver in (calm, wind, pressure, waves, rotated, beach):
            assert isinstance(solver, SurgeSolver)
            assert isinstance(solver.outputs, SurgeOutputs)
        assert stream.ptr != 0
        stream.synchronize()  # constructor/initial state only; outside guard
        for solver in (calm, wind, pressure, waves, rotated, beach):
            assert solver.outputs.eta.shape == shape
            assert solver.outputs.eta.dtype == cp.float32
        eta_pointer = wind.outputs.eta.data.ptr
        flood_pointer = beach.outputs.inundation.data.ptr
        # No silently accepted CPU, shape or dtype conversions.
        import numpy as np
        with pytest.raises(TypeError, match='CUDA'):
            wind.set_waves(WaveOutputs(np.ones(shape, np.float32), zero, zero))
        with pytest.raises(ValueError, match='float32'):
            wind.set_waves(WaveOutputs(cp.ones(shape, cp.float64), zero, zero))
        with pytest.raises(RuntimeError, match='set_forcing'):
            wind.step(1.)
        for value in (0., -1., float('nan'), float('inf')):
            calm.set_forcing(still)
            with pytest.raises(ValueError, match='positive'):
                calm.step(value)

        def forbidden(*args, **kwargs):
            raise AssertionError('host transfer, conversion, synchronization or CUDA graph during ABI step')
        # The native interception guard is optional outside the proof harness.
        guard = ctypes.CDLL(None)
        native_guard = getattr(guard, 'surge_test_guard', None)
        if native_guard:
            native_guard.argtypes = [ctypes.c_int]
            guard.surge_test_stream.argtypes = [ctypes.c_size_t]
            guard.surge_test_stream(stream.ptr)
        with monkeypatch.context() as patch:
            for name in ('asnumpy', 'asarray'):
                patch.setattr(cp, name, forbidden)
            for name in ('deviceSynchronize', 'streamSynchronize', 'memcpy', 'memcpyAsync',
                         'graphLaunch', 'streamBeginCapture'):
                if hasattr(cp.cuda.runtime, name):
                    patch.setattr(cp.cuda.runtime, name, forbidden)
            if native_guard:
                native_guard(1)
            try:
                for solver, forcing in ((calm, still), (wind, stress), (pressure, pf),
                                        (waves, still), (rotated, stress), (beach, stress)):
                    solver.set_forcing(forcing)
                waves.set_waves(wave)
                for _ in range(10):
                    for solver in (calm, wind, pressure, waves, rotated, beach):
                        solver.step(4.)
            finally:
                if native_guard:
                    native_guard(0)
        # Event completion is deliberately only at output cadence.
        stream.synchronize()
        d2d_count = guard.surge_test_d2d_count() if native_guard else None
        if native_guard:
            assert d2d_count > 0  # prove native interception actually ran
        assert wind.outputs.eta.data.ptr == eta_pointer
        assert beach.outputs.inundation.data.ptr == flood_pointer
        assert float(cp.max(cp.abs(calm.outputs.eta)).get()) < 2e-6
        for solver in (calm, wind, pressure, waves, rotated, beach):
            assert bool(cp.isfinite(solver.outputs.eta).all().get())
            assert bool((solver.outputs.inundation >= 0).all().get())
            assert solver.diagnostics.get().tolist() == [0, 0, 0, 0]
        # Read momentum only after completion, never inside the ABI loop.
        def momentum(solver, component):
            return float(solver._state[0].view(cp.float32).reshape(ny, nx, 4)[ny//2,nx//2,component].get())
        assert momentum(wind, 2) == pytest.approx(40/1025, rel=2e-4)
        assert momentum(rotated, 3) == pytest.approx(-40/1025, rel=2e-4)
        assert abs(momentum(rotated, 2)) < 1e-6
        assert momentum(pressure, 2) == pytest.approx(-40/1025*.02*40, rel=2e-4)
        wave_momentum = momentum(waves, 2)
        assert wave_momentum < -1e-4
        assert float(beach.outputs.inundation.max().get()) > 0
        assert float(cp.max(cp.abs(beach.outputs.inundation-cp.where(
            wet>.5, 0, cp.maximum(beach.outputs.eta-beach_bed, 0)))).get()) < 1e-6
        receipt = dict(schema='gpuwm.surge-device-proof.v1', shape=list(shape),
            caller_stream=stream.ptr, simulations=6, steps_each=10, seconds_each=40,
            wall_seconds=time.monotonic()-started, cuda_graphs=False,
            native_host_transfer_sync_guard=bool(native_guard),
            intercepted_device_to_device_copies=d2d_count,
            persistent_outputs=True, diagnostics=[0,0,0,0],
            wind_momentum=momentum(wind,2), pressure_momentum=momentum(pressure,2),
            wave_momentum=wave_momentum, rotated_momentum=momentum(rotated,3),
            allocated_pool_bytes=cp.get_default_memory_pool().total_bytes())
    if os.environ.get('GPUWM_SURGE_DEVICE_RECEIPT'):
        Path(os.environ['GPUWM_SURGE_DEVICE_RECEIPT']).write_text(json.dumps(receipt, indent=2)+'\n')
