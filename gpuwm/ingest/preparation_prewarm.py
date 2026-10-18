"""Tiny, disposable preparations that populate CuPy's source-keyed disk cache.

Use the production orchestration, not replicas of its array expressions: CuPy
specializes on dtype, rank, contiguity and scalar/array parameter kinds. All
three spatial/vertical extents stay nontrivial even though no weather is read.
"""
from __future__ import annotations


def synthetic_preparation(run=None, *, specific=True):
    """Return a small real-dtype source, grid and run, entirely in memory."""
    from dataclasses import replace
    from datetime import datetime

    import numpy as np

    from gpuwm.config import RunConfig
    from gpuwm.ingest.grib import Era5Snapshot
    from gpuwm.ingest.soil_contract import (
        MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE, NOAH_TARGET_SOIL_LAYERS)
    from gpuwm.static.lambert import LambertGrid

    template = run or RunConfig(nx=33, ny=32, nz=7, dx=12000.0, dy=12000.0,
                               ztop=20000.0, dt=30.0, run_seconds=60.0,
                               mp_physics=10, hybrid_opt=2, base_temp=290.0)
    # More than 1024 surface cells selects CuPy's full 512-thread CUB
    # reduction specialization, as a real domain does. A 9x8 toy selected
    # smaller blocks and left all/any/count_nonzero cold on the root.
    cfg = replace(template, nx=33, ny=32, nz=7, dx=12000.0,
                  dy=12000.0, moist=True, terrain_opt=1)
    grid = LambertGrid(ref_lat=40.0, ref_lon=0.0, truelat1=30.0,
                       truelat2=60.0, stand_lon=0.0, dx=cfg.dx, dy=cfg.dy,
                       e_we=cfg.nx + 1, e_sn=cfg.ny + 1)
    levels = np.array([50., 100., 200., 300., 500., 700., 1000.])
    pressure = levels[:, None, None] * 100.0
    shape = (levels.size, 34, 35)

    def column(value):
        return np.broadcast_to(value, shape).copy()

    def plane(value):
        return np.full(shape[1:], value, dtype=np.float64)

    fields = {
        "T": column(215.0 + 72.0 * (pressure / 100000.0) ** 0.20),
        "Z": column(-7800.0 * np.log(pressure / 100000.0) * 9.81),
        "U": column(12.0 + 0.8 * np.log(100000.0 / pressure)),
        "V": column(-3.0 + 0.4 * np.log(100000.0 / pressure)),
        "RH": column(35.0 + 50.0 * pressure / 100000.0),
        "PSFC": plane(96000.0), "T2": plane(286.0), "D2": plane(279.0),
        "U10": plane(11.0), "V10": plane(-2.0),
        "LANDSEA": plane(1.0), "SKINTEMP": plane(286.0),
        MAPPED_SOIL_TEMPERATURE: np.full((NOAH_TARGET_SOIL_LAYERS, *shape[1:]), 283.0),
        MAPPED_SOIL_MOISTURE: np.full((NOAH_TARGET_SOIL_LAYERS, *shape[1:]), 0.25),
    }
    if specific:
        fields.update(PRES=column(pressure),
                      SPFH=column(0.003 * pressure / 100000.0), Q2=plane(0.003))
        fields.update({name: np.zeros(shape) for name in ("QC", "QR", "QI", "QS", "QG")})
    source = Era5Snapshot(valid_time=datetime(2000, 1, 1), levels_hpa=levels,
                          latitude=np.linspace(37.0, 43.0, shape[1]),
                          longitude=np.linspace(-3.0, 3.0, shape[2]), fields=fields)
    return source, grid, cfg


def root_initialize(run=None, *, specific=True, bounded=False):
    """Execute the mapped root's horizontal, start-time and REAL CUDA paths.

    The bounded job also builds an ordinary start through start_time_backend,
    as admission does when the whole start fits, then runs the bounded path.
    Nothing is exported; the only persistent output is the kernel cache.
    """
    import numpy as np
    import cupy as cp

    from gpuwm.core.grid import make_vertical_coord
    from gpuwm.ingest.bounded_cuda import BoundedCudaPreprocessBackend
    from gpuwm.ingest.horiz import interpolate_era5_to_lambert
    from gpuwm.ingest.lateral_bc import StateBoundaryFrames
    from gpuwm.ingest.preprocess_backend import CudaPreprocessBackend, start_time_backend
    from gpuwm.ingest.real import initialize_real

    source, grid, cfg = synthetic_preparation(run, specific=specific)
    engine = (BoundedCudaPreprocessBackend(device_budget_bytes=2**20, host_workers=1)
              if bounded else CudaPreprocessBackend(host_workers=1))
    if bounded:
        engine.selection = {"device_fit": {"unchunked_need_bytes": 2**20,
                                            "free_bytes": 2**21}, "chunking": {}}
    start = start_time_backend(engine)
    for backend in ((start, engine) if start is not engine else (engine,)):
        met = interpolate_era5_to_lambert(
            source, grid, backend=backend,
            target_landmask=np.ones((cfg.ny, cfg.nx), dtype=bool))
        coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt, etac=cfg.etac,
                                    eta_levels=np.linspace(1.0, 0.0, cfg.nz + 1))
        result = initialize_real(
            met, cfg, coord, np.zeros((cfg.ny, cfg.nx)), source_orography=np.zeros((cfg.ny, cfg.nx)),
            p_top=5000.0, preprocess_backend=backend, state_backend="preprocess",
            landmask=np.ones((cfg.ny, cfg.nx)), column_workers=1)
        result.state.set_map_coriolis(
            grid.mapfac_m(), grid.mapfac_u(), grid.mapfac_v(),
            *grid.coriolis_m(), sina=grid.rotation_m()[0], cosa=grid.rotation_m()[1])
        # The root window also couples this state into its first boundary
        # frame; that production path owns face averaging and strided copies.
        forcing = StateBoundaryFrames(spec_bdy_width=cfg.spec_bdy_width,
                                      spec_zone=cfg.spec_zone, relax_zone=cfg.relax_zone)
        forcing.add_state(result.state, index=0)
        del forcing
        del result, met
    cp.cuda.get_current_stream().synchronize()


def preparation_jobs(experiment=None):
    """Independent real op groups, suitable for concurrent cache warming."""
    domains = tuple(getattr(experiment, "domains", ()) or ())
    run = domains[0].run if domains else None
    return [(f"preparation:{'bounded' if bounded else 'cuda'}:{'specific' if specific else 'rh'}",
             dict(run=run, specific=specific, bounded=bounded))
            for bounded in (False, True) for specific in (True, False)]


def warm_preparation_job(options):
    """Each thread owns a stream and device; keep temporary allocations tiny."""
    import cupy as cp

    with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True):
        root_initialize(**options)
