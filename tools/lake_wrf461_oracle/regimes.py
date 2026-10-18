"""Regime column oracle for the WRF v4.6.1 CLM lake: native Fortran against
the production CUDA kernel, every word of every step.

The twelve-column fixture in gpuwm/data/lake/oracle holds its forcing fixed
for 300 steps.  This harness widens the evidence: 112 columns in 28 named
regimes (warm convective days, cold-air outbreaks, stable nights, freeze-up,
snow on ice, thaw, rain on snow, high plateaus at 600 hPa, desert pans,
polar ice, the tropics at zero latitude, gales, calm, and the edge cases of
the ice threshold, zero and default depth, and every soil category), driven
by forcing that changes every step: a diurnal cycle of sunshine and air
temperature, gusting and veering wind, longwave tied to the air temperature,
and timed rain or snow bursts.

Separate stages, so the GPU card is held only for the GPU run:

    python regimes.py forcing    OUT          # inputs, written once
    python regimes.py fortran    OUT BUILD    # native WRF, host libm (CPU)
    python regimes.py fortran-cr OUT BUILD    # native WRF, correctly rounded binary64 libm
    python regimes.py cpp        OUT BUILD    # WOOF's translation on the host
    python regimes.py gpu        OUT          # production kernel trace
    python regimes.py compare    OUT          # word-for-word report

``BUILD`` is the directory tools/lake_wrf461_oracle/build.sh filled.  The GPU
stage loads the kernel through gpuwm.core.kernels and steps it through
gpuwm.core.lake.launch_lake_columns, the forecast's own launcher.  Set
GPUWM_WRF_EXACT=1 for the strict build; leave it unset for default
arithmetic.  The trace files carry the mode in their names.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

from gpuwm.core.lake_schema import (  # noqa: E402
    LAKE_FORCING_WORDS, LAKE_OUTPUT_LAYOUT, LAKE_STATE_LAYOUT,
    LAKE_STATIC_LAYOUT, LAKE_STATE_WORDS, LAKE_STATIC_WORDS, LAKE_OUTPUT_WORDS)

XICE_THRESHOLD = 0.5
#: (dt seconds, steps): 24 h at a 60 s step (a full diurnal cycle) and 6 h
#: at 20 s, the HRRR-like convection-permitting step.
CAMPAIGNS = {"dt60": (60.0, 1440), "dt20": (20.0, 1080)}

# Regime: (name, count, dict of parameters).  Each regime spawns ``count``
# columns whose parameters are perturbed by a fixed seed.  Keys:
#   soil depth tw snow xice   seed (isltyp, lake depth m, tsk K, snow kg/m2, xice)
#   ta amp                    air temperature mean and diurnal half-range (K)
#   ps                        surface pressure (Pa)
#   rh                        relative humidity of the lowest level
#   wind gust                 mean wind and gust amplitude (m/s)
#   sw                        clear-sky peak shortwave (W/m2)
#   cloud                     longwave cloud term (W/m2 added to clear sky)
#   rain                      burst rate (mm/h), active in the bursts below
#   alb emis lat dz hour      input albedo, emissivity, latitude, dz8w, start hour
REGIMES = (
    ("convective_warm_day", 4, dict(soil=8, depth=20, tw=299, ta=297, amp=5, ps=99500, rh=.55, wind=2.5, gust=1.5, sw=950, cloud=10, rain=0, alb=.08, emis=.98, lat=33, dz=50, hour=6)),
    ("convective_thunderstorm", 4, dict(soil=3, depth=12, tw=301, ta=300, amp=6, ps=100200, rh=.7, wind=3, gust=6, sw=1000, cloud=40, rain=45, alb=.08, emis=.98, lat=28, dz=40, hour=10)),
    ("cold_air_outbreak", 4, dict(soil=5, depth=80, tw=281, ta=262, amp=2, ps=101500, rh=.75, wind=12, gust=5, sw=250, cloud=30, rain=2, alb=.08, emis=.98, lat=44, dz=60, hour=0)),
    ("stable_night_fog", 4, dict(soil=4, depth=10, tw=284, ta=294, amp=3, ps=100800, rh=.95, wind=1.2, gust=.4, sw=0, cloud=5, rain=0, alb=.1, emis=.97, lat=40, dz=30, hour=20)),
    ("stable_calm_zero_wind", 4, dict(soil=6, depth=6, tw=288, ta=296, amp=4, ps=100000, rh=.6, wind=0, gust=0, sw=300, cloud=0, rain=0, alb=.1, emis=.99, lat=52, dz=24, hour=22)),
    ("evening_transition", 4, dict(soil=7, depth=35, tw=292, ta=290, amp=7, ps=99800, rh=.65, wind=4, gust=2, sw=800, cloud=0, rain=0, alb=.07, emis=.98, lat=46, dz=50, hour=14)),
    ("tropical_deep_equator", 4, dict(soil=2, depth=250, tw=302.5, ta=300, amp=2.5, ps=100900, rh=.85, wind=3, gust=3, sw=1050, cloud=35, rain=25, alb=.06, emis=.98, lat=0, dz=60, hour=4)),
    ("southern_hemisphere", 4, dict(soil=9, depth=60, tw=290, ta=286, amp=6, ps=101200, rh=.6, wind=6, gust=3, sw=700, cloud=10, rain=4, alb=.08, emis=.98, lat=-38, dz=50, hour=8)),
    ("freeze_up", 4, dict(soil=10, depth=5, tw=275.5, ta=258, amp=4, ps=101800, rh=.7, wind=9, gust=4, sw=150, cloud=0, rain=0, alb=.1, emis=.98, lat=58, dz=40, hour=16)),
    ("ice_with_deep_snow", 4, dict(soil=11, depth=15, tw=268, ta=252, amp=5, ps=102000, rh=.75, wind=4, gust=2, sw=200, cloud=0, rain=0, alb=.75, emis=.99, lat=62, dz=30, hour=0, snow=120, xice=1.0)),
    ("ice_thin_snow_thaw", 4, dict(soil=12, depth=8, tw=271, ta=281, amp=7, ps=100500, rh=.6, wind=5, gust=2, sw=750, cloud=0, rain=0, alb=.55, emis=.98, lat=50, dz=40, hour=5, snow=20, xice=0.9)),
    ("snowfall_on_ice", 4, dict(soil=13, depth=25, tw=270, ta=264, amp=2, ps=100700, rh=.9, wind=6, gust=4, sw=120, cloud=60, rain=6, alb=.6, emis=.99, lat=55, dz=40, hour=9, snow=10, xice=1.0)),
    ("rain_on_snow", 4, dict(soil=1, depth=18, tw=271.5, ta=276.5, amp=1.5, ps=99600, rh=.97, wind=7, gust=5, sw=100, cloud=80, rain=8, alb=.5, emis=.99, lat=47, dz=30, hour=12, snow=60, xice=0.95)),
    ("high_plateau_open", 4, dict(soil=8, depth=40, tw=281, ta=276, amp=10, ps=60000, rh=.35, wind=7, gust=5, sw=1180, cloud=0, rain=0, alb=.06, emis=.98, lat=31, dz=60, hour=6)),
    ("high_plateau_frozen", 4, dict(soil=3, depth=30, tw=269, ta=260, amp=11, ps=64000, rh=.3, wind=9, gust=6, sw=1100, cloud=0, rain=0, alb=.4, emis=.98, lat=34, dz=60, hour=7, snow=5, xice=1.0)),
    ("andes_altiplano", 4, dict(soil=6, depth=100, tw=284, ta=280, amp=8, ps=63500, rh=.45, wind=5, gust=4, sw=1200, cloud=15, rain=10, alb=.06, emis=.98, lat=-16, dz=60, hour=8)),
    ("desert_shallow_hot", 4, dict(soil=1, depth=1.2, tw=304, ta=309, amp=8, ps=98000, rh=.15, wind=5, gust=4, sw=1050, cloud=0, rain=0, alb=.1, emis=.96, lat=24, dz=60, hour=4)),
    ("very_deep_rift", 4, dict(soil=4, depth=900, tw=278, ta=284, amp=6, ps=95000, rh=.5, wind=6, gust=3, sw=850, cloud=10, rain=0, alb=.07, emis=.98, lat=53, dz=50, hour=3)),
    ("polar_ice", 4, dict(soil=16, depth=20, tw=262, ta=238, amp=1.5, ps=101000, rh=.8, wind=6, gust=3, sw=0, cloud=0, rain=.5, alb=.8, emis=.99, lat=79, dz=30, hour=0, snow=40, xice=1.0)),
    ("polar_summer_open", 4, dict(soil=15, depth=10, tw=277, ta=279, amp=1, ps=100900, rh=.85, wind=5, gust=2, sw=550, cloud=20, rain=1, alb=.08, emis=.98, lat=72, dz=30, hour=0)),
    ("gale", 4, dict(soil=5, depth=45, tw=286, ta=282, amp=2, ps=98500, rh=.8, wind=24, gust=8, sw=200, cloud=70, rain=12, alb=.08, emis=.98, lat=48, dz=60, hour=0)),
    ("monsoon_heavy_rain", 4, dict(soil=7, depth=4, tw=300, ta=299, amp=1.5, ps=100300, rh=.95, wind=6, gust=8, sw=500, cloud=90, rain=80, alb=.08, emis=.98, lat=18, dz=40, hour=12)),
    ("xice_threshold_edge", 4, dict(soil=8, depth=12, tw=271, ta=268, amp=3, ps=101000, rh=.7, wind=4, gust=2, sw=300, cloud=0, rain=0, alb=.3, emis=.98, lat=56, dz=40, hour=10, snow=0, xice=0.5)),
    ("zero_depth_uses_default", 4, dict(soil=9, depth=0, tw=289, ta=291, amp=5, ps=100100, rh=.6, wind=3, gust=2, sw=800, cloud=0, rain=0, alb=.08, emis=.98, lat=41, dz=40, hour=6)),
    ("soil_14_water_remap", 4, dict(soil=14, depth=7, tw=287, ta=284, amp=5, ps=100400, rh=.6, wind=4, gust=2, sw=700, cloud=0, rain=2, alb=.08, emis=.98, lat=38, dz=40, hour=9)),
    ("snow_layer_thresholds", 4, dict(soil=2, depth=14, tw=270, ta=262, amp=4, ps=100600, rh=.7, wind=3, gust=2, sw=250, cloud=0, rain=0, alb=.7, emis=.99, lat=61, dz=40, hour=2, snow=4, xice=1.0)),
    ("supercooled_air_open_water", 4, dict(soil=12, depth=3, tw=278, ta=248, amp=2, ps=102500, rh=.6, wind=15, gust=6, sw=50, cloud=0, rain=1, alb=.08, emis=.98, lat=66, dz=24, hour=18)),
    ("tiny_dz_low_level", 4, dict(soil=10, depth=9, tw=291, ta=293, amp=6, ps=100000, rh=.55, wind=2, gust=1, sw=900, cloud=0, rain=0, alb=.08, emis=.98, lat=37, dz=8, hour=7)),
)

# Per-variant edits that pin edge values rather than perturb them.
EDGES = {
    "xice_threshold_edge": [dict(xice=0.5), dict(xice=np.nextafter(np.float32(0.5), np.float32(1))), dict(xice=0.49), dict(xice=1.0, snow=0)],
    "snow_layer_thresholds": [dict(snow=4.0), dict(snow=12.0), dict(snow=29.0), dict(snow=300.0)],
    "zero_depth_uses_default": [dict(depth=0.0), dict(depth=-5.0), dict(depth=0.1), dict(depth=50.0)],
    "stable_calm_zero_wind": [dict(wind=0, gust=0), dict(wind=0, gust=0, lat=0), dict(wind=.2, gust=.1), dict(wind=0, gust=0, sw=0)],
    "tropical_deep_equator": [dict(lat=0), dict(lat=-.5), dict(lat=.5), dict(lat=5)],
    "polar_ice": [dict(lat=79), dict(lat=89.9), dict(lat=-77), dict(lat=85)],
    "desert_shallow_hot": [dict(depth=1.2), dict(depth=0.5), dict(depth=2.0), dict(depth=1.0)],
}


def _qsat(t, p):
    """Saturation specific humidity (Bolton), only to give plausible inputs."""
    tc = t - 273.15
    es = 611.2 * np.exp(17.67 * tc / (tc + 243.5))
    return 0.622 * es / np.maximum(p - 0.378 * es, 1.0)


def columns():
    """Return the regime names and the per-column parameter table."""
    rng = np.random.default_rng(20261007)
    names, table = [], []
    for name, count, base in REGIMES:
        for k in range(count):
            p = dict(snow=0.0, xice=0.0)
            p.update(base)
            if k:  # variant 0 is the regime as written
                p["depth"] = p["depth"] * float(rng.uniform(.6, 1.6)) if p["depth"] > 0 else p["depth"]
                p["tw"] += float(rng.uniform(-1.5, 1.5)) if p["tw"] > 274 else float(rng.uniform(-2, 0))
                p["ta"] += float(rng.uniform(-3, 3))
                p["wind"] = max(0.0, p["wind"] * float(rng.uniform(.7, 1.3)))
                p["rh"] = float(np.clip(p["rh"] + rng.uniform(-.1, .1), .05, 1.0))
                p["lat"] = float(np.clip(p["lat"] + rng.uniform(-4, 4), -89.9, 89.9)) if p["lat"] else 0.0
                p["hour"] = (p["hour"] + 6 * k) % 24
                p["soil"] = int((p["soil"] + 3 * k - 1) % 16 + 1)
                if p["snow"] > 0:
                    p["snow"] *= float(rng.uniform(.5, 2.0))
            if name in EDGES:
                p.update(EDGES[name][k])
            names.append(f"{name}#{k}")
            table.append(p)
    return names, table


def build_inputs(dt, steps):
    """Seed (5, n) and forcing (steps, 13, n), both float32."""
    names, table = columns()
    n = len(table)
    seed = np.zeros((5, n), np.float32)
    forcing = np.zeros((steps, LAKE_FORCING_WORDS, n), np.float32)
    rng = np.random.default_rng(7)
    for c, p in enumerate(table):
        seed[:, c] = (p["soil"], p["depth"], p["tw"], p["snow"], p["xice"])
        t = np.arange(steps) * dt
        hour = (p["hour"] + t / 3600.0) % 24.0
        day = np.cos(2 * np.pi * (hour - 12.0) / 24.0)
        sw = p["sw"] * np.clip(day, 0, None) ** 1.3
        ta = p["ta"] + p["amp"] * np.cos(2 * np.pi * (hour - 15.0) / 24.0)
        # Gusts: two incommensurate periods plus a fixed noise series.
        noise = rng.normal(0, 1, steps)
        speed = np.maximum(0.0, p["wind"] + p["gust"] * (0.6 * np.sin(2 * np.pi * t / 1500.0)
                                                         + 0.4 * np.sin(2 * np.pi * t / 410.0))
                           + 0.15 * p["gust"] * noise)
        direction = np.deg2rad(200 + 60 * np.sin(2 * np.pi * t / 21600.0) + 15 * c)
        u, v = speed * np.cos(direction), speed * np.sin(direction)
        if p["wind"] == 0 and p["gust"] == 0:
            u, v = np.zeros(steps), np.zeros(steps)
        ps = p["ps"] + 150 * np.sin(2 * np.pi * t / 43200.0)
        rho = ps / (287.04 * ta)
        p2 = ps - rho * 9.81 * p["dz"]
        qv = p["rh"] * _qsat(ta, ps)
        qv = qv / (1 - qv)
        clear = 0.70 * 5.67e-8 * ta ** 4 * (1 + 0.15 * p["rh"])
        glw = clear + p["cloud"] * (1 + 0.3 * np.sin(2 * np.pi * t / 7200.0))
        # Bursts: 20 minutes on, then off, cycling with a 3 h period and a
        # column-specific phase; light drizzle columns rain throughout.
        phase = (t + 977.0 * c) % 10800.0
        burst = (phase < 1200.0).astype(float)
        if p["rain"] and p["rain"] < 1.0:
            burst[:] = 1.0
        rain = p["rain"] * burst * dt / 3600.0
        f = forcing[:, :, c]
        f[:, 0] = ta
        f[:, 1] = ps
        f[:, 2] = p2
        f[:, 3] = p["dz"]
        f[:, 4] = qv
        f[:, 5] = u
        f[:, 6] = v
        f[:, 7] = glw
        f[:, 8] = p["emis"]
        f[:, 9] = rain
        f[:, 10] = sw
        f[:, 11] = p["alb"]
        f[:, 12] = p["lat"]
    return names, seed, forcing


def fields(layout):
    out, start = [], 0
    for name, size in layout:
        out += [name if size == 1 else f"{name}[{k}]" for k in range(size)]
        start += size
    return out


STATE_ROWS = fields(LAKE_STATE_LAYOUT)
STATIC_ROWS = fields(LAKE_STATIC_LAYOUT)
OUTPUT_ROWS = fields(LAKE_OUTPUT_LAYOUT)


def _ptr(a, kind):
    return a.ctypes.data_as(ctypes.POINTER(kind))


class _Native:
    """One native library behind the kernel's argument list.

    ``lake_fortran.so`` is the WRF driver wrapper, whose WRF surface driver
    fixes xice_threshold at 0.5 and has no launch window.  ``lake_cpp.so``
    is the production translation compiled for the host (glibc libm, no
    contraction) and takes the kernel's own arguments; it loops every column
    itself and ignores the window and arena.
    """

    def __init__(self, path: Path):
        FP, IP = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
        self.dll = ctypes.CDLL(str(path))
        self.kernel_abi = not path.name.startswith("lake_fortran")
        if self.kernel_abi:
            self.dll.lake_init_columns.argtypes = [ctypes.c_int, FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_float,
                                                   ctypes.c_float, IP, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
            self.dll.lake_step_columns.argtypes = [ctypes.c_int, FP, FP, FP, FP, ctypes.c_float, ctypes.c_float, IP,
                                                   ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        else:
            self.dll.lake_init_columns.argtypes = [ctypes.c_int, FP, FP, FP, ctypes.c_int, ctypes.c_int, ctypes.c_float, IP]
            self.dll.lake_step_columns.argtypes = [ctypes.c_int, FP, FP, FP, FP, ctypes.c_float, IP]

    def lake_init_columns(self, n, seed, state, static, use_depth, depth_flag, default, errors):
        tail = (XICE_THRESHOLD, errors, 0, n, None) if self.kernel_abi else (errors,)
        self.dll.lake_init_columns(n, seed, state, static, use_depth, depth_flag, default, *tail)

    def lake_step_columns(self, n, forcing, state, static, output, dt, errors):
        tail = (XICE_THRESHOLD, errors, 0, n, None) if self.kernel_abi else (errors,)
        self.dll.lake_step_columns(n, forcing, state, static, output, dt, *tail)


def run_fortran(out: Path, build: Path, campaign: str, library: str = "lake_fortran.so", tag: str = "fortran"):
    data = np.load(out / f"inputs-{campaign}.npz")
    seed, forcing = data["seed"], data["forcing"]
    dt = float(data["dt"])
    dll = _Native(build / library)
    n = seed.shape[1]
    state = np.zeros((LAKE_STATE_WORDS, n), np.float32)
    static = np.zeros((LAKE_STATIC_WORDS, n), np.float32)
    output = np.zeros((LAKE_OUTPUT_WORDS, n), np.float32)
    errors = np.zeros(n, np.int32)
    seed = np.ascontiguousarray(seed)
    dll.lake_init_columns(n, _ptr(seed, ctypes.c_float), _ptr(state, ctypes.c_float), _ptr(static, ctypes.c_float),
                          1, 1, 50.0, _ptr(errors, ctypes.c_int))
    assert not errors.any(), errors
    init_state, init_static = state.copy(), static.copy()
    steps = forcing.shape[0]
    trace_state = np.zeros((steps,) + state.shape, np.float32)
    trace_output = np.zeros((steps,) + output.shape, np.float32)
    t0 = time.time()
    for s in range(steps):
        f = np.ascontiguousarray(forcing[s])
        dll.lake_step_columns(n, _ptr(f, ctypes.c_float), _ptr(state, ctypes.c_float), _ptr(static, ctypes.c_float),
                              _ptr(output, ctypes.c_float), dt, _ptr(errors, ctypes.c_int))
        assert not errors.any(), (s, errors)
        trace_state[s], trace_output[s] = state, output
    np.savez(out / f"{tag}-{campaign}.npz", init_state=init_state, init_static=init_static,
             trace_state=trace_state, trace_output=trace_output)
    return dict(library=library, campaign=campaign, columns=n, steps=steps, seconds=round(time.time() - t0, 1))


def run_init_controls_fortran(out: Path, build: Path, library: str = "lake_fortran.so", tag: str = "fortran"):
    data = np.load(out / "inputs-dt60.npz")
    seed = np.ascontiguousarray(data["seed"])
    dll = _Native(build / library)
    n = seed.shape[1]
    states, statics = [], []
    for default in (50.0, 0.0, -1.0, 7.5):
        state = np.zeros((LAKE_STATE_WORDS, n), np.float32)
        static = np.zeros((LAKE_STATIC_WORDS, n), np.float32)
        errors = np.zeros(n, np.int32)
        dll.lake_init_columns(n, _ptr(seed, ctypes.c_float), _ptr(state, ctypes.c_float),
                              _ptr(static, ctypes.c_float), 0, 0, default, _ptr(errors, ctypes.c_int))
        assert not errors.any()
        states.append(state); statics.append(static)
    np.savez(out / f"{tag}-init-controls.npz", defaults=np.float32([50, 0, -1, 7.5]),
             state=np.stack(states), static=np.stack(statics))


def mode():
    return "strict" if os.environ.get("GPUWM_WRF_EXACT", "0") == "1" else "default"


def run_gpu(out: Path, campaign: str):
    import cupy as cp
    import gpuwm  # noqa: F401  (installs the strict compiler hook when selected)
    from gpuwm.core.kernels import get_kernel, module_options, module_source
    from gpuwm.core.lake import lake_arena, launch_lake_columns
    data = np.load(out / f"inputs-{campaign}.npz")
    seed, forcing = data["seed"], data["forcing"]
    dt = np.float32(data["dt"])
    n = seed.shape[1]
    init = get_kernel("lake", "lake_init_columns")
    step = get_kernel("lake", "lake_step_columns")
    ds = cp.asarray(seed)
    state = cp.zeros((LAKE_STATE_WORDS, n), cp.float32)
    static = cp.zeros((LAKE_STATIC_WORDS, n), cp.float32)
    output = cp.zeros((LAKE_OUTPUT_WORDS, n), cp.float32)
    errors = cp.zeros(n, cp.int32)
    arena = lake_arena(n)
    thr = np.float32(XICE_THRESHOLD)
    launch_lake_columns(init, n, (n, ds, state, static, 1, 1, np.float32(50), thr, errors), arena)
    assert not cp.asnumpy(errors).any()
    init_state, init_static = cp.asnumpy(state), cp.asnumpy(static)
    steps = forcing.shape[0]
    trace_state = np.zeros((steps, LAKE_STATE_WORDS, n), np.float32)
    trace_output = np.zeros((steps, LAKE_OUTPUT_WORDS, n), np.float32)
    df = cp.asarray(forcing)
    t0 = time.time()
    for s in range(steps):
        launch_lake_columns(step, n, (n, df[s], state, static, output, dt, thr, errors), arena)
        trace_state[s] = cp.asnumpy(state)
        trace_output[s] = cp.asnumpy(output)
    err = cp.asnumpy(errors)
    import hashlib
    from gpuwm import wrf_exact
    meta = dict(campaign=campaign, mode=mode(), device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
                source_sha256=hashlib.sha256(module_source("lake").encode()).hexdigest(),
                options=list(module_options("lake")),
                effective_options=list(wrf_exact.effective_options(module_options("lake"))) if wrf_exact.ENABLED else None,
                errors=int(np.count_nonzero(err)), seconds=round(time.time() - t0, 1))
    np.savez(out / f"gpu-{mode()}-{campaign}.npz", init_state=init_state, init_static=init_static,
             trace_state=trace_state, trace_output=trace_output)
    controls = None
    if campaign == "dt60":
        states, statics = [], []
        for default in (50.0, 0.0, -1.0, 7.5):
            st = cp.zeros((LAKE_STATE_WORDS, n), cp.float32); sc = cp.zeros((LAKE_STATIC_WORDS, n), cp.float32)
            errors[...] = 0
            launch_lake_columns(init, n, (n, ds, st, sc, 0, 0, np.float32(default), thr, errors), arena)
            assert not cp.asnumpy(errors).any()
            states.append(cp.asnumpy(st)); statics.append(cp.asnumpy(sc))
        np.savez(out / f"gpu-{mode()}-init-controls.npz", state=np.stack(states), static=np.stack(statics))
    (out / f"gpu-{mode()}-{campaign}.json").write_text(json.dumps(meta, indent=2) + "\n")
    return meta


def ulp_distance(a, b):
    from gpuwm.core.fp32_ulp import monotone_fp32_key
    return np.abs(monotone_fp32_key(a) - monotone_fp32_key(b))


def compare_block(ref, act, rows, names, axis_step=True):
    """Per-field report over a (steps, rows, n) or (rows, n) block."""
    if ref.ndim == 2:
        ref, act = ref[None], act[None]
    diff = ref.view(np.uint32) != act.view(np.uint32)
    ulp = ulp_distance(ref, act)
    with np.errstate(invalid="ignore", over="ignore"):
        absd = np.abs(ref.astype(np.float64) - act.astype(np.float64))
    report = {}
    for r, field in enumerate(rows):
        d = diff[:, r, :]
        entry = dict(differing=int(d.sum()), words=int(d.size), max_ulp=int(ulp[:, r, :].max()),
                     max_abs=float(np.nanmax(absd[:, r, :])) if d.any() else 0.0)
        if d.any():
            s, c = np.argwhere(d)[0]
            entry.update(first_step=int(s) + 1, first_column=names[c], reference=float(ref[s, r, c]), actual=float(act[s, r, c]))
        report[field] = entry
    total = dict(differing=int(diff.sum()), words=int(diff.size), max_ulp=int(ulp.max()),
                 max_abs=float(np.nanmax(absd)) if diff.any() else 0.0,
                 nonfinite_reference=int((~np.isfinite(ref)).sum()))
    if diff.any():
        cols = np.unique(np.argwhere(diff)[:, 2])
        total["columns_differing"] = [names[c] for c in cols]
        total["first_step"] = int(np.argwhere(diff)[:, 0].min()) + 1
    return total, report


def compare(out: Path, modes=("gpu-strict", "gpu-default", "cpp"), references=("fortran", "fortran_cr")):
    """``gpu-strict``/``gpu-default`` are the production CUDA kernel under the
    two arithmetic builds; ``cpp`` is the same translation compiled for the
    host, which separates translation defects from device arithmetic.

    ``fortran`` is native WRF linked to the host libm (the reference);
    ``fortran_cr`` is the same native WRF linked to correctly rounded
    binary64 functions (a control that removes the host library's own
    rounding errors).  Keys are ``<actual>/<campaign>`` against ``fortran``
    and ``<actual>/<campaign>@cr`` against ``fortran_cr``."""
    names = json.loads((out / "columns.json").read_text())["names"]
    result = {}
    pairs = [(m, r) for r in references for m in modes]
    if "fortran_cr" in references:
        pairs.append(("fortran", "fortran_cr"))
    for m, reference in pairs:
        suffix = "" if reference == "fortran" else "@cr"
        for campaign in CAMPAIGNS:
            g = out / f"{m}-{campaign}.npz"
            r = out / f"{reference}-{campaign}.npz"
            if not g.exists() or not r.exists():
                continue
            ref, act = np.load(r), np.load(g)
            entry = {}
            entry["init_state"], _ = compare_block(ref["init_state"], act["init_state"], STATE_ROWS, names)
            entry["init_static"], _ = compare_block(ref["init_static"], act["init_static"], STATIC_ROWS, names)
            entry["state"], entry["state_fields"] = compare_block(ref["trace_state"], act["trace_state"], STATE_ROWS, names)
            entry["output"], entry["output_fields"] = compare_block(ref["trace_output"], act["trace_output"], OUTPUT_ROWS, names)
            result[f"{m}/{campaign}{suffix}"] = entry
        g = out / f"{m}-init-controls.npz"
        r = out / f"{reference}-init-controls.npz"
        if g.exists() and r.exists():
            ref, act = np.load(r), np.load(g)
            st, _ = compare_block(ref["state"], act["state"], STATE_ROWS, names)
            sc, _ = compare_block(ref["static"], act["static"], STATIC_ROWS, names)
            result[f"{m}/init-controls{suffix}"] = dict(defaults=ref["defaults"].tolist(), state=st, static=sc)
    (out / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    summary = {}
    for key, entry in result.items():
        if "output" in entry:
            summary[key] = {k: {kk: entry[k][kk] for kk in ("differing", "words", "max_ulp", "max_abs")}
                            for k in ("init_state", "init_static", "state", "output")}
        else:
            summary[key] = {k: {kk: entry[k][kk] for kk in ("differing", "words", "max_ulp")} for k in ("state", "static")}
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("forcing", "fortran", "fortran-cr", "cpp", "gpu", "compare"))
    p.add_argument("out", type=Path)
    p.add_argument("build", type=Path, nargs="?")
    p.add_argument("--campaign", choices=tuple(CAMPAIGNS) + ("all",), default="all")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    chosen = tuple(CAMPAIGNS) if a.campaign == "all" else (a.campaign,)
    if a.stage == "forcing":
        for c in chosen:
            dt, steps = CAMPAIGNS[c]
            names, seed, forcing = build_inputs(dt, steps)
            np.savez_compressed(a.out / f"inputs-{c}.npz", seed=seed, forcing=forcing, dt=np.float32(dt))
        regimes = sorted({n.split("#")[0] for n in names})
        (a.out / "columns.json").write_text(json.dumps(dict(names=names, regimes=regimes, count=len(names)), indent=2) + "\n")
        print(f"{len(names)} columns in {len(regimes)} regimes; finite forcing: "
              f"{bool(np.isfinite(forcing).all())}")
    elif a.stage == "fortran":
        for c in chosen:
            print(json.dumps(run_fortran(a.out, a.build, c)))
        run_init_controls_fortran(a.out, a.build)
    elif a.stage == "fortran-cr":
        for c in chosen:
            print(json.dumps(run_fortran(a.out, a.build, c, "lake_fortran_cr.so", "fortran_cr")))
        run_init_controls_fortran(a.out, a.build, "lake_fortran_cr.so", "fortran_cr")
    elif a.stage == "cpp":
        for c in chosen:
            print(json.dumps(run_fortran(a.out, a.build, c, "lake_cpp.so", "cpp")))
        run_init_controls_fortran(a.out, a.build, "lake_cpp.so", "cpp")
    elif a.stage == "gpu":
        for c in chosen:
            print(json.dumps(run_gpu(a.out, c)))
    else:
        print(json.dumps(compare(a.out), indent=2))
