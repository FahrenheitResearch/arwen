#!/usr/bin/env python3
"""Thompson on saved model columns: WRF v4.6.1's own Fortran against the
port's production adapter run on the host, process by process, for
mp_physics=28 (aerosol aware, the default) or mp_physics=8 (``--mp 8``:
WRF's same module with is_aerosol_aware false).

One run takes a column file written by ``extract_columns.py`` and

1. runs the PRISTINE WRF batch driver (``build_wrf.sh``) on the columns,
2. runs the RATE-INSTRUMENTED WRF driver and requires its outputs to be
   byte-identical to (1), keeping its five checkpoint streams,
3. runs the production adapter -- ``gpuwm.core.microphysics_aerosol.
   _apply_thompson_aerosol`` for mp=28, ``gpuwm.core.microphysics.
   _apply_thompson`` for mp=8 -- on the same float32 inputs through the
   host backend, snapshotting the state after every launcher that ends a
   WRF stage,
4. runs it again with the kernels' rate readback compiled in and requires
   the state to be byte-identical to (3),
5. compares every process rate, every stage state and tendency per
   species, the surface precipitation, reflectivity and effective radii.

Inputs to both codes are identical float32 arrays: the Exner function and
the layer depths are formed once, with the adapter's own expressions, and
handed to WRF.  An mp=8 run reads the same column files and ignores their
droplet and aerosol numbers, which classic Thompson does not carry.

usage:
  real_column_parity.py COLUMNS.npz WRF_BUILD_DIR OUT_DIR [--max-cols N]
      [--seed S] [--keep-streams] [--mp {28,8}]

WRF_BUILD_DIR is the directory ``build_wrf.sh`` wrote.  OUT_DIR receives
``summary.json`` (every metric) and ``worst.json`` (the worst cells).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT), str(HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import host_backend  # noqa: E402

host_backend.install()

from instrument_wrf_rates import (  # noqa: E402
    EXIT_FIELDS, LATE_RATES, RATES, SCHEMA, TENDENCIES,
)
import port_rates  # noqa: E402

f32 = np.float32

SPECIES = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc",
           "nwfa", "nifa")
#: Classic Thompson's prognostic set: no droplet or aerosol number.
SPECIES_MP8 = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr")
SPECIES_BY_MP = {28: SPECIES, 8: SPECIES_MP8}
#: Below these magnitudes a value is "empty" for relative metrics: R1 for
#: masses (module_mp_thompson.F:183), and 1 per kg for the numbers.
FLOOR = {"qv": 1.0e-12, "qc": 1.0e-12, "qr": 1.0e-12, "qi": 1.0e-12,
         "qs": 1.0e-12, "qg": 1.0e-12, "ni": 1.0, "nr": 1.0, "nc": 1.0,
         "nwfa": 1.0, "nifa": 1.0, "T": 0.0, "ng": 1.0e-3}
#: WRF tendency name for each species / temperature.
TEN = {"qv": "qvten", "qc": "qcten", "qr": "qrten", "qi": "qiten",
       "qs": "qsten", "qg": "qgten", "ni": "niten", "nr": "nrten",
       "nc": "ncten", "nwfa": "nwfaten", "nifa": "nifaten", "T": "tten",
       "ng": "ngten"}
ONE_D = {"qv": "qv1d", "qc": "qc1d", "qr": "qr1d", "qi": "qi1d",
         "qs": "qs1d", "qg": "qg1d", "ni": "ni1d", "nr": "nr1d",
         "nc": "nc1d", "nwfa": "nwfa1d", "nifa": "nifa1d", "T": "t1d",
         "ng": "ng1d"}

#: Port launchers after which the state equals a WRF stage.
STAGE_LAUNCHERS = (
    ("cold", "gpuwm.core.thompson_aerosol_cold",
     "launch_aa_cold_network_from_owner"),
    ("warm", "gpuwm.core.thompson_aerosol_warm",
     "launch_aerosol_warm_source_network_from_owner"),
    ("sources", "gpuwm.core.thompson_aerosol_warm", "launch_ncten_balance"),
    ("condensation", "gpuwm.core.thompson_aerosol_sat",
     "launch_aerosol_saturation_adjust"),
    ("rain_evaporation", "gpuwm.core.thompson_aerosol_sat",
     "launch_aerosol_rain_evaporation"),
    ("sedimentation", "gpuwm.core.thompson_aerosol_sed",
     "launch_aa_rain_sedimentation_accumulate"),
    ("cleanup", "gpuwm.core.thompson_aerosol_sed",
     "launch_aa_final_phase_cleanup"),
    ("finalize", "gpuwm.core.thompson_aerosol_state",
     "launch_aerosol_state_finalize"),
)
#: The same stage boundaries in the mp=8 adapter.  Its sources end with the
#: warm network (no droplet-number balance), and its last write of the
#: state is the private graupel number's finalize.
STAGE_LAUNCHERS_MP8 = (
    ("cold", "gpuwm.core.thompson",
     "launch_frozen_vapor_network_from_owner"),
    ("sources", "gpuwm.core.thompson",
     "launch_warm_frozen_source_network_from_owner"),
    ("condensation", "gpuwm.core.thompson", "launch_cloud_saturation_adjust"),
    ("rain_evaporation", "gpuwm.core.thompson", "launch_rain_evaporation"),
    ("sedimentation", "gpuwm.core.thompson", "launch_rain_sedimentation"),
    ("cleanup", "gpuwm.core.thompson", "launch_final_phase_cleanup"),
    ("finalize", "gpuwm.core.thompson",
     "launch_classic_graupel_number_finalize"),
)
STAGE_LAUNCHERS_BY_MP = {28: STAGE_LAUNCHERS, 8: STAGE_LAUNCHERS_MP8}
#: WRF checkpoint whose ``X1d + Xten*DT`` equals the port state after a
#: stage.  Only the two stages before any fallout: the port folds WRF's
#: terminal rain and ice apply (:3972-4082, the size bounds and number
#: rediagnosis) into its sedimentation kernels, so after sedimentation the
#: two codes' working values are different quantities by construction and
#: the comparison moves to the final state (``final`` and ``final_clean``).
STAGE_CHECKPOINT = {"sources": "cp1", "rain_evaporation": "cp2"}
#: Relative difference a column may carry into sedimentation and still count
#: as "clean" for ``final_clean``.
CLEAN_REL = 2.0e-6


# ---------------------------------------------------------------------------
# Inputs.
# ---------------------------------------------------------------------------

def load_columns(path, max_cols=None, seed=0):
    z = np.load(path)
    cols = {k: z[k] for k in z.files}
    ncol = cols["p"].shape[0]
    if max_cols and ncol > max_cols:
        rng = np.random.default_rng(seed)
        keep = np.sort(rng.choice(ncol, size=max_cols, replace=False))
        for k, v in list(cols.items()):
            if isinstance(v, np.ndarray) and v.ndim >= 1 and v.shape[0] == ncol:
                cols[k] = v[keep]
    return cols


def prepare(cols, mp=28):
    """The float32 arrays both codes receive, formed as the adapter forms
    them (gpuwm/core/microphysics_aerosol.py and microphysics.py alike:
    ``pii = (p/P0)**RCP``, ``z8w = (phb + php)/G`` with the whole
    geopotential in ``php``, ``dz = z8w[1:] - z8w[:-1]``)."""
    from gpuwm.core import constants as C
    p = np.ascontiguousarray(cols["p"], dtype=f32)
    geop = np.ascontiguousarray(cols["geop"], dtype=f32)
    zero = np.zeros_like(geop)
    z8w = (zero + geop) / f32(C.G)
    inp = {
        "th": np.ascontiguousarray(cols["th"], dtype=f32),
        "p": p,
        "pii": np.power(p / f32(C.P0), f32(C.RCP)),
        "geop": geop,
        "z8w": z8w,
        "dz": np.ascontiguousarray(z8w[:, 1:] - z8w[:, :-1]),
        "hgt": np.ascontiguousarray(z8w[:, :-1]),
        "w": np.ascontiguousarray(cols["w"], dtype=f32),
    }
    if mp == 28:
        inp["nwfa2d"] = np.ascontiguousarray(cols["nwfa2d"], dtype=f32)
        inp["nifa2d"] = np.ascontiguousarray(cols["nifa2d"], dtype=f32)
    for s in SPECIES_BY_MP[mp]:
        inp[s] = np.ascontiguousarray(cols[s], dtype=f32)
    inp["T"] = inp["th"] * inp["pii"]
    return inp


# ---------------------------------------------------------------------------
# WRF.
# ---------------------------------------------------------------------------

WRF_IN = ("th", "pii", "p", "w_lower", "dz", "hgt", *SPECIES)
WRF_OUT3 = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "nc", "nwfa",
            "nifa", "th", "refl", "re_cloud", "re_ice", "re_snow")
WRF_OUT2 = ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
            "graupelncv", "sr")
#: run_columns_classic.F90's streams.
WRF_IN_MP8 = ("th", "pii", "p", "w_lower", "dz", "hgt", *SPECIES_MP8)
WRF_OUT3_MP8 = ("qv", "qc", "qr", "qi", "qs", "qg", "ni", "nr", "th",
                "refl", "re_cloud", "re_ice", "re_snow")
WRF_IN_BY_MP = {28: WRF_IN, 8: WRF_IN_MP8}
WRF_OUT3_BY_MP = {28: WRF_OUT3, 8: WRF_OUT3_MP8}
#: The driver each scheme's WRF answers come from (build_wrf.sh).
WRF_BINARY = {28: "run_columns_aero", 8: "run_columns_classic"}


def write_wrf_input(path, inp, dt, mp=28):
    ncol, nz = inp["p"].shape
    with open(path, "wb") as fh:
        np.array([ncol, nz], dtype="<i4").tofile(fh)
        np.array([dt], dtype="<f4").tofile(fh)
        for name in WRF_IN_BY_MP[mp]:
            a = inp["w"][:, :nz] if name == "w_lower" else inp[name]
            # Fortran buf(ncol, nz): column index fastest.
            np.ascontiguousarray(a.T, dtype="<f4").tofile(fh)
        if mp == 28:
            for name in ("nwfa2d", "nifa2d"):
                np.ascontiguousarray(inp[name], dtype="<f4").tofile(fh)


def read_wrf_output(path, ncol, nz, mp=28):
    raw = np.fromfile(path, dtype="<f4")
    out3 = WRF_OUT3_BY_MP[mp]
    want = len(out3) * ncol * nz + len(WRF_OUT2) * ncol
    if raw.size != want:
        raise RuntimeError(f"{path}: {raw.size} words, expected {want}")
    out, off = {}, 0
    for name in out3:
        out[name] = raw[off:off + ncol * nz].reshape(nz, ncol).T.copy()
        off += ncol * nz
    for name in WRF_OUT2:
        out[name] = raw[off:off + ncol].copy()
        off += ncol
    return out


def run_wrf(binary, run_dir, in_path, out_path):
    env = dict(os.environ, GFORTRAN_CONVERT_UNIT="big_endian:20")
    done = subprocess.run([str(binary), str(in_path), str(out_path)],
                          cwd=str(run_dir), env=env, capture_output=True,
                          text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"{binary} failed:\n{done.stdout}\n{done.stderr}")


def read_checkpoints(run_dir, ncol, nz):
    """``{cp: {field: (ncol, nz) float64}}`` with NaN where WRF wrote no
    record (columns ``mp_thompson`` left at :2020, ``no_micro``)."""
    out = {}
    for cp, fields in SCHEMA.items():
        raw = np.fromfile(Path(run_dir) / f"wrf-{cp}.bin", dtype="<f8")
        nf = len(fields)
        if raw.size % nf:
            raise RuntimeError(f"wrf-{cp}.bin is not a whole number of rows")
        rows = raw.reshape(-1, nf)
        col = rows[:, 0].astype(np.int64) - 1
        lev = rows[:, 1].astype(np.int64) - 1
        table = {}
        for j, name in enumerate(fields[2:], start=2):
            a = np.full((ncol, nz), np.nan)
            a[col, lev] = rows[:, j]
            table[name] = a
        present = np.zeros(ncol, bool)
        present[col] = True
        table["present"] = present
        out[cp] = table
    return out


# ---------------------------------------------------------------------------
# The port, on the host.
# ---------------------------------------------------------------------------

class ColumnState:
    """The mp=28 ``DomainState`` surface over ``(nz, 1, ncol)`` arrays.

    The scratch protocol is gpuwm/core/state.py's (shape and dtype pinned
    per slot); the attribute set is the one
    tests/test_thompson_aerosol_adapter.py's ``_ColumnState`` drives the
    adapter with.  The whole potential temperature sits in ``thp`` over a
    zero ``thb`` and the whole geopotential in ``php`` over a zero ``phb``,
    so the adapter's own sums reproduce the arrays WRF was given bit for
    bit.
    """

    def __init__(self, inp, mp=28):
        ncol, nz = inp["p"].shape

        def vol(a):
            # Always a COPY: for a single column ``a.T`` is already
            # contiguous and ascontiguousarray would hand back a view, so
            # the adapter's in-place writes would reach the caller's inputs.
            return np.array(a.T.reshape(a.shape[1], 1, ncol), dtype=f32,
                            order="C", copy=True)

        self.p = vol(inp["p"])
        self.thb = np.zeros((nz,), f32)
        self.thp = vol(inp["th"])
        self.phb = np.zeros((nz + 1,), f32)
        self.php = vol(inp["geop"])
        self.w = vol(inp["w"])
        for s in SPECIES_BY_MP[mp]:
            setattr(self, s, vol(inp[s]))
        self.effc = np.zeros((nz, 1, ncol), f32)
        self.effi = np.zeros((nz, 1, ncol), f32)
        self.effs = np.zeros((nz, 1, ncol), f32)
        if mp == 28:
            self.nwfa2d = np.array(inp["nwfa2d"].reshape(1, ncol),
                                   dtype=f32, copy=True)
            self.nifa2d = np.array(inp["nifa2d"].reshape(1, ncol),
                                   dtype=f32, copy=True)
        self.h_diabatic = np.zeros((nz, 1, ncol), f32)
        self.physics = SimpleNamespace(refl_10cm=None, state=self)
        self._scratch = {}

    def scratch(self, shape, slot, dtype=None):
        shape = tuple(shape)
        want = np.dtype(f32 if dtype is None else dtype)
        value = self._scratch.get(slot)
        if value is None:
            value = np.zeros(shape, dtype=want)
            self._scratch[slot] = value
        elif value.shape != shape or value.dtype != want:
            raise ValueError(f"scratch slot {slot!r} reused with a new "
                             f"shape or dtype")
        return value

    def existing_scratch(self, slot):
        return self._scratch.get(slot)


def _cols(a):
    """``(nz, 1, ncol)`` -> ``(ncol, nz)`` float64."""
    a = np.asarray(a)
    return a.reshape(a.shape[0], -1).T.astype(np.float64)


def _snapshot(state, dt=None, species=SPECIES):
    """The port's working state.  Until the terminal apply the port holds
    the qc/nc/nwfa/nifa tendencies (and, on the v4.6.1 generation, the
    qr/nr/qi/ni ones) in per-kilogram accumulators and leaves the state
    arrays at their entry values, so a mid-call snapshot reports
    ``X + Xten*dt`` for those, in float32 the way WRF forms it."""
    snap = {s: _cols(getattr(state, s)) for s in species}
    if dt is not None:
        # Mid-call the port carries WRF's RUNNING vapour qv1d + DT*qvten,
        # which WRF floors only at the terminal apply (:3974); the working
        # value every block reads is MAX(1.E-10, ...) (:3192, :3488, :3569),
        # and that is what WRF's checkpoints are compared as.
        snap["qv"] = np.maximum(snap["qv"], 1.0e-10)
    snap["T"] = _cols(state._scratch["mp_thompson_temperature"])
    if "mp_thompson_graupel_number_shadow" in state._scratch:
        snap["ng"] = _cols(state._scratch["mp_thompson_graupel_number_shadow"])
    for acc, var in (("qcten", "qc"), ("ncten", "nc"), ("nwfaten", "nwfa"),
                     ("nifaten", "nifa"), ("qrten", "qr"), ("nrten", "nr"),
                     ("qiten", "qi"), ("niten", "ni")):
        slot = f"mp_thompson_aero_{acc}"
        if slot in state._scratch:
            snap[acc] = _cols(state._scratch[slot])
            if dt is not None:
                entry = np.asarray(getattr(state, var), f32)
                ten = np.asarray(state._scratch[slot], f32)
                snap[var] = _cols(entry + ten * f32(dt))
    for sfc in ("rainnc", "snownc", "graupelnc"):
        slot = f"mp_{sfc}"
        if slot in state._scratch:
            snap[sfc] = np.asarray(state._scratch[slot]).ravel().astype(
                np.float64)
    return snap


def run_port(inp, dt, *, rates=False, mp=28, thompson_version="wrf_461",
             thompson_fork_snow_fall="blend"):
    """One production adapter call on the host.  Returns
    ``{"final", "stages", "rates"}``.  ``thompson_version`` names the
    Thompson generation the adapter runs (``RunConfig.thompson_version``)."""
    import importlib

    if mp == 28:
        from gpuwm.core.microphysics_aerosol import (
            _apply_thompson_aerosol as apply)
    else:
        from gpuwm.core.microphysics import _apply_thompson as apply

    # Every scheme's instrumented modules are cleared first, so a run of one
    # scheme never compiles the other's readback into a shared unit.
    for scheme in port_rates.ANCHORS_BY_MP:
        for module in port_rates.instrumented_modules(scheme):
            host_backend.set_transform(module, None)
    if rates:
        host_backend.set_rates(True)
        for module in port_rates.instrumented_modules(mp):
            host_backend.set_transform(
                module, lambda text, m=module: port_rates.instrument(
                    m, text, mp))
    else:
        host_backend.set_rates(False)

    species = SPECIES_BY_MP[mp]
    state = ColumnState(inp, mp)
    ncol, nz = inp["p"].shape
    cells = nz * ncol
    buffer = None
    if rates:
        buffer = np.full((port_rates.NSLOTS, cells), np.nan)
        buffer[:len(RATES)] = 0.0
        for module in port_rates.instrumented_modules(mp):
            lib = host_backend.build_module(module).lib
            lib.gpuwm_host_rate_bind.argtypes = [
                np.ctypeslib.ndpointer(np.float64, flags="C_CONTIGUOUS"),
                __import__("ctypes").c_longlong,
                __import__("ctypes").c_int]
            lib.gpuwm_host_rate_bind(buffer, cells, port_rates.NSLOTS)

    stages = {}
    originals = []
    for stage, module_name, attr in STAGE_LAUNCHERS_BY_MP[mp]:
        module = importlib.import_module(module_name)
        original = getattr(module, attr)
        originals.append((module, attr, original))

        def wrapped(*args, _orig=original, _stage=stage, **kwargs):
            result = _orig(*args, **kwargs)
            stages[_stage] = _snapshot(
                state, None if _stage == "finalize" else dt, species)
            return result

        setattr(module, attr, wrapped)
    try:
        cfg = SimpleNamespace(mp_physics=mp, no_mp_heating=0,
                              mp_tend_lim=10.0,
                              thompson_version=thompson_version,
                              thompson_fork_snow_fall=thompson_fork_snow_fall)
        diag = apply(state, cfg, float(dt), refl_10cm_due=True)
    finally:
        for module, attr, original in originals:
            setattr(module, attr, original)
        if rates:
            for module in port_rates.instrumented_modules(mp):
                lib = host_backend.build_module(module).lib
                lib.gpuwm_host_rate_bind(np.zeros((1, 1)), 0, 0)

    final = _snapshot(state, species=species)
    final["th"] = _cols(state._scratch["mp_th"])
    final["refl"] = _cols(state.physics.refl_10cm)
    final["re_cloud"] = _cols(state.effc)
    final["re_ice"] = _cols(state.effi)
    final["re_snow"] = _cols(state.effs)
    for name in ("rainnc", "rainncv", "snownc", "snowncv", "graupelnc",
                 "graupelncv", "sr"):
        final[name] = np.asarray(getattr(diag, name)).ravel().astype(
            np.float64)
    out = {"final": final, "stages": stages}
    if rates:
        out["rates"] = {name: buffer[i].reshape(nz, ncol).T.copy()
                        for i, name in enumerate(RATES)}
        for name in port_rates.PER_STEP[mp]:
            out["rates"][name] = out["rates"][name] / float(dt)
        out["diagnostics"] = {
            name: buffer[port_rates.SLOT[name]].reshape(nz, ncol).T.copy()
            for name in port_rates.DIAGNOSTICS}
    return out


# ---------------------------------------------------------------------------
# Metrics.
# ---------------------------------------------------------------------------

def ulp32(x):
    x = np.abs(np.asarray(x, np.float64)).astype(f32)
    return (np.nextafter(x, np.float32(np.inf)) - x).astype(np.float64)


def compare(port, wrf, *, floor=0.0, mask=None, where=None):
    """Agreement of two same-shaped arrays over cells where either side
    exceeds ``floor`` (and ``mask``)."""
    a = np.asarray(port, np.float64)
    b = np.asarray(wrf, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= mask
    big = np.maximum(np.abs(a), np.abs(b))
    active = ok & (big > floor)
    n = int(active.sum())
    res = {"n_active": n}
    if n == 0:
        return res
    d = np.abs(a - b)[active]
    m = big[active]
    rel = d / m
    ulps = d / np.maximum(ulp32(m), np.finfo(np.float64).tiny)
    res.update({
        "n_exact": int((d == 0).sum()),
        "rel_median": float(np.median(rel)),
        "rel_p99": float(np.quantile(rel, 0.99)),
        "rel_p999": float(np.quantile(rel, 0.999)),
        "rel_max": float(rel.max()),
        "ulp32_p999": float(np.quantile(ulps, 0.999)),
        "ulp32_max": float(ulps.max()),
        "abs_diff_max": float(d.max()),
        "scale_p99": float(np.quantile(m, 0.99)),
        "n_rel_gt_2e-6": int((rel > 2e-6).sum()),
        "n_rel_gt_1e-4": int((rel > 1e-4).sum()),
        "n_rel_gt_1e-2": int((rel > 1e-2).sum()),
    })
    idx = np.argwhere(active)
    worst = int(np.argmax(rel))
    loc = tuple(int(v) for v in idx[worst])
    res["worst"] = {"cell": loc, "port": float(a[loc]), "wrf": float(b[loc]),
                    "rel": float(rel[worst])}
    far = rel > 1e-2
    if far.any():
        # how large the far-off cells are next to the quantity's own scale
        res["far_abs_max"] = float(m[far].max())
    if where is not None:
        res["worst"]["where"] = where(loc)
    return res


#: Fields nudged by one float32 unit for the sensitivity runs: every
#: microphysics input except the surface emission rates.
PERTURBED = ("th", "p", "geop", "w", *SPECIES)
#: Draws of the one-unit nudge.  Each draw picks an independent random sign
#: per cell and field; the sensitivity is the largest response over draws.
SENSITIVITY_DRAWS = 4
#: A cell beyond the 2e-6 rounding gate is "within rounding" when its gap is
#: at most this many times the quantity's own response to a one-unit nudge of
#: every input in the same cell.
SENSITIVITY_FACTOR = 4.0
#: The relative gate for float32 rounding: the port's own end-to-end gate on
#: the 22 committed WRF fixtures, about 17 float32 units.
ROUNDING_REL = 2.0e-6


def perturb_columns(cols, seed):
    """A copy of ``cols`` with every PERTURBED field moved by one float32
    unit toward +inf or -inf, sign drawn per cell.  Zeros stay zero, so no
    empty category gains mass."""
    rng = np.random.default_rng(seed)
    out = dict(cols)
    for key in PERTURBED:
        a = np.asarray(cols[key], dtype=f32)
        up = rng.random(a.shape) < 0.5
        moved = np.where(up, np.nextafter(a, f32(np.inf)),
                         np.nextafter(a, f32(-np.inf))).astype(f32)
        out[key] = np.where(a == 0, a, moved).astype(f32)
    return out


def sensitivity(cols, dt, base, mp=28):
    """Largest response of every rate and final quantity to a one-unit nudge
    of every input, over SENSITIVITY_DRAWS draws, against ``base`` (the
    rate-instrumented port run on the unperturbed inputs).  The draws nudge
    every PERTURBED field whichever scheme runs, so the two schemes see the
    same nudge of every field they share."""
    sens = {"rates": {n: np.zeros_like(v) for n, v in base["rates"].items()},
            "final": {}}
    for draw in range(SENSITIVITY_DRAWS):
        pert = run_port(prepare(perturb_columns(cols, 1000 + draw), mp), dt,
                        rates=True, mp=mp)
        for name, value in base["rates"].items():
            d = np.abs(np.nan_to_num(pert["rates"][name])
                       - np.nan_to_num(value))
            sens["rates"][name] = np.maximum(sens["rates"][name], d)
        for name, value in base["final"].items():
            d = np.abs(np.asarray(pert["final"][name], np.float64)
                       - np.asarray(value, np.float64))
            prev = sens["final"].get(name)
            sens["final"][name] = d if prev is None else np.maximum(prev, d)
    return sens


#: A final-state cell beyond the rounding gate is also "within rounding"
#: when its gap is at most this many float32 units of the largest value the
#: cell held anywhere in the call (entry, every stage of both codes, both
#: finals).  That is cancellation: a level nearly emptied in one step keeps a
#: residual of the size of one unit of what it held, and the two codes' residuals
#: differ by that much (or one code clamps it to zero and the other carries it).
CELL_SCALE_UNITS = 4.0


def classify(port, wrf, sens, *, floor=0.0, mask=None, scale=None):
    """Counts of cells beyond the rounding gate, split by whether the gap is
    within SENSITIVITY_FACTOR times the one-unit response or, where ``scale``
    is given, within CELL_SCALE_UNITS float32 units of the cell's scale."""
    a = np.asarray(port, np.float64)
    b = np.asarray(wrf, np.float64)
    s = np.asarray(sens, np.float64)
    ok = np.isfinite(a) & np.isfinite(b)
    if mask is not None:
        ok &= mask
    big = np.maximum(np.abs(a), np.abs(b))
    active = ok & (big > floor)
    gap = np.abs(a - b)
    rel = np.where(active, gap / np.where(big > 0, big, 1.0), 0.0)
    beyond = active & (rel > ROUNDING_REL)
    by_sens = beyond & (gap <= SENSITIVITY_FACTOR * s)
    by_scale = np.zeros_like(beyond)
    if scale is not None:
        cell = np.maximum(np.nan_to_num(np.asarray(scale, np.float64)), big)
        by_scale = beyond & (gap <= CELL_SCALE_UNITS * ulp32(cell))
    within = by_sens | by_scale
    far = beyond & (rel > 1.0e-2)
    out = {"n_active": int(active.sum()),
           "n_beyond_rounding": int(beyond.sum()),
           "n_beyond_within_sensitivity": int(by_sens.sum()),
           "n_beyond_within_cell_scale": int((by_scale & ~by_sens).sum()),
           "n_beyond_unexplained": int((beyond & ~within).sum()),
           "n_beyond_1e-2": int(far.sum()),
           "n_beyond_1e-2_unexplained": int((far & ~within).sum())}
    if (beyond & ~within).any():
        un = beyond & ~within
        out["unexplained_rel_max"] = float(rel[un].max())
        out["unexplained_abs_max"] = float(gap[un].max())
    return out


def rain_mean_volume_diameter(qr1d, nr1d, rho):
    """WRF's entry rain mean volume diameter (:1878-1898): the number is
    re-diagnosed at 1 mm where it arrives at or below R2 per cubic metre,
    and the diameter is held between 0.75*D0r and 2.5 mm."""
    rr = np.asarray(qr1d, np.float64) * rho
    nn = np.asarray(nr1d, np.float64) * rho
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lam = np.cbrt(np.pi * 1000.0 * np.maximum(nn, 1.0e-6)
                      / np.where(rr > 0, rr, 1.0))
        mvd = np.where(nn <= 1.0e-6, 1.0e-3, 3.672 / lam)
    return np.clip(mvd, 0.75 * 50.0e-6, 2.5e-3)


def wrf_stage(cps, cp, var, dt):
    """WRF's own working value ``X1d + Xten*DT`` at checkpoint ``cp``, in
    float32 exactly as mp_thompson forms it (``qv`` floored at 1e-10)."""
    one_d = cps["cp1"][ONE_D[var]].astype(f32)
    ten = cps[cp][TEN[var]].astype(f32)
    value = (one_d + ten * f32(dt)).astype(np.float64)
    if var == "qv":
        value = np.maximum(value, 1.0e-10)
    return value


# ---------------------------------------------------------------------------
# Driver.
# ---------------------------------------------------------------------------

def run(columns, build, out_dir, *, max_cols=None, seed=0,
        keep_streams=False, dump=False, mp=28):
    t0 = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    build = Path(build)
    run_dir = build / "run"
    cols = load_columns(columns, max_cols=max_cols, seed=seed)
    dt = float(cols["dt"])
    inp = prepare(cols, mp)
    species = SPECIES_BY_MP[mp]
    ncol, nz = inp["p"].shape
    col_j = cols.get("col_j")
    col_i = cols.get("col_i")

    def where(loc):
        c = int(loc[0])
        info = {"column": c}
        if col_j is not None:
            info.update(j=int(col_j[c]), i=int(col_i[c]))
        if len(loc) > 1:
            info["k"] = int(loc[1])
        return info

    summary = {"columns": str(columns), "ncol": ncol, "nz": nz, "dt": dt,
               "mp_physics": mp,
               "kind": str(cols.get("kind", "")), "timings_s": {}}

    # 1-2. WRF, pristine and instrumented.
    in_path = run_dir / "parity-in.bin"
    write_wrf_input(in_path, inp, dt, mp)
    binary = WRF_BINARY[mp]
    t = time.time()
    run_wrf(build / "pristine" / binary, run_dir, in_path,
            run_dir / "parity-pristine.out")
    summary["timings_s"]["wrf_pristine"] = round(time.time() - t, 2)
    t = time.time()
    run_wrf(build / "rates" / binary, run_dir, in_path,
            run_dir / "parity-rates.out")
    summary["timings_s"]["wrf_rates"] = round(time.time() - t, 2)
    same = (run_dir / "parity-pristine.out").read_bytes() == (
        run_dir / "parity-rates.out").read_bytes()
    summary["wrf_instrumentation_neutral"] = bool(same)
    if not same:
        raise RuntimeError("instrumented WRF changed its outputs")
    wrf = read_wrf_output(run_dir / "parity-pristine.out", ncol, nz, mp)
    cps = read_checkpoints(run_dir, ncol, nz)

    # 3-4. The port, pristine and rate-instrumented.
    t = time.time()
    port = run_port(inp, dt, rates=False, mp=mp)
    summary["timings_s"]["port"] = round(time.time() - t, 2)
    t = time.time()
    port_r = run_port(inp, dt, rates=True, mp=mp)
    summary["timings_s"]["port_rates"] = round(time.time() - t, 2)
    neutral = all(np.array_equal(port["final"][k], port_r["final"][k],
                                 equal_nan=True) for k in port["final"])
    summary["port_instrumentation_neutral"] = bool(neutral)
    if not neutral:
        raise RuntimeError("instrumented port kernels changed the state")

    micro = cps["cp1"]["present"]
    summary["columns_with_microphysics"] = int(micro.sum())

    # 4b. The port's own response to a one-unit nudge of every input.
    t = time.time()
    sens = sensitivity(cols, dt, port_r, mp)
    summary["timings_s"]["sensitivity"] = round(time.time() - t, 2)

    # 5a. Process rates.  A rate the scheme's port does not carry
    # (port_rates.NOT_CARRIED) is reported with WRF's own activity only.
    rates = {}
    not_carried = {}
    cold = cps["cp1"]["t1d"] < 273.15
    for name in RATES:
        cp = "cp2" if name in LATE_RATES else "cp1"
        wrf_v = cps[cp][name]
        fin = np.isfinite(wrf_v)
        if name in port_rates.NOT_CARRIED[mp]:
            not_carried[name] = {"n_wrf_nonzero": int(
                (fin & (wrf_v != 0)).sum())}
            continue
        port_v = np.where(np.isnan(wrf_v), np.nan, port_r["rates"][name])
        entry = compare(port_v, wrf_v, floor=0.0, where=where)
        entry["n_wrf_nonzero"] = int((fin & (wrf_v != 0)).sum())
        entry["n_port_nonzero"] = int((fin & (port_v != 0)).sum())
        entry["n_cold_active"] = int((fin & (wrf_v != 0) & cold).sum())
        entry["n_warm_active"] = int((fin & (wrf_v != 0) & ~cold).sum())
        entry["class"] = classify(port_v, wrf_v, sens["rates"][name],
                                  mask=fin)
        rates[name] = entry
    summary["rates"] = rates
    if not_carried:
        summary["rates_not_carried"] = not_carried

    # Rain evaporation at a level the adjustment has just saturated.  WRF
    # evaporates rain wherever the post-adjustment ssatw is below -1e-15
    # (:3501), and after the Newton adjustment that ssatw is a residual of a
    # few float32 units of qv/qvs, so its SIGN is decided by rounding.
    # Measure the cells where the two codes' rain-evaporation rates differ:
    # how far each code's ssatw sits from zero, in float32 epsilons, whether
    # the adjustment ran there, and how much rain WRF evaporates there.
    c2 = cps["cp2"]
    eps32 = float(np.finfo(np.float32).eps)
    wrf_rev = c2["prv_rev"]
    port_rev = np.where(np.isnan(wrf_rev), np.nan, port_r["rates"]["prv_rev"])
    big = np.maximum(np.abs(wrf_rev), np.abs(port_rev))
    diff = np.isfinite(wrf_rev) & (big > 0) & (
        np.abs(wrf_rev - port_rev) > ROUNDING_REL * big)
    if diff.any():
        s_w = np.abs(c2["ssatw"][diff]) / eps32
        s_p = np.abs(np.nan_to_num(
            port_r["diagnostics"]["ssatw_rev"][diff], nan=0.0)) / eps32
        adjusted = c2["prw_vcd"][diff] != 0
        qr_pre = np.maximum(wrf_stage(cps, "cp1", "qr", dt)[diff], 1.0e-30)
        frac = np.abs(wrf_rev - port_rev)[diff] * dt / qr_pre
        near = (s_w <= 16) & adjusted
        summary["rain_evaporation_at_saturation"] = {
            "n_cells_rev_differs": int(diff.sum()),
            "n_adjusted_and_wrf_ssatw_within_16_eps": int(near.sum()),
            "wrf_ssatw_eps_max": float(s_w.max()),
            "wrf_ssatw_eps_max_where_adjusted": float(
                s_w[adjusted].max()) if adjusted.any() else 0.0,
            "port_ssatw_eps_max_where_adjusted": float(
                s_p[adjusted].max()) if adjusted.any() else 0.0,
            "rain_fraction_moved_max": float(frac.max()),
            "rain_fraction_moved_max_outside": float(
                frac[~near].max()) if (~near).any() else 0.0,
            "abs_rate_diff_max_kg_kg_s": float(
                np.abs(wrf_rev - port_rev)[diff].max())}

    # Rain self-collection and break-up (:2159-2176) is 1 - exp(2300*(mvd_r
    # - 1950e-6)) times smooth factors: near the 1950-micron crossing the
    # rate's relative sensitivity to mvd_r is kappa = 2300*mvd*e^x/|1-e^x|.
    # Dividing the measured relative gap by kappa gives the gap in mvd_r
    # itself, which is what rounding can be held to.
    c1 = cps["cp1"]
    rr = c1["qr1d"] * c1["rho"]
    mvd = rain_mean_volume_diameter(c1["qr1d"], c1["nr1d"], c1["rho"])
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        x = 2300.0 * (mvd - 1950.0e-6)
        kappa = np.abs(2300.0 * mvd * np.exp(x)
                       / np.where(np.abs(1.0 - np.exp(x)) > 0,
                                  1.0 - np.exp(x), 1.0e-30))
    w_v = c1["pnr_rcr"]
    p_v = port_r["rates"]["pnr_rcr"]
    big = np.maximum(np.abs(w_v), np.abs(p_v))
    act = np.isfinite(w_v) & (big > 0) & (rr > 1.0e-12)
    if act.any():
        rel = np.abs(p_v - w_v)[act] / big[act]
        summary["conditioning"] = {"pnr_rcr": {
            "n_active": int(act.sum()),
            "rel_max": float(rel.max()),
            "rel_over_kappa_max": float(
                (rel / np.maximum(kappa[act], 1.0)).max()),
            "mvd_um_at_worst": float(mvd[act][np.argmax(rel)] * 1.0e6)}}

    # 5b. Stage states and per-species stage tendencies.
    stages = {}
    for stage, cp in STAGE_CHECKPOINT.items():
        rows = {}
        for var in (*species, "T", "ng"):
            wrf_v = wrf_stage(cps, cp, var, dt)
            port_v = port["stages"][stage][var]
            rows[var] = compare(port_v, wrf_v, floor=FLOOR[var],
                                where=where)
        for acc in ("ncten", "nwfaten", "nifaten"):
            if stage == "sources" and acc in port["stages"][stage]:
                rows[acc] = compare(port["stages"][stage][acc],
                                    cps[cp][acc], floor=0.0, where=where)
        stages[stage] = rows
    summary["stages"] = stages

    tendencies = {}
    order = ["entry", "sources", "rain_evaporation"]
    cp_of = {"entry": None, **STAGE_CHECKPOINT}
    for a_stage, b_stage in zip(order[:-1], order[1:]):
        rows = {}
        for var in (*species, "T"):
            if a_stage == "entry":
                wrf_a = cps["cp1"][ONE_D[var]]
                port_a = inp[var].astype(np.float64)
            else:
                wrf_a = wrf_stage(cps, cp_of[a_stage], var, dt)
                port_a = port["stages"][a_stage][var]
            wrf_b = wrf_stage(cps, cp_of[b_stage], var, dt)
            port_b = port["stages"][b_stage][var]
            if a_stage == "entry":
                port_a = np.where(np.isnan(wrf_a), np.nan, port_a)
            dw = (wrf_b - wrf_a) / dt
            dp = (port_b - port_a) / dt
            # A stage tendency is the difference of two float32 states, so
            # its resolution is the state's ulp over dt: measure the gap in
            # those units, and relatively only where the increment is well
            # resolved (above 1024 ulp of the state per step).
            scale = np.maximum.reduce([np.abs(wrf_a), np.abs(port_a),
                                       np.abs(wrf_b), np.abs(port_b)])
            present = np.isfinite(dw) & np.isfinite(dp) & (scale > FLOOR[var])
            moved = present & (np.maximum(np.abs(dw), np.abs(dp)) > 0)
            gap_ulps = np.abs(dp - dw) * dt / np.maximum(
                ulp32(scale), np.finfo(np.float64).tiny)
            row = {"n_moved": int(moved.sum())}
            if moved.any():
                g = gap_ulps[moved]
                row.update({
                    "gap_ulp_p99": float(np.quantile(g, 0.99)),
                    "gap_ulp_max": float(g.max()),
                    "n_gap_gt_4ulp": int((g > 4).sum()),
                    "n_gap_gt_64ulp": int((g > 64).sum())})
                resolved = moved & (np.maximum(np.abs(dw), np.abs(dp)) * dt
                                    > 1024.0 * ulp32(scale))
                rel = compare(dp, dw, floor=0.0, mask=resolved, where=where)
                row["resolved"] = rel
            rows[var] = row
        tendencies[f"{a_stage}->{b_stage}"] = rows
    summary["stage_tendencies"] = tendencies

    # 5c. Final state and diagnostics.
    final = {}
    for var in species:
        final[var] = compare(port["final"][var], wrf[var],
                             floor=FLOOR[var], where=where)
    final["th"] = compare(port["final"]["th"], wrf["th"], where=where)
    final["ng"] = compare(port["final"]["ng"], cps["cpx"]["ng1d"],
                          floor=FLOOR["ng"], where=where)
    final["T_exit"] = compare(port["final"]["T"], cps["cpx"]["t1d"],
                              where=where)
    final["refl_dbz_abs"] = {
        "max_abs_db": float(np.max(np.abs(port["final"]["refl"]
                                          - wrf["refl"]))),
        "n_cells_gt_0.1db": int((np.abs(port["final"]["refl"]
                                        - wrf["refl"]) > 0.1).sum()),
        "n_cells_ge_0dbz": int((wrf["refl"] >= 0.0).sum())}
    for name, key in (("re_cloud", "re_cloud"), ("re_ice", "re_ice"),
                      ("re_snow", "re_snow")):
        lo, hi = {"re_cloud": (2.49e-6, 50.0e-6),
                  "re_ice": (4.99e-6, 125.0e-6),
                  "re_snow": (9.99e-6, 999.0e-6)}[name]
        want = (np.clip(wrf[key], lo, hi).astype(f32)
                * f32(1.0e6)).astype(np.float64)
        final[name] = compare(port["final"][key], want, where=where)
    for name in ("rainnc", "snownc", "graupelnc", "sr"):
        final[name] = compare(port["final"][name], wrf[name],
                              floor=1.0e-9 if name != "sr" else 0.0,
                              where=lambda loc: where((loc[0],)))
    def cell_scale(var):
        """Largest magnitude ``var`` took in the cell anywhere in the call."""
        out = np.abs(inp[var].astype(np.float64)) if var in inp else 0.0
        for snap in port["stages"].values():
            if var in snap:
                out = np.maximum(out, np.nan_to_num(np.abs(snap[var])))
        for cp in ("cp1", "cp2", "cp3", "cp4"):
            if TEN.get(var) in cps[cp]:
                out = np.maximum(out, np.nan_to_num(
                    np.abs(wrf_stage(cps, cp, var, dt))))
        return out

    for var in species:
        final[var]["class"] = classify(port["final"][var], wrf[var],
                                       sens["final"][var], floor=FLOOR[var],
                                       scale=cell_scale(var))
    final["th"]["class"] = classify(port["final"]["th"], wrf["th"],
                                    sens["final"]["th"])
    final["ng"]["class"] = classify(port["final"]["ng"], cps["cpx"]["ng1d"],
                                    sens["final"]["ng"], floor=FLOOR["ng"],
                                    scale=cell_scale("ng"))
    for name in ("rainnc", "snownc", "graupelnc"):
        final[name]["class"] = classify(port["final"][name], wrf[name],
                                        sens["final"][name], floor=1.0e-9)
    refl_gap = np.abs(port["final"]["refl"] - wrf["refl"])
    final["refl_dbz_abs"]["n_cells_gt_0.1db_beyond_4x_sensitivity"] = int(
        ((refl_gap > 0.1) & (refl_gap > 4.0 * sens["final"]["refl"])).sum())
    summary["final"] = final

    # 5d. Sedimentation, cleanup and the terminal apply on identical inputs:
    # the final state over the columns whose every working value agreed to
    # CLEAN_REL just before the fall-speed search.
    clean = np.ones(ncol, bool)
    for var in (*species, "T"):
        wrf_v = wrf_stage(cps, "cp2", var, dt)
        port_v = port["stages"]["rain_evaporation"][var]
        big = np.maximum(np.abs(wrf_v), np.abs(port_v))
        ok = np.isfinite(wrf_v) & (big > FLOOR[var])
        rel = np.where(ok, np.abs(wrf_v - port_v) / np.where(big > 0, big, 1),
                       0.0)
        clean &= rel.max(axis=1) <= CLEAN_REL
    clean &= micro
    summary["columns_clean_before_sedimentation"] = int(clean.sum())
    final_clean = {}
    cmask = np.repeat(clean[:, None], nz, axis=1)
    for var in species:
        final_clean[var] = compare(port["final"][var], wrf[var],
                                   floor=FLOOR[var], mask=cmask, where=where)
    final_clean["T_exit"] = compare(port["final"]["T"], cps["cpx"]["t1d"],
                                    mask=cmask, where=where)
    for name in ("rainnc", "snownc", "graupelnc"):
        final_clean[name] = compare(port["final"][name], wrf[name],
                                    floor=1.0e-9, mask=clean,
                                    where=lambda loc: where((loc[0],)))
    summary["final_clean"] = final_clean

    # 5e. Size of the final-state differences against the size of the call:
    # column integrals (rho*dz weighted) of |port - WRF| and of WRF's own
    # |exit - entry|, summed over the columns.
    rho = np.where(np.isfinite(cps["cp1"]["rho"]), cps["cp1"]["rho"],
                   inp["p"] / (287.04 * inp["T"]))
    weight = rho * inp["dz"]
    impact = {}
    for var in species:
        entry = inp[var].astype(np.float64)
        diff = np.abs(port["final"][var] - wrf[var].astype(np.float64))
        change = np.abs(wrf[var].astype(np.float64) - entry)
        impact[var] = {"sum_abs_port_minus_wrf": float((diff * weight).sum()),
                       "sum_abs_wrf_change": float((change * weight).sum())}
    summary["impact"] = impact

    # 5f. Cells carrying each named difference (README.md, "Findings"),
    # the repaired ones included: each counter reads what the unrepaired
    # port did, so a repair that regresses shows here by name.
    c1 = cps["cp1"]
    r1_band = {}
    for var in ("qr", "qg", "qs", "qi"):
        q = c1[ONE_D[var]]
        r1_band[var] = int(np.nansum((q > 1.0e-12) & (q * c1["rho"] <= 1.0e-12)))
    ng_tau1 = wrf_stage(cps, "cp1", "ng", dt)
    summary["attribution"] = {
        # G: WRF's terminal apply writes qv = MAX(1.E-10, ...) at every level
        # of a column with microphysics (:3974); the port leaves an entry
        # vapour of exactly zero where no kernel writes the level (before
        # repair G).
        "vapour_floor_at_empty_levels": int(np.nansum(
            (inp["qv"] == 0) & (wrf["qv"] == 1.0e-10)
            & (port["final"]["qv"] == 0))),
        # K: WRF zeroes entry masses at or below R1 (:1845-1914); the port
        # zeroed only cloud water (before repair K).
        "entry_mass_at_or_below_r1": {
            var: int(((inp[var] > 0) & (inp[var] <= 1.0e-12)).sum())
            for var in ("qi", "qr", "qs", "qg")},
        "entry_ice_without_number": int(np.nansum(
            (c1["qi1d"] > 1.0e-12) & (c1["ni1d"] * c1["rho"] <= 1.0e-6))),
        "cold_snow_riming_where_wrf_table_is_zero": int(np.nansum(
            (port_r["rates"]["prs_scw"] != 0) & (c1["prs_scw"] == 0)
            & (c1["t1d"] < 273.15))),
        "graupel_number_removed_by_wrf_at_the_source_stage": int(np.nansum(
            (c1["ng1d"] > 0) & (ng_tau1 <= 0)
            & (port["stages"]["sources"]["ng"] > 0))),
        "entry_mass_at_or_below_r1_concentration": r1_band,
        # T and its kin: condensate the port returns at or below R1 where
        # WRF's terminal apply wrote zero (:4007, :4025, :4042, :4057,
        # :4060).  Below every comparison's floor, so counted here (T and
        # J closed the graupel and cloud/ice cases).
        "left_at_or_below_r1": {
            var: int(((port["final"][var] > 0)
                      & (port["final"][var] <= 1.0e-12)
                      & (wrf[var] == 0)).sum())
            for var in ("qc", "qr", "qi", "qs", "qg")},
    }
    summary["timings_s"]["total"] = round(time.time() - t0, 2)
    if dump:
        arrays = {f"in_{k}": v for k, v in inp.items()}
        arrays.update({f"wrf_{k}": v for k, v in wrf.items()})
        for cp, table in cps.items():
            arrays.update({f"{cp}_{k}": v for k, v in table.items()})
        arrays.update({f"portrate_{k}": v
                       for k, v in port_r["rates"].items()})
        arrays.update({f"portdiag_{k}": v
                       for k, v in port_r["diagnostics"].items()})
        arrays.update({f"sensrate_{k}": v for k, v in sens["rates"].items()})
        arrays.update({f"sensfinal_{k}": v
                       for k, v in sens["final"].items()})
        for stage, snap in port["stages"].items():
            arrays.update({f"port_{stage}_{k}": v for k, v in snap.items()})
        arrays.update({f"portfinal_{k}": v for k, v in port["final"].items()})
        np.savez(Path(dump) / "arrays.npz", **arrays)

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1),
                                          encoding="utf-8")
    if not keep_streams:
        for cp in SCHEMA:
            (run_dir / f"wrf-{cp}.bin").unlink(missing_ok=True)
        for name in ("parity-in.bin", "parity-pristine.out",
                     "parity-rates.out"):
            (run_dir / name).unlink(missing_ok=True)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("columns")
    ap.add_argument("build")
    ap.add_argument("out")
    ap.add_argument("--max-cols", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-streams", action="store_true")
    ap.add_argument("--dump", default=None, metavar="DIR",
                    help="also write every compared array to DIR/arrays.npz")
    ap.add_argument("--mp", type=int, choices=(28, 8), default=28,
                    help="the Thompson variant: 28 aerosol aware, 8 classic")
    args = ap.parse_args(argv)
    summary = run(args.columns, args.build, args.out,
                  max_cols=args.max_cols, seed=args.seed,
                  keep_streams=args.keep_streams, dump=args.dump,
                  mp=args.mp)
    print(json.dumps({k: summary[k] for k in (
        "ncol", "columns_with_microphysics", "wrf_instrumentation_neutral",
        "port_instrumentation_neutral", "timings_s")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
