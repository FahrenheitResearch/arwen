"""Measure the Kessler (mp_physics=1) GPU kernel against WRF v4.6.1's Fortran.

The fixture is written by ``tools/kessler_wrf461_oracle`` (build.sh drives
``run_kessler.F90``, which calls SUBROUTINE kessler of the byte-unmodified
``phys/module_mp_kessler.F``).  It lives under
``tests/data/oracles/kessler/``:

``kessler-inputs.npz``
    float32 ``t, qv, qc, qr, rho, pii, z, dz8w`` of shape ``(nz, ncol)``,
    ``rainnc (ncol,)``, ``dt (ncase,)``, ``nsteps (ncase,)`` and the
    ``regime`` name of every column.
``kessler-wrf461.npz``
    WRF's outputs after every call: ``t, qv, qc, qr`` of shape
    ``(nstep_total, nz, ncol)`` and ``rainnc, rainncv`` of shape
    ``(nstep_total, ncol)``, cases in order, steps in order.

:func:`measure` runs ``launch_kessler`` (the production launcher and the
production compile route) over the same calls and returns, per output field,
the number of words that are not bit-identical, the FP32 total-order ULP
maximum and the largest absolute difference.  Two modes:

``free``
    each case restarts from the inputs and the GPU state evolves on its own,
    exactly as WRF's state did in the oracle run;
``replay``
    every call starts from WRF's previous outputs, so each call is graded on
    its own and an early 1 ULP cannot be amplified downstream.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "kessler")
LEVEL_FIELDS = ("t", "qv", "qc", "qr")
SURFACE_FIELDS = ("rainnc", "rainncv")
OUTPUT_FIELDS = LEVEL_FIELDS + SURFACE_FIELDS
INPUT_FIELDS = ("t", "qv", "qc", "qr", "rho", "pii", "z", "dz8w")


def read_inputs_bin(path) -> dict[str, np.ndarray]:
    """The make_columns.py stream file -> the ``kessler-inputs`` arrays."""
    raw = Path(path).read_bytes()
    ncol, nz, ncase = np.frombuffer(raw[:12], "<i4")
    off = 12
    dt = np.frombuffer(raw, "<f4", ncase, off).copy(); off += 4 * ncase
    nsteps = np.frombuffer(raw, "<i4", ncase, off).copy(); off += 4 * ncase
    out = {"dt": dt, "nsteps": nsteps}
    for name in INPUT_FIELDS:
        out[name] = np.frombuffer(raw, "<f4", nz * ncol, off).reshape(
            nz, ncol).copy()
        off += 4 * nz * ncol
    out["rainnc"] = np.frombuffer(raw, "<f4", ncol, off).copy()
    off += 4 * ncol
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return out


def read_outputs_bin(path, nz: int, ncol: int, nstep_total: int):
    """run_kessler's output stream -> ``{field: (nstep_total, ...)}``."""
    raw = np.fromfile(path, "<f4")
    per_step = 4 * nz * ncol + 2 * ncol
    if raw.size != per_step * nstep_total:
        raise ValueError(f"{path}: {raw.size} words, expected "
                         f"{per_step * nstep_total}")
    raw = raw.reshape(nstep_total, per_step)
    out, off = {}, 0
    for name in LEVEL_FIELDS:
        out[name] = raw[:, off:off + nz * ncol].reshape(nstep_total, nz, ncol)
        off += nz * ncol
    for name in SURFACE_FIELDS:
        out[name] = raw[:, off:off + ncol]
        off += ncol
    return {k: np.ascontiguousarray(v) for k, v in out.items()}


def load_fixture(directory=FIXTURE_DIR):
    directory = Path(directory)
    with np.load(directory / "kessler-inputs.npz") as f:
        inputs = {k: f[k] for k in f.files}
    with np.load(directory / "kessler-wrf461.npz") as f:
        wrf = {k: f[k] for k in f.files}
    return inputs, wrf


def run_gpu(inputs, *, mode: str = "free", wrf=None):
    """Run the production Kessler launcher over every oracle call.

    Returns ``{field: (nstep_total, ...)}`` float32 host arrays laid out as
    the WRF fixture.  ``mode='replay'`` needs ``wrf``.
    """
    import cupy as cp
    from gpuwm.core.microphysics import launch_kessler

    if mode not in ("free", "replay"):
        raise ValueError(mode)
    if mode == "replay" and wrf is None:
        raise ValueError("replay mode needs the WRF outputs")
    nz, ncol = inputs["t"].shape

    def dev(a):
        return cp.asarray(np.ascontiguousarray(a, np.float32).reshape(
            a.shape[0], 1, ncol) if a.ndim == 2 else
            np.ascontiguousarray(a, np.float32).reshape(1, ncol))

    rho, pii = dev(inputs["rho"]), dev(inputs["pii"])
    z, dz8w = dev(inputs["z"]), dev(inputs["dz8w"])
    out = {k: [] for k in OUTPUT_FIELDS}
    n = 0
    for dt, nsteps in zip(inputs["dt"], inputs["nsteps"]):
        state = {k: dev(inputs[k]) for k in LEVEL_FIELDS}
        rainnc = dev(inputs["rainnc"])
        rainncv = cp.zeros((1, ncol), cp.float32)
        for step in range(int(nsteps)):
            if mode == "replay" and step > 0:
                for k in LEVEL_FIELDS:
                    state[k] = dev(wrf[k][n - 1])
                rainnc = dev(wrf["rainnc"][n - 1])
                rainncv = dev(wrf["rainncv"][n - 1])
            launch_kessler(state["t"], state["qv"], state["qc"], state["qr"],
                           rho, pii, z, dz8w, rainnc, rainncv, float(dt))
            for k in LEVEL_FIELDS:
                out[k].append(cp.asnumpy(state[k]).reshape(nz, ncol))
            out["rainnc"].append(cp.asnumpy(rainnc).reshape(ncol))
            out["rainncv"].append(cp.asnumpy(rainncv).reshape(ncol))
            n += 1
    return {k: np.stack(v) for k, v in out.items()}


def compare(got, want, regimes=None) -> dict[str, dict]:
    """Per-field bit-identity, FP32 ULP maximum and max abs difference."""
    table = {}
    for name in OUTPUT_FIELDS:
        g = np.ascontiguousarray(got[name], np.float32)
        w = np.ascontiguousarray(want[name], np.float32)
        mism = g.view(np.uint32) != w.view(np.uint32)
        ulp = fp32_ulp_distance(g, w)
        diff = np.abs(g.astype(np.float64) - w.astype(np.float64))
        row = {"values": int(g.size), "not_bit_identical": int(mism.sum()),
               "max_ulp": int(ulp.max()) if ulp.size else 0,
               "max_abs_diff": float(np.nanmax(diff)) if diff.size else 0.0}
        if regimes is not None and mism.any():
            col = np.nonzero(mism)[-1]
            names, counts = np.unique(np.asarray(regimes)[col],
                                      return_counts=True)
            row["by_regime"] = {str(a): int(b) for a, b in zip(names, counts)}
        table[name] = row
    return table


def measure(directory=FIXTURE_DIR, *, mode: str = "free") -> dict[str, dict]:
    inputs, wrf = load_fixture(directory)
    got = run_gpu(inputs, mode=mode, wrf=wrf)
    return compare(got, wrf, inputs.get("regime"))
