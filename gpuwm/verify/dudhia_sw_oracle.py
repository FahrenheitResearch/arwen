"""Readers and graders for the WRF v4.6.1 Dudhia shortwave column oracle.

tools/dudhia_sw_wrf461_oracle/build.sh writes the inputs and WRF's outputs;
this module reads them, runs WOOF's production Dudhia routine on the GPU on
the same words, and grades every output word bitwise.  It holds no Dudhia
arithmetic of its own.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

INPUT_FIELDS = ("t", "p", "qv", "qc", "qr", "qi", "qs", "qg", "dz8w", "pii",
                "rho")
OUTPUT_FIELDS = ("rthraten", "gsw", "swdown")


def read_inputs_bin(path: Path) -> dict:
    raw = Path(path).read_bytes()
    ncol, nz, ncase = np.frombuffer(raw, "<i4", 3, 0)
    off = 12
    out = {}

    def take(dtype, count, shape):
        nonlocal off
        arr = np.frombuffer(raw, dtype, count, off).reshape(shape)
        off += 4 * count
        return arr.astype(dtype[1:]).copy()

    out["solcon"] = take("<f4", ncase, (ncase,))
    out["swrad_scat"] = take("<f4", ncase, (ncase,))
    out["icloud"] = take("<i4", ncase, (ncase,))
    for name in INPUT_FIELDS:
        out[name] = take("<f4", nz * ncol, (nz, ncol))
    out["coszen"] = take("<f4", ncol, (ncol,))
    out["albedo"] = take("<f4", ncol, (ncol,))
    if off != len(raw):
        raise ValueError(f"{path}: {len(raw) - off} trailing bytes")
    return out


def read_outputs_bin(path: Path, nz: int, ncol: int, ncase: int) -> dict:
    raw = np.fromfile(path, "<f4")
    per_case = nz * ncol + 2 * ncol
    if raw.size != per_case * ncase:
        raise ValueError(f"{path}: {raw.size} words, expected "
                         f"{per_case * ncase}")
    rth, gsw, swd = [], [], []
    for c in range(ncase):
        block = raw[c * per_case:(c + 1) * per_case]
        rth.append(block[:nz * ncol].reshape(nz, ncol))
        gsw.append(block[nz * ncol:nz * ncol + ncol])
        swd.append(block[nz * ncol + ncol:])
    return {"rthraten": np.stack(rth).astype(np.float32),
            "gsw": np.stack(gsw).astype(np.float32),
            "swdown": np.stack(swd).astype(np.float32)}


def _ordered(bits: np.ndarray) -> np.ndarray:
    b = bits.astype(np.int64)
    return np.where(b < 0, -(b & 0x7FFFFFFF), b)


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ia = _ordered(np.asarray(a, np.float32).view(np.int32))
    ib = _ordered(np.asarray(b, np.float32).view(np.int32))
    return np.abs(ia - ib)


def run_gpu(inputs: dict, *, layout: str = "columns") -> dict:
    """WOOF's Dudhia on every case; outputs in the oracle's own layout.

    ``columns``: :func:`gpuwm.core.dudhia.dudhia_shortwave_columns` on
    top-down ``(ncol, nz)`` columns.  ``adapter``: the production
    :class:`gpuwm.core.dudhia.DudhiaShortwaveRadiation` on gpuwm's own
    bottom-up ``(nz, 1, ncol)`` state, with its packing, Exner division and
    result assembly; only the solar geometry is replaced by the oracle's
    COSZEN and SOLCON words (the geometry is the radiation driver's, not
    SWRAD's, and is not graded here).
    """
    import cupy as cp
    from gpuwm.core import dudhia

    nz, ncol = inputs["t"].shape
    fields = ("t", "p", "qv", "qc", "qr", "qi", "qs", "qg", "dz8w")
    rth, gsw, swd = [], [], []
    for c in range(inputs["solcon"].size):
        kw = dict(swrad_scat=float(inputs["swrad_scat"][c]),
                  icloud=int(inputs["icloud"][c]))
        solcon = float(inputs["solcon"][c])
        if layout == "columns":
            dev = [cp.asarray(np.ascontiguousarray(inputs[n][::-1].T))
                   for n in fields]
            exner = cp.asarray(np.ascontiguousarray(inputs["pii"][::-1].T))
            heat, sw, g = dudhia.dudhia_shortwave_columns(
                *dev, cp.asarray(inputs["coszen"]),
                cp.asarray(inputs["albedo"]), exner=exner, solcon=solcon,
                **kw)
            heat = cp.asnumpy(heat)[:, ::-1].T
            sw, g = cp.asnumpy(sw), cp.asnumpy(g)
        elif layout == "adapter":
            heat, sw, g = _run_adapter(inputs, solcon=solcon, **kw)
        else:
            raise ValueError(f"unknown layout {layout!r}")
        rth.append(np.asarray(heat, np.float32).reshape(nz, ncol))
        gsw.append(np.asarray(g, np.float32).reshape(ncol))
        swd.append(np.asarray(sw, np.float32).reshape(ncol))
    return {"rthraten": np.stack(rth), "gsw": np.stack(gsw),
            "swdown": np.stack(swd)}


def _run_adapter(inputs: dict, *, solcon: float, swrad_scat: float,
                 icloud: int):
    from datetime import datetime
    from types import SimpleNamespace
    from unittest import mock
    import cupy as cp
    from gpuwm.config import RunConfig
    from gpuwm.core import dudhia

    nz, ncol = inputs["t"].shape

    def grid(name):
        return cp.asarray(inputs[name].reshape(nz, 1, ncol))

    atmosphere = {"temperature": grid("t"), "pressure": grid("p"),
                  "qv": grid("qv"), "qc": grid("qc"), "qi": grid("qi"),
                  "dz": grid("dz8w"), "exner": grid("pii")}
    state = SimpleNamespace(elapsed_seconds=0.0, qc=grid("qc"),
                            qr=grid("qr"), qi=grid("qi"), qs=grid("qs"),
                            qg=grid("qg"))
    fields = {"albedo": cp.asarray(inputs["albedo"].reshape(1, ncol)),
              "glw": cp.zeros((1, ncol), cp.float32)}
    cfg = RunConfig(nx=ncol, ny=1, nz=nz, dx=1000.0, dy=1000.0,
                    ztop=20000.0, dt=10.0, run_seconds=60.0,
                    ra_lw_physics=0, ra_sw_physics=1, radt_minutes=10.0)
    coszen = cp.asarray(inputs["coszen"].reshape(1, ncol))

    def geometry(*_args, **_kwargs):
        return coszen, solcon

    adapter = dudhia.DudhiaShortwaveRadiation(
        datetime(2026, 6, 21, 18, 0), np.zeros((1, ncol), np.float32),
        np.zeros((1, ncol), np.float32), swrad_scat=swrad_scat,
        icloud=icloud)
    with mock.patch.object(dudhia, "wrf_solar_geometry", geometry):
        result = adapter(atmosphere=atmosphere, fields=fields, state=state,
                         cfg=cfg)
    return (cp.asnumpy(result.rthratensw), cp.asnumpy(result.swdown),
            cp.asnumpy(result.gsw))


def compare(got: dict, wrf: dict, regime: np.ndarray) -> dict:
    """Bitwise grade of every output word, with per-regime miss counts."""
    report = {}
    for name in OUTPUT_FIELDS:
        a = np.asarray(got[name], np.float32)
        b = np.asarray(wrf[name], np.float32)
        if a.shape != b.shape:
            raise ValueError(f"{name}: shape {a.shape} vs {b.shape}")
        diff_bits = a.view(np.uint32) != b.view(np.uint32)
        both_nan = np.isnan(a) & np.isnan(b)
        miss = diff_bits & ~both_nan
        ulp = np.where(miss, ulp_distance(a, b), 0)
        absd = np.where(miss, np.abs(a.astype(np.float64) - b), 0.0)
        # Column axis is the last one for every field.
        per_col = miss.reshape(-1, miss.shape[-1]).any(axis=0)
        regimes = {}
        for r in sorted(set(map(str, regime))):
            sel = np.asarray(regime) == r
            cols_missed = int(per_col[sel].sum())
            if cols_missed:
                regimes[r] = cols_missed
        report[name] = {
            "words": int(a.size),
            "not_bit_identical": int(miss.sum()),
            "max_ulp": int(ulp.max()) if ulp.size else 0,
            "max_abs_diff": float(absd.max()) if absd.size else 0.0,
            "columns_missed_by_regime": regimes,
        }
    return report


FIXTURE_DIR = (Path(__file__).resolve().parents[2] / "tests" / "data"
               / "oracles" / "dudhia_sw")


def load_fixture(fixture_dir: Path = FIXTURE_DIR) -> tuple[dict, dict]:
    """The committed oracle: (inputs, WRF outputs) as float32 arrays."""
    with np.load(fixture_dir / "dudhia-inputs.npz") as f:
        inputs = {k: f[k] for k in f.files}
    with np.load(fixture_dir / "dudhia-wrf461.npz") as f:
        wrf = {k: f[k] for k in f.files}
    return inputs, wrf


def measure(*, layout: str = "columns",
            fixture_dir: Path = FIXTURE_DIR) -> dict:
    """Grade WOOF against the committed fixture; one row per output field."""
    inputs, wrf = load_fixture(fixture_dir)
    return compare(run_gpu(inputs, layout=layout), wrf, inputs["regime"])
