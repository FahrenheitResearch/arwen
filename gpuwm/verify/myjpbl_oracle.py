"""Measure the MYJ PBL kernel against the WRF v4.6.1 Fortran column oracle.

The oracle is ``tools/myjpbl_wrf461_oracle``: WRF's own ``MYJPBL`` from the
byte-unmodified ``phys/module_bl_myjpbl.F``, compiled at gfortran -O0 and run
over the columns ``make_columns.py`` generates, three consecutive calls per
case.  Its fixture (``tests/data/oracles/myjpbl/myjpbl-wrf461.npz``) holds
every call's complete input set and every output, so the kernel can be

* REPLAYED: each call fed the inputs WRF saw on that call, and
* FREE-RUN: call 1 fed WRF's inputs, later calls fed the kernel's own
  carried state (TKE, THZ0, QZ0, QSFC, CT) and columns advanced by the
  kernel's own tendencies with the driver's forward step
  (run_myjpbl.F90: ``X + RXBLTEN*DT``, ``T = TH*EXNER``, moisture floored
  at 0), in float32 exactly as the driver does it.

:func:`measure` returns, per case, call and output field, the maximum ULP
distance, the number of words whose BIT PATTERNS differ (so a -0.0 against
+0.0 counts, which the ULP metric folds to 0) and the maximum absolute
difference.  Nothing here asserts; ``tests/test_myjpbl_wrf461_parity.py`` is
the gate and ``tools/myjpbl_wrf461_oracle/compare_myjpbl.py`` the printer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gpuwm.core.fp32_ulp import fp32_ulp_distance

FIXTURE = (Path(__file__).resolve().parents[2] / "tests" / "data"
           / "oracles" / "myjpbl" / "myjpbl-wrf461.npz")

#: Level outputs, oracle name -> kernel output name.
LEVEL_OUTPUTS = ("tke", "el_myj", "exch_h", "rublten", "rvblten",
                 "rthblten", "rqvblten", "rqcblten", "rqiblten")
#: Surface outputs the kernel writes (kpbl int32).
SURFACE_OUTPUTS = ("pblh", "mixht", "kpbl", "thz0", "qz0", "qsfc", "ct")
STATE = ("thz0", "qz0", "qsfc", "ct")


def load(path: Path | str = FIXTURE) -> dict[str, dict[str, np.ndarray]]:
    """The fixture as ``{case: {record: array}}``."""
    archive = np.load(path)
    cases: dict[str, dict[str, np.ndarray]] = {}
    for key in archive.files:
        case, name = key.split("/", 1)
        cases.setdefault(case, {})[name] = archive[key]
    return cases


def _inputs(fx, step):
    tag = f"s{step}_in_"
    return {name[len(tag):]: fx[name].copy() for name in fx
            if name.startswith(tag)}


def run_kernel(inputs: dict[str, np.ndarray], *, nz: int, flqi: bool,
               dt: float) -> dict[str, np.ndarray]:
    """One MYJ call through the shipped launcher on one call's inputs.

    ``inputs`` are the oracle's records for one call (level fields
    ``(nz+1, ncol)``, surface fields ``(ncol,)``).  Returns the outputs in
    the oracle's layout (level fields ``(nz, ncol)``).
    """
    import cupy as cp

    from gpuwm.core.myjpbl import myj_pbl_step

    ncol = inputs["ht"].shape[0]

    def lev(a):
        return cp.asarray(np.ascontiguousarray(
            a[:nz].astype(np.float32).reshape(nz, 1, ncol)))

    def sfc(a):
        return cp.asarray(np.ascontiguousarray(
            a.astype(np.float32).reshape(1, ncol)))

    columns = {"dz": lev(inputs["dz"]), "u": lev(inputs["u"]),
               "v": lev(inputs["v"]), "t": lev(inputs["t"]),
               "th": lev(inputs["th"]), "exner": lev(inputs["exner"]),
               "qv": lev(inputs["qv"]), "qc": lev(inputs["qcw"]),
               "qi": lev(inputs["qci"]) if flqi else None,
               "p": lev(inputs["pmid"])}
    surface = {"psfc": sfc(inputs["pint"][0]), "ust": sfc(inputs["ustar"]),
               "tsk": sfc(inputs["tsk"]), "chklowq": sfc(inputs["chklowq"]),
               "xland": sfc(inputs["xland"]), "sice": sfc(inputs["sice"]),
               "snow": sfc(inputs["snow"]), "akhs": sfc(inputs["akhs"]),
               "akms": sfc(inputs["akms"]), "elflx": sfc(inputs["elflx"]),
               "uz0": sfc(inputs["uz0"]), "vz0": sfc(inputs["vz0"]),
               "ht": sfc(inputs["ht"])}
    state = {name: sfc(inputs[name]) for name in STATE}
    tke = lev(inputs["tke"])
    out = myj_pbl_step(columns, surface, state, tke, dtturbl=float(dt),
                       flqi=bool(flqi))
    got = {name: cp.asnumpy(out[name]).reshape(nz, ncol)
           for name in LEVEL_OUTPUTS if name != "tke"}
    got["tke"] = cp.asnumpy(tke).reshape(nz, ncol)
    for name in ("pblh", "mixht", "kpbl"):
        got[name] = cp.asnumpy(out[name]).reshape(ncol)
    for name in STATE:
        got[name] = cp.asnumpy(state[name]).reshape(ncol)
    return got


def _advance(inputs, got, *, nz, flqi, dt):
    """run_myjpbl.F90's forward step, in float32, on the kernel's outputs."""
    nxt = {k: v.copy() for k, v in inputs.items()}
    dt = np.float32(dt)
    zero = np.float32(0.0)
    for x, r in (("u", "rublten"), ("v", "rvblten"), ("th", "rthblten")):
        nxt[x][:nz] = inputs[x][:nz] + got[r] * dt
    nxt["t"][:nz] = nxt["th"][:nz] * inputs["exner"][:nz]
    species = [("qv", "rqvblten"), ("qcw", "rqcblten")]
    if flqi:
        species.append(("qci", "rqiblten"))
    for x, r in species:
        nxt[x][:nz] = np.maximum(inputs[x][:nz] + got[r] * dt, zero)
    nxt["tke"][:nz] = got["tke"]
    nxt["exch_h"][:nz] = got["exch_h"]
    for name in STATE:
        nxt[name] = got[name].copy()
    return nxt


def _compare(got, want):
    got = np.ascontiguousarray(got)
    want = np.ascontiguousarray(want)
    if want.dtype.kind == "i":
        diff = np.abs(got.astype(np.int64) - want.astype(np.int64))
        return {"max_ulp": int(diff.max()), "bit_mismatch": int((diff != 0).sum()),
                "max_abs": float(diff.max()), "n": int(want.size),
                "beyond_ftz": int((diff != 0).sum())}
    got = got.astype(np.float32)
    ulp = fp32_ulp_distance(got, want)
    bits = got.view(np.uint32) != want.view(np.uint32)
    tiny = np.finfo(np.float32).tiny
    with np.errstate(invalid="ignore"):
        absd = np.abs(got.astype(np.float64) - want.astype(np.float64))
        # The flush-to-zero class CuPy's default -ftz=true makes: a word
        # that differs by less than FLT_MIN, i.e. by no more than a flushed
        # subnormal operand can move it.  Anything else (or a NaN) is a
        # difference beyond FTZ.
        beyond = bits & ((absd >= tiny) | np.isnan(got) | np.isnan(want))
    return {"max_ulp": int(ulp.max()), "bit_mismatch": int(bits.sum()),
            "max_abs": float(np.nanmax(absd)) if absd.size else 0.0,
            "n": int(want.size), "beyond_ftz": int(beyond.sum()),
            "flushed_subnormal": int(((want != 0) & (np.abs(want) < tiny)
                                      & (got == 0)).sum())}


def measure(fixture=None, *, cases=None, zero_ht: bool = False) -> dict:
    """``{mode: {case: {step: {field: stats}}}}`` for replay and free-run.

    ``zero_ht`` is the negative control: it hands the kernel HT = 0 instead
    of the terrain height WRF saw, i.e. the ground-relative interface
    heights the kernel used before it seeded ZINT with HT.
    """
    fixture = load() if fixture is None else fixture
    report: dict = {"replay": {}, "free_run": {}}
    for case, fx in sorted(fixture.items()):
        if cases is not None and case not in cases:
            continue
        ncol, nz, nstep, flqi = (int(x) for x in fx["meta"])
        dt = float(fx["meta_dt"][0])
        chained = _inputs(fx, 1)
        for mode in ("replay", "free_run"):
            report[mode][case] = {}
        for step in range(1, nstep + 1):
            want = {name: fx[f"s{step}_out_{name}"] for name in
                    LEVEL_OUTPUTS + SURFACE_OUTPUTS}
            for mode in ("replay", "free_run"):
                inputs = _inputs(fx, step) if mode == "replay" else chained
                if zero_ht:
                    inputs = dict(inputs, ht=np.zeros_like(inputs["ht"]))
                got = run_kernel(inputs, nz=nz, flqi=bool(flqi), dt=dt)
                fields = {}
                for name in LEVEL_OUTPUTS:
                    if name == "rqiblten" and not flqi:
                        continue            # WRF does not write it (:647)
                    fields[name] = _compare(got[name], want[name][:nz])
                for name in SURFACE_OUTPUTS:
                    fields[name] = _compare(got[name], want[name])
                report[mode][case][step] = fields
                if mode == "free_run":
                    chained = _advance(chained, got, nz=nz, flqi=bool(flqi),
                                       dt=dt)
    return report


def summarize(report: dict) -> dict:
    """Worst stats per mode and field over every case and call."""
    out: dict = {}
    for mode, by_case in report.items():
        worst: dict = {}
        for by_step in by_case.values():
            for fields in by_step.values():
                for name, st in fields.items():
                    w = worst.setdefault(name, {"max_ulp": 0, "bit_mismatch": 0,
                                                "max_abs": 0.0, "n": 0,
                                                "beyond_ftz": 0,
                                                "flushed_subnormal": 0})
                    w["max_ulp"] = max(w["max_ulp"], st["max_ulp"])
                    w["bit_mismatch"] += st["bit_mismatch"]
                    w["max_abs"] = max(w["max_abs"], st["max_abs"])
                    w["n"] += st["n"]
                    w["flushed_subnormal"] += st.get("flushed_subnormal", 0)
                    w["beyond_ftz"] += st.get("beyond_ftz", 0)
        out[mode] = worst
    return out
