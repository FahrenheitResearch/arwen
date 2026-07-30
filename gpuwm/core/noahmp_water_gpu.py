"""Batched CUDA execution of Noah-MP's WATER subtree.

The arithmetic lives in ``gpuwm/core/kernels/noahmp_water.cu``, whose
``d_water`` body -- CANWATER, SNOWWATER and SOILWATER composed -- is already
held at max_ulp 0 against every ``output`` **and** ``probe`` row of
``noahmp-water.csv`` by ``tests/test_noahmp_water_cuda.py``.  This module
supplies the runtime-facing part: the physical Python call is packed in column
order, all columns are evaluated by one CUDA launch, and each row is unpacked
back into the caller's :class:`~gpuwm.core.noahmp_snow.SnowColumn` and a
:class:`~gpuwm.core.noahmp_water.WaterFluxes`.

Why this leaf is next: with VEGE_FLUX and BARE_FLUX batched, ``cProfile`` over
the whole-column fixtures puts WATER at 13.3% of the ``sflx`` tree -- about
thirty per cent of everything still executing in CPython, and the largest
single remaining subtree.  The measurement is in
``docs/noahmp_device_column_report.md``; it was re-taken after the first two
leaves moved rather than carried over from the pre-conversion ranking.

The slot layout below is the one documented in ``noahmp_water.cu`` and used by
that kernel's fixture gate.  It is duplicated here rather than imported from a
test, and ``tests/test_noahmp_water.py`` checks this packer against the oracle
generator's own column order so a transposition cannot pass silently.

``col`` is mutated in place exactly as :func:`gpuwm.core.noahmp_water.water`
mutates it: ``isnow``, ``snowh``, ``sneqv``, ``snice``, ``snliq``, ``stc``,
``zsnso``, ``dzsnso``, ``sh2o`` and ``sice`` are all INOUT in WRF.  The caller
therefore sees the same object graph either way.

``etrani`` is deliberately left ``None``.  WRF's WATER computes it as a local,
``NOAHMP_SFLX`` never reads it, and the device row does not carry it; returning
a recomputed value would be a second implementation of something nothing
consumes.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from gpuwm.core.noahmp_water import WaterFluxes

NSNOW = 3
NSOIL = 4
NFULL = NSNOW + NSOIL
THREADS = 128

#: The packed snow/soil column, ``noahmp_water.cu``'s ``NSTATE``.
NSTATE = 37
STATE_SLOT = {"snowh": 0, "sneqv": 1, "snice": 2, "snliq": 5, "stc": 8,
              "zsnso": 15, "dzsnso": 22, "sh2o": 29, "sice": 33}
STATE_WIDTH = {"snowh": 1, "sneqv": 1, "snice": NSNOW, "snliq": NSNOW,
               "stc": NFULL, "zsnso": NFULL, "dzsnso": NFULL,
               "sh2o": NSOIL, "sice": NSOIL}

P_STRIDE = 26
P_LAYER = {"smcmax": 0, "smcwlt": 4, "bexp": 8, "dksat": 12, "dwsat": 16}
P_SCALAR = {"kdt": 20, "frzx": 21, "slope": 22, "ch2op": 23, "ssi": 24,
            "snow_ret_fac": 25}

INT_STRIDE = 9
IN_STRIDE = NSTATE + 39
OUT_STRIDE = NSTATE + 36

#: Scalar forcing slots, offset from ``NSTATE``.
IN_SCALAR = {"dt": 0, "fcev": 1, "fctr": 2, "elai": 3, "esai": 4, "fveg": 5,
             "bdfall": 6, "sfctmp": 7, "qvap": 8, "qdew": 9, "qsnow": 10,
             "qrain": 11, "snowhin": 12, "ponding": 13, "canliq": 14,
             "canice": 15, "tv": 16, "wslake": 17, "acc_qinsur": 18,
             "acc_qseva": 19}
IN_VECTOR = {"zsoil": (20, NSOIL), "btrani": (24, NSOIL),
             "acc_etrani": (28, NSOIL), "smc": (32, NSOIL),
             "ficeold": (36, NSNOW)}

OUT_VECTOR = {"smc": (0, NSOIL), "acc_etrani": (10, NSOIL)}
OUT_SCALAR = {"canliq": 4, "canice": 5, "tv": 6, "wslake": 7,
              "acc_qinsur": 8, "acc_qseva": 9, "cmc": 14, "ecan": 15,
              "etran": 16, "fwet": 17, "runsrf": 18, "runsub": 19,
              "qtldrn": 20, "ponding1": 21, "ponding2": 22, "qsnbot": 23,
              "qsnsub": 24, "qsnfro": 25, "qsubc": 26, "qfroc": 27,
              "qfrzc": 28, "qmeltc": 29, "qevac": 30, "qdewc": 31,
              "qseva": 32, "qsdew": 33, "qinsur": 34, "snoflow": 35}

#: The physical positional spelling of :func:`gpuwm.core.noahmp_water.water`.
CALL_NAMES = (
    "parameters", "col", "smc", "dt", "ist", "imelt", "ficeold", "fcev",
    "fctr", "elai", "esai", "fveg", "bdfall", "frozen_canopy",
    "frozen_ground", "sfctmp", "qvap", "qdew", "zsoil", "btrani", "qsnow",
    "qrain", "snowhin", "ponding", "canliq", "canice", "tv", "wslake",
    "acc_qinsur", "acc_qseva", "acc_etrani",
)


def _bind_call(args, kwargs) -> dict:
    if len(args) > len(CALL_NAMES):
        raise TypeError(
            f"WATER received {len(args)} positional arguments; expected at "
            f"most {len(CALL_NAMES)}")
    bound = dict(zip(CALL_NAMES, args))
    for name, value in kwargs.items():
        if name in bound:
            raise TypeError(f"WATER argument {name!r} supplied twice")
        bound[name] = value
    missing = [name for name in CALL_NAMES if name not in bound]
    if missing:
        raise TypeError(f"WATER batch call is missing {missing}")
    col = bound["col"]
    if col.nsnow != NSNOW or col.nsoil != NSOIL:
        raise ValueError(
            "the runtime CUDA WATER is pinned to three snow and four soil "
            f"layers, got {col.nsnow} and {col.nsoil}")
    return bound


def pack_water_calls(calls: Sequence[tuple[tuple, dict]], *, bound=None):
    """Return ``(par, fin, iin)`` for the captured physical WATER calls.

    ``bound`` lets a caller that has already resolved the argument lists pass
    them in.  Binding is a dict build per column and this is the hot path at
    width, so it is done once, not once per helper.
    """
    if bound is None:
        bound = [_bind_call(args, kwargs) for args, kwargs in calls]
    count = len(bound)
    par = np.zeros((count, P_STRIDE), dtype=np.float32)
    fin = np.zeros((count, IN_STRIDE), dtype=np.float32)
    iin = np.zeros((count, INT_STRIDE), dtype=np.int32)

    for row, call in enumerate(bound):
        p = call["parameters"]
        for name, base in P_LAYER.items():
            values = np.asarray(getattr(p, name), dtype=np.float32)
            if values.size != NSOIL:
                raise ValueError(
                    f"WATER parameter {name} must have {NSOIL} layers, "
                    f"got {values.size}")
            par[row, base:base + NSOIL] = values
        for name, slot in P_SCALAR.items():
            par[row, slot] = getattr(p, name)

        col = call["col"]
        for name, base in STATE_SLOT.items():
            width = STATE_WIDTH[name]
            value = getattr(col, name)
            if width == 1:
                fin[row, base] = value
            else:
                fin[row, base:base + width] = np.asarray(value,
                                                         dtype=np.float32)
        for name, offset in IN_SCALAR.items():
            fin[row, NSTATE + offset] = call[name]
        for name, (offset, width) in IN_VECTOR.items():
            values = np.asarray(call[name], dtype=np.float32)
            if values.size != width:
                raise ValueError(
                    f"WATER {name} must have {width} elements, "
                    f"got {values.size}")
            fin[row, NSTATE + offset:NSTATE + offset + width] = values

        imelt = np.asarray(call["imelt"], dtype=np.int32)
        if imelt.size != NSNOW:
            raise ValueError(
                f"WATER imelt must carry the {NSNOW} snow layers, "
                f"got {imelt.size}")
        iin[row] = (int(col.isnow), 1 if p.urban_flag else 0, int(p.nroot),
                    int(call["ist"]), 1 if call["frozen_canopy"] else 0,
                    1 if call["frozen_ground"] else 0,
                    int(imelt[0]), int(imelt[1]), int(imelt[2]))
    return par, fin, iin


def _unpack(call: dict, out_row: np.ndarray, isnow: int) -> WaterFluxes:
    """Write the device row back into ``col`` and build the flux record."""
    col = call["col"]
    col.isnow = int(isnow)
    for name, base in STATE_SLOT.items():
        width = STATE_WIDTH[name]
        if width == 1:
            setattr(col, name, np.float32(out_row[base]))
        else:
            getattr(col, name)[...] = out_row[base:base + width]

    values = {}
    for name, (offset, width) in OUT_VECTOR.items():
        start = NSTATE + offset
        values[name] = np.array(out_row[start:start + width],
                                dtype=np.float32)
    for name, offset in OUT_SCALAR.items():
        values[name] = np.float32(out_row[NSTATE + offset])
    # WRF computes ETRANI as a local and NOAHMP_SFLX never reads it, so the
    # device row does not carry it and nothing here invents one.
    values["etrani"] = None
    return WaterFluxes(**values)


def evaluate_water_calls(
        calls: Sequence[tuple[tuple, dict]]) -> list[WaterFluxes]:
    """Evaluate captured physical WATER calls in one device batch."""
    if not calls:
        return []

    import cupy as cp

    from gpuwm.core.kernels import get_kernel

    bound = [_bind_call(args, kwargs) for args, kwargs in calls]
    par, fin, iin = pack_water_calls(calls, bound=bound)
    count = par.shape[0]
    device_out = cp.zeros((count, OUT_STRIDE), dtype=cp.float32)
    device_isnow = cp.zeros(count, dtype=cp.int32)
    kernel = get_kernel("noahmp_water", "k_water")
    blocks = (count + THREADS - 1) // THREADS
    kernel(
        (blocks,), (THREADS,),
        (cp.asarray(par), cp.asarray(fin), cp.asarray(iin), device_out,
         device_isnow, np.int32(count)),
    )
    host_out = cp.asnumpy(device_out)
    host_isnow = cp.asnumpy(device_isnow)
    return [_unpack(call, host_out[row], host_isnow[row])
            for row, call in enumerate(bound)]


__all__ = [
    "CALL_NAMES",
    "evaluate_water_calls",
    "pack_water_calls",
]
