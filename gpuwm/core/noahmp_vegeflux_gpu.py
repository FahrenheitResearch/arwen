"""Batched CUDA execution of Noah-MP's VEGE_FLUX subtree.

The arithmetic lives in :mod:`gpuwm.core.kernels.noahmp_vegeflux.cu`, whose
``dev_vege_flux`` body is already held at max_ulp 0 against the unmodified
WRF v4.6.1 leaf oracle.  This module supplies the runtime-facing part that the
oracle harness did not: physical calls are packed in column order, evaluated
by one CUDA launch, and unpacked as :class:`VegeFluxState` instances.

No global ``-fmad=false`` option is used here.  The kernel pins every FP32
operation with the round-to-nearest intrinsics and every intended FP64 FMA
explicitly; a translation-unit-wide compiler option would silently change
unrelated arithmetic if this source is composed later.

The coefficient tables are copied from the already-authoritative Python
transcriptions into ``__constant__`` memory after compilation.  Keeping them
out of the CUDA source is load-bearing: ptxas 12.x has been observed to
mis-fold FP32 ties at compile time.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Sequence

import numpy as np

import gpuwm.core.noahmp_libm as _libm
import gpuwm.core.noahmp_vegeflux as _host
from gpuwm.core.noahmp_vegeflux import R4, VegeFluxState

NSNOW = 3
NSOIL = 4
NLAYER = NSNOW + NSOIL
LAYER_OFFSET = NSNOW - 1
N_INPUT = 50
N_OUTPUT = 30
THREADS = 64

PARAMETER_NAMES = (
    "DLEAF", "HVT", "CBIOM", "C3PSN", "KC25", "AKC", "KO25", "AKO",
    "AVCMX", "VCMX25", "BP", "MP", "QE25", "FOLNMX",
)

INPUT_NAMES = (
    "dt", "sav", "sag", "lwdn", "ur", "uu", "vv", "sfctmp", "qair",
    "eair", "rhoair", "snowh", "vai", "gammav", "gammag", "fwet",
    "laisun", "laisha", "cwp", "zlvl", "zpd", "z0m", "fveg", "z0mg",
    "emv", "emg", "canliq", "canice", "rsurf", "latheav", "parsun",
    "parsha", "igs", "foln", "co2air", "o2air", "btran", "sfcprs",
    "rhsur", "pahv", "pahg", "eah", "tah", "tv", "tg", "cm", "ch",
    "qsfc", "psfc", "fsr",
)

OUTPUT_NAMES = (
    "EAH", "TAH", "TV", "TG", "CM", "CH", "TAUXV", "TAUYV", "IRG",
    "IRC", "SHG", "SHC", "EVG", "EVC", "TR", "GH", "T2MV", "PSNSUN",
    "PSNSHA", "CANHS", "QSFC", "Q2V", "CAH2", "CHLEAF", "CHUC",
    "RSSUN", "RSSHA", "SAV", "SAG", "FSR",
)

# Positional spelling of gpuwm.core.noahmp_vegeflux.vege_flux.  Five inputs
# (thair, fsno, latheag, q2 and qc) are retained here even though WRF's body
# never reads them; INPUT_NAMES deliberately omits those dead values.
CALL_NAMES = (
    "p", "nsnow", "nsoil", "isnow", "dt", "sav", "sag", "lwdn", "ur",
    "uu", "vv", "sfctmp", "thair", "qair", "eair", "rhoair", "snowh",
    "vai", "gammav", "gammag", "fwet", "laisun", "laisha", "cwp",
    "dzsnso", "zlvl", "zpd", "z0m", "fveg", "z0mg", "emv", "emg",
    "canliq", "fsno", "canice", "stc", "df", "rsurf", "latheav",
    "latheag", "parsun", "parsha", "igs", "foln", "co2air", "o2air",
    "btran", "sfcprs", "rhsur", "q2", "pahv", "pahg", "eah", "tah",
    "tv", "tg", "cm", "ch", "qc", "qsfc", "psfc", "fsr",
)

assert len(INPUT_NAMES) == N_INPUT
assert len(OUTPUT_NAMES) == N_OUTPUT

_KERNEL = Path(__file__).resolve().parent / "kernels" / "noahmp_vegeflux.cu"


def _flat_pairs(values) -> np.ndarray:
    return np.asarray([word for pair in values for word in pair],
                      dtype=np.float64)


def _constant_tables() -> dict[str, np.ndarray]:
    """Return the exact host values uploaded to CUDA constant memory."""
    phys = np.asarray(
        [_host.GRAV, _host.SB, _host.VKC, _host.TFRZ, _host.CPAIR,
         _host.CWAT, _host.CICE, _host.DENH2O, _host.DENICE],
        dtype=np.float32,
    )
    return {
        "c_exp2f_tab": np.asarray(_libm._EXP2F_TAB, dtype=np.uint64),
        "c_exp2f_poly": np.asarray(_libm._EXP2F_POLY, dtype=np.float64),
        "c_exp2f_poly_scaled": np.asarray(
            _libm._EXP2F_POLY_SCALED, dtype=np.float64),
        "c_exp2f_shift": np.asarray(
            [_libm._EXP2F_SHIFT], dtype=np.float64),
        "c_exp2f_shift_scaled": np.asarray(
            [_libm._EXP2F_SHIFT_SCALED], dtype=np.float64),
        "c_exp2f_invln2_scaled": np.asarray(
            [_libm._EXP2F_INVLN2_SCALED], dtype=np.float64),
        "c_logf_tab": _flat_pairs(_libm._LOGF_TAB),
        "c_logf_ln2": np.asarray([_libm._LOGF_LN2], dtype=np.float64),
        "c_logf_poly": np.asarray(_libm._LOGF_A, dtype=np.float64),
        "c_powf_tab": _flat_pairs(_libm._POWF_TAB),
        "c_powf_poly": np.asarray(_libm._POWF_A, dtype=np.float64),
        "c_atanhi": np.asarray(_libm._ATANF_HI, dtype=np.float32),
        "c_atanlo": np.asarray(_libm._ATANF_LO, dtype=np.float32),
        "c_atan_aT": np.asarray(_libm._ATANF_T, dtype=np.float32),
        "c_phys": phys,
        "c_esat_a": np.asarray(_host._A, dtype=np.float32),
        "c_esat_b": np.asarray(_host._B, dtype=np.float32),
        "c_esat_c": np.asarray(_host._C, dtype=np.float32),
        "c_esat_d": np.asarray(_host._D, dtype=np.float32),
    }


def _copy_constant(module, name: str, value: np.ndarray) -> None:
    import cupy as cp

    pointer = module.get_global(name)
    # CuPy releases have exposed both MemoryPointer and (MemoryPointer, size).
    if isinstance(pointer, tuple):
        pointer = pointer[0]
    target = cp.ndarray(value.shape, dtype=value.dtype, memptr=pointer)
    target[...] = cp.asarray(value)


@lru_cache(maxsize=2)
def _module(use_device_libm: bool = False):
    import cupy as cp

    options = ["-std=c++14"]
    if use_device_libm:
        # Negative control only.  CUDA's device libm is not the WRF function.
        options.append("-DUSE_DEVICE_LIBM")
    module = cp.RawModule(
        code=_KERNEL.read_text(encoding="ascii"),
        backend="nvrtc",
        options=tuple(options),
    )
    module.compile()
    for name, value in _constant_tables().items():
        _copy_constant(module, name, value)
    return module


def _bind_call(args, kwargs) -> dict:
    if len(args) > len(CALL_NAMES):
        raise TypeError(
            f"VEGE_FLUX received {len(args)} positional arguments; "
            f"expected at most {len(CALL_NAMES)}")
    bound = dict(zip(CALL_NAMES, args))
    for name, value in kwargs.items():
        if name in bound:
            raise TypeError(f"VEGE_FLUX argument {name!r} supplied twice")
        bound[name] = value
    missing = [name for name in CALL_NAMES if name not in bound]
    if missing:
        raise TypeError(f"VEGE_FLUX batch call is missing {missing}")
    if int(bound["nsnow"]) != NSNOW or int(bound["nsoil"]) != NSOIL:
        raise ValueError(
            "the runtime CUDA VEGE_FLUX is pinned to three snow and four "
            f"soil layers, got {bound['nsnow']} and {bound['nsoil']}")
    for option, admitted in (
        ("opt_sfc", 1), ("opt_crs", 1), ("opt_crop", 0), ("opt_stc", 1),
    ):
        if int(bound.get(option, admitted)) != admitted:
            raise NotImplementedError(
                f"device VEGE_FLUX admits {option}={admitted} only")
    return bound


def evaluate_vege_flux_calls(
        calls: Sequence[tuple[tuple, dict]], *,
        use_device_libm: bool = False) -> list[VegeFluxState]:
    """Evaluate captured physical VEGE_FLUX calls in one device batch."""
    if not calls:
        return []

    import cupy as cp

    bound = [_bind_call(args, kwargs) for args, kwargs in calls]
    count = len(bound)
    inputs = np.empty((count, N_INPUT), dtype=np.float32)
    params = np.empty((count, len(PARAMETER_NAMES)), dtype=np.float32)
    isnow = np.empty(count, dtype=np.int32)
    dzsnso = np.empty((count, NLAYER), dtype=np.float32)
    stc = np.empty_like(dzsnso)
    df = np.empty_like(dzsnso)

    for row, call in enumerate(bound):
        inputs[row] = [call[name] for name in INPUT_NAMES]
        params[row] = [getattr(call["p"], name) for name in PARAMETER_NAMES]
        isnow[row] = int(call["isnow"])
        for layer in range(-NSNOW + 1, NSOIL + 1):
            slot = layer + LAYER_OFFSET
            dzsnso[row, slot] = call["dzsnso"][layer]
            stc[row, slot] = call["stc"][layer]
            df[row, slot] = call["df"][layer]

    device_out = cp.empty((count, N_OUTPUT), dtype=cp.float32)
    kernel = _module(use_device_libm).get_function("k_vegeflux")
    blocks = (count + THREADS - 1) // THREADS
    kernel(
        (blocks,), (THREADS,),
        (cp.asarray(inputs), cp.asarray(params), cp.asarray(isnow),
         cp.asarray(dzsnso), cp.asarray(stc), cp.asarray(df), device_out,
         np.int32(count)),
    )
    host_out = cp.asnumpy(device_out)

    states = []
    for row in host_out:
        state = VegeFluxState()
        for name, value in zip(OUTPUT_NAMES, row):
            setattr(state, name, R4(value))
        # ENERGY consumes only the numerical return arguments.  The host-only
        # trace fields are diagnostic metadata and are not trajectory state.
        state.iters = 0
        state.branches = {"device_batch": True}
        states.append(state)
    return states


class VegeFluxBatchPending(RuntimeError):
    """Internal control transfer used to pause a column at VEGE_FLUX."""

    def __init__(self, index: int):
        super().__init__(f"VEGE_FLUX call {index} is pending device execution")
        self.index = index


class VegeFluxBatch:
    """Capture VEGE_FLUX calls, execute them once, then replay their results.

    The runtime no longer uses this.  It paused a column by *restarting* it,
    which meant every host statement before VEGE_FLUX ran twice;
    :func:`gpuwm.core.noahmp_energy.energy_steps` replaced it with a
    continuation that resumes the same frame.  What is left of this class is
    the fixture collector: ``tests/test_noahmp_vegeflux_cuda.py`` uses
    ``capture`` to lift the physical argument list out of the unmodified-WRF
    oracle rows, which is still the cleanest way to build that batch.
    """

    def __init__(self):
        self.calls: list[tuple[tuple, dict]] = []
        self.results: list[VegeFluxState] = []
        self._replay_index = 0

    def capture(self, *args, **kwargs):
        index = len(self.calls)
        self.calls.append((args, kwargs))
        raise VegeFluxBatchPending(index)

    def evaluate(self) -> None:
        self.results = evaluate_vege_flux_calls(self.calls)
        if len(self.results) != len(self.calls):
            raise RuntimeError(
                f"VEGE_FLUX batch returned {len(self.results)} results for "
                f"{len(self.calls)} calls")

    def replay(self, *args, **kwargs):
        del args, kwargs
        if self._replay_index >= len(self.results):
            raise RuntimeError("VEGE_FLUX replay consumed beyond its batch")
        result = self.results[self._replay_index]
        self._replay_index += 1
        return result

    def assert_fully_replayed(self) -> None:
        if self._replay_index != len(self.results):
            raise RuntimeError(
                f"VEGE_FLUX replay consumed {self._replay_index} of "
                f"{len(self.results)} results")


__all__ = [
    "VegeFluxBatch",
    "VegeFluxBatchPending",
    "evaluate_vege_flux_calls",
]
