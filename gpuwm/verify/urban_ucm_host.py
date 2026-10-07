"""FFI driver for the compiled CPU twin of ``urban_ucm.cu``.

All numerical work is the same C++ source the GPU compiles. This module
only validates and packs arrays and passes their pointers to the native twin.
"""
from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np

from gpuwm.core.urban_ucm import (
    UCM_COLUMN_INPUTS, UCM_COLUMN_OUTPUTS, UCM_STATE_PLANES,
)


class HostTwin:
    """A compiled ``urban_ucm_host.cpp`` shared library, loaded explicitly."""

    def __init__(self, path: str | Path):
        self.lib = ctypes.CDLL(str(Path(path).resolve()))
        void = ctypes.c_void_p
        self.lib.ucm_host_columns.argtypes = [void] * 7 + [ctypes.c_int]
        self.lib.ucm_host_columns.restype = None

    def run_columns(self, params, inputs, state):
        """Return outputs, renewed state and status codes as host arrays."""
        n = int(np.asarray(inputs["ta"]).size)
        incoming = []
        for name in UCM_COLUMN_INPUTS:
            dtype = np.int32 if name in ("utype", "jmonth") else np.float32
            array = np.ascontiguousarray(inputs[name], dtype=dtype)
            if array.shape != (n,):
                raise ValueError(f"{name}: expected {(n,)}, got {array.shape}")
            incoming.append(array)
        outgoing = [np.zeros(n, np.float32) for _ in UCM_COLUMN_OUTPUTS]
        renewed = []
        for name, layers in UCM_STATE_PLANES:
            shape = (n,) if layers == 0 else (layers, n)
            array = np.array(state.get(name, np.zeros(shape, np.float32)),
                             dtype=np.float32, order="C", copy=True)
            if array.shape != shape:
                raise ValueError(f"{name}: expected {shape}, got {array.shape}")
            renewed.append(array)
        table = np.ascontiguousarray(params.table, dtype=np.float32)
        globals_ = np.ascontiguousarray(params.globals_, dtype=np.float32)
        switches = np.ascontiguousarray(params.switches, dtype=np.int32)
        inptrs = np.asarray([a.ctypes.data for a in incoming], np.uint64)
        outptrs = np.asarray([a.ctypes.data for a in outgoing], np.uint64)
        stateptrs = np.asarray([a.ctypes.data for a in renewed], np.uint64)
        codes = np.zeros(n, np.int32)
        self.lib.ucm_host_columns(
            inptrs.ctypes.data, outptrs.ctypes.data, stateptrs.ctypes.data,
            table.ctypes.data, globals_.ctypes.data, switches.ctypes.data,
            codes.ctypes.data, n,
        )
        return (dict(zip(UCM_COLUMN_OUTPUTS, outgoing)),
                {name: a for (name, _), a in zip(UCM_STATE_PLANES, renewed)},
                codes)
