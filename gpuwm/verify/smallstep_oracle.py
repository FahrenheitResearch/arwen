"""Pinned compiled-WRF acoustic oracle plumbing, with no tolerance policy."""
from __future__ import annotations
import ctypes
import json
import os
from pathlib import Path
import numpy as np

from gpuwm.verify.wrf471_fixtures import require_fixture_dir

# Test data in a source checkout, not package data (gpuwm.verify.wrf471_fixtures).
ORACLE_DIR = Path(__file__).resolve().parents[2]/"tests"/"data"/"wrf471_smallstep"


class WRFOracle:
    """Call unmodified WRF with its real staggered, haloed argument lists."""
    def __init__(self, library_path=None, nx=8, ny=7, nz=44, periodic=False):
        path = library_path or os.environ.get("GPUWM_SMALLSTEP_ORACLE_LIB")
        if path is None:
            raise FileNotFoundError("Set GPUWM_SMALLSTEP_ORACLE_LIB to the pinned compiled oracle library")
        self.path = Path(path)
        self.library = ctypes.CDLL(str(self.path))
        self.schema = json.loads(self.path.with_name("schema.json").read_text())
        self.nx, self.ny, self.nz, self.periodic = nx, ny, nz, periodic
        self.dims = dict(ids=1,ide=nx+1,jds=1,jde=ny+1,kds=1,kde=nz+1,
                         ims=0,ime=nx+2,jms=0,jme=ny+2,kms=1,kme=nz+1,
                         its=1,ite=nx+1,jts=1,jte=ny+1,kts=1,kte=nz+1)

    def array(self, engine_array):
        """Expand (k,j,i), (j,i), or vertical input into WRF memory."""
        a = np.asarray(engine_array, dtype=np.float32)
        if a.ndim == 1:
            return np.asfortranarray(np.pad(a, (0, self.nz+1-a.size), mode="edge"))
        if a.ndim not in (2, 3):
            raise ValueError(a.shape)
        ny, nx = a.shape[-2:]
        # The physical staggered face at nx/ny is present in the input when applicable.
        pad = ((1, self.ny+2-ny), (1, self.nx+2-nx))
        if a.ndim == 3:
            pad = ((0, self.nz+1-a.shape[0]), *pad)
        expanded = np.pad(a, pad, mode="edge")
        if self.periodic:
            # A staggered terminal face aliases the first face. The period
            # remains nx/ny, not the nx+1/ny+1 extent of that storage.
            iy = (np.arange(self.ny+3)-1) % self.ny
            ix = (np.arange(self.nx+3)-1) % self.nx
            expanded = a[..., iy[:,None], ix[None,:]]
            if a.ndim == 3 and a.shape[0] < self.nz+1:
                expanded = np.pad(expanded,((0,self.nz+1-a.shape[0]),(0,0),(0,0)),mode="edge")
        return np.asfortranarray(expanded.transpose(2,0,1) if a.ndim == 3 else expanded.T)

    def extract(self, wrf_array, engine_shape):
        a = np.asarray(wrf_array)
        if len(engine_shape) == 1:
            return np.ascontiguousarray(a[:engine_shape[0]])
        if len(engine_shape) == 2:
            ny,nx = engine_shape
            return np.ascontiguousarray(a[1:1+nx,1:1+ny].T)
        nz,ny,nx = engine_shape
        return np.ascontiguousarray(a[1:1+nx,:nz,1:1+ny].transpose(1,2,0))

    def call(self, routine, **arguments):
        schema = self.schema["routines"][routine]
        pointers, keepalive = [], []
        for name in schema["args"]:
            decl = schema["declarations"][name]
            value = arguments.get(name, self.dims.get(name))
            if value is None:
                raise TypeError(f"{routine} requires {name}")
            if name == "config_flags":
                packed = []
                for field, t in self.schema["config"].items():
                    v = value.get(field, 0)
                    packed.append(np.float32(v).view(np.int32) if t == "real" else int(v))
                value = np.asarray(packed, dtype=np.int32)
            elif not decl["dimensions"]:
                value = np.asarray(value, dtype=np.float32 if decl["type"] == "real" else np.int32)
            else:
                if not isinstance(value, np.ndarray) or value.dtype != np.float32 or not value.flags.f_contiguous:
                    raise TypeError(f"{routine}.{name} must be a Fortran-contiguous float32 array")
                shape = []
                limits = dict(self.dims, **{k:int(v) for k,v in arguments.items() if k in self.dims})
                for extent in decl["dimensions"].split(","):
                    low, high = extent.split(":")
                    low = limits[low] if low in limits else int(low)
                    high = limits[high] if high in limits else int(high)
                    shape.append(high-low+1)
                if value.shape != tuple(shape):
                    raise ValueError(f"{routine}.{name}: WRF memory shape {tuple(shape)} required, got {value.shape}")
            keepalive.append(value)
            pointers.append(ctypes.c_void_p(value.ctypes.data))
        fn = getattr(self.library, "oracle_"+routine)
        fn.argtypes = [ctypes.c_void_p]*len(pointers)
        fn.restype = None
        fn(*pointers)
        return arguments


def load_cases():
    """Real initial and evolved state crops plus explicit stress transformations."""
    directory = require_fixture_dir(ORACLE_DIR, "small-step")
    metadata = json.loads((directory/"cases.json").read_text())
    with np.load(directory/"real-state.npz", allow_pickle=False) as archive:
        for case in metadata["cases"]:
            yield case["name"], {key: archive[case["prefix"]+key].copy()
                                 for key in case["fields"]}, case


def word_metrics(actual, reference):
    """Measure every IEEE-754 word, preserving signed zeros and nonfinite values."""
    from gpuwm.core.fp32_ulp import fp32_ulp_distance
    a, b = np.asarray(actual,dtype=np.float32), np.asarray(reference,dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError((a.shape,b.shape))
    changed = a.view(np.uint32) != b.view(np.uint32)
    finite = np.isfinite(a)&np.isfinite(b)
    ulp = fp32_ulp_distance(a[finite], b[finite])
    return {"words": int(a.size), "different_words": int(changed.sum()),
            "max_ulp": int(ulp.max(initial=0)),
            "nonfinite_different_words": int((changed&~finite).sum()),
            "max_abs": float(np.abs(a[finite].astype(np.float64)-b[finite]).max(initial=0))}
