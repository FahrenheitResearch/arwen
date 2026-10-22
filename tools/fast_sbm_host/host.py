"""Host build of WOOF's mp_physics = 30 device port, for verification only.

The device unit's headers (``gpuwm/core/kernels/fast_sbm_*.cuh``) compile
under a host C++ compiler with ``fsbm_host.cpp`` beside this file, which
drives ``FAST_SBM`` over a WRF tile in WRF's own memory layout and loop
order.  This module builds that library on demand (a C++17 compiler is
needed; tests skip by name without one), loads it with ctypes and offers
the replay helpers the oracle tests use.  Nothing here runs in a forecast.
On Windows the library is a DLL built by MinGW-w64 g++ (the compiler the
windows-2025 runner carries).
"""

from __future__ import annotations

import ctypes
import hashlib
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from gpuwm.core.fast_sbm_contract import FLAG_FIELDS, TABLE_ORDER, STATE_FIELDS

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
KERNEL_DIR = REPO_ROOT / "gpuwm" / "core" / "kernels"
HEADERS = ("fast_sbm_common.cuh", "fast_sbm_coll.cuh", "fast_sbm_aux.cuh",
           "fast_sbm_nucl.cuh", "fast_sbm_core.cuh", "fast_sbm_column.cuh",
           "fast_sbm_init.cuh", "fast_sbm_driver.cuh")
SOURCE = HERE / "fsbm_host.cpp"
# Strict IEEE: no contraction, no fast-math; the arithmetic is the
# Fortran oracle's (-O0 -ffp-contract=off) operation for operation.
CXXFLAGS = ("-std=c++17", "-O2", "-ffp-contract=off", "-fno-fast-math",
            "-fPIC", "-shared")

WINDOWS = os.name == "nt"
#: Windows: link libstdc++, libgcc and winpthread into the DLL.  Without it
#: the DLL imports libstdc++-6.dll and libgcc_s_seh-1.dll from the
#: compiler's folder, and ctypes (Python 3.8+) resolves a DLL's imports
#: from the system folders and the DLL's own folder, never PATH, so the
#: library builds but does not load: "Could not find module ...
#: libfsbm_host_95e0dbb8ce80c686.so (or one of its dependencies)", the
#: windows-2025 cpu leg at 5e0e1ba59.  The math calls still go to the
#: platform libm; static linking moves no arithmetic.
BUILD_FLAGS = CXXFLAGS + ("-static",) if WINDOWS else CXXFLAGS
LIB_SUFFIX = ".dll" if WINDOWS else ".so"


def _gnu_libc() -> str:
    try:
        return os.confstr("CS_GNU_LIBC_VERSION") or ""
    except (AttributeError, ValueError, OSError):  # no confstr off POSIX
        return ""


#: Whether the platform libm is glibc's.  The host build calls the platform
#: libm (fast_sbm_common.cuh "libm"; fast_sbm_init.cuh calls it directly),
#: and the reference bits were computed calling glibc's exp, log and pow:
#: WRF's Fortran oracle, and the packed tables checked against it.  So bit
#: identity with them is the port's property only where that libm is glibc.
GLIBC_LIBM = _gnu_libc().startswith("glibc")
NOT_GLIBC_REASON = (
    "bit identity with WRF's Fortran oracle needs glibc's libm: the oracle's "
    "bits (and the packed tables checked against them) were computed calling "
    "glibc's exp/log/pow, and this host build calls the platform libm, which "
    f"on {sys.platform} is not glibc")

def flags_wrf(hail: int = 1) -> dict:
    """WRF as written (the oracle preset)."""
    f = {name: 0 for name in FLAG_FIELDS}
    f.update(hail=hail, dtcond_carry=1, breakup_retry_cap=64)
    return f


def flags_corrected(hail: int = 1) -> dict:
    """The port's default: every catalogued defect fixed (D8 kept)."""
    f = {name: 1 for name in FLAG_FIELDS}
    f.update(hail=hail, dtcond_carry=0, breakup_retry_cap=64)
    return f


class HostBuildUnavailable(RuntimeError):
    pass


def _compiler() -> str | None:
    for cxx in (os.environ.get("CXX"), "g++", "c++", "clang++"):
        if cxx and shutil.which(cxx):
            return cxx
    return None


def _source_digest() -> str:
    h = hashlib.sha256()
    for name in HEADERS:
        h.update((KERNEL_DIR / name).read_bytes())
    h.update(SOURCE.read_bytes())
    h.update(" ".join(BUILD_FLAGS).encode())
    return h.hexdigest()[:16]


_LIB = None


def library() -> ctypes.CDLL:
    """Build (once per source digest) and load the host library."""
    global _LIB
    if _LIB is not None:
        return _LIB
    cxx = _compiler()
    if cxx is None:
        raise HostBuildUnavailable("no C++ compiler (g++, c++ or clang++) on PATH")
    cache = Path(os.environ.get("GPUWM_FSBM_HOST_CACHE",
                                Path(tempfile.gettempdir()) / "gpuwm-fsbm-host"))
    cache.mkdir(parents=True, exist_ok=True)
    so = cache / f"libfsbm_host_{_source_digest()}{LIB_SUFFIX}"
    if not so.exists():
        tmp = so.with_suffix(f".{os.getpid()}.tmp")
        cmd = [cxx, *BUILD_FLAGS, f"-I{KERNEL_DIR}", "-o", str(tmp), str(SOURCE)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"host build failed:\n{' '.join(cmd)}\n{r.stderr}")
        try:
            os.replace(tmp, so)
        except PermissionError:
            # Windows will not replace a DLL another process has loaded.
            # Every xdist worker imports the oracle tests and builds at
            # collection (the four collection errors at 5e0e1ba59 were one
            # per worker), so a later builder finds the first one's library
            # in use.  Same digest, same build: keep that one.
            tmp.unlink(missing_ok=True)
            if not so.exists():
                raise
    lib = ctypes.CDLL(str(so))
    for p in ("f64", "f32"):
        getattr(lib, f"fsbm_host_new_{p}").restype = ctypes.c_void_p
        getattr(lib, f"fsbm_host_new_{p}").argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        getattr(lib, f"fsbm_host_free_{p}").argtypes = [ctypes.c_void_p]
        getattr(lib, f"fsbm_host_derived_{p}").argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        real = ctypes.c_double if p == "f64" else ctypes.c_float
        getattr(lib, f"fsbm_host_call_{p}").argtypes = [
            ctypes.c_void_p, ctypes.c_int, real, real, real, ctypes.c_void_p,
            ctypes.c_void_p, ctypes.c_void_p]
        getattr(lib, f"fsbm_host_call_{p}").restype = ctypes.c_int
    if lib.fsbm_host_flag_count() != len(FLAG_FIELDS):
        raise RuntimeError("FsbmFlags and FLAG_FIELDS disagree")
    lib.fsbm_host_ledger_f64.argtypes = lib.fsbm_host_call_f64.argtypes + [ctypes.c_void_p]
    lib.fsbm_host_ledger_f64.restype = ctypes.c_int
    _LIB = lib
    return lib


def tables_from_read(read: dict, dtype) -> list[np.ndarray]:
    """The ABI's 27 flat Fortran-order tables from WRF's READ arrays
    (keys like ``ywll_1000mb``, each with Fortran dims)."""
    def f(a):
        return np.ascontiguousarray(np.asarray(a, dtype=dtype).ravel(order="F"))
    out = []
    for name in TABLE_ORDER:
        if name == "ywll":
            arr = np.stack([read["ywll_1000mb"], read["ywll_750mb"], read["ywll_500mb"]], axis=-1)
        elif name in ("ywls", "ywlg", "ywlh", "ywss"):
            arr = np.stack([read[f"{name}_300mb"], read[f"{name}_500mb"], read[f"{name}_750mb"]], axis=-1)
        else:
            arr = read[name]
        out.append(f(arr))
    return out


DERIVED_LAYOUT = (("xl_mg", 33), ("xs_mg", 33), ("xg_mg", 33), ("xh_mg", 33),
                  ("chucm", 1089), ("ima", 1089), ("ecoal_wrf", 1089),
                  ("ecoal_own", 1089), ("prob", 33), ("gain_var_new", 1089),
                  ("dropradii", 33), ("fccnr_con", 43), ("fccnr_mar", 43),
                  ("xccn", 43), ("rccn", 43), ("ikr_spon_break", 1))


TALLY_NAMES = ("clip_mass", "activated_mass", "homfrz_mass", "ice_nucl_mass",
               "freez_mass", "melt_mass", "rime_mass", "homfrz_point_mass",
               "n_cond1", "n_cond2", "n_cond3", "n_coal", "n_freez", "n_melt",
               "n_spont", "n_homfrz_point", "cloud_base_level_first_column")


class Host:
    """One precision's host FAST_SBM with its tables and behaviour."""

    def __init__(self, tables: list[np.ndarray], flags: dict, precision: str = "f64"):
        self.lib = library()
        self.precision = precision
        self.dtype = np.float64 if precision == "f64" else np.float32
        self._tables = [np.ascontiguousarray(t, dtype=self.dtype) for t in tables]
        ptrs = (ctypes.c_void_p * len(self._tables))(*[t.ctypes.data for t in self._tables])
        self._flags = np.array([int(flags[n]) for n in FLAG_FIELDS], dtype=np.int32)
        self.h = getattr(self.lib, f"fsbm_host_new_{precision}")(ptrs, self._flags.ctypes.data)

    def __del__(self):
        h = getattr(self, "h", None)
        if h:
            getattr(self.lib, f"fsbm_host_free_{self.precision}")(h)
            self.h = None

    def derived(self) -> dict[str, np.ndarray]:
        n = self.lib.fsbm_host_derived_count()
        buf = np.zeros(n, dtype=np.float64)
        getattr(self.lib, f"fsbm_host_derived_{self.precision}")(self.h, buf.ctypes.data)
        out, o = {}, 0
        for name, size in DERIVED_LAYOUT:
            out[name] = buf[o:o + size].copy()
            o += size
        return out

    def call(self, itimestep: int, dt: float, dx: float, dy: float, dims, state: dict, *, ledger=None):
        """FAST_SBM on ``state`` (arrays in WRF memory layout, Fortran
        order, this precision), in place.  Returns (status, tally)."""
        arrs = []
        for name in STATE_FIELDS:
            a = state[name]
            if a.dtype != self.dtype or not a.flags.f_contiguous:
                raise TypeError(f"{name} must be Fortran-contiguous {self.dtype}")
            arrs.append(a)
        ptrs = (ctypes.c_void_p * len(arrs))(*[a.ctypes.data for a in arrs])
        d = np.asarray(dims, dtype=np.int32)
        tally = np.zeros(len(TALLY_NAMES), dtype=np.float64)
        args=(self.h,int(itimestep),dt,dx,dy,d.ctypes.data,ptrs,tally.ctypes.data)
        if ledger is None:
            status=getattr(self.lib,f"fsbm_host_call_{self.precision}")(*args)
        else:
            if self.precision!="f64" or ledger.dtype!=np.float64 or ledger.shape!=(10,):
                raise TypeError("the physical water ledger needs ten float64 words and an f64 host")
            status=self.lib.fsbm_host_ledger_f64(*args,ledger.ctypes.data)
        return int(status), dict(zip(TALLY_NAMES, tally))


@dataclass
class Tile:
    """WRF dims of one oracle tile: ids..kte as FAST_SBM receives them."""
    ids: int; ide: int; jds: int; jde: int; kds: int; kde: int
    ims: int; ime: int; jms: int; jme: int; kms: int; kme: int
    its: int; ite: int; jts: int; jte: int; kts: int; kte: int

    def dims(self):
        return [self.ids, self.ide, self.jds, self.jde, self.kds, self.kde,
                self.ims, self.ime, self.jms, self.jme, self.kms, self.kme,
                self.its, self.ite, self.jts, self.jte, self.kts, self.kte]

    def blank(self, dtype) -> dict:
        ni, nk, nj = self.ime - self.ims + 1, self.kme - self.kms + 1, self.jme - self.jms + 1
        s = {}
        for name in STATE_FIELDS:
            if name == "chem":
                s[name] = np.zeros((ni, nk, nj, 143), dtype=dtype, order="F")
            elif name in ("xland", "rainnc", "rainncv", "snownc", "snowncv", "graupelnc", "graupelncv"):
                s[name] = np.zeros((ni, nj), dtype=dtype, order="F")
            else:
                s[name] = np.zeros((ni, nk, nj), dtype=dtype, order="F")
        # the runners' fill values (run_column.F90 / run_kc2.F90 init_state)
        s["dz8w"][...] = 1.0
        s["rho"][...] = 1.0
        s["p"][...] = 1.0e5
        s["pii"][...] = 1.0
        s["th"][...] = 300.0
        s["xland"][...] = 1.0
        return s

    def i(self, i):
        return i - self.ims

    def k(self, k):
        return k - self.kms

    def j(self, j):
        return j - self.jms


KC1_TILE = Tile(1, 5, 1, 2, 1, 41, 0, 5, 0, 2, 1, 41, 1, 4, 1, 1, 1, 40)


def kc2_tile(edge: bool = False) -> Tile:
    t0 = 1 if edge else 2
    return Tile(1, 5, 1, 5, 1, 51, 0, 6, 0, 6, 1, 51, t0, t0 + 2, t0, t0 + 2, 1, 50)
