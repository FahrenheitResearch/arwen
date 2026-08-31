"""Device-resident P3 (``mp_physics`` 50/51): compilation, residency, launch.

This module is the CUDA half of the P3 port.  ``gpuwm/core/p3.py`` keeps the
CPU float32 transcription as the explicit reference/debug path; everything
here runs on the card and leaves the prognostic state THERE.

WHAT RUNS WHERE
---------------
On the device, every timestep: the whole of ``p3_main`` and the WRF
wrapper's precipitation conversions, over prognostic fields that never
leave the card, against lookup tables uploaded once per process.
On the host, once per process: ``p3_init`` -- the SHA-256-validated table
parse and the generated rain tables (``gpuwm/core/p3_tables.py``).  That is
a startup cost, not a per-step one.
On the host, per step: nothing but kernel launches.  There is no
host round trip of any prognostic field, no per-column Python loop and no
transpose, because the kernels address gpuwm's native ``(nz, ny, nx)``
storage directly as ``(nk, ncol)``.

LAYOUT, AND WHY
---------------
One thread per COLUMN, k as a loop index inside the thread.  P3 is not
level-parallel: sedimentation carries an adaptive substep loop whose trip
count comes from a Courant reduction over the column, the two goto targets
are column-scope logical flags, and the flux divergence couples adjacent
levels.  A thread-per-column decomposition makes all of that plain
sequential code and therefore preserves the authority's arithmetic ORDER
by construction rather than by care.

Fields stay LEVEL-MAJOR -- element (k, i) at ``k * ncol + i`` -- so a warp
reading level k touches 32 consecutive floats.  That is fully coalesced AND
it is already how a gpuwm ``DomainState`` stores a 3-D field, so the port
adds no transpose anywhere.  The alternative (column-contiguous, which is
the shape the old host path built) gives every thread its own cache line
per load and costs a full-domain transpose twice per step.

SCRATCH ACCOUNTING
------------------
Eighteen ``(nk, ncol)`` float32 companions: twelve carrying values between
kernels (:data:`SCRATCH_SLOTS`) and six of sedimentation workspace
(:data:`SEDW_SLOTS`), which the three sedimentation kernels share because
they run in sequence.  That is 72 bytes per grid cell, allocated once
through ``DomainState.scratch`` and reused for the life of the run.  Six
diagnostics and seven surface fields are additional and are outputs, not
scratch.  Four candidates were deliberately NOT given arrays -- ``inv_dzq``,
``t_old``, ``ze_ice`` and ``ze_rain`` -- because each is exactly
reproducible from a field that is stored, so recomputing them is bit-exact
and 4 * nk * ncol bytes cheaper each.

ARMS
----
``unfused`` is the reference: nine launches, one per step of the authority,
and the arm every agreement number in the receipts was measured on.
``fused`` composes the same device step functions into three launches and
additionally merges the homogeneous-freezing and final-diagnostics k-loops.
Fusing reorders nothing here, but "reorders nothing" is checked by a byte
gate (``tests/test_p3_cuda.py``) rather than asserted, because this program
has already shipped a vectorisation that was wrong in 39 elements out of
37.8 million and only a byte gate caught it.

CONTRACTION
-----------
``-fmad=false`` on every arm.  The Fortran reference arm was built with
``-ffp-contract=off``, so a contracted ``a*b+c`` on the device is a
different number.  With contraction off, plain infix operators in the .cu
are the ``__fmul_rn``/``__fadd_rn`` intrinsics; the kernel file carries a
two-kernel probe and the test compiles the module both ways to show the
equality holds under ``false`` and breaks under ``true``.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from gpuwm.core.kernels import _ENCODING, _preamble

_KDIR = Path(__file__).resolve().parent / "kernels"

#: The single glibc FP32 transcription in this tree (``r_exp``/``r_log``/
#: ``r_pow``).  P3 borrows it exactly the way ``noahmp_driver_gpu`` does --
#: by compiling after it -- rather than carrying a second copy, which is
#: the rule ``gpuwm/core/noahmp_kernel_sources.py`` states and the reason
#: those routines have that name.
_LIBM_SOURCE = _KDIR / "noahmp_leaves.cu"
_P3_SOURCE = _KDIR / "p3.cu"

#: NVRTC options.  ``-fmad=false`` is a correctness flag here, not a tuning
#: knob: it is what makes the .cu's infix arithmetic equal the Fortran
#: reference arm's ``-ffp-contract=off`` statement order.
DEFAULT_OPTIONS: tuple[str, ...] = ("-std=c++17", "-fmad=false")

#: Prognostic + input field slots.  MUST match the F_* defines in p3.cu;
#: tests/test_p3_cuda.py parses the .cu and pins the two lists together.
FIELD_SLOTS: tuple[str, ...] = (
    "qc", "nc", "qr", "nr", "qi", "qir", "ni", "qib",
    "th", "qv", "th_old", "qv_old", "ssat", "pres", "dz",
)
#: Values carried between kernels.  MUST match the S_* defines in p3.cu.
SCRATCH_SLOTS: tuple[str, ...] = (
    "rho", "inv_rho", "qvs", "qvi", "sup", "supi",
    "rhofacr", "rhofaci", "acn", "t", "tmparr1", "qv_cld",
)
#: Sedimentation workspace, shared by the three sedimentation steps.
SEDW_SLOTS: tuple[str, ...] = ("v_q", "v_n", "flux_q", "flux_n",
                               "flux_qir", "flux_bir")
#: THE FULL DIAGNOSTIC SET, decided once, here.  Adding a seventh later
#: moves the history stream and invalidates receipts registered against it,
#: so the Registry's P3 package names all six and all six are emitted on
#: every call: refl_10cm (as zdbz), re_cloud, re_ice, vmi3d, di3d, rhopo3d.
DIAG_SLOTS: tuple[str, ...] = ("zdbz", "effc", "effi", "vmi", "di", "rhopo")
#: Surface fields: the two p3_main precipitation RATES plus the five WRF
#: accumulators the wrapper converts them into (:892-898).
SURF_SLOTS: tuple[str, ...] = ("prt_liq", "prt_sol", "rainnc", "rainncv",
                               "sr", "snownc", "snowncv")
#: p3_init products, in the order p3.cu's T_* defines expect.
TABLE_SLOTS: tuple[str, ...] = ("itab", "itabcoll", "vn_table", "vm_table",
                                "revap_table")

#: Kernel names per arm, in launch order.
ARM_KERNELS: dict[str, tuple[str, ...]] = {
    "unfused": ("p3k_prep", "p3k_kloop1", "p3k_kloopmain",
                "p3k_sed_cloud", "p3k_sed_rain", "p3k_sed_ice",
                "p3k_homofreeze", "p3k_final", "p3k_saveold_precip"),
    "fused": ("p3k_fused_process", "p3k_fused_sed", "p3k_fused_finish"),
}

#: ``run.p3_backend`` -> the arm it selects.  The config spelling is the
#: user-facing one ("cuda" is what a run asks for); the arm name is the
#: verification one ("unfused" is what every agreement number in
#: evidence/p3-cuda-20260829 was measured on).  They are kept as two names
#: on purpose so a receipt can never be read as naming the other thing.
CONFIG_ARM: dict[str, str] = {"cuda": "unfused", "fused": "fused"}

#: Threads per block.  One thread is one column, so this is a pure
#: occupancy knob and changes no number; the byte gate covers it.
DEFAULT_BLOCK = 128


def p3_source() -> str:
    """The exact translation unit nvrtc is handed.

    Assembled the way ``noahmp_driver_gpu.driver_source`` assembles its
    own: preamble, then the shared libm unit, then this scheme's source.
    """
    return (_preamble()
            + _LIBM_SOURCE.read_text(encoding=_ENCODING)
            + _P3_SOURCE.read_text(encoding=_ENCODING))


@lru_cache(maxsize=None)
def p3_module(options: tuple[str, ...] = DEFAULT_OPTIONS,
              source: str | None = None):
    """Compile (once per option set) the P3 translation unit.

    ``source`` exists so a gate can compile a deliberately perturbed copy
    and show the gate can fail; leave it ``None`` for the real thing.
    """
    import cupy as cp

    code = source if source is not None else p3_source()
    module = cp.RawModule(code=code, options=options)
    module.compile()
    try:
        from gpuwm.certify.kernel_manifest import record_module
    except Exception:                                   # pragma: no cover
        pass
    else:
        record_module("gpuwm.core.p3_device:p3"
                      + ("" if source is None else "(substituted-source)"),
                      source=code, options=options, module=module)
    return module


@dataclass(frozen=True)
class P3DeviceTables:
    """``p3_init``'s products, resident on the card for the whole run.

    Uploaded once per process per table root.  They are 320 KiB in total
    (itab 56 KiB, itabcoll 240 KiB, three rain tables 12 KiB each), which
    is small enough to stay hot in L2 and is why no texture or constant
    binding is used.
    """

    itab: object
    itabcoll: object
    vn_table: object
    vm_table: object
    revap_table: object
    pointers: object          # uint64[5], device
    nbytes: int

    @property
    def arrays(self) -> tuple:
        return (self.itab, self.itabcoll, self.vn_table, self.vm_table,
                self.revap_table)


_TABLE_CACHE: dict[tuple[int, str], P3DeviceTables] = {}


def device_tables(runtime=None, root: str | None = None) -> P3DeviceTables:
    """Upload ``p3_init``'s tables and keep them resident.

    The host-side parse and the rain-table generation stay in
    ``gpuwm/core/p3_tables.py``: they are a once-per-process startup cost
    and they carry the loud SHA-256 refusal, which must run before any
    state is touched.
    """
    import cupy as cp

    from gpuwm.core.p3 import p3_init
    from gpuwm.core.p3_tables import p3_table_root

    if root is None:
        root = p3_table_root()
    key = (int(cp.cuda.Device().id), root)
    cached = _TABLE_CACHE.get(key)
    if cached is not None:
        return cached
    if runtime is None:
        runtime = p3_init()
    arrays = []
    for name in TABLE_SLOTS:
        host = np.ascontiguousarray(getattr(runtime, name), dtype=np.float32)
        arrays.append(cp.asarray(host))
    ptrs = cp.asarray(np.array([int(a.data.ptr) for a in arrays],
                               dtype=np.uint64))
    tables = P3DeviceTables(*arrays, pointers=ptrs,
                            nbytes=int(sum(a.nbytes for a in arrays)))
    _TABLE_CACHE[key] = tables
    return tables


def _pointer_array(arrays):
    import cupy as cp
    return cp.asarray(np.array([int(a.data.ptr) for a in arrays],
                               dtype=np.uint64))


def scratch_bytes_per_cell() -> int:
    """Measured, not estimated: bytes of P3 scratch per grid cell."""
    return 4 * (len(SCRATCH_SLOTS) + len(SEDW_SLOTS))


@dataclass
class P3DeviceWorkspace:
    """The per-domain device buffers, allocated once and reused.

    Held by the caller (the ``DomainState`` adapter caches it on the state)
    so that a normal timestep allocates nothing.
    """

    ncol: int
    nk: int
    carriers: dict
    sedw: dict
    flags: object
    carrier_ptrs: object
    sedw_ptrs: object

    @property
    def nbytes(self) -> int:
        total = sum(a.nbytes for a in self.carriers.values())
        total += sum(a.nbytes for a in self.sedw.values())
        return int(total + self.flags.nbytes)


def make_workspace(ncol: int, nk: int, *, allocate=None) -> P3DeviceWorkspace:
    """Allocate (or fetch) the scratch companions for one domain.

    ``allocate(slot, nlev)`` must return a contiguous float32 array of
    exactly ``nlev * ncol`` elements; its shape is the caller's business,
    which is what lets a ``DomainState`` hand back its own ``(nz, ny, nx)``
    scratch slots and keep every P3 allocation inside the allocation gate.
    """
    import cupy as cp

    def default_alloc(slot, nlev):
        return cp.zeros((nlev, ncol), dtype=cp.float32)

    alloc = allocate or default_alloc

    def take(slot, nlev):
        buf = alloc(slot, nlev)
        if buf.size != nlev * ncol:
            raise ValueError(f"P3 workspace slot {slot!r} has {buf.size} "
                             f"elements, expected {nlev * ncol}")
        if buf.dtype != np.float32:
            raise TypeError(f"P3 workspace slot {slot!r} is {buf.dtype}, "
                            "expected float32")
        return buf

    carriers = {n: take("p3_" + n, nk) for n in SCRATCH_SLOTS}
    sedw = {n: take("p3_sed_" + n, nk) for n in SEDW_SLOTS}
    # The two column-scope logical flags, one pair per column, float so the
    # allocation gate prices them with everything else.
    flags = take("p3_flags", 2)
    return P3DeviceWorkspace(
        ncol=ncol, nk=nk, carriers=carriers, sedw=sedw, flags=flags,
        carrier_ptrs=_pointer_array([carriers[n] for n in SCRATCH_SLOTS]),
        sedw_ptrs=_pointer_array([sedw[n] for n in SEDW_SLOTS]))


def run_p3_device(fields: dict, diag: dict, surf: dict, *,
                  workspace: P3DeviceWorkspace,
                  tables: P3DeviceTables | None = None,
                  dt: float, it: int,
                  log_predictNc: bool = False,
                  clbfact_dep: float = 1.0, clbfact_sub: float = 1.0,
                  arm: str = "unfused", block: int = DEFAULT_BLOCK,
                  options: tuple[str, ...] = DEFAULT_OPTIONS,
                  module=None) -> None:
    """Run one P3 step in place on device arrays.

    ``fields``/``diag``/``surf`` map the slot names above to cupy arrays
    that are ``(nk, ncol)`` (2-D) or ``(nk, ny, nx)`` (3-D with ny*nx =
    ncol); surface arrays are ``(ncol,)`` or ``(ny, nx)``.  Nothing is
    copied to the host.
    """
    if arm not in ARM_KERNELS:
        raise ValueError(
            f"unknown P3 arm {arm!r}: the port ships 'unfused' (the "
            "reference every agreement number was measured on) and "
            "'fused' (the same device step functions in three launches). "
            "A third arm has to be verified against 'unfused' byte for "
            "byte before it can be selected here.")
    ncol, nk = workspace.ncol, workspace.nk
    if tables is None:
        tables = device_tables()
    module = module if module is not None else p3_module(options)

    def flat(arr, want_len):
        v = arr.reshape(-1)
        if v.size != want_len:
            raise ValueError(f"P3 device array has {v.size} elements, "
                             f"expected {want_len}")
        if v.dtype != np.float32:
            raise TypeError("P3 device arrays must be float32, got "
                            f"{v.dtype}")
        return arr

    for name in FIELD_SLOTS:
        if name not in fields:
            raise ValueError(f"P3 device launch is missing field {name!r}")
        flat(fields[name], nk * ncol)
    for name in DIAG_SLOTS:
        flat(diag[name], nk * ncol)
    for name in SURF_SLOTS:
        flat(surf[name], ncol)

    f_ptr = _pointer_array([fields[n] for n in FIELD_SLOTS])
    d_ptr = _pointer_array([diag[n] for n in DIAG_SLOTS])
    p_ptr = _pointer_array([surf[n] for n in SURF_SLOTS])

    args = (f_ptr, workspace.carrier_ptrs, workspace.sedw_ptrs, d_ptr, p_ptr,
            tables.pointers, workspace.flags,
            np.int32(ncol), np.int32(nk), np.float32(dt), np.int32(it),
            np.int32(1 if log_predictNc else 0),
            np.float32(clbfact_dep), np.float32(clbfact_sub))
    grid = ((ncol + block - 1) // block,)
    for name in ARM_KERNELS[arm]:
        module.get_function(name)(grid, (block,), args)
