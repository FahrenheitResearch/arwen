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
The reference uses one thread per column. The level arms divide the
independent process updates among four threads per column, with adjacent
x threads still reading adjacent columns. Column flags are combined with
integer OR. The sedimentation level arm also divides each substep among
four threads: it reduces the Courant maximum, completes all fluxes, then
updates levels. Floating point sums retain their original order.

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
``levels`` uses four launches with level-parallel preparation, process
rates and final diagnostics. ``sedlevels`` also parallelises sedimentation
substeps and is selected by ``cuda``. The sequential arms remain available
as references; ``tests/test_p3_speed_levels.py`` compares their buffers.

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
    "sedlevels": ("p3k_levels_prepare", "p3k_levels_process",
                  "p3k_levels_sed", "p3k_levels_finish"),
    "levels": ("p3k_levels_prepare", "p3k_levels_process",
               "p3k_fused_sed", "p3k_levels_finish"),
}

#: ``run.p3_backend`` -> the arm it selects.  The config spelling is the
#: user-facing one ("cuda" is what a run asks for); the arm name is the
#: verification one. Historical authority receipts name "unfused";
#: default runs select "sedlevels" and explicit "fused" keeps its meaning.
CONFIG_ARM: dict[str, str] = {"cuda": "sedlevels", "fused": "fused"}

#: Launch shape (columns per block at most, level groups) of each level-group
#: kernel.  The group counts and column caps are the shared-array shapes in
#: p3.cu (flags[8][16], flags[16][8], co[16][4]); a column's levels are dealt
#: to its groups round-robin, and only flags and the Courant maximum are
#: combined across groups, so the shape moves speed, never bits.
LEVEL_GROUP_SHAPE: dict[str, tuple[int, int]] = {
    "p3k_levels_prepare": (16, 8),
    "p3k_levels_process": (8, 16),
    "p3k_levels_sed": (4, 16),
}

#: Threads per block.  One thread is one column, so this is a pure
#: occupancy knob and changes no number; the byte gate covers it.
DEFAULT_BLOCK = 128


def p3_source(*, kernel_dir: Path = _KDIR) -> str:
    """The exact translation unit nvrtc is handed.

    Assembled the way ``noahmp_driver_gpu.driver_source`` assembles its
    own: preamble, then the shared libm unit, then this scheme's source.
    ``kernel_dir`` composes it from another tree's kernel files (the A146
    census gates the tree it scans, A193).
    """
    kdir = Path(kernel_dir)
    return (_preamble(kdir)
            + (kdir / _LIBM_SOURCE.name).read_text(encoding=_ENCODING)
            + (kdir / _P3_SOURCE.name).read_text(encoding=_ENCODING))


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

    def upload():
        source = p3_init() if runtime is None else runtime
        arrays = []
        for name in TABLE_SLOTS:
            host = np.ascontiguousarray(getattr(source, name), dtype=np.float32)
            arrays.append(cp.asarray(host))
        ptrs = cp.asarray(np.array([int(a.data.ptr) for a in arrays],
                                   dtype=np.uint64))
        return P3DeviceTables(*arrays, pointers=ptrs,
                              nbytes=int(sum(a.nbytes for a in arrays)))

    # Published with its upload event: every slab on the card reads these
    # tables from its own stream, and an upload still in flight on the first
    # slab's stream is not a table the others may read.
    from gpuwm.core.device_cache import cached_ready
    return cached_ready(cp, _TABLE_CACHE, key, upload)


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
            f"unknown P3 arm {arm!r}: expected one of "
            f"{tuple(ARM_KERNELS)}")
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
        if name == "p3k_levels_finish":
            module.get_function(name)((grid[0], nk), (block,), args)
        elif name in LEVEL_GROUP_SHAPE:
            columns, groups = LEVEL_GROUP_SHAPE[name]
            level_block = min(block, columns)
            module.get_function(name)(
                ((ncol + level_block - 1) // level_block,),
                (level_block, groups), args)
        else:
            module.get_function(name)(grid, (block,), args)


# ---------------------------------------------------------------------------
# The radar observation operator H_Z(x): P3's reflectivity as a PURE function
# of the state.  gpuwm/core/kernels/p3_zdiag.cu carries the statement-level
# account; this is its compile site and launcher, plus the host replay of
# the same statements on the CPU authority's own helpers.
# ---------------------------------------------------------------------------

_ZDIAG_SOURCE = _KDIR / "p3_zdiag.cu"

#: Fields the operator reads, in kernel argument order.  ``th`` is the FULL
#: potential temperature and ``pres`` the full pressure, as the step reads
#: them (gpuwm/core/p3.py apply: ``mp_th = thb + thp``, ``pres = state.p``).
REFLECTIVITY_FIELDS: tuple[str, ...] = (
    "qr", "nr", "qi", "qir", "ni", "qib", "th", "pres")


def _clear_air_dbz() -> float:
    """``10*log10((1e-22 + 1e-22)*1e18)`` in the kernel's float32 order.

    The two accumulator seeds of p3_final_level (WRF module_mp_p3.F
    :2286-2287) with no species added: the scheme's own reflectivity of a
    cell with no rain and no ice.  The device evaluates ``log10`` in double
    and rounds once to float32 (``p3_log10``), which is what this does.
    """
    seed = np.float32(1.0e-22)
    total = np.float32(np.float32(seed + seed) * np.float32(1.0e18))
    return float(np.float32(np.float32(10.0)
                            * np.float32(np.log10(np.float64(total)))))


#: The operator's ONE clear-air value, dBZ: what every cell with no rain and
#: no ice above QSMALL reads (-36.9897).  ``gpuwm.da.obsop`` records it as
#: P3's clear-air floor; tests/test_p3_reflectivity_gpu.py holds the kernel
#: to it bit for bit.
CLEAR_AIR_DBZ = _clear_air_dbz()


def p3_reflectivity_source(*, kernel_dir: Path = _KDIR) -> str:
    """The forecast unit (:func:`p3_source`) with the operator appended.

    The forecast unit's bytes are unchanged, so the forecast module and its
    receipts do not move; the operator compiles as its own module.
    """
    kdir = Path(kernel_dir)
    return (p3_source(kernel_dir=kdir)
            + (kdir / _ZDIAG_SOURCE.name).read_text(encoding=_ENCODING))


@lru_cache(maxsize=None)
def p3_reflectivity_module(options: tuple[str, ...] = DEFAULT_OPTIONS):
    """Compile (once per option set) the operator's translation unit."""
    import cupy as cp

    code = p3_reflectivity_source()
    module = cp.RawModule(code=code, options=options)
    module.compile()
    try:
        from gpuwm.certify.kernel_manifest import record_module
    except Exception:                                   # pragma: no cover
        pass
    else:
        record_module("gpuwm.core.p3_device:p3_reflectivity",
                      source=code, options=options, module=module)
    return module


def run_p3_reflectivity(fields: dict, out, *,
                        tables: P3DeviceTables | None = None,
                        block: int = 256,
                        options: tuple[str, ...] = DEFAULT_OPTIONS):
    """Write P3's reflectivity of ``fields`` into ``out`` (dBZ, float32).

    ``fields`` maps :data:`REFLECTIVITY_FIELDS` to float32 cupy arrays of
    one common size; ``out`` is a contiguous float32 cupy array of that
    size.  The inputs are read only.
    """
    import cupy as cp

    if tables is None:
        tables = device_tables()
    size = int(out.size)
    arrays = []
    for name in REFLECTIVITY_FIELDS:
        if name not in fields:
            raise ValueError(
                f"P3 reflectivity needs field {name!r}; it reads "
                f"{REFLECTIVITY_FIELDS}")
        value = fields[name]
        if int(value.size) != size:
            raise ValueError(
                f"P3 reflectivity field {name!r} has {value.size} elements, "
                f"the output has {size}")
        if value.dtype != np.float32:
            raise TypeError(
                f"P3 reflectivity field {name!r} is {value.dtype}; the "
                "operator runs the scheme's float32 statements")
        arrays.append(cp.ascontiguousarray(value).reshape(-1))
    if out.dtype != np.float32 or not out.flags.c_contiguous:
        raise TypeError("P3 reflectivity output must be contiguous float32")
    module = p3_reflectivity_module(options)
    module.get_function("p3k_reflectivity")(
        ((size + block - 1) // block,), (block,),
        (*arrays, tables.itab, out.reshape(-1), np.int32(size)))
    return out


def p3_reflectivity_host(fields: dict) -> np.ndarray:
    """The same statements on the host, on the CPU authority's helpers.

    ``gpuwm.core.p3`` is the float32 transcription the device port is
    measured against; this replays p3_final_level's Z half per cell with
    its ``get_rain_dsd2``, ``impose_max_total_Ni``, ``calc_bulkRhoRime``,
    ``find_lookupTable_indices_1a`` and ``access_lookup_table`` on local
    values.  One Python iteration per cell holding rain or ice: a
    reference and a small-domain path, not a production one.
    """
    from gpuwm.core import p3 as P

    f32 = np.float32
    host = {name: np.asarray(fields[name], dtype=np.float32)
            for name in REFLECTIVITY_FIELDS}
    shape = host["qr"].shape
    out = np.full(shape, f32(CLEAR_AIR_DBZ), dtype=np.float32)
    active = (host["qr"] >= P.QSMALL) | (host["qi"] >= P.QSMALL)
    if not active.any():
        return out
    itab = P.p3_init().itab
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        for index in zip(*np.nonzero(active)):
            pres = host["pres"][index]
            tm = f32((pres * f32(1.0e-5)) ** (P.RD * P.INV_CP))
            t = f32(host["th"][index] * tm)
            rho = f32(pres / (P.RD * t))
            inv_rho = f32(f32(1.0) / rho)
            ze_rain = ze_ice = f32(1.0e-22)
            qr = host["qr"][index]
            if qr >= P.QSMALL:
                nr, mu_r, lamr, _c, _l = P.get_rain_dsd2(
                    qr, host["nr"][index], f32(1.0))
                lam2 = f32(lamr * lamr)
                lam6 = f32(f32(lam2 * lam2) * lam2)
                ze_rain = f32(rho * nr * (mu_r + f32(6.0))
                              * (mu_r + f32(5.0)) * (mu_r + f32(4.0))
                              * (mu_r + f32(3.0)) * (mu_r + f32(2.0))
                              * (mu_r + f32(1.0)) / lam6)
                ze_rain = max(ze_rain, f32(1.0e-22))
            ni = P.impose_max_total_Ni(host["ni"][index], inv_rho)
            qi = host["qi"][index]
            if qi >= P.QSMALL:
                ni = max(ni, P.NSMALL)
                qir, _bir, rhop = P.calc_bulkRhoRime(
                    qi, host["qir"][index], host["qib"][index])
                dumi, dumjj, dumii, d1, d4, d5 = \
                    P.find_lookupTable_indices_1a(qi, ni, qir, rhop)
                f1pr09 = P.access_lookup_table(itab, dumjj, dumii, dumi, 7,
                                               d1, d4, d5)
                f1pr10 = P.access_lookup_table(itab, dumjj, dumii, dumi, 8,
                                               d1, d4, d5)
                f1pr13 = P.access_lookup_table(itab, dumjj, dumii, dumi, 9,
                                               d1, d4, d5)
                ni = min(ni, f32(f1pr09 * qi))
                ni = max(ni, f32(f1pr10 * qi))
                ze_ice = f32(ze_ice + f32(0.1892) * f1pr13 * ni * rho)
                ze_ice = max(ze_ice, f32(1.0e-22))
            out[index] = f32(f32(10.0) * f32(np.log10(
                np.float64(f32((ze_rain + ze_ice) * f32(1.0e18))))))
    return out


def reflectivity(state, *, temperature=None, pressure=None):
    """H_Z(x) for a P3 state: ``(nz, ny, nx)`` dBZ, the state untouched.

    The state's own fields: ``qr``, ``nr``, ``qi``, ``qir``, ``ni``,
    ``qib``, full theta ``thb + thp`` and full pressure ``p``, exactly the
    arrays the forecast step reads (gpuwm/core/p3.py ``apply``).  The step
    diagnoses no t1d/p1d pair of its own, so ``temperature``/``pressure``
    are refused rather than half honoured.  On a CuPy state the result is
    the state's ``refl_10cm`` scratch slot (copy it to keep it, as for the
    other device routes); on a NumPy state a fresh float32 array.
    """
    if temperature is not None or pressure is not None:
        raise ValueError(
            "the P3 reflectivity operator forms its temperature from the "
            "state's full theta and pressure, as the P3 step does; it takes "
            "no temperature/pressure pair")
    missing = [name for name in ("qr", "nr", "qi", "qir", "ni", "qib")
               if getattr(state, name, None) is None]
    if missing:
        raise ValueError(
            "the P3 reflectivity operator needs the scheme's prognostic "
            f"fields; this state carries no {missing}")
    thb = state.thb if state.thb.ndim == 3 else state.thb[:, None, None]
    shape = tuple(state.p.shape)
    species = {name: getattr(state, name) for name in REFLECTIVITY_FIELDS
               if name not in ("th", "pres")}
    if not hasattr(state.p, "get"):
        species["th"] = np.broadcast_to(
            np.asarray(thb, np.float32) + np.asarray(state.thp, np.float32),
            shape)
        species["pres"] = state.p
        return p3_reflectivity_host(species)
    import cupy as cp

    theta = cp.empty(shape, dtype=cp.float32)
    theta[...] = thb + state.thp
    species["th"] = theta
    species["pres"] = state.p
    out = state.scratch(shape, "refl_10cm")
    run_p3_reflectivity(species, out)
    del theta
    return out
