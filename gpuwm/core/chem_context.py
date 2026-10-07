"""What a chem process reads and writes (DESIGN section 4, "ChemContext").

A chem PROCESS is a module named by :data:`gpuwm.chem_table.CHEM_PROCESS_MODULES`
that exports:

``KEY``
    its process key, e.g. ``"mixing.vertmx"``.
``LEDGER``
    the mass-ledger bucket its mass change is filed under: one of
    :data:`LEDGER_BUCKETS`.  The driver measures every row the process acts on
    before and after :func:`step` and files the difference, so a process never
    books mass itself and cannot book it twice.
``REQUIRES``
    context field names it reads (see :meth:`ChemContext.met`).  A name the
    active scheme set does not produce is refused at the configuration door
    with the process and the field named.
``ALLOCATES``
    a tuple of :class:`ChemAllocation`: the arrays it keeps (accumulations,
    persistence buffers, caches, diagnostics), each with its restart class and
    optional output declaration.  The driver allocates them once and hands
    them over in ``ctx.diag``.
``KERNEL_MODULES``
    the ``gpuwm/core/kernels`` translation units its ``step`` launches (an
    empty tuple for a host-only process).  The forecast preflight prices
    their per-thread local frames from these names and refuses a process
    that does not declare them, rather than pricing its kernels at zero.
``rows(table)``
    the active rows it acts on, normally ``table.rows_for(KEY)``.
``refusal(cfg)`` (optional)
    a reason string when the configuration cannot run this process (a field
    it REQUIRES that no scheme in ``cfg`` publishes, say), else None; the
    configuration door raises it by name.
``init(ctx)`` and ``step(ctx, dt, ktau)``
    once at start (and after a restart restore), then once per chem step.

Everything a process touches is float32 on the device unless its
:class:`ChemAllocation` says otherwise (WRF REAL*8 routines keep float64).
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Mapping

__all__ = [
    "CHEM_ALLOCATION_SHAPES", "CHEM_PREP_FIELDS", "CHEM_PREP_W_FIELDS",
    "CHEM_STEP_ORDER", "LEDGER_BUCKETS", "MONO_IMPLICIT_SLOT",
    "chem_kernel_modules", "MONO_SCRATCH_SLOTS",
    "ChemAllocation", "ChemClock", "ChemContext", "allocation_shape",
    "chem_physics_reads", "chem_processes",
]

#: chem_prep's met fields (gpuwm/core/chem_prep.py, WRF
#: module_chem_utilities.F:8-181), in its kernel's output order, and the
#: ones on w levels, (nz + 1, ny, nx); the rest are (nz, ny, nx).  Spelled
#: here so the device inventory prices them without importing chem_prep.
CHEM_PREP_FIELDS = ('p_phy', 't_phy', 'rho', 'dryrho', 'alt', 'u_phy',
                    'v_phy', 'z_at_w', 'dz8w', 'z', 'rh', 'p8w', 't8w')
CHEM_PREP_W_FIELDS = ('z_at_w', 'p8w', 't8w')

#: The monotonic final stage's (``chem_adv_opt = 2``) mass-shaped workspace
#: in ``DomainState.scratch``: the donor extrema, the two limiter scales and
#: the two tendency diagnostics, every one overwritten by each launch before
#: it is read.  One set serves every row in turn.
MONO_SCRATCH_SLOTS = ("chem_mono_qmin", "chem_mono_qmax", "chem_mono_si",
                      "chem_mono_so", "chem_mono_ht", "chem_mono_zt")
#: WRF's ``wwI`` for that stage, (nz + 1, ny, nx): the explicit-only route
#: uses this zero carrier. With zadvect_implicit on,
#: the scalar split supplies wwI instead. A carrying slot, so no other slot
#: ever shares its backing.
MONO_IMPLICIT_SLOT = "chem_mono_wwi"

#: WRF-Chem's per-step order, as process keys (``chem/chem_driver.F``):
#: emissions with plume rise (:818), optics on radiation steps (:926),
#: deposition velocities then vertical mixing with deposition as its lower
#: boundary (:1049), settling, the GOCART sulfur mechanism (:1242), aging
#: (:1577), large-scale wet removal (:1620/:1689), then the Thompson
#: coupling (an ArWen addition, DESIGN 6.7).
CHEM_STEP_ORDER = (
    "emission.sfire",
    "plumerise.freitas", "emission.fire", "emission.dust",
    "emission.seasalt", "emission.inventory",
    "optics.gocart",
    "drydep.wesely", "drydep.gocart",
    "mixing.vertmx",
    "settling.gocart",
    "chem.sulfur",
    "aging.gocart",
    "wetdep.ls",
    "coupling.thompson",
)


def chem_kernel_modules(table, chem_adv_opt: int) -> frozenset[str]:
    """Every kernel translation unit a chem domain on ``table`` launches.

    The ledger, the history outputs and the outer-domain flow boundary on
    every chem domain; the monotonic final stage under ``chem_adv_opt = 2``
    (the positive-definite one is the core ``pd_advection`` unit); and when
    any process runs, chem_prep's column inputs plus each process's own
    ``KERNEL_MODULES``.  CuPy-free, for the forecast preflight.
    """
    modules = {"chem_bdy", "chem_ledger", "chem_outputs"}
    if int(chem_adv_opt) == 2:
        modules.add("mono_advection")
    processes = chem_processes(table) if table.processes else []
    if processes:
        modules.add("chem_prep")
    for key, module in processes:
        declared = getattr(module, "KERNEL_MODULES", None)
        if declared is None:
            raise ValueError(
                f"chem process {key} ({module.__name__}) declares no "
                "KERNEL_MODULES, so the preflight cannot price the local "
                "frames of the kernels its step launches; add the tuple "
                "(empty for a host-only process)")
        modules.update(declared)
    return frozenset(modules)


def chem_processes(table):
    """``[(key, module)]`` for every process an active row names, in order.

    CuPy-free: a process module imports its device libraries inside
    ``init``/``step``, so the configuration door, the VRAM projection and
    the host-side state allocation can read ``ALLOCATES`` on any machine.
    """
    from gpuwm.chem_table import CHEM_PROCESS_MODULES

    if set(CHEM_STEP_ORDER) != set(CHEM_PROCESS_MODULES):
        raise RuntimeError("every chem process key needs one place in "
                           "CHEM_STEP_ORDER")
    named = set(table.processes)
    return [(key, importlib.import_module(CHEM_PROCESS_MODULES[key]))
            for key in CHEM_STEP_ORDER if key in named]


def chem_physics_reads(cfg) -> frozenset:
    """Every name an active chem process ``REQUIRES`` (empty when chem is off).

    The physics driver keeps a carrier it would otherwise not hold when a
    chem process reads it (GSW outside RUC, for Wesely deposition), and
    preflight prices the same plane from the same answer.
    """
    if not getattr(cfg, "chem_sets", ""):
        return frozenset()
    from gpuwm.chem_table import load

    table = load(cfg)
    if table is None:
        return frozenset()
    names: set = set()
    for _key, module in chem_processes(table):
        names.update(getattr(module, "REQUIRES", ()))
    return frozenset(names)

#: ``ChemAllocation.shape`` values -> how the driver sizes them.
#: ``2d`` (ny, nx); ``3d`` (nz, ny, nx) mass levels; ``3d_w`` (nz + 1, ny, nx)
#: w levels; ``rows_2d`` (nrows, ny, nx) and ``rows_3d`` (nrows, nz, ny, nx)
#: carry one plane/volume per row the process acts on, in ``rows(table)``
#: order.
CHEM_ALLOCATION_SHAPES = ("2d", "3d", "3d_w", "rows_2d", "rows_3d")

#: Where a process's mass change is filed in the per-species ledger.
#: ``transport`` is the dycore's (advection, lateral boundaries, the PD clamp),
#: filed by the driver, never by a process.
LEDGER_BUCKETS = ("emitted", "deposited", "settled", "scavenged",
                  "chemistry", "mixing", "coupling", "transport")


@dataclass(frozen=True)
class ChemAllocation:
    """One array a chem process keeps."""

    name: str
    shape: str
    dtype: str = "float32"
    #: ``serialize`` (cross-step state a restart must carry: accumulations,
    #: persistence buffers, caches read before they are rewritten) or
    #: ``rebuild`` (overwritten before every read).
    restart: str = "serialize"
    #: When set, the array is written to history under this name.
    output_name: str | None = None
    units: str = ""
    description: str = ""

    def __post_init__(self):
        if self.shape not in CHEM_ALLOCATION_SHAPES:
            raise ValueError(f"ChemAllocation {self.name!r}: shape "
                             f"{self.shape!r} not in {CHEM_ALLOCATION_SHAPES}")
        if self.dtype not in ("float32", "float64", "int32"):
            raise ValueError(f"ChemAllocation {self.name!r}: dtype "
                             f"{self.dtype!r} is not float32/float64/int32")
        if self.restart not in ("serialize", "rebuild"):
            raise ValueError(f"ChemAllocation {self.name!r}: restart class "
                             f"{self.restart!r} is not serialize/rebuild")


def allocation_shape(alloc: ChemAllocation, nrows: int, nz: int, ny: int,
                     nx: int) -> tuple[int, ...]:
    """The array shape the driver allocates for ``alloc``."""
    return {"2d": (ny, nx), "3d": (nz, ny, nx), "3d_w": (nz + 1, ny, nx),
            "rows_2d": (nrows, ny, nx),
            "rows_3d": (nrows, nz, ny, nx)}[alloc.shape]


@dataclass(frozen=True)
class ChemClock:
    """WRF chem_driver's clock for one chem step.

    ``gmt`` is the simulation start hour (WRF ``grid%gmt``), ``julday`` the
    start's day of year (``grid%julday``), ``curr_secs`` seconds since the
    simulation start at the START of the step being closed (WRF
    ``curr_secs``: chem_driver runs before the domain clock advances), and
    ``ktau`` the 1-based model step (WRF ``grid%itimestep``).
    """

    gmt: float
    julday: int
    curr_secs: float
    ktau: int
    dt: float


class ChemContext:
    """The per-domain view a chem process is handed (see module docstring).

    ``met(name)`` resolves a context field: first the chem_prep fields
    (``p_phy``, ``t_phy``, ``rho`` -- WRF's MOIST density
    ``1/alt*(1+qv)`` --, ``dryrho`` = ``1/alt``, ``alt``, ``u_phy``, ``v_phy``,
    ``z_at_w`` (nz+1), ``dz8w``, ``z``, ``rh``, ``p8w``, ``t8w`` (nz+1),
    ``qv``/``qc``/``qr`` when present), then the physics driver's fields by
    their WRF names (``ust``, ``rmol``, ``znt``, ``hfx``, ``pblh``,
    ``exch_h``, ``swdown``, ``tsk``, ``u10``, ``v10``, ``xland``, ``ivgtyp``,
    ``isltyp``, ``vegfra``, ``smois``, ``snowh``, ...).  An absent name raises
    ``KeyError`` naming it.

    ``ddvel`` is a (nrows_active, ny, nx) float32 array of dry deposition
    velocities (m s-1), one plane per active row in table order
    (:meth:`row_index`), zeroed at the top of every chem step, written by the
    deposition processes and read by vertical mixing as its lower boundary.
    """

    def __init__(self, state, cfg, table, *, prep=None, physics_fields=None,
                 diag=None, ddvel=None, frames=None, clock=None):
        self.state = state
        self.cfg = cfg
        self.table = table
        self.prep = prep
        self.physics_fields: Mapping = physics_fields or {}
        self.diag: dict = {} if diag is None else diag
        self.ddvel = ddvel
        self.frames = frames
        self.clock = clock
        self._index = {row.name: i for i, row in enumerate(table.rows)}

    def field(self, row):
        """The row's current (nz, ny, nx) float32 field, ``state.chem_<name>``."""
        return getattr(self.state, row.state_attr)

    def row_index(self, row) -> int:
        """The row's position in the active table (its ``ddvel`` plane)."""
        return self._index[row.name]

    def met(self, name: str):
        if self.prep is not None:
            value = self.prep.get(name)
            if value is not None:
                return value
        value = self.physics_fields.get(name)
        if value is None:
            raise KeyError(
                f"chem context field {name!r} is not produced by this run "
                "(neither chem_prep nor the physics driver's fields carry it)")
        return value

    def has(self, name: str) -> bool:
        if self.prep is not None and self.prep.get(name) is not None:
            return True
        return self.physics_fields.get(name) is not None
