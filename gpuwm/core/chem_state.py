"""Chem species storage on a DomainState (DESIGN 2.1).

Each active TRANSPORTED row gets two DomainState attributes, both float32
``(nz, ny, nx)`` like every gpuwm scalar:

* ``chem_<name>``: the current field.  Restart class ``serialize``: WRF
  carries every chem species in its restart stream.
* ``chem0_<name>``: its RK time-t copy.  Restart class ``rebuild``, exactly
  like ``qv0``: ``dycore._save_time_t`` rewrites it at the top of every step
  before anything reads it.

A PRESCRIBED row (``transported: false``, e.g. GOCART's daily-mean
background oxidants) gets ``chem_<name>`` only, class ``serialize``: it is
not advected, so it has no time copy.

All of them are views into ONE arena per state: ``(2, ntransported, nz, ny,
nx)`` for the transported rows (plane 0 current, plane 1 time copy) and
``(nprescribed, nz, ny, nx)`` for the prescribed ones, so a species-axis
kernel can read the whole set contiguously.  :meth:`ChemState.stacked`
returns the arena only while every attribute still aliases it: a relocation
or a restore that REBINDS an attribute (rather than writing into it) drops
the caller back to per-species launches instead of reading stale memory.

``state.chem`` is the :class:`ChemState`.  It is set ONLY when
``cfg.chem_sets`` is non-empty: a default state keeps its object graph
exactly as before (no attribute at all), so every consumer asks
``getattr(state, "chem", None)``.

The mass ledger's running totals and every array a chem process declares
(:class:`gpuwm.core.chem_context.ChemAllocation`) are ALSO DomainState
attributes, allocated here with the state rather than on the first step, so
a checkpoint restored into a fresh state finds every member it wrote:
``chemdiag_<name>`` (restart class ``serialize``) and ``chemwork_<name>``
(``rebuild``), classified by prefix in gpuwm/io/restart.py.
"""

from __future__ import annotations

import numpy as np

from gpuwm.chem_table import ChemTable, load

__all__ = ["ChemState", "chem_state_of", "attach_chem_state",
           "chem_attr_names", "ledger_attr", "process_attr", "LEDGER_TOTALS"]


class ChemState:
    """The chem arena of one DomainState and the table it was built from."""

    def __init__(self, table: ChemTable, nz: int, ny: int, nx: int, xp):
        self.table = table
        self.transported = table.transported
        self.prescribed = table.prescribed
        self.shape = (nz, ny, nx)
        n_t = len(self.transported)
        n_p = len(self.prescribed)
        self.arena = xp.zeros((2, n_t, nz, ny, nx), dtype=np.float32)
        self.prescribed_arena = (xp.zeros((n_p, nz, ny, nx), dtype=np.float32)
                                 if n_p else None)

    def bind(self, state) -> None:
        """Set the ``chem_<name>``/``chem0_<name>`` attributes as arena views."""
        for i, row in enumerate(self.transported):
            setattr(state, row.state_attr, self.arena[0, i])
            setattr(state, row.time_attr, self.arena[1, i])
        for i, row in enumerate(self.prescribed):
            setattr(state, row.state_attr, self.prescribed_arena[i])

    def fields(self, state) -> tuple:
        """Current fields of the transported rows, in table order."""
        return tuple(getattr(state, row.state_attr) for row in self.transported)

    def time_pairs(self, state) -> tuple:
        """``(current, time copy)`` pairs for ``dycore._save_time_t``."""
        return tuple((getattr(state, row.state_attr),
                      getattr(state, row.time_attr))
                     for row in self.transported)

    def stacked(self, state):
        """The (2, n, nz, ny, nx) arena, or None if any attribute was rebound."""
        for i, row in enumerate(self.transported):
            if (getattr(state, row.state_attr) is not None
                    and not _aliases(getattr(state, row.state_attr),
                                     self.arena[0, i])):
                return None
            if not _aliases(getattr(state, row.time_attr), self.arena[1, i]):
                return None
        return self.arena

    @property
    def nbytes(self) -> int:
        total = int(self.arena.nbytes)
        if self.prescribed_arena is not None:
            total += int(self.prescribed_arena.nbytes)
        return total


def _aliases(view, target) -> bool:
    try:
        return (view.shape == target.shape
                and view.data.ptr == target.data.ptr)        # cupy
    except AttributeError:
        return (view.shape == target.shape
                and view.__array_interface__["data"][0]
                == target.__array_interface__["data"][0])  # numpy


def chem_state_of(state) -> ChemState | None:
    """``state.chem`` or None (the attribute is absent on a chem-off state)."""
    return getattr(state, "chem", None)


#: The mass ledger's buckets (gpuwm/core/chem_context.LEDGER_BUCKETS) plus
#: its two running totals, each a (ntransported,) float64 kg vector.
LEDGER_TOTALS = ("initial", "current")


def ledger_attr(name: str) -> str:
    """DomainState attribute of one ledger vector."""
    from gpuwm.state_serialization_contract import CHEM_DIAG_PREFIX
    return f"{CHEM_DIAG_PREFIX}ledger_{name}"


def process_attr(alloc) -> str:
    """DomainState attribute of one process-declared array."""
    from gpuwm.state_serialization_contract import (CHEM_DIAG_PREFIX,
                                                    CHEM_WORK_PREFIX)
    prefix = (CHEM_DIAG_PREFIX if alloc.restart == "serialize"
              else CHEM_WORK_PREFIX)
    return prefix + alloc.name


def attach_chem_state(state, cfg, xp) -> None:
    """Allocate and bind the chem arena when ``cfg.chem_sets`` is non-empty.

    Called at the end of ``DomainState.__init__``.  Chem off: nothing at all
    happens -- no table read, no attribute, no allocation.
    """
    table = load(cfg)
    if table is None:
        return
    chem = ChemState(table, cfg.nz, cfg.ny, cfg.nx, xp)
    state.chem = chem
    chem.bind(state)
    from gpuwm.core.chem_context import LEDGER_BUCKETS

    n = len(chem.transported)
    for name in (*LEDGER_TOTALS, *LEDGER_BUCKETS):
        setattr(state, ledger_attr(name), xp.zeros(n, dtype=np.float64))
    # 0 until the ledger has taken its initial masses (the first chem step
    # after construction; a restored checkpoint brings its own 1).
    setattr(state, ledger_attr("started"), xp.zeros(1, dtype=np.int32))
    if table.processes:
        from gpuwm.core.chem_context import allocation_shape, chem_processes

        seen: set[str] = set()
        for key, module in chem_processes(table):
            rows = module.rows(table)
            for alloc in getattr(module, "ALLOCATES", ()):
                if alloc.name in seen:
                    raise ValueError(f"chem process {key}: array "
                                     f"{alloc.name!r} is declared twice")
                seen.add(alloc.name)
                setattr(state, process_attr(alloc), xp.zeros(
                    allocation_shape(alloc, len(rows), cfg.nz, cfg.ny,
                                     cfg.nx), dtype=alloc.dtype))


def chem_attr_names(state) -> tuple[str, ...]:
    """Every ``chem_<name>`` attribute the state carries (serialized set)."""
    chem = chem_state_of(state)
    if chem is None:
        return ()
    return tuple(row.state_attr for row in (*chem.transported, *chem.prescribed))
