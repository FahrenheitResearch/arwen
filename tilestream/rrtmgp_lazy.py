"""RRTMGP's solver workspace, allocated at the firing and released after it.

WHY THIS CAN EXIST AT ALL
-------------------------
``SharedRRTMGPChunkWorkspace`` is one flat device allocation that every
solver phase lays its live set over, and ``RRTMGP_WORKSPACE_LIFETIME_AUDIT``
(gpuwm/core/rrtmgp.py) names, slot by slot, the kernel or fill that writes
every element before its first read.  If that audit is true then the CONTENTS
of the backing at the top of a radiation call are irrelevant -- and if the
contents are irrelevant, so is the identity of the allocation.  A workspace
that is freed after one firing and freshly allocated for the next must give
bit-identical answers.

That is not an argument for doing it; it is only the argument that doing it
is *legal*.  The argument for doing it is the cadence.  At the production
``radt_minutes = 12, dt = 3 s`` radiation fires once every 240 steps, so the
shipped configuration holds the workspace resident for 239 steps that cannot
read a byte of it.

WHAT IT IS ACTUALLY WORTH -- READ THIS BEFORE BELIEVING IT IS FREE MONEY
-----------------------------------------------------------------------
Releasing bytes between firings lowers what the process HOLDS between
firings.  It lowers the process's PEAK only if the peak is somewhere other
than the radiation step, and at the shipped ``column_chunk = 3125`` the peak
is *on* the radiation step -- measured, see ``tilestream.rrtmgp_bench``.  So
on its own this buys nothing that a capacity bisection can see, and saying
otherwise would be the eighth false result in this project.

It becomes worth its full size in combination with a smaller column chunk.
Chunking shrinks the radiation step until some ordinary step becomes the
peak; from that point on every resident workspace byte is a byte added to a
peak that radiation no longer sets, and releasing it is a straight saving.
The two reclamations are therefore multiplicative, not additive, and neither
is worth much alone.

MECHANISM
---------
The backing is taken from a PRIVATE ``cupy.cuda.MemoryPool`` rather than the
default one.  That is the whole reason the saving is real rather than
bookkeeping: ``del`` on a default-pool array returns the block to the pool,
which keeps holding it against the device, so the device never sees it come
back.  A private pool can be emptied exactly -- ``free_all_blocks()`` on it
touches nothing else in the process -- and ``pool.total_bytes()`` is then a
precise, auditable statement of what radiation is holding from the card at
any instant.

The layout itself is NOT re-derived here.  ``_SizeOnlyModule`` is handed to
the parent as its array module, so the parent runs its own validation and
its own phase-maximum arithmetic and merely allocates nothing at the end of
it.  There is exactly one transcription of the RRTMGP workspace layout in
this repository and it stays in ``gpuwm.core.preflight``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from gpuwm.core.model import SharedRRTMGPChunkWorkspace


__all__ = [
    "LazyRRTMGPChunkWorkspace",
    "RELEASE_HAZARDS",
    "attach_lazy",
]


#: Hazard names the negative control may inject.  Production must pass
#: ``None``; ``build``/``attach_lazy`` do not accept these by accident -- a
#: caller has to name one.
RELEASE_HAZARDS = ("release_between_phases",)


class _SizeOnlyModule:
    """Array-module shim that records the parent's request and allocates none.

    ``SharedRRTMGPChunkWorkspace.__post_init__`` validates its dimensions,
    builds the phase layouts and finishes with ``xp.empty((total,), uint8)``.
    Handing it this object runs every one of those steps unchanged and
    captures ``total`` instead of reserving it, so the lazy subclass inherits
    the parent's layout arithmetic rather than repeating it.
    """

    uint8 = np.uint8

    def __init__(self) -> None:
        self.requested_bytes = 0

    def empty(self, shape, dtype=None):
        self.requested_bytes = int(math.prod(shape))
        # A zero-length real array: sliceable and .view()-able exactly like
        # the real backing, so any accidental use fails loudly on shape
        # rather than silently on None.
        return np.empty((0,), dtype=np.uint8)


@dataclass
class LazyRRTMGPChunkWorkspace(SharedRRTMGPChunkWorkspace):
    """A chunk workspace that exists only while radiation is firing.

    ``phase`` allocates on first use within a call; :meth:`release` returns
    every byte to the device.  ``nbytes`` keeps reporting the FULL size the
    workspace will take when live -- the restart identity record and the
    preflight ledger both compare against it, and a workspace that reported 0
    between firings would silently pass a drift check it should fail.
    :attr:`resident_bytes` is the one that tells you what is held right now.
    """

    #: Release automatically at the end of every ``RRTMGPRadiation.__call__``.
    #: Read by the adapter through ``getattr``, so an ordinary
    #: ``SharedRRTMGPChunkWorkspace`` (which has no such attribute) keeps its
    #: shipped persistent behaviour with no branch of its own.
    release_after_call: bool = True

    #: Negative-control lever.  ``"release_between_phases"`` releases the
    #: backing in the middle of a call, between the optics phase that
    #: produces the carried slots and the RTE phase that reads them at
    #: identical offsets, and lets another allocation take the freed bytes.
    #: That is the "reuse a freed arena while it is still live" hazard, and
    #: the gate must catch it.
    hazard: str | None = None

    _pool: object = field(default=None, init=False, repr=False, compare=False)
    _live: object = field(default=None, init=False, repr=False, compare=False)
    _nbytes: int = field(default=0, init=False, repr=False, compare=False)
    _allocations: int = field(default=0, init=False, repr=False, compare=False)
    _releases: int = field(default=0, init=False, repr=False, compare=False)
    _hazard_fired: int = field(default=0, init=False, repr=False,
                               compare=False)

    def __post_init__(self) -> None:
        if self.hazard is not None and self.hazard not in RELEASE_HAZARDS:
            raise ValueError(
                f"unknown RRTMGP release hazard {self.hazard!r}; "
                f"known: {RELEASE_HAZARDS}")
        sizer = _SizeOnlyModule()
        self._array_module = sizer
        super().__post_init__()
        self._nbytes = int(sizer.requested_bytes)
        if self._nbytes < 1:
            raise RuntimeError(
                "lazy RRTMGP workspace sized itself at zero bytes; the "
                "parent's allocation call did not route through the shim")
        # Drop the shim: nothing may allocate through it after sizing, and
        # leaving it attached would make an accidental re-init silently
        # produce a host array.
        self._array_module = None
        import cupy as cp

        self._pool = cp.cuda.MemoryPool()

    # -- size and residency ------------------------------------------------

    @property
    def nbytes(self) -> int:
        """The full backing size, live or not (the identity/ledger number)."""
        return int(self._nbytes)

    @property
    def resident_bytes(self) -> int:
        """Device bytes this workspace is holding RIGHT NOW.

        Taken from the private pool rather than from ``_live is None``, so it
        counts the block the pool is still holding if a release ever fails to
        empty it.  This is the number every VRAM claim in this lane is made
        against.
        """
        return 0 if self._pool is None else int(self._pool.total_bytes())

    @property
    def allocations(self) -> int:
        """How many times the backing has been materialised."""
        return int(self._allocations)

    @property
    def releases(self) -> int:
        return int(self._releases)

    @property
    def hazard_firings(self) -> int:
        """How many times the injected hazard actually executed.

        A negative control that never ran is not a passing control, so this
        is printed rather than inferred.
        """
        return int(self._hazard_fired)

    # -- lifecycle ---------------------------------------------------------

    def acquire(self):
        """Materialise the backing if it is not already live."""
        if self._live is not None:
            return self._live
        import cupy as cp

        memptr = self._pool.malloc(self._nbytes)
        self._live = cp.ndarray((self._nbytes,), dtype=cp.uint8, memptr=memptr)
        self._storage = self._live
        self._allocations += 1
        return self._live

    def release(self) -> int:
        """Return every byte to the device.  Returns the bytes released.

        Idempotent: releasing an already-released workspace is a no-op, so a
        caller that releases defensively cannot double-count.
        """
        if self._live is None and self.resident_bytes == 0:
            return 0
        held = self.resident_bytes
        self._live = None
        self._storage = None
        self._pool.free_all_blocks()
        self._releases += 1
        remaining = self.resident_bytes
        if remaining:
            raise RuntimeError(
                "lazy RRTMGP workspace release left "
                f"{remaining} bytes in its private pool; the backing is "
                "still referenced somewhere (a phase view outlived the call)")
        return held - remaining

    def phase(self, name: str, ncol: int):
        """Audited live-set views, materialising the backing on demand."""
        self.acquire()
        views = super().phase(name, ncol)
        if (self.hazard == "release_between_phases"
                and name in ("lw_optics", "sw_optics")):
            # THE HAZARD, stated plainly: the optics phase has just been
            # handed views it is about to fill, and the RTE phase that
            # follows reads several of them back at identical offsets.
            # Freeing the backing here and letting another allocation take
            # the same bytes is exactly "reuse a freed arena while it is
            # still live".  The views handed out above stay valid Python
            # objects pointing at memory this process has given away.
            self._live = None
            self._pool.free_all_blocks()
            import cupy as cp

            squatter = cp.empty((self._nbytes,), dtype=cp.uint8)
            squatter.fill(0xA5)
            self._squatter = squatter
            self._hazard_fired += 1
        return views

    # -- adapter hook ------------------------------------------------------

    def on_call_end(self) -> int:
        """Called by ``RRTMGPRadiation.__call__`` after the last chunk."""
        if not self.release_after_call:
            return 0
        self._squatter = None
        return self.release()


def attach_lazy(state, workspace) -> None:
    """Point one state's radiation adapter at ``workspace``.

    Mirrors ``tilestream.shared_workspace.attach`` for the radiation half
    only, so a probe can hold the scratch arena and the dycore workspace
    constant while it varies the radiation one.
    """
    driver = getattr(state, "physics", None)
    if driver is None or workspace is None:
        return
    radiation = getattr(driver, "radiation_callable", None)
    if radiation is None or not hasattr(radiation, "column_chunk"):
        raise TypeError(
            "state's radiation adapter does not take a chunk workspace; "
            f"got {type(radiation).__name__}")
    radiation.column_chunk = int(workspace.column_chunk)
    radiation.chunk_workspace = workspace
