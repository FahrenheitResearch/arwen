"""One workspace for every tile buffer, instead of one per tile buffer.

MEASURED FIRST, THEN BUILT
--------------------------
:mod:`tilestream.vram_probe` armed a live allocation ledger over the whole
life of a full+MYNN+Noah-MP buffer at 128x128x49 and then built three
identical buffers in one process.  The result is the fact this module exists
to act on:

  ==========================  ============  ==================================
  buffer                      pooled bytes  what it paid for
  ==========================  ============  ==================================
  1                           1,299.7 MiB   everything
  2                           1,276.6 MiB   everything except 22.65 MiB
  3                           1,276.6 MiB   everything except 22.65 MiB
  ==========================  ============  ==================================

**98.2% of a tile buffer's device footprint is paid again for every
buffer.**  The only per-PROCESS item in the whole configuration is the
22.65 MiB of RRTMGP k-distribution and cloud-optics tables, which
``functools.lru_cache`` already shares.  Nothing else was shared with
anything.

And the largest single item does not shrink with the tile.  MYNN's declared
workspace is sized against ``MYNN_PBL_COLUMN_CHUNK = 16384`` columns, not
against ``ny*nx``: 52,352 bytes per column, so 818.0 MiB at ``nz = 49`` for
any tile of 16,384 columns or more -- 128x128 exactly reaches it.  Of the
907.2 MiB of arena-eligible scratch a 128x128 tile declares, 836.4 MiB is
MYNN's.  A second buffer therefore costs another 818 MiB of fixed workspace
before it holds a single cell of weather, and on a 12 GB card that is 6.7% of
the whole device per buffer, for a scheme that is running on exactly one
buffer at a time.

WHAT THIS MODULE DOES
---------------------
gpuwm already solved this problem for a different caller.
:class:`gpuwm.core.state.ScratchArena` exists so that the four domains of a
nested tree can share one backing for every step-local scratch slot, and
:mod:`gpuwm.core.preflight` carries the reviewed lifetime audit that decides
which slots may join (``write_before_read`` only -- carrying slots such as
the ``mp_*`` precipitation accumulators and Kain-Fritsch's ``cu_*`` are
excluded by name).  ``SharedDycoreStateWorkspace`` does the same for the
restart-REBUILT state symbols and ``SharedRRTMGPChunkWorkspace`` for the
radiation solver's per-chunk live set.

Tile buffers are the same shape of problem as sibling domains: they hold
different data, they step one at a time, and their scratch is dead between
steps.  So this module builds those three objects ONCE from the tile config
and hands the same three to every buffer.  It adds no new sharing mechanism
and no new lifetime claim; it applies gpuwm's own, and the gate proves the
result is bit-identical.

THE CONDITION THAT MAKES IT LEGAL, AND WHY IT WAS NOT TRUE
----------------------------------------------------------
"They step one at a time" is a claim about the DEVICE, not about the Python
loop, and in ``driver.run_tiled`` it was FALSE.  Each buffer owns a
non-blocking stream and tile *i*'s ``step`` is enqueued on stream
``i % nbuffers`` with nothing ordering it against tile *i+1*'s ``step`` on
the next stream.  The two computations are independent by construction --
that is the whole point of the multi-buffer pipeline -- so the driver may run
them concurrently, and two concurrent steps sharing one scratch arena would
be writing the same bytes.

``run_tiled(share=...)`` therefore also CHAINS THE COMPUTE: an event is
recorded after each tile's ``step`` and the next tile's stream waits on it.
Copies are untouched, so the prefetch pipeline still hides the transfer
behind the compute -- only compute/compute overlap is given up, and
``tilestream.vram_share_bench`` measures what that costs.

``chain_compute=False`` is the negative control and it is not decorative: it
shares the arena WITHOUT the chain, which is precisely "the reclaimed buffer
is reused while it is still live".  On a dry 192x192 domain split 3x3 it
makes ALL NINE carriers differ, three runs out of three, while the same
configuration with the chain on is bit-exact -- ``tilestream/test_share.py``
asserts both.

WHAT IT IS WORTH, MEASURED
--------------------------
Three buffers of an 80x72x49 tile (the gathered extent of a 48x40 tile at
halo 16), full+MYNN+Noah-MP, marginal pooled bytes per buffer:

  ==================================  =========  =========  =========
  configuration                       buffer 1   buffer 2   buffer 3
  ==================================  =========  =========  =========
  private workspaces (as shipped)     472.1 MiB  449.0 MiB  449.0 MiB
  shared, RRTMGP chunk 3125           100.6 MiB  100.6 MiB  100.6 MiB
  shared, RRTMGP chunk 1024           100.6 MiB  100.6 MiB  100.6 MiB
  ==================================  =========  =========  =========

The marginal cost of a buffer falls 4.5x, from 449.0 to 100.6 MiB.  Against
that, one shared allocation of 1,185.1 MiB (624.0 MiB at RRTMGP chunk 1024)
is paid once.  Counting what the DEVICE actually holds -- pool_total, which
includes the blocks the pool retained after radiation freed them -- three
buffers cost 2,100 MiB private, 1,548 MiB shared, and 992 MiB shared at
chunk 1024: 2.1x less.

The shared allocation is eager and complete while ``DomainState.scratch`` is
lazy and partial, so at ``nbuffers == 1`` this is a wash and not a win.
Sharing is worth exactly what a second buffer would have cost.

WHAT IT BUYS ON A 12 GB CARD
----------------------------
Largest square extent at ``nz = 49``, full+MYNN+Noah-MP, that survives two
steps INCLUDING a forced radiation firing at the production 12-minute
cadence.  Bisected one fresh subprocess per trial on an idle RTX 4090 with
the CuPy pool capped at 9.740 GiB -- a 5070's 11.940 GiB total minus the
2.06-2.18 GiB of measured non-pool footprint (CUDA context + NVRTC module
images), which no pool-only accounting counts and which a 12 GB card pays
first.  Every refusal in the sweep cited the pool cap, so no row is a
disguised device-level OOM:

  ===========================================  ==========  ==========
  configuration                                  extent      Mcell
  ===========================================  ==========  ==========
  ONE RESIDENT DOMAIN (the vanilla ceiling)
    as shipped                                     464^2        10.5
    + shared workspaces                            464^2        10.5
    + RRTMGP column_chunk 1024                     480^2        11.3
    + MYNN column chunk 4096                       496^2        12.1
  ONE STREAMED TILE, nbuffers = 2
    as shipped                                     336^2         5.5
    + everything above                             416^2         8.5
  ===========================================  ==========  ==========

The streamed row is the one that matters: the shipped configuration cannot
double-buffer a tile bigger than 336x336 on a 12 GB card, and with the
reclamations it reaches 416x416 -- 1.53x the cells per tile, from the same
device.  A bigger tile is not a bigger domain (the domain lives in host RAM);
it is less halo redundancy per cell and fewer tiles per sweep.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


__all__ = [
    "SharedTileWorkspaces",
    "arena_shapes",
    "attach",
    "build",
    "mynn_column_chunk",
    "set_mynn_column_chunk",
]


class _DomainShim:
    """The one attribute ``preflight``'s arena helpers read off a domain.

    ``shared_scratch_arena_shapes`` and ``shared_scratch_arena_aliases`` take
    a tuple of ``DomainConfig`` but touch only ``dc.run`` unless every member
    also carries ``grid_id``/``parent_id`` (the nested-tree branch).  A tile
    plan has no nest, so a shim with just ``run`` selects the single-domain
    behaviour exactly, without constructing an experiment.
    """

    __slots__ = ("run",)

    def __init__(self, run) -> None:
        self.run = run


def arena_shapes(tile_cfg, *, force_slots: tuple[str, ...] = ()) -> dict:
    """``{slot: shape}`` for the arena-eligible scratch of one tile config.

    ``force_slots`` admits named slots that the lifetime audit EXCLUDES.  It
    exists for one caller: the negative control that shares a carrying slot
    (an ``mp_*`` accumulator) and requires the gate to fail.  Production must
    never pass it, and :func:`build` does not expose it by accident -- a
    caller has to name the slots.
    """
    from gpuwm.core.preflight import (scratch_slot_registry,
                                      scratch_slot_uses_arena,
                                      shared_scratch_arena_shapes)

    shapes = shared_scratch_arena_shapes((_DomainShim(tile_cfg),))
    if force_slots:
        registry = scratch_slot_registry(tile_cfg, n_lbc_intervals=0)
        for slot in force_slots:
            if slot not in registry:
                raise KeyError(
                    f"{slot!r} is not a scratch slot of this configuration; "
                    f"the registry declares {sorted(registry)[:8]}...")
            if scratch_slot_uses_arena(slot):
                raise ValueError(
                    f"{slot!r} is already arena-eligible, so forcing it "
                    "proves nothing")
            shapes[slot] = tuple(registry[slot])
    return shapes


@dataclass
class SharedTileWorkspaces:
    """The three shared allocations, plus what they are worth.

    ``poison`` is the runtime lever the arena's own audit was written
    against: fill every backing with NaN between tiles and require the run
    to stay bit-exact.  A slot that is really carried across the boundary
    turns into NaN and the gate says so.
    """

    arena: object | None = None
    dycore: object | None = None
    rrtmgp: object | None = None
    tile_cfg: object | None = None
    forced_slots: tuple[str, ...] = ()
    _sizes: dict = field(default_factory=dict)

    @property
    def nbytes(self) -> int:
        return sum(self._sizes.values())

    @property
    def sizes(self) -> dict:
        return dict(self._sizes)

    def poison(self) -> None:
        """NaN every shared backing.  Debug/control lever, never production."""
        if self.arena is not None:
            self.arena.poison()
        if self.dycore is not None and hasattr(self.dycore, "poison"):
            self.dycore.poison()

    def describe(self) -> str:
        parts = [f"{name} {value / 2**20:.1f} MiB"
                 for name, value in sorted(self._sizes.items(),
                                           key=lambda kv: -kv[1])]
        return f"{self.nbytes / 2**20:.1f} MiB shared: " + ", ".join(parts)


def build(tile_cfg, *, scratch: bool = True, dycore: bool = True,
          rrtmgp: bool = True, rrtmgp_column_chunk: int | None = None,
          p_top: float | None = None,
          force_slots: tuple[str, ...] = ()) -> SharedTileWorkspaces:
    """Allocate the shared workspaces for tiles of shape ``tile_cfg``.

    Each part can be switched off independently so the probe can price them
    one at a time; ``build`` allocates nothing for a part that is off, so an
    A/B measurement compares two real processes rather than one process with
    a flag.

    ``rrtmgp_column_chunk`` overrides ``gpuwm.config.DEFAULT_COLUMN_CHUNK``.
    The workspace is exactly linear in it (the phase maximum is a sum of
    ``(chunk, nlay, ngpt)`` cubes), and radiation fires once every
    ``radt/dt`` steps -- 240 at the production ``radt = 12 min, dt = 3 s`` --
    so a chunk that costs radiation time buys resident bytes at a rate the
    step average barely notices.  ``RRTMGPRadiation``'s own controller
    benchmark on 250x200x49 is the throughput side of that trade:
    ``256 = 21.9 s, 1024 = 5.54 s, 4096 = 1.71 s, 12500 = 1.12 s`` per call.
    """
    from gpuwm.core.model import SharedRRTMGPChunkWorkspace
    from gpuwm.core.preflight import (shared_dycore_state_workspace_shapes,
                                      shared_scratch_arena_aliases)
    from gpuwm.core.state import (ScratchArena, SharedDycoreStateWorkspace)

    shim = (_DomainShim(tile_cfg),)
    out = SharedTileWorkspaces(tile_cfg=tile_cfg,
                               forced_slots=tuple(force_slots))
    if scratch:
        shapes = arena_shapes(tile_cfg, force_slots=force_slots)
        aliases = shared_scratch_arena_aliases(shim)
        # A forced slot must not also be aliased onto another slot's
        # backing: the control has to demonstrate ONE defect (a carrying
        # slot shared between buffers), not two.
        aliases = {slot: target for slot, target in aliases.items()
                   if slot not in out.forced_slots}
        out.arena = ScratchArena(shapes, slot_aliases=aliases)
        out._sizes["scratch_arena"] = int(out.arena.nbytes)
    if dycore:
        out.dycore = SharedDycoreStateWorkspace(
            shared_dycore_state_workspace_shapes(shim))
        out._sizes["dycore_state"] = int(
            getattr(out.dycore, "nbytes", 0)
            or sum(4 * math.prod(shape) for shape
                   in shared_dycore_state_workspace_shapes(shim).values()))
    if rrtmgp and _radiation_active(tile_cfg):
        # Capping at the tile's own column count is not a tuning choice: a
        # workspace wider than the tile is bytes no chunk can ever fill.
        ncol = int(tile_cfg.ny) * int(tile_cfg.nx)
        chunk = (int(rrtmgp_column_chunk) if rrtmgp_column_chunk
                 else _default_column_chunk())
        chunk = max(1, min(chunk, ncol))
        out.rrtmgp = SharedRRTMGPChunkWorkspace(
            nz=int(tile_cfg.nz), column_chunk=chunk,
            p_top=(_p_top_of(tile_cfg) if p_top is None else float(p_top)))
        out._sizes["rrtmgp_chunk"] = int(out.rrtmgp.nbytes)
    return out


def attach(state, shared: SharedTileWorkspaces | None) -> None:
    """Point ``state`` at the shared workspaces.

    Call BEFORE the state's first ``scratch`` request.  ``DomainState.scratch``
    caches the first buffer it hands out per slot, so a slot already created
    privately keeps its private allocation and the sharing silently buys
    nothing for it -- which is why the tile factories inject the arena at
    construction rather than afterwards.
    """
    if shared is None:
        return
    if shared.arena is not None:
        state._scratch_arena = shared.arena
    driver = getattr(state, "physics", None)
    if driver is None or shared.rrtmgp is None:
        return
    radiation = getattr(driver, "radiation_callable", None)
    if radiation is None or not hasattr(radiation, "column_chunk"):
        return
    radiation.column_chunk = int(shared.rrtmgp.column_chunk)
    radiation.chunk_workspace = shared.rrtmgp


# --------------------------------------------------------------------------
# the MYNN column chunk, which is a capacity knob wearing a throughput name
# --------------------------------------------------------------------------

#: Every module that bound ``MYNN_PBL_COLUMN_CHUNK`` at import time.  All
#: three have to move together: ``mynn_pbl_runtime`` decides how wide a call
#: is, ``mynn_pbl_gpu`` sizes its launches, and ``preflight`` -- through
#: ``mynn_pbl_scratch_slots`` -> ``mynn_pbl_column_chunk`` -- decides how wide
#: the SHARED ARENA's MYNN slots are.  Patching only the runtime leaves the
#: arena allocated at the old width and the saving does not appear: MEASURED
#: as an identical 480^2 ceiling at chunk 4096 and chunk 16384 before this
#: tuple existed.
_MYNN_CHUNK_MODULES = (
    "gpuwm.core.mynn_pbl_scratch",
    "gpuwm.core.mynn_pbl_runtime",
    "gpuwm.core.mynn_pbl_gpu",
)


def mynn_column_chunk() -> int:
    """The chunk width MYNN will actually use for a wide tile."""
    from gpuwm.core import mynn_pbl_runtime

    return int(mynn_pbl_runtime.MYNN_PBL_COLUMN_CHUNK)


def set_mynn_column_chunk(chunk: int) -> int:
    """Set the process-wide MYNN column chunk; returns the previous value.

    ``mynn_pbl_step`` takes ``column_chunk`` as an argument but
    ``PhysicsDriver._run_mynn_pbl`` does not pass one, so the module constant
    is the only handle a caller has today.  Plumbing it through ``RunConfig``
    is the right home for it and is NOT done here -- this function exists so
    the capacity measurement can be made honestly, and so the number it
    produces can justify that plumbing.

    Every width is bit-identical: each MYNN kernel gives one thread one whole
    column and reads no neighbour, and ``tests/test_mynn_pbl_scratch.py``
    asserts the split matches the single wide call rather than assuming it.
    The workspace is exactly 52,352 bytes per column at ``nz = 49``
    (818.0 MiB at the shipped 16,384), and the scheme's own timing table puts
    the plateau at 8k-25k columns: ``8,192 = 1.5609 us/column`` against
    ``16,384 = 1.4929`` -- 4.6% slower for half the workspace -- while
    ``4,096 = 2.5064`` is 68% slower for a quarter.
    """
    import importlib

    chunk = int(chunk)
    if chunk < 1:
        raise ValueError("MYNN column chunk must be positive")
    previous = mynn_column_chunk()
    for name in _MYNN_CHUNK_MODULES:
        module = importlib.import_module(name)
        if not hasattr(module, "MYNN_PBL_COLUMN_CHUNK"):
            raise AttributeError(
                f"{name} no longer binds MYNN_PBL_COLUMN_CHUNK; the set of "
                "modules that must move together has changed and this "
                "function would silently leave one of them behind")
        module.MYNN_PBL_COLUMN_CHUNK = chunk
    return previous


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _radiation_active(cfg) -> bool:
    from gpuwm.core.physics import radiation_enabled

    return bool(radiation_enabled(cfg))


def _default_column_chunk() -> int:
    from gpuwm.config import DEFAULT_COLUMN_CHUNK

    return int(DEFAULT_COLUMN_CHUNK)


def _p_top_of(cfg) -> float:
    """The base state's model-top pressure, without building a DomainState.

    ``SharedRRTMGPChunkWorkspace`` is shape-checked against the adapter's
    ``state.p_top`` on every radiation call and rrtmgp.py:2069-2078 RAISES on
    drift, so the workspace has to be built with the number the tile states
    will actually carry.  ``p_top`` is a function of the base state's theta
    profile, so this reproduces
    :func:`tilestream.physics_inventory.default_builder`'s WK82 sounding
    rather than a convenient constant: a constant-300 K column at the same
    20 km top gives 2,509.5 Pa against WK82's 5,717.8 Pa, and the adapter
    rejects the mismatch on the first radiation call.  A caller using a
    different builder passes ``p_top=`` to :func:`build` explicitly.
    """
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.state import DTYPE
    from gpuwm.verify.cases.wk82 import wk82_sounding

    coord = make_vertical_coord(cfg.nz, hybrid_opt=cfg.hybrid_opt,
                                etac=cfg.etac)
    base = make_base_state(coord, lambda z: wk82_sounding(z)[0],
                           p_surf=cfg.p_surf, ztop=cfg.ztop, terrain_z=None)
    # The float32 round trip is not cosmetic: load_base stores p_top as
    # float32 and the adapter compares the two as Python floats, so the
    # float64 base value 5717.781854825707 fails equality against the
    # state's 5717.78173828125 and the run dies on the first radiation call.
    return float(DTYPE(base.p_top))
