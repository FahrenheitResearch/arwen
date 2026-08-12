"""Timing for the tiled streaming pipeline, with correctness attached.

Everything in here exists because a streaming bug that skips work looks
EXACTLY like a speedup.  So:

* every timed configuration also produces a SHA-256 over the persisted set
  (:func:`tilestream.harness.hash_state`), and the runner refuses to report a
  number it could have checked but did not;
* :class:`TiledRunner` -- the object that is actually timed -- is checked
  against :func:`tilestream.driver.run_tiled`, the loop the bit-exact gate
  validated, by :func:`selftest_runner_matches_driver`.  A faster loop that
  computes something else is not a result.

Why a separate runner at all
----------------------------
``run_tiled`` builds its plan, its tile ``DomainState`` s, its streams and its
shadow store on every call.  That is right for a correctness gate and wrong
for a measurement: at a 544-cell tile the setup is milliseconds against a
sweep of hundreds, but at a 1056-cell tile it is not, and the setup cost would
silently masquerade as tiling overhead.  :class:`TiledRunner` hoists all of it
into ``__init__`` and exposes :meth:`TiledRunner.sweep`, which is one full
domain step and nothing else.  The semantics -- shadow buffers, the swap, the
pre-loop ``deviceSynchronize`` that a non-blocking stream needs -- are
reproduced exactly; ``selftest_runner_matches_driver`` is what makes that
claim checkable rather than asserted.

Timing method
-------------
Wall time is ``perf_counter`` around a region bracketed by full
``deviceSynchronize`` calls, because the pipeline spans several streams and a
single stream's events cannot see the whole thing.  The per-phase split
(:meth:`TiledRunner.sweep` with ``events=True``) uses CUDA events recorded on
each tile's own stream, which measures how long each phase OCCUPIED that
stream -- including any stall waiting for a shared engine.  That is the number
that answers "is the copy actually hidden?", and it is why the sum of the
parts exceeding the wall clock is the signal of successful overlap rather
than an inconsistency.

Seeding a domain too big to build
---------------------------------
:func:`seed_host_store` fills a pinned host store directly, without ever
constructing a full-domain ``DomainState`` -- necessary once the domain
exceeds VRAM, since ``harness.make_state`` is a device object.  It reproduces
``harness.make_state``'s amplitudes on top of the at-rest profile, which is
horizontally uniform under milestone-one settings (``terrain_opt=0``,
``map_proj=0``) and so can be read off a tiny probe state and broadcast.
``p``/``al``/``alt`` are left at their at-rest values rather than being made
consistent with the seeded ``thp``/``php``/``mup``: ``dycore.step`` recomputes
all three from the prognostic set through ``calc_p_alpha`` four times per
step, so the inconsistency does not survive the first step and never leaves
the O(1) perturbation range.  It is NOT the same state
``harness.make_state`` builds and must not be compared against one -- both
sides of any comparison here are filled from the same store.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
import time
from typing import Any, Callable, Sequence

import numpy as np

from tilestream import gather as _gather
from tilestream import harness as _harness
from tilestream import spec as _spec


__all__ = [
    "BenchResult",
    "PhaseBreakdown",
    "TiledRunner",
    "device_store_like",
    "free_vram_gib",
    "gpu_probe",
    "host_mem_gib",
    "repeat_stats",
    "seed_host_store",
    "selftest_runner_matches_driver",
    "time_monolithic",
]

_NS_PER_S = 1e9


# --------------------------------------------------------------------------
# machine state
# --------------------------------------------------------------------------

def free_vram_gib() -> tuple[float, float]:
    """``(free, total)`` device memory in GiB, from ``cudaMemGetInfo``."""
    import cupy as cp

    free, total = cp.cuda.runtime.memGetInfo()
    return free / 2 ** 30, total / 2 ** 30


def gpu_probe() -> dict[str, float]:
    """``{util_pct, mem_used_mib, sm_clock_mhz, power_w, temp_c}``.

    Sampled from ``nvidia-smi`` so it sees the WHOLE card, including other
    processes.  Every timed result carries one of these taken before and one
    after, because this box is shared and a neighbour's job showed up mid-run
    once already -- and a 1.5x swing that is really contention looks exactly
    like a 1.5x regression.  ``mem_used_mib`` includes this process, so the
    useful signal is how much it MOVES while a stage runs.
    """
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,clocks.sm,"
             "power.draw,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip()
        vals = [float(v) for v in out.split(",")]
    except Exception:
        return {}
    keys = ("util_pct", "mem_used_mib", "sm_clock_mhz", "power_w", "temp_c")
    return dict(zip(keys, vals))


def host_mem_gib() -> dict[str, float]:
    """``MemTotal``/``MemFree``/``MemAvailable`` in GiB, from ``/proc/meminfo``.

    Watched during the big runs: pinned pages cannot be swapped, so driving
    ``MemAvailable`` to zero does not degrade the measurement, it destroys the
    machine.
    """
    out: dict[str, float] = {}
    with open("/proc/meminfo", "r", encoding="ascii") as fh:
        for line in fh:
            key, _, rest = line.partition(":")
            if key in ("MemTotal", "MemFree", "MemAvailable", "Mlocked"):
                out[key] = float(rest.strip().split()[0]) * 1024 / 2 ** 30
    return out


# --------------------------------------------------------------------------
# results
# --------------------------------------------------------------------------

@dataclass
class PhaseBreakdown:
    """Stream-occupancy totals for one sweep, in milliseconds.

    Each figure is the sum over tiles of the CUDA-event interval bracketing
    that phase on that tile's stream.  ``total_parts`` exceeding ``wall`` is
    the direct evidence that phases on different streams ran concurrently;
    ``total_parts`` equal to ``wall`` means they serialised.

    ``stall_ms`` is the interval between the end of a tile's gather and the
    start of its compute ON THE SAME STREAM -- time the stream sat idle
    holding a fully gathered tile.  It is the number that distinguishes "the
    copy is slow" from "the copy finished long ago and the launch was late".
    """

    gather_ms: float = 0.0
    compute_ms: float = 0.0
    scatter_ms: float = 0.0
    stall_ms: float = 0.0
    wall_ms: float = 0.0
    tiles: int = 0

    @property
    def total_parts_ms(self) -> float:
        return self.gather_ms + self.compute_ms + self.scatter_ms

    @property
    def overlap_ratio(self) -> float:
        """``sum(parts) / wall``.  1.0 = fully serial, >1 = overlapping."""
        return self.total_parts_ms / self.wall_ms if self.wall_ms else float("nan")

    @property
    def transfer_exposed_ms(self) -> float:
        """Wall time not covered by the compute phase of any stream.

        With perfect overlap this is ~0: every byte moved while a kernel ran.
        Equal to ``gather_ms + scatter_ms`` means nothing was hidden at all.
        """
        return self.wall_ms - self.compute_ms

    @property
    def hidden_fraction(self) -> float:
        """Fraction of the transfer time that compute actually covered."""
        moved = self.gather_ms + self.scatter_ms
        if moved <= 0:
            return float("nan")
        return max(0.0, 1.0 - self.transfer_exposed_ms / moved)


def repeat_stats(samples: Sequence[float]) -> dict[str, float]:
    """``min``/``median``/``max``/``spread`` over repeated timings.

    THE HEADLINE STATISTIC IS THE MINIMUM, and deliberately so.  Every source
    of noise on a shared GPU is additive -- a neighbour's kernels, a power-cap
    clock drop, a page fault -- so the fastest observed run is the closest
    estimate of the work's true cost, while the mean is an estimate of the
    machine's business.  ``spread`` (max/min) is reported alongside so a
    reader can see when the minimum is not well determined; anything above
    about 1.15 means the box was busy and the number should be distrusted.
    """
    xs = sorted(float(s) for s in samples)
    if not xs:
        return {}
    mid = len(xs) // 2
    median = xs[mid] if len(xs) % 2 else 0.5 * (xs[mid - 1] + xs[mid])
    return {"min": xs[0], "median": median, "max": xs[-1],
            "spread": xs[-1] / xs[0] if xs[0] else float("nan"),
            "n": len(xs)}


@dataclass
class BenchResult:
    """One timed configuration, with the hash that makes it trustworthy."""

    label: str
    domain: tuple[int, int, int]        # (nz, ny, nx)
    ms_per_step: float
    digest: str | None = None
    tiles: int = 1
    tile_interior: tuple[int, int] | None = None
    compute_shape: tuple[int, int, int] | None = None
    halo: int = 0
    nbuffers: int = 1
    steps: int = 0
    gathered_bytes: int = 0
    scattered_bytes: int = 0
    phases: PhaseBreakdown | None = None
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def cells(self) -> int:
        nz, ny, nx = self.domain
        return nz * ny * nx

    @property
    def ns_per_cell(self) -> float:
        """Nanoseconds per USEFUL cell per step -- the comparable figure.

        Useful means a cell of the real domain.  Halo cells are recomputed
        work, not extra domain, so they are deliberately NOT in the
        denominator; that is what makes the tiling overhead visible instead of
        hidden.
        """
        return self.ms_per_step * 1e6 / self.cells

    @property
    def gathered_ns_per_cell(self) -> float:
        """Nanoseconds per COMPUTED cell -- what the kernels actually saw."""
        if self.compute_shape is None:
            return self.ns_per_cell
        nz, cny, cnx = self.compute_shape
        return self.ms_per_step * 1e6 / (nz * cny * cnx * self.tiles)

    @property
    def redundancy(self) -> float:
        """Computed cells divided by useful cells."""
        if self.compute_shape is None:
            return 1.0
        nz, cny, cnx = self.compute_shape
        return (nz * cny * cnx * self.tiles) / self.cells

    @property
    def pcie_gbps(self) -> float:
        """Aggregate H2D+D2H rate over the wall clock, in GB/s (10^9)."""
        moved = self.gathered_bytes + self.scattered_bytes
        if not moved or not self.steps:
            return 0.0
        return moved / (self.ms_per_step * 1e-3 * self.steps) / 1e9

    def row(self) -> str:
        return (f"{self.label:<38s} {self.ms_per_step:10.2f} ms/step "
                f"{self.ns_per_cell:8.3f} ns/cell "
                f"{(self.digest[:16] if self.digest else '-'):>16s}")


# --------------------------------------------------------------------------
# monolithic baseline
# --------------------------------------------------------------------------

def time_monolithic(state, cfg, *, steps: int = 4, warmup: int = 2,
                    label: str = "monolithic", hash_after: bool = True,
                    ) -> BenchResult:
    """Time ``dycore.step`` on a whole-domain state, in place.

    ``warmup`` steps run first (NVRTC compilation, pool growth and the first
    autotune all land there), then ``steps`` are timed between two full device
    synchronizations.  The state is advanced by ``warmup + steps`` steps in
    total and the digest, if requested, is taken at the end -- so a caller
    comparing against a tiled run must advance the tiled run by the same
    total.
    """
    import cupy as cp

    from gpuwm.core.dycore import step

    for _ in range(int(warmup)):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()

    t0 = time.perf_counter()
    for _ in range(int(steps)):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    elapsed = time.perf_counter() - t0

    digest = _harness.hash_state(state) if hash_after else None
    return BenchResult(
        label=label, domain=(cfg.nz, cfg.ny, cfg.nx),
        ms_per_step=elapsed * 1e3 / max(1, int(steps)),
        digest=digest, steps=int(steps),
        compute_shape=(cfg.nz, cfg.ny, cfg.nx),
    )


# --------------------------------------------------------------------------
# stores
# --------------------------------------------------------------------------

def device_store_like(manifest_source, cfg, *, fill: float | None = None
                      ) -> dict[str, Any]:
    """A full-domain ``{name: cupy array}`` store, for the in-VRAM control.

    Shapes come from ``manifest_source`` -- a ``HostDomainStore``, a
    ``DomainState`` or any ``{name: array}`` -- so the device store is
    guaranteed to carry the same inventory and staggering as the host one it
    is being compared against, rather than a re-derived guess.
    """
    import cupy as cp

    src = manifest_source
    inner = getattr(src, "arrays", None)
    if isinstance(inner, dict):
        src = inner
    src = _gather.inventory(src)
    out: dict[str, Any] = {}
    for name, arr in src.items():
        buf = cp.empty(arr.shape, dtype=arr.dtype)
        if fill is not None:
            buf.fill(fill)
        out[name] = buf
    return out


def seed_host_store(store, cfg, *, seed: int = _harness.DEFAULT_SEED,
                    chunk_rows: int = 256) -> None:
    """Fill a pinned host store with a seeded, at-rest-based state.

    See the module docstring for why this is not ``harness.make_state`` and
    what it does differently.  Written in row chunks so a 745-Mcell domain
    never materialises a full-domain float64 temporary (``standard_normal``
    produces float64; the chunk is cast on the way in).
    """
    from dataclasses import replace

    from tilestream import driver as _driver

    arrays = getattr(store, "arrays", None)
    if not isinstance(arrays, dict):
        arrays = _gather.inventory(store)

    # At-rest profiles.  Horizontally uniform under milestone-one settings,
    # so a 4x4 probe carries every value a full domain would have.
    probe_cfg = replace(cfg, nx=4, ny=4)
    probe = _driver.make_tile_state(probe_cfg)
    profiles: dict[str, np.ndarray] = {}
    for name, arr in _gather.inventory(probe).items():
        host = arr.get()
        profiles[name] = host[..., 0:1, 0:1] if host.ndim >= 2 else host
    del probe
    import cupy as cp
    cp.get_default_memory_pool().free_all_blocks()

    amps = {name: amp for name, amp, _x, _y in _harness._SEED_FIELDS}
    rng = np.random.default_rng(int(seed))

    for name in sorted(arrays):
        dest = arrays[name]
        prof = profiles.get(name)
        if prof is None:
            raise KeyError(f"probe state has no {name!r}; cannot seed it")
        amp = amps.get(name, 0.0)
        nrow = dest.shape[-2] if dest.ndim >= 2 else 1
        for r0 in range(0, nrow, int(chunk_rows)):
            r1 = min(nrow, r0 + int(chunk_rows))
            view = dest[..., r0:r1, :]
            view[...] = prof
            if amp:
                view += (amp * rng.standard_normal(view.shape)).astype(
                    dest.dtype, copy=False)
        if name == "u":                       # periodic duplicate column
            dest[..., :, -1] = dest[..., :, 0]
        elif name == "v":
            dest[..., -1, :] = dest[..., 0, :]
        elif name == "w":                     # rigid lid and ground
            dest[0] = 0.0
            dest[-1] = 0.0


# --------------------------------------------------------------------------
# the timed loop
# --------------------------------------------------------------------------

class TiledRunner:
    """A set-up-once tiled integrator whose ``sweep`` is one domain step.

    Reproduces :func:`tilestream.driver.run_tiled`'s semantics exactly (see
    the module docstring), with the planning, the tile ``DomainState`` s, the
    CUDA streams and the shadow store built in ``__init__`` so that
    :meth:`sweep` times the integration and nothing else.

    ``store`` is updated in place.  Under ``write_mode="shadow"`` the live data
    is back in ``store`` after an EVEN number of sweeps and in the shadow after
    an odd one, and :meth:`settle` copies it home; under ``write_mode="ring"``
    (the default) there is only one store, so it is always live and
    :meth:`settle` is a no-op.  :attr:`live` always names the buffer holding
    the current state.

    The ring's save and patch are issued inside the gather bracket, so the
    event-instrumented ``gather_ms`` counts them: they are part of the cost of
    handing a tile a time-t window and putting them anywhere else would flatter
    the overlap ratio.
    """

    def __init__(self, store, cfg, tile_nx: int, tile_ny: int, *,
                 halo: int | None = None, nbuffers: int = 2,
                 periodic: bool = True, shadow=None, poison: bool = False,
                 names: Sequence[str] | None = None,
                 allow_pageable: bool = False, pipeline: str = "prefetch",
                 write_mode: str = "ring", ring_ordering: str = "events",
                 tile_state_factory=None, inventory_fn=None, nz=None,
                 scalars=None, geography=None, geography_names=None,
                 impose_geography_flags: bool = True):
        import cupy as cp

        from tilestream import driver as _driver

        self._cp = cp
        # PHYSICS HOOKS.  The four keywords below are exactly ``run_tiled``'s
        # (see its docstring for why each is structural rather than optional).
        # They are threaded through this class instead of being re-implemented
        # so that the timed loop and the gated loop share their semantics, and
        # :func:`selftest_runner_matches_driver` checks that they agree on the
        # digest at a physics rung as well as a dry one.
        self.inventory_fn = inventory_fn
        self._take = inventory_fn or _gather.inventory
        self.nz_hint = None if nz is None else int(nz)
        # The DOMAIN clock.  Reset onto a buffer before EVERY tile step and
        # advanced once per SWEEP -- a buffer serving k tiles would otherwise
        # run k*dt ahead of the domain inside one sweep and take different
        # cadence branches.  ``None`` for a dry run, which has no clock.
        self.clock = None if scalars is None else dict(scalars)
        self._physics = None
        if self.clock is not None:
            from tilestream import physics_inventory as _physinv

            self._physics = _physinv

        inner = getattr(store, "arrays", None)
        self.home = self._take(inner if isinstance(inner, dict) else store,
                               names)
        if not self.home:
            raise ValueError("store holds none of the persisted attributes")
        self.nz, self.ny, self.nx = _gather.domain_extents(self.home,
                                                           nz=self.nz_hint)
        if (self.nz, self.ny, self.nx) != (cfg.nz, cfg.ny, cfg.nx):
            raise ValueError(
                f"store is {self.nz}x{self.ny}x{self.nx} but cfg is "
                f"{cfg.nz}x{cfg.ny}x{cfg.nx}")

        self.cfg = cfg
        self.halo = int(_harness.halo_radius(cfg) if halo is None else halo)
        self.nbuffers = max(1, int(nbuffers))
        self.names = names
        self.allow_pageable = bool(allow_pageable)
        if pipeline not in ("prefetch", "naive"):
            raise ValueError(f"pipeline must be 'prefetch' or 'naive', "
                             f"got {pipeline!r}")
        self.pipeline = pipeline

        self.specs = _spec.plan_tiles(self.nx, self.ny, int(tile_nx),
                                      int(tile_ny), self.halo, periodic)
        _spec.validate_plan(self.specs, self.ny, self.nx)
        self.cnx, self.cny = self.specs[0].cnx, self.specs[0].cny
        self.tile_cfg = _harness.tile_config(cfg, self.cnx, self.cny)

        factory = tile_state_factory or _driver.make_tile_state
        self.tiles = [factory(self.tile_cfg) for _ in range(self.nbuffers)]
        self.streams = [cp.cuda.Stream(non_blocking=True)
                        for _ in range(self.nbuffers)]

        # GEOGRAPHY.  Read-only input, gathered when a buffer changes tile and
        # never scattered.  Omitting it leaves every tile with the geography it
        # REBUILT from tile_cfg, which is centred on the tile rather than on
        # the domain and is correct only for the exactly-centred tile.
        self.geo_home = None
        self.geography_names = geography_names
        self._geo_tile: list[int | None] = [None] * self.nbuffers
        if geography is not None:
            self.geo_home = _driver.geography_inventory(geography,
                                                        geography_names)
            if not self.geo_home:
                raise ValueError("geography= holds no gatherable arrays")
            if impose_geography_flags:
                flags = _driver.geography_scalars(self.geo_home)
                for tile in self.tiles:
                    _driver._pin_scheme_geography(tile)
                    for name, value in flags.items():
                        setattr(tile, name, value)

        if write_mode not in ("ring", "shadow"):
            raise ValueError(f"write_mode must be 'ring' or 'shadow', "
                             f"got {write_mode!r}")
        self.write_mode = write_mode
        self.ring = None
        self.ring_events = None
        if write_mode == "ring":
            from tilestream import rings as _rings

            if shadow is not None:
                raise ValueError(
                    "a shadow buffer was supplied but write_mode='ring'; the "
                    "ring path keeps a single store and would ignore it")
            kinds = sorted({k for _n, k, _d, _i
                            in _rings.field_geometry(self.home,
                                                     nz=self.nz_hint)})
            self.ring_plan = _rings.build_ring_plan(self.specs, kinds)
            self.ring = _rings.RingArena(
                self.ring_plan, self.home, self.tiles, nz=self.nz_hint,
                names=names, inventory_fn=inventory_fn,
                allow_pageable=allow_pageable)
            if ring_ordering == "events":
                self.ring_events = [cp.cuda.Event() for _ in self.specs]
            elif ring_ordering != "submission":
                raise ValueError(f"ring_ordering must be 'events' or "
                                 f"'submission', got {ring_ordering!r}")
            shadow = None
        elif shadow is None:
            shadow = _driver._empty_like_store(self.home, poison=poison)
        else:
            inner = getattr(shadow, "arrays", None)
            shadow = _gather.inventory(
                inner if isinstance(inner, dict) else shadow, names)
        if shadow is not None:
            for name, arr in self.home.items():
                if tuple(shadow[name].shape) != tuple(arr.shape):
                    raise ValueError(f"shadow mis-shaped for {name!r}")
        self.shadow = shadow

        self._src = self.home
        self._dst = self.home if shadow is None else self.shadow
        self._events = None
        self.gathered_bytes = 0
        self.scattered_bytes = 0
        self.sweeps = 0

        # Same MEASURED hazard run_tiled documents: non-blocking streams do
        # not order against the legacy default stream, and the tile states
        # were just built (and their setup arrays uploaded) on it.  Without
        # this the first tile reads setup that has not landed and comes out
        # all-NaN.
        cp.cuda.runtime.deviceSynchronize()

        # Resolve and cache every transfer plan up front so the first timed
        # sweep does not pay for plan construction.
        for tspec in self.specs:
            _gather.make_plan(self._src, self._take(self.tiles[0], names),
                              tspec, "gather", allow_pageable=self.allow_pageable,
                              nz=self.nz_hint)
            _gather.make_plan(self._take(self.tiles[0], names), self._dst,
                              tspec, "scatter", allow_pageable=self.allow_pageable,
                              nz=self.nz_hint)

    # -- description ----------------------------------------------------

    @property
    def live(self) -> dict[str, Any]:
        """The buffer currently holding the domain state."""
        return self._src

    @property
    def redundancy(self) -> float:
        return (len(self.specs) * self.cnx * self.cny) / float(self.nx * self.ny)

    def bytes_per_sweep(self) -> tuple[int, int]:
        """``(gathered, scattered)`` bytes for one sweep, from the plans."""
        tinv = self._take(self.tiles[0], self.names)
        g = sum(_gather.make_plan(self._src, tinv, s, "gather",
                                  allow_pageable=self.allow_pageable,
                                  nz=self.nz_hint).nbytes
                for s in self.specs)
        s_ = sum(_gather.make_plan(tinv, self._dst, s, "scatter",
                                   allow_pageable=self.allow_pageable,
                                   nz=self.nz_hint).nbytes
                 for s in self.specs)
        return g, s_

    # -- the loop -------------------------------------------------------

    # -- issue order, which is the whole ballgame on one copy engine ----
    #
    # MEASURED.  This card reports asyncEngineCount == 1: there is a SINGLE
    # DMA queue serving every stream, and it is drained in SUBMISSION order.
    # The obvious loop --
    #
    #     for i: gather(i); step(i); scatter(i)
    #
    # submits scatter(i), which cannot start until step(i) finishes, AHEAD of
    # gather(i+1), which could start immediately.  The copy engine stops at
    # the head of its queue and idles for the whole of step(i).  Nothing
    # overlaps, and the cost of the transfers is added to the wall clock in
    # full.  Measured at 1950^2, tile 512, host store: sum(parts)/wall = 0.909
    # and 1037 ms/step against the 771 ms/step of the same tiling with the
    # store in VRAM -- the entire 267 ms gap is the 275 ms of gather+scatter,
    # completely exposed.
    #
    # ``pipeline="prefetch"`` submits gather(i+depth) BEFORE step(i) and
    # scatter(i), so a ready copy is always ahead of a blocked one in the
    # queue.  Same arithmetic, same digest, transfers hidden.  ``depth`` is
    # ``nbuffers - 1``: a tile can only be prefetched into a buffer whose
    # previous occupant has already been scattered, and stream ordering on
    # that buffer's stream enforces exactly that with no extra events.
    #
    # ``pipeline="naive"`` keeps the original order so the difference can be
    # measured rather than asserted.  Both produce identical digests.

    def _event_slots(self) -> list[list]:
        """Preallocated, reused CUDA events -- five per tile.

        Created once and re-recorded every sweep.  Allocating fresh events
        inside the loop measurably perturbs the prefetch pipeline (the
        create/destroy traffic lands between the queued copies), which is the
        one thing an instrument must not do.
        """
        if self._events is None:
            cp = self._cp
            self._events = [[cp.cuda.Event() for _ in range(5)]
                            for _ in self.specs]
        return self._events

    def _issue_gather(self, itile: int, slots, events: bool,
                      skip_transfer: bool = False) -> None:
        b = itile % self.nbuffers
        stream = self.streams[b]
        with stream:
            if events:
                slots[itile][0].record(stream)
            if not skip_transfer:
                # Geography first, and only when this buffer changes tile.
                # Counted inside the gather bracket: it is a real transfer the
                # pipeline pays for, just an infrequent one.
                if self.geo_home is not None and self._geo_tile[b] != itile:
                    from tilestream import driver as _driver

                    _gather.gather_tile(
                        self.geo_home, self.tiles[b], self.specs[itile], stream,
                        allow_pageable=self.allow_pageable,
                        names=self.geography_names,
                        inventory_fn=_driver.geography_inventory,
                        nz=self.nz_hint)
                    self._geo_tile[b] = itile
                _gather.gather_tile(self._src, self.tiles[b], self.specs[itile],
                                    stream, allow_pageable=self.allow_pageable,
                                    names=self.names,
                                    inventory_fn=self.inventory_fn,
                                    nz=self.nz_hint)
                # The ring save and patch are timed INSIDE the gather bracket
                # deliberately: they are part of what it costs to hand this
                # tile a time-t window, and hiding them somewhere else would
                # make the overlap ratio flatter than the truth.
                if self.ring is not None:
                    tinv = self._take(self.tiles[b], self.names)
                    self.ring.save(itile, tinv, stream)
                    if self.ring_events is not None:
                        self.ring_events[itile].record(stream)
                        for k in self.ring_plan.patch_deps[itile]:
                            if k % self.nbuffers != b:
                                stream.wait_event(self.ring_events[k])
                    self.ring.patch(itile, tinv, stream)
            if events:
                slots[itile][1].record(stream)

    def _issue_compute_scatter(self, itile: int, slots, events: bool,
                               skip_transfer: bool = False,
                               skip_compute: bool = False) -> None:
        from gpuwm.core.dycore import step

        b = itile % self.nbuffers
        stream = self.streams[b]
        # The domain clock, onto this buffer, before this tile's step.  Outside
        # the event bracket because it is a handful of Python attribute writes,
        # not GPU work -- but it must happen, and per TILE.
        if self.clock is not None and not skip_compute:
            self._physics.set_carrier_scalars(self.tiles[b], self.clock)
        with stream:
            if events:
                slots[itile][2].record(stream)
            if not skip_compute:
                step(self.tiles[b], self.tile_cfg)
            if events:
                slots[itile][3].record(stream)
            if not skip_transfer:
                if self.ring_events is not None:
                    for j in self.ring_plan.war_deps[itile]:
                        if j % self.nbuffers != b:
                            stream.wait_event(self.ring_events[j])
                _gather.scatter_tile(self.tiles[b], self._dst,
                                     self.specs[itile], stream,
                                     allow_pageable=self.allow_pageable,
                                     names=self.names,
                                     inventory_fn=self.inventory_fn,
                                     nz=self.nz_hint)
            if events:
                slots[itile][4].record(stream)

    def sweep(self, n: int = 1, *, events: bool = False,
              pipeline: str | None = None, skip_transfer: bool = False,
              skip_compute: bool = False,
              progress: Callable | None = None) -> PhaseBreakdown:
        """Advance the domain ``n`` steps.  Returns wall time (+ phases).

        ``pipeline`` selects the issue order -- see the block comment above;
        it defaults to the runner's ``pipeline`` attribute (``"prefetch"``).

        With ``events=True`` preallocated CUDA events bracket each phase of
        every tile on that tile's own stream, plus the gap between the end of
        the gather and the start of the compute.  Off by default so the
        headline wall-clock numbers come from an uninstrumented loop.

        ``skip_transfer`` and ``skip_compute`` produce the two ATTRIBUTION
        VARIANTS used to decide whether transfers really hide under compute.
        Neither is a correct integration -- ``skip_transfer`` re-steps a stale
        tile buffer, ``skip_compute`` moves bytes nobody updated -- so their
        wall times are diagnostics only and the state they leave behind is
        garbage.  They exist because the non-invasive comparison
        ``wall_full`` vs ``max(wall_compute_only, wall_transfer_only)`` answers
        the overlap question without an instrument that perturbs the thing it
        measures.
        """
        cp = self._cp

        mode = self.pipeline if pipeline is None else pipeline
        if mode not in ("prefetch", "naive"):
            raise ValueError(f"pipeline must be 'prefetch' or 'naive', got {mode!r}")
        ntiles = len(self.specs)
        # A tile may only be prefetched into a buffer whose previous occupant
        # has already been scattered, so the depth can never exceed
        # nbuffers-1.  With a single buffer there is nowhere to prefetch to
        # and the order degenerates to naive -- silently allowing depth 1
        # there would have tile i+1 overwrite the tile i being computed.
        depth = self.nbuffers - 1 if mode == "prefetch" else 0
        if depth < 1:
            mode = "naive"
        slots = self._event_slots() if events else None

        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _istep in range(int(n)):
            if mode == "prefetch":
                # Prime the pipeline.  Never across a sweep boundary: the
                # shadow buffers swap there, so a tile gathered early would
                # read the wrong generation.
                for i in range(min(depth, ntiles)):
                    self._issue_gather(i, slots, events, skip_transfer)
            for itile in range(ntiles):
                if mode == "prefetch":
                    nxt = itile + depth
                    if nxt < ntiles:
                        self._issue_gather(nxt, slots, events, skip_transfer)
                else:
                    self._issue_gather(itile, slots, events, skip_transfer)
                self._issue_compute_scatter(itile, slots, events,
                                            skip_transfer, skip_compute)
                if progress is not None:
                    progress(_istep, itile, self.specs[itile])
            for stream in self.streams:
                stream.synchronize()
            cp.cuda.runtime.deviceSynchronize()
            # ONE domain step per SWEEP, not per tile.  ``skip_compute`` steps
            # nothing, so advancing there would desynchronise the clock from
            # the state it describes.
            if self.clock is not None and not skip_compute:
                from tilestream import driver as _driver

                used = min(self.nbuffers, ntiles)
                self.clock = _driver._advance_clock(
                    self.clock, self.tiles[:used],
                    (ntiles - 1) % self.nbuffers, self._physics)
            if self.shadow is not None:
                self._src, self._dst = self._dst, self._src
            self.sweeps += 1
        cp.cuda.runtime.deviceSynchronize()
        wall = time.perf_counter() - t0

        out = PhaseBreakdown(wall_ms=wall * 1e3 / max(1, int(n)),
                             tiles=ntiles)
        if events:
            for e0, e1, e2, e3, e4 in slots:
                out.gather_ms += cp.cuda.get_elapsed_time(e0, e1)
                out.stall_ms += cp.cuda.get_elapsed_time(e1, e2)
                out.compute_ms += cp.cuda.get_elapsed_time(e2, e3)
                out.scatter_ms += cp.cuda.get_elapsed_time(e3, e4)
        g, s_ = self.bytes_per_sweep()
        self.gathered_bytes = g
        self.scattered_bytes = s_
        return out

    def attribute(self, n: int = 2, *, pipeline: str | None = None
                  ) -> dict[str, float]:
        """Non-invasive overlap attribution, in ms per sweep.

        Times the same loop three ways -- everything, compute suppressed,
        transfers suppressed -- with no events in any of them.  Interpretation:

        * ``full ~= max(compute_only, transfer_only)`` -- overlap is working;
        * ``full ~= compute_only + transfer_only`` -- the phases serialised;
        * ``overlap_efficiency`` puts that on a 0..1 scale, 1 meaning the
          cheaper phase is entirely hidden inside the more expensive one.

        DESTROYS the domain state (see :meth:`sweep`): call it last, or
        re-seed afterwards.
        """
        cp = self._cp

        full = self.sweep(n, pipeline=pipeline).wall_ms
        # compute-only starts from the tile buffers the full sweep just
        # gathered, so the kernels chew on real data rather than on whatever
        # was left in memory.  The check below is what makes that claim
        # rather than hoping: a tile that drifted to NaN/Inf would be timing
        # arithmetic the real run never does.
        comp = self.sweep(n, pipeline=pipeline, skip_transfer=True).wall_ms
        finite = all(bool(cp.isfinite(t.p).all()) for t in self.tiles)
        xfer = self.sweep(n, pipeline=pipeline, skip_compute=True).wall_ms
        serial, ideal = comp + xfer, max(comp, xfer)
        denom = serial - ideal
        return {
            "full_ms": full, "compute_only_ms": comp, "transfer_only_ms": xfer,
            "serial_ms": serial, "ideal_ms": ideal,
            "overlap_efficiency": (serial - full) / denom if denom > 0 else float("nan"),
            "exposed_transfer_ms": full - comp,
            "compute_only_finite": finite,
        }

    def settle(self) -> None:
        """Ensure ``store`` (not the shadow) holds the live state.

        A no-op in ring mode: there is only one store, so the live data is
        never anywhere else and an odd sweep count costs nothing.  That is
        not only a memory saving -- the shadow path's full-domain copy-home
        is the reason the bench insists on an even step count.
        """
        from tilestream import driver as _driver

        if self.shadow is not None and self._src is not self.home:
            _driver._copy_into(self.home, self._src)
            self._src, self._dst = self.home, self.shadow

    def digest(self) -> str:
        """SHA-256 of the live buffer, in the harness's canonical order.

        Under a physics ``inventory_fn`` this covers the WHOLE carrier
        manifest -- ``state._scratch`` and ``state.physics`` included -- in
        sorted name order.  It has to: the dry order walks
        ``STATE_SERIALIZED_ATTRS`` only, which is 25 of the 229 carriers at
        the top rung, and a timing whose hash cannot see 204 of the arrays it
        moved is not a check.
        """
        import hashlib

        live = self.live
        if self.inventory_fn is not None:
            from tilestream import physics_inventory as _physinv

            per_field = _physinv.field_digests(live)
            digest = hashlib.sha256()
            for name in sorted(per_field):
                digest.update(name.encode("utf-8"))
                digest.update(bytes.fromhex(per_field[name]))
            return digest.hexdigest()
        digest = hashlib.sha256()
        for name in _gather.serialized_attrs():
            if name in live:
                digest.update(_harness._digest_payload(f"state.{name}",
                                                       live[name]))
        return digest.hexdigest()

    def result(self, phases: PhaseBreakdown, label: str, *,
               digest: str | None = None, steps: int = 1) -> BenchResult:
        return BenchResult(
            label=label, domain=(self.nz, self.ny, self.nx),
            ms_per_step=phases.wall_ms, digest=digest, tiles=len(self.specs),
            tile_interior=(self.specs[0].interior_ny, self.specs[0].interior_nx),
            compute_shape=(self.tile_cfg.nz, self.cny, self.cnx),
            halo=self.halo, nbuffers=self.nbuffers, steps=int(steps),
            gathered_bytes=self.gathered_bytes,
            scattered_bytes=self.scattered_bytes,
            phases=phases,
        )

    def ring_bytes(self) -> int:
        """Bytes the ring arena holds; 0 under ``write_mode='shadow'``."""
        return 0 if self.ring is None else self.ring.nbytes

    def _dst_home(self):
        """The write target when the state is back in ``home``.

        ``home`` itself under ``write_mode='ring'`` -- there is no second
        buffer -- and the shadow otherwise.  A caller resetting the sweep
        parity wants this rather than ``self.shadow``, which is ``None`` in
        ring mode and would silently make the next sweep write nowhere.
        """
        return self.home if self.shadow is None else self.shadow

    def free(self) -> None:
        self.tiles.clear()
        self.streams.clear()
        self._events = None
        self.shadow = None
        self.ring_events = None
        if self.ring is not None:
            self.ring.free()
            self.ring = None
        self._src = self._dst = self.home
        gc.collect()
        self._cp.get_default_memory_pool().free_all_blocks()


# --------------------------------------------------------------------------
# the check that makes the numbers mean something
# --------------------------------------------------------------------------

def selftest_runner_matches_driver(nx: int = 192, ny: int = 192, nz: int = 49,
                                   tile: int = 64, halo: int = 16,
                                   nsteps: int = 2, seed: int = 20260731,
                                   ) -> dict[str, str]:
    """``TiledRunner`` vs ``run_tiled`` vs monolithic, all three digests.

    The timed loop in this module is not the loop the bit-exact gate ran.
    This runs all three on the same seeded domain and returns their digests;
    the caller asserts they are equal.  Without this, every number below is a
    measurement of an unvalidated loop.
    """
    import cupy as cp

    from tilestream import driver as _driver

    cfg = _harness.make_config(nx, ny, nz)

    ref = _harness.make_state(cfg, seed=seed)
    _harness.run_steps(ref, cfg, nsteps)
    mono = _harness.hash_state(ref)
    del ref
    cp.get_default_memory_pool().free_all_blocks()

    a = _harness.make_state(cfg, seed=seed)
    store_a = {n: v.copy() for n, v in _harness.state_arrays(a).items()}
    del a
    cp.get_default_memory_pool().free_all_blocks()
    _driver.run_tiled(store_a, cfg, tile, tile, halo=halo, nsteps=nsteps,
                      nbuffers=2)
    import hashlib
    h = hashlib.sha256()
    for name in _gather.serialized_attrs():
        if name in store_a:
            h.update(_harness._digest_payload(f"state.{name}", store_a[name]))
    driver_digest = h.hexdigest()
    del store_a
    cp.get_default_memory_pool().free_all_blocks()

    out = {"monolithic": mono, "run_tiled": driver_digest}
    for mode, nb, wm in (("naive", 2, "ring"), ("prefetch", 2, "ring"),
                         ("prefetch", 3, "ring"), ("prefetch", 2, "shadow"),
                         ("naive", 1, "ring")):
        b = _harness.make_state(cfg, seed=seed)
        store_b = {n: v.copy() for n, v in _harness.state_arrays(b).items()}
        del b
        cp.get_default_memory_pool().free_all_blocks()
        runner = TiledRunner(store_b, cfg, tile, tile, halo=halo, nbuffers=nb,
                             pipeline=mode, write_mode=wm)
        runner.sweep(nsteps)
        runner.settle()
        out[f"TiledRunner/{wm}/{mode}/nb={nb}"] = runner.digest()
        runner.free()
        del store_b
        cp.get_default_memory_pool().free_all_blocks()
    return out


def physics_rung(name: str = "full+MYNN+Noah-MP") -> dict:
    """Config overrides for one rung of the physics ladder.

    The same definitions ``test_gate`` runs its bit-exact matrix over, kept
    here so a benchmark can name a rung without importing the gate.
    ``ztop=20000`` is load-bearing: at the harness default of 8 km RRTMGP pads
    to 140 layers against its own limit of 128 and raises.
    """
    moist = dict(moist=True, mp_physics=10, ztop=20000.0)
    full = dict(moist, km_opt=4, sf_sfclay_physics=91, bl_pbl_physics=1,
                bldt=0.0, sf_surface_physics=2, ra_sw_physics=4,
                ra_lw_physics=4, radt_minutes=12.0, cu_physics=1,
                cudt_minutes=5.0)
    rungs = {
        "dry": {},
        "mp10": dict(moist),
        "full": dict(full),
        "full+MYNN": dict(full, sf_sfclay_physics=5, bl_pbl_physics=5),
        "full+MYNN+Noah-MP": dict(full, sf_sfclay_physics=5, bl_pbl_physics=5,
                                  sf_surface_physics=4),
    }
    if name not in rungs:
        raise KeyError(f"unknown rung {name!r}; have {sorted(rungs)}")
    return rungs[name]


def selftest_physics_runner_matches_driver(
        nx: int = 96, ny: int = 80, nz: int = 49, tile_x: int = 48,
        tile_y: int = 40, nsteps: int = 2, rung: str = "full+MYNN+Noah-MP",
        ) -> dict[str, str]:
    """The physics version of :func:`selftest_runner_matches_driver`.

    ``TiledRunner`` grew the four physics hooks (carrier inventory, explicit
    ``nz``, the domain clock, geography) by hand; ``run_tiled`` is the loop the
    bit-exact gate certified.  Two implementations of the same semantics is
    exactly the situation this project has been burned by, so this runs the
    monolithic, the gated and the TIMED loops on one seeded physics domain and
    returns their carrier digests for the caller to compare.

    Digests here cover the WHOLE manifest (229 carriers at the default rung),
    not ``STATE_SERIALIZED_ATTRS``: a check that cannot see ``state.physics``
    would pass while the timed loop dropped every field in it.
    """
    import hashlib

    import cupy as cp

    from tilestream import driver as _driver
    from tilestream import physics_inventory as _physinv

    cfg = _harness.make_config(nx, ny, nz, **physics_rung(rung))
    halo = _harness.halo_radius(cfg)

    def manifest_digest(arrays) -> str:
        per = _physinv.field_digests(arrays)
        h = hashlib.sha256()
        for name in sorted(per):
            h.update(name.encode("utf-8"))
            h.update(bytes.fromhex(per[name]))
        return h.hexdigest()

    def seeded():
        state, drv = _harness.make_physics_state(cfg)
        _harness.run_steps(state, cfg, 1)          # lazy carriers (KF w0avg)
        return state, drv

    ref, _drv = seeded()
    _harness.run_steps(ref, cfg, nsteps)
    mono = manifest_digest(_physinv.carrier_inventory(ref))
    del ref, _drv
    cp.get_default_memory_pool().free_all_blocks()

    out = {"monolithic": mono}

    src, _drv = seeded()
    start = {k: v.copy() for k, v in _physinv.carrier_inventory(src).items()}
    scalars = _physinv.carrier_scalars(src)
    del src, _drv
    cp.get_default_memory_pool().free_all_blocks()

    store = {k: v.copy() for k, v in start.items()}
    _driver.run_tiled(store, cfg, tile_x, tile_y, halo=halo, nsteps=nsteps,
                      nbuffers=2,
                      **{**_driver.physics_run_kwargs(cfg, None),
                         "scalars": dict(scalars)})
    out["run_tiled"] = manifest_digest(store)
    del store
    cp.get_default_memory_pool().free_all_blocks()

    for mode, nb in (("prefetch", 2), ("naive", 2), ("prefetch", 1)):
        store = {k: v.copy() for k, v in start.items()}
        kw = _driver.physics_run_kwargs(cfg, None)
        runner = TiledRunner(
            store, cfg, tile_x, tile_y, halo=halo, nbuffers=nb, pipeline=mode,
            tile_state_factory=kw["tile_state_factory"],
            inventory_fn=kw["inventory_fn"], nz=int(cfg.nz),
            scalars=dict(scalars))
        runner.sweep(nsteps)
        runner.settle()
        out[f"TiledRunner/{mode}/nb={nb}"] = runner.digest()
        runner.free()
        del store, runner
        cp.get_default_memory_pool().free_all_blocks()
    return out


if __name__ == "__main__":                    # pragma: no cover - manual
    d = selftest_runner_matches_driver()
    for k, v in d.items():
        print(f"  {k:<12s} {v}")
    print("AGREE" if len(set(d.values())) == 1 else "DISAGREE")
