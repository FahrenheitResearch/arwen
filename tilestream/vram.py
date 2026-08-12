"""Where the device memory actually goes -- allocation by allocation.

Every VRAM number this project has quoted so far came from one of two places:
``cupy.get_default_memory_pool().used_bytes()`` (a single scalar) or
``cudaMemGetInfo`` (a single scalar that also counts every other process on
the card).  Neither can answer the question this workstream exists to answer,
which is not "how much" but "WHICH ALLOCATION, and would it still be needed if
we did something else".  A category -- "radiation", "physics", "the packer" --
is not an answer either: the survey that produced the 7.4 GiB reclaimable
figure attributed to categories, and a category cannot be freed.

So this module keeps a LIVE LEDGER: every pooled device allocation, its size,
and the source line that asked for it, from the moment the ledger is armed
until the moment each block is returned.  At any point it can say what is
resident right now, broken down by ``file:line``, and it reconciles against
CuPy's own counter -- :meth:`DeviceLedger.check` raises if the ledger's live
sum and ``pool.used_bytes()`` disagree, so a silent gap in the accounting
fails loudly instead of producing a tidy, wrong table.

THE FOUR NUMBERS, AND WHY ALL FOUR ARE NEEDED
---------------------------------------------
==========================  ===============================================
``pool_used``               bytes the program is holding through live CuPy
                            arrays.  This is what a domain-size formula
                            should be built from.
``pool_total``              bytes the pool has taken from the driver and
                            not given back.  Freed-but-retained blocks live
                            in the gap; they are unavailable to any other
                            allocation with a different size class, so they
                            are real occupancy even though nothing points
                            at them.
``device_used``             ``total - free`` from ``cudaMemGetInfo``.  Sees
                            the CUDA context, every JIT-compiled module,
                            cuBLAS/cuFFT handles and any non-pool
                            allocation -- none of which the hook can see.
``ledger_live``             this module's own sum, attributed to source
                            lines.  Must equal ``pool_used``.
==========================  ===============================================

``device_used - pool_total`` is the non-pool device footprint.  On this
codebase it is dominated by the CUDA context plus NVRTC module images, it is
per-PROCESS, and it does not shrink when a domain shrinks -- which is exactly
the kind of fact a per-cell formula hides.

WHAT THE HOOK CAN AND CANNOT SEE
--------------------------------
CuPy calls the hook for pool ``malloc``/``free`` (a logical allocation) and
for ``alloc`` (an actual ``cudaMalloc`` when the pool has to grow).  Both are
recorded and both are attributed, because they answer different questions:
``malloc`` says who ASKED, ``alloc`` says what the card actually gave out.
A site with a large malloc total and a small alloc total is being served from
recycled blocks and costs nothing new; a site with a large alloc total is the
one that grew the pool.

The hook cannot see: the CUDA context, NVRTC module loads, memory allocated
by a library through its own allocator, or pinned HOST memory (that is
:mod:`tilestream.hoststore`'s business).  ``device_used`` catches the first
three in aggregate and nothing catches them individually from inside the
process.

COST
----
One ``traceback.extract_stack`` per pooled allocation.  MEASURED on an RTX
5090 by :func:`measure_ledger_overhead` (20,000 four-byte allocations, best
of three): 1.01 us per pooled allocation bare against 19.08 us armed, a
factor of 18.8.  That is fine for attribution -- the probe's slowest
configuration takes seconds -- and disqualifying inside a timed window, so
nothing in the benchmark path arms a ledger, and :func:`device_snapshot`
(four scalar reads, no hook) is what a timed lane should use if it wants a
number at all.  Re-run :func:`measure_ledger_overhead` rather than trusting
this line on other hardware.
"""

from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import os
import traceback

from cupy.cuda import memory_hook


__all__ = [
    "DeviceLedger",
    "LedgerMismatch",
    "device_snapshot",
    "format_sites",
    "format_snapshot",
    "trim_pool",
]


#: Repository root, used to turn absolute filenames into the short
#: ``gpuwm/core/rrtmgp.py`` form the reports are keyed by.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Frames inside these path fragments are never a useful attribution: they
#: are the allocator itself, or this ledger.  Attribution walks past them to
#: the deepest frame that is genuinely the caller's code.
_SKIP_FRAGMENTS = (
    os.path.join("site-packages", "cupy"),
    os.path.join("site-packages", "cupyx"),
    os.path.abspath(__file__),
)


class LedgerMismatch(RuntimeError):
    """The ledger's live sum disagrees with CuPy's own pool counter."""


def _short(filename: str) -> str:
    """``<any-absolute-tree>/gpuwm/core/rrtmgp.py`` -> ``gpuwm/core/rrtmgp.py``."""
    try:
        rel = os.path.relpath(filename, _REPO_ROOT)
    except ValueError:                        # different drive on Windows
        return filename
    return filename if rel.startswith(os.pardir) else rel


def _is_skipped(filename: str) -> bool:
    return any(fragment in filename for fragment in _SKIP_FRAGMENTS)


def device_snapshot() -> dict:
    """The four scalars, with no hook armed and no traceback walked.

    Cheap enough (four driver/pool queries) to call inside a loop.  Returns
    plain ints so a caller can subtract two snapshots.
    """
    import cupy as cp

    pool = cp.get_default_memory_pool()
    free, total = cp.cuda.runtime.memGetInfo()
    return {
        "pool_used": int(pool.used_bytes()),
        "pool_total": int(pool.total_bytes()),
        "device_used": int(total - free),
        "device_free": int(free),
        "device_total": int(total),
        "nonpool": int(total - free) - int(pool.total_bytes()),
    }


def trim_pool() -> None:
    """Return every free block to the driver, then synchronize.

    Called before and after a measurement so ``pool_total`` means "blocks
    that are actually in use or actively retained", not "every size class
    this process has ever touched".  The synchronize is not optional: a
    pending free is not visible to the pool until the stream it was freed on
    has been reached.
    """
    import cupy as cp

    cp.cuda.runtime.deviceSynchronize()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()


class DeviceLedger(memory_hook.MemoryHook):
    """Live pooled-allocation ledger, attributed to ``file:line``.

    Arm it as a context manager::

        with DeviceLedger() as ledger:
            state, driver = physics_inventory.default_builder(cfg)
            ledger.mark("state built")
            harness.run_steps(state, cfg, 1)
        print(ledger.report())

    ``mark`` records a named checkpoint of the four scalars plus the live
    total; ``phase`` tags every allocation made inside it, which is how the
    per-process / per-tile-buffer split is measured (build buffer one under
    one phase label and buffer two under another; whatever is charged only to
    the first phase is a one-time cost).

    Only allocations made while the ledger is armed are tracked.  Frees of
    blocks allocated BEFORE arming are counted separately as
    ``untracked_frees`` rather than being silently dropped, because a large
    untracked-free count means the caller armed the ledger too late and the
    live total is not the whole picture.
    """

    name = "TilestreamDeviceLedger"

    def __init__(self, *, chain_depth: int = 3, key_frames: int = 2) -> None:
        if chain_depth < 1:
            raise ValueError("chain_depth must be at least 1")
        if key_frames < 1:
            raise ValueError("key_frames must be at least 1")
        self._chain_depth = int(chain_depth)
        # Two frames, not one, because gpuwm routes most persistent
        # allocation through generic helpers: EVERY named scratch slot in
        # the process is allocated by ``state.py:822`` and every DomainState
        # field by ``state.py:381``.  Keying on the deepest frame alone put
        # 991 of 1300 resident MiB on one line and said nothing about which
        # slot or which caller -- measured at 128x128x49, full+MYNN+Noah-MP.
        self._key_frames = int(key_frames)
        #: pmem_id -> (bytes, site, phase)
        self._live: dict[int, tuple[int, str, str]] = {}
        self._live_by_site: dict[str, int] = defaultdict(int)
        self._live_by_phase: dict[str, int] = defaultdict(int)
        self._live_bytes = 0
        self._peak_bytes = 0
        self.malloc_bytes: dict[str, int] = defaultdict(int)
        self.malloc_calls: dict[str, int] = defaultdict(int)
        self.alloc_bytes: dict[str, int] = defaultdict(int)
        self.alloc_calls: dict[str, int] = defaultdict(int)
        self.peak_live_bytes: dict[str, int] = defaultdict(int)
        self.chains: dict[str, str] = {}
        self.marks: list[tuple[str, dict]] = []
        self.untracked_frees = 0
        self.untracked_free_bytes = 0
        self._phase = "unlabelled"
        self._armed = False

    # -- arming -----------------------------------------------------------

    def __enter__(self) -> "DeviceLedger":
        self._armed = True
        return super().__enter__()

    def __exit__(self, *exc) -> None:
        self._armed = False
        super().__exit__(*exc)

    @contextmanager
    def phase(self, label: str):
        """Tag every allocation made inside the block with ``label``."""
        previous = self._phase
        self._phase = str(label)
        try:
            yield self
        finally:
            self._phase = previous

    # -- callbacks --------------------------------------------------------

    def malloc_postprocess(self, **kw) -> None:
        site = self._site()
        size = int(kw["mem_size"])
        phase = self._phase
        self.malloc_bytes[site] += size
        self.malloc_calls[site] += 1
        self._live[int(kw["pmem_id"])] = (size, site, phase)
        self._live_bytes += size
        self._live_by_site[site] += size
        self._live_by_phase[phase] += size
        if self._live_bytes > self._peak_bytes:
            self._peak_bytes = self._live_bytes
        if self._live_by_site[site] > self.peak_live_bytes[site]:
            self.peak_live_bytes[site] = self._live_by_site[site]

    def free_postprocess(self, **kw) -> None:
        entry = self._live.pop(int(kw["pmem_id"]), None)
        if entry is None:
            # A block allocated before this ledger was armed.  Counted, not
            # ignored: a large count here means the live total below is only
            # part of the story and the caller armed the ledger too late.
            self.untracked_frees += 1
            self.untracked_free_bytes += int(kw["mem_size"])
            return
        size, site, phase = entry
        self._live_bytes -= size
        self._live_by_site[site] -= size
        self._live_by_phase[phase] -= size

    def alloc_postprocess(self, **kw) -> None:
        site = self._site()
        self.alloc_bytes[site] += int(kw["mem_size"])
        self.alloc_calls[site] += 1

    # -- attribution ------------------------------------------------------

    def _site(self) -> str:
        """Deepest stack frame that is not CuPy and not this module."""
        frames = traceback.extract_stack()
        useful = [f for f in frames if not _is_skipped(f.filename)]
        if not useful:
            return "<unattributed>"
        key = useful[-self._key_frames:]
        site = " <- ".join(f"{_short(f.filename)}:{f.lineno}"
                           for f in reversed(key))
        if site not in self.chains:
            tail = useful[-self._chain_depth:]
            self.chains[site] = " <- ".join(
                f"{_short(f.filename)}:{f.lineno}:{f.name}"
                for f in reversed(tail))
        return site

    # -- reading ----------------------------------------------------------

    @property
    def live_bytes(self) -> int:
        return self._live_bytes

    @property
    def peak_bytes(self) -> int:
        return self._peak_bytes

    def live_by_site(self) -> dict[str, int]:
        return {site: value for site, value in self._live_by_site.items()
                if value}

    def live_by_phase(self) -> dict[str, int]:
        return {phase: value for phase, value in self._live_by_phase.items()
                if value}

    def live_by_module(self) -> dict[str, int]:
        """Grouped by the file that ran the allocation."""
        out: dict[str, int] = defaultdict(int)
        for site, value in self._live_by_site.items():
            if value:
                out[site.split(":", 1)[0]] += value
        return dict(out)

    def live_by_owner(self) -> dict[str, int]:
        """Grouped by the OUTERMOST file in the key -- who asked for it.

        ``state.py`` allocates almost everything; this says on whose behalf.
        """
        out: dict[str, int] = defaultdict(int)
        for site, value in self._live_by_site.items():
            if value:
                out[site.rsplit("<- ", 1)[-1].split(":", 1)[0]] += value
        return dict(out)

    def mark(self, label: str) -> dict:
        """Record a named checkpoint; returns the snapshot it recorded."""
        snap = device_snapshot()
        snap["ledger_live"] = self._live_bytes
        snap["ledger_peak"] = self._peak_bytes
        self.marks.append((str(label), snap))
        return snap

    def check(self, *, tolerance: int = 0) -> None:
        """Raise unless the ledger's live sum equals the pool's.

        The two are computed by completely separate machinery -- this module
        counts hook callbacks, CuPy counts inside the pool -- so agreement is
        real evidence that nothing is being missed.  ``tolerance`` exists
        only for callers that armed the ledger after some allocation already
        existed; pass the known pre-existing byte count.
        """
        import cupy as cp

        pool_used = int(cp.get_default_memory_pool().used_bytes())
        if abs(pool_used - self._live_bytes - tolerance) > 0:
            raise LedgerMismatch(
                f"ledger live {self._live_bytes} + tolerance {tolerance} "
                f"!= pool used {pool_used} "
                f"(delta {pool_used - self._live_bytes - tolerance}); "
                f"{self.untracked_frees} untracked frees of "
                f"{self.untracked_free_bytes} bytes")

    def report(self, *, top: int = 20, chains: bool = True) -> str:
        lines = [format_sites(self.live_by_site(), title="LIVE NOW", top=top,
                              chains=self.chains if chains else None)]
        lines.append(format_sites(dict(self.alloc_bytes),
                                  title="GREW THE POOL (cudaMalloc)",
                                  top=top, chains=None))
        if self.marks:
            lines.append("MARKS")
            for label, snap in self.marks:
                lines.append(
                    f"  {label:<38s} live {snap['ledger_live'] / 2**20:9.1f} "
                    f"MiB  pool_used {snap['pool_used'] / 2**20:9.1f}  "
                    f"pool_total {snap['pool_total'] / 2**20:9.1f}  "
                    f"device {snap['device_used'] / 2**20:9.1f}")
        return "\n".join(lines)


def format_sites(sites: dict, *, title: str, top: int = 20,
                 chains: dict | None = None) -> str:
    """Render a ``{site: bytes}`` mapping largest-first."""
    total = sum(sites.values())
    lines = [f"{title}  ({total / 2**20:.1f} MiB over {len(sites)} sites)"]
    ordered = sorted(sites.items(), key=lambda kv: -kv[1])
    for site, value in ordered[:top]:
        share = 100.0 * value / total if total else 0.0
        lines.append(f"  {value / 2**20:10.2f} MiB  {share:5.1f}%  {site}")
        if chains is not None and site in chains:
            lines.append(f"                            {chains[site]}")
    if len(ordered) > top:
        rest = sum(v for _, v in ordered[top:])
        lines.append(f"  {rest / 2**20:10.2f} MiB  {'':5s}   "
                     f"... {len(ordered) - top} more sites")
    return "\n".join(lines)


def resident_inventory(root, *, max_depth: int = 5,
                       label: str = "state") -> dict[str, int]:
    """Every distinct device BLOCK reachable from ``root``, by attribute path.

    The traceback ledger says which SOURCE LINE allocated a byte; this says
    which OBJECT is still holding it, which is the half a reclamation plan
    actually needs.  Two rules make the total meaningful:

    * blocks, not arrays.  Views into a shared arena and slices of a parent
      array report their own ``nbytes``, and adding those up double-counts.
      Each distinct underlying ``Memory`` object is charged ONCE, to the
      first path that reached it, and its size is the BLOCK's size.
    * first path wins, and the walk is breadth-first, so a block held by
      both ``state._scratch['cu_nca']`` and ``state.physics.cu_nca`` is
      charged to the shorter path rather than to whichever the walk saw
      last.

    The returned total is therefore directly comparable with
    ``pool.used_bytes()`` minus whatever the caller holds outside ``root``.
    """
    import cupy as cp

    seen_blocks: set[int] = set()
    seen_objects: set[int] = set()
    out: dict[str, int] = {}
    queue = [(root, label, 0)]
    while queue:
        obj, path, depth = queue.pop(0)
        if depth > max_depth or id(obj) in seen_objects:
            continue
        seen_objects.add(id(obj))
        if isinstance(obj, cp.ndarray):
            block = obj.data.mem
            if id(block) not in seen_blocks:
                seen_blocks.add(id(block))
                out[path] = out.get(path, 0) + int(block.size)
            continue
        if isinstance(obj, dict):
            for key, value in obj.items():
                queue.append((value, f"{path}[{key!r}]", depth + 1))
            continue
        if isinstance(obj, (list, tuple, set)):
            for index, value in enumerate(obj):
                queue.append((value, f"{path}[{index}]", depth + 1))
            continue
        members = getattr(obj, "__dict__", None)
        if members is None:
            continue
        for name, value in list(members.items()):
            queue.append((value, f"{path}.{name}", depth + 1))
    return out


def group_inventory(inventory: dict[str, int], *, depth: int = 2
                    ) -> dict[str, int]:
    """Roll a :func:`resident_inventory` up to its first ``depth`` path parts."""
    out: dict[str, int] = defaultdict(int)
    for path, value in inventory.items():
        parts = path.replace("[", ".[").split(".")
        out[".".join(parts[:depth]).replace(".[", "[")] += value
    return dict(out)


def measure_ledger_overhead(n: int = 20_000) -> dict:
    """Microseconds per pooled allocation, armed against unarmed.

    Allocates and frees ``n`` tiny arrays twice.  Tiny on purpose: the point
    is to isolate the hook's fixed per-call cost, and a large allocation
    would bury it under the driver's.  Reported so the docstring above can
    quote a measurement rather than an impression.
    """
    import time

    import cupy as cp

    def loop() -> float:
        cp.cuda.runtime.deviceSynchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            cp.empty((4,), dtype=cp.float32)
        cp.cuda.runtime.deviceSynchronize()
        return (time.perf_counter() - t0) / n * 1e6

    cp.empty((4,), dtype=cp.float32)          # warm the size class
    bare = min(loop() for _ in range(3))
    with DeviceLedger():
        armed = min(loop() for _ in range(3))
    return {"bare_us": bare, "armed_us": armed,
            "ratio": armed / bare if bare else float("nan")}


def format_snapshot(snap: dict, label: str = "") -> str:
    return (f"{label:<28s} pool_used {snap['pool_used'] / 2**30:7.3f} GiB  "
            f"pool_total {snap['pool_total'] / 2**30:7.3f}  "
            f"device_used {snap['device_used'] / 2**30:7.3f}  "
            f"nonpool {snap['nonpool'] / 2**30:7.3f}")
