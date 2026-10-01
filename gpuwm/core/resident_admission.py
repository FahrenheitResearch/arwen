"""A domain held whole on the card is admitted, or refused by name, first.

THE BREAKAGE THIS PREVENTS.  With no ``[tiles]`` block a domain is resident,
and the resident road had no admission at all: the reload doors and the
public loaders (:func:`gpuwm.ingest.prepared_cache.restore_prepared_cache`,
the native wrfinput restore, :func:`gpuwm.core.state.init_at_rest`) went
straight into ``DomainState``.  A 1792x1024x55 mp=18 prepared cache, or an
mp=8 one retaining 74 eager boundary intervals (25,782,246,220 bytes of state
and boundary tables before any physics), stopped in a CUDA out-of-memory part
way through its constructor instead of being refused by name.  Whether a
domain streams and whether it fits are two questions; ``mode = 'off'``
answers the first, and this module answers the second for the resident road.

Two figures, asked in two places:

* the CONSTRUCTOR FLOOR (:func:`admit_construction`): the exact
  ``DomainState`` inventory plus the lateral boundary tables the loader
  attaches, against what this process can still allocate on its card.
  Every byte of it is allocated by the loader that asks, so the floor
  never refuses a construction that would have completed.
* the FORECAST (:func:`gpuwm.core.streaming.admit_resident_road`): state,
  physics, boundary tables, workspaces and context -- the peak envelope
  every review prices -- taken at the door before anything is restored.

The two are not the same kind of figure.  The floor is exact.  The envelope
is the measured UPPER bound of the peak, and it sits above the true peak by
design (a 552x552x49 child on a 5070 Ti card: envelope 14,922,267,728 B,
pool peak 12,428,445,696 B), so a forecast inside that margin fits a card
the envelope says it does not.  ``gpuwm go`` has always let a reader skip its
own envelope gate for exactly that reason; ``--no-memory-gate`` at any door
(:data:`MEMORY_GATE_OVERRIDE_ENV` below it) skips the forecast envelope here
too, and the floor still refuses a state that cannot be built at all.

A card that cannot be read never refuses: the answer is then ``None`` and
the construction proceeds as it always did.

Module scope is the standard library and the refusal class; CuPy and the
estimators are imported where they are used.
"""
from __future__ import annotations

import contextlib
import math
import os
import sys

from gpuwm.ingest.memory_refusal import ResidentMemoryRefused

GIB = 1024 ** 3

#: The switch every door's ``--no-memory-gate`` sets, read where an ENVELOPE
#: admission is taken (never where a constructor floor is).
#:
#: An environment variable rather than an argument because those admissions
#: sit one or more processes below the door that was typed: ``gpuwm go``
#: runs its forecast stage as a subprocess or hosts it, ``gpuwm run`` runs a
#: supervised worker process, and the remote worker reaches both through
#: ``gpuwm go --no-memory-gate``.  Every one of those inherits this
#: environment, the way the checkpoint retention already reaches the same
#: runners (:data:`gpuwm.resume.KEEP_CHECKPOINTS_ENV`).  Without it the
#: override stopped at go's own gate, and the runner go launched refused the
#: same envelope again after the fetch and the preparation.
MEMORY_GATE_OVERRIDE_ENV = "GPUWM_NO_MEMORY_GATE"

#: How an envelope refusal names the way past it, at every door.
MEMORY_GATE_OVERRIDE_HINT = (
    f"--no-memory-gate (or {MEMORY_GATE_OVERRIDE_ENV}=1) runs it anyway, "
    "because this figure is a priced upper bound and the card's own "
    "allocation then decides")


def memory_gate_overridden() -> bool:
    """Has the door that launched this process said ``--no-memory-gate``?"""
    value = os.environ.get(MEMORY_GATE_OVERRIDE_ENV, "")
    return value.strip().lower() in ("1", "true", "yes", "on")


@contextlib.contextmanager
def memory_gate_override(enabled):
    """Hold :data:`MEMORY_GATE_OVERRIDE_ENV` on while a door runs, then put back.

    ``enabled`` false changes nothing, so a door can wrap itself
    unconditionally.  The previous value is restored on the way out, so a
    process that hosts a door (the remote worker, a GUI) does not carry one
    job's override into the next.
    """
    if not enabled:
        yield
        return
    previous = os.environ.get(MEMORY_GATE_OVERRIDE_ENV)
    os.environ[MEMORY_GATE_OVERRIDE_ENV] = "1"
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(MEMORY_GATE_OVERRIDE_ENV, None)
        else:
            os.environ[MEMORY_GATE_OVERRIDE_ENV] = previous

#: What a resident refusal offers, in the words of the settings that change
#: the answer.  ``[tiles] mode = 'auto'`` is the road that exists for a
#: domain the card cannot hold whole.
RESIDENT_REMEDY = (
    "run the domain with [tiles] mode = 'auto' (or --tiles auto), which "
    "streams a domain the card cannot hold through host memory; free card "
    "memory (close other programs using the GPU); or use a smaller domain "
    "or fewer levels")


def _gib(value) -> str:
    return f"{int(value) / GIB:.2f} GiB"


def terms_text(terms) -> str:
    """``model state 19.70 GiB, physics 7.80 GiB`` -- the non-zero terms."""
    return ", ".join(f"{name} {_gib(value)}"
                     for name, value in dict(terms).items() if int(value))


def device_free_bytes() -> int | None:
    """What this process can still allocate on its current card, or ``None``.

    ``cudaMemGetInfo``'s free plus the bytes the default pool holds unused
    (a freed block is reusable without the driver).  Under a private bounded
    pool -- a reconstruction reservation routing this thread's allocations
    through its own arena -- the answer is that pool's remaining limit,
    because every allocation made now is drawn from it and nowhere else.
    ``None`` when there is no card to ask, which admits.
    """
    try:
        import cupy as cp
    except Exception:                         # noqa: BLE001 - no runtime
        return None
    try:
        allocator = cp.cuda.get_allocator()
        owner = getattr(allocator, "__self__", None)
        private = getattr(owner, "pool", None)
        limit = (int(private.get_limit())
                 if private is not None and hasattr(private, "get_limit")
                 else 0)
        if limit > 0:
            return max(0, limit - int(private.used_bytes()))
        free, _total = cp.cuda.runtime.memGetInfo()
        return int(free) + int(cp.get_default_memory_pool().free_bytes())
    except Exception:                         # noqa: BLE001 - unread card
        return None


def admit(what: str, terms, *, free_bytes=None, stage: str,
          remedy: str = RESIDENT_REMEDY, card: str | None = None,
          envelope: bool = False):
    """Weigh ``terms`` against the card's free memory; refuse or record.

    ``free_bytes`` is a reading already taken (a door's cold card); left
    ``None`` it is :func:`device_free_bytes` now.  Returns the admission
    record, with ``fits`` ``None`` when the card could not be read.  Raises
    :class:`ResidentMemoryRefused` naming the terms, the free figure, the
    out-of-memory it prevents and the remedy.

    ``envelope`` says ``terms`` are the measured UPPER bound of a peak
    rather than bytes the caller is about to allocate.  Such a refusal says
    the out-of-memory is expected rather than certain, names
    ``--no-memory-gate``, and is skipped under it
    (:func:`memory_gate_overridden`): the record then carries
    ``overridden`` and one warning line says what was skipped.  A floor
    (``envelope`` false) refuses whatever the override says, because its
    bytes are allocated by the very call that asks.
    """
    terms = {str(name): int(value) for name, value in dict(terms).items()}
    need = sum(terms.values())
    if free_bytes is None:
        free_bytes = device_free_bytes()
    record = {"what": what, "need_bytes": need, "terms": terms,
              "free_bytes": None if free_bytes is None else int(free_bytes),
              "fits": None if free_bytes is None else need <= int(free_bytes)}
    if free_bytes is None or need <= int(free_bytes):
        return record
    where = f"the card has {_gib(free_bytes)} free" if card is None else card
    if envelope and memory_gate_overridden():
        print(f"warning: {what}: its peak envelope is {_gib(need)} on the "
              f"card ({terms_text(terms)}) and {where}; --no-memory-gate "
              "skips that check, so the run proceeds and the card's own "
              "allocation decides", file=sys.stderr, flush=True)
        return dict(record, overridden=True)
    if envelope:
        body = (f"Its peak envelope is {_gib(need)} on the card "
                f"({terms_text(terms)}) and {where}.  Started, it is "
                f"expected to stop with a CUDA out-of-memory {stage}.\n"
                f"  remedy: {remedy}; or, when you know its true peak "
                f"fits: {MEMORY_GATE_OVERRIDE_HINT}, and a model state that "
                "cannot be built at all is still refused by name")
    else:
        body = (f"It needs {_gib(need)} on the card ({terms_text(terms)}) "
                f"and {where}.  Started, it would stop with a CUDA "
                f"out-of-memory {stage}.\n  remedy: {remedy}")
    raise ResidentMemoryRefused(
        f"{what}: refused before anything was allocated.  {body}",
        need_bytes=need, free_bytes=free_bytes, terms=terms, what=what)


def resident_forecast_terms(estimate) -> dict:
    """A resident forecast's envelope, as the terms a refusal names.

    Model state, physics and the lateral boundary tables as the itemized
    estimate carries them, and the rest of
    :attr:`~gpuwm.core.preflight.ExperimentMemoryEstimate.peak_envelope_bytes`
    -- shared workspaces, step transients, the CUDA context and the
    allocator headroom -- as one remainder, so the terms add up to the
    envelope every review prices.
    """
    domains = tuple(getattr(estimate, "domains", ()) or ())
    state = sum(int(d.category_bytes("state")) for d in domains)
    physics = sum(int(d.category_bytes("physics")) for d in domains)
    tables = sum(int(d.category_bytes("lbc")) for d in domains)
    total = int(estimate.peak_envelope_bytes)
    return {
        "model state": state, "physics": physics,
        "lateral boundary tables": tables,
        "workspaces, step transients, CUDA context and allocator headroom":
            max(0, total - state - physics - tables),
    }


def state_bytes(cfg, *, shared_symbols=()) -> int:
    """Bytes ``DomainState(cfg)`` allocates on the card.

    The exact constructor inventory
    (:func:`gpuwm.core.device_inventory.state_array_shapes`, cross-checked
    against the constructor by test), float32.  ``shared_symbols`` are the
    arrays a shared dycore workspace hands the constructor as views of its
    own backing, which the constructor therefore does not allocate.
    """
    from gpuwm.core.device_inventory import state_array_shapes

    shared = frozenset(shared_symbols)
    return sum(4 * math.prod(shape)
               for name, shape in state_array_shapes(cfg).items()
               if name not in shared)


def admit_construction(what: str, cfg, *, boundary_values: int = 0,
                       shared_symbols=(), free_bytes=None,
                       remedy: str = RESIDENT_REMEDY):
    """The constructor floor: state plus the boundary tables, before either.

    ``boundary_values`` is the count of float32 values the loader attaches
    as lateral boundary tables on the card (value, tendency and any time-law
    coefficients, every retained interval).
    """
    terms = {"model state": state_bytes(cfg, shared_symbols=shared_symbols)}
    if boundary_values:
        terms["lateral boundary tables"] = 4 * int(boundary_values)
    return admit(what, terms, free_bytes=free_bytes,
                 stage=(f"while building the {int(cfg.nx)}x{int(cfg.ny)}x"
                        f"{int(cfg.nz)} domain state"),
                 remedy=remedy)


def slab_device_peak_bytes(cfg, rows: int, *, p_top: float,
                           column_chunk: int | None = None) -> int:
    """One loader slab's device peak: state, physics, carriers, transients.

    The figure the tile planner already charges the store loader
    (:meth:`gpuwm.core.prepared_tile_memory.PreparedTileMemory.fixed_terms`,
    ``loader_pool_peak_bytes``): the itemized domain estimate at slab height
    with no boundary tables, because the store road keeps every forcing
    interval on the host.
    """
    from dataclasses import replace
    from types import SimpleNamespace

    from gpuwm.config import DEFAULT_COLUMN_CHUNK
    from gpuwm.core.preflight import estimate_domain

    run = replace(cfg, ny=int(rows))
    estimate = estimate_domain(
        SimpleNamespace(run=run, grid_id=int(cfg.grid_id), parent_id=0),
        n_lbc_intervals=0, p_top=float(p_top),
        column_chunk=int(column_chunk or DEFAULT_COLUMN_CHUNK))
    return int(estimate.resident_bytes + estimate.transient_bytes)


def admitted_slab_rows(cfg, rows: int, *, p_top: float, log=print,
                       free_bytes=None) -> int:
    """The slab height a store load may use on this card, before its first slab.

    The requested height when its slab fits; otherwise the tallest slab that
    does, said in one line (the slab height partitions the same cache bytes
    and changes no operand); a refusal by name when not even a one-row slab
    fits.  An unread card keeps the request.
    """
    rows = max(1, int(rows))
    if free_bytes is None:
        free_bytes = device_free_bytes()
    if free_bytes is None:
        return rows
    free = int(free_bytes)

    def price(height):
        return slab_device_peak_bytes(cfg, height, p_top=p_top)

    need = price(rows)
    if need <= free:
        return rows
    admit(f"a store load in {rows}-row slabs",
          {"one row slab (state, physics and carriers)": price(1)},
          free_bytes=free,
          stage="while building the first slab state",
          remedy=("free card memory (close other programs using the GPU) "
                  "or use a larger card"))
    lo, hi = 1, rows - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if price(mid) <= free:
            lo = mid
        else:
            hi = mid - 1
    log(f"    store: a {rows}-row slab needs {_gib(need)} on the card "
        f"(state, physics and carriers) and the card has {_gib(free)} "
        f"free, so the cache loads in {lo}-row slabs; the slab height "
        "partitions the same cache bytes and changes no operand")
    return lo


__all__ = [
    "MEMORY_GATE_OVERRIDE_ENV", "MEMORY_GATE_OVERRIDE_HINT",
    "RESIDENT_REMEDY", "ResidentMemoryRefused", "admit",
    "admit_construction", "admitted_slab_rows", "device_free_bytes",
    "memory_gate_override", "memory_gate_overridden",
    "resident_forecast_terms", "slab_device_peak_bytes", "state_bytes",
    "terms_text",
]
