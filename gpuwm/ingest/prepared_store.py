"""The prepared cache, loaded straight into a pinned host store.

The resident road -- :func:`gpuwm.ingest.prepared_cache.restore_prepared_cache`
followed by :func:`gpuwm.ingest.hrrr_physics.initialize_prepared_physics` --
builds ONE full-domain ``DomainState`` on the card and then lets
:func:`gpuwm.core.streaming.attach` copy it, carrier by carrier, into pinned
host RAM.  Every byte of that state is dead the moment the copy finishes: a
streamed forecast integrates out of the store and never reads it again.  It is
also the ceiling.  MEASURED on a 16 GB card, nz = 49: the bare state costs
11 276.5 B per column and the prepared case about 15 780, so 1024 x 1024
refuses inside ``initialize_physics`` while the forecast it would have fed
needs about 6 GiB and would have run comfortably.  The streamed engine on the
same card has stepped 2624 x 2624.

This module removes the intermediary.  It reads the cache one ROW SLAB at a
time, builds a slab-height state, attaches physics to THAT, and scatters the
result into full-domain pinned host arrays -- so the largest device
allocation in a preparation is one slab, and the domain the card can carry
stops being a property of the card at all.

:func:`tilestream.bigdomain.build_store_by_slabs` already does exactly this
for an analytic initial condition, and this is deliberately its structure:
price the manifest, refuse before the first pinned byte, then fill.  The two
differences are both about where the rows come from.

*The rows come off disk, not out of a sounding.*  The cache's integrity
guarantee is a SHA-256 over each array WHOLE, so it is spent first and
separately: one streaming pass re-hashes every array through
``PreparedCacheReader.read_array`` and drops it, proving the bundle for the
cost of one array's memory.  Rows are then served from read-only memory maps,
so what stays resident is page cache the kernel may evict rather than a
process that has swallowed its own input.  Neither pass puts anything on the
device at domain shape.  See :class:`_CachePayload` -- holding the payload
whole would have cost about 11 GiB of prognostics at 1024 x 1024 x 49 on top
of the 9 GiB store, and multiplied on every stretch rung above it.

*Physics is attached per slab.*  The cache carries
``STATE_SERIALIZED_ATTRS`` only -- the dynamics and microphysics prognostics.
Everything the store also needs (``fields/*``, ``driver/*``, ``cumulus/*``,
``scratch/*``) is made by ``initialize_prepared_physics``, which is
column-local given windowed inputs: its one apparent reduction,
``GREENFRAC.min(axis=0)``, is over the twelve MONTHS and not over space.  The
two places a slab could disagree with the domain are named and handled --
``center_lat`` is passed down explicitly rather than re-derived from the
slab's own grid (``initialize_landuse`` reads it for the LANDUSE.TBL season),
and the y-staggered inputs take one extra row so the ``V10`` face average at
a slab's last row sees the same neighbour the whole domain would.

That the two roads agree is not argued from this docstring: a domain small
enough for both produces bit-identical frames, which is the parity gate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields, is_dataclass, replace
from types import MappingProxyType, SimpleNamespace

import numpy as np

__all__ = [
    "PreparedStore",
    "PreparedStoreError",
    "store_from_prepared_cache",
]


class PreparedStoreError(RuntimeError):
    """A prepared cache could not be loaded into a host store."""


@dataclass(frozen=True)
class PreparedStore:
    """What a streamed run needs, with no resident domain state in it.

    ``store`` and ``geography`` are dicts of pinned full-domain host arrays,
    keyed exactly as this run's own ``inventory_fn`` -- by default
    :func:`gpuwm.core.streaming.streamed_store_inventory`, the rule the tile
    buffers are built by -- and :func:`tilestream.driver.geography_inventory`
    key them, which is what :class:`tilestream.driver.TiledRun` requires of
    both.

    ``base`` is the WHOLE domain's base state, and the word to hold onto is
    DOMAIN-SHAPED rather than domain-invariant.  Flat terrain makes it look
    invariant -- :class:`gpuwm.core.grid.BaseState` is then a scalar ``mub``
    and 1-D profiles that broadcast against any height -- but with terrain
    ``mub`` is ``(ny, nx)`` and ``pb``/``alb``/``thb``/``phb`` are full 3-D
    ``(nz[,+1], ny, nx)`` arrays.  Every slab is therefore built against a
    ROW-WINDOWED copy (:func:`_slab_base`), while the bundle carries the
    un-windowed original: a consumer holding this object wants the DOMAIN's
    base state, not whichever slab happened to be built last.

    ``template`` is a SLAB-HEIGHT state, not a domain-shaped one.  It exists
    because :func:`gpuwm.core.streaming.prepared_tile_state_factory` reads
    three things off "the domain": its scheme adapters (to twin per buffer),
    its vertical coordinate, and the ``ndim < 2`` half of its setup arrays.
    All three are identical at every height -- the eta table does not vary by
    row -- so a slab carries them faithfully and costs a fraction of a
    domain.  The horizontally-varying setup arrays are NOT taken from it;
    those are gathered per tile out of ``geography``.
    """

    store: dict
    geography: dict
    scalars: dict
    template: object
    coord: object
    base: object
    boundaries: object
    missing: tuple
    receipt: object


def _row_window(arr, j0: int, rows: int, ny: int):
    """``rows`` rows of ``arr`` at ``j0``, one more if it is y-staggered.

    The horizontal axes are the last two -- the rule the whole store is
    derived from (``hoststore.manifest_from_arrays``).  A field whose
    second-to-last extent is ``ny + 1`` carries the closing face, so slab k's
    last row IS slab k+1's first; both are the same number, so the overlap is
    a duplicate rather than a conflict.  Anything else is refused loudly:
    silently windowing an array of unexpected height is how a slab ends up
    holding a shifted copy of the domain.
    """
    if arr is None:
        return None
    if isinstance(arr, (str, bytes, bool, int, float)):
        # A static bundle carries scalars and identity strings beside its
        # fields (MMINLU and friends).  Windowing is a horizontal operation
        # and these have no horizontal extent; passing them through
        # unchanged is what keeps a slab's static mapping the same KIND of
        # object the whole-domain one is, rather than a 0-d array that the
        # consumer will silently treat as a number.
        return arr
    array = np.asarray(arr)
    if array.ndim < 2:
        return array
    height = int(array.shape[-2])
    if height == int(ny):
        take = int(rows)
    elif height == int(ny) + 1:
        take = int(rows) + 1
    else:
        raise PreparedStoreError(
            f"cannot window an array of height {height} against a domain of "
            f"{ny} mass rows: it is neither mass- nor y-staggered, so which "
            "rows belong to a slab is undefined")
    return array[..., j0:j0 + take, :]


def _window_mapping(mapping, j0: int, rows: int, ny: int):
    """Every array in a static/surface/met mapping, windowed by rows."""
    if mapping is None:
        return None
    return {name: _row_window(value, j0, rows, ny)
            for name, value in mapping.items()}


def _slab_base(base, j0: int, rows: int, ny: int):
    """``base`` with every ARRAY field row-windowed, its scalars untouched.

    THE BASE STATE IS NOT DOMAIN-INVARIANT, and believing that it was is the
    defect this function exists to remove.  Flat terrain makes it look
    invariant: :class:`gpuwm.core.grid.BaseState` is then a SCALAR ``mub``
    and four 1-D columns that broadcast against ``(nz, ny, nx)`` at any
    height, so a slab handed the domain's base gets the right numbers by
    accident.  WITH TERRAIN -- which is every real case -- ``mub`` is
    ``(ny, nx)``, ``pb``/``alb``/``thb`` are ``(nz, ny, nx)``, ``phb`` is
    ``(nz + 1, ny, nx)`` and ``terrain_z`` is ``(ny, nx)``;
    :meth:`gpuwm.core.state.DomainState.load_base` assigns each into a
    SLAB-shaped device array with ``dev[...] = ...``, which broadcasts only
    while the heights agree and refuses the moment they do not.  MEASURED on
    the parity gate at 384 x 384 with 64-row slabs, three times on real
    data: ``ValueError: operands could not be broadcast together with shapes
    (49, 384, 384) (49, 64, 384)``.  The store's own unit tests missed it
    because they use a flat base and an ``ny`` that fits in one slab, and a
    one-slab domain-height array broadcasts.

    The field list is DERIVED, from :func:`dataclasses.fields`, so a field
    added to ``BaseState`` is windowed the day it is added rather than being
    handed to every slab whole -- the same failure again, in a new array,
    with the unit tests still green.  Each field goes through
    :func:`_row_window`, so the mass/y-staggered rule and its loud refusal
    are the store's ONE rule rather than a second copy of it that can drift;
    ``p_top``, and ``mub`` when the terrain is flat, fall through it as
    themselves, which is what keeps a flat-terrain base exactly the object
    it was.
    """
    if not is_dataclass(base):
        raise PreparedStoreError(
            f"the prepared cache's base state is a {type(base).__name__}, "
            "which is not a dataclass: the per-slab window is derived from "
            "dataclasses.fields precisely so that no field can be missed, "
            "and an object it cannot enumerate is one it cannot window "
            "safely")
    return replace(base, **{
        spec.name: _row_window(getattr(base, spec.name), j0, rows, ny)
        for spec in fields(base) if spec.init})


def _as_host(arr) -> np.ndarray:
    import cupy as cp

    return cp.asnumpy(arr) if isinstance(arr, cp.ndarray) else np.asarray(arr)


def _scatter_rows(dst: np.ndarray, src, j0: int) -> None:
    """Write ``src``'s rows into ``dst`` at ``j0``, inferring the staggering.

    :func:`tilestream.bigdomain._scatter_rows`' rule, and for the same
    reason: the destination row span is the source's own second-to-last
    extent, so a v-staggered field writes ``rows + 1`` rows.
    """
    host = _as_host(src)
    dst[..., j0:j0 + host.shape[-2], :] = host


def _seam_disagreement(dst: np.ndarray, src: np.ndarray, j0: int,
                       rows: int) -> int:
    """Differing values on a y-staggered carrier's SHARED face.

    A y-staggered field's first row at ``j0`` is the same face as the
    previous slab's last row, so it is written twice.  For anything COPIED
    out of the cache that is a duplicate and harmless: both slabs read the
    same number.  For anything a slab COMPUTES it need not be, and the
    failure is specific and nasty -- slab k writes that face as its own
    domain-TOP edge, slab k+1 overwrites it as its own domain-BOTTOM edge,
    and neither is the interior value a whole domain would have produced.
    That is exactly the signature ``tilestream.test_realdata`` records for
    ``y_halo = 0``: one field, wrong in ``nslabs - 1`` rows, by an amount
    small enough to look like weather.

    So the overlap is not assumed to be a duplicate, it is COUNTED, over
    raw bits so that a signed zero or a NaN pair is judged the way the
    parity gate will judge it.  Returns how many values on the shared face
    the second writer would change.
    """
    # ONLY a y-staggered carrier has a shared face.  A mass-height field's
    # row j0 is the first row THIS slab owns and nobody has written it yet,
    # so comparing it against whatever the buffer happens to hold reports a
    # disagreement that does not exist -- measured on a zero-filled store,
    # every mass field scored a full row of false positives.  The staggering
    # is inferred the way the rest of this module infers it: from the
    # source's own height against the rows the slab owns.
    if j0 <= 0 or int(src.shape[-2]) != int(rows) + 1:
        return 0
    existing = dst[..., j0, :]
    incoming = src[..., 0, :]
    if existing.shape != incoming.shape or existing.dtype != incoming.dtype:
        return 0
    if existing.dtype.kind not in ("f", "i", "u"):
        return 0
    width = existing.dtype.itemsize
    return int(np.count_nonzero(
        np.ascontiguousarray(existing).view(f"u{width}")
        != np.ascontiguousarray(incoming).view(f"u{width}")))


def _plan_slabs(ny: int, rows_per_slab: int) -> tuple[tuple[int, int], ...]:
    """``(j0, rows)`` pairs partitioning ``[0, ny)`` exactly.

    A partition, not an overlay: every mass row is written by exactly one
    slab.  The last slab is short rather than the loop refusing an ny that
    ``rows_per_slab`` does not divide -- ``build_store_by_slabs`` may demand
    divisibility because it also sizes an analytic noise field by it, and
    nothing here does.
    """
    ny = int(ny)
    rows_per_slab = max(1, int(rows_per_slab))
    if ny < 1:
        raise PreparedStoreError(f"a domain of {ny} rows cannot be slabbed")
    out = []
    j0 = 0
    while j0 < ny:
        out.append((j0, min(rows_per_slab, ny - j0)))
        j0 += rows_per_slab
    return tuple(out)


def _slab_state(cfg_slab, coord, base_slab, static_slab, cache_rows,
                state_names):
    """One slab-height state, carrying the cache's own rows.

    The construction is ``restore_prepared_cache``'s, at slab height:
    ``DomainState`` for the shapes, ``load_base`` for the vertical
    coordinate and the base state, and ``set_map_coriolis`` for the slab's
    own map factors.  Then the prognostics, which is the only part that
    reads the cache.

    THE TWO ARGUMENTS ARE NOT THE SAME KIND OF THING, and an earlier version
    of this docstring said they were.  ``coord`` is the DOMAIN's and is
    passed whole because the eta table has no horizontal extent -- it is
    identical at every row, at any slab height.  ``base_slab`` is the
    domain's base state ALREADY WINDOWED to this slab's rows by
    :func:`_slab_base`, because under terrain the base carries an
    ``(ny, nx)`` dry mass and full 3-D profiles.  The window is the caller's
    to make rather than this function's so that the SAME windowed object
    also reaches ``CachedInitialResult`` -- the state and the physics have
    to agree about which rows the base describes, and building it twice is
    how they would stop agreeing.
    """
    import cupy as cp

    from gpuwm.core.state import DomainState

    state = DomainState(cfg_slab)
    state.load_base(coord, base_slab)
    state.set_map_coriolis(
        static_slab["MAPFAC_M"], static_slab["MAPFAC_U"],
        static_slab["MAPFAC_V"], static_slab["F"], static_slab["E"],
        sina=static_slab["SINALPHA"], cosa=static_slab["COSALPHA"])
    for name in state_names:
        target = getattr(state, name, None)
        if target is None:
            raise PreparedStoreError(
                f"the cache carries state/{name} but a slab-height state has "
                "no such array; the active config does not match the cache")
        rows = cache_rows(name)
        if tuple(rows.shape) != tuple(target.shape) \
                or rows.dtype != target.dtype:
            raise PreparedStoreError(
                f"prepared cache state/{name} windowed to {rows.shape} "
                f"{rows.dtype} against a slab expecting {target.shape} "
                f"{target.dtype}")
        # ascontiguousarray, not asarray: a row window of a memory-mapped
        # array is a strided VIEW, and a host->device copy of one is either
        # a silent extra copy or a refusal depending on the backend's mood.
        # Made explicit so the cost is visible and the behaviour is not a
        # property of which CuPy is installed.
        target[...] = cp.asarray(np.ascontiguousarray(rows))
    return state


class _CachePayload:
    """Row-windowed access to a prepared cache, verified once, mapped after.

    Two requirements pull against each other here.  The cache's integrity
    guarantee is a SHA-256 taken over each array WHOLE, so a windowed read
    can never check itself; but holding every array whole is the thing this
    module exists to stop doing -- at 1024 x 1024 x 49 the prognostics alone
    are about 11 GiB, and the pinned store is another 9, and the stretch
    rungs multiply both.

    So the payload is read twice and held never.  ``verify`` walks every
    array through :meth:`PreparedCacheReader.read_array`, which re-hashes it
    and drops it, so the whole bundle is proven against its own manifest for
    the cost of one streaming pass and one array's worth of memory.  After
    that, rows are served from a read-only ``np.load(mmap_mode="r")``: the
    kernel pages in the rows a slab touches and evicts them under pressure,
    so the resident cost is the page cache's problem rather than the
    process's.  The shape and dtype of every map are still checked against
    the manifest at open time, because a mapping that silently disagreed
    with the header is exactly what the digest pass would no longer be
    covering.
    """

    def __init__(self, reader, *, verify: bool = True, log=print):
        self._reader = reader
        self._maps: dict = {}
        if verify:
            t0 = time.perf_counter()
            for key in sorted(reader.arrays):
                reader.read_array(key)
            log(f"    cache payload verified against its manifest: "
                f"{len(reader.arrays)} arrays, "
                f"{reader.payload_bytes / hoststore_gib():.2f} GiB, "
                f"in {time.perf_counter() - t0:.1f}s")

    def __getitem__(self, key: str):
        mapped = self._maps.get(key)
        if mapped is None:
            try:
                spec = self._reader.arrays[key]
            except KeyError as exc:
                raise PreparedStoreError(
                    f"prepared cache is missing array {key!r}") from exc
            mapped = np.load(self._reader.path / spec["file"],
                             mmap_mode="r", allow_pickle=False)
            if list(mapped.shape) != list(spec["shape"]) \
                    or str(mapped.dtype) != str(spec["dtype"]):
                raise PreparedStoreError(
                    f"prepared cache array {key!r} maps as {mapped.shape} "
                    f"{mapped.dtype} against a manifest declaring "
                    f"{spec['shape']} {spec['dtype']}")
            self._maps[key] = mapped
        return mapped

    def close(self) -> None:
        self._maps.clear()


def hoststore_gib() -> int:
    from tilestream import hoststore

    return hoststore.GIB


def _boundaries_from_cache(reader, metadata):
    """The domain's ``LateralBoundaries``, on the HOST and attached to nothing.

    ``restore_prepared_cache`` calls ``attach_lateral_boundaries(state, ...)``
    because it has a state to attach them to.  A streamed run does not: the
    tables are windowed per tile by
    :func:`gpuwm.core.streaming.tile_boundary_tables` and each buffer holds
    only its own edge, which is the single-slot streaming bind.  So they are
    returned as they come off disk -- the whole series, host-side -- and the
    windowing decides what ever reaches the card.
    """
    from gpuwm.ingest.lateral_bc import (BoundaryInterval, FieldBoundary,
                                         LateralBoundaries, SideBoundary)

    lbc = metadata["lbc"]
    intervals = []
    for index, interval_meta in enumerate(lbc["intervals"]):
        field_map = {}
        for name in interval_meta["fields"]:
            sides = {}
            for side_name in ("west", "east", "south", "north"):
                prefix = f"lbc/{index}/{name}/{side_name}"
                sides[side_name] = SideBoundary(
                    reader.read_array(f"{prefix}/value"),
                    reader.read_array(f"{prefix}/tendency"))
            field_map[name] = FieldBoundary(**sides)
        intervals.append(BoundaryInterval(
            float(interval_meta["start_seconds"]),
            float(interval_meta["end_seconds"]), field_map))
    return LateralBoundaries(
        tuple(intervals), int(lbc["spec_bdy_width"]), int(lbc["spec_zone"]),
        int(lbc["relax_zone"]))


def store_from_prepared_cache(path, *, expected_identity, cfg, static,
                              landuse_attrs, grid, valid_time,
                              rows_per_slab: int = 64,
                              budget_bytes: int | None = None,
                              verify_payload: bool = True,
                              center_lat=None, constant_glw_wm2=None,
                              inventory_fn=None,
                              log=print) -> PreparedStore:
    """Load a prepared cache into pinned host arrays, slab by slab.

    No domain-shaped device array is ever allocated.  The peak device
    residency is one slab's state plus its physics; the peak HOST transient
    beyond the store itself is one full-domain array, because that is the
    unit ``PreparedCacheReader`` can verify against its digest.

    The budget guard is ``build_store_by_slabs``' and runs in the same place:
    the whole request is priced from the manifest the first slab reveals, and
    :func:`tilestream.hoststore.check_allocatable` refuses BEFORE the first
    ``alloc_pinned_array``.  A guard that fires on the way up has already
    taken most of what it is refusing, and pinned pages cannot be swapped.

    ``inventory_fn`` IS THE RULE THE TILE BUFFERS ARE BUILT BY, and it is a
    parameter rather than a literal so that the store and the buffers cannot
    be given two different ones.  It defaults to
    :func:`gpuwm.core.streaming.streamed_store_inventory` -- the exact
    callable ``store_domain_builder`` and ``prepared_domain_builder`` hand
    :func:`gpuwm.core.streaming.attach` -- and :class:`tilestream.driver.
    TiledRun` compares the two inventories key for key before it steps.
    Harvesting each slab with ``physics_inventory.carrier_inventory``
    instead cost the store exactly one carrier: ``scratch/refl_10cm`` is
    REBUILT scratch, excluded from the plain manifest BY CONSTRUCTION, so
    ``prime_lazy_carriers`` allocated it on every slab and nothing collected
    it.  MEASURED, before the first step: ``TiledRunError: tile state
    inventory [143] != store inventory [142] ... in TILE not in STORE:
    ['scratch/refl_10cm']``.
    """
    import cupy as cp

    from gpuwm.core.streaming import (prime_lazy_carriers,
                                      streamed_store_inventory)
    from gpuwm.ingest.hrrr_physics import initialize_prepared_physics
    from gpuwm.ingest.prepared_cache import (CachedInitialResult,
                                             PreparedCacheReader)
    from tilestream import driver as tdriver
    from tilestream import hoststore
    from tilestream import physics_inventory as physinv
    from tilestream import realdata as _realdata

    nz, ny, nx = int(cfg.nz), int(cfg.ny), int(cfg.nx)
    reader = PreparedCacheReader(path, expected_identity=expected_identity)
    metadata = reader.header["metadata"]

    from gpuwm.core.grid import BaseState, VerticalCoord

    coord_values = dict(metadata["coord_scalars"])
    for name in metadata["coord_arrays"]:
        coord_values[name] = reader.read_array(f"coord/{name}")
    coord = VerticalCoord(**coord_values)
    base_values = dict(metadata["base_scalars"])
    for name in metadata["base_arrays"]:
        base_values[name] = reader.read_array(f"base/{name}")
    base = BaseState(**base_values)

    state_names = list(metadata["state_names"])
    surface_names = list(metadata.get("surface_fields", []))
    met_names = list(metadata["met_fields"])

    # Resolved once, here, so every slab is harvested by the same object the
    # buffers will be built by; see the docstring for the one carrier the
    # plain carrier inventory silently left out of the store.
    inventory_fn = (streamed_store_inventory() if inventory_fn is None
                    else inventory_fn)

    slabs = _plan_slabs(ny, rows_per_slab)
    log(f"    {len(slabs)} row slabs of <= {rows_per_slab} rows, "
        f"{len(state_names)} prognostics + {len(surface_names)} surface "
        f"+ {len(met_names)} met from the cache")
    cached = _CachePayload(reader, verify=verify_payload, log=log)

    hydrometeors = MappingProxyType(
        dict(metadata.get("hydrometeor_initialization", {})))
    cen_lat = (float(getattr(grid, "cen_lat", grid.ref_lat))
               if center_lat is None else float(center_lat))

    store: dict = {}
    geo_store: dict = {}
    scalars: dict = {}
    seen: set = set()
    seams: dict = {}
    allocated = False
    manifest = None
    template = None

    for index, (j0, rows) in enumerate(slabs):
        t0 = time.perf_counter()
        cfg_slab = replace(cfg, ny=int(rows))
        static_slab = _window_mapping(static, j0, rows, ny)
        surface_slab = SimpleNamespace(fields=MappingProxyType({
            name: _row_window(cached[f"surface/{name}"], j0, rows, ny)
            for name in surface_names}))
        met_slab = SimpleNamespace(fields=MappingProxyType({
            name: _row_window(cached[f"met/{name}"], j0, rows, ny)
            for name in met_names}))
        grid_slab = _realdata.window_grid(grid, 0, j0, nx, int(rows))

        # The base state is windowed exactly as the statics are, and for
        # exactly the same reason: with terrain it is domain-SHAPED, so a
        # slab handed the whole thing dies inside load_base's dev[...]
        # assignment.  Built ONCE per slab and used twice -- the state and
        # the physics must agree about which rows the base describes.
        base_slab = _slab_base(base, j0, rows, ny)

        state = _slab_state(
            cfg_slab, coord, base_slab, static_slab,
            lambda name: _row_window(cached[f"state/{name}"], j0, rows, ny),
            state_names)
        result = CachedInitialResult(
            state=state, coord=coord, base=base_slab,
            surface_pressure=_row_window(
                cached["result/surface_pressure"], j0, rows, ny),
            surface_qv=_row_window(
                cached["result/surface_qv"], j0, rows, ny),
            hydrometeor_initialization=hydrometeors)
        # center_lat is the DOMAIN's, passed rather than re-derived:
        # initialize_landuse reads it for the LANDUSE.TBL season, and a slab
        # that used its own centre would pick a different season for the same
        # column depending on which slab it landed in.
        initialize_prepared_physics(
            result, cfg_slab, met_slab, surface_slab, static_slab,
            landuse_attrs, grid_slab, valid_time,
            center_lat=cen_lat, constant_glw_wm2=constant_glw_wm2)
        # Exactly where the resident road primes the DOMAIN before attach
        # (prepared_domain_builder), and for the same reason: REFL_10CM and
        # the other lazily-allocated slots are absent from a state that has
        # never stepped, so a store built without them has nowhere for the
        # tiles' windows to join and the first due history frame is refused
        # an hour into an otherwise healthy forecast.  Primed per slab so
        # the full-domain store carries the slot the buffers carry.
        prime_lazy_carriers(state, cfg_slab)

        # The RUN's inventory rule, not the plain carrier one: the store and
        # the tile buffers have to be built by the same rule or TiledRun
        # refuses the pair, and REFL_10CM is the key the two disagreed on.
        # ``names=None`` because a slab reveals the manifest rather than
        # being asked for a known one -- streaming_inventory takes the
        # argument, carrier_inventory defaults it, and the call has to be
        # right for whichever rule a caller passes.
        inv = inventory_fn(state, None)
        ginv = tdriver.geography_inventory(state)
        if not allocated:
            allocated = True
            manifest = hoststore.manifest_from_arrays(
                inv, nz, int(rows), nx)
            geo_shapes = {
                name: (tuple(arr.shape[:-2])
                       + (ny + (int(arr.shape[-2]) - int(rows)),
                          int(arr.shape[-1])))
                for name, arr in ginv.items()}
            # THE WHOLE REQUEST, BEFORE THE FIRST BYTE OF IT IS TAKEN.
            planned = sum(spec.nbytes(nz, ny, nx) for spec in manifest)
            planned += sum(int(np.prod(shape))
                           * np.dtype(ginv[name].dtype).itemsize
                           for name, shape in geo_shapes.items())
            hoststore.check_allocatable(planned, budget_bytes=budget_bytes)
            log(f"    store: {planned / hoststore.GIB:.2f} GiB pinned "
                f"({len(manifest)} carriers + {len(geo_shapes)} geography), "
                "budget checked")
            for spec in manifest:
                store[spec.name] = hoststore.alloc_pinned_array(
                    spec.shape(nz, ny, nx), spec.dtype)
            for name, shape in geo_shapes.items():
                geo_store[name] = hoststore.alloc_pinned_array(
                    shape, np.dtype(ginv[name].dtype))
            scalars = physinv.carrier_scalars(state)

        for name, arr in inv.items():
            if name not in store:
                raise PreparedStoreError(
                    f"slab {index} produced carrier {name!r} that slab 0 did "
                    "not; the manifest is not slab-invariant")
            host = _as_host(arr)
            disagreed = _seam_disagreement(store[name], host, j0, rows)
            if disagreed:
                seams[name] = seams.get(name, 0) + disagreed
            _scatter_rows(store[name], host, j0)
            seen.add(name)
        for name, arr in ginv.items():
            host = _as_host(arr)
            disagreed = _seam_disagreement(geo_store[name], host, j0, rows)
            if disagreed:
                seams[name] = seams.get(name, 0) + disagreed
            _scatter_rows(geo_store[name], host, j0)

        if index == len(slabs) - 1:
            # Kept, not rebuilt: the last slab already carries the scheme
            # adapters, the eta table and the ndim < 2 setup arrays that
            # prepared_tile_state_factory reads off "the domain", and every
            # one of those is the same at any height.  See PreparedStore.
            template = state
        else:
            del state
        del inv, ginv, result
        cp.get_default_memory_pool().free_all_blocks()
        log(f"    slab {index + 1}/{len(slabs)} rows {j0}..{j0 + rows} "
            f"in {time.perf_counter() - t0:.1f}s")

    cached.close()
    if seams:
        # Not a warning.  A carrier whose shared face disagrees between the
        # two slabs that own it is wrong in nslabs - 1 rows by construction,
        # and it is wrong by an amount that looks like weather rather than
        # like a bug -- which is precisely why it must stop the load rather
        # than annotate it.  The fix is a y-halo on the offending carrier,
        # not a smaller tolerance.
        detail = ", ".join(f"{name} ({count} values)"
                           for name, count in sorted(seams.items()))
        raise PreparedStoreError(
            "these y-staggered carriers disagree with themselves on the "
            f"faces two slabs share: {detail}.  A y-staggered field's first "
            "row is the previous slab's last row, so a carrier the physics "
            "COMPUTES rather than copies is written there twice -- once as "
            "one slab's top edge and once as the next slab's bottom edge, "
            "and neither is the interior value the whole domain would have "
            "produced.  Slabbing this cache needs a y-halo for those "
            "carriers before its store can be trusted.")
    boundaries = (None if metadata["lbc"].get("intervals") is None
                  else _boundaries_from_cache(reader, metadata))
    missing = tuple(sorted(set(store) - seen))
    receipt = MappingProxyType({
        "schema": "gpuwm-prepared-store-v1",
        "status": "LOADED",
        "path": str(reader.path.resolve()),
        "content_sha256": reader.content_sha256,
        "payload_bytes": reader.payload_bytes,
        "slabs": len(slabs),
        "rows_per_slab": int(rows_per_slab),
        "carriers": len(store),
        "geography": len(geo_store),
        "store_bytes": int(sum(a.nbytes for a in store.values())),
        "geography_bytes": int(sum(a.nbytes for a in geo_store.values())),
        "missing": missing,
    })
    return PreparedStore(
        store=store, geography=geo_store, scalars=scalars, template=template,
        # THE WHOLE-DOMAIN base, deliberately, and never a slab's: the
        # windowed copies exist only to build the slabs above, and a
        # consumer that reads bundle.base wants the domain's dry mass and
        # profiles -- publishing the last slab's would be a base state
        # silently one slab tall.
        coord=coord, base=base, boundaries=boundaries, missing=missing,
        receipt=receipt)
