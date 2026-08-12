"""Slice a GLOBAL domain state across ranks: gathered, never rebuilt.

The multi-GPU path builds every rank's state by calling an analytic builder at
the rank's LOCAL shape (``physics_inventory.default_builder`` on a WK82
sounding).  That is the right thing for a bit-exact transport gate -- the gate
never inspects a value -- and it is useless for a forecast, for two separate
reasons:

* the sounding is a function of ``z`` alone, so eight ranks build eight
  IDENTICAL tiles and the "domain" has no horizontal structure to advect;
* there is no analysis anywhere in it.

The missing operation is not a new initialiser.  It is the one the tiled path
already performs on GEOGRAPHY: take the DOMAIN-extent arrays, put them in a
host store, and hand each rank ITS WINDOW through
:class:`tilestream.spec.TileSpec` -- staggering, halo and periodic wrap all
handled by the code the data path already uses.

Rebuilding per rank is not a slower equivalent of slicing; it is a DIFFERENT
DOMAIN.  ``harness.make_geography`` centres its projection on whatever
``nx``/``ny`` it is handed, so a rank that rebuilds is somewhere else on the
earth -- up to 1,022 km away on an eight-way split of a CONUS domain.  The
same argument applies verbatim to the initial condition, which is why this
module exists.

What one rank needs, in full:

``carriers``
    every entry of :func:`tilestream.physics_inventory.carrier_inventory`
    with a horizontal extent -- the prognostic state, the tendency bundles,
    the surface and soil fields.  Sliced.
``geography``
    every entry of :func:`tilestream.driver.geography_inventory` -- the
    horizontally-varying INPUT: terrain, base state, map factors, Coriolis,
    and the per-scheme latitude/longitude grids that no carrier manifest
    mentions.  Sliced.
``scalars``
    the carrier scalars, ``elapsed_seconds`` first among them.  BROADCAST, not
    sliced.  Ranks that disagree about elapsed time disagree about which
    schemes are due on the next step.
``setup flags``
    ``has_msf`` and ``rotational``, computed ONCE over the domain arrays and
    re-imposed on every rank.  ``DomainState.set_map_coriolis`` derives them
    from ``.any()`` over whatever window it was handed, so a rank whose window
    happens to be uniform silently takes a different branch in the Coriolis
    kernel.
``vertical setup``
    ``znu``, ``znw``, ``dnw``, ``c1h..c4f``, ``fnp``, ``fnm`` and friends are
    pure functions of ``nz``/``hybrid_opt``/``etac``/``p_top``: a rank rebuilds
    them exactly.  NOT sliced -- and :func:`assert_slice_faithful` checks that
    claim rather than assuming it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from tilestream import gather as _gather
from tilestream import spec as _spec

__all__ = [
    "DecompositionError",
    "DomainStore",
    "rank_specs",
    "describe_plan",
    "variant_of",
    "store_from_state",
    "store_from_arrays",
    "slice_array",
    "install_slice",
    "reassemble",
    "assert_slice_faithful",
    "seam_statistics",
]


class DecompositionError(RuntimeError):
    """A rank plan that cannot be a decomposition of the domain."""


# --------------------------------------------------------------------------
# the plan
# --------------------------------------------------------------------------

def rank_specs(nx: int, ny: int, px: int, py: int, halo: int, *,
               periodic: bool = True) -> list[_spec.TileSpec]:
    """One :class:`~tilestream.spec.TileSpec` per rank, in rank order.

    A multi-GPU decomposition IS a tile plan with exactly one tile per rank,
    so this is deliberately thin: it defers every geometric question --
    staggered faces, the shared-face ownership rule, windows that run off a
    periodic edge -- to :func:`tilestream.spec.plan_tiles`, which the tiled
    data path already trusts with them.

    Rank ``r`` is ``(r // px, r % px)`` in ``(y, x)``, matching the
    ``jy * px + jx`` ordering the rescued 8-GPU workers use, so a plan built
    here can be handed to them unchanged.

    Exact divisibility is REQUIRED, not merely preferred: ``plan_tiles`` is
    happy to make ragged edge tiles, but every rank in a resident multi-GPU
    run allocates its state once and they must all be the same shape for the
    exchange planes and the per-rank VRAM budget to mean anything.
    """
    if px < 1 or py < 1:
        raise DecompositionError(f"px={px}, py={py} must both be >= 1")
    if nx % px or ny % py:
        raise DecompositionError(
            f"domain {ny}x{nx} does not divide evenly over a {py}x{px} rank "
            f"grid ({ny}/{py}={ny / py}, {nx}/{px}={nx / px}); a resident "
            f"multi-GPU run needs identical per-rank shapes")
    tile_nx, tile_ny = nx // px, ny // py
    if halo < 0:
        raise DecompositionError(f"halo={halo} must be >= 0")
    if not periodic and halo > 0:
        raise DecompositionError(
            "periodic=False clamps the compute window at the domain edge, "
            "which makes per-rank shapes RAGGED; the resident multi-GPU path "
            "needs uniform shapes, so it decomposes periodically and lets the "
            "boundary zone be imposed by the lateral-boundary series")
    specs = _spec.plan_tiles(nx, ny, tile_nx, tile_ny, halo, periodic=periodic)
    _spec.validate_plan(specs, ny, nx)
    if len(specs) != px * py:
        raise DecompositionError(
            f"plan_tiles produced {len(specs)} tiles for a {py}x{px} rank grid")
    ordered = sorted(specs, key=lambda s: (s.ty, s.tx))
    for r, s in enumerate(ordered):
        if (s.ty, s.tx) != (r // px, r % px):
            raise DecompositionError(
                f"rank {r} landed at tile {(s.ty, s.tx)}, expected "
                f"{(r // px, r % px)}")
        if s.ragged_x or s.ragged_y:
            raise DecompositionError(f"rank {r} is ragged: {s.describe()}")
    return ordered


def describe_plan(specs: Sequence[_spec.TileSpec], nz: int) -> str:
    s0 = specs[0]
    interior = s0.interior_ny * s0.interior_nx * nz
    local = s0.cny * s0.cnx * nz
    return (f"{len(specs)} ranks, interior {s0.interior_ny}x{s0.interior_nx}, "
            f"local-with-halo {s0.cny}x{s0.cnx} (halo {s0.halo}), "
            f"{interior / 1e6:.2f} Mcell owned / {local / 1e6:.2f} Mcell "
            f"resident per rank")


# --------------------------------------------------------------------------
# the store
# --------------------------------------------------------------------------

def variant_of(name: str, shape: Sequence[int], nz: int, ny: int,
               nx: int) -> str:
    """The stagger variant of a domain-extent array, or raise saying which.

    ``layers_ok=True`` because the carrier set legitimately contains leading
    axes that are not vertical -- 4 soil levels, 3 snow layers, a 7-entry
    snow-soil coordinate.  Those are ordinary horizontal windows.
    """
    try:
        return _gather.classify(shape, nz, ny, nx, layers_ok=True)
    except _gather.SpecError as exc:
        raise DecompositionError(f"{name}: {exc}") from None


def _as_host(array) -> np.ndarray:
    mod = type(array).__module__
    if mod.startswith("cupy"):
        return array.get()
    return np.ascontiguousarray(array)


@dataclass
class DomainStore:
    """DOMAIN-extent arrays every rank gathers its window from.

    ``arrays`` and ``geography`` are kept apart because they are installed
    differently -- carriers into the carrier inventory, geography onto the
    state's setup attributes and the driver's scheme objects -- not because
    the transport treats them differently.  It does not: both are horizontal
    windows through the same :class:`~tilestream.spec.TileSpec`.
    """

    nx: int
    ny: int
    nz: int
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    geography: dict[str, np.ndarray] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    geo_flags: dict[str, bool] = field(default_factory=dict)
    elapsed_seconds: float = 0.0
    provenance: dict[str, Any] = field(default_factory=dict)

    # -- introspection ----------------------------------------------------

    @property
    def names(self) -> list[str]:
        return sorted(self.arrays)

    def variant(self, name: str) -> str:
        source = self.arrays.get(name)
        if source is None:
            source = self.geography[name]
        return variant_of(name, source.shape, self.nz, self.ny, self.nx)

    def nbytes(self) -> int:
        return (sum(a.nbytes for a in self.arrays.values())
                + sum(a.nbytes for a in self.geography.values()))

    def summary(self) -> str:
        return (f"DomainStore {self.nz}x{self.ny}x{self.nx}: "
                f"{len(self.arrays)} carriers + {len(self.geography)} "
                f"geography arrays, {self.nbytes() / 2 ** 30:.2f} GiB host, "
                f"clock {self.elapsed_seconds:.1f} s, "
                f"has_msf={self.geo_flags.get('has_msf')} "
                f"rotational={self.geo_flags.get('rotational')}")


def store_from_state(state, *, nz: int | None = None,
                     names: Iterable[str] | None = None,
                     provenance: Mapping[str, Any] | None = None
                     ) -> DomainStore:
    """Host copy of a MONOLITHIC state's carriers and geography.

    This is the reference path: it is how the bit-exactness gate gets a
    domain to slice, and how a domain that fits one card is initialised from
    the production ingest.  A domain too large to build monolithically needs
    :func:`store_from_arrays` instead, which never materialises the state.
    """
    from tilestream import driver as _driver
    from tilestream import physics_inventory as _physics

    carriers = _physics.carrier_inventory(state, names)
    geo = _driver.geography_inventory(state)
    sample = carriers.get("state/p")
    if nz is None:
        if sample is None:
            raise DecompositionError(
                "cannot infer nz: no 'state/p' carrier; pass nz=")
        nz = int(sample.shape[0])
    ny, nx = (int(sample.shape[-2]), int(sample.shape[-1])) if sample is not None \
        else (None, None)
    if ny is None:
        raise DecompositionError("cannot infer domain extents without state/p")

    store = DomainStore(nx=nx, ny=ny, nz=nz)
    for name, value in carriers.items():
        if not _gather._is_array(value):
            continue
        host = _as_host(value)
        if host.ndim >= 2:
            variant_of(name, host.shape, nz, ny, nx)   # refuse odd shapes here
        store.arrays[name] = host
    for key, value in geo.items():
        host = _as_host(value)
        if host.ndim >= 2:
            variant_of(key, host.shape, nz, ny, nx)
        store.geography[key] = host
    store.scalars = dict(_physics.carrier_scalars(state))
    store.geo_flags = dict(_driver.geography_scalars(store.geography))
    store.elapsed_seconds = float(getattr(state, "elapsed_seconds", 0.0))
    store.provenance = dict(provenance or {})
    return store


def store_from_arrays(nx: int, ny: int, nz: int, arrays: Mapping[str, Any],
                      geography: Mapping[str, Any] | None = None, *,
                      scalars: Mapping[str, Any] | None = None,
                      geo_flags: Mapping[str, bool] | None = None,
                      elapsed_seconds: float = 0.0,
                      provenance: Mapping[str, Any] | None = None
                      ) -> DomainStore:
    """A store assembled field by field, with no monolithic state anywhere.

    The out-of-core route: the ingest writes each domain-extent field straight
    into host memory (or a memmap) and this wraps them.  Nothing here ever
    allocates ``nz*ny*nx`` on a device.
    """
    store = DomainStore(nx=int(nx), ny=int(ny), nz=int(nz))
    for name, value in arrays.items():
        host = _as_host(value)
        if host.ndim >= 2:
            variant_of(name, host.shape, nz, ny, nx)
        store.arrays[name] = host
    for key, value in (geography or {}).items():
        host = _as_host(value)
        if host.ndim >= 2:
            variant_of(key, host.shape, nz, ny, nx)
        store.geography[key] = host
    store.scalars = dict(scalars or {})
    if geo_flags is None:
        from tilestream import driver as _driver
        geo_flags = _driver.geography_scalars(store.geography)
    store.geo_flags = dict(geo_flags)
    store.elapsed_seconds = float(elapsed_seconds)
    store.provenance = dict(provenance or {})
    return store


# --------------------------------------------------------------------------
# the slice
# --------------------------------------------------------------------------

def slice_array(source, spec: _spec.TileSpec, variant: str, *,
                out=None, xp=None):
    """This rank's window of a domain-extent array, halo included.

    The index arithmetic is pure delegation to
    :meth:`TileSpec.apply_gather` -- the wrap, the staggered closing face and
    the several rectangles a window that runs off an edge decomposes into are
    all its business, and reimplementing them here is how a decomposition
    acquires an off-by-one that only shows on the ranks touching a boundary.

    The gather is ASSEMBLED ON THE HOST and then crosses to the device once,
    as a single contiguous copy.  ``TileSpec``'s rectangles are fancy-index
    assignments, and a device destination cannot take a host source that way;
    doing it per rectangle would also mean up to nine small transfers per
    field instead of one.  (The streamed path issues the same rectangles as
    ``cudaMemcpy3DAsync`` through :mod:`tilestream.gather` because it is on
    the step path and the copies must overlap compute.  An initialiser is not
    on the step path.)
    """
    source = _as_host(source)
    if source.ndim < 2:
        raise DecompositionError(
            f"array of shape {source.shape} has no horizontal extent to slice")
    lead = tuple(int(s) for s in source.shape[:-2])
    ey, ex = _spec.stagger(variant)
    shape = lead + (spec.cny + ey, spec.cnx + ex)
    window = np.empty(shape, dtype=source.dtype)
    spec.apply_gather(source, window, variant)
    if out is None:
        if xp is None or xp is np:
            return window
        return xp.asarray(window)
    if tuple(out.shape) != shape:
        raise DecompositionError(
            f"destination shape {tuple(out.shape)} != window shape {shape} "
            f"for variant {variant!r}")
    if type(out).__module__.startswith("cupy"):
        import cupy as cp
        out[...] = cp.asarray(window)
    else:
        out[...] = window
    return out


def install_slice(state, store: DomainStore, spec: _spec.TileSpec, *,
                  strict: bool = True) -> dict[str, Any]:
    """Fill ONE rank's state with its window of ``store``.

    Order matters and is not cosmetic:

    1. geography first, because ``mub2d``/``ht``/``thb``/``phb`` are what the
       prognostic fields are defined against;
    2. carriers second;
    3. scalars and the two setup FLAGS last, over the top of whatever the
       rank's own builder derived from its window.

    Returns a report naming everything that was installed and everything the
    store had no entry for, so a caller can refuse a partial slice instead of
    stepping one.
    """
    from tilestream import driver as _driver
    from tilestream import physics_inventory as _physics

    installed: list[str] = []
    missing: list[str] = []
    skipped_vertical: list[str] = []

    # -- 1. geography ------------------------------------------------------
    local_geo = _driver.geography_inventory(state)
    for key, target in local_geo.items():
        source = store.geography.get(key)
        if source is None:
            missing.append(key)
            continue
        variant = variant_of(key, source.shape, store.nz, store.ny, store.nx)
        slice_array(source, spec, variant, out=target)
        installed.append(key)
    for key in store.geography:
        if key not in local_geo:
            missing.append(key)

    # -- 2. carriers -------------------------------------------------------
    local = _physics.carrier_inventory(state)
    for name, target in local.items():
        if not _gather._is_array(target):
            continue
        source = store.arrays.get(name)
        if source is None:
            missing.append(name)
            continue
        if getattr(target, "ndim", 0) < 2:
            # A carrier with no horizontal extent is a domain constant; it is
            # broadcast, not windowed.
            target[...] = source
            skipped_vertical.append(name)
            continue
        variant = variant_of(name, source.shape, store.nz, store.ny, store.nx)
        slice_array(source, spec, variant, out=target)
        installed.append(name)

    # -- 3. scalars and the domain-wide setup flags ------------------------
    if store.scalars:
        _physics.set_carrier_scalars(state, store.scalars)
    state.elapsed_seconds = float(store.elapsed_seconds)
    for flag, value in store.geo_flags.items():
        setattr(state, flag, bool(value))

    report = dict(rank=spec.index, installed=len(installed),
                  missing=sorted(set(missing)),
                  broadcast=sorted(skipped_vertical))
    if strict and report["missing"]:
        raise DecompositionError(
            f"rank {spec.index}: the store has no entry for "
            f"{report['missing'][:8]}"
            + (f" (+{len(report['missing']) - 8} more)"
               if len(report['missing']) > 8 else "")
            + " -- a partial slice would leave analytic air in those fields")
    return report


def reassemble(specs: Sequence[_spec.TileSpec], blocks: Sequence[Mapping[str, Any]],
               store: DomainStore, *, names: Iterable[str] | None = None
               ) -> dict[str, np.ndarray]:
    """Scatter every rank's INTERIOR back into domain-extent host arrays.

    The inverse of the slice, and the only honest way to ask whether the slice
    was right: ``TileSpec.scatter``'s shared-face ownership rule means every
    point of the domain is written exactly once across the plan
    (:func:`tilestream.spec.coverage_counts` is what proves that), so the
    result is comparable to the monolithic array BIT FOR BIT rather than
    approximately.

    Destinations start as ``nan``-filled (or a sentinel for integer dtypes) so
    a point no rank claimed cannot masquerade as a match against a zero.
    """
    keys = list(names) if names is not None else sorted(
        set(store.arrays) | set(store.geography))
    out: dict[str, np.ndarray] = {}
    for key in keys:
        source = store.arrays.get(key)
        if source is None:
            source = store.geography.get(key)
        if source is None or source.ndim < 2:
            continue
        variant = variant_of(key, source.shape, store.nz, store.ny, store.nx)
        dest = np.empty_like(source)
        if dest.dtype.kind == "f":
            dest[...] = np.nan
        else:
            dest[...] = np.iinfo(dest.dtype).min if dest.dtype.kind in "iu" else 0
        for spec, block in zip(specs, blocks):
            value = block.get(key)
            if value is None:
                continue
            spec.apply_scatter(_as_host(value), dest, variant)
        out[key] = dest
    return out


# --------------------------------------------------------------------------
# the checks
# --------------------------------------------------------------------------

def assert_slice_faithful(specs: Sequence[_spec.TileSpec],
                          blocks: Sequence[Mapping[str, Any]],
                          store: DomainStore, *,
                          names: Iterable[str] | None = None
                          ) -> dict[str, Any]:
    """Prove the reassembled slice is the monolithic domain, bit for bit.

    Compared on the BIT PATTERN, not with ``allclose``: the claim being tested
    is that a rank holds exactly the bits the monolithic run holds at the same
    global position, and a tolerance would pass a slice that is merely close,
    which is precisely the failure mode of an off-by-one in a smooth field.

    ``nan`` is compared by bit pattern too, so an unclaimed point (which the
    reassembly leaves as ``nan``) can never match.
    """
    rebuilt = reassemble(specs, blocks, store, names=names)
    bad: dict[str, dict[str, Any]] = {}
    checked = 0
    for key, got in rebuilt.items():
        ref = store.arrays.get(key)
        if ref is None:
            ref = store.geography[key]
        checked += 1
        a = np.ascontiguousarray(got)
        b = np.ascontiguousarray(ref)
        if a.dtype != b.dtype or a.shape != b.shape:
            bad[key] = dict(reason="shape/dtype",
                            got=(a.shape, str(a.dtype)),
                            want=(b.shape, str(b.dtype)))
            continue
        av = a.view(_bit_dtype(a.dtype)) if a.dtype.kind == "f" else a
        bv = b.view(_bit_dtype(b.dtype)) if b.dtype.kind == "f" else b
        differ = av != bv
        n = int(differ.sum())
        if n:
            idx = np.argwhere(differ)
            first = tuple(int(v) for v in idx[0])
            with np.errstate(invalid="ignore"):
                delta = float(np.nanmax(np.abs(
                    a[differ].astype(np.float64)
                    - b[differ].astype(np.float64)))) if a.dtype.kind == "f" \
                    else float(np.max(np.abs(a[differ].astype(np.int64)
                                             - b[differ].astype(np.int64))))
            bad[key] = dict(reason="bits", n_differ=n, size=int(a.size),
                            first_index=first, max_abs_delta=delta)
    # ``ok`` used to be ``not bad``, which makes ``checked == 0`` a pass: a
    # reassembly that compared nothing reported bit-exact.  The count was
    # printed by the gate but was never a CONDITION, which is the shape
    # ``d998bb667`` / ``9b1b99289`` fixed on the release line's own gates.
    # The floor is one, not the full carrier count, so a deliberately
    # narrowed comparison is still a result.
    return dict(checked=checked, mismatched=bad,
                the_comparison_is_not_empty=checked >= 1,
                ok=(checked >= 1) and not bad)


def _bit_dtype(dtype):
    return {4: np.uint32, 8: np.uint64, 2: np.uint16}[dtype.itemsize]


def seam_statistics(field: np.ndarray, specs: Sequence[_spec.TileSpec], *,
                    nx: int, ny: int, variant: str = "mass") -> dict[str, Any]:
    """Is the field CONTINUOUS across the sub-domain boundaries?

    Answered the only way that means anything: by comparing like with like.
    The mean absolute first difference is taken across the columns that ARE a
    rank boundary and across the columns that are NOT, and the statistic is
    their ratio.  A field with real weather in it has a large absolute
    gradient everywhere, so an absolute threshold on the seam difference is
    uninterpretable; the ratio is not.  A correct slice gives a ratio near 1,
    because a seam column is an ordinary column that happens to have a rank
    boundary drawn on it.

    Reported separately for x and y seams: a decomposition can be right in one
    axis and transposed in the other, and a combined number hides that.
    """
    if field.ndim != 2:
        raise DecompositionError(
            f"seam_statistics wants a 2-D horizontal field, got {field.shape}")
    a = np.asarray(field, dtype=np.float64)
    x_seams = sorted({int(s.i0) for s in specs if s.i0 != 0})
    y_seams = sorted({int(s.j0) for s in specs if s.j0 != 0})

    def axis_stats(arr, seams, n):
        # |f[k] - f[k-1]| indexed by k, so k in `seams` is the difference
        # ACROSS the boundary between rank k-1's last column and rank k's
        # first.
        diff = np.abs(np.diff(arr, axis=0))
        k = np.arange(1, arr.shape[0])
        is_seam = np.isin(k, seams)
        if not is_seam.any():
            return None
        seam_mean = float(np.nanmean(diff[is_seam]))
        bulk_mean = float(np.nanmean(diff[~is_seam]))
        # A field with no horizontal structure cannot gate a seam: every
        # difference is zero on both sides of the comparison, and 0/0 is not a
        # discontinuity, it is an absence of evidence.  Say so, rather than
        # reporting `inf` and letting a uniform field read as a failure.
        degenerate = bulk_mean == 0.0 and seam_mean == 0.0
        if degenerate:
            ratio = 1.0
        elif bulk_mean > 0.0:
            ratio = seam_mean / bulk_mean
        else:
            ratio = float("inf")
        return dict(seam_mean=seam_mean, bulk_mean=bulk_mean, ratio=ratio,
                    degenerate=degenerate,
                    n_seam_lines=int(is_seam.sum()),
                    n_bulk_lines=int((~is_seam).sum()))

    return dict(
        x=axis_stats(a.T, x_seams, nx),   # transpose: diff along columns
        y=axis_stats(a, y_seams, ny),
        x_seam_columns=x_seams, y_seam_rows=y_seams)
