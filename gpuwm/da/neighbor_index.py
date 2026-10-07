"""The observation-to-gridpoint localisation index, built once per analysis.

:func:`gpuwm.da.letkf.analyze` needs, for every analysis gridpoint, the
observations inside its Gaspari-Cohn lens and their localisation weights.
The bounded host-staged route used to find them FORWARD, chunk by chunk:
for every gridpoint of a chunk it enumerated every stencil slot of every
batch that could reach the chunk, evaluated the full geometric weight on
the host, and then threw away the slots that held no observation.  On the
241 x 241 x 49 recent case that is roughly 2.8 million gridpoints x ~2,000
slots x every reaching batch, in single-threaded numpy, repeated for every
chunk: most of the 23 minutes the analysis took.

This module finds the same pairs BACKWARD, once.  An observation at grid
point ``o`` and a stencil offset ``s`` name exactly one analysis gridpoint
``g = o - s``, so enumerating ``observations x slots`` visits only pairs
that can carry weight.  Observations are sparse (the filter's own masks
are), so the work is the observed points times the stencil, not the grid
times the stencil, and it runs in whichever array namespace the solve uses
(cupy on a card, numpy on the host).

The answer is the SAME roster, in the SAME order, with the SAME bits:

* the horizontal weight is a function of ``(analysis column, offset)``
  only, so it is tabulated once per stencil on the HOST with the same
  numpy expression the forward path evaluates -- transcendental functions
  (the haversine) never run on the device;
* the vertical weight ``gaspari_cohn(|z[o] - z[g]|)`` uses only
  subtraction, absolute value, products, quotients, ``minimum``/``where``
  and ``maximum``, each a single correctly rounded IEEE operation in
  either namespace (one operation per ufunc call, so no fused
  multiply-add can form);
* the geometric weight is their single product, filtered on ``> 0``, cast
  to the solve dtype and filtered on ``> 0`` again, exactly as before;
* each gridpoint's neighbours are ordered by (batch, slot), which is the
  order the forward path's per-batch ``nonzero`` and its packing loop
  produced.

So a transform fed from this index sees byte-identical packed arrays.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import time

import numpy as np


#: Horizontal weight tables are built in column blocks of this many
#: analysis columns, so a continental grid does not allocate the whole
#: (columns x offsets) haversine scratch at once.
_TABLE_COLUMN_BLOCK = 1 << 16

#: Upper bound on candidate (observation, level) x horizontal-offset pairs
#: evaluated in one vectorised step.  About 40 bytes of scratch per pair.
DEFAULT_PAIR_BUDGET = 1 << 25

#: ``resident="auto"`` keeps the roster on the card while its worst case
#: (every candidate pair kept, observation id + weight) is at most this
#: fraction of the card's free memory, and on the host otherwise.  The
#: breakage it prevents: on the 3 km CONUS grid at the deck's localisation
#: (4.3 M observations, about 5.8 G neighbours, about 70 GB of roster) the
#: card-resident build exhausted a 32 GB card in its per-slab sort
#: (measured 2026-10-05, lane conus-da-obs), while the same slabs fit
#: easily once finished slabs leave the card.
RESIDENT_FREE_FRACTION = 0.5

#: On a card, one slab (a run of whole levels, or a row band of one level)
#: holds at most this fraction of the free memory, priced at this many
#: bytes per candidate pair: int64 key, observation id, float64 weight,
#: the argsort order and the sorted copies.  The breakage it prevents: a
#: single 3 km CONUS level with a 100 km surface localisation (about a
#: billion candidates) sorted as one slab exhausted a 32 GB RTX 5090
#: (measured 2026-10-05, lane conus-da-obs).
SLAB_FREE_FRACTION = 0.25
SLAB_BYTES_PER_PAIR = 64

#: One vectorised step's bytes per candidate pair, used only when the build
#: is given a ``transient_bytes`` cap (see :func:`build_neighbor_index`).
STEP_BYTES_PER_PAIR = 128


def _namespace_name(xp) -> str:
    return getattr(xp, "__name__", "numpy")


def _to_host(xp, value):
    if xp is np or _namespace_name(xp) == "numpy":
        return np.asarray(value)
    return xp.asnumpy(value)


class _Uploads:
    """``xp.asarray`` of host arrays, with the bytes that crossed counted."""

    def __init__(self, xp):
        self.xp = xp
        self.bytes = 0

    def __call__(self, value):
        value = np.asarray(value)
        self.bytes += int(value.nbytes)
        return self.xp.asarray(value)


def _host(value):
    """A host numpy view of a small array from either namespace."""
    if hasattr(value, "get") and not isinstance(value, np.ndarray):
        return value.get()
    return np.asarray(value)


@dataclass
class NeighborIndex:
    """CSR roster of positive-weight neighbours, one row per gridpoint.

    ``row_ptr`` and ``counts`` live on the host (they steer the host chunk
    loop); ``obs``, ``weight`` and the compact observation arrays live in
    ``xp``.  ``obs`` indexes the compact arrays, which hold every batch's
    observed points back to back in batch order.
    """

    xp: object
    npts: int
    counts: np.ndarray            # (npts,) int64, host
    row_ptr: np.ndarray           # (npts + 1,) int64, host
    obs: object                   # (nnz,) int32/int64, xp
    weight: object                # (nnz,) solve dtype, xp
    sim: object                   # (members, nobs) work dtype, xp
    values: object                # (nobs,) work dtype, xp
    err2: object                  # (nobs,) solve dtype, xp
    batch_offsets: tuple          # start of each batch in the compact arrays
    build_seconds: float = 0.0
    #: True when ``obs``/``weight`` are host numpy arrays (a roster larger
    #: than the card); :meth:`pack` then gathers each chunk's slice on the
    #: host and uploads only that slice.
    host_resident: bool = False
    #: Host bytes the build copied into ``xp``, and the bytes the most
    #: recent :meth:`pack` copied (its row offsets).
    build_upload_bytes: int = 0
    last_pack_upload_bytes: int = 0
    receipt: dict = field(default_factory=dict)

    @property
    def nnz(self) -> int:
        return int(self.row_ptr[-1])

    @property
    def nbytes(self) -> int:
        return int(sum(int(getattr(a, "nbytes", 0)) for a in (
            self.obs, self.weight, self.sim, self.values, self.err2)))

    def pack(self, gpts, local_max: int, members: int, solve_dtype):
        """The transform's packed (sim, values, err2, weight) for ``gpts``.

        Row ``r`` of the result is gridpoint ``gpts[r]``; its neighbours
        fill slots ``0..count-1`` in (batch, slot) order and the tail keeps
        the forward path's fill values (zero simulation, zero value, unit
        error, zero weight).  Returned arrays are in ``xp``.
        """
        xp = self.xp
        gpts = np.asarray(gpts, dtype=np.int64)
        ng = int(gpts.size)
        cnt = self.counts[gpts]
        starts = self.row_ptr[gpts]
        total = int(cnt.sum())
        self.last_pack_upload_bytes = 0
        packed_s = xp.zeros((members, ng, local_max), dtype=solve_dtype)
        packed_v = xp.zeros((ng, local_max), dtype=solve_dtype)
        packed_e = xp.ones((ng, local_max), dtype=solve_dtype)
        packed_w = xp.zeros((ng, local_max), dtype=solve_dtype)
        if total == 0:
            return packed_s, packed_v, packed_e, packed_w
        up = _Uploads(xp)
        if self.host_resident:
            # The same entry -> (row, rank, source) map, on the host, and
            # only this chunk's observation ids and weights cross.
            ends_h = np.cumsum(cnt)
            entry_h = np.arange(total, dtype=np.int64)
            rows_h = np.searchsorted(ends_h, entry_h, side="right")
            rank_h = entry_h - (ends_h - cnt)[rows_h]
            src_h = starts[rows_h] + rank_h
            rows, rank = up(rows_h), up(rank_h)
            oid = up(self.obs[src_h])
            wt = up(self.weight[src_h])
            self.last_pack_upload_bytes = up.bytes
            packed_s[:, rows, rank] = self.sim[:, oid]
            packed_v[rows, rank] = self.values[oid]
            packed_e[rows, rank] = self.err2[oid]
            packed_w[rows, rank] = wt
            return packed_s, packed_v, packed_e, packed_w
        # Entry e belongs to row r(e) at rank q(e); its source position in
        # the CSR is starts[r] + q.  Built on the host as int64 (cheap: one
        # entry per packed neighbour) only when the namespace is numpy; on a
        # card it is built there.
        ends = np.cumsum(cnt)
        d_ends = up(ends)
        d_first = d_ends - up(cnt)
        entry = xp.arange(total, dtype=np.int64)
        rows = xp.searchsorted(d_ends, entry, side="right")
        rank = entry - d_first[rows]
        src = up(starts)[rows] + rank
        self.last_pack_upload_bytes = up.bytes
        oid = self.obs[src]
        packed_s[:, rows, rank] = self.sim[:, oid]
        packed_v[rows, rank] = self.values[oid]
        packed_e[rows, rank] = self.err2[oid]
        packed_w[rows, rank] = self.weight[src]
        return packed_s, packed_v, packed_e, packed_w


def horizontal_weight_table(dj, di, hcut, *, ny, nx, horizontal_distance,
                            gaspari_cohn):
    """``(ny*nx, n_h)`` horizontal Gaspari-Cohn weight, host float64.

    Row ``c`` is analysis column ``c``; column ``h`` is the weight to the
    column offset ``(dj[h], di[h])`` from it, evaluated with the forward
    path's own expression: the neighbour column clipped into the grid, the
    distance from ``horizontal_distance(analysis, neighbour)``.  Entries
    whose neighbour falls off the grid are computed against the clipped
    column and never read.
    """
    dj = np.asarray(dj, dtype=np.int64)
    di = np.asarray(di, dtype=np.int64)
    ncol = ny * nx
    table = np.empty((ncol, int(dj.size)), dtype=np.float64)
    for lo in range(0, ncol, _TABLE_COLUMN_BLOCK):
        hi = min(ncol, lo + _TABLE_COLUMN_BLOCK)
        ccol = np.arange(lo, hi, dtype=np.int64)
        jj = ccol // nx
        ii = ccol - jj * nx
        j2 = np.clip(jj[:, None] + dj[None, :], 0, ny - 1)
        i2 = np.clip(ii[:, None] + di[None, :], 0, nx - 1)
        col = j2 * nx + i2
        table[lo:hi] = np.asarray(gaspari_cohn(
            horizontal_distance(ccol[:, None], col), hcut))
    return table


def observed_column_weight_table(ocols, dj, di, hcut, *, ny, nx,
                                 horizontal_distance, gaspari_cohn):
    """``(len(ocols), n_h)`` horizontal weight, host float64, observed rows.

    Row ``u`` is OBSERVED column ``o = ocols[u]``; column ``h`` is the
    weight between the analysis column ``o - (dj[h], di[h])`` and ``o``,
    evaluated as :func:`horizontal_weight_table` evaluates it at that
    analysis column: ``gaspari_cohn(horizontal_distance(analysis, o))``,
    one IEEE operation per element in the same order.  Entries whose
    analysis column falls off the grid are computed against ``o`` itself
    and never read.

    The full table is ``columns x offsets``: at 3 km over CONUS with a
    300 km conventional cutoff that is 1.79 M x 32,600 float64, 455 GiB
    (measured 2026-10-05, lane conus-da-obs), and at 9 km it took 6.4 GB of
    a 32 GB card.  Only observed columns are ever read, and a CONUS hour
    has a few thousand stations and well under a million radar columns.
    """
    ocols = np.asarray(ocols, dtype=np.int64)
    dj = np.asarray(dj, dtype=np.int64)
    di = np.asarray(di, dtype=np.int64)
    nh = int(dj.size)
    table = np.empty((int(ocols.size), nh), dtype=np.float64)
    block = max(1, _TABLE_COLUMN_BLOCK * 64 // max(1, nh))
    for lo in range(0, int(ocols.size), block):
        hi = min(int(ocols.size), lo + block)
        o = ocols[lo:hi]
        oj = o // nx
        oi = o - oj * nx
        cj = oj[:, None] - dj[None, :]
        ci = oi[:, None] - di[None, :]
        inside = (cj >= 0) & (cj < ny) & (ci >= 0) & (ci < nx)
        ccol = np.where(inside, cj * nx + ci, o[:, None])
        table[lo:hi] = np.asarray(gaspari_cohn(
            horizontal_distance(ccol, np.broadcast_to(o[:, None], ccol.shape)),
            hcut))
    return table


def _slab_plan(level_pairs, compact, level_ptr, *, ny, pair_budget):
    """``(k0, k1, ja, jb)`` slabs: analysis levels ``[k0, k1)``, rows ``[ja, jb)``.

    Whole levels are grouped while their candidate pairs stay inside the
    budget, exactly as before.  A single level whose candidates alone
    exceed the budget is cut into row bands, each priced exactly (per
    batch, per vertical offset, the observed rows convolved with the
    stencil's row offsets) and filled up to the budget.  The breakage this
    prevents: on the 3 km CONUS grid with a 100 km surface localisation a
    single level held about a billion candidates, and its one-slab sort
    exhausted a 32 GB RTX 5090 (std::bad_alloc in thrust argsort, measured
    2026-10-05, lane conus-da-obs).  The roster is keyed by (gridpoint,
    batch, offset slot), unique per neighbour, so where the slab
    boundaries fall does not change a bit of it.
    """
    nz = int(level_pairs.size)
    k0 = 0
    while k0 < nz:
        k1 = k0 + 1
        while k1 < nz and int(level_pairs[k0:k1 + 1].sum()) <= pair_budget:
            k1 += 1
        if k1 > k0 + 1 or int(level_pairs[k0]) <= pair_budget:
            yield k0, k1, 0, ny
            k0 = k1
            continue
        # Observed rows feeding level k0, summed per stencil (batches that
        # share a stencil share its row offsets), then correlated with the
        # stencil's row-offset multiplicities: candidates per analysis row.
        hist = {}
        for (lidx, k, j, i, dk, dj, di, hkey), ptr in zip(compact, level_ptr):
            if int(lidx.size) == 0 or int(dj.size) == 0:
                continue
            h = hist.setdefault(hkey, np.zeros(ny, dtype=np.int64))
            for d in dk.tolist():
                kk = k0 + int(d)        # analysis level = observed level - dk
                if 0 <= kk < nz and ptr[kk + 1] > ptr[kk]:
                    h += np.bincount(j[ptr[kk]:ptr[kk + 1]], minlength=ny)
        rows = np.zeros(ny, dtype=np.int64)
        for hkey, h in hist.items():
            dj = np.asarray(hkey[0], dtype=np.int64)
            vmin, vmax = int(dj.min()), int(dj.max())
            mult = np.bincount(dj - vmin)                 # m(v) at v - vmin
            # rows[g] = sum_v h[g + v] m(v), analysis row = observed row - dj
            padded = np.concatenate((np.zeros(max(0, -vmin), np.int64), h,
                                     np.zeros(max(0, vmax), np.int64)))
            # full[n] = sum_v padded[n - vmax + v] m(v), padded[t] = h[t - a]
            full = np.convolve(padded, mult[::-1])
            shift = max(0, -vmin) + vmax
            rows += full[shift:shift + ny]
        cum = np.concatenate(([0], np.cumsum(rows)))
        ja = 0
        while ja < ny:
            jb = int(np.searchsorted(cum, cum[ja] + pair_budget, side="right")) - 1
            jb = min(ny, max(ja + 1, jb))
            yield k0, k1, ja, jb
            ja = jb
        k0 = k1


def _band_segments(kj, lo, hi, *, ny, jlo, jhi):
    """Observation index ranges at levels ``[lo, hi)`` with rows in ``[jlo, jhi]``.

    A batch's observations are in ascending flat order (level major, then
    row), so ``kj = k * ny + j`` is sorted and each level's rows are one
    contiguous run.
    """
    out = []
    for kk in range(lo, hi):
        a = int(np.searchsorted(kj, kk * ny + max(0, jlo), side="left"))
        e = int(np.searchsorted(kj, kk * ny + min(ny - 1, jhi), side="right"))
        if e > a:
            out.append((a, e))
    return out


def build_neighbor_index(stencils, *, shape, zflat, horizontal_distance,
                         members, solve_dtype, xp,
                         pair_budget: int = DEFAULT_PAIR_BUDGET,
                         gaspari_cohn=None, parallel=None,
                         resident: str = "auto",
                         transient_bytes: int | None = None) -> NeighborIndex:
    """Build the index from :func:`gpuwm.da.letkf.analyze`'s stencils.

    ``stencils`` are the analysis' own per-batch records (flat host
    ``mask``/``values``/``err2``/``sim``, the batch window, the stencil
    offsets and cutoffs); ``zflat`` the host float64 gridpoint heights;
    ``horizontal_distance`` the analysis' own host metric.  ``xp`` is where
    the index is built and kept.  ``parallel(levels, pairs)``, when given,
    returns the host worker count for a numpy build (1 keeps one process).
    ``transient_bytes``, when given, caps the build's working set on a card
    (each vectorised step and each slab sort, at ``SLAB_BYTES_PER_PAIR`` per
    candidate pair): the analysis passes the scratch its admission priced.
    Breakage it prevents: the card-resident roster route (2ac908c52) sized
    its slabs from a quarter of the free memory with a 32 M pair floor,
    outside the analysis price, and the test's 10-member leg peaked at
    148.6 MB against a 115.2 MB admission (node-4, 2026-10-06); in a packed
    cycle that unpriced grab is another member's memory.  Any slab plan
    assembles the same roster (see ``_slab``), so the cap moves no byte.
    """
    if gaspari_cohn is None:
        from gpuwm.da import letkf as _letkf    # noqa: PLC0415
        gaspari_cohn = _letkf.gaspari_cohn
    t0 = time.perf_counter()
    nz, ny, nx = (int(v) for v in shape)
    plane = ny * nx
    npts = nz * plane
    solve_dtype = np.dtype(solve_dtype)
    nbatch = len(stencils)
    smax = max([1] + [int(_host(st["dk"]).size)
                      * int(_host(st["dj"]).size) for st in stencils])
    bbits = max(1, int(math.ceil(math.log2(max(2, nbatch)))))
    sbits = max(1, int(math.ceil(math.log2(max(2, smax)))))
    if int(math.ceil(math.log2(max(2, npts)))) + bbits + sbits > 62:
        raise ValueError("neighbour index key does not fit in int64")

    up = _Uploads(xp)

    # -- compact observations, batch order --------------------------------
    tables = {}
    compact = []
    offsets = []
    nobs = 0
    for st in stencils:
        mask = _host(st["mask"]).reshape(-1)
        lidx = np.flatnonzero(mask)
        nj, ni = int(st["nj"]), int(st["ni"])
        k = lidx // (nj * ni)
        rem = lidx - k * (nj * ni)
        jw = rem // ni
        iw = rem - jw * ni
        j = jw + int(st["j0"])
        i = iw + int(st["i0"])
        offsets.append(nobs)
        nobs += int(lidx.size)
        dk = _host(st["dk"]).astype(np.int64)
        dj = _host(st["dj"]).astype(np.int64)
        di = _host(st["di"]).astype(np.int64)
        hkey = (tuple(dj.tolist()), tuple(di.tolist()), float(st["hcut"]))
        tables.setdefault(hkey, []).append(j * nx + i)
        compact.append((lidx, k, j, i, dk, dj, di, hkey))
    # One horizontal table per stencil, over the columns its batches
    # observe; each observation carries its row in that table.
    table_rows = {}
    for hkey, parts in tables.items():
        ocols = np.unique(np.concatenate(parts)) if parts else np.zeros(0, np.int64)
        dj = np.asarray(hkey[0], dtype=np.int64)
        di = np.asarray(hkey[1], dtype=np.int64)
        table_rows[hkey] = ocols
        tables[hkey] = up(observed_column_weight_table(
            ocols, dj, di, hkey[2], ny=ny, nx=nx,
            horizontal_distance=horizontal_distance,
            gaspari_cohn=gaspari_cohn))
    device_trow = [up(np.searchsorted(table_rows[c[7]], c[2] * nx + c[3]))
                   if int(c[0].size) else None for c in compact]
    table_bytes = int(sum(int(t.nbytes) for t in tables.values()))
    # Observation coordinates and stencil offsets cross once per batch.
    device_coords = [None if int(c[0].size) == 0 else
                     (up(c[1]), up(c[2]), up(c[3]), up(c[4]), up(c[5]),
                      up(c[6]))
                     for c in compact]

    def gathered(name, lead):
        parts = []
        for st, c in zip(stencils, compact):
            source = st[name]
            if isinstance(source, np.ndarray):
                parts.append(up(source[..., c[0]]))
            else:
                # Already in a device namespace: gather there.
                parts.append(xp.asarray(source[..., xp.asarray(c[0])]))
        if not parts:
            return xp.zeros(lead + (0,), dtype=np.float64)
        return xp.concatenate(parts, axis=-1)

    sim_c = gathered("sim", (members,))
    values_c = gathered("values", ())
    err2_c = gathered("err2", ())

    z_dev = up(np.asarray(zflat, dtype=np.float64))
    obs_dtype = np.int32 if nobs < 2**31 - 1 else np.int64

    # -- level slabs ------------------------------------------------------
    # Per batch, the observations at each level (k is the major axis of the
    # ascending flat index, so each level is a contiguous run).
    level_ptr = [np.searchsorted(c[1], np.arange(nz + 1)) for c in compact]
    # Candidate pairs whose ANALYSIS level is gk, summed over batches and
    # vertical offsets: the observations at level gk + dk, times n_h.
    level_pairs = np.zeros(nz, dtype=np.int64)
    for (lidx, k, j, i, dk, dj, di, hkey), ptr in zip(compact, level_ptr):
        per_level = np.diff(ptr)
        for d in dk.tolist():
            lo, hi = max(0, -d), min(nz, nz - d)
            if hi > lo:
                level_pairs[lo:hi] += per_level[lo + d:hi + d] * int(dj.size)

    # Where finished slabs live.  The slab arithmetic is the same either
    # way; only the destination of each finished slab differs.
    if resident not in ("auto", "device", "host"):
        raise ValueError(f"resident must be auto, device or host, got {resident!r}")
    worst = int(level_pairs.sum()) * (np.dtype(obs_dtype).itemsize
                                      + solve_dtype.itemsize)
    free = None
    if _namespace_name(xp) != "numpy":
        free = int(xp.cuda.runtime.memGetInfo()[0])
    if resident == "auto":
        on_host = free is not None and worst > RESIDENT_FREE_FRACTION * free
    else:
        on_host = resident == "host"
    keep_slab = (lambda a: _to_host(xp, a)) if on_host else (lambda a: a)
    store = np if on_host else xp

    # A slab's sort holds every kept pair of the slab at once (key, id,
    # weight, order and their sorted copies), so on a card its size follows
    # the free memory; pair_budget still bounds each vectorised step.
    slab_pairs = int(pair_budget)
    if free is not None:
        slab_pairs = max(slab_pairs, int(SLAB_FREE_FRACTION * free)
                         // SLAB_BYTES_PER_PAIR)
        if transient_bytes is not None:
            # Half for the slab's kept pairs and their sort, half for one
            # vectorised step's candidates, which hold about sixteen
            # eight-byte temporaries each (gk, gj, gi, the masks, g, og,
            # the table row, dz and the weights) at once.
            half = max(1, int(transient_bytes) // 2)
            slab_pairs = min(slab_pairs, max(1, half // SLAB_BYTES_PER_PAIR))
            pair_budget = min(int(pair_budget),
                              max(1, half // STEP_BYTES_PER_PAIR))
    band_kj = {}

    def _slab(k0, k1, ja, jb):
        """The roster rows of analysis levels ``k0..k1``, rows ``ja..jb``.

        ``(slab_counts, oid, wt, evaluated)``, or ``None`` arrays when no
        pair lands in the slab.  Each kept pair's key (gridpoint, batch,
        slot) is unique, so the sorted rows do not depend on how the slab
        is blocked over observations, nor on where the slab or band
        boundaries fall: any slab plan assembles the same roster.
        """
        banded = (ja, jb) != (0, ny)
        g0 = k0 * plane + ja * nx
        g1 = (k1 - 1) * plane + jb * nx
        keys, oids, weights = [], [], []
        evaluated = 0
        for b, ((lidx, k, j, i, dk, dj, di, hkey), ptr, st) in enumerate(
                zip(compact, level_ptr, stencils)):
            nh = int(dj.size)
            if nh == 0 or int(lidx.size) == 0:
                continue
            table = tables[hkey]
            d_trow = device_trow[b]
            d_k, d_j, d_i, d_dk, d_dj, d_di = device_coords[b]
            vcut = float(st["vcut"])
            base = offsets[b]
            # Every vertical offset at once: the observations whose level
            # minus SOME offset lands in the slab, paired with each offset
            # that does, then with every horizontal offset.  Blocked over
            # observations so (pairs x n_h) stays inside the pair budget.
            ndk = int(dk.size)
            lo = min(nz, max(0, k0 + int(dk.min())))
            hi = min(nz, max(0, k1 + int(dk.max())))
            o0, o1 = int(ptr[lo]), int(ptr[hi])
            if o1 <= o0:
                continue
            ids = None
            if banded:
                # Only observations whose row can reach rows [ja, jb),
                # gathered into one id list so the blocks stay as large as
                # an unbanded slab's (one block per level segment was
                # millions of tiny kernel launches on the 3 km grid).
                if b not in band_kj:
                    band_kj[b] = k * ny + j
                segments = _band_segments(band_kj[b], lo, hi, ny=ny,
                                          jlo=ja + int(dj.min()),
                                          jhi=jb - 1 + int(dj.max()))
                if not segments:
                    continue
                ids = up(np.concatenate([np.arange(a, e, dtype=np.int64)
                                         for a, e in segments]))
                o0, o1 = 0, int(ids.size)
            step = max(1, pair_budget // max(1, ndk * nh))
            for s0 in range(o0, o1, step):
                s1 = min(o1, s0 + step)
                if ids is None:
                    sel = slice(s0, s1)
                else:
                    sel = ids[s0:s1]
                ok = d_k[sel]
                oj = d_j[sel]
                oi = d_i[sel]
                trow = d_trow[sel]
                gk_all = ok[:, None] - d_dk[None, :]
                ro, aa = xp.nonzero((gk_all >= k0) & (gk_all < k1))
                if int(ro.size) == 0:
                    continue
                gk = gk_all[ro, aa]
                gj = oj[ro][:, None] - d_dj[None, :]
                gi = oi[ro][:, None] - d_di[None, :]
                valid = (gj >= ja) & (gj < jb) & (gi >= 0) & (gi < nx)
                evaluated += int(gj.size)
                rows, hs = xp.nonzero(valid)
                if int(rows.size) == 0:
                    continue
                gcol = gj[rows, hs] * nx + gi[rows, hs]
                g = gk[rows] * plane + gcol
                obs_row = ro[rows]
                og = (ok[obs_row] * ny + oj[obs_row]) * nx + oi[obs_row]
                wh = table[trow[obs_row], hs]
                dz = xp.abs(z_dev[og] - z_dev[g])
                geometric = gaspari_cohn(dz, vcut) * wh
                keep = geometric > 0
                w = geometric[keep].astype(solve_dtype)
                positive = w > 0
                w = w[positive]
                if int(w.size) == 0:
                    continue
                pick = xp.nonzero(keep)[0][positive]
                slot = aa[rows[pick]] * nh + hs[pick]
                key = (((g[pick] - g0) << bbits) | b) << sbits | slot
                keys.append(key)
                if ids is None:
                    oids.append((obs_row[pick] + (base + s0)).astype(obs_dtype))
                else:
                    oids.append((sel[obs_row[pick]] + base).astype(obs_dtype))
                weights.append(w)
        if not keys:
            return None, None, None, evaluated
        key = xp.concatenate(keys)
        order = xp.argsort(key)
        key = key[order]
        oid = xp.concatenate(oids)[order]
        wt = xp.concatenate(weights)[order]
        del order
        g_local = key >> (bbits + sbits)
        # The keys are sorted, so each gridpoint's count is the gap between
        # its first key and the next one's.  Not xp.bincount: cupy routes it
        # through CUB's histogram, whose temporary storage was the build's
        # peak, 68.8 MB on a 10-member 64 x 64 x 10 leg whose analysis
        # admission is 115.2 MB in all (node-4, 2026-10-06).  Same int64
        # counts, byte for byte.
        bounds = xp.searchsorted(g_local, xp.arange(g1 - g0 + 1,
                                                    dtype=g_local.dtype))
        slab_counts = _to_host(xp, xp.diff(bounds))
        del bounds
        return slab_counts, oid, wt, evaluated

    # On the host with workers to spare, one level per slab on forked
    # workers (gpuwm.da.letkf_host_parallel): the slab plan does not change
    # a byte (see _slab), and a level is the finest slab the plan allows.
    # Otherwise the priced plan, whose over-budget levels are row bands.
    workers = 1
    if parallel is not None and _namespace_name(xp) == "numpy":
        workers = int(parallel(nz, int(level_pairs.sum())))
    if workers > 1:
        plan = [(k, k + 1, 0, ny) for k in range(nz)]
        from gpuwm.da import letkf_host_parallel as _hp   # noqa: PLC0415
        # Largest level first so the last slabs to start are short ones.
        order = sorted(range(len(plan)),
                       key=lambda n: -int(level_pairs[plan[n][0]:plan[n][1]].sum()))
        results = [None] * len(plan)
        try:
            for n, result in _hp.run(lambda n: (n, _slab(*plan[n])), order,
                                     min(workers, len(plan))):
                results[n] = result
        except (_hp.HostWorkerLost, MemoryError):
            # A lost or out-of-memory worker: build the same slabs here
            # (the slab plan does not change a byte) rather than hang or
            # fail the analysis.
            results = [None] * len(plan)
            for n in order:
                results[n] = _slab(*plan[n])
    else:
        plan = list(_slab_plan(level_pairs, compact, level_ptr, ny=ny,
                               pair_budget=slab_pairs))
        results = (_slab(*span) for span in plan)

    counts = np.zeros(npts, dtype=np.int64)
    obs_parts = []
    weight_parts = []
    evaluated = 0
    kept_total = 0
    slabs = 0
    bands = 0
    for (k0, k1, ja, jb), (slab_counts, oid, wt, slab_evaluated) in zip(
            plan, results):
        evaluated += slab_evaluated
        if oid is not None:
            g0 = k0 * plane + ja * nx
            counts[g0:g0 + int(slab_counts.size)] = slab_counts
            obs_parts.append(keep_slab(oid))
            weight_parts.append(keep_slab(wt))
            kept_total += int(oid.size)
        slabs += 1
        bands += int((ja, jb) != (0, ny))
    del results

    row_ptr = np.zeros(npts + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    # Assembled into exact-size arrays one slab at a time, each slab freed
    # as it lands: peak is the roster plus one slab, not twice the roster.
    obs = store.empty(kept_total, dtype=obs_dtype)
    weight = store.empty(kept_total, dtype=solve_dtype)
    at = 0
    while obs_parts:
        part_obs, part_weight = obs_parts.pop(0), weight_parts.pop(0)
        n = int(part_obs.size)
        obs[at:at + n] = part_obs
        weight[at:at + n] = part_weight
        at += n
        del part_obs, part_weight
    if _namespace_name(xp) != "numpy":
        xp.cuda.runtime.deviceSynchronize()
    seconds = time.perf_counter() - t0
    index = NeighborIndex(
        xp=xp, npts=npts, counts=counts, row_ptr=row_ptr, obs=obs,
        weight=weight, sim=sim_c, values=values_c, err2=err2_c,
        batch_offsets=tuple(offsets), build_seconds=seconds,
        build_upload_bytes=up.bytes, host_resident=bool(on_host))
    index.receipt = {
        "namespace": _namespace_name(xp),
        "observations": int(nobs),
        "pairs_evaluated": int(evaluated),
        "neighbours": int(kept_total),
        "slabs": int(slabs),
        "row_band_slabs": int(bands),
        "slab_pairs": int(slab_pairs),
        "horizontal_tables": len(tables),
        "horizontal_table_bytes": table_bytes,
        "bytes": index.nbytes,
        "upload_bytes": int(up.bytes),
        "build_seconds": round(seconds, 4),
        "resident": "host" if on_host else _namespace_name(xp),
        "worst_case_roster_bytes": int(worst),
        "free_bytes_at_build": free,
    }
    return index
