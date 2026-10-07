"""The NumPy transcription of WPS metgrid's masked-field chain: a test oracle.

This is the float64 host code ``gpuwm.ingest.horiz`` ran for every masked
surface field (soil moisture and temperature, snow, skin temperature and
sea ice) until the chain moved into the Rust preprocessing library
(``tools/grib1_bridge/src/wps_masked.rs``, entry
``gpuwm_wps_masked_chain_f64``).  It is kept verbatim, and ONLY as the
reference the native chain is held to byte for byte, values and repair
counts alike (``tests/test_wps_masked_chain_native.py``).

Nothing outside ``gpuwm/verify`` and ``tests`` may import it: it ran on one
core and took about a minute per forcing time on a 1792 x 1024 grid, and a
runtime that reached it would be a silent fallback to that.  A test
refuses any such import.
"""
from __future__ import annotations

import numpy as np

from gpuwm.ingest.horiz import (
    _DONOR_SPAN_TOLERANCE,
    _LAND_FIELD_IN_RANGE_SHARE,
    _LAND_QUANTITY,
    _WPS_FULL_CHAIN,
    _regular_coordinates,
    source_value_in_range,
)

_WPS_SEARCH_DEPTH = 1200
#: Target-by-candidate distances :func:`_wps_search` evaluates at once
#: (about 16 MB of float64 per table).  A memory bound only: every block
#: gives the same answers.
_WPS_SEARCH_BLOCK_CELLS = 1 << 21


def _wps_oned(x, a, b, c, d):
    """Vectorized float64 transcription of metgrid ``oned`` (interp_module.F).

    One-dimensional overlapping-parabolic interpolation with WPS's exact
    zero-value special cases: a zero ``b`` or ``c`` collapses the result to
    0 unless ``x`` is exactly 0 or 1, and a zero ``a`` or ``d`` selects the
    one-sided parabola or the linear form.
    """
    x = np.asarray(x, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    d = np.asarray(d, dtype=np.float64)
    result = np.zeros(np.broadcast(x, a, b, c, d).shape, dtype=np.float64)
    result = np.where(x == 0.0, b, result)
    result = np.where(x == 1.0, c, result)
    parab_b = b + x * (0.5 * (c - a) + x * (0.5 * (c + a) - b))
    parab_c = c + (1.0 - x) * (0.5 * (b - d) + (1.0 - x) * (0.5 * (b + d) - c))
    linear = b * (1.0 - x) + c * x
    both = (1.0 - x) * parab_b + x * parab_c
    inner = np.where(
        (a == 0.0) & (d == 0.0), linear,
        np.where(a != 0.0,
                 np.where(d != 0.0, both, parab_b),
                 parab_c))
    return np.where(b * c != 0.0, inner, result)


def _wps_sixteen_pt(field, valid, yy, xx, todo):
    """Metgrid ``sixteen_pt`` on active cells; NaN marks fall-through.

    ``field``/``valid`` are (ny,nx) float64/bool source arrays; ``yy``/``xx``
    are zero-based fractional source coordinates of the target cells.  The
    16-point overlapping parabolic requires every stencil point unmasked
    (interp_module.F:1262-1272); edge stencils clamp indices exactly like
    the Fortran (``kk``/``ll`` clipping, :1227-1241).  WPS's REAL*4 quirk of
    substituting 1e-20 for exact zeros before ``oned`` and mapping an exact
    1e-20 result back to zero (:1255-1257,1299) is transcribed as-is.
    """
    ny, nx = field.shape
    out = np.full(yy.shape, np.nan, dtype=np.float64)
    if not np.any(todo):
        return out
    i = np.floor(xx + 1.0e-5).astype(np.int64)
    j = np.floor(yy + 1.0e-5).astype(np.int64)
    xf = xx - i
    yf = yy - j
    near = (np.abs(xf) <= 1.0e-4) & (np.abs(yf) <= 1.0e-4)
    # Coincident-point branch (interp_module.F:1301-1332): take the source
    # point when usable, otherwise fall through.
    sel = todo & near
    if np.any(sel):
        jj = np.clip(j[sel], 0, ny - 1)
        ii = np.clip(i[sel], 0, nx - 1)
        ok = valid[jj, ii]
        vals = np.where(ok, field[jj, ii], np.nan)
        out[sel] = vals
    sel = todo & ~near
    if np.any(sel):
        i_s = i[sel]
        j_s = j[sel]
        stl = np.empty((4, 4) + i_s.shape, dtype=np.float64)
        all_ok = np.ones(i_s.shape, dtype=bool)
        for k in range(4):          # x offset -1..2
            kk = np.clip(i_s + (k - 1), 0, nx - 1)
            for l in range(4):      # y offset -1..2
                ll = np.clip(j_s + (l - 1), 0, ny - 1)
                value = field[ll, kk]
                all_ok &= valid[ll, kk]
                stl[k, l] = np.where(value == 0.0, 1.0e-20, value)
        a = _wps_oned(xf[sel], stl[0, 0], stl[1, 0], stl[2, 0], stl[3, 0])
        b = _wps_oned(xf[sel], stl[0, 1], stl[1, 1], stl[2, 1], stl[3, 1])
        c = _wps_oned(xf[sel], stl[0, 2], stl[1, 2], stl[2, 2], stl[3, 2])
        d = _wps_oned(xf[sel], stl[0, 3], stl[1, 3], stl[2, 3], stl[3, 3])
        value = _wps_oned(yf[sel], a, b, c, d)
        value = np.where(value == 1.0e-20, 0.0, value)
        out[sel] = np.where(all_ok, value, np.nan)
    return out


def _wps_four_pt(field, valid, yy, xx, todo, *, average):
    """Metgrid ``four_pt`` bilinear or ``four_pt_average``; NaN falls through.

    ``four_pt`` requires all four corners usable (interp_module.F:1099-1148)
    and handles integer-coordinate degeneracy exactly (:1150-1169);
    ``average_4pt`` renormalizes over the usable corners (:691-732).
    """
    ny, nx = field.shape
    out = np.full(yy.shape, np.nan, dtype=np.float64)
    if not np.any(todo):
        return out
    # Every value below belongs to one target, so only the targets still
    # waiting for an answer are evaluated: the same arithmetic per target,
    # without the rest of the grid computed and thrown away.
    yy = yy[todo]
    xx = xx[todo]
    fx = np.floor(xx).astype(np.int64)
    cx = np.ceil(xx).astype(np.int64)
    fy = np.floor(yy).astype(np.int64)
    cy = np.ceil(yy).astype(np.int64)
    fx = np.clip(fx, 0, nx - 1)
    cx = np.clip(cx, 0, nx - 1)
    fy = np.clip(fy, 0, ny - 1)
    cy = np.clip(cy, 0, ny - 1)
    v_ff = field[fy, fx]
    v_fc = field[cy, fx]
    v_cf = field[fy, cx]
    v_cc = field[cy, cx]
    ok_ff = valid[fy, fx]
    ok_fc = valid[cy, fx]
    ok_cf = valid[fy, cx]
    ok_cc = valid[cy, cx]
    if average:
        w_ff = np.where(ok_ff, 1.0, 0.0)
        w_fc = np.where(ok_fc, 1.0, 0.0)
        w_cf = np.where(ok_cf, 1.0, 0.0)
        w_cc = np.where(ok_cc, 1.0, 0.0)
        wsum = w_ff + w_fc + w_cf + w_cc
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(
                wsum > 0.0,
                (w_ff * v_ff + w_fc * v_fc + w_cf * v_cf + w_cc * v_cc)
                / np.where(wsum > 0.0, wsum, 1.0),
                np.nan)
        out[todo] = value
        return out
    all_ok = ok_ff & ok_fc & ok_cf & ok_cc
    x_int = fx == cx
    y_int = fy == cy
    lin_x = v_ff * (cx - xx) + v_cf * (xx - fx)
    lin_y = v_ff * (cy - yy) + v_fc * (yy - fy)
    bilinear = ((yy - fy) * (v_fc * (cx - xx) + v_cc * (xx - fx))
                + (cy - yy) * (v_ff * (cx - xx) + v_cf * (xx - fx)))
    value = np.where(
        x_int, np.where(y_int, v_ff, lin_y),
        np.where(y_int, lin_x, bilinear))
    out[todo] = np.where(all_ok, value, np.nan)
    return out


def _wps_wt_average(field, valid, yy, xx, todo, *, sixteen):
    """Metgrid ``wt_average_4pt``/``wt_average_16pt``; NaN falls through.

    Weights are ``max(0, 1-d)`` on the four corners (interp_module.F:776-779)
    or ``max(0, 2-d)`` on the 4x4 stencil (:1011-1024), zeroed on unusable
    points and renormalized; the 16-point form rejects stencils that would
    leave the array (:993-998) instead of clamping.
    """
    ny, nx = field.shape
    out = np.full(yy.shape, np.nan, dtype=np.float64)
    if not np.any(todo):
        return out
    # Per-target arithmetic, so only the targets still waiting are
    # evaluated (see _wps_four_pt).
    yy = yy[todo]
    xx = xx[todo]
    if sixteen:
        fx = np.floor(xx).astype(np.int64)
        fy = np.floor(yy).astype(np.int64)
        inside = (fx >= 1) & (fx <= nx - 3) & (fy >= 1) & (fy <= ny - 3)
        num = np.zeros(yy.shape, dtype=np.float64)
        den = np.zeros(yy.shape, dtype=np.float64)
        fx_c = np.clip(fx, 1, max(nx - 3, 1))
        fy_c = np.clip(fy, 1, max(ny - 3, 1))
        for dx in (-1, 0, 1, 2):
            for dy in (-1, 0, 1, 2):
                ii = fx_c + dx
                jj = fy_c + dy
                w = np.maximum(
                    0.0, 2.0 - np.sqrt((xx - ii) ** 2 + (yy - jj) ** 2))
                w = np.where(valid[jj, ii], w, 0.0)
                num += w * field[jj, ii]
                den += w
        with np.errstate(invalid="ignore", divide="ignore"):
            value = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0),
                             np.nan)
        out[todo] = np.where(inside, value, np.nan)
        return out
    fx = np.clip(np.floor(xx).astype(np.int64), 0, nx - 1)
    cx = np.clip(np.ceil(xx).astype(np.int64), 0, nx - 1)
    fy = np.clip(np.floor(yy).astype(np.int64), 0, ny - 1)
    cy = np.clip(np.ceil(yy).astype(np.int64), 0, ny - 1)
    num = np.zeros(yy.shape, dtype=np.float64)
    den = np.zeros(yy.shape, dtype=np.float64)
    for ii, jj in ((fx, fy), (fx, cy), (cx, fy), (cx, cy)):
        w = np.maximum(0.0, 1.0 - np.sqrt((xx - ii) ** 2 + (yy - jj) ** 2))
        w = np.where(valid[jj, ii], w, 0.0)
        num += w * field[jj, ii]
        den += w
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(den > 0.0, num / np.where(den > 0.0, den, 1.0),
                         np.nan)
    out[todo] = value
    return out


def _wps_search_single(field, valid, yy, xx, visited, stamp):
    """One-target transcription of metgrid ``search_extrap``.

    Four-connected FIFO breadth-first search from ``NINT(xx), NINT(yy)``
    (interp_module.F:484-563): expansion stops once the first usable point
    is DEQUEUED (that iteration still enqueues its neighbours, exactly like
    the Fortran loop body), then only points remaining IN THE QUEUE compete
    on squared Euclidean distance with a strict ``<`` (:565-607), so the
    first-found point wins ties and never-enqueued points never win even if
    globally nearer.  Neighbour order is x-1, x+1, y-1, y+1, and the depth
    counter reproduces WRF's in-place ``qdata%depth`` mutation (:521-559),
    capped by ``interp_opts`` (default 1200, :267).  Distances are float64
    here versus WPS REAL; the substitution is bounded by the FP32 final
    cast and can only differ inside FP32 rounding of a distance tie.
    Returns NaN when no usable point is reachable.
    """
    ny, nx = field.shape
    # Fortran NINT for non-negative arguments.
    ix = int(np.floor(xx + 0.5))
    jy = int(np.floor(yy + 0.5))
    if ix < 0 or ix >= nx or jy < 0 or jy >= ny:
        return np.nan
    from collections import deque
    queue = deque()
    queue.append((ix, jy, 0))
    visited[jy, ix] = stamp
    found = None
    while queue and found is None:
        i, j, depth = queue.popleft()
        if valid[j, i]:
            found = (i, j)
        dd = depth
        for ni, nj in ((i - 1, j), (i + 1, j), (i, j - 1), (i, j + 1)):
            if 0 <= ni < nx and 0 <= nj < ny and visited[nj, ni] != stamp:
                if dd < _WPS_SEARCH_DEPTH:
                    dd += 1
                    queue.append((ni, nj, dd))
                    visited[nj, ni] = stamp
    if found is None:
        return np.nan
    fi, fj = found
    best_d2 = (float(fi) - xx) ** 2 + (float(fj) - yy) ** 2
    best = field[fj, fi]
    while queue:
        i, j, _ = queue.popleft()
        if valid[j, i]:
            d2 = (float(i) - xx) ** 2 + (float(j) - yy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = field[j, i]
    return best


def _wps_search_candidates(valid_bytes, ny, nx, start):
    """The source cells metgrid's search compares from one start cell.

    The breadth-first walk of :func:`_wps_search_single` -- its queue
    order, its depth counter and the point where it stops -- is decided by
    the start cell ``NINT(x), NINT(y)``, the grid's shape and the usable
    mask, and by nothing else: the target's fractional position only
    enters the distance comparison after the walk has stopped.  So the
    walk is taken once per start cell and returns what that comparison
    runs over, in the order it runs: the first usable point dequeued,
    then every usable point still in the queue.  Flat indices
    (``row * nx + column``); ``None`` when no usable point is reachable.
    """
    from collections import deque
    cap = _WPS_SEARCH_DEPTH
    visited = bytearray(ny * nx)
    visited[start] = 1
    queue = deque(((start, 0),))
    popleft = queue.popleft
    append = queue.append
    last_column = nx - 1
    last_row = ny - 1
    found = -1
    while queue:
        flat, depth = popleft()
        row, column = divmod(flat, nx)
        dd = depth
        # Neighbour order x-1, x+1, y-1, y+1, each counted into the depth
        # as it is enqueued, exactly as _wps_search_single does.
        if column > 0 and dd < cap and not visited[flat - 1]:
            dd += 1
            append((flat - 1, dd))
            visited[flat - 1] = 1
        if column < last_column and dd < cap and not visited[flat + 1]:
            dd += 1
            append((flat + 1, dd))
            visited[flat + 1] = 1
        if row > 0 and dd < cap and not visited[flat - nx]:
            dd += 1
            append((flat - nx, dd))
            visited[flat - nx] = 1
        if row < last_row and dd < cap and not visited[flat + nx]:
            dd += 1
            append((flat + nx, dd))
            visited[flat + nx] = 1
        if valid_bytes[flat]:
            found = flat
            break
    if found < 0:
        return None
    rest = [flat for flat, _ in queue if valid_bytes[flat]]
    return np.asarray([found, *rest], dtype=np.int64)


def _wps_search(field, valid, yy, xx, todo, reach=None):
    """Metgrid ``search_extrap`` over the active cells; NaN falls through.

    Per-cell FIFO/queue-limited semantics (see :func:`_wps_search_single`);
    a global-nearest shortcut is NOT equivalent -- WPS only compares points
    already enqueued when the first usable point is dequeued.

    The walk is taken once per start cell and shared by every target that
    starts there (:func:`_wps_search_candidates`); each target then runs
    the same strict-``<`` distance comparison over the same candidates in
    the same order, which is a first-minimum ``argmin``.  Every answer is
    the one :func:`_wps_search_single` gives that target alone.  Walking
    once per TARGET is what this replaced: a 1 km grid puts some 700
    targets in each 0.25 degree source cell, and every target of an island
    a coarse source holds as sea walked the same ocean to the same land.
    Measured on a real island tile (GDAS 0.25 degree over a 230 x 230 1 km
    grid), the root forcing stage went past 30 minutes that way.

    ``reach`` is ``(limited, cells)``: a target flagged in ``limited``
    whose nearest candidate lies farther than ``cells`` source cells
    (squared distance past ``cells * cells``) gets NaN, the inland-water
    reach of ``gpuwm_wps_masked_chain_reach_f64``.
    """
    out = np.full(yy.shape, np.nan, dtype=np.float64)
    limited_t = None
    if reach is not None:
        reach2 = float(reach[1]) * float(reach[1])
    if not np.any(todo) or not np.any(valid):
        return out
    ny, nx = field.shape
    rows_t, columns_t = np.nonzero(todo)
    if reach is not None:
        limited_t = np.asarray(reach[0], dtype=bool)[rows_t, columns_t]
    target_y = np.asarray(yy, dtype=np.float64)[rows_t, columns_t]
    target_x = np.asarray(xx, dtype=np.float64)[rows_t, columns_t]
    # Fortran NINT for non-negative arguments; a start off the source grid
    # has no search, as in _wps_search_single.
    start_x = np.floor(target_x + 0.5)
    start_y = np.floor(target_y + 0.5)
    inside = ((start_x >= 0) & (start_x < nx)
              & (start_y >= 0) & (start_y < ny))
    if not np.any(inside):
        return out
    rows_t = rows_t[inside]
    columns_t = columns_t[inside]
    if limited_t is not None:
        limited_t = limited_t[inside]
    target_y = target_y[inside]
    target_x = target_x[inside]
    starts = (start_y[inside].astype(np.int64) * nx
              + start_x[inside].astype(np.int64))
    valid_bytes = np.ascontiguousarray(valid, dtype=np.uint8).tobytes()
    flat_field = np.asarray(field, dtype=np.float64).ravel()
    order = np.argsort(starts, kind="stable")
    unique_starts, first = np.unique(starts[order], return_index=True)
    bounds = np.append(first, order.size)
    for group, start in enumerate(unique_starts):
        members = order[bounds[group]:bounds[group + 1]]
        candidates = _wps_search_candidates(valid_bytes, ny, nx, int(start))
        if candidates is None:
            continue
        cand_row, cand_column = np.divmod(candidates, nx)
        cand_x = cand_column.astype(np.float64)[None, :]
        cand_y = cand_row.astype(np.float64)[None, :]
        # Bounded blocks of targets, so a fine grid over a coarse source
        # (tens of thousands of targets per start cell) never holds the
        # whole target-by-candidate distance table at once.
        block = max(1, _WPS_SEARCH_BLOCK_CELLS // candidates.size)
        for lo in range(0, members.size, block):
            part = members[lo:lo + block]
            dx = cand_x - target_x[part, None]
            dy = cand_y - target_y[part, None]
            distance = dx * dx + dy * dy
            nearest = np.argmin(distance, axis=1)
            values = flat_field[candidates[nearest]]
            if limited_t is not None:
                far = limited_t[part] & (
                    distance[np.arange(part.size), nearest] > reach2)
                values = np.where(far, np.nan, values)
            out[rows_t[part], columns_t[part]] = values
    return out


def _sixteen_pt_donor_span(field, yy, xx, targets):
    """The least and greatest source value in each target's ``sixteen_pt`` stencil.

    The same 4x4 stencil, with the same index clipping, that
    :func:`_wps_sixteen_pt` reads; a target it answered had every one of
    those points usable, so the span is its donors'.
    """
    ny, nx = field.shape
    i = np.floor(xx[targets] + 1.0e-5).astype(np.int64)
    j = np.floor(yy[targets] + 1.0e-5).astype(np.int64)
    least = np.full(i.shape, np.inf)
    greatest = np.full(i.shape, -np.inf)
    for k in range(4):
        kk = np.clip(i + (k - 1), 0, nx - 1)
        for m in range(4):
            value = field[np.clip(j + (m - 1), 0, ny - 1), kk]
            least = np.minimum(least, value)
            greatest = np.maximum(greatest, value)
    return least, greatest


def _unusable_source_within_reach(unusable, yy, xx, targets):
    """Which ``targets`` had an unusable source cell within two source cells.

    Two source cells is the reach of every operator before ``search`` in
    metgrid's chains (``wt_average_16pt`` weighs ``max(0, 2 - d)`` over the
    4x4 stencil), so a target the search answered with such a cell in reach
    lost its donors to their values, not to the land-sea mask.
    """
    ny, nx = unusable.shape
    rows = yy[targets]
    columns = xx[targets]
    base_row = np.floor(rows).astype(np.int64)
    base_column = np.floor(columns).astype(np.int64)
    hit = np.zeros(rows.shape, dtype=bool)
    for dy in (-1, 0, 1, 2):
        for dx in (-1, 0, 1, 2):
            jj = base_row + dy
            ii = base_column + dx
            inside = (jj >= 0) & (jj < ny) & (ii >= 0) & (ii < nx)
            near = inside & ((columns - ii) ** 2 + (rows - jj) ** 2 < 4.0)
            hit |= near & unusable[np.clip(jj, 0, ny - 1),
                                   np.clip(ii, 0, nx - 1)]
    found = np.zeros(yy.shape, dtype=bool)
    found[targets] = hit
    return found


def wps_masked_field_interpolate(field, latitude, longitude, target_lat,
                                 target_lon, *, source_valid, target_active,
                                 chain, fill_value, physical_range=None,
                                 tally=None, reach=None):
    """WPS metgrid masked-field interpolation chain on the host in float64.

    Transcribes metgrid's ``interp_sequence`` fall-through semantics
    (interp_module.F:304-367): each operator either produces a value or
    defers to the next; targets never produced -- including every cell
    outside ``target_active``, exactly like metgrid's landmask-restricted
    processing -- receive ``fill_value`` (process_domain fill_missing).
    ``source_valid`` folds the field's interp_mask and missing-value
    exclusions into one usable-source predicate.

    ``physical_range`` is ``(low, high)`` for a field that cannot leave
    that range, such as volumetric soil moisture (0..1).  It changes only
    values outside the range, so a field whose source and WPS result both
    stay inside it is byte-identical to WPS:

    * A source value outside the range by more than packing roundoff
      (:func:`source_value_in_range`) is not a value of the field (a fill
      value, a decode slip), so it is treated the way metgrid treats a
      missing value: it is not a donor, and every operator that would
      have used it falls through.  A value within the roundoff stays a
      donor, unchanged.
    * ``sixteen_pt`` is the one operator in metgrid's chains that is not
      a weighted mean of its donors: its overlapping parabolas swing past
      the donors on a sharp step.  On HRRR's 1.6 m soil moisture, where a
      block of dry land cells near 0.002 sits among cells near 0.30, they
      put a 1 km grid's land cells at -0.055.  A ``sixteen_pt`` value
      outside the range and outside the span of its own donors is treated
      as not produced and the target falls through to ``four_pt``,
      exactly as it does when a stencil point is masked, so it takes a
      weighted mean of the same usable source.  Every other operator is
      such a mean and cannot leave the range its donors span.
    * Donors admitted with packing roundoff past a bound (a saturated ice
      sheet stored at 1.0003) hand that roundoff on; every answer outside
      the range is left only by it, and goes on the bound.

    A deliberate divergence from metgrid, confined to those values.

    ``tally``, when given, is a mutable mapping that accumulates what the
    chain did, as counts of target cells unless named otherwise:
    ``sixteen_pt_outside_range`` (answered by a later operator instead),
    ``search`` (no source cell of the target's surface within two source
    cells, so the WPS search supplied the nearest usable one: a land-sea
    mask disagreement between the source and the target when the field
    is masked), ``search_past_unusable`` (the WPS search supplied it
    because every source cell of the surface within two source cells was
    missing its value or outside the range), ``fill`` (no operator
    answered, so ``fill_value`` stands), ``source_outside_range``
    (SOURCE values under the target's footprint that the range kept from
    being donors), and ``source_roundoff_at_bound`` (answers past a bound
    only by the packing roundoff their donors carry, put on the bound).

    Arithmetic is float64 where metgrid computes in REAL: a known
    non-bitwise substitution, bounded by the FP32 final cast -- it can
    only change a result where WPS's FP32 rounding sits exactly on a
    stencil-rejection or distance-comparison boundary.
    """
    field = np.asarray(field, dtype=np.float64)
    source_valid = np.asarray(source_valid, dtype=bool)
    if field.shape != source_valid.shape:
        raise ValueError("field and source_valid shapes differ")
    yy, xx = _regular_coordinates(latitude, longitude, target_lat, target_lon)
    target_active = np.asarray(target_active, dtype=bool)
    if target_active.shape != yy.shape:
        raise ValueError("target_active shape does not match target grid")
    usable = source_valid & np.isfinite(field)
    outside_source = None
    if physical_range is not None:
        low, high = (float(bound) for bound in physical_range)
        if not low < high:
            raise ValueError("physical_range must be (low, high) with low < high")
        outside_source = usable & ~source_value_in_range(field, low, high)
        usable = usable & ~outside_source
    # Missing source values must never enter a stencil product even at zero
    # weight (0*NaN pollutes); neutralize them outside the usable set.
    safe = np.where(usable, field, 0.0)
    result = np.full(yy.shape, np.float64(fill_value), dtype=np.float64)
    todo = target_active.copy()
    counts = {"sixteen_pt_outside_range": 0, "search": 0,
              "search_past_unusable": 0, "fill": 0,
              "source_outside_range": 0, "source_roundoff_at_bound": 0}
    if outside_source is not None and np.any(target_active) \
            and np.any(outside_source):
        # Only the source cells a target stencil can reach are this
        # domain's business; the rest of a continental grid is not.
        ny, nx = field.shape
        rows = yy[target_active]
        columns = xx[target_active]
        j0 = max(int(np.floor(rows.min())) - 1, 0)
        j1 = min(int(np.floor(rows.max())) + 3, ny)
        i0 = max(int(np.floor(columns.min())) - 1, 0)
        i1 = min(int(np.floor(columns.max())) + 3, nx)
        counts["source_outside_range"] = int(
            np.count_nonzero(outside_source[j0:j1, i0:i1]))
    for op in chain:
        if not np.any(todo):
            break
        if op == "sixteen_pt":
            got = _wps_sixteen_pt(safe, usable, yy, xx, todo)
            if physical_range is not None:
                outside = todo & np.isfinite(got) & ((got < low) | (got > high))
                if np.any(outside):
                    # Past the range AND past its own donors is the
                    # parabola's overshoot.  Past the range but inside the
                    # donors is their packing roundoff at the bound (a
                    # saturated ice sheet stored at 1.0003), which the
                    # bound repair below answers.
                    least, greatest = _sixteen_pt_donor_span(
                        safe, yy, xx, outside)
                    value = got[outside]
                    swing = _DONOR_SPAN_TOLERANCE * (high - low)
                    beyond = np.zeros(yy.shape, dtype=bool)
                    beyond[outside] = ((value < least - swing)
                                       | (value > greatest + swing))
                    counts["sixteen_pt_outside_range"] = int(
                        np.count_nonzero(beyond))
                    got = np.where(beyond, np.nan, got)
        elif op == "four_pt":
            got = _wps_four_pt(safe, usable, yy, xx, todo, average=False)
        elif op == "average_4pt":
            got = _wps_four_pt(safe, usable, yy, xx, todo, average=True)
        elif op == "wt_average_4pt":
            got = _wps_wt_average(safe, usable, yy, xx, todo, sixteen=False)
        elif op == "wt_average_16pt":
            got = _wps_wt_average(safe, usable, yy, xx, todo, sixteen=True)
        elif op == "search":
            got = _wps_search(safe, usable, yy, xx, todo, reach=reach)
        else:
            raise ValueError(f"unknown WPS interpolation operator {op!r}")
        produced = todo & np.isfinite(got)
        if op == "search" and np.any(produced):
            # A source cell of the right surface whose value is missing or
            # outside the range is not a donor; one within reach means the
            # search answered for its value, not for the land-sea mask.
            past = _unusable_source_within_reach(
                source_valid & ~usable, yy, xx, produced)
            counts["search_past_unusable"] = int(np.count_nonzero(past))
            counts["search"] = int(np.count_nonzero(produced & ~past))
        result[produced] = got[produced]
        todo &= ~produced
    counts["fill"] = int(np.count_nonzero(todo))
    if physical_range is not None:
        # What is left outside the range came from donors admitted with
        # their packing roundoff past a bound; it goes on the bound.
        answered = target_active & ~todo
        with np.errstate(invalid="ignore"):
            roundoff = answered & ((result < low) | (result > high))
        counts["source_roundoff_at_bound"] = int(np.count_nonzero(roundoff))
        result[roundoff] = np.clip(result[roundoff], low, high)
    if tally is not None:
        for key, value in counts.items():
            tally[key] = tally.get(key, 0) + value
    return result


def _land_pass_with_fractional_second_chance(
        slab, latitude, longitude, target_lat, target_lon, *,
        land_donors, partial_land_donors, target_active, chain, fill_value,
        physical_range=None, tally=None):
    """The WPS land pass, then the fraction ungrib discarded, then the fill.

    Pass one is byte-for-byte WPS: ``ungrib`` binarizes an ECMWF land-sea
    mask at the half mark (rrpr.F:869-876) and metgrid interpolates a
    masked=water field from what survives.  Where that produces a value --
    everywhere a domain shares a coastline with a source cell the flag
    calls land -- this function IS metgrid, unchanged.

    Pass two exists because the binarization is lossy in one direction
    that matters.  ERA5's mask is an area fraction on a 0.25 degree grid,
    so an island smaller than roughly half a source cell rounds to ocean
    everywhere near it; the land pass then has no donor at all, and every
    land target of a domain fine enough to RESOLVE that island takes
    METGRID.TBL fill_missing -- 0 K skin temperature, 285 K soil, 1.0 soil
    moisture.  Stock WRF papers over the first of those in real.exe
    (module_initialize_real.F:3283-3292, TSK <- TMN) and carries the other
    two into the forecast.  The fraction those cells carry is real: IFS
    integrates a land tile in any cell with LANDSEA > 0, so its soil and
    skin state there is a land state, and it is a far better initial
    condition for the island than a saturated 285 K column.

    A deliberate divergence from WPS, and a deliberately narrow one: pass
    two runs ONLY when the binarized flag marks no source land anywhere in
    the crop, which is the one situation where WPS is guaranteed to fill
    every land target.  Any domain that shares its source with real
    flagged land -- every continental case -- takes pass one alone and is
    bit-identical to before.  (The gate is the donor SET, not the
    per-target outcome, because the snow family's four_pt+average_4pt
    chain has no ``search`` and legitimately leaves cells for the fill
    even where donors are plentiful.)

    ``physical_range`` reaches both passes unchanged, and ``tally``
    accumulates both passes' counts with ``fill`` counted once, after
    pass two (see :func:`wps_masked_field_interpolate`).

    Returns ``(values, recovered)``; ``recovered`` counts what pass two
    supplied, and is zero on every WPS-identical call.
    """
    active = np.asarray(target_active, dtype=bool)
    passes: dict[str, int] = {}
    values = wps_masked_field_interpolate(
        slab, latitude, longitude, target_lat, target_lon,
        source_valid=land_donors, target_active=active,
        chain=chain, fill_value=np.nan, physical_range=physical_range,
        tally=passes)
    recovered = 0
    if (not np.any(land_donors) and np.any(active)
            and np.any(partial_land_donors)):
        starved = active & ~np.isfinite(values)
        second = wps_masked_field_interpolate(
            slab, latitude, longitude, target_lat, target_lon,
            source_valid=partial_land_donors, target_active=starved,
            chain=chain, fill_value=np.nan, physical_range=physical_range,
            tally=passes)
        supplied = starved & np.isfinite(second)
        values[supplied] = second[supplied]
        recovered = int(supplied.sum())
    if tally is not None:
        passes["fill"] = int(np.count_nonzero(active & ~np.isfinite(values)))
        for key, value in passes.items():
            tally[key] = tally.get(key, 0) + value
    return np.where(np.isfinite(values), values, fill_value), recovered


def _skin_temperature_on_both_surfaces(
        slab, latitude, longitude, target_lat, target_lon, *,
        land_donors, partial_land_donors, target_land, fill_value,
        physical_range=None, tally=None, reach=None):
    """METGRID.TBL ``masked=both`` skin temperature, with no 0 K on a surface.

    Land targets take the land pass (with its second chance) and water
    targets the source's water, each through the full chain, exactly as
    before.  The chain ends in the WPS search, which reaches the whole
    source array, so a target is left without a value only when the
    source holds no usable cell of the target's own surface at all: a
    regional crop of a coarse source over an inland domain holds no water
    for its lakes, one over open ocean no land for its islands.  WPS
    writes fill_missing there, 0 K, and the water-temperature assembly
    refused every such lake while the soil initializer refused every such
    island.

    Skin temperature is a field of the whole surface, so such a target
    takes the source's skin temperature of the other surface at the same
    place, through the same chain: a lake the source has as land takes
    that land's skin, an island it has as sea takes the sea's.  That is
    the source model's own surface state where the target lies, never
    another basin's.  Every such value is counted as ``other_surface`` in
    ``tally``, and ``fill`` counts only the land targets that still have
    nothing, which needs a source with no usable skin temperature on
    either surface.

    Returns ``(values, recovered)`` as the land pass does.

    ``reach`` is ``(limited, cells)``, the inland-water reach of the water
    pass's search: a flagged water target the search would answer only
    from source water farther than ``cells`` source cells keeps NaN (not
    the other surface's skin, not the fill), counted as ``beyond_reach``.
    An infinite or non-positive reach is no reach, as in the native entry.
    """
    asked = reach is not None
    if asked and not (np.isfinite(reach[1]) and reach[1] > 0.0):
        reach = None
    target_land = np.asarray(target_land, dtype=bool)
    land_donors = np.asarray(land_donors, dtype=bool)
    counts: dict[str, int] = {}
    land_part, recovered = _land_pass_with_fractional_second_chance(
        slab, latitude, longitude, target_lat, target_lon,
        land_donors=land_donors, partial_land_donors=partial_land_donors,
        target_active=target_land, chain=_WPS_FULL_CHAIN,
        fill_value=np.nan, physical_range=physical_range, tally=counts)
    # The water half's surface is the water-temperature assembly's, which
    # carries its own receipt, so its chain counts are not tallied here.
    water_part = wps_masked_field_interpolate(
        slab, latitude, longitude, target_lat, target_lon,
        source_valid=~land_donors, target_active=~target_land,
        chain=_WPS_FULL_CHAIN, fill_value=np.nan,
        physical_range=physical_range, reach=reach)
    combined = np.where(target_land, land_part, water_part)
    unresolved = np.zeros(target_land.shape, dtype=bool)
    if reach is not None:
        unresolved = (~target_land & np.asarray(reach[0], dtype=bool)
                      & ~np.isfinite(combined))
    other_surface = 0
    for starved, donors in ((~target_land & ~np.isfinite(combined)
                             & ~unresolved, land_donors),
                            (target_land & ~np.isfinite(combined),
                             ~land_donors)):
        if not np.any(starved):
            continue
        answer = wps_masked_field_interpolate(
            slab, latitude, longitude, target_lat, target_lon,
            source_valid=donors, target_active=starved,
            chain=_WPS_FULL_CHAIN, fill_value=np.nan,
            physical_range=physical_range)
        took = starved & np.isfinite(answer)
        combined[took] = answer[took]
        other_surface += int(np.count_nonzero(took))
    counts["fill"] = int(np.count_nonzero(
        target_land & ~np.isfinite(combined)))
    counts["other_surface"] = other_surface
    if asked:
        counts["beyond_reach"] = int(np.count_nonzero(unresolved))
    if tally is not None:
        for key, value in counts.items():
            tally[key] = tally.get(key, 0) + value
    values = np.where(np.isfinite(combined), combined, fill_value)
    return np.where(unresolved, np.nan, values), recovered


def _refuse_land_field_not_in_its_unit(
        slab, *, name, layer, bounds, fill, land_donors, partial_land_donors,
        target_active):
    """Refuse a source whose soil field is missing or not in its unit.

    A source land cell with no value, or one outside the physical range,
    is simply not a donor, and the target land near it takes the WPS
    chain's answer from the land around it.  That answers a fill value or
    a decode slip on a few cells.  It cannot answer a field that is not
    in its unit at all, soil moisture in percent or soil temperature in
    Celsius: a few of its values still lie inside the range (percent soil
    moisture on land drier than 1%), and they would be the only donors,
    so the WPS search would hand them to the whole domain's land and
    blame the land-sea mask.  So the source is judged on the share of the
    values it carries on its land that lie inside the range, and refused
    under :data:`_LAND_FIELD_IN_RANGE_SHARE`.  A source that carries the
    field on none of its land is a missing field, and WPS would write
    METGRID.TBL fill_missing on every land cell (saturated soil at 1.0, a
    285 K column): refused the same way.  A source with no land at all is
    not this case: an island the source cannot resolve keeps the
    second-chance pass and WPS's fill, as before.

    Nor is a source whose only land here is cells it calls under half
    land and which carries nothing on them.  A source that keeps a soil
    state only on the cells it calls land (a native mesh remapped to a
    regular window does, leaving the rest missing) has no land to give a
    window of small islands; that is the unresolved island above, not a
    missing field.  Named breakage: every such window was refused as
    "the source carries no soil temperature on any of its N land cell(s)"
    though the field is present wherever the source has land.  When any
    of those part-land cells carries a value, the share test still judges
    its unit.
    """
    target_active = np.asarray(target_active, dtype=bool)
    if not np.any(target_active):
        return
    part_land_only = not np.any(land_donors)
    donors = (partial_land_donors if part_land_only else land_donors)
    donors = np.asarray(donors, dtype=bool)
    land_cells = int(np.count_nonzero(donors))
    if land_cells == 0:
        return
    low, high = bounds
    values = np.asarray(slab, dtype=np.float64)[donors]
    carried = int(np.count_nonzero(np.isfinite(values)))
    inside = int(np.count_nonzero(source_value_in_range(values, low, high)))
    if carried and inside >= _LAND_FIELD_IN_RANGE_SHARE * carried:
        return
    if not carried and part_land_only:
        return
    quantity = _LAND_QUANTITY[name]
    where = "" if layer is None else f" in source layer {int(layer) + 1}"
    if not carried:
        raise ValueError(
            f"the source carries no {quantity} on any of its {land_cells} "
            f"land cell(s){where}, so there is no {quantity} to initialize "
            "this domain's land from; the field is missing, and WPS would "
            f"write METGRID.TBL fill_missing ({fill:g}) on every land cell "
            "of the domain")
    finite = values[np.isfinite(values)]
    raise ValueError(
        f"only {inside} of the {carried} {quantity} value(s) the source "
        f"carries on its land{where} lie inside {low:g}..{high:g} (its land "
        f"values span {finite.min():.6g}..{finite.max():.6g}), so the field "
        "is not in the unit its name states; mapped anyway, those "
        f"{inside} value(s) would be the only donors and the WPS search "
        "would hand them to this domain's land")


__all__ = [
    "_land_pass_with_fractional_second_chance",
    "_refuse_land_field_not_in_its_unit",
    "_skin_temperature_on_both_surfaces",
    "_wps_four_pt",
    "_wps_oned",
    "_wps_search",
    "_wps_sixteen_pt",
    "_wps_wt_average",
    "wps_masked_field_interpolate",
]
