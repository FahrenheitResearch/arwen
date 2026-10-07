"""The radial-velocity dispersion gate.

THE BREAKAGE THIS PREVENTS
--------------------------
Radial velocity updates theta and vapour through the ensemble's
cross-covariances.  Where the ensemble is under-dispersed in Vr (its
innovations far larger than its spread and the observation error explain),
those covariances are noise with a large gain, and the analysis writes
that noise into the thermodynamics.  On a storm-scale four-minute cycle (a
3 km parent and a 1 km child, one radar) the first analysis, where the
spun-up ensemble first meets the radar, put 6.44 Mt of vapour into one
box of the child in which a run with Vr kept off theta and vapour removed
0.64 Mt, and the model built storms there that no radar saw.  Keeping Vr
off theta and vapour everywhere removed the dump but starved the cycled
storm (footprint rain 0.083 against 0.396 with Vr on both).  So Vr is
withheld from theta and vapour only where its own ensemble cannot explain
its innovations.

THE RULE
--------
Per radial-velocity batch and per column of the domain: the ratio ``sum w
d^2 / sum w (ensemble variance + observation error variance)`` over the
gates of the columns within the batch's horizontal cutoff (``d`` the
innovation against the ensemble-mean H(x), ``w`` the Gaspari-Cohn weight at
the grid's nominal spacing, every level; 1 for an ensemble whose spread and
error explain its innovations).  In a batch whose ratio over ALL its gates
exceeds :data:`DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO`, a column whose
ratio exceeds :data:`DEFAULT_VELOCITY_DISPERSION_RATIO` withholds that
batch from :data:`VELOCITY_DISPERSION_FIELDS` at every level
(:class:`DispersionGate`).  :func:`withhold` then solves those fields again
without the withheld batches and takes them from that solve in the
withheld columns only; every other field and column is the joint solve's.
The columns are split by which gates hold there, with no limit on the
number of gated batches (a continental network gates one batch per radar
at its first analysis), and each of those solves is cut to the box around
its own columns and the observations within their reach, so it costs what
its columns cost and not what the domain does.

Every analysis records the ratio distribution per batch, gate on or off
(:func:`velocity_dispersion`'s receipt), so the thresholds keep being
checked against the cycles that run them.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable, Mapping, Sequence

import numpy as np

DISPERSION_SCHEMA = "gpuwm-da.velocity-dispersion.v1"

#: The column ratio above which a gated batch is withheld from theta and
#: vapour.  Measured on CPU replays of the first child analysis of the cycle
#: the module docstring describes: withheld above 4 the dump box still
#: gains 2.59 Mt of vapour, above 3 0.25 Mt, above 2 it loses 0.79 Mt, as
#: clean as keeping Vr off both everywhere (-0.64 Mt).  In the cycled
#: ensembles after the first analysis the column ratio has median 0.68 to
#: 0.88 and 99th percentile 2.1 to 5.1.
DEFAULT_VELOCITY_DISPERSION_RATIO = 2.0

#: The batch condition: a Vr batch is gated at all only when its ratio over
#: all its gates exceeds this.  The column test alone, once cycled,
#: withheld Vr from theta and vapour in 21 to 57 percent of the observed
#: storm's 35 dBZ columns in the storm hour.  The batch ratio separates the
#: dump from the storm: 4.68 (child) and 3.72 (parent) at the first
#: analysis, at most 1.27 (child) and 1.97 (parent) at every cycled one
#: after it.  3 sits between: every first-analysis batch is gated and no
#: cycled batch is.
DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO = 3.0

#: The fields a withheld batch may not update: the two the dump moved.
VELOCITY_DISPERSION_FIELDS = ("thp", "qv")

#: The ratios whose exceedance every receipt counts, per batch.
DISPERSION_RATIO_LADDER = (1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0)


class DispersionGateError(ValueError):
    """A dispersion-gate setting or gate the rule cannot mean."""


def is_velocity_batch(name: str) -> bool:
    """A radial-velocity batch, by its name (``vr:<site>``, window slots
    add ``@label``)."""

    from gpuwm.da.obs_radar import VELOCITY_PREFIX

    return str(name).split("@", 1)[0].split(":", 1)[0] == VELOCITY_PREFIX


def check_ratio(value, label: str):
    """``None`` (off) or a finite positive ratio; anything else refused."""

    if value is None:
        return None
    try:
        threshold = float(value)
    except (TypeError, ValueError):
        threshold = float("nan")
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise DispersionGateError(
            f"the velocity dispersion {label} must be finite and positive, "
            f"or None to switch it off; got {value!r}")
    return threshold


@dataclass(frozen=True)
class DispersionGate:
    """One observation batch withheld from ``fields`` in ``columns``
    (``(ny, nx)`` bool, every level)."""

    batch: str
    fields: tuple[str, ...]
    columns: object
    #: Fraction of the joint solve kept in the gated columns: the gated
    #: fields there are ``keep * joint + (1 - keep) * withheld``.  0, the
    #: default, withholds the batch entirely.  Every gate holding at one
    #: column must state the same keep (one blend of two solves).
    keep: float = 0.0

    def __post_init__(self) -> None:
        keep = float(self.keep)
        if not np.isfinite(keep) or not 0.0 <= keep < 1.0:
            raise DispersionGateError(
                f"{self.batch}: keep is a fraction in [0, 1), got "
                f"{self.keep!r}; 1 would keep the joint solve, which is no "
                "gate at all")
        if not isinstance(self.fields, tuple) or not self.fields:
            raise DispersionGateError(
                f"{self.batch}: a dispersion gate withholds a non-empty "
                "tuple of fields")
        if np.ndim(self.columns) != 2:
            raise DispersionGateError(
                f"{self.batch}: a dispersion gate's columns are (ny, nx), "
                f"got shape {np.shape(self.columns)}")

    def payload(self) -> dict:
        out = {"batch": self.batch, "fields": list(self.fields),
               "columns": int(np.count_nonzero(np.asarray(self.columns)))}
        if float(self.keep):
            out["keep"] = float(self.keep)
        return out


def _zone_keep(withheld) -> float:
    """The one keep of a zone's gates, or a refusal when they differ."""
    keeps = {float(gate.keep) for gate in withheld}
    if len(keeps) != 1:
        raise DispersionGateError(
            f"gates {sorted(gate.batch for gate in withheld)} hold at the "
            f"same columns with different keeps {sorted(keeps)}; a zone "
            "blends one joint and one withheld solve, so one keep")
    return keeps.pop()


def _host(array):
    return array.get() if hasattr(array, "__cuda_array_interface__") \
        else np.asarray(array)


def _stencil(dx_m: float, dy_m: float, cutoff_m: float):
    """``(dj, di, weight)``: the offsets inside the horizontal cutoff and
    their Gaspari-Cohn weights, on the grid's nominal spacing."""

    from gpuwm.da.letkf import gaspari_cohn

    hy = int(np.floor(cutoff_m / float(dy_m)))
    hx = int(np.floor(cutoff_m / float(dx_m)))
    dj, di = np.mgrid[-hy:hy + 1, -hx:hx + 1]
    weight = gaspari_cohn(np.hypot(di * float(dx_m), dj * float(dy_m)),
                          float(cutoff_m))
    keep = weight > 0.0
    return dj[keep], di[keep], weight[keep]


def _neighbourhood_sum(field, dj, di, weight):
    """``out[..., j, i] = sum_s weight_s * field[..., j + dj_s, i + di_s]``
    over the offsets that stay inside the array."""

    out = np.zeros_like(field)
    ny, nx = field.shape[-2:]
    for a, b, w in zip(dj.tolist(), di.tolist(), weight.tolist()):
        tj = slice(max(0, -a), min(ny, ny - a))
        ti = slice(max(0, -b), min(nx, nx - b))
        sj = slice(max(0, a), min(ny, ny + a))
        si = slice(max(0, b), min(nx, nx + b))
        out[..., tj, ti] += w * field[..., sj, si]
    return out


def _batch_dispersion(batch, *, dx_m, dy_m, cutoff_m, shape,
                      prior_inflation=1.0):
    """One velocity batch's column ratio on the domain, and its receipt.

    Per gate: the innovation ``d = y - mean_k H(x_k)``, the ensemble
    variance of ``H(x_k)`` (ddof 1, times the filter's prior inflation, the
    background covariance the transform uses) and the observation error
    variance (the batch as the filter receives it).  Per column: the sums
    of ``d^2`` and of the two variances over the column's gates, gathered
    over the columns within the batch's horizontal cutoff with Gaspari-Cohn
    weights, and their ratio.  Returns ``(ratio, entry)``; ``ratio`` is
    ``(ny, nx)`` with NaN where no gate is within the cutoff, or ``None``
    when the batch holds no gate.
    """

    nz, ny, nx = (int(n) for n in shape)
    mask = _host(batch.mask).astype(bool)
    count = int(np.count_nonzero(mask))
    entry = {"batch": batch.name, "observations": count,
             "horizontal_m": float(cutoff_m)}
    if not count:
        return None, entry
    y = _host(batch.values)[mask].astype(np.float64)
    sim = _host(batch.simulated)[:, mask].astype(np.float64)
    errors = _host(batch.errors)
    err = (np.full(y.shape, float(errors)) if np.ndim(errors) == 0
           else np.broadcast_to(errors, mask.shape)[mask].astype(np.float64))
    d2 = (y - sim.mean(axis=0)) ** 2
    ens = (float(prior_inflation) * sim.var(axis=0, ddof=1)
           if sim.shape[0] > 1 else np.zeros_like(y))
    expected = ens + err ** 2
    window = getattr(batch, "window", None)
    j0, i0 = (0, 0) if window is None else (int(window[0]), int(window[2]))
    wnj, wni = mask.shape[1], mask.shape[2]
    _, jj, ii = np.nonzero(mask)
    flat = jj * wni + ii
    per_column = np.zeros((3, wnj * wni))
    per_column[0] = np.bincount(flat, weights=d2, minlength=wnj * wni)
    per_column[1] = np.bincount(flat, weights=expected, minlength=wnj * wni)
    per_column[2] = np.bincount(flat, minlength=wnj * wni)
    per_column = per_column.reshape(3, wnj, wni)
    dj, di, weight = _stencil(dx_m, dy_m, cutoff_m)
    hy, hx = int(np.abs(dj).max()), int(np.abs(di).max())
    b0, b1 = max(0, j0 - hy), min(ny, j0 + wnj + hy)
    c0, c1 = max(0, i0 - hx), min(nx, i0 + wni + hx)
    box = np.zeros((2, b1 - b0, c1 - c0))
    box[:, j0 - b0:j0 - b0 + wnj, i0 - c0:i0 - c0 + wni] = per_column[:2]
    gathered = _neighbourhood_sum(box, dj, di, weight)
    ratio = np.full((ny, nx), np.nan)
    inside = gathered[1] > 0.0
    local = np.full(inside.shape, np.nan)
    local[inside] = gathered[0][inside] / gathered[1][inside]
    ratio[b0:b1, c0:c1] = local
    observed = per_column[2] > 0
    at_observed = ratio[j0:j0 + wnj, i0:i0 + wni][observed]
    gates_at = per_column[2][observed]
    entry.update({
        "innovation_variance": float(d2.mean()),
        "ensemble_variance": float(ens.mean()),
        "error_variance": float((err ** 2).mean()),
        "batch_ratio": float(d2.sum() / expected.sum()),
        "observed_columns": int(at_observed.size),
        "column_ratio": {
            "p10": float(np.percentile(at_observed, 10)),
            "p50": float(np.percentile(at_observed, 50)),
            "p90": float(np.percentile(at_observed, 90)),
            "p99": float(np.percentile(at_observed, 99)),
            "max": float(at_observed.max())},
        "columns_above": {f"{rung:g}": int(np.count_nonzero(at_observed > rung))
                          for rung in DISPERSION_RATIO_LADDER},
        "observations_above": {
            f"{rung:g}": int(gates_at[at_observed > rung].sum())
            for rung in DISPERSION_RATIO_LADDER},
    })
    return ratio, entry


def velocity_dispersion(batches, *, dx_m: float, dy_m: float,
                        localization, shape,
                        ratio: float | None = DEFAULT_VELOCITY_DISPERSION_RATIO,
                        fields: Sequence[str] = VELOCITY_DISPERSION_FIELDS,
                        prior_inflation: float = 1.0,
                        batch_ratio: float | None =
                        DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO):
    """``(gates, receipt)``: the dispersion gates of an analysis's
    radial-velocity batches and the receipt of their ratios.

    Every velocity batch holding a gate gets its column ratio (over its own
    horizontal cutoff, else ``localization``'s).  With ``ratio`` set, a
    batch whose batch ratio exceeds ``batch_ratio`` is withheld from
    ``fields`` in the columns where its column ratio exceeds ``ratio``;
    ``batch_ratio=None`` gates every batch on its columns alone, and
    ``ratio=None`` switches the gate off and records the ratios all the
    same.
    """

    ratio = check_ratio(ratio, "ratio")
    batch_ratio = check_ratio(batch_ratio, "batch ratio")
    fields = tuple(fields)
    gates, entries = [], []
    for batch in batches:
        if not is_velocity_batch(batch.name):
            continue
        spec = batch.localization if batch.localization is not None \
            else localization
        local, entry = _batch_dispersion(
            batch, dx_m=dx_m, dy_m=dy_m, cutoff_m=float(spec.horizontal_m),
            shape=shape, prior_inflation=prior_inflation)
        if local is not None and ratio is not None:
            columns = np.nan_to_num(local, nan=0.0) > ratio
            gated = batch_ratio is None or entry["batch_ratio"] > batch_ratio
            entry["batch_gated"] = bool(gated)
            entry["columns_above_gate"] = int(np.count_nonzero(columns))
            entry["withheld_columns"] = (int(np.count_nonzero(columns))
                                         if gated else 0)
            if gated and columns.any():
                gates.append(DispersionGate(batch=batch.name, fields=fields,
                                            columns=columns))
        entries.append(entry)
    receipt = {
        "schema": DISPERSION_SCHEMA,
        "ratio": ratio,
        "batch_ratio_gate": batch_ratio,
        "fields": list(fields),
        "rule": ("per radial-velocity batch and column: sum w d^2 / sum w "
                 "(ensemble variance + observation error variance) over the "
                 "gates of the columns within the batch's horizontal cutoff, "
                 "w the Gaspari-Cohn weight at the grid's nominal spacing, "
                 "every level; in a batch whose sum d^2 / sum (ensemble "
                 "variance + observation error variance) over all its gates "
                 "exceeds the batch ratio gate (every batch when that is "
                 "None), a column above the ratio withholds the batch from "
                 "the fields at every level"),
        "ladder": [float(rung) for rung in DISPERSION_RATIO_LADDER],
        "batches": entries,
        "withheld_columns": sum(entry.get("withheld_columns", 0)
                                for entry in entries),
    }
    return gates, receipt


def _zones(gates, ny: int, nx: int):
    """``(codes, inverse, gated)``: the columns split by which gates hold
    there, with no limit on the number of gates.

    Gate ``g`` is bit ``7 - g % 8`` of byte ``g // 8`` of a column's code,
    the layout ``np.packbits`` gives an ``(n_gates, ny*nx)`` bool stack
    along its gate axis; it is packed gate by gate so the unpacked stack
    is never formed.  ``gated`` is the flat index of every column at least
    one gate holds, ``codes`` ``(n_bytes, n_zones)`` the distinct codes
    among them and ``inverse`` each gated column's zone.
    """

    packed = np.zeros(((len(gates) + 7) // 8, ny * nx), dtype=np.uint8)
    for index, gate in enumerate(gates):
        if np.shape(gate.columns) != (ny, nx):
            raise DispersionGateError(
                f"{gate.batch}: gate columns {np.shape(gate.columns)} against "
                f"{(ny, nx)}; every gate of one analysis is on one grid")
        bit = np.uint8(1 << (7 - index % 8))
        packed[index // 8][
            np.asarray(_host(gate.columns), dtype=bool).reshape(-1)] |= bit
    gated = np.flatnonzero(packed.any(axis=0))
    if not gated.size:
        return (np.zeros((packed.shape[0], 0), np.uint8),
                np.zeros(0, np.int64), gated)
    codes, inverse = np.unique(packed[:, gated], axis=1, return_inverse=True)
    return codes, np.asarray(inverse).reshape(-1), gated


def _dilate(where, dj, di):
    """``where`` ``(ny, nx)`` dilated by the offsets ``(dj, di)``: every
    column that is ``c + (dj_s, di_s)`` for a column ``c`` of ``where``.

    Worked in the box of ``where`` padded by the offsets' reach, one row
    offset at a time: a Gaspari-Cohn disc's column offsets at one row
    offset are one contiguous run, so each row is a running window over a
    cumulative sum (a row whose offsets are not one run is shifted offset
    by offset)."""

    ny, nx = where.shape
    rows = np.flatnonzero(where.any(axis=1))
    cols = np.flatnonzero(where.any(axis=0))
    out = np.zeros((ny, nx), dtype=bool)
    if not rows.size:
        return out
    dj = np.asarray(dj, dtype=np.int64)
    di = np.asarray(di, dtype=np.int64)
    hy, hx = int(np.abs(dj).max()), int(np.abs(di).max())
    j0, j1 = max(0, int(rows[0]) - hy), min(ny, int(rows[-1]) + hy + 1)
    i0, i1 = max(0, int(cols[0]) - hx), min(nx, int(cols[-1]) + hx + 1)
    src = where[j0:j1, i0:i1]
    box = out[j0:j1, i0:i1]
    bj, bi = src.shape
    running = np.zeros((bj, bi + 1), dtype=np.int64)
    np.cumsum(src, axis=1, out=running[:, 1:])
    x = np.arange(bi)
    for a in np.unique(dj).tolist():
        offsets = np.sort(di[dj == a])
        lo, hi = int(offsets[0]), int(offsets[-1])
        if offsets.size == hi - lo + 1:
            # reached[j, x]: some src[j, i] with lo <= x - i <= hi
            first = np.clip(x - hi, 0, bi)
            last = np.clip(x - lo + 1, 0, bi)
            reached = (running[:, last] - running[:, first]) > 0
        else:
            reached = np.zeros((bj, bi), dtype=bool)
            for b in offsets.tolist():
                if b >= 0:
                    reached[:, b:] |= src[:, :bi - b]
                else:
                    reached[:, :bi + b] |= src[:, -b:]
        if a >= 0:
            box[a:] |= reached[:bj - a]
        else:
            box[:bj + a] |= reached[-a:]
    return out


def _local_batches(batches, where, geometry, localization, stencils):
    """``(batches, dropped, box)``: each batch with its mask ANDed with the
    columns within its horizontal cutoff of ``where``; a batch left with
    no observation is dropped.  ``box`` ``(j0, j1, i0, i1)`` bounds every
    such reach (so ``where`` and every observation kept); with no batch
    at all it bounds ``where`` itself, since a column no observation
    reaches reads only its own prior.

    An analysis column takes weight only from observations inside its
    batch's horizontal cutoff (Gaspari-Cohn is zero at and past it), so in
    ``where`` the solve on these batches is the solve on the whole
    batches.  The reach is :func:`gpuwm.da.letkf._horizontal_stencil`,
    the filter's own superset of the offsets that carry weight anywhere on
    ``geometry``; ``stencils`` caches it per cutoff across zones."""

    from dataclasses import replace

    from gpuwm.da.letkf import _horizontal_stencil

    ny, nx = where.shape
    reach, box = {}, {}
    kept, dropped = [], 0
    for batch in batches:
        spec = batch.localization if batch.localization is not None \
            else localization
        key = float(spec.horizontal_m)
        if key not in reach:
            if key not in stencils:
                stencils[key] = _horizontal_stencil(spec, geometry, nx, ny)
            reach[key] = _dilate(where, *stencils[key])
            rows = np.flatnonzero(reach[key].any(axis=1))
            cols = np.flatnonzero(reach[key].any(axis=0))
            box[key] = (int(rows[0]), int(rows[-1]), int(cols[0]),
                        int(cols[-1]))
        window = getattr(batch, "window", None)
        j0, j1, i0, i1 = box[key]
        if window is not None and (
                int(window[1]) < j0 or int(window[0]) > j1
                or int(window[3]) < i0 or int(window[2]) > i1):
            # a window that misses the zone's reach costs no array work
            dropped += 1
            continue
        mask = _host(batch.mask).astype(bool)
        region = (reach[key] if window is None else
                  reach[key][int(window[0]):int(window[1]) + 1,
                             int(window[2]):int(window[3]) + 1])
        local = mask & region[None, :, :]
        count = int(np.count_nonzero(local))
        if not count:
            dropped += 1
            continue
        if count == int(np.count_nonzero(mask)):
            kept.append(batch)
        elif getattr(batch, "points", None) is not None:
            # A point batch's PointSet must shrink with its mask, or the
            # observation-sparse route would assimilate the dropped points.
            kept.append(_point_subset(batch, local, mask.shape, (0, 0)))
        else:
            kept.append(replace(batch, mask=local))
    if not box:
        rows = np.flatnonzero(where.any(axis=1))
        cols = np.flatnonzero(where.any(axis=0))
        return kept, dropped, (int(rows[0]), int(rows[-1]), int(cols[0]),
                               int(cols[-1]))
    edges = np.array(list(box.values()))
    return kept, dropped, (int(edges[:, 0].min()), int(edges[:, 1].max()),
                           int(edges[:, 2].min()), int(edges[:, 3].max()))


def _point_subset(batch, keep, full_shape, origin, *, shape=None):
    """``batch``'s points where ``keep`` (a ``full_shape`` boolean) holds,
    as a new point batch on ``shape`` (default ``full_shape``) whose cells
    are offset by ``origin = (j0, i0)``.  Values, errors and H(x) are the
    same numbers; only the indexing changes."""

    from gpuwm.da.letkf import PointSet, point_batch

    pts = batch.points
    flat = np.asarray(pts.flat_index, np.int64)
    k, j, i = np.unravel_index(flat, tuple(int(v) for v in full_shape))
    sel = np.asarray(keep, dtype=bool)[k, j, i]
    shape = tuple(int(v) for v in (shape or full_shape))
    local = np.ravel_multi_index(
        (k[sel], j[sel] - int(origin[0]), i[sel] - int(origin[1])), shape)
    subset = PointSet(
        flat_index=local,
        values=np.asarray(pts.values, np.float64)[sel],
        errors=np.asarray(pts.errors, np.float64)[sel],
        simulated=np.asarray(pts.simulated, np.float64)[:, sel])
    return point_batch(batch.name, shape, subset,
                       localization=batch.localization)


def _crop(prior, batches, geometry, box):
    """``(prior, batches, geometry)`` cut to the box ``(j0, j1, i0, i1)``.

    Every batch must hold its observations inside the box
    (:func:`_local_batches` guarantees it); its arrays are cut to where its
    extent meets the box and its window restated in the box's indices.  A
    column's analysis reads only the prior at that column and the
    observations within its cutoff, so in the columns the box holds the
    cut solve is the solve on the whole grid, at the box's cost."""

    from dataclasses import replace

    from gpuwm.da.letkf import GridGeometry

    j0, j1, i0, i1 = box
    sub_prior = {name: value[..., j0:j1 + 1, i0:i1 + 1].copy()
                 for name, value in prior.items()}
    sub_batches = []
    for batch in batches:
        mask_shape = np.shape(batch.mask)
        if getattr(batch, "points", None) is not None and batch.window is None:
            # A point batch is whole-grid flat indices.  Restating it with a
            # window (the dense cut below) handed the device solver a point
            # batch with a window, which it refuses because the indices would
            # land in the wrong cells: every conventional/surface point batch
            # in a dispersion-gated zone stopped the cycle (Iowa da-obs-iau,
            # 2026-10-06).  Re-index the points into the box instead.
            nz = mask_shape[0]
            cut = np.zeros(mask_shape, dtype=bool)
            cut[:, j0:j1 + 1, i0:i1 + 1] = True
            sub_batches.append(_point_subset(
                batch, cut, mask_shape, (j0, i0),
                shape=(nz, j1 - j0 + 1, i1 - i0 + 1)))
            continue
        wj0, wj1, wi0, wi1 = (tuple(int(v) for v in batch.window)
                              if batch.window is not None else
                              (0, mask_shape[-2] - 1, 0, mask_shape[-1] - 1))
        cj0, cj1 = max(wj0, j0), min(wj1, j1)
        ci0, ci1 = max(wi0, i0), min(wi1, i1)
        cut = (Ellipsis, slice(cj0 - wj0, cj1 - wj0 + 1),
               slice(ci0 - wi0, ci1 - wi0 + 1))
        errors = batch.errors
        if np.ndim(errors):
            errors = errors[cut]
        sub_batches.append(replace(
            batch, values=batch.values[cut], errors=errors,
            simulated=batch.simulated[cut], mask=batch.mask[cut],
            window=(cj0 - j0, cj1 - j0, ci0 - i0, ci1 - i0)))
    heights = geometry.heights_m
    if np.ndim(heights) == 3:
        heights = heights[:, j0:j1 + 1, i0:i1 + 1]
    lat, lon = geometry.lat_deg, geometry.lon_deg
    if lat is not None:
        lat = lat[j0:j1 + 1, i0:i1 + 1]
        lon = lon[j0:j1 + 1, i0:i1 + 1]
    sub_geometry = GridGeometry(
        dx_m=geometry.dx_m, dy_m=geometry.dy_m, heights_m=heights,
        lat_deg=lat, lon_deg=lon, earth_radius_m=geometry.earth_radius_m)
    return sub_prior, sub_batches, sub_geometry


def withhold(solve: Callable[[Mapping, list, tuple, object], Mapping], prior,
             batches, increments, gates: Sequence[DispersionGate],
             analysis_fields, *, geometry, localization):
    """``(increments, receipt)`` with the gated batches withheld from their
    fields in their columns.

    ``solve(prior, batches, fields, geometry)`` runs the filter on
    ``batches`` for ``fields`` over the grid ``geometry`` describes and
    returns their increments ``{field: (R, nz, ny, nx)}`` on the prior's
    grid.  The columns are split into zones by which gates withhold there
    (:func:`_zones`: one zone per distinct set, any number of gates); each
    zone's fields are solved once without its batches and written into
    that zone only.  Each such solve is local: every batch it keeps is cut
    to the columns within its horizontal cutoff of the zone
    (:func:`_local_batches` on ``geometry``, the filter's grid, and
    ``localization``, the analysis's default cutoff), and the prior, the
    batches and the grid are cut to the box around that reach
    (:func:`_crop`), so its cost follows the zone and not the domain, and
    in the zone it is the whole-domain solve.  A zone no kept batch
    reaches, whether batches are left elsewhere or none is, is solved on
    no observation over the box around the zone: the filter's transform of
    an unobserved column, zero at prior inflation 1 and not otherwise.
    The joint solve's increments stand everywhere else and for every other
    field.  No gate: the increments come back unchanged, with no solve.
    """

    receipt = {"gates": [gate.payload() for gate in gates], "solves": []}
    if not gates:
        return increments, receipt
    fields = tuple(name for name in analysis_fields
                   if any(name in gate.fields for gate in gates))
    if not fields:
        return increments, receipt
    ny, nx = np.shape(gates[0].columns)
    codes, inverse, gated = _zones(gates, ny, nx)
    order = np.argsort(inverse, kind="stable")
    bounds = np.searchsorted(inverse[order], np.arange(codes.shape[1] + 1))
    # Only the fields a gate rewrites are copied; every other field's
    # increment is passed through as the same array.  Copying all of them
    # held a second whole-ensemble float64 copy of every analysed field on
    # the host for the length of the gated solves.
    out = {name: (np.array(_host(value), copy=True) if name in fields
                  else value)
           for name, value in increments.items()}
    stencils = {}
    for zone in range(codes.shape[1]):
        held = np.unpackbits(codes[:, zone], count=len(gates)).astype(bool)
        withheld = [gate for gate, on in zip(gates, held) if on]
        names = {gate.batch for gate in withheld}
        these = tuple(name for name in fields
                      if any(name in gate.fields for gate in withheld))
        flat = gated[order[bounds[zone]:bounds[zone + 1]]]
        where = np.zeros(ny * nx, dtype=bool)
        where[flat] = True
        where = where.reshape(ny, nx)
        started = time.perf_counter()
        candidates = [batch for batch in batches if batch.name not in names]
        kept, dropped, box = _local_batches(candidates, where, geometry,
                                            localization, stencils)
        # No batch within reach of the zone, whether others are left in the
        # domain or none is: the solve still runs, on no observation,
        # because the filter's transform of an unobserved column is not
        # zero under prior inflation, and the zone must hold what the
        # whole-domain solve without its batches holds there.
        j0, j1, i0, i1 = box
        sub_prior, sub_batches, sub_geometry = _crop(
            {name: prior[name] for name in these}, kept, geometry, box)
        solved = solve(sub_prior, sub_batches, these, sub_geometry)
        inside = where[j0:j1 + 1, i0:i1 + 1]
        keep = _zone_keep(withheld)
        for name in these:
            value = _host(solved[name])[..., inside]
            if keep:
                # keep * joint + (1 - keep) * withheld, the device's order.
                value = keep * out[name][..., where] + (1.0 - keep) * value
            out[name][..., where] = value
        receipt["solves"].append({
            "withheld": sorted(names), "fields": list(these),
            "columns": int(flat.size),
            "keep": keep,
            "box": [int(v) for v in box],
            "batches_kept": len(kept),
            "batches_out_of_reach": dropped,
            "wall_seconds": round(time.perf_counter() - started, 3)})
    return out, receipt


def column_plan(gates: Sequence[DispersionGate], analysis_fields):
    """``(fields, column_zone, zones, receipt)`` for withholding the gates in
    the filter's own pass, or ``None`` when :func:`withhold` must do it.

    The zones are :func:`withhold`'s (:func:`_zones`): one per distinct set
    of gates holding at a column.  ``column_zone`` is ``(ny * nx,)`` int32,
    each column's zone or -1; ``zones[z]`` the batch names zone ``z``
    withholds.  At a point of a zoned column, a solve that skips exactly
    those batches is the whole-domain solve without them -- the solve
    :func:`withhold` cuts to the zone's box and reach -- because a batch the
    point does not read adds no term to anything the point computes.

    ``None`` when no gate rewrites an analysed field, or when the gates
    name different field sets (one pass re-solves one field set).
    """

    if not gates:
        return None
    fields = tuple(name for name in analysis_fields
                   if any(name in gate.fields for gate in gates))
    if not fields:
        return None
    if len({tuple(gate.fields) for gate in gates}) != 1:
        return None
    ny, nx = np.shape(gates[0].columns)
    codes, inverse, gated = _zones(gates, ny, nx)
    column_zone = np.full(ny * nx, -1, dtype=np.int32)
    column_zone[gated] = inverse.astype(np.int32)
    zones, receipt = [], []
    counts = np.bincount(inverse, minlength=codes.shape[1])
    for zone in range(codes.shape[1]):
        held = np.unpackbits(codes[:, zone], count=len(gates)).astype(bool)
        names = tuple(sorted({gate.batch for gate, on in zip(gates, held)
                              if on}))
        zones.append(names)
        entry = {"withheld": list(names), "fields": list(fields),
                 "columns": int(counts[zone]),
                 "keep": _zone_keep([gate for gate, on in zip(gates, held)
                                     if on])}
        receipt.append(entry)
    return fields, column_zone, tuple(zones), receipt


__all__ = ["DEFAULT_VELOCITY_DISPERSION_BATCH_RATIO",
           "DEFAULT_VELOCITY_DISPERSION_RATIO", "DISPERSION_RATIO_LADDER",
           "DISPERSION_SCHEMA", "DispersionGate", "DispersionGateError",
           "VELOCITY_DISPERSION_FIELDS", "check_ratio", "column_plan",
           "is_velocity_batch",
           "velocity_dispersion", "withhold"]
