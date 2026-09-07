"""Conservative vertical remapping between two eta ladders on one column.

An offline child may carry a deeper vertical ladder than the archived parent
it is built from (a 100 m LES child off a 1 km parent wants the levels, not
just the columns).  Moving a state between two ladders has to conserve dry
mass and every water substance it carries, or the child starts from an
atmosphere that holds a different amount of water than the parent did.

The coordinate makes that conservation exact rather than approximate.  WRF's
hybrid coordinate defines the reference dry pressure at an interface as
``pd_f[k] = c3f[k]*mu + c4f[k] + p_top`` (``gpuwm/core/grid.py`` ::
``compute_hybrid_coeffs``).  Because ``c3f[0] = 1``, ``c4f[0] = 0``,
``c3f[nz] = 0`` and ``c4f[nz] = 0`` for every admissible ladder, that
expression collapses at both ends::

    pd_f[0]  = mu + p_top = p_s        (surface)
    pd_f[nz] = p_top                   (model top)

So two ladders that share ``p_top``, ``hybrid_opt`` and ``etac`` partition
*exactly the same* physical mass interval ``[0, mu]``, with coincident
endpoints and no extrapolation anywhere.  Remapping is then rebinning of an
integral between two partitions of one interval -- exact by construction, not
exact to a tolerance.  Those three parameters are therefore required to match
(:func:`require_shared_column_basis`); allowing a per-domain ``p_top`` would
break the endpoint identity this module rests on.

Layer dry mass is ``dm[k] = -dnw[k]*(c1h[k]*mu + c2h[k])``, which is the same
weight ``gpuwm/offline_child.py`` :: ``_couple_parent`` already forms as
``chm`` and the same one ``gpuwm/core/grid.py`` uses in its discrete
hydrostatic recurrence.  This module computes it through
``compute_hybrid_coeffs`` so it is the tree's own arithmetic and not a second
transcription of it.

Everything here is host NumPy in float64 and runs once at prepare time.  The
integration loop is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gpuwm.core import constants as c
from gpuwm.core.grid import compute_hybrid_coeffs


class VerticalRemapRefusal(ValueError):
    """A remap that would not conserve, or a coordinate that is folded."""


@dataclass(frozen=True)
class ColumnBasis:
    """The three coordinate parameters two ladders must share.

    Sharing them is what makes ``pd_f`` span exactly ``[p_top, p_s]`` on both
    ladders; see the module docstring.
    """

    p_top: float
    hybrid_opt: int
    etac: float


@dataclass(frozen=True)
class VerticalRemapReceipt:
    """What one remap actually did to one field.

    Reported rather than assumed: a remap that silently failed to conserve
    would otherwise reach the child as a plausible-looking atmosphere.
    """

    field: str
    source_levels: int
    target_levels: int
    source_integral: float
    target_integral: float
    max_relative_drift: float
    min_source_layer_mass_pa: float
    min_target_layer_mass_pa: float
    minimum_value: float

    def summary(self) -> str:
        return (
            f"{self.field}: {self.source_levels}->{self.target_levels} levels, "
            f"column integral {self.source_integral:.9e} -> "
            f"{self.target_integral:.9e} "
            f"(max relative drift {self.max_relative_drift:.3e}), "
            f"min layer mass {self.min_target_layer_mass_pa:.4g} Pa, "
            f"min value {self.minimum_value:.6g}")


def require_shared_column_basis(source: ColumnBasis, target: ColumnBasis,
                                *, context: str) -> None:
    """Refuse two ladders whose columns are not the same mass interval.

    ``pd_f[0] = p_s`` and ``pd_f[nz] = p_top`` hold identically only when
    ``p_top``, ``hybrid_opt`` and ``etac`` agree.  Let one of them differ and
    the two ladders stop spanning the same physical column: the operator would
    have to extrapolate above the model top, where there is no reference
    state to extrapolate from.  Refused rather than silently rescaled.
    """

    drift = []
    if float(source.p_top) != float(target.p_top):
        drift.append(f"p_top {source.p_top!r} -> {target.p_top!r}")
    if int(source.hybrid_opt) != int(target.hybrid_opt):
        drift.append(f"hybrid_opt {source.hybrid_opt!r} -> {target.hybrid_opt!r}")
    if float(source.etac) != float(target.etac):
        drift.append(f"etac {source.etac!r} -> {target.etac!r}")
    if drift:
        raise VerticalRemapRefusal(
            f"{context}: a vertical remap requires both ladders to share "
            f"p_top, hybrid_opt and etac, but {', '.join(drift)}.  Only then "
            "does the hybrid reference dry pressure span exactly "
            "[p_top, p_s] on both, giving the two ladders coincident "
            "endpoints; with any of the three changed the remap would have "
            "to extrapolate above the model top, where there is no state to "
            "extrapolate from.  Refine the level COUNT and leave these three "
            "shared.")


def _column_shape(edges: np.ndarray, name: str) -> tuple[int, ...]:
    if edges.ndim < 1 or edges.shape[0] < 2:
        raise VerticalRemapRefusal(
            f"{name} must carry at least two interfaces, got shape "
            f"{edges.shape}")
    return edges.shape[1:]


def dry_mass_edges(znw, *, hybrid_opt: int, etac: float, p_top: float,
                   mu) -> np.ndarray:
    """Cumulative dry mass at every interface, surface (``mu``) to top (0).

    ``mu`` is the FULL column dry mass (``MUB + MU``) and may carry trailing
    horizontal axes; the result is ``(nz + 1,) + np.shape(mu)``.

    Computed through ``compute_hybrid_coeffs`` so this is the tree's own
    coordinate arithmetic rather than a second transcription of it:
    ``m[k] = c3f[k]*mu + c4f[k]``, which is ``pd_f[k] - p_top``.
    """

    znw = np.asarray(znw, dtype=np.float64)
    if znw.ndim != 1 or znw.size < 2:
        raise VerticalRemapRefusal(
            f"znw must be one ladder of at least two interfaces, got shape "
            f"{znw.shape}")
    if znw[0] != 1.0 or znw[-1] != 0.0 or not np.all(np.diff(znw) < 0.0):
        raise VerticalRemapRefusal(
            "znw must decrease strictly from 1.0 at the surface to 0.0 at "
            f"the model top, got {znw[0]!r} .. {znw[-1]!r}")
    mu = np.asarray(mu, dtype=np.float64)
    hy = compute_hybrid_coeffs(znw, int(hybrid_opt), float(etac),
                               float(c.P0), float(p_top))
    index = (slice(None),) + (None,) * mu.ndim
    edges = hy["c3f"][index] * mu[None] + hy["c4f"][index]

    # WRF's own coordinate-validity check (compute_vcoord_1d_coeffs): the
    # reference dry pressure must decrease with k in every column.  At
    # hybrid_opt=2 the Klemp cubic's dB/deta exceeds 1 near the ground, so a
    # column over high terrain (small mu) can reach c1h*mu + c2h <= 0.  That
    # is a folded coordinate, and alt = dphi/dm would divide by it and carry
    # an inf into the child rather than raising.
    dm = edges[:-1] - edges[1:]
    if not np.all(dm > 0.0):
        raise VerticalRemapRefusal(
            f"hybrid coordinate is folded (hybrid_opt={int(hybrid_opt)}, "
            f"etac={float(etac)}, p_top={float(p_top)} Pa): the smallest "
            f"layer dry mass is {float(dm.min()):.2f} Pa, so the reference "
            "dry pressure stops decreasing with height and the column is not "
            "orderable (WRF compute_vcoord_1d_coeffs validity check).  "
            "Terrain is too high for this etac -- reduce etac.  Refused "
            "rather than returning the inf that alt = dphi/dm would give.")
    return edges


def layer_masses(edges) -> np.ndarray:
    """Per-layer dry mass ``dm`` from interface cumulative masses."""

    edges = np.asarray(edges, dtype=np.float64)
    _column_shape(edges, "edges")
    return edges[:-1] - edges[1:]


def column_integral(edges, values) -> np.ndarray:
    """Mass-weighted column integral ``sum(values*dm)`` of a layer field."""

    edges = np.asarray(edges, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    dm = layer_masses(edges)
    if values.shape != dm.shape:
        raise VerticalRemapRefusal(
            f"layer field shape {values.shape} does not match the "
            f"{dm.shape} layers of its ladder")
    return (values * dm).sum(axis=0)


def _flatten(edges: np.ndarray, name: str):
    """Interfaces as ``(n+1, ncol)`` measured UP from 0 at the surface."""

    cols = _column_shape(edges, name)
    ncol = int(np.prod(cols)) if cols else 1
    flat = edges.reshape(edges.shape[0], ncol)
    # Increasing from 0 at the surface; np.searchsorted-style logic below
    # needs a monotonically increasing coordinate.
    return (flat[:1] - flat), cols, ncol


def _bin_index(x: np.ndarray, xs: np.ndarray, nbin: int) -> np.ndarray:
    """Index of the source bin containing ``x``, per column."""

    return np.clip(np.count_nonzero(xs <= x, axis=0) - 1, 0, nbin - 1)


def remap_layer_means(src_edges, values, dst_edges) -> np.ndarray:
    """Exact integral rebinning of a layer-mean field between two ladders.

    Conservative, monotone and positivity preserving: the result is the exact
    layer mean of the piecewise-constant source density over each target
    layer.  A target layer contained in one source layer takes that source
    value with NO arithmetic at all, which makes a matched pair of ladders
    the exact bitwise identity and makes a pure refinement exact wherever a
    child layer nests inside a parent layer.
    """

    src = np.asarray(src_edges, dtype=np.float64)
    dst = np.asarray(dst_edges, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    xs, cols, ncol = _flatten(src, "src_edges")
    xd, dcols, dncol = _flatten(dst, "dst_edges")
    if cols != dcols:
        raise VerticalRemapRefusal(
            f"source columns {cols} and target columns {dcols} differ; a "
            "remap moves one column between two ladders")
    ns, nd = xs.shape[0] - 1, xd.shape[0] - 1
    if values.shape != (ns,) + cols:
        raise VerticalRemapRefusal(
            f"layer field shape {values.shape} does not match the "
            f"{(ns,) + cols} layers of the source ladder")
    v = values.reshape(ns, ncol)

    # Exact piecewise-linear cumulative of the source density.
    cum = np.empty((ns + 1, ncol), dtype=np.float64)
    cum[0] = 0.0
    np.cumsum(v * np.diff(xs, axis=0), axis=0, out=cum[1:])

    col = np.arange(ncol)
    out = np.empty((nd, ncol), dtype=np.float64)
    for j in range(nd):
        a, b = xd[j], xd[j + 1]
        ia = _bin_index(a, xs, ns)
        # A target layer that lies inside ONE source layer is that source
        # value exactly -- no multiply, no divide, no rounding.
        contained = (xs[ia, col] <= a) & (b <= xs[ia + 1, col])
        ib = _bin_index(b, xs, ns)
        lower = _cumulative_at(a, ia, xs, cum, col)
        upper = _cumulative_at(b, ib, xs, cum, col)
        out[j] = np.where(contained, v[ia, col], (upper - lower) / (b - a))
    return out.reshape((nd,) + cols)


def _cumulative_at(x: np.ndarray, idx: np.ndarray, xs: np.ndarray,
                   cum: np.ndarray, col: np.ndarray) -> np.ndarray:
    """Exact piecewise-linear cumulative integral evaluated at ``x``."""

    x0, x1 = xs[idx, col], xs[idx + 1, col]
    c0, c1 = cum[idx, col], cum[idx + 1, col]
    return c0 + (c1 - c0) * (x - x0) / (x1 - x0)


def remap_interface_values(src_edges, values, dst_edges) -> np.ndarray:
    """Remap an interface field (``w``) linearly in dry mass.

    Exact at both ends, where the two ladders' interfaces coincide, and exact
    at any interior interface the two ladders share.
    """

    src = np.asarray(src_edges, dtype=np.float64)
    dst = np.asarray(dst_edges, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    xs, cols, ncol = _flatten(src, "src_edges")
    xd, dcols, _ = _flatten(dst, "dst_edges")
    if cols != dcols:
        raise VerticalRemapRefusal(
            f"source columns {cols} and target columns {dcols} differ; a "
            "remap moves one column between two ladders")
    ns, nd = xs.shape[0] - 1, xd.shape[0] - 1
    if values.shape != (ns + 1,) + cols:
        raise VerticalRemapRefusal(
            f"interface field shape {values.shape} does not match the "
            f"{(ns + 1,) + cols} interfaces of the source ladder")
    v = values.reshape(ns + 1, ncol)

    col = np.arange(ncol)
    out = np.empty((nd + 1, ncol), dtype=np.float64)
    for j in range(nd + 1):
        x = xd[j]
        raw = np.clip(np.count_nonzero(xs <= x, axis=0) - 1, 0, ns)
        # An interface the two ladders share takes the source value exactly.
        exact = xs[raw, col] == x
        idx = np.minimum(raw, ns - 1)
        x0, x1 = xs[idx, col], xs[idx + 1, col]
        v0, v1 = v[idx, col], v[idx + 1, col]
        out[j] = np.where(exact, v[raw, col],
                          v0 + (v1 - v0) * (x - x0) / (x1 - x0))
    return out.reshape((nd + 1,) + cols)


def remap_receipt(field: str, src_edges, values, dst_edges,
                  remapped) -> VerticalRemapReceipt:
    """Measure one completed layer-field remap.

    Reported for every remap that runs.  A conservation number that is never
    printed is a conservation number nobody checked.
    """

    src = np.asarray(src_edges, dtype=np.float64)
    dst = np.asarray(dst_edges, dtype=np.float64)
    before = np.atleast_1d(column_integral(src, values))
    after = np.atleast_1d(column_integral(dst, remapped))
    # A column whose integral is exactly zero (a species this scheme does not
    # carry) scales by 1 instead of dividing by zero; its drift is then an
    # absolute number, and for an all-zero field that is exactly 0.
    scale = np.where(np.abs(before) > 0.0, np.abs(before), 1.0)
    return VerticalRemapReceipt(
        field=str(field),
        source_levels=int(src.shape[0] - 1),
        target_levels=int(dst.shape[0] - 1),
        source_integral=float(before.sum()),
        target_integral=float(after.sum()),
        max_relative_drift=float(np.abs((after - before) / scale).max()),
        min_source_layer_mass_pa=float(layer_masses(src).min()),
        min_target_layer_mass_pa=float(layer_masses(dst).min()),
        minimum_value=float(np.asarray(remapped).min()),
    )


def geopotential_thickness_per_mass(phi, edges) -> np.ndarray:
    """``alt = dphi/dm``, the quantity the dycore itself diagnoses.

    ``gpuwm/core/diagnostics.py`` :: ``update_diagnostics`` computes exactly
    this at ``hypsometric_opt=1`` as ``-dphi*rdnw/(c1h*mu + c2h)``.
    """

    phi = np.asarray(phi, dtype=np.float64)
    dm = layer_masses(np.asarray(edges, dtype=np.float64))
    if phi.shape != (dm.shape[0] + 1,) + dm.shape[1:]:
        raise VerticalRemapRefusal(
            f"geopotential shape {phi.shape} does not match the "
            f"{(dm.shape[0] + 1,) + dm.shape[1:]} interfaces of its ladder")
    return (phi[1:] - phi[:-1]) / dm


def rebuild_geopotential(phi_surface, alt, edges) -> np.ndarray:
    """Integrate ``phi[k+1] = phi[k] + alt[k]*dm[k]`` from the surface.

    The exact inverse of :func:`geopotential_thickness_per_mass`, so a child
    built this way satisfies the dycore's own discrete hydrostatic relation
    rather than merely coming close to it.  Interpolating PHI directly in
    some other coordinate (eta, height) would preserve the column DEPTH --
    both ladders share the endpoints -- while leaving the child's diagnosed
    ``alt`` inconsistent with the thermodynamics it was remapped alongside.
    """

    alt = np.asarray(alt, dtype=np.float64)
    dm = layer_masses(np.asarray(edges, dtype=np.float64))
    if alt.shape != dm.shape:
        raise VerticalRemapRefusal(
            f"alt shape {alt.shape} does not match the {dm.shape} layers of "
            "its ladder")
    phi = np.empty((dm.shape[0] + 1,) + dm.shape[1:], dtype=np.float64)
    phi[0] = phi_surface
    for k in range(dm.shape[0]):
        phi[k + 1] = phi[k] + alt[k] * dm[k]
    return phi


__all__ = [
    "ColumnBasis",
    "VerticalRemapReceipt",
    "VerticalRemapRefusal",
    "column_integral",
    "dry_mass_edges",
    "geopotential_thickness_per_mass",
    "layer_masses",
    "rebuild_geopotential",
    "remap_interface_values",
    "remap_layer_means",
    "remap_receipt",
    "require_shared_column_basis",
]
