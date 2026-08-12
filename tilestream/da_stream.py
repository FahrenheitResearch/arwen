"""The DA/ensemble lane against a streamed domain: what tiles, what cannot.

``gpuwm.da.perturb`` and ``gpuwm.da.letkf`` were written for a domain that
fits on the card.  ``tilestream`` exists for one that does not.  Nothing in
either lane mentions the other -- ``grep -rn 'perturb\\|letkf' tilestream/``
and ``grep -rn 'streaming\\|hoststore' gpuwm/da/`` are both empty -- so the
question "does DA work with streaming on" has never had an implementation to
be true or false about.  This module is that implementation, plus the
controls that say which half of it is a design decision and which half is a
law.

THE LAW: AN FFT IS A GLOBAL TRANSFORM
-------------------------------------
``perturb.gaussian_random_field`` draws unit white noise and multiplies its
``rfftn`` by a separable Gaussian amplitude filter.  A per-tile FFT of a
windowed field is NOT the window of the domain FFT, at ANY halo width, and
that is not a halo-tuning problem: the filter's impulse response is
``exp(-r^2/L^2)`` with unbounded support, and the transform of a window is
the domain spectrum CONVOLVED with the window's own transform.  Widening the
halo shrinks the error but never removes it, and the leftover lives exactly
where a reviewer will not look for it -- as a smooth, plausible-looking
noise field with a scale discontinuity at every tile seam.
:func:`tile_filter_control` measures it.

THE DESIGN DECISION: THE DRAW ALREADY HAPPENS ON THE HOST
---------------------------------------------------------
``perturb._white_noise`` runs NumPy's Philox on the HOST even when the
filtering will run on the GPU, precisely so ``noise_sha256`` identifies the
perturbation and not the machine.  So the escape hatch is already half
built: keep the FILTER on the host too and the whole draw is a host-side
operation over the pinned store, which never touches a tile and therefore
never meets the law above.  What was missing was (a) something that lets
``apply_perturbations`` see a store at all, and (b) a way to make the
resident arm agree with it bit for bit -- because ``perturb``'s own
docstring admits "cuFFT and pocketfft round differently".  Both are in this
module and in ``perturb.PerturbationConfig.fft_host``.

WHAT LETKF IS, WHICH IS THE OPPOSITE SHAPE
------------------------------------------
``letkf.analyze`` is pointwise under Gaspari-Cohn localisation: a gridpoint's
analysis reads only observations inside its cutoff.  It is therefore
chunkable, and it already chunks (``chunk_points_for_budget``).  Its problem
is not structure but the LEDGER: the prior is ``R`` times the whole domain,
``_validate_prior`` materialises it with ``xp.asarray(..., float64)``, and
the mean/perturbation step allocates ``xb`` and ``increments`` on top --
three full-domain float64 copies per field per member before a single solve.
:func:`letkf_ledger` measures the multiplier and extrapolates it, because
pinned pages cannot be swapped and this project has already frozen a machine
by guessing.


THE MEASUREMENTS, SO THE TWO SHAPES CAN BE READ SIDE BY SIDE
------------------------------------------------------------
All at 192x160x49, 8 members, ``length_scale_km = 6`` (12 cells), the
streaming lane's own halo 16 = ``harness.halo_radius``.  Run
``python -m tilestream.test_da_stream``.

*The FFT does not tile.*  The same white noise filtered per 48x40 tile
differs from the domain filter by ``8.3e-2`` relative, RMS ``7.2e-3``, and
the error is ``83.6x`` concentrated on the tile seams.  Widening the halo
buys a Gaussian decay and nothing else -- ``5.7e-3`` at 24, ``2.1e-4`` at
32, ``2.3e-8`` at 48 -- and only reaches float64 rounding at halo 80, whose
window is 200x208 on a 160x192 domain.  A window that covers the domain is
not a tile, and even it is not bit-exact, because a transform of a
differently shaped array is different arithmetic.  Static row decomposition
(the multi-GPU shape) is the same law with a bigger seam ratio: ``5.1e-2``
at 2 ranks, ``6.0e-2`` at 4.

*The analysis does tile, exactly.*  The same 4-slab decomposition run
through :func:`letkf_slab_control` is BIT-IDENTICAL to the monolithic
analysis at halo 2 -- 0 of 12,042,240 values differ -- and differs in 3,360
values by ``1.8e-1`` relative at halo 1.  Halo 2 is not a swept number: it
is ``floor(horizontal_m / dx)`` straight out of ``_horizontal_stencil``,
because Gaspari-Cohn has compact support.  That contrast is the finding:
one operator has a dependency cone and the other has none, and no amount of
halo tuning moves either of them across the line.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np

from gpuwm.da import perturb as _perturb


# --------------------------------------------------------------------------
# Letting apply_perturbations see a store
# --------------------------------------------------------------------------

#: Every attribute ``gpuwm.da.perturb`` reads off a state, and what each one
#: is for.  ``thb`` is the one that is NOT in ``STATE_SERIALIZED_ATTRS`` --
#: it is a setup array, so a store built from the persisted contract cannot
#: supply it, and a store-backed perturbation therefore cannot evaluate the
#: supersaturation cap or the Exner conversion unless the caller passes it
#: in separately.  That is a real restriction and it is stated rather than
#: papered over.
STORE_READABLE = ("thp", "qv", "u", "v", "p")
STORE_EXTRA = ("thb",)


class StoreStateView:
    """A ``DomainState``-shaped read/write window onto a store's arrays.

    ``gpuwm.da.perturb`` addresses the state by attribute (``state.thp``,
    ``state.u``, ...) and mutates in place.  A :class:`tilestream.hoststore
    .HostDomainStore` addresses the same fields by string key in ``.arrays``,
    and a physics carrier store addresses them as ``state/thp``.  This view
    is the whole adapter: it binds the store's arrays -- the pinned buffers
    THEMSELVES, not copies -- to the attribute names the DA lane uses, so
    ``apply_perturbations(view, seed, cfg)`` perturbs the domain that the
    streamed run will actually integrate.

    It is deliberately a view and not a copy.  A copy would perturb a
    throwaway and hand the streamed run its unperturbed background, which is
    the exact failure ``gpuwm.ensemble.perturbation`` refuses to allow for
    the resolution path ("an ensemble that believes it was perturbed and was
    not is worse than one that refuses to start").

    ``extra`` supplies arrays the persisted contract does not carry --
    ``thb`` above all.  Nothing is invented: an attribute that neither the
    store nor ``extra`` provides reads as ``None``, which is what
    ``perturb._resolve_target`` turns into a named refusal.
    """

    def __init__(self, arrays: Mapping[str, Any], *,
                 extra: Mapping[str, Any] | None = None,
                 prefix: str = ""):
        self._arrays = arrays
        self._extra = dict(extra or {})
        self._prefix = prefix
        # Bind eagerly so a typo in a key is an AttributeError here rather
        # than a silent None three frames into the DA lane.
        for name in STORE_READABLE:
            key = f"{prefix}{name}"
            if key in arrays:
                object.__setattr__(self, name, arrays[key])
        for name, value in self._extra.items():
            object.__setattr__(self, name, value)

    def __getattr__(self, name: str):
        # Only reached for names __init__ did not bind.  perturb treats a
        # missing field as absent (None) and says so by name.
        if name.startswith("_"):
            raise AttributeError(name)
        return None

    def total_theta(self):
        """Full potential temperature, when the caller supplied ``thb``."""
        thb = getattr(self, "thb", None)
        if thb is None:
            raise ValueError(
                "the store view has no thb: base-state potential temperature "
                "is a SETUP array and is not in STATE_SERIALIZED_ATTRS, so a "
                "store built from the persisted contract cannot supply it. "
                "Pass extra={'thb': ...} to perturb 't' or to let the "
                "supersaturation cap fire on qv.")
        base = thb if getattr(thb, "ndim", 0) == 3 else thb[:, None, None]
        return base + self.thp

    def field_digests(self, names: Sequence[str] = STORE_READABLE
                      ) -> dict[str, str]:
        """SHA-256 per bound field, for a byte-level comparison."""
        out = {}
        for name in names:
            arr = getattr(self, name, None)
            if arr is None:
                continue
            out[name] = _digest(arr)
        return out


def perturb_member(target, seed: int, cfg):
    """``apply_perturbations`` with the filter pinned to the host, always.

    The entry point an ENSEMBLE should use, resident or streamed, and the
    reason it exists is that the choice cannot be made correctly one member
    at a time.  ``PerturbationConfig.fft_host`` decides which library
    filters the draw; a resident member filters on whichever backend its
    state lives on and a streamed member's domain lives in pinned host RAM,
    so an ensemble with both kinds of member in it -- which is exactly what
    ``[tiles] mode = "auto"`` produces, since the decision is per
    domain and depends on whether it fits -- would draw its members from
    two different filters and call the difference spread.

    Forcing the flag here rather than defaulting it in
    :class:`~gpuwm.da.perturb.PerturbationConfig` is deliberate: flipping
    that default would change every existing resident member's bytes, and a
    reproducibility fix that silently rewrites past results is not a fix.
    A caller that wants the old behaviour still has it; a caller that wants
    one ensemble reproducible across execution modes calls this.
    """
    from gpuwm.da.perturb import apply_perturbations

    if not cfg.fft_host:
        cfg = replace(cfg, fft_host=True)
    return apply_perturbations(target, seed, cfg)


def _digest(array) -> str:
    """SHA-256 over dtype, shape and C-order bytes, backend-agnostic."""
    host = array if isinstance(array, np.ndarray) else np.asarray(array.get())
    host = np.ascontiguousarray(host)
    sha = hashlib.sha256()
    sha.update(str(host.dtype).encode())
    sha.update(np.asarray(host.shape, dtype=np.int64).tobytes())
    sha.update(host.tobytes())
    return sha.hexdigest()


def _digest_inventory(arrays: Mapping[str, Any]) -> str:
    """One SHA-256 over a whole ``{name: array}`` inventory, in name order.

    Deliberately the same payload shape as ``harness._digest_payload`` (name,
    dtype, int64 shape, C-order bytes) so a shape change cannot masquerade as
    a value change, and so this digest is comparable with the gate's.
    """
    sha = hashlib.sha256()
    for name in sorted(arrays):
        sha.update(_digest(arrays[name]).encode("ascii"))
        sha.update(name.encode("utf-8"))
    return sha.hexdigest()


def host_mirror(state, names: Sequence[str] = STORE_READABLE,
                extra: Sequence[str] = STORE_EXTRA) -> StoreStateView:
    """A NumPy copy of ``state``'s perturbable fields, as a view.

    The control arm: same numbers as the resident state, host namespace, no
    store.  It separates "the store broke it" from "the backend broke it",
    which is the distinction the resident-versus-streamed verdict turns on.
    """
    arrays = {}
    for name in names:
        arr = getattr(state, name, None)
        if arr is not None:
            arrays[name] = np.ascontiguousarray(_to_host(arr))
    ex = {}
    for name in extra:
        arr = getattr(state, name, None)
        if arr is not None:
            ex[name] = np.ascontiguousarray(_to_host(arr))
    return StoreStateView(arrays, extra=ex)


def _to_host(array) -> np.ndarray:
    return array if isinstance(array, np.ndarray) else np.asarray(array.get())


# --------------------------------------------------------------------------
# The negative control: the same white noise, filtered per tile
# --------------------------------------------------------------------------

def _windowed_filter(noise: np.ndarray, dx_km: float, dy_km: float,
                     length_scale_km: float, vertical_scale_levels: float,
                     *, normalize_by=None) -> np.ndarray:
    """``gaussian_random_field``'s filter, on whatever extents it is given.

    This is exactly what a tiled implementation would do to a gathered
    window: transform the window, multiply by the separable filter built
    for the WINDOW's extents, transform back, normalise by the window's own
    analytic variance.  Every one of those steps is the only thing a tile
    could do, and the sum of them is not the domain's answer.
    """
    nz, ny, nx = noise.shape
    spectrum = np.fft.rfftn(noise, axes=(0, 1, 2))
    hz = _perturb._axis_filter(nz, 1.0, vertical_scale_levels, half=False)
    hy = _perturb._axis_filter(ny, dy_km, length_scale_km, half=False)
    hx = _perturb._axis_filter(nx, dx_km, length_scale_km, half=True)
    kernel = (hz[:, None, None] * hy[None, :, None]
              * hx[None, None, :]).astype(noise.dtype, copy=False)
    field = np.fft.irfftn(spectrum * kernel, s=noise.shape, axes=(0, 1, 2))
    if normalize_by is None:
        variance = (
            _perturb._analytic_variance(nz, 1.0, vertical_scale_levels)
            * _perturb._analytic_variance(ny, dy_km, length_scale_km)
            * _perturb._analytic_variance(nx, dx_km, length_scale_km))
        normalize_by = math.sqrt(variance)
    return (field / normalize_by).astype(noise.dtype, copy=False)


def tile_filter_control(shape, *, seed: int, name: str, dx_km: float,
                        dy_km: float, length_scale_km: float,
                        vertical_scale_levels: float,
                        tile_ny: int, tile_nx: int, halo: int,
                        domain_normalization: bool = False) -> dict:
    """Filter one white-noise draw whole-domain and per tile; compare.

    THIS CONTROL MUST FIRE.  It is the negative control the streamed
    perturbation rests on: if a per-tile FFT reproduced the domain FFT the
    whole "the draw must stay whole-domain" claim would be empty, and the
    honest conclusion would be that the filter's length scale is too short
    for the test rather than that tiling works.

    The tile gather is periodic (``np.roll``-style wraparound), which is the
    FRIENDLIEST possible case: the domain filter is itself periodic, so a
    wraparound halo has no edge effect of its own to confuse the answer.
    Under specified boundaries the tiled error can only be larger.

    ``domain_normalization`` re-runs the tiled arm normalised by the
    DOMAIN's analytic variance instead of the window's.  It separates two
    different errors that would otherwise be reported as one: a scale factor
    (which a careful implementer might fix) from the seams (which nobody
    can).
    """
    nz, ny, nx = (int(s) for s in shape)
    noise, key, digest = _perturb._white_noise(
        (nz, ny, nx), seed=seed, name=name, dtype=np.float64)

    whole = _windowed_filter(noise, dx_km, dy_km, length_scale_km,
                             vertical_scale_levels)
    domain_norm = None
    if domain_normalization:
        var = (_perturb._analytic_variance(nz, 1.0, vertical_scale_levels)
               * _perturb._analytic_variance(ny, dy_km, length_scale_km)
               * _perturb._analytic_variance(nx, dx_km, length_scale_km))
        domain_norm = math.sqrt(var)

    tiled = np.empty_like(whole)
    seam_rows: list[int] = []
    seam_cols: list[int] = []
    for j0 in range(0, ny, tile_ny):
        seam_rows.append(j0)
        for i0 in range(0, nx, tile_nx):
            if j0 == 0:
                seam_cols.append(i0)
            j1 = min(j0 + tile_ny, ny)
            i1 = min(i0 + tile_nx, nx)
            rows = np.arange(j0 - halo, j1 + halo) % ny
            cols = np.arange(i0 - halo, i1 + halo) % nx
            window = np.ascontiguousarray(
                noise[:, rows[:, None], cols[None, :]])
            filtered = _windowed_filter(
                window, dx_km, dy_km, length_scale_km,
                vertical_scale_levels, normalize_by=domain_norm)
            tiled[:, j0:j1, i0:i1] = filtered[:, halo:halo + (j1 - j0),
                                              halo:halo + (i1 - i0)]

    return _difference_report(whole, tiled, ny, nx, seam_rows, seam_cols,
                              tile_ny, tile_nx, halo)


def slab_filter_control(shape, *, seed: int, name: str, dx_km: float,
                        dy_km: float, length_scale_km: float,
                        vertical_scale_levels: float,
                        nranks: int, halo: int) -> dict:
    """The MULTI-GPU shape of the same law: N row slabs, not a tile grid.

    Static domain decomposition across ``nranks`` cards gives each rank a
    contiguous slab plus a halo.  A rank that filters its own slab is doing
    the tile control with one very wide tile, and the answer is the same
    law with a smaller constant: the error concentrates on the rank
    boundaries.  Run separately from :func:`tile_filter_control` because the
    multi-GPU verdict has to be about the decomposition multi-GPU actually
    uses.
    """
    ny = int(shape[1])
    rows = ny // int(nranks)
    if rows * int(nranks) != ny:
        raise ValueError(f"{ny} rows do not divide into {nranks} slabs")
    return tile_filter_control(
        shape, seed=seed, name=name, dx_km=dx_km, dy_km=dy_km,
        length_scale_km=length_scale_km,
        vertical_scale_levels=vertical_scale_levels,
        tile_ny=rows, tile_nx=int(shape[2]), halo=halo)


def _difference_report(whole: np.ndarray, other: np.ndarray, ny: int, nx: int,
                       seam_rows, seam_cols, tile_ny, tile_nx,
                       halo) -> dict:
    """Max/RMS difference, and whether it lives on the seams."""
    diff = np.abs(other - whole)
    scale = float(np.abs(whole).max())
    # A relative measure against the FIELD's own amplitude, not pointwise:
    # the draw crosses zero everywhere, so a pointwise ratio is dominated by
    # cells where the denominator is noise.
    rel = diff / scale

    seam_mask = np.zeros((ny, nx), dtype=bool)
    for j0 in seam_rows:
        for d in (-1, 0):
            seam_mask[(j0 + d) % ny, :] = True
    for i0 in seam_cols:
        for d in (-1, 0):
            seam_mask[:, (i0 + d) % nx] = True

    # "Interior" = at least a quarter tile from any seam in both directions,
    # so the two populations do not overlap.
    pad_y = max(1, tile_ny // 4)
    pad_x = max(1, tile_nx // 4)
    interior = np.ones((ny, nx), dtype=bool)
    for j0 in seam_rows:
        for d in range(-pad_y, pad_y + 1):
            interior[(j0 + d) % ny, :] = False
    for i0 in seam_cols:
        for d in range(-pad_x, pad_x + 1):
            interior[:, (i0 + d) % nx] = False

    per_cell = rel.max(axis=0)
    return {
        "bitwise_identical": bool(np.array_equal(other, whole)),
        "max_abs": float(diff.max()),
        "field_max_abs": scale,
        "max_rel": float(rel.max()),
        "rms_rel": float(np.sqrt((rel ** 2).mean())),
        "seam_max_rel": float(per_cell[seam_mask].max()),
        "seam_mean_rel": float(per_cell[seam_mask].mean()),
        "interior_max_rel": (float(per_cell[interior].max())
                             if interior.any() else None),
        "interior_mean_rel": (float(per_cell[interior].mean())
                              if interior.any() else None),
        "seam_over_interior_mean": (
            float(per_cell[seam_mask].mean() / per_cell[interior].mean())
            if interior.any() and per_cell[interior].mean() > 0 else None),
        "tile": (tile_ny, tile_nx),
        "halo": halo,
    }


# --------------------------------------------------------------------------
# LETKF: the operator that DOES tile, and the halo that makes it
# --------------------------------------------------------------------------

def letkf_halo(loc, grid) -> tuple[int, int]:
    """``(dj, di)`` half-widths of the localisation stencil, in cells.

    This is the LETKF's halo and it is not a measurement, it is a property
    of the operator: Gaspari-Cohn has COMPACT support, so
    ``letkf._horizontal_stencil`` returns a bounded index box whose
    half-widths are ``floor(horizontal_m / min_spacing)`` clamped to the
    grid.  A slab that carries that many rows of neighbour sees every
    observation any of its interior points can read, so its interior
    analysis is the whole-domain analysis EXACTLY -- not approximately, and
    not at some width that has to be swept for.

    Contrast :func:`tile_filter_control`, where no such number exists.  That
    is the whole difference between the two halves of this lane.
    """
    from gpuwm.da import letkf

    dj, di = letkf._horizontal_stencil(loc, grid, int(grid.lat_deg.shape[1])
                                       if grid.lat_deg is not None else 1,
                                       int(grid.lat_deg.shape[0])
                                       if grid.lat_deg is not None else 1)
    return int(np.abs(dj).max()), int(np.abs(di).max())


def letkf_slab_control(prior, obs, grid, config, *, nslabs: int,
                       halo: int) -> dict:
    """Analyse the domain in row slabs and compare to the whole-domain run.

    The streamed shape of an LETKF: the prior is ``R`` times the domain and
    an out-of-core ensemble cannot hold it, so the analysis has to be done a
    band at a time out of the pinned stores.  This runs that, in ONE
    namespace so the comparison is about geometry and not about linear
    algebra, and reports whether the stitched increments are bit-identical
    to the monolithic ones.

    ``halo`` below the stencil half-width is the negative control and MUST
    fail: a slab that cannot see far enough drops observations its interior
    points were entitled to, which shows up as an increment that is too
    small near the seams and is otherwise perfectly plausible.
    """
    from gpuwm.da import letkf

    fields = tuple(config.analysis_fields)
    whole = letkf.analyze(prior, obs, grid, config, letkf.LetkfDiagnostics())

    shape = np.asarray(prior[fields[0]]).shape
    members, nz, ny, nx = shape
    rows = ny // int(nslabs)
    if rows * int(nslabs) != ny:
        raise ValueError(f"{ny} rows do not divide into {nslabs} slabs")

    stitched = {f: np.empty_like(np.asarray(whole[f])) for f in fields}
    heights = np.asarray(grid.heights_m)
    for s in range(int(nslabs)):
        j0, j1 = s * rows, (s + 1) * rows
        g0, g1 = max(0, j0 - halo), min(ny, j1 + halo)
        sub_prior = {f: np.ascontiguousarray(
            np.asarray(prior[f])[:, :, g0:g1, :]) for f in fields}
        sub_obs = []
        for o in obs:
            sub_obs.append(letkf.GriddedObs(
                name=o.name,
                values=np.ascontiguousarray(np.asarray(o.values)[:, g0:g1, :]),
                errors=(o.errors if np.ndim(o.errors) == 0 else
                        np.ascontiguousarray(
                            np.asarray(o.errors)[:, g0:g1, :])),
                simulated=np.ascontiguousarray(
                    np.asarray(o.simulated)[:, :, g0:g1, :]),
                mask=np.ascontiguousarray(np.asarray(o.mask)[:, g0:g1, :]),
                localization=o.localization))
        sub_grid = letkf.GridGeometry(
            dx_m=grid.dx_m, dy_m=grid.dy_m,
            heights_m=(heights if heights.ndim == 1
                       else np.ascontiguousarray(heights[:, g0:g1, :])),
            lat_deg=(None if grid.lat_deg is None
                     else np.ascontiguousarray(grid.lat_deg[g0:g1, :])),
            lon_deg=(None if grid.lon_deg is None
                     else np.ascontiguousarray(grid.lon_deg[g0:g1, :])),
            earth_radius_m=grid.earth_radius_m)
        part = letkf.analyze(sub_prior, sub_obs, sub_grid, config,
                             letkf.LetkfDiagnostics())
        for f in fields:
            stitched[f][:, :, j0:j1, :] = np.asarray(
                part[f])[:, :, j0 - g0:j0 - g0 + (j1 - j0), :]

    out = {"nslabs": int(nslabs), "halo": int(halo), "fields": {}}
    identical = True
    for f in fields:
        a = np.asarray(whole[f], dtype=np.float64)
        b = stitched[f].astype(np.float64)
        eq = bool(np.array_equal(a, b))
        identical = identical and eq
        scale = float(np.abs(a).max())
        out["fields"][f] = {
            "bitwise_identical": eq,
            "max_abs": float(np.abs(a - b).max()),
            "max_rel": (float(np.abs(a - b).max() / scale) if scale else 0.0),
            "differing_cells": int(np.count_nonzero(a != b)),
            "cells": int(a.size),
        }
    out["bitwise_identical"] = identical
    return out


# --------------------------------------------------------------------------
# The LETKF ledger
# --------------------------------------------------------------------------

def letkf_ledger(nz: int, ny: int, nx: int, members: int, nfields: int,
                 *, store_bytes_per_member: int | None = None,
                 itemsize: int = 8) -> dict:
    """What ``analyze`` costs before it solves anything.

    ``_validate_prior`` calls ``xp.asarray(prior[f], dtype=float64)`` on
    every analysis field, then ``analyze`` builds ``xb = pri - mean`` and
    ``increments`` -- so THREE full ``(R, nz, ny, nx)`` float64 arrays per
    analysis field exist simultaneously before the chunk loop starts, on top
    of whatever the caller's prior already occupied.  The chunk budget
    (``memory_budget_mib``) bounds the solve's scratch and explicitly not
    this.  Every observation batch adds a fourth ``(R, nz, ny, nx)``
    (``simulated``) plus two ``(nz, ny, nx)``.

    Reported separately from the pinned figure because the two live in
    different pools: the store is pinned and unswappable, the analysis
    arrays are ordinary pageable host memory (or VRAM, in the resident arm).
    """
    cells = int(nz) * int(ny) * int(nx)
    one = cells * members * itemsize
    return {
        "cells": cells,
        "members": members,
        "analysis_fields": nfields,
        "prior_bytes_one_field": one,
        # pri + xb + increments, per field, all live at the same moment
        "analysis_peak_bytes": 3 * one * nfields,
        "per_obs_batch_bytes": one + 2 * cells * itemsize,
        "pinned_store_bytes": (None if store_bytes_per_member is None
                               else store_bytes_per_member * members),
    }


def container_memory_limit() -> dict:
    """The cgroup's limit, and how far ``/proc/meminfo`` is from it.

    ``hoststore.host_memory`` reads ``/proc/meminfo``, and
    ``pinned_ceiling_bytes`` multiplies its ``MemTotal`` by
    :data:`~tilestream.hoststore.PINNED_CEILING_FRACTION`.  Inside an
    unprivileged container ``/proc/meminfo`` is the HOST's, not the
    container's, so both numbers can be far too generous.  MEASURED on this
    project's four rented boxes:

        box        cgroup limit   /proc/meminfo   overstatement
        5090         120.6 GiB      125.6 GiB          1.04x
        2x4090       241.7 GiB      503.6 GiB          2.08x
        4090         342.0 GiB     1007.8 GiB          2.93x

    A single streamed domain has survived that so far because it asks for
    one store.  An ENSEMBLE asks for ``R`` of them, so the same mis-read
    reaches the wall ``R`` times sooner -- and pinned pages cannot be
    swapped, which is how this project already froze a machine.  Read the
    cgroup, cap against it, and treat ``/proc/meminfo`` as an upper bound
    that may be wrong by a factor of three.
    """
    out: dict[str, Any] = {"cgroup_bytes": None, "cgroup_path": None}
    for path in ("/sys/fs/cgroup/memory.max",
                 "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            with open(path, "r", encoding="ascii") as handle:
                raw = handle.read().strip()
        except OSError:
            continue
        if raw == "max":
            out["cgroup_path"] = path
            break
        try:
            value = int(raw)
        except ValueError:
            continue
        # cgroup v1 spells "no limit" as a number near 2**63; anything at or
        # above the machine's own RAM is not a limit, it is the absence of
        # one, and reporting it as a limit would be worse than reporting
        # none at all.
        out["cgroup_bytes"] = value
        out["cgroup_path"] = path
        break

    from tilestream import hoststore

    mem = hoststore.host_memory()
    out["meminfo_total"] = mem["total"]
    out["meminfo_available"] = mem["available"]
    out["pinned_ceiling_from_meminfo"] = hoststore.pinned_ceiling_bytes()
    limit = out["cgroup_bytes"]
    if limit and mem["total"]:
        out["overstatement"] = mem["total"] / limit
        out["pinned_ceiling_from_cgroup"] = int(
            hoststore.PINNED_CEILING_FRACTION * limit)
    return out


def fmt_bytes(n) -> str:
    if n is None:
        return "n/a"
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if abs(n) < 1024.0 or unit == "TiB":
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} TiB"
