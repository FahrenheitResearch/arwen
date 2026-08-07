"""Config-driven initial-state theta bubbles for real-data experiments.

The [perturbation] block (:mod:`gpuwm.experiment`) declares warm bubbles
in geographic coordinates; this module evaluates them on one domain's
grid and writes them into that domain's initial theta (and, under
``rh_preserve``, qv) columns inside
:func:`gpuwm.ingest.real.initialize_real`, before the specific volume
and geopotential are formed.  The shape is WRF's em_quarter_ss bubble
(module_initialize_ideal.F; the port's transcription is
gpuwm/verify/cases/moist_bubble.py): peak ``amplitude_k`` at the
center, ``cos^2(pi/2 * r)`` taper on the normalized ellipse radius
``r = sqrt((d_h/radius)^2 + ((z - z_c)/depth)^2)``, exactly zero at and
beyond ``r = 1``.

PER-DOMAIN APPLICATION, deliberately.  gpuwm's real-data nest init is
real.exe-per-domain, not ndown: every child re-ingests the source
analysis on its own grid (gpuwm/ingest/nest_init.py), so a bubble
written into the parent's prepared state would be invisible to a
child's initial conditions.  Each domain that initializes at the
experiment start time therefore evaluates the same geographic bubble on
its own grid inside its own ``initialize_real`` -- the one init path
every domain shares -- and the per-domain receipts record what each
grid actually received.  A domain with a delayed start initializes from
the analysis at its activation time, by which point the parent has
already evolved the bubble, so it takes no fresh analytic bubble.

OFF is absolute: with no [perturbation] block
:func:`build_initial_state_perturbation` is never called, callers pass
``initial_perturbation=None``, and ``initialize_real`` executes not one
instruction of this module -- the prepared state is byte-identical to a
build without this file.

Refusals (loud, before any integration):

* a bubble center outside the coarse domain (``require_containment``);
* an enabled bubble whose center is inside the domain but which touches
  zero cells (radius below the grid's resolving power) -- a refusal,
  never a silent no-op, per the treatment-proof rule.

This is an ArWen-over-WRF extension (PROVENANCE.md): stock WRF v4.6.1
has no config-driven real-data initial-condition perturbation; its warm
bubbles exist only in the idealized initializers.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gpuwm.core import constants as c
from gpuwm.static.projection import EARTH_RADIUS_M

APPLICATION_SCHEMA = "gpuwm-initial-perturbation-apply-v1"


def _great_circle_km(lat, lon, center_lat, center_lon) -> np.ndarray:
    """Haversine distance in km on the WPS sphere (EARTH_RADIUS_M)."""
    lat = np.asarray(lat, dtype=np.float64) * (np.pi / 180.0)
    lon = np.asarray(lon, dtype=np.float64) * (np.pi / 180.0)
    lat0 = float(center_lat) * (np.pi / 180.0)
    lon0 = float(center_lon) * (np.pi / 180.0)
    half_dlat = 0.5 * (lat - lat0)
    half_dlon = 0.5 * (lon - lon0)
    h = (np.sin(half_dlat) ** 2
         + np.cos(lat) * np.cos(lat0) * np.sin(half_dlon) ** 2)
    return (2.0 * EARTH_RADIUS_M / 1000.0) * np.arcsin(np.sqrt(h))


@dataclass(frozen=True)
class _PlacedBubble:
    """One bubble resolved onto a specific domain's grid."""

    spec: object                 # gpuwm.experiment.BubbleConfig
    index: int                   # 1-based position in the config block
    center_x: float              # 1-based projection coordinate (mass i)
    center_y: float              # 1-based projection coordinate (mass j)
    inside: bool
    horizontal_km: np.ndarray | None = field(repr=False, default=None)


class InitialStatePerturbation:
    """The [perturbation] block evaluated for one domain's grid.

    Built by :func:`build_initial_state_perturbation` (which returns
    ``None`` for an absent block -- the OFF contract), consumed exactly
    once by ``initialize_real``.  ``apply`` mutates the host FP64
    theta/qv columns in place and returns the per-domain receipt.
    """

    def __init__(self, bubbles, grid, *, grid_id: int,
                 require_containment: bool):
        self.grid_id = int(grid_id)
        nx = int(grid.e_we) - 1
        ny = int(grid.e_sn) - 1
        lat = lon = None
        placed = []
        for index, spec in enumerate(bubbles, start=1):
            x, y = grid.latlon_to_ij(spec.center_lat, spec.center_lon)
            x, y = float(x), float(y)
            # Mass point (i, j) sits at projection coordinate (i, j),
            # 1-based, i in [1, e_we-1], j in [1, e_sn-1] (static/lambert.py
            # grid-registration contract).
            inside = (1.0 <= x <= float(nx)) and (1.0 <= y <= float(ny))
            if require_containment and not inside:
                raise ValueError(
                    f"perturbation.bubbles #{index} center "
                    f"({spec.center_lat:g}, {spec.center_lon:g}) lies "
                    f"outside domain d{self.grid_id:02d} (projection "
                    f"coordinate ({x:.2f}, {y:.2f}) of a {nx} x {ny} mass "
                    "grid); every bubble must sit inside the coarse "
                    "domain. Move the bubble or widen the domain.")
            horizontal_km = None
            if inside:
                if lat is None:
                    lat, lon = grid.latlon_mass()
                horizontal_km = _great_circle_km(
                    lat, lon, spec.center_lat, spec.center_lon)
            placed.append(_PlacedBubble(
                spec=spec, index=index, center_x=x, center_y=y,
                inside=inside, horizontal_km=horizontal_km))
        self._placed = tuple(placed)

    def apply(self, *, theta, qv, pressure, z_half_agl) -> dict:
        """Add every contained bubble to ``theta`` (and qv) in place.

        ``theta``/``qv``/``pressure`` are the final host FP64
        ``(nz, ny, nx)`` columns of ``initialize_real``; ``z_half_agl``
        the matching half-level heights AGL.  Cells outside every
        bubble's ``r < 1`` ellipse are byte-untouched.  Returns the
        per-domain application receipt; raises when an in-domain bubble
        touches zero cells.
        """
        theta = np.asarray(theta)
        rows = []
        for placed in self._placed:
            spec = placed.spec
            if not placed.inside:
                rows.append({
                    "bubble": placed.index,
                    "applied": False,
                    "reason": "center outside this domain",
                    "center_xy": [placed.center_x, placed.center_y],
                    "cells_touched": 0,
                })
                continue
            radial = np.sqrt(
                (placed.horizontal_km[None, :, :] / spec.radius_km) ** 2
                + ((np.asarray(z_half_agl, dtype=np.float64)
                    - spec.center_height_m) / spec.depth_m) ** 2)
            mask = radial < 1.0
            cells = int(np.count_nonzero(mask))
            if cells == 0:
                raise ValueError(
                    f"perturbation.bubbles #{placed.index} is enabled and "
                    f"centered inside domain d{self.grid_id:02d} "
                    f"(projection coordinate ({placed.center_x:.2f}, "
                    f"{placed.center_y:.2f})) but touches zero cells: "
                    f"radius_km = {spec.radius_km:g} / depth_m = "
                    f"{spec.depth_m:g} are below this grid's resolving "
                    "power. An applied perturbation that writes nothing "
                    "is a refusal, not a silent no-op -- enlarge the "
                    "bubble or drop it.")
            delta = np.zeros_like(theta[mask])
            delta[...] = (spec.amplitude_k
                          * np.cos(0.5 * np.pi * radial[mask]) ** 2)
            row = {
                "bubble": placed.index,
                "applied": True,
                "center_xy": [placed.center_x, placed.center_y],
                "cells_touched": cells,
                "max_theta_added_k": float(delta.max()),
                "rh_preserve": bool(spec.rh_preserve),
            }
            if spec.rh_preserve:
                row["max_qv_delta_kg_kg"] = self._preserve_rh(
                    theta, qv, pressure, mask, delta)
            theta[mask] += delta
            rows.append(row)
        return {
            "schema": APPLICATION_SCHEMA,
            "grid_id": self.grid_id,
            "bubbles": rows,
        }

    @staticmethod
    def _preserve_rh(theta, qv, pressure, mask, delta) -> float:
        """Adjust qv on ``mask`` so RH survives the theta change.

        WRF-faithful pair from :mod:`gpuwm.ingest.real` (imported lazily
        -- real.py must not import this module back): RH diagnosed from
        the unperturbed T/p/qv with the exact algebraic inverse of
        rh_to_mxrat1, then qv rebuilt from that RH at the perturbed
        temperature and the SAME analyzed pressure.  Only masked cells
        are written.
        """
        from gpuwm.ingest.real import (
            _mixing_ratio_to_relative_humidity,
            _saturation_mixing_ratio,
            _temperature_from_potential_temperature,
        )

        qv = np.asarray(qv)
        p_cells = np.asarray(pressure, dtype=np.float64)[mask]
        theta_cells = np.asarray(theta, dtype=np.float64)[mask]
        t_old = _temperature_from_potential_temperature(
            theta_cells, p_cells)
        # allow_wps_undershoot: the native-HRRR lane legitimately
        # carries bounded negative SPFH undershoot in its columns (the
        # same envelope real.py's own RH diagnosis admits); such a cell
        # diagnoses a negative RH, which the saturation relation below
        # clips to WRF's own 1e-6 floor.
        rh = _mixing_ratio_to_relative_humidity(
            t_old, p_cells, qv[mask], allow_wps_undershoot=True)
        t_new = _temperature_from_potential_temperature(
            theta_cells + delta, p_cells)
        qv_new = _saturation_mixing_ratio(t_new, p_cells, rh)
        if not np.isfinite(qv_new).all() or np.any(qv_new < 0.0):
            raise ValueError(
                "rh_preserve produced an invalid qv inside the bubble")
        max_delta = float(np.max(np.abs(qv_new - qv[mask])))
        qv[mask] = qv_new
        return max_delta


    def apply_to_state(self, state) -> dict:
        """Add the bubbles to an already-initialized :class:`DomainState`.

        The prepared-cache route (``gpuwm.prepared_domain_tree_forecast``)
        restores sealed, unperturbed per-domain states and never passes
        through ``initialize_real``; this is that route's application
        point, run after restore and BEFORE ``initialize_prepared_physics``
        whose ``update_diagnostics`` then rederives p/al/alt from the
        perturbed prognostics.  The bubble is evaluated in FP64 on host
        against the restored full theta, analyzed pressure ``state.p``
        and geopotential heights AGL, then written back to ``thp`` (and
        ``qv`` under ``rh_preserve``) on the bubble cells ONLY -- every
        other cell keeps its exact restored bytes.  Geopotential is NOT
        rebalanced here: this is WRF's own moist-bubble convention (the
        port's ``init_moist_balanced``, gpuwm/core/moist.py -- the bubble
        is added without recomputing the balance), and the receipt names
        the route so the two application points stay distinguishable.
        """
        def host(value):
            return np.asarray(value.get() if hasattr(value, "get")
                              else value, dtype=np.float64)

        thb = host(state.thb)
        if thb.ndim == 1:                 # flat base state: 1-D column
            thb = thb[:, None, None]
        theta = thb + host(state.thp)
        theta_before = theta.copy()
        qv = host(state.qv)
        qv_before = qv.copy()
        pressure = host(state.p)
        full_phi = host(state.phb) + host(state.php)
        z_half_agl = (0.5 * (full_phi[:-1] + full_phi[1:])
                      - full_phi[:1]) / c.G
        receipt = self.apply(theta=theta, qv=qv, pressure=pressure,
                             z_half_agl=z_half_agl)
        receipt["application_point"] = "restored-prepared-state"
        changed = theta != theta_before
        if changed.any():
            xp_mask = type(state.thp).__module__.startswith("cupy")
            if xp_mask:
                import cupy as cp
                mask = cp.asarray(changed)
                state.thp[mask] = cp.asarray(
                    (theta - thb)[changed].astype(np.float32))
            else:
                state.thp[changed] = (theta - thb)[changed].astype(
                    np.float32)
        qv_changed = qv != qv_before
        if qv_changed.any():
            if type(state.qv).__module__.startswith("cupy"):
                import cupy as cp
                state.qv[cp.asarray(qv_changed)] = cp.asarray(
                    qv[qv_changed].astype(np.float32))
            else:
                state.qv[qv_changed] = qv[qv_changed].astype(np.float32)
        return receipt


def build_initial_state_perturbation(perturbation, grid, *, grid_id: int,
                                     require_containment: bool):
    """The construction entry: ``None`` unless a block is configured.

    ``perturbation`` is ``ExperimentConfig.perturbation``.  The ``None``
    return is the whole OFF contract -- the caller holds no object and
    ``initialize_real`` executes no instruction of this module.
    ``require_containment`` is set by the COARSE domain (and the
    single-domain path): a center outside it is a configuration error.
    A child legitimately may not contain a bubble; its receipt says so.
    """
    if perturbation is None:
        return None
    return InitialStatePerturbation(
        perturbation.bubbles, grid, grid_id=grid_id,
        require_containment=require_containment)


__all__ = [
    "APPLICATION_SCHEMA",
    "InitialStatePerturbation",
    "build_initial_state_perturbation",
]
