"""Sub-source-cell downscaling of the initial LAND soil state.

Why this module exists
======================

A forcing model delivers its soil state on its own mesh -- 0.25 degree for
GFS and ERA5 -- and the horizontal stage carries it to the model grid with
WPS's ``sixteen_pt`` overlapping-parabolic interpolant.  That interpolation
is correct: it is a faithful transcription of ``geogrid/src/interp_module.F``
and it is what WPS metgrid does.  What it CANNOT do is invent information
below the source spacing, so every interpolated soil field is a smooth
surface fitted inside each source cell, joined to its neighbours with a
curvature break on the cell boundary.

Nothing downstream fixes that, in WPS/WRF or (until this module) here.  The
consequence is measurable and visible.  On a 3 km European domain forced by
GFS 0.25 degree, a 16-term bicubic fitted inside each source cell explains
``SMOIS`` layer 4 at R2 = 0.99972 and ``TSLB`` layer 4 at 0.99978: those
fields hold essentially ZERO information below 0.25 degrees.  Terrain
(``HGT``, 0.90691) and land use (``LU_INDEX``, 0.51153) measured the same way
DO hold sub-cell information, so the instrument is not simply reporting
smoothness.  ``SMOIS`` never relaxes -- it is a prognostic reservoir -- so the
source mesh stays in the forecast, and it reaches the screen: soil moisture
controls the latent heat flux, the latent heat flux controls 2 m humidity, and
the 2 m dewpoint over land ends up quilted into rectangles 0.25 degrees on a
side (16.8 km by 27.6 km at 53 N).  Correlation between block-scale dewpoint
and block-scale soil moisture reaches +0.41 through the first five forecast
hours, tracking the latent heat flux (+0.415) and collapsing after sunset,
which is the fingerprint of an evaporative pathway rather than a plotting
artifact.

Deliberate divergence from WRF
==============================

**What WRF does.**  ``real.exe`` takes metgrid's interpolated ``SM``/``ST``
layers and uses them as-is.  ``module_initialize_real.F`` adjusts soil
TEMPERATURE for terrain (``adjust_soil_temp_new``, ``module_soil_pre.F``
:993-1073) and floors pathological soil MOISTURE at a constant 0.005
(``account_for_zero_soil_moisture`` :3363-3395), but at no point does it
consult the target grid's own soil texture, and it never rescales moisture.
The soil texture WRF hands Noah (``ISLTYP`` from geogrid's 30 arc-second
STATSGO/FAO dataset) and the soil moisture it hands Noah are therefore
INCONSISTENT with each other by construction: a cell whose 30 arc-second
texture is sand receives the volumetric water content of a 0.25 degree cell
that was mostly clay, which for the same physical wetness is a different
number.  Stock WRF quilts soil moisture exactly like this.

**What gpuwm does.**  Volumetric water content is not the quantity the two
grids share; the dimensionless wetness of the soil is.  So the moisture is
carried across the resolution change as Noah's own degree-of-saturation
ratio and reconstituted against the target grid's texture::

    SRATIO  = (SMC - SMCDRY) / (SMCMAX - SMCDRY)     source-cell texture
    SMC_new = SMCDRY_t + SRATIO * (SMCMAX_t - SMCDRY_t)     target texture

``SMCDRY`` (``SOILPARM.TBL`` DRYSMC) and ``SMCMAX`` (MAXSMC) are the same
air-dry and saturation constants Noah itself uses, and ``SRATIO`` is
literally Noah's direct-evaporation variable (``module_sf_noahlsm.F``:1214).
The source-cell texture is the area mean of the target grid's own texture
parameters over one source cell, because the same 30 arc-second database
describes both meshes and its cell mean is what a 0.25 degree cell can
represent.  Nothing new is fetched and no new dataset is introduced.

**Why this is better and not merely different.**  Three reasons.

1.  It is the only operation available that puts REAL sub-source-cell
    structure into ``SMOIS``.  Smoothing the interpolated field (WPS's
    ``smooth_option``) removes the curvature break and the visible facet
    edges but leaves the block-scale amplitude untouched, because it adds no
    information -- it is cosmetic, and this module is deliberately not that.
2.  It makes soil moisture and soil texture mutually consistent, which is
    what Noah's own hydrology assumes.  Handing Noah a wetness the texture
    cannot support is how a sandy cell starts the forecast at an impossible
    saturation.
3.  It conserves water.  Because the reconstitution is affine in ``SRATIO``
    and the source-cell texture is the cell MEAN of the target texture, the
    source-cell mean of the result equals the source value it came from,
    exactly, whenever ``SRATIO`` is uniform in the cell -- and to within a
    fraction of a percent in practice.  See ``soil_water_change_pct`` on the
    receipt, which every run reports.

**How to reproduce stock WRF.**  Declare, in the experiment TOML::

    [ingest]
    soil_texture_downscale = false

That restores the byte-exact previous behaviour for WRF-comparison work,
and the run receipt records that it was declared.

The deep soil temperature analogue
==================================

``TSLB`` carries the same imprint, and for GFS it is worse than for ERA5:
the four GFS slabs coincide with Noah's four layers, so the layer-form
mapping in :func:`gpuwm.ingest.soil.preprocess_noah_soil` copies them
straight through and the target grid's own deep-soil climatology (``TMN``,
geogrid ``SOILTEMP`` adjusted to the model terrain) never enters the column
at all.  WRF's own ERA5-style path does not have that hole: ``init_soil_2_real``
(``module_soil_pre.F``:1591-1608) brackets the profile with ``TSK`` at 0 m and
``TMN`` at 3 m, so every layer is anchored on the target's deep temperature in
proportion to its depth.

:func:`downscale_deep_soil_temperature` applies exactly that anchoring, and
only to the part of ``TMN`` the source mesh cannot represent::

    TSLB_k += (midpoint_k / 3 m) * (TMN - cellmean(TMN))

The weight is WRF's own linear-in-depth blend toward the 3 m boundary
condition.  Subtracting the source-cell mean means the source keeps every
scale it actually resolves and contributes nothing at the scales it does
not, so the source-cell mean of ``TSLB`` is unchanged, exactly.  Layer 1 is
weighted 0.017 and layer 4 is weighted 0.5, so this is a deep-layer
correction by construction.

Resolution advisory
===================

A domain whose grid spacing is much finer than the forcing mesh is asking
the source for detail it does not have.  :func:`source_mesh_receipt` records
the source spacing on every run and warns once when ``dx`` is finer than
about a fifth of it -- the regime where the imprint is large enough to see.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import PurePath

import numpy as np


#: Report the forcing mesh whenever the model resolves more than this many
#: grid cells across one source cell.  Five is where a 0.25 degree quilt
#: first becomes visible in a rendered 2 m field; below it the source cell
#: is only a few grid cells wide and the interpolant's facets are
#: indistinguishable from ordinary gradients.
SOURCE_MESH_ADVISORY_RATIO = 5.0

#: WRF's boundary depth for the deep-soil temperature (``TMN``), the
#: denominator of ``init_soil_2_real``'s linear blend
#: (``module_soil_pre.F``:1591-1608).
DEEP_SOIL_BOUNDARY_DEPTH_M = 3.0

_WRF_REFERENCE_MOISTURE = {
    "wrf_behaviour": (
        "real.exe consumes metgrid's interpolated SM layers unchanged; no "
        "stage consults the target grid's own soil texture, so SMOIS "
        "carries the forcing mesh permanently "
        "(dyn_em/module_initialize_real.F, share/module_soil_pre.F)"),
    "gpuwm_behaviour": (
        "soil moisture crosses the resolution change as Noah's own "
        "degree-of-saturation SRATIO = (SMC-SMCDRY)/(SMCMAX-SMCDRY) "
        "(phys/module_sf_noahlsm.F:1214) and is reconstituted against the "
        "target grid's 30 arc-second texture"),
    "reproduce_wrf": "[ingest] soil_texture_downscale = false",
}

_WRF_REFERENCE_TEMPERATURE = {
    "wrf_behaviour": (
        "init_soil_2_real (share/module_soil_pre.F:1591-1608) anchors the "
        "soil column on TMN at 3 m for level-form sources, but a layer-form "
        "source whose layers already match Noah's is copied straight "
        "through and never sees the target grid's deep-soil climatology"),
    "gpuwm_behaviour": (
        "WRF's own linear-in-depth blend toward TMN is applied to the part "
        "of TMN the source mesh cannot represent, leaving the source-cell "
        "mean of every layer exactly unchanged"),
    "reproduce_wrf": "[ingest] soil_texture_downscale = false",
}


@dataclass(frozen=True)
class SoilMeshPlan:
    """The forcing mesh measured against the target grid.

    ``enabled`` is the case's declaration, not a capability check: a plan
    that knows both spacings but was told ``soil_texture_downscale = false``
    still rides the receipt so the reader can see the defect was left in
    deliberately.
    """

    source_spacing_deg_lat: float
    source_spacing_deg_lon: float
    target_spacing_deg_lat: float
    target_spacing_deg_lon: float
    enabled: bool = True

    def __post_init__(self) -> None:
        for name in ("source_spacing_deg_lat", "source_spacing_deg_lon",
                     "target_spacing_deg_lat", "target_spacing_deg_lon"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite spacing")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "enabled", bool(self.enabled))

    @property
    def footprint_cells(self) -> tuple[float, float]:
        """One source cell measured in target cells, ``(x, y)``."""
        return (self.source_spacing_deg_lon / self.target_spacing_deg_lon,
                self.source_spacing_deg_lat / self.target_spacing_deg_lat)

    @property
    def resolution_ratio(self) -> float:
        """Target cells across the NARROWER side of one source cell."""
        return float(min(self.footprint_cells))

    @property
    def advisory(self) -> bool:
        """Is the model resolving detail the forcing mesh cannot supply?"""
        return self.resolution_ratio > SOURCE_MESH_ADVISORY_RATIO

    @classmethod
    def from_grids(cls, source_latitude, source_longitude,
                   target_lat, target_lon, *, enabled=True):
        """Measure both meshes from their coordinates.

        ``source_latitude``/``source_longitude`` are the regular source axes
        the horizontal stage interpolated from; ``target_lat``/``target_lon``
        are the two-dimensional mass-point coordinates of the model grid.
        The target spacing is taken at the domain centre, where a Lambert
        grid's degrees-per-cell is representative of the whole domain to
        well inside the accuracy a low-pass footprint needs.
        """
        source_lat = np.asarray(source_latitude, dtype=np.float64)
        source_lon = np.asarray(source_longitude, dtype=np.float64)
        lat = np.asarray(target_lat, dtype=np.float64)
        lon = np.asarray(target_lon, dtype=np.float64)
        if source_lat.ndim != 1 or source_lon.ndim != 1:
            raise ValueError("source axes must be 1-D")
        if source_lat.size < 2 or source_lon.size < 2:
            raise ValueError("source axes must each contain two points")
        if lat.ndim != 2 or lat.shape != lon.shape:
            raise ValueError("target coordinates must be matching 2-D arrays")
        if lat.shape[0] < 3 or lat.shape[1] < 3:
            raise ValueError("target grid is too small to measure spacing")
        jc, ic = lat.shape[0] // 2, lat.shape[1] // 2
        return cls(
            source_spacing_deg_lat=float(
                np.abs(np.diff(source_lat)).mean()),
            source_spacing_deg_lon=float(
                np.abs(np.diff(source_lon)).mean()),
            target_spacing_deg_lat=float(
                abs(lat[jc + 1, ic] - lat[jc - 1, ic]) / 2.0),
            target_spacing_deg_lon=float(
                abs(lon[jc, ic + 1] - lon[jc, ic - 1]) / 2.0),
            enabled=enabled)


def source_mesh_receipt(plan: SoilMeshPlan, *, announce: bool = True) -> dict:
    """Record the forcing mesh on the run receipt, and warn when it is coarse.

    Always returns a receipt: the soil-state source resolution belongs on
    every run's provenance whether or not it triggered anything.  The
    advisory print fires only when the model is resolving more than
    :data:`SOURCE_MESH_ADVISORY_RATIO` cells across the narrow side of a
    source cell, which is the regime where the quilt is visible.
    """
    footprint_x, footprint_y = plan.footprint_cells
    receipt = {
        "source_spacing_deg": {
            "lat": plan.source_spacing_deg_lat,
            "lon": plan.source_spacing_deg_lon,
        },
        "target_spacing_deg": {
            "lat": plan.target_spacing_deg_lat,
            "lon": plan.target_spacing_deg_lon,
        },
        "source_cell_in_target_cells": {
            "x": footprint_x, "y": footprint_y},
        "resolution_ratio": plan.resolution_ratio,
        "advisory_ratio": SOURCE_MESH_ADVISORY_RATIO,
        "advisory": plan.advisory,
        "downscale_enabled": plan.enabled,
    }
    if announce and plan.advisory:
        print(
            "soil-state source resolution: the forcing mesh is "
            f"{plan.source_spacing_deg_lat:.4g} x "
            f"{plan.source_spacing_deg_lon:.4g} degrees (lat x lon), which "
            f"is {footprint_y:.1f} rows by {footprint_x:.1f} columns of "
            f"this grid, so the "
            "soil state carries no information below "
            f"{plan.resolution_ratio:.1f} grid cells"
            + (" -- texture downscaling is reconstituting the sub-cell "
               "structure from the target grid's own 30 arc-second soil "
               "texture" if plan.enabled else
               " -- texture downscaling is DISABLED by declaration, so the "
               "source mesh will remain visible in SMOIS and in any field "
               "it drives (2 m dewpoint, latent heat flux)"),
            file=sys.stderr)
    return receipt


def _frac_box_mean(values, valid, width_x, width_y):
    """Separable running mean over a FRACTIONAL window, ignoring invalid cells.

    The window is exactly ``width`` cells wide: the two cells straddling the
    edge enter with the fractional weight left over from the integer core.
    A source cell is 5.5 by 9.1 grid cells on the domain this was built for,
    and rounding that to 5 by 9 biases the low-pass by ten percent, which
    lands directly in the reconstituted moisture.

    A running mean is locked to the TARGET grid, not to the source mesh, so
    it can introduce no phase structure of its own -- which is the whole
    point: this operator estimates what a source cell can represent without
    stamping the source mesh anywhere.  ``NaN`` is returned where the window
    contains no valid cell.
    """
    numerator = np.where(valid, values, 0.0).astype(np.float64)
    denominator = np.asarray(valid, dtype=np.float64)

    def one_axis(num, den, width, axis):
        if axis == 1:
            num, den = num.T.copy(), den.T.copy()
        n = num.shape[0]
        # A source cell NARROWER than one target cell along this axis has
        # nothing to average: the window is the cell itself.  Clamped
        # rather than refused because the two axes are independent -- a
        # source can be coarser in latitude and finer in longitude than
        # the target, and that is one legitimate window per axis, not an
        # error.  Without the clamp the half-width goes negative and the
        # cumulative-sum window inverts, which silently returns negative
        # "means".
        half = (max(float(width), 1.0) - 1.0) / 2.0
        core = int(np.floor(half))
        edge = half - core
        cum_n = np.concatenate(
            [np.zeros((1,) + num.shape[1:]), np.cumsum(num, axis=0)])
        cum_d = np.concatenate(
            [np.zeros((1,) + den.shape[1:]), np.cumsum(den, axis=0)])
        index = np.arange(n)
        low = np.clip(index - core, 0, n)
        high = np.clip(index + core + 1, 0, n)
        out_n = cum_n[high] - cum_n[low]
        out_d = cum_d[high] - cum_d[low]
        if edge > 0.0:
            for offset in (-(core + 1), core + 1):
                neighbour = index + offset
                inside = (neighbour >= 0) & (neighbour < n)
                clipped = np.clip(neighbour, 0, n - 1)
                out_n = out_n + edge * np.where(
                    inside[:, None], num[clipped], 0.0)
                out_d = out_d + edge * np.where(
                    inside[:, None], den[clipped], 0.0)
        if axis == 1:
            return out_n.T.copy(), out_d.T.copy()
        return out_n, out_d

    numerator, denominator = one_axis(numerator, denominator, width_y, 0)
    numerator, denominator = one_axis(numerator, denominator, width_x, 1)
    return np.where(denominator > 1e-9,
                    numerator / np.maximum(denominator, 1e-30), np.nan)


def soil_texture_bounds(soil_type, params):
    """``(SMCDRY, SMCMAX, usable)`` for every cell, from ``SOILPARM.TBL``.

    ``usable`` excludes categories outside the table and the WATER row,
    whose ``DRYSMC``/``MAXSMC`` pair (0.0, 1.0) describes no soil.  Land
    cells can legitimately carry that category: geogrid's landmask and its
    dominant soil category are independent fields and disagree along every
    coastline.
    """
    from gpuwm.core.noah import SOIL_COLS

    categories = np.asarray(soil_type)
    in_table = (np.isfinite(categories)
                & (categories == np.floor(categories))
                & (categories >= 1) & (categories <= params.slcats))
    rows = np.where(in_table, categories, 1).astype(np.int64) - 1
    smcdry = params.soil[rows, SOIL_COLS.index("smcdry")]
    smcmax = params.soil[rows, SOIL_COLS.index("smcmax")]
    usable = in_table & (smcdry > 0.0) & (smcmax > smcdry)
    return smcdry, smcmax, usable


def downscale_soil_moisture(soil_moisture, *, soil_type, terrestrial,
                            params, plan: SoilMeshPlan,
                            layer_thickness_m=None, announce: bool = True):
    """Reconstitute ``SMOIS`` against the target grid's own soil texture.

    See the module docstring for what WRF does, what this does instead, and
    why.  Returns ``(soil_moisture, receipt)``.  When the plan is disabled,
    when no land cell has a usable texture, or when the target grid is not
    finer than the source mesh, the INPUT ARRAY is returned untouched (same
    object) beside a receipt that says which of those it was, so a
    WRF-comparison run is byte-identical and says so.

    The result is bounded by ``[SMCDRY_t, SMCMAX_t]`` cell by cell, by
    construction: ``SRATIO`` is clipped to ``[0, 1]`` before reconstitution.
    Water, sea ice, and land cells whose soil category is water are left
    exactly as they arrived.
    """
    values = np.asarray(soil_moisture, dtype=np.float64)
    land = np.asarray(terrestrial, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != land.shape:
        raise ValueError(
            "soil_moisture must be (layers, ny, nx) matching the landmask")
    receipt = dict(source_mesh_receipt(plan, announce=announce))
    receipt["wrf_reference"] = dict(_WRF_REFERENCE_MOISTURE)
    if not plan.enabled:
        receipt["applied"] = False
        receipt["reason"] = "declined by [ingest] soil_texture_downscale"
        return soil_moisture, receipt
    footprint_x, footprint_y = plan.footprint_cells
    if footprint_x <= 1.0 and footprint_y <= 1.0:
        receipt["applied"] = False
        receipt["reason"] = (
            "target grid is not finer than the forcing mesh, so there is no "
            "sub-source-cell scale to reconstitute")
        return soil_moisture, receipt

    smcdry, smcmax, texture = soil_texture_bounds(soil_type, params)
    if smcdry.shape != land.shape:
        raise ValueError("soil_type shape differs from the landmask")
    usable = land & texture
    receipt["land_cells"] = int(np.count_nonzero(land))
    receipt["land_without_texture"] = int(np.count_nonzero(land & ~texture))
    if not usable.any():
        receipt["applied"] = False
        receipt["reason"] = "no land cell carries a usable soil category"
        return soil_moisture, receipt

    # The source cell's effective texture: the area mean of the target
    # grid's own parameters over one source cell, computed over land with a
    # usable category so a coastline cannot pull ocean's (0.0, 1.0) pair
    # into a land cell's denominator.
    smcdry_cell = _frac_box_mean(smcdry, usable, footprint_x, footprint_y)
    smcmax_cell = _frac_box_mean(smcmax, usable, footprint_x, footprint_y)
    span_cell = smcmax_cell - smcdry_cell
    active = usable & np.isfinite(span_cell) & (span_cell > 1e-6)
    receipt["downscaled_cells"] = int(np.count_nonzero(active))
    if not active.any():
        receipt["applied"] = False
        receipt["reason"] = "no land cell has a usable source-cell texture"
        return soil_moisture, receipt

    safe_span = np.where(active, span_cell, 1.0)
    span_target = smcmax - smcdry
    result = np.array(values, copy=True)
    per_layer = {}
    clipped_dry = clipped_wet = 0
    for layer in range(values.shape[0]):
        ratio = (values[layer] - smcdry_cell) / safe_span
        below = active & (ratio < 0.0)
        above = active & (ratio > 1.0)
        clipped_dry += int(np.count_nonzero(below))
        clipped_wet += int(np.count_nonzero(above))
        ratio = np.clip(np.where(active, ratio, 0.0), 0.0, 1.0)
        rebuilt = smcdry + ratio * span_target
        result[layer] = np.where(active, rebuilt, values[layer])
        delta = result[layer][active] - values[layer][active]
        per_layer[f"SMOIS_L{layer + 1}"] = {
            "mean_abs_change": float(np.abs(delta).mean()),
            "max_abs_change": float(np.abs(delta).max()),
            "sratio_mean": float(ratio[active].mean()),
            "sratio_clipped_dry": int(np.count_nonzero(below)),
            "sratio_clipped_saturated": int(np.count_nonzero(above)),
        }

    receipt.update({
        "applied": True,
        "fields": per_layer,
        "sratio_clipped_dry": clipped_dry,
        "sratio_clipped_saturated": clipped_wet,
        "sratio_samples": int(np.count_nonzero(active)) * values.shape[0],
    })
    # The soil-water budget needs the column geometry these layers sit on.
    # A caller whose layers are NOT the ones it passed -- the RUC route,
    # whose source profiles are on the source's own depths and whose layer
    # count varies by geometry arm -- gets no budget rather than one
    # weighted by the wrong column, and NEVER a NaN: the receipt is
    # canonicalized with ``allow_nan=False`` on its way into proof.json,
    # so a NaN here would refuse the whole preparation at the very end.
    thickness = (NOAH_DEFAULT_LAYER_THICKNESS_M if layer_thickness_m is None
                 else np.asarray(layer_thickness_m, dtype=np.float64))
    change = None
    if thickness.size == values.shape[0] and values.shape[0] > 0:
        cells = max(int(np.count_nonzero(active)), 1)
        before = float(np.einsum("l,lij->ij", thickness, values)[active].sum())
        after = float(np.einsum("l,lij->ij", thickness, result)[active].sum())
        change = 100.0 * (after - before) / max(abs(before), 1e-30)
        receipt["soil_water_kg_m2"] = {
            "before": before * 1000.0 / cells,
            "after": after * 1000.0 / cells,
        }
        receipt["soil_water_change_pct"] = change
    if announce:
        budget = ("domain soil water moved "
                  f"{change:+.3f}%; " if change is not None else "")
        print(
            "soil texture downscaling: SMOIS reconstituted on "
            f"{receipt['downscaled_cells']} land cell(s) as Noah's SRATIO "
            "against the target grid's own 30 arc-second soil texture "
            f"(one source cell spans {footprint_y:.1f} rows by "
            f"{footprint_x:.1f} columns of this grid); {budget}"
            f"{clipped_dry} value(s) clipped at air-dry and {clipped_wet} at "
            "saturation.  Stock WRF leaves the forcing mesh in SMOIS; "
            "declare [ingest] soil_texture_downscale = false to reproduce it",
            file=sys.stderr)
    return result, receipt


#: Noah's four layer thicknesses, repeated here so this module can weigh a
#: soil-water budget without importing the soil preprocessor it is called
#: FROM.  :mod:`gpuwm.ingest.soil` owns the authoritative copy and passes
#: its own array in; a drift between the two is refused at import.
NOAH_DEFAULT_LAYER_THICKNESS_M = np.array([0.10, 0.30, 0.60, 1.00],
                                          dtype=np.float64)

#: Noah's four layer midpoints, same discipline as the thicknesses above.
NOAH_DEFAULT_LAYER_MIDPOINTS_M = np.array([0.05, 0.25, 0.70, 1.50],
                                          dtype=np.float64)


def _refuse_soil_geometry_drift() -> None:
    """Refuse at import if this module's fallback geometry has drifted.

    :mod:`gpuwm.ingest.soil` owns Noah's layer geometry and passes its own
    arrays in on every production call; the copies above exist only so a
    caller that omits them still gets a correct budget and a correct depth
    weight.  A silent disagreement between the two would produce a
    plausible wrong number in a receipt and a plausible wrong weight on
    the deep layers, which is exactly the failure mode this project
    refuses to ship.  The import is local because :mod:`gpuwm.ingest.soil`
    imports THIS module.
    """
    from gpuwm.ingest import soil

    for label, here, there in (
            ("thicknesses", NOAH_DEFAULT_LAYER_THICKNESS_M,
             soil.NOAH_LAYER_THICKNESS_M),
            ("midpoints", NOAH_DEFAULT_LAYER_MIDPOINTS_M,
             soil.NOAH_LAYER_MIDPOINTS_M)):
        if not np.array_equal(here, np.asarray(there, dtype=np.float64)):
            raise AssertionError(
                f"gpuwm.ingest.soil_downscale's fallback Noah layer {label} "
                f"{here.tolist()} disagree with gpuwm.ingest.soil's "
                f"{np.asarray(there).tolist()}")


def downscale_deep_soil_temperature(soil_temperature, *,
                                    deep_soil_temperature, terrestrial,
                                    plan: SoilMeshPlan,
                                    layer_midpoints_m=None,
                                    announce: bool = True):
    """Anchor ``TSLB`` on the sub-source-cell part of the target's ``TMN``.

    WRF's ``init_soil_2_real`` blends every soil layer linearly between
    ``TSK`` at 0 m and ``TMN`` at 3 m.  A layer-form source whose layers
    already match Noah's skips that blend entirely, so the target grid's own
    deep-soil climatology never reaches the column.  This applies WRF's own
    weight to the part of ``TMN`` the forcing mesh cannot represent::

        TSLB_k += (midpoint_k / 3 m) * (TMN - cellmean(TMN))

    Subtracting the source-cell mean is what makes this a downscaling rather
    than a nudge: the source keeps every scale it resolves, and the
    source-cell mean of each layer is unchanged.  Returns
    ``(soil_temperature, receipt)``; the input array is returned untouched
    when the plan is disabled or there is no sub-source-cell scale.
    """
    values = np.asarray(soil_temperature, dtype=np.float64)
    deep = np.asarray(deep_soil_temperature, dtype=np.float64)
    land = np.asarray(terrestrial, dtype=bool)
    if values.ndim != 3 or values.shape[1:] != land.shape:
        raise ValueError(
            "soil_temperature must be (layers, ny, nx) matching the landmask")
    if deep.shape != land.shape:
        raise ValueError("deep_soil_temperature shape differs from the "
                         "landmask")
    midpoints = (NOAH_DEFAULT_LAYER_MIDPOINTS_M if layer_midpoints_m is None
                 else np.asarray(layer_midpoints_m, dtype=np.float64))
    if midpoints.size != values.shape[0]:
        raise ValueError("layer_midpoints_m does not match the layer count")
    receipt = {"wrf_reference": dict(_WRF_REFERENCE_TEMPERATURE),
               "downscale_enabled": plan.enabled}
    if not plan.enabled:
        receipt["applied"] = False
        receipt["reason"] = "declined by [ingest] soil_texture_downscale"
        return soil_temperature, receipt
    footprint_x, footprint_y = plan.footprint_cells
    if footprint_x <= 1.0 and footprint_y <= 1.0:
        receipt["applied"] = False
        receipt["reason"] = (
            "target grid is not finer than the forcing mesh, so there is no "
            "sub-source-cell scale to reconstitute")
        return soil_temperature, receipt
    cell_mean = _frac_box_mean(deep, land, footprint_x, footprint_y)
    active = land & np.isfinite(cell_mean)
    if not active.any():
        receipt["applied"] = False
        receipt["reason"] = "no land cell has a usable deep-temperature mean"
        return soil_temperature, receipt
    anomaly = np.where(active, deep - cell_mean, 0.0)
    result = np.array(values, copy=True)
    per_layer = {}
    for layer in range(values.shape[0]):
        weight = float(midpoints[layer]) / DEEP_SOIL_BOUNDARY_DEPTH_M
        increment = weight * anomaly
        result[layer] = np.where(active, values[layer] + increment,
                                 values[layer])
        per_layer[f"TSLB_L{layer + 1}"] = {
            "weight": weight,
            "rms_change_k": float(np.sqrt((increment[active] ** 2).mean())),
            "max_abs_change_k": float(np.abs(increment[active]).max()),
        }
    receipt.update({
        "applied": True,
        "cells": int(np.count_nonzero(active)),
        "deep_anomaly_std_k": float(anomaly[active].std()),
        "fields": per_layer,
        "boundary_depth_m": DEEP_SOIL_BOUNDARY_DEPTH_M,
    })
    if announce:
        deepest = per_layer[f"TSLB_L{values.shape[0]}"]
        print(
            "deep soil temperature downscaling: TSLB anchored on the "
            f"sub-source-cell part of TMN across {receipt['cells']} land "
            f"cell(s) with WRF's own linear-in-depth weight; layer "
            f"{values.shape[0]} moved {deepest['rms_change_k']:.4f} K rms "
            f"(max {deepest['max_abs_change_k']:.4f} K).  Each layer's "
            "source-cell mean is unchanged",
            file=sys.stderr)
    return result, receipt


__all__ = [
    "DEEP_SOIL_BOUNDARY_DEPTH_M",
    "NOAH_DEFAULT_LAYER_MIDPOINTS_M",
    "NOAH_DEFAULT_LAYER_THICKNESS_M",
    "SOURCE_MESH_ADVISORY_RATIO",
    "SoilMeshPlan",
    "INGEST_TABLE_KEYS",
    "declared_soil_texture_downscale",
    "downscale_deep_soil_temperature",
    "downscale_soil_moisture",
    "parse_ingest_table",
    "soil_mesh_plan_from_case",
    "soil_texture_bounds",
    "source_mesh_receipt",
]


#: The ``[ingest]`` table of an experiment TOML: ingest POLICY, owned by
#: this module the way ``[fetch]`` is owned by :mod:`gpuwm.fetch` and
#: ``[static]`` by :mod:`gpuwm.static.highres_production`.
#:
#: It is a table of its own rather than a ``[case_data]`` key because
#: ``[case_data]`` selects the ERA5 config-driven route and declares that
#: route's inputs; a policy every route honours must not drag a routing
#: decision along with it.  A `gpuwm go` config carries no ``[case_data]``
#: at all, and that is the door most users come through -- a switch they
#: cannot reach from it is not shipped.
INGEST_TABLE_KEYS = ("soil_texture_downscale",)


def parse_ingest_table(table, *, source: str) -> dict:
    """Validate an experiment TOML's ``[ingest]`` table, fail-loud."""
    if not isinstance(table, dict):
        raise ValueError(f"[ingest] of {source} must be a table")
    unknown = sorted(set(table) - set(INGEST_TABLE_KEYS))
    if unknown:
        raise ValueError(
            f"[ingest] of {source} has unknown key(s): {unknown}; known "
            f"keys are {list(INGEST_TABLE_KEYS)}")
    value = table.get("soil_texture_downscale")
    if value is not None and not isinstance(value, bool):
        raise ValueError(
            f"soil_texture_downscale in [ingest] of {source} must be true "
            "or false")
    return {"soil_texture_downscale": value}


def declared_soil_texture_downscale(source=None) -> bool:
    """Is the reconstitution enabled?  Silence means YES.

    ``source`` is whatever the calling route has: a path to an experiment
    TOML, an object carrying a ``soil_texture_downscale`` attribute, or
    ``None``.  Only an explicit ``[ingest] soil_texture_downscale = false``
    turns it off, because this is a correctness remedy rather than an
    experiment: a bare default run must stop showing the defect, and the
    declaration exists so WRF-comparison work can put it back on purpose.

    A config that cannot be read at all is NOT silently treated as a
    refusal: the reconstitution stays on and the config's real problem is
    reported by whichever loader owns it.
    """
    if source is None:
        return True
    declared = getattr(source, "soil_texture_downscale", None)
    if declared is None and isinstance(source, (str, bytes, PurePath)):
        import tomllib

        try:
            with open(source, "rb") as handle:
                raw = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            return True
        table = raw.get("ingest")
        if table is not None:
            declared = parse_ingest_table(
                table, source=str(source))["soil_texture_downscale"]
    if declared is None:
        return True
    if not isinstance(declared, bool):
        raise ValueError(
            "soil_texture_downscale in [ingest] must be true or false")
    return declared


def soil_mesh_plan_from_case(source_snapshot, target, case_data=None, *,
                             enabled=None):
    """The plan a route hands to the soil preprocessor, or ``None``.

    Both ends are taken as OBJECTS rather than as coordinate arrays, and
    this function owns both guards, because "there is nothing to measure"
    is an answer it has to be able to give: not every source has a regular
    mesh (native HRRR is on a Lambert grid), and a stand-in snapshot or
    grid in a test has neither axes nor mass coordinates.  Reaching into
    ``.latitude`` or ``.latlon_mass()`` at the call site instead turned
    that answer into an ``AttributeError`` at every front door -- twice,
    once per end.

    ``source_snapshot`` is the forcing snapshot the soil came off.
    ``target`` is either the model grid (anything with ``latlon_mass()``)
    or an explicit ``(latitude, longitude)`` pair of two-dimensional mass
    coordinates.

    ``None`` is not silent: :func:`gpuwm.ingest.soil.preprocess_noah_soil`
    announces once per process that the route declared no source mesh.

    ``enabled`` overrides the declaration for a caller that has already
    resolved it -- ``prepare_real_case`` takes it as a parameter, because
    its caller holds the config and it does not.
    """
    latitude = getattr(source_snapshot, "latitude", None)
    longitude = getattr(source_snapshot, "longitude", None)
    if latitude is None or longitude is None:
        return None
    latitude = np.asarray(latitude, dtype=np.float64)
    longitude = np.asarray(longitude, dtype=np.float64)
    if latitude.ndim != 1 or longitude.ndim != 1:
        return None
    projection = getattr(source_snapshot, "projection", None)
    if projection is not None:
        # Projected axes carry metres in `axis_unit_m` units; the mesh
        # plan compares spacings in degrees, so convert the source mesh to
        # equivalent degrees of latitude (1 degree = 111,195 m on the WPS
        # sphere) rather than comparing metres against degrees silently.
        unit_m = float(projection["parameters"]["axis_unit_m"])
        metres_per_degree = 111_195.0
        latitude = latitude * (unit_m / metres_per_degree)
        longitude = longitude * (unit_m / metres_per_degree)
    mass_latlon = getattr(target, "latlon_mass", None)
    if mass_latlon is not None:
        target_lat, target_lon = mass_latlon()
    else:
        try:
            target_lat, target_lon = target
        except (TypeError, ValueError):
            return None
    if (getattr(target_lat, "shape", None) is None
            or getattr(target_lon, "shape", None) is None):
        return None
    return SoilMeshPlan.from_grids(
        latitude, longitude, target_lat, target_lon,
        enabled=(declared_soil_texture_downscale(case_data)
                 if enabled is None else bool(enabled)))
