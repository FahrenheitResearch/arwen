"""GOES cloud water path, from two packs to one gridded observation set.

This is the consumer the bridge's separate-pack ruling names.  ``rw_goes``
emits the 2 km CWP pack and the 10 km cloud-top pack and **never regrids**
between them, on the grounds that a resample buried in an ingest tool is
an interpolation no science reviewed.  This module is where that
interpolation happens, where it is chosen by name, and where it is written
into a receipt: :func:`join_cloud_top` states its method, its coverage,
and what it refused.

Three separate things happen here, and they are separate on purpose.

**The join** (:func:`join_cloud_top`) puts the 10 km cloud-top height onto
the 2 km CWP grid.  Default ``nearest``, in the geostationary fixed-grid
scan-angle space both packs carry -- not because nearest is more accurate
than bilinear in general, but because cloud-top height is *discontinuous
at a cloud edge* and a bilinear blend across that edge invents a height no
pixel observed.  ``bilinear`` is available and is recorded when used.

**The QC** (:func:`grid_cwp`) is three gates in series.  The bridge's DQF
gate already ran: condemned pixels arrive as ``NaN`` and this module
honours that by construction, carrying the rule and condemn mask into its
own receipt so the screening can be read downstream.  Then a *derivation
cross-check*: CWP is re-derived here from the pack's own ``cod``/``cps``/
``phase`` planes using the pack's own declared coefficients, and a pack
whose ``cwp`` plane does not reproduce is refused.  Then the superob
gates: minimum valid pixels, minimum valid fraction, and phase-class
uniformity.

**The error model** (:class:`CwpErrorModel`) requires explicit constants.
The automatic window caller supplies documented provisional values;
explicit pack gridding can supply another policy. Every value used is
written into the product's receipt and labelled UNCALIBRATED.

Units are g m-2 throughout, the pack's own unit and the unit
:func:`gpuwm.da.obsop_cwp.simulated_cloud_water_path` returns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from gpuwm.obs.goes_pack import (CLOUDTOP_SCHEMAS, CWP_SCHEMAS,
                                 PROJECTION_KEYS, GoesPack, read_goes_pack)

#: ABI ACTP cloud-top phase codes, as ``rw_sat::cwp::CloudPhase`` decodes
#: them (``flag_values`` 0..=5, verified against real GOES-19 granules).
PHASE_CLEAR = 0
PHASE_LIQUID = 1
PHASE_SUPERCOOLED = 2
PHASE_MIXED = 3
PHASE_ICE = 4
PHASE_UNKNOWN = 5

#: The three classes the error model and the uniformity gate work in.
#: Supercooled liquid rides with liquid and mixed-phase rides with ice,
#: which is the same branch split ``rw_sat::cwp`` uses to pick a density
#: -- so the class a cell is given here is the class its CWP was derived
#: under, not a second, disagreeing opinion.
CLASS_NONE = -1
CLASS_CLEAR = 0
CLASS_LIQUID = 1
CLASS_ICE = 2

CLASS_NAMES = {CLASS_NONE: "none", CLASS_CLEAR: "clear",
               CLASS_LIQUID: "liquid", CLASS_ICE: "ice"}

_PHASE_TO_CLASS = {
    PHASE_CLEAR: CLASS_CLEAR,
    PHASE_LIQUID: CLASS_LIQUID,
    PHASE_SUPERCOOLED: CLASS_LIQUID,
    PHASE_MIXED: CLASS_ICE,
    PHASE_ICE: CLASS_ICE,
}

#: What ``cloud_top_height_m`` is measured from.  ABI ACHA publishes cloud
#: top height above the geoid, and :class:`~gpuwm.obs.target_grid.TargetGrid`
#: ``z_w`` is above mean sea level, so the two are used as the same datum.
#: NAMED ASSUMPTION: a geoid/ellipsoid/MSL offset of tens of metres is not
#: corrected here.  It is far below a model layer's depth at cloud-top
#: altitudes and far above zero, so it is stated rather than silently
#: absorbed.
CLOUD_TOP_DATUM = ("ABI ACHA height above the geoid, used directly as "
                   "height above mean sea level against TargetGrid.z_w; "
                   "no geoid-to-MSL correction is applied")

#: Join methods :func:`join_cloud_top` will express.
JOIN_METHODS = ("nearest", "bilinear")

#: DCOMP quality bits the operator spec inflates rather than gates.  They
#: are deliberately OUTSIDE the bridge's condemn mask (88 = snow/sea-ice 8
#: | twilight 16 | glint 64, measured on the live packs), so pixels
#: carrying them arrive as observations and it is this stage's job to say
#: they are worth less -- not to drop them.
#: Numeric masks follow the COD/CPS CF ``flag_masks`` and
#: ``flag_meanings``: 256 is thick cloud, 512 is thin cloud.
DCOMP_THIN_BIT = 512
DCOMP_THICK_BIT = 256

#: Which products' DQF words carry those bits.  Both are DCOMP outputs and
#: measured identical on the live granule, but the union is taken rather
#: than either alone: "they agreed on this scan" is an observation about
#: one scan, not a property of the format.
DCOMP_PRODUCTS = ("COD", "CPS")


class GoesCwpError(ValueError):
    """The packs, the grid and the policy cannot be reconciled."""


# ---------------------------------------------------------------------------
# the error model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CwpErrorModel:
    """Observation error standard deviations for gridded CWP, in g m-2.

    Every field is required.  This project has not measured a CWP
    observation-error covariance and will not present a borrowed number as
    one, so there is nothing here to default to; the caller states the
    numbers and this class records them where the receipt can find them.

    clear_g_m2
        The error on a clear-sky zero.  Small: within the DQF gate the
        retrieval survived, "there is no cloud here" is the highest
        confidence statement it makes, and it is the statement that lets
        the analysis *remove* invented cloud.

    rel_liquid / floor_liquid_g_m2
        ``max(rel_liquid * CWP, floor_liquid_g_m2)`` for a liquid or
        supercooled-liquid cell.

    rel_ice / floor_ice_g_m2
        The same form for an ice or mixed-phase cell.  ``rel_ice`` must be
        at least ``rel_liquid``: the ice branch of the upstream CWP
        relation is PROVISIONAL (``rw_sat::cwp``, spherical-particle form
        at bulk ice density, flagged in every pack), and an error model
        that gave ice the same confidence as liquid would be contradicting
        the pack it reads.
    """

    clear_g_m2: float
    rel_liquid: float
    floor_liquid_g_m2: float
    rel_ice: float
    floor_ice_g_m2: float
    #: Multiply sigma_o for a pixel whose DCOMP thin-cloud (512) or
    #: thick-cloud (256) bit is set.  1.0 -- the default -- is the v1
    #: behaviour: no inflation, because a v1 pack cannot support it.
    #: Requesting >1.0 against a pack with no per-pixel DQF plane is a
    #: refusal, not a silent no-op.  Like every other number here these
    #: are UNCALIBRATED.
    thin_inflation: float = 1.0
    thick_inflation: float = 1.0

    def validate(self) -> None:
        for name in ("clear_g_m2", "floor_liquid_g_m2", "floor_ice_g_m2",
                     "rel_liquid", "rel_ice"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise GoesCwpError(
                    f"CwpErrorModel.{name} must be finite and positive, got "
                    f"{value!r}. A zero or negative observation error is an "
                    "infinite weight on a number nobody calibrated")
        for name in ("thin_inflation", "thick_inflation"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 1.0:
                raise GoesCwpError(
                    f"CwpErrorModel.{name} must be finite and >= 1, got "
                    f"{value!r}. These bits mark retrievals that are harder "
                    "to trust; deflating their error would claim the "
                    "opposite of what the flag says")
        if float(self.rel_ice) < float(self.rel_liquid):
            raise GoesCwpError(
                f"rel_ice ({self.rel_ice!r}) is below rel_liquid "
                f"({self.rel_liquid!r}). The ice branch of the upstream CWP "
                "relation is flagged PROVISIONAL in every pack; an error "
                "model that trusts it more than the liquid branch "
                "contradicts its own observations")

    def wants_inflation(self) -> bool:
        """Whether this model asks for anything a v1 pack cannot give."""

        return (float(self.thin_inflation) > 1.0
                or float(self.thick_inflation) > 1.0)

    def to_payload(self, *, applied: bool | None = None,
                   reason: str | None = None, counts: dict | None = None
                   ) -> dict:
        self.validate()
        return {
            "units": "g m-2",
            "clear_g_m2": float(self.clear_g_m2),
            "rel_liquid": float(self.rel_liquid),
            "floor_liquid_g_m2": float(self.floor_liquid_g_m2),
            "rel_ice": float(self.rel_ice),
            "floor_ice_g_m2": float(self.floor_ice_g_m2),
            "form": ("clear: constant; cloudy: max(rel * CWP, floor), "
                     "per phase class"),
            "calibration": "UNCALIBRATED",
            "calibration_note": (
                "These are the caller's stated constants, not a measured "
                "CWP observation-error covariance. No such covariance has "
                "been established for this system. They are required "
                "arguments precisely so that no stage can inherit a "
                "confident-looking default nobody earned, and they are "
                "recorded here so the scoreboard can A/B them."),
            "thin_thick_inflation": {
                "applied": bool(self.wants_inflation() if applied is None
                                else applied),
                "thin_bit": DCOMP_THIN_BIT,
                "thick_bit": DCOMP_THICK_BIT,
                "thin_inflation": float(self.thin_inflation),
                "thick_inflation": float(self.thick_inflation),
                "products": list(DCOMP_PRODUCTS),
                "combination": (
                    "a pixel is thin/thick if EITHER product's DQF word "
                    "says so; a pixel carrying both takes the product of "
                    "both factors"),
                "cell_rule": (
                    "a cell's sigma_o is multiplied by the MEAN inflation "
                    "factor over the valid pixels averaged into it, "
                    "because the cell's value is their areal mean"),
                "calibration": "UNCALIBRATED",
                **({"reason": reason} if reason else {}),
                **({"counts": counts} if counts else {}),
            },
        }


# ---------------------------------------------------------------------------
# the superob policy
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SuperobPolicy:
    """The geometric gates between satellite pixels and one model cell.

    Unlike :class:`CwpErrorModel` these do have defaults, and the split is
    deliberate: these govern *whether* a cell becomes an observation, and
    the conservative setting is knowable without a calibration campaign --
    admit less.  The error constants govern *how much the analysis moves*,
    where there is no conservative direction to default to.

    min_pixels
        Contributing satellite pixels a cell needs.

    min_valid_fraction
        Of the pixels that land in a cell, the fraction that must have
        survived the DQF gate and produced a finite CWP.  A cell that is
        mostly condemned pixels is mostly not observed.

    phase_uniform_fraction
        The share of a cell's valid pixels that must agree on a phase
        class.  1.0 -- the default -- is the design note's rule verbatim:
        "a cell half clear, half deep ice is not one observation".  Below
        1.0 the cell is admitted with the *areal mean* of every valid
        pixel's CWP (the physically correct superob) and the *dominant*
        class's error model, and both counts are recorded.

    fallback_placement_agl_m
        Where a column's observation is centred when no cloud-top height
        reached it -- every clear-sky zero, and any cloudy cell the join
        could not serve.  See :func:`grid_cwp` on what placement means for
        a column integral: it is the centre of a vertical localisation
        lens, not a claim about where the water is.
    """

    min_pixels: int = 1
    min_valid_fraction: float = 0.5
    phase_uniform_fraction: float = 1.0
    fallback_placement_agl_m: float = 3000.0

    def validate(self) -> None:
        if int(self.min_pixels) < 1:
            raise GoesCwpError(
                f"min_pixels must be >= 1, got {self.min_pixels!r}")
        for name in ("min_valid_fraction", "phase_uniform_fraction"):
            value = float(getattr(self, name))
            if not (0.0 < value <= 1.0):
                raise GoesCwpError(
                    f"{name} must be in (0, 1], got {value!r}")
        height = float(self.fallback_placement_agl_m)
        if not np.isfinite(height) or height <= 0.0:
            raise GoesCwpError(
                f"fallback_placement_agl_m must be finite and positive, got "
                f"{height!r}")

    def to_payload(self) -> dict:
        self.validate()
        return {
            "min_pixels": int(self.min_pixels),
            "min_valid_fraction": float(self.min_valid_fraction),
            "phase_uniform_fraction": float(self.phase_uniform_fraction),
            "fallback_placement_agl_m": float(self.fallback_placement_agl_m),
            "reduction": "areal mean of every valid pixel's CWP in the cell",
            "class_rule": ("the cell takes the error model of its dominant "
                           "phase class; a cell below phase_uniform_fraction "
                           "is masked and counted"),
        }


# ---------------------------------------------------------------------------
# reading the two families
# ---------------------------------------------------------------------------


def read_cwp_pack(path) -> GoesPack:
    """One ``gpuwm-obs.goes-cwp.v1`` pack, family-checked."""

    pack = read_goes_pack(path, expected_schema=CWP_SCHEMAS)
    for name in ("cwp", "phase", "cod", "cps", "lat", "lon"):
        pack.plane(name)
    if "coefficients" not in pack.meta:
        raise GoesCwpError(
            f"{pack.path.name}: a CWP pack must carry the coefficient table "
            "its cwp plane was derived with; without it the derivation "
            "cannot be re-proved and the PROVISIONAL ice flag is lost")
    return pack


def read_cloudtop_pack(path) -> GoesPack:
    """One ``gpuwm-obs.goes-cloudtop.v1`` pack, family-checked."""

    pack = read_goes_pack(path, expected_schema=CLOUDTOP_SCHEMAS)
    for name in ("cloud_top_height_m", "lat", "lon"):
        pack.plane(name)
    return pack


def phase_class(phase_plane) -> np.ndarray:
    """Map decoded ABI phase codes to the three classes, ``(ny, nx)`` int8.

    ``NaN`` (fill or DQF-gated), a non-integral value, a value outside
    0..=5, and ``unknown`` (5) all become :data:`CLASS_NONE`.  This is
    ``rw_sat::cwp::CloudPhase::from_decoded`` followed by that function's
    own phase dispatch, restated so the Python side classifies exactly
    what the Rust side derived rather than a near-miss of it.
    """

    values = np.asarray(phase_plane, dtype=np.float64)
    out = np.full(values.shape, CLASS_NONE, dtype=np.int8)
    usable = (np.isfinite(values) & (values >= 0.0) & (values <= 5.0)
              & (np.floor(values) == values))
    codes = np.where(usable, values, -1.0).astype(np.int64)
    for code, klass in _PHASE_TO_CLASS.items():
        out[usable & (codes == code)] = klass
    return out


def rederive_cwp(pack: GoesPack) -> np.ndarray:
    """Re-derive the pack's CWP plane from its own inputs and coefficients.

    ``(2/3) * rho(phase) * tau * r_e`` in float32, in ``rw_sat::cwp``'s own
    operand order, with the densities the pack declares rather than any
    this module holds an opinion about.  Clear sky is ``0.0``; unknown
    phase, missing phase, and missing or negative inputs are ``NaN``.

    This exists so that :func:`grid_cwp` can refuse a pack whose payload
    does not reproduce from its own metadata -- a digest proves the bytes
    are the bytes the writer wrote, not that they are the numbers the
    writer said it was computing.
    """

    coefficients = pack.meta["coefficients"]
    rho_liquid = np.float32(coefficients["liquid_density_g_cm3"])
    rho_ice = np.float32(coefficients["ice_density_g_cm3"])
    cod = np.asarray(pack.plane("cod"), dtype=np.float32)
    cps = np.asarray(pack.plane("cps"), dtype=np.float32)
    classes = phase_class(pack.plane("phase"))

    two_thirds = np.float32(2.0) / np.float32(3.0)
    out = np.full(cod.shape, np.nan, dtype=np.float32)
    out[classes == CLASS_CLEAR] = np.float32(0.0)
    inputs_ok = (np.isfinite(cod) & np.isfinite(cps)
                 & (cod >= np.float32(0.0)) & (cps >= np.float32(0.0)))
    for klass, density in ((CLASS_LIQUID, rho_liquid), (CLASS_ICE, rho_ice)):
        where = (classes == klass) & inputs_ok
        if not np.any(where):
            continue
        out[where] = ((two_thirds * density) * cod[where]) * cps[where]
    return out


# ---------------------------------------------------------------------------
# the cross-grid join
# ---------------------------------------------------------------------------


def _nearest_index(target, source):
    """Index into ``source`` of the nearest entry to each ``target``."""

    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    order = np.argsort(src)
    ordered = src[order]
    if ordered.size == 1:
        return np.zeros(tgt.shape, dtype=np.intp), np.abs(tgt - ordered[0])
    position = np.clip(np.searchsorted(ordered, tgt), 1, ordered.size - 1)
    left = ordered[position - 1]
    right = ordered[position]
    take_left = np.abs(tgt - left) <= np.abs(right - tgt)
    picked = np.where(take_left, position - 1, position)
    offset = np.abs(tgt - ordered[picked])
    return order[picked], offset


def _bracket(target, source):
    """``(lower, upper, weight)`` for linear interpolation in ``source``."""

    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    order = np.argsort(src)
    ordered = src[order]
    if ordered.size == 1:
        zeros = np.zeros(tgt.shape, dtype=np.intp)
        return zeros, zeros, np.zeros(tgt.shape), np.abs(tgt - ordered[0])
    position = np.clip(np.searchsorted(ordered, tgt), 1, ordered.size - 1)
    left = ordered[position - 1]
    right = ordered[position]
    span = right - left
    weight = np.where(span > 0.0, (tgt - left) / np.where(span > 0.0, span,
                                                          1.0), 0.0)
    # Distance outside the source's own span, zero for anything bracketed.
    # A target inside the span is served exactly; only extrapolation is
    # a coverage question.
    outside = np.where((tgt < ordered[0]) | (tgt > ordered[-1]),
                       np.minimum(np.abs(tgt - ordered[0]),
                                  np.abs(tgt - ordered[-1])), 0.0)
    return (order[position - 1], order[position], np.clip(weight, 0.0, 1.0),
            outside)


def join_cloud_top(cwp_pack: GoesPack, cloudtop_pack: GoesPack, *,
                   method: str = "nearest",
                   coverage_slack: float = 0.5) -> tuple[np.ndarray, dict]:
    """Put the cloud-top pack's height onto the CWP pack's grid.

    The interpolation the bridge refused to do, done here where it is
    named.  Both packs carry the geostationary fixed-grid scan angles and
    the projection those angles are swept in, so the resample is exact in
    the coordinate the instrument actually samples -- no reprojection
    through lat/lon, and no assumption that the two grids' rows line up.

    ``method`` is ``"nearest"`` (the default) or ``"bilinear"``.  Nearest
    is the default because cloud-top height is discontinuous at a cloud
    edge: bilinear across that edge returns a height that lies between
    "cloud at 12 km" and "no cloud", which no pixel observed and which
    would place a CWP observation in clear air.

    ``coverage_slack`` is how far outside the cloud-top grid, in multiples
    of that grid's own spacing, a CWP pixel may sit and still be served.
    Beyond it the pixel gets ``NaN`` and is counted: the 10 km pack is a
    coarser sampling of the same sector, not a smaller one, so this should
    catch mismatched windows rather than edges.

    Returns ``(height_m_on_cwp_grid, receipt)``.  The receipt is the record
    the coordinator's ruling demands: it names the method, both grids, the
    pairing proof, and the counts.
    """

    if method not in JOIN_METHODS:
        raise GoesCwpError(
            f"join method must be one of {list(JOIN_METHODS)}, got "
            f"{method!r}")
    if cwp_pack.schema not in CWP_SCHEMAS:
        raise GoesCwpError(
            f"{cwp_pack.path.name} is a {cwp_pack.schema!r} pack, the join "
            f"target must be one of {list(CWP_SCHEMAS)}")
    if cloudtop_pack.schema not in CLOUDTOP_SCHEMAS:
        raise GoesCwpError(
            f"{cloudtop_pack.path.name} is a {cloudtop_pack.schema!r} pack, "
            f"the join source must be one of {list(CLOUDTOP_SCHEMAS)}")

    # -- the pairing, proved rather than assumed from a file name --------
    if cwp_pack.pairing_key != cloudtop_pack.pairing_key:
        raise GoesCwpError(
            f"these packs are not two halves of one scan: CWP is "
            f"{cwp_pack.pairing_key}, cloud-top is "
            f"{cloudtop_pack.pairing_key}. The pairing key is (satellite, "
            "sector, scan_start) and both packs state it")
    mismatched = [key for key in PROJECTION_KEYS
                  if cwp_pack.meta["projection"][key]
                  != cloudtop_pack.meta["projection"][key]]
    if mismatched:
        raise GoesCwpError(
            f"the packs disagree on the geostationary projection {mismatched}"
            "; a resample between two different perspectives is not an "
            "interpolation, it is a fabrication")
    sibling = cloudtop_pack.meta.get("sibling")
    sibling_proof = "absent: pairing rests on (satellite, sector, scan_start)"
    if sibling is not None:
        expected = str(cwp_pack.meta["content_sha256"])
        if str(sibling.get("content_sha256")) != expected:
            raise GoesCwpError(
                f"{cloudtop_pack.path.name} was written --pairs-with a CWP "
                f"pack whose payload hashes to "
                f"{sibling.get('content_sha256')}, but the CWP pack given "
                f"here hashes to {expected}. Refusing to join a proven pair "
                "to the wrong file")
        sibling_proof = f"content_sha256 pinned and matched: {expected}"

    source_y = np.asarray(cloudtop_pack.meta["y_scan_rad"], dtype=np.float64)
    source_x = np.asarray(cloudtop_pack.meta["x_scan_rad"], dtype=np.float64)
    target_y = np.asarray(cwp_pack.meta["y_scan_rad"], dtype=np.float64)
    target_x = np.asarray(cwp_pack.meta["x_scan_rad"], dtype=np.float64)
    heights = np.asarray(cloudtop_pack.plane("cloud_top_height_m"),
                         dtype=np.float64)

    step_y = (float(np.min(np.abs(np.diff(source_y)))) if source_y.size > 1
              else np.inf)
    step_x = (float(np.min(np.abs(np.diff(source_x)))) if source_x.size > 1
              else np.inf)
    slack_y = float(coverage_slack) * step_y
    slack_x = float(coverage_slack) * step_x

    if method == "nearest":
        jj, off_y = _nearest_index(target_y, source_y)
        ii, off_x = _nearest_index(target_x, source_x)
        joined = heights[np.ix_(jj, ii)]
        outside = ((off_y[:, None] > slack_y) | (off_x[None, :] > slack_x))
    else:
        j0, j1, wy, out_y = _bracket(target_y, source_y)
        i0, i1, wx, out_x = _bracket(target_x, source_x)
        corners = (heights[np.ix_(j0, i0)], heights[np.ix_(j0, i1)],
                   heights[np.ix_(j1, i0)], heights[np.ix_(j1, i1)])
        wy2 = wy[:, None]
        wx2 = wx[None, :]
        joined = ((1.0 - wy2) * ((1.0 - wx2) * corners[0] + wx2 * corners[1])
                  + wy2 * ((1.0 - wx2) * corners[2] + wx2 * corners[3]))
        # A bilinear cell that touches even one NaN corner has no defined
        # value: NaN is "no observation", and averaging it away would
        # manufacture a height out of the absence of one.
        any_nan = np.zeros(joined.shape, dtype=bool)
        for corner in corners:
            any_nan |= ~np.isfinite(corner)
        joined = np.where(any_nan, np.nan, joined)
        outside = ((out_y[:, None] > slack_y) | (out_x[None, :] > slack_x))

    joined = np.where(outside, np.nan, joined)
    served = int(np.count_nonzero(np.isfinite(joined)))
    receipt = {
        "performed": True,
        "method": method,
        "space": ("geostationary fixed-grid scan angle (radians); both "
                  "packs carry x_scan_rad/y_scan_rad and the same "
                  "projection, so no reprojection through lat/lon occurs"),
        "why_this_method": (
            "cloud-top height is discontinuous at a cloud edge; nearest "
            "never returns a height between 'cloud' and 'no cloud'"
            if method == "nearest" else
            "caller-selected bilinear; a cell touching any NaN corner is "
            "left NaN rather than blended"),
        "source_schema": cloudtop_pack.schema,
        "source_pack": cloudtop_pack.path.name,
        "source_pack_sha256": cloudtop_pack.pack_sha256,
        "source_grid": [int(cloudtop_pack.meta["ny"]),
                        int(cloudtop_pack.meta["nx"])],
        "target_schema": cwp_pack.schema,
        "target_pack": cwp_pack.path.name,
        "target_pack_sha256": cwp_pack.pack_sha256,
        "target_grid": [int(cwp_pack.meta["ny"]), int(cwp_pack.meta["nx"])],
        "pairing_key": list(cwp_pack.pairing_key),
        "sibling_digest_proof": sibling_proof,
        "coverage_slack_grid_cells": float(coverage_slack),
        "pixels_served": served,
        "pixels_outside_source_coverage": int(np.count_nonzero(outside)),
        "pixels_without_a_top": int(joined.size - served),
        "datum": CLOUD_TOP_DATUM,
        "bridge_ruling": (
            "gpuwm-obs.goes-cloudtop.v1 states regrid: none. The bridge "
            "never resamples; this receipt is the consumer's record of the "
            "resample it chose, per the 2026-08-06 separate-pack ruling"),
        "source_dqf_policy": cloudtop_pack.dqf_policy(),
    }
    return joined, receipt


def _ecef_from_geodetic(lat_deg, lon_deg, height_m, a: float, b: float):
    """Geodetic to Earth-centred, Earth-fixed metres on the (a, b) ellipsoid."""

    lat = np.radians(np.asarray(lat_deg, dtype=np.float64))
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64))
    h = np.asarray(height_m, dtype=np.float64)
    e2 = 1.0 - (b * b) / (a * a)
    n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
    x = (n + h) * np.cos(lat) * np.cos(lon)
    y = (n + h) * np.cos(lat) * np.sin(lon)
    z = (n * (1.0 - e2) + h) * np.sin(lat)
    return x, y, z


def _geodetic_from_ecef(x, y, z, a: float, b: float):
    """The inverse of :func:`_ecef_from_geodetic`, iterated to convergence.

    Six fixed-point iterations bring the latitude to well under a
    micro-degree for every point near the surface, which is a millimetre
    on the ground and far below the pixel size this is applied to.
    """

    e2 = 1.0 - (b * b) / (a * a)
    lon = np.arctan2(y, x)
    p = np.hypot(x, y)
    lat = np.arctan2(z, p * (1.0 - e2))
    h = np.zeros_like(p)
    for _ in range(6):
        n = a / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        h = p / np.cos(lat) - n
        lat = np.arctan2(z, p * (1.0 - e2 * n / (n + h)))
    return np.degrees(lat), np.degrees(lon), h


def parallax_displacement(lat_deg, lon_deg, height_m, projection: dict):
    """Where a cloud top seen at a nominal ground point really is.

    A geostationary imager assigns every pixel the ground point its line
    of sight reaches, the *nominal* location.  A cloud top at height
    ``height_m`` on that line sits nearer the satellite than the ground
    point does, so it is displaced from the nominal location toward the
    sub-satellite point by close to ``height_m * tan(view zenith)``: about
    a kilometre per kilometre of cloud top over the mid-latitudes.  This
    returns the ground point directly beneath the cloud top, which is
    where a column observation of that cloud belongs on a model grid.

    The satellite sits ``perspective_point_height_m`` above the equator
    at ``longitude_of_projection_origin_deg``; the surface at height
    ``height_m`` is the reference ellipsoid grown by that height on both
    axes, which is exact for a sphere and within metres of the geodetic
    height surface for the Earth.  ``height_m`` is the ACHA cloud-top
    height, taken as height above the ellipsoid (:data:`CLOUD_TOP_DATUM`:
    the geoid offset of tens of metres moves the answer by centimetres).

    Every argument broadcasts.  A pixel whose height is not finite or not
    positive is not moved and its displacement is 0: a clear pixel is
    where the ground is, and a cloudy pixel without a retrieved height has
    no support for any other place.

    Returns ``(lat_deg, lon_deg, displacement_m, view_zenith_deg)``: the
    displaced ground point, how far it moved along the ground (the chord
    between the two ground points, in metres), and the view zenith angle
    at the nominal point.
    """

    a = float(projection["semi_major_axis_m"])
    b = float(projection["semi_minor_axis_m"])
    satellite_radius = float(projection["perspective_point_height_m"]) + a
    sub_lon = float(projection["longitude_of_projection_origin_deg"])
    lat0 = np.asarray(lat_deg, dtype=np.float64)
    lon0 = np.asarray(lon_deg, dtype=np.float64)
    top = np.asarray(height_m, dtype=np.float64)
    lat0, lon0, top = np.broadcast_arrays(lat0, lon0, top)
    movable = np.isfinite(lat0) & np.isfinite(lon0) & np.isfinite(top) \
        & (top > 0.0)

    sx = satellite_radius * math.cos(math.radians(sub_lon))
    sy = satellite_radius * math.sin(math.radians(sub_lon))
    sz = 0.0
    px, py, pz = _ecef_from_geodetic(np.where(movable, lat0, 0.0),
                                     np.where(movable, lon0, 0.0), 0.0, a, b)
    # The line of sight, satellite to nominal ground point.
    dx, dy, dz = px - sx, py - sy, pz - sz
    # View zenith at the nominal point: the angle between the geodetic
    # up there and the direction to the satellite.
    lat_r = np.radians(np.where(movable, lat0, 0.0))
    lon_r = np.radians(np.where(movable, lon0, 0.0))
    ux, uy, uz = (np.cos(lat_r) * np.cos(lon_r), np.cos(lat_r) * np.sin(lon_r),
                  np.sin(lat_r))
    to_sat = np.sqrt(dx * dx + dy * dy + dz * dz)
    cos_zenith = -(dx * ux + dy * uy + dz * uz) / np.where(to_sat > 0, to_sat, 1)
    zenith_deg = np.degrees(np.arccos(np.clip(cos_zenith, -1.0, 1.0)))

    # First intersection of the line of sight with the ellipsoid grown by
    # the cloud-top height: solve |X|_(a+h, b+h) = 1 for X = S + t * d.
    ah = a + np.where(movable, top, 0.0)
    bh = b + np.where(movable, top, 0.0)
    qa = (dx * dx + dy * dy) / (ah * ah) + (dz * dz) / (bh * bh)
    qb = 2.0 * (sx * dx + sy * dy) / (ah * ah) + 2.0 * sz * dz / (bh * bh)
    qc = (sx * sx + sy * sy) / (ah * ah) + (sz * sz) / (bh * bh) - 1.0
    discriminant = np.maximum(qb * qb - 4.0 * qa * qc, 0.0)
    t = (-qb - np.sqrt(discriminant)) / (2.0 * np.where(qa > 0, qa, 1))
    cx, cy, cz = sx + t * dx, sy + t * dy, sz + t * dz
    lat_c, lon_c, _ = _geodetic_from_ecef(cx, cy, cz, a, b)
    gx, gy, gz = _ecef_from_geodetic(lat_c, lon_c, 0.0, a, b)
    moved = np.sqrt((gx - px) ** 2 + (gy - py) ** 2 + (gz - pz) ** 2)

    lat_out = np.where(movable, lat_c, lat0)
    lon_out = np.where(movable, lon_c, lon0)
    displacement = np.where(movable, moved, 0.0)
    zenith = np.where(movable, zenith_deg, np.nan)
    return lat_out, lon_out, displacement, zenith


def no_join_receipt(reason: str) -> dict:
    """The receipt written when no cloud-top pack was supplied.

    Silence is not an acceptable record of an interpolation that did not
    happen: without this, a product with every observation at the fallback
    placement height looks the same as one placed at real cloud tops.
    """

    return {
        "performed": False,
        "method": None,
        "reason": reason,
        "consequence": (
            "every observation is centred at "
            "SuperobPolicy.fallback_placement_agl_m above its own terrain, "
            "because no retrieved cloud-top height reached this product"),
    }


# ---------------------------------------------------------------------------
# the gridded product
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GriddedCwp:
    """CWP superobbed onto a model grid, with everything that qualified it.

    All 2-D fields are ``(ny, nx)`` on the target grid's mass points.

    cwp_obs
        g m-2.  Meaningless where ``cwp_mask`` is 0.
    cwp_mask
        int8, 1 where an observation exists.  No "maybe".
    cwp_err
        g m-2, observation error STANDARD DEVIATION.  Positive under the
        mask.
    cwp_class
        int8, :data:`CLASS_CLEAR` / :data:`CLASS_LIQUID` / :data:`CLASS_ICE`
        for an observation, :data:`CLASS_NONE` elsewhere.  This is the
        class the operator composes model condensate under, and the class
        whose error model was applied.
    cwp_count
        int32, valid satellite pixels that made the cell.
    cwp_pixels
        int32, satellite pixels that landed in the cell at all, valid or
        condemned.  ``cwp_count / cwp_pixels`` is the fraction the
        ``min_valid_fraction`` gate tested.
    cloud_top_height_m
        The joined retrieved top, cell-averaged over the cloudy pixels
        that had one.  NaN where none did.
    obs_level
        int32, the model level each observation is centred at, or -1.  See
        :func:`grid_cwp` for what this does and does not claim.
    counts
        Every pixel and every cell that did not become an observation,
        by reason.  Never silence.
    """

    cwp_obs: np.ndarray
    cwp_mask: np.ndarray
    cwp_err: np.ndarray
    cwp_class: np.ndarray
    cwp_count: np.ndarray
    cwp_pixels: np.ndarray
    cloud_top_height_m: np.ndarray
    obs_level: np.ndarray
    counts: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)


def pixel_inflation(cwp_pack: GoesPack,
                    error_model: CwpErrorModel) -> tuple[np.ndarray, dict]:
    """Per-pixel sigma_o inflation from the DCOMP thin/thick DQF bits.

    The operator spec's one row that a v1 pack could not serve.  Bits 256
    and 512 sit OUTSIDE the bridge's condemn mask by design, so those
    pixels are observations; the spec says they are worth less rather than
    worthless.  ``gpuwm-obs.goes-cwp.v2`` carries the per-pixel DQF word
    that makes "which pixels" answerable.

    Returns ``(factor, receipt)`` with ``factor`` all-ones when nothing is
    asked for.  NaN DQF pixels -- the ones the pack counts as
    ``dqf_missing`` -- take a factor of 1.0 and are counted separately:
    their flag was unreadable, they are already gated out of ``cwp``, and
    ``astype(uint16)`` on a NaN returns garbage rather than raising, so
    they are never read as a bitfield.
    """

    ny, nx = cwp_pack.shape
    ones = np.ones((ny, nx), dtype=np.float64)
    if not error_model.wants_inflation():
        return ones, {"applied": False,
                      "reason": "the error model asks for no inflation "
                                "(thin_inflation and thick_inflation are "
                                "both 1.0)"}
    if not cwp_pack.has_dqf_planes:
        raise GoesCwpError(
            f"{cwp_pack.path.name} is a {cwp_pack.schema} pack, which "
            "carries DQF counts and the condemn mask but no per-pixel DQF "
            "plane, and the error model asks for thin/thick inflation "
            f"(thin={error_model.thin_inflation}, "
            f"thick={error_model.thick_inflation}). Which pixels carry "
            "bits 256/512 is not recoverable from this pack, and guessing "
            "it from cod would be inventing the gate rather than honouring "
            "it. Use a v2 pack, or set both factors to 1.0 and accept that "
            "those retrievals are assimilated at un-inflated error")

    thin = np.zeros((ny, nx), dtype=bool)
    thick = np.zeros((ny, nx), dtype=bool)
    unreadable = np.zeros((ny, nx), dtype=bool)
    products = []
    for product in DCOMP_PRODUCTS:
        try:
            word, valid = cwp_pack.dqf_plane(product)
        except Exception as error:              # noqa: BLE001
            raise GoesCwpError(
                f"{cwp_pack.path.name}: the error model needs {product}'s "
                f"per-pixel DQF and it could not be read: {error}"
            ) from error
        products.append(product)
        thin |= valid & ((word & DCOMP_THIN_BIT) != 0)
        thick |= valid & ((word & DCOMP_THICK_BIT) != 0)
        unreadable |= ~valid

    factor = ones.copy()
    factor[thin] *= float(error_model.thin_inflation)
    factor[thick] *= float(error_model.thick_inflation)
    receipt = {
        "applied": True,
        "products_unioned": products,
        "pixels_thin": int(thin.sum()),
        "pixels_thick": int(thick.sum()),
        "pixels_both": int((thin & thick).sum()),
        "pixels_dqf_unreadable": int(unreadable.sum()),
        "pixels_inflated": int((factor > 1.0).sum()),
        "max_factor": float(factor.max()),
    }
    return factor, receipt


def grid_cwp(cwp_pack: GoesPack, grid, *, error_model: CwpErrorModel,
             policy: SuperobPolicy | None = None,
             cloud_top_m=None, join_receipt: dict | None = None,
             max_derivation_mismatch_fraction: float = 0.0) -> GriddedCwp:
    """Superob one CWP pack onto a :class:`TargetGrid`.

    **On horizontal placement.**  The pack's ``lat``/``lon`` planes are
    the ground points the satellite's lines of sight reach, not the ground
    beneath the clouds it saw.  A cloudy pixel whose joined cloud top is
    known is moved to the ground beneath that top
    (:func:`parallax_displacement`), a clear pixel stays where the ground
    is, and a cloudy pixel with no retrieved top stays at the ground and is
    counted in the receipt rather than moved by a height nobody measured.

    **On vertical placement.**  CWP is a column integral: it has no
    height, and the ``obs_level`` this returns is not a claim that it
    does.  The LETKF
    localises in metres about an observation's gridpoint, so a column
    observation has to be *centred* somewhere, and the centre chosen here
    is the retrieved cloud top where the join supplied one and
    ``fallback_placement_agl_m`` above the terrain where it did not.  The
    reach around that centre is the caller's vertical localisation radius,
    and for a column integral it must be set to span the column -- a
    4 km radius on a column observation is a 4 km-tall observation, which
    is not what was observed.  :mod:`gpuwm.da.obs_goes` states this again
    at the point where the radius is applied.

    The mask carries exactly one observation per column.  A CWP value
    repeated down a column would be the same measurement counted ``nz``
    times, and the filter has no way to know it.
    """

    policy = SuperobPolicy() if policy is None else policy
    policy.validate()
    error_model.validate()

    ny_grid, nx_grid = int(grid.ny), int(grid.nx)
    cells = ny_grid * nx_grid

    # -- gate 1: the pack reproduces from its own metadata ---------------
    stated = np.asarray(cwp_pack.plane("cwp"), dtype=np.float32)
    rederived = rederive_cwp(cwp_pack)
    both_finite = np.isfinite(stated) & np.isfinite(rederived)
    finite_disagrees = np.zeros(stated.shape, dtype=bool)
    finite_disagrees[both_finite] = ~np.isclose(
        stated[both_finite], rederived[both_finite], rtol=1.0e-5, atol=1.0e-3)
    presence_disagrees = np.isfinite(stated) != np.isfinite(rederived)
    mismatches = int(np.count_nonzero(finite_disagrees | presence_disagrees))
    fraction = mismatches / float(stated.size) if stated.size else 0.0
    if fraction > float(max_derivation_mismatch_fraction):
        raise GoesCwpError(
            f"{cwp_pack.path.name}: {mismatches} of {stated.size} pixels "
            f"({fraction:.3%}) do not reproduce from the pack's own cod, "
            "cps, phase and coefficient table. The payload digest proves "
            "the bytes are the writer's bytes; it does not prove they are "
            "the numbers the writer said it was computing, which is what "
            "this checks. Raise max_derivation_mismatch_fraction only with "
            "a reason recorded")

    # -- pixels to cells --------------------------------------------------
    inflation_plane, inflation_receipt = pixel_inflation(cwp_pack,
                                                        error_model)
    inflation_flat = inflation_plane.reshape(-1)

    lat = np.asarray(cwp_pack.plane("lat"), dtype=np.float64).reshape(-1)
    lon = np.asarray(cwp_pack.plane("lon"), dtype=np.float64).reshape(-1)
    values = stated.reshape(-1).astype(np.float64)
    classes = phase_class(cwp_pack.plane("phase")).reshape(-1)

    geolocated = np.isfinite(lat) & np.isfinite(lon)
    cloudy = (classes != CLASS_NONE) & (classes != CLASS_CLEAR)

    # -- the joined cloud top -------------------------------------------
    if cloud_top_m is None:
        tops = np.full(values.size, np.nan)
        if join_receipt is None:
            join_receipt = no_join_receipt("no cloud-top pack was supplied")
    else:
        tops = np.asarray(cloud_top_m, dtype=np.float64).reshape(-1)
        if tops.size != values.size:
            raise GoesCwpError(
                f"the joined cloud-top field holds {tops.size} values but "
                f"the CWP pack's grid holds {values.size}; the join did not "
                "land on this pack's grid")
        if join_receipt is None:
            raise GoesCwpError(
                "a joined cloud-top field was supplied without the receipt "
                "join_cloud_top produced. The interpolation is the "
                "consumer's recorded choice; an unrecorded one is the thing "
                "the bridge's separate-pack ruling exists to prevent")

    # -- where each pixel's cloud really is ------------------------------
    # The pack's lat/lon are the ground points the satellite's lines of
    # sight reach.  A cloud top on such a line sits nearer the satellite
    # than the ground does, so it is over a point displaced toward the
    # sub-satellite point by close to height * tan(view zenith): six
    # kilometres and more for a mid-latitude cloud, which is a cell or two
    # of a storm-scale grid.  A cloudy pixel with a retrieved top is moved
    # to the ground beneath its top; a clear pixel is where the ground is;
    # a cloudy pixel with no height is left at the ground and counted,
    # because there is no support for any other place.
    has_height = np.isfinite(tops) & (tops > 0.0)
    displaced = geolocated & cloudy & has_height
    placed_lat, placed_lon, moved_m, zenith_deg = parallax_displacement(
        np.where(geolocated, lat, 0.0), np.where(geolocated, lon, 0.0),
        np.where(displaced, tops, np.nan), cwp_pack.projection())

    i_frac, j_frac = grid.mass_index(placed_lat, placed_lon)
    i_index = np.rint(i_frac).astype(np.int64)
    j_index = np.rint(j_frac).astype(np.int64)
    on_grid = geolocated & grid.inside(i_index, j_index)
    flat = np.where(on_grid, j_index * nx_grid + i_index, 0)

    valid = on_grid & (classes != CLASS_NONE) & np.isfinite(values)

    landed = np.bincount(flat[on_grid], minlength=cells).astype(np.int64)
    counted = np.bincount(flat[valid], minlength=cells).astype(np.int64)
    total = np.bincount(flat[valid], weights=values[valid],
                        minlength=cells)
    per_class = {
        klass: np.bincount(flat[valid & (classes == klass)],
                           minlength=cells).astype(np.int64)
        for klass in (CLASS_CLEAR, CLASS_LIQUID, CLASS_ICE)
    }
    inflation_sum = np.bincount(flat[valid], weights=inflation_flat[valid],
                                minlength=cells)
    stacked = np.stack([per_class[CLASS_CLEAR], per_class[CLASS_LIQUID],
                        per_class[CLASS_ICE]])
    dominant = np.argmax(stacked, axis=0).astype(np.int8)
    dominant_count = stacked.max(axis=0)

    # -- the cloudy cells' retrieved top, credited where the cloud is ----
    has_top = valid & (classes != CLASS_CLEAR) & np.isfinite(tops)
    top_sum = np.bincount(flat[has_top], weights=tops[has_top],
                          minlength=cells)
    top_count = np.bincount(flat[has_top], minlength=cells).astype(np.int64)

    displaced_on_grid = displaced & on_grid
    if np.any(displaced):
        moved = moved_m[displaced]
        displacement_stats = {
            "median": float(np.median(moved)),
            "p95": float(np.percentile(moved, 95.0)),
            "max": float(moved.max()),
            "mean": float(moved.mean()),
            "view_zenith_deg_median": float(np.median(zenith_deg[displaced])),
        }
    else:
        displacement_stats = None

    # -- the gates ---------------------------------------------------------
    with np.errstate(invalid="ignore", divide="ignore"):
        valid_fraction = np.where(landed > 0, counted / np.maximum(landed, 1),
                                  0.0)
        uniformity = np.where(counted > 0,
                              dominant_count / np.maximum(counted, 1), 0.0)
    enough = counted >= int(policy.min_pixels)
    filled = valid_fraction >= float(policy.min_valid_fraction)
    uniform = uniformity >= float(policy.phase_uniform_fraction)
    admitted = enough & filled & uniform

    cwp_obs = np.zeros(cells)
    cwp_obs[admitted] = total[admitted] / counted[admitted]
    cwp_class = np.where(admitted, dominant, CLASS_NONE).astype(np.int8)

    cwp_err = np.zeros(cells)
    is_clear = admitted & (cwp_class == CLASS_CLEAR)
    is_liquid = admitted & (cwp_class == CLASS_LIQUID)
    is_ice = admitted & (cwp_class == CLASS_ICE)
    cwp_err[is_clear] = float(error_model.clear_g_m2)
    cwp_err[is_liquid] = np.maximum(
        float(error_model.rel_liquid) * cwp_obs[is_liquid],
        float(error_model.floor_liquid_g_m2))
    cwp_err[is_ice] = np.maximum(
        float(error_model.rel_ice) * cwp_obs[is_ice],
        float(error_model.floor_ice_g_m2))

    # The thin/thick inflation, applied to the cell's sigma_o as the MEAN
    # factor over the pixels averaged into it -- the cell's value is their
    # areal mean, so its error follows the same reduction.  Applied AFTER
    # the class floors, so a floor is a floor on the un-inflated error and
    # the flag can still raise sigma above it.
    cell_inflation = np.ones(cells)
    np.divide(inflation_sum, counted, out=cell_inflation,
              where=admitted & (counted > 0))
    cwp_err[admitted] *= cell_inflation[admitted]

    top_mean = np.full(cells, np.nan)
    served = admitted & (top_count > 0)
    top_mean[served] = top_sum[served] / top_count[served]

    # -- placement ---------------------------------------------------------
    shape2 = (ny_grid, nx_grid)
    jj, ii = np.divmod(np.arange(cells), nx_grid)
    terrain = np.asarray(grid.terrain_m, dtype=np.float64).reshape(-1)
    centre = np.where(np.isfinite(top_mean), top_mean,
                      terrain + float(policy.fallback_placement_agl_m))
    level = grid.level_index(ii, jj, centre)
    placed = admitted & (level >= 0)

    counts = {
        "pack_pixels": int(values.size),
        "pixels_not_geolocated": int(np.count_nonzero(~geolocated)),
        "pixels_off_grid": int(np.count_nonzero(geolocated & ~on_grid)),
        "pixels_on_grid": int(np.count_nonzero(on_grid)),
        "pixels_no_phase": int(np.count_nonzero(
            on_grid & (classes == CLASS_NONE))),
        "pixels_no_cwp": int(np.count_nonzero(
            on_grid & (classes != CLASS_NONE) & ~np.isfinite(values))),
        "pixels_valid": int(np.count_nonzero(valid)),
        "pixels_clear": int(np.count_nonzero(valid & (classes == CLASS_CLEAR))),
        "pixels_liquid": int(np.count_nonzero(
            valid & (classes == CLASS_LIQUID))),
        "pixels_ice": int(np.count_nonzero(valid & (classes == CLASS_ICE))),
        # Horizontal placement: cloudy pixels moved to the ground beneath
        # their retrieved top, and cloudy pixels that had no top to move by.
        "pixels_displaced_to_cloud_top": int(np.count_nonzero(displaced)),
        "pixels_displaced_on_grid": int(np.count_nonzero(displaced_on_grid)),
        "pixels_cloudy_without_height_at_ground": int(np.count_nonzero(
            geolocated & cloudy & ~has_height)),
        "cells_touched": int(np.count_nonzero(landed > 0)),
        "cells_below_min_pixels": int(np.count_nonzero(
            (landed > 0) & ~enough)),
        "cells_below_min_valid_fraction": int(np.count_nonzero(
            (landed > 0) & enough & ~filled)),
        "cells_phase_mixed": int(np.count_nonzero(
            (landed > 0) & enough & filled & ~uniform)),
        "cells_admitted": int(np.count_nonzero(admitted)),
        "cells_unplaceable": int(np.count_nonzero(admitted & (level < 0))),
        "observations": int(np.count_nonzero(placed)),
        "observations_clear": int(np.count_nonzero(placed
                                                   & (cwp_class == CLASS_CLEAR))),
        "observations_liquid": int(np.count_nonzero(
            placed & (cwp_class == CLASS_LIQUID))),
        "observations_ice": int(np.count_nonzero(
            placed & (cwp_class == CLASS_ICE))),
        "observations_at_retrieved_top": int(np.count_nonzero(
            placed & np.isfinite(top_mean))),
        "observations_at_fallback_height": int(np.count_nonzero(
            placed & ~np.isfinite(top_mean))),
        "derivation_mismatched_pixels": mismatches,
        "cells_error_inflated": int(np.count_nonzero(
            placed & (cell_inflation > 1.0))),
        "mean_cell_inflation_where_applied": float(
            cell_inflation[placed & (cell_inflation > 1.0)].mean())
        if np.any(placed & (cell_inflation > 1.0)) else 1.0,
    }

    provenance = {
        "pack": cwp_pack.provenance(),
        "cwp_counts_from_pack": dict(cwp_pack.meta["cwp_counts"]),
        "coefficients_from_pack": dict(cwp_pack.meta["coefficients"]),
        "derivation_cross_check": {
            "performed": True,
            "form": "(2/3) * rho(phase) * cod * cps, float32, pack's own rho",
            "mismatched_pixels": mismatches,
            "mismatch_fraction": float(fraction),
            "tolerance_fraction": float(max_derivation_mismatch_fraction),
        },
        "dqf_policy": cwp_pack.dqf_policy(),
        "dqf_honoured": (
            "the bridge's condemn mask ran upstream; condemned pixels reach "
            "this stage as NaN and become no observation. The rule and mask "
            "are recorded above rather than re-applied, because the pack "
            "carries no per-pixel DQF plane"),
        "join": join_receipt,
        "superob": policy.to_payload(),
        "error_model": error_model.to_payload(
            applied=inflation_receipt["applied"],
            reason=inflation_receipt.get("reason"),
            counts={key: value for key, value
                    in inflation_receipt.items()
                    if key not in ("applied", "reason")} or None),
        "pack_schema_version": cwp_pack.schema_version,
        "per_pixel_dqf_available": cwp_pack.has_dqf_planes,
        "placement": {
            "rule": ("retrieved cloud top where the join served one, "
                     "otherwise fallback_placement_agl_m above the cell's "
                     "own terrain"),
            "meaning": (
                "CWP is a column integral and has no height. obs_level is "
                "the centre of the vertical localisation lens, not a claim "
                "about where the condensate is. The lens radius must span "
                "the column or the observation is silently truncated"),
            "one_observation_per_column": True,
            "horizontal": {
                "rule": (
                    "a cloudy pixel with a retrieved top is placed at the "
                    "ground beneath that top, displaced from the pack's "
                    "nominal ground point toward the sub-satellite point "
                    "along the satellite's line of sight; a clear pixel "
                    "stays at the ground; a cloudy pixel without a "
                    "retrieved top stays at the ground and is counted"),
                "geometry": {
                    "sub_satellite_longitude_deg": float(
                        cwp_pack.projection()[
                            "longitude_of_projection_origin_deg"]),
                    "perspective_point_height_m": float(
                        cwp_pack.projection()["perspective_point_height_m"]),
                    "height_datum": CLOUD_TOP_DATUM,
                },
                "pixels_displaced": int(np.count_nonzero(displaced)),
                "pixels_cloudy_without_height_at_ground": int(
                    np.count_nonzero(geolocated & cloudy & ~has_height)),
                "displacement_m": displacement_stats,
            },
        },
        "counts": counts,
    }

    return GriddedCwp(
        cwp_obs=np.where(placed, cwp_obs, 0.0).reshape(shape2),
        cwp_mask=placed.astype(np.int8).reshape(shape2),
        cwp_err=np.where(placed, cwp_err, 0.0).reshape(shape2),
        cwp_class=np.where(placed, cwp_class, CLASS_NONE).astype(
            np.int8).reshape(shape2),
        cwp_count=np.where(placed, counted, 0).astype(np.int32).reshape(shape2),
        cwp_pixels=landed.astype(np.int32).reshape(shape2),
        cloud_top_height_m=np.where(placed, top_mean,
                                    np.nan).reshape(shape2),
        obs_level=np.where(placed, level, -1).astype(np.int32).reshape(shape2),
        counts=counts,
        provenance=provenance)
