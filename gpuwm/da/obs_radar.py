"""EXPERIMENTAL: ``gpuwm-obs.radar-grid.v1`` -> LETKF observation batches.

The radar lane writes the file; :mod:`gpuwm.da.letkf` consumes
:class:`~gpuwm.da.letkf.GriddedObs`.  Neither lane owns the translation, and
the translation is where the two schemas actually differ, so it lives here.

Three properties of the file are load-bearing and are the reason this is a
module rather than four lines at a call site.

**Radial velocity carries a leading radar axis, and it must survive.**
``vr_obs`` is ``(radar, level, south_north, west_east)``.  Two radars looking
at one cell measure two different projections of one wind; their mean is a
number with no observation operator behind it, and dropping one throws away
half the information.  Each radar therefore becomes its **own**
:class:`GriddedObs`, named ``vr:<SITE>``.  The filter batches observation
types independently and localises each on its own, so N radars is N entries in
the ``obs`` sequence and nothing else changes.

**The beam unit vectors are the observation operator direction.**
``vr_beam_east/north/up`` ship with every retained velocity precisely so that
the assimilating side does not re-derive beam geometry from the radar position
and the cell centre -- duplicated work, and a place for the two stages to
disagree about where a cell is.  :func:`simulated_radial_velocity` projects a
member's earth-relative wind onto **the file's own vectors**.  It deliberately
does not call :func:`gpuwm.da.obsop.beam_geometry`: that function evaluates
the beam exactly at the mass point, while the file's vectors are the
normalised sum over the gates that actually contributed to the superob cell.
Those agree only to within a superob cell, and the observation's own geometry
is the one an innovation must be taken against.

**The grid is bound by its arrays, not by its name.**  Every entry point
here takes the caller's own :class:`~gpuwm.obs.target_grid.TargetGrid` and
recomputes the digest table the writer bound to the file, ``z_w`` included.
``grid_identity_sha256`` on its own cannot do that job: it digests ``lat``,
``lon``, ``z_w`` and ``terrain_m`` at float64, while the file stores three
of the four at float32 and no vertical coordinate at all.  A file whose
identity attribute has been relabelled to name a different grid passes
every check a reader can make by itself, and assimilating it puts every
observation in the wrong column.  So the string is accepted only as an
extra demand to cross-check, never as the binding.

**Observation errors are standard deviations.**  ``z_err`` and ``vr_err`` are
``sqrt(base^2/n + in-cell variance)`` with a floor -- already a sigma.  The
filter's contract is also a sigma.  Nothing is squared here, and a
non-positive error under a True mask is a refusal rather than a quietly
dropped point.

Nothing in this module is wired into a default route.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

from gpuwm.da.letkf import GriddedObs, Localization

#: The observation contract this adapter reads.
OBS_SCHEMA = "gpuwm-obs.radar-grid.v1"

#: Provenance stamp for the adaptation itself.
ADAPTER_SCHEMA = "gpuwm-da.radar-obs-adapter.v1"

#: Accepted reductions for the reflectivity observation.  ``z_obs`` is the
#: file's own chosen reduction (``superob_params.z_reduce``, ``max`` by
#: default); ``z_max`` is the in-cell maximum regardless of that choice.
#: ``z_mean`` is deliberately absent: it is a linear-Z mean expressed in
#: dBZ, so assimilating it against a dBZ forward operator is a units trap
#: of exactly the kind this lane exists to avoid making silently.
Z_SOURCES = ("z_obs", "z_max")

#: Name given to the reflectivity batch.  Reflectivity is merged across
#: radars by the writer (max of maxima, count-weighted linear mean), so
#: unlike velocity there is exactly one of it.
REFLECTIVITY_NAME = "z"

#: Prefix for the per-radar velocity batches.
VELOCITY_PREFIX = "vr"


class RadarObsAdapterError(ValueError):
    """The file and the filter cannot be reconciled.  Never a warning."""


def read_document(source, *, expected_grid,
                  expected_grid_identity: str | None = None):
    """Accept a path or an already-read document, uniformly and bound.

    ``expected_grid`` is a :class:`~gpuwm.obs.target_grid.TargetGrid` and it
    is **required**, because the identity string alone cannot close the
    door this function stands in.

    ``grid_identity_sha256`` is a digest of the projection descriptor and
    of ``lat``, ``lon``, ``z_w`` and ``terrain_m`` at float64.  The file
    stores three of those four arrays, at float32, and no vertical
    coordinate at all -- so a reader can prove the stored arrays match the
    file's own digest table, and can never recompute the identity from
    them.  A file whose ``grid_identity_sha256`` attribute has been
    relabelled to name a different grid is therefore internally consistent
    to every check a reader can make alone, and assimilating it places
    every observation in the wrong column.  Only a caller holding the grid
    can recompute the digest table, and ``z_w`` is reachable no other way.

    So this takes the grid, not the string.  ``expected_grid_identity`` is
    still accepted, and a caller passing both a grid and an identity that
    disagree is refused rather than served either one.
    """

    if expected_grid is None:
        raise RadarObsAdapterError(
            "expected_grid is required: the DA read path binds a file to "
            "the caller's own TargetGrid, not to an identity string. The "
            "string is a claim the file makes about arrays it only partly "
            "contains -- relabel it and no reader-side check can tell, "
            "because z_w is not in the file at all")

    from gpuwm.obs.radar_grid import (require_grid_binding,  # noqa: PLC0415
                                      read_radar_grid)

    if isinstance(source, (str, Path)):
        return read_radar_grid(
            source, expected_grid_identity=expected_grid_identity,
            expected_grid=expected_grid)
    if not isinstance(source, Mapping):
        raise RadarObsAdapterError(
            f"expected a path or a read_radar_grid document, got "
            f"{type(source).__name__}")
    if source.get("schema") != OBS_SCHEMA:
        raise RadarObsAdapterError(
            f"document declares schema {source.get('schema')!r}, this "
            f"adapter reads {OBS_SCHEMA!r}")
    demanded = expected_grid.identity_sha256()
    if (expected_grid_identity is not None
            and expected_grid_identity != demanded):
        raise RadarObsAdapterError(
            f"expected_grid hashes to {demanded} but expected_grid_identity "
            f"demands {expected_grid_identity}; the caller is asking for two "
            "different grids")
    require_grid_binding(source, expected_grid, label="document")
    return source


def observation_shape(document) -> tuple[int, int, int]:
    """``(nz, ny, nx)`` -- the model grid the file was written onto."""

    dims = document["dims"]
    return (int(dims["level"]), int(dims["south_north"]),
            int(dims["west_east"]))


def beam_unit_vectors(document, radar_index: int):
    """``(east, north, up)``, each ``(nz, ny, nx)``, for one radar.

    These are the observation operator direction, straight out of the file.
    """

    variables = document["variables"]
    missing = [name for name in ("vr_beam_east", "vr_beam_north",
                                 "vr_beam_up") if name not in variables]
    if missing:
        raise RadarObsAdapterError(
            f"the file carries no {missing}; a radial velocity without its "
            "look direction is not an observation, and this adapter will "
            "not re-derive the beam geometry the writer already computed")
    return tuple(np.asarray(variables[name][radar_index], dtype=np.float64)
                 for name in ("vr_beam_east", "vr_beam_north", "vr_beam_up"))


def simulated_radial_velocity(u_east, v_north, w_up, beam):
    """Project one member's earth-relative wind onto the file's beam.

    ``Vr = u*east + v*north + w*up``, positive **away** from the radar --
    the radar lane's sign convention and ``gpuwm.da.obsop``'s, which agree.

    ``w_up`` is the vertical air motion **already net of any hydrometeor
    fall speed** (``w - vt``).  The fall speed is a property of the forward
    operator, not of the beam, so it is the caller's to subtract; passing
    plain ``w`` gives an air-motion-only H(x), which is a defensible choice
    and a different one.
    """

    east, north, up = beam
    u_east = np.asarray(u_east, dtype=np.float64)
    v_north = np.asarray(v_north, dtype=np.float64)
    w_up = np.asarray(w_up, dtype=np.float64)
    if not (u_east.shape == v_north.shape == w_up.shape == east.shape):
        raise RadarObsAdapterError(
            f"wind components {u_east.shape}/{v_north.shape}/{w_up.shape} "
            f"and beam vectors {east.shape} must share the mass-point shape")
    return u_east * east + v_north * north + w_up * up


def _batch(name, values, errors, mask, simulated, localization, shape):
    """One validated :class:`GriddedObs`.  Every failure is named."""

    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    mask = np.asarray(mask).astype(bool)
    simulated = np.asarray(simulated, dtype=np.float64)
    if values.shape != shape:
        raise RadarObsAdapterError(
            f"{name}: values have shape {values.shape}, the file's grid is "
            f"{shape}")
    if mask.shape != shape or errors.shape != shape:
        raise RadarObsAdapterError(
            f"{name}: mask {mask.shape} and errors {errors.shape} must both "
            f"be {shape}")
    if simulated.ndim != 4 or simulated.shape[1:] != shape:
        raise RadarObsAdapterError(
            f"{name}: simulated must be (R, {shape[0]}, {shape[1]}, "
            f"{shape[2]}), got {simulated.shape}")
    if not mask.any():
        # Legitimate -- a radar that saw nothing this cycle.  The filter
        # handles an all-False mask as a no-observation type.
        return GriddedObs(name=name, values=values, errors=errors,
                          simulated=simulated, mask=mask,
                          localization=localization)
    bad_value = int(np.count_nonzero(~np.isfinite(values[mask])))
    if bad_value:
        raise RadarObsAdapterError(
            f"{name}: {bad_value} unmasked observation(s) are not finite. "
            "The mask is the file's statement about what is usable; a "
            "non-finite value under a True mask means the two disagree, "
            "and guessing which to believe is how a NaN reaches a solve")
    bad_error = int(np.count_nonzero(
        ~np.isfinite(errors[mask]) | (errors[mask] <= 0.0)))
    if bad_error:
        raise RadarObsAdapterError(
            f"{name}: {bad_error} unmasked observation(s) carry a "
            "non-positive or non-finite error. These are standard "
            "deviations, and a zero sigma is an infinitely confident "
            "observation, not a missing one")
    bad_sim = int(np.count_nonzero(
        ~np.isfinite(simulated[:, mask])))
    if bad_sim:
        raise RadarObsAdapterError(
            f"{name}: H(x_k) is not finite at {bad_sim} unmasked "
            "member-gridpoint(s); the forward operator produced something "
            "the filter cannot use")
    return GriddedObs(name=name, values=values, errors=errors,
                      simulated=simulated, mask=mask,
                      localization=localization)


def radar_grid_to_gridded_obs(
    source,
    *,
    expected_grid,
    reflectivity_simulated=None,
    velocity_simulated: Callable[[int, Mapping], object] | None = None,
    z_source: str = "z_obs",
    expected_grid_identity: str | None = None,
    reflectivity_localization: Localization | None = None,
    velocity_localization: Localization | None = None,
    radars: Sequence[str] | None = None,
) -> tuple[list[GriddedObs], dict]:
    """Adapt one radar-grid file to the filter's observation batches.

    ``expected_grid`` is the caller's own
    :class:`~gpuwm.obs.target_grid.TargetGrid` and is required.  See
    :func:`read_document` for why an identity string is not enough: it is a
    claim about four arrays of which the file stores three, at a narrower
    dtype, and none of them vertical.

    ``reflectivity_simulated`` is ``H(x_k)`` for reflectivity, ``(R, nz, ny,
    nx)``, or None to leave reflectivity out.  ``velocity_simulated`` is a
    callable ``(radar_index, radar_record) -> (R, nz, ny, nx)``; it is a
    callable rather than an array because the operator differs **per radar**
    -- each one has its own beam vectors -- and handing the caller the index
    keeps :func:`beam_unit_vectors` on this side of the seam without this
    module needing to know what a member state looks like.

    Returns ``(batches, provenance)``.  The provenance is JSON-serialisable
    and belongs in the cycle manifest's assimilation slot.
    """

    if z_source not in Z_SOURCES:
        raise RadarObsAdapterError(
            f"z_source must be one of {Z_SOURCES}, got {z_source!r}; "
            "'z_mean' is excluded on purpose -- it is a linear-Z mean "
            "expressed in dBZ and does not difference against a dBZ H(x)")
    document = read_document(
        source, expected_grid=expected_grid,
        expected_grid_identity=expected_grid_identity)
    shape = observation_shape(document)
    variables = document["variables"]

    batches: list[GriddedObs] = []
    used: list[dict] = []

    if reflectivity_simulated is not None:
        if z_source not in variables:
            raise RadarObsAdapterError(
                f"the file carries no {z_source!r}; available reflectivity "
                f"reductions: "
                f"{sorted(n for n in variables if n.startswith('z_'))}")
        z_mask = np.asarray(variables["z_mask"]).astype(bool)
        batches.append(_batch(
            REFLECTIVITY_NAME, variables[z_source], variables["z_err"],
            z_mask, reflectivity_simulated, reflectivity_localization, shape))
        used.append({
            "name": REFLECTIVITY_NAME, "kind": "reflectivity",
            "source_variable": z_source, "units": "dBZ",
            "merged_across_radars": True,
            "observed_points": int(np.count_nonzero(z_mask)),
        })

    if velocity_simulated is not None:
        wanted = None if radars is None else {str(r) for r in radars}
        for index, radar in enumerate(document["radars"]):
            site = str(radar["id"])
            if wanted is not None and site not in wanted:
                continue
            vr_mask = np.asarray(variables["vr_mask"][index]).astype(bool)
            name = f"{VELOCITY_PREFIX}:{site}"
            batches.append(_batch(
                name, variables["vr_obs"][index], variables["vr_err"][index],
                vr_mask, velocity_simulated(index, radar),
                velocity_localization, shape))
            rejected = variables.get("vr_rejected")
            used.append({
                "name": name, "kind": "radial_velocity",
                "radar": site, "radar_index": index,
                "units": "m s-1", "sign_convention": "positive away",
                "operator_direction": "vr_beam_east/north/up from the file",
                "observed_points": int(np.count_nonzero(vr_mask)),
                "gates_rejected": (None if rejected is None
                                   else int(np.sum(rejected[index]))),
            })
        if wanted is not None:
            found = {entry.get("radar") for entry in used}
            absent = sorted(wanted - {site for site in found if site})
            if absent:
                raise RadarObsAdapterError(
                    f"requested radar(s) {absent} are not in the file, "
                    f"which carries "
                    f"{[str(r['id']) for r in document['radars']]}")

    if not batches:
        raise RadarObsAdapterError(
            "neither reflectivity_simulated nor velocity_simulated was "
            "given, so this call would assimilate nothing. An intentional "
            "no-observation cycle is an empty obs sequence at the analyze() "
            "call, not an empty adaptation here")

    provenance = {
        "schema": ADAPTER_SCHEMA,
        "stability": "experimental",
        "obs_schema": document.get("schema"),
        "obs_status": document.get("status"),
        "valid_time": document.get("valid_time"),
        "grid_identity_sha256": document.get("grid_identity_sha256"),
        # The per-array digests the identity was made from, recorded because
        # the identity string alone is the thing this adapter refuses to
        # trust.  `z_w` is here and nowhere in the observation file.
        "grid_coordinate_sha256": document.get("grid_coordinate_sha256"),
        "grid_binding": "full TargetGrid, including z_w",
        "grid_shape": list(shape),
        "z_source": z_source if reflectivity_simulated is not None else None,
        "dealiasing": document.get("provenance", {}).get("dealiasing"),
        "superob_params": document.get("superob_params"),
        "batches": used,
        "radars": [{"id": str(r["id"]), "lat_deg": r["lat_deg"],
                    "lon_deg": r["lon_deg"], "alt_m": r["alt_m"]}
                   for r in document["radars"]],
        "notes": [
            "per-radar velocity batches; projections are never averaged",
            "H direction is the file's own beam unit vectors, not "
            "re-derived geometry",
            "errors are standard deviations, passed through unsquared",
            "the file was bound to the caller's TargetGrid arrays, not to "
            "its grid_identity_sha256 string; z_w is checked and is not "
            "stored in the file",
        ],
    }
    return batches, provenance


def letkf_grid_geometry(target_grid):
    """The filter's localisation metric from the radar lane's target grid.

    The whole 3-D georeference goes across, because all of it is already on
    the file and every reduction of it is a lie the filter cannot detect:

    * **layer-midpoint heights per column**, ``(nz, ny, nx)``, the same
      midpoint reduction of ``z_w`` the observation operator uses.  A single
      representative column -- the domain mean, say -- reports one height
      per level, so every same-level pair is zero metres apart vertically
      and takes a vertical localisation weight of exactly 1 no matter what
      the terrain between them does.  Over a ridge that is not a small
      approximation: 2 km of relief under a 4 km cutoff is the difference
      between a weight of 1 and Gaspari-Cohn's 5/24.
    * **mass-point latitude/longitude**, so horizontal separation is a
      geodesic.  ``dx_m`` is the projection's nominal spacing; the physical
      spacing on a conformal projection is ``dx/mapfac``, and a fixed radius
      in metres therefore reaches a different number of columns depending on
      where the domain sits relative to the standard parallels.

    ``dx_m``/``dy_m`` still go across as well: they are what sizes the
    horizontal stencil in the degenerate no-coordinates case, and they are
    part of the grid's identity.
    """

    from gpuwm.da.letkf import GridGeometry                # noqa: PLC0415

    z_w = np.asarray(target_grid.z_w, dtype=np.float64)
    midpoints = 0.5 * (z_w[:-1] + z_w[1:])
    return GridGeometry(dx_m=float(target_grid.dx_m),
                        dy_m=float(target_grid.dy_m),
                        heights_m=np.ascontiguousarray(midpoints),
                        lat_deg=np.asarray(target_grid.lat, dtype=np.float64),
                        lon_deg=np.asarray(target_grid.lon, dtype=np.float64))
