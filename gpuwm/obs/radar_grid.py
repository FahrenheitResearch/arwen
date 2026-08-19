"""``gpuwm-obs.radar-grid.v1`` — gridded radar observations, the DA front door.

This is the cross-lane contract: the DA lanes read this file and nothing
upstream of it.  Everything a data assimilation step needs to use an
observation is in here, and everything it must not have to guess is
refused rather than defaulted.

Dimensions
----------
``level``, ``south_north``, ``west_east``
    The target model grid, exactly.  A reader that disagrees on any of the
    three, or on ``grid_identity_sha256``, must stop.
``radar``
    Contributing radars, in the order they were merged.
``nchar``, ``ntime``
    Fixed widths for the radar id and ISO8601 timestamp strings.

Variables
---------
``z_obs`` [dBZ], ``z_mask``, ``z_err``
    Reflectivity, merged across radars.  ``z_max`` and ``z_mean`` ship
    beside ``z_obs`` because the choice between them is the consumer's, and
    neither is recoverable from the other.  ``z_mean`` is the linear-Z mean
    expressed in dBZ.
``vr_obs`` [m/s], ``vr_mask``, ``vr_err``
    Radial velocity, with a leading ``radar`` axis.  **This is a deliberate
    widening of the nominal ``(level, south_north, west_east)`` shape and
    the one place this writer adds a dimension rather than collapsing
    one.**  Two radars looking at one cell measure two different
    projections of the same wind; merging them produces a number with no
    observation operator, and dropping one throws away half the
    information.  The trailing three axes are still the target grid, so a
    consumer that wants a single radar indexes ``vr_obs[r]`` and is back on
    the nominal shape.
``vr_beam_east|north|up``
    The unit look vector per retained cell — ``Vr = u*east + v*north +
    w*up``.  A radial velocity without its look direction is not an
    observation, so the operator ships with the value.

Masks are ``int8``: ``1`` where a usable observation exists, ``0``
everywhere else.  There is no "maybe".  ``vr_rejected`` counts the gates
that were dropped for aliasing risk, so a zero mask over a storm reads as
"masked, and here is how many" rather than "no data".

**Dealiasing is off unless asked for.**  With ``SuperobParams.dealias``
unset -- the default -- see :mod:`gpuwm.obs.superob` for the fail-closed
rules that mask aliasing rather than correct it, and this file is written
exactly as it always has been.  With it set, :mod:`gpuwm.obs.dealias` has
unfolded the velocities and ``provenance.dealias`` carries the three-state
account of every gate it resolved, left alone, or refused.  The
``dealiasing`` attribute states which of the two this file is, and is the
first thing a consumer should read.

Which library moves the bytes
-----------------------------
Both halves are Drew's Rust on the bare default.  The file is WRITTEN by the
classic writer at ``tools/rustwx/crates/netcdf-writer`` through
:mod:`gpuwm.obs.grid_product`, in the CDF-5 container;
``GPUWM_OBS_GRID_WRITER=python`` reaches the netCDF4/HDF5 writer this schema
used through 2.4 and is an announced workaround, never a fallback.  Every
OBSERVATION is READ back through :mod:`gpuwm.netcdf_bridge`.

The file's METADATA -- its global attributes and the two fixed-width
character variables ``radar_id`` and ``radar_valid_time`` -- comes off one
netCDF4 open, the only netCDF4 read left in this module.  That is the split
``gpuwm.downscale._parent_geometry`` already draws, and here it is forced:
the vendored netcrust reader exposes no character read at all, and it cannot
see an HDF5 attribute that landed in an object-header continuation block --
it returns ``None`` for one rather than failing.  A two-radar product's
``provenance`` is past that point, and every radar-grid file written before
this release is HDF5, so reading the metadata on the bridge turned "this
build cannot read the attribute" into "the file does not carry one" for the
whole assimilation record.  Values stay on the bridge; metadata does not.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import uuid

import numpy as np

from gpuwm.obs.superob import (CLEAR_AIR_SOURCE, CLEAR_AIR_SOURCE_CENSOR,
                              CLEAR_AIR_SOURCES, GriddedObservations,
                              SuperobParams)
from gpuwm.obs.target_grid import GridMismatchError, TargetGrid

#: The contract string. Consumers pin this exact value.
RADAR_GRID_SCHEMA = "gpuwm-obs.radar-grid.v1"

#: Status of a product this lane can honestly claim: the geometry and the
#: schema are proven by tests, and the observation-error model is a
#: documented parameterization rather than a tuned one.  Real cycling
#: assimilations HAVE consumed this schema (four 2026-07-30 analyses, the
#: ``evidence/da-cycling/`` receipts), so "no case has yet assimilated
#: from it" is no longer the reason.  The label stays because those runs
#: were a first demonstration, not a certification: no shakedown case has
#: yet passed the radar-DA demonstration program's full control battery
#: (registered claim rule, ablation, chaos-pricing null, persistence,
#: wrong-day, dual-run screens).  Promotion gate: that program's
#: shakedown package with every registered control green, decided by the
#: owner, not inferred from one experiment.
RADAR_GRID_STATUS = "READY_NOT_YET_ASSIMILATION_CERTIFIED"

#: Width of the fixed-length radar id strings.
ID_WIDTH = 8

#: Width of the fixed-length ISO8601 timestamp strings ("...T20:03:16Z").
TIME_WIDTH = 24

_FILL_F4 = np.float32(-9.99e30)

_PLANE = ("level", "south_north", "west_east")
_VOLUME = ("radar",) + _PLANE
_MASS = ("south_north", "west_east")

#: Every variable this schema defines, with the **dimension tuple** it must
#: carry, its storage type, and its units.
#:
#: The dimension tuple is the load-bearing part, and it is here rather than
#: implied by the writer because a reader that checks only array *shape* is
#: not checking anything.  A file whose ``z_obs`` is declared
#: ``('level', 'west_east', 'south_north_stag')`` on a square grid has the
#: right shape, the right rank, the right dtype, and a transposed,
#: half-cell-shifted field inside it; the only thing that distinguishes it
#: from a correct file is the tuple.  ``None`` units means the variable
#: declares none (the masks).
CANONICAL_VARIABLES: dict[str, tuple[tuple[str, ...], str, str | None]] = {
    "XLAT": (_MASS, "float32", "degree_north"),
    "XLONG": (_MASS, "float32", "degree_east"),
    "HGT": (_MASS, "float32", "m"),
    "z_obs": (_PLANE, "float32", "dBZ"),
    "z_mask": (_PLANE, "int8", None),
    "z_err": (_PLANE, "float32", "dBZ"),
    "z_max": (_PLANE, "float32", "dBZ"),
    "z_mean": (_PLANE, "float32", "dBZ"),
    "z_count": (_PLANE, "int32", "count"),
    "vr_obs": (_VOLUME, "float32", "m s-1"),
    "vr_mask": (_VOLUME, "int8", None),
    "vr_err": (_VOLUME, "float32", "m s-1"),
    "vr_count": (_VOLUME, "int32", "count"),
    "vr_rejected": (_VOLUME, "int32", "count"),
    "vr_beam_east": (_VOLUME, "float32", "1"),
    "vr_beam_north": (_VOLUME, "float32", "1"),
    "vr_beam_up": (_VOLUME, "float32", "1"),
    "radar_id": (("radar", "nchar"), "S1", None),
    "radar_lat": (("radar",), "float64", "degree_north"),
    "radar_lon": (("radar",), "float64", "degree_east"),
    "radar_alt": (("radar",), "float64", "m"),
    "radar_valid_time": (("radar", "ntime"), "S1", None),
}

#: Clear-air ("zero") observations: an **optional extension**, present in
#: files written after the capability landed and absent from every file
#: written before it.
#:
#: Optional rather than canonical on purpose.  ``read_radar_grid`` refuses
#: a file missing any :data:`CANONICAL_VARIABLES` entry, so promoting these
#: would retroactively invalidate every observation file already on disk --
#: including the ones the cycling receipts in ``evidence/da-cycling`` were
#: produced from, which must stay readable for those runs to be
#: reproducible.  When present they are checked with exactly the same
#: dimension-tuple rigor as the canonical set.
#:
#: There is no ``z0_obs``.  The value a clear-air observation differences
#: against is the forward operator's clear-air floor, which is a property
#: of the microphysics scheme being run, not of the radar; see
#: :class:`gpuwm.obs.superob.GriddedObservations`.
CLEAR_AIR_VARIABLES: dict[str, tuple[tuple[str, ...], str, str | None]] = {
    "z0_mask": (_PLANE, "int8", None),
    "z0_count": (_PLANE, "int32", "count"),
    "z0_err": (_PLANE, "float32", "dBZ"),
}

#: How the clear-air observations in a file were established, re-exported
#: from :mod:`gpuwm.obs.superob` where the two regimes are defined and
#: documented.  Written as a file attribute so a consumer can tell what it
#: is trusting: :data:`CLEAR_AIR_SOURCE` is the measurement-only regime, and
#: :data:`CLEAR_AIR_SOURCE_CENSOR` is the one that additionally reads the
#: Message-31 "below threshold" code the decoder now preserves.  A consumer
#: that does not recognise the value it finds must refuse the file rather
#: than assume either.
__all_clear_air__ = (CLEAR_AIR_SOURCE, CLEAR_AIR_SOURCE_CENSOR,
                     CLEAR_AIR_SOURCES)

#: Fixed-width dimensions, and the width each must have.
FIXED_DIMENSIONS = {"nchar": ID_WIDTH, "ntime": TIME_WIDTH}

#: Largest departure from unit length tolerated in a beam vector under a
#: true velocity mask.  Float32 storage of a normalised double is good to
#: about 1e-7; 1e-5 is far below any scaling error worth catching and far
#: above the round trip.
BEAM_NORM_TOLERANCE = 1.0e-5

#: What this file's velocities have and have not been through, stated so
#: that a consumer cannot read the masks as an alias-proof quality gate.
#:
#: The previous wording -- "aliasing-risk gates are masked and counted" --
#: was true and was read as more than it said.  Masks find risk signatures.
#: They do not find aliasing.  A region of gates that folds *together* has
#: a present and plausible Nyquist, speeds well inside the 0.8 threshold,
#: no in-cell spread, and no gate-to-gate jump in its interior: a smooth,
#: plausible, wrong wind field that every rule here passes.  The
#: ``fold_suspicion`` block in provenance carries what the shear scan did
#: see, per sweep, including the sweeps where it saw nothing.
_DEALIASING_STATEMENT = (
    "none. Velocities are masked on four risk signatures -- speed above "
    "nyquist_reject_fraction of the sweep Nyquist, a missing or "
    "implausible Nyquist, in-cell spread above nyquist_spread_fraction, "
    "and gates flanking a gate-to-gate jump above shear_fold_fraction of "
    "the full Nyquist interval -- and every rejection is counted into "
    "vr_rejected and provenance.counts. These are signatures of risk, not "
    "detections of aliasing: a spatially coherent fold covering a whole "
    "region passes all four, because its interior is smooth and its speeds "
    "are small. Excluding that case requires true dealiasing, which is "
    "region-global unwrapping and is not implemented here. See "
    "provenance.fold_suspicion for the per-sweep shear-scan record."
)

#: What this file's velocities have been through when dealiasing DID run.
#:
#: The statement above is not merely reworded here, it is *retired* for this
#: file: its central claim -- "excluding that case requires true dealiasing,
#: which is region-global unwrapping and is not implemented here" -- stops
#: being true the moment the unfolder runs, and a provenance string that
#: understates what happened is as much a lie as one that overstates it.  A
#: consumer distinguishes the two files by this attribute and by the
#: presence of ``provenance.dealias``; neither can be forged by a rebuild,
#: because both are written from the same parameter object that produced the
#: velocities.
_DEALIASED_STATEMENT = (
    "region-based unfolding (gpuwm.obs.dealias). Each sweep is segmented "
    "into regions whose adjacent gates differ by less than "
    "region_join_fraction of Nyquist, so no fold lies inside a region; "
    "regions are anchored to an absolute fold by a fold-aware VAD (Browning "
    "& Wexler 1968) fitted per range band and cross-checked against a wind "
    "profile pooled from every cut in the volume, which is believed at a "
    "height only where two or more different elevations reached it. A region "
    "may anchor only where it still looks environmental after unfolding, so "
    "a rotational couplet is never flattened onto the background wind. "
    "Remaining regions resolve across their strongest confident "
    "inter-region boundary. EVERY velocity gate ends in exactly "
    "one of three states -- confidently unfolded, confidently unchanged, or "
    "REJECTED with a recorded reason -- and all three are counted in "
    "provenance.dealias, whose totals are asserted to balance against the "
    "gates offered. Where the evidence is ambiguous the gate is rejected, "
    "not guessed: a region that is never anchored and never linked, a region "
    "on a contradicted boundary, a fold beyond max_fold, a speed beyond "
    "max_speed_ms, or a departure from the reference beyond "
    "max_reference_departure_ms. Because a resolved gate's fold state is "
    "KNOWN, it is "
    "bounded by max_speed_ms rather than by nyquist_reject_fraction of "
    "Nyquist; that fraction still governs every gate on a sweep the unfolder "
    "declined. See provenance.dealias.sweeps for the per-sweep account and "
    "provenance.fold_suspicion for the shear scan, which now runs on the "
    "UNFOLDED field and so measures what the unfolder missed."
)


#: What this file's velocities have been through when the REGION-GLOBAL
#: engine ran.
#:
#: A third statement rather than a footnote on the second, for the reason
#: the second is not a footnote on the first: the claim that separates them
#: is load-bearing.  ``_DEALIASED_STATEMENT`` promises that every gate is
#: confidently unfolded, confidently unchanged, or REJECTED, and that where
#: the evidence is ambiguous the gate is rejected rather than guessed.
#: This engine makes no such promise -- it has no environmental reference
#: to be ambiguous against, and it assigns a fold to every region it
#: resolved.  A consumer reading a file made by this engine must not be
#: told it was made by an abstaining one.
_DEALIASED_STATEMENT_REGION_GLOBAL = (
    "region-global unfolding (gpuwm.obs.dealias_region, driving the "
    "vendored region-global-dealias crate at tools/region_global_dealias -- "
    "a Rust port of Py-ART's dealias_region_based, Helmus & Collis 2016, "
    "verified fold-for-fold identical to Py-ART 2.2.5 on this pipeline's "
    "own decoded sweeps). Each sweep is segmented into regions that are "
    "internally fold-free, and the whole region network is unfolded "
    "JOINTLY: the strongest region pair merges first and every remaining "
    "connection is re-weighted with what that merge established, rather "
    "than each boundary being committed from local information alone. "
    "Regions separated by up to 100 gates of no-data are linked. THIS "
    "ENGINE DOES NOT ABSTAIN: it carries no environmental reference, and "
    "every region it resolved is assigned a fold, so provenance.dealias "
    "records gates as unchanged or unfolded and rejects only gates that "
    "were not numbers to begin with and gates on a sweep whose Nyquist "
    "could not be believed. It publishes no region-graph counts across its "
    "ABI, so provenance.dealias.totals reports those as null rather than "
    "as zero. Velocities are exact: a gate the solver did not move is "
    "returned bit-identically, and a gate it moved differs by a whole "
    "number of Nyquist intervals and nothing else. Because a resolved "
    "gate's fold state is known, it is bounded by max_speed_ms rather than "
    "by nyquist_reject_fraction of Nyquist. See provenance.dealias.sweeps "
    "for the per-sweep account and provenance.fold_suspicion for the shear "
    "scan, which runs on the UNFOLDED field and so measures what the "
    "unfolder missed."
)


def dealiasing_statement(params: SuperobParams) -> str:
    """The ``dealiasing`` attribute these parameters entail.

    A function rather than a constant because the file's honest description
    of itself depends on what ran, and the only thing that knows what ran is
    the parameter set the velocities were made with -- now including *which
    engine* the parameter set selected.
    """

    from gpuwm.obs.dealias import ENGINE_REGION_GLOBAL     # noqa: PLC0415

    if params.dealias is None:
        return _DEALIASING_STATEMENT
    if params.dealias.engine == ENGINE_REGION_GLOBAL:
        return _DEALIASED_STATEMENT_REGION_GLOBAL
    return _DEALIASED_STATEMENT


class RadarGridSchemaError(ValueError):
    """A radar-grid file that does not satisfy the contract."""


def write_radar_grid(path: str | Path, observations: GriddedObservations,
                     grid: TargetGrid, *, valid_time: str,
                     params: SuperobParams,
                     provenance: dict | None = None,
                     overwrite: bool = False) -> dict:
    """Write one ``gpuwm-obs.radar-grid.v1`` file, atomically.

    Returns the receipt: schema, status, path, byte count, sha256, and the
    grid identity the file is bound to.
    """

    from gpuwm.obs.grid_product import (                 # noqa: PLC0415
        open_obs_grid_product)

    path = Path(path)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {path}")
    # The parameters go into the file's ``superob_params`` attribute and
    # into provenance, where a consumer reads them as the description of how
    # these observations were made.  A set that could not have produced them
    # must not be published beside them, so it is re-checked here rather
    # than assumed to have come straight from a validated construction.
    params.validate()
    shape = (grid.nz, grid.ny, grid.nx)
    if observations.z_obs.shape != shape:
        raise GridMismatchError(
            f"z_obs has shape {observations.z_obs.shape}, target grid is "
            f"{shape}")
    n_radar = len(observations.radars)
    if observations.vr_obs.shape != (n_radar,) + shape:
        raise GridMismatchError(
            f"vr_obs has shape {observations.vr_obs.shape}, expected "
            f"{(n_radar,) + shape}")
    if n_radar == 0:
        raise ValueError("a radar-grid product needs at least one radar")

    identity = grid.identity_sha256()
    coordinates = _coordinate_digests(grid)
    temp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    try:
        with open_obs_grid_product(temp) as dataset:
            dataset.createDimension("level", grid.nz)
            dataset.createDimension("south_north", grid.ny)
            dataset.createDimension("west_east", grid.nx)
            dataset.createDimension("radar", n_radar)
            dataset.createDimension("nchar", ID_WIDTH)
            dataset.createDimension("ntime", TIME_WIDTH)

            dataset.setncatts({
                "schema": RADAR_GRID_SCHEMA,
                "status": RADAR_GRID_STATUS,
                "valid_time": valid_time,
                "grid_identity_sha256": identity,
                "grid_coordinate_sha256": _canonical_json(coordinates),
                "grid_name": grid.name,
                "grid_source": grid.source,
                "MAP_PROJ": np.int32(grid.projection.wrf_map_proj),
                "MAP_PROJ_CHAR": grid.projection.map_proj_char,
                "TRUELAT1": np.float64(grid.truelat1),
                "TRUELAT2": np.float64(grid.truelat2),
                "STAND_LON": np.float64(grid.stand_lon),
                "CEN_LAT": np.float64(grid.projection.cen_lat),
                "CEN_LON": np.float64(grid.projection.cen_lon),
                "DX": np.float64(grid.dx_m),
                "DY": np.float64(grid.dy_m),
                "GRIDTYPE": "C",
                "dealiasing": dealiasing_statement(params),
                "superob_params": _canonical_json(params.to_payload()),
                "provenance": _canonical_json(
                    _provenance_payload(observations, params, provenance)),
            })

            _grid_var(dataset, "XLAT", ("south_north", "west_east"),
                      grid.lat, "LATITUDE, SOUTH IS NEGATIVE", "degree_north")
            _grid_var(dataset, "XLONG", ("south_north", "west_east"),
                      grid.lon, "LONGITUDE, WEST IS NEGATIVE", "degree_east")
            _grid_var(dataset, "HGT", ("south_north", "west_east"),
                      grid.terrain_m, "Terrain Height", "m")

            plane = ("level", "south_north", "west_east")
            volume = ("radar",) + plane
            _obs_var(dataset, "z_obs", plane, observations.z_obs, "dBZ",
                     "superobbed reflectivity (see z_reduce in "
                     "superob_params)")
            _mask_var(dataset, "z_mask", plane, observations.z_mask,
                      "1 where z_obs is a usable observation")
            _obs_var(dataset, "z_err", plane, observations.z_err, "dBZ",
                     "observation error standard deviation for z_obs")
            _obs_var(dataset, "z_max", plane, observations.z_max, "dBZ",
                     "in-cell maximum reflectivity")
            _obs_var(dataset, "z_mean", plane, observations.z_mean, "dBZ",
                     "in-cell mean reflectivity, averaged in linear Z")
            _count_var(dataset, "z_count", plane, observations.z_count,
                       "contributing reflectivity gates")

            # Written only when the product actually carries a clear-air
            # assessment.  A caller that assembled observations without one
            # gets a file with no z0 variables and no clear_air_source, and
            # the DA side reads that as "zeroes were never established"
            # rather than as "no clear air was found".
            clear_air = (observations.z0_mask, observations.z0_count,
                         observations.z0_err)
            if all(part is not None for part in clear_air):
                _mask_var(dataset, "z0_mask", plane, observations.z0_mask,
                          "1 where the radar(s) measured this cell and every "
                          "one of them found no significant echo; 0 also "
                          "means 'never measured', which is not a clear-air "
                          "observation")
                _count_var(dataset, "z0_count", plane, observations.z0_count,
                           "measured gates below the reflectivity floor "
                           "supporting the clear-air observation")
                _obs_var(dataset, "z0_err", plane, observations.z0_err, "dBZ",
                         "observation error standard deviation for a "
                         "clear-air zero")
                source = observations.clear_air_source
                if source not in CLEAR_AIR_SOURCES:
                    raise RadarGridSchemaError(
                        f"refusing to write clear_air_source {source!r}: a "
                        "reader that does not recognise the value must "
                        "refuse the file, so writing an unknown one "
                        f"publishes unreadable zeroes. Known: "
                        f"{sorted(CLEAR_AIR_SOURCES)}")
                dataset.setncattr("clear_air_source", source)

            _obs_var(dataset, "vr_obs", volume, observations.vr_obs, "m s-1",
                     "superobbed radial velocity, positive away from the "
                     "radar")
            _mask_var(dataset, "vr_mask", volume, observations.vr_mask,
                      "1 where vr_obs is a usable observation")
            _obs_var(dataset, "vr_err", volume, observations.vr_err, "m s-1",
                     "observation error standard deviation for vr_obs")
            _count_var(dataset, "vr_count", volume, observations.vr_count,
                       "contributing radial-velocity gates")
            _count_var(dataset, "vr_rejected", volume,
                       observations.vr_rejected,
                       "gates dropped for aliasing risk (not dealiased)")
            for name, array, axis in (
                    ("vr_beam_east", observations.vr_beam_east, "east"),
                    ("vr_beam_north", observations.vr_beam_north, "north"),
                    ("vr_beam_up", observations.vr_beam_up, "up")):
                _obs_var(dataset, name, volume, array, "1",
                         f"{axis} component of the beam unit vector; "
                         "Vr = u*east + v*north + w*up")

            ids = dataset.createVariable("radar_id", "S1", ("radar", "nchar"))
            ids.description = "contributing radar site id"
            ids[:] = _char_array(
                [radar["id"] for radar in observations.radars], ID_WIDTH)
            for key, unit, description in (
                    ("lat_deg", "degree_north", "radar latitude"),
                    ("lon_deg", "degree_east", "radar longitude"),
                    ("alt_m", "m", "radar antenna height above sea level")):
                variable = dataset.createVariable(
                    f"radar_{key.split('_')[0]}", "f8", ("radar",))
                variable.units = unit
                variable.description = description
                variable[:] = np.array(
                    [radar[key] for radar in observations.radars],
                    dtype=np.float64)
            times = dataset.createVariable("radar_valid_time", "S1",
                                           ("radar", "ntime"))
            times.description = ("per-radar volume start time; the file-level "
                                 "valid_time is the analysis time")
            times[:] = _char_array(
                [radar["valid_time"] for radar in observations.radars],
                TIME_WIDTH)

        payload = temp.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()

    return {
        "schema": RADAR_GRID_SCHEMA,
        "status": RADAR_GRID_STATUS,
        "path": str(path),
        "bytes": len(payload),
        "sha256": digest,
        "grid_identity_sha256": identity,
        "valid_time": valid_time,
        "radars": [radar["id"] for radar in observations.radars],
        "dims": {"level": grid.nz, "south_north": grid.ny,
                 "west_east": grid.nx, "radar": n_radar},
    }


def read_radar_grid(path: str | Path, *,
                    expected_grid_identity: str | None = None,
                    expected_grid: TargetGrid | None = None) -> dict:
    """Read a radar-grid file back, checking the contract before the data.

    ``expected_grid_identity`` is how a consumer fails closed on *which*
    grid: pass the identity of the grid it intends to assimilate into, and a
    file written for any other grid raises instead of being silently
    regridded.

    ``expected_grid`` is how it fails closed on the grid *itself*, and it is
    what the DA read path always passes -- see
    :func:`gpuwm.da.obs_radar.read_document`, which requires it.  The
    identity attribute is a claim the file makes about arrays it does not
    all contain: it is computed over ``lat``, ``lon``, ``z_w`` and terrain
    at float64, and only the first, second and fourth are stored, at
    float32.  A reader alone therefore cannot recompute it, so an attribute
    relabelled to name a different grid is consistent with everything a
    reader can check by itself.  Passing the consumer's own
    :class:`TargetGrid` re-derives the whole digest chain from arrays in
    hand and compares it to what the file recorded, which is the only check
    that reaches ``z_w``.  It implies ``expected_grid_identity``.

    Whatever the caller demands, the structural contract is always checked:
    every canonical variable present, with its exact dimension **tuple**,
    storage type and units, and every stored coordinate array hashing to
    the digest the writer bound to the identity.  A file that carries the
    right identity string over transposed or staggered variables is a
    refusal, not a regrid.
    """

    from gpuwm import netcdf_bridge                      # noqa: PLC0415

    path = Path(path)
    if expected_grid is not None:
        demanded = expected_grid.identity_sha256()
        if (expected_grid_identity is not None
                and expected_grid_identity != demanded):
            raise GridMismatchError(
                f"expected_grid hashes to {demanded} but "
                f"expected_grid_identity demands {expected_grid_identity}; "
                "the caller is asking for two different grids")
        expected_grid_identity = demanded

    # One netCDF4 open for the METADATA -- the global attributes and the two
    # character variables -- before any decode, because the bridge can see
    # neither.  See :func:`_read_metadata`.
    attributes, ids, times = _read_metadata(path)

    # Every OBSERVATION -- value, error, count, mask, beam component and
    # coordinate -- is decoded by ``rw_netcdf``.  Masking is turned off for
    # the same reason the satellite twin turns it off: this schema writes
    # ``_FillValue = -9.99e30`` and its consumers read that sentinel back
    # deliberately, so handing them NaN would change what the DA side sees.
    with netcdf_bridge.open_dataset(path) as dataset:
        dataset.set_auto_mask(False)
        schema = attributes.get("schema")
        if schema != RADAR_GRID_SCHEMA:
            raise RadarGridSchemaError(
                f"{path.name}: schema {schema!r}, expected "
                f"{RADAR_GRID_SCHEMA!r}")
        identity = attributes.get("grid_identity_sha256")
        if not identity:
            raise RadarGridSchemaError(
                f"{path.name}: no grid_identity_sha256; a gridded "
                "observation set without its grid identity cannot be "
                "assimilated safely")
        if (expected_grid_identity is not None
                and identity != expected_grid_identity):
            raise GridMismatchError(
                f"{path.name} is bound to grid {identity}, the caller "
                f"requires {expected_grid_identity}")
        _require_structure(path, dataset)
        stored = _require_coordinates(path, dataset)
        if expected_grid is not None:
            _require_grid(path.name, expected_grid, stored)

        result = {
            "schema": schema,
            "status": attributes.get("status"),
            "valid_time": attributes.get("valid_time"),
            "grid_identity_sha256": identity,
            "grid_coordinate_sha256": stored,
            "dims": {name: len(dataset.dimensions[name])
                     for name in ("level", "south_north", "west_east",
                                  "radar")},
            # None for a file written before the clear-air extension; the
            # DA side reads this to decide whether a zero batch may be
            # built at all, so its absence must be visible rather than
            # defaulted to the value this build happens to write.  Absent
            # now means absent FROM THE FILE: the attributes come off
            # netCDF4, which sees every one the writer stored.
            "clear_air_source": attributes.get("clear_air_source"),
            "superob_params": json.loads(
                attributes.get("superob_params", "{}")),
            "provenance": json.loads(attributes.get("provenance", "{}")),
            "radars": [],
            "variables": {},
        }
        for name in dataset.variables:
            if name in _CHARACTER_VARIABLES:
                continue
            result["variables"][name] = np.asarray(dataset.variables[name][:])
        _require_beam_vectors(path, result["variables"])
        n_radar = len(dataset.dimensions["radar"])

    if len(ids) != n_radar:
        raise RadarGridSchemaError(
            f"{path.name}: {len(ids)} radar ids for {n_radar} radars")
    # The per-radar geometry comes out of the arrays already decoded above
    # rather than being read a second time: one decode per variable is the
    # bridge's unit of work, and re-indexing the file three times per radar
    # would be three more of them for numbers already in hand.
    for index, site in enumerate(ids):
        result["radars"].append({
            "id": site,
            "lat_deg": float(result["variables"]["radar_lat"][index]),
            "lon_deg": float(result["variables"]["radar_lon"][index]),
            "alt_m": float(result["variables"]["radar_alt"][index]),
            "valid_time": times[index],
        })
    return result


#: The two variables the Rust decoder cannot hand back.
_CHARACTER_VARIABLES = ("radar_id", "radar_valid_time")


def _is_character(variable) -> bool:
    """Is this variable a fixed-width character array?"""

    declared = getattr(variable, "is_character", None)
    if isinstance(declared, bool):
        return declared
    return np.dtype(variable.dtype).kind in ("S", "U", "O")


def _read_metadata(path: Path) -> tuple[dict, list[str], list[str]]:
    """The global attributes, radar ids and timestamps, in ONE netCDF4 open.

    This is the ONLY netCDF4 read left in this module.  Two things force it,
    and both are the same shape of quiet wrong:

    * the vendored netcrust reader promotes numeric variables to f64 and
      exposes no character read at all (upstream gap 3), so handing the
      caller an empty list of ids would say "this file has no radars";
    * it cannot see an HDF5 global attribute that landed in an object-header
      continuation block, and answers ``None`` for one instead of failing.
      Seven attributes with one large among them is enough to spill, which a
      two-radar product's ``provenance`` reaches, so reading the metadata on
      the bridge said "this file records nothing about how it was made" for
      every radar-grid file written before this release -- all of them HDF5.

    Reading both off one open keeps the netCDF4 surface at exactly one
    ``Dataset``, which :mod:`tests.test_obs_grid_writer` pins.
    """

    import netCDF4                                       # noqa: PLC0415

    with netCDF4.Dataset(path, "r") as dataset:
        attributes = {name: dataset.getncattr(name)
                      for name in dataset.ncattrs()}
        # A file that carries neither character variable is not refused
        # here: the schema and structure checks in the caller run first and
        # say WHICH contract the file breaks.  An absent id array reaches
        # the caller as an empty list, which its radar-count check names.
        strings = [
            _read_strings(dataset.variables[name][:])
            if name in dataset.variables else []
            for name in _CHARACTER_VARIABLES]
        return attributes, strings[0], strings[1]


def _require_structure(path: Path, dataset) -> None:
    """Every canonical variable, with its exact dimension tuple and type."""

    for dimension in ("level", "south_north", "west_east", "radar"):
        if dimension not in dataset.dimensions:
            raise RadarGridSchemaError(
                f"{path.name}: missing dimension {dimension!r}")
    for dimension, width in FIXED_DIMENSIONS.items():
        if dimension not in dataset.dimensions:
            raise RadarGridSchemaError(
                f"{path.name}: missing dimension {dimension!r}")
        actual = len(dataset.dimensions[dimension])
        if actual != width:
            raise RadarGridSchemaError(
                f"{path.name}: dimension {dimension!r} is {actual}, this "
                f"schema fixes it at {width}")

    missing = [name for name in CANONICAL_VARIABLES
               if name not in dataset.variables]
    if missing:
        raise RadarGridSchemaError(
            f"{path.name}: missing variables {missing}")

    # Clear-air variables are optional but not individually optional: a file
    # carrying z0_mask without z0_err is a file whose zeroes cannot be
    # assimilated, and reading it half-way is how one of the three gets
    # defaulted.  All three or none.
    present = [name for name in CLEAR_AIR_VARIABLES
               if name in dataset.variables]
    if present and len(present) != len(CLEAR_AIR_VARIABLES):
        raise RadarGridSchemaError(
            f"{path.name}: carries clear-air variables {sorted(present)} but "
            f"not {sorted(set(CLEAR_AIR_VARIABLES) - set(present))}. The "
            "clear-air extension is all three variables or none of them; a "
            "partial set has no error to assimilate with or no count to "
            "justify the mask")
    if present:
        source = getattr(dataset, "clear_air_source", None)
        if source not in CLEAR_AIR_SOURCES:
            raise RadarGridSchemaError(
                f"{path.name}: declares clear_air_source {source!r}, this "
                f"reader understands {sorted(CLEAR_AIR_SOURCES)}. How a zero "
                "was established decides what it means -- gates measured "
                "below the floor and a decoder's below-threshold flag are "
                "different observations with different coverage, and "
                "assimilating one as the other misstates both the coverage "
                "and the error")

    checked = dict(CANONICAL_VARIABLES)
    checked.update({name: CLEAR_AIR_VARIABLES[name] for name in present})
    for name, (dims, dtype, units) in checked.items():
        variable = dataset.variables[name]
        if tuple(variable.dimensions) != dims:
            raise RadarGridSchemaError(
                f"{path.name}: {name} is declared over "
                f"{tuple(variable.dimensions)}, this schema defines it over "
                f"{dims}. Dimension *names and order* are the contract, not "
                "array shape: a transposed field, or one on a staggered "
                "dimension of equal length, has the right shape and the "
                "wrong values, and only the tuple can tell them apart")
        if np.dtype(dtype).kind == "S":
            # A fixed-width character array.  The decoder spells that
            # ``Char`` in a classic container and ``String`` in an HDF5 one
            # -- the same bytes on disk, two reader spellings -- so the
            # question asked here is the one the schema actually means: is
            # this a character variable?  Comparing numpy dtypes instead
            # would refuse every radar-grid file written before the writer
            # moved to the classic container, which is a compatibility
            # break dressed up as a schema violation.
            if not _is_character(variable):
                raise RadarGridSchemaError(
                    f"{path.name}: {name} is stored as "
                    f"{np.dtype(variable.dtype)}, this schema defines it as "
                    f"{np.dtype(dtype)}")
        elif np.dtype(variable.dtype) != np.dtype(dtype):
            raise RadarGridSchemaError(
                f"{path.name}: {name} is stored as {np.dtype(variable.dtype)}, "
                f"this schema defines it as {np.dtype(dtype)}")
        if units is None:
            continue
        declared = getattr(variable, "units", None)
        if declared != units:
            raise RadarGridSchemaError(
                f"{path.name}: {name} declares units {declared!r}, this "
                f"schema defines {units!r}. Errors are standard deviations "
                "and beams are unit vectors; a unit that says otherwise is "
                "a different quantity wearing the same name")


def _require_coordinates(path: Path, dataset) -> dict:
    """The stored coordinates must be the ones the identity was made from."""

    raw = getattr(dataset, "grid_coordinate_sha256", None)
    if not raw:
        raise RadarGridSchemaError(
            f"{path.name}: no grid_coordinate_sha256. The identity is a "
            "digest of arrays this file only partly contains, so without "
            "the per-array digests nothing here can be checked against it "
            "-- the identity string would be an unbound claim")
    try:
        stored = json.loads(raw)
    except json.JSONDecodeError as error:
        raise RadarGridSchemaError(
            f"{path.name}: grid_coordinate_sha256 is not JSON: "
            f"{error}") from error
    expected_keys = {"XLAT", "XLONG", "HGT", "z_w"}
    if set(stored) != expected_keys:
        raise RadarGridSchemaError(
            f"{path.name}: grid_coordinate_sha256 carries {sorted(stored)}, "
            f"expected {sorted(expected_keys)}")
    for name in ("XLAT", "XLONG", "HGT"):
        actual = _digest(dataset.variables[name][:], np.float32)
        if actual != stored[name]:
            raise GridMismatchError(
                f"{path.name}: the stored {name} hashes to {actual}, but the "
                f"file's own grid_coordinate_sha256 says {stored[name]}. The "
                "coordinates are not the ones the grid identity was computed "
                "from, so the identity attribute describes some other grid")
    return stored


def require_grid_binding(document, grid: TargetGrid, *,
                         label: str = "document") -> None:
    """Bind an **already-read** document to a grid the caller holds.

    :func:`read_radar_grid` does this from the file when it is given an
    ``expected_grid``.  A consumer that was handed the document instead of
    the path -- the DA adapter's second input shape, and the one a cycle
    uses when several batches come off one read -- has no file to re-open,
    so the same binding runs off the digests the reader carried out in
    ``grid_coordinate_sha256``.

    Both halves matter and neither implies the other.  The identity string
    is a claim; the digest table is the arrays.  A document whose identity
    was relabelled to some other grid still carries the digests of the
    arrays it was written from, and that is what catches it -- including
    ``z_w``, which the file does not store at all.
    """

    identity = document.get("grid_identity_sha256")
    demanded = grid.identity_sha256()
    if identity != demanded:
        raise GridMismatchError(
            f"{label} is bound to grid {identity}, the caller requires "
            f"{demanded}")
    stored = document.get("grid_coordinate_sha256")
    expected_keys = {"XLAT", "XLONG", "HGT", "z_w"}
    if not isinstance(stored, dict) or set(stored) != expected_keys:
        raise RadarGridSchemaError(
            f"{label}: grid_coordinate_sha256 is "
            f"{sorted(stored) if isinstance(stored, dict) else type(stored).__name__}, "
            f"expected {sorted(expected_keys)}. Without the per-array "
            "digests the identity string is an unbound claim and nothing "
            "here can be checked against the caller's grid")
    _require_grid(label, grid, stored)


def _require_grid(label: str, grid: TargetGrid, stored: dict) -> None:
    """Bind the file's declared coordinates to a grid the caller holds.

    This is the check that reaches ``z_w``.  The file stores float32
    ``XLAT``/``XLONG``/``HGT`` and no vertical coordinate at all, so a
    reader alone can never recompute the writer's digest; a caller who
    brought the grid can.
    """

    expected = _coordinate_digests(grid)
    disagree = sorted(name for name in expected
                      if expected[name] != stored.get(name))
    if disagree:
        raise GridMismatchError(
            f"{label}: the caller's target grid disagrees with the file "
            f"on {disagree}. The identity strings match, so this is the file "
            "claiming a grid whose arrays it was not written from"
            + (" -- including z_w, which the file does not store and which "
               "nothing but this comparison can check" if "z_w" in disagree
               else ""))


def _require_beam_vectors(path: Path, variables: dict) -> None:
    """A retained velocity without a unit look vector is not an observation.

    ``Vr = u*east + v*north + w*up`` is only that projection if the vector
    has unit length.  A file whose beams were scaled -- by a regrid, by a
    well-meant "normalisation" against a different norm, by float16 storage
    somewhere upstream -- changes every H(x) it touches by that factor, and
    silently: the innovations stay finite and plausible.
    """

    mask = np.asarray(variables["vr_mask"]).astype(bool)
    if not mask.any():
        return
    east = np.asarray(variables["vr_beam_east"], dtype=np.float64)
    north = np.asarray(variables["vr_beam_north"], dtype=np.float64)
    up = np.asarray(variables["vr_beam_up"], dtype=np.float64)
    if not (east.shape == north.shape == up.shape == mask.shape):
        raise RadarGridSchemaError(
            f"{path.name}: beam components {east.shape}/{north.shape}/"
            f"{up.shape} do not match vr_mask {mask.shape}")
    components = np.stack([east[mask], north[mask], up[mask]])
    if not np.all(np.isfinite(components)):
        raise RadarGridSchemaError(
            f"{path.name}: "
            f"{int(np.count_nonzero(~np.isfinite(components)))} beam "
            "component(s) under a true vr_mask are not finite")
    norm = np.sqrt((components ** 2).sum(axis=0))
    worst = float(np.max(np.abs(norm - 1.0)))
    if worst > BEAM_NORM_TOLERANCE:
        raise RadarGridSchemaError(
            f"{path.name}: {int(np.count_nonzero(np.abs(norm - 1.0) > BEAM_NORM_TOLERANCE))} "
            f"beam vector(s) under a true vr_mask are not unit length "
            f"(worst departure {worst:.3e}, tolerance "
            f"{BEAM_NORM_TOLERANCE:.0e}). Vr = u*east + v*north + w*up is "
            "that projection only for a unit look direction; a scaled one "
            "rescales every innovation it touches without ever going "
            "non-finite")


def _char_array(strings, width: int) -> np.ndarray:
    """``(n, width)`` of single bytes — NETCDF4_CLASSIC has no string type.

    Built by hand rather than through ``netCDF4.stringtochar``, whose
    encode path disagrees with a bytes-dtype input on this netCDF4 build.
    """

    rows = [list(value[:width].ljust(width).encode("ascii", "replace"))
            for value in strings]
    return np.array(rows, dtype="u1").view("S1").reshape(len(rows), width)


def _read_strings(chars) -> list[str]:
    """Inverse of :func:`_char_array`."""

    array = np.asarray(chars)
    return [b"".join(row.tolist()).decode("ascii").strip()
            for row in array]


# The four variable helpers hand the writer the caller's OWN array and let
# the declared external type do the cast, which is what netCDF4 has always
# done on assignment and what the classic facade does at write time.  Casting
# here instead would leave a float32 copy of every volume alive until the file
# closes -- several hundred megabytes for a multi-radar product, for nothing.
def _grid_var(dataset, name, dims, values, description, units):
    variable = dataset.createVariable(name, "f4", dims)
    variable.description = description
    variable.units = units
    variable.coordinates = "XLONG XLAT"
    variable[:] = values


def _obs_var(dataset, name, dims, values, units, description):
    variable = dataset.createVariable(name, "f4", dims,
                                      fill_value=_FILL_F4)
    variable.units = units
    variable.description = description
    variable.coordinates = "XLONG XLAT"
    variable[:] = values


def _mask_var(dataset, name, dims, values, description):
    variable = dataset.createVariable(name, "i1", dims)
    variable.description = description
    variable.flag_values = np.array([0, 1], dtype=np.int8)
    variable.flag_meanings = "no_observation valid"
    variable.coordinates = "XLONG XLAT"
    variable[:] = values


def _count_var(dataset, name, dims, values, description):
    variable = dataset.createVariable(name, "i4", dims)
    variable.description = description
    variable.units = "count"
    variable.coordinates = "XLONG XLAT"
    variable[:] = values


def _digest(array, dtype) -> str:
    """Digest an array with its dtype and shape, C-ordered.

    Same construction as :func:`gpuwm.obs.target_grid._array_sha256`, but
    at the dtype the file actually stores rather than always float64: the
    point here is to hash what is *on disk*, so that a reader can recompute
    it without the writer's inputs.
    """

    contiguous = np.ascontiguousarray(np.asarray(array), dtype=dtype)
    hasher = hashlib.sha256()
    hasher.update(contiguous.dtype.str.encode("ascii"))
    hasher.update(b";")
    hasher.update(json.dumps(list(contiguous.shape)).encode("ascii"))
    hasher.update(b";")
    hasher.update(contiguous.tobytes(order="C"))
    return hasher.hexdigest()


def _coordinate_digests(grid: TargetGrid) -> dict:
    """The per-array digests that bind a file's contents to its identity.

    Three of the four are stored in the file at float32 and are therefore
    recomputable by any reader.  ``z_w`` is not stored -- the schema is a
    horizontal product -- so its digest travels as the only handle a
    consumer has on the vertical structure the observations were placed
    against.  Checking it needs the consumer's own grid; see
    :func:`_require_grid`.
    """

    return {
        "XLAT": _digest(grid.lat, np.float32),
        "XLONG": _digest(grid.lon, np.float32),
        "HGT": _digest(grid.terrain_m, np.float32),
        "z_w": _digest(grid.z_w, np.float64),
    }


def _canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"),
                      allow_nan=False)


def _provenance_payload(observations: GriddedObservations,
                        params: SuperobParams,
                        extra: dict | None) -> dict:
    payload = {
        "superob_params": params.to_payload(),
        "volumes": list(observations.provenance),
        "counts": list(observations.counts),
        "radars": list(observations.radars),
        "dealiasing": dealiasing_statement(params),
        "fold_suspicion": list(observations.fold_suspicion),
    }
    # Present only when the unfolder ran.  A file written without it carries
    # the same provenance key set it always has -- which is what makes the
    # masking-only path byte-identical rather than merely equivalent -- and
    # a file written with it cannot omit the account of what was refused.
    dealias = list(getattr(observations, "dealias", ()) or ())
    if dealias:
        payload["dealias"] = dealias
    # Same rule for the dual-pol mask, and for the same reason.  The
    # per-radar list carries an empty entry for a radar whose mask never
    # ran, so the key appears only when at least one radar's did: a file
    # built without --cc-qc keeps the provenance key set it always had.
    cc_qc = list(getattr(observations, "cc_qc", ()) or ())
    if any(cc_qc):
        payload["cc_qc"] = cc_qc
    if extra:
        payload["context"] = extra
    return payload
