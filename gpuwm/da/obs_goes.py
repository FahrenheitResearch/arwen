"""Adapt one ``gpuwm-obs.goes-grid.v1`` file to the filter's batches.

The satellite twin of :mod:`gpuwm.da.obs_radar`, with one structural
difference that governs everything else here: **CWP is a column integral,
so there is exactly one observation per column.**

:class:`gpuwm.da.letkf.GriddedObs` wants ``(nz, ny, nx)``.  A column
observation is expressed in that shape by setting the mask True at one
level -- the ``obs_level`` the gridding stage recorded -- and False
everywhere else in the column.  It is *not* expressed by repeating the
value down the column: that would hand the filter ``nz`` copies of one
measurement, each with the full weight of an independent observation, and
the filter has no way to detect it.  A 50-level column would arrive
carrying 50x the information it has, and the analysis would follow it.

Two consequences the caller has to own, both recorded in the provenance
this module returns:

* **The vertical localisation radius is doing the column integration's
  work.**  The observation constrains the whole column but is centred at
  one level, so a radius that spans only part of the column silently
  assimilates a column integral as if it were a layer measurement.  This
  module measures the radius against the actual column depth and records
  the ratio; it does not refuse, because what fraction is right is a
  scoreboard question, not one this module gets to settle.
* **Suppression is the point, not a side effect.**  A clear-sky zero
  against a cloudy model is a large negative innovation, and with
  condensate in the analysis fields the ensemble covariance turns it into
  negative condensate increments across the localisation lens.  That is
  how obs-clear removes model cloud.  It only works if the lens reaches
  the levels where the model put the cloud -- which is the previous point,
  stated in the direction that matters most.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from gpuwm.da.letkf import GriddedObs, Localization
from gpuwm.da.obs_radar import _batch
from gpuwm.obs.goes_grid import (GOES_GRID_SCHEMA, read_goes_grid,
                                 require_goes_grid_binding)

#: Provenance stamp for the adaptation itself.
ADAPTER_SCHEMA = "gpuwm-da.goes-obs-adapter.v1"

#: Name given to the CWP batch.  One batch: the product is merged across
#: satellite pixels by the gridding stage, so unlike radial velocity there
#: is exactly one of it.
CWP_NAME = "cwp"


class GoesObsAdapterError(ValueError):
    """The file and the filter cannot be reconciled.  Never a warning."""


def read_document(source, *, expected_grid,
                  expected_grid_identity: str | None = None):
    """Accept a path or an already-read document, uniformly and bound.

    ``expected_grid`` is required for the reason
    :func:`gpuwm.da.obs_radar.read_document` gives -- the identity is a
    digest over four arrays of which the file stores three -- plus one
    that is specific to this product: ``obs_level`` is an index into
    ``z_w``, and ``z_w`` is the array the file does not carry.  A
    goes-grid file bound to the wrong vertical structure centres every
    column observation in the wrong layer, and no reader-side check can
    see it.
    """

    if expected_grid is None:
        raise GoesObsAdapterError(
            "expected_grid is required: obs_level indexes the grid's z_w, "
            "which is not in the file. Binding to an identity string alone "
            "cannot check the one array that matters here")
    if isinstance(source, (str, Path)):
        return read_goes_grid(source,
                              expected_grid_identity=expected_grid_identity,
                              expected_grid=expected_grid)
    if not isinstance(source, Mapping):
        raise GoesObsAdapterError(
            f"expected a path or a read_goes_grid document, got "
            f"{type(source).__name__}")
    if source.get("schema") != GOES_GRID_SCHEMA:
        raise GoesObsAdapterError(
            f"document declares schema {source.get('schema')!r}, this "
            f"adapter reads {GOES_GRID_SCHEMA!r}")
    demanded = expected_grid.identity_sha256()
    if (expected_grid_identity is not None
            and expected_grid_identity != demanded):
        raise GoesObsAdapterError(
            f"expected_grid hashes to {demanded} but expected_grid_identity "
            f"demands {expected_grid_identity}; the caller is asking for two "
            "different grids")
    require_goes_grid_binding(source, expected_grid, label="document")
    return source


def observation_shape(document, grid) -> tuple[int, int, int]:
    """``(nz, ny, nx)`` -- the model volume the batch is expressed in."""

    dims = document["dims"]
    shape = (int(dims["nz"]), int(dims["south_north"]),
             int(dims["west_east"]))
    expected = (int(grid.nz), int(grid.ny), int(grid.nx))
    if shape != expected:
        raise GoesObsAdapterError(
            f"the file describes a {shape} volume, the caller's grid is "
            f"{expected}")
    return shape


def column_class(document) -> np.ndarray:
    """``(ny, nx)`` phase class per observed column, for the operator.

    The operator composes model condensate under the *observation's* phase
    (see :mod:`gpuwm.da.obsop_cwp`), so this is how the class crosses from
    the file to H(x).  Unobserved columns carry
    :data:`gpuwm.obs.goes_cwp.CLASS_NONE`, which the operator integrates
    under the clear composition -- a complete field, never a hole.
    """

    return np.asarray(document["variables"]["cwp_class"], dtype=np.int8)


def expand_to_volume(document, shape, simulated):
    """Place the 2-D column product at its levels inside a 3-D volume.

    Returns ``(values, errors, mask, simulated3, placed)``.  ``placed`` is
    the ``(ny, nx)`` boolean of columns that became an observation, which
    is the file's own mask -- returned so a caller can count what it
    assimilated without re-deriving it from the volume.
    """

    nz, ny, nx = shape
    variables = document["variables"]
    mask2 = np.asarray(variables["cwp_mask"]).astype(bool)
    if mask2.shape != (ny, nx):
        raise GoesObsAdapterError(
            f"cwp_mask is {mask2.shape}, the grid is {(ny, nx)}")
    values2 = np.asarray(variables["cwp_obs"], dtype=np.float64)
    errors2 = np.asarray(variables["cwp_err"], dtype=np.float64)
    levels2 = np.asarray(variables["obs_level"], dtype=np.int64)

    simulated = np.asarray(simulated, dtype=np.float64)
    if simulated.ndim != 3 or simulated.shape[1:] != (ny, nx):
        raise GoesObsAdapterError(
            f"cwp_simulated must be (R, {ny}, {nx}) -- H(x) for a column "
            f"integral is a 2-D field per member -- got {simulated.shape}")
    members = simulated.shape[0]

    jj, ii = np.nonzero(mask2)
    kk = levels2[jj, ii]
    if jj.size and not np.all((kk >= 0) & (kk < nz)):
        raise GoesObsAdapterError(
            f"obs_level is outside 0..{nz - 1} where cwp_mask claims an "
            "observation; the file passed its own consistency check against "
            "a different nz than the caller's grid")

    values = np.zeros((nz, ny, nx), dtype=np.float64)
    # 1.0, not 0.0: errors under a False mask are never read, but a zero
    # standard deviation anywhere is one squeezed sign-check away from an
    # infinite weight, and this array is handed to code that squares it.
    errors = np.ones((nz, ny, nx), dtype=np.float64)
    mask = np.zeros((nz, ny, nx), dtype=bool)
    simulated3 = np.zeros((members, nz, ny, nx), dtype=np.float64)

    values[kk, jj, ii] = values2[jj, ii]
    errors[kk, jj, ii] = errors2[jj, ii]
    mask[kk, jj, ii] = True
    simulated3[:, kk, jj, ii] = simulated[:, jj, ii]
    return values, errors, mask, simulated3, mask2


def _vertical_reach(grid, localization: Localization) -> dict:
    """How much of the column the localisation radius actually reaches."""

    z_w = np.asarray(grid.z_w, dtype=np.float64)
    depth = z_w[-1] - z_w[0]
    radius = float(localization.vertical_m)
    return {
        "vertical_localization_m": radius,
        "median_column_depth_m": float(np.median(depth)),
        "fraction_of_median_column_reached": float(
            min(1.0, radius / max(float(np.median(depth)), 1.0))),
        "note": (
            "CWP is a column integral centred at one level. A vertical "
            "radius well below the column depth assimilates it as if it "
            "were a layer measurement, and in particular stops a clear-sky "
            "zero from reaching model cloud at other heights. This is "
            "recorded, not enforced: what fraction is right is a scoreboard "
            "question"),
    }


def goes_grid_to_gridded_obs(source, *, expected_grid, cwp_simulated,
                             expected_grid_identity: str | None = None,
                             localization: Localization | None = None,
                             error_inflation: float = 1.0,
                             ) -> tuple[list[GriddedObs], dict]:
    """Adapt one goes-grid file to the filter's observation batches.

    ``cwp_simulated`` is ``H(x_k)`` for CWP, ``(R, ny, nx)`` in g m-2 --
    two-dimensional per member, because the observable is a column
    integral.  :func:`gpuwm.da.obsop_cwp.simulated_cloud_water_path`
    returns exactly that shape.

    Returns ``(batches, provenance)``.
    """

    inflation = float(error_inflation)
    if not np.isfinite(inflation) or inflation < 1.0:
        raise GoesObsAdapterError(
            f"error_inflation must be finite and >= 1, got "
            f"{error_inflation!r}; deflating a stated observation error is a "
            "claim of skill nobody measured, and this one is UNCALIBRATED "
            "to begin with")
    document = read_document(source, expected_grid=expected_grid,
                             expected_grid_identity=expected_grid_identity)
    shape = observation_shape(document, expected_grid)
    values, errors, mask, simulated, placed = expand_to_volume(
        document, shape, cwp_simulated)
    errors = errors * inflation

    batch = _batch(CWP_NAME, values, errors, mask, simulated, localization,
                   shape)

    classes = column_class(document)
    counted = {
        "observed_columns": int(np.count_nonzero(placed)),
        "clear_sky_zeros": int(np.count_nonzero(placed & (classes == 0))),
        "liquid": int(np.count_nonzero(placed & (classes == 1))),
        "ice": int(np.count_nonzero(placed & (classes == 2))),
    }
    reach = None
    if localization is not None:
        reach = _vertical_reach(expected_grid, localization)

    provenance = {
        "schema": ADAPTER_SCHEMA,
        "stability": "experimental",
        "obs_schema": document.get("schema"),
        "obs_status": document.get("status"),
        "valid_time": document.get("valid_time"),
        "grid_identity_sha256": document.get("grid_identity_sha256"),
        "grid_coordinate_sha256": document.get("grid_coordinate_sha256"),
        "grid_binding": "full TargetGrid, including z_w",
        "grid_shape": list(shape),
        "batches": [{
            "name": CWP_NAME,
            "kind": "cloud_water_path",
            "units": "g m-2",
            "observable": "column integral; one observation per column",
            "placement": ("mask is True at obs_level only; the value is "
                          "never repeated down the column, which would be "
                          "one measurement counted nz times"),
            "observed_points": int(np.count_nonzero(mask)),
            **counted,
        }],
        "error_inflation": inflation,
        "error_model": document.get("error_model"),
        "join": document.get("join"),
        "dqf_policy": document.get("dqf_policy"),
        "superob_params": document.get("superob_params"),
        "vertical_reach": reach,
        "notes": [
            "errors are standard deviations, passed through unsquared",
            "the error model is UNCALIBRATED; see error_model.calibration",
            "the cross-grid cloud-top join is this consumer's recorded "
            "choice, not the bridge's; see join.method",
            "a clear-sky zero is a genuine observation of 0 g m-2 and is "
            "what lets the analysis remove model cloud",
        ],
    }
    return [batch], provenance
