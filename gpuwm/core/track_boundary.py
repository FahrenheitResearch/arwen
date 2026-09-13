"""Detect a tracked centre that is no longer enclosed by its parent grid."""
from __future__ import annotations

import numpy as np


def boundary_reason(plane, box, center, *, extremum: str,
                    radius_cells=None, signal: str | None = None) -> str | None:
    """A tracked extremum at the actual array edge, not a clipped window edge.

    This is a conservative diagnostic limit: it does not infer a position
    outside the model. Tied extrema with any interior cell, missing data, flat
    fields and an extremum on an interior search-window edge do not prove exit.

    ``extremum`` NAMES WHICH END OF THE FIELD IS THE CENTRE, and it has no
    default on purpose. A cyclone is a pressure MINIMUM, rotation and echo
    are MAXIMA, and a tracked attribute is whichever its own configuration
    says -- so a caller that did not state the sense would be asking whether
    the field's weakest cell had reached the edge, which for every maximum
    signal is a different question with the same answer shape: silently
    wrong, and wrong in the direction that ends a live track. The one caller
    reads the sense from :func:`gpuwm.core.storm_tracking.extremum_kind`,
    which is also what the receipt's ``extremum_kind`` is written from, so
    the row and this test cannot disagree about what was being tracked.
    """
    if extremum not in ("minimum", "maximum"):
        raise ValueError(
            "boundary_reason needs extremum='minimum' or 'maximum'; it is "
            "which end of the tracked field is the centre, and there is no "
            "sense to assume for a field whose configuration states one")
    values = np.asarray(plane)
    if values.ndim != 2 or min(values.shape) < 3:
        return None
    ny, nx = values.shape
    js, is_ = box
    window = values[js, is_]
    if window.size == 0:
        return None
    jj, ii = np.mgrid[js.start:js.stop, is_.start:is_.stop]
    mask = np.isfinite(window)
    if radius_cells is not None:
        ci, cj = center
        mask &= (ii - ci) ** 2 + (jj - cj) ** 2 <= float(radius_cells) ** 2
    finite = window[mask]
    if finite.size < 2 or float(finite.max()) == float(finite.min()):
        return None
    peak = float(finite.min()) if extremum == "minimum" else float(finite.max())
    extrema = mask & (window == peak)
    boundary = (ii == 0) | (ii == nx - 1) | (jj == 0) | (jj == ny - 1)
    if np.any(extrema) and np.all(boundary[extrema]):
        return (f"tracked {signal or 'signal'} {extremum} reached the "
                "parent-domain boundary; an enclosed center is no longer "
                "resolved")
    return None
