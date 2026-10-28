"""Select analyzed initial vegetation with the existing climatology fallback."""
import sys

import numpy as np

#: How far outside [0, 100] % an analyzed value may sit and still be the
#: analysis itself.  A cycle whose fully vegetated cells publish exactly
#: 100 % (HRRR 2025-05-19 18Z: 86 cells) interpolates to a few units in the
#: last place above 100 wherever the weights do not sum to one exactly; such
#: a value is the same fraction and is clipped, not replaced.
_ROUNDING_PERCENT = 1.0e-3


def _climatology(static, valid_time):
    from gpuwm.static.build import monthly_interp_to_date
    return 100.0 * monthly_interp_to_date(static["GREENFRAC"], valid_time)


def initial_vegetation_fraction(met, static, valid_time):
    """Use the analyzed fraction in percent when the source carries one.

    Cells the analysis leaves non-finite or outside 0-100 % (missing
    points, a sentinel, a donor that does not cover the cell) take the
    GREENFRAC monthly climatology at ``valid_time``, the value every start
    read before analyzed vegetation was retained, cell by cell; the rest of
    the field keeps the analysis.  A field on another grid is refused: it
    describes a different area, not missing cells of this one.
    """
    field = met.fields.get("VEGFRA")
    if field is None:
        return _climatology(static, valid_time)
    host = field.get() if hasattr(field, "get") else field
    values = np.asarray(host)
    if values.shape != np.shape(static["LANDMASK"]):
        raise ValueError(
            f"analyzed VEGFRA is on a {values.shape} grid, not the mass grid "
            f"{np.shape(static['LANDMASK'])}; it would initialize a different "
            "vegetated area")
    finite = np.isfinite(values)
    if finite.all() and not (values < 0.0).any() and not (values > 100.0).any():
        return field
    with np.errstate(invalid="ignore"):
        analyzed = (finite & (values >= -_ROUNDING_PERCENT)
                    & (values <= 100.0 + _ROUNDING_PERCENT))
    rounded = int(np.count_nonzero(
        analyzed & ((values < 0.0) | (values > 100.0))))
    replaced = int(values.size - np.count_nonzero(analyzed))
    climatology = np.broadcast_to(_climatology(static, valid_time), values.shape)
    repaired = np.where(analyzed, np.clip(np.where(finite, values, 0.0), 0.0, 100.0),
                        climatology).astype(values.dtype, copy=False)
    parts = []
    if rounded:
        parts.append(f"{rounded} cell(s) within {_ROUNDING_PERCENT:g} % of the "
                     "0-100 % range clipped to it")
    if replaced:
        land = np.asarray(static["LANDMASK"]) > 0.5
        on_land = int(np.count_nonzero(~analyzed & land))
        parts.append(f"{replaced} missing or out-of-range cell(s) ({on_land} on "
                     "land) take the GREENFRAC monthly climatology")
    print(f"warning: analyzed VEGFRA ({values.size} cells): " + "; ".join(parts),
          file=sys.stderr, flush=True)
    if hasattr(field, "get"):
        import cupy as cp
        return cp.asarray(repaired)
    return repaired
