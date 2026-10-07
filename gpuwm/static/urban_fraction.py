"""Urban fraction from mapped land-cover area, computed in Rust.

The source classes provide area fractions and urban types, not a measured
imperviousness plane. The selected URBPARM rows provide each class's built
fraction. Rust forms their weighted sum and keeps WRF's dominant-urban
mask. Python selects table rows and carries buffers through the bridge.
"""
from __future__ import annotations

import ctypes
import json

URBAN_FRACTION_ALGORITHM = "landcover-area-urbparm-weighted-v1"
NLCD_FRACTION_ALGORITHM = "nlcd-impervious-class-midpoint-area-v1"
# MRLC's class ranges, expressed as fractions: <20%, 20-49%, 50-79%,
# 80-100%. These are class-midpoint estimates, not measured imperviousness.
NLCD_IMPERVIOUS_CLASS_MIDPOINTS = {21: 0.10, 22: 0.345, 23: 0.645, 24: 0.90}


def estimate_urban_fraction(fields, *, option: int, use_wudapt_lcz: int,
                            source_fractions=None, source_weight=None):
    """Return the FRC_URB2D plane and its estimate receipt.

    This operation has no Python numerical fallback. A bridge predating
    this operation cannot silently prepare canopy inputs from table-only
    fractions while the configuration requests area-weighted fractions.
    """
    from gpuwm.core.urban_tables import load_urban_params, urban_category_set
    from . import rust_bridge as bridge
    from .highres import _fieldset_new, MODIS21_ISURBAN

    library = bridge.load()
    if source_fractions is not None:
        try:
            capability = library.gpuwm_static_highres_urban_fraction_source_v1
        except AttributeError:
            raise bridge.StaticBridgeError(
                "the static-fields library predates source-class urban fractions: "
                "rebuild it to avoid silently using URBPARM table fractions "
                "instead of the requested land-cover imperviousness estimate") from None
        capability.argtypes = []
        capability.restype = ctypes.c_uint32
        if capability() != 1:
            raise bridge.StaticBridgeError(
                "the static-fields source-class urban-fraction contract is incompatible")
    try:
        entry = library.gpuwm_static_highres_urban_fraction
    except AttributeError:
        raise bridge.StaticBridgeError(
            "the static-fields library predates land-cover urban fractions: "
            "rebuild it to avoid silently replacing mixed-cell built area "
            "with a whole-cell table fraction") from None
    categories = urban_category_set(isurban=MODIS21_ISURBAN)
    params = load_urban_params(option, use_wudapt_lcz)
    lookup = categories.utype_lookup(use_wudapt_lcz)
    weights = [(category, float(params.FRC_URB_TBL[int(lookup[category]) - 1]))
               for category in categories.urban
               if 0 < int(lookup[category]) <= len(params.FRC_URB_TBL)]
    request_values = {"category_built_fractions": weights}
    source_fields = {name: fields[name]
                     for name in ("LANDUSEF", "LU_INDEX", "LANDMASK")}
    if source_fractions is not None:
        if source_weight is None:
            raise ValueError("source fractions need their land-cover coverage weight")
        source_fields["URBAN_SOURCEF"] = source_fractions
        source_fields["URBAN_SOURCE_WEIGHT"] = source_weight
        request_values.update({
            "source_category_built_fractions": [
                (n, value) for n, value in
                enumerate(NLCD_IMPERVIOUS_CLASS_MIDPOINTS.values(), start=2)],
            "source_algorithm": NLCD_FRACTION_ALGORITHM,
            "source_interpretation": "Estimate: NLCD developed-class imperviousness range midpoints, area averaged on dominant urban land cells; not observed imperviousness",
        })
    elif source_weight is not None:
        raise ValueError("source coverage weight needs source fractions")
    request = json.dumps(request_values).encode("utf-8")
    payload = (ctypes.c_uint8 * len(request)).from_buffer_copy(request)
    entry.argtypes = [ctypes.c_uint64, ctypes.POINTER(ctypes.c_uint8),
                      ctypes.c_size_t, ctypes.POINTER(ctypes.c_uint64)]
    entry.restype = ctypes.c_int32
    source = _fieldset_new(bridge, source_fields)
    output = ctypes.c_uint64(0)
    try:
        code = entry(source, payload, len(request), ctypes.byref(output))
        if code != 0:
            raise bridge.StaticBridgeError(bridge.last_error(library))
        values = bridge.fieldset_to_dict(output.value)
        audit = bridge.highres_audit_json(output.value)
        expected_algorithm = request_values.get("source_algorithm", URBAN_FRACTION_ALGORITHM)
        if audit.get("algorithm") != expected_algorithm:
            raise bridge.StaticBridgeError(
                f"urban fraction returned algorithm {audit.get('algorithm')!r}, "
                f"expected {expected_algorithm!r}: the built area was computed "
                "with different inputs than the preparation identity records")
        return values["FRC_URB2D"], audit
    finally:
        bridge.fieldset_free(source)
        if output.value:
            bridge.highres_audit_drop(output.value)
            bridge.fieldset_free(output.value)
