"""Every front door hands the land-surface ingest the RECONCILED soil category.

The GFS+RUC route refusal was retired, and the death it pre-empted -- a
shoreline land column carrying the water soil category (14) under a land
``LU_INDEX`` reaching RUC's ``soilvegin`` and evaluating ``0./0.`` into
MAVAIL on the first surface call -- had a root-cause fix
(``gpuwm.core.landuse.reconciled_soil_category``, real.exe's
``module_initialize_real.F:3608-3650``) wired into the ERA5 config door and
the nested child only.  The GFS and mapped doors still handed the raw
``SCT_DOM`` to ``preprocess_land_surface_soil`` (ENG-009).  One assembly of
the reconciler's arguments now serves all four callers.
"""

from __future__ import annotations

import inspect

import numpy as np

from gpuwm.ingest.soil import (SOIL_TEMPERATURE_RECONCILER_NAMES,
                               door_reconciled_soil_category)

#: MODIS-with-lakes identity, as the doors read it off the land-use table.
_ATTRS = {"ISWATER": 17, "ISLAKE": 21, "ISICE": 15}
_CROPLAND = 12
_WATER_SOIL = 14


def _shoreline():
    """Two land columns with the water soil category, one water column."""
    lu = np.array([[_CROPLAND, _CROPLAND, 17]], dtype=np.int32)
    sct = np.array([[_WATER_SOIL, 3, _WATER_SOIL]], dtype=np.int32)
    static = {"LU_INDEX": lu, "SCT_DOM": sct}
    # The GFS per-layer soil-temperature spelling is what the door decodes.
    fields = {SOIL_TEMPERATURE_RECONCILER_NAMES[2]: np.full((1, 3), 290.0, np.float32),
              "SKINTEMP": np.full((1, 3), 288.0, np.float32)}
    return static, fields


def test_a_land_column_with_the_water_soil_category_is_reconciled_to_land():
    static, fields = _shoreline()
    out = np.asarray(door_reconciled_soil_category(static, fields, _ATTRS))
    assert out.shape == (1, 3)
    assert out[0, 0] != _WATER_SOIL, "the shoreline column reached RUC as water soil"
    assert out[0, 1] == 3, "an agreeing land column is left alone"
    assert out[0, 2] == _WATER_SOIL, "a water column keeps the water soil category"
    # Idempotent: the reconciled field has no mismatching column left.
    again = door_reconciled_soil_category({**static, "SCT_DOM": out}, fields, _ATTRS)
    assert np.array_equal(np.asarray(again), out)


def test_without_a_land_use_table_the_raw_category_is_returned_and_said(capsys):
    static, fields = _shoreline()
    out = door_reconciled_soil_category(static, fields, None, route="GFS")
    assert out is static["SCT_DOM"]
    err = capsys.readouterr().err
    assert "not reconciled" in err and "--geog-root" in err
    door_reconciled_soil_category(static, fields, None)
    assert capsys.readouterr().err == ""


def _soil_initializer_calls(source: str):
    """The argument text of every ``preprocess_land_surface_soil(`` call."""
    for chunk in source.split("preprocess_land_surface_soil(")[1:]:
        depth, end = 1, 0
        for end, char in enumerate(chunk):
            depth += (char == "(") - (char == ")")
            if depth == 0:
                break
        yield chunk[:end]


def test_every_door_routes_the_soil_ingest_through_the_reconciled_category():
    """Source-level pin: no door hands the raw SCT_DOM to the SOIL initializer.

    ``initialize_landuse`` legitimately takes the raw category -- the
    reconciliation lives inside it and is idempotent -- so the pin is on the
    ``preprocess_land_surface_soil`` calls, which is where RUC's soil column
    is built and where the raw category did the damage.
    """
    from gpuwm import gfs_direct, mapped_direct, runtime
    from gpuwm.ingest import nest_init
    for module in (gfs_direct, mapped_direct, runtime, nest_init):
        source = inspect.getsource(module)
        assert "door_reconciled_soil_category(" in source, module.__name__
        calls = list(_soil_initializer_calls(source))
        assert calls, module.__name__
        for arguments in calls:
            assert 'soil_type=static["SCT_DOM"]' not in arguments, module.__name__
            assert 'soil_type=static_fields["SCT_DOM"]' not in arguments, module.__name__
