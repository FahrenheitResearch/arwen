"""What the landmask/soil-category reconciler is allowed to read.

``dyn_em/module_initialize_real.F:3608-3650`` decides a column whose land
mask disagrees with its soil category from its soil temperature, then its
SST, and aborts on ``mismatch_landmask_ivgtyp`` with neither.  gpuwm
transcribes all three arms (tests/test_ruc_shoreline_soil_category.py); what
this file covers is the step BEFORE them -- finding the evidence in a met
field mapping whose spellings differ per source.

That lookup has failed twice, and both times the soil column was present in
the mapping the reconciler was handed:

* 2026-08-06, native HRRR: the chain did not know ``SOILT`` and a nested
  1 km preparation aborted on 38 shoreline columns;
* 2026-08-08, nested GFS: the chain did not know ``GFS_ST000010`` and a
  3 km child aborted on 73 inland-water columns, which made nested-GFS
  preparation impossible for essentially any CONUS child holding a
  reservoir or a river.

Both fixes were a name added to an inline chain at one call site.  These
tests hold the replacement -- ONE table in ``gpuwm/ingest/soil.py``, read by
every call site -- and, just as hard, hold the refusal: a mapping that
genuinely carries no evidence must still abort, because the third arm is
correct and fail-open here would fabricate a soil column.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from gpuwm.core.landuse import _top_soil_level, reconciled_soil_category
from gpuwm.ingest.soil import (SOIL_TEMPERATURE_RECONCILER_NAMES,
                               SST_RECONCILER_NAMES,
                               reconciler_soil_temperature, reconciler_sst)
from gpuwm.ingest.soil_contract import MAPPED_SOIL_TEMPERATURE

#: MODIS-with-lakes identity, as every real preparation runs it.
_ISWATER = 17
_ISLAKE = 21
_ISICE = 15
_LAND_VEGETATION = 12          # croplands, the live child's mismatch class
_WATER_SOIL = 14
_LAND_SOIL = 8                 # real.exe's "artificial silty clay loam"
_ATTRS = dict(iswater=_ISWATER, islake=_ISLAKE, isice=_ISICE)

#: The spelling and array rank each source hands the reconciler.  ERA5 and
#: GFS name their top layer directly (2-D); the mapped and native-HRRR lanes
#: hand over a stacked column whose FIRST level is the one WRF reads.
_SPELLINGS = {
    MAPPED_SOIL_TEMPERATURE: 4,
    "ST000007": 0,
    "GFS_ST000010": 0,
    "SOILT": 9,
}


def _profile(name, shape, value):
    """One source's soil-temperature field, at that source's rank."""

    levels = _SPELLINGS[name]
    if levels == 0:
        return np.full(shape, value, np.float32)
    # A deliberately COLDER stack below the top level: a lookup that read
    # the wrong level would resolve the column the other way.
    column = np.stack([np.full(shape, value - 40.0 * index, np.float32)
                       for index in range(levels)])
    return column


def _one_mismatch(shape=(3, 4)):
    """A land grid whose single interior column carries the water soil."""

    lu_index = np.full(shape, _LAND_VEGETATION, np.float32)
    soil_type = np.full(shape, 6, np.float32)
    soil_type[1, 2] = _WATER_SOIL
    return lu_index, soil_type


def _reconcile(fields, lu_index, soil_type):
    """The exact call both production call sites now make."""

    return reconciled_soil_category(
        lu_index, soil_type=soil_type, xice=fields.get("XICE", 0.0),
        soil_temperature=reconciler_soil_temperature(fields),
        sst=reconciler_sst(fields), **_ATTRS)


# ---------------------------------------------------------------------------
# The table itself
# ---------------------------------------------------------------------------

def test_the_table_carries_every_source_spelling_in_lookup_order():
    """One table, and it is the union of what the lanes actually write.

    Pinned by value rather than by count: a source added to the ingest but
    not to this table is the whole defect, and a length assertion would not
    say which name went missing.
    """

    assert SOIL_TEMPERATURE_RECONCILER_NAMES == (
        MAPPED_SOIL_TEMPERATURE,   # "RW_SOIL_TEMPERATURE"
        "ST000007",                # classic per-layer Vtable
        "GFS_ST000010",            # GFS per-layer
        "SOILT",                   # native HRRR stacked nodes
    )
    # The two per-layer entries are the TOP layer of the lane's own tuple,
    # not a second transcription of it that can drift.
    from gpuwm.ingest.soil import _GFS_TEMP_NAMES, _TEMP_NAMES

    assert SOIL_TEMPERATURE_RECONCILER_NAMES[1] == _TEMP_NAMES[0]
    assert SOIL_TEMPERATURE_RECONCILER_NAMES[2] == _GFS_TEMP_NAMES[0]


def test_the_sst_table_is_real_exes_own_precedence():
    """SST first, then the skin temperature it substitutes.

    ``module_initialize_real.F:2844-2866`` keeps exactly that column's
    SKINTEMP wherever SST has no valid support, so the mismatch pass reads
    a skin temperature there.  ``TSK`` is the same quantity under the
    prepared/physics inventory's spelling
    (``gpuwm/ingest/hrrr_physics.py``), so one call serves either mapping.
    """

    assert SST_RECONCILER_NAMES == ("SST", "SKINTEMP", "TSK")


@pytest.mark.parametrize("name", list(_SPELLINGS))
def test_every_spelling_resolves_the_disagreeing_column(name):
    """The live failure, once per source, at that source's own rank."""

    lu_index, soil_type = _one_mismatch()
    fields = {name: _profile(name, lu_index.shape, 288.0),
              "SKINTEMP": np.full(lu_index.shape, 291.0, np.float32)}

    assert reconciler_soil_temperature(fields) is not None
    resolved = _reconcile(fields, lu_index, soil_type)

    # The warm-soil arm: land, and WRF's artificial silty clay loam.
    assert int(resolved[1, 2]) == _LAND_SOIL
    # Nothing else moved.
    untouched = np.full(lu_index.shape, 6, np.int32)
    untouched[1, 2] = _LAND_SOIL
    np.testing.assert_array_equal(resolved, untouched)


def test_a_mapping_with_no_evidence_at_all_still_refuses():
    """The third arm is CORRECT; the fix must not fail open.

    A mapping carrying none of the spellings has nothing to decide the
    column with, and WRF aborts rather than guessing.  So does this.
    """

    lu_index, soil_type = _one_mismatch()
    fields = {"LANDSEA": np.ones(lu_index.shape, np.float32),
              "SNOW": np.zeros(lu_index.shape, np.float32)}

    assert reconciler_soil_temperature(fields) is None
    assert reconciler_sst(fields) is None
    with pytest.raises(ValueError, match="mismatch_landmask_ivgtyp"):
        _reconcile(fields, lu_index, soil_type)


def test_a_cold_soil_column_with_no_sea_evidence_still_refuses():
    """Present-but-cold is not the same state as absent, and neither passes.

    The soil temperature is found, read, and rejected by WRF's own > 1 K
    test; with no sea evidence behind it the column is still unresolvable.
    """

    lu_index, soil_type = _one_mismatch()
    fields = {"GFS_ST000010": np.zeros(lu_index.shape, np.float32)}

    assert reconciler_soil_temperature(fields) is not None
    with pytest.raises(ValueError, match="mismatch_landmask_ivgtyp"):
        _reconcile(fields, lu_index, soil_type)


# ---------------------------------------------------------------------------
# The SST fallback
# ---------------------------------------------------------------------------

def test_the_sst_fallback_prefers_a_real_sst_then_skin_then_tsk():
    """A name speaks only when every name before it is absent."""

    shape = (2, 2)
    sst = np.full(shape, 275.0, np.float32)
    skintemp = np.full(shape, 285.0, np.float32)
    tsk = np.full(shape, 295.0, np.float32)

    everything = {"SST": sst, "SKINTEMP": skintemp, "TSK": tsk}
    np.testing.assert_array_equal(reconciler_sst(everything), sst)
    np.testing.assert_array_equal(
        reconciler_sst({"SKINTEMP": skintemp, "TSK": tsk}), skintemp)
    np.testing.assert_array_equal(reconciler_sst({"TSK": tsk}), tsk)
    assert reconciler_sst({"T2": tsk}) is None


def test_the_skin_fallback_resolves_the_column_real_exe_calls_water():
    """The GFS lane has no SST field at all, and this is what that costs.

    Without the fallback the SST arm cannot fire for any GFS child, so a
    cold-soil mismatch column has no second chance and the preparation
    aborts.  With it, the column becomes water exactly as real.exe leaves
    it once SKINTEMP has stood in for the missing SST.
    """

    lu_index, soil_type = _one_mismatch()
    fields = {"GFS_ST000010": np.zeros(lu_index.shape, np.float32),
              "SKINTEMP": np.full(lu_index.shape, 290.0, np.float32)}

    resolved = _reconcile(fields, lu_index, soil_type)
    assert int(resolved[1, 2]) == _WATER_SOIL


# ---------------------------------------------------------------------------
# The live shape
# ---------------------------------------------------------------------------

def test_the_child_shape_that_aborted_on_seventy_three_columns():
    """The refusal as it was observed, and the same grid after the fix.

    Shape, count and field inventory are the probe's dump of what
    ``finalize_prepared_child`` actually handed the reconciler on the
    12-3 km nested-GFS tree: a 160x192 child, 73 disagreeing columns
    scattered across the whole grid (reservoirs and rivers, not an edge
    artifact), and a mapping whose only soil temperature is spelled
    ``GFS_ST000010``.
    """

    shape = (160, 192)
    lu_index = np.full(shape, _LAND_VEGETATION, np.float32)
    soil_type = np.full(shape, 6, np.float32)
    rng = np.random.default_rng(73)
    flat = rng.choice(shape[0] * shape[1], size=73, replace=False)
    rows, columns = np.unravel_index(flat, shape)
    soil_type[rows, columns] = _WATER_SOIL

    fields = {
        "GFS_SM000010": np.full(shape, 0.3, np.float32),
        "GFS_ST000010": np.full(shape, 289.0, np.float32),
        "GHT": np.zeros(shape, np.float32),
        "LANDSEA": np.ones(shape, np.float32),
        "PSFC": np.full(shape, 98000.0, np.float32),
        "SKINTEMP": np.full(shape, 292.0, np.float32),
        "SNOW": np.zeros(shape, np.float32),
        "T2": np.full(shape, 295.0, np.float32),
        "XICE": np.zeros(shape, np.float32),
    }
    # No SST key at all, exactly as the GFS lane leaves it.
    assert "SST" not in fields

    # As it was: neither piece of evidence found, and the count in the
    # message is the count of columns.
    with pytest.raises(ValueError, match=r"73 column\(s\) disagree"):
        reconciled_soil_category(
            lu_index, soil_type=soil_type, xice=fields["XICE"],
            soil_temperature=None, sst=None, **_ATTRS)

    resolved = _reconcile(fields, lu_index, soil_type)
    np.testing.assert_array_equal(
        resolved[rows, columns], np.full(73, _LAND_SOIL, np.int32))
    assert int((resolved == _WATER_SOIL).sum()) == 0


# ---------------------------------------------------------------------------
# Device residency
# ---------------------------------------------------------------------------

class _DeviceArrayDouble:
    """A LOUD stand-in for ``cupy.ndarray``, host-side.

    Reproduces the only two behaviours this module's marshalling depends
    on -- ``__module__`` naming cupy, and ``np.asarray`` refusing with
    CuPy's own TypeError while ``.get()`` succeeds -- so the contract is
    tested on a machine whose card is busy, and in CI with no card at all.
    Anything else about it will fail loudly rather than pass quietly.
    """

    def __init__(self, host):
        self._host = np.asarray(host)

    def __array__(self, *args, **kwargs):
        raise TypeError("Implicit conversion to a NumPy array is not allowed. "
                        "Please use `.get()` to construct a NumPy array "
                        "explicitly.")

    def get(self):
        return self._host


# The check under test is ``type(value).__module__``, so the double must
# claim cupy's module the way a real device array does.
_DeviceArrayDouble.__module__ = "cupy"


def test_the_double_refuses_host_conversion_like_the_real_thing():
    """Validate the instrument before trusting what it proves."""

    double = _DeviceArrayDouble(np.zeros((2, 2), np.float32))
    with pytest.raises(TypeError, match="Implicit conversion"):
        np.asarray(double, dtype=np.float64)
    assert double.get().shape == (2, 2)


def test_the_top_soil_read_accepts_a_device_resident_column():
    """``_top_soil_level`` is reached ONLY on a grid that has a mismatch.

    So a bare ``np.asarray`` here does not fail at setup on a device-backed
    lane -- it fails deep inside preparation, on exactly the grids that
    needed the reconciliation, and never on the ones that did not.
    """

    shape = (3, 4)
    stacked = np.stack([np.full(shape, 290.0 - 40.0 * level, np.float32)
                        for level in range(4)])

    top = _top_soil_level(_DeviceArrayDouble(stacked), shape,
                          "soil_temperature")
    np.testing.assert_array_equal(top, stacked[0])
    # A 2-D single-layer field (the GFS/ERA5 rank) takes the same path.
    plane = np.full(shape, 288.0, np.float32)
    np.testing.assert_array_equal(
        _top_soil_level(_DeviceArrayDouble(plane), shape, "soil_temperature"),
        plane)
    assert _top_soil_level(None, shape, "soil_temperature") is None


def test_the_reconciler_takes_device_evidence_through_initialize_landuse():
    """The OTHER entry point, which does not marshal its arguments.

    ``reconciled_soil_category`` hosts everything it is handed on the way
    in; ``initialize_landuse`` does not, and its callers pass host statics
    beside met fields that can still be device-resident.  Both pieces of
    evidence are read inside the reconciliation, so both take the
    marshalling -- the SST read sits one line below the soil read, and
    fixing only the first moves the TypeError rather than removing it.
    """

    from datetime import datetime

    from gpuwm.core.landuse import initialize_landuse

    shape = (1, 5)
    lu_index = np.full(shape, _LAND_VEGETATION, np.int32)
    soil_type = np.array([[_WATER_SOIL, 6, 4, 11, 6]], np.int32)
    landuse = initialize_landuse(
        lu_index, soil_type=soil_type,
        landmask=np.ones(shape), snow=0.0, xice=0.0,
        valid_time=datetime(2026, 8, 8, 6), cen_lat=35.0,
        mminlu="MODIFIED_IGBP_MODIS_NOAH", **_ATTRS,
        soil_temperature=_DeviceArrayDouble(
            np.zeros((4,) + shape, np.float32)),
        sst=_DeviceArrayDouble(np.full(shape, 290.0, np.float32)))

    # Cold soil, warm sea: WRF's second arm, on device-resident inputs.
    np.testing.assert_array_equal(
        landuse.ivgtyp,
        [[_ISWATER, _LAND_VEGETATION, _LAND_VEGETATION, _LAND_VEGETATION,
          _LAND_VEGETATION]])
    np.testing.assert_array_equal(landuse.isltyp, [[14, 6, 4, 11, 6]])


def test_a_device_resident_mismatch_resolves_to_the_host_answer():
    """Both pieces of evidence arrive on the device, and both are read."""

    lu_index, soil_type = _one_mismatch()
    warm = np.full(lu_index.shape, 288.0, np.float32)
    sea = np.full(lu_index.shape, 291.0, np.float32)

    on_host = reconciled_soil_category(
        lu_index, soil_type=soil_type, xice=0.0,
        soil_temperature=warm, sst=sea, **_ATTRS)
    on_device = reconciled_soil_category(
        lu_index, soil_type=soil_type, xice=0.0,
        soil_temperature=_DeviceArrayDouble(warm),
        sst=_DeviceArrayDouble(sea), **_ATTRS)

    assert isinstance(on_device, np.ndarray)
    np.testing.assert_array_equal(on_host, on_device)
    assert int(on_device[1, 2]) == _LAND_SOIL


# ---------------------------------------------------------------------------
# The call sites
# ---------------------------------------------------------------------------

def test_no_call_site_spells_the_evidence_inline():
    """The bug class, not the two instances of it.

    Both reconciler call sites -- the root case and the nested child --
    must read the table.  An inline ``fields.get("ST000007")`` chain
    growing back at either one is how this defect reached a shipped
    release twice, so it is refused here by name.
    """

    from gpuwm.ingest.nest_init import finalize_prepared_child
    from gpuwm.runtime import prepare_real_case

    for function in (finalize_prepared_child, prepare_real_case):
        source = inspect.getsource(function)
        where = function.__qualname__
        assert "reconciled_soil_category(" in source, where
        assert "soil_temperature=reconciler_soil_temperature(" in source, where
        assert "sst=reconciler_sst(" in source, where
        for name in SOIL_TEMPERATURE_RECONCILER_NAMES:
            assert f'"{name}"' not in source, (
                f"{where} spells {name} inline instead of reading the table")
