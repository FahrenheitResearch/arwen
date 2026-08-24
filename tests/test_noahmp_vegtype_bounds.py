"""A land-use class the vegetation table does not cover is a named refusal.

WHY THIS FILE EXISTS
--------------------
Driving a generated MPAS mesh through the dycore on node-2's 5090 stopped in
`noahmp_cold_start` with:

    File "gpuwm/core/noahmp_runtime.py", line 863, in veg_value
      return float(entry[int(vegtyp) - 1] if len(entry) > 1 else entry[0])
    IndexError: tuple index out of range

The mesh refines to 15 km over 41.5N 88W, which resolves the Great Lakes. The
geography archive on that box is `modis_landuse_20class_30s_with_lakes`, whose
lake class is 21. `MODIFIED_IGBP_MODIS_NOAH` declares NVEG = 20. So 115 cells
of the native static (1,074 of the Rust static) carried `ivgtyp = 21` and the
cold start indexed one past the end of a 20-entry tuple.

This is not a property of that mesh. It is reachable by anyone who refines
over a lake with that archive, and the published x1/x4 meshes hide it only
because they are too coarse to resolve one -- both top out at ivgtyp 19.

`IndexError: tuple index out of range` names nothing: not the class, not the
table, not the cells, not the way through. A refusal that does not name the
breakage it prevents is a defect in its own right.
"""
from __future__ import annotations

import pytest

from gpuwm.core import noahmp_runtime


@pytest.fixture(scope="module")
def params():
    return noahmp_runtime.NoahmpRuntimeParameters()


def test_a_covered_class_still_reads_its_row(params):
    """The guard must not cost the working case anything."""
    value = params.veg_value("SLA", 1)
    assert isinstance(value, float)
    # And the last covered class reads, which is where an off-by-one would
    # show up first.
    assert isinstance(params.veg_value("SLA", params.n_vegetation_classes), float)


def test_a_class_past_the_table_is_refused_by_name(params):
    """The exact failure a 15 km mesh over the Great Lakes produced."""
    nveg = params.n_vegetation_classes
    with pytest.raises(ValueError) as excinfo:
        params.veg_value("SLA", nveg + 1)
    message = str(excinfo.value)
    # It names the class that was asked for, the table that does not have it,
    # and how many classes the table does have.
    assert str(nveg + 1) in message
    assert str(nveg) in message
    assert "SLA" in message
    assert params.dataset_identifier in message
    # And it names the concrete breakage, not just the bounds.
    assert "land use" in message.lower() or "land-use" in message.lower()


def test_a_zero_or_negative_class_is_refused_too(params):
    """`vegtyp - 1` makes 0 index the LAST row and -1 the second last.

    Fortran is 1-based and this table is indexed `entry[vegtyp - 1]`, so a 0
    does not raise -- it silently reads the wrong row, which is worse than a
    crash because the run completes and the numbers are wrong.
    """
    for bad in (0, -1):
        with pytest.raises(ValueError) as excinfo:
            params.veg_value("SLA", bad)
        assert str(bad) in str(excinfo.value)


def test_the_class_count_is_read_from_the_table_not_hardcoded(params):
    """NVEG comes from the loaded table, so a different dataset moves it.

    A hardcoded 20 would make this guard wrong the day someone selects USGS,
    which has 24 classes.
    """
    assert params.n_vegetation_classes == int(params._categories.scalar("NVEG"))
    assert params.n_vegetation_classes >= 1
