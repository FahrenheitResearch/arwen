"""The tiles real-case preparers assemble their water temperature.

The 2.0 battery's twin_a row: every streamed real-case run on an ERA5
source carrying SST was refused at prepare, because
``tilestream/realcase.py`` handed soil preprocessing a raw SST beside its
SKINTEMP with no assembled water temperature -- the exact configuration
task #168 made a refusal (``gpuwm.ingest.water_temperature.
require_assembled_water_temperature``).  The mainline routes adopted
``assemble_for_route``; the tiles route did not.

Two halves, matching the shape of the mainline gate in
``tests/test_water_temperature.py``:

RECEIPT-IMPLIES-CONSUMPTION, structurally -- every soil-router call in
``tilestream/realcase.py`` forwards ``water_temperature`` and names its
``route``, checked by AST so a revert on EITHER preparer goes red.

THE ASSEMBLY ITSELF, functionally -- ``realcase.assembled_water_
temperature`` on a tiny synthetic domain produces a field the soil
router's refusal accepts, chooses per-body providers rather than the
per-cell fuse, and the refusal control shows what the fix closed.

CPU only: the assembly is host NumPy end to end.
"""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REALCASE = (Path(__file__).resolve().parents[1]
             / "tilestream" / "realcase.py")


# ---------------------------------------------------------------------------
# Structural: assembles implies forwards, on both preparers at once
# ---------------------------------------------------------------------------

def test_every_soil_router_call_in_realcase_forwards_the_assembly():
    """The false-receipt gate, extended to the tiles module.

    The mainline version of this gate (tests/test_water_temperature.py)
    scans the ``gpuwm`` package only, which is exactly why the tiles
    route could regress unseen: ``tilestream`` was outside its scope.
    Same bar here -- a module that assembles a water temperature must
    forward the result into every soil-router call it makes and must
    name its route while doing it.
    """
    text = _REALCASE.read_text(encoding="utf-8")
    assert "assemble_for_route" in text, (
        "tilestream/realcase.py no longer assembles a water temperature; "
        "its ERA5 sources carry SST and the soil router will refuse them")
    calls = [node for node in ast.walk(ast.parse(text))
             if isinstance(node, ast.Call)
             and (getattr(node.func, "id", None)
                  or getattr(node.func, "attr", None))
             == "preprocess_land_surface_soil"]
    assert len(calls) >= 2, (
        "both tiles preparers (low-water and slabbed) route soil through "
        "preprocess_land_surface_soil; finding fewer means a preparer "
        "was rewired around the one seam this gate watches")
    for node in calls:
        keywords = {keyword.arg for keyword in node.keywords}
        assert "water_temperature" in keywords, (
            f"tilestream/realcase.py:{node.lineno} calls the soil router "
            "without forwarding the assembled water temperature; that is "
            "the twin_a refusal (or worse, the silent per-cell fuse)")
        assert "route" in keywords, (
            f"tilestream/realcase.py:{node.lineno} calls the soil router "
            "without naming its route, so a missing assembly could not "
            "be attributed")


# ---------------------------------------------------------------------------
# Functional: the assembly on a synthetic domain, and its refusal control
# ---------------------------------------------------------------------------

_MODIS_LAKE = 21
_ATTRS = {"ISWATER": 17, "ISLAKE": _MODIS_LAKE, "ISICE": 15,
          "MMINLU": "MODIS"}


def _domain():
    """4x6, one ocean strip and one inland lake, MODIS categories."""
    landmask = np.ones((4, 6))
    landmask[:, :2] = 0.0                      # ocean, columns 0-1
    landmask[1:3, 4] = 0.0                     # a 2-cell inland lake
    lu_index = np.where(landmask >= 0.5, 10.0, 17.0)
    lu_index[1:3, 4] = float(_MODIS_LAKE)
    static = {"LANDMASK": landmask, "LU_INDEX": lu_index}
    lat, lon = np.meshgrid(np.linspace(40.0, 41.5, 4),
                           np.linspace(-90.0, -87.5, 6), indexing="ij")
    fields = {
        # Mapped SST: warm and finite over the ocean, garbage over the
        # lake (the mainline defect: a coarse SST has no lake support).
        "SST": np.where(landmask >= 0.5, 285.0, 290.0),
        "SKINTEMP": np.full((4, 6), 288.0),
    }
    fields["SST"][1:3, 4] = 150.0              # inadmissible over the lake
    # ERA5-shaped source: 1-D regular coordinate axes, like the
    # HorizontalSnapshot the real preparers hand this seam.
    source = SimpleNamespace(
        fields={"SST": np.full((7, 9), 290.0)},
        latitude=np.linspace(39.0, 42.0, 7),
        longitude=np.linspace(-91.0, -87.0, 9))
    data = SimpleNamespace(water_temperature_policy=None,
                           water_temperature_overlay=None)
    return static, fields, source, data, lat, lon


def test_the_assembly_satisfies_the_soil_routers_refusal():
    from gpuwm.ingest.water_temperature import (
        require_assembled_water_temperature)
    from tilestream import realcase

    static, fields, source, data, lat, lon = _domain()
    assembly, policy = realcase.assembled_water_temperature(
        fields, static=static, landuse_attrs=_ATTRS, data=data,
        source_snapshot=source, target_lat=lat, target_lon=lon,
        route=realcase.WATER_ROUTE_SLABBED)
    assert policy == "era5_class_coherent"
    values = np.asarray(assembly.values)
    assert values.shape == (4, 6)
    assert np.isfinite(values).all()
    # Every water cell carries a physical temperature -- the inadmissible
    # 150 K lake SST cannot survive a per-body assembly.
    water = np.asarray(static["LANDMASK"]) < 0.5
    assert (values[water] > 200.0).all() and (values[water] < 350.0).all()
    assert assembly.receipt["route"] == realcase.WATER_ROUTE_SLABBED
    assert assembly.receipt["lake_class"] == "declared"
    # The refusal that killed twin_a accepts this exact forward.
    require_assembled_water_temperature(
        route=realcase.WATER_ROUTE_SLABBED, fields=fields,
        water_temperature=values, policy=policy)


def test_the_refusal_control_is_the_twin_a_failure():
    """RED-ON-REVERT: withholding the assembly is the battery's crash."""
    from gpuwm.ingest.water_temperature import (
        require_assembled_water_temperature)
    from tilestream import realcase

    static, fields, source, data, lat, lon = _domain()
    with pytest.raises(ValueError,
                       match="no assembled water temperature"):
        require_assembled_water_temperature(
            route=realcase.WATER_ROUTE_SLABBED, fields=fields,
            water_temperature=None, policy="era5_class_coherent")


def test_a_source_without_skintemp_is_refused_by_name():
    from tilestream import realcase

    static, fields, source, data, lat, lon = _domain()
    del fields["SKINTEMP"]
    with pytest.raises(realcase.RealCaseError, match="SKINTEMP"):
        realcase.assembled_water_temperature(
            fields, static=static, landuse_attrs=_ATTRS, data=data,
            source_snapshot=source, target_lat=lat, target_lon=lon,
            route=realcase.WATER_ROUTE_LOW_WATER)


def test_a_declared_policy_travels_from_the_case_to_the_assembly():
    """The case's own declaration wins, exactly as on the mainline route."""
    from tilestream import realcase

    static, fields, source, data, lat, lon = _domain()
    data = SimpleNamespace(water_temperature_policy="wrf_compat",
                           water_temperature_overlay=None)
    assembly, policy = realcase.assembled_water_temperature(
        fields, static=static, landuse_attrs=_ATTRS, data=data,
        source_snapshot=source, target_lat=lat, target_lon=lon,
        route=realcase.WATER_ROUTE_SLABBED)
    assert policy == "wrf_compat"
    assert assembly.receipt["policy"] == "wrf_compat"
