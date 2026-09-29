"""Surface pressure reduced from mean-sea-level pressure, lead by lead.

AIGFS and AIGEFS derive their surface pressure from each valid time's own
mean-sea-level pressure through ``surface_pressure_from_sea_level``, WRF
real's sfcprs3 relation, at the terrain height the analysis donor
supplies.  The breakage this prevents: AIGFS borrowed the analysis
surface pressure, which carried the f000 column mass to every lead, so
the lateral boundaries never saw a pressure system move; AIGEFS files
from NOMADS carry no surface pressure at all, and the record the AWS
mirror appends sits on the AI model's own orography rather than the
terrain the composition pairs it with.

The expected numbers are the mapped engine's own unit-test numbers
(``derive.rs``), so the two engines are held to the same arithmetic.
"""

from __future__ import annotations

import copy
import json
import math

import numpy as np
import pytest

from gpuwm.mapped_source import (CanonicalField, _evaluate_derivation,
                                 _surface_pressure_from_sea_level,
                                 load_mapping)
from gpuwm.source_authorities import packaged_authorities

#: One column on five isobaric levels, top first as a pressure-level
#: source declares them: 500 to 1000 hPa.
LEVELS_PA = np.array([50_000.0, 70_000.0, 85_000.0, 92_500.0, 100_000.0])
HEIGHTS_M = np.array([5_600.0, 3_000.0, 1_500.0, 800.0, 100.0])


def _column(slp: float, terrain: float, *, levels=LEVELS_PA,
            heights=HEIGHTS_M) -> float:
    values = _surface_pressure_from_sea_level(
        np.array([[slp]]), heights[:, None, None], levels[:, None, None],
        np.array([[terrain]]), "surface_pressure")
    return float(values[0, 0])


def test_a_low_surface_takes_sea_level_pressure_along_the_lowest_gradient():
    expected = 101_300.0 + (100_000.0 - 92_500.0) / (100.0 - 800.0) * 30.0
    assert _column(101_300.0, 30.0) == expected
    assert _column(101_300.0, 0.0) == 101_300.0


def test_a_bracketed_surface_is_interpolated_in_log_pressure():
    zm = 1_200.0
    expected = math.exp(
        (math.log(92_500.0) * (zm - 1_500.0)
         + math.log(85_000.0) * (800.0 - zm)) / (800.0 - 1_500.0))
    got = _column(101_300.0, zm)
    assert got == pytest.approx(expected, rel=1e-15)
    assert 85_000.0 < got < 92_500.0


def test_a_surface_under_every_level_interpolates_from_sea_level_to_the_second():
    zm = 60.0
    expected = math.exp(
        (math.log(101_300.0) * (zm - 800.0)
         + math.log(92_500.0) * (0.0 - zm)) / (0.0 - 800.0))
    assert _column(101_300.0, zm) == pytest.approx(expected, rel=1e-15)


def test_a_higher_sea_level_pressure_gives_a_higher_surface_pressure():
    for terrain in (0.0, 30.0, 60.0, 1_200.0, 2_500.0):
        assert _column(101_800.0, terrain) >= _column(100_800.0, terrain)


def test_the_source_level_order_does_not_change_the_answer():
    for terrain in (0.0, 30.0, 60.0, 1_200.0):
        assert _column(101_300.0, terrain, levels=LEVELS_PA[::-1],
                       heights=HEIGHTS_M[::-1]) == _column(101_300.0, terrain)
    # Reordering copies the same numbers into the same operations, so the
    # answer is the same bytes, not merely close.


def test_every_column_is_reduced_on_its_own():
    slp = np.array([[101_300.0, 99_000.0]])
    terrain = np.array([[30.0, 1_200.0]])
    heights = np.stack([HEIGHTS_M, HEIGHTS_M + 50.0], axis=-1)[:, None, :]
    levels = np.broadcast_to(LEVELS_PA[:, None, None], heights.shape).copy()
    values = _surface_pressure_from_sea_level(
        slp, heights, levels, terrain, "surface_pressure")
    assert values.shape == (1, 2)
    assert values[0, 0] == _column(101_300.0, 30.0)
    assert values[0, 1] == pytest.approx(
        _column(99_000.0, 1_200.0, heights=HEIGHTS_M + 50.0), rel=1e-15)


def test_a_column_out_of_the_first_columns_order_refuses_by_position():
    levels = np.array([[85_000.0, 100_000.0], [92_500.0, 92_500.0],
                       [100_000.0, 85_000.0]])[:, None, :]
    heights = np.array([[1_500.0, 100.0], [800.0, 800.0],
                        [100.0, 1_500.0]])[:, None, :]
    with pytest.raises(ValueError) as caught:
        _surface_pressure_from_sea_level(
            np.full((1, 2), 101_300.0), heights, levels,
            np.full((1, 2), 10.0), "surface_pressure")
    message = str(caught.value)
    assert "row 0, column 1" in message
    assert "strict order" in message


def test_a_non_finite_sea_level_pressure_refuses_by_position():
    with pytest.raises(ValueError, match="row 0, column 0: the sea-level"):
        _column(float("nan"), 30.0)


def _field(name, axes, values, *, location="surface"):
    return CanonicalField(
        name=name, units="", axes=tuple(axes), location=location,
        staggering="none", values=np.asarray(values, dtype=np.float64),
        missing_count=0, source_references=(f"@test.{name}",))


def test_the_derivation_reads_its_four_declared_operands_and_waits_for_terrain():
    operation = {
        "name": "psfc", "operation": "surface_pressure_from_sea_level",
        "sea_level_pressure": "air_pressure_at_mean_sea_level",
        "level_height": "geopotential_height",
        "pressure": "air_pressure",
        "surface_height": "terrain_height",
    }
    field = {"source_axes": ["y", "x"], "target_axes": ["y", "x"],
             "units": {"source": "Pa", "target": "Pa"},
             "location": "surface", "missing": {"kind": "reject"}}
    available = {
        "air_pressure_at_mean_sea_level": _field(
            "air_pressure_at_mean_sea_level", ("y", "x"),
            [[101_300.0, 101_300.0]]),
        "geopotential_height": _field(
            "geopotential_height", ("vertical", "y", "x"),
            np.repeat(HEIGHTS_M[:, None, None], 2, axis=2), location="mass"),
        "air_pressure": _field(
            "air_pressure", ("vertical", "y", "x"),
            np.repeat(LEVELS_PA[:, None, None], 2, axis=2), location="mass"),
    }
    # Terrain is borrowed in a composed frame and may arrive last: until
    # it does, the derivation asks to be retried rather than refusing.
    with pytest.raises(KeyError):
        _evaluate_derivation(operation, available, None, field,
                             "surface_pressure")
    available["terrain_height"] = _field("terrain_height", ("y", "x"),
                                         [[30.0, 1_200.0]])
    values, axes, references = _evaluate_derivation(
        operation, available, None, field, "surface_pressure")
    assert axes == ("y", "x")
    assert values[0, 0] == _column(101_300.0, 30.0)
    assert values[0, 1] == pytest.approx(_column(101_300.0, 1_200.0),
                                         rel=1e-15)
    assert references[0] == "@derived.sea_level_reduction"
    assert "@test.terrain_height" in references


def _sea_level_mapping(*drop):
    """A real 13-level mapping whose surface pressure is reduced from MSLP."""

    mapping = copy.deepcopy(json.loads(packaged_authorities(
        "aigfs-gdas-hybrid-grib2-v1")["mapping"].read_text(encoding="utf-8")))
    mapping["fields"]["surface_pressure"] = {
        "selectors": [], "units": {"source": "Pa", "target": "Pa"},
        "source_axes": ["y", "x"], "target_axes": ["y", "x"],
        "location": "surface", "staggering": "none",
        "missing": {"kind": "reject"},
        "derivation": "surface-pressure-from-sea-level"}
    mapping["fields"]["air_pressure_at_mean_sea_level"] = {
        "selectors": [{"format": "grib2", "discipline": 0, "category": 3,
                       "parameter": 1, "level_type": 101}],
        "units": {"source": "Pa", "target": "Pa"},
        "source_axes": ["y", "x"], "target_axes": ["y", "x"],
        "location": "surface", "staggering": "none",
        "missing": {"kind": "reject"}}
    derivation = {
        "name": "surface-pressure-from-sea-level",
        "operation": "surface_pressure_from_sea_level",
        "sea_level_pressure": "air_pressure_at_mean_sea_level",
        "level_height": "geopotential_height",
        "pressure": "air_pressure",
        "surface_height": "terrain_height"}
    for key in drop:
        del derivation[key]
    mapping["derivations"] = [
        item for item in mapping["derivations"]
        if item["name"] != derivation["name"]] + [derivation]
    return mapping


def test_a_mapping_declaring_the_four_operands_loads(tmp_path):
    path = tmp_path / "sea-level.mapping.json"
    path.write_text(json.dumps(_sea_level_mapping()), encoding="utf-8")
    assert load_mapping(path)["fields"]["surface_pressure"]["derivation"] == (
        "surface-pressure-from-sea-level")


def test_a_mapping_that_omits_an_operand_refuses_at_load(tmp_path):
    path = tmp_path / "broken.mapping.json"
    path.write_text(json.dumps(_sea_level_mapping("surface_height")),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="missing=\\['surface_height'\\]"):
        load_mapping(path)
