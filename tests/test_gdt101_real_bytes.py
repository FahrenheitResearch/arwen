"""Real DWD octets, against the selectors the shipped document declares.

The fixture beside this file holds GRIB2 Sections 0 through 6, verbatim, of
twenty-six objects downloaded from opendata.dwd.de for the 2026-09-15 12 UTC
ICON global cycle.  Section 7 is left out on purpose: these octets are the
field IDENTITY, which is exactly what a selector addresses, and carrying the
packed values would add megabytes that prove nothing more.

Everything the normalization document claims about those bytes is checked here
against the bytes: originating centre, grid template, cell count, product
template, reference time, the forecast unit and lead, and every field's
discipline, category, parameter and both fixed surfaces.  A selector edited in
the document without DWD having changed its product fails this test.

Source: Deutscher Wetterdienst (CC BY 4.0).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from gpuwm import source_normalization as norm

FIXTURE = (Path(__file__).with_name("fixtures")
           / "icon-global-2026091512-headers.json")
MISSING_OCTET = 0xFF
MISSING_WORD = 0xFFFF_FFFF


def sections(header: bytes) -> dict[int, bytes]:
    """Split one GRIB2 message prefix into its numbered sections."""

    assert header[:4] == b"GRIB" and header[7] == 2
    found: dict[int, bytes] = {}
    offset = 16
    while offset < len(header):
        length = int.from_bytes(header[offset:offset + 4], "big")
        number = header[offset + 4]
        assert 5 <= length <= len(header) - offset
        found[number] = header[offset:offset + length]
        offset += length
    return found


def surface(scale_octet: int, raw: int) -> float:
    """A GRIB2 scaled fixed-surface pair, or 0.0 when it is not encoded."""

    if scale_octet == MISSING_OCTET or raw == MISSING_WORD:
        return 0.0
    scale = -(scale_octet & 0x7F) if scale_octet & 0x80 else scale_octet
    value = -(raw & 0x7FFF_FFFF) if raw & 0x8000_0000 else raw
    return value / 10.0 ** scale


@pytest.fixture(scope="module")
def fixture():
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert document["schema"] == "gpuwm-source-real-byte-headers-v1"
    assert document["licence"] == "CC BY 4.0"
    assert document["objects"]
    return document


@pytest.fixture(scope="module")
def spec():
    return norm.load_normalization("icon-gdt101-pressure-v1")


def observed(row) -> dict[str, object]:
    header = base64.b64decode(row["header_base64"])
    part = sections(header)
    s1, s3, s4 = part[1], part[3], part[4]
    word = lambda s, a: int.from_bytes(s[a:a + 4], "big")   # noqa: E731
    half = lambda s, a: int.from_bytes(s[a:a + 2], "big")   # noqa: E731
    return {
        "discipline": header[6],
        "centre": half(s1, 5),
        "reference_time": (half(s1, 12), s1[14], s1[15], s1[16], s1[17], s1[18]),
        "cells": word(s3, 6),
        "grid_template": half(s3, 12),
        "product_template": half(s4, 7),
        "category": s4[9],
        "parameter": s4[10],
        "time_unit": s4[17],
        "forecast_time": word(s4, 18),
        "level_type": s4[22],
        "level_value": surface(s4[23], word(s4, 24)),
        "second_level_type": s4[28],
        "second_level_value": surface(s4[29], word(s4, 30)),
    }


def test_the_fixture_covers_every_field_the_document_declares(fixture, spec):
    named = set()
    for row in fixture["objects"]:
        named.add(norm.parse_object(spec, Path("/f") / row["object"]).field)
    assert named == set(spec.fields), sorted(set(spec.fields) - named)


@pytest.mark.parametrize("index", range(26))
def test_every_real_object_carries_the_declared_identity(fixture, spec, index):
    row = fixture["objects"][index]
    seen = observed(row)
    declared = spec.native_grid
    assert seen["grid_template"] == declared["grid_template"]
    assert seen["cells"] == declared["cells"]
    assert seen["centre"] == declared["originating_centre"]
    # Only product template 0 is addressed by a seven-value selector; an
    # ensemble or statistically-processed template would need its own.
    assert seen["product_template"] == 0
    assert seen["reference_time"] == (2026, 9, 15, 12, 0, 0)


@pytest.mark.parametrize("index", range(26))
def test_every_real_selector_matches_the_document(fixture, spec, index):
    row = fixture["objects"][index]
    obj = norm.parse_object(spec, Path("/not-downloaded") / row["object"])
    seen = observed(row)
    declared = obj.selector
    assert (seen["discipline"], seen["category"], seen["parameter"]) == declared[:3]
    assert seen["level_type"] == declared[3]
    assert seen["level_value"] == pytest.approx(declared[4], abs=1e-9)
    assert seen["second_level_type"] == declared[5]
    assert seen["second_level_value"] == pytest.approx(declared[6], abs=1e-9)


@pytest.mark.parametrize("index", range(26))
def test_the_forecast_lead_is_minutes_and_matches_the_object_name(
        fixture, spec, index):
    """DWD stamps the lead in MINUTES (Code Table 4.4 value 0), not hours."""

    row = fixture["objects"][index]
    obj = norm.parse_object(spec, Path("/not-downloaded") / row["object"])
    seen = observed(row)
    assert seen["time_unit"] == 0
    assert seen["forecast_time"] * 60 == (obj.lead or 0) * 3600


def test_the_soil_filename_five_really_is_half_a_centimetre(fixture, spec):
    """The one number in this product a reader would get wrong by division."""

    row = next(r for r in fixture["objects"] if r["object"].endswith("_5_T_SO.grib2"))
    assert observed(row)["level_value"] == pytest.approx(0.005)
    deeper = next(r for r in fixture["objects"]
                  if r["object"].endswith("_2_T_SO.grib2"))
    assert observed(deeper)["level_value"] == pytest.approx(0.02)


def test_the_water_column_layers_are_bounded_pairs(fixture, spec):
    row = next(r for r in fixture["objects"] if r["object"].endswith("_3_W_SO.grib2"))
    seen = observed(row)
    assert seen["level_type"] == 106 and seen["second_level_type"] == 106
    assert (seen["level_value"], seen["second_level_value"]) == pytest.approx((0.03, 0.09))


def test_terrain_rides_a_bounded_surface_not_a_bare_ground_level(fixture, spec):
    """HSURF's second surface is type 101 (mean sea level), and the selector says so."""

    row = next(r for r in fixture["objects"] if r["object"].endswith("_HSURF.grib2"))
    seen = observed(row)
    assert (seen["discipline"], seen["category"], seen["parameter"]) == (0, 3, 6)
    assert (seen["level_type"], seen["second_level_type"]) == (1, 101)
    assert spec.selector_for("HSURF", None) == (0, 3, 6, 1, 0.0, 101, 0.0)
