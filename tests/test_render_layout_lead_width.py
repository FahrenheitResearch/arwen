"""The layout's own seams: the lead grammar, the frame clock, the delivery.

These are the parts of :mod:`gpuwm.render_layout` that are pure path and
filename work, so they run with no renderer binary, no netCDF and no
``wrf`` package -- which is what separates this file from
``tests/test_render_layout.py``, whose fixtures build real wrfout frames
and which therefore skips wherever ``wrf`` is not installed.  A seam
nothing can exercise on a plain CPU box is a seam that breaks unnoticed.
"""
from __future__ import annotations

import pytest

from gpuwm import render_layout


# ------------------------------------------------- the lead is a width

@pytest.mark.parametrize("lead,day", [
    ("f999", "2026-09-30"), ("f1000", "2026-09-30"), ("f1001", "2026-09-30")])
def test_a_lead_past_999_hours_parses_like_any_other(lead, day):
    """Three digits is the zero-padding width, never a ceiling."""
    name = f"arwen_wrf_20260820_0z_{lead}_d01-12km_2m_temperature.png"
    parsed = render_layout.parse_engine_output(name)
    assert parsed is not None, name
    assert parsed[0] == "d01-12km"
    assert parsed[1] == "2m_temperature"
    assert parsed[2] == day


def test_a_lead_past_999_hours_counts_every_hour_of_itself():
    """``f1000`` is 1000 hours, not 100 with a digit left over."""
    hundred = render_layout.parse_engine_output(
        "arwen_wrf_20260820_0z_f100_d01-12km_2m_temperature.png")
    thousand = render_layout.parse_engine_output(
        "arwen_wrf_20260820_0z_f1000_d01-12km_2m_temperature.png")
    assert hundred[2] == "2026-08-24"
    assert thousand[2] == "2026-09-30"


@pytest.mark.parametrize("lead", ["f999", "f1000", "f1001"])
def test_a_lead_past_999_hours_round_trips_through_its_two_folders(lead):
    name = f"arwen_wrf_20260820_0z_{lead}_d01-12km_2m_temperature.png"
    delivered = render_layout.delivered_name(
        name, domain="d01-12km", product="2m_temperature")
    assert "d01-12km" not in delivered
    assert delivered == f"arwen_wrf_20260820_0z_{lead}.png"
    assert render_layout.engine_name(
        delivered, domain="d01-12km", product="2m_temperature") == name
