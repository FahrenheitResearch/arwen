"""The HRRR index subset selects what the configuration's preparation reads.

The native bridge reads HRRR's analyzed aerosol numbers (QNWFA/QNIFA,
indexed PMTF/PMTC) when a file carries them, and an mp=28 domain asking
for the analyzed aerosol needs them.  The subset selection was a fixed
561-record list without them, so a subset fetch for that configuration
handed the preparation a file that refused "missing QNIFA, QNWFA".
"""
from __future__ import annotations

from pathlib import Path

import pytest

from gpuwm.experiment import load_experiment
from gpuwm.fetch_bars import CERTIFIED_RECORD_BARS
from gpuwm.preparation_assets import (analyzed_aerosol_domains,
                                      analyzed_aerosol_fetch_hints)
from tools import download_hrrr_native_subset as subset

ROOT = Path(__file__).resolve().parents[1]


def _rows(*, aerosol: bool):
    rows = []
    names = subset.HYBRID_FIELDS + (subset.ANALYZED_AEROSOL_FIELDS if aerosol else ())
    for level in range(1, 51):
        for name in names:
            rows.append((name, f"{level} hybrid level"))
    rows += list(subset.SURFACE_FIELDS)
    return tuple(subset.IndexRow(sequence=index + 1, offset=index * 10,
                                 variable=variable, level=level,
                                 raw=f"{index + 1}:{index * 10}:d=2026100221:{variable}:{level}:anl:")
                 for index, (variable, level) in enumerate(rows))


def test_the_counts_and_the_certified_bars_agree():
    assert subset.atmosphere_record_count() == 561 == subset.ATMOSPHERE_RECORD_COUNT
    assert subset.atmosphere_record_count(analyzed_aerosol=True) == 661
    assert CERTIFIED_RECORD_BARS["hrrr-atmosphere-aerosol"] == 661
    plain = subset.atmosphere_selectors()
    aerosol = subset.atmosphere_selectors(analyzed_aerosol=True)
    assert len(plain) == 561 and len(aerosol) == 661
    assert "PMTF:1 hybrid level" in aerosol and "PMTC:50 hybrid level" in aerosol
    assert not any(selector.startswith("PMT") for selector in plain)


def test_the_selection_follows_the_preparation():
    rows = _rows(aerosol=True)
    assert len(subset._atmosphere_selection(rows)) == 561
    assert len(subset._atmosphere_selection(rows, analyzed_aerosol=True)) == 661
    with pytest.raises(subset.IndexInventoryError, match="PMTF"):
        subset._atmosphere_selection(_rows(aerosol=False), analyzed_aerosol=True)


def test_the_analyzed_aerosol_is_derived_from_the_configuration():
    gsd41 = load_experiment(ROOT / "configs/recipes/hrrr_v4_gsd41.toml")
    assert analyzed_aerosol_domains(gsd41) == (1,)
    assert analyzed_aerosol_fetch_hints(gsd41, {"source": "hrrr"}) == {
        "source": "hrrr", "analyzed_aerosol": True}
    # Another source's acquisition is not the HRRR subset.
    assert analyzed_aerosol_fetch_hints(gsd41, {"source": "gfs"}) == {"source": "gfs"}
    from dataclasses import replace

    plain = load_experiment(ROOT / "configs/recipes/hrrr_configuration_clock.toml")
    off = replace(plain, domains=tuple(
        replace(domain, run=replace(domain.run, use_rap_aero_icbc=False,
                                    mp28_aerosol_source="auto"))
        for domain in plain.domains))
    assert analyzed_aerosol_domains(off) == ()
    assert analyzed_aerosol_fetch_hints(off, {"source": "hrrr"}) == {"source": "hrrr"}


def test_gpuwm_fetch_takes_the_flag():
    from gpuwm.cli import build_parser

    args = build_parser().parse_args(
        ["fetch", "--source", "hrrr", "--cycle", "2026-10-02T21",
         "--hours", "1", "--analyzed-aerosol", "--out", "x"])
    assert args.analyzed_aerosol is True
