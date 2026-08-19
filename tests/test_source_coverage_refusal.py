"""A source that cannot reach the domain is refused BY THE FRONT DOOR.

The refusal itself has been complete for a long time: it names the first
uncovered corner, the source index it maps to, and the window the source
actually covers -- enough to tell a crop that is too small from a source
whose grid does not reach the target, which have opposite remedies.  What
was missing is ownership.  The preparation adapters called the interpolator
with no handler, so a correct refusal arrived as ``ValueError`` at the end
of ten lines of internal call stack (measured 2026-08-17: a regional
European source against a central-US domain, ``gpuwm prep`` rc=1, one
``Traceback`` header, ten ``File ...`` frames).  A user reads the internals
of ``horiz.py`` before the sentence that tells them what to do.

The breakage these tests prevent: a preparation front door -- including the
one a NEW model gets for free through the mapped route -- relaying an
internal traceback instead of the refusal it already wrote.
"""
from __future__ import annotations

import importlib
import re

import numpy as np
import pytest

from gpuwm.ingest.horiz import _regular_coordinates
from gpuwm.ingest.source_coverage import (
    SOURCE_COVERAGE_EXIT_CODE,
    SOURCE_COVERAGE_REMEDY,
    SourceCoverageRefusal,
    owns_source_coverage_refusal,
    report_source_coverage_refusal,
)
from gpuwm.static.lambert import LambertGrid


def _domain(centre_lat: float, centre_lon: float) -> LambertGrid:
    """A small Lambert domain centred where the caller asks."""

    return LambertGrid(
        ref_lat=centre_lat, ref_lon=centre_lon,
        truelat1=centre_lat - 10.0, truelat2=centre_lat + 10.0,
        stand_lon=centre_lon, dx=3000.0, dy=3000.0, e_we=61, e_sn=61)


def _regional_axes(lat0: float, lat1: float, lon0: float, lon1: float,
                   step: float = 0.0625):
    """A regional source window: the whole grid such a model publishes."""

    return (np.arange(lat0, lat1 + 0.5 * step, step, dtype=np.float64),
            np.arange(lon0, lon1 + 0.5 * step, step, dtype=np.float64))


# Two unrelated regional windows and a domain each one cannot reach.  The
# refusal is a property of "regional source, distant domain", not of any
# one model, so nothing here names a model or a case.
_UNREACHABLE = (
    # An eastern-hemisphere mid-latitude window; domain in the central US.
    ((29.5, 70.5, -23.5, 62.5), (38.5, -97.5)),
    # A southern-hemisphere window; domain in the northern mid-latitudes.
    ((-45.0, -10.0, 110.0, 155.0), (45.0, 10.0)),
)


@pytest.mark.parametrize("window,centre", _UNREACHABLE)
def test_out_of_coverage_is_a_named_refusal_class_for_any_regional_source(
        window, centre):
    latitude, longitude = _regional_axes(*window)
    lat, lon = _domain(*centre).latlon_mass()
    with pytest.raises(SourceCoverageRefusal) as raised:
        _regular_coordinates(latitude, longitude, lat, lon)
    message = str(raised.value)
    assert "fall outside the source grid" in message
    # The three numbers that make the refusal actionable survive the class.
    assert re.search(r"target point \(\d+, \d+\) at lat/lon", message)
    assert re.search(r"maps to source index x=-?[\d.]+ y=-?[\d.]+", message)
    assert re.search(r"the source covers x=0\.\.\d+ \(lon ", message)


def test_the_refusal_class_stays_a_valueerror_for_library_callers():
    """Every existing ``except ValueError`` around interpolation still fires."""

    assert issubclass(SourceCoverageRefusal, ValueError)


def test_a_domain_inside_the_window_is_not_refused():
    """The negative control: the same code path passes when it should."""

    latitude, longitude = _regional_axes(29.5, 70.5, -23.5, 62.5)
    lat, lon = _domain(50.0, 10.0).latlon_mass()
    y, x = _regular_coordinates(latitude, longitude, lat, lon)
    assert np.isfinite(y).all() and np.isfinite(x).all()


def test_the_report_prints_the_named_breakage_and_the_remedy(capsys):
    refusal = SourceCoverageRefusal("target points fall outside the source grid: X")
    code = report_source_coverage_refusal(refusal)
    captured = capsys.readouterr()
    assert code == SOURCE_COVERAGE_EXIT_CODE != 0
    assert "target points fall outside the source grid: X" in captured.err
    assert SOURCE_COVERAGE_REMEDY in captured.err
    assert "Traceback" not in captured.err
    assert captured.out == ""


def test_the_remedy_names_both_repairs():
    """A crop too small and a source that cannot reach have opposite fixes."""

    remedy = SOURCE_COVERAGE_REMEDY.lower()
    assert "re-fetch" in remedy or "wider" in remedy
    assert "--list-sources" in remedy


def _prep_adapter_modules() -> tuple[str, ...]:
    """The modules ``gpuwm prep`` launches, read from its own routing code."""

    source = (
        importlib.import_module("gpuwm.source_cli").__file__)
    with open(source, encoding="utf-8") as stream:
        text = stream.read()
    found = re.findall(r'"-m",\s*\n\s*"(gpuwm\.[a-z0-9_]+)"', text)
    return tuple(dict.fromkeys(found))


def test_the_adapter_list_is_read_from_the_routing_code_and_is_not_empty():
    """Validate the instrument: an empty list would make the gate vacuous."""

    modules = _prep_adapter_modules()
    assert len(modules) >= 5
    assert "gpuwm.mapped_direct" in modules


@pytest.mark.parametrize("module_name", _prep_adapter_modules())
def test_every_prep_adapter_door_owns_the_coverage_refusal(module_name):
    """A new adapter that forgets this delivers a traceback to a user."""

    module = importlib.import_module(module_name)
    assert getattr(module.main, "owns_source_coverage_refusal", False) is True


def test_the_door_wrapper_returns_the_refusal_instead_of_raising(capsys):
    message = "target points fall outside the source grid: window"

    @owns_source_coverage_refusal
    def main(argv=None) -> int:
        raise SourceCoverageRefusal(message)

    assert main([]) == SOURCE_COVERAGE_EXIT_CODE
    captured = capsys.readouterr()
    assert message in captured.err
    assert "Traceback" not in captured.err


def test_the_door_wrapper_does_not_swallow_other_failures():
    """Only the named refusal is owned; a real defect still raises."""

    @owns_source_coverage_refusal
    def main(argv=None) -> int:
        raise ValueError("some other problem")

    with pytest.raises(ValueError, match="some other problem"):
        main([])


def test_the_mapped_door_reports_the_refusal_without_a_traceback(
        monkeypatch, tmp_path, capsys):
    """The observed defect, at the door every new model route goes through."""

    from gpuwm import mapped_direct

    message = (
        "target points fall outside the source grid: target point (0, 0) at "
        "lat/lon (34.2972, -102.4465) maps to source index x=-1263.144 "
        "y=76.756, and the source covers x=0..1376 (lon -23.5..62.5) "
        "y=0..656 (lat 29.5..70.5)")

    def refuse(**kwargs):
        raise SourceCoverageRefusal(message)

    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", refuse)
    monkeypatch.setattr(mapped_direct, "load_mapping",
                        lambda path: {"format": "netcdf"})
    mapping = tmp_path / "mapping.json"
    mapping.write_text("{}", encoding="utf-8")
    composition = tmp_path / "composition.json"
    composition.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "inputs.json"
    manifest.write_text("{}", encoding="utf-8")
    code = mapped_direct.main([
        "--source-format", "netcdf",
        "--mapping", str(mapping),
        "--composition", str(composition),
        "--input", str(tmp_path / "one.nc"),
        "--input-manifest", str(manifest),
        "--input-manifest-sha256", "0" * 64,
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--geog-root", str(tmp_path / "geog"),
        "--experiment-config", str(tmp_path / "case.toml"),
        "--output-root", str(tmp_path / "out"),
    ])
    captured = capsys.readouterr()
    assert code == SOURCE_COVERAGE_EXIT_CODE
    assert message in captured.err
    assert SOURCE_COVERAGE_REMEDY in captured.err
    assert "Traceback" not in captured.err
    assert "horiz.py" not in captured.err


# ---------------------------------------------------------------------------
# The second member of the family: a series too short to bound a forecast.
#
# MEASURED 2026-08-17 on the tip: a single staged valid time through the
# preparation door raised a bare ``ValueError`` from the mapped engine, so
# a complete sentence -- "lateral-boundary forcing requires at least two
# times" -- arrived at the end of a 28-line traceback through a door that
# already owns its refusals.  Three sibling routes spelled the same
# refusal their own way, each with its own traceback.
# ---------------------------------------------------------------------------

def _single_time_mapped_inputs(tmp_path):
    """One staged valid time, written by the mapped suite's own builders.

    Borrowed rather than re-typed: a second copy of the fixture would
    drift from the corpus the decode is actually tested against, and this
    test is about the DELIVERY of a refusal the real engine raises on
    real bytes, not about a bespoke file shape.
    """

    import test_mapped_source as corpus

    mapping = tmp_path / "mapping.json"
    source = tmp_path / "source.nc"
    manifest = tmp_path / "inputs.json"
    corpus._write_mapping(mapping, corpus._mapping())
    corpus._write_source(source, times=(0.0,))
    manifest_sha = corpus._write_manifest(manifest, mapping, source)
    return mapping, source, manifest, manifest_sha


def test_a_one_time_series_is_a_named_refusal_not_a_bare_valueerror(tmp_path):
    """The engine states it; the class is what lets a door deliver it."""

    from gpuwm.ingest.source_coverage import ForcingSeriesRefusal
    from gpuwm.mapped_source import decode_mapped_source

    mapping, source, manifest, manifest_sha = _single_time_mapped_inputs(
        tmp_path)
    with pytest.raises(ForcingSeriesRefusal) as raised:
        decode_mapped_source(mapping, [source], input_manifest=manifest,
                             input_manifest_sha256=manifest_sha)
    # The sentence was never the defect; it survives the class unchanged.
    assert "at least two times" in str(raised.value)


def test_the_forcing_series_refusal_stays_a_valueerror_for_library_callers():
    from gpuwm.ingest.source_coverage import ForcingSeriesRefusal

    assert issubclass(ForcingSeriesRefusal, ValueError)


def test_the_series_remedy_names_what_to_stage():
    from gpuwm.ingest.source_coverage import FORCING_SERIES_REMEDY

    remedy = FORCING_SERIES_REMEDY.lower()
    assert "at least two" in remedy
    assert "lateral boundar" in remedy


def test_the_prep_door_delivers_the_series_refusal_as_two_sentences(
        monkeypatch, tmp_path, capsys):
    """The observed defect, at the door a user actually types.

    The refusal is raised by the REAL decode of a REAL one-time file --
    only the surrounding preparation (statics, namelist, geography) is
    stood aside, because none of it is reached before the series is read.
    """

    from gpuwm import mapped_direct
    from gpuwm.ingest.source_coverage import (
        FORCING_SERIES_REMEDY, PREPARATION_REFUSAL_EXIT_CODE)
    from gpuwm.mapped_source import decode_mapped_source

    mapping, source, manifest, manifest_sha = _single_time_mapped_inputs(
        tmp_path)

    def prepare(**kwargs):
        return decode_mapped_source(
            mapping, [source], input_manifest=manifest,
            input_manifest_sha256=manifest_sha)

    monkeypatch.setattr(mapped_direct, "prepare_mapped_wrf", prepare)
    monkeypatch.setattr(mapped_direct, "load_mapping",
                        lambda path: {"format": "netcdf"})
    code = mapped_direct.main([
        "--source-format", "netcdf",
        "--mapping", str(mapping),
        "--composition", str(tmp_path / "composition.json"),
        "--input", str(source),
        "--input-manifest", str(manifest),
        "--input-manifest-sha256", manifest_sha,
        "--wps-namelist", str(tmp_path / "namelist.wps"),
        "--geog-root", str(tmp_path / "geog"),
        "--experiment-config", str(tmp_path / "case.toml"),
        "--output-root", str(tmp_path / "out"),
    ])
    captured = capsys.readouterr()
    assert code == PREPARATION_REFUSAL_EXIT_CODE
    assert "at least two times" in captured.err
    assert FORCING_SERIES_REMEDY in captured.err
    assert "Traceback" not in captured.err
    assert "mapped_source.py" not in captured.err
    assert captured.out == ""


#: Every route that reads a staged valid-time series states this refusal
#: in its own words.  The words are fine; the CLASS is what a door can
#: own, so a route that goes back to a bare ``ValueError`` hands a user a
#: traceback again.  Read as source text because the four call sites take
#: four different argument shapes -- what is asserted is that none of
#: them raises the unowned class.
_SERIES_REFUSAL_MODULES = (
    "gpuwm/mapped_source.py",
    "gpuwm/era5_direct.py",
    "gpuwm/gfs_direct.py",
    "gpuwm/twentycrv3_direct.py",
)


@pytest.mark.parametrize("relative_path", _SERIES_REFUSAL_MODULES)
def test_no_route_states_the_short_series_refusal_as_a_bare_valueerror(
        relative_path):
    from pathlib import Path as _Path

    import gpuwm

    text = (_Path(gpuwm.__file__).parent.parent / relative_path).read_text(
        encoding="utf-8")
    offenders = [line.strip() for line in text.splitlines()
                 if "at least two times" in line
                 and "raise ValueError(" in line]
    assert not offenders, offenders
    # Validate the instrument: the refusal is still stated in this module,
    # so an empty file or a renamed message cannot make the gate vacuous.
    assert "at least two times" in text


def test_every_short_series_refusal_raises_the_owned_family():
    """Both spellings that can be exercised without staged bytes."""

    from gpuwm.ingest.source_coverage import ForcingSeriesRefusal
    from gpuwm.gfs_direct import _read_series

    import tempfile
    from pathlib import Path as _Path

    with tempfile.TemporaryDirectory() as directory:
        series = _Path(directory) / "series.tsv"
        (_Path(directory) / "f018.grib2").write_bytes(b"GRIB")
        series.write_text("18\tf018.grib2\t96\n", encoding="utf-8")
        with pytest.raises(ForcingSeriesRefusal):
            _read_series(series)
