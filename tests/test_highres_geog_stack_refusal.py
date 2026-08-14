"""The high-resolution terrain path refuses BEFORE it downloads.

2.3.2's failure was not that a dependency could be missing.  It was the
ORDER: the only check for rasterio lived inside the mosaic step, which
runs after the tile fetch, so a missing library cost the user 160.7 MiB
of Copernicus downloads and then produced a bare ``RuntimeError``
traceback at exit 1 with no remedy in it.

Three properties are pinned here, and each one failed in 2.3.2:

- the refusal fires with the network provably untouched;
- it is a named :class:`HighresRefusal` carrying the exact install
  command, not a raw exception class;
- ``on_refuse = "fallback-30s"`` does NOT swallow it.  That setting is a
  statement about source *coverage*; letting it apply to a broken install
  would hand back 900 m baseline terrain because a library was missing,
  which is the silent degradation the whole module refuses.
"""
from __future__ import annotations

from datetime import date

import pytest

from gpuwm.static import geog_stack
from gpuwm.static.highres_production import (HighresRefusal,
                                             apply_highres_statics,
                                             require_geography_stack)


class _Config:
    """Minimal stand-in for a validated [static.highres] block."""

    def __init__(self, on_refuse="error"):
        self.enabled = True
        self.cache_root = "unused-because-we-refuse-first"
        self.on_refuse = on_refuse
        self.terrain_source = "auto"
        self.fields = "auto"

    def echo(self):
        return {}


def _blind(monkeypatch):
    """Make the geography stack look absent, however it is installed."""
    monkeypatch.setattr(geog_stack, "find_spec", lambda name: None)


def _exploding_urlopen(*args, **kwargs):
    raise AssertionError(
        "the network was touched before the dependency was checked")


@pytest.mark.parametrize("on_refuse", ["error", "fallback-30s"])
def test_missing_geography_stack_refuses_before_any_fetch(monkeypatch,
                                                          on_refuse):
    _blind(monkeypatch)
    with pytest.raises(HighresRefusal) as caught:
        apply_highres_statics(
            {"HGT_M": None}, object(), config=_Config(on_refuse),
            domain_id=1, case_date=date(2024, 6, 1), landuse_attrs={},
            urlopen=_exploding_urlopen)
    assert caught.value.reason == "geography-stack-missing"


def test_the_refusal_names_the_exact_command_to_run(monkeypatch):
    _blind(monkeypatch)
    with pytest.raises(HighresRefusal) as caught:
        require_geography_stack()
    message = str(caught.value)
    # The remedy has to be pasteable, and it has to name both libraries.
    assert "pip install --upgrade gpuwm" in message
    assert "rasterio" in message and "pyproj" in message
    assert "gpuwm doctor" in message


def test_fallback_30s_does_not_swallow_a_broken_install(monkeypatch):
    """A coverage policy must not silently answer an install question."""
    _blind(monkeypatch)
    with pytest.raises(HighresRefusal):
        apply_highres_statics(
            {"HGT_M": None}, object(), config=_Config("fallback-30s"),
            domain_id=1, case_date=date(2024, 6, 1), landuse_attrs={},
            urlopen=_exploding_urlopen)


def test_a_disabled_block_is_still_the_identity(monkeypatch):
    """Nothing is checked, fetched or printed when the block is off."""
    _blind(monkeypatch)

    class Off:
        enabled = False

    baseline = {"HGT_M": None}
    fields, receipt = apply_highres_statics(
        baseline, object(), config=Off(), domain_id=1,
        case_date=date(2024, 6, 1), landuse_attrs={},
        urlopen=_exploding_urlopen)
    assert fields is baseline
    assert receipt is None


def test_the_probe_reports_the_truth_on_this_box():
    """Instrument check: with the real stack installed, nothing missing.

    rasterio and pyproj are runtime dependencies as of 2.3.3, so any
    environment that can import gpuwm at all has them.  A failure here
    means the packaging change regressed.
    """
    assert geog_stack.missing_geog_modules() == ()


def test_the_probe_is_not_vacuously_green(monkeypatch):
    """Both directions: it must also be able to report absence."""
    _blind(monkeypatch)
    assert set(geog_stack.missing_geog_modules()) == {"rasterio", "pyproj"}
