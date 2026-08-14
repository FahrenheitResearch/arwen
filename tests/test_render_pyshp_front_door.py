"""The pyshp prerequisite is answered BEFORE the work, not after it.

The DA nowcast's render stage is the last thing a run does -- after the
survey, the fetch, the preparation, the free forecast and every DA cycle
-- and it opened its basemap with a bare ``import shapefile``.  A run
that was always going to fail therefore failed at the end, with
``ModuleNotFoundError: No module named 'shapefile'`` and no remedy:
the most work destroyed and the least information given of any failure
in the reachability audit.

These tests bind the fix in both directions, which is the only way a
capability check is worth anything: the refusal must FIRE when pyshp is
absent and must STAY SILENT when it is present.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gpuwm import rustwx


def _nowcast_args(**overrides):
    """The narrowest args object validate_analysis_flags reads."""

    defaults = dict(
        reflectivity_analysis=False, hydrometeors=False,
        positivity_policy=None, dealias=False, dealias_refinement=None,
        dealias_engine=None, clear_air_analysis=False, goes_cwp=None,
        cwp_vertical_loc_m=None, surface_obs=None, sfc_t2_sigma_k=None,
        sfc_wspd_sigma_ms=None, stop_after=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# The capability check itself, both directions
# ---------------------------------------------------------------------------

def test_pyshp_available_answers_true_when_the_reader_imports():
    """The instrument must not report a gap that is not there."""

    pytest.importorskip("shapefile")
    assert rustwx.pyshp_available() is True
    # ... and the gate it drives stays silent.
    rustwx.require_pyshp()


def test_pyshp_available_answers_false_when_the_reader_is_absent(
        monkeypatch):
    """The other direction, which is the one that was never asked."""

    import importlib.util

    real = importlib.util.find_spec

    def absent(name, *args, **kwargs):
        if name == "shapefile":
            return None
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib.util, "find_spec", absent)
    assert rustwx.pyshp_available() is False
    with pytest.raises(ImportError) as caught:
        rustwx.require_pyshp()
    assert "pyshp" in str(caught.value)


def test_the_remedy_names_a_command_that_can_actually_supply_it():
    """A remedy that cannot supply what it names is worse than none.

    Both halves have to be there: the reader arrives by pip, the
    geometry it reads arrives in the bridge bundle, and naming only the
    first sends a reader to a second failure one step later.
    """

    remedy = rustwx.PYSHP_REMEDY
    assert "pip install" in remedy
    assert "gpuwm[render]" in remedy
    assert "gpuwm fetch-bridges" in remedy


# ---------------------------------------------------------------------------
# The position: the front door, before the forecast
# ---------------------------------------------------------------------------

def test_the_nowcast_front_door_refuses_before_any_stage_runs(monkeypatch):
    """FAILS before the fix: nothing asked, so nothing refused."""

    from tools import da_nowcast

    monkeypatch.setattr(rustwx, "pyshp_available", lambda: False)
    with pytest.raises(da_nowcast.FrontDoorError) as caught:
        da_nowcast.validate_analysis_flags(_nowcast_args())
    message = str(caught.value)
    assert "render" in message
    assert "pyshp" in message
    # It must also name the escape that keeps the rest of the run: a
    # user who cannot install pyshp right now can still get every
    # receipt except the gallery.
    assert "--stop-after cycle" in message


def test_the_front_door_is_silent_when_pyshp_is_installed(monkeypatch):
    """The instrument, validated the other way."""

    from tools import da_nowcast

    monkeypatch.setattr(rustwx, "pyshp_available", lambda: True)
    da_nowcast.validate_analysis_flags(_nowcast_args())


def test_a_run_that_stops_before_render_is_not_refused(monkeypatch):
    """A prerequisite for work this run will not do is not a gap.

    `--stop-after cycle` never reaches the render stage, so refusing it
    for a missing map reader would be a refusal for a capability the run
    does not need.
    """

    from tools import da_nowcast

    monkeypatch.setattr(rustwx, "pyshp_available", lambda: False)
    for stage in ("survey", "domain", "fetch", "prepare", "forecast",
                  "obs", "cycle"):
        da_nowcast.validate_analysis_flags(_nowcast_args(stop_after=stage))
    # ... but `--stop-after render` DOES run the render stage.
    with pytest.raises(da_nowcast.FrontDoorError):
        da_nowcast.validate_analysis_flags(_nowcast_args(stop_after="render"))


def test_run_reaches_render_is_derived_from_the_stage_list():
    """Not a hand-written list of the stages that stop early today."""

    from tools import da_nowcast

    for stage in da_nowcast.STAGES:
        expected = stage == "render"
        assert da_nowcast.run_reaches_render(
            _nowcast_args(stop_after=stage)) is expected, stage
    assert da_nowcast.run_reaches_render(_nowcast_args(stop_after=None))


# ---------------------------------------------------------------------------
# The direct doors: the renderer run on its own
# ---------------------------------------------------------------------------

def test_no_basemap_call_site_still_bare_imports_the_reader():
    """The three sites the audit named, checked as source.

    A test that only exercised the nowcast front door would pass while
    `python -m tools.da_nowcast_render` still died on a bare import.
    """

    for relative in ("tools/da_nowcast_render.py",
                     "tools/da_nowcast_launcher.py"):
        text = (REPO_ROOT / relative).read_text(encoding="utf-8")
        for index, line in enumerate(text.splitlines(), start=1):
            if line.strip() != "import shapefile":
                continue
            window = "\n".join(text.splitlines()[max(0, index - 12):index])
            assert "require_pyshp" in window, (
                f"{relative}:{index} imports shapefile with no named "
                "refusal above it; a missing reader there is a bare "
                "ModuleNotFoundError at the end of a whole run")
