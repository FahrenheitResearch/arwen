"""An ERA5 intent on the config-driven route downloads its own forcing.

The breakage this guards: the run door skipped the generated config's
[fetch] recipe for every intent plan, so an ERA5 intent (the point-and-date
door every front end drives) was refused after the plan was accepted with
"declared input(s) this run needs are not on disk" and never downloaded
anything.  The fetch itself is replaced here; what is checked is that the
door asks for it, with the keyless provider and the generated area.
"""

from __future__ import annotations

import json

from gpuwm import runplan as rp
from gpuwm.gui.api import region_polygon


class _Fetched(Exception):
    pass


def _plan(tmp_path, monkeypatch, *, geog=True):
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")
    # The door refuses an install without cupy at plan acceptance, before the
    # fetch this file is about; these tests stop at or before the fetch, so
    # they hold on a CPU-only install as well.
    from gpuwm import capabilities
    monkeypatch.setattr(capabilities, "require", lambda *a, **k: None)
    monkeypatch.chdir(tmp_path)
    # The geography tree no download supplies: present, or not, before the run starts.
    monkeypatch.setenv("GPUWM_CASE_DATA_ROOT", str(tmp_path / "case-data"))
    if geog:
        (tmp_path / "case-data" / "WPS_GEOG").mkdir(parents=True)
    (tmp_path / "box.geojson").write_text(json.dumps(region_polygon(40.76, -111.9, 300, 300)))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps({
        "schema": "gpuwm.run-plan.v1", "name": "era5-intent", "route": "experiment",
        "config": {"intent": {"source": "era5", "cycle": "1999-08-11T12", "hours": 6,
                              "root_dx_km": 12, "polygon": str(tmp_path / "box.geojson"),
                              "card": "16gb"}},
        "output_root": str(tmp_path / "out")}))
    return plan_path


def test_an_era5_intent_asks_its_own_fetch_for_the_forcing(tmp_path, monkeypatch):
    plan_path = _plan(tmp_path, monkeypatch)
    asked = []

    def fake_fetch(arguments, run_dir, **_):
        asked.append(list(arguments))
        raise _Fetched

    monkeypatch.setattr(rp, "_run_fetch", fake_fetch)
    plan = rp.load_plan(plan_path)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with rp.EventStream(plan.run_dir / rp.EVENTS_FILENAME, mirror=None) as events:
        # The run stops at the replaced fetch; the door reports that as a failed stage.
        assert rp.execute_plan(plan, events=events) != 0
    assert len(asked) == 1
    arguments = asked[0]
    assert arguments[arguments.index("--source") + 1] == "era5"
    assert arguments[arguments.index("--era5-provider") + 1] == "arco"
    assert arguments[arguments.index("--cycle") + 1] == "1999-08-11T12"
    assert "--area" in arguments


def test_a_missing_geography_tree_is_refused_before_the_download(tmp_path, monkeypatch):
    """A run found its geography tree missing only after a nine minute ERA5 download; now it is refused first."""
    plan_path = _plan(tmp_path, monkeypatch, geog=False)
    asked = []
    monkeypatch.setattr(rp, "_run_fetch", lambda arguments, run_dir, **_: asked.append(arguments))
    plan = rp.load_plan(plan_path)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    with rp.EventStream(plan.run_dir / rp.EVENTS_FILENAME, mirror=None) as events:
        assert rp.execute_plan(plan, events=events) != 0
    assert asked == []
    failed = [e for e in rp.read_events(plan.run_dir / rp.EVENTS_FILENAME) if e["event"] == "failed"]
    assert "geog_root" in failed[0]["message"] and "before the download" in failed[0]["message"]
    # What to do is the command that sets the tree up, not "fix the plan document": the plan was not wrong.
    assert failed[0]["remedy"] == ("Set up the geography data once on this computer with "
                                   "gpuwm fetch-geog --datasets wrf.")
