"""Acquisition owners reach setup through registry facts and pure hooks."""
import argparse
from datetime import datetime
from importlib import import_module
import json
from pathlib import Path
import tomllib

import pytest

from gpuwm import cyclone_setup as tc
from gpuwm import cyclone_sources as cs
from gpuwm import fetch
from gpuwm import source_adapters as registry


def owned_source():
    row = next(r for r in registry.source_adapters() if r.acquisition_owner)
    owner = import_module(row.acquisition_owner)
    # Exercise an unclaimed model shorthand, with no source/model name here.
    alias = next(a for a in owner.source_aliases()
                 if registry.get_source_adapter(a).acquisition_owner == row.acquisition_owner
                 and owner.acquisition_hints(a))
    row = registry.get_source_adapter(alias)
    model = owner.acquisition_hints(alias)["model"]
    return row, alias, model


def test_owner_menu_alias_and_member_contract():
    row, alias, _ = owned_source()
    assert cs.source_adapter(alias) == row
    assert row.source_id in cs.source_ids()
    assert row.source_id in {r["source"] for r in cs.source_options()}
    assert cs.selected_member(alias) is None
    assert cs.follow_statics(alias)["integrates_moving_nest"]
    assert cs.plan_route(alias) == "prepared"


def test_alias_and_explicit_model_author_identical_fetch_contract():
    row, alias, model = owned_source()
    arguments = dict(cycle="2024-09-24T12", point=(20., -86.), hours=72, tiles="off")
    text, _ = tc.configuration_text(**arguments, forcing_source=alias)
    canonical, _ = tc.configuration_text(**arguments, forcing_source=row.source_id, model=model)
    a, b = (tomllib.loads(t)["fetch"] for t in (text, canonical))
    assert a == b
    assert a["source"] == row.source_id and a["model"] == model
    assert a["cycle"] == "2024-09-24T12" and a["hours"] == 72
    assert "area" in a
    fetch.validate_fetch_hints(a, source="owner fixture")
    assert fetch.source_argument(alias) == row.source_id


def test_owner_area_reserves_parent_plus_margin():
    row, alias, _ = owned_source()
    text, _ = tc.configuration_text(cycle="2024-09-24T12", point=(20., -86.),
        hours=72, forcing_source=alias, tiles="off")
    raw = tomllib.loads(text)
    from gpuwm import domain_wizard as dw
    parent = raw["domain"][0]
    grid = dw._root_grid(raw["projection"], parent["nx"], parent["ny"], parent["dx"])
    south, west, north, east = map(float, raw["fetch"]["area"].split(","))
    margin = dw._fetch_margin_deg(row.source_id)
    corners = [grid.ij_to_latlon(x, y) for x in (.5, parent["nx"]+.5)
               for y in (.5, parent["ny"]+.5)]
    assert south <= min(p[0] for p in corners)-margin+.02
    assert north >= max(p[0] for p in corners)+margin-.02
    assert west <= min(p[1] for p in corners)-margin+.02
    assert east >= max(p[1] for p in corners)+margin-.02


def test_model_on_unowned_source_refuses():
    source = next(s for s in cs.source_ids() if not cs.source_adapter(s).acquisition_owner)
    with pytest.raises(ValueError, match="no model selection"):
        cs.acquisition_hints(source, "arbitrary-model")
    with pytest.raises(ValueError, match="model requires"):
        fetch.validate_fetch_hints({"source": source, "model": "arbitrary-model"}, source="fixture")


def test_track_cli_preserves_alias_model_through_review(monkeypatch, tmp_path, capsys):
    from gpuwm import domain_wizard as dw
    from types import SimpleNamespace
    row, alias, model = owned_source()
    monkeypatch.setattr(dw, "_sizing_phases", lambda *a, **k:
        SimpleNamespace(peak_envelope_bytes=10**9, binding_phase="forecast"))
    monkeypatch.setattr(dw, "sizing_budget_bytes", lambda *a, **k: 90*dw.GIB)
    fixture = json.loads((Path(__file__).parent/"fixtures/cyclone-tracks.json").read_text())
    start = datetime(2024, 9, 24, 12)
    from datetime import timedelta
    for member in fixture["members"]:
        member.update(source=row.source_id, init=start.strftime("%Y-%m-%dT%H:%M:%SZ"))
    for track in fixture["tracks"]:
        for fix in track["fixes"]:
            fix["valid"] = (start+timedelta(hours=fix["lead_hours"])).strftime("%Y-%m-%dT%H:%M:%SZ")
    path = tmp_path/"tracks.json"
    path.write_text(json.dumps(fixture))
    parser = argparse.ArgumentParser()
    tc.register_cli(parser.add_subparsers())
    args = parser.parse_args(["cyclone-setup", "--source", alias, "--tracks", str(path),
        "--storm", "auto", "--cycle", "2024-09-24T12", "--start-hour", "0", "--hours", "72",
        "--card", "96gb", "--tiles", "off", "--out", str(tmp_path/"setup.toml")])
    assert tc.main(args) == 0
    proposal = json.loads(capsys.readouterr().out)
    assert proposal["fitting"]["review_required"]
    args.accept_fit = proposal["fitting"]["fit_id"]
    assert tc.main(args) == 0
    result = json.loads(capsys.readouterr().out)
    hints = tomllib.loads(result["config_text"])["fetch"]
    assert hints["model"] == model and hints["source"] == row.source_id
    assert result["created"] and not result["forecast_started"]
    assert result["track_placement"]["all_fixes_with_follower_clearance"]
    assert result["lineage"]["cyclone_tracks"]["member"]["source"] == row.source_id
