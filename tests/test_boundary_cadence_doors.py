"""A boundary cadence a door accepts is one the preparation takes.

``gpuwm domain --cadence N``, the ``[fetch]`` table's config-load check and
``gpuwm fetch`` each accept a cadence and write it into the run: the fetch
downloads that ladder and the companion namelist carries it as
``interval_seconds``.  The decode then holds the fetched series to the
packaged mapping's ``target.boundary_interval_seconds``.  When the two
disagreed the doors said yes, the whole window was downloaded, and the
decode refused it.

These tests hold the doors and the decode to one answer, read from the same
packaged mapping, for every source whose preparation is a packaged mapped
profile.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from gpuwm import fetch, fetch_routes
from gpuwm.cli import main as cli_main
from gpuwm.source_adapters import get_source_adapter
from gpuwm.source_authorities import (
    BOUNDARY_MULTIPLES_KEY,
    boundary_interval_refusal,
    packaged_authorities,
    packaged_mapping_target,
)


def _mapped_front_door_sources() -> list[str]:
    return sorted(
        source for source in fetch.fetch_front_door_sources()
        if get_source_adapter(source).runner == "mapped_composition_v1"
        and get_source_adapter(source).packaged_profile)


def _offered_cadences(source: str) -> tuple[int, ...]:
    """Every cadence the fetch grammar accepts for SOURCE, in hours."""

    if source in fetch_routes.route_ids():
        return tuple(fetch_routes.route_for(source).cadences)
    if not fetch.fetch_accepts_cadence(source):
        return ()
    # A container ladder: every whole-hour spacing with a window on the
    # published hours is a cadence its planner accepts.
    horizon = get_source_adapter(source).max_forecast_hour
    accepted = []
    for cadence in range(1, horizon + 1):
        try:
            fetch.container_forecast_hours(source, cadence, cadence)
        except ValueError:
            continue
        accepted.append(cadence)
    return tuple(accepted)


@pytest.mark.parametrize("source", _mapped_front_door_sources())
def test_every_cadence_a_fetch_offers_is_one_its_preparation_takes(source):
    target = packaged_mapping_target(get_source_adapter(source).packaged_profile)
    offered = _offered_cadences(source)
    refused = {cadence: boundary_interval_refusal(target, cadence * 3600)
               for cadence in offered}
    assert not {c: why for c, why in refused.items() if why is not None}, (
        f"{source}: the fetch offers cadences its preparation refuses")
    for cadence in offered:
        assert fetch.preparation_cadence_refusal(source, cadence) is None


def test_the_gdas_decode_contract_takes_the_three_hour_ladder():
    target = packaged_mapping_target(
        get_source_adapter("gdas").packaged_profile)
    assert target["boundary_interval_seconds"] == 3600
    assert target[BOUNDARY_MULTIPLES_KEY] is True
    for cadence in (1, 2, 3, 9):
        assert boundary_interval_refusal(target, cadence * 3600) is None
    assert boundary_interval_refusal(target, 5400) is not None


def test_the_door_writes_a_cadence_the_gdas_preparation_takes(tmp_path):
    """Through the wizard, then through the prep contract check."""

    from gpuwm import mapped_direct
    from gpuwm.experiment import load_experiment

    out = tmp_path / "coarse.toml"
    rc = cli_main([
        "domain", "--point=35.5,-97.5", "--root-dx", "3", "--card", "5gb",
        "--hours", "3", "--cadence", "3", "--source", "gdas",
        "--cycle", "2023-03-31T18", "--out", str(out)])
    assert rc == 0
    namelist = out.with_name("coarse.namelist.wps").read_text(encoding="utf-8")
    assert "interval_seconds = 10800," in namelist
    mapping = json.loads(packaged_authorities(
        get_source_adapter("gdas").packaged_profile)["mapping"]
        .read_text(encoding="utf-8"))
    receipt = mapped_direct._validate_target_contract(
        mapping, load_experiment(out), 10800, hierarchy=False,
        experiment_config=out)
    assert receipt["boundary_interval_seconds"] == 10800


def test_a_cadence_the_mapping_refuses_is_refused_at_every_door(
        tmp_path, monkeypatch, capsys):
    """A mapping that takes one spacing only is refused before a download."""

    import gpuwm.source_authorities as authorities

    exact = {"boundary_interval_seconds": 3600,
             "require_lateral_boundaries": True}
    monkeypatch.setattr(authorities, "packaged_mapping_target",
                        lambda profile_id: exact)
    with pytest.raises(ValueError) as config_refusal:
        fetch.validate_fetch_hints(
            {"source": "gdas", "hours": 3, "cadence": 3}, source="gdas.toml")
    message = str(config_refusal.value)
    assert "cadence 3" in message and "takes 1 h and no other spacing" in message

    rc = cli_main(["fetch", "--source", "gdas", "--cycle", "2023-03-31T18",
                   "--hours", "3", "--cadence", "3",
                   "--area", "30,-100,40,-90", "--out", str(tmp_path / "out")])
    assert rc == 2
    assert "no other spacing" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()

    out = tmp_path / "refused.toml"
    rc = cli_main([
        "domain", "--point=35.5,-97.5", "--root-dx", "3", "--card", "5gb",
        "--hours", "3", "--cadence", "3", "--source", "gdas",
        "--cycle", "2023-03-31T18", "--out", str(out)])
    assert rc == 2
    assert "no other spacing" in capsys.readouterr().err
    assert not out.exists()


def test_a_bare_gdas_fetch_takes_the_row_spacing():
    """An omitted cadence is the registry row's, so any whole window plans."""

    assert fetch.container_default_cadence("gdas") == 1
    assert fetch.container_default_cadence("gfs") == 3
    assert fetch.container_forecast_hours("gdas", 1) == (0, 1)
    assert fetch.container_forecast_hours("gdas", 3) == (0, 1, 2, 3)
    assert fetch.gdas_forecast_hours(2) == (0, 1, 2)
    fetch.validate_fetch_hints(
        {"source": "gdas", "cycle": "2023-03-31T18", "hours": 1},
        source=fetch.COMMAND_LINE_HINTS)
    # The GFS default is the same number it always was.
    assert fetch.container_forecast_hours("gfs", 6) == (0, 3, 6)
