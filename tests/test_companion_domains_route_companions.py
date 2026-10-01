"""A candidate edit carries every file the route it is on will read.

The domain-edit door writes a NEW configuration beside the one it was
given.  For a configuration on the native regional route that is not one
file: the route reads two namelists and a target-domain document beside
the TOML and refuses the run before any fetch if one of them is absent
(``gpuwm/runplan.py``, the ``route_input_paths`` precheck at the top of
``_hrrr_chain``).  The emission door has written the whole set since the
route landed; the edit door wrote the TOML and the WPS namelist, so
every edit it made to such a configuration -- any action, any grid --
produced a candidate that could not start.

The root grid hid it, and the reason is worth stating because it is why
the report that started this said only nests were affected: the root's
output interval has a SECOND door, ``gpuwm domain --history-interval``,
which re-emits and writes all five files.  A nest interval has only this
one.

The edit door is not the only door that saves a changed copy of a
forecast.  ``gpuwm domain-fit`` and ``gpuwm domain-tiles`` save one too
-- a refitted layout and an automatic tile plan -- and both published
the same short set, so a fitted or tiled copy of a regional forecast was
refused at the same precheck.  The last section here drives those two.

Every assertion here runs the real doors over real bytes.  The only
substitution is the device observation the tile planner prices against,
which is arithmetic over a card and cannot run without one.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
from pathlib import Path

import pytest

from gpuwm import companion_domains
import gpuwm.runplan as runplan_module
from gpuwm.cli import main as cli_main
from gpuwm.experiment import load_experiment
from gpuwm.hrrr_route_inputs import (ROUTE_SHARED_DOMAIN_KEYS,
                                     route_input_paths,
                                     verify_round_trip)
from gpuwm.namelist_import import parse_namelist_text
from gpuwm.runplan import PLAN_SCHEMA, load_plan

#: A small two-domain emission inside the regional source's own grid.
#: The wizard sizes the layout itself; the values here only have to be a
#: chain it accepts on a declared card.
_EMISSION = ("--cycle", "2026-07-29T18", "--hours", "1", "--root-dx", "3",
             "--chain", "3", "--card", "32gb", "--point", "38.0,-98.0")


def _emit(tmp_path, source="hrrr", name="base"):
    """The emission door, run for real, with no network and no device."""

    out = tmp_path / f"{name}.toml"
    with contextlib.redirect_stdout(io.StringIO()):
        rc = cli_main(["domain", "--name", name, "--source", source,
                       *_EMISSION, "--out", str(out)])
    assert rc == 0
    return out


def _edit(config, output, action):
    """The edit door, through its own request document."""

    return companion_domains.edit_configuration({
        "schema": companion_domains.REQUEST_SCHEMA,
        "config_path": str(config),
        "expected_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "output_path": str(output),
        "action": action,
    })


def _interval_column(namelist_input: Path):
    """The per-domain output cadence the route's own namelist declares.

    ``render_namelist_input`` writes ``history_interval_s`` -- SECONDS,
    one whole number per domain -- not stock WRF's ``history_interval``
    in minutes, so the value asserted here is the value the edit set.
    """
    tables = parse_namelist_text(namelist_input.read_text(encoding="utf-8"))
    return [int(value)
            for value in tables["time_control"]["history_interval_s"]]


def _chain_reaches_fetch(tmp_path, config, name, monkeypatch):
    """Drive the route's real precheck and stop at the fetch seam.

    The precheck sits above the fetch stage, so a plan that gets past it
    reaches the seam below and a plan that does not never reaches it.
    The seam raises rather than downloading, so neither answer needs a
    network.
    """

    class _Reached(Exception):
        pass

    plan_path = tmp_path / f"plan-{name}.json"
    plan_path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": name, "route": "prepared",
        "config": {"path": str(config)},
        "output_root": str(tmp_path / f"run-{name}")}), encoding="utf-8")
    plan = load_plan(plan_path)

    class _Observer:
        def __getattr__(self, attribute):
            return lambda *args, **kwargs: None

    def _refuse_to_fetch(*args, **kwargs):
        raise _Reached

    monkeypatch.setattr(runplan_module, "_run_fetch", _refuse_to_fetch)
    with contextlib.redirect_stdout(io.StringIO()):
        try:
            runplan_module._hrrr_chain(
                plan, config_path=Path(config),
                exp=load_experiment(config), observer=_Observer(),
                run_dir=plan.run_dir)
        except _Reached:
            return True, ""
        except runplan_module.PlanError as refusal:
            return False, str(refusal)
    return True, ""


def test_a_nest_output_edit_leaves_a_candidate_the_route_can_read(tmp_path):
    base = _emit(tmp_path)
    assert load_experiment(base).domains[1].history_interval_s == 900.0

    candidate = tmp_path / "nest-interval.toml"
    result = _edit(base, candidate, {
        "kind": "set_output", "grid_id": 2,
        "history_interval_s": 300.0, "restart_interval_s": 3600.0})

    assert result["changes"] == [{"field": "domain[1].history_interval_s",
                                  "before": 900.0, "after": 300.0}]
    paths = route_input_paths(candidate)
    absent = sorted(role for role, path in paths.items()
                    if not path.is_file())
    assert absent == []
    assert set(result["route_companions"]) == {str(path)
                                               for path in paths.values()}

    edited = load_experiment(candidate)
    verify_round_trip(edited, paths["wps_namelist"], paths["namelist_input"])
    assert _interval_column(paths["namelist_input"]) == [3600, 300]
    assert _interval_column(paths["stock_namelist_input"]) == [3600, 300]


def test_a_root_output_edit_leaves_the_same_complete_candidate(tmp_path):
    base = _emit(tmp_path)
    candidate = tmp_path / "root-interval.toml"
    result = _edit(base, candidate, {
        "kind": "set_output", "grid_id": 1,
        "history_interval_s": 900.0, "restart_interval_s": 3600.0})

    assert result["changes"] == [{"field": "domain[0].history_interval_s",
                                  "before": 3600.0, "after": 900.0}]
    paths = route_input_paths(candidate)
    assert all(path.is_file() for path in paths.values())
    assert _interval_column(paths["namelist_input"]) == [900, 900]


def test_the_route_precheck_accepts_an_edited_candidate(tmp_path, monkeypatch):
    base = _emit(tmp_path)
    candidate = tmp_path / "edited.toml"
    _edit(base, candidate, {"kind": "set_output", "grid_id": 2,
                            "history_interval_s": 300.0,
                            "restart_interval_s": 3600.0})
    reached, refusal = _chain_reaches_fetch(
        tmp_path, candidate, "edited", monkeypatch)
    assert reached, refusal


def test_a_candidate_missing_a_companion_runs_the_set_its_configuration_renders(
        tmp_path, monkeypatch):
    """The precheck that reported this defect is retired, not weakened.

    It refused a configuration with an incomplete set beside it, and it
    did so three seconds into the real run, after `gpuwm go --dry-run`
    had passed it.  Since 2.8.1 a run with no complete set beside its
    configuration writes the whole set from the configuration into its
    own run folder, through the same writer and round trip the doors
    use, and a configuration that set cannot carry is refused at the
    door (tests/test_hrrr_route_bare_configuration.py).  What the
    precheck protected still holds: the run never mixes a partial set
    with rendered files, and what it runs is the door's set, byte for
    byte.
    """

    base = _emit(tmp_path)
    candidate = tmp_path / "stripped.toml"
    _edit(base, candidate, {"kind": "set_output", "grid_id": 2,
                            "history_interval_s": 300.0,
                            "restart_interval_s": 3600.0})
    beside = route_input_paths(candidate)
    door = {role: path.read_bytes() for role, path in beside.items()}
    beside["namelist_input"].unlink()
    reached, refusal = _chain_reaches_fetch(
        tmp_path, candidate, "stripped", monkeypatch)
    assert reached, refusal
    rendered, = (tmp_path / "run-stripped").rglob(
        f"route-inputs/{beside['namelist_input'].name}")
    written = route_input_paths(rendered.parent / candidate.name)
    for role, content in door.items():
        assert written[role].read_bytes() == content, role
    assert not beside["namelist_input"].exists()


#: Every kind the door advertises, named here so the sweep below is a
#: list a reader can count rather than a dictionary order.
_KINDS = ("add_nest", "move_domain", "remove_nest", "resize_domain",
          "resize_domain_cells", "resize_domain_edges", "set_activation",
          "set_follow", "set_output", "set_physics", "set_placement",
          "set_spawn", "set_targets", "set_tiles")


def _actions(exp):
    """One valid request per kind the door advertises.

    Derived from the emitted experiment rather than from constants, so a
    wizard that sizes the layout differently does not turn this sweep
    into a set of refusals that pass for coverage.
    """
    nest = exp.domain(2)
    center = exp.projection.ref_lat, exp.projection.ref_lon
    later = exp.start_time.replace(microsecond=0).isoformat() + "Z"
    return {
        "add_nest": {"kind": "add_nest", "parent_id": nest.grid_id,
                     "nx": 90, "ny": 90, "parent_grid_ratio": 3,
                     "parent_time_step_ratio": 3,
                     "history_interval_s": 120.0,
                     "placement": {"kind": "parent_cells",
                                   "i_parent_start": 20,
                                   "j_parent_start": 20}},
        "remove_nest": {"kind": "remove_nest", "grid_id": nest.grid_id,
                        "include_children": False},
        "move_domain": {"kind": "move_domain", "grid_id": nest.grid_id,
                        "latitude": center[0] + 0.1,
                        "longitude": center[1] + 0.1},
        "resize_domain": {"kind": "resize_domain", "grid_id": nest.grid_id,
                          "bounds": {"south": center[0] - 1.2,
                                     "west": center[1] - 1.2,
                                     "north": center[0] + 1.2,
                                     "east": center[1] + 1.2}},
        "resize_domain_edges": {
            "kind": "resize_domain_edges", "grid_id": nest.grid_id,
            "handle": "e",
            "start": {"latitude": center[0], "longitude": center[1] + 1.0},
            "end": {"latitude": center[0], "longitude": center[1] + 0.8}},
        "resize_domain_cells": {
            "kind": "resize_domain_cells", "grid_id": nest.grid_id,
            "handle": "e", "nx": nest.run.nx - 60, "ny": nest.run.ny},
        "set_output": {"kind": "set_output", "grid_id": nest.grid_id,
                       "history_interval_s": 300.0,
                       "restart_interval_s": 3600.0},
        "set_tiles": {"kind": "set_tiles", "mode": "off"},
        "set_placement": {"kind": "set_placement", "grid_id": nest.grid_id,
                          "placement": {
                              "kind": "parent_cells",
                              "i_parent_start": nest.i_parent_start + 2,
                              "j_parent_start": nest.j_parent_start + 2}},
        "set_activation": {"kind": "set_activation",
                           "grid_id": nest.grid_id, "mode": "immediate"},
        "set_follow": {"kind": "set_follow", "grid_id": nest.grid_id,
                       "settings": None},
        "set_spawn": {"kind": "set_spawn", "grid_id": nest.grid_id,
                      "settings": {"trigger": "time", "at_s": 900.0}},
        "set_targets": {"kind": "set_targets", "grid_id": nest.grid_id,
                        "points": [], "max_move_parent_cells": 6,
                        "min_overlap_fraction": 0.7},
        "set_physics": {"kind": "set_physics", "grid_id": nest.grid_id,
                        "settings": {"diff_6th_factor": 0.11}},
    }


def test_the_sweep_below_covers_every_kind_the_door_advertises(tmp_path):
    """A kind added to the door without a row here is a gap, not a pass."""

    base = _emit(tmp_path)
    advertised = set(companion_domains.capabilities()["actions"])
    assert advertised == set(_KINDS) == set(_actions(load_experiment(base)))


@pytest.mark.parametrize("kind", _KINDS)
def test_every_edit_kind_leaves_a_candidate_the_route_can_read(tmp_path, kind):
    base = _emit(tmp_path)
    candidate = tmp_path / f"{kind}.toml"
    result = _edit(base, candidate, _actions(load_experiment(base))[kind])
    paths = route_input_paths(candidate)
    absent = sorted(role for role, path in paths.items()
                    if not path.is_file())
    assert absent == []
    assert set(result["route_companions"]) == {str(path)
                                               for path in paths.values()}
    verify_round_trip(load_experiment(candidate), paths["wps_namelist"],
                      paths["namelist_input"])


def test_the_edit_door_refuses_to_replace_an_existing_companion(tmp_path):
    base = _emit(tmp_path)
    candidate = tmp_path / "occupied.toml"
    route_input_paths(candidate)["namelist_input"].write_text(
        "earlier work", encoding="utf-8")
    with pytest.raises((FileExistsError, ValueError)):
        _edit(base, candidate, {"kind": "set_output", "grid_id": 2,
                                "history_interval_s": 300.0,
                                "restart_interval_s": 3600.0})
    assert not candidate.exists()
    assert route_input_paths(candidate)["namelist_input"].read_text(
        encoding="utf-8") == "earlier work"


def test_a_candidate_off_the_native_route_keeps_only_its_wps_namelist(
        tmp_path):
    """The other routes read the TOML; nothing new is written for them."""

    base = _emit(tmp_path, source="gfs", name="global")
    candidate = tmp_path / "global-edit.toml"
    _edit(base, candidate, {"kind": "set_output", "grid_id": 2,
                            "history_interval_s": 300.0,
                            "restart_interval_s": 3600.0})
    paths = route_input_paths(candidate)
    assert paths["wps_namelist"].is_file()
    assert not paths["namelist_input"].exists()
    assert not paths["stock_namelist_input"].exists()
    assert not paths["target_domain"].exists()


#: A synthetic card for the tile planner: real planner arithmetic over a
#: declared capacity, and no CUDA context anywhere.  ``domain-tiles``
#: takes no capacity flag -- it plans for the machine in front of it --
#: so a box with no device cannot drive that door at all without this.
@pytest.fixture
def tile_machine(monkeypatch):
    from gpuwm import domain_wizard as dw
    from gpuwm.core import preflight, streaming
    from tilestream import autoplan

    probe = dict(total_bytes=32 * dw.GIB, free_bytes=int(30.0 * dw.GIB),
                 profile=dict(name="NVIDIA GeForce RTX 5090",
                              multiprocessor_count=170,
                              max_threads_per_multiprocessor=1536,
                              default_stack_limit_bytes=1024,
                              bare_context_bytes=182452224))
    monkeypatch.setattr(dw, "device_memory_probe_subprocess", lambda: probe)
    monkeypatch.setattr(dw, "device_memory_probe_reason",
                        lambda: "declared capacity, no device contact")
    monkeypatch.setattr(preflight, "device_physical_total_bytes",
                        lambda: probe["total_bytes"])
    monkeypatch.setattr(preflight, "live_device_local_memory_profile",
                        lambda: preflight.profile_from_device_probe(probe))
    monkeypatch.setattr(streaming, "_host_total_bytes", lambda: 64 * dw.GIB)
    monkeypatch.setattr(preflight, "host_available_bytes", lambda: 48 * dw.GIB)
    monkeypatch.setattr(autoplan.Machine, "detect", classmethod(
        lambda cls, **kwargs: pytest.fail("Planning must not create a CUDA context")))
    return probe


def _fit(base, out, *arguments):
    """The fit door, run for real on a declared card."""

    with contextlib.redirect_stdout(io.StringIO()) as printed:
        rc = cli_main(["domain-fit", str(base), "--point", "38.0,-98.0",
                       "--card", "32gb", "--out", str(out), "--write",
                       *arguments])
    assert rc == 0, printed.getvalue()
    return printed.getvalue()


def _tiles(base, out, *arguments):
    """The tile door, run for real against the declared card above."""

    with contextlib.redirect_stdout(io.StringIO()) as printed:
        rc = cli_main(["domain-tiles", str(base), "--out", str(out),
                       "--mode", "on", "--write", *arguments])
    assert rc == 0, printed.getvalue()
    return printed.getvalue()


def _receipt(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_fitted_copy_carries_the_route_files(tmp_path, monkeypatch):
    """A refitted copy is a saved copy, so it answers the same question."""

    base = _emit(tmp_path)
    fitted = tmp_path / "fitted.toml"
    printed = _fit(base, fitted)

    paths = route_input_paths(fitted)
    absent = sorted(role for role, path in paths.items() if not path.is_file())
    assert absent == []
    receipt = _receipt(fitted.with_suffix(".fit.json"))
    assert set(receipt["route_companions"]) == {str(path)
                                                for path in paths.values()}
    for path in paths.values():
        assert str(path) in printed

    verify_round_trip(load_experiment(fitted), paths["wps_namelist"],
                      paths["namelist_input"])
    reached, refusal = _chain_reaches_fetch(
        tmp_path, fitted, "fitted", monkeypatch)
    assert reached, refusal


def test_a_fitted_copy_declares_the_layout_it_just_fitted(tmp_path):
    """The namelists are rendered from the fitted tables, not copied."""

    base = _emit(tmp_path)
    fitted = tmp_path / "refitted.toml"
    _fit(base, fitted, "--hours", "2")

    edited = load_experiment(fitted)
    paths = route_input_paths(fitted)
    tables = parse_namelist_text(paths["namelist_input"].read_text(
        encoding="utf-8"))
    assert [int(value) for value in tables["domains"]["e_we"]] == [
        domain.run.nx + 1 for domain in edited.domains]
    assert _interval_column(paths["namelist_input"]) == [
        int(domain.history_interval_s) for domain in edited.domains]


def test_a_fitted_copy_off_the_native_route_keeps_only_its_wps_namelist(
        tmp_path):
    base = _emit(tmp_path, source="gfs", name="global")
    fitted = tmp_path / "global-fit.toml"
    _fit(base, fitted)
    paths = route_input_paths(fitted)
    assert paths["wps_namelist"].is_file()
    assert not paths["namelist_input"].exists()
    assert not paths["stock_namelist_input"].exists()
    assert not paths["target_domain"].exists()


def test_the_fit_door_refuses_to_replace_an_existing_route_companion(
        tmp_path):
    base = _emit(tmp_path)
    fitted = tmp_path / "occupied-fit.toml"
    route_input_paths(fitted)["namelist_input"].write_text(
        "earlier work", encoding="utf-8")
    with contextlib.redirect_stdout(io.StringIO()):
        assert cli_main(["domain-fit", str(base), "--point", "38.0,-98.0",
                         "--card", "32gb", "--out", str(fitted),
                         "--write"]) != 0
    assert not fitted.exists()
    assert route_input_paths(fitted)["namelist_input"].read_text(
        encoding="utf-8") == "earlier work"


def test_a_tiled_copy_carries_the_route_files(tmp_path, tile_machine,
                                              monkeypatch):
    """An automatic tile copy is a saved copy, so it answers it too."""

    base = _emit(tmp_path)
    tiled = tmp_path / "tiled.toml"
    _tiles(base, tiled)

    paths = route_input_paths(tiled)
    absent = sorted(role for role, path in paths.items() if not path.is_file())
    assert absent == []
    receipt = _receipt(tiled.with_suffix(".tiles.json"))
    assert set(receipt["route_companions"]) == {str(path)
                                                for path in paths.values()}
    verify_round_trip(load_experiment(tiled), paths["wps_namelist"],
                      paths["namelist_input"])
    reached, refusal = _chain_reaches_fetch(
        tmp_path, tiled, "tiled", monkeypatch)
    assert reached, refusal


def test_a_tiled_copy_off_the_native_route_keeps_only_its_wps_namelist(
        tmp_path, tile_machine):
    base = _emit(tmp_path, source="gfs", name="global")
    tiled = tmp_path / "global-tiles.toml"
    _tiles(base, tiled)
    paths = route_input_paths(tiled)
    assert paths["wps_namelist"].is_file()
    assert not paths["namelist_input"].exists()
    assert not paths["stock_namelist_input"].exists()
    assert not paths["target_domain"].exists()


def test_an_authored_following_nest_configuration_carries_the_route_files(
        tmp_path):
    """The authoring door publishes a configuration on any planable source.

    It is not an edit and not a copy, but it publishes the same pair the
    short doors published -- the configuration and its WPS namelist --
    and it accepts the native regional source, so the configuration it
    authored was refused at the same precheck.
    """
    import argparse

    from gpuwm import cyclone_setup

    out = tmp_path / "authored.toml"
    parser = argparse.ArgumentParser()
    cyclone_setup.register_cli(parser.add_subparsers(dest="command",
                                                     required=True))
    arguments = parser.parse_args([
        "cyclone-setup", "--source", "hrrr", "--cycle", "2026072918",
        "--point=38.5,-97.5", "--hours", "6", "--name", "authored",
        "--card", "32gb", "--out", str(out)])
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        rc = cyclone_setup.main(arguments)
    result = json.loads(printed.getvalue())
    assert rc == 0, result.get("error")
    assert result["created"] and not result["forecast_started"]

    paths = route_input_paths(out)
    absent = sorted(role for role, path in paths.items() if not path.is_file())
    assert absent == []
    receipt = _receipt(out.with_suffix(".cyclone.json"))
    assert set(receipt["route_companions"]) == {str(path)
                                                for path in paths.values()}
    verify_round_trip(load_experiment(out), paths["wps_namelist"],
                      paths["namelist_input"])
#: One value per key that the emission does not already carry, so each
#: edit below is a real change.  The WAY OUT is deliberately not written
#: here: it is read off the refusal the door raises, because a remedy
#: copied into a test is a remedy nobody measured.
_PROBE_VALUE = {"bldt": 0.5, "diff_6th_opt": 0, "isfflx": 0,
                "mp_physics": 6, "sf_sfclay_physics": 1}


def _all_domains_publishes(base, tmp_path, settings, name):
    """Does this edit, at every domain, leave a candidate that runs?

    The real door over real bytes, which is the only thing that settles
    whether a remedy is a remedy.
    """

    candidate = tmp_path / f"{name}.toml"
    try:
        _edit(base, candidate, {"kind": "set_physics", "grid_id": 0,
                                "settings": dict(settings)})
    except (ValueError, NotImplementedError):
        return False
    return sorted(role for role, path in route_input_paths(candidate).items()
                  if not path.is_file()) == []


def test_every_route_shared_key_is_one_the_door_takes_per_domain():
    """A key the schema does not scope per domain needs no route reading."""

    from gpuwm.case_catalog import _native_contract

    _, _, domain_keys = _native_contract()
    assert set(ROUTE_SHARED_DOMAIN_KEYS) <= set(domain_keys)
    assert set(_PROBE_VALUE) == set(ROUTE_SHARED_DOMAIN_KEYS)


def test_a_microphysics_change_on_one_domain_is_refused_with_its_way_out(
        tmp_path):
    """The reported shape: one grid selected, a per-domain key changed.

    The route reads one microphysics column for the whole tree, so this
    edit cannot be written into the files the run reads.  It is refused
    at the door, in the sentence the door already uses for a tree-wide
    setting, rather than in the importer's words after the candidate has
    been rendered.
    """

    base = _emit(tmp_path)
    candidate = tmp_path / "microphysics.toml"
    with pytest.raises(ValueError) as refusal:
        _edit(base, candidate, {"kind": "set_physics", "grid_id": 1,
                                "settings": {"mp_physics": 6}})
    assert "mp_physics" in str(refusal.value)
    assert "Select All domains to change them" in str(refusal.value)
    assert not candidate.exists()
    assert not route_input_paths(candidate)["namelist_input"].exists()


def test_the_same_microphysics_change_still_publishes_on_another_route(
        tmp_path):
    """Another route reads the configuration, so it keeps the per-domain edit."""

    base = _emit(tmp_path, source="gfs", name="global")
    candidate = tmp_path / "global-microphysics.toml"
    result = _edit(base, candidate, {"kind": "set_physics", "grid_id": 1,
                                     "settings": {"mp_physics": 6}})

    assert result["created"]
    assert load_experiment(candidate).domain(1).run.mp_physics == 6
    assert load_experiment(candidate).domain(2).run.mp_physics != 6
    paths = route_input_paths(candidate)
    assert paths["wps_namelist"].is_file()
    assert not paths["namelist_input"].exists()


@pytest.mark.parametrize("key", ROUTE_SHARED_DOMAIN_KEYS)
def test_a_route_shared_key_is_refused_on_a_nest_with_its_way_out(
        tmp_path, key):
    base = _emit(tmp_path)
    candidate = tmp_path / f"{key}-nest.toml"
    with pytest.raises(ValueError) as refusal:
        _edit(base, candidate, {"kind": "set_physics", "grid_id": 2,
                                "settings": {key: _PROBE_VALUE[key]}})
    assert key in str(refusal.value)
    assert "Select All domains to change them" in str(refusal.value)
    assert not candidate.exists()


@pytest.mark.parametrize("key", ROUTE_SHARED_DOMAIN_KEYS)
def test_the_way_out_the_refusal_names_publishes_a_candidate_that_runs(
        tmp_path, key):
    """The remedy is FOLLOWED, not restated.

    The refusal carries the tree-wide edit the door drove and found
    publishing, and names every field of it in the sentence a reader
    reads.  Doing exactly that has to leave a candidate the route can
    read; anything else is a way out that does not work, told to a
    reader as though it had been tried.
    """

    base = _emit(tmp_path)
    with pytest.raises(companion_domains.DomainScopeError) as refusal:
        _edit(base, tmp_path / f"{key}-nest.toml",
              {"kind": "set_physics", "grid_id": 2,
               "settings": {key: _PROBE_VALUE[key]}})

    remedy = refusal.value.all_domains_settings
    assert remedy is not None and remedy[key] == _PROBE_VALUE[key]
    for field in remedy:
        assert field in str(refusal.value)

    candidate = tmp_path / f"{key}-all.toml"
    result = _edit(base, candidate, {"kind": "set_physics", "grid_id": 0,
                                     "settings": remedy})

    assert result["created"]
    edited = load_experiment(candidate)
    for field, value in remedy.items():
        assert [getattr(domain.run, field) for domain in edited.domains] == [
            value for _ in edited.domains]
    paths = route_input_paths(candidate)
    assert sorted(role for role, path in paths.items()
                  if not path.is_file()) == []
    verify_round_trip(edited, paths["wps_namelist"], paths["namelist_input"])


def test_the_second_step_is_named_only_where_the_route_needs_one(tmp_path):
    """Which keys need more than the scope, measured one door call each.

    Three of the five are carried tree-wide by the namelist columns
    alone.  The other two select a physics suite, and this route states
    a runtime switch for the suite its namelists have no key for, so the
    tree-wide edit is refused until that switch is stated with it.
    Pinned as a measurement because it is the case the remedy used to
    promise its way past.
    """

    base = _emit(tmp_path)
    beyond = {}
    for key, value in _PROBE_VALUE.items():
        with pytest.raises(companion_domains.DomainScopeError) as refusal:
            _edit(base, tmp_path / f"{key}-scope.toml",
                  {"kind": "set_physics", "grid_id": 2,
                   "settings": {key: value}})
        beyond[key] = sorted(set(refusal.value.all_domains_settings) - {key})

    assert beyond == {"bldt": [], "diff_6th_opt": [], "isfflx": [],
                      "mp_physics": ["moist_cq"],
                      "sf_sfclay_physics": ["moist_cq"]}
    for key in ("mp_physics", "sf_sfclay_physics"):
        assert not _all_domains_publishes(base, tmp_path,
                                          {key: _PROBE_VALUE[key]},
                                          f"{key}-alone")


def test_a_per_domain_key_the_route_does_carry_still_reaches_its_namelist(
        tmp_path):
    """The reverse control: this door did not become a tree-wide door."""

    base = _emit(tmp_path)
    candidate = tmp_path / "per-domain.toml"
    result = _edit(base, candidate, {"kind": "set_physics", "grid_id": 2,
                                     "settings": {"radt": 6.0}})

    assert result["created"]
    paths = route_input_paths(candidate)
    tables = parse_namelist_text(paths["namelist_input"].read_text(
        encoding="utf-8"))
    assert [float(value) for value in tables["physics"]["radt"]][1] == 6.0
    assert [float(value) for value in tables["physics"]["radt"]][0] != 6.0
    verify_round_trip(load_experiment(candidate), paths["wps_namelist"],
                      paths["namelist_input"])


def test_the_physics_panel_calls_a_route_shared_key_a_scope_refusal(tmp_path):
    """What the panel shows for the cell, measured at the panel's door.

    The panel greys a cell with the door's own sentence and the remedy
    that sentence offers.  Reported as an ordinary combination refusal,
    the same cell would carry no remedy at all.
    """

    import hashlib

    from gpuwm import companion_physics

    base = _emit(tmp_path)
    answer = companion_physics.availability({
        "schema": companion_domains.REQUEST_SCHEMA,
        "config_path": str(base),
        "expected_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
        "action": {"kind": "set_physics", "grid_id": 2,
                   "settings": {"mp_physics": 6}}})

    reasons = answer["draft"]["reasons"]
    assert [reason["kind"] for reason in reasons][:1] == [
        companion_physics.SHARED_SCOPE]
    assert "Select All domains to change them" in " ".join(
        (reason["reason"] + " " + (reason["detail"] or ""))
        for reason in reasons)


def _availability(config, action):
    """The physics panel, through its own request document."""

    import hashlib

    from gpuwm import companion_physics

    return companion_physics.availability({
        "schema": companion_domains.REQUEST_SCHEMA,
        "config_path": str(config),
        "expected_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "action": action})


@pytest.mark.parametrize("key", ROUTE_SHARED_DOMAIN_KEYS)
def test_the_panel_claims_the_all_domains_remedy_only_where_it_works(
        tmp_path, key):
    """The remedy field is a measurement, so it has to match the door.

    A front end reads it as "this was tried and it worked" and prints
    "select All domains to use this" on the strength of it.  It was
    emitted for all five of these keys; for the two that select a
    physics suite, taking that advice met the route importer's words
    about a runtime switch instead of a saved file.  The claim and the
    door are driven here in the same test so they cannot answer
    differently.
    """

    from gpuwm import companion_physics

    base = _emit(tmp_path)
    reasons = _availability(base, {"kind": "set_physics", "grid_id": 2,
                                   "settings": {key: _PROBE_VALUE[key]}}
                            )["draft"]["reasons"]
    works = _all_domains_publishes(base, tmp_path, {key: _PROBE_VALUE[key]},
                                   f"{key}-panel")

    assert reasons[0]["kind"] == companion_physics.SHARED_SCOPE
    assert reasons[0]["closes"]
    if works:
        assert reasons[0]["remedy"] == companion_physics.ALL_DOMAINS
        assert len(reasons) == 1
    else:
        assert reasons[0]["remedy"] is None
        assert len(reasons) > 1
        assert "moist_cq" in " ".join(
            reason["reason"] + " " + (reason["detail"] or "")
            for reason in reasons[1:])


def test_every_all_domains_remedy_a_cell_carries_is_driven(tmp_path):
    """Every promise the panel makes about a cell, through the save door.

    A front end prints "select All domains to use this" on the strength
    of this field, so each one is taken up here exactly as a reader
    would: the draft plus that option, at every domain, through the
    door that saves.  Before the renderer was asked, cells carried it
    for edits whose save met the route importer's words.
    """

    from gpuwm import companion_physics

    base = _emit(tmp_path)
    action = {"kind": "set_physics", "grid_id": 2,
              "settings": {"mp_physics": 6}}
    installed = {component["id"]: {option["id"]: option["settings"]
                                   for option in component["options"]}
                 for component in companion_domains.physics_components()}
    answer = _availability(base, action)
    claimed = [(component["id"], option["id"])
               for component in answer["components"]
               for option in component["options"]
               if any(reason["remedy"] == companion_physics.ALL_DOMAINS
                      for reason in option["reasons"])]

    assert claimed, "the panel claims this remedy somewhere, or nothing is held"
    for component_id, option_id in claimed:
        settings = dict(action["settings"], **installed[component_id][option_id])
        name = "".join(character if character.isalnum() else "-"
                       for character in f"{component_id}-{option_id}")
        assert _all_domains_publishes(base, tmp_path, settings, name), (
            f"{component_id}/{option_id} carries the All domains remedy "
            f"and {settings} at every domain does not publish")
