"""Capacity, refusal and launch contracts for resident transported smoke."""
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import copy
import tomllib

import pytest

from gpuwm.core.devices import DeviceOptions, DevicesRefused, validate_ranked_physics
from gpuwm.experiment import load_experiment
from gpuwm.smoke_execution import (
    HRRR_SMOKE_SETTINGS, apply_smoke_settings, at_launch, capacity_reason, receipt)
from gpuwm.toml_document import emit_experiment_toml

RECIPE = Path(__file__).resolve().parents[1] / 'gpuwm/data/recipes/hrrr_v4_gsd41.toml'


def _smoke_run():
    """The recipe's run with --smoke's settings (2.9.4 ships smoke off)."""
    return replace(load_experiment(RECIPE).root.run, **HRRR_SMOKE_SETTINGS)


@pytest.mark.parametrize('sets', ['dust', 'cams_aq', 'gocart_primary', 'smoke,dust'])
def test_other_chemistry_keeps_named_refusal(sets):
    cfg = replace(_smoke_run(), chem_sets=sets)
    with pytest.raises(DevicesRefused, match='chemistry|chem_sets'):
        validate_ranked_physics(cfg)


def test_ranked_smoke_passes_physics_admission():
    validate_ranked_physics(_smoke_run())


@pytest.mark.parametrize('mode,physics_bytes', [('on', 0), ('off', 0), ('auto', 0), ('off', 2)])
def test_capacity_fallback_is_per_run_and_keeps_recipe(tmp_path, monkeypatch, capsys, mode, physics_bytes):
    from gpuwm.core import preflight
    original = RECIPE.read_bytes()
    # A config with smoke on (as --smoke writes it): the capacity fallback
    # still applies on top of the opt-in.
    raw = apply_smoke_settings(tomllib.loads(original.decode()))
    raw['tiles'] = {'mode':mode}
    # A crop keeps config emission quick; force its one-card price over free.
    raw['domain'][0].update(nx=96, ny=80)
    config = tmp_path / 'original.toml'
    config.write_bytes(emit_experiment_toml(raw).encode())
    monkeypatch.setattr(preflight, 'device_memory_probe_subprocess', lambda: {'free_bytes':1})
    monkeypatch.setattr(preflight, 'profile_from_device_probe', lambda p: None)
    monkeypatch.setattr(preflight, 'estimate_experiment', lambda exp, **kw:
        SimpleNamespace(peak_envelope_bytes=2 if exp.root.run.chem_sets else physics_bytes))
    args = SimpleNamespace(config=config, outdir=tmp_path/'run', devices=None, wps_namelist=None)
    path, off = at_launch(args, config, copy.deepcopy(raw))
    assert path != config
    effective = load_experiment(path)
    assert effective.root.run.chem_sets == ''
    assert effective.root.run.simple_dir_fdb == 0
    assert effective.tiles.mode == ('auto' if mode == 'off' and physics_bytes == 2 else mode)
    if mode == 'off' and physics_bytes == 2:
        assert 'automatic tiling is selected' in receipt(effective.root.run)['smoke_off']
    assert 'smoke off for this run:' in capsys.readouterr().out
    assert 'run with --devices 2 to carry it' in receipt(effective.root.run)['smoke_off']
    assert RECIPE.read_bytes() == original
    assert tomllib.loads(config.read_text())['shared']['chem_sets'] == 'smoke'
    # The route's emitted namelist is also chemistry off, so preparation
    # cannot restore smoke by importing an old companion.
    assert path.with_suffix('.namelist.wps').is_file()
    native = path.with_suffix('.namelist.input')
    if native.is_file():
        assert 'chem_opt = 18' not in native.read_text()
    # Smoke off is chemistry off: the recipe's chem transport switches are
    # back at their defaults, so the HRRR route's emitted pair reproduces the
    # config (2026-10-21: "d01 chem_adv_opt: config 2 vs namelist 1;
    # chem_mix6_off: config True vs namelist False; smoke_off_reason ...").
    assert (effective.root.run.chem_adv_opt, effective.root.run.chem_mix6_off) == (1, False)
    from gpuwm.hrrr_route_inputs import route_input_paths, verify_round_trip
    paths = route_input_paths(path)
    if paths['wps_namelist'].is_file() and paths['namelist_input'].is_file():
        verify_round_trip(effective, paths['wps_namelist'], paths['namelist_input'])
    # Preparation wire settings preserve the reason and the run manifest
    # carries the same sentence a user saw before downloading anything.
    from gpuwm.hrrr_configuration import resolved_run_settings
    assert resolved_run_settings(effective.root.run)['smoke_off_reason']
    from gpuwm.runplan import build_plan, write_manifest, PLAN_SCHEMA
    import json
    plan = build_plan({'schema':PLAN_SCHEMA, 'name':'capacity fallback', 'route':'prepared',
        'config':{'path':str(path)}, 'output_root':str(tmp_path/'manifest')},
        source='test', base_dir=tmp_path, sha256='0'*64)
    root = tmp_path/'manifest'
    root.mkdir()
    manifest = write_manifest(plan, run_dir=root, events_path=root/'events.jsonl',
        run_id='smoke-capacity', started_at_utc='2026-10-21T16:00:00Z')
    assert json.loads(manifest.read_text())['smoke_off'] == receipt(effective.root.run)['smoke_off']


def test_ranked_or_unknown_card_never_disables_smoke():
    exp = load_experiment(RECIPE)
    exp = replace(exp, domains=tuple(replace(d, run=replace(d.run, **HRRR_SMOKE_SETTINGS))
                                     for d in exp.domains))
    assert capacity_reason(exp) is None
    exp = replace(exp, devices=DeviceOptions(count=2))
    assert capacity_reason(exp, probe={'free_bytes':1}) is None


def test_go_tiles_reaches_launch_with_chemistry_off(tmp_path, monkeypatch, capsys):
    from gpuwm import go_cli
    from gpuwm.cli import main
    captured = {}

    def launch(args, *, config, payload, **kw):
        captured.update(path=config, config=load_experiment(config), payload=payload)
        return 0

    monkeypatch.setattr(go_cli, '_registered_launch', launch)
    assert main(['go', str(RECIPE), '--smoke', '--tiles', 'on', '--outdir', str(tmp_path/'run'),
                 '--wps-namelist', str(tmp_path/'input.namelist.wps')]) == 0
    assert captured['config'].root.run.chem_sets == ''
    assert captured['config'].tiles.mode == 'on'
    assert captured['path'] != RECIPE
    assert receipt(captured['config'].root.run)['smoke_off'] in capsys.readouterr().out
