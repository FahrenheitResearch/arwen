"""CPU contracts for the generated local cycling configuration."""
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import tomllib

import pytest

from gpuwm.local_da import (
    Card, Request, PlanError, build_plan, configuration, publish,
    derive_rung, main, price_rung, protocol_document, register_cli, SCHEMA,
    ALTERNATIVE_FIELDS, FALLBACK_REFUSAL_CODE, LAUNCH_REFUSAL_CODES,
    PRE_EXECUTION_FUNCTIONS, REFUSAL_CODES, REFUSAL_EFFECTS, REFUSAL_SOURCES,
    REMEDY_CADENCE, REMEDY_CARD_MEMORY, REMEDY_FORCING, REMEDY_HOST_MEMORY,
    REMEDY_TIME_BUDGET,
    REVIEW_REFUSAL_CODES, RUN_REFUSAL_CODES,
)

EPOCH = '2026-09-10T12:00:00Z'


def availability(*args):
    return [{'id': 'radar', 'status': 'candidate', 'reason': 'test route'},
            {'id': 'surface', 'status': 'candidate', 'reason': 'test route'}]


def price(request, rung, experiment, streams):
    # Deliberately monotonic, so the fitter can be checked independently
    # of the installed radiation tables and native libraries.
    n = rung['members']
    cells = rung['nx'] * rung['ny'] * experiment.domains[0].run.nz
    return dict(forecast_peak_bytes=int(cells * 4000),
                analysis_peak_bytes=int(cells * max(n, 8) * 40),
                host_peak_bytes=int(cells * max(n, 8) * 100),
                cycle_seconds=40. * n / request.card.speed_factor,
                forecast_seconds=80. * n / request.card.speed_factor,
                preparation_seconds=30., basis=['injected test price'],
                measured=False, observation_slots=405,
                solve_memory_mib=128)


def request(**kwargs):
    return Request(epoch=EPOCH, point=(40., -100.), card=Card(vram_gib=10.),
                   budget_seconds=3600., **kwargs)


def test_first_rung_is_one_forecast_trajectory_and_static_analysis():
    p = build_plan(request(), availability=availability, price=price)
    assert p['selected']['members'] == 1
    assert p['selected']['cycles'] == 1
    assert p['selected']['analysis'] == 'static-covariance-oi'
    assert p['selected']['covariance_members'] >= 2
    raw = tomllib.loads(p['configuration']['experiment'])
    assert len(raw['domain']) == 1
    assert raw['experiment']['restart_interval_s'] == p['selected']['cadence_seconds']
    assert p['clock']['cycle_ticks'] % p['clock']['parent_step_ticks'] == 0
    assert p['memory']['peak_bytes'] <= p['memory']['budget_bytes']
    assert p['schema'] == SCHEMA


@pytest.mark.parametrize('vram', [10., 16., 24., 32., 96.])
def test_estimated_fit_never_selects_a_lower_requested_rung(vram):
    req = Request(epoch=EPOCH, point=(40., -100.), card=Card(vram_gib=vram),
                  budget_seconds=1800., scale=9)
    plan = build_plan(req, availability=availability, price=price)
    assert plan['selected'] == derive_rung(req, req.scale)
    assert len(plan['alternatives']) == req.scale
    assert not plan['changed_scale']
    assert plan['selected']['cycles'] == req.scale


def test_vram_is_not_used_as_a_speed_measure():
    slow = request(scale=3)
    fast = Request(epoch=EPOCH, point=slow.point,
                   card=Card(vram_gib=10., speed_factor=2.),
                   budget_seconds=3600., scale=3)
    a = build_plan(slow, availability=availability, price=price)
    b = build_plan(fast, availability=availability, price=price)
    assert a['selected']['scale'] == b['selected']['scale']
    assert a['wall']['cycle_seconds'] == 2 * b['wall']['cycle_seconds']


def test_estimated_cadence_overrun_names_lag_and_optional_remedies():
    def too_slow(req, rung, exp, streams):
        return dict(price(req, rung, exp, streams), cycle_seconds=4000.)
    plan = build_plan(request(), availability=availability, price=too_slow)
    text = ' '.join(plan['warnings'])
    for remedy in ('fewer members', 'coarser', 'longer cadence'):
        assert remedy in text
    assert plan['status'] == 'ready'
    assert plan['cadence_overrun']['outcome'] == 'queued'


def test_missing_observations_are_receipted_not_converted_to_data():
    p = build_plan(request(), availability=lambda *a: [
        {'id': 'radar', 'status': 'unavailable', 'reason': 'no matching site'},
        {'id': 'surface', 'status': 'unavailable', 'reason': 'no station'}], price=price)
    assert p['observations'][0]['status'] == 'unavailable'
    assert any('forecast-only' in note for note in p['warnings'])


def test_unmeasured_is_not_an_admission_failure():
    p = build_plan(request(), availability=availability, price=price)
    assert not p['wall']['measured']
    assert p['status'] == 'ready'


def test_declared_region_is_contained_and_dateline_is_short_arc():
    req = Request(epoch=EPOCH, region=(178., -10., -179., -8.),
                  card=Card(vram_gib=96.), budget_seconds=36000., scale=1)
    p = build_plan(req, availability=availability, price=price)
    assert p['selected']['point'][1] == pytest.approx(179.5)
    assert p['selected']['nx'] < 1000
    assert p['region_checks']['contained']


@pytest.mark.parametrize('kwargs', [
    {'epoch': '2026-09-10T12:00:00'},
    {'epoch': 'not-a-time'},
    {'point': (float('nan'), 0)},
    {'point': (91., 0.)},
    {'scale': 0}, {'scale': True}, {'budget_seconds': float('inf')},
    {'region': (0., 0., 0., 1.), 'point': None},
])
def test_bad_inputs_refused_before_writes(kwargs):
    args = dict(epoch=EPOCH, point=(40., -100.), card=Card(vram_gib=10.),
                budget_seconds=3600.)
    args.update(kwargs)
    with pytest.raises((PlanError, ValueError)):
        build_plan(Request(**args), availability=availability, price=price)


def test_card_rejects_nonfinite_or_inconsistent_capacity():
    for kwargs in ({'vram_gib': float('nan')}, {'vram_gib': 10., 'free_gib': 11.},
                   {'vram_gib': 10., 'speed_factor': 0.},
                   {'vram_gib': 10., 'host_gib': -1.}):
        with pytest.raises(PlanError):
            Card(**kwargs)


def test_rung_members_follow_owner(monkeypatch):
    import gpuwm.ensemble.config as owner
    monkeypatch.setattr(owner, 'MAX_MEMBERS', 7)
    r = derive_rung(request(scale=5), 5)
    assert r['members'] == 7


def test_dry_run_writes_nothing_and_never_dispatches_runtime(tmp_path, monkeypatch, capsys):
    import argparse
    import gpuwm.local_da as module
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers(dest='command'))
    out = tmp_path / 'absent'
    args = parser.parse_args(['local-da', '--point', '40,-100', '--epoch', EPOCH,
                              '--vram-gib', '10', '--out', str(out), '--dry-run'])
    real = module.build_plan
    monkeypatch.setattr(module, 'build_plan', lambda req, **kwargs: real(
        req, availability=availability, price=price, background_probe=lambda url: True))
    assert main(args) == 0
    assert not out.exists()
    data = json.loads(capsys.readouterr().out)
    assert data['schema'] == SCHEMA
    assert data['forecast_started'] is False


def test_publish_is_complete_and_refuses_overwrite(tmp_path):
    p = build_plan(request(), availability=availability, price=price)
    root = tmp_path / 'case'
    receipt = publish(p, root)
    assert (root / 'experiment.toml').is_file()
    assert (root / 'ensemble.toml').is_file()
    assert (root / 'experiment.namelist.wps').is_file()
    assert (root / 'local-da.json').is_file()
    from gpuwm.ensemble.config import load_ensemble_config
    c = load_ensemble_config(root / 'ensemble.toml')
    assert c.n_members == 1 and c.perturbation == 'none'
    assert receipt['plan_path'] == str((root / 'local-da.json').resolve())
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    with pytest.raises(PlanError, match='exists'):
        publish(p, root)
    assert before == {p.name: p.read_bytes() for p in root.iterdir()}


def test_review_hash_changes_with_card_or_observation_inputs():
    a = build_plan(request(), availability=availability, price=price)
    b = build_plan(Request(epoch=EPOCH, point=(40., -100.),
                           card=Card(vram_gib=12.), budget_seconds=3600.),
                   availability=availability, price=price)
    assert a['review_sha256'] != b['review_sha256']


def test_composed_price_counts_analysis_storage_not_parallel_forecasts(monkeypatch):
    from types import SimpleNamespace
    import gpuwm.core.preflight as memory_owner
    import gpuwm.core.pace as pace_owner
    from gpuwm.local_da import price_rung
    monkeypatch.setattr(memory_owner, 'estimate_phases', lambda *a, **k: SimpleNamespace(peak_envelope_bytes=2 * 2**30))
    monkeypatch.setattr(pace_owner, 'estimate_pace', lambda exp, **k: SimpleNamespace(
        wall_seconds_high=90., basis='injected forecast timing', reference_card='fixture'))
    req = request()
    rung = derive_rung(req, 1)
    _, _, exp, _ = configuration(req, rung)
    low = price_rung(req, rung, exp, [{'status': 'ready', 'batch_bound': 2}])
    high = price_rung(req, rung, exp, [{'status': 'ready', 'batch_bound': 10}])
    assert high['forecast_peak_bytes'] == low['forecast_peak_bytes'] == 2 * 2**30
    assert high['analysis_peak_bytes'] > low['analysis_peak_bytes']
    assert high['observation_slots'] == 5 * low['observation_slots']
    assert max(high['forecast_peak_bytes'], high['analysis_peak_bytes']) < req.card.budget_bytes
    assert not high['measured']


def test_registry_inspection_restores_environment(monkeypatch):
    import os
    from gpuwm.local_da_observations import registry_inspection
    monkeypatch.setenv('GPUWM_FIXTURE_OLD', 'old')
    monkeypatch.delenv('GPUWM_FIXTURE_NEW', raising=False)
    with registry_inspection():
        os.environ['GPUWM_FIXTURE_OLD'] = 'changed'
        os.environ['GPUWM_FIXTURE_NEW'] = 'new'
    assert os.environ['GPUWM_FIXTURE_OLD'] == 'old'
    assert 'GPUWM_FIXTURE_NEW' not in os.environ


def test_registry_dateline_bounds_remain_two_small_boxes():
    import numpy as np
    from gpuwm.local_da_observations import _box_pieces
    pieces = _box_pieces(np.array([[1., 2.]]), np.array([[179., -179.]]), 180.)
    assert len(pieces) == 2
    assert sum(e - w for w, s, e, n in pieces) == pytest.approx(2.)


def test_companion_discovery_is_read_only(tmp_path, monkeypatch, capsys):
    import argparse
    import gpuwm.local_da as module
    parser = argparse.ArgumentParser()
    register_cli(parser.add_subparsers())
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(module, 'build_plan', lambda *a, **k: pytest.fail('discovery priced a domain'))
    assert main(parser.parse_args(['local-da', '--capabilities'])) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['schema'] == 'arwen.companion-local-da.v1'
    assert result['commands']['launch'] == ['local-da', '--launch', '{plan_path}']
    assert not list(tmp_path.iterdir())


def test_an_unpriced_rung_is_admitted_at_the_most_conservative_basis(monkeypatch):
    """Unmeasured is not an admission reason: price it and say so."""
    from types import SimpleNamespace
    from tilestream import autoplan
    import gpuwm.core.preflight as memory_owner

    monkeypatch.setattr(memory_owner, 'estimate_phases',
                        lambda *a, **k: SimpleNamespace(peak_envelope_bytes=2 * 2**30))
    monkeypatch.setattr(autoplan, 'rung_of', lambda cfg: 'a-rung-nobody-timed')
    req = request()
    rung = derive_rung(req, 1)
    _, _, exp, _ = configuration(req, rung)
    costs = price_rung(req, rung, exp, [{'status': 'ready', 'batch_bound': 2}])
    assert costs['pace_substituted'] and not costs['measured']
    assert costs['cycle_seconds'] > 0. and costs['forecast_seconds'] > 0.
    assert any('slowest rate on record' in line for line in costs['basis'])


def test_the_substituted_basis_is_warned_about_once_and_the_rung_still_runs():
    def unpriced(req, rung, exp, streams):
        row = price(req, rung, exp, streams)
        row['pace_substituted'] = True
        return row
    p = build_plan(request(), availability=availability, price=unpriced)
    assert p['status'] == 'ready'
    named = [note for note in p['warnings'] if 'slowest rate on record' in note]
    assert len(named) == 1


def test_the_cadence_verdict_comes_from_the_cadence_owner(monkeypatch):
    import gpuwm.da.cadence as owner
    seen = []
    real = owner.check_overrun

    def spy(plan, *, cycle_cost_seconds, policy='refuse', cost_basis='measured'):
        seen.append((policy, float(cycle_cost_seconds), len(plan.cycles)))
        return real(plan, cycle_cost_seconds=cycle_cost_seconds, policy=policy,
                    cost_basis=cost_basis)

    monkeypatch.setattr(owner, 'check_overrun', spy)
    p = build_plan(request(), availability=availability, price=price)
    assert seen, 'the ladder decided the cadence question without its owner'
    assert p['cadence_overrun']['schema'] == 'gpuwm-da.cadence-overrun.v1'
    assert p['cadence_overrun']['outcome'] == 'clear'
    assert p['cadence_overrun']['cycle_cost_seconds'] == p['wall']['cycle_seconds']
    # Every row of a successful review is priced, so the record is on
    # every one of them; the hedge that used to skip a row without the
    # key went with the row shape that could drop it.
    assert all(row['priced'] for row in p['alternatives'])
    assert all(row['cadence']['outcome'] == 'clear' for row in p['alternatives'])


def test_a_cadence_advisory_carries_the_owners_queue_record():
    def too_slow(req, rung, exp, streams):
        return dict(price(req, rung, exp, streams), cycle_seconds=4000.)
    plan = build_plan(request(), availability=availability, price=too_slow)
    records = [row['cadence'] for row in plan['alternatives'] if row['priced']]
    assert records and all(record['outcome'] == 'queued' for record in records)
    assert all(record['cycles_shorter_than_cost'] >= 1 for record in records)
    assert all(record['policy'] == 'queue' for record in records)
    assert all(record['cost_basis'] == 'estimated' for record in records)
    assert all('projection_policy' not in record for record in records)


def test_a_neutral_table_with_no_decoder_is_refused_with_the_way_out(
        monkeypatch):
    """Rule 1: a refusal owes the breakage AND the way out, in the text."""
    import builtins
    import importlib
    import gpuwm.da.obs_point as owner

    real_import = builtins.__import__

    def without_the_decoder(name, *args, **kwargs):
        if name.startswith('arwen_global'):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, '__import__', without_the_decoder)
    with pytest.raises(ImportError) as caught:
        owner.read_tables(['does-not-matter.csv'])
    text = str(caught.value)
    assert 'arwen_global.obs_table' in text, 'the refusal must name what is absent'
    assert 'gpuwm-global' in text and '--obs-table' in text, (
        'the refusal must name both ways out: install the decoder, or'
        ' review without the flag')
    assert importlib is not None


def test_every_local_da_option_says_what_it_does():
    import argparse
    parser = argparse.ArgumentParser()
    door = register_cli(parser.add_subparsers())
    silent = sorted(action.option_strings[0] for action in door._actions
                    if action.option_strings and not (action.help or '').strip())
    assert not silent, f"these options carry no help text: {silent}"


def test_the_companion_document_and_the_review_do_not_disagree():
    from pathlib import Path as _Path
    contract = protocol_document()
    plan = build_plan(request(), availability=availability, price=price)
    assert contract['review_fields'] == sorted(plan)
    root = _Path(__file__).resolve().parents[1]
    text = (root / contract['document']).read_text(encoding='utf-8')
    missing = [field for field in contract['review_fields']
               if f'`{field}`' not in text]
    assert not missing, f"the companion document does not describe {missing}"
    assert contract['review_schema'] == plan['schema'] == SCHEMA

    # The refusal roster is checked the same way the field roster is: a
    # code the door can print must be in the published contract and must
    # have a line in the document, because a companion switching on
    # `code` has no arm for one it was never told about.
    codes = contract['refusal_codes']
    assert codes['review'] == sorted(REVIEW_REFUSAL_CODES)
    assert codes['launch'] == sorted(LAUNCH_REFUSAL_CODES)
    assert codes['run'] == sorted(RUN_REFUSAL_CODES)
    assert codes['fallback'] == FALLBACK_REFUSAL_CODE
    undocumented = [code for code in REFUSAL_CODES if f'`{code}`' not in text]
    assert not undocumented, (
        f"the companion document does not describe {undocumented}")

    # What a group leaves on disk is part of the same contract, because a
    # companion told to expect no execution document will not read the one
    # that is there.  Every group, and the promise it makes, has a line.
    effects = contract['refusal_effects']
    assert set(effects) == {group for group, _, _ in REFUSAL_EFFECTS}
    assert set(effects) | {'fallback'} == set(codes)
    for group, document, started in REFUSAL_EFFECTS:
        assert effects[group] == dict(execution_document=document,
                                      forecast_started_can_be_true=started)
        assert f'`refusal_codes.{group}`' in text, (
            f"the companion document does not describe the {group} group")
    assert contract['refusal_discriminator'] == 'error'
    assert contract['refusal_discriminator'] in contract['refusal_fields']
    assert 'schema' in contract['refusal_fields'], (
        'a refusal carries a schema string, so the roster must name it')
    assert '`refusal_discriminator`' in text
    assert '`arwen.local-da-execution.v1`' in text, (
        'the page must name the document a run refusal leaves behind')


def test_the_refusal_roster_is_every_code_the_door_can_print():
    """The roster is checked against source, not maintained beside it.

    Group membership is read from WHERE the code is raised, not from a
    list a maintainer keeps in step by hand: a launch-side raise inside
    one of :data:`PRE_EXECUTION_FUNCTIONS` runs before the case's
    execution document is opened, and a raise anywhere else in those
    modules runs after it, which is the whole of the difference the
    published contract promises a companion.

    What this proves is that the roster and the source agree.  It does
    NOT prove a code can escape the door: a `code=` literal caught and
    folded into a ladder row is still a literal.  That half is proved by
    driving the door, in
    ``test_a_run_past_the_source_horizon_is_refused_as_a_horizon_problem``
    and ``test_the_horizon_refusal_reaches_the_door_with_its_own_code``,
    which is where an unemittable published code was found.
    """
    import ast
    import re
    from pathlib import Path as _Path
    root = _Path(__file__).resolve().parents[1]

    def enclosing(tree, line):
        names = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                if node.lineno <= line <= node.end_lineno:
                    names.append(node.name)
        return names

    found: dict[str, set[str]] = {'review': set(), 'launch': set(), 'run': set()}
    for door, relative in REFUSAL_SOURCES:
        source = (root / relative).read_text(encoding='utf-8')
        tree = ast.parse(source)
        for number, line in enumerate(source.splitlines(), 1):
            match = re.search(r"""code=['"]([A-Z_]+)['"]""", line)
            if match is None:
                continue
            group = door
            if door == 'launch' and not set(enclosing(tree, number)) & set(PRE_EXECUTION_FUNCTIONS):
                group = 'run'
            found[group].add(match.group(1))
    # The default on PlanError itself is the malformed-request code and is
    # raised by every bare `raise PlanError(...)`, so it belongs to review.
    found['review'].add('INVALID_PLAN')
    assert sorted(found['review']) == sorted(REVIEW_REFUSAL_CODES), (
        "gpuwm/local_da.py raises review codes the roster does not carry,"
        " or the roster carries codes it no longer raises")
    assert sorted(found['launch']) == sorted(LAUNCH_REFUSAL_CODES), (
        "the launch modules raise codes before the execution document is"
        " opened that the launch roster does not carry, or the roster"
        " carries codes they no longer raise there")
    assert sorted(found['run']) == sorted(RUN_REFUSAL_CODES), (
        "the launch modules raise codes after the execution document is"
        " opened that the run roster does not carry, or a check moved"
        " across that line without moving groups")
    assert FALLBACK_REFUSAL_CODE not in set().union(*found.values())
    assert len(REFUSAL_CODES) == len(set(REFUSAL_CODES))


@pytest.mark.parametrize('group', ['launch', 'run'])
def test_a_refusal_leaves_behind_what_the_published_contract_promises(tmp_path, group):
    """One code of each launch-side group, measured against the contract.

    The two groups differ in exactly what a companion has to do next, so
    both are measured here: whether an execution document is on disk and
    whether ``forecast_started`` can be true.  A test that covered only
    the tampered-case arm would be green while the page described the
    other group wrongly, which is how the wrong description shipped.
    """
    import gpuwm.local_da_runtime as runtime
    # Imported inside the test: that module imports this one at collection.
    from test_local_da_runtime import Backend
    plan = build_plan(request(), availability=availability, price=price)
    root = tmp_path / 'case'
    publication = publish(plan, root)
    arguments = {'backend': Backend()}
    if group == 'launch':
        (root / 'experiment.toml').write_text(
            (root / 'experiment.toml').read_text(encoding='utf-8')
            + chr(10) + '# appended after review' + chr(10),
            encoding='utf-8')
    else:
        arguments['roster_reader'] = lambda *a, **kw: {}
    with pytest.raises(PlanError) as raised:
        runtime.launch(publication['plan_path'], **arguments)

    code = raised.value.code
    contract = protocol_document()
    owners = [name for name, codes in contract['refusal_codes'].items()
              if isinstance(codes, list) and code in codes]
    assert owners == [group], (
        f"{code} was raised by the {group} stage and the published roster"
        f" puts it in {owners}")
    effects = contract['refusal_effects'][group]
    document = root / 'execution.json'
    assert document.exists() == effects['execution_document'], (
        f"the contract says a {group} refusal writes"
        f"{'' if effects['execution_document'] else ' no'} execution"
        f" document, and {code} did the opposite")
    started = bool(getattr(raised.value, 'forecast_started', False))
    if not effects['forecast_started_can_be_true']:
        assert not started, (
            f"the contract says forecast_started is false for a {group}"
            f" refusal, and {code} raised it true")
    if document.exists():
        report = json.loads(document.read_text(encoding='utf-8'))
        assert report['schema'] == 'arwen.local-da-execution.v1'
        assert report['status'] == 'FAILED'
        assert report['forecast_started'] == started
    if group == 'run':
        assert started, (
            f"{code} is raised after members have integrated, which is why"
            " the contract says forecast_started can be true")


def test_the_published_case_is_the_shape_the_preparation_owner_reads(tmp_path):
    """The launch path's first stage reads the namelist this publishes."""
    from gpuwm import go_cli
    from pathlib import Path as _Path

    plan = build_plan(request(), availability=availability, price=price)
    root = tmp_path / 'case'
    publish(plan, root)
    prepared = go_cli.plan_from_config(root / 'experiment.toml',
                                       outdir=tmp_path / 'run', run_stamp=False)
    assert _Path(prepared['wps_namelist']).is_file(), (
        "the preparation owner reads "
        f"{prepared['wps_namelist']}, which the publisher did not write")
    assert _Path(prepared['config']).is_file()


# --- the ladder's two unpriced rows and the refusal each of them causes ---
#
# Both of these paths stop the ladder before a rung can be priced, and
# both hand their row to a companion that was told to draw every row.  The
# row shape and the emitted code are checked here rather than inferred
# from a source grep, because a code that only appears as a `code=`
# literal is a code a consumer will never receive.


def forcing_request(**kwargs):
    """A request whose run needs forcing past the source horizon."""
    return Request(epoch=EPOCH, point=(40., -100.), card=Card(vram_gib=10.),
                   budget_seconds=1.e8, forecast_seconds=1.44e6, **kwargs)


def test_a_run_past_the_source_horizon_is_refused_as_a_horizon_problem():
    with pytest.raises(PlanError) as caught:
        build_plan(forcing_request(), availability=availability, price=price)
    assert caught.value.code == 'FORCING_HORIZON', (
        'the door published FORCING_HORIZON to companions as a review code,'
        ' so a companion switching on `code` must receive it')
    assert 'horizon' in str(caught.value)
    # The way out must be one that can move a source horizon.  Neither
    # memory nor a longer time budget can, and offering them is the
    # failure this test exists to prevent.
    assert 'Supply the required frames' in str(caught.value)
    assert 'free memory' not in str(caught.value)
    assert 'time budget' not in str(caught.value)


def test_the_horizon_refusal_reaches_the_door_with_its_own_code(capsys):
    parser = __import__('argparse').ArgumentParser()
    register_cli(parser.add_subparsers(dest='command'))
    args = parser.parse_args(['local-da', '--point', '40,-100', '--epoch', EPOCH,
                              '--vram-gib', '10', '--forecast-seconds', '1440000',
                              '--budget-seconds', '100000000', '--dry-run'])
    assert main(args) == 1
    document = json.loads(capsys.readouterr().out)
    assert document['code'] == 'FORCING_HORIZON'
    assert document['forecast_started'] is False
    assert 'Supply the required frames' in document['error']


def test_resource_advice_names_options_for_the_reason_it_quotes():
    def slow_preparation(req, rung, experiment, streams):
        return dict(price(req, rung, experiment, streams), preparation_seconds=1.e5)
    plan = build_plan(request(), availability=availability, price=slow_preparation)
    text = ' '.join(plan['warnings'])
    assert 'time budget' in text and REMEDY_TIME_BUDGET in text
    assert REMEDY_CARD_MEMORY not in text and REMEDY_HOST_MEMORY not in text
    def enormous(req, rung, experiment, streams):
        return dict(price(req, rung, experiment, streams), forecast_peak_bytes=1 << 45)
    plan = build_plan(request(), availability=availability, price=enormous)
    text = ' '.join(plan['warnings'])
    assert REMEDY_CARD_MEMORY in text and REMEDY_TIME_BUDGET not in text


def test_cadence_and_memory_advisories_keep_each_optional_control():
    def starved_and_slow(req, rung, exp, streams):
        return dict(price(req, rung, exp, streams), forecast_peak_bytes=1 << 45,
                    host_peak_bytes=1 << 46, cycle_seconds=4000.)
    plan = build_plan(request(scale=2), availability=availability, price=starved_and_slow)
    text = ' '.join(plan['warnings'])
    assert 'exceeds cadence' in text
    for remedy in (REMEDY_CADENCE, REMEDY_CARD_MEMORY, REMEDY_HOST_MEMORY):
        assert remedy.rstrip('.') in text
        assert any(remedy in row['remedies'] for row in plan['alternatives'])
    assert plan['selected']['scale'] == 2 and plan['selected']['cycles'] == 2


def test_shipped_pricer_keeps_the_requested_region_despite_estimated_lag():
    req = Request(epoch=EPOCH, region=(-99.5, 33.5, -95.5, 37.),
                  card=Card(vram_gib=10., free_gib=2., host_gib=6.),
                  budget_seconds=3600., scale=4)
    plan = build_plan(req, availability=availability)
    assert plan['selected'] == derive_rung(req, req.scale)
    assert REMEDY_CADENCE.rstrip('.') in ' '.join(plan['warnings'])


def test_every_ladder_row_is_the_shape_the_protocol_publishes():
    contract = protocol_document()
    assert contract['alternative_fields'] == sorted(ALTERNATIVE_FIELDS)
    assert contract['unpriced_row_discriminator'] == 'priced'
    starved = Request(epoch=EPOCH, point=(40., -100.),
        card=Card(vram_gib=10., host_gib=1.e-6), budget_seconds=3600.)
    plan = build_plan(starved, availability=availability, price=price)
    for row in plan['alternatives']:
        assert sorted(row) == sorted(ALTERNATIVE_FIELDS)
        assert row['priced'] and not row['fits']
        assert REMEDY_HOST_MEMORY in row['remedies']
        assert row['cadence']['schema'] == 'gpuwm-da.cadence-overrun.v1'
    with pytest.raises(PlanError) as horizon:
        build_plan(forcing_request(), availability=availability, price=price)
    rows = horizon.value.details['alternatives']
    assert rows
    for row in rows:
        assert sorted(row) == sorted(ALTERNATIVE_FIELDS)
    unpriced = [row for row in rows if not row['priced']]
    assert len(unpriced) == 1
    row = unpriced[0]
    assert not row['fits'] and row['remedies'][-1] == REMEDY_FORCING
    assert len(row['remedies']) == len(row['reasons'])
    for key in ('members', 'dx_m', 'cycles', 'peak_bytes', 'cycle_seconds', 'total_seconds', 'cadence'):
        assert row[key] is None


def test_the_companion_document_describes_the_ladder_row():
    from pathlib import Path as _Path
    contract = protocol_document()
    text = (_Path(__file__).resolve().parents[1] / contract['document']).read_text(encoding='utf-8')
    missing = [field for field in contract['alternative_fields'] if f'`{field}`' not in text]
    assert not missing, f'the companion document does not describe {missing}'
    assert '`unpriced_row_discriminator`' in text
    assert '`alternative_fields`' in text


def test_the_cadence_record_says_the_review_priced_the_cost():
    """A reviewed cadence verdict rests on a price, and says so.

    Nothing here is priced by stopwatch, so a record reading `measured`
    beside `wall.measured` false would be two statements about one
    number.
    """
    p = build_plan(request(), availability=availability, price=price)
    assert p['wall']['measured'] is False
    assert p['cadence_overrun']['cost_basis'] == 'estimated'
    assert all(row['cadence']['cost_basis'] == 'estimated'
               for row in p['alternatives'])

    def timed(req, rung, experiment, streams):
        return dict(price(req, rung, experiment, streams), measured=True)
    q = build_plan(request(), availability=availability, price=timed)
    assert q['cadence_overrun']['cost_basis'] == 'measured'
