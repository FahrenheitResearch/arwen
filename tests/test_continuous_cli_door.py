"""The continuous door: what a user types, what the plan carries, what --status and --stop answer."""
import argparse
import json

import pytest

from gpuwm.local_da import (build_plan, main, protocol_document, publish, register_cli,
                            request_from_json, PlanError, REQUEST_SCHEMA, RUN_REFUSAL_CODES)
from gpuwm.local_da_controller import STATUSES, TERMINAL_STATUSES, STATUS_SCHEMA
from test_local_da_plan import request, availability, price


def door():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    register_cli(sub)
    return parser


def parse(*argv):
    return door().parse_args(['local-da', *argv])


def test_continuous_takes_a_window_count_and_status_and_stop_name_a_plan(tmp_path):
    args = parse('--point', '35.2,-97.4', '--epoch', '2026-09-10T12:00:00Z', '--vram-gib', '16', '--continuous', '3')
    assert args.continuous == 3 and args.status is None and args.stop is None
    args = parse('--status', str(tmp_path / 'local-da.json'))
    assert str(args.status).endswith('local-da.json') and args.continuous is None
    args = parse('--stop', str(tmp_path / 'local-da.json'))
    assert str(args.stop).endswith('local-da.json')
    with pytest.raises(SystemExit):
        parse('--continuous', 'three')


def test_the_help_text_says_what_each_continuous_option_does():
    actions = {action.option_strings[0]: action.help for action in door()._subparsers._group_actions[0].choices['local-da']._actions
               if action.option_strings}
    assert 'analysis windows' in actions['--continuous'] and 'renewed' in actions['--continuous']
    assert 'status document' in actions['--status']
    assert 'stop' in actions['--stop'] and 'durable' in actions['--stop']
    assert 'clears it and resumes' in actions['--stop'] and 'ask again' in actions['--stop']
    assert actions['--continuous'].startswith('cycle continuously for WINDOWS')


def test_a_continuous_review_carries_the_contract_and_publishes_its_paths(tmp_path):
    from dataclasses import replace
    plan = build_plan(replace(request(), continuous_windows=4), availability=availability, price=price)
    continuous = plan['continuous']
    assert continuous['enabled'] is True and continuous['windows'] == 4
    assert continuous['status_schema'] == STATUS_SCHEMA
    assert continuous['status_relative_path'] == 'continuous/status.json'
    assert continuous['control_relative_path'] == 'continuous/stop'
    assert continuous['window_relative_path'] == 'continuous/window_{index:06d}'
    assert continuous['product_policy'] == 'every-window-in-order'
    result = publish(plan, tmp_path / 'case')
    assert result['status_path'] == str(tmp_path / 'case' / 'continuous' / 'status.json')
    assert result['control_path'] == str(tmp_path / 'case' / 'continuous' / 'stop')
    saved = json.loads((tmp_path / 'case' / 'local-da.json').read_text())
    assert saved['continuous'] == continuous and saved['request']['continuous_windows'] == 4
    finite = build_plan(request(), availability=availability, price=price)
    assert 'continuous' not in finite and finite['request']['continuous_windows'] == 0


@pytest.mark.parametrize('value', [-1, 1.5, True, '2'])
def test_a_continuous_request_needs_a_whole_window_count(value):
    from dataclasses import replace
    with pytest.raises(PlanError, match='continuous_windows must be a whole number'):
        replace(request(), continuous_windows=value).validate()


def test_a_request_document_may_ask_for_continuous_windows():
    raw = dict(schema=REQUEST_SCHEMA, epoch='2026-09-10T12:00:00Z', point=[35.2, -97.4],
               card=dict(vram_gib=16.), continuous_windows=2)
    assert request_from_json(raw).continuous_windows == 2


def test_capabilities_publish_the_continuous_contract_and_its_refusal_code():
    contract = protocol_document()
    continuous = contract['continuous']
    assert continuous['supported'] is True
    assert continuous['request_field'] == 'continuous_windows' and continuous['review_field'] == 'continuous'
    assert continuous['statuses'] == list(STATUSES) and continuous['terminal_statuses'] == list(TERMINAL_STATUSES)
    assert continuous['status_command'] == ['local-da', '--status', '{plan_path}']
    assert continuous['stop_command'] == ['local-da', '--stop', '{plan_path}']
    assert continuous['resume_command'] == ['local-da', '--launch', '{plan_path}']
    assert contract['optional_review_fields'] == ['continuous']
    assert 'continuous_windows' in contract['request_fields']
    assert 'CONTINUOUS_WINDOW_FAILED' in contract['refusal_codes']['run'] and 'CONTINUOUS_WINDOW_FAILED' in RUN_REFUSAL_CODES


def test_status_and_stop_doors_print_one_document_each(tmp_path, capsys):
    from dataclasses import replace
    plan = build_plan(replace(request(), continuous_windows=2), availability=availability, price=price)
    path = publish(plan, tmp_path / 'case')['plan_path']
    assert main(parse('--status', path)) == 0
    status = json.loads(capsys.readouterr().out)
    assert status['status'] == 'NOT_STARTED' and status['stop_requested'] is False
    assert status['windows'] == 2 and status['plan_path'] == path
    assert main(parse('--stop', path)) == 0
    stopped = json.loads(capsys.readouterr().out)
    assert stopped['stop_requested'] is True and stopped['status'] == 'NOT_STARTED'
    assert main(parse('--status', path)) == 0
    assert json.loads(capsys.readouterr().out)['stop_requested'] is True


def test_status_on_a_finite_plan_and_contradictory_arguments_are_refused(tmp_path, capsys):
    plan = build_plan(request(), availability=availability, price=price)
    path = publish(plan, tmp_path / 'finite')['plan_path']
    assert main(parse('--status', path)) == 1
    refusal = json.loads(capsys.readouterr().out)
    assert 'error' in refusal and '--continuous' in refusal['error']
    assert main(parse('--status', path, '--stop', path)) == 1
    assert 'each name one saved plan' in json.loads(capsys.readouterr().out)['error']
    assert main(parse('--score', path, '--stop', path)) == 1
    assert 'each name one saved plan' in json.loads(capsys.readouterr().out)['error']
    assert main(parse('--score', path, '--dry-run')) == 1
    assert 'each name one saved plan' in json.loads(capsys.readouterr().out)['error']
    assert main(parse('--continuous', '0', '--point', '1,2', '--epoch', '2026-09-10T12:00:00Z', '--vram-gib', '8', '--dry-run')) == 1
    assert 'positive number of windows' in json.loads(capsys.readouterr().out)['error']
    assert main(parse('--continuous', '2', '--launch', path)) == 1
    assert 'decided at review' in json.loads(capsys.readouterr().out)['error']


def test_the_documents_describe_the_continuous_door():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    protocol = (root / 'docs' / 'local-da-companion-protocol.md').read_text(encoding='utf-8')
    for name in ('`continuous_windows`', '`continuous`', '`CONTINUOUS_WINDOW_FAILED`', '`WAITING_FORCING`',
                 '`status_relative_path`', '`renewal_policy`', 'renewal.json'):
        assert name in protocol, name
    for status in STATUSES:
        assert f'`{status}`' in protocol, status
    page = (root / 'docs' / 'local-da-rapid-cycling.md').read_text(encoding='utf-8')
    assert '--continuous 3' in page and '--status' in page and '--stop' in page
    reference = (root / 'docs' / 'public' / 'CLI-OPTIONS.md').read_text(encoding='utf-8')
    assert '| `--continuous WINDOWS` |' in reference or '| `--continuous` |' in reference
    assert '| `--status' in reference and '| `--stop' in reference
