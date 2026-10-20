"""Native forecast argument parsing must not import unrelated command trees."""
import argparse

import pytest

from gpuwm import cli


def _run_parser(parser):
    sub = next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction))
    return sub.choices['run']


def test_run_parser_preserves_help_and_all_option_defaults():
    full = cli.build_parser()
    focused = cli.build_parser(run_only=True)
    assert _run_parser(full).format_help() == _run_parser(focused).format_help()
    for argv in [
        ['run', 'forecast.toml'],
        ['run', '--wrfinput', '/inputs', '--benchmark', '--rrtmg-variant', 'rrtmg_legacy'],
        ['run', '--met-em', '/met', '--vertical-levels', '49', '--run-seconds', '3600',
         '--members', '36', '--allow-shared-gpu', '--explain'],
    ]:
        expected = vars(full.parse_args(argv))
        # The check registrar leaks its private callback into later parsers;
        # only combined_check_main consumes it, never the run dispatcher.
        expected.pop('ingest_preflight_handler', None)
        assert expected == vars(focused.parse_args(argv))


def test_run_dispatch_registers_only_its_own_command(capsys, monkeypatch):
    register = cli._lazy_register
    requested = []

    def relevant(module):
        requested.append(module)
        assert module == 'gpuwm.supervisor', module
        return register(module)

    monkeypatch.setattr(cli, '_lazy_register', relevant)
    with pytest.raises(SystemExit) as stop:
        cli.main(['run', '--help'])
    assert stop.value.code == 0
    assert requested == ['gpuwm.supervisor']
    assert '--wrfinput' in capsys.readouterr().out
