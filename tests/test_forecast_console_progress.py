"""The supervised front door reports live work and only confirms a real success."""
from io import StringIO
from types import SimpleNamespace

import pytest

from gpuwm.progress import ForecastProgress
from gpuwm.supervisor import SupervisorError, supervise_experiment, supervise_from_cli
from test_supervisor import _heartbeat, _install_scripted_supervisor


def _args(config, tmp_path):
    return SimpleNamespace(config=config, outdir=tmp_path / 'out', restart=None,
        gpu_uuid=None, supervisor_max_restarts=0, prep_timeout=None,
        allow_shared_gpu=False, health_debug=False)


def test_long_preparation_stays_visible_without_flooding_a_redirected_log(monkeypatch):
    import gpuwm.progress as progress
    now = [0.]
    monkeypatch.setattr(progress.time, 'monotonic', lambda: now[0])
    stream = StringIO()
    reporter = ForecastProgress(stream=stream)
    preparing = _heartbeat(status='preparing:prepare-case', outer_step=0,
                           model_elapsed_seconds=0.)
    reporter(preparing)
    for second in range(1, 30):
        now[0] = second
        reporter(preparing)
    assert len(stream.getvalue().splitlines()) == 1
    now[0] = 30.
    reporter(preparing)
    now[0] = 31.
    reporter(_heartbeat(model_elapsed_seconds=90061., outer_step=43))
    lines = stream.getvalue().splitlines()
    assert len(lines) == 3
    assert 'loading and preparing inputs' in lines[0]
    assert 'elapsed 00:00:30' in lines[1]
    assert '25:01:01 simulated; step 43' in lines[2]


def test_closed_progress_pipe_does_not_interrupt_forecast_work():
    class ClosedPipe(StringIO):
        def write(self, text):
            raise BrokenPipeError('reader closed')
    reporter = ForecastProgress(stream=ClosedPipe())
    reporter(_heartbeat())
    reporter(_heartbeat(status='complete'))
    assert not reporter.enabled


def test_ordinary_cli_reports_progress_outputs_and_detail_logs(monkeypatch, tmp_path, capsys):
    config, _, processes = _install_scripted_supervisor(monkeypatch, tmp_path, [[
        {'status': 'preparing:prepare-case', 'step': 0},
        {'status': 'integrating', 'step': 1},
        {'status': 'complete', 'step': 2, 'exit': 0},
    ]])
    original = config.read_bytes()
    assert supervise_from_cli(_args(config, tmp_path)) == 0
    output = capsys.readouterr()
    assert 'Starting forecast.' in output.err
    assert 'loading and preparing inputs' in output.err
    assert 'simulated; step 1' in output.err
    assert 'Forecast complete:' in output.out
    assert 'worker-01.stdout.log' in output.out
    assert 'worker-01.stderr.log' in output.out
    assert str((tmp_path / 'out').resolve()) in output.out
    assert "{'run_id':" not in output.out
    assert len(processes) == 1 and config.read_bytes() == original


def test_progress_never_uses_a_heartbeat_from_a_different_worker(monkeypatch, tmp_path):
    config, _, _ = _install_scripted_supervisor(monkeypatch, tmp_path, [[
        {'status': 'integrating', 'step': 1, 'heartbeat_pid': 52001},
        {'status': 'integrating', 'step': 2, 'heartbeat_pid': 52002},
    ]])
    seen = []
    with pytest.raises(SupervisorError, match='heartbeat identity violation'):
        supervise_experiment(config, tmp_path / 'out', on_progress=seen.append)
    assert seen and all(heartbeat.pid == 52001 for heartbeat in seen)
    assert all(heartbeat.outer_step == 1 for heartbeat in seen)


def test_terminal_heartbeat_does_not_print_success_after_failed_exit(monkeypatch, tmp_path, capsys):
    config, _, _ = _install_scripted_supervisor(monkeypatch, tmp_path, [[
        {'status': 'complete', 'step': 2},
        {'exit': 1},
    ]])
    with pytest.raises(SupervisorError):
        supervise_from_cli(_args(config, tmp_path))
    output = capsys.readouterr()
    assert 'Finishing forecast' in output.err
    assert 'Forecast complete' not in output.out


def test_closed_stream_is_safe_before_worker_launch():
    stream = StringIO()
    stream.close()
    reporter = ForecastProgress(stream=stream)
    reporter.write('Starting forecast')
    reporter(_heartbeat())
    assert not reporter.enabled


def test_final_output_publication_is_not_labeled_preparation():
    stream = StringIO()
    reporter = ForecastProgress(stream=stream)
    reporter(_heartbeat(status='finalizing:write-capsule'))
    reporter(_heartbeat(status='failed'))
    text = stream.getvalue()
    assert 'Finishing: write capsule' in text
    assert 'reported a failure' in text
    assert 'Preparing:' not in text
    assert 'Forecast complete' not in text
