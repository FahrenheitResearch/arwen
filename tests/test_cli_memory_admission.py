"""Real admission failures meet the CLI boundary without exhausting memory."""

from types import SimpleNamespace

import pytest

from gpuwm import cli


@pytest.mark.parametrize("capacity", ["device_budget_bytes", "host_available_bytes"])
@pytest.mark.parametrize("explain", [False, True])
def test_initialization_budget_refusal_is_readable_at_cli(
        tmp_path, monkeypatch, capsys, capacity, explain):
    from gpuwm.config import RunConfig
    from gpuwm.ingest.case_store import CaseStoreRequest, admit_case_initialization

    cfg = RunConfig(nx=31, ny=27, nz=9, dx=10000., dy=10000., ztop=16000.,
                    dt=30., run_seconds=60.)
    met = SimpleNamespace(fields={"actual": SimpleNamespace(shape=(17, 27, 31))})
    request = CaseStoreRequest(tmp_path / "cache", resources={capacity: 1})
    monkeypatch.setattr(cli, "_dispatch", lambda args:
                        admit_case_initialization(request, cfg, met, (0, 1, 2)))
    # The diagnostic command needs no CUDA preflight. Only the dispatch is
    # substituted: the refusal is produced by the actual admission estimator.
    assert cli.main(["version"] + (["--explain"] if explain else [])) == 2
    terminal = capsys.readouterr()
    assert "initialization" in terminal.err and "GiB" in terminal.err
    assert "reduce the prepared grid" in terminal.err
    assert ("Traceback" in terminal.err) == explain
    assert len(request.admissions) == 1
    assert not (tmp_path / "cache").exists()


def test_pinned_store_budget_guard_is_readable_without_allocating(monkeypatch, capsys):
    from tilestream.hoststore import GIB, check_allocatable
    monkeypatch.setattr(cli, "_dispatch", lambda args:
                        check_allocatable(GIB, budget_bytes=GIB // 2))
    assert cli.main(["version"]) == 2
    error = capsys.readouterr().err
    assert "store needs 1.00 GiB but budget_bytes is 0.50 GiB" in error
    assert "Reduce the prepared grid" in error and "Traceback" not in error


def test_measured_host_store_guard_refuses_without_allocating(monkeypatch, capsys):
    from tilestream.hoststore import check_allocatable
    # One PiB fails against this machine's measured available RAM. The
    # guard only reads capacity; it never attempts the allocation.
    monkeypatch.setattr(cli, "_dispatch", lambda args: check_allocatable(1 << 50))
    assert cli.main(["version"]) == 2
    error = capsys.readouterr().err
    assert "is available" in error and "host-store memory refused" in error
    assert "Traceback" not in error


@pytest.mark.parametrize("error", [MemoryError("actual allocation failed"),
                                     RuntimeError("unrelated runtime defect")])
def test_unrelated_failures_are_not_masked(monkeypatch, error):
    def dispatch(args):
        raise error
    monkeypatch.setattr(cli, "_dispatch", dispatch)
    with pytest.raises(type(error), match=str(error)):
        cli.main(["version"])
