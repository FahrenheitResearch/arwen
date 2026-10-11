"""Controls for diagnostic timing, never a substitute for a real-run gate."""

import pytest

from tools.timeline.profile_cli import Budget, install


def test_nested_budget_is_not_double_counted_and_returns_the_real_result():
    budget = Budget()
    inner = budget.wrap("inner", lambda value: value + 1)
    outer = budget.wrap("outer", lambda value: inner(value) * 2)
    assert outer(3) == 8
    assert budget.rows["inner"]["calls"] == 1
    assert budget.rows["inner"]["outermost_seconds"] == 0
    assert budget.occupied_seconds == budget.rows["outer"]["outermost_seconds"]


def test_failed_work_keeps_its_exception_and_restores_nesting_depth():
    budget = Budget()
    def failed():
        raise RuntimeError("real work failed")
    with pytest.raises(RuntimeError, match="real work failed"):
        budget.wrap("failed", failed)()
    assert budget.local.depth == 0
    assert budget.rows["failed"]["calls"] == 1
    assert budget.wrap("next", lambda: 7)() == 7
    assert budget.rows["next"]["outermost_seconds"] > 0


def test_installed_budget_measures_real_emission_without_charging_monitor_sleep(tmp_path, monkeypatch):
    import gpuwm.timeline.ledger as ledger
    import gpuwm.stage_timing as timing
    # Preserve module methods so the diagnostic cannot affect another test.
    for cls in (ledger.Ledger, timing.ForecastLedger, timing.LedgerObserver):
        for name, function in list(vars(cls).items()):
            if callable(function):
                monkeypatch.setattr(cls, name, function)
    for name in ("register_ledger_observer", "current_ledger", "end_admission", "bind_ledger", "ledger_main"):
        monkeypatch.setattr(timing, name, getattr(timing, name))
    watch = timing.ForecastLedger._watch
    monkeypatch.delenv("GPUWM_TIME_LEDGER", raising=False)
    budget = Budget()
    install(budget)
    assert timing.ForecastLedger._watch is watch
    assert ledger.Ledger(tmp_path / "ledger.jsonl").emit("boot.ready")
    assert budget.rows["Ledger.emit"]["calls"] == 1
    assert budget.occupied_seconds > 0
    @timing.ledger_main
    def real_work():
        return 17
    assert real_work() == 17
    assert budget.rows["scope:real_work"]["calls"] == 1


def test_profile_entry_preserves_every_scientific_argument_and_binds_source(tmp_path):
    from tools.timeline.qualification import profile_command, ROOT
    argv = ["python", "-m", "gpuwm.cli", "go", "input.toml", "--prepared-root", "/prepared"]
    assert profile_command(argv, ROOT)[2:] == argv[3:]
    assert argv[1:3] == ["-m", "gpuwm.cli"]
    with pytest.raises(ValueError, match="misidentify the measured engine"):
        profile_command(argv, tmp_path)
    with pytest.raises(ValueError, match="different entry point"):
        profile_command(["other"], ROOT)
