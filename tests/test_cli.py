# tests/test_cli.py
"""CLI verify-driver tests.

The gate plumbing is CPU-only: the single-sourced GATES exports (each case
module owns its gate intervals; cli._GATES and the benchmark tests consume
that one export), the _failing_gate semantics (strict bounds, None =
unbounded on EITHER side, NaN fails), and the exit-0/exit-1 paths via a
stubbed case runner.  The end-to-end straka subprocess run stays a GPU
test.
"""
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import requires_gpu

import gpuwm.cli as cli


def _passing_straka() -> dict:
    """Metrics inside every straka gate (measured Phase 1 values)."""
    return {"nan": False, "theta_min": -9.58, "front_km": 15.34,
            "symmetry_err": 0.014, "w_max": 20.0}


def test_gates_single_sourced_from_case_modules():
    """cli._GATES holds each case module's own GATES export (identical
    object, not a copy), for every discovered case.

    Asserted as a SUPERSET, not an exact set: the case tables are now
    discovery-driven, so dropping a module into gpuwm/verify/cases/ must
    not turn into a failing driver test.  What is pinned is that the cases
    that were registered by hand are still reachable, and that every
    discovered case's gate table is well formed."""
    assert {"straka", "igw", "hill2d", "moist_bubble", "wk82",
            "real74_d01"} <= set(cli._CASES)
    assert set(cli._GATES) == set(cli._CASES)
    for name, mod in cli._CASES.items():
        assert cli._GATES[name] is mod.GATES, name
        for metric, (lo, hi) in mod.GATES.items():
            assert lo is not None or hi is not None, (name, metric)
            if lo is not None and hi is not None:
                assert lo < hi, (name, metric)


def test_failing_gate_passes_and_names_failures():
    m = _passing_straka()
    assert cli._failing_gate("straka", m) is None
    assert cli._failing_gate("straka", {**m, "nan": True}) == "nan"
    assert cli._failing_gate("straka", {**m, "theta_min": -7.9}) \
        == "theta_min"
    # bounds are strict: landing exactly on a bound fails
    assert cli._failing_gate("straka", {**m, "front_km": 17.0}) == "front_km"
    # NaN metric values compare False and fail their gate
    assert cli._failing_gate("straka", {**m, "front_km": float("nan")}) \
        == "front_km"


def test_failing_gate_none_unbounded_on_either_side():
    """None means unbounded on THAT side: (None, hi) has no lower bound
    and (lo, None) no upper bound.  The Phase 1 implementation only
    handled lo=None and raised TypeError on (lo, None) gates."""
    m = _passing_straka()
    # straka symmetry_err is (None, 0.05): arbitrarily low passes
    assert cli._failing_gate("straka", {**m, "symmetry_err": -1.0e9}) is None
    # hill2d w_corr is (0.95, None): arbitrarily high passes
    hm = {"nan": False, "w_corr": 1.0e9, "mflux_dev": 0.05,
          "mflux_mean": -0.01, "mflux_ratio": 1.0}
    assert cli._failing_gate("hill2d", hm) is None
    assert cli._failing_gate("hill2d", {**hm, "w_corr": 0.5}) == "w_corr"


def test_cli_exit_codes(monkeypatch, capsys):
    """main() returns 0 when every gate passes and 1 naming the failing
    gate on stderr -- exercised CPU-only through a stubbed case runner
    (the real gate table stays in force)."""
    good = _passing_straka()
    monkeypatch.setitem(cli._CASES, "straka",
                        SimpleNamespace(run=lambda outdir: dict(good)))
    assert cli.main(["verify", "straka"]) == 0
    captured = capsys.readouterr()
    assert "theta_min" in captured.out
    assert "FAIL" not in captured.err

    bad = {**good, "w_max": 55.0}                 # w_max gate is (None, 40)
    monkeypatch.setitem(cli._CASES, "straka",
                        SimpleNamespace(run=lambda outdir: bad))
    assert cli.main(["verify", "straka"]) == 1
    err = capsys.readouterr().err
    assert "FAIL" in err and "w_max" in err


def test_cli_case_choices():
    """The Phase 2 cases are wired into the verify subcommand; an unknown
    case is an argparse usage error (exit 2)."""
    for case in ("straka", "igw", "hill2d", "moist_bubble", "wk82",
                 "real74_d01"):
        assert case in cli._GATES, case
    with pytest.raises(SystemExit) as exc:
        cli.main(["verify", "nosuchcase"])
    assert exc.value.code == 2


def test_real_case_subcommands_dispatch_loaded_config(monkeypatch, tmp_path,
                                                       capsys):
    """static/ingest/run are config-driven and dispatch to the named real
    case without duplicating case-specific pipeline logic in the CLI."""
    calls = []
    cfg = SimpleNamespace(case="real74_d01")
    fake = SimpleNamespace(
        write_static=lambda loaded, output: calls.append(
            ("static", loaded, output)) or output,
        write_ingest=lambda loaded, output: calls.append(
            ("ingest", loaded, output)) or output,
        run_config=lambda loaded, outdir, restart=None: calls.append(
            ("run", loaded, outdir, restart)) or SimpleNamespace(
                wrfout_paths=(), completed_seconds=3600.0, nan_free=True),
    )
    monkeypatch.setattr(cli, "load_config", lambda path: cfg)
    monkeypatch.setitem(cli._REAL_CASES, "real74_d01", fake)
    config_path = tmp_path / "real74.toml"
    static_path = tmp_path / "static.npz"
    ingest_path = tmp_path / "initial.npz"
    run_dir = tmp_path / "run"

    assert cli.main(["static", str(config_path),
                     "--output", str(static_path)]) == 0
    assert cli.main(["ingest", str(config_path),
                     "--output", str(ingest_path)]) == 0
    assert cli.main(["run", str(config_path),
                     "--outdir", str(run_dir)]) == 0

    assert calls == [
        ("static", cfg, static_path),
        ("ingest", cfg, ingest_path),
        ("run", cfg, run_dir, None),
    ]
    output = capsys.readouterr().out
    assert str(static_path) in output
    assert str(ingest_path) in output
    assert "real74_d01" in output


def test_real_case_config_name_is_required(monkeypatch, tmp_path, capsys):
    # User-facing config refusals follow the uniform CLI boundary: the
    # ValueError message is printed, exit 2, no traceback.
    monkeypatch.setattr(
        cli, "load_config", lambda path: SimpleNamespace(case=""))
    rc = cli.main(["static", str(tmp_path / "missing-case.toml")])
    assert rc == 2
    err = capsys.readouterr().err
    assert err.startswith("gpuwm static:") and "case" in err


def test_missing_cupy_is_a_clean_install_gap_message(monkeypatch,
                                                     tmp_path, capsys):
    """Base-install users (no [gpu] extra) hitting a GPU-needing command
    get the exact pip remedy at exit 2, never a raw ModuleNotFoundError
    traceback (acceptance F3)."""
    import gpuwm.domain_wizard as domain_wizard

    def missing_cupy(**kwargs):
        raise ModuleNotFoundError("No module named 'cupy'", name="cupy")

    monkeypatch.setattr(domain_wizard, "fit_ladder", missing_cupy)
    rc = cli.main(["domain", "--point", "35.2,-97.4",
                   "--cycle", "2026-07-28T06",
                   "--out", str(tmp_path / "area.toml")])
    assert rc == 2
    err = capsys.readouterr().err
    assert "gpuwm[gpu]" in err and "cupy-cuda12x" in err
    assert "gpuwm doctor" in err and "Traceback" not in err


def test_committed_real74_config_matches_case_defaults():
    from gpuwm.config import load_config
    from gpuwm.verify.cases.real74_d01 import phase3_config

    assert load_config(Path("configs/real74_d01.toml")) == phase3_config()


@pytest.mark.gpu
@requires_gpu
def test_cli_straka(tmp_path):
    r = subprocess.run([sys.executable, "-m", "gpuwm.cli", "verify", "straka",
                        "--outdir", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "theta_min" in r.stdout
    assert any(p.name.startswith("wrfout_") for p in tmp_path.iterdir())
