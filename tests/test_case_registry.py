# tests/test_case_registry.py
"""Cases are data: the driver discovers them instead of naming them.

The load-bearing assertion is :func:`test_a_new_case_needs_no_driver_edit`
-- a case module written into a directory this repository does not contain
becomes a ``gpuwm verify`` choice, runs, and has its gates enforced, with
no edit to ``gpuwm/cli.py``.  Everything else pins the classification that
makes that safe: capabilities come from the entry points a module declares,
a helper module is not a case, and enumerating cases does not import them.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from gpuwm.verify import cases

ROOT = Path(__file__).resolve().parents[1]

_VERIFY_CASE = '''\
"""Scratch verification case planted outside the repository."""
GATES = {"peak": (1.0, 3.0)}
PEAK = 2.0


def run(outdir=None) -> dict:
    return {"nan": False, "peak": PEAK}
'''

_HELPER_MODULE = '''\
"""Shared helper: declares no case entry point, so it is not a case."""
SHARED_CONSTANT = 3


def helper(value):
    return value + SHARED_CONSTANT
'''

_CONFIG_CASE = '''\
"""Scratch config-driven case: the static/ingest/run entry points."""


def write_static(cfg, output):
    return output


def write_ingest(cfg, output):
    return output


def run_config(cfg, outdir, restart=None):
    raise NotImplementedError
'''


def _plant(directory: Path, name: str, body: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.py"
    path.write_text(body, encoding="utf-8")
    return path


def test_capabilities_come_from_declared_entry_points(tmp_path):
    verify = _plant(tmp_path, "probe_verify", _VERIFY_CASE)
    config = _plant(tmp_path, "probe_config", _CONFIG_CASE)
    helper = _plant(tmp_path, "probe_helper", _HELPER_MODULE)
    assert cases.capabilities_of(verify) == {cases.VERIFY}
    assert cases.capabilities_of(config) == {cases.CONFIG}
    assert cases.capabilities_of(helper) == frozenset()


def test_shipped_cases_are_discovered_with_their_capabilities():
    """The two hand-written driver dicts are reproduced by discovery.

    Superset assertions: adding a case must not fail this test.  What is
    pinned is that nothing that used to be dispatchable stopped being so,
    and that the shared nest helper is still not mistaken for a case."""
    found = cases.discover()
    assert {"straka", "igw", "hill2d", "moist_bubble", "wk82",
            "real74_d01"} <= set(cases.names(cases.VERIFY))
    assert "real74_d01" in cases.names(cases.CONFIG)
    assert cases.CONFIG in found["real74_d01"].capabilities
    assert cases.VERIFY in found["real74_d01"].capabilities
    # nest_ideal_common is a helper shared by the nest cases, not a case.
    assert "nest_ideal_common" not in found


def test_enumerating_cases_does_not_import_them():
    """Discovery is a static read, so listing cases stays cheap and free
    of import side effects -- the property that lets the CLI build its
    --help without pulling netCDF/cupy through a real-data case.

    Membership and the argparse choices are covered too: the Mapping
    default answers ``in`` by indexing, which would import the module just
    to learn that it exists."""
    probe = subprocess.run(
        [sys.executable, "-c",
         "import sys\n"
         "import gpuwm.cli as cli\n"
         "from gpuwm.verify import cases\n"
         "records = cases.manifest()\n"
         "assert records, 'no cases discovered'\n"
         "assert sorted(cli._CASES)\n"
         "assert 'no_such_case' not in cli._CASES\n"
         "assert 'no_such_case' not in cli._GATES\n"
         "leaked = [m for m in sys.modules\n"
         "          if m.startswith('gpuwm.verify.cases.')]\n"
         "print(leaked)\n"],
        capture_output=True, text=True, cwd=ROOT)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    assert probe.stdout.strip() == "[]", probe.stdout


def test_a_new_case_needs_no_driver_edit(tmp_path, monkeypatch, capsys):
    """A case module written outside this repository becomes a verify
    choice, runs, and is gated -- with no edit to gpuwm/cli.py.

    Both directions are asserted: a case whose metric sits inside its own
    GATES exits 0, and one whose metric sits outside exits 1 naming the
    gate.  A discovered case that could not fail would prove nothing."""
    import gpuwm.cli as cli

    external = tmp_path / "extra_cases"
    _plant(external, "probe_pass", _VERIFY_CASE)
    _plant(external, "probe_fail",
           _VERIFY_CASE.replace("PEAK = 2.0", "PEAK = 9.0"))
    monkeypatch.setenv(cases.CASE_PATH_ENV, str(external))

    assert {"probe_pass", "probe_fail"} <= set(cases.names(cases.VERIFY))
    assert "probe_pass" in cli._CASES
    assert cli._GATES["probe_pass"] == {"peak": (1.0, 3.0)}
    assert cli.main(["verify", "probe_pass"]) == 0
    assert cli.main(["verify", "probe_fail"]) == 1
    err = capsys.readouterr().err
    assert "FAIL" in err and "peak" in err


def test_external_case_cannot_shadow_a_shipped_one(tmp_path, monkeypatch):
    """GPUWM_CASE_PATH extends the registry; it never overrides it."""
    external = tmp_path / "extra_cases"
    _plant(external, "straka", _VERIFY_CASE)
    monkeypatch.setenv(cases.CASE_PATH_ENV, str(external))
    record = cases.discover()["straka"]
    assert record.packaged
    assert record.path.parent == Path(cases.__file__).resolve().parent


def test_unknown_case_and_wrong_capability_fail_loud(tmp_path, monkeypatch):
    external = tmp_path / "extra_cases"
    _plant(external, "probe_config_only", _CONFIG_CASE)
    monkeypatch.setenv(cases.CASE_PATH_ENV, str(external))
    with pytest.raises(KeyError, match="not a discovered"):
        cases.entry("no_such_case")
    with pytest.raises(ValueError, match="does not provide"):
        cases.require("probe_config_only", cases.VERIFY)


def test_manifest_is_json_serializable_for_a_front_end():
    payload = json.loads(json.dumps(cases.manifest()))
    assert payload
    for record in payload:
        assert set(record) == {"name", "module", "path", "packaged",
                               "capabilities"}
        assert record["capabilities"]
        assert set(record["capabilities"]) <= set(cases.CAPABILITY_CONTRACTS)


def test_cases_subcommand_prints_the_registry():
    probe = subprocess.run(
        [sys.executable, "-m", "gpuwm.cli", "cases", "--json"],
        capture_output=True, text=True, cwd=ROOT)
    assert probe.returncode == 0, probe.stdout + probe.stderr
    records = json.loads(probe.stdout)
    assert {record["name"] for record in records} == set(cases.discover())
