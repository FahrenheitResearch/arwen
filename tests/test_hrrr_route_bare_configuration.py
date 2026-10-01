"""A native HRRR configuration with no namelists beside it.

The native HRRR route runs WRF namelists, not the TOML, and until 2.8.1
it found those files only when the chain started: a configuration no
door had saved -- written by a program, by hand, or by
``gpuwm import-namelist`` -- passed ``gpuwm go --dry-run`` and was
refused three seconds into the real run.  Now the run writes the set
from the configuration into its own folder, and a configuration the set
cannot carry is refused by the dry run and the real run alike, before
anything is fetched, with the source of the same model whose route
reads the configuration itself named as the way out.

CPU-only: the stages are captured at the ``run_stage`` seam.
"""

from __future__ import annotations

import contextlib
import io
import json
import tomllib
from pathlib import Path

import pytest

import gpuwm.capabilities as capabilities
import gpuwm.go_cli as go_cli
import gpuwm.prepared_domain_tree_forecast as tree
import gpuwm.runplan as runplan_module
from gpuwm.cli import main
from gpuwm.hrrr_route_inputs import render_namelist_input, route_input_paths
from gpuwm.runplan import PlanError, generate_intent_config
from gpuwm.toml_document import emit_experiment_toml

from test_runplan_hrrr_tree import _Observer, _plan, _stage


def _authored(tmp_path):
    """A nested HRRR configuration exactly as `gpuwm domain` writes it."""

    plan = _plan(tmp_path)
    with contextlib.redirect_stdout(io.StringIO()):
        config, _ = generate_intent_config(plan, destination=tmp_path / "authored")
    return plan, config


def _bare(config, *, keep_wps=False, edit=None):
    """The same configuration with its route namelists taken away."""

    companions = {role: path.read_bytes()
                  for role, path in route_input_paths(config).items()}
    for role, path in route_input_paths(config).items():
        if role == "wps_namelist" and keep_wps:
            continue
        path.unlink()
    if edit is not None:
        raw = tomllib.loads(config.read_text(encoding="utf-8"))
        edit(raw)
        config.write_text(emit_experiment_toml(raw), encoding="utf-8")
    return companions


def _experiment(config):
    from gpuwm.domain_wizard import experiment_from_text

    return experiment_from_text(config.read_text(encoding="utf-8"),
                                source=str(config))


def _moist_cq_the_namelists_cannot_state(raw):
    """moist_cq opposite to what the route derives for this suite."""

    raw["shared"]["moist_cq"] = not raw["shared"]["moist_cq"]


def _drive_chain(plan, config, exp, monkeypatch, observer=None):
    staged: list[tuple[str, list[str]]] = []
    tree_root = plan.run_dir / "chain" / "hrrr-hierarchy"

    def run_stage(label, command, **kwargs):
        staged.append((label, [str(part) for part in command]))
        if label == "hierarchy":
            tree_root.mkdir(parents=True, exist_ok=True)
            (tree_root / "receipt.json").write_text(json.dumps(
                {"schema": "gpuwm-native-hrrr-hierarchy-direct-v1",
                 "status": "PASS"}), encoding="utf-8")

    def fake_fetch(arguments, run_dir, **_kwargs):
        out = Path(arguments[arguments.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "SHA256SUMS").write_text("x", encoding="utf-8")
        return {}

    monkeypatch.setattr(go_cli, "run_stage", run_stage)
    monkeypatch.setattr(runplan_module, "_run_fetch", fake_fetch)
    monkeypatch.setattr(tree, "main", lambda argv, *, observer=None: 0)
    monkeypatch.setattr(runplan_module, "_chain_render",
                        lambda plan, **kwargs: {"ok": True})
    with contextlib.redirect_stdout(io.StringIO()):
        runplan_module._hrrr_chain(plan, config_path=config, exp=exp,
                                   observer=observer or _Observer(),
                                   run_dir=plan.run_dir)
    return staged


def test_a_bare_configuration_runs_from_namelists_written_into_its_run_folder(
        tmp_path, monkeypatch):
    plan, config = _authored(tmp_path)
    written = _bare(config, keep_wps=True)
    exp = _experiment(config)

    staged = _drive_chain(plan, config, exp, monkeypatch)

    prepare = _stage(staged, "prepare")
    hierarchy = _stage(staged, "hierarchy")
    into = plan.run_dir / "chain" / "route-inputs"
    used = route_input_paths(into / config.name)
    assert prepare[prepare.index("--namelist-input") + 1] == str(used["namelist_input"])
    assert prepare[prepare.index("--domain-spec") + 1] == str(used["target_domain"])
    assert (hierarchy[hierarchy.index("--stock-wrf-namelist-input") + 1]
            == str(used["stock_namelist_input"]))
    # The files the run wrote are byte for byte the ones the door wrote,
    # because both come from one renderer over one configuration.
    for role in ("target_domain", "namelist_input", "stock_namelist_input"):
        assert used[role].read_bytes() == written[role], role
    # Nothing is written beside the user's configuration.
    assert not any(path.exists() for role, path in route_input_paths(config).items()
                   if role != "wps_namelist")


def test_a_bare_configuration_with_no_wps_namelist_gets_one_rendered(
        tmp_path, monkeypatch):
    plan, config = _authored(tmp_path)
    _bare(config)
    staged = _drive_chain(plan, config, _experiment(config), monkeypatch)

    prepare = _stage(staged, "prepare")
    wps = Path(prepare[prepare.index("--wps-namelist") + 1])
    assert wps.parent == plan.run_dir / "chain" / "route-inputs"
    assert "interval_seconds = 3600" in wps.read_text(encoding="utf-8")


def _front_door_sees_the_gpu_runtime(monkeypatch):
    """The install preflight answers as on an install with CuPy.

    A real ``gpuwm go`` passes the front door's install preflight
    (``gpuwm.cli.main``) before it dispatches, and on an install without
    CuPy that refuses first, for a reason that is not this
    configuration's, so the comparison below would read the install
    refusal against the dry run's route refusal.  Only the preflight's
    question is answered; the runtime itself is not provided.  On an
    install without CuPy, then, any import of it on the way to the route
    check still fails and ``gpuwm go`` reports the missing runtime, so
    this test also proves the real run refuses this configuration
    without loading CUDA.
    """

    installed = capabilities.is_installed
    monkeypatch.setattr(
        capabilities, "is_installed",
        lambda module: (str(module).split(".", 1)[0]
                        == capabilities.GPU_RUNTIME.module
                        or installed(module)))


def test_the_dry_run_and_the_real_run_refuse_what_the_namelists_cannot_carry(
        tmp_path, monkeypatch, capsys):
    plan, config = _authored(tmp_path)
    _bare(config, keep_wps=True, edit=_moist_cq_the_namelists_cannot_state)
    exp = _experiment(config)
    capsys.readouterr()

    def refusal(err):
        return [line for line in err.splitlines()
                if line.startswith("gpuwm go:")]

    with monkeypatch.context() as doors:
        _front_door_sees_the_gpu_runtime(doors)
        assert main(["go", str(config), "--dry-run"]) == 2
        dry = refusal(capsys.readouterr().err)
        outdir = tmp_path / "real-run"
        assert main(["go", str(config), "--outdir", str(outdir)]) == 2
        real = refusal(capsys.readouterr().err)
    with pytest.raises(PlanError) as chain:
        _drive_chain(plan, config, exp, monkeypatch)

    assert len(dry) == 1
    for said in (dry[0], real[0], str(chain.value)):
        assert "moist_cq" in said
        assert "namelist.input" in said or "namelist_input" in said
        assert '[fetch] source = "hrrr-prs"' in said
    # The same sentence at both doors, and the real run refused before it
    # claimed a run folder or fetched a byte.
    assert dry == real
    assert not outdir.exists()


def test_the_way_out_is_read_off_the_source_table():
    from gpuwm.hrrr_route_inputs import configuration_reading_sources

    assert configuration_reading_sources("hrrr") == ("hrrr-prs",)
    assert "hrrr" not in configuration_reading_sources("hrrr")
    assert configuration_reading_sources("not-a-source") == ()


def test_an_imported_configuration_gets_the_namelists_it_was_imported_from(
        tmp_path):
    """`gpuwm import-namelist` writes radiation in the aggregate spelling.

    The route's renderer wrote that as ``ra_lw_physics = -1``, which its
    own importer refuses, so an uploaded WRF namelist could never run
    on the native HRRR route.
    """

    from gpuwm.namelist_import import import_namelists

    _plan_unused, config = _authored(tmp_path)
    paths = route_input_paths(config)
    text, _report = import_namelists(paths["wps_namelist"],
                                     paths["namelist_input"], name="uploaded")
    raw = tomllib.loads(text)
    assert raw["shared"].get("ra_physics") == 4
    assert "ra_lw_physics" not in raw["shared"]
    raw["fetch"] = tomllib.loads(config.read_text(encoding="utf-8"))["fetch"]
    uploaded = tmp_path / "upload" / "uploaded.toml"
    uploaded.parent.mkdir()
    uploaded.write_text(emit_experiment_toml(raw), encoding="utf-8")
    exp = _experiment(uploaded)

    from gpuwm.namelist_import import parse_namelist_text

    rendered = render_namelist_input(exp)
    physics = parse_namelist_text(rendered)["physics"]
    assert physics["ra_lw_physics"] == [4] * len(exp.domains)
    assert physics["ra_sw_physics"] == [4] * len(exp.domains)

    from gpuwm.hrrr_route_inputs import run_route_inputs

    assert run_route_inputs(uploaded, exp, raw=raw) is None
    written = run_route_inputs(uploaded, exp, raw=raw,
                               into=tmp_path / "run" / "route-inputs")
    assert written["namelist_input"].read_text(encoding="utf-8") == rendered


# ---------------------------------------------------------------------------
# A child that never mentions radiation.  `radt = 0` means "not stated
# here", so the child runs radt_minutes (twelve minutes), and the route's
# namelist states that cadence as radt = 12.  The round trip compared the
# raw fields and refused the pair as a different forecast, which turned
# away every bare nested configuration whose child leaves radt unset.


def _child_states_no_radiation_cadence(raw):
    for row in raw["domain"][1:]:
        row.pop("radt", None)
        row.pop("radt_minutes", None)


def _config_plan(tmp_path, config):
    from gpuwm.runplan import PLAN_SCHEMA, load_plan

    path = tmp_path / "bare-plan.json"
    path.write_text(json.dumps({
        "schema": PLAN_SCHEMA, "name": "bare-plan", "route": "prepared",
        "config": {"path": str(config)},
        "output_root": str(tmp_path / "bare-out")}), encoding="utf-8")
    return load_plan(path)


def test_a_child_with_no_radt_resolves_and_runs_on_its_effective_cadence(
        tmp_path, monkeypatch):
    from gpuwm.config import effective_radt_minutes
    from gpuwm.hrrr_route_inputs import run_route_inputs
    from gpuwm.namelist_import import parse_namelist_text
    from gpuwm.runplan import resolve_plan

    plan, config = _authored(tmp_path)
    _bare(config, edit=_child_states_no_radiation_cadence)
    exp = _experiment(config)
    raw = tomllib.loads(config.read_text(encoding="utf-8"))
    child = exp.domains[1].run
    # The case under test: the child's row says nothing, and it runs the
    # twelve minutes radt_minutes carries.
    assert child.radt == 0.0
    assert effective_radt_minutes(child) == 12.0

    assert run_route_inputs(config, exp, raw=raw) is None
    resolve_plan(_config_plan(tmp_path, config), require_inputs=False)

    staged = _drive_chain(plan, config, exp, monkeypatch)
    prepare = _stage(staged, "prepare")
    namelist = Path(prepare[prepare.index("--namelist-input") + 1])
    radt = parse_namelist_text(
        namelist.read_text(encoding="utf-8"))["physics"]["radt"]
    assert [float(value) for value in radt] == [
        effective_radt_minutes(domain.run) for domain in exp.domains]


def test_the_identity_compares_the_radiation_cadence_a_domain_runs():
    from gpuwm.ingest.prepared_cache import effective_prepared_domain_config

    def cadence(radt, radt_minutes):
        return effective_prepared_domain_config(
            {"run": {"radt": radt, "radt_minutes": radt_minutes}})["run"]

    # Unstated radt runs radt_minutes; a positive radt overrides it.
    assert cadence(0.0, 12.0) == cadence(12.0, 12.0) == cadence(12.0, 5.0)
    # A different cadence is still a different forecast.
    assert cadence(0.0, 12.0) != cadence(6.0, 12.0)
    assert cadence(0.0, 0.0) != cadence(0.0, 12.0)


class _Warnings(_Observer):
    """Records the chain's warnings and swallows the rest."""

    def __init__(self):
        self.said = []

    def warn(self, code, message, **fields):
        self.said.append((code, message, fields))


def test_a_partial_set_says_the_namelists_beside_it_are_not_what_runs(
        tmp_path, monkeypatch):
    """A hand-edited namelist.input beside an incomplete set does not run.

    The run writes the whole set from the configuration, so an edit to
    the one file left beside it would look applied and not be.  The run
    says which files it did not read and where the set it ran is.
    """

    plan, config = _authored(tmp_path)
    beside = route_input_paths(config)
    beside["target_domain"].unlink()
    observer = _Warnings()
    _drive_chain(plan, config, _experiment(config), monkeypatch,
                 observer=observer)

    (code, message, fields), = [said for said in observer.said
                                if said[0] == "route_inputs_rendered"]
    assert fields["unread"] == [beside["namelist_input"].name,
                                beside["stock_namelist_input"].name]
    assert fields["missing"] == [beside["target_domain"].name]
    assert fields["written"] == str(plan.run_dir / "chain" / "route-inputs")
    assert beside["namelist_input"].name in message


def test_a_complete_set_is_read_where_it_sits_and_says_nothing(
        tmp_path, monkeypatch):
    plan, config = _authored(tmp_path)
    observer = _Warnings()
    staged = _drive_chain(plan, config, _experiment(config), monkeypatch,
                          observer=observer)
    prepare = _stage(staged, "prepare")
    assert (prepare[prepare.index("--namelist-input") + 1]
            == str(route_input_paths(config)["namelist_input"]))
    assert not [said for said in observer.said
                if said[0] == "route_inputs_rendered"]
