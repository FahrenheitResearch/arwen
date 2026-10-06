"""Direct Python entry points bind constants before preparation or execution."""
from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from types import SimpleNamespace

import pytest

from gpuwm import physics_params as pp
from gpuwm.config import RunConfig
from gpuwm.core import kernels, model
from gpuwm.experiment import experiment_from_run_config


class _ReachedBoundary(Exception):
    def __init__(self, experiment):
        self.experiment = experiment


@pytest.fixture(autouse=True)
def _binding(monkeypatch):
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    pp.reset_for_tests()
    yield
    pp.reset_for_tests()


def _experiment(pset=None):
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=10.0, run_seconds=60.0,
                    bl_pbl_physics=5, sf_sfclay_physics=5)
    return replace(experiment_from_run_config(cfg, datetime(2026, 10, 3)),
                   physics_params=pset)


def _set():
    return pp.make_set("api-set", {"mynn.prandtl": 0.8})


def _route(monkeypatch, name, tmp_path):
    """Stop at the first orchestration boundary, before any numeric work."""
    from gpuwm import runtime
    from gpuwm.ensemble import runtime_context

    calls = []
    if name == "run":
        class Session:
            def run_experiment(self, runner, exp, data, **kwargs):
                calls.append("dispatch")
                raise _ReachedBoundary(exp)
        monkeypatch.setattr(runtime_context, "current_session", lambda: Session())
        return lambda exp: runtime.run_experiment(exp, None, tmp_path / "output"), calls

    def catalog(data):
        calls.append("source preparation")
        return SimpleNamespace()

    def adaptation(exp, data, catalog):
        raise _ReachedBoundary(exp)

    monkeypatch.setattr(runtime, "_runtime_input_catalog", catalog)
    monkeypatch.setattr(model, "_adapt_experiment_vertical_for_case", adaptation)
    return lambda exp: model.build_experiment(exp, None), calls


@pytest.mark.parametrize("route", ("run", "build"))
def test_manual_parameter_set_binds_before_python_api_preparation(monkeypatch, tmp_path, route):
    exp = _experiment(_set())
    default_source = kernels.module_source("mynn_pbl")
    invoke, calls = _route(monkeypatch, route, tmp_path)
    with pytest.raises(_ReachedBoundary) as reached:
        invoke(exp)
    bound = reached.value.experiment
    assert bound is exp and pp.active() == exp.physics_params
    assert calls
    source = kernels.module_source("mynn_pbl")
    assert source != default_source
    assert source.count("const real pr = 0.8f,") == 3
    assert model.restart_identity_payload(bound)["physics_params"] == pp.document(pp.active())
    assert pp.wrfout_global_attrs() == {
        "GPUWM_PHYSICS_PARAMS": "api-set",
        "GPUWM_PHYSICS_PARAMS_SHA256": exp.physics_params.sha256(),
    }


@pytest.mark.parametrize("route", ("run", "build"))
def test_default_object_under_a_previous_set_refuses_before_preparation(monkeypatch, tmp_path, route):
    pp.declare(_set(), source="earlier experiment")
    invoke, calls = _route(monkeypatch, route, tmp_path)
    with pytest.raises(ValueError, match="One process runs one set"):
        invoke(_experiment())
    assert calls == []
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("route", ("run", "build"))
@pytest.mark.parametrize("already_issued", ("kernel", "tables"))
def test_a_late_set_refuses_issued_default_constants(monkeypatch, tmp_path, route, already_issued):
    if already_issued == "kernel":
        pp.note_compiled("mynn_pbl")
    else:
        assert pp.ruc_bundle_for_forecast(lambda: pytest.fail("default loader must stay untouched")) is None
    invoke, calls = _route(monkeypatch, route, tmp_path)
    with pytest.raises(ValueError, match="default constants"):
        invoke(_experiment(_set()))
    assert calls == [] and pp.active() is None


@pytest.mark.parametrize("route", ("run", "build"))
def test_environment_selection_is_attached_to_the_python_api_experiment(monkeypatch, tmp_path, route):
    path = tmp_path / "member.toml"
    path.write_text('[physics_params]\nname = "api-set"\nvalues = {"mynn.prandtl" = 0.8}\n',
                    encoding="utf-8")
    monkeypatch.setenv(pp.ENV_VAR, str(path))
    base = _experiment()
    invoke, _ = _route(monkeypatch, route, tmp_path)
    with pytest.raises(_ReachedBoundary) as reached:
        invoke(base)
    bound = reached.value.experiment
    assert bound is not base and base.physics_params is None
    assert bound.physics_params == pp.active()
    assert model.restart_identity_payload(bound)["physics_params"] == pp.document(pp.active())


@pytest.mark.parametrize("route", ("run", "build"))
def test_default_python_api_retains_config_identity_source_and_attrs(monkeypatch, tmp_path, route):
    exp = _experiment()
    identity = model.restart_identity_payload(exp)
    source = kernels.module_source("mynn_pbl")
    invoke, _ = _route(monkeypatch, route, tmp_path)
    with pytest.raises(_ReachedBoundary) as reached:
        invoke(exp)
    assert reached.value.experiment is exp
    assert model.restart_identity_payload(exp) == identity
    assert kernels.module_source("mynn_pbl") == source
    assert "physics_params" not in identity and pp.wrfout_global_attrs() == {}


def _tree(exp=None):
    tree = model.ExperimentState(root=None, nodes_by_grid_id={}, schedule=None,
                                 memory_ledger=None, experiment_fingerprint="metadata-only")
    if exp is not None:
        model.publish_declared_experiment(tree, exp)
    return tree


def _execution_boundary(monkeypatch):
    calls = []
    def policy(value):
        calls.append("execution")
        raise _ReachedBoundary(None)
    monkeypatch.setattr(model, "_resolve_pool_trim_policy", policy)
    return calls


def test_executor_cannot_first_apply_a_set_to_existing_state(monkeypatch):
    calls = _execution_boundary(monkeypatch)
    with pytest.raises(ValueError, match="already-built tree"):
        model.execute_experiment(_tree(), experiment=_experiment(_set()))
    assert calls == [] and pp.active() is None


def test_default_executor_keeps_the_original_experiment_argument_fallback():
    exp = _experiment()
    tree = _tree(exp)
    assert model._bind_execution_physics_params(tree, None) is None
    assert model._bind_execution_physics_params(tree, exp) is exp
    assert tree._declared_experiment is exp


def test_executor_refuses_default_state_under_a_previous_set(monkeypatch):
    pp.declare(_set(), source="earlier experiment")
    calls = _execution_boundary(monkeypatch)
    with pytest.raises(ValueError, match="One process runs one set"):
        model.execute_experiment(_tree(_experiment()))
    assert calls == []


def test_executor_retains_the_set_declared_before_tree_construction(monkeypatch):
    exp = _experiment(_set())
    pp.declare(exp.physics_params, source="before tree construction")
    pp.note_compiled("mynn_pbl")
    tree = _tree(exp)
    calls = _execution_boundary(monkeypatch)
    with pytest.raises(_ReachedBoundary):
        model.execute_experiment(tree)
    assert calls == ["execution"]
    assert tree._declared_experiment is exp and pp.active() == exp.physics_params


def test_executor_without_a_declaration_cannot_guess_active_constants(monkeypatch):
    pp.declare(_set(), source="other state")
    calls = _execution_boundary(monkeypatch)
    with pytest.raises(ValueError, match="without its physics parameter declaration"):
        model.execute_experiment(_tree())
    assert calls == []


def test_executor_records_environment_attachment_for_the_same_built_set(monkeypatch, tmp_path):
    path = tmp_path / "member.toml"
    path.write_text('[physics_params]\nname = "api-set"\nvalues = {"mynn.prandtl" = 0.8}\n',
                    encoding="utf-8")
    monkeypatch.setenv(pp.ENV_VAR, str(path))
    pset = _set()
    pp.declare(pset, source="before construction")
    pp.note_compiled("mynn_pbl")
    tree = _tree(_experiment(pset))
    base = _experiment()
    _execution_boundary(monkeypatch)
    with pytest.raises(_ReachedBoundary):
        model.execute_experiment(tree, experiment=base)
    assert tree._declared_experiment is not base
    assert tree._declared_experiment.physics_params == pset
    assert model.restart_identity_payload(tree._declared_experiment)["physics_params"] \
        == pp.document(pset)
