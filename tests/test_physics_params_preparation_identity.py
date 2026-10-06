"""Forecast constants bind trajectories while retaining preparation identity."""

from __future__ import annotations

import dataclasses
from datetime import datetime
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import physics_params as pp
from gpuwm.ensemble import runtime_preparation as preparation


@pytest.fixture(autouse=True)
def _binding(monkeypatch):
    monkeypatch.delenv(pp.ENV_VAR, raising=False)
    pp.reset_for_tests()
    yield
    pp.reset_for_tests()


def _experiment():
    from gpuwm.config import RunConfig
    from gpuwm.experiment import experiment_from_run_config

    cfg = RunConfig(nx=64, ny=64, nz=8, dx=3000.0, dy=3000.0,
                    ztop=20000.0, dt=10.0, run_seconds=3600.0)
    return dataclasses.replace(experiment_from_run_config(cfg, datetime(2026, 10, 3)),
                               feedback=1)


def _historical_document(exp):
    """Serialize the old dataclass shape, independently of preparation._plain."""
    fields = [field for field in dataclasses.fields(exp)
              if field.name != "physics_params"]
    old_type = dataclasses.make_dataclass("PreParameterExperiment",
        [(field.name, field.type) for field in fields])
    old = old_type(**{field.name: getattr(exp, field.name) for field in fields})

    def convert(value):
        if isinstance(value, dict):
            return {str(key): convert(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(item) for item in value]
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, Path):
            return str(value.resolve())
        if isinstance(value, np.ndarray):
            return {"shape": list(value.shape), "dtype": value.dtype.str,
                    "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}
        if isinstance(value, np.generic):
            return {"dtype": value.dtype.str, "words": value.tobytes().hex()}
        return value

    return convert(dataclasses.asdict(old))


def _canonical_bytes(document):
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode()


def test_preparation_document_and_digest_match_the_prior_dataclass_shape():
    exp = _experiment()
    old = _historical_document(exp)
    current = preparation._preparation_experiment_identity(exp)
    expected_bytes = _canonical_bytes(old)
    assert _canonical_bytes(current) == expected_bytes
    assert preparation._digest(current) == hashlib.sha256(expected_bytes).hexdigest()
    assert set(current) == {field.name for field in dataclasses.fields(exp)} - {"physics_params"}


def test_parameter_values_share_preparation_but_bind_distinct_trajectories():
    from gpuwm.core.model import restart_identity_payload
    from gpuwm.experiment import experiment_config_document

    base = _experiment()
    first = dataclasses.replace(base, physics_params=pp.make_set("first", {"mynn.prandtl": 0.8}))
    second = dataclasses.replace(base, physics_params=pp.make_set("second", {"mynn.prandtl": 0.9}))
    expected = _canonical_bytes(_historical_document(base))
    for exp in (base, first, second):
        assert _canonical_bytes(preparation._preparation_experiment_identity(exp)) == expected
    assert restart_identity_payload(first) != restart_identity_payload(second)
    for exp in (first, second):
        assert experiment_config_document(exp)["physics_params"] == pp.document(exp.physics_params)
    assert "physics_params" not in experiment_config_document(base)


@pytest.mark.parametrize("field,value", [("run_seconds", 7200.0), ("feedback", 0),
                                        ("column_chunk", 123)])
def test_every_existing_preparation_field_remains_bound(field, value):
    base = _experiment()
    assert hasattr(base, field)
    moved = dataclasses.replace(base, **{field: value})
    assert preparation._digest(preparation._preparation_experiment_identity(base)) \
        != preparation._digest(preparation._preparation_experiment_identity(moved))


def test_absent_set_adds_no_default_resolution_or_emitted_plan_key():
    from gpuwm.runplan import _schema_default_resolutions, _config_snapshot

    exp = _experiment()
    defaults = _schema_default_resolutions({})
    assert not any(row["key"] == "physics_params" for row in defaults)
    assert "physics_params" not in _config_snapshot(exp, None)["experiment"]
    active = dataclasses.replace(exp, physics_params=pp.make_set("member", {"mynn.prandtl": 0.8}))
    assert _config_snapshot(active, None)["experiment"]["physics_params"] \
        == pp.document(active.physics_params)


def test_root_cache_prepares_once_for_default_and_parameter_members(tmp_path, monkeypatch):
    from gpuwm import runtime
    from test_ensemble_runtime_preparation import _root_fixture

    exp, data, catalog, prepared, inputs = _root_fixture(tmp_path)
    monkeypatch.setattr(preparation, "_device_id", lambda: 7)
    monkeypatch.setattr(runtime, "case_static_fields", lambda *args, **kwargs: prepared.static_fields)
    source = preparation.RuntimePreparationSource(tmp_path / "owned")
    calls = []

    def build():
        calls.append("prepare")
        source.capture_root_inputs(**inputs)
        return prepared

    options = dict(grid=prepared.grid, selection=None, catalog=catalog,
                   scratch_arena=None, dycore_state_workspace=None, store_request=None, build=build)
    restored = object()
    monkeypatch.setattr(source, "_restore_root", lambda *args, **kwargs: calls.append("restore") or restored)
    with source.scope():
        assert source.prepare_root(exp, data, **options) is prepared
        for pset in (None, pp.make_set("first", {"mynn.prandtl": 0.8}),
                     pp.make_set("second", {"mynn.prandtl": 0.9})):
            member = SimpleNamespace(**vars(exp), physics_params=pset)
            assert source.prepare_root(member, data, **options) is restored
    assert calls == ["prepare", "restore", "restore", "restore"]
    assert len(source._roots) == 1
    assert source.receipt()["counts"]["root_preparations"] == 1
    source.close()
