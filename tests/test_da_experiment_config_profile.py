"""``--physics-profile experiment-config`` binds the same way on the DA
controller and its packed workers (CPU).

Breakage reproduced here: the controller mapped the name to ``None`` for
its own preflight and then named its ensemble identity
``"experiment-config"``; its workers replayed the shared argument record
(``null``) and rebuilt the identity with profile ``"None"``, so every
member leg was refused ("worker ensemble identity differs from the public
prepared binding", box S 2026-10-06 18:51Z) and the cycle never ran a
forecast.  The string now stays the string on both sides and
``preflight_prepared_forecast`` binds it as the experiment config's own
physics.
"""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest

from gpuwm import prepared_single_domain_forecast as psdf
from gpuwm.da import member_transport as transport
from tools.da_ensemble_state import EnsembleIdentity

import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "_da_member_transport_fixtures",
    Path(__file__).with_name("test_da_member_transport.py"))
_fixtures = importlib.util.module_from_spec(_spec)
import sys
sys.modules[_spec.name] = _fixtures
_spec.loader.exec_module(_fixtures)
HotStart, Perturbation = _fixtures.HotStart, _fixtures.Perturbation
make_context, patch_bindings, request = (
    _fixtures.make_context, _fixtures.patch_bindings, _fixtures.request)


def test_the_experiment_config_name_binds_as_no_named_profile():
    assert psdf.EXPERIMENT_CONFIG_PROFILE == "experiment-config"
    assert psdf.physics_profile_binding("experiment-config") is None
    assert psdf.physics_profile_binding(None) is None
    assert psdf.physics_profile_binding("some-named-profile") == "some-named-profile"


def _context_with_profile(tmp_path, *, args_profile, identity_profile):
    from tools.da_member_leg import MemberLegContext
    ctx = make_context(tmp_path)
    identity = EnsembleIdentity(
        members=2, nx=3, ny=4, nz=5, dt_s=15.0, mp_physics=6,
        physics_profile=identity_profile, prepared_content_sha256="0" * 64)
    args = argparse.Namespace(**{**vars(ctx.args),
                                 "physics_profile": args_profile})
    return MemberLegContext(
        args=args, inputs=ctx.inputs, identity=identity,
        cfg_perturb=Perturbation(), hot_cfg=HotStart(),
        leg=0, absolute_leg_number=0, t_start=0.0, t_end=60.0,
        stage_root=tmp_path / "stage", out=tmp_path / "out",
        setup_arrays=ctx.setup_arrays, thb_host=ctx.thb_host)


def test_worker_rebinds_the_controllers_experiment_config_identity(tmp_path, monkeypatch):
    """The controller's identity is built from ``str(args.physics_profile)``
    (tools/da_cycle_prepared.cycle) and the worker's from the same string
    out of the shared argument record: with the name kept as a string the
    two agree and the leg is admitted."""
    ctx = _context_with_profile(
        tmp_path, args_profile=psdf.EXPERIMENT_CONFIG_PROFILE,
        identity_profile=str(psdf.EXPERIMENT_CONFIG_PROFILE))
    patch_bindings(monkeypatch, ctx)
    path, job = request(tmp_path, ctx)
    loaded, _ = transport.read_context(path, expected_hash=job["request_hash"])
    assert loaded.identity == ctx.identity
    assert loaded.identity.physics_profile == "experiment-config"


def test_the_nulled_controller_argument_is_the_refusal_the_box_saw(tmp_path, monkeypatch):
    """The shape the controller produced before the fix: argument ``None``
    in the shared record, identity named ``"experiment-config"``.  The
    worker's ``str(None)`` is ``"None"`` and the leg is refused with the
    exact sentence from the box's control.log."""
    ctx = _context_with_profile(
        tmp_path, args_profile=None, identity_profile="experiment-config")
    patch_bindings(monkeypatch, ctx)
    path, job = request(tmp_path, ctx)
    with pytest.raises(ValueError, match="worker ensemble identity differs from the public prepared binding"):
        transport.read_context(path, expected_hash=job["request_hash"])


def test_the_controller_no_longer_nulls_the_name():
    """The controller keeps the parsed string; the preflight binds it."""
    import inspect
    from tools import da_cycle_prepared
    source = inspect.getsource(da_cycle_prepared.cycle)
    assert "args.physics_profile = None" not in source
    assert 'physics_profile or "experiment-config"' not in source
    assert "physics_profile=str(args.physics_profile)," in source
