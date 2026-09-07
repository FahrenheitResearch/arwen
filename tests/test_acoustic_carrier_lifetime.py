"""Checkpoint-required acoustic flux belongs to a domain across tree turns."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu
from gpuwm.config import RunConfig


def test_checkpoint_acoustic_carrier_does_not_alias_shared_tree_workspace(monkeypatch):
    from gpuwm.core import preflight, state as state_module
    from gpuwm.io.restart import classify_state_attr, state_manifest
    monkeypatch.setattr(state_module, "cp", np)
    large_cfg = RunConfig(nx=11, ny=7, nz=5, dx=500., dy=500., dt=3., ztop=8000., run_seconds=0.)
    small_cfg = replace(large_cfg, nx=8, ny=6, nz=4)
    domains = tuple(SimpleNamespace(run=cfg) for cfg in (large_cfg, small_cfg))
    workspace = state_module.build_shared_dycore_state_workspace(domains)
    parent, child = (state_module.DomainState(dc.run, dycore_state_workspace=workspace)
                     for dc in domains)
    parent.ww_pp.fill(7.)
    child.ww_pp.fill(11.)
    np.testing.assert_array_equal(parent.ww_pp, np.full(parent.ww_pp.shape, 7., np.float32))
    assert not np.shares_memory(parent.ww_pp, child.ww_pp)
    assert "ww_pp" not in preflight.shared_dycore_state_symbols()
    assert "ww_pp" not in preflight.shared_dycore_state_workspace_shapes(domains)
    assert classify_state_attr("ww_pp") == "checkpoint_only"
    assert state_manifest(parent)["acoustic/ww_pp"] is parent.ww_pp
    assert workspace.nbytes == preflight.shared_dycore_state_workspace_bytes(domains)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("specified", [True, False])
@pytest.mark.parametrize("mapped", [True, False])
def test_acoustic_boundary_flux_survives_until_actual_sumflux_consumption(specified, mapped):
    import cupy as cp
    from gpuwm.core.acoustic import acoustic_substep_explicit
    from gpuwm.core.dycore import _sumflux_launch
    from gpuwm.verify.npref import random_acoustic_state

    state, cfg = random_acoustic_state(seed=814, nz=8, ny=8, nx=12,
                                       msf_amp=0.1 if mapped else 0.)
    cfg = replace(cfg, specified=specified, spec_zone=1, relax_zone=4,
                  spec_bdy_width=5)
    sentinel = cp.float32(7.)
    state.ww_pp.fill(sentinel)
    acoustic_substep_explicit(state, cfg, dtau=0.5, first=True)
    targets = tuple(cp.zeros_like(a) for a in (state.u_pp, state.v_pp, state.ww_pp))
    _sumflux_launch("accumulate_sumflux", targets,
                    (state.u_pp, state.v_pp, state.ww_pp))
    got = cp.asnumpy(targets[2])
    frame = np.zeros((cfg.ny, cfg.nx), bool)
    frame[[0, -1], :] = True
    frame[:, [0, -1]] = True
    if specified:
        np.testing.assert_array_equal(got[:, frame], np.full(got[:, frame].shape, 7., np.float32))
    else:
        assert not np.any(got == 7.)
    assert not np.any(got[:, 1:-1, 1:-1] == 7.)
