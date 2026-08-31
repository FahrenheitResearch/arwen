"""Run a real mp=50 step, then classify what the step actually produced.

The CPU sibling (``tests/test_p3_restart_manifest.py``) reads
``gpuwm/core/p3.py`` and classifies the attribute names it finds assigned.
That is a source scan, and a source scan can be defeated by an attribute
assigned somewhere else -- inside ``gpuwm/core/p3_device.py``, or through a
helper.  This one takes the state a real device step leaves behind and
classifies every attribute ON IT, which is the artifact rather than the
source.

MEASURED 2026-08-29: without ``_p3_workspace`` classified, this test's
``state_manifest`` call raises exactly the error that killed every mp=50
forecast in ``canonical_state_digest`` before its first history frame.
"""
from __future__ import annotations

import pytest

from gpuwm.config import RunConfig, validate_run_config

pytestmark = pytest.mark.gpu


def _state():
    from gpuwm.core.state import DomainState
    nz, ny, nx = 12, 6, 6
    cfg = validate_run_config(RunConfig(
        nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, ztop=20000.0, dt=20.0,
        run_seconds=40.0, mp_physics=50, moist=True))
    return DomainState(cfg), cfg


def test_a_real_mp50_step_leaves_a_classifiable_state():
    pytest.importorskip("cupy")
    from gpuwm.core import microphysics
    from gpuwm.io import restart as restart_io

    state, cfg = _state()
    restart_io.state_manifest(state)            # clean before the step
    microphysics.apply(state, cfg, 20.0)
    assert hasattr(state, "_p3_workspace"), (
        "the device adapter no longer caches a workspace on the state; if "
        "that is deliberate, retire this test and the restart row with it")
    manifest = restart_io.state_manifest(state)
    assert manifest


def test_every_attribute_a_real_step_leaves_is_classified():
    pytest.importorskip("cupy")
    from gpuwm.core import microphysics
    from gpuwm.io.restart import (RestartManifestError, classify_state_attr)

    state, cfg = _state()
    microphysics.apply(state, cfg, 20.0)
    unclassified = []
    for name in vars(state):
        try:
            classify_state_attr(name)
        except RestartManifestError:
            unclassified.append(name)
    assert not unclassified, (
        "a real mp=50 step left DomainState attributes that "
        "gpuwm/io/restart.py cannot classify: " + ", ".join(unclassified))
