"""Plan review is what asks the legacy-RRTMG readiness gate (CPU only).

NO DEVICE IS IMPORTED HERE, and that is the reason this module exists
apart from ``tests/test_rrtmg_legacy_selection.py``.  That module imports
cupy in a helper, so ``tests/conftest.py`` marks every test in it ``gpu``
and skips them all under ``GPUWM_NO_LOCAL_GPU=1`` -- which is the
invocation this lane and the merge suites run.  A caller pin that skips
is not a pin: with the call deleted from ``validate_run_config`` the
whole module still reported green.  Nothing below touches a card.
"""
from __future__ import annotations

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.physics_compat import (RRTMG_VARIANT_LEGACY,
                                  require_rrtmg_legacy_executable,
                                  require_rrtmg_legacy_ready)


def _cfg(**updates):
    values = dict(nx=2, ny=1, nz=4, dx=1000.0, dy=1000.0,
                  ztop=8000.0, dt=10.0, run_seconds=60.0)
    values.update(updates)
    return RunConfig(**values)


def test_plan_review_is_what_calls_the_readiness_helper(monkeypatch):
    """The defect audit R-030 named was that NOTHING production called it.

    The helper was complete and tested directly, and the refusal of record
    was still the adapter constructor -- reached from ``initialize_physics``,
    which on the mid-run twin rebuild (``gpuwm/core/streaming.py``) runs
    with a forecast already in hand.  So the thing pinned here is the
    CALLER: deleting the call from ``validate_run_config`` restores the
    exact defect, and this cell is what notices.
    """
    import gpuwm.physics_compat as compat

    calls = []
    monkeypatch.setattr(compat, "require_rrtmg_legacy_ready",
                        lambda: calls.append("asked"), raising=True)

    validate_run_config(_cfg(ra_physics=4,
                             ra_rrtmg_variant=RRTMG_VARIANT_LEGACY))
    assert calls == ["asked"], "plan review does not ask the readiness gate"

    # The default 4/4 pair is RTE+RRTMGP and owns no legacy assets, so it
    # must not be asked; nor must a run with no radiation at all.
    calls.clear()
    validate_run_config(_cfg(ra_physics=4))
    validate_run_config(_cfg())
    assert calls == []

    # And the gate refuses the PLAN, not the constructor: a stripped
    # install is refused by validate_run_config itself.
    def _broken():
        raise NotImplementedError("assets missing")

    monkeypatch.setattr(compat, "require_rrtmg_legacy_ready", _broken,
                        raising=True)
    with pytest.raises(NotImplementedError, match="assets missing"):
        validate_run_config(_cfg(ra_physics=4,
                                 ra_rrtmg_variant=RRTMG_VARIANT_LEGACY))


def test_readiness_helper_passes_here_and_receipts_when_broken(
        monkeypatch):
    """require_rrtmg_legacy_ready: silent on a complete installation,
    one complete receipt when anything is missing."""
    import gpuwm.physics_compat as compat

    require_rrtmg_legacy_ready()          # complete install: no raise
    assert require_rrtmg_legacy_executable is require_rrtmg_legacy_ready
    monkeypatch.setattr(
        compat, "_RRTMG_LEGACY_ASSETS",
        compat._RRTMG_LEGACY_ASSETS + ("data/wrf_radiation/NOT_A_FILE",),
        raising=True)
    with pytest.raises(NotImplementedError) as excinfo:
        compat.require_rrtmg_legacy_ready()
    message = str(excinfo.value)
    assert "NOT_A_FILE" in message
    assert "no silent fallback" in message
