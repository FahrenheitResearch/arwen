"""The [tiles] gate harness declares its downward longwave, or leaves it alone.

The GLW lane made downward longwave a source with a disposition: a run
either has a longwave scheme, or declares a constant, or is refused.  The
tilestream gate arrived from a tree that predates that lane and its
``+Noah LSM`` rungs pair a land-surface scheme with radiation off, which
is the ``"consumed"`` case: no scheme computes GLW and Noah reads it every
surface step.  Five gate cases raised the refusal on the first run from
the integrated tree, which is the seam the integration map predicted and
nothing in the 233/0 had ever seen.

``harness.declared_glw_kwargs`` is the fix and this suite is its contract.
Two properties, and the second is the one that keeps the fix honest:

  * a rung whose GLW is consumed or published DECLARES the constant, so
    the refusal cannot fire and the receipt names an origin;
  * a rung that has a longwave scheme, or no consumer at all, is left
    untouched, so nothing overrides a scheme's own field and no digest in
    the gate moves.

The classification is never restated here or in the harness: both read
``physics_compat.downward_longwave_disposition``, and this suite drives
that same function to build its expectations, so a future change to the
disposition moves the harness and the test together instead of leaving
one of them describing a rule that no longer exists.

CPU only, no card: the whole question is which keyword argument gets
passed, and that is decided before any allocation.
"""

from __future__ import annotations

import pytest

from gpuwm.core.physics import DECLARED_CONSTANT_GLW_WM2
from gpuwm.physics_compat import downward_longwave_disposition
from tilestream import harness


class _Cfg:
    """The three selectors the disposition reads, and nothing else."""

    def __init__(self, ra_lw_physics, ra_sw_physics, sf_surface_physics):
        self.ra_lw_physics = ra_lw_physics
        self.ra_sw_physics = ra_sw_physics
        self.sf_surface_physics = sf_surface_physics


# (ra_lw, ra_sw, sf_surface) triples spanning every branch of the
# disposition, named by the branch each one lands in.
_SELECTORS = [
    (0, 0, 2),    # consumed  -- Noah LSM, radiation entirely off
    (0, 1, 2),    # consumed  -- Noah LSM, shortwave only
    (0, 1, 4),    # consumed  -- Noah-MP, shortwave only
    (0, 1, 0),    # published -- no land surface, shortwave on
    (0, 0, 0),    # unused    -- nothing reads it and nothing writes it
    (4, 4, 2),    # scheme    -- RRTMG-class longwave writes it
    (4, 4, 0),    # scheme    -- longwave on, no land surface
]


@pytest.mark.parametrize("ra_lw, ra_sw, sf_surface", _SELECTORS)
def test_the_harness_declares_exactly_what_has_no_source(
        ra_lw, ra_sw, sf_surface) -> None:
    kind, _consumer = downward_longwave_disposition(
        ra_lw_physics=ra_lw, ra_sw_physics=ra_sw,
        sf_surface_physics=sf_surface)
    got = harness.declared_glw_kwargs(_Cfg(ra_lw, ra_sw, sf_surface))
    if kind in ("consumed", "published"):
        assert got == {"glw": DECLARED_CONSTANT_GLW_WM2}, (
            f"{kind} must be declared or initialize_physics refuses it")
    else:
        assert got == {}, (
            f"{kind} needs no declaration and must not be overridden")


def test_every_branch_of_the_disposition_is_covered() -> None:
    """The parametrisation is only meaningful if it spans the classifier.

    A selector list that happened to miss ``scheme`` would let a harness
    that declares GLW unconditionally pass, and that harness would
    overwrite a longwave scheme's own initial field.
    """
    kinds = {downward_longwave_disposition(
        ra_lw_physics=lw, ra_sw_physics=sw, sf_surface_physics=sf)[0]
        for lw, sw, sf in _SELECTORS}
    assert kinds == {"consumed", "published", "unused", "scheme"}


def test_a_declaration_from_the_caller_is_not_overridden() -> None:
    """The harness merges under caller kwargs, never over them.

    A negative control that wants a deliberately wrong longwave has to be
    able to ask for one, which is only true if the caller's value wins.
    """
    cfg = _Cfg(0, 1, 2)
    merged = {**harness.declared_glw_kwargs(cfg), **{"glw": 410.0}}
    assert merged["glw"] == 410.0
