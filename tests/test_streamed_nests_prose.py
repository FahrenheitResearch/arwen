"""What the streaming-x-nesting seam actually does, pinned to the capability.

``refuse_streamed_nests`` is the last trace of a refusal that no longer
exists: it ignores nesting entirely and refuses only a moving ``[relocation]``
domain configured with ``tiles.store = 'device'``.  Several comment blocks
went on asserting a load-time refusal of ``[tiles] mode = 'on'`` over a
domain tree, and one of them named a second refusal -- an edge with BOTH
ends streamed in ``NestCoupler.force`` -- that no code has ever made.

These tests are written in the house style of the other source-pinning
gates: assert the presence of the SEAM, not merely the absence of a string,
so deleting the capability cannot make them pass.
"""

from __future__ import annotations

import inspect

import pytest

from gpuwm.core import model as model_mod
from gpuwm.core import streaming
from gpuwm.core.nest import NestCoupler


def test_the_dispatch_does_not_claim_a_both_ends_streamed_refusal():
    """The claim goes, and the capability it misdescribed is pinned here.

    ``gpuwm/core/model.py``'s feedback dispatch used to end its note with
    "(cross-scheme edges off a streamed parent in ``_coupled_parent_field``,
    an edge with BOTH ends streamed in ``force``)".  The second half was
    never true and the first half is now a capability, so both halves are
    gone.  The two ``NestWindowSource`` constructions below are what makes
    the second half false: ``force`` builds one over the PARENT's state and
    one over the CHILD's, in the same branch, which is an edge with both
    ends streamed running rather than being refused.
    """
    dispatch = inspect.getsource(model_mod)
    assert "BOTH ends streamed in" not in dispatch, (
        "the feedback dispatch still advertises a refusal of an edge with "
        "both ends streamed; NestCoupler.force implements that edge")

    force = inspect.getsource(NestCoupler.force)
    assert "NestWindowSource(parent.state)" in force, (
        "force no longer windows the PARENT through NestWindowSource; the "
        "both-ends-streamed capability this test pins is gone")
    assert "NestWindowSource(node.state)" in force, (
        "force no longer windows the CHILD through NestWindowSource; the "
        "both-ends-streamed capability this test pins is gone")


def test_the_coupler_no_longer_refuses_a_cross_scheme_edge_off_a_store():
    """The other half of the retired parenthetical, pinned the same way."""
    coupled = inspect.getsource(NestCoupler._coupled_parent_field)
    assert "unimplemented" not in coupled
    assert "_sync_in(" in coupled, (
        "the cross-scheme arm no longer pulls through the store seam; a "
        "streamed parent would map its attach-time air into the edge")


def test_a_streamed_tree_with_no_relocation_is_admitted():
    """The behavioural control, GREEN before and after the prose fix.

    The comments this change corrects claim ``mode = 'on'`` over a tree is
    refused at load.  Build exactly that tree -- two domains, one tree-wide
    ``[tiles] mode = 'on'``, no ``[relocation]`` -- and the seam every front
    door calls admits it.  Without this the prose correction could be waved
    through on a route that really does refuse.
    """
    from test_streaming import _build, _raw_nested_experiment

    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "on"}
    exp = _build(raw)

    assert [dc.grid_id for dc in exp.domains] == [1, 2]
    assert exp.tiles.mode == "on"
    assert streaming.refuse_streamed_nests(exp, source="test") is None


def test_the_seam_still_refuses_the_moving_device_store_it_is_for():
    """The negative control: the one thing that function does still fires.

    A test that only shows an admission would also pass against a function
    gutted to ``return``.  The moving-domain host-store contract is the
    real breakage the seam names, so it has to keep naming it.
    """
    from test_streaming import _build, _raw_nested_experiment

    raw = _raw_nested_experiment()
    raw["tiles"] = {"mode": "on", "store": "device"}
    raw["relocation"] = {
        "enabled": True, "grid_id": 2,
        "move": [{"at_seconds": 120.0, "di_parent_cells": 1,
                  "dj_parent_cells": 0}]}
    with pytest.raises(streaming.StreamingRefused) as refusal:
        _build(raw)
    assert "host store" in str(refusal.value)
