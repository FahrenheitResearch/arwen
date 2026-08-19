"""``--chain`` past three nests: the 12 km -> 250 m LES ladder.

The program's stated objective is a ladder that starts on a synoptic
root and lands on a large-eddy-permitting inner nest -- 12 -> 3 -> 1 ->
0.5 -> 0.25 km, four nests.  Until this file existed the wizard died on
that ladder with a bare ``IndexError: tuple index out of range``: two
per-depth constants tabulated at the certified FOUR-domain layout's
depth were indexed by nest depth with no bound.

* ``_CHILD_SPAN_FRACTION`` (3 entries) -> ``_dims_for_scale`` raised at
  the FOURTH nest, before anything had a chance to refuse politely.
* ``_DIFF6_FACTORS`` (4 entries) -> ``_domain_tables`` raised one nest
  later, so fixing only the first moved the traceback rather than the
  defect.

Both now clamp to their innermost certified entry, which is the defined
behaviour: a nest deeper than the tabulated ladder inherits the tightest
proportion and the weakest sixth-order damping that were ever certified,
rather than an undefined one.  Nothing here relaxes a refusal -- a nest
that genuinely cannot be hosted inside its parent with the boundary
clearance still raises :class:`DomainFitError` with a message that names
the extent, and the fit loop still prices every candidate.
"""
from __future__ import annotations

import pytest

from gpuwm.cli import main as cli_main
from gpuwm.domain_wizard import (_CHILD_SPAN_FRACTION, _CLEARANCE_ROWS,
                                 _DIFF6_FACTORS, _MAX_SCALE, _MIN_SCALE,
                                 DomainFitError, _child_span_fraction,
                                 _diff6_factor, _dims_for_scale,
                                 _domain_tables, _min_hosting_scale,
                                 experiment_from_text)


# ---------------------------------------------------------------------------
# The two per-depth tables, at and past their tabulated depth.
# ---------------------------------------------------------------------------

def test_per_depth_tables_are_defined_at_every_depth():
    """Tabulated where tabulated; clamped, never undefined, past the end."""

    for depth, value in enumerate(_CHILD_SPAN_FRACTION):
        assert _child_span_fraction(depth) == value
    for depth, value in enumerate(_DIFF6_FACTORS):
        assert _diff6_factor(depth) == value
    # Past the tables: the innermost certified entry, at every depth,
    # forever.  Asserted against the table's own last element so a future
    # extension of either ladder keeps this test honest.
    for depth in range(len(_CHILD_SPAN_FRACTION), 24):
        assert _child_span_fraction(depth) == _CHILD_SPAN_FRACTION[-1]
    for depth in range(len(_DIFF6_FACTORS), 24):
        assert _diff6_factor(depth) == _DIFF6_FACTORS[-1]
    # Negative depth is not a thing a caller can mean; it reads as the
    # root rather than wrapping around to the innermost nest.
    assert _child_span_fraction(-3) == _CHILD_SPAN_FRACTION[0]
    assert _diff6_factor(-3) == _DIFF6_FACTORS[0]


@pytest.mark.parametrize("nests", [4, 5, 6])
def test_dims_for_scale_survives_four_five_and_six_nests(nests):
    """The IndexError regression, pinned at 4, 5 and 6 nests.

    Scale 6.0 is generous enough that clearance is not the binding
    constraint at these depths, so a failure here is the indexing bug and
    not a legitimately too-small parent.
    """
    ratios = tuple([2] * nests)
    dims = _dims_for_scale(6.0, ratios)
    assert len(dims) == nests + 1
    for (pnx, pny), (cnx, cny), ratio in zip(dims, dims[1:], ratios):
        # Child mass dimensions are span * ratio, and the span must clear
        # the parent boundary on both sides -- the invariant the loader
        # re-validates.
        assert cnx % ratio == 0 and cny % ratio == 0
        assert cnx // ratio <= pnx - 2 * _CLEARANCE_ROWS
        assert cny // ratio <= pny - 2 * _CLEARANCE_ROWS
        assert cnx >= 12 and cny >= 12


@pytest.mark.parametrize("nests", [4, 5, 6])
def test_domain_tables_survive_four_five_and_six_nests(nests):
    """The SECOND IndexError: diff_6th_factor past the tabulated depth."""

    ratios = tuple([2] * nests)
    dims = _dims_for_scale(6.0, ratios)
    tables = _domain_tables(dims, ratios)
    assert len(tables) == nests + 1
    factors = [t["diff_6th_factor"] for t in tables]
    # Every domain carries one, it never increases with depth, and past
    # the certified ladder it holds at the innermost certified value.
    assert len(factors) == nests + 1
    assert all(a >= b for a, b in zip(factors, factors[1:]))
    assert factors[len(_DIFF6_FACTORS):] == (
        [_DIFF6_FACTORS[-1]] * (len(factors) - len(_DIFF6_FACTORS)))


def test_the_les_ladder_12km_to_250m_builds():
    """12 -> 3 -> 1 -> 0.5 -> 0.25 km: the program's stated objective."""

    ratios = (4, 3, 2, 2)
    dims = _dims_for_scale(8.0, ratios)
    assert len(dims) == 5
    tables = _domain_tables(dims, ratios, root_dx_m=12000.0)
    assert [t["grid_id"] for t in tables] == [1, 2, 3, 4, 5]
    assert [t["parent_id"] for t in tables] == [0, 1, 2, 3, 4]
    assert [t["parent_grid_ratio"] for t in tables] == [1, 4, 3, 2, 2]


# ---------------------------------------------------------------------------
# Genuinely invalid input still refuses -- with a message.
# ---------------------------------------------------------------------------

def test_a_nest_that_cannot_be_hosted_refuses_with_a_message():
    """Not an IndexError: a named extent and the clearance that bound it.

    Deep enough at a small scale and the parent runs out of interior for
    the boundary clearance.  That is a real limit and it stays a refusal
    -- what it must never be again is a bare traceback.
    """
    with pytest.raises(DomainFitError) as caught:
        _dims_for_scale(0.35, tuple([2] * 6))
    message = str(caught.value)
    assert "cannot host a nest" in message
    assert str(_CLEARANCE_ROWS) in message


# ---------------------------------------------------------------------------
# The minimum layout is a property of the ladder, not a constant.
# ---------------------------------------------------------------------------

def test_min_hosting_scale_leaves_every_preset_exactly_where_it_was():
    """No preset moves: the probe is a floor-finder, not a resize.

    ``_MIN_SCALE`` hosts every preset and every chain of three nests or
    fewer, so the probe must return it unchanged for all of them -- a
    deep-ladder repair that quietly enlarged the shallow ones would be a
    regression against v2.4.1 dressed as a fix.
    """
    from gpuwm.domain_wizard import LADDER_RATIOS

    for ratios in LADDER_RATIOS.values():
        assert _min_hosting_scale(ratios) == _MIN_SCALE
    for ratios in ((4,), (4, 3), (4, 3, 2), (3, 3, 3), (4, 3, 2, 2)):
        assert _min_hosting_scale(ratios) == _MIN_SCALE


@pytest.mark.parametrize("nests", [4, 5, 6])
def test_min_hosting_scale_finds_a_floor_for_deep_ratio_two_chains(nests):
    """Four ratio-2 nests do NOT fit at ``_MIN_SCALE`` -- and used to be
    refused for it, because the fit loop probed that one scale and gave
    up.  The floor it finds must actually host the ladder, and must be
    the SMALLEST such layout to within the scan step.
    """
    ratios = tuple([2] * nests)
    with pytest.raises(DomainFitError):
        _dims_for_scale(_MIN_SCALE, ratios)
    scale = _min_hosting_scale(ratios)
    assert _MIN_SCALE < scale <= _MAX_SCALE
    dims = _dims_for_scale(scale, ratios)          # hosts it
    assert len(dims) == nests + 1
    # ...and it is a floor: a step below it, the ladder does not fit.
    with pytest.raises(DomainFitError):
        _dims_for_scale(scale / 1.05, ratios)


def test_deep_chains_never_raise_indexerror_at_any_depth():
    """The whole point: no depth produces a bare IndexError any more.

    Every depth up to twice the blessed maximum either builds or refuses
    with a :class:`DomainFitError` that carries an explanation.
    """
    from gpuwm.domain_wizard import MAX_CHAIN_DEPTH

    for nests in range(1, 2 * MAX_CHAIN_DEPTH + 1):
        ratios = tuple([2] * nests)
        try:
            dims = _dims_for_scale(4.0, ratios)
        except DomainFitError as error:
            assert str(error)
            continue
        _domain_tables(dims, ratios)


def test_a_ladder_too_deep_to_host_anywhere_refuses_with_a_remedy():
    """The bracket's own limit, said in words instead of a traceback."""

    with pytest.raises(DomainFitError) as caught:
        _min_hosting_scale(tuple([2] * 64))
    message = str(caught.value)
    assert "64 nests" in message
    assert "clearance" in message
    assert "fewer nests" in message and "--chain ratios" in message


# ---------------------------------------------------------------------------
# The front door.  Engine-proven is not shipped: these drive `gpuwm
# domain --chain` exactly as a user types it, at 4, 5 and 6 nests.
# ---------------------------------------------------------------------------

@pytest.fixture
def version_identity_bound(monkeypatch):
    """Stand down ONE unrelated, pre-existing refusal for the door tests.

    On the 2.5.0 line as cut, ``pyproject.toml`` still declares 2.4.1
    while the installed distribution says something else, so
    :func:`gpuwm.provenance_gate.version_identity_refusal` refuses EVERY
    ``gpuwm`` front-door command -- presets, single domains and custom
    chains alike, and the wizard's own pre-existing CLI tests fail the
    same way at ``b83d62b90``.  That defect belongs to the version lane;
    it is not this lane's, and it is not about nest depth.

    So this fixture silences exactly that one check and nothing else --
    the fit loop, the experiment loader, ``render_config`` and ``gpuwm
    check`` all still run for real underneath.  It is deliberately
    self-retiring: once the version lane binds the two numbers, the
    early return fires and the door tests exercise a completely
    unpatched CLI.
    """
    import gpuwm.provenance_gate as gate

    if gate.version_identity_refusal() is None:
        return                                  # already bound; no patch
    monkeypatch.setattr(gate, "version_identity_refusal",
                        lambda prov=None: None)


@pytest.mark.parametrize("nests", [4, 5, 6])
def test_the_chain_door_emits_at_four_five_and_six_nests(
        tmp_path, capsys, nests, version_identity_bound):
    """``--chain`` with four or more nests used to be an ``IndexError``.

    Ratio-2 steps on purpose: that is the chain shape that failed both
    ways -- the unbounded per-depth lookup first, and then the
    single-scale minimum-layout probe.
    """
    out = tmp_path / f"chain{nests}.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "32gb",
                   "--root-dx", "12", "--chain", ",".join(["2"] * nests),
                   "--source", "gfs", "--cycle", "2026-07-29T18",
                   "--hours", "3", "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "gpuwm check: PASS (rc 0)" in printed

    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    dxs = [float(exp.dx_exact(d.grid_id)) for d in exp.domains]
    assert len(dxs) == nests + 1
    assert dxs[0] == 12000.0
    assert dxs[-1] == pytest.approx(12000.0 / (2 ** nests), rel=1e-12)


def test_the_les_ladder_door_reaches_250_m(tmp_path, capsys,
                                           version_identity_bound):
    """12 -> 3 -> 1 -> 0.5 -> 0.25 km through the door a user types.

    This is the program's stated objective ladder, and before this repair
    the command below died with ``IndexError: tuple index out of range``.
    """
    out = tmp_path / "les.toml"
    rc = cli_main(["domain", "--point=35.3,-97.5", "--card", "32gb",
                   "--root-dx", "12", "--chain", "4,3,2,2",
                   "--source", "gfs", "--cycle", "2026-07-29T18",
                   "--hours", "3", "--out", str(out)])
    printed = capsys.readouterr().out
    assert rc == 0, printed
    assert "gpuwm check: PASS (rc 0)" in printed

    exp = experiment_from_text(out.read_text(encoding="utf-8"),
                               source=str(out))
    dxs = [float(exp.dx_exact(d.grid_id)) for d in exp.domains]
    assert dxs == [12000.0, 3000.0, 1000.0, 500.0, 250.0]
    # diff_6th_factor is defined on every domain, including the two past
    # the certified table, and never increases inward.
    factors = [float(d.run.diff_6th_factor) for d in exp.domains]
    assert len(factors) == 5
    assert all(a >= b for a, b in zip(factors, factors[1:]))
