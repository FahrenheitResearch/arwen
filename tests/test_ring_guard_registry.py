"""Ring-guard coverage for EVERY dispatched scheme, derived (CPU only).

NO DEVICE IS IMPORTED HERE, and that is why this module is apart from
``tests/test_mp_spec_zone_ring.py``.  That module opens a card in its
state fixture, so ``tests/conftest.py`` marks the whole file ``gpu`` and
skips it under ``GPUWM_NO_LOCAL_GPU=1`` -- the invocation the lane and
the merge suites run.  These cells are exactly the half of audit R-066 a
parametrised live test could never cover (mp=8 and 28 need the Thompson
tables; 9, 18 and 50 need their own initialisations), so a placement that
skips them on a CPU-only runner and fails to compile on a card-bearing one
retires the coverage it was written to buy.

What the live three-scheme ring pins measure is unchanged and stays in
``tests/test_mp_spec_zone_ring.py``, beside the WRF oracle that motivates
them.
"""
import re

import pytest

from gpuwm.config import RunConfig


# ---------------------------------------------------------------------------
# ring guard coverage, EVERY scheme (CPU)
# ---------------------------------------------------------------------------

def _apply_dispatched_schemes() -> tuple[int, ...]:
    """Every selector ``gpuwm.core.microphysics.apply`` dispatches.

    DERIVED, not typed.  The GPU pins below run three schemes; these CPU
    cells run every one, which is the half of audit R-066 a parametrized
    live test could never cover -- mp=8 and 28 need the Thompson tables,
    and 9, 18 and 50 need their own initialisations, so a scheme's ring
    guard used to be unexamined until somebody built a fixture for it.
    mp=18 was unexamined for exactly that reason and its nine number and
    volume moments were left advancing in a ring WRF's clipped tiles never
    touch.

    A hand-typed tuple here would have reintroduced the shape of the defect
    the item retired: a scheme added to the registry and to
    ``microphysics._dispatch_scheme`` would silently drop out of all three
    cells below.  The registry's implemented microphysics options ARE the
    schemes apply() dispatches -- ``tests/test_registry_reachability.py``
    holds those two to each other -- so the set comes from there, less the
    one selector that is not a scheme: mp=0 is "no microphysics", apply()
    returns without dispatching anything, and it has no ring guard row
    because it advances nothing to guard.
    """
    from gpuwm.physics_registry import implemented_selector_values

    return tuple(value for value
                 in sorted(implemented_selector_values("microphysics"))
                 if value != 0)


_APPLY_DISPATCHED_SCHEMES = _apply_dispatched_schemes()


def test_the_dispatched_set_is_the_registrys_and_excludes_only_mp0():
    """The derivation is the whole derivation, and 0's absence is a row.

    Two halves: every scheme in the set has a ring guard row (the cells
    below would fail loudly if not), and the one selector left out is left
    out because the registry itself says it has nothing to guard -- not
    because a tuple here forgot it.
    """
    from gpuwm.physics_registry import implemented_selector_values
    from gpuwm.core.physics_inventory import ring_guard_row

    implemented = set(implemented_selector_values("microphysics"))
    assert set(_APPLY_DISPATCHED_SCHEMES) == implemented - {0}
    assert 0 in implemented
    with pytest.raises(ValueError, match="no consumers.ring_guard row"):
        ring_guard_row(0)


def test_every_dispatched_scheme_has_a_ring_guard_row():
    """No scheme apply() runs is missing from the guard's own registry."""
    from gpuwm.core.physics_inventory import ring_guard_row

    for mp in _APPLY_DISPATCHED_SCHEMES:
        row = ring_guard_row(mp)
        assert row["state_fields"], mp
        assert "thp" in row["state_fields"], mp
        assert "qv" in row["state_fields"], mp


def test_every_scheme_the_guard_captures_is_a_field_the_guard_knows():
    """Priced set == captured set, for every scheme, at import.

    ``ring_guard_state_fields`` is the union the capture walks -- read here
    from the module that derives it rather than from
    ``gpuwm.core.microphysics``, which imports cupy at module scope and so
    cannot be read at all on a card-free runner -- and ``ring_guard_row``
    is the per-scheme row the preflight registry prices.  A scheme whose
    row names a field the union lacks is captured for nothing; the union is
    derived from the rows, so this is the arithmetic that says the
    derivation is the whole derivation.
    """
    from gpuwm.core.physics_inventory import (ring_guard_row,
                                              ring_guard_state_fields)

    union = set(ring_guard_state_fields())
    for mp in _APPLY_DISPATCHED_SCHEMES:
        missing = set(ring_guard_row(mp)["state_fields"]) - union
        assert not missing, (mp, sorted(missing))
    assert "qndrop" in union and "qvolh" in union, (
        "audit R-066: NSSL-2's registry moment names must be capturable; "
        "the typed tuple this union replaced listed none of them")


@pytest.mark.parametrize("mp", _APPLY_DISPATCHED_SCHEMES)
def test_ring_save_slots_are_priced_for_every_scheme(mp):
    """The preflight registry enumerates one slot per captured array
    and per non-empty ring edge, for every scheme apply() dispatches --
    not only for the three a GPU fixture exists for.  A scheme priced
    short allocates ``mp_ring_save_*`` buffers behind the allocation gate
    on its first guarded call."""
    from gpuwm.core.physics_inventory import (ring_guard_row,
                                              spec_zone_ring_save_slots)

    cfg = RunConfig(nx=9, ny=8, nz=40, dx=1000.0, dy=1000.0, ztop=10000.0,
                    dt=30.0, run_seconds=0.0, moist=True, mp_physics=mp,
                    specified=True)
    slots = spec_zone_ring_save_slots(cfg)
    row = ring_guard_row(mp)
    # This geometry (9 x 8, sz = 1) has all four ring edges non-empty, so
    # every captured array is priced exactly four times.
    edges = 4

    def priced(name):
        # The edge suffix is an index, and the match has to require it: a
        # prefix test counts P3's qv_old slots as qv's and reports eight
        # edges for a four-edge ring.
        return [slots[key] for key in slots
                if re.fullmatch(rf"mp_ring_save_{re.escape(name)}_\d+", key)]

    for name in list(row["state_fields"]) + ["refl_10cm"]:
        shapes = priced(name)
        assert len(shapes) == edges, (mp, name, len(shapes), edges)
        assert all(len(shape) == 3 for shape in shapes), (mp, name)
    for name in row["surface_slots"]:
        shapes = priced(name)
        assert len(shapes) == edges, (mp, name)
        assert all(len(shape) == 2 for shape in shapes), (mp, name)
    # ...and nothing else is priced: an over-priced family is a budget the
    # run cannot meet for buffers it never allocates.
    priced = {key.rsplit("_", 1)[0][len("mp_ring_save_"):] for key in slots}
    assert priced == set(row["state_fields"]) | set(row["surface_slots"]) | {
        "refl_10cm"}, (mp, sorted(priced))
