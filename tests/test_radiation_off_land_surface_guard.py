"""A land surface with no radiation at all: the guard, and the sky it gets.

Provenance, 2026-08-09, found by execution.

The 1.7.1 nocturnal guard tests ``ra_sw_physics > 0 and ra_lw_physics
== 0``.  A suite with BOTH streams off walks past it -- and
:func:`gpuwm.core.physics.initialize_physics` computes
``radiation_active = bool(ra_lw_physics or ra_sw_physics)``, so in that
case it attaches no radiation adapter at all.  Nothing ever writes
``fields["glw"]``.  Noah (``gpuwm/core/noah.py``), Noah-MP
(``gpuwm/core/noahmp_runtime.py``) and RUC (``gpuwm/core/ruc.py``) read
it on every surface step regardless, so the whole run's downward
longwave was the constructor's seed -- which was ``300.0``, a plausible
clear-sky number that looked like a measurement in every wrfout it
reached, on the same argument line whose ``swdown`` was already ``0.0``.

Two things changed, and this file pins both:

1.  THE NUMBER -- or rather, the absence of one.  There is no seed at
    all: ``initialize_physics``' ``glw`` argument has NO DEFAULT, and a
    longwave-off run whose land surface reads GLW is REFUSED rather than
    filled (``gpuwm.core.physics._resolve_initial_glw``).  That is
    stronger than replacing 300.0 with WRF's zero, which is what this
    lane originally did: WRF's zero is honest about being nothing, but a
    surface budget closing against zero is still not a forecast, and a
    default of any value is a number the caller did not choose.  A run
    that genuinely wants a fixed sky types it -- and says so in the
    config, with :data:`CONSTANT_DOWNWARD_LONGWAVE_ACK`, which is a
    SECOND and separate declaration from this file's own token.
2.  THE GUARD.  A real experiment (it has a ``[projection]``) that runs
    a land-surface model with both radiation streams off refuses to
    load unless it declares
    :data:`RADIATION_OFF_LAND_SURFACE_ACK`.

Both directions, always: the negative controls below are what say the
guard does not fire on the legitimate cases, and they are bound to the
positive case so that a tree where the guard was never wired in fails
them rather than passing them.
"""
from __future__ import annotations


from pathlib import Path

import pytest
import tomllib

from gpuwm.experiment import build_experiment
from gpuwm.physics_compat import (ASYMMETRIC_RADIATION_NOCTURNAL_ACK,
                                  CONSTANT_DOWNWARD_LONGWAVE_ACK,
                                  MORRISON_PROFILE_ID,
                                  RADIATION_OFF_LAND_SURFACE_ACK,
                                  radiation_off_land_surface_refusal)

ROOT = Path(__file__).parents[1]

#: The shipped case that HAS this shape: a four-domain 60-second VRAM and
#: step-cost probe running Noah with ``ra_physics = 0``.  Read from the
#: tree rather than synthesized, because the audit found it in the tree.
_NORAD = ROOT / "configs" / "real74_4dom_mynn_norad.toml"

#: The other one: ``_nocu``, the same probe without Kain-Fritsch.
#:
#: The two-domain GFS hierarchy proof WAS in this set on this lane, and is
#: not any more.  It is one of the three descriptors a new user copies
#: first, and the composed 1.8.8 answer for those three is that they
#: COMPUTE their radiation (ra_lw_physics 4 / ra_sw_physics 4) rather than
#: declare their way out of it -- so its own suite is no longer
#: radiation-off and there is nothing here for it to declare.
#: ``tests/test_shipped_acknowledgement_justifications.py`` pins that from
#: the other side: those three ship with ``acknowledgements == ()``.
_DECLARING = (
    _NORAD,
    ROOT / "configs" / "real74_4dom_mynn_norad_nocu.toml",
)


def _raw(path: Path) -> dict:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    raw.pop("fetch", None)
    raw.pop("case_data", None)
    return raw


def _undeclared(path: Path) -> dict:
    raw = _raw(path)
    raw["experiment"].pop("acknowledgements", None)
    return raw


# ---------------------------------------------------------------------------
# The refusal.
# ---------------------------------------------------------------------------

def test_radiation_off_with_a_land_surface_refuses_at_the_shared_loader():
    """The positive arm, on the shipped file, minus its declaration."""
    with pytest.raises(ValueError) as caught:
        build_experiment(_undeclared(_NORAD), source="<norad>")
    message = str(caught.value)
    assert "radiation switched entirely OFF" in message
    assert "ra_lw_physics 0 and ra_sw_physics 0" in message   # the selectors
    assert "sf_surface_physics 2 = noah" in message           # the consumer
    assert "1, 2, 3, 4" in message                            # which domains
    assert "zero" in message                                  # what GLW is
    # All three remedies, named.
    assert MORRISON_PROFILE_ID in message                     # 1: turn it on
    assert "sf_surface_physics = 0" in message                # 2: no LSM
    assert RADIATION_OFF_LAND_SURFACE_ACK in message          # 3: declare
    # And the mechanism, on the --explain layer, with WRF's own citation.
    assert "surface energy budget" in message
    assert "module_physics_init.F:1168-1170" in message


def test_the_three_shipped_cases_that_declare_it_still_load():
    """NEGATIVE 1: the declaration lifts the refusal, on the real files.

    Contrasted against the same files with the declaration removed, so
    this cannot pass on a tree where the guard does nothing.
    """
    for path in _DECLARING:
        exp = build_experiment(_raw(path), source=str(path))
        assert RADIATION_OFF_LAND_SURFACE_ACK in exp.acknowledgements, path
        with pytest.raises(ValueError, match="radiation switched entirely"):
            build_experiment(_undeclared(path), source=str(path))


def test_radiation_off_without_a_land_surface_is_not_guarded():
    """NEGATIVE 2: nothing reads GLW, so there is nothing to guard.

    ``sf_surface_physics = 0`` is a prescribed skin temperature; it is
    the remedy the refusal names, and it has to actually work.
    """
    raw = _undeclared(_NORAD)
    raw["shared"]["sf_surface_physics"] = 0
    raw["shared"]["bl_pbl_physics"] = 0
    raw["shared"]["sf_sfclay_physics"] = 0
    build_experiment(raw, source="<norad-no-lsm>")


def test_full_radiation_with_a_land_surface_is_not_guarded():
    """NEGATIVE 3: the ordinary case, which is most of the tree."""
    raw = _undeclared(_NORAD)
    raw["shared"]["ra_physics"] = 4
    build_experiment(raw, source="<full-radiation-noah>")


def test_the_asymmetric_pairing_is_not_this_guard_s_business():
    """NEGATIVE 4: (0, 1) belongs to the nocturnal guard, not this one.

    The two guards are deliberately separate: this one asks nothing
    about the clock, and the nocturnal one is the authority on a
    shortwave-only suite.  Asserted on the FUNCTION rather than through
    the loader so the nocturnal guard cannot supply the refusal and make
    this look right for the wrong reason.
    """
    from types import SimpleNamespace

    asymmetric = SimpleNamespace(grid_id=1, ra_physics=0, ra_lw_physics=0,
                                 ra_sw_physics=1, sf_surface_physics=2)
    assert radiation_off_land_surface_refusal([asymmetric]) is None
    # ...and the same domain with the shortwave half off too does trip it.
    both_off = SimpleNamespace(grid_id=1, ra_physics=0, ra_lw_physics=0,
                               ra_sw_physics=0, sf_surface_physics=2)
    assert radiation_off_land_surface_refusal([both_off]) is not None


def test_the_nocturnal_declaration_does_not_lift_this_refusal():
    """A declaration is about ONE thing, and it is not a skeleton key.

    ``configs/gfs_wrf_hierarchy_proof.toml`` had exactly this gap from
    1.7.1 to 2026-08-09: the nocturnal acknowledgement was the only one
    it carried, and the nocturnal guard cannot look at a suite whose
    shortwave is off.  It declares both now.
    """
    raw = _undeclared(_NORAD)
    raw["experiment"]["acknowledgements"] = [
        ASYMMETRIC_RADIATION_NOCTURNAL_ACK]
    with pytest.raises(ValueError, match="radiation switched entirely"):
        build_experiment(raw, source="<wrong-declaration>")


def test_one_offending_domain_in_a_tree_is_enough_and_is_named():
    """Per-domain, not per-experiment, and the refusal says WHICH.

    Asserted on the function rather than through a config because the
    radiation selectors are shared-table-only in the experiment schema
    (``ra_physics`` on a ``[[domain]]`` is refused by name one gate
    earlier).  The guard still walks the resolved per-domain configs, so
    the per-domain shape has to be exercised somewhere -- here.
    """
    from types import SimpleNamespace

    def domain(grid_id, *, lw, sw, surface=2):
        return SimpleNamespace(grid_id=grid_id, ra_physics=0,
                               ra_lw_physics=lw, ra_sw_physics=sw,
                               sf_surface_physics=surface)

    tree = [domain(1, lw=4, sw=4), domain(2, lw=4, sw=4),
            domain(3, lw=4, sw=4), domain(4, lw=0, sw=0)]
    refusal = radiation_off_land_surface_refusal(tree)
    assert refusal is not None
    assert "domain(s) 4 run a land-surface model" in refusal
    # ...and a tree where nobody offends stays silent.
    assert radiation_off_land_surface_refusal(tree[:3]) is None


def test_an_idealized_experiment_is_not_guarded():
    """NEGATIVE 5: the documented exemption, pinned in its own suite.

    PHYSICS.md and the guard's wiring both promise that an idealized
    experiment -- one with no ``[projection]``, so it has no place and
    no clock -- is not guarded, and the wiring honours it by sitting
    inside ``if experiment.projection is not None:`` in
    ``build_experiment``.  Nothing in this file pinned that, and the
    guard itself is deliberately clock-free ("no clock" in its own
    docstring), so hoisting the call OUT of the projection gate reads
    like a tidy-up rather than a behaviour change.  It is a behaviour
    change: it would refuse every idealized case that runs a land
    surface with radiation off, which is a large fraction of the
    idealized tree.

    Exercised through the real loader on the real offending physics --
    the same file as the positive arm, with its projection removed --
    so it fails if the call moves.
    """
    raw = _undeclared(_NORAD)
    raw.pop("projection")
    # map_proj = 1 without a [projection] table is refused one gate
    # earlier (the F1 amendment), so an idealized case says so.
    raw["shared"]["map_proj"] = 0

    experiment = build_experiment(raw, source="<idealized-norad>")

    assert experiment.projection is None
    # The physics really is the guarded shape: this passes because the
    # experiment is idealized, not because the selectors are innocent.
    run = experiment.domains[0].run
    assert (run.ra_lw_physics, run.ra_sw_physics) == (-1, -1)
    assert run.sf_surface_physics == 2
    assert radiation_off_land_surface_refusal(
        [dc.run for dc in experiment.domains]) is not None


# ---------------------------------------------------------------------------
# The omission case: radiation defaults to off, so silence is in scope.
# ---------------------------------------------------------------------------

def test_a_config_that_never_names_radiation_is_refused_and_told_so():
    """The wider half of the compatibility break, through the real loader.

    Radiation defaults to off, so a real config that simply OMITS a
    selector resolves to the same (0, 0) a config spelling two zeros
    does, and is refused on the same physics.  That is the class that
    breaks previously-loading files -- it is what took
    ``tests/test_physics_mode.py`` from 30 passed to 7 failed -- so the
    message may not quote back lines the file does not contain.

    Built by DELETING the shipped file's ``ra_physics = 0`` rather than
    by hand, so the arm is the shipped case minus one line.
    """
    raw = _undeclared(_NORAD)
    assert raw["shared"].pop("ra_physics") == 0     # the only selector it had

    with pytest.raises(ValueError) as caught:
        build_experiment(raw, source="<omits-radiation>")
    message = str(caught.value)

    assert "NO RADIATION SELECTOR SET AT ALL" in message
    assert "names none of ra_physics, ra_lw_physics, ra_sw_physics" in message
    # The remedy an omitting reader needs FIRST is the missing line.
    assert "the radiation line is simply missing" in message
    # It is still told what the omission RESOLVES to -- that is the
    # physics -- but not that it wrote it.
    assert "resolves to ra_lw_physics 0 and ra_sw_physics 0" in message
    assert "with radiation switched entirely OFF" not in message
    # The other two remedies survive the branch.
    assert "sf_surface_physics = 0" in message
    assert RADIATION_OFF_LAND_SURFACE_ACK in message
    # ...and the declaration lifts THIS refusal here exactly as it does
    # elsewhere -- proven by the token being what changes the answer.
    raw["experiment"]["acknowledgements"] = [RADIATION_OFF_LAND_SURFACE_ACK]
    with pytest.raises(ValueError) as still:
        build_experiment(raw, source="<omits-radiation-declared>")
    assert "radiation switched entirely OFF" not in str(still.value)
    assert "NO RADIATION SELECTOR SET AT ALL" not in str(still.value)

    # What is left is the OTHER guard, and that is the 1.8.8 composition
    # rather than a hole in this one.  Both radiation streams off under a
    # land-surface scheme is the one class both guards are about, and
    # they ask different questions: this file's token says "nothing
    # computes my sky", and constant-downward-longwave-v1 says "the
    # number my land surface integrates is one I typed".  Neither implies
    # the other, so a config in the overlap states both -- which is what
    # the two shipped probes in _DECLARING do, and why they load.
    assert CONSTANT_DOWNWARD_LONGWAVE_ACK in str(still.value)
    raw["experiment"]["acknowledgements"] = [RADIATION_OFF_LAND_SURFACE_ACK,
                                             CONSTANT_DOWNWARD_LONGWAVE_ACK]
    build_experiment(raw, source="<omits-radiation-declared>")


def test_a_spelled_zero_is_not_reported_as_an_omission():
    """The other direction, and the reason it needs the RAW table.

    ``configs/real74_4dom_mynn_norad.toml`` selects radiation with the
    legacy aggregate ``ra_physics = 0`` and leaves the split pair at its
    -1 sentinel -- so on the resolved ``RunConfig`` it is indistinguishable
    from a file that wrote nothing.  It DID choose, and telling its author
    they named no selector would send them hunting for a line that is
    right there.  A branch keyed on the RunConfig fields instead of the
    declared set passes the test above and fails this one.
    """
    with pytest.raises(ValueError) as caught:
        build_experiment(_undeclared(_NORAD), source="<spelled-legacy-zero>")
    message = str(caught.value)

    assert "NO RADIATION SELECTOR SET AT ALL" not in message
    assert "with radiation switched entirely OFF" in message
    # ...and it names the key the reader should actually go and edit.
    assert "written here as ra_physics" in message


def test_the_declared_selector_set_is_read_off_the_raw_shared_table():
    """Presence, not value, and all three spellings count as a choice."""
    from gpuwm.physics_compat import declared_radiation_selectors

    assert declared_radiation_selectors({}) == ()
    assert declared_radiation_selectors({"mp_physics": 6}) == ()
    assert declared_radiation_selectors({"ra_physics": 0}) == ("ra_physics",)
    # A file turning radiation ON is just as much a declaration as one
    # turning it off: the guard reads presence, and the value question
    # belongs to radiation_scheme_ids.
    assert declared_radiation_selectors({"ra_physics": 4}) == ("ra_physics",)
    assert declared_radiation_selectors(
        {"ra_lw_physics": 0, "ra_sw_physics": 0}) == (
            "ra_lw_physics", "ra_sw_physics")


def test_an_unknown_provenance_never_claims_the_author_wrote_nothing():
    """``None`` is not ``()``: a caller who cannot say gets the safe text.

    Direct callers of the guard (the tests above, and anything outside
    ``build_experiment``) have no raw table to read.  Defaulting them to
    the omission wording would put "you named no selector" in front of
    readers whose files do.
    """
    from types import SimpleNamespace

    domain = SimpleNamespace(grid_id=1, ra_physics=0, ra_lw_physics=0,
                             ra_sw_physics=0, sf_surface_physics=2)
    unknown = radiation_off_land_surface_refusal([domain])
    assert "NO RADIATION SELECTOR SET AT ALL" not in unknown
    assert "with radiation switched entirely OFF" in unknown
    # An explicit empty set is the one thing that unlocks the other text.
    omitted = radiation_off_land_surface_refusal([domain],
                                                 declared_selectors=())
    assert "NO RADIATION SELECTOR SET AT ALL" in omitted


# ---------------------------------------------------------------------------
# The seed itself: zero, and WRF's zero.
# ---------------------------------------------------------------------------

def test_the_downward_longwave_has_no_default_at_all():
    """The contract that there IS no argument default to inherit.

    Read off the signature rather than by calling ``initialize_physics``,
    which needs a device.  ``tests/test_hrrr_route_downward_longwave.py``
    is where the refusal is measured on real hardware.

    This lane first replaced the plausible 300.0 with WRF's zero.  The
    composed 1.8.8 answer goes further and removes the default entirely,
    so no run inherits a downward longwave from an argument line: the
    number is either computed by a scheme, typed by the caller, or the
    call is refused.  ``DECLARED_CONSTANT_GLW_WM2`` still exists as the
    historical idealised value a caller may TYPE -- a named constant is
    not a default, and that distinction is the whole fix.
    """
    import inspect

    from gpuwm.core import physics as physics_module
    from gpuwm.core.physics import (DECLARED_CONSTANT_GLW_WM2,
                                    initialize_physics)

    parameters = inspect.signature(initialize_physics).parameters
    assert parameters["glw"].default is None
    # The sibling on the same line was always 0.0; the asymmetry between
    # them is what made a missing scheme look like weather.  1.9's carrier
    # contract removes that default too, and for a reason the note here
    # used to have backwards: SWDOWN IS consumed as a surface energy term
    # -- Noah and Noah-MP read it every surface step whether or not a
    # shortwave scheme is attached -- so "the caller said nothing" and
    # "the caller typed zero" had to become different states.  The BUFFER
    # is still allocated as zeros, so every existing trajectory is
    # byte-identical; what changed is that gpuwm.core.radiation_carriers
    # can now tell the two apart and refuse the first before an LSM eats
    # it.
    assert parameters["swdown"].default is None
    # A value a caller may type, and NOT a seed anything hands out.
    assert DECLARED_CONSTANT_GLW_WM2 == 300.0
    assert not hasattr(physics_module, "GLW_NO_LONGWAVE_SCHEME"), (
        "a constant-GLW seed came back; the 1.8.8 composition removed it "
        "so that no run inherits a downward longwave from a default")
