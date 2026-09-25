"""The positivity policy's field list, against the contract it comes from."""

from __future__ import annotations

import pytest


def _every_scheme():
    """Every microphysics scheme the registry publishes a moment row for.

    Enumerated through the registry and :func:`scheme_moments`, not from
    a list here, so a scheme that joins by adding its row joins this test
    at the same time.  NSSL resolves its own set from its namelist
    switches, so each of its switch combinations is asked separately.
    """
    from gpuwm.core import nssl2_contract
    from gpuwm.da.moments import scheme_moments
    from gpuwm.physics_registry import consumer_rows_by_selector

    for mp, row in sorted(
            consumer_rows_by_selector("microphysics", "moments").items()):
        if not isinstance(row, dict):
            continue                       # microphysics off
        if "resolved_by" in row:
            continue                       # asked below, per switch set
        yield scheme_moments(int(mp))
    for hail in (0, 1):
        for ccn in (0, 1):
            yield scheme_moments(
                nssl2_contract.MP_PHYSICS,
                nssl_hail_on=hail, nssl_ccn_on=ccn)


def _number_spellings():
    """Every spelling a prognostic NUMBER moment has in this tree.

    The moment tables are the source: ``SchemeMoments.number_fields``
    for the paired numbers each scheme advances, and
    :func:`_unpaired_and_volume_spellings` for the ones with no mass to
    be paired with.  ``gpuwm.core.moist`` carries the transport tuples
    for the same numbers, and it is used as a CROSS-CHECK rather than as
    a second source, because two lists that can disagree are how a state
    gets truncated.
    """
    names: set[str] = set()
    for scheme in _every_scheme():
        names.update(scheme.number_fields)
    return names


def _transport_tuples():
    """``gpuwm.core.moist``'s number tuples, where they can be read.

    That module imports cupy at :64, so on a machine with no GPU stack
    it cannot be read at all.  These assertions are about a contract and
    have to hold on a machine with no device, so the cross-check is
    skipped there rather than the contract going unchecked -- and the
    tables above, which are what the analysis actually reads, need no
    device.
    """
    try:
        from gpuwm.core import moist
    except Exception:                                # noqa: BLE001
        return None
    return {name: getattr(moist, name) for name in dir(moist)
            if name.endswith("NUMBER_SPECIES")}


def _unpaired_and_volume_spellings():
    """Prognostic moments that are neither a mass nor a paired number.

    NSSL's predicted-density volumes and predicted CCN, and P3's rime
    mass and rime volume.  Each is a quantity of something, so each is
    non-negative for the same reason a mixing ratio is.
    """
    names: set[str] = set()
    for scheme in _every_scheme():
        names.update(scheme.unpaired)
        names.update(scheme.volume_fields)
    return names - _number_spellings()


def test_the_transport_tuples_name_no_number_the_moment_tables_miss():
    """The cross-check: moist's tuples must be covered, never additional.

    A spelling there and not in the tables would mean the analysis reads
    one list and the transport another, which is the shape of the defect
    this file exists for.  Skipped where the module cannot be imported;
    see :func:`_transport_tuples`.
    """
    tuples = _transport_tuples()
    if tuples is None:
        pytest.skip("gpuwm.core.moist needs a GPU stack to import")
    assert tuples, "no transport tuples found; the cross-check is vacuous"
    reached = _number_spellings() | _unpaired_and_volume_spellings()
    for name, spellings in sorted(tuples.items()):
        extra = sorted(set(spellings) - reached)
        assert not extra, (
            f"gpuwm.core.moist.{name} carries {extra}, which the moment "
            "tables do not; the analysis and the transport disagree about "
            "what this scheme advances")


def test_the_derivation_reaches_the_schemes_it_is_supposed_to_reach():
    """Validate the instrument before trusting what it reports.

    A derivation that quietly resolved to nothing would make every
    assertion below pass while checking nothing at all, which is the
    failure mode a typed list at least could not have.  These are
    spellings the tree is known to carry, named here only so an empty or
    truncated enumeration is a failure rather than a green run.

    Asserted against the UNION of the two derivations, because which
    side of the split a given moment falls on is a description of the
    scheme's structure and not a promise this test makes: NSSL's
    predicted CCN is a number, and it arrives as an unpaired moment
    because it has no mass to be paired with.  Both sides are checked
    against the same policy below, so the split costs no coverage.
    """
    reached = _number_spellings() | _unpaired_and_volume_spellings()
    for known in ("nc", "nr", "ni", "ns", "ng", "nh", "nn",
                  "nwfa", "nifa", "qnr", "qni", "qns", "qng", "qnn",
                  "qir", "qib"):
        assert known in reached, (
            f"{known!r} is a prognostic moment this tree carries and the "
            "derivation did not reach it; the enumeration is broken, not "
            "the policy")
    assert len(list(_every_scheme())) >= 8


def test_every_number_concentration_the_contract_carries_is_constrained():
    """The list fell behind the prognostic contract once; not twice.

    A real mp 28 radar cycle analysed nwfa with the rest of the scheme's
    state, the filter put one cell at -1.005e8 kg^-1, this module had no
    opinion about the field, and the next leg's health check refused the
    state.  Every number concentration the restart contract carries is
    physically non-negative, so the test is the contract itself rather
    than a second typed-out list.

    The candidate spellings are DERIVED from the moment tables rather
    than typed here.  A typed set could only catch a number the author of
    the set had heard of; this one catches a number moment that joins the
    contract under a name nobody here has seen.
    """

    from gpuwm.da.positivity import NON_NEGATIVE_FIELDS
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    carried = _number_spellings().intersection(STATE_SERIALIZED_ATTRS)
    assert carried, "the contract carries no number concentrations at all"
    missing = sorted(carried.difference(NON_NEGATIVE_FIELDS))
    assert not missing, (
        f"{missing} are prognostic number concentrations the analysis can "
        "move and the positivity policy has no opinion about")


def test_every_unpaired_and_volume_moment_the_contract_carries_is_constrained():
    """The same question for the moments that are not numbers.

    ``NON_NEGATIVE_FIELDS`` says it covers "mixing ratios, number
    concentrations, and the two-moment volume variables", and the
    unpaired moments are exactly where that sentence had been trusted
    rather than checked: it reached NSSL's qvolg and qvolh and missed
    P3's rime mass and rime volume, which the scheme itself will not hold
    below zero (``gpuwm/core/p3.py:1310-1312`` sets both to zero the
    moment the rime mass goes negative).
    """

    from gpuwm.da.positivity import NON_NEGATIVE_FIELDS
    from gpuwm.state_serialization_contract import STATE_SERIALIZED_ATTRS

    carried = _unpaired_and_volume_spellings().intersection(
        STATE_SERIALIZED_ATTRS)
    assert carried, "the contract carries no unpaired prognostic moments"
    missing = sorted(carried.difference(NON_NEGATIVE_FIELDS))
    assert not missing, (
        f"{missing} are prognostic moments the analysis can move and the "
        "positivity policy has no opinion about")
