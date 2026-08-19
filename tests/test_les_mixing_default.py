"""The sub-kilometre closure this product RECOMMENDS must be a stable one.

The anisotropic-K criterion ``mix_upper_bound*(dz_max/dx)^2`` has an
explicit limit of 0.25, and this project's own 250 m LES child reads
0.702 against it -- 2.8x, in the tier where an explicit Laplacian
AMPLIFIES a 2-grid-interval mode in ``w`` rather than damping it.  It is
not a theoretical bound: it came out of a real tornado-run crash traced
to WRF-inherited anisotropic-K instability.

Until this file existed the product handled that with two advisories and
an opt-in remedy, which under "fixed means default" is a workaround
rather than a fix.  The specific hole was that the ONLY place the
product tells anyone how to configure sub-kilometre turbulence --
:func:`gpuwm.domain_wizard.gray_zone_advisory` -- recommended ``km_opt =
3`` / ``km_opt = 2`` with ``bl_pbl_physics = 0`` and said NOTHING about
the mixing length.  ``mix_isotropic`` defaults to 0 (WRF's Registry
value), so a user who followed the product's own printed recipe at 250 m
landed on exactly the exposed configuration, at 0.702, and learned about
it only if they later read stderr at config load.

THE RULING PINNED HERE (REWRITTEN for Drew's 2026-08-16 auto-switch;
the recipe-only ruling it replaces lives in this file's history).  The
remedy is default-on: a domain whose config leaves ``mix_isotropic``
UNSET (or writes ``"auto"``) and violates the criterion runs
``mix_isotropic = 1`` -- resolved once, at the shared experiment load
(``gpuwm.experiment.resolve_auto_mix_isotropic``), announced by one
loud line, repeated by ``gpuwm check``, and named at both restart
doors.  ``tests/test_les_mixing_autoswitch.py`` pins that machinery.
What this file keeps pinning is the boundary the auto-switch does NOT
cross, because each line of it was nearly crossed once:

* An EXPLICIT ``mix_isotropic = 0`` is honoured, danger zone included,
  with the advisory (now carrying the override state).  Refusing at
  load would make the frozen crash records unloadable, which deletes
  the ability to reproduce a committed crash.
  ``test_it_is_still_an_advisory_and_records_load`` holds that line.
* Only UNSET flips, and never silently: flipping a key the user WROTE
  would mutate a physics selector they chose -- and ``mix_isotropic``
  is inside the RunConfig fingerprint ``gpuwm/io/restart.py`` writes as
  ``configuration_sha256``, so the flip is loud and the fingerprint
  consequence is stated where it happens and again at restart load.
* The ``RunConfig.mix_isotropic`` DATACLASS default stays 0, WRF's
  Registry value: code-constructed configs (the frozen verify cases,
  wrapped legacy RunConfigs) mean what their authors wrote, and the
  auto semantics live in the experiment loader where "unset" is
  knowable.

The recipe change (``gray_zone_advisory`` naming ``mix_isotropic = 1``)
stands, and the cost of the remedy is still stated where the remedy is
offered.

CPU only, no card, no data.
"""
from __future__ import annotations

import pytest

from gpuwm.config import (EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT,
                          EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT,
                          anisotropic_w_mixing_advice,
                          anisotropic_w_mixing_ratio)
from gpuwm.domain_wizard import _SHARED_CERTIFIED, gray_zone_advisory

#: The measured geometry of this project's own 250 m LES child, read off
#: ``configs/frozen/les_nest_250m_km3.toml`` through the real loader:
#: dx 250 m, base-state layers up to 662.4 m deep, the default
#: ``mix_upper_bound``.  This is the configuration the criterion was
#: written for and the one the receipts record at 0.702.
LES_250M = dict(km_opt=3, mix_upper_bound=0.1, dx=250.0, dy=250.0,
                dz_max=662.4320250962883)


# ---------------------------------------------------------------------------
# The number, both ways round.
# ---------------------------------------------------------------------------

def test_the_recipe_without_a_mixing_length_is_the_defect():
    """0.702 vs 0.25, and what the remedy does to it.

    The A/B that justifies the whole change: the same geometry, the same
    closure, one key apart.
    """
    exposed = anisotropic_w_mixing_ratio(mix_isotropic=0, **LES_250M)
    assert exposed == pytest.approx(0.702, abs=5e-4)
    assert exposed > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT
    assert exposed > EXPLICIT_HORIZONTAL_DIFFUSION_GROWTH_LIMIT, (
        "0.702 is in the AMPLIFIES tier, not merely the INVERTS one")

    # The remedy does not lower the ratio -- it takes the domain off the
    # per-axis path altogether, so the criterion has no subject at all.
    remedied = anisotropic_w_mixing_ratio(mix_isotropic=1, **LES_250M)
    assert remedied is None
    ratio, advice = anisotropic_w_mixing_advice(
        where="d03", mix_isotropic=1, **LES_250M)
    assert (ratio, advice) == (None, None)


def test_the_recommended_subkm_closure_names_the_mixing_length():
    """The product's own recipe must be a configuration that holds.

    ``gray_zone_advisory`` is the only place the product tells a user how
    to configure turbulence below a kilometre.  It recommended km_opt 3
    or 2 with the PBL off and said nothing about ``mix_isotropic``, whose
    default is the exposed value -- so following the printed advice at
    250 m produced 0.702.  The recipe now carries the mixing length.
    """
    lines = gray_zone_advisory([3.0, 0.75, 0.25], _SHARED_CERTIFIED)
    assert len(lines) == 1, "still one honest sentence, not a lecture"
    recipe = lines[0]
    # The closure it already recommended, unchanged...
    assert "km_opt = 3" in recipe and "km_opt = 2" in recipe
    assert "bl_pbl_physics = 0" in recipe
    # ...and the mixing length that makes it a stable one.
    assert "mix_isotropic = 1" in recipe
    # Named as part of the recipe, with the criterion it satisfies, so a
    # reader knows it is not decoration.
    assert "mix_upper_bound*(dz_max/dx)^2" in recipe
    assert str(EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT) in recipe


def test_the_gray_zone_headline_still_leads_with_the_finding():
    """The recipe grew; the one-line stdout headline must not."""

    from gpuwm.domain_wizard import gray_zone_headline

    head = gray_zone_headline([3.0, 0.75, 0.25], _SHARED_CERTIFIED)
    assert len(head) == 1
    assert head[0].startswith("GRAY ZONE:")
    assert "mix_isotropic" not in head[0], (
        "the headline is the finding; the remedy lives in the advisory")


# ---------------------------------------------------------------------------
# The remedy's cost, stated where the remedy is offered.
# ---------------------------------------------------------------------------

def test_the_over_limit_advisory_states_the_restart_fingerprint_cost():
    """``mix_isotropic`` is inside ``configuration_sha256``.

    The remedy is not free and the user must not discover that by losing
    a campaign: a checkpoint written under ``mix_isotropic = 0`` cannot
    be resumed under 1.  Offering a remedy without its cost is how a
    stability fix becomes a data-loss incident.
    """
    ratio, advice = anisotropic_w_mixing_advice(
        where="d03", mix_isotropic=0, **LES_250M)
    assert ratio == pytest.approx(0.702, abs=5e-4)
    assert advice is not None
    assert "configuration_sha256" in advice
    assert "restart" in advice.lower()
    # And it says WHEN to take it, which is the actionable half.
    assert "t = 0" in advice


def test_the_remedy_is_stated_as_the_remedy_not_as_one_of_two_options():
    """Two co-equal options is how an opt-in workaround reads.

    Lowering ``mix_upper_bound`` does satisfy the criterion, so it stays
    in the sentence -- but it weakens the subgrid mixing everywhere to
    fix a horizontal-operator problem, and the advisory has to say which
    one is the remedy.
    """
    _, advice = anisotropic_w_mixing_advice(
        where="d03", mix_isotropic=0, **LES_250M)
    assert "The remedy is mix_isotropic = 1" in advice
    # The alternative is still offered, still with its admitted value.
    assert "mix_upper_bound" in advice


# ---------------------------------------------------------------------------
# What must NOT change: this is an advisory, and records must load.
# ---------------------------------------------------------------------------

def test_it_is_still_an_advisory_and_records_load():
    """Refusals are load-bearing -- and so is this one's ABSENCE.

    Converting the criterion into a load-time refusal would make every
    frozen record unloadable, deleting the ability to reproduce a
    committed crash, and would overturn an evidence-backed ruling without
    one.  The advisory reports; it never blocks.
    """
    ratio, advice = anisotropic_w_mixing_advice(
        where="d03", mix_isotropic=0, **LES_250M)
    assert ratio is not None and advice is not None
    assert "ADVISORY, not a refusal" in advice

    # The whole call chain returns rather than raising, at 0.702 and at
    # the 4.23 the 100 m tree recorded when it aborted.
    from gpuwm.config import warn_anisotropic_w_mixing

    for dz_max in (LES_250M["dz_max"], 6.5 * 100.0):
        got = warn_anisotropic_w_mixing(
            where="d04", km_opt=3, mix_isotropic=0, mix_upper_bound=0.1,
            dx=100.0, dy=100.0, dz_max=dz_max)
        assert got is not None and got > EXPLICIT_HORIZONTAL_DIFFUSION_LIMIT


def test_the_runconfig_default_still_matches_wrfs_registry():
    """The dataclass default is where WRF's value is recorded.

    The auto-switch lives in the experiment loader, where "unset" is
    knowable, NOT here: code-constructed configs (frozen verify cases,
    wrapped legacy RunConfigs) mean what their authors wrote.  A
    namelist import that omits the key emits a TOML that omits it, so
    on a criterion-violating grid the RUN takes the auto-selected
    isotropic length -- a deliberate, loudly announced divergence from
    WRF's unstable default, escapable with an explicit
    ``mix_isotropic = 0`` -- while this recorded Registry value keeps
    the importer's fall-through meaning WRF's for every stable grid.
    """
    import dataclasses

    from gpuwm.config import RunConfig

    defaults = {f.name: f.default for f in dataclasses.fields(RunConfig)}
    assert defaults["mix_isotropic"] == 0
    assert defaults["mix_upper_bound"] == 0.1
