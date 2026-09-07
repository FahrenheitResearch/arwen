"""CPU controls for complete microphysics-to-radiation declarations."""
import pytest


#: Every mp_physics selector ``validate_run_config`` accepts.  Asserted
#: against the validator below so this tuple cannot drift away from the
#: contract it claims to mirror -- and it HAD drifted: 9 (Milbrandt-Yau)
#: and 50 (P3 one-category) were admitted by production while this tuple
#: still read the pre-1.9 eight, so the three cross-checks that iterate it
#: had quietly stopped covering two schemes.  RE-DERIVED by running the
#: production validator over range(0, 60), exactly as
#: ``test_the_accepted_selector_list_this_module_uses_is_the_real_one``
#: does; not hand-extended.
_ACCEPTED_MP_PHYSICS = (0, 1, 6, 8, 9, 10, 16, 18, 28, 50)

#: The admitted selector without a cloud-optics coupling. P3 now has a
#: dedicated single-ice remap and independent cloud-fraction flags; the
#: device controls in test_mpas_column_batch_gpu exercise that full path.
_NO_RTE_RRTMGP_CLOUD_OPTICS = (9,)

#: The selectors the RTE+RRTMGP cross-checks below may actually drive.
#: Derived, so a scheme cannot be dropped from coverage by hand.
_RTE_RRTMGP_COUPLED_MP_PHYSICS = tuple(
    mp for mp in _ACCEPTED_MP_PHYSICS
    if mp not in _NO_RTE_RRTMGP_CLOUD_OPTICS)


def test_every_accepted_selector_is_judged():
    """The partition is exact, and the uncoupled half really refuses.

    Every selector ``validate_run_config`` admits is either coupled to
    RTE+RRTMGP by a row in ``_MP_CLOUD_OPTICS_SCHEME`` or refused against
    the 4/4 pair by name, with the remedy stated.  Nothing may be in
    neither set: that is the state mp=50 shipped in at 1.9 -- admitted,
    uncoupled, unrefused -- and it died at the first radiation call rather
    than at the door.
    """
    from gpuwm.config import RunConfig, validate_run_config
    from gpuwm.core.rrtmgp import _MP_CLOUD_OPTICS_SCHEME

    coupled = set(_MP_CLOUD_OPTICS_SCHEME)
    uncoupled = set(_NO_RTE_RRTMGP_CLOUD_OPTICS)
    assert coupled.isdisjoint(uncoupled)
    assert coupled | uncoupled == set(_ACCEPTED_MP_PHYSICS), (
        "these accepted selectors are neither coupled nor refused: "
        f"{sorted(set(_ACCEPTED_MP_PHYSICS) - coupled - uncoupled)}")

    for mp_physics in sorted(uncoupled):
        cfg = RunConfig(
            nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0, dt=10.0,
            run_seconds=0.0, time_step_sound=4, moist=True,
            mp_physics=mp_physics, ra_lw_physics=4, ra_sw_physics=4)
        with pytest.raises(NotImplementedError) as caught:
            validate_run_config(cfg)
        message = str(caught.value)
        assert "has no cloud-optics coupling" in message, mp_physics
        assert "ra_rrtmg_variant='rrtmg_legacy'" in message, mp_physics
        # And the remedy is admitted, so the refusal is not a dead end.
        validate_run_config(RunConfig(
            nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0, dt=10.0,
            run_seconds=0.0, time_step_sound=4, moist=True,
            mp_physics=mp_physics, ra_lw_physics=0, ra_sw_physics=1))


def test_the_uncoupled_selectors_are_a_decision_in_the_module_that_raises():
    """Keep exclusions explicit and retire them when the coupling exists."""
    from gpuwm.core.rrtmgp import (
        _CLOUD_OPTICS_REMEDY,
        _MP_CLOUD_OPTICS_SCHEME, _NO_CLOUD_OPTICS_COUPLING,
        cloud_optics_scheme, scheme_is_ice_active, scheme_has_snow_species)

    # 1. The partition is stated HERE, in the module that raises.
    coupled = set(_MP_CLOUD_OPTICS_SCHEME)
    recorded = set(_NO_CLOUD_OPTICS_COUPLING)
    assert coupled.isdisjoint(recorded)
    assert coupled | recorded == set(_ACCEPTED_MP_PHYSICS), (
        "these accepted selectors have neither a cloud-optics row nor a "
        "recorded reason for not having one: "
        f"{sorted(set(_ACCEPTED_MP_PHYSICS) - coupled - recorded)}")
    assert (tuple(sorted(_NO_CLOUD_OPTICS_COUPLING))
            == tuple(sorted(_NO_RTE_RRTMGP_CLOUD_OPTICS)))

    # 2. The refusal reads as a decision, and names both live adapters.
    for selector in sorted(_NO_CLOUD_OPTICS_COUPLING):
        with pytest.raises(NotImplementedError) as caught:
            cloud_optics_scheme(selector)
        message = str(caught.value)
        assert f"mp_physics={selector}" in message
        assert "cloud-optics" in message
        assert "deliberately" in message
        assert "add a row" not in message, (
            f"mp_physics={selector} is a recorded exclusion but its "
            "refusal still tells the reader to widen the table")
        assert _CLOUD_OPTICS_REMEDY in message

    # P3's implemented coupling consumes its single ice category without
    # claiming a snow species or asking for an unallocated snow radius.
    assert cloud_optics_scheme(50) == "p3"
    assert scheme_is_ice_active("p3")
    assert not scheme_has_snow_species("p3")
    assert 50 not in recorded

    # 5. The record does not soften the gate.  An unjudged selector still
    #    fails closed, and is told how to make the omission speak.
    with pytest.raises(NotImplementedError) as unjudged:
        cloud_optics_scheme(51)
    assert "add a row" in str(unjudged.value)
    assert "_NO_CLOUD_OPTICS_COUPLING" in str(unjudged.value)


def test_the_module_and_the_registry_builder_refuse_the_same_selectors():
    """Two authorities, one partition.

    ``tools/build_registry.py`` writes a ``refused_when`` constraint into
    the shipped registry for every implemented scheme with no
    cloud-optics row, keyed by its own reason table.  If that table and
    the module's diverge, the registry refuses a pairing the runtime
    couples or -- the direction that costs a forecast -- couples one the
    runtime raises on at the first radiation call.
    """
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from tools.build_registry import _NO_RTE_RRTMGP_CLOUD_OPTICS_REASON

    from gpuwm.core.rrtmgp import _NO_CLOUD_OPTICS_COUPLING

    assert (sorted(_NO_CLOUD_OPTICS_COUPLING)
            == sorted(_NO_RTE_RRTMGP_CLOUD_OPTICS_REASON))


def test_the_accepted_selector_list_this_module_uses_is_the_real_one():
    """``_ACCEPTED_MP_PHYSICS`` must be exactly what RunConfig admits.

    The three cross-checks below iterate this tuple; if it silently drifted
    away from ``gpuwm/config.py`` they would stop covering a selector
    without failing.

    ``admits`` classifies a refusal by WHICH SELECTOR it names, not by one
    sentence's wording.  The generic tail says "mp_physics must be ...",
    but a scheme gpuwm deliberately does not port gets a NAMED refusal
    instead -- WDM5/WDM7 (gpuwm/config.py), and the P3 siblings on their
    own lane -- and those say why rather than reciting the admitted set.
    Matching one wording made this helper re-raise the named refusals, so
    the drift detector this test exists to be became an ERROR at the first
    scheme that got a good refusal message.  It still cannot be silenced: a
    ValueError that does not name ``mp_physics`` at all still propagates,
    and the loop below is what decides admission, never a literal.
    """
    from gpuwm.config import RunConfig, validate_run_config

    refusals = {}

    def admits(mp_physics):
        try:
            validate_run_config(RunConfig(
                nx=4, ny=3, nz=12, dx=2000.0, dy=2000.0, ztop=8000.0,
                dt=10.0, run_seconds=0.0, time_step_sound=4, moist=True,
                mp_physics=mp_physics))
        except ValueError as error:
            message = str(error)
            if not message.startswith("mp_physics"):
                raise
            refusals[mp_physics] = message
            return False
        return True

    admitted = tuple(mp for mp in range(0, 60) if admits(mp))
    assert admitted == _ACCEPTED_MP_PHYSICS
    # The named refusals are the reason the helper cannot key on one
    # sentence; pin that they exist, so a lane cannot "fix" a future
    # mismatch by deleting the explanation.
    assert "WDM5" in refusals[14] and "WDM7" in refusals[26]
    assert all("mp_physics must be" in refusals[mp]
               for mp in (2, 5, 55) if mp in refusals)


def test_thompsonaero_mp28_resolves_to_the_thompson_cloud_optics_coupling():
    """mp_physics=28 must get classic Thompson's radiative coupling.

    WRF v4.6.1 authority, all in the stock tree:

    * ``Registry/Registry.EM_COMMON:3036`` --
      ``package thompsonaero mp_physics==28 - moist:qv,qc,qr,qi,qs,qg;
      scalar:...;state:re_cloud,re_ice,re_snow`` -- the same ``moist``
      inventory and the same three ``re_*`` state fields as line 3024's
      ``package thompson mp_physics==8``.
    * ``phys/module_physics_init.F:1005-1006`` names THOMPSON and
      THOMPSONAERO in ONE disjunction setting
      ``has_reqc = has_reqi = has_reqs = 1`` (:1021-1023); the P3 /
      Jensen-Ishmael ``has_reqs = 0`` override (:1027-1033) does not list
      THOMPSONAERO.
    * ``phys/module_radiation_driver.F``'s ``cal_cldfra1`` branches on
      ``mp_physics`` only for Ferrier (:3926-3937); both Thompson packages
      take the ``F_QI .and. F_QC .and. F_QS`` arm at :3870-3877.  The RRTMG
      wrappers likewise test the selector only for Ferrier/HWRF
      (``module_ra_rrtmg_lw.F:12131-12136``,
      ``module_ra_rrtmg_sw.F:10732-10737``).
    """
    from gpuwm.core.rrtmgp import (
        _MP_CLOUD_OPTICS_SCHEME, cloud_optics_scheme, scheme_is_ice_active)

    assert cloud_optics_scheme(28) == "thompson"
    assert cloud_optics_scheme(28) == cloud_optics_scheme(8)
    assert scheme_is_ice_active(cloud_optics_scheme(28)) is True
    # Every accepted selector is judged; nothing falls through.  A
    # selector either has a cloud-optics row or is refused against
    # RTE+RRTMGP by name -- see test_every_accepted_selector_is_judged.
    assert (tuple(sorted(_MP_CLOUD_OPTICS_SCHEME))
            == tuple(mp for mp in _ACCEPTED_MP_PHYSICS
                     if mp not in _NO_RTE_RRTMGP_CLOUD_OPTICS))


def test_wdm6_mp16_resolves_to_the_wsm6_cloud_optics_coupling():
    """mp_physics=16 must get WSM6's radiative coupling, not Kessler's.

    WRF v4.6.1 authority, all in the stock tree:

    * ``Registry/Registry.EM_COMMON:3031`` --
      ``package wdm6scheme mp_physics==16 - moist:qv,qc,qr,qi,qs,qg;
      scalar:qnn,qnc,qnr;state:re_cloud,re_ice,re_snow`` -- WSM6's
      (:3021) ``moist`` inventory character for character, plus three
      transported numbers and the same three ``re_*`` state fields.
    * ``phys/module_physics_init.F:1013`` names WDM6SCHEME in the same
      ``use_mp_re`` disjunction as WSM6SCHEME (:1010), setting
      ``has_reqc = has_reqi = has_reqs = 1`` (:1021-1023); the P3 /
      Jensen-Ishmael ``has_reqs = 0`` override (:1027-1033) omits it.
    * ``module_radiation_driver.F``'s ``cal_cldfra1`` branches on
      ``mp_physics`` only for Ferrier (:3926-3937); WDM6 takes the
      ``F_QI .and. F_QC .and. F_QS`` arm at :3870-3877.

    The consequence of getting this wrong is not subtle and is the reason
    the table fails closed: an unmapped selector used to fall through to
    Kessler, and mp=28 spent four waves radiating overcast ice as clear
    sky.  What is CHECKED here is that mp=16 lands on the explicit-radius,
    ice-active branch -- WDM6's droplet radius is built from prognostic nc
    (effectRad_wdm6, module_mp_wdm6.F:3203-3213), but that changes the
    values the scheme supplies, not the branch that consumes them.
    """
    from gpuwm.core.rrtmgp import (_MP_CLOUD_OPTICS_SCHEME,
                                   cloud_optics_scheme, scheme_is_ice_active)

    assert cloud_optics_scheme(16) == "wsm6"
    assert cloud_optics_scheme(16) == cloud_optics_scheme(6)
    assert scheme_is_ice_active(cloud_optics_scheme(16)) is True
    assert cloud_optics_scheme(16) != "kessler"
    # Every accepted selector is judged; nothing falls through.  A
    # selector either has a cloud-optics row or is refused against
    # RTE+RRTMGP by name -- see test_every_accepted_selector_is_judged.
    assert (tuple(sorted(_MP_CLOUD_OPTICS_SCHEME))
            == tuple(mp for mp in _ACCEPTED_MP_PHYSICS
                     if mp not in _NO_RTE_RRTMGP_CLOUD_OPTICS))
    # ... and the legacy engine agrees, which is the pairing that used to
    # make ice clouds appear or vanish with the radiation selector.
    from gpuwm.core.rrtmg_legacy import _MP_DECLARES_RADII, legacy_ice_active

    assert legacy_ice_active(16) is True
    assert _MP_DECLARES_RADII[16] is True


def test_cloud_optics_scheme_fails_closed_on_an_unmapped_selector():
    """No silent Kessler default.

    A ``.get(mp, "kessler")`` default is exactly how mp=28 spent four waves
    radiating its ice clouds as clear sky, so an unmapped selector must
    raise instead of inheriting constant radii and an ice-free cloud
    fraction.
    """
    from gpuwm.core.rrtmgp import cloud_optics_scheme

    for unmapped in (2, 5, 51, 95):
        with pytest.raises(NotImplementedError, match="cloud-optics"):
            cloud_optics_scheme(unmapped)


def test_both_radiation_engines_agree_on_which_schemes_carry_ice():
    """The RTE+RRTMGP and legacy-RRTMG adapters must not disagree.

    ``gpuwm/core/rrtmg_legacy.py`` already carried the WRF judgement for
    mp=28 (``_MP_DECLARES_RADII[28] = True``,
    ``_LEGACY_ICE_ACTIVE_MICROPHYSICS`` contains 28) while the default
    engine did not, so an operator's ice clouds appeared or vanished
    depending on which radiation engine they selected.  Both sides derive
    from the same Registry ``moist`` package, so they are pinned equal here
    for every selector gpuwm accepts.
    """
    from gpuwm.core.rrtmgp import (
        cloud_optics_scheme, scheme_is_ice_active, scheme_has_snow_species)
    from gpuwm.core.rrtmg_legacy import (
        _MP_DECLARES_RADII, legacy_cloud_fraction_flags)

    for mp_physics in _RTE_RRTMGP_COUPLED_MP_PHYSICS:
        scheme = cloud_optics_scheme(mp_physics)
        flags = (scheme_is_ice_active(scheme), scheme_has_snow_species(scheme))
        assert flags == legacy_cloud_fraction_flags(mp_physics), (mp_physics, flags)
        # One-way implication: a scheme WRF hands its radii to
        # (module_physics_init.F:1005-1024 has_req*) must not land on
        # Kessler's constant 10 um / 50 um pair here.  Morrison is False in
        # the legacy table by WRF's own omission and is exempt from the
        # converse -- the RRTMGP adapter reconstructs its radii from the
        # number moments instead.
        if _MP_DECLARES_RADII[mp_physics]:
            assert scheme != "kessler", (
                f"mp_physics={mp_physics} declares effective radii to WRF's "
                "radiation but resolves to the constant-radius branch")


