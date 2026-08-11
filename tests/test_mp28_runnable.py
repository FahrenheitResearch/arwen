"""WP-11a: the front door for ``mp_physics=28`` (Thompson aerosol-aware).

Before this package landed, eight validated CUDA translation units and a
complete adapter existed and NO USER COULD REACH ANY OF THEM.  Three
independent gates refused the selector:

* ``gpuwm/config.py`` -- ``mp_physics not in (0, 1, 6, 8, 10, 18)`` raised;
* ``gpuwm/core/microphysics.py`` -- ``apply`` refused, and ``_dispatch_scheme``
  fell through to the NSSL arm, so even a widened ``apply`` would have run
  the wrong scheme;
* ``gpuwm/core/refl.py`` -- ``compute_refl_10cm`` raised, so the one
  user-facing product field the fixtures reach 52 dBZ on was unreachable.

Two configuration fields the scope pin talks about did not exist at all:
``RunConfig`` had no ``aer_init_opt`` and no ``wif_input_opt``, so there was
nothing a user could set and, more importantly, nothing that could REFUSE.
A scope pin that cannot be violated is not a scope pin.

Every test below fails on the pre-WP-11a tree.  The mechanism of each
failure is stated in its docstring, because "this test is new" and "this
test caught something" are different claims.

WHAT THIS MODULE DELIBERATELY DOES NOT TEST
-------------------------------------------
The mp=28 numerics.  ``tests/test_thompson_aerosol_adapter.py`` owns the G3
oracle gate against all 19 WRF column fixtures and publishes its residuals;
duplicating a weaker version of that here would only create a second, laxer
authority.  What this module tests is that the scheme is REACHABLE, that it
is reachable only in configurations ArWen can actually honour, and that the
one thing it must never do -- change a field mp=28 does not own -- it does
not do.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

# Only the two symbols that predate WP-11a are imported at module scope.
# Everything this package ADDS is imported inside the test that needs it, so
# that on the pre-WP-11a tree each test fails with its own specific
# ImportError/AttributeError instead of the whole module failing to collect.
# A collection error proves the module is new; a per-test failure proves what
# each test is actually about.
from gpuwm.config import RunConfig, validate_run_config


def _cfg(**overrides) -> RunConfig:
    """A minimal admissible single-domain config, mp=28 by default."""
    values = dict(
        nx=4, ny=3, nz=8, dx=1000.0, dy=1000.0, ztop=10000.0,
        dt=10.0, run_seconds=10.0, moist=True, mp_physics=28,
        sf_sfclay_physics=91, sf_surface_physics=2, bl_pbl_physics=1,
        num_soil_layers=4, cu_physics=0,
    )
    values.update(overrides)
    return RunConfig(**values)


# ---------------------------------------------------------------------------
# 1.  config.py admits the selector.
# ---------------------------------------------------------------------------

def test_validate_run_config_admits_mp_physics_28():
    """The whole package in one line.

    Pre-WP-11a this raised ``ValueError: mp_physics must be 0 (off), 1
    (Kessler), 6 (WSM6), 8 (Thompson), 10 (Morrison two-moment), or 18
    (NSSL two-moment), got 28.``
    """
    cfg = _cfg()
    assert validate_run_config(cfg) is cfg


def test_the_schema_message_names_28_when_it_refuses_a_neighbour():
    """A refusal must list what IS available, or 28 is invisible.

    ``38`` is WRF's Thompson graupel-hail aerosol scheme -- the nearest
    unported neighbour, and exactly the value a user who read the WRF
    Registry might try next.  Pre-WP-11a the message it got did not mention
    28 at all.
    """
    with pytest.raises(ValueError) as caught:
        validate_run_config(_cfg(mp_physics=38))
    message = str(caught.value)
    assert "28 (Thompson aerosol-aware)" in message
    assert "got 38" in message


def test_mp28_still_requires_moisture():
    """Admission must not have widened the moist requirement."""
    with pytest.raises(ValueError, match="requires moist=true"):
        validate_run_config(_cfg(moist=False))


# ---------------------------------------------------------------------------
# 2.  The scope pin exists, defaults to WRF's Registry default, and refuses.
# ---------------------------------------------------------------------------

def test_the_two_aerosol_source_selectors_exist_as_config_fields():
    """Pre-WP-11a ``RunConfig`` had neither field.

    An auditor found the scope pin was unenforceable for the simplest
    possible reason: ``RunConfig(aer_init_opt=1)`` raised ``TypeError:
    __init__() got an unexpected keyword argument``, and every declaration
    that the port was pinned to ``aer_init_opt=0`` was therefore a claim
    about a field that did not exist.
    """
    from dataclasses import fields as dataclass_fields

    from gpuwm.config import MP28_AEROSOL_SOURCE_OPTIONS

    names = {field.name for field in dataclass_fields(RunConfig)}
    assert {"aer_init_opt", "wif_input_opt"} <= names
    default = RunConfig(nx=2, ny=2, nz=4, dx=1.0, dy=1.0, ztop=1.0,
                        dt=1.0, run_seconds=0.0)
    # WRF's own Registry defaults: Registry.EM_COMMON:2656 and
    # registry.new3d_wif:17.  These must be WRF's, not merely "off": the
    # prepared-forecast runner compares physics_compat's
    # _SINGLE_DOMAIN_RUNTIME_SWITCHES rows for EXACT equality, so a nonzero
    # default would silently change every shipped profile.
    assert default.aer_init_opt == 0
    assert default.wif_input_opt == 0
    assert {name: only for name, (only, _c, _w)
            in MP28_AEROSOL_SOURCE_OPTIONS.items()} == {
        "aer_init_opt": 0, "wif_input_opt": 0}


@pytest.mark.parametrize("name,value", [
    ("aer_init_opt", 1),
    ("aer_init_opt", 2),
    ("wif_input_opt", 1),
    ("wif_input_opt", 2),
])
def test_every_other_aerosol_source_value_fails_closed(name, value):
    """Fail CLOSED, and name what is missing.

    ``aer_init_opt`` 1/2 and ``wif_input_opt`` 1/2 are the complete set of
    non-default values WRF defines (Registry.EM_COMMON:2656,
    registry.new3d_wif:17/:80-82).  Each is refused, and the refusal names
    the missing capability rather than the number: WIF metgrid ingest for
    both, plus nbca for ``wif_input_opt=2``'s use_wif_input_bc package.
    """
    with pytest.raises(NotImplementedError) as caught:
        validate_run_config(_cfg(**{name: value}))
    message = str(caught.value)
    assert f"{name}={value!r}" in message
    assert "WIF" in message
    assert "metgrid" in message
    if name == "wif_input_opt":
        assert "qnbca" in message or "nbca" in message


def test_the_pin_holds_under_every_scheme_not_only_28():
    """A value that MEANS something under mp=28 must never ride in silently.

    Under mp=6 these keys are inert in WRF too, so accepting them would be
    defensible -- until the config is used to seed an mp=28 restart or a
    nest whose child is mp=28, at which point an unhonoured setting has
    crossed into a run that would have honoured it.  Refusing everywhere is
    the cheaper invariant.
    """
    from gpuwm.config import validate_aerosol_source_options

    with pytest.raises(NotImplementedError):
        validate_run_config(_cfg(mp_physics=6, wif_input_opt=1))
    with pytest.raises(NotImplementedError):
        validate_aerosol_source_options(_cfg(mp_physics=0, aer_init_opt=2))


def test_the_deliberate_deviation_is_published_and_cites_the_wrf_refusal():
    """The one thing a user must not learn from a code comment.

    WRF's real.exe FATALs ``mp_physics=28`` with ``wif_input_opt=0``
    (dyn_em/module_initialize_real.F:2735-2736).  ArWen runs that exact
    configuration, on thompson_init's synthetic profile.  The physics is
    WRF's; the INITIALIZATION is one WRF's initializer refuses to produce,
    which means an ArWen mp=28 run and a WIF-initialized WRF mp=28 run are
    not directly comparable.  A reader who does not know that can draw a
    false conclusion from a true comparison, so the statement is carried in
    a published constant and reaches the namelist importer's printed
    receipt -- it is not buried in a comment.
    """
    from gpuwm.config import MP28_AEROSOL_SOURCE_DEVIATION

    text = MP28_AEROSOL_SOURCE_DEVIATION
    assert "dyn_em/module_initialize_real.F:2735-2736" in text
    assert "wif_input_opt=0 but mp_physics=28" in text
    assert "thompson_init" in text
    assert "phys/module_mp_thompson.F:482-559" in text
    # It must say the comparison is NOT equivalent, not merely that the
    # setup differs.
    assert "not be reported as one" in text.replace("\n", " ")
    # And every refusal must carry it, so the user who trips the pin is the
    # user most likely to need it.
    with pytest.raises(NotImplementedError) as caught:
        validate_run_config(_cfg(wif_input_opt=1))
    assert MP28_AEROSOL_SOURCE_DEVIATION in str(caught.value)


# ---------------------------------------------------------------------------
# 3.  The driver dispatches 28, and to the right adapter.
# ---------------------------------------------------------------------------

class _StubState:
    """Just enough state for ``microphysics.apply``'s guards."""

    def __init__(self):
        self.qv = object()
        self.p = np.zeros((4, 3, 2), dtype=np.float32)


def test_apply_dispatches_28_to_the_aerosol_adapter(monkeypatch):
    """Pre-WP-11a this raised ``unknown mp_physics=28``.

    And widening only ``apply``'s accepted set would have been WORSE than
    the raise: ``_dispatch_scheme``'s trailing ``else`` is the NSSL arm, so
    an mp=28 request would have demanded an NSSL production binding and, on
    a state that had one, run NSSL two-moment under an mp=28 label.  The
    assertion is therefore about WHICH adapter, not merely that something
    ran.
    """
    from gpuwm.core import microphysics
    import gpuwm.core.microphysics_aerosol as aerosol

    calls = []

    def _record(state, cfg, dt, *, refl_10cm_due=False):
        calls.append((state, cfg, dt, refl_10cm_due))
        return "aerosol-diagnostics"

    monkeypatch.setattr(aerosol, "_apply_thompson_aerosol", _record)
    state, cfg = _StubState(), _cfg()
    assert microphysics.apply(state, cfg, 12.5, refl_10cm_due=True) == \
        "aerosol-diagnostics"
    assert calls == [(state, cfg, 12.5, True)]


def test_apply_still_refuses_an_unported_scheme_and_lists_28():
    from gpuwm.core import microphysics

    with pytest.raises(ValueError) as caught:
        microphysics.apply(_StubState(), _cfg(mp_physics=38), 1.0)
    assert "28 = Thompson aerosol-aware" in str(caught.value)


def test_mp28_dispatch_does_not_disturb_the_frozen_mp8_route(monkeypatch):
    """The mp=8 arm must still reach ``_apply_thompson``, unchanged.

    ``gpuwm/core/thompson.py`` and ``kernels/thompson.cu`` are byte-frozen;
    the guarantee this test adds is that the DISPATCH still selects them.
    A refactor that folded 8 and 28 into one arm would be invisible to
    tests/test_mp8_frozen.py's source hashes.
    """
    from gpuwm.core import microphysics
    import gpuwm.core.microphysics_aerosol as aerosol

    seen = []
    monkeypatch.setattr(microphysics, "_apply_thompson",
                        lambda *a, **k: seen.append("classic"))
    monkeypatch.setattr(aerosol, "_apply_thompson_aerosol",
                        lambda *a, **k: seen.append("aerosol"))
    microphysics.apply(_StubState(), _cfg(mp_physics=8), 1.0)
    microphysics.apply(_StubState(), _cfg(mp_physics=28), 1.0)
    assert seen == ["classic", "aerosol"]


# ---------------------------------------------------------------------------
# 3b.  The one-time scheme init hook (WRF's microphysics_init).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("mp", [0, 1, 6, 8, 10, 18])
def test_microphysics_init_is_an_empty_receipt_for_every_other_scheme(mp):
    """Unconditionally callable, so the init path needs no scheme branch.

    A receipt of ``{}`` is the answer for every scheme gpuwm shipped before
    mp=28: none of them has a domain-construction step.  The state object
    here would raise ``AttributeError`` on ANY field access, which is how
    "does nothing" is asserted rather than assumed.
    """
    from gpuwm.core import microphysics

    class _Untouchable:
        def __getattr__(self, name):
            raise AssertionError(
                f"microphysics_init touched state.{name} for mp_physics={mp}")

    assert microphysics.microphysics_init(
        _Untouchable(), _cfg(mp_physics=mp, moist=bool(mp))) == {}


@requires_gpu
def test_microphysics_init_fills_wrfs_synthetic_ccn_profile_for_mp28():
    """The hook exists because nothing in production calls the fill.

    ``thompson_aerosol_init_fill`` landed with WP-09 and had NO caller
    anywhere outside its own test: an mp=28 forecast therefore integrated
    with nwfa/nifa at the cold-start zero, which is not an error anywhere --
    the terminal apply's clamps (module_mp_thompson.F:3972-4021) hold the
    aerosol at its floors and the run stays finite and silently inert.
    This is the named hook a domain-construction path can call; the caller
    itself is gpuwm/core/physics.py::initialize_physics, which WP-11a does
    not own (integration request).

    The assertion is WRF's own arithmetic, not "the field became nonzero":
    ``nwfa2d(i,j) = nwfa(i,1,j) * 0.000196 * (50./z1)`` with
    ``z1 = hgt(i,2,j) - hgt(i,1,j)`` (module_mp_thompson.F:509-510), and
    the profile floors at ``naCCN1 = 50.0E6`` / ``naIN1 = 0.5E6`` (:94-97).
    """
    import cupy as cp

    from gpuwm.core import microphysics

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CUDA device")
    _tables_or_skip()

    cfg = validate_run_config(_cfg(nz=12))
    state = _mp28_domain_state(cp, cfg)
    state.nwfa[...] = 0.0
    state.nifa[...] = 0.0
    state.nwfa2d[...] = 0.0

    receipt = microphysics.microphysics_init(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert receipt == {"thompson_aerosol_profile": {"ccn": True, "in": True}}

    nwfa = cp.asnumpy(state.nwfa)
    nifa = cp.asnumpy(state.nifa)
    nwfa2d = cp.asnumpy(state.nwfa2d)
    # WRF's floors, and the boundary-layer-following decay with height.
    assert nwfa.min() >= 50.0e6 * (1.0 - 1.0e-6)
    assert nifa.min() >= 0.5e6 * (1.0 - 1.0e-6)
    assert nwfa[0].min() > nwfa[-1].max()
    assert nifa[0].min() > nifa[-1].max()
    # nwfa2d from :509-510, on the same z8w the fill used.
    z8w = cp.asnumpy((state.phb.reshape(-1, 1, 1) if state.phb.ndim == 1
                      else state.phb) + state.php) / 9.81
    z1 = z8w[1] - z8w[0]
    np.testing.assert_allclose(
        nwfa2d, nwfa[0] * 0.000196 * (50.0 / z1), rtol=1.0e-5)

    # Idempotent: the second call finds aerosol present and declines both
    # fills, bitwise.  A per-step call would overwrite an advected,
    # activated and scavenged field with the synthetic profile.
    before = (nwfa.copy(), nifa.copy(), nwfa2d.copy())
    again = microphysics.microphysics_init(state, cfg)
    cp.cuda.Stream.null.synchronize()
    assert again == {"thompson_aerosol_profile": {"ccn": False, "in": False}}
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa), before[0])
    np.testing.assert_array_equal(cp.asnumpy(state.nifa), before[1])
    np.testing.assert_array_equal(cp.asnumpy(state.nwfa2d), before[2])


@requires_gpu
def test_the_unfilled_aerosol_profile_is_physics_visible_not_cosmetic():
    """Why the missing caller is a defect and not a tidiness point.

    Two identical mp=28 cold-start columns -- no droplets, no aerosol --
    one initialized through ``microphysics_init`` and one left at the
    allocation zero.  ONE step apart they must differ materially, or the
    unwired init would be a cosmetic gap and the integration request noise.

    MEASURED on this tree, one 10 s step, nz=12:

      nwfa max  1.110e+07 unfilled  vs  9.699e+07 filled   (8.74x)
      nifa max  5.000e+03 unfilled  vs  9.776e+05 filled   (195x)
      effc      differs by up to 27.4% of the unfilled value

    The unfilled column is not "zero aerosol": WRF's terminal clamps
    (module_mp_thompson.F:3972-4021) lift nwfa to its 11.1E6 floor and nifa
    to naIN1*0.01 = 5.0E3 on the first step, so the run has aerosol -- the
    WRONG aerosol, at a floor, everywhere, with no vertical structure and
    no NaN, no negative and no health trip to say so.  That is exactly the
    failure mode this hook exists to prevent, and a 27% error in the
    radiation-facing cloud effective radius after ONE step is what it costs.
    """
    import cupy as cp

    from gpuwm.core import microphysics

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CUDA device")
    _tables_or_skip()

    cfg = validate_run_config(_cfg(nx=4, ny=3, nz=12))
    results = {}
    for filled in (False, True):
        state = _mp28_domain_state(cp, cfg)
        # A true cold start: no aerosol AND no droplets.  Leaving the
        # helper's nc = 1e8 in place would let cloud-droplet evaporation
        # hand the aerosol-free column ~1e8 CCN back within the same step
        # (WRF's droplet-evaporation nwfaten source), which is real physics
        # and would mask the initialization difference this test is about.
        state.nwfa[...] = 0.0
        state.nifa[...] = 0.0
        state.nwfa2d[...] = 0.0
        state.nc[...] = 0.0
        if filled:
            microphysics.microphysics_init(state, cfg)
        microphysics.apply(state, cfg, cfg.dt)
        cp.cuda.Stream.null.synchronize()
        results[filled] = {
            name: cp.asnumpy(getattr(state, name)).astype(np.float64)
            for name in ("qc", "nc", "qr", "effc", "nwfa", "nifa")
        }
    unfilled, filled = results[False], results[True]
    # The aerosol state itself: the unfilled column sits on WRF's floors.
    assert unfilled["nwfa"].max() == pytest.approx(11.1e6, rel=1e-6)
    assert unfilled["nifa"].max() == pytest.approx(5.0e3, rel=1e-6)
    assert filled["nwfa"].max() > 5.0 * unfilled["nwfa"].max()
    assert filled["nifa"].max() > 50.0 * unfilled["nifa"].max()
    # ... and it propagates into the cloud field within one step, including
    # the effective radius radiation consumes.
    assert not np.array_equal(filled["qc"], unfilled["qc"])
    assert not np.array_equal(filled["nc"], unfilled["nc"])
    effc_relative = np.abs(filled["effc"] - unfilled["effc"]) / np.maximum(
        unfilled["effc"], 1.0e-30)
    assert effc_relative.max() > 0.10, effc_relative.max()


# ---------------------------------------------------------------------------
# 4.  The specified-zone ring guard covers mp=28's fields.
# ---------------------------------------------------------------------------

def test_ring_save_slots_cover_the_exact_mp28_field_set():
    """WRF's clipped tiles never touch the ring, so the guard must restore
    every field mp=28's adapter writes -- and only those.

    Pre-WP-11a ``spec_zone_ring_save_slots`` had no mp=28 arm at all, so an
    mp=28 specified/nested domain would have captured only thp/qv/qc/qr and
    silently kept ring qi/qs/qg/nc/nr/ni/nwfa/nifa/eff* the microphysics had
    mutated.  ``nwfa2d``/``nifa2d`` must NOT appear: they are OPTIONAL,
    INTENT(IN) on mp_gt_driver (module_mp_thompson.F:1098), read at :1247
    and :1320-1321 and written nowhere, so there is nothing to restore.
    """
    from gpuwm.core.microphysics import spec_zone_ring_save_slots

    cfg = _cfg(specified=True, spec_zone=1)
    slots = spec_zone_ring_save_slots(cfg)
    prefixes = {name.rsplit("_", 1)[0][len("mp_ring_save_"):]
                for name in slots}
    assert prefixes == {
        "thp", "qv", "qc", "qr", "qi", "qs", "qg",
        "nr", "ni", "nc", "nwfa", "nifa",
        "effc", "effi", "effs",
        "refl_10cm",
        "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
        "mp_graupelnc", "mp_graupelncv", "mp_sr",
    }
    assert "nwfa2d" not in prefixes and "nifa2d" not in prefixes
    # effr belongs to Morrison only; adding it for mp=28 would budget a slot
    # for a field the aerosol adapter never writes.
    assert "effr" not in prefixes


def test_ring_state_fields_gained_the_two_aerosol_scalars_only():
    from gpuwm.core.microphysics import _RING_STATE_FIELDS

    assert "nwfa" in _RING_STATE_FIELDS and "nifa" in _RING_STATE_FIELDS
    assert "nwfa2d" not in _RING_STATE_FIELDS
    assert "nifa2d" not in _RING_STATE_FIELDS


def test_mp8_ring_slot_inventory_is_byte_unchanged():
    """The mp=28 arm must not have moved mp=8's budget.

    ``spec_zone_ring_save_slots`` feeds ``preflight.scratch_slot_registry``
    and therefore the VRAM estimate and the allocation gate.  An accidental
    widening of one of the shared ``in (6, 8, 10)`` guards would change
    every mp=8 run's arena.
    """
    from gpuwm.core.microphysics import spec_zone_ring_save_slots

    mp8 = spec_zone_ring_save_slots(_cfg(mp_physics=8, specified=True,
                                         spec_zone=1))
    prefixes = {name.rsplit("_", 1)[0][len("mp_ring_save_"):] for name in mp8}
    assert prefixes == {
        "thp", "qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni",
        "effc", "effi", "effs", "refl_10cm",
        "mp_rainnc", "mp_rainncv", "mp_snownc", "mp_snowncv",
        "mp_graupelnc", "mp_graupelncv", "mp_sr",
    }
    assert "nwfa" not in prefixes and "nifa" not in prefixes and \
        "nc" not in prefixes


# ---------------------------------------------------------------------------
# 5.  Reflectivity.
# ---------------------------------------------------------------------------

def test_compute_refl_10cm_admits_28_and_names_it_when_refusing():
    """Pre-WP-11a: ``do_radar_ref needs an active microphysics scheme
    (mp_physics 1, 6, 8, or 10), got 28``.

    REFL_10CM is the one user-facing product field of this port, present at
    up to 52 dBZ in the oracle fixtures, and it was unreachable.
    """
    from gpuwm.core.refl import compute_refl_10cm

    with pytest.raises(ValueError) as caught:
        compute_refl_10cm(_StubState(), _cfg(mp_physics=38))
    # The list widened when WDM6 landed; what this test is about is that 28
    # is IN it and that the refusal names the admitted set rather than a
    # bare "unsupported".
    message = str(caught.value)
    assert "1, 6, 8, 10, 16, or 28" in message
    assert "28" in message


def test_mp28_reflectivity_requires_the_same_graupel_shadow_as_mp8():
    """mp=28 routes through the mp=8 branch, including its preconditions."""
    from gpuwm.core.refl import compute_refl_10cm

    class _S:
        qv = np.zeros((4, 3, 2), dtype=np.float32)
        qr = nr = qs = qg = np.zeros((4, 3, 2), dtype=np.float32)
        p = np.zeros((4, 3, 2), dtype=np.float32)

        @staticmethod
        def scratch(shape, _name):
            return np.zeros(shape, dtype=np.float32)

    temperature = np.zeros((4, 3, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="graupel number shadow"):
        compute_refl_10cm(_S(), _cfg(), temperature=temperature,
                          pressure=_S.p)
    # The same precondition, worded for the selector that asked.
    with pytest.raises(ValueError, match="mp_physics=28"):
        compute_refl_10cm(_S(), _cfg(), temperature=temperature,
                          pressure=_S.p)


@requires_gpu
def test_refl_10cm_is_bit_identical_under_two_very_different_nc_fields():
    """WRF's ``calc_refl10cm`` reads no droplet number.  Prove ArWen's does
    not either.

    ``calc_refl10cm`` (module_mp_thompson.F:5710-6028) is ONE routine shared
    by mp=8 and mp=28.  Its argument list (:5710-5711) has no ``nc1d``, no
    ``nwfa`` and no ``nifa``; cloud water enters only as
    ``rc(k) = MAX(R1, qc1d(k)*rho(k))`` at :5764 and ``rc`` is read nowhere
    else in the routine's 319 lines.  So the prognostic droplet number that
    DEFINES mp=28 contributes exactly zero to REFL_10CM.

    This test exists because that is a fact nothing else in the port
    checks, and because it is exactly the kind of thing a well-meaning
    future change ("mp=28 knows nc, so let it improve the Rayleigh sum")
    would break while looking like an improvement.  Two nc fields four
    orders of magnitude apart, everything else bit-identical, must give
    bit-identical dBZ -- not close, IDENTICAL.
    """
    import cupy as cp

    from gpuwm.core.refl import compute_refl_10cm

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CUDA device")

    nz, ny, nx = 8, 3, 4
    rng = np.random.default_rng(28)

    class _S:
        pass

    def _make(nc_value):
        state = _S()
        state.p = cp.asarray(np.linspace(
            95000.0, 30000.0, nz, dtype=np.float32)[:, None, None]
            * np.ones((1, ny, nx), np.float32))
        state.qv = cp.asarray(np.full((nz, ny, nx), 6.0e-3, np.float32))
        state.qc = cp.asarray(np.full((nz, ny, nx), 5.0e-4, np.float32))
        state.qr = cp.asarray(np.full((nz, ny, nx), 4.0e-4, np.float32))
        state.nr = cp.asarray(np.full((nz, ny, nx), 3.0e5, np.float32))
        state.qs = cp.asarray(np.full((nz, ny, nx), 6.0e-5, np.float32))
        state.qg = cp.asarray(np.full((nz, ny, nx), 8.0e-5, np.float32))
        state.nc = cp.asarray(np.full((nz, ny, nx), nc_value, np.float32))
        state.scratch_slots = {}
        state.scratch = lambda shape, name, _s=state: (
            _s.scratch_slots.setdefault(
                name, cp.zeros(shape, dtype=cp.float32)))
        return state

    temperature = cp.asarray(np.linspace(
        290.0, 220.0, nz, dtype=np.float32)[:, None, None]
        * np.ones((1, ny, nx), np.float32))
    graupel_number = cp.asarray(
        rng.uniform(1.0e2, 1.0e4, (nz, ny, nx)).astype(np.float32))

    results = []
    for nc_value in (2.0e6, 1.9e9):
        state = _make(nc_value)
        refl = compute_refl_10cm(
            state, _cfg(), temperature=temperature, pressure=state.p,
            thompson_graupel_number=graupel_number)
        results.append(cp.asnumpy(refl).copy())

    assert np.array_equal(results[0], results[1]), (
        "REFL_10CM moved when only nc changed; WRF's calc_refl10cm cannot "
        "see nc at all")
    # And the field must be a real radar field, not an all-fill sentinel --
    # otherwise "identical" would be trivially true.
    assert results[0].max() > 0.0


@requires_gpu
def test_mp28_and_mp8_reflectivity_agree_bitwise_on_identical_inputs():
    """One WRF routine, so one ArWen answer.

    If mp=28 ever grew its own reflectivity branch, this is what would
    catch it: the same state evaluated under both selectors must produce
    the same bits, because in WRF it is literally the same call.
    """
    import cupy as cp

    from gpuwm.core.refl import compute_refl_10cm

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CUDA device")

    nz, ny, nx = 6, 2, 3

    class _S:
        pass

    def _make():
        state = _S()
        state.p = cp.asarray(np.linspace(
            95000.0, 40000.0, nz, dtype=np.float32)[:, None, None]
            * np.ones((1, ny, nx), np.float32))
        state.qv = cp.asarray(np.full((nz, ny, nx), 7.0e-3, np.float32))
        state.qr = cp.asarray(np.full((nz, ny, nx), 6.0e-4, np.float32))
        state.nr = cp.asarray(np.full((nz, ny, nx), 4.0e5, np.float32))
        state.qs = cp.asarray(np.full((nz, ny, nx), 9.0e-5, np.float32))
        state.qg = cp.asarray(np.full((nz, ny, nx), 1.2e-4, np.float32))
        state.scratch_slots = {}
        state.scratch = lambda shape, name, _s=state: (
            _s.scratch_slots.setdefault(
                name, cp.zeros(shape, dtype=cp.float32)))
        return state

    temperature = cp.asarray(np.linspace(
        288.0, 230.0, nz, dtype=np.float32)[:, None, None]
        * np.ones((1, ny, nx), np.float32))
    graupel_number = cp.asarray(np.full((nz, ny, nx), 5.0e3, np.float32))

    out = []
    for mp in (8, 28):
        state = _make()
        refl = compute_refl_10cm(
            state, _cfg(mp_physics=mp), temperature=temperature,
            pressure=state.p, thompson_graupel_number=graupel_number)
        out.append(cp.asnumpy(refl).copy())
    assert np.array_equal(out[0], out[1])


# ---------------------------------------------------------------------------
# 6.  wrfout inventory.
# ---------------------------------------------------------------------------

def test_the_four_aerosol_fields_carry_wrf_registry_metadata_verbatim():
    """Transcribed, not improved.

    ``QNIFA2D`` really does say "dust" in WRF for what mp=28 uses as the
    generic ice-friendly aerosol emission (Registry.EM_COMMON:493), and
    ``QNWFA``/``QNIFA``'s descriptions really are cut off mid-word at "con"
    (registry.new3d_wif:88/:90).  A gpuwm wrfout that "fixed" either would
    disagree with a WRF one about a field they both write.
    """
    from gpuwm.io.wrf_output_schema import (
        HISTORY_FIELDS_BY_NETCDF_NAME, OUTPUT_FIELDS_BY_NETCDF_NAME,
        PRECIPITATION_OUTPUT_FIELDS, SCHEME_OUTPUT_FIELDS,
        THOMPSON_AEROSOL_OUTPUT_FIELDS, WRF_FIELD_TYPE_REAL)

    expected = {
        "QNWFA": ("water-friendly aerosol number con", "  kg(-1)"),
        "QNIFA": ("ice-friendly aerosol number con", "  kg(-1)"),
        "QNWFA2D": ("Surface aerosol number conc emission", "kg-1 s-1"),
        "QNIFA2D": ("Surface dust number conc emission", "kg-1 s-1"),
    }
    assert set(THOMPSON_AEROSOL_OUTPUT_FIELDS) == set(expected)
    for name, (description, units) in expected.items():
        field = THOMPSON_AEROSOL_OUTPUT_FIELDS[name]
        assert field.netcdf_name == name
        assert field.description == description
        assert field.units == units
        assert field.dtype == "f4"
        assert field.field_type == WRF_FIELD_TYPE_REAL
        assert field.stagger == ""
        assert field.wrf_history is True
        # And the writer must actually see them: HISTORY_FIELDS_BY_NETCDF_NAME
        # is what gpuwm.io.wrfout consults.
        assert HISTORY_FIELDS_BY_NETCDF_NAME[name] is field
        # ... while the SCHEME-selector inventory must NOT grow.  These are
        # transported model state, not a physics driver's published
        # diagnostics, and tests/test_wrf_output_schema.py counts that map
        # as exactly the driver-published rows.  Adding them there broke
        # test_no_two_gpuwm_fields_claim_one_netcdf_name (95 != 85 + 6).
        assert name not in OUTPUT_FIELDS_BY_NETCDF_NAME


def test_the_two_netcdf_name_maps_stand_in_the_declared_relation():
    """The wider map is exactly the narrower one plus the aerosol rows.

    Stated as a property so neither map can drift: the selector inventory
    keeps the cardinality ``tests/test_wrf_output_schema.py`` pins, the
    writer's map is a strict superset, and the only difference is the four
    mp=28 aerosol carriers.
    """
    from gpuwm.io.wrf_output_schema import (
        HISTORY_FIELDS_BY_NETCDF_NAME, OUTPUT_FIELDS_BY_NETCDF_NAME,
        PRECIPITATION_OUTPUT_FIELDS, SCHEME_OUTPUT_FIELDS,
        THOMPSON_AEROSOL_OUTPUT_FIELDS)

    assert len(OUTPUT_FIELDS_BY_NETCDF_NAME) == (
        len(SCHEME_OUTPUT_FIELDS) + len(PRECIPITATION_OUTPUT_FIELDS))
    assert set(HISTORY_FIELDS_BY_NETCDF_NAME) == (
        set(OUTPUT_FIELDS_BY_NETCDF_NAME)
        | set(THOMPSON_AEROSOL_OUTPUT_FIELDS))
    assert (set(HISTORY_FIELDS_BY_NETCDF_NAME)
            - set(OUTPUT_FIELDS_BY_NETCDF_NAME)) == {
        "QNWFA", "QNIFA", "QNWFA2D", "QNIFA2D"}
    for name, field in OUTPUT_FIELDS_BY_NETCDF_NAME.items():
        assert HISTORY_FIELDS_BY_NETCDF_NAME[name] is field


def test_aerosol_metadata_lives_in_exactly_one_table():
    """Two tables describing one field is how two tables disagree.

    ``gpuwm.io.wrfout._VAR_META`` is the writer's FALLBACK for names with no
    transcribed schema row.  These four have one, so they must not also
    appear there -- the same rule that moved RAINC/RAINNC out of _VAR_META.
    """
    from gpuwm.io.wrfout import _VAR_META

    for name in ("QNWFA", "QNIFA", "QNWFA2D", "QNIFA2D"):
        assert name not in _VAR_META


def test_live_state_history_fields_publishes_the_aerosol_inventory():
    """Both writers read this one function, so both inventories agree."""
    from gpuwm.io.wrfout import _live_state_history_fields

    class _S:
        qi = np.zeros((4, 3, 2), np.float32)
        nc = np.full((4, 3, 2), 1.0e8, np.float32)
        nwfa = np.full((4, 3, 2), 2.0e8, np.float32)
        nifa = np.full((4, 3, 2), 3.0e5, np.float32)
        nwfa2d = np.full((3, 2), 1.0e4, np.float32)
        nifa2d = np.zeros((3, 2), np.float32)

    fields = _live_state_history_fields(_S())
    assert fields["QNWFA"] is _S.nwfa
    assert fields["QNIFA"] is _S.nifa
    assert fields["QNWFA2D"] is _S.nwfa2d
    assert fields["QNIFA2D"] is _S.nifa2d
    # QNCLOUD is not a new row: it has mapped to state.nc since Morrison
    # landed.  Under mp=28 it starts carrying a PROGNOSTIC droplet number,
    # which is a change WRF makes too -- in the values, not the inventory.
    assert fields["QNCLOUD"] is _S.nc


def test_a_scheme_without_aerosol_state_publishes_no_aerosol_fields():
    """Presence-guarded: no other run's wrfout inventory changes.

    mp=10 allocates ``state.nc`` and no aerosol at all, so it must keep
    exactly the inventory it had.
    """
    from gpuwm.io.wrfout import _live_state_history_fields

    class _S:
        qi = np.zeros((4, 3, 2), np.float32)
        nc = np.zeros((4, 3, 2), np.float32)

    fields = _live_state_history_fields(_S())
    assert "QNCLOUD" in fields
    assert not ({"QNWFA", "QNIFA", "QNWFA2D", "QNIFA2D"} & set(fields))


def test_the_2d_emission_fields_route_onto_the_surface_axis():
    """``_dims_for`` routes by SHAPE, so a 2-D field must not be handed a
    ``bottom_top`` axis.  A wrong axis is a schema lie a reader cannot
    detect, which is why ``_create_variable`` cross-checks the stagger."""
    from gpuwm.io.wrfout import WrfoutWriter

    writer = WrfoutWriter.__new__(WrfoutWriter)
    writer.nz, writer.ny, writer.nx = 8, 3, 4
    writer.soil_layers = 4
    assert writer._dims_for("QNWFA2D", (3, 4)) == (
        "Time", "south_north", "west_east")
    assert writer._dims_for("QNWFA", (8, 3, 4)) == (
        "Time", "bottom_top", "south_north", "west_east")


# ---------------------------------------------------------------------------
# 7.  Namelist import.
# ---------------------------------------------------------------------------

def test_mp28_is_a_translation_not_a_substitution():
    """28 -> 28.  Mapping it to 8 would be a silent physics downgrade.

    Classic Thompson pins ``nc`` at ``Nt_c = 100e6`` and runs Cooper
    nucleation; 28 carries prognostic nc/nwfa/nifa, CCN activation and
    iceDeMott.  Because the mapped value equals the WRF value, PHYSICS.md's
    "exactly three ratified substitutions" claim is untouched.
    """
    from gpuwm.namelist_import import _MP_MAP

    assert _MP_MAP[28][0] == 28
    assert _MP_MAP[28][1] == _MP_MAP[28][2] == "Thompson aerosol-aware"
    # The microphysics map's ONLY substitution stays 55 (ISHMAEL -> Morrison).
    substitutions = {wrf for wrf, (mapped, _w, _g) in _MP_MAP.items()
                     if mapped != wrf}
    assert substitutions == {55}


def test_every_wrf_mp28_aerosol_namelist_key_is_answered():
    """``_Section.finish()`` refuses ANY unconsumed key.

    So each of WRF's mp=28 aerosol knobs must be answered explicitly, or an
    otherwise valid namelist becomes unimportable with a message that names
    no reason.  The set is WRF's, not a selection.
    """
    from gpuwm.namelist_import import _MP28_AEROSOL_NAMELIST_KEYS

    assert set(_MP28_AEROSOL_NAMELIST_KEYS["physics"]) == {
        "use_aero_icbc", "use_rap_aero_icbc", "qna_update", "scalar_pblmix",
        "grav_settling", "wif_fire_emit", "wif_fire_inj", "dust_emis"}
    assert set(_MP28_AEROSOL_NAMELIST_KEYS["domains"]) == {
        "wif_input_opt", "num_wif_levels"}
    for keys in _MP28_AEROSOL_NAMELIST_KEYS.values():
        for key, why in keys.items():
            assert len(why) > 40, key


def test_the_refusal_reasons_cite_wrfs_silent_overwrites():
    """ArWen refuses where WRF silently reconfigures.

    ``share/module_check_a_mundo.F`` forces ``grav_settling`` to 0
    (:2459-2474) and ``scalar_pblmix`` to 1 (:2477-2495) under mp=28
    without failing.  A user who wrote the other value would otherwise be
    handed a different model than the one they wrote down, with no signal.
    """
    from gpuwm.namelist_import import _MP28_AEROSOL_NAMELIST_KEYS

    physics = _MP28_AEROSOL_NAMELIST_KEYS["physics"]
    assert "module_check_a_mundo.F:2459-2474" in physics["grav_settling"]
    assert "module_check_a_mundo.F:2477-2495" in physics["scalar_pblmix"]
    domains = _MP28_AEROSOL_NAMELIST_KEYS["domains"]
    assert "module_initialize_real.F:2735-2736" in domains["wif_input_opt"]


# ---------------------------------------------------------------------------
# 8.  Readiness authority, registry resolution and vertical bounds.
# ---------------------------------------------------------------------------

def test_the_readiness_authority_appends_no_blocker_for_28():
    """Admission is an edit in physics_compat, not a widened number check.

    ``pending_wrf_physics_components`` is the single readiness authority;
    config.py accepts every selector value in its schema tables and lets
    this receipt do the refusing.  mp=28 has a dispatch row and a complete
    adapter, so it appends nothing -- while what is genuinely missing (the
    WIF ingest) is refused by name in ``validate_aerosol_source_options``.
    """
    from gpuwm.physics_compat import pending_wrf_physics_components

    blockers = pending_wrf_physics_components(
        mp_physics=28, sf_sfclay_physics=91, bl_pbl_physics=1,
        sf_surface_physics=2, num_soil_layers=4)
    assert blockers == ()


def test_the_registry_resolves_28_to_the_named_option_and_bounds_it():
    """The vertical dispatch must key on the id the registry actually uses.

    A renamed option would silently stop being bounds-checked, because the
    ``elif`` chain simply would not match -- no error, no check.
    """
    from gpuwm.physics_compat import (
        MP28_REGISTRY_OPTION_ID, validate_resolved_physics_vertical_levels)

    receipt = validate_resolved_physics_vertical_levels(_cfg(nz=49))
    assert receipt["resolved_components"]["microphysics"] == \
        MP28_REGISTRY_OPTION_ID
    labels = [check["component"] for check in receipt["checks"]]
    assert "Thompson aerosol-aware microphysics" in labels


def test_mp28_vertical_bounds_are_the_runtime_bounds():
    """The standalone contract constant must equal the kernels' own bound.

    ``gpuwm/physics_vertical_contract.py`` deliberately imports no forecast
    executor, so it restates the number; this is what stops the restatement
    drifting from ``thompson_aerosol_sed``'s
    ``THOMPSON_AA_KMAX_GENERIC``.
    """
    from gpuwm.core.thompson_aerosol import VERTICAL_LEVEL_BOUNDS
    from gpuwm.physics_vertical_contract import (
        THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS)

    assert THOMPSON_AEROSOL_VERTICAL_LEVEL_BOUNDS == VERTICAL_LEVEL_BOUNDS


def test_an_impossible_vertical_grid_is_refused_for_28():
    from gpuwm.physics_compat import (
        PhysicsVerticalPreflightError,
        validate_resolved_physics_vertical_levels)

    with pytest.raises(PhysicsVerticalPreflightError,
                       match="Thompson aerosol-aware microphysics"):
        validate_resolved_physics_vertical_levels(_cfg(nz=400))


def test_the_direct_hierarchy_path_knows_28():
    from gpuwm.hrrr_hierarchy_direct import _SUPPORTED_MICROPHYSICS

    assert 28 in _SUPPORTED_MICROPHYSICS
    assert {1, 6, 8, 10, 18} <= _SUPPORTED_MICROPHYSICS


def test_the_wrf_authority_matrix_cites_the_registry_line_for_28():
    from gpuwm.wrf461_compatibility import MP_OPTIONS, compatibility_cell

    assert 28 in MP_OPTIONS
    cell = compatibility_cell(
        mp_physics=28, bl_pbl_physics=1, sf_sfclay_physics=91,
        sf_surface_physics=2, radiation="off", cu_physics=0)
    anchors = {citation.anchor for citation in cell.citations}
    assert "Registry/Registry.EM_COMMON:3036" in anchors


# ---------------------------------------------------------------------------
# 9.  The end-to-end proof: a real DomainState, through the real driver.
# ---------------------------------------------------------------------------

def _tables_or_skip():
    """Skip only when CCN_ACTIVATE.BIN is genuinely absent.

    The asset ships as of 2026-08-01 (MP28_PORT_SPEC.md blocking unknown 1,
    reversed), so this guard does not fire on a clean checkout.  It stays as
    defence for a tree missing the file or pointed elsewhere by an override,
    and names that one asset rather than swallowing every load failure.
    """
    from gpuwm.core.thompson_aerosol_contract import (
        MissingAerosolTableAsset, resolve_aerosol_table_root,
        resolve_ccn_activation_path)
    try:
        resolve_ccn_activation_path(None, resolve_aerosol_table_root(None))
    except MissingAerosolTableAsset as exc:                # pragma: no cover
        pytest.skip(f"CCN_ACTIVATE.BIN unavailable: {exc}")


def _mp28_domain_state(cp, cfg):
    from gpuwm.core.state import DomainState

    state = DomainState(cfg)
    nz = cfg.nz
    state.p[...] = cp.asarray(
        np.linspace(95000.0, 30000.0, nz, dtype=np.float32)[:, None, None])
    state.thb[...] = cp.asarray(np.full(nz, 300.0, np.float32))
    state.thp[...] = 0.0
    state.phb[...] = cp.asarray(
        np.linspace(0.0, 9.81 * 10000.0, nz + 1, dtype=np.float32))
    state.php[...] = 0.0
    state.qv[...] = 0.006
    state.qc[...] = 2.0e-4
    state.qr[...] = 1.0e-4
    state.nr[...] = 2.0e5
    state.qi[...] = 1.0e-6
    state.ni[...] = 1.0e4
    state.qs[...] = 2.0e-5
    state.qg[...] = 3.0e-5
    state.nc[...] = 1.0e8
    state.nwfa[...] = 1.0e8
    state.nifa[...] = 1.0e5
    state.nwfa2d[...] = 1.0e4
    return state


@requires_gpu
def test_microphysics_apply_runs_mp28_end_to_end_on_a_real_domainstate():
    """The whole package, executed rather than asserted about.

    Not ``_apply_thompson_aerosol`` directly -- ``microphysics.apply``, the
    function ``dycore.step`` calls, on a ``DomainState`` built from a
    ``RunConfig`` that ``validate_run_config`` accepted.  Every layer this
    package touches is on the path, and pre-WP-11a it raised at the first
    of them.
    """
    import cupy as cp

    from gpuwm.core import microphysics

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CUDA device")
    _tables_or_skip()

    cfg = validate_run_config(_cfg(nz=8))
    state = _mp28_domain_state(cp, cfg)
    diagnostics = microphysics.apply(state, cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()

    assert diagnostics is not None
    for name in ("qv", "qc", "qr", "qi", "qs", "qg", "nc", "nr", "ni",
                 "nwfa", "nifa", "effc", "effi", "effs", "thp"):
        value = getattr(state, name)
        assert bool(cp.all(cp.isfinite(value))), name
    for name in ("rainnc", "snownc", "graupelnc"):
        accumulator = getattr(diagnostics, name)
        assert bool(cp.all(cp.isfinite(accumulator))), name
        assert float(accumulator.min()) >= 0.0, name


@requires_gpu
def test_mp28_specified_zone_ring_is_bit_restored_including_the_aerosols():
    """WRF's clipped tiles never dispatch the ring, so ArWen's must not
    either -- for every field, aerosols included.

    Pre-WP-11a there was no mp=28 arm anywhere in the ring guard, so this
    could not even be asked.  The check is BITWISE, on the raw bytes, and
    it covers ``nwfa``/``nifa`` (which the terminal apply and the surface
    emission both write) alongside the hydrometeors.
    """
    import cupy as cp

    from gpuwm.core import microphysics

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CUDA device")
    _tables_or_skip()

    cfg = validate_run_config(
        _cfg(nx=8, ny=7, nz=8, specified=True, spec_zone=1))
    state = _mp28_domain_state(cp, cfg)
    # Make the ring distinguishable from the interior, so a restore that
    # merely re-wrote the interior value everywhere would fail.
    tracked = ("qv", "qc", "qr", "qi", "qs", "qg",
               "nc", "nr", "ni", "nwfa", "nifa", "thp")
    for index, name in enumerate(tracked):
        array = getattr(state, name)
        array[:, 0, :] *= np.float32(1.0 + 0.01 * (index + 1))
        array[:, -1, :] *= np.float32(1.0 - 0.005 * (index + 1))
    before = {name: getattr(state, name).copy() for name in tracked}
    emission = (state.nwfa2d.copy(), state.nifa2d.copy())

    microphysics.apply(state, cfg, cfg.dt)
    cp.cuda.Stream.null.synchronize()

    sz = int(cfg.spec_zone)
    for name in tracked:
        now = getattr(state, name)
        for slc in microphysics.spec_zone_ring_slices(cfg.ny, cfg.nx, sz):
            assert cp.array_equal(now[slc], before[name][slc]), name
        interior = (Ellipsis, slice(sz, cfg.ny - sz), slice(sz, cfg.nx - sz))
        if name in ("qc", "nc", "thp"):
            assert not cp.array_equal(now[interior], before[name][interior]), (
                f"{name} did not change in the interior, so the ring "
                "comparison proves nothing")
    # h_diabatic's ring is PINNED to WRF's exact 0, not restored.
    assert float(cp.abs(state.h_diabatic[:, 0, :]).max()) == 0.0
    # And the two INTENT(IN) emission constants are untouched everywhere.
    assert cp.array_equal(state.nwfa2d, emission[0])
    assert cp.array_equal(state.nifa2d, emission[1])


# ---------------------------------------------------------------------------
# 10.  The namelist importer, exercised rather than inspected.
# ---------------------------------------------------------------------------

_WPS_TEXT = """\
&share
 wrf_core = 'ARW',
 max_dom = 1,
 start_date = '1999-05-03_12:00:00',
 end_date   = '1999-05-03_18:00:00',
 interval_seconds = 21600,
 io_form_geogrid = 2,
/
&geogrid
 parent_id         = 1,
 parent_grid_ratio = 1,
 i_parent_start    = 1,
 j_parent_start    = 1,
 e_we              = 101,
 e_sn              = 81,
 geog_data_res     = 'default',
 dx = 12000,
 dy = 12000,
 map_proj = 'lambert',
 ref_lat   = 39.7,
 ref_lon   = -83.9,
 truelat1  = 30.0,
 truelat2  = 60.0,
 stand_lon = -83.9,
 geog_data_path = '/geog',
/
&ungrib
 out_format = 'WPS',
 prefix = 'ERA5',
/
&metgrid
 fg_name = 'ERA5',
/
"""

from gpuwm.physics_compat import CONSTANT_DOWNWARD_LONGWAVE_ACK

#: The imported namelists below run Dudhia shortwave with longwave OFF
#: under Noah, which is now a DECLARED constant downward longwave
#: (gpuwm.physics_compat.constant_longwave_refusal) -- and a namelist WRF
#: v4.6.1 itself refuses, since its lwrad_select has no lw = 0 case.  A
#: WRF namelist has no spelling for a gpuwm governance declaration, so the
#: importer takes it through ``--ack`` / ``acknowledgements=``, which is
#: what these fixtures do.  The radiation is incidental to every claim in
#: this file; the declaration is here so the microphysics claims can be
#: made against a config that loads.
_CONSTANT_GLW_ACK = (CONSTANT_DOWNWARD_LONGWAVE_ACK,)


_INPUT_TEMPLATE = """\
&time_control
 run_hours = 6,
 start_year = 1999,
 start_month = 05,
 start_day = 03,
 start_hour = 12,
 end_year = 1999,
 end_month = 05,
 end_day = 03,
 end_hour = 18,
 interval_seconds = 21600,
 input_from_file = .true.,
 history_interval = 60,
 restart = .false.,
 restart_interval = 60,
/
&domains
 time_step = 60,
 max_dom = 1,
 e_we = 101,
 e_sn = 81,
 e_vert = 9,
 eta_levels = 1.0, 0.9, 0.8, 0.7, 0.6,
              0.5, 0.4, 0.2, 0.0,
 p_top_requested = 5000,
 dx = 12000.0,
 dy = 12000.0,
 grid_id = 1,
 parent_id = 0,
 i_parent_start = 1,
 j_parent_start = 1,
 parent_grid_ratio = 1,
 parent_time_step_ratio = 1,
 feedback = 0,
 smooth_option = 0,
{domains_extra}/
&physics
 mp_physics = {mp},
 ra_lw_physics = 0,
 ra_sw_physics = 1,
 radt = 12,
 sf_sfclay_physics = 91,
 sf_surface_physics = 2,
 bl_pbl_physics = 1,
 bldt = 0,
 cu_physics = 0,
 cudt = 0,
{physics_extra}/
&dynamics
 hybrid_opt = 2,
 etac = 0.2,
 w_damping = 1,
 epssm = 0.5,
 diff_opt = 2,
 km_opt = 4,
 mix_full_fields = .true.,
 diff_6th_opt = 2,
 diff_6th_factor = 0.12,
 diff_6th_slopeopt = 1,
 base_temp = 2.90D2,
 damp_opt = 3,
 zdamp = 5000.,
 dampcoef = 0.2,
 khdif = 0,
 kvdif = 0,
 non_hydrostatic = .true.,
 use_theta_m = 0,
 moist_adv_opt = 1,
/
&bdy_control
 spec_bdy_width = 5,
 spec_zone = 1,
 relax_zone = 4,
 specified = .true.,
/
"""


def _namelist_pair(tmp_path, *, mp=28, physics_extra="", domains_extra=""):
    wps = tmp_path / "namelist.wps"
    inp = tmp_path / "namelist.input"
    # utf-8 and LF named explicitly.  The default encoding is cp1252
    # on Windows, and the default newline turns every "\n" in
    # these templates into CRLF -- and these two files are the
    # literal input to the namelist reader under test.
    wps.write_text(_WPS_TEXT, encoding="utf-8", newline="\n")
    inp.write_text(_INPUT_TEMPLATE.format(
        mp=mp, physics_extra=physics_extra, domains_extra=domains_extra),
        encoding="utf-8", newline="\n")
    return wps, inp


def test_a_plain_mp28_namelist_imports_and_stays_28(tmp_path):
    """Pre-WP-11a: ``&physics mp_physics = 28: no ratified gpuwm mapping
    (implemented: [0, 1, 6, 8, 10, 18, 55])``.

    And the emitted TOML must say 28.  A translation that quietly wrote 8
    would hand the user classic Thompson -- constant Nt_c, Cooper
    nucleation, no aerosol at all -- under a file they believe selects the
    aerosol-aware scheme.
    """
    from gpuwm.namelist_import import import_namelists

    wps, inp = _namelist_pair(tmp_path)
    toml_text, report = import_namelists(wps, inp, name="mp28-plain",
        acknowledgements=_CONSTANT_GLW_ACK)
    assert "mp_physics = 28" in toml_text
    assert not [s for s in report.substitutions if s.key == "mp_physics"]


def test_the_import_receipt_carries_the_deviation_where_a_user_sees_it(
        tmp_path):
    """Not a comment, not a docstring: the printed report.

    ``SubstitutionReport.format()`` renders ``defaults_applied`` under
    "gpuwm-supplied values without a WRF Registry source", which is the
    importer's never-silent last mile.  This is the channel that tells a
    user their mp=28 run uses thompson_init's synthetic aerosol profile and
    that WRF's own real.exe refuses the same configuration.
    """
    from gpuwm.namelist_import import import_namelists

    wps, inp = _namelist_pair(tmp_path)
    _toml, report = import_namelists(wps, inp, name="mp28-receipt",
        acknowledgements=_CONSTANT_GLW_ACK)
    rendered = report.format()
    assert "thompson_init" in rendered
    assert "dyn_em/module_initialize_real.F:2735-2736" in rendered
    entries = [a for a in report.defaults_applied
               if "aerosol" in a.key]
    assert len(entries) == 1, [a.key for a in report.defaults_applied]

    # And it must appear ONLY for mp=28: a WSM6 import must not grow a
    # paragraph about aerosol it never had.
    wps6, inp6 = _namelist_pair(tmp_path / "six", mp=6) \
        if (tmp_path / "six").mkdir() or True else (None, None)
    _toml6, report6 = import_namelists(wps6, inp6, name="mp6-receipt",
        acknowledgements=_CONSTANT_GLW_ACK)
    assert "thompson_init" not in report6.format()


@pytest.mark.parametrize("section,key,value", [
    ("physics", "use_aero_icbc", ".true."),
    ("physics", "use_rap_aero_icbc", ".true."),
    ("physics", "qna_update", ".true."),
    ("physics", "scalar_pblmix", "1"),
    ("physics", "grav_settling", "1"),
    ("physics", "wif_fire_emit", "1"),
    ("physics", "wif_fire_inj", "1"),
    ("physics", "dust_emis", "1"),
    ("domains", "wif_input_opt", "1"),
    ("domains", "num_wif_levels", "30"),
])
def test_each_mp28_aerosol_key_is_refused_by_name(tmp_path, section, key,
                                                  value):
    """Refuse, with the reason, rather than accept-and-ignore.

    Pre-WP-11a every one of these produced the importer's generic
    ``unmapped key(s)`` message from ``_Section.finish()``, which names the
    key and nothing else -- no WRF citation, no statement of what ArWen is
    missing.  A user could not tell an unported knob from a typo.
    """
    from gpuwm.namelist_import import import_namelists

    extra = {"physics": "", "domains": ""}
    extra[section] = f" {key} = {value},\n"
    wps, inp = _namelist_pair(
        tmp_path, mp=28, physics_extra=extra["physics"],
        domains_extra=extra["domains"])
    with pytest.raises(ValueError) as caught:
        import_namelists(wps, inp, name=f"mp28-{key}",
        acknowledgements=_CONSTANT_GLW_ACK)
    message = str(caught.value)
    assert f"&{section} {key}" in message
    assert "unmapped key" not in message
    # Each refusal names a capability, not just a number.
    assert any(token in message for token in (
        "WIF", "aerosol", "fog settling", "dust", "fire"))


@pytest.mark.parametrize("section,key,value", [
    ("physics", "use_aero_icbc", ".false."),
    ("physics", "grav_settling", "1"),
    ("domains", "wif_input_opt", "0"),
])
def test_mp28_aerosol_keys_drop_as_inert_under_another_scheme(
        tmp_path, section, key, value):
    """Inert in WRF too, so inert here -- with the reason recorded.

    WRF consumes these only inside the ``thompsonaero`` package
    (Registry/Registry.EM_COMMON:3036).  Under WSM6 they change nothing in
    either model, so refusing would make importable WRF namelists
    unimportable for no scientific reason.
    """
    from gpuwm.namelist_import import import_namelists

    extra = {"physics": "", "domains": ""}
    extra[section] = f" {key} = {value},\n"
    wps, inp = _namelist_pair(
        tmp_path, mp=6, physics_extra=extra["physics"],
        domains_extra=extra["domains"])
    _toml, report = import_namelists(wps, inp, name=f"mp6-{key}",
        acknowledgements=_CONSTANT_GLW_ACK)
    dropped = {(d.section, d.key): d.reason for d in report.dropped}
    assert (section, key) in dropped
    assert "thompsonaero" in dropped[(section, key)]


# ---------------------------------------------------------------------------
# 11.  A real wrfout file, reopened and read back.
# ---------------------------------------------------------------------------

def test_a_written_wrfout_carries_the_aerosol_fields_wrf_would_carry(
        tmp_path):
    """Assert on the FILE, not on the table that describes it.

    Every claim above about names, units, descriptions, NetCDF type,
    ``FieldType`` and axes is a claim about what lands on disk.  This writes
    a frame through the real writer, reopens it with netCDF4 and reads the
    attributes back, so a schema row that is right and a writer path that
    never reaches it cannot both pass.
    """
    import netCDF4

    from gpuwm.io.wrfout import WrfoutWriter

    nz, ny, nx = 6, 4, 5
    volume = np.arange(nz * ny * nx, dtype=np.float32).reshape(nz, ny, nx)
    surface = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
    fields = {
        "QNWFA": volume * np.float32(1.0e6),
        "QNIFA": volume * np.float32(1.0e3),
        "QNWFA2D": surface * np.float32(1.0e4),
        "QNIFA2D": np.zeros((ny, nx), np.float32),
    }
    path = tmp_path / "wrfout_d01_mp28"
    writer = WrfoutWriter(path, nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0)
    writer.write_frame("1999-05-03_12:00:00", dict(fields))
    writer.close()

    expected = {
        "QNWFA": ("water-friendly aerosol number con", "  kg(-1)",
                  ("Time", "bottom_top", "south_north", "west_east")),
        "QNIFA": ("ice-friendly aerosol number con", "  kg(-1)",
                  ("Time", "bottom_top", "south_north", "west_east")),
        "QNWFA2D": ("Surface aerosol number conc emission", "kg-1 s-1",
                    ("Time", "south_north", "west_east")),
        "QNIFA2D": ("Surface dust number conc emission", "kg-1 s-1",
                    ("Time", "south_north", "west_east")),
    }
    with netCDF4.Dataset(path, "r") as ds:
        for name, (description, units, dims) in expected.items():
            variable = ds.variables[name]
            assert variable.dimensions == dims, name
            assert variable.dtype == np.float32, name
            assert variable.description == description, name
            assert variable.units == units, name
            assert int(variable.FieldType) == 104, name
            assert variable.stagger == "", name
            np.testing.assert_array_equal(
                np.asarray(variable[0]), fields[name])


@requires_gpu
def test_a_real_mp28_state_reaches_the_file_through_state_frame(tmp_path):
    """The loop closed: DEVICE state -> frame builder -> NetCDF -> readback.

    The test above writes a hand-built frame, which proves the writer and
    the schema agree.  This proves the FRAME BUILDER agrees with them: an
    inventory row that never picks up ``state.nwfa`` would pass every other
    test in this module and ship a wrfout with no aerosol in it.  The
    values are compared against the device arrays themselves, so a mapping
    that carried the right name onto the wrong array also fails.
    """
    import cupy as cp
    import netCDF4

    from gpuwm.io.wrfout import WrfoutWriter, state_frame

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CUDA device")

    cfg = validate_run_config(_cfg(nx=5, ny=4, nz=6))
    state = _mp28_domain_state(cp, cfg)
    # Distinguishable, non-constant values: a frame that transposed two
    # fields or wrote a broadcast constant could not pass.
    shape = state.nwfa.shape
    ramp = cp.asarray(
        np.arange(int(np.prod(shape)), dtype=np.float32).reshape(shape))
    state.nwfa[...] = ramp * np.float32(1.0e5)
    state.nifa[...] = ramp * np.float32(1.0e2) + np.float32(7.0)
    state.nwfa2d[...] = cp.asarray(
        np.arange(cfg.ny * cfg.nx, dtype=np.float32).reshape(cfg.ny, cfg.nx))
    state.nifa2d[...] = 0.0

    frame = state_frame(state)
    for name in ("QNWFA", "QNIFA", "QNWFA2D", "QNIFA2D", "QNCLOUD"):
        assert name in frame, name

    path = tmp_path / "wrfout_d01_mp28_state"
    writer = WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                          dx=cfg.dx, dy=cfg.dy)
    writer.write_frame("1999-05-03_12:00:00", dict(frame))
    writer.close()

    with netCDF4.Dataset(path, "r") as ds:
        np.testing.assert_array_equal(
            np.asarray(ds.variables["QNWFA"][0]), cp.asnumpy(state.nwfa))
        np.testing.assert_array_equal(
            np.asarray(ds.variables["QNIFA"][0]), cp.asnumpy(state.nifa))
        np.testing.assert_array_equal(
            np.asarray(ds.variables["QNWFA2D"][0]),
            cp.asnumpy(state.nwfa2d))
        np.testing.assert_array_equal(
            np.asarray(ds.variables["QNIFA2D"][0]),
            cp.asnumpy(state.nifa2d))
        # QNCLOUD is the pre-existing row that starts carrying a PROGNOSTIC
        # droplet number under mp=28 -- same inventory, different values.
        np.testing.assert_array_equal(
            np.asarray(ds.variables["QNCLOUD"][0]), cp.asnumpy(state.nc))
        assert ds.variables["QNWFA"].units == "  kg(-1)"
        assert ds.variables["QNWFA2D"].units == "kg-1 s-1"


# ---------------------------------------------------------------------------
# 12.  The experiment front door -- the schema production runs are written in.
# ---------------------------------------------------------------------------

_EXPERIMENT_TOML = """
[experiment]
name = "mp28-front-door"
run_seconds = 60.0
start_time = 1999-05-03T12:00:00
restart_interval_s = 0.0

[shared]
nz = 30
ztop = 15000.0

[[domain]]
grid_id = 1
parent_id = 0
i_parent_start = 1
j_parent_start = 1
parent_grid_ratio = 1
parent_time_step_ratio = 1
history_interval_s = 60.0
nx = 20
ny = 16
dx = 3000.0
dy = 3000.0
time_step = 10
moist = true
mp_physics = 28
"""


def test_an_experiment_toml_can_select_mp28():
    """The production front door, not just RunConfig.

    ``build_experiment`` runs the same ``validate_run_config`` battery per
    domain, so pre-WP-11a this raised the mp_physics schema ValueError and
    no experiment TOML could name 28.
    """
    import tomllib

    from gpuwm.experiment import build_experiment

    experiment = build_experiment(
        tomllib.loads(_EXPERIMENT_TOML), source="mp28-front-door")
    run = experiment.domains[0].run
    assert run.mp_physics == 28
    assert run.aer_init_opt == 0 and run.wif_input_opt == 0


@pytest.mark.parametrize("key", ["aer_init_opt", "wif_input_opt"])
def test_the_experiment_schema_also_fails_closed_on_the_aerosol_keys(key):
    """Both front doors refuse; they just say different things.

    ``gpuwm/experiment.py``'s per-domain key allow-list is independent of
    ``RunConfig``'s field set, so these two keys are refused there as
    UNKNOWN keys rather than as unimplemented ones.  That is a weaker
    message than ``validate_aerosol_source_options`` gives on the legacy
    ``load_config`` path, and it is recorded here as measured behaviour
    rather than assumed away: what matters for the scope pin is that
    neither door lets a nonzero value through, and neither does.
    """
    import tomllib

    from gpuwm.experiment import build_experiment

    text = _EXPERIMENT_TOML + f"{key} = 1\n"
    with pytest.raises(ValueError) as caught:
        build_experiment(tomllib.loads(text), source="mp28-front-door")
    assert key in str(caught.value)


def test_the_legacy_config_path_gives_the_named_refusal(tmp_path):
    """``load_config`` derives its key set from ``dataclasses.fields``, so
    this is the path on which the two selectors are settable -- and
    therefore the path on which the named refusal has to fire."""
    from gpuwm.config import load_config

    path = tmp_path / "mp28.toml"
    path.write_text(
        "[grid]\nnx = 4\nny = 3\nnz = 8\ndx = 1000.0\ndy = 1000.0\n"
        "ztop = 10000.0\n\n"
        "[run]\ndt = 10.0\nrun_seconds = 10.0\nmoist = true\n"
        "mp_physics = 28\nwif_input_opt = 1\n",
        encoding="utf-8", newline="\n")
    with pytest.raises(NotImplementedError) as caught:
        load_config(path)
    assert "wif_input_opt=1" in str(caught.value)

    path.write_text(
        "[grid]\nnx = 4\nny = 3\nnz = 8\ndx = 1000.0\ndy = 1000.0\n"
        "ztop = 10000.0\n\n"
        "[run]\ndt = 10.0\nrun_seconds = 10.0\nmoist = true\n"
        "mp_physics = 28\n",
        encoding="utf-8", newline="\n")
    cfg = load_config(path)
    assert cfg.mp_physics == 28


def test_a_mixed_column_asking_for_28_anywhere_still_refuses_the_wif_key(
        tmp_path):
    """The &domains half of the sweep peeks, and must peek at the COLUMN.

    ``wif_input_opt`` lives in &domains (registry.new3d_wif:17) while
    ``mp_physics`` lives in &physics, and ``_Section.finish()`` for
    &domains fires long before mp_physics is mapped -- so the sweep reads
    the unconsumed &physics entry.  Reading only its ROOT element would let
    ``mp_physics = 6, 28`` drop a WIF key as inert on its way to the
    separate non-uniform-column rejection, i.e. a refusal that names the
    wrong thing.  Any domain asking for 28 is enough.
    """
    from gpuwm.namelist_import import import_namelists

    wps, inp = _namelist_pair(
        tmp_path, mp="6, 28", domains_extra=" wif_input_opt = 1,\n")
    with pytest.raises(ValueError) as caught:
        import_namelists(wps, inp, name="mixed-mp-column",
        acknowledgements=_CONSTANT_GLW_ACK)
    message = str(caught.value)
    assert "&domains wif_input_opt" in message
    assert "module_initialize_real.F:2735-2736" in message


def test_mp28_is_a_component_override_and_never_a_default():
    """The registry decides reachability, and this pins what it decided.

    The comment in ``pending_wrf_physics_components`` justifies appending
    no blocker for mp=28 partly on the ground that the registry keeps it
    off every template and out of every route's source template list.  That
    is a checkable claim about a shipped document, so it is checked --
    otherwise a later registry edit could quietly make aerosol-aware
    Thompson somebody's default while this file still says it cannot be.
    """
    from gpuwm.physics_compat import MP28_REGISTRY_OPTION_ID
    from gpuwm.physics_registry import DEFAULT_TEMPLATE_ID, physics_registry

    registry = physics_registry()
    option = registry["components"]["microphysics"]["options"][
        MP28_REGISTRY_OPTION_ID]
    assert option["selectors"] == {"mp_physics": 28}
    assert option["implemented"] is True
    assert option["reachability"]["state"] == "component-override"

    by_template = {
        name: template.get("components", {}).get("microphysics")
        for name, template in registry["templates"].items()
    }
    assert MP28_REGISTRY_OPTION_ID not in by_template.values()
    assert by_template.get(DEFAULT_TEMPLATE_ID) != MP28_REGISTRY_OPTION_ID
    for route in registry["runner_routes"].values():
        for template_ids in route.get("source_template_ids", {}).values():
            for template_id in template_ids:
                assert by_template.get(template_id) != MP28_REGISTRY_OPTION_ID


def test_the_aerosol_units_are_wrfs_resolved_value_not_the_registry_line():
    """WRF blanks every ``#`` inside a quoted Registry string.

    ``Registry/registry.new3d_wif:88`` spells QNWFA's units ``"# kg(-1)"``,
    but ``tools/reg_parse.c:201-208`` -- "check line and zap any # characters
    that are in double quotes", ``else if ( *p == '#' && inquote ) *p = ' '``
    -- runs over the raw line BEFORE tokenizing, so the value WRF actually
    carries is ``"  kg(-1)"``.  WRF's own generated table in the built tree
    confirms it: ``scalar_units_table( idomain, P_qnwfa ) = '  kg(-1)'``
    (``inc/scalar_indices.inc:2586``; :2600 for ``P_qnifa``).

    This test exists because the first version of the schema row here
    transcribed the Registry LINE and shipped ``"# kg(-1)"`` -- a units
    string no WRF file contains.  It fails on that version.

    The five number-concentration rows in ``_VAR_META`` that were verified
    against a stock v4.6.1 wrfout are the in-tree corroboration, so the
    aerosol rows are additionally required to agree with QNCLOUD's spelling
    rather than merely to avoid a literal.
    """
    from gpuwm.io.wrf_output_schema import THOMPSON_AEROSOL_OUTPUT_FIELDS
    from gpuwm.io.wrfout import _VAR_META

    verified_number_units = _VAR_META["QNCLOUD"][1]
    assert "#" not in verified_number_units
    for name in ("QNWFA", "QNIFA"):
        units = THOMPSON_AEROSOL_OUTPUT_FIELDS[name].units
        assert "#" not in units, (
            f"{name} units {units!r} carries a '#' that WRF's own Registry "
            "parser blanks (tools/reg_parse.c:206)")
        assert units == verified_number_units
    for field in THOMPSON_AEROSOL_OUTPUT_FIELDS.values():
        assert "#" not in field.description
        assert "#" not in field.units


@requires_gpu
def test_an_output_due_mp28_call_produces_a_real_radar_field():
    """REFL_10CM is the port's one user-facing product field, and until
    WP-11a nothing in the port compared it at all.

    Pre-WP-11a this path could not run: ``compute_refl_10cm`` raised on
    mp_physics=28 before it looked at anything.  The assertion is
    deliberately about the VALUES as well as the plumbing -- a field that
    is entirely the -35 dBZ floor would satisfy "a field was produced"
    while proving nothing, so the column is required to contain real
    returns from the hydrometeors the state carries.
    """
    import types

    import cupy as cp

    from gpuwm.core import microphysics
    from gpuwm.core.refl import consume_refl_10cm, refl_10cm_is_stashed

    try:
        cp.cuda.runtime.getDeviceCount()
    except Exception:                                   # pragma: no cover
        pytest.skip("no CUDA device")
    _tables_or_skip()

    cfg = validate_run_config(_cfg(nx=6, ny=5, nz=12))
    state = _mp28_domain_state(cp, cfg)
    # stash_refl_10cm requires the one-frame handoff slot on the driver;
    # a namespace is enough and keeps this test off the PhysicsDriver.
    state.physics = types.SimpleNamespace(refl_10cm=None, state=state)

    microphysics.apply(state, cfg, cfg.dt, refl_10cm_due=True)
    cp.cuda.Stream.null.synchronize()

    assert refl_10cm_is_stashed(state)
    refl = cp.asnumpy(consume_refl_10cm(state))
    assert np.all(np.isfinite(refl))
    # WRF clamps to -35 dBZ (mp_gt_driver:1461); a real column must exceed
    # the floor somewhere, and must stay in a physical range.
    assert refl.min() >= -35.0 - 1e-5
    assert refl.max() > -35.0
    assert refl.max() < 100.0
    assert not refl_10cm_is_stashed(state)


# ---------------------------------------------------------------------------
# 11.  REFL_10CM against WRF's own calc_refl10cm, on all 19 fixtures.
#
# The 19 aerosol column fixtures each carry a 23rd column, ``refl_dbz``,
# produced by the oracle's own ``mp_gt_driver`` call with
# ``diagflag=.true., do_radar_ref=1`` (tools/thompson_wrf461_oracle/
# run_column_aero.F90:296), i.e. by unmodified WRF v4.6.1
# ``calc_refl10cm``.  NOTHING in the port read that column before WP-11a:
# the adapter gate compares 16 fields and reflectivity is not one of them.
# So the one user-facing PRODUCT field mp=28 publishes was, until this
# test, the only field in the scheme with no oracle comparison at all.
# ---------------------------------------------------------------------------

#: dBZ agreement gate for every fixture whose 16 state fields already clear
#: the adapter's own 2e-6 relative gate.  2.0e-04 is a PIN: it must be
#: tightened, never widened.
#:
#: RE-MEASURED IN WP-14, ON THE WHOLE 22-COLUMN DECK, AND IT IMPROVED.  Every
#: fixture now agrees with WRF to <= 3.4332e-05 dBZ over its signal levels --
#: the gate has a 5.8x margin on the worst column in the deck, where the
#: measurement this constant was originally sized from had 1.3x.  19 of the 22
#: are inside 1.15e-05 dBZ and SEVEN are BIT-EXACT (aero-ccn-activate,
#: aero-ccn-sweep, aero-ice-demott-dep, aero-ice-koop, aero-init-profile,
#: aero-sfc-emit, wp08-melt).  Worst three: aero-cold-overlap 3.4332e-05,
#: aero-reduces-to-classic 3.2425e-05, wp08-freeze 2.0981e-05.
#:
#: The gate is NOT tightened to match, and that is a deliberate refusal
#: rather than an oversight: 2.0e-04 dB is the number
#: ``tests/test_thompson_aerosol_adapter.py::_REFL_DB_GATE`` carries and the
#: number the public documents quote, and having the two files hold the same
#: constant is worth more than a 5x tightening in one of them.  The margin is
#: published here instead, and the ratchet that actually catches a
#: reflectivity regression is the ULP table in the adapter file, where the
#: same residuals read 0 to 9 float32 ulps of the dBZ value.
_REFL_DBZ_GATE = 2.0e-4

#: The one fixture whose reflectivity residual was INHERITED, with its cause
#: measured rather than asserted.  ``aero-reduces-to-classic`` is the only
#: fixture carrying a declared end-to-end carve-out
#: (``tests/test_thompson_aerosol_adapter.py::_END_TO_END_BOUNDS``, which is
#: now ``{'nr_per_kg': 1.0e-5}`` -- it read ``{'qr': 0.0025,
#: 'nr_per_kg': 0.0025}`` when the paragraph below was written, and the
#: sequence since has been 2.5e-03 -> 1.0e-04 -> qr deleted, nr 1.0e-05, i.e.
#: 250x STRICTER with nothing widened), and the carve-out lives at
#: level 6.  Level 6 is also where its dBZ residual lived, and the two agreed
#: quantitatively:
#:
#:   qr  got 2.538754870e-06  want 2.543626124e-06  rel 1.915082e-03
#:   nr  got 1.504648906e+05  want 1.507545938e+05  rel 1.921687e-03
#:   dBZ got -25.227710724    want -25.219974518    diff 7.736206e-03 dB
#:
#: 7.736206e-03 dB is 10*log10(1 + eps_Z) with eps_Z = 1.7827e-03 -- the
#: same order as the qr/nr residual that produced it, as it must be for a
#: reflectivity that is a smooth function of the rain moments and of
#: nothing else.  So this is not a reflectivity defect: it is the rain
#: residual WP-09 declared, expressed in dB.  The gate is stated separately
#: instead of raising the global one, so a regression anywhere else stays
#: caught at 2.0e-04.
#:
#: RETIRED BY WP-13a, AND KEPT AS AN EMPTY DICT ON PURPOSE.  Restoring WRF's
#: level-wise :3237-vs-:3568 sedimentation density took aero-reduces-to-
#: classic's worst |dBZ - WRF| from 5.283e-04 dB (already down from the
#: 7.736e-03 this carve-out was sized for) to 3.242e-05 dB, so EVERY fixture
#: in the deck now clears the flat 2.0e-04 dB gate and the exemption buys
#: nothing.  Leaving the dict in place keeps `_REFL_DBZ_INHERITED_GATE.get`
#: as the single lookup and makes re-adding an entry a deliberate act.
_REFL_DBZ_INHERITED_GATE: dict[str, float] = {}


def _refl_oracle_harness():
    """The WP-09 fixture harness, or a skip naming what is missing.

    Imported rather than reimplemented: the entry-state reconstruction
    (pressure/theta/geopotential preimages that reproduce the fixture's
    recorded temperature and dz bitwise) is the hard part of driving these
    fixtures, it is already reviewed, and a second copy of it here would be
    a second authority on what the oracle's "before" column MEANS.
    """
    try:
        import test_thompson_aerosol_adapter as harness
    except ImportError as exc:                          # pragma: no cover
        pytest.skip(f"aerosol oracle harness unavailable: {exc}")
    if not harness._FIXTURES:                           # pragma: no cover
        pytest.skip("no aerosol column fixtures present")
    return harness


@requires_gpu
def test_refl_10cm_reproduces_wrfs_calc_refl10cm_on_all_nineteen_fixtures():
    """The reflectivity oracle gate, which did not exist before WP-11a.

    Fails on the pre-WP-11a tree at ``compute_refl_10cm``, which raised
    ``do_radar_ref needs an active microphysics scheme (mp_physics 1, 6, 8,
    or 10), got 28`` before reading a single array -- so no version of this
    comparison could run at all.

    What is compared: the field a due mp=28 call stashes, against the
    ``refl_dbz`` column WRF's own ``calc_refl10cm`` wrote into each fixture,
    level by level, in dBZ.  Three independent claims:

    1. THE CLAMP IS BITWISE.  Every level where WRF reports its -35 dBZ
       floor (``refl_10cm(i,k,j) = MAX(-35., dBZ(k))``,
       module_mp_thompson.F:1461) must be exactly -35.0 here -- RE-MEASURED
       IN WP-14 over the whole 22-column deck: 426 of the 528 levels, all
       426 bitwise.  (The prose used to say 367 of 456, which was the
       19-column deck before the three WP-08 sedimentation columns joined
       it.)  A port that computed dBZ correctly but clamped at, say,
       -35.0000001 would pass a tolerance test and fail this one.
    2. THE SIGNAL AGREES.  The 102 levels with real returns must agree to
       :data:`_REFL_DBZ_GATE`.  ``_REFL_DBZ_INHERITED_GATE`` is now EMPTY --
       WP-13a closed the one fixture that needed it -- so this is the flat
       gate on every fixture, with no exception at all.
    3. THE COMPARISON IS NOT VACUOUS.  The fixtures must actually contain
       strong returns; the oracle's maximum is 51.98 dBZ, and a suite of
       all-floor columns would satisfy 1 and 2 while proving nothing.
    """
    import cupy as cp

    harness = _refl_oracle_harness()
    _tables_or_skip()

    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    floor = np.float32(-35.0)
    floor_levels = floor_exact = signal_levels = 0
    max_oracle_dbz = -1.0e30
    worst = {}
    for name in harness._FIXTURES:
        state, cfg, dt, _before, after, _surface, _report = \
            harness._build_case(cp, name)
        # stash_refl_10cm writes the one-frame handoff onto state.physics;
        # the harness's column stand-in has no driver, and a namespace is
        # all the contract requires.
        state.physics = SimpleNamespace(refl_10cm=None, state=state)
        _apply_thompson_aerosol(state, cfg, dt, refl_10cm_due=True)
        cp.cuda.Stream.null.synchronize()

        assert state.physics.refl_10cm is not None, name
        got = cp.asnumpy(state.physics.refl_10cm).ravel()
        want = harness._column(after, "refl_dbz")
        assert got.shape == want.shape, name
        assert np.all(np.isfinite(got)), name

        is_floor = want == floor
        floor_levels += int(is_floor.sum())
        floor_exact += int((got[is_floor] == floor).sum())
        signal = ~is_floor
        signal_levels += int(signal.sum())
        max_oracle_dbz = max(max_oracle_dbz, float(want.max()))
        # WRF's floor is a floor for gpuwm too, everywhere.
        assert float(got.min()) >= -35.0, name

        gate = _REFL_DBZ_INHERITED_GATE.get(name, _REFL_DBZ_GATE)
        diff = np.abs(got.astype(np.float64) - want.astype(np.float64))
        worst[name] = float(diff.max())
        assert worst[name] <= gate, (
            f"{name}: max |dBZ - WRF| = {worst[name]:.6e} exceeds {gate:.1e} "
            f"at level {int(np.argmax(diff)) + 1} "
            f"(got {got[int(np.argmax(diff))]}, "
            f"want {want[int(np.argmax(diff))]})")

    # 1. the clamp, bitwise, on every floor level in the suite
    assert floor_exact == floor_levels, (
        f"{floor_levels - floor_exact} of {floor_levels} oracle-floor levels "
        "are not bitwise -35.0 dBZ")
    assert floor_levels >= 300
    # 3. non-vacuous: the suite really does carry strong returns
    assert signal_levels >= 80, signal_levels
    assert max_oracle_dbz > 50.0, max_oracle_dbz
    # and the fixtures that clear the adapter gate on all 16 state fields
    # clear the reflectivity gate by two further decades
    clean = [value for name, value in worst.items()
             if name not in _REFL_DBZ_INHERITED_GATE]
    assert max(clean) <= _REFL_DBZ_GATE


@requires_gpu
def test_the_reflectivity_residual_is_the_declared_rain_residual_in_db():
    """Attribution, DERIVED from WRF's own moment algebra, not a pinned level.

    ``aero-reduces-to-classic`` USED TO BE the one fixture whose dBZ column
    missed the 2.0e-04 dB gate (see "WHAT WP-13a CHANGED" below -- it now
    clears it with a 6x margin), and the claim this test owns is that its
    residual, then and now, is the DECLARED rain-moment residual seen through
    ``calc_refl10cm`` -- not an independent reflectivity defect wearing an
    "inherited" label.

    THE DERIVATION (module_mp_thompson.F, all in ``calc_refl10cm``):

        :5915   ze_rain = N0_r * crg(4) * ilamr**cre(4)
        :5770   N0_r    = nr * org2 * lamr**cre(2)
        :5768   lamr    = (am_r*crg(3)*org2*nr/rr)**obmr,  obmr = 1/bm_r (:721)

    with ``mu_r = 0`` (:103), so ``cre(2) = mu_r + 1 = 1`` and
    ``cre(4) = bm_r*2 + mu_r + 1 = 7`` (:706, :708) and ``bm_r = 3``.
    Substituting:

        ze_rain  =  const * nr * lamr**(1 - 7)
                 =  const * nr * (nr/rr)**(-2)
                 =  const * rr**2 / nr                     EXACTLY

    so to first order the reflectivity-factor error is
    ``eps_Z = 2*eps_rr - eps_nr`` and the dB residual is
    ``10*log10(1 + 2*eps_qr - eps_nr)``, SIGNED.  Rain is the only
    contributor here: ``qs`` and ``qg`` are bitwise equal to WRF at every
    level of this column, so ``ze_snow`` and ``ze_graupel`` (:5980) cancel out
    of the difference exactly.

    WHAT WP-13a CHANGED, AND WHY THE ASSERTIONS INVERT.  This fixture no
    longer misses the 2.0e-04 dB gate at all.  Restoring WRF's level-wise
    :3237-vs-:3568 sedimentation density took the worst |dBZ - WRF| on this
    column from 5.283e-04 dB to 3.242e-05 dB -- a 16x margin INSIDE the flat
    gate -- which is why ``_REFL_DB_BOUNDS`` in
    tests/test_thompson_aerosol_adapter.py and
    :data:`_REFL_DBZ_INHERITED_GATE` in this file were both retired.  So the
    old assertions "the worst level's rain residual is above the G3 gate" and
    "2.0e-04 dB < worst < 1.0e-03 dB" are now FALSE and are replaced by their
    closure counterparts rather than relaxed.

    AND THE MECHANISM CLAIM HAD TO CHANGE UNITS WITH IT.  At 5e-04 dB the
    linearisation above was tested as a RATIO to 5%.  At 3e-05 dB it cannot
    be: the residual is now within a few float32 ulps of the dBZ value itself
    (1.9e-06 dB near -25 dBZ), and the old docstring's claim that ze_snow and
    ze_graupel "cancel out of the difference exactly" because qs and qg are
    bitwise equal is only true to first order -- calc_refl10cm also reads the
    temperature, and temp_k on this column differs by 1.393e-07 relative.  So
    the claim asserted below is the one that survives at this magnitude and is
    still falsifiable: at EVERY non-floor level the measured dB residual
    exceeds the rain-moment prediction by no more than 2.0e-05 dB, i.e. no
    level carries an independent reflectivity error bigger than a tenth of the
    gate.  A genuine reflectivity defect would break it; float32 dBZ rounding
    cannot.

    RE-MEASURED after the 1.4.1 merge (RTX 5090, cupy 14.1.1), predicted vs
    measured dB.  Every number below MOVED TOWARDS ZERO; the previous
    measurement is in parentheses where it differs materially:

        0-based level 0   -1.448e-07  vs  +5.722e-06
        0-based level 1   +0.000e+00  vs  +6.676e-06
        0-based level 2   -1.371e-06  vs  +6.676e-06
        0-based level 3   -2.965e-07  vs  +8.583e-06
        0-based level 4   -4.559e-07  vs  +6.914e-06
        0-based level 5   +1.801e-06  vs  +9.537e-06   <- worst
                          (was +2.631e-05  vs  +3.242e-05)

    So the worst |dBZ - WRF| on this column is now 9.537e-06 dB, a 21x
    margin inside the flat 2.0e-04 dB gate, and the rain-moment prediction
    is below the float32 dBZ ulp at EVERY non-floor level.  Assertion 4
    below records that as a closure.
    """
    import cupy as cp

    harness = _refl_oracle_harness()
    _tables_or_skip()

    from gpuwm.core.microphysics_aerosol import _apply_thompson_aerosol

    name = "aero-reduces-to-classic"
    assert name in harness._FIXTURES
    state, cfg, dt, _before, after, _surface, _report = \
        harness._build_case(cp, name)
    state.physics = SimpleNamespace(refl_10cm=None, state=state)
    _apply_thompson_aerosol(state, cfg, dt, refl_10cm_due=True)
    cp.cuda.Stream.null.synchronize()

    got = cp.asnumpy(state.physics.refl_10cm).ravel().astype(np.float64)
    want = harness._column(after, "refl_dbz").astype(np.float64)

    def signed(field, key):
        mine = cp.asnumpy(field).ravel().astype(np.float64)
        theirs = harness._column(after, key).astype(np.float64)
        return np.where(theirs != 0.0, (mine - theirs)
                        / np.where(theirs != 0.0, theirs, 1.0), 0.0)

    eps_qr = signed(state.qr, "qr")
    eps_nr = signed(state.nr, "nr_per_kg")

    # Nothing else feeding calc_refl10cm disagrees ANYWHERE in this column,
    # so ze_snow and ze_graupel cancel out of the difference identically.
    for field, key in ((state.qs, "qs"), (state.qg, "qg")):
        mine = cp.asnumpy(field).ravel().astype(np.float64)
        assert np.array_equal(mine, harness._column(after, key)
                              .astype(np.float64)), key

    # WRF's -35 dBZ clamp (mp_gt_driver:1461) destroys the signal at the
    # levels it fires on, so the prediction cannot be tested there; require
    # bitwise agreement instead, which is stronger than any tolerance.
    floor = want == -35.0
    assert np.all(got[floor] == -35.0)
    assert int(floor.sum()) >= 18, int(floor.sum())

    difference = got - want
    level = int(np.argmax(np.abs(difference)))
    assert not floor[level]

    # 1. THE CLOSURE.  WP-13a's sedimentation-density fix put this column
    #    inside the flat gate with a 16x margin; it used to sit at 5.283e-04
    #    dB and need a 1.0e-03 dB carve-out.  Asserted as an upper bound so a
    #    regression re-opens it here instead of quietly re-adding a bound.
    assert abs(difference[level]) <= _REFL_DBZ_GATE, (
        f"level {level}: |dBZ - WRF| = {abs(difference[level]):.6e} dB is "
        f"back outside the flat {_REFL_DBZ_GATE:.1e} dB gate WP-13a put it "
        "inside")
    # RATCHETED by the 1.4.1 merge from 5.0e-5 to 2.0e-5 dB: the inherited
    # mp=8 sedimentation reconciliations took the worst |dBZ - WRF| on this
    # column from 3.242e-05 dB to 9.537e-06 dB, a further 3.4x.
    assert abs(difference[level]) <= 2.0e-5, abs(difference[level])
    # Non-vacuous: the column really does carry the returns being compared.
    assert float(want.max()) > 8.0, float(want.max())

    # 2. THE ATTRIBUTION.  The worst dBZ level is the level carrying the
    #    largest rain residual on the column -- the dB miss lands where the
    #    rain moments' miss is, not somewhere else.
    #    Restricted to the levels that carry a return: 0-based level 6 has
    #    this column's biggest rain residual by far (3.155e-03) and WRF puts
    #    it AT the -35 dBZ floor, where mp_gt_driver:1461's clamp destroys the
    #    signal and both ports are required to be bitwise -35.0 above.
    rain_eps = np.where(floor, 0.0, np.abs(2.0 * eps_qr - eps_nr))
    assert level == int(np.argmax(rain_eps)), (level, int(np.argmax(rain_eps)))
    assert floor[int(np.argmax(np.abs(2.0 * eps_qr - eps_nr)))], (
        "the column's largest rain residual is no longer at the dBZ floor; "
        "this test's restriction to non-floor levels needs re-deriving")

    # 3. THE MECHANISM, in the only units that still resolve it.  At every
    #    non-floor level the measured dB residual may exceed the rain-moment
    #    prediction by no more than a tenth of the gate.  Everything above the
    #    prediction is float32 dBZ representation plus the temp_k residual
    #    reaching ze_snow/ze_graupel; both are bounded by construction, an
    #    independent reflectivity defect is not.
    predicted = 10.0 * np.log10(1.0 + 2.0 * eps_qr - eps_nr)
    unexplained = 0.0
    resolvable = 0
    for k in range(want.size):
        if floor[k]:
            continue
        excess = abs(difference[k]) - abs(predicted[k])
        unexplained = max(unexplained, excess)
        assert excess <= 2.0e-5, (
            f"level {k}: measured {difference[k]:.6e} dB exceeds the "
            f"rain-moment prediction {predicted[k]:.6e} dB by {excess:.6e} "
            "dB; that is more than a tenth of the gate and needs its own "
            "explanation")
        if abs(predicted[k]) > 1.0e-5:
            resolvable += 1
            ratio = difference[k] / predicted[k]
            assert 0.5 < ratio < 2.0, (k, ratio, difference[k], predicted[k])
    # 4. THE SECOND CLOSURE, and this assertion INVERTS for the same reason
    #    the dB carve-out did.  It used to read "the prediction is above the
    #    dBZ ulp at exactly one level, and that level is ALSO the worst
    #    level".  The 1.4.1 merge inherited the mp=8 lane's two sedimentation
    #    reconciliations (5e4af4e3, cb765336) into the frozen kernel this
    #    port shares for rain fallout, level 5's nr residual went 5.700e-06
    #    -> 4.146e-07, and the rain-moment prediction with it: 2.631e-05 dB
    #    -> 1.801e-06 dB.  There is now NO non-floor level where the
    #    prediction is resolvable against float32 dBZ at all.
    #
    #    That is a closure, not a loss of evidence, and it is asserted as
    #    one: the attribution above (assertion 2 -- the worst dB level is the
    #    worst rain-residual level) still holds and is still falsifiable, and
    #    a regression that re-grows the rain residual makes the prediction
    #    resolvable again and fails HERE.
    assert resolvable == 0, (
        f"the rain-moment prediction is resolvable again at {resolvable} "
        "non-floor level(s); the residual this test attributes has grown "
        "back above the float32 dBZ ulp and assertion 3's ratio test should "
        "be restored")
    assert abs(predicted[level]) <= 1.0e-5, predicted[level]
    assert unexplained > 0.0
