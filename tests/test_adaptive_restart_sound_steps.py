"""An adaptive run must be able to restart from its own checkpoints.

THE BUG THIS PINS.  ``AdaptiveClockDriver._apply`` overwrites TWO
``RunConfig`` fields on the live config every root step -- ``dt`` and
``time_step_sound`` -- while ``_require_config_match``'s identity walk
exempted only the first.  A checkpoint therefore stored the DERIVED
sound-step count and the walk compared it against the config's DECLARED
one and refused:

    time_step_sound: restart=6 run=4

so every checkpoint an adaptive run wrote became unrestartable as soon as
its step grew enough to move that count.  On a sub-km nest that is almost
immediate -- :func:`wrf_num_sound_steps` leaves 4 the moment
``300*dt/spacing >= 2`` -- which defeated ``restart_interval_s`` on
exactly the runs most likely to need it: the long ones, and the ones that
die.

The fix is NOT to stop comparing ``time_step_sound``.  Under a fixed
clock it is a real model difference and a resume that changes it must
still be refused; these gates pin both halves.
"""

from __future__ import annotations

from fractions import Fraction

import pytest

from gpuwm.core.adaptive_clock import (          # noqa: E402
    ADAPTIVE_DERIVED_RUN_FIELDS, wrf_num_sound_steps,
)


# ------------------------------------------------- the two ends agree

def test_the_restart_walk_reads_the_same_object_the_clock_derives_from():
    """One authority, not two lists that were kept in step by hand.

    The whole defect was that one call site mutated two fields and the
    other end knew about one of them.
    """
    from gpuwm.core import adaptive_clock
    from gpuwm.io import restart
    assert restart.ADAPTIVE_DERIVED_RUN_FIELDS is (
        adaptive_clock.ADAPTIVE_DERIVED_RUN_FIELDS)


def test_both_mutated_fields_are_named():
    assert ADAPTIVE_DERIVED_RUN_FIELDS == {"dt", "time_step_sound"}


def test_a_field_named_but_not_derived_raises_rather_than_being_exempted():
    """The dangerous direction, so it fails loudly and early.

    A field listed as derived but never computed would be exempted from
    the restart comparison while still holding its stale configured
    value -- quieter than the bug it replaced, and worse.
    """
    from gpuwm.core.adaptive_clock import AdaptiveClockDriver

    drv = AdaptiveClockDriver.__new__(AdaptiveClockDriver)
    drv.max_msft = {}

    class _Run:
        dx = dy = 2000.0
        grid_id = 1

    with pytest.raises(KeyError, match="does not compute it"):
        drv._derive_run_field("epssm", Fraction(30), _Run())


def test_every_named_field_is_actually_derivable():
    """The other direction: nothing in the set is unimplemented."""
    from gpuwm.core.adaptive_clock import AdaptiveClockDriver

    drv = AdaptiveClockDriver.__new__(AdaptiveClockDriver)
    drv.max_msft = {}

    class _Run:
        dx = dy = 667.0
        grid_id = 3

    for name in ADAPTIVE_DERIVED_RUN_FIELDS:
        value = drv._derive_run_field(name, Fraction(6), _Run())
        assert value is not None, name


# ------------------------------------ the boundary the run actually hit

def test_the_sound_step_count_really_does_move_on_a_sub_km_nest():
    """Why this bug was not theoretical.

    d03 at 667 m left 4 immediately; the refused checkpoint stored 6.
    """
    at30 = wrf_num_sound_steps(30.0, 667.0, 667.0, 1.0)
    at6 = wrf_num_sound_steps(6.0, 667.0, 667.0, 1.0)
    assert at6 != at30 or at30 != 4, (at30, at6)
    # And on a coarse root it stays put across the same range, which is
    # why a 10 km single-domain run would never have shown this.
    assert (wrf_num_sound_steps(30.0, 10000.0, 10000.0, 1.0)
            == wrf_num_sound_steps(50.0, 10000.0, 10000.0, 1.0) == 4)


# --------------------------------------------------- the walk's behaviour

def _cfg(**over):
    """A real RunConfig -- the walk calls dataclasses.asdict on it."""
    from gpuwm.config import RunConfig
    base = dict(nx=12, ny=12, nz=8, dx=1.0e4, dy=1.0e4, ztop=1.5e4,
                dt=30.0, run_seconds=300.0)
    base.update(over)
    return RunConfig(**base)


def _walk(stored_over, live_over):
    """Run the identity walk; return the refusal text, or None if it passed.

    ``stored_over`` is applied to a full asdict of the live config so the
    two differ ONLY in the keys under test -- otherwise every unrelated
    default would show up as a difference and the gate would pass (or
    fail) for reasons that have nothing to do with this bug.
    """
    import dataclasses
    from gpuwm.io.restart import _require_config_match

    cfg = _cfg(**live_over)
    stored = dataclasses.asdict(cfg)
    stored.update(stored_over)
    try:
        _require_config_match(stored, cfg, "checkpoint.npz")
    except Exception as exc:                       # noqa: BLE001
        return str(exc)
    return None


def test_a_moved_sound_step_count_is_forgiven_when_both_sides_are_adaptive():
    # live config declares 30 s / 4 sound steps; the run adapted to 6 s
    # and the driver rewrote the count to 6, which is what got stored.
    msg = _walk({"dt": 6.0, "time_step_sound": 6},
                {"use_adaptive_time_step": True, "dt": 30.0,
                 "time_step_sound": 4})
    assert msg is None or "time_step_sound" not in msg, msg


def test_a_moved_sound_step_count_is_STILL_refused_under_a_fixed_clock():
    """The half that must not regress.

    Under a fixed dt the declared value is the one the model ran, so a
    resume that changes it is a different model and must be refused.
    """
    msg = _walk({"time_step_sound": 6},
                {"use_adaptive_time_step": False, "dt": 30.0,
                 "time_step_sound": 4})
    assert msg is not None and "time_step_sound" in msg, msg


def test_flipping_the_feature_is_still_refused():
    """`use_adaptive_time_step` itself is never exempt."""
    msg = _walk({"use_adaptive_time_step": True, "dt": 6.0,
                 "time_step_sound": 6},
                {"use_adaptive_time_step": False, "dt": 30.0,
                 "time_step_sound": 4})
    assert msg is not None, "a resume that turns adaptive off was allowed"


# ------------------------------- controller policy may change on a resume

def test_the_prepared_cache_no_longer_compares_controller_policy():
    """It never described the artifact.

    Nothing on the preparation path reads a target or a clamp, so a tree
    prepared under one value is byte-for-byte a tree prepared under
    another.  Comparing them refused caches that described exactly the
    state the live config needed -- which blocked the one thing
    checkpoints are most wanted for: resuming a dead run with the
    setting that would have saved it.
    """
    from gpuwm.ingest.prepared_cache import compare_prepared_domain_config

    cached = {"run": {"target_cfl": 1.2, "min_time_step": 6, "nz": 61}}
    live = {"run": {"target_cfl": 0.9, "min_time_step": 1, "nz": 61}}
    _tolerated, differing = compare_prepared_domain_config(cached, live)
    assert differing == [], differing


def test_the_prepared_cache_still_refuses_a_field_that_DOES_describe_it():
    """The positive control: this is a skip list, not an off switch."""
    from gpuwm.ingest.prepared_cache import compare_prepared_domain_config

    cached = {"run": {"target_cfl": 1.2, "nz": 61}}
    live = {"run": {"target_cfl": 1.2, "nz": 49}}
    _tolerated, differing = compare_prepared_domain_config(cached, live)
    assert "run.nz" in differing, differing


def _notices(stored_over, live_over, drop=()):
    """Run the identity walk and collect the notices it published.

    The retune report goes through :func:`gpuwm.explain.warn`, not
    ``warnings.warn``, so it is read here through that module's observer
    seam -- the same seam a machine consumer uses to receive it as a
    field rather than as a line to recognize.

    ``drop`` removes keys from the STORED side, which ``_walk`` cannot
    do: it builds the stored dict from a full asdict, and a policy field
    absent from a header is exactly the case the sentinel used to leak
    into.
    """
    import dataclasses
    from gpuwm.explain import add_warning_observer, remove_warning_observer
    from gpuwm.io.restart import _require_config_match

    cfg = _cfg(**live_over)
    stored = dataclasses.asdict(cfg)
    stored.update(stored_over)
    for key in drop:
        stored.pop(key, None)
    seen = []
    add_warning_observer(seen.append)
    try:
        _require_config_match(stored, cfg, "checkpoint.npz")
    except Exception as exc:                       # noqa: BLE001
        return str(exc), [record["action"] for record in seen]
    finally:
        remove_warning_observer(seen.append)
    return None, [record["action"] for record in seen]


def test_a_resume_may_retune_the_controller_and_is_told_that_it_did():
    """Allowed, and never silent."""
    # stored side carries the OLD policy (what the dead leg ran);
    # live side is the retune the operator wants to resume with.
    msg, notices = _notices(
        {"dt": 6.0, "time_step_sound": 6,
         "min_time_step": 6, "target_cfl": 1.2},
        {"use_adaptive_time_step": True, "dt": 30.0,
         "time_step_sound": 4, "min_time_step": 1, "target_cfl": 0.9})
    assert msg is None, msg
    assert len(notices) == 1, notices
    text = notices[0]
    assert "min_time_step" in text and "target_cfl" in text, text
    # BOTH values of each field, not merely the names
    assert "6 -> 1" in text and "1.2 -> 0.9" in text, text


def test_the_retune_notice_is_not_a_refusal_under_error_filters():
    """A ``warnings.warn`` here made the ALLOWED retune an exception.

    Under ``-W error`` -- which any caller with
    ``simplefilter("error")`` runs, and CI does -- it raised
    ``UserWarning`` out of the middle of the identity gate, so the one
    recovery this branch exists to permit failed with a traceback
    instead of resuming.  The project's own channel prints and returns.
    """
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        msg, notices = _notices(
            {"dt": 6.0, "time_step_sound": 6, "target_cfl": 1.2},
            {"use_adaptive_time_step": True, "dt": 30.0,
             "time_step_sound": 4, "target_cfl": 0.9})
    assert msg is None, msg
    assert len(notices) == 1, notices


def test_a_policy_field_the_checkpoint_never_had_is_named_as_absent():
    """Never repr() the sentinel.

    A header written between two builds carries
    ``use_adaptive_time_step`` but not every policy field beside it, and
    the notice whose whole job is to state the old value precisely then
    printed ``<object object at 0x...>`` -- a memory address that moves
    between runs and says nothing about why the clock changed.
    """
    from gpuwm.core.model import ADAPTIVE_POLICY_RUN_FIELDS

    field = sorted(ADAPTIVE_POLICY_RUN_FIELDS)[0]
    msg, notices = _notices(
        {"dt": 6.0, "time_step_sound": 6},
        {"use_adaptive_time_step": True, "dt": 30.0, "time_step_sound": 4},
        drop=(field,))
    assert msg is None, msg
    assert len(notices) == 1, notices
    text = notices[0]
    assert field in text, text
    assert "object object at" not in text, text
    assert "absent from the restart file" in text, text


def test_an_unchanged_policy_says_nothing_at_all():
    """The warning must mean something, so it cannot fire on every resume."""
    msg, notices = _notices(
        {"dt": 6.0, "time_step_sound": 6},
        {"use_adaptive_time_step": True, "dt": 30.0,
         "time_step_sound": 4})
    assert msg is None, msg
    assert notices == [], notices


def test_the_policy_set_excludes_the_feature_flag():
    """Flipping adaptive off leaves the carried controller state meaningless."""
    from gpuwm.core.model import ADAPTIVE_POLICY_RUN_FIELDS
    assert "use_adaptive_time_step" not in ADAPTIVE_POLICY_RUN_FIELDS
    assert len(ADAPTIVE_POLICY_RUN_FIELDS) == 11


# ------------------------------- the third gate: the physics fingerprint

def _fp(**over):
    from gpuwm.io.restart import _configuration_fingerprint
    return _configuration_fingerprint(_cfg(**over))


def test_the_configuration_fingerprint_forgives_what_the_walk_forgives():
    """Three gates, one argument -- and it had to be made in all three.

    The identity walk allowed a retuned resume and the fingerprint then
    refused it anyway, with a message that named no field.  A gate that
    contradicts the gate beside it is worse than either alone.
    """
    base = dict(use_adaptive_time_step=True, target_cfl=1.2,
                min_time_step=6, time_step_sound=4, dt=2.0)
    moved = dict(base, target_cfl=1.1, min_time_step=1,
                 time_step_sound=6, dt=6.0)
    assert _fp(**base) == _fp(**moved)


def test_a_fixed_clock_fingerprint_still_binds_the_derived_values():
    base = dict(use_adaptive_time_step=False, time_step_sound=4)
    assert _fp(**base) != _fp(**dict(base, time_step_sound=6))


def test_the_fingerprint_still_moves_for_a_real_field():
    """Positive control: this is an exemption, not an off switch."""
    base = dict(use_adaptive_time_step=True)
    assert _fp(**base) != _fp(**dict(base, nz=61))
