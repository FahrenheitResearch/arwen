"""The adaptive-timestep namelist surface: scoping, refusals, cache identity.

Behaviour is not wired yet -- these gate the SURFACE, which is the part
that has historically broken things in this package without anyone
noticing until an upgrade refused every prepared tree in the field.
"""

from __future__ import annotations

import dataclasses
from fractions import Fraction

import pytest

from gpuwm.config import RunConfig, validate_run_config
from gpuwm.experiment import _DOMAIN_RUN_OVERRIDES
from gpuwm.ingest.prepared_cache import (DEFAULT_TOLERANT_IDENTITY_FIELDS,
                                         PREPARATION_INERT_RUN_FIELDS,
                                         undelayed_identity_defaults)

#: Every field this campaign added to RunConfig.
ADAPTIVE_FIELDS = (
    "use_adaptive_time_step", "step_to_output_time", "adaptation_domain",
    "target_cfl", "target_hcfl", "max_step_increase_pct",
    "starting_time_step", "starting_time_step_den",
    "max_time_step", "max_time_step_den",
    "min_time_step", "min_time_step_den",
)

#: WRF Registry.EM_COMMON scope `1` -- one scalar for the whole run.
GLOBAL_SCOPE = ("use_adaptive_time_step", "step_to_output_time",
                "adaptation_domain")


def _cfg(**over) -> RunConfig:
    base = dict(nx=12, ny=12, nz=8, dx=1.0e4, dy=1.0e4, ztop=1.5e4,
                dt=30.0, run_seconds=300.0)
    base.update(over)
    return RunConfig(**base)


# --------------------------------------------------------------- defaults

def test_defaults_match_the_wrf_registry():
    """Registry.EM_COMMON:2269-2281.  A wrong default is a silent divergence."""
    cfg = _cfg()
    assert cfg.use_adaptive_time_step is False
    assert cfg.step_to_output_time is True      # Registry default IS true
    assert cfg.adaptation_domain == 1
    assert cfg.target_cfl == pytest.approx(1.2)
    assert cfg.target_hcfl == pytest.approx(0.84)
    assert cfg.max_step_increase_pct == 5
    for name in ("starting_time_step", "max_time_step", "min_time_step"):
        assert getattr(cfg, name) == -1, name
        assert getattr(cfg, f"{name}_den") == 0, name


def test_the_feature_is_off_by_default():
    """Everything below rests on this: defaulted off must stay bit-inert."""
    assert _cfg().use_adaptive_time_step is False


# ------------------------------------------------------- cache identity

def test_every_new_field_carries_a_prepared_cache_ruling():
    """The trap this table's own comment records four instances of.

    A field that joins RunConfig with NO ruling in any of the
    prepared-cache tables refuses EVERY tree prepared before it, with an
    error that points at the user's experiment TOML when the cause was a
    package upgrade.  Either ruling closes that hole, and each of the two
    the block uses is checked by name below.
    """
    ruled = DEFAULT_TOLERANT_IDENTITY_FIELDS | PREPARATION_INERT_RUN_FIELDS
    missing = [f"run.{n}" for n in ADAPTIVE_FIELDS
               if f"run.{n}" not in ruled]
    assert not missing, (
        f"these adaptive fields joined RunConfig without a prepared-cache "
        f"ruling: {missing}.  Every tree prepared before them would be "
        f"refused.")


def test_the_controller_policy_is_skipped_outright_not_merely_tolerated():
    """Tolerance is not enough for a field a RECOVERY resume must change.

    Tolerating a field forgives it ABSENT from an older header and only
    at its default; the eleven controller settings have to be forgiven at
    ANY value and in both directions, or a tree prepared under one target
    refuses the run that needs another -- which is exactly the run whose
    controller setting killed it.  Nothing on the preparation path reads
    them, so the stronger ruling is the true one.
    """
    for name in ADAPTIVE_FIELDS:
        key = f"run.{name}"
        if name == "use_adaptive_time_step":
            # The flag itself is NOT inert: a bundle prepared with the
            # feature on describes a tree built for a different clock.
            assert key in DEFAULT_TOLERANT_IDENTITY_FIELDS, key
            assert key not in PREPARATION_INERT_RUN_FIELDS, key
            continue
        assert key in PREPARATION_INERT_RUN_FIELDS, (
            f"{key} governs how the run INTEGRATES and is read by nothing "
            f"on the preparation path, so it belongs in the skipped-"
            f"outright partition rather than the tolerance table")
        assert key not in DEFAULT_TOLERANT_IDENTITY_FIELDS, (
            f"{key} carries two contradictory rulings")


def test_tolerance_defaults_are_read_from_the_dataclass():
    """The not-in-use value must BE the dataclass default, not a copy of it."""
    defaults = undelayed_identity_defaults(
        type("E", (), {"start_time": None})())
    fields = {f.name: f for f in dataclasses.fields(RunConfig)}
    for name in ADAPTIVE_FIELDS:
        key = f"run.{name}"
        if key not in DEFAULT_TOLERANT_IDENTITY_FIELDS:
            # A skipped-outright field needs no not-in-use value: the
            # comparison never reaches it at any value.
            assert key not in defaults, (
                f"{key} is skipped outright but still declares a "
                f"not-in-use value, which is a second ruling on one field")
            continue
        assert key in defaults, f"{key} has no declared not-in-use value"
        assert defaults[key] == fields[name].default, (
            f"{key} not-in-use value {defaults[key]!r} disagrees with "
            f"RunConfig's default {fields[name].default!r}")


# ------------------------------------------------------------- scoping

def test_per_domain_keys_are_overridable():
    """Registry `max_domains` keys must be settable on a [[domain]]."""
    for name in ADAPTIVE_FIELDS:
        if name in GLOBAL_SCOPE:
            continue
        assert name in _DOMAIN_RUN_OVERRIDES, (
            f"{name} is declared max_domains by WRF but cannot be set per "
            f"domain here")


def test_global_scope_keys_are_NOT_per_domain():
    """Registry scope `1` keys must be [shared]-only, enforced not documented.

    A per-domain `use_adaptive_time_step` would let one domain of a tree
    adapt while its parent did not, which upstream has no representation
    for at all.
    """
    for name in GLOBAL_SCOPE:
        assert name not in _DOMAIN_RUN_OVERRIDES, (
            f"{name} is WRF scope 1 (one scalar per run) but is listed as "
            f"a per-domain override")


# ------------------------------------------------------------ refusals

@pytest.mark.parametrize("over,fragment", [
    (dict(target_cfl=0.0), "target_cfl must be positive"),
    (dict(target_cfl=-1.0), "target_cfl must be positive"),
    (dict(target_hcfl=0.0), "target_hcfl must be positive"),
    (dict(max_step_increase_pct=-5), "max_step_increase_pct must be >= 0"),
    (dict(adaptation_domain=0), "adaptation_domain must be 1"),
    # ABOVE 1 IS REFUSED TOO, and that is the fix: the child-driven arm
    # nest_dt_from_parent(adapt_using_child=True) has no caller, so a
    # value above 1 was accepted, filed in three identity tables and then
    # ignored -- the run adapted parent-first and said nothing.
    (dict(adaptation_domain=2), "not wired to the driver"),
    (dict(min_time_step=60, min_time_step_den=0,
          max_time_step=30, max_time_step_den=0),
     "exceeds max_time_step"),
    (dict(starting_time_step=90, starting_time_step_den=0,
          max_time_step=60, max_time_step_den=0),
     "exceeds max_time_step"),
    (dict(starting_time_step=5, starting_time_step_den=0,
          min_time_step=10, min_time_step_den=0),
     "is below min_time_step"),
    (dict(min_time_step=10, min_time_step_den=-1), "must be >= 0"),
    (dict(use_adaptive_time_step=True, step_to_output_time=False),
     "needs step_to_output_time = true"),
])
def test_incoherent_combinations_are_refused(over, fragment):
    with pytest.raises(ValueError, match=fragment):
        validate_run_config(_cfg(**over))


def test_the_min_over_max_refusal_names_upstreams_silent_behaviour():
    """A refusal that only says 'invalid' teaches nothing.

    WRF does not error on min > max: it clamps max first and min second,
    so the minimum wins and the model runs ABOVE the maximum it was told
    to respect.  The message has to say that, because the user's next
    question is 'why does WRF accept it'.
    """
    with pytest.raises(ValueError) as excinfo:
        validate_run_config(_cfg(min_time_step=60, max_time_step=30))
    text = str(excinfo.value)
    assert "adapt_timestep_em.F" in text
    assert "the minimum wins" in text


# -------------------------------------------------- coherent configs pass

@pytest.mark.parametrize("over", [
    {},
    dict(use_adaptive_time_step=True),
    dict(use_adaptive_time_step=True, target_cfl=2.0, target_hcfl=1.0),
    dict(use_adaptive_time_step=True, min_time_step=10, max_time_step=60,
         starting_time_step=30),
    dict(max_step_increase_pct=51),                 # WRF's nest guidance
    dict(min_time_step=1, min_time_step_den=2),     # 0.5 s, rational form
])
def test_coherent_configurations_are_accepted(over):
    assert validate_run_config(_cfg(**over)) is not None


def test_rational_pairs_resolve_the_way_wrf_reads_them():
    """`_den = 0` means whole seconds; `-1` with `_den = 0` means unset."""
    from gpuwm.config import _adaptive_interval
    assert _adaptive_interval(-1, 0, "min_time_step") is None
    assert _adaptive_interval(30, 0, "min_time_step") == Fraction(30)
    assert _adaptive_interval(1, 2, "min_time_step") == Fraction(1, 2)
