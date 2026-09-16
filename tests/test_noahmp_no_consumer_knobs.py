"""Noah-MP knobs that reach no gpuwm code are carried, not refused.

``gpuwm.config.NOAHMP_OPTION_IDENTITY_EVIDENCE`` pins every Noah-MP
option to one value and gives the evidence for each.  Three of its rows
read "declared only", and they mean something different from the rest:
the knob has NO consumer at any value.  ``opt_pedo`` selects a
pedotransfer branch that ``opt_soil=1`` makes unreachable;
``noahmp_output`` and ``noahmp_acc_dt`` drive WRF's ``module_diag_misc``
accumulator block, which gpuwm has no counterpart to.  Nothing reads
them, so no value of them can make a run wrong.

Refusing one therefore named no breakage: the reason in the table is
"pinned to keep it that way", which is about tidiness.  They are
admitted with one warning saying the knob reaches nothing -- the fact a
user setting it actually needs -- while the rows whose other value
selects code that was never transcribed stay refused, because that code
is MISSING and substituting a nearby branch for it is how a run becomes
silently wrong.

Both doors reach that verdict from the same table: the run door
(``validate_run_config``) and the namelist importer.
"""
from __future__ import annotations

from pathlib import Path
import sys

import pytest

from gpuwm.config import (NOAHMP_OPTIONS_WITHOUT_CONSUMER,
                          NOAHMP_OPTION_IDENTITY_EVIDENCE,
                          RunConfig, validate_run_config)

#: ``tests`` is not a package, so the sibling suite that owns the
#: namelist fixtures is reached by path the way conftest is.
TESTS_DIR = Path(__file__).resolve().parent


def _sibling(name):
    """A helper from a sibling suite, reached the way conftest is."""

    if str(TESTS_DIR) not in sys.path:
        sys.path.insert(0, str(TESTS_DIR))
    import importlib

    return importlib.import_module(name)


def _import_namelist_with(tmp_path, extra_input):
    """The real importer, on the shipped namelist pair plus one section."""

    return _sibling("test_namelist_import")._import_with(
        tmp_path, extra_input=extra_input)


def _noahmp_cfg(**kw) -> RunConfig:
    return RunConfig(nx=12, ny=12, nz=8, dx=3000.0, dy=3000.0,
                     ztop=12000.0, dt=6.0, run_seconds=60.0,
                     moist=True, sf_surface_physics=4,
                     sf_sfclay_physics=1, bl_pbl_physics=1, **kw)


def test_the_no_consumer_set_is_derived_from_the_evidence_it_cites():
    """The set and the prose cannot drift: one is built from the other."""

    assert NOAHMP_OPTIONS_WITHOUT_CONSUMER == {
        "opt_pedo", "noahmp_output", "noahmp_acc_dt"}
    for name in NOAHMP_OPTIONS_WITHOUT_CONSUMER:
        assert NOAHMP_OPTION_IDENTITY_EVIDENCE[name][1].startswith(
            "declared only")
    # CONTROL: a row whose other value selects untranscribed code is not
    # in the set, so this cannot pass by naming every row.
    assert "opt_crs" not in NOAHMP_OPTIONS_WITHOUT_CONSUMER
    assert "opt_sfc" not in NOAHMP_OPTIONS_WITHOUT_CONSUMER


@pytest.mark.parametrize("name,value", [("opt_pedo", 2),
                                        ("noahmp_output", 0),
                                        ("noahmp_acc_dt", 60.0)])
def test_a_knob_with_no_consumer_runs_and_says_it_reaches_nothing(
        name, value, capsys):
    cfg = _noahmp_cfg(**{name: value})
    validate_run_config(cfg)

    said = capsys.readouterr().err
    rows = [row for row in said.splitlines()
            if f"{name}=" in row and "reaches no gpuwm code" in row]
    assert len(rows) == 1, said
    assert "changes nothing this run does" in rows[0]
    admitted = NOAHMP_OPTION_IDENTITY_EVIDENCE[name][0]
    assert f"behaves as {name}={admitted!r}" in rows[0]


def test_the_admitted_value_of_a_no_consumer_knob_says_nothing(capsys):
    """The control: a configuration at the pin is silent, as before."""

    validate_run_config(_noahmp_cfg())
    said = capsys.readouterr().err
    assert "reaches no gpuwm code" not in said


def test_a_knob_whose_other_branch_was_never_transcribed_is_still_refused():
    """The refusal that stands, because the code it selects is MISSING."""

    with pytest.raises(ValueError, match="outside the admitted Noah-MP"):
        validate_run_config(_noahmp_cfg(opt_crs=2))
    with pytest.raises(ValueError, match="outside the admitted Noah-MP"):
        validate_run_config(_noahmp_cfg(opt_sfc=2))


def test_the_wrong_type_is_still_refused_and_names_the_type():
    """A float knob given a string is self-contradictory, not unmeasured."""

    with pytest.raises(ValueError, match="where Noah-MP takes a float"):
        validate_run_config(_noahmp_cfg(noahmp_acc_dt="60"))


def test_the_importer_agrees_with_the_run_door_about_the_same_knob(tmp_path):
    """Two doors, one table: a namelist setting one is imported.

    The importer refused the whole namelist over a knob that changes
    nothing, which is the same refusal in the other door.  It records
    the value instead, with the reason, and keeps refusing the rows
    whose branch is untranscribed.  The call is the real importer, so
    this fails if the branch behind the constant breaks.
    """
    _toml, report = _import_namelist_with(
        tmp_path, "&noah_mp\n opt_pedo = 2,\n/\n")
    fixed = {(row.section, row.key): row for row in report.fixed}
    row = fixed[("noah_mp", "opt_pedo")]
    assert row.values == (2,)
    assert row.fixed_value == NOAHMP_OPTION_IDENTITY_EVIDENCE["opt_pedo"][0]
    assert "reaches no gpuwm code" in row.reason
    assert "changes nothing" in row.reason


def test_the_importer_still_refuses_the_knob_whose_branch_is_missing(
        tmp_path):
    """The control for the case above, through the same door.

    ``opt_crs = 2`` selects CANRES+CALHUM, which was never transcribed,
    so the namelist is refused rather than reinterpreted.  Without this
    the test above would pass against an importer that accepted
    everything.
    """
    with pytest.raises(ValueError, match="opt_crs"):
        _import_namelist_with(tmp_path, "&noah_mp\n opt_crs = 2,\n/\n")


def test_the_importer_reads_the_verdict_from_the_config_table(tmp_path):
    """The table half: neither door keeps a second list of its own."""

    from gpuwm import namelist_import

    source = open(namelist_import.__file__, encoding="utf-8").read()
    assert "NOAHMP_OPTIONS_WITHOUT_CONSUMER" in source
    assert NOAHMP_OPTIONS_WITHOUT_CONSUMER, (
        "the set must not be empty or this proves nothing")


# ---------------------------------------------------------------------------
# the third door: the physics registry
# ---------------------------------------------------------------------------


def _noahmp_single_domain_plan():
    """The shipped Noah-MP profile as a one-domain plan."""

    module = _sibling("test_physics_registry")
    return module._single_plan(module.NOAHMP_PROFILE_ID)


@pytest.mark.parametrize("name,value", [("opt_pedo", 2),
                                        ("noahmp_output", 0),
                                        ("noahmp_acc_dt", 60.0)])
def test_plan_review_does_not_refuse_a_knob_with_no_consumer(name, value):
    """The third door, pinned to the other two.

    The registry carried a ``required_settings`` row per Noah-MP knob,
    so ``validate_physics_plan`` refused a value the run door admits,
    with "Set opt_pedo=1 for it, or select another land_surface
    option."  Two plan-review doors disagreeing about one configuration
    is the defect; the rows for these three came out of
    tools/build_registry.py and the registry was regenerated.
    """
    from gpuwm.physics_registry import validate_physics_plan

    plan = _noahmp_single_domain_plan()
    plan["domains"][0]["parameters"] = {name: value}
    report = validate_physics_plan(plan)
    offending = [error for error in report["errors"]
                 if error["code"] == "component-required-setting"
                 and name in error["message"]]
    assert offending == [], offending


def test_plan_review_still_refuses_the_knob_whose_branch_is_missing():
    """The control: the rows that stayed are still enforced.

    Without this the test above would pass against a registry that had
    lost every Noah-MP row rather than the three that reach no code.
    """
    from gpuwm.physics_registry import validate_physics_plan

    plan = _noahmp_single_domain_plan()
    plan["domains"][0]["parameters"] = {"opt_crs": 2}
    report = validate_physics_plan(plan)
    refusals = [error for error in report["errors"]
                if error["code"] == "component-required-setting"
                and "opt_crs" in error["message"]]
    assert len(refusals) == 1, report["errors"]
    assert report["launchable"] is False
