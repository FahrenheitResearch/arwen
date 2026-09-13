"""One rule, one sentence, every door: the offline child's root forcing.

THE DEFECT THIS PINS.  "the run's root takes specified boundaries" was
spelled TWICE, in two different wordings -- ``gpuwm/offline_child_run.py``
said "child config must set specified=true and nested=false" at admission,
``gpuwm/offline_child.py`` said "standalone offline child requires
specified=True and nested=False" in the state builder -- and fired at
NEITHER plan-review door.  ``gpuwm downscale --dry-run`` never read
``cfg.nested`` at all, so the earliest a reviewer could learn that a
``nested = true`` child config was unusable was after the real run had
started, and the second copy landed only after the whole parent archive
had been interpolated.  Neither sentence named a breakage.

Now one function owns the rule, all three doors call it, and it fires at
plan review.

WHY TWO IMPORTS SIT INSIDE TEST BODIES BELOW.  The proof this file exists
to carry is that the FIRST test is red on the base commit for its own
reason: ``--dry-run`` exits 0 and says nothing.  Importing
``OFFLINE_CHILD_ROOT_FORCING_REFUSAL`` or
``require_offline_child_root_forcing`` at module level errors the whole
file at COLLECTION on that commit, which reports as one collection error
and proves nothing about the defect.  So the names this change adds are
imported by the tests that need them, and the front-door test imports
only what the base already has.  Do not lift them back to the top.
"""

from datetime import datetime, timedelta

import pytest

from gpuwm.cli import main as cli_main
from gpuwm.downscale import _derive_child_run_config, _render_child_toml
from gpuwm.offline_child import (
    OfflineChildContractError,
    build_offline_child_domain_state,
)
from test_downscale_cli import _PARENT_CONFIG
from test_offline_child import _history


def _parent_archive(tmp_path):
    start = datetime(1974, 4, 3, 12)
    for index in range(3):
        _history(tmp_path / f"wrfout_d03_1974-04-03_{12 + index:02d}_00_00",
                 start + timedelta(hours=index), ny=18, nx=20)
    namelist = tmp_path / "namelist.input"
    namelist.write_text("&physics\n mp_physics = 8,\n/\n", encoding="utf-8")
    return namelist


def _child_toml(tmp_path, *, replace=None):
    parent = {"nx": 20, "ny": 18, "dx": 1000.0, "dy": 1000.0}
    merged = _derive_child_run_config(
        _PARENT_CONFIG, parent=parent, ratio=1, child_nx=12, child_ny=10,
        run_seconds=600.0, output_interval_s=300.0)
    text = _render_child_toml(merged)
    if replace is not None:
        old, new = replace
        assert old in text
        text = text.replace(old, new)
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "child.toml"
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _args(tmp_path, namelist, child_config, outdir):
    return [
        "downscale", str(tmp_path), "--parent-domain", "3",
        "--parent-namelist", str(namelist),
        "--child-config", str(child_config), "--ratio", "1",
        "--i-parent-start", "4", "--j-parent-start", "4",
        "--accept-parent-cadence", "--out", str(outdir), "--dry-run"]


class _Cfg:
    """The two fields the rule reads, and nothing else."""

    def __init__(self, *, specified, nested):
        self.specified = specified
        self.nested = nested


def test_a_nested_root_is_refused_at_plan_review(tmp_path, capsys):
    """RED ON THE BASE: --dry-run exited 0 and said nothing.

    The refusal has to reach the reviewer before the run, and it has to
    name what breaks: no Davies forcing, so the child drifts off the
    archive that is its only forcing source.
    """

    namelist = _parent_archive(tmp_path)
    child = _child_toml(tmp_path / "nested",
                        replace=("nested = false", "nested = true"))
    outdir = tmp_path / "child-run"
    assert cli_main(_args(tmp_path, namelist, child, outdir)) != 0
    err = capsys.readouterr().err
    assert "specified" in err
    assert "Davies" in err
    assert "drift" in err
    # A refusal that still hands --out back.
    assert not outdir.exists()


def test_the_three_doors_speak_one_sentence(tmp_path, capsys):
    """One function, three doors, byte-identical text."""

    from gpuwm.offline_child import (
        OFFLINE_CHILD_ROOT_FORCING_REFUSAL,
        require_offline_child_root_forcing,
    )

    # The function that owns the rule.
    with pytest.raises(OfflineChildContractError) as owner:
        require_offline_child_root_forcing(
            _Cfg(specified=True, nested=True))
    assert str(owner.value) == OFFLINE_CHILD_ROOT_FORCING_REFUSAL

    # The deep floor, where the interpolated parent state is uploaded.
    with pytest.raises(OfflineChildContractError) as builder:
        build_offline_child_domain_state(
            None, _Cfg(specified=True, nested=True))
    assert str(builder.value) == OFFLINE_CHILD_ROOT_FORCING_REFUSAL

    # Plan review, which did not have the rule at all.
    namelist = _parent_archive(tmp_path)
    child = _child_toml(tmp_path / "nested",
                        replace=("nested = false", "nested = true"))
    assert cli_main(
        _args(tmp_path, namelist, child, tmp_path / "child-run")) != 0
    assert OFFLINE_CHILD_ROOT_FORCING_REFUSAL in capsys.readouterr().err


def test_no_door_carries_a_second_wording_of_the_rule():
    """The census rule 4 asks for: the retired spellings are gone.

    The admission door (``gpuwm.offline_child_run._run``) is the third
    caller.  It cannot be executed on a CPU-only runner -- its first
    statement imports ``cupy`` -- so what is asserted here is the thing
    that made the two doors disagree in the first place: a second wording
    living in a second file.
    """
    import inspect
    from pathlib import Path

    import gpuwm.downscale
    import gpuwm.offline_child
    import gpuwm.offline_child_run

    retired = (
        "child config must set specified=true and nested=false",
        "standalone offline child requires specified=True and nested=False",
    )
    root = Path(gpuwm.offline_child.__file__).parent
    for path in sorted(root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for sentence in retired:
            assert sentence not in text, f"{path} still spells the rule"

    for module in (gpuwm.downscale, gpuwm.offline_child,
                   gpuwm.offline_child_run):
        source = inspect.getsource(module)
        assert "require_offline_child_root_forcing(" in source


def test_the_ordinary_child_still_plans_and_an_unspecified_one_refuses(
        tmp_path, capsys):
    """NEGATIVE CONTROLS, green on both sides of the fix."""

    from gpuwm.offline_child import OFFLINE_CHILD_ROOT_FORCING_REFUSAL

    namelist = _parent_archive(tmp_path)
    ordinary = _child_toml(tmp_path / "ordinary")
    assert cli_main(
        _args(tmp_path, namelist, ordinary, tmp_path / "ordinary-run")) == 0

    unspecified = _child_toml(
        tmp_path / "unspecified",
        replace=("specified = true", "specified = false"))
    capsys.readouterr()
    assert cli_main(
        _args(tmp_path, namelist, unspecified,
              tmp_path / "unspecified-run")) != 0
    assert OFFLINE_CHILD_ROOT_FORCING_REFUSAL in capsys.readouterr().err
