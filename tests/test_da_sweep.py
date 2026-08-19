"""The sweep's two load-bearing claims, checked mechanically.

1. The neighborhood is a square SIDE LENGTH, not a radius.  Every
   comparison in ``docs/da-vs-wofs.md`` against published WoFS skill
   turns on this, because the WoFS literature uses both conventions and
   they differ by a factor of two in linear scale.

2. The scorer uses the gallery's constants, not its own.  A sweep that
   scored itself on a different threshold or a different box than the
   figures would produce numbers that look comparable and are not.

3. There is exactly ONE scorer of record -- ``tools/da_sweep_score.py``
   -- and any other scorer is a caller of it.  Two scorers with two
   copies of the same constants agreed at 3 km and only at 3 km; the
   copy meant a different metric on any other grid while reporting it
   under the same name.  The last block below holds the unification.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SWEEP = REPO / "evidence" / "da-demo" / "sweep"


def test_neighborhood_is_a_side_length_not_a_radius():
    """27 km at 3 km spacing must be 9 cells across, not 19."""

    from tools.da_sweep_score import metric_constants

    const, _ = metric_constants()
    half_width = max(1, round(const["FSS_BOX_KM"] / 2.0 / 3.0))
    assert half_width == 4
    assert 2 * half_width + 1 == 9, "the box is 9 cells across"
    assert (2 * half_width + 1) * 3.0 == 27.0, "which is 27 km ACROSS"
    # If it were a radius, the box would span 2*27/3+1 = 19 cells.
    assert 2 * half_width + 1 != 19


def test_scorer_takes_its_constants_from_the_renderer():
    from tools import da_nowcast_render as render
    from tools.da_sweep_score import metric_constants

    const, source = metric_constants()
    assert source == "tools.da_nowcast_render"
    assert const["FSS_BOX_KM"] == render.FSS_BOX_KM
    assert const["FSS_THRESHOLD_DBZ"] == render.FSS_THRESHOLD_DBZ
    assert const["MISSING_OBS_FILL_DBZ"] == render.MISSING_OBS_FILL_DBZ
    assert const["COLUMN_THRESHOLD_DBZ"] == render.COLUMN_THRESHOLD_DBZ


# ---------------------------------------------------------------------------
# One scorer of record; the other is a caller
# ---------------------------------------------------------------------------

def _ensemble_scorer():
    """Import ``tools/ens_sweep/score_free_forecast.py`` by path.

    It is not in a package -- its own siblings import it by bare name
    after cd'ing into the directory -- so a dotted import cannot reach
    it and the test loads the file the shell scripts actually run.
    """

    import importlib.util

    path = REPO / "tools" / "ens_sweep" / "score_free_forecast.py"
    spec = importlib.util.spec_from_file_location("_ens_scorer", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ens_scorer"] = module
    spec.loader.exec_module(module)
    return module


def test_the_ensemble_scorer_imports_the_record_constants_rather_than_copying():
    """The unification, at the level that actually prevents divergence."""

    from tools.da_sweep_score import metric_constants

    const, _ = metric_constants()
    scorer = _ensemble_scorer()
    assert scorer.THRESHOLD_DBZ == const["FSS_THRESHOLD_DBZ"]
    assert scorer.COLUMN_THRESHOLD_DBZ == const["COLUMN_THRESHOLD_DBZ"]
    assert scorer.MISSING_OBS_FILL_DBZ == const["MISSING_OBS_FILL_DBZ"]
    assert scorer.FSS_BOX_KM == const["FSS_BOX_KM"]


def test_the_ensemble_scorer_declares_no_metric_constant_of_its_own():
    """A hard-coded VALUE here is the divergence.

    Read the assignments, not the text: the module's own comments
    quote the literals they replaced, and a substring search would
    match the explanation of the bug as though it were the bug.
    """

    import ast

    source = (REPO / "tools" / "ens_sweep" / "score_free_forecast.py"
              ).read_text(encoding="utf-8")
    metric_names = {"THRESHOLD_DBZ", "COLUMN_THRESHOLD_DBZ",
                    "MISSING_OBS_FILL_DBZ", "FSS_BOX_KM", "HALF_WIDTH_CELLS"}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in metric_names:
                assert not isinstance(node.value, ast.Constant), (
                    f"{target.id} is assigned a literal here; it must be "
                    "derived from tools.da_sweep_score so the two scorers "
                    "cannot drift apart")


def test_the_ensemble_scorer_derives_its_neighborhood_from_the_spacing():
    """4 cells is the 27 km box at 3 km, and is wrong at any other dx."""

    from tools.da_sweep_score import half_width_cells, metric_constants

    const, _ = metric_constants()
    scorer = _ensemble_scorer()
    assert scorer.HALF_WIDTH_CELLS == half_width_cells(scorer.DX_KM, const)
    assert scorer.HALF_WIDTH_CELLS == 4, "unchanged at the 3 km it runs on"
    # The derivation is the point: a finer grid needs more cells for the
    # same physical box, which a hard-coded 4 would silently get wrong.
    assert half_width_cells(1.5, const) > 4
    assert half_width_cells(6.0, const) < 4


def test_fss_of_a_field_against_itself_is_one():
    from gpuwm.verify.field_metrics import fss_distance

    rng = np.random.default_rng(20260805)
    field = rng.normal(20.0, 12.0, size=(40, 40))
    assert fss_distance(field, field, threshold=30.0, half_width=4) == 0.0


def test_fss_is_symmetric_in_its_two_arguments():
    """Both fields are smoothed; this is the fractional-truth family."""

    from gpuwm.verify.field_metrics import fss_distance

    rng = np.random.default_rng(7)
    left = rng.normal(20.0, 12.0, size=(48, 48))
    right = rng.normal(20.0, 12.0, size=(48, 48))
    a = fss_distance(left, right, threshold=30.0, half_width=4)
    b = fss_distance(right, left, threshold=30.0, half_width=4)
    assert a == pytest.approx(b)


def test_plan_arms_vary_exactly_one_flag_within_family_a():
    """The ensemble-size answer is only a measurement if nothing else moves."""

    plan = json.loads((SWEEP / "sweep-plan.json").read_text(encoding="utf-8"))
    cycles = {}
    for arm in plan["arms"]:
        if not arm["family"].startswith("A"):
            continue
        argv = next(s["argv"] for s in arm["steps"] if s["name"] == "cycle")
        # Normalise away the two things that legitimately differ: the
        # member count itself and the per-arm output directories.
        norm = []
        skip = False
        for index, token in enumerate(argv):
            if skip:
                skip = False
                continue
            if token in ("--members", "--out", "--stage-dir"):
                skip = True
                continue
            norm.append(token)
        cycles[arm["name"]] = norm

    assert len(cycles) == 2, "family A should be exactly two arms"
    (first, left), (second, right) = cycles.items()
    assert left == right, (
        f"{first} and {second} differ in more than --members; the "
        "ensemble-size comparison would be confounded")


#: The plan's spelling for the prepared case it runs on, and the
#: variable that binds it.  Read from the validator rather than retyped:
#: a token these tests spelled themselves would let the plan and the
#: validator agree on one word while this file asserted about another.
def _case_root_contract() -> tuple[str, str]:
    sys.path.insert(0, str(SWEEP))
    try:
        import validate_sweep_plan as validator

        return validator.CASE_ROOT_TOKEN, validator.CASE_ROOT_ENV
    finally:
        sys.path.remove(str(SWEEP))


def _bind_case_root(root: Path) -> dict:
    """Materialize every ``${CASE_ROOT}`` input the plan names, and no more.

    Built FROM the plan, never from a list typed here: a fixture that
    enumerated the case tree by hand would keep passing after the plan
    started naming a seventh observation file, which is the exact class
    of drift this validator exists to catch.

    Prepared cases are host-bound and none is committed, so what stands
    in for one is a tree with the right SHAPE -- the validator's question
    is existence, and answering it with an empty file is answering the
    question it actually asks.
    """

    token, variable = _case_root_contract()
    plan = json.loads((SWEEP / "sweep-plan.json").read_text(encoding="utf-8"))

    # Directory-valued flags and file-valued flags, split as the tools
    # themselves read them.
    directories = {"--prepared-root", "--authority-dir", "--obs-dir",
                   "--composites"}
    files = {"--obs", "--grid-wrfout", "--domain-polygon"}
    wanted: list[tuple[str, str]] = [
        (str(root / needed), "dir")
        for needed in ("go/prepared", "go/authority", "obs", "obsverify")]
    for arm in plan["arms"]:
        for step in arm["steps"]:
            argv = step["argv"]
            for flag, value in zip(argv, argv[1:]):
                if not value.startswith(token):
                    continue
                kind = "dir" if flag in directories else (
                    "file" if flag in files else None)
                assert kind is not None, (
                    f"{flag} names a {token} path and this fixture does not "
                    "know whether it is a file or a directory")
                wanted.append((value.replace(token, str(root)), kind))

    assert wanted, "the plan names no case-root inputs; nothing to bind"
    for path_text, kind in wanted:
        path = Path(path_text)
        if kind == "dir":
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()
    return {variable: str(root)}


def _validate(env_overrides: dict | None = None):
    environment = dict(os.environ)
    environment.pop("GPUWM_DA_SWEEP_CASE_ROOT", None)
    environment.update(env_overrides or {})
    return subprocess.run(
        [sys.executable, str(SWEEP / "validate_sweep_plan.py")],
        cwd=str(REPO), capture_output=True, text=True, env=environment)


def test_plan_validates_against_the_real_parsers(tmp_path):
    """The staged plan must still parse; a drifted flag fails here first.

    Bound to a case tree built from the plan's own declarations, so the
    only thing this can fail on is the plan: a flag the real parser no
    longer accepts, an argv the real entry point refuses, or an input
    the plan names and the case does not carry.
    """

    proc = _validate(_bind_case_root(tmp_path / "case"))
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_plan_refuses_when_no_box_has_bound_a_case_root():
    """A template is refused, by name, before any card is waited for.

    The plan cannot carry the case root: prepared cases are host-bound
    and the one these argv were recorded from was an ephemeral
    scratchpad that has since been reaped.  Unbound, the queue would
    hand ``tools.da_cycle_prepared`` the literal token as a prepared
    root -- inside a GPU slot it waited hours for -- so the refusal
    belongs here, and it has to name the variable that fixes it.
    """

    token, variable = _case_root_contract()
    proc = _validate()
    assert proc.returncode != 0
    message = proc.stdout + proc.stderr
    assert token in message
    assert variable in message
    assert "go/prepared" in message  # what the reader must point it at


def test_a_bound_case_root_missing_one_input_is_still_refused(tmp_path):
    """The existence check keeps its teeth once the token resolves.

    The instrument is tested in both directions on purpose: a validator
    that went green on an unbound token would have been "fixed" into
    saying nothing at all.
    """

    root = tmp_path / "case"
    overrides = _bind_case_root(root)
    victim = sorted((root / "obs").glob("*"))
    assert victim, "the plan names no observation files under obs/"
    victim[0].unlink()

    proc = _validate(overrides)
    assert proc.returncode != 0
    message = proc.stdout + proc.stderr
    assert "REFUSED" in message
    assert victim[0].name in message


def test_a_nest_composite_is_not_scored_as_an_extra_member(tmp_path):
    r"""The nest is a view of a member, never an extra member.

    tools/da_cycle_prepared.py writes nest composites into the SAME
    composites/ directory as the parent members, named
    leg{NN}_{name}_d{GG}.npz.  The scorer's name pattern is
    ^leg(\d+)_(.+)\.npz$, which happily reads "leg00_3_d02" as a member
    called "3_d02".  Averaging a fine-grid field into the ensemble mean
    moves every FSS number in the sweep and raises nothing.

    This only became reachable when the nested-forecast lane and the sweep
    landed in the same tree, and the committed live-fire-3 fixtures predate
    the nest, so nothing else in this file would catch it.
    """

    from tools.da_sweep_score import member_names

    composites = tmp_path / "composites"
    composites.mkdir()
    for name in ("0", "1", "2", "control"):
        (composites / f"leg00_{name}.npz").write_bytes(b"")
    # The same three members, as seen by the nest.
    for name in ("0", "1", "2"):
        (composites / f"leg00_{name}_d02.npz").write_bytes(b"")

    assert member_names(composites, 0) == ["0", "1", "2"]


def test_the_nest_filter_does_not_eat_ordinary_member_names(tmp_path):
    """Guard the guard: a member may legitimately end in a digit."""

    from tools.da_sweep_score import member_names

    composites = tmp_path / "composites"
    composites.mkdir()
    for name in ("0", "10", "d2", "memberd02x"):
        (composites / f"leg03_{name}.npz").write_bytes(b"")

    assert member_names(composites, 3) == ["0", "10", "d2", "memberd02x"]
