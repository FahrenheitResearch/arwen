"""The capability gates, measured from outside.

``tests/conftest.py`` asks, once per session, whether the staged Rust
estate can do the thing the gated cases need, and skips those cases with
the command that stages it.  Two properties of that arrangement are not
visible from inside a normal run, and both were defects:

* a probe that RAISES takes collection down, and with it every test in
  the session that has nothing to do with the artifact.  ``GPUWM_RW_NETCDF``
  naming a path that no longer exists does exactly that:
  ``gpuwm.netcdf_bridge.find_netcdf_bin`` refuses a missing override on
  purpose, so a stale override collected nothing at all; and
* a gate written at function granularity skips the parametrizations that
  never open the artifact.  Three parametrized cases refuse their input in
  Python, before any bridge is reached, and those are the refusal tests --
  the class least worth switching off for want of a binary.

Both are measured here by running pytest in a subprocess, which is the
only way to see a collection-time verdict at all.
"""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = REPOSITORY_ROOT / "tests"

#: Overrides that name a path no file is at.  Each resolver refuses a
#: missing explicit override rather than falling through to another
#: artifact, so this is the supported way to say "this box has none".
NO_ESTATE = {
    "GPUWM_RW_NETCDF": str(REPOSITORY_ROOT / "no-such-rw-netcdf"),
    "GPUWM_CPU_PREPROCESS_BRIDGE": str(
        REPOSITORY_ROOT / "no-such-preprocess-library.so"),
}

#: The cases that refuse their input before the bridge's own operation is
#: reached, with the sibling in the same parametrization that does open
#: the artifact and the capability they are gated on.  Each row is one
#: function, so a gate that returns to function granularity fails here.
SPLIT_CASES = (
    ("eta",
     "tests/test_wrf_eta.py::"
     "test_automatic_eta_rejects_invalid_actual_algorithm_inputs",
     ("test_automatic_eta_rejects_invalid_actual_algorithm_inputs"
      "[auto_levels_opt-True-integer]",
      "test_automatic_eta_rejects_invalid_actual_algorithm_inputs"
      "[auto_levels_opt-3-1 or 2]"),
     "test_automatic_eta_rejects_invalid_actual_algorithm_inputs"
     "[max_dz-0-max_dz]"),
    ("netcdf",
     "tests/test_wrfinput_identity.py::"
     "test_malformed_file_identity_is_refused",
     ("test_malformed_file_identity_is_refused[midpoint-ZNU]",
      "test_malformed_file_identity_is_refused[missing_eta-ZNW]",
      "test_malformed_file_identity_is_refused[fractional_flag-finite "
      "integer]"),
     "test_malformed_file_identity_is_refused[time-Times]"),
    ("netcdf",
     "tests/test_analyzed_scalar_boundaries.py::"
     "test_supplied_aerosol_boundary_failures_are_not_dropped",
     ("test_supplied_aerosol_boundary_failures_are_not_dropped[identity]",),
     "test_supplied_aerosol_boundary_failures_are_not_dropped[missing]"),
)


def _normalised(text: str) -> str:
    return " ".join((text or "").split())


def _capability_gap(capability: str) -> str | None:
    """The conftest probe's own verdict for this box, unmodified."""

    from conftest import cpu_preprocess_gap, netcdf_bridge_gap

    if capability == "eta":
        return cpu_preprocess_gap("gpuwm_wrf_eta_f32")
    return netcdf_bridge_gap()


def _environment(overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ, **(overrides or {}))
    env["GPUWM_NO_LOCAL_GPU"] = "1"
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(REPOSITORY_ROOT), env.get("PYTHONPATH"))
        if part)
    return env


def _pytest(*arguments: str,
            overrides: dict[str, str] | None = None
            ) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "no:randomly",
         "-p", "no:cacheprovider", "--no-header", "-q", *arguments],
        cwd=REPOSITORY_ROOT, env=_environment(overrides), capture_output=True,
        text=True, timeout=900)


def _outcomes(report: Path) -> dict[str, tuple[str, str]]:
    """name -> (outcome, the skip reason or the empty string)."""

    cases: dict[str, tuple[str, str]] = {}
    for case in ET.parse(report).getroot().iter("testcase"):
        outcome, detail = "passed", ""
        for child in case:
            if child.tag in ("failure", "error"):
                outcome = "failed"
            elif child.tag == "skipped":
                outcome, detail = "skipped", child.get("message") or ""
        cases[case.get("name")] = (outcome, detail)
    return cases


def test_a_stale_bridge_override_still_collects_the_whole_suite(tmp_path):
    """An override naming a missing file is a gap, never a crash.

    Measured before this: ``GPUWM_RW_NETCDF=/nonexistent pytest
    tests/test_config.py`` raised FileNotFoundError out of conftest at
    import and collected NOTHING -- a supported environment variable
    going stale switched off every test on the box, including every one
    that has never heard of NetCDF.
    """

    report = tmp_path / "collected.xml"
    completed = _pytest("tests/test_config.py", "--collect-only",
                        f"--junitxml={report}", overrides=NO_ESTATE)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "FileNotFoundError" not in completed.stdout, completed.stdout
    assert "no-such-rw-netcdf" not in completed.stdout, completed.stdout
    assert report.exists()


@pytest.mark.parametrize(
    "capability,node,unaffected,gated", SPLIT_CASES,
    ids=[case[1].split("::")[-1] for case in SPLIT_CASES])
def test_a_gate_skips_only_the_cases_that_open_the_artifact(
        tmp_path, capability, node, unaffected, gated):
    """The gate is per case, not per function.

    Each of these functions parametrizes one refusal over several
    inputs.  Some are refused before the bridge's own operation is
    reached and pass against a staged estate too old for the rest; a
    function-level gate skipped those too, which is coverage lost to a
    binary the case never opens, and what it lost was refusal tests.
    Measured on the Linux CPU box with its ordinary staged estate: six
    such cases went from passing to skipped, and came back.

    The subprocess runs in THIS box's environment, unmodified, and the
    case recognises three estates:

    * the capability is PRESENT: the gate does not fire, there is no
      split to see, and the case skips itself;
    * the artifact is present but STALE: the gate fires, the unaffected
      refusal cases must run, and the artifact-opening control must be
      skipped by the gate.  This is the estate the gate exists for;
    * the artifact is ABSENT altogether (a fresh clone, or a wheel
      install that has not run ``tools/stage_wheel_bridges.py``): the
      file's own ``backend()``/``native()`` helper refuses the
      unaffected cases before the gate is the question, so the split is
      not observable and the case skips itself.

    The second and third are told apart by comparing the sibling's skip
    reason with the gate's own reason, which this process already holds
    as ``gap``: a skip that IS the gate's reason is the defect this case
    exists for, and a skip with any other reason is the file's helper.
    Guessing from the wording would not do: the helper's reason on the
    bare estate embeds the same staging command the gate's does, because
    both come from the bridge remedy text.
    """

    gap = _capability_gap(capability)
    if gap is None:
        pytest.skip(
            f"this box has the {capability} capability, so the gate does "
            "not fire and there is no split to see; run it on a box whose "
            "staged estate is older than the tree")

    report = tmp_path / "split.xml"
    _pytest(node, f"--junitxml={report}")
    cases = _outcomes(report)
    assert cases, "the subprocess collected nothing"

    for name in unaffected:
        outcome, detail = cases.get(name, ("absent", ""))
        assert outcome != "absent", f"{name} was not collected at all"
        if outcome == "skipped" and _normalised(detail) != _normalised(gap):
            pytest.skip(
                f"{name} is skipped by its own file's helper rather than "
                f"by the capability gate ({detail}), so the split is not "
                "observable here: with the artifact ABSENT the case does "
                "not run whether the gate fires or not. Run it on a box "
                "whose staged estate is present but older than the tree, "
                "which is the estate the gate exists for")
        assert outcome != "skipped", (
            f"{name} is refused before the bridge's operation is reached "
            f"and the gate skipped it anyway ({detail})")

    outcome, detail = cases.get(gated, ("absent", ""))
    assert outcome == "skipped", (
        f"{gated} opens the staged artifact and was {outcome} with the "
        "estate forced absent, so the control for this split is gone")
    assert _normalised(detail) == _normalised(gap), (
        f"the skip reason for {gated} is not the gate's own: {detail!r}")
    assert "cargo build" in detail or "stage_wheel_bridges" in detail, (
        f"the skip reason for {gated} names no way to stage it: {detail!r}")
