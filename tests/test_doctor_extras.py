"""``gpuwm doctor``'s extras block: every extra, both directions.

A 2026-08-14 reachability audit installed the published wheel into a
clean venv, ran ``gpuwm doctor --explain``, and measured 40,351
characters of report in which the strings ``scipy``, ``pyshp``,
``shapefile``, ``rasterio`` and ``pyproj`` appeared **zero times**.
Doctor probed cupy, wrf and matplotlib and nothing else on the Python
side, so four extras were invisible to the one command whose whole job
is telling a user what their install cannot do.  The same report
labelled ``[render]`` "wrf-rust + matplotlib" -- matplotlib is a base
dependency and the extra's second package is pyshp -- and exited 0 with
"0 of them blocking" on a box that could not run a forecast.

Every test here is written to fail in BOTH directions: it fires when the
thing is missing and stays silent when it is present.  A check that
cannot fail is worse than no check, and this program has had five
instruments give confident wrong answers in one night.
"""

from __future__ import annotations

import importlib.metadata
from pathlib import Path
import sys
import tomllib

import pytest

from gpuwm import doctor


REPO_ROOT = Path(__file__).resolve().parent.parent


def _pyproject_extras() -> dict[str, list[str]]:
    """The checkout's declared extras -- for TESTS only.

    Doctor itself must never read this file (that is the whole point of
    :func:`gpuwm.doctor.declared_requirements`), but a test may, and
    this is what lets one assert the hand-written tables in doctor keep
    up with the packaging.
    """

    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)["project"]["optional-dependencies"]


def _requirement_name(text: str) -> str:
    match = doctor._REQUIREMENT_NAME.match(text)
    assert match is not None, text
    return doctor._canonical(match.group(1))


def _fake_metadata(monkeypatch, requires, extras):
    """Point doctor at a fabricated INSTALLED distribution.

    The point of the indirection is the audit's own instrument: doctor
    must read the extras from installed metadata, so a test that wants
    to control what doctor sees has to control the metadata, not the
    checkout's pyproject.
    """

    class _Metadata:
        @staticmethod
        def get_all(field):
            return list(extras) if field == "Provides-Extra" else []

    monkeypatch.setattr(importlib.metadata, "requires",
                        lambda name: list(requires))
    monkeypatch.setattr(importlib.metadata, "metadata",
                        lambda name: _Metadata())


def _by_name(checks):
    return {check.name: check for check in checks}


# ---------------------------------------------------------------------------
# H14: every extra gets a check, and the report says its packages out loud
# ---------------------------------------------------------------------------

def test_every_extra_the_install_declares_gets_its_own_check():
    """No extra may be invisible.  This is the audit's headline row."""

    declared = doctor.declared_requirements()
    assert declared is not None, "this test needs an installed gpuwm"
    _base, extras = declared
    reported = _by_name(doctor._extras_checks())
    for extra in extras:
        assert f"pip extra [{extra}]" in reported, extra


def test_the_report_names_the_packages_the_audit_could_not_find(monkeypatch):
    """scipy, pyshp, rasterio, pyproj -- zero mentions before this.

    Measured on the shipped wheel: `gpuwm doctor --explain` contained
    none of these four strings.  They are the whole of `[obs]`,
    `[dealias]`, half of `[render]` and (through 2.3.2) `[geog]`.
    """

    _fake_metadata(
        monkeypatch,
        requires=["numpy>=2.0",
                  'wrf-rust>=0.2.35,<0.3; extra == "render"',
                  'pyshp>=2.3; extra == "render"',
                  'scipy>=1.11; extra == "obs"',
                  'scipy>=1.11; extra == "dealias"'],
        extras=["render", "obs", "dealias"])
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (False,
                                                           "not installed"))
    monkeypatch.setattr(importlib.metadata, "version",
                        _raise_not_found)
    text = doctor.format_report(doctor._extras_checks())
    for token in ("scipy", "pyshp", "wrf-rust",
                  "pip install 'gpuwm[obs]'", "pip install 'gpuwm[dealias]'",
                  "pip install 'gpuwm[render]'"):
        assert token in text, token


def _raise_not_found(name):
    raise importlib.metadata.PackageNotFoundError(name)


def test_an_extra_fires_when_absent_and_is_silent_when_present(monkeypatch):
    """BOTH directions on one extra, in one test, so neither can rot."""

    requires = ['scipy>=1.11; extra == "obs"']
    _fake_metadata(monkeypatch, requires=requires, extras=["obs"])

    # Direction 1: absent.  A gap, with the exact install line.
    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (False,
                                                           "not installed"))
    absent = _by_name(doctor._extras_checks())["pip extra [obs]"]
    assert absent.status == "missing"
    assert absent.action == "pip install 'gpuwm[obs]'"
    assert "tools.obs_battery_score" in absent.detail

    # Direction 2: present.  Not a gap, and no install line to run.
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "1.14.0")
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (True, "1.14.0"))
    present = _by_name(doctor._extras_checks())["pip extra [obs]"]
    assert present.status == "verified"
    assert present.action is None
    assert doctor.blocking_gaps([present]) == []


def test_an_extra_doctor_has_never_been_taught_is_reported_loudly(
        monkeypatch):
    """A new extra must not slip through as silence.

    The failure this closes is not hypothetical -- it is exactly how
    `[obs]` and `[dealias]` came to be shipped, documented nowhere and
    probed by nothing.
    """

    _fake_metadata(monkeypatch,
                   requires=['somepkg>=1; extra == "brand-new"'],
                   extras=["brand-new"])
    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
    check = _by_name(doctor._extras_checks())["pip extra [brand-new]"]
    assert check.status == "missing"
    assert "NOT RECORDED" in check.detail
    assert check.action == "pip install 'gpuwm[brand-new]'"


# ---------------------------------------------------------------------------
# H23: the label has to be true
# ---------------------------------------------------------------------------

def test_the_render_extra_never_claims_matplotlib(monkeypatch):
    """`[render]` is wrf-rust + pyshp.  matplotlib is a base dependency.

    The old label read "render extra (wrf-rust + matplotlib)" in three
    places including the remedy, so a reader whose basemaps were dying
    on a missing shapefile reader was pointed at a package they already
    had, and never told about the one they did not.
    """

    # The false claim, verbatim as it shipped, must be gone -- and the
    # hint must say the true thing rather than merely omitting the
    # false one, because "matplotlib" legitimately appears in the name
    # of the engine the extra unlocks.
    assert "installs wrf-rust + matplotlib" not in doctor.RENDER_EXTRA_HINT
    assert "matplotlib is NOT in this extra" in doctor.RENDER_EXTRA_HINT
    assert "pyshp" in doctor.RENDER_EXTRA_HINT
    assert "wrf-rust" in doctor.RENDER_EXTRA_HINT

    _fake_metadata(monkeypatch,
                   requires=['wrf-rust>=0.2.35,<0.3; extra == "render"',
                             'pyshp>=2.3; extra == "render"'],
                   extras=["render"])
    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (False,
                                                           "not installed"))
    check = _by_name(doctor._extras_checks())["pip extra [render]"]
    assert "pyshp" in check.detail
    assert "matplotlib" not in check.brief
    assert "wrf-rust + matplotlib" not in check.detail


def test_matplotlib_is_reported_as_the_base_dependency_it_is(monkeypatch):
    _fake_metadata(monkeypatch,
                   requires=["numpy>=2.0", "matplotlib>=3.8"], extras=[])
    monkeypatch.setattr(importlib.metadata, "version", lambda name: "3.9.0")
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (True, "3.9.0"))
    checks = _by_name(doctor._extras_checks())
    base = checks["base dependencies (installed by `pip install gpuwm`)"]
    assert "matplotlib" in base.detail
    assert base.status == "verified"


def test_a_missing_base_dependency_is_broken_and_blocks(monkeypatch):
    _fake_metadata(monkeypatch,
                   requires=["numpy>=2.0", "matplotlib>=3.8"], extras=[])
    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (False,
                                                           "not installed"))
    base = _by_name(doctor._extras_checks())[
        "base dependencies (installed by `pip install gpuwm`)"]
    assert base.status == "missing" and base.blocking
    assert base.severity == doctor.SEVERITY_BROKEN


def test_pillow_is_reported_as_transitive_and_not_as_the_render_extra(
        monkeypatch):
    """`--pair`'s own refusal names the wrong package; doctor may not.

    `gpuwm render --pair` prints "needs Pillow (installed with the
    render extra)".  Pillow is not in `[render]`; it arrives with
    matplotlib, a base dependency, so a reader following that message
    installs an extra that cannot supply what they are missing.  Fixing
    the refusal itself belongs to the CLI lane; saying the true thing
    here does not.
    """

    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (True, "10.0.0"))
    check = doctor._transitive_dependency_check()
    assert "matplotlib" in check.detail
    assert "NOT with the [render] extra" in check.detail

    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (False,
                                                           "not installed"))
    absent = doctor._transitive_dependency_check()
    assert absent.status == "missing"
    assert "gpuwm render --pair" in absent.detail
    # Never the render extra: it cannot install Pillow.
    assert "gpuwm[render]" not in (absent.remedy or "")


# ---------------------------------------------------------------------------
# The metadata contract: installed distribution, never a checkout's pyproject
# ---------------------------------------------------------------------------

def test_extras_come_from_the_installed_metadata_not_this_checkout(
        monkeypatch):
    """The substitution this module exists to refuse.

    A checkout beside a wheel is the configuration where a
    transcription and the truth part company, so the test fabricates an
    extra that appears in NO pyproject and asserts doctor reports it.
    """

    _fake_metadata(monkeypatch,
                   requires=['pyshp>=2.3; extra == "invented-by-this-test"'],
                   extras=["invented-by-this-test"])
    monkeypatch.setattr(importlib.metadata, "version", _raise_not_found)
    reported = _by_name(doctor._extras_checks())
    assert "pip extra [invented-by-this-test]" in reported
    assert "invented-by-this-test" not in _pyproject_extras()


def test_no_installed_distribution_says_not_tested_never_ok(monkeypatch):
    """A source tree on PYTHONPATH declares no extras.  Say so."""

    def _no_distribution(name):
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "requires", _no_distribution)
    monkeypatch.setattr(importlib.metadata, "metadata", _no_distribution)
    checks = doctor._extras_checks()
    assert len(checks) == 1
    assert checks[0].status == "untested"
    assert checks[0].detail.startswith("not tested")


def test_a_shared_import_name_does_not_make_the_other_wheel_look_present(
        monkeypatch):
    """cupy-cuda12x and cupy-cuda13x both import as ``cupy``.

    An import probe therefore cannot tell them apart, and reporting
    both extras "verified" off one successful `import cupy` is the
    same class of false green as the pairing failure this module was
    built around.  Metadata first, import second.
    """

    _fake_metadata(monkeypatch,
                   requires=['cupy-cuda12x>=13.0; extra == "gpu-cu12"',
                             'cupy-cuda13x>=13.6; extra == "gpu-cu13"'],
                   extras=["gpu-cu12", "gpu-cu13"])
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (True, "14.0.1"))
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 13)

    def _only_cu13(name):
        if doctor._canonical(name) == "cupy-cuda13x":
            return "14.0.1"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _only_cu13)
    reported = _by_name(doctor._extras_checks())
    cu12 = reported["pip extra [gpu-cu12]"]
    cu13 = reported["pip extra [gpu-cu13]"]
    assert "cupy-cuda12x>=13.0 (imports as cupy) not installed" in cu12.detail
    assert "cupy-cuda13x>=13.6 (imports as cupy) installed" in cu13.detail
    # And neither is a gap: they are alternatives, so a correctly
    # installed box is missing one of them by definition.
    assert cu12.status == "info" and cu13.status == "info"
    # The CUDA major read off the driver is preserved and used.
    assert "THIS is the pair's matching extra" in cu13.detail


# ---------------------------------------------------------------------------
# The tables doctor declares by hand must keep up with the packaging
# ---------------------------------------------------------------------------

def test_every_declared_requirement_has_a_known_import_name():
    """A package doctor cannot import is a package doctor cannot judge.

    Without this, adding a dependency silently downgrades its check to
    "installed per metadata, not imported" and nobody notices.
    """

    declared = doctor.declared_requirements()
    assert declared is not None
    base, extras = declared
    everything = list(base) + [item for items in extras.values()
                               for item in items]
    unknown = sorted({item.distribution for item in everything
                      if not item.alias
                      and item.distribution not in doctor._IMPORT_NAME})
    assert not unknown, (
        f"teach gpuwm.doctor._IMPORT_NAME the import name(s) of: {unknown}")


def test_every_extra_the_packaging_declares_has_recorded_facts():
    """"What does this extra unlock" is not in any metadata.

    So it is hand-declared, and this is what stops the hand-declared
    half from falling behind the packaging half -- which is exactly how
    `[obs]` and `[dealias]` shipped with no documented install line.
    """

    missing = sorted(extra for extra, requirements in
                     _pyproject_extras().items()
                     if extra not in doctor._EXTRA_FACTS
                     and any(not _requirement_name(text).startswith("gpuwm")
                             for text in requirements))
    assert not missing, (
        f"teach gpuwm.doctor._EXTRA_FACTS what these unlock: {missing}")


def test_the_gpu_extras_doctor_names_are_extras_that_exist():
    """A remedy that names a nonexistent extra fails when pasted."""

    extras = _pyproject_extras()
    for extra in doctor._GPU_EXTRA_BY_MAJOR.values():
        assert extra in extras


# ---------------------------------------------------------------------------
# H19: the exit code has to mean something
# ---------------------------------------------------------------------------

def test_the_cupy_gap_blocks_and_the_geog_tree_does_not(monkeypatch):
    """The severity model, at both of its ends, in one test.

    Before 2.3.3 a bare install printed "3 gap(s), 0 of them blocking
    (exit 0)" with CuPy absent -- a green light over a box where
    `gpuwm run` dies in a raw ModuleNotFoundError after fetching
    gigabytes.  The fix must not swing the other way and start failing
    installers for the ~16 GB terrain download they deliberately
    skipped.
    """

    monkeypatch.setattr(doctor, "find_spec", lambda name: None)
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 12)
    cupy = doctor._cupy_check()
    assert cupy.status == "missing"
    assert cupy.blocking, "a box that cannot run a forecast is not green"
    assert cupy.severity == doctor.SEVERITY_UNREACHABLE
    assert doctor.blocking_gaps([cupy]) == [cupy]

    geog = doctor._geog_tree_checks(Path("nowhere-at-all"))
    assert [check.status for check in geog] == ["missing"]
    assert not geog[0].blocking, "an explicit opt-in must not fail a box"
    assert geog[0].severity == doctor.SEVERITY_OPT_IN
    assert doctor.blocking_gaps(geog) == []

    # And the summary says which is which, rather than one number for
    # two questions.
    text = doctor.format_brief([cupy, *geog])
    assert "1 of them blocking (exit 1)" in text
    assert "1 unreachable" in text and "1 opt-in" in text


def test_present_cupy_leaves_no_gap_at_all(monkeypatch):
    """The other direction: the check must go quiet when CuPy is there."""

    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (True, "14.0.1"))
    monkeypatch.setattr(doctor, "_installed_cupy_wheels",
                        lambda: [("cupy-cuda12x", 12)])
    monkeypatch.setattr(doctor, "_cublas_pairing_probe",
                        lambda: {"cublas": "ok", "driver": 12080,
                                 "wheel_runtime": 12060, "devices": 1})
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    check = doctor._cupy_check()
    assert check.status == "verified"
    assert doctor.blocking_gaps([check]) == []


def test_every_gap_carries_a_severity_and_the_exit_code_follows_it():
    """No finding may be unclassified, and the two must agree."""

    checks = doctor.collect_checks()
    for check in checks:
        if check.status == "missing":
            assert check.severity in (
                doctor.SEVERITY_BROKEN, doctor.SEVERITY_UNREACHABLE,
                doctor.SEVERITY_DEGRADED, doctor.SEVERITY_OPT_IN), check.name
            if check.severity in (doctor.SEVERITY_DEGRADED,
                                  doctor.SEVERITY_OPT_IN):
                assert not check.blocking, check.name
    census = doctor.severity_census(checks)
    assert sum(census.values()) == sum(
        1 for check in checks if check.status == "missing")


def test_a_legacy_check_without_a_severity_still_gets_one():
    """The translation, not a guess: `blocking` always meant this."""

    assert doctor.Check("x", "missing", "d").severity == doctor.SEVERITY_BROKEN
    assert doctor.Check("x", "missing", "d",
                        blocking=False).severity == doctor.SEVERITY_OPT_IN
    assert doctor.Check("x", "verified", "d").severity is None


def test_run_plan_probe_readiness_follows_the_same_verdict(monkeypatch):
    """`run-plan --probe` reported "ready": true on a box with no CuPy.

    It reads `blocking_gaps`, so it inherits the fix rather than
    needing its own -- which is the point of there being one severity
    rule.  Pinned here because the inheritance is what makes the fix
    reach that door at all.
    """

    from gpuwm import runplan

    cupy_gap = doctor.Check(
        "cupy (GPU runtime)", "missing", "not installed",
        severity=doctor.SEVERITY_UNREACHABLE)
    monkeypatch.setattr(doctor, "collect_checks", lambda: [cupy_gap])
    document = runplan.probe_environment(readiness=True)
    assert document["readiness"]["ready"] is False
    assert document["readiness"]["blocking_gaps"] == 1


if __name__ == "__main__":  # pragma: no cover - convenience
    sys.exit(pytest.main([__file__, "-q"]))
