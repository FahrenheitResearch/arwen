"""F1: what ``[render]`` may promise, on the interpreters wheels exist for.

MEASURED against the index, 2026-08-17 -- ``pypi.org/pypi/wrf-rust/0.2.39``
publishes 26 artifacts::

    wrf_rust-0.2.39-cp310-cp310-{macosx_10_12_x86_64,macosx_11_0_arm64,
        manylinux_2_17_aarch64,manylinux_2_17_x86_64,win_amd64}.whl
    ... cp311 ... cp312 ... cp313 ... cp314 (the same five each)
    wrf_rust-0.2.39.tar.gz

cp314 wheels on ALL FIVE platforms, so the gap this file used to pin --
0.2.38 publishing no cp314 wheel, pip falling back to an sdist whose pyo3
caps at 3.13, and the whole ``pip install 'gpuwm[gpu-cu13,render]'``
resolution dying with it -- is closed at the source.  What replaces the
environment marker is the FLOOR: the extra resolves 0.2.39 or newer, which
is the oldest release with a wheel for every interpreter gpuwm supports.

Two things are pinned here, and the second is the reason the file survives
its own fix:

* the 0.2.39 reality -- no marker, an install floor that names why it is
  where it is, and no shipped sentence claiming a 3.14 gap that no longer
  exists;
* the MECHANISM that named the gap.  A ceiling is a fact about the index,
  and the index moves: 0.2.39 publishes nothing for 3.15.  So doctor still
  reports a marker-excluded science core as ``excluded`` with a named
  reason rather than ``missing`` with a pip line that installs nothing --
  exercised here against a hypothetical future ceiling, because the whole
  cost of the 2026-08-17 outage was that this path did not exist.
"""

from __future__ import annotations

import pathlib
import re
import sys
import tomllib

import pytest

from gpuwm import doctor, science_core

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _render_requirements() -> list[str]:
    return _pyproject()["project"]["optional-dependencies"]["render"]


# --------------------------------------------------------------------------
# The packaging half.
# --------------------------------------------------------------------------

def test_the_science_core_records_the_python_ceiling_the_index_has_wheels_for():
    """The ceiling is an availability fact, and the fact moved to 3.14."""
    assert science_core.SCIENCE_CORE_PYTHON_CEILING == (3, 14)
    assert science_core.python_supports_science_core((3, 13, 7))
    assert science_core.python_supports_science_core((3, 14, 4))
    # 0.2.39 publishes cp310..cp314 and nothing above it.
    assert not science_core.python_supports_science_core((3, 15, 0))


def test_the_stale_gap_constant_is_gone():
    """A hard-coded sentence about 3.14 is now a false sentence.

    Deleted rather than reworded: the sentence has to follow the ceiling,
    and a constant cannot.  :func:`science_core.python_gap_sentence` is
    the replacement and every door reads it.
    """
    assert not hasattr(science_core, "SCIENCE_CORE_PYTHON_GAP")
    assert "SCIENCE_CORE_PYTHON_GAP" not in science_core.__all__


def test_the_gap_sentence_is_derived_from_the_ceiling_not_written_down():
    """The mechanism, on a box above whatever the ceiling is."""
    sentence = science_core.python_gap_sentence((3, 15, 0))
    assert "3.15" in sentence, sentence
    assert "3.14" in sentence, sentence
    assert science_core.SCIENCE_CORE_DISTRIBUTION in sentence, sentence


def test_the_render_extra_carries_no_python_marker():
    """THE remedy.  0.2.39 has wheels everywhere, so nothing is excluded.

    A marker left behind after upstream closed the gap is worse than the
    gap: pip silently skips a package that installs perfectly, and the
    reader gets `gpuwm render --engine matplotlib` failing on a box whose
    install line said it was covered.
    """
    requirements = _render_requirements()
    wrf = [item for item in requirements if item.startswith("wrf-rust")]
    assert len(wrf) == 1, requirements
    assert ";" not in wrf[0], (
        f"{wrf[0]!r} still carries an environment marker; wrf-rust 0.2.39 "
        "publishes cp310-cp314 wheels on all five platforms, so there is no "
        "interpreter left for pip to skip it on")


def test_the_install_floor_is_the_oldest_release_with_wheels_everywhere():
    """0.2.39, and the extra installs exactly that window."""
    assert science_core.SCIENCE_CORE_INSTALL_FLOOR == "0.2.39"
    assert science_core.SCIENCE_CORE_REQUIREMENT == "wrf-rust>=0.2.39,<0.3"
    wrf = [item for item in _render_requirements()
           if item.startswith("wrf-rust")][0]
    assert wrf.strip() == science_core.SCIENCE_CORE_REQUIREMENT


def test_the_runtime_window_still_accepts_the_cores_that_run_here():
    """The install floor is not a licence to refuse a working core.

    pip resolving 0.2.38 on a 3.14 box is what took the install down; a
    0.2.38 that is ALREADY installed on 3.13 renders every product it ever
    did.  Two axes, each naming its own breakage -- raising the runtime
    floor to 0.2.39 would be a refusal with nothing behind it.
    """
    assert science_core.SCIENCE_CORE_FLOOR == "0.2.35"
    assert science_core.version_supported("0.2.35")
    assert science_core.version_supported("0.2.38")
    assert science_core.version_supported("0.2.39")
    assert not science_core.version_supported("0.2.34")


def test_the_certified_release_is_the_one_the_suites_ran_on():
    """Re-recorded, not guessed: the counts belong to 0.2.39."""
    assert science_core.SCIENCE_CORE_CERTIFIED == "0.2.39"
    assert science_core.version_supported(science_core.SCIENCE_CORE_CERTIFIED)


def test_the_one_liners_still_reach_the_renderer():
    """``[all-cu12]``/``[all-cu13]`` keep naming it."""
    extras = _pyproject()["project"]["optional-dependencies"]
    for name in ("all-cu12", "all-cu13"):
        assert "gpuwm[render]" in extras[name], extras[name]


# --------------------------------------------------------------------------
# The doctor half: the ordinary gap is back, and the mechanism stays.
# --------------------------------------------------------------------------

_RENDER_LINE = 'wrf-rust>=0.2.39,<0.3; extra == "render"'

#: The same requirement as a FUTURE ceiling would file it.  Not a
#: prediction: the shape doctor has to keep handling, exercised now so the
#: next wheel-matrix gap is a named line instead of a dead pip command.
_FUTURE_LINE = ('wrf-rust>=0.2.39,<0.3; python_version < "3.15" '
                'and extra == "render"')


def _render_extra_check(monkeypatch, *, python, installed, line=_RENDER_LINE):
    """The ``pip extra [render]`` line as an interpreter would see it."""
    import importlib.metadata

    monkeypatch.setattr(doctor, "_python_version", lambda: python)
    monkeypatch.setattr(
        doctor, "_import_probe",
        lambda module, distribution=None: (installed, "0.2.39" if installed
                                           else "not installed"))

    def _version(name):
        if installed:
            return "0.2.39"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", _version)
    requirement = doctor._parse_requirement(line)
    return doctor._extra_check("render", (requirement,), {})


def test_the_unmarked_requirement_survives_the_round_trip():
    """doctor reads ``Requires-Dist``: the specifier lands, no marker left."""
    requirement = doctor._parse_requirement(_RENDER_LINE)
    assert requirement.distribution == "wrf-rust"
    assert requirement.specifier == "wrf-rust>=0.2.39,<0.3"
    assert requirement.marker == ""


def test_doctor_hands_a_3_14_box_the_pip_line_that_now_works(monkeypatch):
    """THE correction.  pip installs it on 3.14, so doctor says so.

    Reporting 3.14 as "not installable on this interpreter" after upstream
    shipped cp314 wheels sends the reader away from a command that would
    have fixed their box in one line.
    """
    check = _render_extra_check(monkeypatch, python=(3, 14, 4), installed=False)
    assert check.status == "missing"
    assert check.action == "pip install 'gpuwm[render]'"
    assert "not installable on this interpreter" not in check.detail


def test_doctor_still_reports_the_ordinary_gap_below_the_ceiling(monkeypatch):
    """On 3.13 nothing changed: absent means absent, and pip fixes it."""
    check = _render_extra_check(monkeypatch, python=(3, 13, 7), installed=False)
    assert check.status == "missing"
    assert check.action == "pip install 'gpuwm[render]'"


def test_doctor_names_a_future_ceiling_instead_of_a_dead_pip_line(monkeypatch):
    """THE MECHANISM, kept.  A marker-excluded core is named, not commanded.

    The 2026-08-17 outage cost what it cost because this path did not
    exist: doctor reported ``wrf-rust ... not installed`` with ``pip
    install 'gpuwm[render]'`` on an interpreter where that command
    installs nothing, reports success, and leaves doctor saying MISSING
    forever.  The wheel matrix will move again; the path stays.
    """
    monkeypatch.setattr(science_core, "SCIENCE_CORE_PYTHON_CEILING", (3, 14))
    check = _render_extra_check(monkeypatch, python=(3, 15, 0),
                                installed=False, line=_FUTURE_LINE)
    assert check.status != "missing", (
        "a requirement this interpreter is EXCLUDED from is not a gap the "
        "reader can close; it is a gap the project has to name")
    assert science_core.python_gap_sentence((3, 15, 0)) in check.detail
    assert "gpuwm[render]" not in (check.action or "")
    assert not check.blocking


def test_an_excluded_requirement_with_no_recorded_reason_prints_its_marker(
        monkeypatch):
    """An unnamed exclusion surfaces; it never hides behind a blank line."""
    check = _render_extra_check(
        monkeypatch, python=(3, 15, 0), installed=False,
        line='pyshp>=2.3; python_version < "3.15" and extra == "render"')
    assert "python_version" in check.detail
    assert "has not recorded why" in check.detail


def test_a_box_above_the_ceiling_that_has_the_core_is_reported_as_working(
        monkeypatch):
    """A source build, or a newer upstream: a marker is not a refusal."""
    monkeypatch.setattr(science_core, "SCIENCE_CORE_PYTHON_CEILING", (3, 14))
    check = _render_extra_check(monkeypatch, python=(3, 15, 0),
                                installed=True, line=_FUTURE_LINE)
    assert check.status == "verified"


def test_the_matplotlib_fallback_note_is_an_ordinary_gap_on_3_14(monkeypatch):
    """3.14 is inside the ceiling now, so the note prints the pip route."""
    monkeypatch.setattr(doctor, "_python_version", lambda: (3, 14, 4))
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (False, "x"))
    note = doctor._matplotlib_engine_note()
    assert "[render]" in note
    assert "pip cannot supply" not in note


def test_the_matplotlib_fallback_note_names_a_future_ceiling(monkeypatch):
    """The mechanism's second door, above whatever the ceiling is."""
    monkeypatch.setattr(science_core, "SCIENCE_CORE_PYTHON_CEILING", (3, 14))
    monkeypatch.setattr(doctor, "_python_version", lambda: (3, 15, 0))
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (False, "x"))
    note = doctor._matplotlib_engine_note()
    assert science_core.python_gap_sentence((3, 15, 0)) in note


def test_the_python_version_helper_reports_this_interpreter():
    """The instrument, against the known answer."""
    assert doctor._python_version()[:2] == sys.version_info[:2]


# --------------------------------------------------------------------------
# The documentation half.  A closed gap named in shipped prose is a reader
# skipping an install line that would have worked.
# --------------------------------------------------------------------------

@pytest.mark.parametrize("relative", ("README.md", "docs/install.md"))
def test_the_shipped_docs_no_longer_claim_a_3_14_gap(relative):
    text = (REPO_ROOT / relative).read_text(encoding="utf-8")
    assert "render extra needs Python <= 3.13" not in text, (
        f"{relative} still tells a 3.14 reader the render extra installs "
        "nothing; wrf-rust 0.2.39 publishes cp314 wheels")
    assert not re.search(r'python_version\s*<\s*["\']3\.14["\']', text), (
        f"{relative} still documents the environment marker")


def test_docs_record_the_release_that_closed_it():
    """The reader is told which release they need, not just that it exists."""
    text = (REPO_ROOT / "docs" / "install.md").read_text(encoding="utf-8")
    assert science_core.SCIENCE_CORE_INSTALL_FLOOR in text


def test_the_gap_sentence_is_written_in_exactly_one_place():
    """Two spellings of one gap is how a reader learns to trust neither."""
    source = (REPO_ROOT / "gpuwm" / "doctor.py").read_text(encoding="utf-8")
    assert "publishes no wheel for Python" not in source, (
        "doctor.py spells the gap sentence itself; call "
        "science_core.python_gap_sentence() instead")
