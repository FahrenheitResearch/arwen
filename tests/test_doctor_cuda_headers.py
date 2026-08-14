"""``gpuwm doctor``'s verdict on whether this box can COMPILE a CUDA kernel.

This line exists because the gap it reports is silent by construction and
the advice that used to cover it was actively wrong.

Silent: cupy imports, cuBLAS loads, a matmul returns the right answer, and
every cheaper probe therefore passes on a box whose next uncached reduction
will not build.  A warm kernel cache hides it for weeks -- which is exactly
how it was found, on a box that had moved to a CUDA-13 toolkit and kept
serving cubins compiled under 12.

Wrong: the remedy on offer was ``pip install 'gpuwm[gpu-cu13]'``.  NVRTC --
the compiler -- ships INSIDE the CuPy wheel, and no CuPy wheel has ever
carried a CUDA header tree.  So that command reinstalls the piece that is
already present and supplies nothing that is missing, while pip reports
success.  A buyer on a fresh box follows it and watches the fault survive.

Every branch is forced here rather than left to whatever this box has,
because on a healthy box only one of them is reachable and the other three
are the ones a reader in trouble will actually see.
"""

from __future__ import annotations

import pathlib
import re
import subprocess

import pytest

from gpuwm import doctor


def _check(monkeypatch, payload, *, box_major=13):
    """The check with the probe and the box's CUDA major both forced."""
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.setattr(doctor, "_nvrtc_header_probe", lambda: payload)
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: box_major)
    return doctor._cuda_headers_check()


_HEADERS_MISSING = {
    "self_contained": "ok",
    "toolkit_headers": "CompileException: cuda_fp8.hpp: No such file",
    "cuda_path": "",
}
_WHEEL_BROKEN = {
    "self_contained": "ImportError: libnvrtc.so.12: cannot open",
    "toolkit_headers": "ImportError: libnvrtc.so.12: cannot open",
    "cuda_path": "/usr/local/cuda",
}
_BOTH_OK = {
    "self_contained": "ok", "toolkit_headers": "ok",
    "cuda_path": "/usr/local/cuda",
}


# --------------------------------------------------------------------------
# The probe itself: it must COMPILE, and it must compile COLD.
# --------------------------------------------------------------------------

def test_the_probe_compiles_rather_than_importing():
    """Nothing cheaper than a compile detects this fault."""
    source = doctor._NVRTC_HEADER_PROBE
    # A self-contained kernel, to exercise NVRTC on its own...
    assert "RawModule" in source
    assert "__global__" in source
    # ...and a cupy reduction, which is what drags in the toolkit headers.
    assert "cupy.arange" in source and "sum()" in source
    # And it checks the ANSWER, not merely that the call returned.
    assert "2016" in source


def test_the_probe_runs_against_a_cold_kernel_cache():
    """A warm cache is precisely what hid this fault in the field."""
    import inspect
    source = inspect.getsource(doctor._nvrtc_header_probe)
    assert "CUPY_CACHE_DIR" in source
    assert "TemporaryDirectory" in source


# --------------------------------------------------------------------------
# The scenarios.
# --------------------------------------------------------------------------

def test_both_kernels_compiling_is_verified_and_prints_no_remedy(monkeypatch):
    check = _check(monkeypatch, _BOTH_OK)
    assert check.status == "verified"
    assert check.remedy is None and check.action is None
    assert not check.blocking


@pytest.mark.parametrize("box_major", (12, 13, None))
def test_headers_missing_is_named_as_headers_and_never_as_a_wheel(
        monkeypatch, box_major):
    """THE correction: a header gap must not be answered with a wheel.

    ``pip install 'gpuwm[gpu-cu13]'`` reinstalls the compiler that is
    already installed.  It must not appear on this branch at any major.
    """
    check = _check(monkeypatch, _HEADERS_MISSING, box_major=box_major)
    assert check.status == "missing"
    assert check.brief == "toolkit headers missing"
    assert not check.blocking
    # It says which piece is missing, in those words.
    assert "NVRTC works" in check.detail
    assert "header" in check.detail.lower()
    # The wheel remedy is absent from BOTH the remedy and the one command.
    assert "gpuwm[gpu-cu12]" not in check.remedy
    assert "gpuwm[gpu-cu13]" not in check.remedy
    assert "gpuwm[gpu-" not in (check.action or "")
    # The real remedy: an external toolkit, and CUDA_PATH pointed at it.
    # The `nvidia` channel leads because that is the line that fixed a real
    # 2.2.1 box; conda-forge survives as the named equivalent.
    assert "conda install -c nvidia cuda-toolkit" in check.remedy
    assert "conda install -c conda-forge cuda-toolkit" in check.remedy
    assert "CUDA_PATH" in check.remedy
    assert check.action.startswith("conda install -c nvidia cuda-toolkit")


@pytest.mark.parametrize("box_major", (12, 13))
def test_the_toolkit_remedy_is_pinned_to_the_boxs_cuda_major(
        monkeypatch, box_major):
    """A CUDA-12 box must not be handed a CUDA-13 toolkit, or the reverse."""
    check = _check(monkeypatch, _HEADERS_MISSING, box_major=box_major)
    assert f"cuda-toolkit={box_major}" in check.remedy
    other = 12 if box_major == 13 else 13
    assert f"cuda-toolkit={other}" not in check.remedy


def test_an_unreadable_major_tells_the_reader_to_look_it_up(monkeypatch):
    """A silent default is how a CUDA-13 box ends up on a cu12 toolkit."""
    check = _check(monkeypatch, _HEADERS_MISSING, box_major=None)
    assert "nvidia-smi" in check.remedy
    assert "cuda-toolkit=12" not in check.remedy
    assert "cuda-toolkit=13" not in check.remedy


@pytest.mark.parametrize("box_major", (12, 13))
def test_no_box_is_ever_told_to_install_a_tombstone_wheel(monkeypatch,
                                                          box_major):
    """The shadow trap, in the fallback half of the headers remedy.

    The pip alternative names the CUDA runtime and NVRTC wheels.  NVIDIA
    has deprecated BOTH suffixed spellings of those, and a deprecation
    tombstone installs cleanly and supplies nothing.
    """
    check = _check(monkeypatch, _HEADERS_MISSING, box_major=box_major)
    for name in ("nvidia-cuda-runtime", "nvidia-cuda-nvrtc"):
        assert f"{name}-cu13" not in check.remedy
        assert f"{name}-cu12" not in check.remedy
        # The unsuffixed name, pinned to the major the box actually serves.
        assert f'"{name}=={box_major}.*"' in check.remedy
    assert "pip install --no-deps " in check.remedy
    other = 12 if box_major == 13 else 13
    assert f"=={other}.*" not in check.remedy


def test_a_broken_wheel_is_named_as_the_wheel_and_gets_the_wheel_remedy(
        monkeypatch):
    """The other half of the distinction the check exists to draw."""
    check = _check(monkeypatch, _WHEEL_BROKEN, box_major=13)
    assert check.status == "missing"
    assert check.brief == "nvrtc unusable"
    assert "wheel rather than the toolkit headers" in check.detail
    # THIS branch is the one a wheel install genuinely fixes.
    assert "gpuwm[gpu-cu13]" in check.remedy
    assert "conda install" not in check.remedy


# --------------------------------------------------------------------------
# The branches that must not judge.
# --------------------------------------------------------------------------

def test_the_local_gpu_switch_stops_the_check_touching_the_device(
        monkeypatch):
    """Compiling is device contact, and the switch means what it says."""
    monkeypatch.setenv("GPUWM_NO_LOCAL_GPU", "1")

    def _must_not_run():
        raise AssertionError("probe ran under GPUWM_NO_LOCAL_GPU")

    monkeypatch.setattr(doctor, "_nvrtc_header_probe", _must_not_run)
    check = doctor._cuda_headers_check()
    assert check.status == "info"
    assert not check.blocking
    assert "not judged" in check.detail


@pytest.mark.parametrize("payload,expected", [
    ({"cupy": "not installed"}, "needs cupy"),
    ({"devices": 0}, "no device"),
    # A slow first compile is not a missing toolkit.  Reporting it as one
    # would send a reader to install headers they already have.
    ({"slow": "no answer within 180 s"}, "compile timed out"),
])
def test_a_box_that_cannot_be_judged_says_so_rather_than_passing(
        monkeypatch, payload, expected):
    """``info``, honestly.  A box with no device must not fail doctor."""
    check = _check(monkeypatch, payload)
    assert check.status == "info"
    assert check.brief == expected
    assert not check.blocking
    assert check.remedy is None


def test_a_probe_that_would_not_run_still_prints_the_real_remedy(monkeypatch):
    """The branch whose remedy/action pair is assembled by unpacking."""
    check = _check(monkeypatch, {"probe": "did not run: timeout"},
                   box_major=13)
    assert check.status == "missing"
    assert not check.blocking
    assert "conda install -c nvidia cuda-toolkit=13" in check.remedy
    assert check.action == "conda install -c nvidia cuda-toolkit=13"


def test_the_check_is_in_the_default_estate(monkeypatch):
    """A check behind a flag is a check nobody runs."""
    monkeypatch.setattr(doctor, "_nvrtc_header_probe", lambda: _BOTH_OK)
    names = [c.name for c in doctor.collect_checks(sources=())]
    assert "CUDA kernel headers" in names


@pytest.mark.parametrize("box_major", (12, 13, None))
def test_no_doctor_remedy_recommends_a_tombstone_wheel(monkeypatch, box_major):
    """The class sweep for (c): audit EVERY remedy, not just the two fixed.

    No remedy doctor can assemble, at any detected major, may name a
    SUFFIXED NVIDIA package: ``-cu13`` was a tombstone from the start and
    NVIDIA has since deprecated ``-cu12`` the same way, so both spellings
    now install cleanly and supply nothing.  ``gpuwm[gpu-cu12]`` is
    deliberately NOT caught here -- that is this project's own extra, it
    resolves, and it installs the CuPy wheel it names.
    """
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: box_major)
    offenders = []
    for check in doctor.collect_checks(sources=()):
        for text in (check.remedy or "", check.action or ""):
            for token in text.replace('"', " ").split():
                token = token.strip("'\"(),")
                if not token.startswith("nvidia-"):
                    continue
                if token.endswith("-cu13") or token.endswith("-cu12"):
                    offenders.append((check.name, token))
    assert not offenders, offenders


# --------------------------------------------------------------------------
# Shell correctness, on BOTH platforms.
#
# The estate-wide sweep in test_doctor.py cannot reach these strings: this
# check returns no remedy at all under GPUWM_NO_LOCAL_GPU, which is how the
# suite runs.  So the remedy's own contract is asserted here, against the
# same helper, for the shell the remedy was generated for AND the other one.
# --------------------------------------------------------------------------

from test_doctor import (  # noqa: E402
    _assert_remedy_lines_are_commands_or_comments,
    _force_shell,
)


@pytest.mark.parametrize("box_major", (12, 13, None))
@pytest.mark.parametrize("windows", (False, True))
def test_the_headers_remedy_is_shell_correct_on_both_platforms(
        monkeypatch, windows, box_major):
    _force_shell(monkeypatch, windows)
    remedy, action = doctor._cuda_headers_remedy(box_major)
    _assert_remedy_lines_are_commands_or_comments(remedy, windows=windows)
    _assert_remedy_lines_are_commands_or_comments(action, windows=windows)
    # The CUDA_PATH line is spelled for the shell it was generated for.
    if windows:
        assert "$env:CUDA_PATH" in remedy
        assert "export CUDA_PATH" not in remedy
    else:
        assert "export CUDA_PATH" in remedy
        assert "$env:" not in remedy


@pytest.mark.parametrize("box_major", (12, 13, None))
@pytest.mark.parametrize("windows", (False, True))
def test_the_cusolver_remedy_is_shell_correct_on_both_platforms(
        monkeypatch, windows, box_major):
    _force_shell(monkeypatch, windows)
    remedy, action = doctor._cusolver_hint(box_major)
    _assert_remedy_lines_are_commands_or_comments(remedy, windows=windows)


# --------------------------------------------------------------------------
# The cupy LINE's own remedy, which is where a fresh box meets this fault
# first.  A 2.2.1 user on Ubuntu got
#
#     MISSING  cupy (GPU runtime)  ... Failed to find CUDA headers ...
#     remedy:  pip install 'gpuwm[gpu-cu13]'
#
# The wheel was already installed.  What was missing was the toolkit, which
# no gpuwm extra has ever carried, so pip reported success and the fault
# survived.  The CUDA-kernel-headers line above cannot cover this case: it
# declines to judge when cupy will not import, which is exactly when this
# one speaks.
# --------------------------------------------------------------------------

def _cupy_check_with(monkeypatch, evidence, *, box_major=13):
    monkeypatch.delenv("GPUWM_NO_LOCAL_GPU", raising=False)
    monkeypatch.setattr(doctor, "_import_probe",
                        lambda module, distribution=None: (False, evidence))
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: box_major)
    return doctor._cupy_check()


_FIELD_EVIDENCE = ("installed but failed to import: RuntimeError: CuPy is "
                   "not correctly installed. Failed to find CUDA headers")


@pytest.mark.parametrize("box_major", (12, 13, None))
def test_the_field_case_gets_the_toolkit_and_not_the_gpuwm_extra(
        monkeypatch, box_major):
    """THE 2.2.1 correction, on the line the user actually read."""
    check = _cupy_check_with(monkeypatch, _FIELD_EVIDENCE,
                             box_major=box_major)
    assert check.status == "missing"
    assert "conda install -c nvidia cuda-toolkit" in check.remedy
    assert check.action.startswith("conda install -c nvidia cuda-toolkit")
    # The advice that could not work is gone from this branch entirely.
    assert "gpuwm[gpu-" not in check.remedy
    assert "gpuwm[gpu-" not in check.action


def test_a_missing_shared_library_still_gets_the_wheel_remedy(monkeypatch):
    """The other half of the distinction: this one IS a wheel fault."""
    check = _cupy_check_with(
        monkeypatch,
        "installed but failed to import: ImportError: libcublas.so.12: "
        "cannot open shared object file")
    assert "gpuwm[gpu-cu13]" in check.remedy
    assert "conda install" not in check.remedy


def test_an_absent_cupy_still_gets_the_wheel_extra(monkeypatch):
    """The regression guard on the branch that was always right."""
    check = _cupy_check_with(monkeypatch, "not installed")
    assert check.action == "pip install 'gpuwm[gpu-cu13]'"
    assert "conda install" not in check.remedy


def test_an_unrecognised_import_failure_prints_both_labelled_by_symptom(
        monkeypatch):
    """No signature in the message is not licence to guess one."""
    check = _cupy_check_with(
        monkeypatch, "installed but failed to import: RuntimeError: boom")
    assert "SYMPTOM" in check.remedy
    assert "conda install -c nvidia cuda-toolkit" in check.remedy
    assert "gpuwm[gpu-cu13]" in check.remedy


@pytest.mark.parametrize("windows", (False, True))
@pytest.mark.parametrize("evidence", (
    _FIELD_EVIDENCE,
    "installed but failed to import: RuntimeError: boom",
))
def test_the_cupy_import_remedy_is_shell_correct_on_both_platforms(
        monkeypatch, windows, evidence):
    """Including the two-remedy block, which is assembled by concatenation."""
    _force_shell(monkeypatch, windows)
    remedy, action = doctor._cupy_import_failure_remedy(evidence, 13)
    _assert_remedy_lines_are_commands_or_comments(remedy, windows=windows)
    _assert_remedy_lines_are_commands_or_comments(action, windows=windows)


# --------------------------------------------------------------------------
# The shipped documentation, which no remedy guard could ever see.
# --------------------------------------------------------------------------
#
# The guards above grep doctor's REMEDY STRINGS.  They cannot see a doc, and
# 2.3.0 shipped its doctor fix beside a public quickstart still telling a
# CUDA-12 reader to `pip install nvidia-cusolver-cu12 ...` -- the exact
# tombstone the release removed from the code.  A user follows the doc, pip
# reports success, and the fault survives: "fixed" that a default reader
# still hits is not fixed.  So the rule extends over docs/.
#
# The rule is about INSTRUCTIONS, not mentions: a doc may name a tombstone to
# warn against it (the quickstart does), but no line that installs may name
# one.

_TOMBSTONE_NAMES = re.compile(r"nvidia-[a-z0-9-]+-cu1[23]")


def _repo_root():
    """Derived from this file, never an absolute path baked in at write time.

    A path pinned to the tree the test was written in grades that tree from
    wherever it is copied -- the wrong-tree failure the battery's PYTHONPATH
    discipline exists to prevent.
    """
    return pathlib.Path(__file__).resolve().parent.parent


def _tracked_docs():
    root = _repo_root()
    out = subprocess.run(
        ["git", "ls-files", "-z", "docs"],
        cwd=root, capture_output=True, check=True)
    return [root / name
            for name in out.stdout.decode("utf-8").split("\0") if name]


def test_the_docs_tree_is_tracked_and_non_empty():
    """A guard over an empty list passes forever and proves nothing."""
    docs = _tracked_docs()
    assert len(docs) > 5, docs
    assert any(d.name == "da-nowcast-quickstart.md" for d in docs)


def test_no_shipped_doc_tells_a_reader_to_install_a_tombstone_wheel():
    offenders = []
    for path in _tracked_docs():
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if "pip install" not in line:
                continue
            for hit in _TOMBSTONE_NAMES.findall(line):
                offenders.append(
                    "{}:{}: {}".format(
                        path.relative_to(_repo_root()).as_posix(),
                        lineno, hit))
    assert not offenders, (
        "shipped docs install a deprecation tombstone:\n"
        + "\n".join(offenders))


def test_the_docs_guard_would_catch_the_line_that_shipped():
    """The instrument, tested against the known answer, both directions."""
    shipped = ("pip install nvidia-cusolver-cu12 nvidia-cublas-cu12"
               " nvidia-cusparse-cu12")
    assert "pip install" in shipped
    assert _TOMBSTONE_NAMES.findall(shipped) == [
        "nvidia-cusolver-cu12", "nvidia-cublas-cu12", "nvidia-cusparse-cu12"]
    # And the spelling that replaced it is clean.
    fixed = ('pip install --no-deps "nvidia-cusolver==12.*"'
             ' "nvidia-cublas==12.*" "nvidia-cusparse==12.*"')
    assert _TOMBSTONE_NAMES.findall(fixed) == []
    # A prose mention carries no install and must NOT be flagged.
    prose = "`nvidia-cusolver-cu12` resolves as a deprecation tombstone."
    assert "pip install" not in prose
