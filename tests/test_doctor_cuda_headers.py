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
    assert "conda install -c conda-forge cuda-toolkit" in check.remedy
    assert "CUDA_PATH" in check.remedy
    assert check.action.startswith("conda install -c conda-forge cuda-toolkit")


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


def test_a_cuda13_box_is_never_told_to_install_a_tombstone_wheel(monkeypatch):
    """The shadow trap, in the fallback half of the headers remedy.

    The pip alternative names a CUDA runtime wheel.  At CUDA 13 the
    ``-cu13`` spelling of that wheel is a deprecation tombstone that
    installs cleanly and supplies nothing.
    """
    check = _check(monkeypatch, _HEADERS_MISSING, box_major=13)
    assert "nvidia-cuda-runtime-cu13" not in check.remedy
    assert "nvidia-cuda-runtime-cu12" not in check.remedy
    assert "pip install --no-deps nvidia-cuda-runtime" in check.remedy


def test_a_cuda12_box_gets_the_suffixed_runtime_wheel(monkeypatch):
    """Negative control: the suffix is correct at 12."""
    check = _check(monkeypatch, _HEADERS_MISSING, box_major=12)
    assert "pip install nvidia-cuda-runtime-cu12" in check.remedy
    assert "--no-deps" not in check.remedy


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
    assert "conda install -c conda-forge cuda-toolkit=13" in check.remedy
    assert check.action == "conda install -c conda-forge cuda-toolkit=13"


def test_the_check_is_in_the_default_estate(monkeypatch):
    """A check behind a flag is a check nobody runs."""
    monkeypatch.setattr(doctor, "_nvrtc_header_probe", lambda: _BOTH_OK)
    names = [c.name for c in doctor.collect_checks(sources=())]
    assert "CUDA kernel headers" in names


def test_no_doctor_remedy_recommends_a_cuda13_tombstone_wheel(monkeypatch):
    """The class sweep for (c): audit EVERY remedy, not just the two fixed.

    On a CUDA-13 box no remedy doctor can assemble may name a ``-cu13``
    NVIDIA library wheel, because every one of them is a tombstone.
    """
    monkeypatch.setattr(doctor, "_driver_cuda_major", lambda: 13)
    offenders = []
    for check in doctor.collect_checks(sources=()):
        for text in (check.remedy or "", check.action or ""):
            for token in text.split():
                if token.startswith("nvidia-") and token.endswith("-cu13"):
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
