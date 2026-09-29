"""Bare-default preprocessing must reach the CPU backend without cupy.

The 2.5.0 persona walks measured the gap this file pins: the prep doors
defaulted ``--preprocess-backend`` to ``cuda``, so a CPU-only install
died mid-preparation in ``RuntimeError: CuPy is required for GPU
horizontal interpolation`` -- while ``--preprocess-backend cpu`` walked
the same route to completion, and ``gpuwm doctor --explain`` promised
that the whole preprocessing half runs without cupy.  A capability that
only a flag can reach is a workaround, not a fix (fixed-means-default).

The fixed contract:

* the three prep front doors default to ``auto``;
* ``auto`` resolves to the CPU backend when CUDA is unusable and prints
  ONE line saying so and why;
* ``cuda`` typed explicitly on an install with no cupy is a NAMED
  refusal at resolve time -- what is missing, the install line, and the
  flag that runs the same preparation on the CPU -- never a bare
  RuntimeError from the first kernel that went looking.
"""

from __future__ import annotations

import pytest

from gpuwm.ingest import preprocess_backend as backend_module
from gpuwm.ingest.preprocess_backend import resolve_preprocess_backend


@pytest.fixture()
def _fresh_announcement(monkeypatch):
    """Each test sees the one-per-process announcement fresh."""

    monkeypatch.setattr(backend_module, "_ANNOUNCED_AUTO_REASONS", set())


def _cupyless(monkeypatch):
    """Make the resolver's view of cupy an ImportError, both probes."""

    from gpuwm.ingest import horiz

    def _no_cupy():
        raise RuntimeError(
            "CuPy is required for GPU horizontal interpolation")

    monkeypatch.setattr(horiz, "_cupy", _no_cupy)
    monkeypatch.setattr(
        backend_module, "_gpu_runtime_installed", lambda: False)


def test_the_three_prep_doors_default_to_auto():
    """The real parsers, asked for the real default.

    ``cuda`` as the default is exactly the unreachable-CPU defect: a
    bare run on a CPU-only box selected a backend that cannot exist
    there.  ``auto`` is the only default that serves both installs.
    """

    from gpuwm import era5_direct, gfs_direct, mapped_direct

    for module in (mapped_direct, gfs_direct, era5_direct):
        parser = module._parser()
        assert parser.get_default("preprocess_backend") == "auto", \
            module.__name__


def test_the_hrrr_tree_preparer_defaults_to_auto_like_the_other_doors():
    """``tools/prepare_hrrr_wrf`` is the door ``gpuwm go`` and the source
    CLI drive for HRRR, and neither passes a backend.  Its ``cuda``
    default sent a bare HRRR preparation on a CPU-only install to the
    named cuda refusal, while the same preparation with
    ``--preprocess-backend cpu`` ran."""

    from tools import prepare_hrrr_wrf

    assert prepare_hrrr_wrf._parser().get_default(
        "preprocess_backend") == "auto"


def test_the_hrrr_root_receipt_keeps_why_auto_chose_the_cpu(
        monkeypatch, _fresh_announcement):
    """Pinning the CPU worker budget is not a second choice of backend.

    The HRRR root preparation re-resolves the CPU backend to fix one
    explicit worker total, and that re-resolution used to replace the
    receipt's selection with ``requested: cpu`` and "named by the
    caller" -- on a bare run where nobody named anything and auto had
    fallen to the CPU because cupy was absent.
    """

    import os
    from types import SimpleNamespace

    from tools import hrrr_single_domain_benchmark as benchmark

    class _Cpu:
        name = "cpu"

        def __init__(self, workers=None, bridge=None):
            self.workers = workers
            self.selection = None

    _cupyless(monkeypatch)
    monkeypatch.setattr(backend_module, "ParallelCpuPreprocessBackend", _Cpu)

    bare = SimpleNamespace(preprocess_backend="auto",
                           preprocess_workers=None, cpu_preprocess_bridge=None)
    chosen, workers = benchmark._budgeted_preprocess_backend(bare)
    assert workers == int(os.cpu_count() or 1)
    assert chosen.workers == workers
    assert chosen.selection["requested"] == "auto"
    assert chosen.selection["backend"] == "cpu"
    assert "cupy" in chosen.selection["reason"]

    named = SimpleNamespace(preprocess_backend="cpu",
                            preprocess_workers=None, cpu_preprocess_bridge=None)
    chosen, _workers = benchmark._budgeted_preprocess_backend(named)
    assert chosen.selection == {
        "requested": "cpu", "backend": "cpu",
        "reason": backend_module.NAMED_BY_CALLER}


def test_auto_without_cupy_resolves_cpu_and_says_so(
        monkeypatch, capsys, _fresh_announcement):
    from types import SimpleNamespace

    _cupyless(monkeypatch)
    cpu = SimpleNamespace(name="cpu")
    monkeypatch.setattr(
        backend_module, "ParallelCpuPreprocessBackend",
        lambda **_kwargs: cpu)
    assert resolve_preprocess_backend("auto") is cpu
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "cpu" in err.lower()
    assert "cupy" in err.lower()


def test_the_auto_line_prints_once_per_process(
        monkeypatch, capsys, _fresh_announcement):
    """nest initialization re-resolves per child; four lines is noise."""

    from types import SimpleNamespace

    _cupyless(monkeypatch)
    monkeypatch.setattr(
        backend_module, "ParallelCpuPreprocessBackend",
        lambda **_kwargs: SimpleNamespace(name="cpu"))
    resolve_preprocess_backend("auto")
    resolve_preprocess_backend("auto")
    resolve_preprocess_backend("auto")
    assert capsys.readouterr().err.count("\n") == 1


def test_cuda_without_cupy_is_a_named_refusal(monkeypatch,
                                              _fresh_announcement):
    """Explicitly requested GPU preprocessing keeps a refusal with the
    remedy in it -- and gets it at resolve time, before any bytes are
    decoded, rather than as a RuntimeError out of the first kernel."""

    _cupyless(monkeypatch)
    with pytest.raises(ValueError) as caught:
        resolve_preprocess_backend("cuda")
    text = str(caught.value)
    assert "cupy" in text
    assert "pip install" in text
    assert "--preprocess-backend cpu" in text


def test_cuda_with_cupy_present_stays_lazy(monkeypatch,
                                           _fresh_announcement):
    """The presence probe is the only new gate: with cupy resolvable the
    cuda branch returns the same lazy backend it always has, importing
    nothing at resolve time."""

    monkeypatch.setattr(
        backend_module, "_gpu_runtime_installed", lambda: True)
    assert resolve_preprocess_backend("cuda").name == "cuda"


def test_auto_with_unusable_runtime_names_the_runtime(
        monkeypatch, capsys, _fresh_announcement):
    """cupy installed but on a runtime with no certification row: the
    one line names the runtime it declined, not a missing install."""

    from types import SimpleNamespace

    class _Runtime:
        @staticmethod
        def runtimeGetVersion():
            return 14_000

        @staticmethod
        def getDeviceCount():
            return 1

    cuda = SimpleNamespace(
        name="cuda",
        array_module=SimpleNamespace(
            __version__="14.2.0",
            cuda=SimpleNamespace(runtime=_Runtime())))
    monkeypatch.setattr(
        backend_module, "CudaPreprocessBackend", lambda: cuda)
    cpu = SimpleNamespace(name="cpu")
    monkeypatch.setattr(
        backend_module, "ParallelCpuPreprocessBackend",
        lambda **_kwargs: cpu)
    assert resolve_preprocess_backend("auto") is cpu
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "14000" in err
    assert "CUDA 14" in err


def test_auto_with_cupy_installed_and_no_device_does_not_say_not_installed(
        monkeypatch, capsys, _fresh_announcement):
    """cupy installed, no device answering (CUDA_VISIBLE_DEVICES empty):
    the line said "cupy is not installed here", which sends a reader with
    a working cupy to reinstall it.  It names the device error instead."""

    from types import SimpleNamespace

    class _Runtime:
        @staticmethod
        def runtimeGetVersion():
            raise RuntimeError(
                "cudaErrorNoDevice: no CUDA-capable device is detected")

        @staticmethod
        def getDeviceCount():
            raise RuntimeError(
                "cudaErrorNoDevice: no CUDA-capable device is detected")

    cuda = SimpleNamespace(
        name="cuda",
        array_module=SimpleNamespace(
            __version__="14.2.0",
            cuda=SimpleNamespace(runtime=_Runtime())))
    monkeypatch.setattr(
        backend_module, "CudaPreprocessBackend", lambda: cuda)
    monkeypatch.setattr(
        backend_module, "_gpu_runtime_installed", lambda: True)
    cpu = SimpleNamespace(name="cpu")
    monkeypatch.setattr(
        backend_module, "ParallelCpuPreprocessBackend",
        lambda **_kwargs: cpu)
    assert resolve_preprocess_backend("auto") is cpu
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "not installed" not in err
    assert "no CUDA device is visible here" in err
    assert "no CUDA-capable device is detected" in err


def test_auto_with_cupy_installed_but_unloadable_says_it_could_not_be_loaded(
        monkeypatch, capsys, _fresh_announcement):
    """cupy resolves but its import fails (a missing CUDA library, say):
    the real loader wraps that in "CuPy is required ...", and the line
    quoted that wrapper after "no CUDA device answered", naming neither
    the failure nor its cause.  It says cupy could not be loaded and
    quotes the import error."""

    import sys

    from types import SimpleNamespace

    monkeypatch.setitem(sys.modules, "cupy", None)
    monkeypatch.setattr(
        backend_module, "_gpu_runtime_installed", lambda: True)
    cpu = SimpleNamespace(name="cpu")
    monkeypatch.setattr(
        backend_module, "ParallelCpuPreprocessBackend",
        lambda **_kwargs: cpu)
    assert resolve_preprocess_backend("auto") is cpu
    err = capsys.readouterr().err
    assert err.count("\n") == 1
    assert "cupy is installed but could not be loaded" in err
    assert "import of cupy halted" in err
    assert "device" not in err
    assert "CuPy is required" not in err


# ---------------------------------------------------------------------------
# The remedy names what is certified, not what merely installs
# ---------------------------------------------------------------------------

def _remedy_install_lines():
    """The lines the reader is told to RUN, not the ones explaining."""

    from gpuwm.ingest.preprocess_backend import _GPU_PREPROCESS_REMEDY

    return [line.strip() for line in _GPU_PREPROCESS_REMEDY.splitlines()
            if line.strip().startswith("remedy:")]


def test_the_gpu_preprocess_remedy_names_exactly_the_certified_pairs():
    """One install line per certified CUDA major, and nothing else.

    A remedy that installs a wheel for a runtime ``auto`` still declines
    leaves the reader on the CPU backend after doing what it said; a
    certified major with no line leaves that box without a remedy.
    """

    from gpuwm.ingest.preprocess_backend import (
        CERTIFIED_PREPROCESS_CUDA_MAJORS)

    lines = _remedy_install_lines()
    assert lines, "the remedy stopped naming a command to run"
    expected = [f"remedy: pip install 'gpuwm[{row['extra']}]'"
                for _major, row in sorted(
                    CERTIFIED_PREPROCESS_CUDA_MAJORS.items())]
    assert lines == expected


def test_the_remedy_states_the_certified_majors():
    from gpuwm.ingest.preprocess_backend import _GPU_PREPROCESS_REMEDY

    assert "certified on CUDA 12 and CUDA 13" in _GPU_PREPROCESS_REMEDY
    assert "--preprocess-backend cpu" in _GPU_PREPROCESS_REMEDY


def test_the_certification_is_the_table_not_a_restated_literal():
    """The resolver asks the table; the sealed bundle keeps its own pin.

    A literal runtime range inside the resolver is how the answer to
    "what is certified" stayed CUDA 12 after CUDA 13 passed the same
    parity certification.  The sealed native-WRF distribution ships one
    CuPy wheel and pins its runtime family separately.
    """

    import inspect

    from gpuwm import ingest
    from gpuwm.gpu_stack_identity import CUDA_RUNTIME_RANGE

    source = inspect.getsource(ingest.preprocess_backend)
    assert "CERTIFIED_PREPROCESS_CUDA_MAJORS.get(major)" in source
    assert "12_000 <= runtime_version < 13_000" not in source
    resolver = source.split("def resolve_preprocess_backend", 1)[1]
    assert "CUDA_RUNTIME_RANGE" not in resolver
    assert CUDA_RUNTIME_RANGE == (12_000, 13_000)
