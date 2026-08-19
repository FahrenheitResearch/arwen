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


def test_auto_without_cupy_resolves_cpu_and_says_so(
        monkeypatch, capsys, _fresh_announcement):
    _cupyless(monkeypatch)
    cpu = object()
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

    _cupyless(monkeypatch)
    monkeypatch.setattr(
        backend_module, "ParallelCpuPreprocessBackend",
        lambda **_kwargs: object())
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
    """cupy installed but outside the certified family: the one line
    names the runtime it declined, not a missing install."""

    from types import SimpleNamespace

    class _Runtime:
        @staticmethod
        def runtimeGetVersion():
            return 13_000

        @staticmethod
        def getDeviceCount():
            return 1

    cuda = SimpleNamespace(
        name="cuda",
        array_module=SimpleNamespace(
            __version__="13.1.0",
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
    assert "13" in err
