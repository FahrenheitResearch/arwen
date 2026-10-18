"""Fast numerics is the default; strict is the opt-in (owner ruling 2026-10-16).  CPU only."""
import importlib
import subprocess
import sys
import types


def _fresh(monkeypatch, **env):
    for name in ("GPUWM_NUMERICS", "GPUWM_WRF_EXACT"):
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    import gpuwm.numerics as numerics
    numerics._mode = None
    return numerics


def test_default_is_fast_and_drops_only_the_no_contraction_request(monkeypatch):
    numerics = _fresh(monkeypatch)
    assert numerics.mode() == "fast"
    assert numerics.effective_options(("-std=c++17", "--fmad=false", "--ftz=false")) == \
        ("-std=c++17", "--ftz=false")
    assert numerics.effective_options(("-fmad=false",)) == ()
    numerics._mode = None


def test_strict_and_wrf_exact_compile_as_written(monkeypatch):
    tuple_ = ("-std=c++17", "--fmad=false", "--ftz=false")
    numerics = _fresh(monkeypatch, GPUWM_NUMERICS="strict")
    assert numerics.effective_options(tuple_) == tuple_
    numerics = _fresh(monkeypatch, GPUWM_WRF_EXACT="1")
    assert numerics.mode() == "strict" and numerics.effective_options(tuple_) == tuple_
    numerics._mode = None


def test_compile_boundary_hook(monkeypatch):
    numerics = _fresh(monkeypatch)
    seen = []

    class Program:
        def compile(self, options=(), log_stream=None):
            seen.append(("program", tuple(options)))

    compiler = types.SimpleNamespace(
        _compile_with_cache_cuda=lambda source, options, *a, **k: seen.append(("cache", tuple(options))),
        _NVRTCProgram=Program)
    numerics.install(compiler)
    numerics.install(compiler)  # idempotent
    compiler._compile_with_cache_cuda("src", ("-std=c++17", "--fmad=false"))
    Program().compile(("--fmad=false", "--ftz=false"))
    assert seen == [("cache", ("-std=c++17",)), ("program", ("--ftz=false",))]
    numerics._mode = None


def test_import_gpuwm_does_not_import_cupy():
    code = "import sys, gpuwm; print('cupy' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                            env={**__import__("os").environ, "GPUWM_NUMERICS": "fast"})
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "False"
