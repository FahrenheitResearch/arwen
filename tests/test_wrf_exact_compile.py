"""CPU checks for strict compiler isolation and the final CuPy FTZ override."""

from types import SimpleNamespace

from gpuwm import wrf_exact


def test_strict_options_remove_last_occurrence_conflicts():
    options = wrf_exact.effective_options((
        "-std=c++17", "--fmad=true", "-ftz=true", "--ftz=false",
        "--use_fast_math", "--prec-div=false", "--prec-sqrt=false",
        "-Iheaders", "-DGPUWM_WRF_EXACT=0"))
    assert options == ("-std=c++17", "-Iheaders") + wrf_exact.STRICT_OPTIONS


def test_default_never_installs_a_compiler_hook(monkeypatch):
    monkeypatch.setattr(wrf_exact, "ENABLED", False)
    compiler = SimpleNamespace()
    wrf_exact.install(compiler)
    assert vars(compiler) == {}


def test_cache_key_and_final_program_both_get_strict_options(monkeypatch):
    calls = []

    class Program:
        src = "source"

        def compile(self, options=(), log_stream=None):
            calls.append(("program", options))
            return "compiled"

    def cache(source, options, *args, **kwargs):
        calls.append(("cache", source, options, args, kwargs))
        # Reproduce CuPy adding FTZ after the caller's switches.
        return Program().compile(options + ("-ftz=true",))

    compiler = SimpleNamespace(_compile_with_cache_cuda=cache,
                               _NVRTCProgram=Program)
    monkeypatch.setattr(wrf_exact, "ENABLED", True)
    wrf_exact.install(compiler)
    wrapped = compiler._compile_with_cache_cuda
    wrf_exact.install(compiler)
    assert compiler._compile_with_cache_cuda is wrapped
    assert wrapped("source", ("-std=c++17",), "120", backend="nvrtc") == "compiled"
    expected = ("-std=c++17",) + wrf_exact.STRICT_OPTIONS
    assert calls == [("cache", "source", expected, ("120",), {"backend": "nvrtc"}),
                     ("program", expected)]
    calls.clear()
    Program().compile(("--ftz=true",))
    assert calls == [("program", wrf_exact.STRICT_OPTIONS)]
