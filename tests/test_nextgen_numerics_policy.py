"""Process-isolated launch/realization policy; no CUDA or hydrology surrogate."""
import os
import subprocess
import sys
import textwrap

import pytest


@pytest.mark.parametrize("boundary", ["go", "resolve"])
@pytest.mark.parametrize("requested,inherited,exact,fixed,expected,error", [
    (None, None, False, None, "fast", None),
    (None, "strict", False, None, "strict", None),
    (None, "fast", True, None, "strict", None),
    ("strict", None, False, None, "strict", None),
    ("fast", "strict", False, None, "fast", None),
    (None, None, False, "strict", "strict", None),
    ("fast", None, True, None, None, "requires strict"),
    ("fast", None, False, "strict", None, "needs a fresh process"),
    ("invalid", None, False, None, None, "must be fast or strict"),
])
def test_prepared_hydro_preserves_process_arithmetic(
        tmp_path, boundary, requested, inherited, exact, fixed, expected, error):
    config = tmp_path / "hydro.toml"
    config.write_text('[hydro]\nrealization="lower-colorado-hyfeatures"\n'
                      + (f'numerics="{requested}"\n' if requested else '')
                      + '[hydro.settings_bindings]\nfabric_manifest="/prepared/manifest.json"\n')
    script = textwrap.dedent('''
        import sys
        from types import SimpleNamespace
        from gpuwm import numerics
        from gpuwm.hydro import runtime
        from gpuwm.hydro.config import configure
        if sys.argv[3] != "None":
            numerics.select(sys.argv[3])
        settings = runtime.read_settings(sys.argv[1])
        original = dict(settings)
        def check(settings):
            resolved = runtime.resolve_settings(settings)
            assert resolved["numerics"] == numerics.mode() == sys.argv[4]
            # The sealed realization must agree with its actual factory mode.
            assert configure(resolved) == resolved
            explicit = dict(original, numerics=sys.argv[4])
            explicit.pop("activation_reason", None)
            assert resolved["realization_identity"] == configure(explicit)["realization_identity"]
        if sys.argv[2] == "go":
            @runtime.go_scope
            def launch(args):
                assert runtime.current_settings()["numerics"] == sys.argv[4]
                check(runtime.current_settings())
            launch(SimpleNamespace(config=sys.argv[1], hydro=None))
        else:
            check(settings)
        assert settings == original
        print("MODE", numerics.mode())
    ''')
    environment = dict(os.environ)
    environment.pop("GPUWM_NUMERICS", None)
    environment.pop("GPUWM_WRF_EXACT", None)
    if inherited is not None:
        environment["GPUWM_NUMERICS"] = inherited
    if exact:
        environment["GPUWM_WRF_EXACT"] = "1"
    result = subprocess.run([sys.executable, "-B", "-c", script, str(config), boundary,
                             str(fixed), str(expected)], env=environment,
                            capture_output=True, text=True, timeout=30)
    if error:
        assert result.returncode != 0
        assert error in result.stderr, result.stderr
    else:
        assert result.returncode == 0, result.stdout + result.stderr
        assert f"MODE {expected}" in result.stdout
