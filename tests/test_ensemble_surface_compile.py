"""The fixed surface arm keeps its explicit floating point compiler policy."""
from types import SimpleNamespace
import importlib
from gpuwm.ensemble import surface_recipe


def test_surface_arm_compiles_without_cupy_adding_a_second_ftz(monkeypatch):
    calls = []
    class Module:
        def load(self, image):
            calls.append(('load', image))
    def compile_native(source, options, arch, name):
        assert options == ('--std=c++17', '--fmad=false', '--ftz=false')
        calls.append(('compile', source, options, name))
        return b'ptx', {}
    xp = SimpleNamespace(cuda=SimpleNamespace(
        runtime=SimpleNamespace(getDevice=lambda: 0),
        compiler=SimpleNamespace(compile_using_nvrtc=compile_native),
        function=SimpleNamespace(Module=Module)))
    def refuse_raw(**kwargs):
        raise RuntimeError('CuPy RawModule appends -ftz=true')
    xp.RawModule = refuse_raw
    monkeypatch.setattr(surface_recipe, '_MODULES', {})
    records = []
    monkeypatch.setattr(importlib.import_module('gpuwm.certify.kernel_manifest'),
                        'record_module', lambda *args, **kwargs: records.append(kwargs))
    first = surface_recipe._module(xp)
    assert surface_recipe._module(xp) is first
    assert [row[0] for row in calls] == ['compile', 'load']
    assert records[0]['options'] == calls[0][2]
    assert records[0]['module'].ptx == b'ptx'
