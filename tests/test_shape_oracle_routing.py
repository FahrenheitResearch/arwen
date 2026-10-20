"""CPU regression for shape-aware native verification launch selection."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


def test_vertical_selector_forwards_dimensions(monkeypatch):
    # Compile the actual selector without importing the GPU-only oracle setup.
    root = Path(__file__).resolve().parents[1]
    tree = ast.parse((root / 'gpuwm/verify/smallstep_vertical_oracle.py').read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'selected_vertical_launch')
    calls, lookups = [], []
    acoustic = SimpleNamespace()

    def get_kernel(module, name, shape=()):
        lookups.append((name, shape))
        return lambda grid, block, args: calls.append((name, args))

    def w_kernel(name, nz, ny=0, nx=0):
        return acoustic.get_kernel('acoustic', name, (nz, ny, nx))

    def prepare(*args):
        mu = acoustic.get_kernel('acoustic', 'advance_mu_th_msf', (44, 70, 120))
        w = acoustic._w_phi_kernel('advance_w_phi_msf', 44, 70, 120)
        def launch(*, first):
            mu((33,), (256,), ('mu',))
            w((33,), (256,), ('w',))
        return launch

    acoustic.get_kernel = get_kernel
    acoustic._w_phi_kernel = w_kernel
    acoustic.prepare_acoustic_substep_launch = prepare
    import gpuwm.core
    monkeypatch.setattr(gpuwm.core, 'acoustic', acoustic, raising=False)
    namespace = {'patch': patch}
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<native-selector>', 'exec'), namespace)
    assert namespace['selected_vertical_launch'](None, None, 18., ()) == 'advance_w_phi_msf'
    assert lookups == [('advance_mu_th_msf', (44, 70, 120)),
                       ('advance_w_phi_msf', (44, 70, 120))]
    assert calls == [('advance_w_phi_msf', ('w',))]
