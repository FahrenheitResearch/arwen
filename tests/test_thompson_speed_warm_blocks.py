"""Compare network launch widths against the former 256-thread launch."""
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


def _reference_kernel(loader):
    def get_kernel(module, symbol):
        kernel = loader(module, symbol)
        def launch(grid, block, args):
            size = int(args[-1])
            return kernel(((size + 255) // 256,), (256,), args)
        return launch
    return get_kernel


def _bits(groups):
    result = {}
    for index, group in enumerate(groups):
        for name, value in group.items():
            result[index, name] = value.view(np.uint32).copy()
    return result



def test_warm_launch_bits(monkeypatch):
    cp = pytest.importorskip('cupy')
    import test_thompson_aerosol_warm_gpu as fixture
    from gpuwm.core import thompson_aerosol_warm as launcher
    column = fixture._warm_column(n=517)
    def run():
        groups = fixture._run_network(column)[:3]
        return _bits([{name: cp.asnumpy(value) for name, value in group.items()} for group in groups])
    candidate = run()
    monkeypatch.setattr(launcher, 'aerosol_kernel', _reference_kernel(launcher.aerosol_kernel))
    reference = run()
    for key in reference:
        np.testing.assert_array_equal(candidate[key], reference[key], err_msg=str(key))
