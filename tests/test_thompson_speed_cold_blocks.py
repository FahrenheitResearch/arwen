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


@pytest.mark.parametrize('scenario', ['aero-scav-frozen', 'aero-ice-demott-dep', 'aero-ice-koop'])
def test_cold_launch_bits(monkeypatch, scenario):
    pytest.importorskip('cupy')
    import test_thompson_aerosol_cold_gpu as fixture
    from gpuwm.core import thompson_aerosol_cold as launcher
    from gpuwm.core.thompson_runtime import load_classic_device_tables
    root = fixture._classic_table_root()
    if root is None:
        pytest.skip('canonical Thompson tables are not staged')
    tables = load_classic_device_tables(str(root))
    host = fixture._host
    # Repeat the real column across partial and complete blocks.
    monkeypatch.setattr(fixture, '_host', lambda rows, name: np.tile(host(rows, name), 17))
    candidate = _bits(fixture._run_cold_network(scenario, 10.0, tables))
    monkeypatch.setattr(launcher, 'get_kernel', _reference_kernel(launcher.get_kernel))
    reference = _bits(fixture._run_cold_network(scenario, 10.0, tables))
    for key in reference:
        np.testing.assert_array_equal(candidate[key], reference[key], err_msg=str(key))
