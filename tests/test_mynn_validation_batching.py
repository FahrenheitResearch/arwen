"""Immediate validation masks retain their predicates while batching scans."""
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize('count', [4, 6, 50])
def test_device_sized_grid_preserves_variable_length_tail_masks(count, monkeypatch):
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu as gpu
    stream = cp.cuda.Stream(non_blocking=True)
    with stream:
        arrays = [cp.zeros(0 if i == 0 else 131073 + i * 193, cp.float32) for i in range(count)]
    expected = [False] * count
    for index, value in ((1, cp.nan), (count - 1, cp.inf)):
        with stream:
            arrays[index][-1] = value
        expected[index] = True
    actual_factory = gpu._validation_batch_kernel
    grids = []
    def factory(kind, width):
        actual = actual_factory(kind, width)
        def launch(grid, block, arguments):
            grids.append(grid)
            return actual(grid, block, arguments)
        return launch
    monkeypatch.setattr(gpu, '_validation_batch_kernel', factory)
    with stream:
        flags = cp.full(64, -1, cp.int32)
        assert gpu._flag_mask(gpu._nonfinite(), arrays, flags) == expected
        budget = 32 * cp.cuda.runtime.getDeviceProperties(cp.cuda.runtime.getDevice())['multiProcessorCount']
        assert grids == [(min((max(a.size for a in arrays) + 127) // 128,
                              max(1, budget // count)), count)]
        for array in arrays:
            array.fill(1.)
        assert gpu._flag_mask(gpu._nonfinite(), arrays, flags) == [False] * count


@pytest.mark.parametrize('kind', ['nonfinite', 'nonpositive', 'nonzero'])
@pytest.mark.parametrize('dtype', ['float32', 'float64'])
@pytest.mark.parametrize('capacity', [4, 64])
def test_masks_match_independent_reductions_across_blocks_and_edge_values(kind, dtype, capacity):
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu as gpu
    kernel = getattr(gpu, '_' + kind)()
    tiny = np.finfo(np.float32).tiny
    sub = np.nextafter(np.float32(0), np.float32(1))
    values = [0., -0., sub, -sub, tiny, -tiny, tiny/2, 1., -1., np.inf, -np.inf, np.nan]
    arrays = []
    for i in range(69):
        # Vary lengths, including beyond one scan block and an empty field.
        array = cp.zeros(0 if i == 68 else (4097 if i % 3 == 0 else 37), dtype=dtype)
        if array.size:
            array[-1] = values[i % len(values)]
        arrays.append(array)
    expected = [bool(int(kernel(array))) for array in arrays]
    flags = cp.full(capacity, -1, cp.int32)
    assert gpu._flag_mask(kernel, arrays, flags) == expected
    for array in arrays:
        array.fill(1. if kind == 'nonpositive' else 0.)
    assert gpu._flag_mask(kernel, arrays, flags) == [False]*len(arrays)


def test_large_native_group_avoids_per_array_reduction_launches(monkeypatch):
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu as gpu
    actual = gpu._nonfinite()
    class CountedReduction:
        name = actual.name
        calls = 0
        def __call__(self, *args, **kwargs):
            self.calls += 1
            return actual(*args, **kwargs)
    observed = CountedReduction()
    monkeypatch.setattr(gpu, '_nonfinite', lambda: observed)
    arrays = [cp.zeros((17, 50 if i % 2 else 51), cp.float32) for i in range(24)]
    arrays[21][-1, -1] = cp.nan
    flags = cp.zeros(64, cp.int32)
    assert gpu._flag_mask(observed, arrays, flags) == [i == 21 for i in range(24)]
    assert observed.calls == 0, 'each native input still launches its own reduction'


def test_noncontiguous_and_custom_predicates_keep_the_original_contract():
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu as gpu
    arrays = [cp.arange(40, dtype=cp.float32)[::2] for _ in range(6)]
    kernel = cp.ReductionKernel('T x', 'int32 y', 'x > 10 ? 1 : 0', 'a | b', 'y = a', '0', 'custom_threshold')
    flags = cp.zeros(4, cp.int32)
    assert gpu._flag_mask(kernel, arrays, flags) == [True]*6
    assert gpu._flag_mask(gpu._nonfinite(), arrays, flags) == [False]*6


def test_a_custom_predicate_cannot_borrow_a_native_name():
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu as gpu
    custom = cp.ReductionKernel('T x', 'int32 y', 'x > 10 ? 1 : 0',
        'a | b', 'y = a', '0', 'mynn_pbl_nonzero')
    arrays = [cp.ones(13, cp.float32) for _ in range(6)]
    assert gpu._flag_mask(custom, arrays, cp.zeros(64, cp.int32)) == [False]*6


def test_streams_keep_separate_input_arguments_and_status_buffers():
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu as gpu
    gpu._flag_mask(gpu._nonzero(), [cp.zeros(12, cp.float32)]*8, cp.zeros(64, cp.int32))
    def run(which):
        with cp.cuda.Device(0), cp.cuda.Stream(non_blocking=True):
            arrays = [cp.zeros(1001, cp.float32) for _ in range(8)]
            flags = cp.full(64, -1, cp.int32)
            for turn in range(10):
                hit = (which + turn) % len(arrays)
                for array in arrays:
                    array.fill(0.)
                arrays[hit][-1] = 1.
                assert gpu._flag_mask(gpu._nonzero(), arrays, flags) == [i == hit for i in range(8)]
        return True
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert all(pool.map(run, [0, 3]))


@pytest.mark.parametrize('scenario', ['nonfinite', 'positive', 'named_forcing'])
def test_real_tendency_door_preserves_immediate_error_order_and_input_bytes(scenario, monkeypatch):
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu as gpu
    from test_mynn_pbl import _tendencies_inputs, _tendencies_oracle
    _, fields = _tendencies_oracle()
    inputs = {name:cp.asarray(value) for name,value in _tendencies_inputs(fields).items()}
    if scenario == 'nonfinite':
        inputs['dz'][0,0] = cp.nan
        inputs['rho'][0,0] = 0.
        message = 'MYNN tendency inputs must be finite'
    elif scenario == 'positive':
        inputs['rho'][0,0] = 0.
        inputs['exner'][0,0] = 0.
        message = 'MYNN tendency dz and rho must be positive'
    else:
        inputs['s_awthl'].fill(1.)
        inputs['sub_thl'].fill(1.)
        message = ('this MYNN tendency lane admits only zero mass-flux, subsidence '
                   'and detrainment forcing; nonzero: s_awthl, sub_thl')
    before = {name:cp.asnumpy(value).tobytes() for name,value in inputs.items()}
    monkeypatch.setattr(gpu, '_tendency_launch', lambda *a,**k:pytest.fail('invalid inputs reached the scientific solver'))
    with pytest.raises(ValueError) as caught:
        gpu.mynn_tendencies_nomf_cuda(inputs)
    assert str(caught.value) == message
    assert {name:cp.asnumpy(value).tobytes() for name,value in inputs.items()} == before


@pytest.mark.parametrize('tripped, expected', [
    ((), None),
    ((0,), 'finite'),
    ((0, 2), 'finite'),
    ((2,), 'third'),
    ((1, 3), 'second'),
    ((3,), 'fourth'),
])
def test_one_read_per_call_site_keeps_the_first_refusal_in_order(tripped, expected):
    """A call site that queues every check and reads the words once refuses
    with the message of the first check, in the caller's order, that the
    one-read-per-check sequence refused with."""
    import cupy as cp
    from gpuwm.core import mynn_pbl_gpu as gpu
    finite = [cp.ones((7, 5), cp.float32, order='F') for _ in range(6)]
    second = [cp.ones(9, cp.float32), cp.ones((3, 4), cp.float32)]
    third = [cp.ones(11, cp.float32)]
    fourth = [cp.ones((4, 6), cp.float32, order='F') for _ in range(5)]
    if 0 in tripped:
        finite[4][6, 2] = cp.nan
    if 1 in tripped:
        second[1][2, 3] = 0.0
    if 2 in tripped:
        third[0][10] = -1.0
    if 3 in tripped:
        fourth[3][0, 5] = 0.0
    checks = ((gpu._nonfinite(), finite, 'finite'),
              (gpu._nonpositive(), second, 'second'),
              (gpu._nonpositive(), third, 'third'),
              (gpu._nonpositive(), fourth, 'fourth'))
    flags = cp.full(64, -1, cp.int32)
    assert gpu._first_refusal(checks, flags) == expected
    sequential = next((message for kernel, arrays, message in checks
                       if gpu._tripped(kernel, arrays, cp.zeros(64, cp.int32))),
                      None)
    assert sequential == expected
