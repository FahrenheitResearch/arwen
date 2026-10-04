"""Moist EOS words stay equal to independent original column diagnostics."""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("count", (1, 4, 10, 20))
@pytest.mark.parametrize("terrain", (False, True))
@pytest.mark.parametrize("hypso", (1, 2))
def test_moist_diagnostic_words(count, terrain, hypso):
    import cupy as cp
    from test_ensemble_batch_acoustic_gpu import _pack_physical
    from gpuwm.core.diagnostics import update_diagnostics
    from gpuwm.ensemble.batch_diagnostics import prepare_update_diagnostics
    state, references, cfg, _ = _pack_physical(count, moist=True, terrain=terrain, mapped=terrain)
    # Require a fresh evaluation from changed prognostic words.
    state.thp += np.float32(0.125)
    state.qv *= np.float32(1.015625)
    for scalar in references:
        scalar.thp += np.float32(0.125)
        scalar.qv *= np.float32(1.015625)
    launch = prepare_update_diagnostics(state, hypso)
    launch()
    cp.cuda.get_current_stream().synchronize()
    for member, scalar in enumerate(references):
        update_diagnostics(scalar, hypso)
        for name in ("p", "al", "alt"):
            actual = cp.asnumpy(state.member_view(name, member)).view(np.uint32)
            expected = cp.asnumpy(getattr(scalar, name)).view(np.uint32)
            assert actual.tobytes() == expected.tobytes(), (member, hypso, name)
    allocations = []
    original = cp.cuda.get_allocator()
    def observed(nbytes):
        allocations.append(int(nbytes))
        return original(nbytes)
    with cp.cuda.using_allocator(observed):
        launch()
    cp.cuda.get_current_stream().synchronize()
    assert not allocations
