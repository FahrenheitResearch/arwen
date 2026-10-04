"""RK copies preserve all words, including NaN payloads and odd slab tails."""
import os

import numpy as np
import pytest

cp = pytest.importorskip("cupy")
pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get("GPUWM_NO_LOCAL_GPU", "") not in ("", "0"),
    reason="GPUWM_NO_LOCAL_GPU is set")]

from gpuwm.ensemble.batch_bookkeeping import prepare_bookkeeping


@pytest.mark.parametrize("members", [1, 4, 10, 20, 40])
@pytest.mark.parametrize("shape", [(3, 5, 7), (2, 8, 16)])
def test_member_word_copy_and_zero_equal_scalar(members, shape):
    rng = np.random.default_rng(714)
    host = rng.integers(0, 2**32, size=(members,) + shape, dtype=np.uint32)
    host.reshape(-1)[:4] = (0x7fc00001, 0xffc00123, 0x80000000, 0x00000000)
    src = cp.asarray(host.view(np.float32))
    dst = cp.empty_like(src)
    aux = cp.ones((members, 5, 9), cp.float32)
    copied = cp.empty_like(aux)
    prepare_bookkeeping(((src, dst), (aux, copied)), members=members)()
    assert cp.asnumpy(dst).view(np.uint32).tobytes() == host.tobytes()
    assert cp.asnumpy(copied).tobytes() == cp.asnumpy(aux).tobytes()
    prepare_bookkeeping(((dst, dst), (copied, copied)), members=members, zero=True)()
    assert not cp.asnumpy(dst).view(np.uint32).any()
    assert not cp.asnumpy(copied).view(np.uint32).any()


def test_batch_bookkeeping_refuses_cross_row_and_self_alias():
    a = cp.zeros((4, 3, 5), cp.float32)
    b = cp.zeros_like(a)
    with pytest.raises(ValueError, match="overlap"):
        prepare_bookkeeping(((a, a),), members=4)
    with pytest.raises(ValueError, match="overlap"):
        prepare_bookkeeping(((a, b), (a, b)), members=4)
