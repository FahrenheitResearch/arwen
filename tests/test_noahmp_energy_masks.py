"""The batched masks must preserve layer boundaries and floating words."""

import numpy as np
from conftest import requires_gpu


@requires_gpu
def test_energy_masks_match_original_for_strided_words_and_fallback():
    import cupy as cp
    from gpuwm.core.noahmp_energy_slab import (
        energy_seg4, _energy_seg4_reference, NSNOW, NSOIL)

    words = np.array([0, 0x80000000, 1, 0x80000001, 0x7fc12345,
                      0x7f800000, 0xff800000, 0x3f800000], np.uint32)
    for n in (257, 0):
        def floats(width=1):
            a = cp.empty(n * width * 2, dtype=cp.float32)[::2]
            a[...] = cp.asarray(np.resize(words, n * width).view(np.float32))
            return a if width == 1 else a.reshape(n, width)
        s = {"isnow": cp.asarray(np.arange(n, dtype=np.int32) % (NSNOW+1)-NSNOW),
             "nroot": cp.asarray(np.arange(n, dtype=np.int32) % (NSOIL+1)),
             "cosz": floats()}
        e = {"dt_soil": cp.ones(n, cp.float32), "btrani": floats(NSOIL)}
        out = {k: floats() for k in ("eflxb", "fsrv", "fsrg", "bgap", "wgap")}
        out.update(hcpct=floats(NSNOW+NSOIL),
                   imelt=cp.arange(n*(NSNOW+NSOIL), dtype=cp.int32)
                           .reshape(n, NSNOW+NSOIL))
        out.update({k: floats(NSNOW) for k in ("snicev", "snliqv", "epore")})
        for fallback in (False, True):
            if fallback:
                s["nroot"] = s["nroot"].astype(cp.int64)
                out["hcpct"] = out["hcpct"].astype(cp.float64)
            expected = _energy_seg4_reference(s, {}, e, out)
            actual = energy_seg4(s, {}, e, out)
            assert actual.keys() == expected.keys()
            for k in expected:
                assert actual[k].dtype == expected[k].dtype
                np.testing.assert_array_equal(
                    cp.asnumpy(actual[k]).view(np.uint8),
                    cp.asnumpy(expected[k]).view(np.uint8), err_msg=k)
