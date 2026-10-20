"""The compact VEGE_FLUX handoff must retain every output and fallback word."""

import numpy as np

from conftest import requires_gpu


@requires_gpu
def test_vege_expansion_preserves_words_and_bare_fallbacks():
    import cupy as cp
    from gpuwm.core.noahmp_column_slab import (
        _spread_vege_flux, _VEGE_SPREAD_NAMES, _VEGE_SPREAD_FALLBACK)

    n = 257  # Partial final block, with mixed bare and vegetated columns.
    words = np.array([0, 0x80000000, 1, 0x80000001, 0x7fc12345,
                      0x7f800000, 0xff800000, 0x3f800000], np.uint32)
    entry = {name: cp.asarray(np.resize(np.roll(words, k), n).view(np.float32))
             for k, name in enumerate(_VEGE_SPREAD_FALLBACK.values())}
    for selected in (np.arange(1, n, 3), np.empty(0, dtype=np.int64)):
        # Strided source views exercise the raw pointer boundary as well.
        vf = {source: cp.asarray(np.resize(np.roll(words, k), selected.size * 2)
                                .view(np.float32))[::2]
              for k, source in enumerate(_VEGE_SPREAD_NAMES.values())}
        out = _spread_vege_flux(vf, cp.asarray(selected), entry.__getitem__, n)
        for name, source in _VEGE_SPREAD_NAMES.items():
            fallback = _VEGE_SPREAD_FALLBACK.get(name)
            expected = (cp.asnumpy(entry[fallback]).view(np.uint32).copy()
                        if fallback else np.zeros(n, np.uint32))
            expected[selected] = cp.asnumpy(vf[source]).view(np.uint32)
            assert out[name].flags.c_contiguous
            np.testing.assert_array_equal(cp.asnumpy(out[name]).view(np.uint32),
                                          expected, err_msg=name)
