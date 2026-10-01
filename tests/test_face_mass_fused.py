"""The face helper keeps the eager sum, multiply and periodic alias words."""
import numpy as np
import pytest


@pytest.mark.gpu
@pytest.mark.parametrize('shape', [(1, 1), (1, 19), (17, 1), (13, 17)])
def test_face_words_match_eager(shape):
    import cupy as cp
    from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces
    rng = np.random.default_rng(2819)
    mu = cp.asarray(rng.uniform(-100000, 100000, shape), dtype=cp.float32)
    for fn, axis in [(mu_at_u_faces, 1), (mu_at_v_faces, 0)]:
        mass = .5 * (mu + cp.roll(mu, 1, axis=axis))
        expected = cp.concatenate([mass, mass[:, :1] if axis else mass[:1]], axis=axis)
        actual = fn(mu)
        np.testing.assert_array_equal(actual.view(cp.uint32).get(),
                                      expected.view(cp.uint32).get())


@pytest.mark.gpu
def test_noncontiguous_and_double_keep_eager_path():
    import cupy as cp
    from gpuwm.core.state import mu_at_u_faces, mu_at_v_faces
    for mu in [cp.arange(40, dtype=cp.float32).reshape(5, 8)[:, ::2],
               cp.arange(20, dtype=cp.float64).reshape(5, 4)]:
        for fn, axis in [(mu_at_u_faces, 1), (mu_at_v_faces, 0)]:
            mass = .5 * (mu + cp.roll(mu, 1, axis=axis))
            expected = cp.concatenate([mass, mass[:, :1] if axis else mass[:1]], axis=axis)
            np.testing.assert_array_equal(fn(mu).get(), expected.get())
