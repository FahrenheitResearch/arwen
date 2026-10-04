"""Moist large-step bindings keep original arithmetic and planned carriers."""
import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


@pytest.mark.parametrize("count", (1, 4, 10, 20))
@pytest.mark.parametrize("terrain", (False, True))
@pytest.mark.parametrize("mp", (0, 8))
def test_moist_pgf_buoyancy_and_heating_finish_words(count, terrain, mp):
    import cupy as cp
    from test_ensemble_batch_acoustic_gpu import _pack_physical
    from gpuwm.core import dycore, acoustic
    from gpuwm.ensemble import batch_acoustic, batch_bigstep
    state, references, cfg, _ = _pack_physical(count, moist=True, mp_physics=mp,
                                             terrain=terrain, mapped=terrain)
    cq = batch_acoustic.prepare_moist_cq(state, cfg)
    batch_bigstep.prepare_slow_pgf(state, cfg, cq=cq)()
    batch_bigstep.prepare_slow_buoyancy(state, cfg)()
    batch_bigstep.prepare_small_step_init(state, cfg)()
    interval = np.float32(3) if mp else np.float32(0)
    if mp:
        values = np.random.default_rng(147).uniform(-0.001, 0.001, state.h_diabatic.shape).astype(np.float32)
        state.h_diabatic.set(values)
        for member, scalar in enumerate(references):
            scalar.h_diabatic.set(values[member])
    batch_bigstep.prepare_small_step_finish(state, cfg, hdiab_dt=interval)()
    for member, scalar in enumerate(references):
        scalar_cq = acoustic.prepare_moist_cq(scalar, cfg)
        dycore._launch_slow_pgf(scalar, cfg, cq=scalar_cq)
        dycore._launch_slow_buoyancy(scalar, cfg)
        dycore._prepare_small_step_init_launch(scalar, cfg)()
        dycore._prepare_small_step_finish_launch(scalar, cfg, interval)()
        for name in ("ru_t", "rv_t", "rw_t", "u", "v", "w", "thp", "php", "mup",
                     "u_pp", "v_pp", "th_pp", "ph_pp", "mu_pp"):
            actual = cp.asnumpy(state.member_view(name, member)).view(np.uint32)
            expected = cp.asnumpy(getattr(scalar, name)).view(np.uint32)
            assert actual.tobytes() == expected.tobytes(), (member, mp, name)
