"""The seeded initializer uses identical words for one member and a roster."""
from types import SimpleNamespace
import numpy as np
import pytest

pytestmark = pytest.mark.gpu


@pytest.mark.parametrize("members", [1, 4, 10, 20])
def test_uniform_wind_initializer_matches_every_independent_member(members):
    cp = pytest.importorskip("cupy")
    from gpuwm.ensemble.batch_perturbation import initialize_member_winds
    from gpuwm.ensemble.seeds import member_seed
    cfg = SimpleNamespace(nx=7, ny=5, nz=4)
    rng = np.random.default_rng(9371)
    u = rng.uniform(-15, 15, (members, 4, 5, 8)).astype(np.float32)
    v = rng.uniform(-15, 15, (members, 4, 6, 7)).astype(np.float32)
    seeds = tuple(member_seed(7284, index) for index in range(members))
    batch = SimpleNamespace(u=cp.asarray(u), v=cp.asarray(v))
    receipt = initialize_member_winds(state=batch, cfg=cfg, member_indices=tuple(range(members)),
                                      seeds=seeds, phase="after_physics_before_step")
    for index, seed in enumerate(seeds):
        scalar = SimpleNamespace(u=cp.asarray(u[index]), v=cp.asarray(v[index]))
        initialize_member_winds(state=scalar, cfg=cfg, member_indices=(index,),
                                seeds=(seed,), phase="after_physics_before_step")
        for field in ("u", "v"):
            np.testing.assert_array_equal(getattr(batch, field)[index].get().view(np.uint32),
                                          getattr(scalar, field).get().view(np.uint32))
    assert receipt["seed_allocation_bytes"] == 8 * members
    assert receipt["seed_pool_rounded_bytes"] == 512
    assert receipt["kind"] == "illustrative uncalibrated uniform IC wind ensemble"
    assert not np.array_equal(batch.u.get().view(np.uint32), u.view(np.uint32))


def test_uniform_wind_initializer_refuses_pre_bootstrap_phase():
    from gpuwm.ensemble.batch_perturbation import initialize_member_winds
    with pytest.raises(ValueError, match="initialized physics before stepping"):
        initialize_member_winds(state=None, cfg=None, member_indices=(0,), seeds=(1,),
                                phase="after_restore_before_physics")
