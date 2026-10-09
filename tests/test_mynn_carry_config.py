"""The configuration door and published carry contract for both generations."""
import pytest
from gpuwm.config import RunConfig, validate_run_config
from gpuwm.physics_registry import physics_registry


@pytest.mark.parametrize("version", ("wrf_461", "gsd_41"))
def test_cycled_mynn_config_is_admitted(version):
    cfg = RunConfig(nx=8, ny=8, nz=30, dx=3000., dy=3000., ztop=16000.,
                    dt=20., run_seconds=60., bl_pbl_physics=5,
                    sf_sfclay_physics=5, moist=True, mp_physics=8,
                    bl_mynn_version=version, cycling=True)
    validate_run_config(cfg)


def test_registry_names_the_cycling_divergence_without_the_retired_refusal():
    row = physics_registry()["components"]["pbl"]["options"]["mynn"]
    text = " ".join(row["warnings"])
    assert "cycling must be false" not in text
    assert "three-assignment repaired referee" in text
    assert row["consumers"]["restart_algorithm_identity"] == (
        "mynn-edmf-pbl-wrf-v4.6.1-v3-exact-driver")
