"""The GF default choices must be explicit in the registry."""
from gpuwm.physics_registry import physics_registry


def test_gf_default_velocity_and_top_bound_are_declared():
    registry = physics_registry()
    extensions = registry["components"]["cumulus"]["options"]["grell-freitas"]["extensions"]
    assert extensions["shallow_closure_top_bound"]["default"] == "min(kpbl,ktf-1)"
    velocity = extensions["vertical_velocity_convention"]
    assert velocity["default"] == velocity["strict"] == "lower-face"


def test_gf_checkpoint_identity_binds_the_changed_defaults():
    from gpuwm.checkpoint_identity import CUMULUS_ALGORITHM_IDENTITIES
    assert CUMULUS_ALGORITHM_IDENTITIES[3] == "grell-freitas-wrf461-gfdrv-corrected-k22-v2"


def test_gf_strict_arm_is_declared_separately_from_default():
    arm=physics_registry()["components"]["cumulus"]["options"]["grell-freitas"]["extensions"]["strict_arithmetic"]
    assert arm["selector"] == "GPUWM_WRF_EXACT=1"
    assert arm["default_unchanged"] is True
    assert arm["cold_condensate_without_ice"] == "retain-stock-split-and-ignore-ice-slot"
