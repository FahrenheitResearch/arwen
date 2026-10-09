"""Deferred named physics ports cannot enter a default release run."""

import pytest

from gpuwm.config import MP_PHYSICS_ACCEPTED, RunConfig, validate_run_config
from gpuwm.microphysics_schemes import NAMED_SCHEMES, resolve_mp_physics, scheme


def test_deferred_named_microphysics_is_not_admitted():
    assert NAMED_SCHEMES == {}
    assert scheme(900) is None
    assert 900 not in MP_PHYSICS_ACCEPTED
    with pytest.raises(ValueError, match="not a scheme name"):
        resolve_mp_physics("tempo")
    with pytest.raises(ValueError, match="mp_physics"):
        validate_run_config(RunConfig(
            nx=12, ny=12, nz=16, dx=2000., dy=2000., ztop=8000.,
            dt=2., run_seconds=2., moist=True, mp_physics=900))


def test_existing_pbl_900_is_preserved():
    from gpuwm.config import SASE_PBL_SCHEME
    from gpuwm.checkpoint_identity import PBL_ALGORITHM_IDENTITIES

    assert SASE_PBL_SCHEME == 900
    assert PBL_ALGORITHM_IDENTITIES[900] == "sase-experimental-v1"
