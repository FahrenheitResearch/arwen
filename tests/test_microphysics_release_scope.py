"""Named physics ports that enter the 2.8.8 release, and only those.

2.8.8 ships one named microphysics port, TEMPO (WOOF id 900), because the
RRFS v2 hex preset is TEMPO.  Any other named scheme stays out of a release
run until its own intake admits it.
"""

from gpuwm.config import MP_PHYSICS_ACCEPTED, RunConfig, validate_run_config
from gpuwm.microphysics_schemes import NAMED_SCHEMES, resolve_mp_physics, scheme


def test_tempo_is_the_only_named_microphysics_admitted():
    assert set(NAMED_SCHEMES) == {"tempo"}
    assert scheme(900).name == "tempo"
    assert 900 in MP_PHYSICS_ACCEPTED
    assert resolve_mp_physics("tempo") == 900
    cfg = validate_run_config(RunConfig(
        nx=12, ny=12, nz=16, dx=2000., dy=2000., ztop=8000.,
        dt=2., run_seconds=2., moist=True, mp_physics="tempo"))
    assert cfg.mp_physics == 900


def test_existing_pbl_900_is_preserved():
    from gpuwm.config import SASE_PBL_SCHEME
    from gpuwm.checkpoint_identity import PBL_ALGORITHM_IDENTITIES

    assert SASE_PBL_SCHEME == 900
    assert PBL_ALGORITHM_IDENTITIES[900] == "sase-experimental-v1"
