"""Public WRF package consumers for selector 4."""
from types import SimpleNamespace

def test_offline_scalar_package_matches_the_declared_wrf_package():
    from gpuwm.offline_child import _transported_source_fields
    assert set(_transported_source_fields(4)) == {'qr', 'qs', 'qi', 'qc', 'qv'}

def test_single_moment_number_flags_and_snow_classification():
    from gpuwm.config import MYNN_QN_FLAG_SPECIES, mynn_mixscalars_unported_species
    from gpuwm.core.mynn_pbl_runtime import mynn_flag_qs
    assert MYNN_QN_FLAG_SPECIES[4] == ()
    assert mynn_mixscalars_unported_species(SimpleNamespace(
        mp_physics=4, bl_mynn_mixscalars=1, bl_pbl_physics=5)) == ()
    assert mynn_flag_qs(4) is True
