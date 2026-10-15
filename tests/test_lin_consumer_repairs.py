import pytest
from gpuwm.offline_child import _transported_source_fields
from gpuwm.core import preflight as pf

def test_lin_offline_parent_carries_all_six_masses_without_numbers():
    assert _transported_source_fields(2) == (
        'qv', 'qc', 'qr', 'qi', 'qs', 'qg')

@pytest.mark.parametrize('levels,lin,radar',[(128,37808,7040),(256,45488,14080),(512,60848,28160)])
def test_lin_pricing_models_every_driver_measured_tier(levels,lin,radar):
    assert pf.LIN_TIER_FRAME.frame_bytes(levels) == lin
    assert pf.LIN_REFL_TIER_FRAME.frame_bytes(levels) == radar

def test_lin_radar_module_has_a_recorded_price():
    assert pf.KERNEL_MAX_LOCAL_SIZE_BYTES['lin_refl'] == 14080


def test_lin_generic_offer_retains_its_unverified_scope():
    from gpuwm.prepared_single_domain_forecast import runner_capabilities
    from tools.hrrr_single_domain_benchmark import NATIVE_BENCHMARK_PHYSICS_PROFILES
    profile = 'lin-mp2-ysu-mm5-noah-rrtmg-legacy-implemented-unverified-v1'
    sources = runner_capabilities()['source_profiles']
    for source in sources.values():
        if source['physics_profile_ids']:
            assert profile in source['physics_profile_ids']
            assert 'source-independent' in source['physics_profile_ids_semantics']
            assert 'implemented-unverified' in source['physics_profile_ids_semantics']
    assert profile not in NATIVE_BENCHMARK_PHYSICS_PROFILES
