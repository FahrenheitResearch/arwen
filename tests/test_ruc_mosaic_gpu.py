"""Device column and fused runtime checks for both RUC mosaic controls."""

from dataclasses import fields
import numpy as np
import pytest

from conftest import requires_gpu
from ruc_mosaic_fixture import surface_cases, driver_calls

pytestmark = requires_gpu


def test_cuda_mosaic_surface_matches_wrf_bits():
    import cupy as cp
    from gpuwm.core.ruc_gpu import ruc_surface_parameters_cuda
    for row, inputs, keywords in surface_cases():
        result = ruc_surface_parameters_cuda(*inputs, **keywords)
        for descriptor in fields(result):
            name = descriptor.name
            dtype = np.int32 if name == "iforest" else np.float32
            expected = np.array([float(row[name])], dtype=dtype)
            np.testing.assert_array_equal(cp.asnumpy(getattr(result, name)).view(np.uint32),
                                          expected.view(np.uint32), err_msg=f"{row['case']}:{name}")


def test_resident_mosaic_columns_match_wrf_bits():
    import cupy as cp
    from gpuwm.core.ruc import ruc_land_surface_step
    from gpuwm.core.ruc_runtime import ruc_device_sfctmp_sets
    leaves, stages, arrays = ruc_device_sfctmp_sets()
    for label, values, keywords, expected in driver_calls():
        values = {name: cp.asarray(value) for name, value in values.items()}
        result = ruc_land_surface_step(values, **keywords, leaves=leaves, stages=stages, arrays=arrays)
        for name, reference in expected.items():
            np.testing.assert_array_equal(cp.asnumpy(getattr(result, name)).view(np.uint32),
                                          reference.view(np.uint32), err_msg=f"{label}:{name}")


@pytest.mark.parametrize("mosaic_lu,mosaic_soil", [(1, 0), (0, 1), (1, 1)])
@pytest.mark.parametrize("lakemodel", [0, 1])
@pytest.mark.parametrize("irrigation", ["wrf_45", "wrf_461"])
def test_fused_mosaic_runtime_matches_resident_columns(mosaic_lu, mosaic_soil, lakemodel,
                                                       irrigation):
    import cupy as cp
    from types import SimpleNamespace
    from test_ruc_lsm_fused import _case, _copy, _equal
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing

    bench, driver, atmosphere, cold = _case(5917, 9, "mixed")
    shape = driver.fields["tsk"].shape
    lu = cp.zeros((21,) + shape, cp.float32)
    so = cp.zeros((19,) + shape, cp.float32)
    lu[11] = .55
    lu[9] = .30
    lu[0] = .15
    so[5] = .35
    so[7] = .50
    so[13] = .15
    driver.fields["landusef"] = lu
    driver.fields["soilctop"] = so
    reference = _copy(driver.fields)
    for k in range(1, 4):
        bench.forcing(driver, k, 11, shape, cold)
        bench.forcing(SimpleNamespace(fields=reference), k, 11, shape, cold)
        results = []
        for function, target in ((ruc_lsm_step, driver.fields),
                                 (_ruc_lsm_step_reference, reference)):
            results.append(function(target, atmosphere, params=driver.ruc_params,
                           precipitation=SurfacePrecipitationForcing.from_fields(target),
                           dt=12.0, itimestep=k, mosaic_lu=mosaic_lu,
                           mosaic_soil=mosaic_soil, flag_sm_adj=0,
                           spp_lsm=0, lakemodel=lakemodel,
                           ruc_irrigation=irrigation))
        assert results[0] == results[1]
        _equal(driver.fields, reference)


@pytest.mark.parametrize("qvg_cold_start", ["wrf", "air"])
def test_fused_qvg_cold_start_matches_the_host_twin(qvg_cold_start):
    """Both QVG cold starts, fused against the reference driver, from k=1."""
    import cupy as cp
    from types import SimpleNamespace
    from test_ruc_lsm_fused import _case, _copy, _equal
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing

    bench, driver, atmosphere, cold = _case(1499, 9, "mixed")
    shape = driver.fields["tsk"].shape
    driver.fields["qvg"][...] = cp.float32(0.0)
    driver.fields["qcg"][...] = cp.float32(0.0)
    reference = _copy(driver.fields)
    for k in range(1, 4):
        bench.forcing(driver, k, 11, shape, cold)
        bench.forcing(SimpleNamespace(fields=reference), k, 11, shape, cold)
        results = []
        for function, target in ((ruc_lsm_step, driver.fields),
                                 (_ruc_lsm_step_reference, reference)):
            results.append(function(target, atmosphere, params=driver.ruc_params,
                           precipitation=SurfacePrecipitationForcing.from_fields(target),
                           dt=12.0, itimestep=k, mosaic_lu=0, mosaic_soil=0,
                           flag_sm_adj=0, spp_lsm=0, lakemodel=0,
                           ruc_qvg_cold_start=qvg_cold_start))
        assert results[0] == results[1]
        _equal(driver.fields, reference)


@pytest.mark.parametrize("diagnostic", ["flux", "log_profile"])
def test_fused_2m_diagnostic_matches_the_host_twin(diagnostic):
    """Both 2 m forms, fused epilogue against the host SFCDIAGS twin."""
    from types import SimpleNamespace
    from test_ruc_lsm_fused import _case, _copy, _equal
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing

    bench, driver, atmosphere, cold = _case(1499, 9, "mixed")
    shape = driver.fields["tsk"].shape
    reference = _copy(driver.fields)
    for k in range(1, 4):
        bench.forcing(driver, k, 11, shape, cold)
        bench.forcing(SimpleNamespace(fields=reference), k, 11, shape, cold)
        results = []
        for function, target in ((ruc_lsm_step, driver.fields),
                                 (_ruc_lsm_step_reference, reference)):
            results.append(function(target, atmosphere, params=driver.ruc_params,
                           precipitation=SurfacePrecipitationForcing.from_fields(target),
                           dt=12.0, itimestep=k, mosaic_lu=0, mosaic_soil=0,
                           flag_sm_adj=0, spp_lsm=0, lakemodel=0,
                           ruc_2m_diagnostic=diagnostic))
        assert results[0] == results[1]
        _equal(driver.fields, reference)
