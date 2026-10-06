"""Whole-call bitwise checks for the full-width RUC driver kernels."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu


def _harness():
    path = Path(__file__).with_name("ruc_fused_fixture.py")
    spec = importlib.util.spec_from_file_location("ruc_bench_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(width, nzs, scenario):
    import cupy as cp
    bench = _harness()
    _, _, driver, atmosphere, cold = bench.build(97, 61, 20, nzs, scenario, 11)
    index = cp.arange(width) % (97 * 61)

    def resize(array):
        if isinstance(array, cp.ndarray) and array.shape[-2:] == (61, 97):
            return cp.ascontiguousarray(array.reshape(array.shape[:-2] + (-1,))[..., index]
                                       .reshape(array.shape[:-2] + (1, width)))
        return array.copy() if hasattr(array, "copy") else array

    fields = {name: resize(array) for name, array in driver.fields.items()}
    atmosphere = {name: resize(array) for name, array in atmosphere.items()}
    cold = cold.reshape(-1)[np.arange(width) % (97 * 61)].reshape(1, width)
    return bench, SimpleNamespace(fields=fields, ruc_params=driver.ruc_params), atmosphere, cold


def _copy(fields):
    return {name: array.copy() if hasattr(array, "copy") else array
            for name, array in fields.items()}


def _equal(left, right):
    import cupy as cp
    for name in left:
        if isinstance(left[name], cp.ndarray):
            np.testing.assert_array_equal(cp.asnumpy(left[name]).view(np.uint8),
                                          cp.asnumpy(right[name]).view(np.uint8),
                                          err_msg=name)


def _call(function, fields, atmosphere, params, k):
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing
    return function(fields, atmosphere, params=params,
                    precipitation=SurfacePrecipitationForcing.from_fields(fields),
                    dt=12.0, itimestep=k, mosaic_lu=0, mosaic_soil=0,
                    flag_sm_adj=0, spp_lsm=0, lakemodel=1)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [9, 6])
@pytest.mark.parametrize("scenario", ["mixed", "warm"])
@pytest.mark.parametrize("width", [1, 3, 17, 5917, 100000])
def test_every_field_after_every_call(width, nzs, scenario):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    bench, driver, atmosphere, cold = _case(width, nzs, scenario)
    reference = _copy(driver.fields)
    if width >= 5917 and scenario == "mixed":
        # Two fixed fractions supplement the randomized mixed fixture.
        for fields in (driver.fields, reference):
            fields["xice"][0, 0:2] = cp.asarray([0.55, 0.85], dtype=cp.float32)
    for k in range(1, 8):
        bench.forcing(driver, k, 11, (1, width), cold)
        bench.forcing(SimpleNamespace(fields=reference), k, 11, (1, width), cold)
        actual = _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, k)
        expected = _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, k)
        assert actual == expected
        _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
def test_negative_control_detects_one_word():
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    _, driver, atmosphere, _ = _case(17, 9, "warm")
    reference = _copy(driver.fields)
    reference["psfc"].view(cp.uint32)[0, 0] ^= cp.uint32(1)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 1)
    _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, 1)
    with pytest.raises(AssertionError):
        _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize('nzs', [6, 9])
@pytest.mark.parametrize('mosaic_lu', [0, 1])
def test_generic_ruc_defaults_match_named_legacy_after_every_gpu_call(nzs, mosaic_lu):
    """Omitted selectors reproduce legacy arithmetic throughout mixed columns.

    The independent pre-switch source anchors are in
    test_ruc_default_selection.py. This covers runtime dispatch, successive
    state writes and 2 m diagnostics, including snow and an irrigation witness.
    """
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing

    bench, driver, atmosphere, cold = _case(5917, nzs, 'mixed')
    base = driver.fields
    shape = base['tsk'].shape
    crop = ((base['xland'] < 1.5) & (base['xice'] == 0)
            & (base['snow'] == 0))
    base['ivgtyp'][crop] = 12
    base['isltyp'][crop] = 6
    base['vegfra'][crop] = 80
    base['smois'][:, crop] = cp.float32(.06)
    base['sh2o'][:, crop] = cp.float32(.06)
    if mosaic_lu:
        fractions = cp.zeros((21,) + shape, cp.float32)
        for category in range(21):
            fractions[category] = (base['ivgtyp'] == category + 1).astype(cp.float32)
        fractions[:, crop] = 0
        fractions[11, crop] = cp.float32(.05)
        fractions[9, crop] = cp.float32(.95)
        base['landusef'] = fractions
    arms = {name: _copy(base) for name in ('generic', 'named', 'reference', 'fork')}
    changed = False
    for k in range(1, 7):
        for target in arms.values():
            bench.forcing(SimpleNamespace(fields=target), k, 29, shape, cold)
        kwargs = dict(params=driver.ruc_params, dt=12. if k % 2 else 20.,
                      itimestep=k, mosaic_lu=mosaic_lu, mosaic_soil=0,
                      flag_sm_adj=0, spp_lsm=0, lakemodel=0)
        for name, target in arms.items():
            selected = {} if name == 'generic' else dict(
                ruc_soilprop='wrf_45', ruc_irrigation='wrf_461', ruc_snow='wrf_461',
                ruc_qvg_cold_start='wrf', ruc_2m_diagnostic='flux')
            if name == 'fork':
                selected.update(ruc_irrigation='wrf_45', ruc_snow='wrf_45')
            function = _ruc_lsm_step_reference if name == 'reference' else ruc_lsm_step
            function(target, atmosphere,
                     precipitation=SurfacePrecipitationForcing.from_fields(target),
                     **kwargs, **selected)
        _equal(arms['generic'], arms['named'])
        _equal(arms['generic'], arms['reference'])
        for name in ('snowc', 'tsk', 'smois'):
            changed |= bool(cp.any(arms['generic'][name].view(cp.uint32)
                                   != arms['fork'][name].view(cp.uint32)))
    assert changed, 'the heterogeneous snow and cropland columns never reached a fork difference'


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
@pytest.mark.parametrize("ncol", [17, 500000])
def test_full_width_workspace_allocation_census(nzs, ncol, monkeypatch):
    """Admission counts real arrays, including the generated snow workspace."""
    from gpuwm.core import ruc_fused, ruc_gpu, ruc_memory

    monkeypatch.setattr(ruc_gpu, "_SFCTMP_SCRATCH", {})
    workspace = ruc_fused._Workspace((1, ncol), nzs)
    driver = {name: getattr(workspace, name) for name in
              ("storage", "integer", "run", "flag_slab", "sptr", "iptr", "optr", "cptr")}
    scratch = ruc_gpu._sfctmp_scratch(ncol, nzs)
    surface = {"scratch": scratch[0], "pointers": scratch[2],
               "private_flags": scratch[7], "alive": scratch[8]}
    for actual, expected in (
            (driver, ruc_memory.driver_workspace_allocations(ncol, nzs)),
            (surface, ruc_memory.sfctmp_workspace_allocations(ncol, nzs))):
        assert {name: (array.shape, str(array.dtype)) for name, array in actual.items()} == expected
        assert len({array.data.ptr for array in actual.values()}) == len(actual)
        assert sum(array.nbytes for array in actual.values()) == ruc_memory.allocation_bytes(
            expected, rounded=False)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
def test_full_width_output_allocation_census(nzs, monkeypatch):
    """Full-width outputs coexist with both workspaces until the commit."""
    from gpuwm.core import ruc_fused, ruc_memory
    from gpuwm.core.ruc_runtime import ruc_lsm_step

    _, driver, atmosphere, _ = _case(17, nzs, "mixed")
    original = ruc_fused._RUC_SFCTMP_FULL_WIDTH
    observed = []

    def measure(*args, **kwargs):
        result = original(*args, **kwargs)
        observed.append({name: (array.shape, str(array.dtype))
                         for name, array in result.items()})
        return result

    monkeypatch.setattr(ruc_fused, "_RUC_SFCTMP_FULL_WIDTH", measure)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 1)
    assert observed == [ruc_memory.sfctmp_output_allocations(17, nzs)]


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name,value", [("psfc", np.nan), ("psfc", 0.0),
                                       ("cqs2", np.nan), ("chs2", np.nan),
                                       ("psfc", np.asarray(0x7fc01234, dtype=np.uint32).view(np.float32)[()])])
def test_diagnostic_nan_words_match_numpy(name, value):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    _, driver, atmosphere, _ = _case(17, 9, "warm")
    driver.fields[name][0, 0] = cp.float32(value)
    reference = _copy(driver.fields)
    _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, 2)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 2)
    _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("reverse", [False, True])
def test_diagnostic_signed_zero_ties(reverse):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    _, driver, atmosphere, _ = _case(17, 9, "warm")
    driver.fields["lakemask"][0, 0] = 1.0
    driver.fields["xice"][0, 0] = 0.0
    driver.fields["tsk"][0, 0] = cp.float32(0.0 if reverse else -0.0)
    driver.fields["qsfc"][0, 0] = cp.float32(0.0 if reverse else -0.0)
    driver.fields["chs2"][0, 0] = cp.float32(1.0e-7)
    driver.fields["cqs2"][0, 0] = cp.float32(1.0e-7)
    atmosphere["temperature"][0, 0, 0] = cp.float32(-0.0 if reverse else 0.0)
    atmosphere["qv"][0, 0, 0] = cp.float32(-0.0 if reverse else 0.0)
    reference = _copy(driver.fields)
    _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, 2)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 2)
    _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("kind", ["input", "category", "output", "driver_output",
                                  "qsn_overflow", "epilogue_qsn", "surface_derived_nan", "precedence"])
def test_refusal_matches_and_never_commits(kind, monkeypatch):
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    _, driver, atmosphere, _ = _case(17, 9, "warm")
    # These first columns are bare land in the warm fixture.
    driver.fields["xice"].fill(0)
    if kind in ("input", "precedence"):
        driver.fields["smois"][0, 0, 0] = cp.nan
    if kind in ("category", "precedence"):
        driver.fields["ivgtyp"][0, 1] = 100
    if kind == "output":
        # qkms is built inside the driver as FLQC/RHO/MAVAIL.  Finite zero
        # inputs produce a NaN, which sfctmp admits before any leaf runs.
        driver.fields["flqc"][0, 0] = 0.0
        driver.fields["mavail"][0, 0] = 0.0
    if kind == "surface_derived_nan":
        driver.fields["ivgtyp"][0, 0] = 1
        driver.fields["shdmin"][0, 0] = -cp.finfo(cp.float32).max
        driver.fields["shdmax"][0, 0] = cp.finfo(cp.float32).max
        driver.fields["vegfra"][0, 0] = cp.finfo(cp.float32).max
    k = 2
    if kind == "qsn_overflow":
        # The prologue qsn refusal must precede the overflowing TSNAV and
        # every sfctmp or driver-output check in this lake column.
        k = 1
        driver.fields["lakemask"][0, 0] = 1.0
        driver.fields["tsk"][0, 0] = cp.finfo(cp.float32).max
        driver.fields["tslb"][0, 0, 0] = cp.finfo(cp.float32).max
    if kind == "epilogue_qsn":
        driver.fields["xland"][0, 0] = 2.0
        atmosphere["temperature"][0, 0, 0] = cp.finfo(cp.float32).max
    if kind == "driver_output":
        from dataclasses import replace
        from gpuwm.core import ruc
        original = ruc.ruc_surface_temperature_step
        def large_finite_flux(*args, **kwargs):
            result = original(*args, **kwargs)
            return replace(result, eeta=cp.full_like(result.eeta, cp.finfo(cp.float32).max))
        monkeypatch.setattr(ruc, "ruc_surface_temperature_step", large_finite_flux)
        # The fused path runs sfctmp as kernels, not through the host
        # function, so the same overflow is injected at its hook.
        from gpuwm.core import ruc_fused
        fused = ruc_fused._RUC_SFCTMP_FULL_WIDTH
        def large_finite_flux_fused(*args, **kwargs):
            result = fused(*args, **kwargs)
            result["eeta"] = cp.full_like(result["eeta"], cp.finfo(cp.float32).max)
            return result
        monkeypatch.setattr(ruc_fused, "_RUC_SFCTMP_FULL_WIDTH", large_finite_flux_fused)
    reference = _copy(driver.fields)
    before = _copy(driver.fields)
    with pytest.raises(Exception) as expected:
        _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, k)
    with pytest.raises(type(expected.value)) as actual:
        _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, k)
    assert str(actual.value) == str(expected.value)
    _equal(driver.fields, before)
    _equal(reference, before)


@pytest.mark.gpu
@requires_gpu
def test_old_surface_leaf_differs_on_derived_nan():
    import cupy as cp
    from gpuwm.core import ruc, ruc_gpu
    maximum = cp.finfo(cp.float32).max
    args = (cp.ones(1, dtype=cp.int32), cp.ones(1, dtype=cp.int32),
            cp.full(1, -maximum, dtype=cp.float32),
            cp.full(1, maximum, dtype=cp.float32),
            cp.full(1, maximum, dtype=cp.float32),
            cp.full(1, 0.1, dtype=cp.float32), cp.full(1, 2.0, dtype=cp.float32))
    driver = ruc.ruc_surface_parameters(*args, arrays=ruc_gpu.RUC_DEVICE_ARRAYS)
    leaf = ruc_gpu.ruc_surface_parameters_cuda(*args)
    assert bool(cp.isnan(driver.lai).all())
    assert bool(cp.isfinite(leaf.lai).all())


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
@pytest.mark.parametrize("width", [1, 17])
@pytest.mark.parametrize("surface", ["water", "lake"])
def test_no_sfctmp_columns(width, nzs, surface):
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    bench, driver, atmosphere, cold = _case(width, nzs, "warm")
    driver.fields["xice"].fill(0)
    driver.fields["xland"].fill(2 if surface == "water" else 1)
    driver.fields["lakemask"].fill(1 if surface == "lake" else 0)
    reference = _copy(driver.fields)
    for k in range(1, 8):
        bench.forcing(driver, k, 11, (1, width), cold)
        bench.forcing(SimpleNamespace(fields=reference), k, 11, (1, width), cold)
        assert _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, k) == (
            _call(_ruc_lsm_step_reference, reference, atmosphere, driver.ruc_params, k))
        _equal(driver.fields, reference)


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
def test_the_forecast_call_is_six_kernels_and_no_orchestration_reads(nzs, monkeypatch):
    """The speed property itself, held so it cannot quietly regress.

    The array orchestration this path replaced launched about 1,800 kernels
    per call on a production-shaped grid and made 17 to 41 blocking reads
    (admission batch flushes, dispatch-arm index conversions and leaf
    checks).  The fused call is six kernels and reads the host only for the
    SFCDIAGS power pair's input and the one flag slab at the end.
    """
    import cupy as cp
    from gpuwm.core import ruc, ruc_fused, ruc_tier, ruc_validation
    from gpuwm.core.ruc_runtime import ruc_lsm_step
    width = 5917
    bench, driver, atmosphere, cold = _case(width, nzs, "mixed")
    bench.forcing(driver, 1, 11, (1, width), cold)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 1)
    launched = []
    original = ruc_tier.ruc_fused_kernel

    def counting(func, levels, *lineage):
        kernel = original(func, levels, *lineage)

        def launch(*args):
            launched.append(func)
            return kernel(*args)
        return launch

    def forbidden(*args, **kwargs):
        raise AssertionError("the fused RUC call took an orchestration read")

    monkeypatch.setattr(ruc_tier, "ruc_fused_kernel", counting)
    monkeypatch.setattr(ruc_fused, "ruc_fused_kernel", counting)
    monkeypatch.setattr(ruc_validation, "_host_list", forbidden)
    monkeypatch.setattr(ruc, "_selected", forbidden)
    monkeypatch.setattr(cp, "asnumpy", forbidden)
    bench.forcing(driver, 2, 11, (1, width), cold)
    _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, 2)
    assert launched == ["ruc_driver_prologue", "ruc_sfctmp_stage0",
                        "ruc_sfctmp_stage1", "ruc_sfctmp_stage2",
                        "ruc_driver_epilogue", "ruc_driver_commit"], launched


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [6, 9])
def test_t2_and_th2_do_not_reach_the_hosts_numpy_power(nzs, monkeypatch):
    # A145.  The call's two host powers (SFCDIAGS_RUCLSM's exner factors)
    # were NumPy's float32 power, the host's own function, so an AVX-512
    # product box wrote other T2 and TH2 words.  Here every np.power answer
    # moves one ULP, as another host's would, and no field may move.
    from gpuwm.core.ruc_runtime import ruc_lsm_step
    bench, driver, atmosphere, cold = _case(5917, nzs, "mixed")
    moved = _copy(driver.fields)
    real = np.power

    def another_hosts_power(*args, **kwargs):
        answer = np.asarray(real(*args, **kwargs))
        return np.nextafter(answer, np.asarray(np.inf, dtype=answer.dtype))

    for k in range(1, 4):
        bench.forcing(driver, k, 11, (1, 5917), cold)
        bench.forcing(SimpleNamespace(fields=moved), k, 11, (1, 5917), cold)
        _call(ruc_lsm_step, driver.fields, atmosphere, driver.ruc_params, k)
        with monkeypatch.context() as patch:
            patch.setattr(np, "power", another_hosts_power)
            assert np.power(np.float32(1.5), np.float32(0.3)) != real(
                np.float32(1.5), np.float32(0.3))
            _call(ruc_lsm_step, moved, atmosphere, driver.ruc_params, k)
        _equal(driver.fields, moved)


def _operational_params(params, nzs, *, rdlai2d, fractional_seaice):
    from gpuwm.core.ruc_runtime import RucRuntimeParameters
    resolved = RucRuntimeParameters(
        params.bundle, dataset_identifier=params.dataset_identifier,
        seaice_albedo_default=params.seaice_albedo_default,
        num_soil_layers=nzs, rdlai2d=rdlai2d,
        fractional_seaice=fractional_seaice)
    resolved.iswater, resolved.isice = params.iswater, params.isice
    return resolved


def _fractional_ice_and_monthly_lai(fields, width):
    """Sea-ice fractions across WRF's 0.02 threshold and a prescribed LAI.

    Every fractional ice cell of the mixed fixture moves into [0.02, 0.5),
    the cells the 0.5 pin left open water and HRRR's fractional_seaice = 1
    blends; two sit exactly on 0.02 and one just below it.  The LAI is a
    field no table produces, as real.exe's LAI12M interpolation is.
    """
    import cupy as cp
    rng = np.random.default_rng(29)
    xice = cp.asnumpy(fields["xice"]).copy()
    partial = (xice > 0.0) & (xice < 1.0)
    xice[partial] = rng.uniform(0.02, 0.5, xice.shape)[partial].astype(np.float32)
    edge = np.flatnonzero(partial.reshape(-1))[:3]
    flat = xice.reshape(-1)
    flat[edge] = np.asarray([0.02, 0.02, np.nextafter(np.float32(0.02),
                                                      np.float32(0.0))],
                            dtype=np.float32)[:edge.size]
    fields["xice"][...] = cp.asarray(xice, dtype=fields["xice"].dtype)
    # landuse_init's fractional blend (module_physics_init.F:1638-1639) for
    # the moved cells, so the driver's de-blend recovers an ice albedo.
    one = np.float32(1.0)
    moved = cp.asarray(partial)
    fraction = fields["xice"]
    fields["albedo"][...] = cp.where(
        moved, fraction * np.float32(0.55) + (one - fraction) * np.float32(0.08),
        fields["albedo"])
    fields["emiss"][...] = cp.where(
        moved, fraction * np.float32(0.98) + (one - fraction) * np.float32(0.98),
        fields["emiss"])
    fields["lai"][...] = cp.asarray(
        rng.uniform(0.05, 4.5, (1, width)), dtype=fields["lai"].dtype)
    return int(np.count_nonzero((xice >= np.float32(0.02))
                                & (xice < np.float32(0.5))))


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("nzs", [9, 6])
@pytest.mark.parametrize("width", [17, 5917])
def test_rdlai2d_and_the_fractional_threshold_fused_matches_reference(width, nzs):
    # Operational HRRR's RUC switches (hrrr_wrf.nl:149 rdlai2d, :154
    # fractional_seaice = 1 -> xice_threshold 0.02 at
    # module_surface_driver.F:1365-1368): the fused driver must write every
    # word the per-kernel reference path writes, on cells the new threshold
    # admits, and SOILVEGIN must leave the prescribed LAI alone
    # (module_sf_ruclsm.F:7028, :7042, :7061, :7075).
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step, _ruc_lsm_step_reference
    bench, driver, atmosphere, cold = _case(width, nzs, "mixed")
    params = _operational_params(driver.ruc_params, nzs, rdlai2d=True,
                                 fractional_seaice=1)
    assert params.xice_threshold == 0.02
    admitted = _fractional_ice_and_monthly_lai(driver.fields, width)
    assert admitted > 0
    reference = _copy(driver.fields)
    prescribed = cp.asnumpy(driver.fields["lai"]).copy()
    for k in range(1, 8):
        bench.forcing(driver, k, 11, (1, width), cold)
        bench.forcing(SimpleNamespace(fields=reference), k, 11, (1, width), cold)
        actual = _call(ruc_lsm_step, driver.fields, atmosphere, params, k)
        expected = _call(_ruc_lsm_step_reference, reference, atmosphere, params, k)
        assert actual == expected
        _equal(driver.fields, reference)
        np.testing.assert_array_equal(
            cp.asnumpy(driver.fields["lai"]).view(np.uint32),
            prescribed.view(np.uint32))


@pytest.mark.gpu
@requires_gpu
def test_the_new_switches_are_live_on_the_fused_path():
    # Negative control for the test above: the same fields under the
    # previous pins (rdlai2d false, threshold 0.5) must write other words,
    # in the LAI everywhere on land and in the fluxes of the admitted cells.
    import cupy as cp
    from gpuwm.core.ruc_runtime import ruc_lsm_step
    width, nzs = 5917, 9
    bench, driver, atmosphere, cold = _case(width, nzs, "mixed")
    _fractional_ice_and_monthly_lai(driver.fields, width)
    pinned = _copy(driver.fields)
    xice = cp.asnumpy(driver.fields["xice"]).reshape(-1)
    admitted = (xice >= np.float32(0.02)) & (xice < np.float32(0.5))
    new = _operational_params(driver.ruc_params, nzs, rdlai2d=True,
                              fractional_seaice=1)
    old = _operational_params(driver.ruc_params, nzs, rdlai2d=False,
                              fractional_seaice=0)
    bench.forcing(driver, 1, 11, (1, width), cold)
    bench.forcing(SimpleNamespace(fields=pinned), 1, 11, (1, width), cold)
    _call(ruc_lsm_step, driver.fields, atmosphere, new, 1)
    _call(ruc_lsm_step, pinned, atmosphere, old, 1)
    lai_new = cp.asnumpy(driver.fields["lai"]).reshape(-1)
    lai_old = cp.asnumpy(pinned["lai"]).reshape(-1)
    assert np.count_nonzero(lai_new != lai_old) > width // 2
    for name in ("hfx", "tsk", "albedo"):
        moved = (cp.asnumpy(driver.fields[name]).reshape(-1)
                 != cp.asnumpy(pinned[name]).reshape(-1))
        assert np.count_nonzero(moved & admitted) > 0, name
