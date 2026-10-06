"""Original fused RUC versus real member-indexed science launches."""
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _case(width, nzs, scenario, seed):
    import cupy as cp
    spec = spec_from_file_location("packed_ruc_fixture", Path(__file__).with_name("ruc_fused_fixture.py"))
    fixture = module_from_spec(spec)
    spec.loader.exec_module(fixture)
    _, _, driver, atmosphere, cold = fixture.build(97, 61, 20, nzs, scenario, seed)
    index = cp.arange(width) % (97 * 61)
    def resize(array):
        if isinstance(array, cp.ndarray) and array.shape[-2:] == (61, 97):
            return cp.ascontiguousarray(array.reshape(array.shape[:-2] + (-1,))[..., index]
                                       .reshape(array.shape[:-2] + (1, width)))
        return array.copy() if hasattr(array, "copy") else array
    fields = {name: resize(value) for name, value in driver.fields.items()}
    atmosphere = {name: resize(value) for name, value in atmosphere.items()}
    cold = cold.reshape(-1)[np.arange(width) % (97 * 61)].reshape(1, width)
    return fixture, SimpleNamespace(fields=fields, ruc_params=driver.ruc_params), atmosphere, cold


def _stack(mappings):
    import cupy as cp
    return {name: cp.stack([value[name] for value in mappings])
            if isinstance(mappings[0][name], cp.ndarray) else mappings[0][name]
            for name in mappings[0]}


def _equal(actual, expected):
    import cupy as cp
    for name, array in expected.items():
        if isinstance(array, cp.ndarray):
            np.testing.assert_array_equal(cp.asnumpy(actual[name]).view(np.uint8),
                                          cp.asnumpy(array).view(np.uint8), err_msg=name)


def _ordinary(driver, atmosphere, dt, count, lineage, **options):
    from gpuwm.core.ruc_fused import step
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing
    return step(driver.fields, atmosphere, params=driver.ruc_params,
                precipitation=SurfacePrecipitationForcing.from_fields(driver.fields),
                dt=dt, itimestep=count, lakemodel=1, soilprop=lineage, **options)


@pytest.mark.parametrize("members,width,nzs,scenario,lineage", [
    (4, 17, 6, "warm", "wrf_45"), (4, 5917, 6, "mixed", "wrf_45"),
    (8, 17, 6, "mixed", "wrf_45"), (2, 17, 9, "mixed", "wrf_461")])
def test_all_field_words_and_census_match_each_member_alone(members, width, nzs, scenario, lineage, monkeypatch):
    import cupy as cp
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing
    from gpuwm.ensemble import batch_ruc
    from gpuwm.ensemble.batch_ruc import PackedRucDriver, workspace_allocations
    from gpuwm.core.ruc_memory import allocation_bytes
    cases = [_case(width, nzs, scenario, 11 + member) for member in range(members)]
    if width == 5917 and scenario == "mixed":
        # Exercise the operational prescribed-LAI and 0.02 fractional-ice
        # hooks, while keeping the table and source lineage common.
        from gpuwm.core.ruc_runtime import RucRuntimeParameters
        for _, driver, _, _ in cases:
            original_params = driver.ruc_params
            params = RucRuntimeParameters(original_params.bundle,
                dataset_identifier=original_params.dataset_identifier,
                seaice_albedo_default=original_params.seaice_albedo_default,
                num_soil_layers=nzs, rdlai2d=True, fractional_seaice=1)
            params.iswater, params.isice = original_params.iswater, original_params.isice
            driver.ruc_params = params
            driver.fields["lai"].fill(cp.float32(1.375))
            partial = (driver.fields["xice"] > 0) & (driver.fields["xice"] < 1)
            driver.fields["xice"][...] = cp.where(partial, cp.float32(0.15), driver.fields["xice"])
            fraction = driver.fields["xice"]
            driver.fields["albedo"][...] = cp.where(partial,
                fraction * cp.float32(0.55) + (cp.float32(1) - fraction) * cp.float32(0.08),
                driver.fields["albedo"])
    fields = _stack([driver.fields for _, driver, _, _ in cases])
    atmosphere = _stack([atmosphere for _, _, atmosphere, _ in cases])
    bank = PackedRucDriver(members, (1, width), nzs=nzs,
        available_bytes=allocation_bytes(workspace_allocations(members, (1, width), nzs)))
    assert {name: (array.shape, str(array.dtype)) for name, array in bank.arrays.items()} == workspace_allocations(members, (1, width), nzs)
    assert len({array.data.ptr for array in bank.arrays.values()}) == len(bank.arrays)
    assert sum(array.nbytes for array in bank.arrays.values()) == allocation_bytes(workspace_allocations(members, (1, width), nzs), rounded=False)
    launched, original = [], batch_ruc._module
    class CountModule:
        def __init__(self, module): self.module = module
        def get_function(self, name):
            kernel = self.module.get_function(name)
            def launch(grid, block, args):
                launched.append((name, grid))
                return kernel(grid, block, args)
            return launch
    def measured(*args, **kwargs):
        module, digest = original(*args, **kwargs)
        return CountModule(module), digest
    monkeypatch.setattr(batch_ruc, "_module", measured)
    try:
        for call in range(1, 8):
            # Both paths receive the same independently changing forcings.
            for member, (fixture, driver, _, cold) in enumerate(cases):
                fixture.forcing(driver, call, 11 + member, (1, width), cold)
                packed_driver = SimpleNamespace(fields={name: value[member] if isinstance(value, cp.ndarray) else value
                                                       for name, value in fields.items()})
                fixture.forcing(packed_driver, call, 11 + member, (1, width), cold)
            dt = np.asarray([6 + 2 * member + call for member in range(members)], dtype=np.float32)
            counts = np.asarray([call + 3 * member for member in range(members)], dtype=np.int32)
            expected = tuple(_ordinary(driver, atmo, dt[member], counts[member], lineage)
                             for member, (_, driver, atmo, _) in enumerate(cases))
            launched.clear()
            census = bank.step(fields, atmosphere, params=cases[0][1].ruc_params,
                precipitation=SurfacePrecipitationForcing.from_fields(fields), dt=dt,
                itimestep=counts, lakemodel=1, soilprop=lineage)
            assert census == expected
            assert [name for name, _ in launched] == ["packed_" + name for name in batch_ruc.ENTRIES[:4]] + ["packed_ruc_bind_exner"] + ["packed_" + name for name in batch_ruc.ENTRIES[4:]]
            assert all(grid == ((width + 127) // 128, members) for _, grid in launched)
            for member, (_, driver, _, _) in enumerate(cases):
                _equal({name: value[member] if isinstance(value, cp.ndarray) else value for name, value in fields.items()}, driver.fields)
            assert bank.last_receipt["science_launches"] == 6
            assert bank.last_receipt["members"] == members
    finally:
        bank.close()
        assert not bank.arrays and not bank.driver and not bank.sf_scratch
        assert bank.flag_slab is None and bank.integer is None and bank.run is None


@pytest.mark.parametrize("options", [{}, {"irrigation": "wrf_45", "qvg_cold_start": "air",
                                          "diagnostic_2m": "log_profile", "snow": "wrf_45"}])
def test_one_member_refusal_preserves_its_words_and_other_member_commit(options):
    import cupy as cp
    from gpuwm.core.ruc_memory import allocation_bytes
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing
    from gpuwm.ensemble.batch_ruc import PackedRucDriver, workspace_allocations
    cases = [_case(17, 6, "warm", 21 + member) for member in range(2)]
    cases[1][1].fields["smois"][0, 0, 0] = cp.nan
    fields, atmosphere = _stack([case[1].fields for case in cases]), _stack([case[2] for case in cases])
    before = {name: array.copy() for name, array in cases[1][1].fields.items() if isinstance(array, cp.ndarray)}
    good = _ordinary(cases[0][1], cases[0][2], 12, 2, "wrf_45", **options)
    with pytest.raises(ValueError) as ordinary:
        _ordinary(cases[1][1], cases[1][2], 12, 2, "wrf_45", **options)
    bank = PackedRucDriver(2, (1, 17), available_bytes=allocation_bytes(workspace_allocations(2, (1, 17))))
    try:
        with pytest.raises(ValueError) as packed:
            bank.step(fields, atmosphere, params=cases[0][1].ruc_params,
                precipitation=SurfacePrecipitationForcing.from_fields(fields), dt=[12, 12], itimestep=[2, 2], lakemodel=1, **options)
        assert str(packed.value) == str(ordinary.value)
        assert packed.value.member_id == 1
        _equal({name: array[0] for name, array in fields.items() if isinstance(array, cp.ndarray)}, cases[0][1].fields)
        _equal({name: array[1] for name, array in fields.items() if isinstance(array, cp.ndarray)}, before)
    finally:
        bank.close()


def test_bank_refuses_an_unreserved_allocation_and_another_stream():
    import cupy as cp
    from gpuwm.core.ruc_memory import allocation_bytes
    from gpuwm.ensemble.batch_ruc import PackedRucDriver, workspace_allocations
    required = allocation_bytes(workspace_allocations(2, (1, 17)))
    with pytest.raises(MemoryError, match="admitted device reservation"):
        PackedRucDriver(2, (1, 17), available_bytes=required - 1)
    bank = PackedRucDriver(2, (1, 17), available_bytes=required)
    try:
        with cp.cuda.Stream(non_blocking=True):
            with pytest.raises(ValueError, match="owning device and stream"):
                bank._owned()
    finally:
        bank.close()


@pytest.mark.parametrize("irrigation", ["wrf_45", "wrf_461"])
@pytest.mark.parametrize("qvg_cold_start", ["air", "wrf"])
@pytest.mark.parametrize("diagnostic_2m", ["flux", "log_profile"])
@pytest.mark.parametrize("snow", ["wrf_45", "wrf_461"])
def test_current_four_selectors_match_every_standalone_field_over_changing_periods(
        irrigation, qvg_cold_start, diagnostic_2m, snow):
    _selector_words(4, irrigation, qvg_cold_start, diagnostic_2m, snow)


def test_eight_members_keep_private_legacy_irrigation_air_start_log_profile_and_snow_words():
    _selector_words(8, "wrf_45", "air", "log_profile", "wrf_45")


def _selector_counts(call, members):
    # Every missing ground-humidity carrier must enter the stock ktau==1
    # initialization before later members use divergent carried clocks.
    return np.asarray([1 if call == 1 else call + member for member in range(members)], np.int32)


def _selector_words(members, irrigation, qvg_cold_start, diagnostic_2m, snow):
    import cupy as cp
    from gpuwm.core.ruc_memory import allocation_bytes
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing
    from gpuwm.ensemble.batch_ruc import PackedRucDriver, workspace_allocations
    width, nzs = 5917, 6
    cases = [_case(width, nzs, "mixed", 73 + member) for member in range(members)]
    for member, (_, driver, _, _) in enumerate(cases):
        # The two cold-start branches must actually encounter missing ground
        # vapour/condensate, and stable columns exercise the log diagnostics.
        driver.fields["qvg"].fill(cp.float32(-1))
        driver.fields["qcg"].fill(cp.float32(-1))
        vegetation = driver.ruc_params.bundle.vegetation_for(driver.ruc_params.dataset_identifier)
        crop, natural = (int(vegetation.scalars[name]) for name in ("CROP", "NATURAL"))
        count = len(vegetation.rows)
        driver.fields["landusef"] = cp.zeros((count, 1, width), cp.float32)
        driver.fields["landusef"][crop - 1].fill(cp.float32((.1, .2, .3, .4, .5, .6, .7, .8)[member]))
        driver.fields["landusef"][natural - 1].fill(cp.float32((.9, .8, .7, .6, .5, .4, .3, .2)[member]))
    fields = _stack([case[1].fields for case in cases])
    atmosphere = _stack([case[2] for case in cases])
    options = dict(irrigation=irrigation, qvg_cold_start=qvg_cold_start,
                   diagnostic_2m=diagnostic_2m, snow=snow)
    bank = PackedRucDriver(members, (1, width), nzs=nzs,
        available_bytes=allocation_bytes(workspace_allocations(members, (1, width), nzs)))
    try:
        for call in range(1, 5):
            for member, (fixture, driver, _, cold) in enumerate(cases):
                fixture.forcing(driver, call, 73 + member, (1, width), cold)
                packed = SimpleNamespace(fields={name: value[member] if isinstance(value, cp.ndarray) else value
                                                for name, value in fields.items()})
                fixture.forcing(packed, call, 73 + member, (1, width), cold)
            dt = np.asarray((9, 13, 17, 21, 11, 15, 19, 23)[:members], np.float32)
            counts = _selector_counts(call, members)
            expected = tuple(_ordinary(driver, atmo, dt[member], counts[member], "wrf_45", **options)
                for member, (_, driver, atmo, _) in enumerate(cases))
            actual = bank.step(fields, atmosphere, params=cases[0][1].ruc_params,
                precipitation=SurfacePrecipitationForcing.from_fields(fields), dt=dt,
                itimestep=counts, lakemodel=1, **options)
            assert actual == expected
            assert all(all(census[name] > 0 for name in ("land", "water", "lake", "sea_ice")) for census in actual)
            for member, (_, driver, _, _) in enumerate(cases):
                _equal({name: value[member] if isinstance(value, cp.ndarray) else value
                        for name, value in fields.items()}, driver.fields)
            assert bank.last_receipt["selectors"] == {"ruc_irrigation": irrigation,
                "ruc_soilprop": "wrf_45", "ruc_qvg_cold_start": qvg_cold_start,
                "ruc_2m_diagnostic": diagnostic_2m, "ruc_snow": snow}
            assert bank.last_receipt["landusef_member_private"] is (irrigation == "wrf_45")
    finally:
        bank.close()


@pytest.mark.parametrize("name,values", [
    ("irrigation", ["wrf_45", "wrf_461"]), ("qvg_cold_start", ["air", "wrf"]),
    ("diagnostic_2m", ["flux", "log_profile"]), ("snow", ["wrf_45", "wrf_461"]),
])
def test_mixed_selector_metadata_refuses_without_committing_any_member(name, values):
    import cupy as cp
    from gpuwm.core.ruc_memory import allocation_bytes
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing
    from gpuwm.ensemble.batch_ruc import PackedRucDriver, workspace_allocations
    cases = [_case(17, 6, "mixed", 31 + member) for member in range(2)]
    fields, atmosphere = _stack([case[1].fields for case in cases]), _stack([case[2] for case in cases])
    before = {name: value.copy() for name, value in fields.items() if isinstance(value, cp.ndarray)}
    bank = PackedRucDriver(2, (1, 17), available_bytes=allocation_bytes(workspace_allocations(2, (1, 17))))
    try:
        with pytest.raises(ValueError):
            bank.step(fields, atmosphere, params=cases[0][1].ruc_params,
                precipitation=SurfacePrecipitationForcing.from_fields(fields), dt=[12, 12],
                itimestep=[1, 1], **{name: values})
        _equal(fields, before)
    finally:
        bank.close()
