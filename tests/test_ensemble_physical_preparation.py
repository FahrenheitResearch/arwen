"""Native source alignment, storage and member selection proofs."""
from dataclasses import replace
from datetime import datetime, timedelta
import json
import os

import numpy as np
import pytest

from gpuwm.ensemble.native_preparation import NativeEnsemblePreparation
from gpuwm.ensemble.physical_recenter import RecenteredPhysicalPreparation
from gpuwm.ensemble.physical_store import NativePhysicalStore, digest_file, validate_physical_input_binding, physical_static_identity
from gpuwm.ensemble.recentered import FieldBounds, recenter_field
from gpuwm.ingest.horiz import HorizontalSnapshot

pytestmark = pytest.mark.skipif(not os.environ.get("GPUWM_ENSEMBLE_PREPARATION_BRIDGE"),
                                reason="requires the built native ensemble preparation bridge")


@pytest.fixture
def bridge():
    return os.environ["GPUWM_ENSEMBLE_PREPARATION_BRIDGE"]


def snapshot(hour=0, offset=0.0):
    levels = np.array([900., 800., 700., 600., 500.], dtype=np.float64)
    fields = {
        "PRES": np.broadcast_to((levels*100).astype("f4")[:,None,None], (5,2,3)).copy(),
        "PSFC": np.full((2,3), 100000., "f4"),
        "TT": np.full((5,2,3), 280.+offset, "f4"),
        "SPFH": np.full((5,2,3), 0.006, "f4"),
        "UU": np.full((5,2,4), 3.+offset, "f4"),
        "VV": np.full((5,3,3), 2.-offset, "f4"),
        "GHT": np.broadcast_to(np.array([1000.,2000.,3000.,4000.,5000.], "f4")[:,None,None],(5,2,3)).copy(),
        "T2": np.full((2,3), 280.+offset, "f4"),
        "Q2": np.full((2,3), 0.006, "f4"),
        "U10": np.full((2,4), 3.+offset, "f4"),
        "V10": np.full((3,3), 2.-offset, "f4"),
        "SOURCE_OROGRAPHY": np.zeros((2,3), "f8"),
        "ST000010": np.full((2,3), 285., "f4"),
    }
    return HorizontalSnapshot(datetime(2024,1,1)+timedelta(hours=hour), levels, fields,
                              analyzed_species=(), specific_humidity_authority=True,
                              specific_humidity_undershoot_floor=0.,
                              soil_no_source_land=np.zeros((2,3), bool))


def make_store(root, frames):
    from physical_field_fixtures import analytic_field_contract
    grid = {"mass_shape":[2,3],"fixture":"native-field-test"}
    contract = analytic_field_contract(grid)
    if "PRES" not in frames[0].fields:
        contract["vertical"]["pressure_field"] = None
    store = NativePhysicalStore(root, grid_identity=grid,
                                source_identity={"input_manifest_sha256":"a"*64,
                                    "static_identity":physical_static_identity({"HGT_M":np.zeros((2,3))})},
                                field_contract=contract)
    for frame in frames:
        store.write(frame)
    store.seal()
    return root


def test_native_store_exact_roundtrip_and_changed_frame_refusal(tmp_path, bridge):
    original = snapshot()
    root = make_store(tmp_path/"store", [original])
    store = NativePhysicalStore(root)
    read = store.read(0)
    for name, value in original.fields.items():
        assert read.fields[name].dtype == value.dtype
        assert read.fields[name].tobytes() == value.tobytes()
    assert read.soil_no_source_land.dtype == np.dtype(bool)
    assert read.soil_no_source_land.tobytes() == original.soil_no_source_land.tobytes()
    path = root/store.document["frames"][0]["file"]
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="differ from the sealed"):
        store.read(0)


def test_native_population_partition_and_worker_identity(bridge):
    random = np.random.default_rng(421)
    base = random.uniform(-4,4,(7,9)).astype("f4")
    donors = random.uniform(-9,9,(30,7,9)).astype("f4")
    ids = tuple(f"p{i:02d}" for i in range(1,31))
    kwargs = dict(donor_ids=ids, bounds=FieldBounds("m s-1", 1.3,-10.,10.,1.7),
                  mapped_grid_sha256="a"*64, array_module=np, cpu_bridge=bridge)
    all_values, _ = recenter_field(base,donors,selected_ids=ids,workers=2,**kwargs)
    one, _ = recenter_field(base,donors,selected_ids=(ids[17],),workers=1,**kwargs)
    assert one[0].tobytes() == all_values[17].tobytes()
    assert np.all(np.abs(all_values.astype("f8")-base) <= 1.3)
    off, _ = recenter_field(base,donors,selected_ids=ids,workers=2,
                            **{**kwargs,"bounds":FieldBounds("m s-1",1.3,-10.,10.,0.)})
    assert all(value.tobytes() == base.tobytes() for value in off)


def test_native_time_and_pressure_stagger(bridge):
    native = NativeEnsemblePreparation(bridge)
    left = np.array([-0.,1.,2.,3.], "f4")
    right = np.array([4.,5.,6.,7.], "f4")
    assert native.time_blend(left,right,0.).tobytes() == left.tobytes()
    assert native.time_blend(left,right,1.).tobytes() == right.tobytes()
    np.testing.assert_array_equal(native.time_blend(left,right,.25), [1.,2.,3.,4.])
    pressure = np.array([[[100.,200.,300.],[400.,500.,600.]]], "f4")
    np.testing.assert_array_equal(native.pressure_stagger(pressure,2), [[[100.,150.,250.,300.],[400.,450.,550.,600.]]])
    np.testing.assert_array_equal(native.pressure_stagger(pressure,1), [[[100.,200.,300.],[250.,350.,450.],[400.,500.,600.]]])
    levels = native.pressure_levels(np.array([1000.,850.,500.]), (2,3))
    np.testing.assert_array_equal(levels[:,0,0], [100000.,85000.,50000.])
    assert all(levels[:,j,i].tobytes()==levels[:,0,0].tobytes() for j in range(2) for i in range(3))


def test_physical_time_alignment_subset_replay_and_portable_binding(tmp_path, bridge):
    base = make_store(tmp_path/"base",[snapshot(0),snapshot(1),snapshot(2)])
    donors = {
        "p01":make_store(tmp_path/"donor1",[snapshot(0,-1.),snapshot(3,-4.)]),
        "p02":make_store(tmp_path/"donor2",[snapshot(0,1.),snapshot(3,4.)])}
    full = RecenteredPhysicalPreparation(base,donors,amplitude=1.,cpu_bridge=bridge,workers=2)
    full.prepare({"p01":tmp_path/"full1","p02":tmp_path/"full2"})
    single = RecenteredPhysicalPreparation(base,donors,amplitude=1.,cpu_bridge=bridge,workers=1)
    single.prepare({"p02":tmp_path/"single"})
    first, second = NativePhysicalStore(tmp_path/"full2"), NativePhysicalStore(tmp_path/"single")
    for index in range(3):
        all_frame, one_frame = first.read(index), second.read(index)
        for name in all_frame.fields:
            assert all_frame.fields[name].tobytes() == one_frame.fields[name].tobytes()
        np.testing.assert_array_equal(all_frame.fields["T2"], 281.+index)
        assert all_frame.fields["PRES"].tobytes() == snapshot(index).fields["PRES"].tobytes()
    binding = {"schema":"gpuwm-ensemble-physical-input-binding.v1", "manifest":first.document,
               "manifest_sha256":digest_file(first.manifest_path)}
    validate_physical_input_binding(binding,input_manifest_sha256="a"*64)
    with pytest.raises(ValueError, match="base source differs"):
        validate_physical_input_binding(binding,input_manifest_sha256="b"*64)
    broken = json.loads(json.dumps(binding))
    broken["manifest"]["source"]["selected_member"]="p03"
    with pytest.raises(ValueError, match="bound digest"):
        validate_physical_input_binding(broken,input_manifest_sha256="a"*64)


@pytest.mark.parametrize("specific_authority", [False, True])
def test_native_humidity_undershoot_is_preserved(tmp_path, bridge, specific_authority):
    initial = snapshot()
    fields = dict(initial.fields)
    fields["SPFH"] = fields["SPFH"].copy()
    fields["SPFH"][0, 0, 0] = np.float32(-2.9e-6)
    fields["SPFH"][0, 0, 1] = np.float32(0.)
    initial = replace(initial, fields=fields, specific_humidity_authority=specific_authority,
                      specific_humidity_undershoot_floor=None)
    base = make_store(tmp_path/"base", [initial])
    donor1, donor2 = snapshot(offset=-1.), snapshot(offset=1.)
    donor1.fields["SPFH"][:] = 0.001
    donor2.fields["SPFH"][:] = 0.009
    donors = {"p01":make_store(tmp_path/"donor1", [donor1]),
              "p02":make_store(tmp_path/"donor2", [donor2])}
    operator = RecenteredPhysicalPreparation(base, donors, amplitude=1., cpu_bridge=bridge)
    invalid_bounds=dict(operator.bounds,TT=replace(operator.bounds["TT"],units="m"))
    with pytest.raises(ValueError,match="bounds units differ"):
        RecenteredPhysicalPreparation(base,donors,amplitude=1.,cpu_bridge=bridge,bounds=invalid_bounds)
    operator.prepare({key:tmp_path/key for key in donors})
    for key in donors:
        store = NativePhysicalStore(tmp_path/key)
        values = store.read(0).fields["SPFH"]
        assert values[0,0,0].tobytes() == initial.fields["SPFH"][0,0,0].tobytes()
        assert np.all(values[initial.fields["SPFH"] >= 0.] >= 0.)
        bounds = store.document["source"]["frame_operations"][0]["fields"]["SPFH"]["bounds"]
        assert bounds["lower"] == 0.
        assert bounds["input_lower"] == float(initial.fields["SPFH"][0,0,0])

    invalid_fields = dict(initial.fields)
    invalid_fields["SPFH"] = initial.fields["SPFH"].copy()
    invalid_fields["SPFH"][0,0,0] = -1.
    invalid = make_store(tmp_path/"invalid", [replace(initial, fields=invalid_fields)])
    with pytest.raises(ValueError, match="native initializer's undershoot envelope"):
        RecenteredPhysicalPreparation(invalid, donors, amplitude=1., cpu_bridge=bridge).prepare(
            {"p01":tmp_path/"invalid-output"})


def test_native_humidity_representation_matches_existing_initializer(bridge):
    from gpuwm.ingest import real
    native = NativeEnsemblePreparation(bridge)
    temperature = np.linspace(210.,310.,97,dtype="f4")
    pressure = np.linspace(10000.,100000.,97,dtype="f4")
    specific = np.linspace(-0.00001,0.02,97,dtype="f4")
    expected = real._mixing_ratio_to_relative_humidity(
        temperature, pressure, real._specific_humidity_to_mixing_ratio(specific, allow_wps_undershoot=True),
        allow_wps_undershoot=True).astype("f4")
    actual = native.humidity(temperature, pressure, specific, to_relative=True)
    np.testing.assert_array_equal(actual,expected)
    rh = np.linspace(-5.,105.,97,dtype="f4")
    mixing = real._saturation_mixing_ratio(temperature,pressure,rh)
    expected = (mixing/(1.+mixing)).astype("f4")
    np.testing.assert_array_equal(native.humidity(temperature,pressure,rh,to_relative=False),expected)


def test_relative_humidity_base_keeps_its_native_zero_and_clipped_words(tmp_path,bridge):
    initial = snapshot()
    fields = {key:value for key,value in initial.fields.items() if key not in ("SPFH","Q2","PRES")}
    fields["RH"] = np.full(fields["TT"].shape,50.,"f4")
    fields["RH2"] = np.full(fields["T2"].shape,50.,"f4")
    fields["RH"][0,0,:] = [-3.,104.,0.]
    initial = replace(initial,fields=fields,specific_humidity_authority=False,
                      specific_humidity_undershoot_floor=None)
    base = make_store(tmp_path/"base",[initial])
    donors = {"p01":make_store(tmp_path/"donor1",[snapshot(offset=-1.)]),
              "p02":make_store(tmp_path/"donor2",[snapshot(offset=1.)])}
    operator=RecenteredPhysicalPreparation(base,donors,amplitude=1.,cpu_bridge=bridge)
    operator.prepare_variants({
        "off":{"amplitude":0.,"outputs":{"p01":tmp_path/"off"}},
        "on":{"amplitude":1.,"outputs":{"p01":tmp_path/"on"}}})
    off=NativePhysicalStore(tmp_path/"off").read(0)
    on=NativePhysicalStore(tmp_path/"on").read(0)
    assert off.fields.keys()==on.fields.keys()==initial.fields.keys()
    assert off.specific_humidity_authority is on.specific_humidity_authority is False
    for name,value in initial.fields.items():assert off.fields[name].tobytes()==value.tobytes()
    assert on.fields["RH"][0,0,:2].tobytes()==initial.fields["RH"][0,0,:2].tobytes()
    assert on.fields["RH"][0,0,2] >= 0.
    assert not {"SPFH","PRES","Q2"} & on.fields.keys()
    separate=RecenteredPhysicalPreparation(base,donors,amplitude=1.,cpu_bridge=bridge)
    separate.prepare({"p01":tmp_path/"separate"})
    assert (tmp_path/"on/frame-0000.nc").read_bytes()==(tmp_path/"separate/frame-0000.nc").read_bytes()


def test_square_native_grid_still_rejects_transposed_temperature_authority(tmp_path,bridge):
    from physical_field_fixtures import analytic_field_contract
    initial=snapshot()
    fields={name:values[...,:3 if name in ("UU","U10") else 2].copy()
            for name,values in initial.fields.items()}
    initial=replace(initial,fields=fields,soil_no_source_land=np.zeros((2,2),bool))
    grid={"mass_shape":[2,2]}
    contract=analytic_field_contract(grid)
    contract["arrays"]["field__TT"]["dimensions"]=["level","x","y"]
    writer=NativePhysicalStore(tmp_path/"transposed",grid_identity=grid,
        source_identity={"input_manifest_sha256":"a"*64},field_contract=contract)
    writer.write(initial)
    writer.seal()
    store=NativePhysicalStore(writer.root)
    store.read(0)  # A square shape alone cannot detect its transposed semantics.
    with pytest.raises(ValueError,match="source-qualified TT.*dimensions"):
        RecenteredPhysicalPreparation._verify_field_contract(store)
