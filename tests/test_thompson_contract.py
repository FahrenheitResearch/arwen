"""CPU-only gates for the staged WRF Thompson mp=8 port."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import struct
from types import SimpleNamespace
from types import MappingProxyType

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.core import preflight
from gpuwm.core.kf import KFPhaseMode, kf_phase_mode_for_microphysics
from gpuwm.core.moist import extra_moist_species
from gpuwm.core.thompson_contract import (
    AUXILIARY_TABLE_RECORDS,
    BACKGROUND_EFFECTIVE_RADIUS_UM,
    CLASSIC_TABLE_ASSETS,
    CLASSIC_GRAUPEL_DENSITY_KG_M3,
    GENERATED_TABLE_FILES,
    MASS_SPECIES,
    NUMBER_SPECIES,
    TABLE_SET_ID,
    ClassicTableSet,
    TableAsset,
    TableRecord,
    read_sequential_records,
    sequential_file_bytes,
    validate_table_assets,
)
from gpuwm.core.thompson_runtime import (
    RAIN_FREEZING_TABLE_NAMES,
    RAIN_GRAUPEL_TABLE_NAMES,
    RAIN_SNOW_TABLE_NAMES,
    DeviceClassicTableSet,
    _upload_table_set,
)
from gpuwm.verify.npref import np_calc_cq


_TINY = dict(nx=8, ny=6, nz=4, dx=1000.0, dy=1000.0,
             ztop=10000.0, dt=1.0, run_seconds=10.0)
_ORACLE = Path(__file__).parents[1] / "gpuwm" / "data" / "thompson" / "oracle"
_ORACLE_SHA256 = {
    "cloud-condense-sed-column.csv":
        "1bb6ba2f924dd87fd2367d3968f82b5f9c45e7d77cb5f2117d160374f13e8e76",
    "cloud-condense-sed-surface.csv":
        "877c740872e674fe25bd475360b042394f186b08eb04359a29b5f71958b0379f",
    "cloud-condense-nofall-column.csv":
        "a51bd9a5a8d05d9e8626d8de6b7a0cdda97c92e4a305aa612a3327aeb9c88b55",
    "cloud-condense-nofall-surface.csv":
        "f45ba9f8bde3db3c4b3fe0511b27790ad3f32f4e14a3702c85a3bcc295302142",
    "cloud-rain-condense-sed-column.csv":
        "172c63ae8b77c9c114c932b40fc7f83de8af21227a11945fd03312827319c3df",
    "cloud-rain-condense-sed-surface.csv":
        "d31cb0f1dbfe17472af2d0fb1a55fa5025c6b9a22d23e8415a694e15f03d8e4f",
    "cloud-rain-condense-nofall-column.csv":
        "bda73c4a278bf8e9a67e7585c5bff8d2bd2b56b5b32358816d9275aa4543be23",
    "cloud-rain-condense-nofall-surface.csv":
        "f13d9e0638e36d9966e2c9a343a4db3d3e691696ad0bdc4ca7aea1f915f1fa64",
    "condense-fall-attempt-column.csv":
        "fc1799e6355d64854cfa4d7504110190237f52102565ca428eeef66fded0bbc6",
    "condense-fall-attempt-surface.csv":
        "c715772e065a89e21dbcb04ee19ccce6f00552ce62639d6af68bc60275c9afb1",
    "cloud-sed-column.csv":
        "3048e24ecd8f463671b8e127215d0ffe81d01267325f020de3772695e1530a6f",
    "cloud-sed-surface.csv":
        "792324c4b813873c41c7406e2cd9a5256a3f0c9773941a77d68d9df9e2bc0a93",
    "graupel-sed-column.csv":
        "92f78d858e695af72e6731e7eb3bbc20bd55a0724609522827cb57546a641d44",
    "graupel-sed-surface.csv":
        "738fb73bbf1e9d0f64af0552ba9ef94845ae5cd0d73f41668aed2c5fcbaca4c0",
    "condense-column.csv":
        "5a3f3ac205a99bf0635f57d00fbc0001d6a94f91710e59892408e56322ec2a17",
    "condense-surface.csv":
        "f0bd26981a13c93a783f3485959aadb6e3386fc4783d37fa731138364d97f866",
    "ice-column.csv":
        "9c39aaa346862703294f9ee18100cbe252f109b2ee4f85c46daa5fb39f57d072",
    "ice-surface.csv":
        "0207096c26dd4d8064614a52f5b0cef4696b67c3e021e76ca7a13d13093fcec2",
    "ice-sed-column.csv":
        "5e5400206b5ab7e0886a190a592bca019cf9fa61b0236e9252203f43a0245329",
    "ice-sed-surface.csv":
        "a08dbd776fcd97f8c8632a782d8d5f3ce831b86d40389c829810847285dd92ce",
    "mixed-column.csv":
        "3bb4efd6f006c56ad6686f3bc7945f02bc3825de2133f04df6431ca268c7f793",
    "mixed-surface.csv":
        "5a76db934273b636a186f79d84e9d7bd1171179730812b3bf931d330a27b1a8a",
    "rain-sed-column.csv":
        "2e73e319fff8818e2ae39bb3f602de99f923c30dc98c5ed14641340d98ec75f8",
    "rain-sed-surface.csv":
        "eda5fd0d5b9e89ef2b3b1bbef56db5c4358ed1c91c35146fca9b10439289689a",
    "rain-self-column.csv":
        "347a91365d6aba915d843c72eebe6979bfe0ef4ce48821016a49a7d1d2df6d5d",
    "rain-self-surface.csv":
        "a1378ce44d3c8d13ec5f730397c8e4394d6aa8a9844c412e5e00af9c3b10ba94",
    "snow-sed-column.csv":
        "bed1f5e56d926070f69872b4b4d71b48c9b90bf9f4cc0417de4e809b3d438ef9",
    "snow-sed-surface.csv":
        "24d3f5f7126b393898ae9c812b712075b6edc023c3c48ccb5fc3e788f8029cfb",
    "warm-column.csv":
        "ccd170351b57e81dbc5715f86958faf7ae1e9c2e299349b3fa976866c5409b1e",
    "warm-surface.csv":
        "70b9b973b8813c175fcd525045683ad3f0182d26fbec8bc13e68d33fb5b316ca",
    "warm-auto-column.csv":
        "dc2636c7792148dc5521d3c1742622e39d3f9b9b087119794a366803b4116547",
    "warm-auto-surface.csv":
        "ed2757ad2bdf077d13937f37785d89b5776d334c16d136700c3051f9e6d99662",
    "warm-accrete-column.csv":
        "9ae5fe00afbfef20346e738018b63c4caefeafc1bc622f338f8a42122571b836",
    "warm-accrete-surface.csv":
        "7344d13e33b65c204284b2ed6c30d1095f1c6a490229c34c41e2113b2e28ec31",
    "warm-overlap-column.csv":
        "01d866bb8e0de74c95433a2eab798e889ba61c3aaa29fef27ff0b6ce25765e1a",
    "warm-overlap-surface.csv":
        "3e9d1b6b99024342b6be2fad80473cbc547055c33aeb5fe54f5164cab85647e9",
    "warm-frozen-subsat-column.csv":
        "f68c8d8131c9c2028c921bd1104811117d7e8f76425cc14bc48edb992cb1430a",
    "warm-frozen-subsat-surface.csv":
        "1bd0433eb10a4095386c0425b34084d6f85a2cfafde3885c2d5b4027e5d48df0",
    "rain-snow-graupel-overlap-column.csv":
        "41538afe3290ad43287fc78b2f416dc60a6fa1d5043ddf540fc0bdc8101d95e1",
    "rain-snow-graupel-overlap-surface.csv":
        "b9e7507a343fa8427e00f99980b11d7e8872c0f430eb5beb20f0cc5d05c75827",
    "rain-ice-graupel-overlap-column.csv":
        "a01ab68ab9d84e8a04f1dcb428ad311bd7b4016b924c57790ecdc13fa844e49d",
    "rain-ice-graupel-overlap-surface.csv":
        "68041b3b5e5ec9260dc65889410dbce4cf96d5ccaf9248ec01a81bd0d60af7b5",
    "frozen-vapor-overlap-column.csv":
        "dfc497f228a4052e27bcdd9485339ff743dbc3dcf5a9c199351799ac9e070fa5",
    "frozen-vapor-overlap-surface.csv":
        "b959a14c612389861566d6144787b2f2400a8a7b6ffbc1f95c96fcd76443649c",
    "cold-cloud-overlap-column.csv":
        "012616aefa5e81f7c2bb202868b7352526706842a1cab35b1067927e017ba454",
    "cold-cloud-overlap-surface.csv":
        "8ef4c162e79bbe4858b4b79749002ac7ae7cbb842942e481ce35d25dcfab55e2",
    "frozen-vapor-nucleation-overlap-column.csv":
        "7e3d2894336c9a773dc64f03e12dbded949e317b4e4713e83dbc0f9a996cf531",
    "frozen-vapor-nucleation-overlap-surface.csv":
        "3e833c00297f3cafeae45732def3ddacc33f01d23c9e547f8e97080fc84baaf8",
    "cold-ice-rain-overlap-column.csv":
        "6e964a6cf7fef3801a80e4d241c4a2b402081b2681e44dcf75a4c0594a006c44",
    "cold-ice-rain-overlap-surface.csv":
        "6ea219231ce8d90cb4edb9c6fc31acb9d55368265ac3edcee0553b8591b6977d",
    "cold-full-overlap-column.csv":
        "848d4c7f5efc13da67b7b1447a97eecd3acdbf1dc4cfebd05a56a748bd82af4b",
    "cold-full-overlap-surface.csv":
        "4d72461e4fc2f137889f9abf87a0d261b7a9c59a547383d00d3f723ac39dd71d",
    "cold-cloud-rain-overlap-column.csv":
        "76da2ac7be669cd78b0b69f3940abb9e5ec817919c55f4d60da9f9893b4d9d96",
    "cold-cloud-rain-overlap-surface.csv":
        "9097a49eeee045cff8f9aeec62ad2ac28147e9d0a4daad96fd72531d0fca0ec4",
    "ice-auto-column.csv":
        "24235d243f2194f0923cf9b8650c8dd0b086406618e63df8ac12256b904177c1",
    "ice-auto-surface.csv":
        "03197e816306b270b9bf97904bb47b337b07631137f29eba6fe141971faf02d6",
    "rain-evap-column.csv":
        "f5105e593ba1d5dd556a33c4b5010afbb8a6c77f0fd051570d7e4b5f18917e68",
    "rain-evap-surface.csv":
        "f12e6d78b78b605650c37b66e43efe45472a645e0724d72d0367972eb249056b",
    "snow-subl-column.csv":
        "a8a0436d81b6adb077d6d17f8b4fff4506ca5846d316873571cf19515fb5c1aa",
    "snow-subl-surface.csv":
        "437c9d6a2e0f0377a31c37b25ce7f2b770fa36929a2c448e34686e5d43094104",
    "graupel-subl-column.csv":
        "dd9fbae02bc9449dadc960277f75996b6bbd4f39b2f811b189409a49aa059e73",
    "graupel-subl-surface.csv":
        "e3f442d05292946958e701cf29bcd3f8017078949dc6d263739625e7653f5a25",
    "ice-dep-column.csv":
        "52e4c4e832fa07a9449316155014b9b6754df3335b4090dadcc2c97d00080882",
    "ice-dep-surface.csv":
        "9e03f2a22915fd381850256733d3f993892e695e30e5b09809bb6d564b0aff24",
    "ice-nuc-column.csv":
        "40c247a388f558c9fcbeaa4ef54909377431c7c32937bd6521e6ea41ae8e7fcf",
    "ice-nuc-surface.csv":
        "ca598639d2b9be1a0e01aba3073119dbb0f9a7c7cebd0b34b3f0f2f9fe1d6d10",
    "snow-dep-column.csv":
        "8303c02e0168d341fca74ebebfebf7745f449f61007b0342e9269560a5c43d4f",
    "snow-dep-surface.csv":
        "354dad333fbe4002ab8f1d1df429173ffe06c93d8581d57a4b672cf6de235eca",
    "graupel-dep-column.csv":
        "33946f5964ac81aeaf42ccfb5c1d346c46aecb2fa36bfc8aa827ccc760a359f2",
    "graupel-dep-surface.csv":
        "e455846e716b72dbb295880fa3a7cafc53e114ae9a115fa5518587a87fcf4179",
    "snow-melt-column.csv":
        "8e0029ce5fc32fc5c594d33d7beed3fc507f22f77b8bead6a95fa84ef4ba04c3",
    "snow-melt-surface.csv":
        "34f3673a96967e3cdff978bc94d7f7db100b9f2c75055b73efea685653c428dc",
    "graupel-melt-column.csv":
        "cba622bb65ef671771b2f292e8e4c6d021342e921e57c0bec1767386070dcfca",
    "graupel-melt-surface.csv":
        "40b079d4943ae78623d953ac88fd57ebd6c59cb6f261484e76f1f01236016462",
}


def _cfg(**changes):
    return RunConfig(**_TINY, moist=True, mp_physics=8, **changes)


def test_thompson_registry_state_and_background_constants_are_pinned():
    assert MASS_SPECIES == ("qv", "qc", "qr", "qi", "qs", "qg")
    assert NUMBER_SPECIES == ("ni", "nr")
    assert BACKGROUND_EFFECTIVE_RADIUS_UM == {
        "effc": 2.49, "effi": 4.99, "effs": 9.99}
    assert CLASSIC_GRAUPEL_DENSITY_KG_M3 == 400.0


def test_thompson_generated_table_layout_and_exact_sizes_are_pinned():
    assert sequential_file_bytes(
        GENERATED_TABLE_FILES["qr_acr_qg_V4.dat"]) == 74_966_480
    assert sequential_file_bytes(
        GENERATED_TABLE_FILES["qr_acr_qsV2.dat"]) == 43_764_288
    assert sequential_file_bytes(
        GENERATED_TABLE_FILES["freezeH2O.dat"]) == 254_944_848
    assert sum(sequential_file_bytes(records) for records in
               GENERATED_TABLE_FILES.values()) == 373_675_616
    assert sequential_file_bytes(AUXILIARY_TABLE_RECORDS) == 6_164_536
    assert TABLE_SET_ID == "wrf-v4.6.1-classic-thompson-mp8-gfortran13-v1"
    assert sum(item.bytes for item in CLASSIC_TABLE_ASSETS) == 379_840_152
    assert tuple(item.filename for item in CLASSIC_TABLE_ASSETS) == (
        "qr_acr_qg_V4.dat", "qr_acr_qsV2.dat", "freezeH2O.dat",
        "thompson_aux_tables.dat")


def test_table_asset_identity_rejects_resize_and_hash_substitution(tmp_path):
    path = tmp_path / "tiny.dat"
    path.write_bytes(b"official-table-bytes")
    expected = (TableAsset(
        path.name, path.stat().st_size,
        hashlib.sha256(path.read_bytes()).hexdigest()),)
    assert validate_table_assets(tmp_path, expected) == expected

    path.write_bytes(b"official-table-byteS")
    with pytest.raises(ValueError, match="SHA-256"):
        validate_table_assets(tmp_path, expected)
    path.write_bytes(b"short")
    with pytest.raises(ValueError, match="expected"):
        validate_table_assets(tmp_path, expected)


def test_sequential_reader_preserves_fortran_index_order_and_rejects_junk(
        tmp_path):
    records = (TableRecord("first", (2, 3)), TableRecord("second", (2,)))
    expected = {
        "first": np.arange(1.0, 7.0).reshape((2, 3), order="F"),
        "second": np.array([13.0, 21.0]),
    }
    payload = bytearray()
    for record in records:
        raw = np.asarray(expected[record.name], dtype="<f8",
                         order="F").tobytes(order="F")
        payload += struct.pack("<i", len(raw)) + raw + struct.pack("<i", len(raw))
    path = tmp_path / "table.dat"
    path.write_bytes(payload)
    got = read_sequential_records(path, records)
    for name in expected:
        np.testing.assert_array_equal(got[name], expected[name])
        assert got[name].flags.f_contiguous

    path.write_bytes(payload + b"x")
    with pytest.raises(ValueError, match="unexpected bytes"):
        read_sequential_records(path, records)


def test_runtime_table_owner_checks_layout_hashes_and_round_trip(tmp_path):
    records = (
        TableRecord("large", (2, 3)),
        TableRecord("small", (4,)),
    )
    arrays = {
        "large": np.array(
            np.arange(6.0).reshape((2, 3), order="F"),
            dtype=np.float64, order="F"),
        "small": np.array([2.0, 3.0, 5.0, 7.0], dtype=np.float64,
                          order="F"),
    }
    for value in arrays.values():
        value.flags.writeable = False
    assets = (TableAsset("synthetic.dat", 1, "0" * 64),)
    table_set = ClassicTableSet(
        root=tmp_path, arrays=MappingProxyType(arrays), assets=assets)

    class CopyBackend:
        float64 = np.float64

        @staticmethod
        def asarray(value, dtype, order):
            return np.array(value, dtype=dtype, order=order, copy=True)

    device = _upload_table_set(
        table_set, CopyBackend,
        records=records, expected_assets=assets, device_id=None,
        verify_roundtrip=True)
    assert tuple(device.arrays) == ("large", "small")
    assert device.payload_bytes == sum(record.payload_bytes for record in records)
    assert device.roundtrip_verified
    assert len(device.identity_sha256) == 64
    assert set(device.array_sha256) == {"large", "small"}
    np.testing.assert_array_equal(device.large, arrays["large"])
    assert device.large.flags.f_contiguous


def test_runtime_table_owner_exposes_exact_cold_source_group_order(tmp_path):
    names = (
        "tpi_ide", "tps_iaus", "tni_iaus",
        *RAIN_SNOW_TABLE_NAMES,
        *RAIN_GRAUPEL_TABLE_NAMES,
        *RAIN_FREEZING_TABLE_NAMES,
        "t_Efrw", "tpi_qcfz", "tni_qcfz",
    )
    sentinels = {name: object() for name in names}
    owner = DeviceClassicTableSet(
        root=tmp_path,
        arrays=MappingProxyType(sentinels),
        payload_bytes=0,
        array_sha256=MappingProxyType({}),
        identity_json="{}",
        identity_sha256="a" * 64,
        device_id=None,
        upload_seconds=0.0,
        verification_seconds=0.0,
        roundtrip_verified=True,
    )

    groups = owner.cold_source_tables
    assert groups.ice_deposition_partition is sentinels["tpi_ide"]
    assert groups.ice_to_snow_mass is sentinels["tps_iaus"]
    assert groups.ice_to_snow_number is sentinels["tni_iaus"]
    assert groups.rain_snow_tables == tuple(
        sentinels[name] for name in RAIN_SNOW_TABLE_NAMES)
    assert groups.rain_graupel_tables == tuple(
        sentinels[name] for name in RAIN_GRAUPEL_TABLE_NAMES)
    assert groups.rain_freezing_tables == tuple(
        sentinels[name] for name in RAIN_FREEZING_TABLE_NAMES)
    assert groups.rain_cloud_efficiency is sentinels["t_Efrw"]
    assert groups.cloud_freezing_tables == (
        sentinels["tpi_qcfz"], sentinels["tni_qcfz"])
    assert groups.identity_sha256 == owner.identity_sha256


def test_runtime_table_owner_rejects_missing_mutable_and_changed_device_data(
        tmp_path):
    record = TableRecord("only", (2, 2))
    value = np.array([[1.0, 2.0], [3.0, 4.0]], order="F")
    assets = (TableAsset("synthetic.dat", 1, "0" * 64),)

    def table(arrays):
        return ClassicTableSet(
            root=tmp_path, arrays=MappingProxyType(arrays), assets=assets)

    with pytest.raises(ValueError, match="missing"):
        _upload_table_set(
            table({}), np, records=(record,), expected_assets=assets,
            device_id=None, verify_roundtrip=True)

    with pytest.raises(ValueError, match="immutable"):
        _upload_table_set(
            table({"only": value}), np, records=(record,),
            expected_assets=assets, device_id=None, verify_roundtrip=True)

    value.flags.writeable = False

    class CorruptBackend:
        float64 = np.float64

        @staticmethod
        def asarray(source, dtype, order):
            result = np.array(source, dtype=dtype, order=order, copy=True)
            result[0, 0] += 1.0
            return result

    with pytest.raises(RuntimeError, match="round trip changed"):
        _upload_table_set(
            table({"only": value}), CorruptBackend, records=(record,),
            expected_assets=assets, device_id=None, verify_roundtrip=True)


def test_thompson_state_preflight_transport_nesting_and_radii(monkeypatch):
    import gpuwm.core.state as state_module

    monkeypatch.setattr(state_module, "cp", np)
    state = state_module.DomainState(_cfg())
    shape = (4, 6, 8)
    for name in ("qi", "qs", "qg", "nr", "ni",
                 "qi0", "qs0", "qg0", "nr0", "ni0",
                 "effc", "effi", "effs"):
        assert getattr(state, name).shape == shape
    for name in ("nc", "ns", "ng", "ns0", "ng0", "effr"):
        assert getattr(state, name, None) is None
    np.testing.assert_array_equal(state.effc, np.float32(2.49))
    np.testing.assert_array_equal(state.effi, np.float32(4.99))
    np.testing.assert_array_equal(state.effs, np.float32(9.99))
    assert extra_moist_species(state) == (
        "qi", "qs", "qg", "nr", "ni")

    shapes = preflight.state_array_shapes(_cfg())
    assert set(("qi", "qs", "qg", "nr", "ni",
                "qi0", "qs0", "qg0", "nr0", "ni0")) <= shapes.keys()
    assert not set(("nc", "ns", "ng", "ns0", "ng0", "effr")) & shapes.keys()
    assert preflight.nest_field_kinds(_cfg()) == (
        "u", "v", "w", "t", "ph", "mu", "qv", "qc", "qr",
        "qi", "qs", "qg", "nr", "ni")


def test_thompson_fixed_diffusion_and_kf_phase_contracts():
    slots = preflight.scratch_slot_registry(
        _cfg(km_opt=4, diff_6th_opt=2))
    for species in ("qv", "qc", "qr", "qi", "qs", "qg", "nr", "ni"):
        assert "smag_r" + species in slots
    assert "smag_rns" not in slots
    assert "smag_rng" not in slots
    assert kf_phase_mode_for_microphysics(8) == \
        KFPhaseMode.SEPARATE_ICE_SNOW


def test_thompson_acoustic_cq_uses_mass_but_not_number_moments():
    shape = (3, 2, 4)
    moisture = {
        name: np.full(shape, index * 1.0e-4, dtype=np.float64)
        for index, name in enumerate(MASS_SPECIES, start=1)
    }
    moisture.update(nr=np.full(shape, 9.0e7), ni=np.full(shape, 4.0e6))
    got = np_calc_cq(moisture, mp_physics=8)
    reference = np_calc_cq(moisture, mp_physics=6)
    for actual, expected in zip(got, reference):
        np.testing.assert_array_equal(actual, expected)


def test_generic_transport_does_not_infer_morrison_only_numbers():
    state = SimpleNamespace(qi=object(), qs=object(), qg=object(),
                            nr=object(), ni=object(), ns=None, ng=None)
    assert extra_moist_species(state) == (
        "qi", "qs", "qg", "nr", "ni")


@pytest.mark.parametrize(
    "scenario",
    ("warm", "mixed", "ice", "condense", "rain-sed", "ice-sed",
     "cloud-sed", "cloud-condense-sed", "cloud-condense-nofall",
     "cloud-rain-condense-sed", "cloud-rain-condense-nofall",
     "condense-fall-attempt",
     "snow-sed", "graupel-sed", "warm-auto",
     "rain-self", "warm-accrete", "warm-overlap",
     "rain-snow-graupel-overlap", "frozen-vapor-overlap",
     "cold-cloud-overlap", "frozen-vapor-nucleation-overlap",
     "cold-ice-rain-overlap", "cold-full-overlap",
     "cold-cloud-rain-overlap",
     "ice-auto", "rain-evap", "snow-subl",
     "graupel-subl", "ice-dep", "ice-nuc", "snow-dep",
     "graupel-dep", "snow-melt", "graupel-melt",
     "warm-frozen-subsat"))
def test_official_wrf_column_oracles_are_hash_pinned_finite_and_conservative(
        scenario):
    for suffix in ("column", "surface"):
        path = _ORACLE / f"{scenario}-{suffix}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == \
            _ORACLE_SHA256[path.name]

    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48
    assert [row["phase"] for row in rows] == ["before"] * 24 + ["after"] * 24
    assert [int(row["k"]) for row in rows[:24]] == list(range(1, 25))
    assert [int(row["k"]) for row in rows[24:]] == list(range(1, 25))
    assert all(None not in row for row in rows)

    numeric_names = tuple(name for name in rows[0] if name not in ("phase", "k"))
    values = np.asarray([
        [float(row[name]) for name in numeric_names] for row in rows])
    assert np.isfinite(values).all()
    before, after = rows[:24], rows[24:]
    for name in ("qv", "qc", "qr", "qi", "qs", "qg",
                 "ni_per_kg", "nr_per_kg"):
        assert min(float(row[name]) for row in after) >= 0.0
    assert min(float(row["refl_dbz"]) for row in after) >= -35.0
    assert all(2.49e-6 <= float(row["effc_m"]) <= 50.0e-6
               for row in after)
    assert all(float(np.float32(4.99e-6)) <= float(row["effi_m"]) <=
               float(np.float32(125.0e-6))
               for row in after)
    assert all(float(np.float32(9.99e-6)) <= float(row["effs_m"]) <=
               float(np.float32(999.0e-6))
               for row in after)

    qnames = ("qv", "qc", "qr", "qi", "qs", "qg")
    before_q = np.asarray([[float(row[name]) for name in qnames]
                           for row in before])
    after_q = np.asarray([[float(row[name]) for name in qnames]
                          for row in after])
    pressure = np.asarray([float(row["p_pa"]) for row in before])
    temperature = np.asarray([float(row["temp_k"]) for row in before])
    dz = np.asarray([float(row["dz_m"]) for row in before])
    density = 0.622 * pressure / (
        287.04 * temperature * (before_q[:, 0] + 0.622))
    column_loss = float(np.sum(
        density * dz * (before_q.sum(axis=1) - after_q.sum(axis=1))))
    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    precipitation = float(surface["rainnc_mm"])
    if scenario == "cold-full-overlap":
        # This cap-forcing vector deliberately inherits classic WRF's held
        # pre-cap prg_rci source after pri/prr_rci are scaled.  At the
        # diagnostic 100,000-km layer thickness that legacy asymmetry is
        # enormous; pin it explicitly instead of mislabeling it conservative.
        assert precipitation == 0.0
        assert column_loss == pytest.approx(
            -1623599.5398851773, abs=2.0e-6)
        return
    if scenario in {
            "frozen-vapor-nucleation-overlap", "cold-ice-rain-overlap",
            "cold-cloud-rain-overlap", "warm-frozen-subsat"}:
        # This source-isolation vector uses 100,000-km layers so that the
        # sedimentation calls remain present but dynamically negligible.
        # Do not turn its ordinary FP32 mixing-ratio rounding into a loose
        # absolute column budget: pin zero fallout and a relative water-mass
        # residual instead.
        initial_water = float(np.sum(
            density * dz * before_q.sum(axis=1)))
        assert precipitation == 0.0
        assert abs(column_loss) / initial_water < 2.0e-8
        return
    assert column_loss == pytest.approx(precipitation, abs=2.0e-5)


def test_rain_ice_graupel_oracle_pins_wrf_group_cap_mass_asymmetry():
    """Pin, rather than conceal, classic WRF's post-cap held qg source."""
    scenario = "rain-ice-graupel-overlap"
    for suffix in ("column", "surface"):
        path = _ORACLE / f"{scenario}-{suffix}.csv"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == \
            _ORACLE_SHA256[path.name]

    with (_ORACLE / f"{scenario}-column.csv").open(
            newline="", encoding="ascii") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 48
    numeric_names = tuple(
        name for name in rows[0] if name not in ("phase", "k"))
    values = np.asarray([
        [float(row[name]) for name in numeric_names] for row in rows])
    assert np.isfinite(values).all()
    before, after = rows[:24], rows[24:]
    for name in ("qv", "qc", "qr", "qi", "qs", "qg",
                 "ni_per_kg", "nr_per_kg"):
        assert min(float(row[name]) for row in after) >= 0.0

    qnames = ("qv", "qc", "qr", "qi", "qs", "qg")
    before_q = np.asarray([[float(row[name]) for name in qnames]
                           for row in before])
    after_q = np.asarray([[float(row[name]) for name in qnames]
                          for row in after])
    pressure = np.asarray([float(row["p_pa"]) for row in before])
    temperature = np.asarray([float(row["temp_k"]) for row in before])
    dz = np.asarray([float(row["dz_m"]) for row in before])
    density = 0.622 * pressure / (
        287.04 * temperature * (before_q[:, 0] + 0.622))
    column_loss = float(np.sum(
        density * dz * (before_q.sum(axis=1) - after_q.sum(axis=1))))
    with (_ORACLE / f"{scenario}-surface.csv").open(
            newline="", encoding="ascii") as stream:
        surface = next(csv.DictReader(stream))
    precipitation = float(surface["rainnc_mm"])

    # WRF scales prr_rci in the rain group but retains the already formed
    # prg_rci source.  The intentionally cap-limited vector therefore creates
    # this exact water-budget asymmetry; an exact-WRF mode must preserve it.
    assert column_loss == pytest.approx(0.017860315863411014, abs=2.0e-12)
    assert precipitation == pytest.approx(
        0.021886643022298813, abs=2.0e-12)
    assert precipitation - column_loss == pytest.approx(
        0.004026327158887799, abs=3.0e-12)
