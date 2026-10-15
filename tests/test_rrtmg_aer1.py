"""Tegen six-type acceptance tests, WRF 4.6.1 authority."""
import importlib
import pytest


def test_device_climatology_contract_exists():
    module = importlib.import_module("gpuwm.core.rrtmg_aerosol_optics")
    for name in ("aer1_climatology_device", "aer1_time_pressure_device", "aer1_sw_optics_device"):
        assert callable(getattr(module, name, None)), f"missing {name}"


def test_aer1_legacy_admission():
    from gpuwm.config import RunConfig, validate_radiation_driver_options
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000., dy=1000., ztop=10000., dt=5., run_seconds=10., aer_opt=1, ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rrtmg_legacy")
    validate_radiation_driver_options(cfg)


def test_aer1_modern_refusal_names_mapping():
    from gpuwm.config import RunConfig, validate_radiation_driver_options
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=1000., dy=1000., ztop=10000., dt=5., run_seconds=10., aer_opt=1, ra_lw_physics=4, ra_sw_physics=4, ra_rrtmg_variant="rte-rrtmgp")
    with pytest.raises(ValueError, match="six-type.*mapping|mapping.*six-type"):
        validate_radiation_driver_options(cfg)


def test_aer1_climatology_one_bit_is_refused(tmp_path):
    import json
    from pathlib import Path
    import numpy as np
    from gpuwm import data_assets
    from gpuwm.ingest.rrtmg_coeffs import DATA_DIR
    from gpuwm.core.rrtmg_sw import (AER1_CLIMATOLOGY_ASSET, decode_aer1_climatology,
                                     encode_aer1_climatology, load_sw_tables)
    manifest = json.loads((DATA_DIR / "rrtmg_aer1_manifest.json").read_text())
    (tmp_path / "rrtmg_aer1_manifest.json").write_bytes((DATA_DIR / "rrtmg_aer1_manifest.json").read_bytes())
    with np.load(data_assets.data_path(AER1_CLIMATOLOGY_ASSET)) as source:
        arrays = {key: source[key] for key in source.files}
    climatology = decode_aer1_climatology(
        arrays.pop("climatology_xz"), manifest["arrays"]["climatology"]["shape"]).copy()
    climatology.view(np.uint32).flat[0] ^= np.uint32(1)
    np.savez(tmp_path / "rrtmg_aer1_climatology.npz",
             climatology_xz=encode_aer1_climatology(climatology), **arrays)
    with pytest.raises(ValueError, match="climatology manifest hash mismatch"):
        load_sw_tables(aer_opt=1, data_dir=tmp_path)


def test_aer1_optics_one_bit_is_refused():
    import numpy as np
    from gpuwm.core.rrtmg_sw import load_sw_tables, validate_aer1_optics_tables
    tables = load_sw_tables(aer_opt=1)
    tables.rsrtaua = tables.rsrtaua.copy()
    tables.rsrtaua.view(np.uint32).flat[0] ^= np.uint32(1)
    with pytest.raises(ValueError, match="rsrtaua manifest hash mismatch"):
        validate_aer1_optics_tables(tables)


def test_registry_publishes_state_output_and_doors():
    import json
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    registry=json.loads((root/"gpuwm/physics_registry_v2.json").read_text())
    row=registry["parameters"]["aer_opt"]["implementation_rows"]["aeropt1"]
    assert row["state"] == ["aerodm"]
    assert row["outputs"] == ["AODTOT"]
    assert row["doors"]["ra_sw_physics"] == 4
    assert row["doors"]["nested"] is False


def test_aodtot_is_preserved_as_a_held_restart_diagnostic():
    from gpuwm.io.restart import DRIVER_CHECKPOINT_ONLY_ATTRS
    assert "aodtot" in DRIVER_CHECKPOINT_ONLY_ATTRS


def test_aer1_history_planner_keeps_both_aerosol_and_absorbed_shortwave():
    from gpuwm.config import RunConfig
    from gpuwm.io.history_layout import produced_history_shapes
    from gpuwm.core.preflight import physics_array_shapes
    cfg=RunConfig(nx=8,ny=8,nz=12,dx=3000.,dy=3000.,ztop=20000.,dt=15.,run_seconds=300.,
        moist=True,mp_physics=1,ra_physics=4,ra_rrtmg_variant="rrtmg_legacy",aer_opt=1,
        sf_sfclay_physics=91,sf_surface_physics=2,bl_pbl_physics=1)
    allocated=physics_array_shapes(cfg)
    assert allocated["fields/gsw"] == (8,8)
    history=produced_history_shapes(cfg)
    assert history["GSW"] == history["AODTOT"]


def test_monthly_aerosol_has_a_setup_derived_state_classification():
    from types import SimpleNamespace
    import numpy as np
    from gpuwm.io.restart import classify_state_attr, state_manifest, PHYSICS_ASSET_PATHS
    monthly=np.zeros((1,12,12,6),np.float32)
    assert classify_state_attr("aerodm") == "derived_setup"
    assert state_manifest(SimpleNamespace(aerodm=monthly)) == {}
    assert "wrf_aer1_climatology" in PHYSICS_ASSET_PATHS
    assert "wrf_aer1_manifest" in PHYSICS_ASSET_PATHS


def test_aer1_climatology_encoding_round_trips_and_the_shipped_asset_decodes_to_its_pin():
    """The companion carries the climatology re-encoded (2.9.0 wheel-size fix).

    The loader decodes it; the decoded float32 words must be the ones the
    manifest pins, and the codec must invert itself on arbitrary words
    (signed zeros, subnormals, a NaN payload) bit for bit.
    """
    import hashlib
    import json
    import numpy as np
    from gpuwm import data_assets
    from gpuwm.ingest.rrtmg_coeffs import DATA_DIR
    from gpuwm.core.rrtmg_sw import (AER1_CLIMATOLOGY_ASSET, decode_aer1_climatology,
                                     encode_aer1_climatology)
    rng = np.random.default_rng(461)
    words = rng.integers(0, 2**32, size=(2, 3, 5), dtype=np.uint64).astype(np.uint32)
    words.flat[:4] = [0x80000000, 0x00000001, 0x7fc00001, 0xffffffff]
    sample = words.view(np.float32)
    back = decode_aer1_climatology(encode_aer1_climatology(sample), sample.shape)
    assert back.dtype == np.float32 and back.tobytes() == sample.tobytes()
    manifest = json.loads((DATA_DIR / "rrtmg_aer1_manifest.json").read_text())
    pin = manifest["arrays"]["climatology"]
    with np.load(data_assets.data_path(AER1_CLIMATOLOGY_ASSET)) as source:
        decoded = decode_aer1_climatology(source["climatology_xz"], pin["shape"])
    assert list(decoded.shape) == pin["shape"]
    assert hashlib.sha256(decoded.tobytes()).hexdigest() == pin["sha256"]
