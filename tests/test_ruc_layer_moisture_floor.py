"""A layer source whose top layer is far drier than the next reaches RUC in range.

Breakage it prevents: ``init_soil_3_real``'s linear surface anchor off the top
two layers goes negative over dry ground (ECMWF open-data 12Z 7 Oct 2026: 44
land columns, levels 1-2, down to -0.0214), and ``gpuwm sim`` refused the
prepared start with "RUC smois must remain within 0..1 m3 m-3".  real.exe
takes ``SMOIS = MAX(SMOIS, 0.005)`` for a RUC layer source
(``module_initialize_real.F`` v4.6.1 :3435-3446) and nothing for a level
source.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.ruc import ruc_initialize_cold_start
from gpuwm.ingest.ruc_soil import preprocess_ruc_soil


def _dry_top_era5(ny: int = 3, nx: int = 4) -> dict:
    land = np.ones((ny, nx))
    land[:, -1] = 0.0
    return {
        "LANDSEA": land,
        "SKINTEMP": np.full((ny, nx), 288.0),
        "SST": np.full((ny, nx), 286.0),
        "XICE": np.zeros((ny, nx)),
        "ST000007": np.full((ny, nx), 289.0),
        "ST007028": np.full((ny, nx), 287.5),
        "ST028100": np.full((ny, nx), 285.0),
        "ST100289": np.full((ny, nx), 282.0),
        # The measured column: swvl1 0.0095 over swvl2 0.0897.
        "SM000007": np.full((ny, nx), 0.0095),
        "SM007028": np.full((ny, nx), 0.0897),
        "SM028100": np.full((ny, nx), 0.12),
        "SM100289": np.full((ny, nx), 0.13),
        "TMN": np.full((ny, nx), 281.0),
        "SNOW": np.zeros((ny, nx)),
    }


def test_a_dry_top_layer_source_initializes_ruc_inside_0_1() -> None:
    fields = _dry_top_era5()
    shape = fields["LANDSEA"].shape
    state = preprocess_ruc_soil(
        fields, soil_type=np.full(shape, 6.0),
        water_temperature_policy="wrf_compat")
    land = state.landmask >= 0.5
    smois = state.soil_moisture
    assert smois.min() >= np.float32(0.005)
    # The extrapolated anchor (level 1) is what real.exe floors; deeper
    # levels keep their remapped values.
    np.testing.assert_array_equal(smois[0][land], np.float32(0.005))
    assert float(smois[4][land].min()) > 0.05
    np.testing.assert_array_equal(smois[:, ~land], np.float32(1.0))
    np.testing.assert_array_equal(state.liquid_moisture, smois)
    assert state.soil_moisture_floor["floored_columns"] == int(land.sum())
    assert state.soil_moisture_floor["min_pre_floor"] < 0.0
    # And the launch preflight that refused the run now admits it.
    ruc_initialize_cold_start(
        state.soil_temperature, smois, np.full(shape, 6), np.full(shape, 7),
        np.zeros(shape, dtype=np.float32))


def test_the_floor_is_the_layer_arm_only_and_identity_when_unneeded() -> None:
    from gpuwm.ingest.ruc_soil import account_for_zero_ruc_soil_moisture
    wet = np.full((9, 2, 2), 0.2, dtype=np.float32)
    same, receipt = account_for_zero_ruc_soil_moisture(wet, "layers")
    assert same is wet and receipt == {}
    dry = wet.copy()
    dry[0, 0, 0] = -0.02
    kept, receipt = account_for_zero_ruc_soil_moisture(dry, "levels")
    assert kept is dry and receipt == {}
    floored, receipt = account_for_zero_ruc_soil_moisture(dry, "layers")
    assert floored.dtype == np.float32
    assert floored[0, 0, 0] == np.float32(0.005)
    assert receipt["floored_values"] == 1
    np.testing.assert_array_equal(floored[1:], dry[1:])


def test_mapped_ifs_layers_use_the_shipped_ecmwf_soil_contract():
    from gpuwm.mapped_composition import load_composition
    from gpuwm.source_authorities import packaged_authorities
    from gpuwm.source_adapters import get_source_adapter
    from gpuwm.ingest.soil_contract import MAPPED_SOIL_MOISTURE, MAPPED_SOIL_TEMPERATURE
    authorities = packaged_authorities(get_source_adapter("ecmwf-open-data").packaged_profile)
    composition = load_composition(authorities["composition"], authorities["mapping"])
    fields = _dry_top_era5()
    fields[MAPPED_SOIL_TEMPERATURE] = np.stack([fields.pop(n) for n in
        ("ST000007", "ST007028", "ST028100", "ST100289")]).astype(np.float32)
    fields[MAPPED_SOIL_MOISTURE] = np.stack([fields.pop(n) for n in
        ("SM000007", "SM007028", "SM028100", "SM100289")]).astype(np.float32)
    state = preprocess_ruc_soil(fields, soil_type=np.full((3, 4), 6.),
        soil_layer_contract=composition["soil_layers"], water_temperature_policy="wrf_compat")
    assert state.soil_moisture_floor["floored_columns"] == 9
    assert state.soil_moisture.dtype == np.float32
    assert state.soil_moisture.min() == np.float32(0.005)
    ruc_initialize_cold_start(state.soil_temperature, state.soil_moisture,
        np.full((3, 4), 6), np.full((3, 4), 7), np.zeros((3, 4), dtype=np.float32))


@pytest.mark.parametrize("missing", [np.nan, np.inf, -np.inf])
def test_the_floor_preserves_missing_values_for_the_launch_refusal(missing):
    from gpuwm.ingest.ruc_soil import account_for_zero_ruc_soil_moisture
    moisture = np.full((9, 2, 2), 0.2, dtype=np.float32)
    moisture[0, 0, 0] = -0.02
    moisture[0, 0, 1] = missing
    floor, _ = account_for_zero_ruc_soil_moisture(moisture, "layers")
    assert not np.isfinite(floor[0, 0, 1])
    with pytest.raises(ValueError, match="finite"):
        ruc_initialize_cold_start(np.full_like(floor, 285.), floor,
            np.full((2, 2), 6), np.full((2, 2), 7), np.zeros((2, 2), dtype=np.float32))


def test_a_fired_floor_is_a_note_the_prep_door_puts_on_screen(capsys):
    """D-02 (2.8.8 acceptance): the floor's line went to stdout without a
    ``note:`` prefix, so the prep door kept it in the details log and the
    terminal never said the floor fired."""
    from gpuwm.ingest.ruc_soil import account_for_zero_ruc_soil_moisture
    moisture = np.full((9, 2, 2), 0.2, dtype=np.float32)
    moisture[0, 0, 0] = -0.029
    account_for_zero_ruc_soil_moisture(moisture, "layers")
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith(
        "note: RUC soil moisture: 1 value(s) in 1 column(s) below 0.005 on")
    account_for_zero_ruc_soil_moisture(np.full((9, 2, 2), 0.2, np.float32),
                                       "layers")
    assert capsys.readouterr().err == ""
