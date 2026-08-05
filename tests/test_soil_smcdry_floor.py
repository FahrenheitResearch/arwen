"""The SMCDRY floor for sub-physical ERA5 land soil moisture.

The defect this pins: an ERA5 swvl value a hair below zero (GRIB packing,
interpolation undershoot) was admitted by the overshoot band and clipped
to EXACTLY 0.0 on a land point.  Noah's thermal conductivity divides by
SMC (TDFCND: ``xunfroz = sh2o/smc``, the ``ake`` divide,
``powf(smcmax/smc, bexp)``), so ONE such cell was NaN conductivity ->
NaN GRDFLX -> NaN HFX and the run died at step 0 blamed on the PBL
scheme.  The threshold was a hard zero: 1e-12 survived, 0.0 did not.

The fix floors each land layer below the soil category's SMCDRY (air-dry,
SOILPARM.TBL DRYSMC) to that SMCDRY.  WRF v4.6.1 real.exe instead resets
the whole column to the constant 0.005 when the TOP layer is below 0.005
(dyn_em/module_initialize_real.F:3363-3395); the divergence ledger lives
on gpuwm.ingest.soil._floor_land_moisture_at_smcdry.
"""

from __future__ import annotations

import numpy as np
import pytest

from gpuwm.core.noah import (
    SOIL_COLS, load_tables, pack_params, sh2o_init)
from gpuwm.ingest.soil import (
    ERA5_LAYER_MIDPOINTS_M, NOAH_LAYER_MIDPOINTS_M, _interp_nodes,
    preprocess_noah_soil)
from gpuwm.verify.npref import np_noah_column

DT = 30.0
DZS = (0.10, 0.30, 0.60, 1.00)

_TEMP_NAMES = ("ST000007", "ST007028", "ST028100", "ST100289")
_MOIST_NAMES = ("SM000007", "SM007028", "SM028100", "SM100289")


@pytest.fixture(scope="module")
def params():
    return pack_params(load_tables())


def _smcdry(params, category: int) -> float:
    return float(params.soil[category - 1, SOIL_COLS.index("smcdry")])


def _era5_fields(shape, *, tslb=285.0, smois=0.30):
    """A warm land ERA5 layer-form input, uniform unless overridden."""
    fields = {
        "LANDSEA": np.ones(shape),
        "SKINTEMP": np.full(shape, tslb),
        "TMN": np.full(shape, tslb),
    }
    fields.update({name: np.full(shape, tslb) for name in _TEMP_NAMES})
    fields.update({name: np.full(shape, smois) for name in _MOIST_NAMES})
    return fields


def _driver_column(smois, tslb, sh2o, *, veg=10, soil=6):
    """One land column for the WRF v4.6.1 driver mirror, moderate forcing."""
    sfctmp = 287.0
    sfcprs = 96000.0
    es = 611.2 * np.exp(17.67 * (sfctmp - 273.15) / (sfctmp - 29.65))
    qgh = 0.622 * es / (sfcprs - es)
    return dict(
        psfc=96500.0, sfcprs=sfcprs, sfctmp=sfctmp,
        qv1=0.5 * qgh, qgh=qgh, dz8w1=60.0,
        glw=0.85 * 5.67051e-8 * sfctmp ** 4, swdown=500.0,
        rainbl=0.0, sr=0.0,
        chs=0.01, cqs2=0.02, chs2=0.01, rib=0.0,
        ivgtyp=veg, isltyp=soil, vegfra=60.0,
        shdmin=10.0, shdmax=90.0, tmn=283.0,
        xland=1.0, xice=0.0, snoalb=0.65, embck=0.95,
        tsk=288.0, canwat=0.0, snow=0.0, snowh=0.0, snowc=0.0,
        smois=np.asarray(smois, np.float64),
        tslb=np.asarray(tslb, np.float64),
        sh2o=np.asarray(sh2o, np.float64),
        albedo=0.19, albbck=0.19, emiss=0.95,
        z0=0.1, znt=0.1, snotime=0.0, lai=2.0,
        sfcrunoff=0.0, udrunoff=0.0, acsnow=0.0, acsnom=0.0,
        snopcx=0.0, potevp=0.0,
        hfx=0.0, qfx=0.0, lh=0.0, grdflx=0.0, qsfc=0.0,
    )


# ---------------------------------------------------------------------------
# the user's failure mode, and the perturbed control that keeps it honest
# ---------------------------------------------------------------------------

def test_smois_exact_zero_column_is_nan_when_the_floor_is_bypassed(params):
    """Perturbed control: raw SMOIS=SH2O=0 through the WRF v4.6.1 driver
    mirror IS the reported failure chain.  TDFCND divides ``sh2o/smc``:
    the CUDA kernel's float 0/0 is NaN (NaN conductivity -> NaN GRDFLX ->
    NaN HFX); the float64 mirror images the same divide as
    ZeroDivisionError.  Either way the step does not produce a finite
    flux.  If Noah ever stops dividing by SMC this control fails and the
    floor's justification must be re-read."""
    col = _driver_column(np.zeros(4), np.full(4, 285.0), np.zeros(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        try:
            out = np_noah_column(col, params, DT, DZS)
        except ZeroDivisionError:
            return  # the mirror's image of the kernel's 0/0 NaN
    assert out["skip"] == 0  # a land run, not a masked column
    assert not np.isfinite(out["hfx"]) or not np.isfinite(out["grdflx"])


def test_smois_zero_single_column_through_ingest_runs_clean(params):
    """The user's exact failure mode: SMOIS=0.0 on one land column, through
    ingest and then the real driver path.  Before the floor the driver
    produced NaN HFX at step 0; now the column leaves ingest AT the soil
    type's SMCDRY and integrates finite."""
    soil_cat = 6  # LOAM, SMCDRY = 0.066
    state = preprocess_noah_soil(
        _era5_fields((1, 1), smois=0.0),
        soil_type=np.full((1, 1), soil_cat))
    expected = _smcdry(params, soil_cat)
    assert state.soil_moisture.shape == (4, 1, 1)
    np.testing.assert_array_equal(
        state.soil_moisture[:, 0, 0], np.full(4, expected))
    # sibling: SH2O is derived AFTER the floor, so it carries the same
    # protection (warm column: wholly liquid, equal to SMOIS).
    np.testing.assert_array_equal(
        state.liquid_moisture[:, 0, 0], state.soil_moisture[:, 0, 0])

    out = np_noah_column(
        _driver_column(
            state.soil_moisture[:, 0, 0],
            state.soil_temperature[:, 0, 0],
            state.liquid_moisture[:, 0, 0],
            soil=soil_cat),
        params, DT, DZS)
    assert out["skip"] == 0
    for name in ("hfx", "qfx", "lh", "grdflx", "tsk"):
        assert np.isfinite(out[name]), name
    assert np.isfinite(np.asarray(out["smois"])).all()

    receipt = state.moisture_floor
    assert receipt["policy"] == "land-below-smcdry-floored-to-smcdry"
    assert receipt["total_floored_cells"] == 4
    assert receipt["min_pre_floor"] == 0.0
    assert sorted(receipt["fields"]) == [
        "SMOIS_L1", "SMOIS_L2", "SMOIS_L3", "SMOIS_L4"]
    for entry in receipt["fields"].values():
        assert entry["floored_cells"] == 1
        assert entry["min_pre_floor"] == 0.0
        assert entry["smcdry_applied_min"] == expected
    assert "module_initialize_real.F" in \
        receipt["wrf_reference"]["wrf_citation"]


def test_hard_zero_threshold_is_gone_positive_subphysical_also_floored(
        params):
    """1e-12 survived the old clip and 0.0 did not; both are sub-physical
    and both now leave ingest at SMCDRY."""
    state = preprocess_noah_soil(
        _era5_fields((1, 1), smois=1e-12),
        soil_type=np.full((1, 1), 6))
    np.testing.assert_array_equal(
        state.soil_moisture[:, 0, 0], np.full(4, _smcdry(params, 6)))
    assert state.moisture_floor["total_floored_cells"] == 4


# ---------------------------------------------------------------------------
# a negative-input ERA5 array is floored and counted
# ---------------------------------------------------------------------------

def test_negative_era5_values_floored_counted_and_healthy_cells_untouched(
        params):
    shape = (2, 3)
    fields = _era5_fields(shape, smois=0.30)
    # Cell (0,0): the whole ERA5 column undershoots to -1e-4 --
    # overshoot-band admissible, previously clipped to exactly 0.0 on
    # every Noah layer.  Cell (1,1): the shallow layer alone undershoots;
    # the vertical interpolation dilutes it to a POSITIVE sub-SMCDRY
    # value at Noah L1 (the other half of the ruling's mask).
    for name in _MOIST_NAMES:
        fields[name] = np.full(shape, 0.30)
        fields[name][0, 0] = -1e-4
    fields["SM000007"][1, 1] = -1e-4
    soil_type = np.full(shape, 3)  # SANDY LOAM, SMCDRY = 0.047
    state = preprocess_noah_soil(fields, soil_type=soil_type)

    smcdry = _smcdry(params, 3)
    receipt = state.moisture_floor
    assert receipt["total_floored_cells"] == 5  # 4 layers + 1 diluted L1
    assert receipt["min_pre_floor"] == -1e-4  # pre-CLIP honesty, not 0.0
    touched = receipt["fields"]
    assert sorted(touched) == [
        "SMOIS_L1", "SMOIS_L2", "SMOIS_L3", "SMOIS_L4"]
    assert touched["SMOIS_L1"]["floored_cells"] == 2
    assert touched["SMOIS_L1"]["min_pre_floor"] == -1e-4
    for name in ("SMOIS_L2", "SMOIS_L3", "SMOIS_L4"):
        assert touched[name]["floored_cells"] == 1
        assert touched[name]["min_pre_floor"] == -1e-4
    np.testing.assert_array_equal(
        state.soil_moisture[:, 0, 0], np.full(4, smcdry))
    assert state.soil_moisture[0, 1, 1] == smcdry
    assert (state.soil_moisture >= 0.0).all()
    # Healthy land cells are byte-untouched by the floor.
    healthy = np.ones(shape, dtype=bool)
    healthy[0, 0] = False
    healthy[1, 1] = False
    reference = preprocess_noah_soil(
        _era5_fields(shape, smois=0.30), soil_type=soil_type)
    np.testing.assert_array_equal(
        state.soil_moisture[:, healthy], reference.soil_moisture[:, healthy])


# ---------------------------------------------------------------------------
# do no harm: a healthy array is byte-identical through ingest
# ---------------------------------------------------------------------------

def test_healthy_array_is_byte_identical_and_receiptless(params):
    shape = (3, 4)
    rng = np.random.default_rng(7)
    tslb = 284.0
    fields = _era5_fields(shape, tslb=tslb)
    moist = {}
    for name in _MOIST_NAMES:
        # Everything at or above the wettest SMCDRY in the table.
        moist[name] = 0.15 + 0.25 * rng.random(shape)
        fields[name] = moist[name]
    # One water column too: water handling must stay on its own path.
    fields["LANDSEA"][2, 3] = 0.0
    soil_type = np.full(shape, 6)
    state = preprocess_noah_soil(fields, soil_type=soil_type)

    # The receipt is EMPTY: zero floored cells means zero receipt noise.
    assert state.moisture_floor == {}

    # Byte-identity against the module's own unfloored pipeline: layer
    # values at ERA5 midpoints bracketed by the repeated shallow/deep
    # layer, interpolated to Noah's midpoints, clipped to 0..1.
    moistures = [fields[name] for name in _MOIST_NAMES]
    zsource = np.concatenate(([0.0], ERA5_LAYER_MIDPOINTS_M, [3.0]))
    nodes = np.stack([moistures[0], *moistures, moistures[-1]])
    expected = np.clip(
        _interp_nodes(nodes, zsource, NOAH_LAYER_MIDPOINTS_M), 0.0, 1.0)
    expected[:, 2, 3] = 1.0  # open water fill
    assert state.soil_moisture.tobytes() == expected.tobytes()
    expected_sh2o = sh2o_init(
        expected, state.soil_temperature, soil_type, params)
    assert state.liquid_moisture.tobytes() == expected_sh2o.tobytes()


def test_water_and_sea_ice_columns_are_never_floored():
    shape = (1, 2)
    fields = _era5_fields(shape, tslb=274.0, smois=0.0)
    fields["LANDSEA"] = np.zeros(shape)      # all water...
    fields["SEAICE"] = np.array([[0.0, 1.0]])  # ...one cell iced over
    fields["SST"] = np.full(shape, 274.0)
    state = preprocess_noah_soil(fields, soil_type=np.full(shape, 6))
    assert state.moisture_floor == {}
    np.testing.assert_array_equal(
        state.soil_moisture, np.ones((4,) + shape))
    # sea-ice SH2O=0 (LSMSCHEME postprocessing) is untouched by the floor
    np.testing.assert_array_equal(
        state.liquid_moisture[:, 0, 1], np.zeros(4))


def test_land_cell_with_soilparm_water_category_is_left_alone():
    """SOILPARM row 14 (WATER) has SMCDRY=0: no positive floor exists, so
    the cell passes through exactly as before the fix."""
    state = preprocess_noah_soil(
        _era5_fields((1, 1), smois=0.20),
        soil_type=np.full((1, 1), 14))
    assert state.moisture_floor == {}
    np.testing.assert_allclose(state.soil_moisture[:, 0, 0], 0.20)
