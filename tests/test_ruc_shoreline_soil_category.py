"""A land column carrying the water soil category, end to end.

The reopening battery prepared native HRRR RUC cleanly and then died on the
very first surface call with ``ValueError: mavail must be finite``, zero model
time advanced.  All 31 non-finite columns of 30,720 carried ``SOILTYP = 14``
(water) while packed into the land execution mask -- the Lake Erie shoreline,
where a land ``LU_INDEX`` meets a water ``SCT_DOM``.

WRF never lets that column reach the solver.  ``real.exe``'s final
landmask/category reconciliation
(``dyn_em/module_initialize_real.F:3608-3650``) rewrites it -- to land
(``IVGTYP 5``, ``ISLTYP 8``) when it has a soil temperature, to water when it
has an SST, and aborts with ``mismatch_landmask_ivgtyp`` when it has neither.
RUC LSM assumes that and never re-verifies it: ``soilvegin``
(``phys/module_sf_ruclsm.F:6973-6984``) has no ``else`` for ``isltyp == 14``,
so the column keeps the zeroed ``intent(out)`` parameters of
``:6917-6926`` -- ``ref == qmin == 0`` -- and ``:913`` evaluates ``0./0.``
into ``MAVAIL``.  The gpuwm transcription reaches the same arithmetic at
``gpuwm/core/ruc.py:9231-9240``.

So the fix belongs where WRF puts it, at initialization, and this file proves
both halves: the reconciled column runs a finite first surface step, and the
UNreconciled column -- the battery's exact input -- still kills it.  A gate
nobody has seen fail is not evidence.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from gpuwm.core.landuse import initialize_landuse
from gpuwm.core.ruc import ruc_land_surface_step
from gpuwm.core.ruc_runtime import (C1SN, C2SN, DEFINED_ILNB, ISNCOVR_OPT,
                                    RucRuntimeParameters)

#: MODIS-with-lakes identity, the one native HRRR preparation uses.
_MMINLU = "MODIFIED_IGBP_MODIS_NOAH"
_ISWATER = 17
_ISLAKE = 21
_ISICE = 15
_NSOIL = 9

#: The battery's own histogram of the 31 bad columns: a water soil category
#: (14) under ordinary vegetated land categories.  12 is croplands.
_SHORELINE_VEGETATION = 12
_WATER_SOIL = 14

_PARAMS = RucRuntimeParameters()


def _surface_values(n: int, *, mavail) -> dict:
    """``n`` warm columns in LSMRUC's own argument names."""

    values = {
        "soilmois": np.full((_NSOIL, n), 0.28, np.float32),
        "sh2o": np.full((_NSOIL, n), 0.28, np.float32),
        "tso": np.stack([np.full(n, 297.0 - level, np.float32)
                         for level in np.linspace(0.0, 8.0, _NSOIL)]),
        "smfr3d": np.zeros((_NSOIL, n), np.float32),
        "keepfr3dflag": np.zeros((_NSOIL, n), np.float32),
    }
    scalars = {
        "snow": 0.0, "snowh": 0.0, "snowc": 0.0, "canwat": 0.0,
        "snoalb": 0.7, "alb": 0.2, "emiss": 0.98, "lai": 2.0,
        "sfcexc": 0.02, "z0": 0.05, "znt": 0.05,
        "soilt": 303.0, "hfx": 0.0, "qfx": 0.0, "lh": 0.0,
        "sfcevp": 0.0, "sfcrunoff": 0.0, "udrunoff": 0.0, "acrunoff": 0.0,
        "grdflx": 0.0, "acsnow": 0.0, "snom": 0.0, "qvg": 0.011, "qcg": 0.0,
        "dew": 0.0, "qsfc": 0.011, "qsg": 0.011, "chklowq": 1.0,
        "soilt1": 303.0, "tsnav": -6.0, "smavail": 0.0, "smmax": 0.0,
        "rhosnf": 200.0, "precipfr": 0.0, "snowfallac": 0.0,
        "z3d": 40.0, "p8w": 95000.0, "t3d": 300.0, "qv3d": 0.011,
        "qc3d": 0.0, "rho3d": 1.2, "rainbl": 0.0, "frzfrac": 1.0,
        "glw": 340.0, "gsw": 560.0, "chs": 0.02, "flqc": 0.02, "flhc": 20.0,
        "albbck": 0.2, "xland": 1.0, "xice": 0.0, "tbot": 288.0,
        "shdmin": 10.0, "shdmax": 80.0, "vegfra": 60.0,
    }
    values.update({name: np.full(n, value, np.float32)
                   for name, value in scalars.items()})
    values["mavail"] = np.asarray(mavail, dtype=np.float32)
    return values


def _first_surface_step(*, ivgtyp, isltyp, xland, mavail):
    n = int(np.size(isltyp))
    values = _surface_values(n, mavail=mavail)
    values["xland"] = np.asarray(xland, dtype=np.float32)
    return ruc_land_surface_step(
        values, dt=12.0, ktau=1, zs=_PARAMS.zs,
        ivgtyp=np.asarray(ivgtyp, np.int32),
        isltyp=np.asarray(isltyp, np.int32),
        em_core=0, iswater=_ISWATER, isice=_ISICE,
        ilnb=DEFINED_ILNB, ilnb_chain=False,
        c1sn=C1SN, c2sn=C2SN, isncovr_opt=ISNCOVR_OPT,
        mminlu=_PARAMS.dataset_identifier, parameters=_PARAMS.bundle)


def _shoreline_domain(**overrides):
    """One row: mismatch, pure water, and three ordinary land categories.

    The soil dimension is tested at more than one value on purpose -- the
    battery found category 14 to be the only failing class, and a fix that
    only ever sees 14 and 8 cannot show that.
    """

    inputs = dict(
        lu_index=np.array(
            [[_SHORELINE_VEGETATION, _ISWATER, _SHORELINE_VEGETATION,
              _SHORELINE_VEGETATION, _SHORELINE_VEGETATION]], np.int32),
        soil_type=np.array([[_WATER_SOIL, _WATER_SOIL, 4, 6, 11]], np.int32),
        landmask=np.array([[1.0, 0.0, 1.0, 1.0, 1.0]]),
        snow=0.0, xice=0.0,
        valid_time=datetime(2026, 7, 29, 23),
        cen_lat=41.5, mminlu=_MMINLU, iswater=_ISWATER, islake=_ISLAKE,
        isice=_ISICE,
    )
    inputs.update(overrides)
    return initialize_landuse(**inputs)


def test_real_reconciles_a_land_column_carrying_the_water_soil_category():
    """dyn_em/module_initialize_real.F:3615-3624, the warm-soil arm."""

    landuse = _shoreline_domain(
        soil_temperature=np.full((_NSOIL, 1, 5), 291.0, np.float32))

    # The mismatched column becomes WRF's artificial silty clay loam under
    # cropland/grassland mosaic, and stays land.
    np.testing.assert_array_equal(landuse.isltyp, [[8, 14, 4, 6, 11]])
    np.testing.assert_array_equal(
        landuse.ivgtyp,
        [[5, _ISWATER, _SHORELINE_VEGETATION, _SHORELINE_VEGETATION,
          _SHORELINE_VEGETATION]])
    np.testing.assert_array_equal(landuse.xland, [[1.0, 2.0, 1.0, 1.0, 1.0]])
    np.testing.assert_array_equal(landuse.landmask, [[1.0, 0.0, 1.0, 1.0, 1.0]])

    # Pure water and the three ordinary land categories are untouched: the
    # reconciliation is not a blanket rewrite.
    assert landuse.isltyp[0, 1] == _WATER_SOIL
    assert landuse.ivgtyp[0, 1] == _ISWATER


def test_a_water_column_keeps_the_water_soil_category():
    """The pure-water control: nothing here is a mismatch."""

    landuse = _shoreline_domain(
        lu_index=np.array([[_ISWATER, _ISWATER]], np.int32),
        soil_type=np.array([[_WATER_SOIL, _WATER_SOIL]], np.int32),
        landmask=np.array([[0.0, 0.0]]),
        soil_temperature=np.full((_NSOIL, 1, 2), 291.0, np.float32))

    np.testing.assert_array_equal(landuse.isltyp, [[14, 14]])
    np.testing.assert_array_equal(landuse.ivgtyp, [[_ISWATER, _ISWATER]])
    np.testing.assert_array_equal(landuse.xland, [[2.0, 2.0]])


def test_a_mismatched_column_with_only_an_sst_becomes_water():
    """dyn_em/module_initialize_real.F:3625-3630, the SST arm."""

    landuse = _shoreline_domain(
        lu_index=np.array([[_SHORELINE_VEGETATION, _SHORELINE_VEGETATION]],
                          np.int32),
        soil_type=np.array([[_WATER_SOIL, 6]], np.int32),
        landmask=np.array([[1.0, 1.0]]),
        soil_temperature=np.zeros((_NSOIL, 1, 2), np.float32),
        sst=np.array([[290.0, 290.0]]))

    np.testing.assert_array_equal(
        landuse.ivgtyp, [[_ISWATER, _SHORELINE_VEGETATION]])
    np.testing.assert_array_equal(landuse.isltyp, [[14, 6]])
    np.testing.assert_array_equal(landuse.xland, [[2.0, 1.0]])


def test_a_mismatched_column_with_no_evidence_is_refused_like_wrf():
    """dyn_em/module_initialize_real.F:3631-3641, ``wrf_error_fatal``.

    WRF aborts rather than guessing, and so does this.  No clamp is invented
    for a column the source refuses to run.
    """

    with pytest.raises(ValueError, match="mismatch_landmask_ivgtyp"):
        _shoreline_domain(
            soil_temperature=np.zeros((_NSOIL, 1, 5), np.float32))

    # Omitting the evidence entirely is the same answer, not a silent pass.
    with pytest.raises(ValueError, match="mismatch_landmask_ivgtyp"):
        _shoreline_domain()


def test_the_reconciled_shoreline_column_runs_a_finite_first_surface_step():
    """The battery's failure, and the control that shows the gate can fire.

    ``mavail must be finite`` was raised out of
    ``ruc_surface_temperature_step`` before any model time advanced.  The
    reconciled categories run that same call to completion; the raw ones the
    battery fed it still do not.
    """

    landuse = _shoreline_domain(
        soil_temperature=np.full((_NSOIL, 1, 5), 291.0, np.float32))
    land = np.ravel(landuse.xland) < 1.5
    reconciled = _first_surface_step(
        ivgtyp=np.ravel(landuse.ivgtyp)[land],
        isltyp=np.ravel(landuse.isltyp)[land],
        xland=np.ravel(landuse.xland)[land],
        mavail=np.ravel(landuse.mavail)[land])
    for name in ("mavail", "soilt", "hfx", "qfx", "lh"):
        value = np.asarray(getattr(reconciled, name), dtype=np.float64)
        assert np.all(np.isfinite(value)), f"{name} is not finite"
    availability = np.asarray(reconciled.mavail, dtype=np.float64)
    assert np.all((availability >= 0.0) & (availability <= 1.0))

    # Control: the unreconciled shoreline column, exactly as the battery's
    # HRRR preparation handed it over -- a land XLAND with SOILTYP 14 and an
    # ordinary incoming MAVAIL.  The NaN is manufactured inside the driver by
    # ``ref - qmin == 0``, not supplied.
    with pytest.raises(ValueError, match="mavail must be finite"):
        _first_surface_step(
            ivgtyp=np.array([_SHORELINE_VEGETATION], np.int32),
            isltyp=np.array([_WATER_SOIL], np.int32),
            xland=np.array([1.0], np.float32),
            mavail=np.array([0.5], np.float32))
