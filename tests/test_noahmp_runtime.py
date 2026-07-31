"""Noah-MP, actually forecasting.

Every other Noah-MP test in this repository measures a routine against an
oracle CSV.  This one runs the model: ``sf_surface_physics=4`` over mixed land
and water, stepped through RK3, and it asserts the things that separate
"ported" from "runs":

* the selector reaches Noah-MP and not Noah, and the census proves it ran on
  land columns rather than skipping every one of them;
* the cold start produces a usable state -- the arrays that
  ``initialize_physics`` allocates at zero are not zero after
  ``NOAHMP_INIT``/``SNOW_INIT``, because ``TV = TG = 0 K`` is not a cold state
  but a state whose saturation vapour pressure is negative;
* the surface fluxes and the soil column actually move, which is what says
  the scheme is coupled rather than merely called;
* nothing goes non-finite;
* a restart round trip reproduces every carried array bit for bit **and** the
  next step from the restored state is bit identical -- with a companion test
  that perturbs one word and proves that comparison can fail; and
* a real wrfout is written and read back, including the snow stack on its own
  vertical axis.

The grid is small on purpose.  Seven of the column's leaves are device-batched
and the rest is still host FP32, about 0.55 ms per land column at width on the
reference card (see ``test_the_column_cost_is_what_the_registry_says``), so a
36-land-column domain is a few tens of milliseconds per LSM call; a 50x20
domain is still minutes.  That cost is the scheme's scaling blocker, it is
published in the registry warnings, and it is why this file does not grow the
grid to make the assertions look better.
"""

from __future__ import annotations

from datetime import datetime
import hashlib

import numpy as np
import pytest

from conftest import requires_gpu

import cupy as cp

from gpuwm.config import RunConfig
from gpuwm.core.noahmp_runtime import (NOAHMP_DIAGNOSTICS_2D,
                                       NOAHMP_STATE_2D,
                                       NOAHMP_STATE_INT_2D,
                                       NOAHMP_STATE_SNOWSOIL_3D,
                                       NOAHMP_STATE_SNOW_3D,
                                       NSNOW)

#: MODIS/IGBP categories out of the packaged MPTABLE land-use block.  Named
#: here rather than inlined so a reader can see that the vegetated columns are
#: grassland and that the glacier test really does select ISICE.
_GRASSLAND = 10
_ICE = 15
_LOAM = 6
_WATER_SOIL = 14

#: A summer mid-afternoon over the central United States: solar time is about
#: 11:20 local, so COSZ is well above zero and the shortwave forcing below is
#: physically consistent with it.  Noah-MP reads COSZ, XLAT, JULIAN and YR;
#: none of them has a defensible default, so the run binds all four.
_START = datetime(2026, 7, 1, 18, 0, 0)
_LATITUDE = 40.0
_LONGITUDE = -100.0


def _build(*, nx: int = 8, ny: int = 6, nz: int = 40, vegtyp: int = _GRASSLAND,
           soiltyp: int = _LOAM, water_columns: int = 2, snow_mm: float = 0.0,
           snow_depth_m: float = 0.0, xice=0.0, radiation=None,
           ra_physics: int = 0, radt_minutes: float = 12.0,
           mp_physics: int = 6, sf_sfclay_physics: int = 1,
           bl_pbl_physics: int = 1):
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    cfg = RunConfig(nx=nx, ny=ny, nz=nz, dx=3000.0, dy=3000.0, ztop=16000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    mp_physics=mp_physics,
                    sf_sfclay_physics=sf_sfclay_physics,
                    sf_surface_physics=4,
                    bl_pbl_physics=bl_pbl_physics, bldt=0.0,
                    ra_physics=ra_physics, radt_minutes=radt_minutes)

    def theta(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 300.0,
                        np.where(z < 1700.0, 300.0 + 0.030 * (z - 1500.0),
                                 306.0 + 0.0045 * (z - 1700.0)))

    def qvapor(z):
        z = np.asarray(z, np.float64)
        return np.where(z < 1500.0, 0.0110,
                        np.maximum(0.0110 - 5.0e-6 * (z - 1500.0), 1.0e-5))

    coord = make_vertical_coord(cfg.nz, stretch=2.0)
    base = make_base_state(coord, theta, p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, qvapor)
    state.u[...] = cp.float32(5.0)
    state.v[...] = cp.float32(1.0)

    landmask = np.ones((cfg.ny, cfg.nx), np.float64)
    if water_columns:
        landmask[:, -water_columns:] = 0.0
    land = landmask == 1.0
    # The land surface is deliberately WARMER than the 300 K boundary layer.
    # An earlier version of this file used a 297 K ground and asserted a
    # positive sensible heat flux, which is not a defect in the port: the
    # first model level sits near 300 K, so the flux was correctly negative.
    # Choosing an unstable surface exercises the branch a daytime forecast
    # actually takes instead of asserting a sign the setup forbids.
    tsk = np.full((cfg.ny, cfg.nx), 303.0)
    tsk[~land] = 294.0
    soil_t = np.stack([tsk - 1.0, tsk - 2.0, tsk - 3.0, tsk - 4.0])
    soil_m = np.full((4, cfg.ny, cfg.nx), 0.28)
    soil_m[:, ~land] = 1.0
    latitude = np.full((cfg.ny, cfg.nx), _LATITUDE, np.float64)
    longitude = np.full((cfg.ny, cfg.nx), _LONGITUDE, np.float64)

    driver = initialize_physics(
        state, cfg, landmask=landmask, tsk=tsk,
        soil_temperature=soil_t, soil_moisture=soil_m,
        liquid_moisture=soil_m,
        ivgtyp=np.where(land, vegtyp, 17),
        isltyp=np.where(land, soiltyp, _WATER_SOIL),
        vegfra=60.0, tmn=288.0, swdown=700.0, glw=340.0, pblh=500.0,
        snow=snow_mm, snow_depth=snow_depth_m, xice=xice,
        noahmp_start_time=_START, noahmp_latitude=latitude,
        noahmp_longitude=longitude, radiation=radiation)
    return state, cfg, driver


_CARRIED = (*NOAHMP_STATE_2D, *NOAHMP_STATE_INT_2D,
            *NOAHMP_STATE_SNOW_3D, *NOAHMP_STATE_SNOWSOIL_3D,
            "tslb", "smois", "sh2o", "snow", "snowh", "canwat", "lai",
            "tsk", "hfx", "qfx", "lh", "grdflx", "albedo", "snowc", "emiss",
            "qsfc", "z0", "znt", "sfcrunoff", "udrunoff", "acsnow", "acsnom")


def _carried_digest(driver):
    """The historical Noah-MP digest spelling, including dtype identity."""
    digest = hashlib.sha256()
    for name in sorted(_CARRIED):
        array = np.ascontiguousarray(cp.asnumpy(driver.fields[name]))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# admission
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_selector_reaches_noahmp_and_the_cold_start_is_usable():
    state, cfg, driver = _build()
    assert driver.scheme_dispatch["sf_surface_physics"] == "_run_noahmp"
    assert driver.noahmp_params is not None
    assert driver.noahmp_geometry is not None

    # NOAHMP_INIT/SNOW_INIT ran at construction.  A runner that skipped them
    # would leave TV/TG at the allocation zeros, which is the single most
    # likely way for this to "work" and be wrong.
    for name in ("tvxy", "tgxy", "tahxy", "eahxy", "alboldxy"):
        array = cp.asnumpy(driver.fields[name])
        assert np.isfinite(array).all(), name
        assert float(np.abs(array).min()) > 0.0, name
    tg = cp.asnumpy(driver.fields["tgxy"])
    assert 200.0 < float(tg.min()) and float(tg.max()) < 350.0
    # ZSNSO's soil half is the interface depth ladder, negative downward.
    zsnso = cp.asnumpy(driver.fields["zsnsoxy"])
    np.testing.assert_allclose(zsnso[NSNOW:, 0, 0],
                               [-0.1, -0.4, -1.0, -2.0], rtol=1e-6)


@requires_gpu
@pytest.mark.parametrize("build_kwargs", [
    {"nx": 8, "ny": 6, "water_columns": 2},
    {
        "nx": 6, "ny": 4, "water_columns": 1,
        "snow_mm": 45.0, "snow_depth_m": 0.16,
    },
])
def test_device_leaves_preserve_the_six_step_carried_state(
        build_kwargs, monkeypatch):
    """Pin the full runtime seam, not only the leaves' own outputs.

    The earlier report called this a 47-array digest; ``_CARRIED`` has always
    contained 45 names.  Its snowpack digest remains the historical value, but
    the bare absolute digest has moved with shared surface-layer work outside
    this lane.  Run the unmodified host routines beside the device batches at
    the current tree instead of pinning somebody else's changing shared input.

    Both batched leaves are covered: BARE_FLUX runs on **every** land column,
    so a defect in its packing would move every column of both domains.
    """
    from gpuwm.core.dycore import step
    import gpuwm.core.noahmp_runtime as runtime_module

    device_state, device_cfg, device_driver = _build(**build_kwargs)
    for _ in range(6):
        step(device_state, device_cfg)

    monkeypatch.setattr(runtime_module, "LEAF_BATCH_EVALUATOR",
                        runtime_module.evaluate_leaf_batch_on_host)
    host_state, host_cfg, host_driver = _build(**build_kwargs)
    for _ in range(6):
        step(host_state, host_cfg)

    assert len(_CARRIED) == 45
    assert _carried_digest(device_driver) == _carried_digest(host_driver)
    for name in _CARRIED:
        np.testing.assert_array_equal(
            cp.asnumpy(device_driver.fields[name]),
            cp.asnumpy(host_driver.fields[name]),
            err_msg=f"device/host {name}",
        )


#: The smallest perturbation of one output word of each batched leaf that the
#: six-step carried-state digest can see, **measured** on this domain rather
#: than assumed.  VEGE_FLUX's TV and WATER's RUNSRF are carried into TVXY and
#: SFCRUNOFF unweighted, so one ULP survives.  BARE_FLUX's TGB is not: this
#: domain is grassland at SHDMAX 60%, so :2289's tile average forms
#: ``FVEG*TGV + (1-FVEG)*TGB`` and one ULP of TGB, scaled by 0.4, falls below
#: half an ULP of a ~300 K sum and rounds away.  Two ULP does not.
#:
#: This is a property of the tile average, not a weakness of the leaf gate:
#: ``tests/test_noahmp_bareflux_cuda.py`` holds BARE_FLUX itself to max_ulp 0
#: on every fixture row, where nothing is weighted.
_SMALLEST_VISIBLE_NUDGE = (
    ("vege_flux", "TV", 1),
    ("bare_flux", "tgb", 2),
    ("water", "runsrf", 1),
)


@requires_gpu
@pytest.mark.parametrize("leaf,attribute,ulps", _SMALLEST_VISIBLE_NUDGE)
def test_the_six_step_state_comparison_can_fail(leaf, attribute, ulps,
                                                monkeypatch):
    """The paired-authority gate above has to be able to reject a device leaf.

    A comparison of two runs that share every code path except the leaf batch
    proves nothing until it is shown failing.  One perturbed output word per
    batched leaf, at the smallest magnitude measured to be observable, and the
    digest must move for all three.
    """
    import struct

    from gpuwm.core.dycore import step
    import gpuwm.core.noahmp_runtime as runtime_module

    build_kwargs = {"nx": 4, "ny": 3, "water_columns": 1}
    state, cfg, driver = _build(**build_kwargs)
    for _ in range(6):
        step(state, cfg)
    honest = _carried_digest(driver)

    def nudged(requested, calls):
        out = runtime_module.evaluate_leaf_batch_on_device(requested, calls)
        if requested == leaf and out:
            value = float(getattr(out[0], attribute))
            bits = struct.unpack("<I", struct.pack("<f", value))[0]
            setattr(out[0], attribute, np.float32(
                struct.unpack("<f", struct.pack("<I", bits + ulps))[0]))
        return out

    monkeypatch.setattr(runtime_module, "LEAF_BATCH_EVALUATOR", nudged)
    nudged_state, nudged_cfg, nudged_driver = _build(**build_kwargs)
    for _ in range(6):
        step(nudged_state, nudged_cfg)

    assert _carried_digest(nudged_driver) != honest, (
        f"a {ulps}-ULP nudge of one {leaf} {attribute} left the six-step "
        "carried state digest unchanged; the gate cannot see a leaf defect")


@requires_gpu
def test_one_ulp_of_bare_flux_tgb_really_is_absorbed_by_the_tile_average():
    """Record the measurement behind the ``2`` in the table above.

    Asserting the smaller nudge is invisible is what stops somebody "tidying"
    the 2 back to a 1 and getting a test that fails for a reason nobody
    understands -- and it is the evidence that the 2 was measured rather than
    picked until the test went green.
    """
    import struct

    from gpuwm.core.dycore import step
    import gpuwm.core.noahmp_runtime as runtime_module

    build_kwargs = {"nx": 4, "ny": 3, "water_columns": 1}
    state, cfg, driver = _build(**build_kwargs)
    for _ in range(6):
        step(state, cfg)
    honest = _carried_digest(driver)

    previous = runtime_module.LEAF_BATCH_EVALUATOR

    def one_ulp(requested, calls):
        out = previous(requested, calls)
        if requested == "bare_flux" and out:
            bits = struct.unpack("<I", struct.pack("<f", float(out[0].tgb)))[0]
            out[0].tgb = np.float32(
                struct.unpack("<f", struct.pack("<I", bits + 1))[0])
        return out

    runtime_module.LEAF_BATCH_EVALUATOR = one_ulp
    try:
        nudged_state, nudged_cfg, nudged_driver = _build(**build_kwargs)
        for _ in range(6):
            step(nudged_state, nudged_cfg)
        absorbed = _carried_digest(nudged_driver) == honest
    finally:
        runtime_module.LEAF_BATCH_EVALUATOR = previous

    assert absorbed, (
        "one ULP of BARE_FLUX TGB is now visible in the six-step digest; the "
        "2-ULP entry in _SMALLEST_VISIBLE_NUDGE can be tightened to 1")


@requires_gpu
def test_runtime_does_not_execute_the_host_leaves(monkeypatch):
    """Host tripwires prove the runtime path reaches both device batches.

    One tripwire per batched leaf.  BARE_FLUX needs its own because it runs on
    every column including the ones that never reach VEGE_FLUX, so the
    VEGE_FLUX tripwire alone would say nothing about it.
    """
    from gpuwm.core.dycore import step
    import gpuwm.core.noahmp_energy as energy_module

    state, cfg, driver = _build(nx=4, ny=2, water_columns=0)

    def tripwire(leaf):
        def fire(*args, **kwargs):
            del args, kwargs
            raise AssertionError(f"runtime executed CPython {leaf}")
        return fire

    monkeypatch.setattr(energy_module, "vege_flux", tripwire("VEGE_FLUX"))
    monkeypatch.setattr(energy_module, "bare_flux", tripwire("BARE_FLUX"))
    step(state, cfg)
    assert driver.last_noahmp_census == {
        "land": cfg.nx * cfg.ny, "water": 0, "sea_ice": 0}


@requires_gpu
def test_the_host_tripwires_would_fire_on_the_host_authority(monkeypatch):
    """...and the tripwires above are live, not decorative."""
    from gpuwm.core.dycore import step
    import gpuwm.core.noahmp_energy as energy_module
    import gpuwm.core.noahmp_runtime as runtime_module

    for leaf in ("vege_flux", "bare_flux"):
        state, cfg, _driver = _build(nx=4, ny=2, water_columns=0)
        monkeypatch.setattr(runtime_module, "LEAF_BATCH_EVALUATOR",
                            runtime_module.evaluate_leaf_batch_on_host)

        def fire(*args, **kwargs):
            del args, kwargs
            raise AssertionError(f"host {leaf} ran")

        monkeypatch.setattr(energy_module, leaf, fire)
        with pytest.raises(AssertionError, match=f"host {leaf} ran"):
            step(state, cfg)
        monkeypatch.undo()


@requires_gpu
def test_the_column_tiling_cannot_change_the_answer(monkeypatch):
    """Noah-MP has no horizontal coupling, so the batch tile must be inert.

    ``COLUMN_BATCH`` bounds how many suspended column frames are held at once.
    It is a memory knob, and a memory knob that moved the forecast would be a
    coupling defect.  One column per batch is the extreme case.
    """
    from gpuwm.core.dycore import step
    import gpuwm.core.noahmp_runtime as runtime_module

    build_kwargs = {"nx": 5, "ny": 3, "water_columns": 1}
    wide_state, wide_cfg, wide_driver = _build(**build_kwargs)
    for _ in range(3):
        step(wide_state, wide_cfg)

    monkeypatch.setattr(runtime_module, "COLUMN_BATCH", 1)
    tiled_state, tiled_cfg, tiled_driver = _build(**build_kwargs)
    for _ in range(3):
        step(tiled_state, tiled_cfg)

    assert _carried_digest(wide_driver) == _carried_digest(tiled_driver)


@requires_gpu
def test_out_of_identity_configurations_are_refused_before_the_run():
    """The gate has to fire at configuration time, not three hours in."""
    from gpuwm.config import (NOAHMP_OPTION_IDENTITY, RunConfig,
                              validate_run_config)

    base = dict(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                sf_sfclay_physics=1, sf_surface_physics=4, bl_pbl_physics=1)
    validate_run_config(RunConfig(**base))
    for name, admitted in NOAHMP_OPTION_IDENTITY.items():
        other = (admitted + 1) if isinstance(admitted, int) else admitted + 1.0
        with pytest.raises(ValueError, match="Noah-MP option identity"):
            validate_run_config(RunConfig(**base, **{name: other}))


@requires_gpu
def test_a_noah_run_gains_no_noahmp_arrays():
    """The negative control for the allocation gate.

    Forty-nine extra arrays in every Noah run would change that run's restart
    inventory, its VRAM budget and its health-descriptor count.
    """
    from gpuwm.core.grid import make_base_state, make_vertical_coord
    from gpuwm.core.moist import init_moist_balanced
    from gpuwm.core.physics import initialize_physics

    cfg = RunConfig(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                    dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                    sf_sfclay_physics=1, sf_surface_physics=2,
                    bl_pbl_physics=1)
    coord = make_vertical_coord(cfg.nz)
    base = make_base_state(coord, lambda z: 300.0 + 0.004 * np.asarray(z),
                           p_surf=cfg.p_surf, ztop=cfg.ztop)
    state = init_moist_balanced(cfg, coord, base, lambda z: 0.008 + 0.0 * z)
    driver = initialize_physics(state, cfg)
    for name in (*NOAHMP_STATE_2D, *NOAHMP_STATE_INT_2D,
                 *NOAHMP_STATE_SNOW_3D, *NOAHMP_STATE_SNOWSOIL_3D,
                 *NOAHMP_DIAGNOSTICS_2D):
        assert name not in driver.fields, name
    assert driver.noahmp_params is None


@requires_gpu
def test_the_vram_preflight_counts_the_noahmp_arrays():
    """Under-counting VRAM is a correctness bar on this hardware."""
    from gpuwm.core.preflight import physics_array_shapes

    noah = RunConfig(nx=6, ny=4, nz=20, dx=3000.0, dy=3000.0, ztop=12000.0,
                     dt=12.0, run_seconds=0.0, time_step_sound=4, moist=True,
                     sf_sfclay_physics=1, sf_surface_physics=2,
                     bl_pbl_physics=1)
    noahmp = RunConfig(**{**{f.name: getattr(noah, f.name)
                             for f in noah.__dataclass_fields__.values()},
                          "sf_surface_physics": 4})
    added = set(physics_array_shapes(noahmp)) - set(physics_array_shapes(noah))
    for name in (*NOAHMP_STATE_2D, *NOAHMP_STATE_INT_2D,
                 *NOAHMP_DIAGNOSTICS_2D):
        assert f"fields/{name}" in added, name
    shapes = physics_array_shapes(noahmp)
    for name in NOAHMP_STATE_SNOW_3D:
        assert shapes[f"fields/{name}"] == (NSNOW, noah.ny, noah.nx), name
    for name in NOAHMP_STATE_SNOWSOIL_3D:
        assert shapes[f"fields/{name}"] == (NSNOW + 4, noah.ny, noah.nx), name


@requires_gpu
def test_noahmp_coszen_is_carried_at_radiation_cadence_and_species_are_threaded(
        monkeypatch):
    """The surface sees the last radiation COSZEN and all six ARW amounts."""
    from gpuwm.core import physics
    from gpuwm.core.physics import RadiationResult
    from gpuwm.core.surface_forcing import (
        SURFACE_PRECIPITATION_FIELD_NAMES,
    )

    radiation_calls = []

    def radiation(**kwargs):
        state = kwargs["state"]
        radiation_calls.append(float(state.elapsed_seconds))
        nz, ny, nx = state.p.shape
        zeros = cp.zeros((nz, ny, nx), cp.float32)
        value = cp.float32(0.25 if len(radiation_calls) == 1 else 0.75)
        return RadiationResult(
            zeros, zeros, cp.full((ny, nx), 700.0, cp.float32),
            cp.full((ny, nx), 340.0, cp.float32),
            coszen=cp.full((ny, nx), value, cp.float32))

    state, cfg, driver = _build(
        nx=4, ny=2, water_columns=1, radiation=radiation,
        ra_physics=90, radt_minutes=1.0)
    species_values = {
        "rain_convective": 0.1,
        "rain_nonconvective": 0.2,
        "rain_shallow_convective": 0.3,
        "snow_nonconvective": 0.4,
        "graupel_nonconvective": 0.5,
        "hail_nonconvective": 0.6,
    }
    for member, value in species_values.items():
        driver.fields[SURFACE_PRECIPITATION_FIELD_NAMES[member]][...] = \
            cp.float32(value)
    seen = []

    def capture(_fields, _atmosphere, *, precipitation, coszen, **_kwargs):
        seen.append((
            cp.asnumpy(coszen).copy(),
            {name: cp.asnumpy(getattr(precipitation, name)).copy()
             for name in species_values},
        ))
        return {"land": 6, "water": 2, "sea_ice": 0}

    monkeypatch.setattr(physics, "noahmp_lsm_step", capture)
    driver.compute(state, cfg)
    state.elapsed_seconds = 12.0
    driver.compute(state, cfg)
    state.elapsed_seconds = 60.0
    driver.compute(state, cfg)

    assert radiation_calls == [0.0, 60.0]
    np.testing.assert_array_equal(seen[0][0], np.full((2, 4), 0.25,
                                                      np.float32))
    np.testing.assert_array_equal(seen[1][0], seen[0][0])
    np.testing.assert_array_equal(seen[2][0], np.full((2, 4), 0.75,
                                                      np.float32))
    for member, value in species_values.items():
        np.testing.assert_array_equal(
            seen[0][1][member], np.full((2, 4), value, np.float32))
        np.testing.assert_array_equal(
            seen[1][1][member], np.zeros((2, 4), np.float32))


@requires_gpu
def test_radiation_free_noahmp_seeds_coszen_at_wrfs_half_interval():
    from gpuwm.core.dudhia import wrf_solar_geometry

    _state, cfg, driver = _build(nx=4, ny=2, water_columns=1)
    expected, _ = wrf_solar_geometry(
        _START,
        np.full((2, 4), _LATITUDE, np.float64),
        np.full((2, 4), _LONGITUDE, np.float64),
        hour_offset_seconds=0.5 * cfg.radt_minutes * 60.0)
    np.testing.assert_array_equal(
        cp.asnumpy(driver.fields["coszen"]),
        np.asarray(expected, np.float32))


# ---------------------------------------------------------------------------
# forecasting
# ---------------------------------------------------------------------------

@requires_gpu
def test_noahmp_forecasts_and_stays_finite():
    from gpuwm.core.dycore import step, stability_report

    state, cfg, driver = _build()
    before = {name: cp.asnumpy(driver.fields[name]).copy()
              for name in _CARRIED}

    for _ in range(20):
        step(state, cfg)
        assert not stability_report(state, cfg)["nan"]

    census = driver.last_noahmp_census
    assert census is not None
    assert census["land"] == cfg.ny * (cfg.nx - 2)
    assert census["water"] == cfg.ny * 2
    assert census["sea_ice"] == 0

    for name in _CARRIED:
        array = cp.asnumpy(driver.fields[name])
        assert np.isfinite(array).all(), name

    # The coupled quantities must have MOVED on land.  A runner that returned
    # its inputs would pass every finiteness assertion above.
    #
    # TAUSSXY is deliberately absent from this list, and its absence is a
    # measured fact rather than a convenience: SNOW_AGE
    # (module_sf_noahmplsm.F) sets TAGE = 0 whenever SNEQV <= 0 and then
    # TAUSS = TAGE, so on a snow-free column the snow age is pinned at its
    # cold zero for the whole run.  It is asserted in
    # ``test_a_snowpack_column_runs_the_snow_stack`` instead, where a
    # snowpack exists for it to age.
    for name in ("tgxy", "tvxy", "tslb", "sh2o", "hfx", "lh", "grdflx",
                 "emiss", "albedo", "lai", "chxy", "cmxy"):
        got = cp.asnumpy(driver.fields[name])
        was = before[name]
        moved = (got[..., :cfg.nx - 2] != was[..., :cfg.nx - 2])
        assert moved.any(), f"{name} did not move on any land column"

    # Water columns are skipped by WRF's own CYCLE ILOOP, so every carrier
    # that only the LSM writes must be untouched there.
    for name in ("tgxy", "tvxy", "taussxy"):
        got = cp.asnumpy(driver.fields[name])[..., cfg.nx - 2:]
        np.testing.assert_array_equal(
            got, before[name][..., cfg.nx - 2:], err_msg=f"water/{name}")

    # A daytime, vegetated, unsaturated column absorbs shortwave, transpires
    # and warms the ground relative to the deep soil.
    fsa = cp.asnumpy(driver.fields["fsaxy"])[..., :cfg.nx - 2]
    assert float(fsa.min()) > 0.0
    etran = cp.asnumpy(driver.fields["etranxy"])[..., :cfg.nx - 2]
    assert float(etran.max()) > 0.0
    albedo = cp.asnumpy(driver.fields["albedo"])[..., :cfg.nx - 2]
    assert 0.05 < float(albedo.min()) and float(albedo.max()) < 0.9

    # The sensible heat flux must follow the ground-air temperature
    # difference on every land column.  That is a weaker claim than "HFX is
    # positive" and a much harder one to satisfy by accident: it ties the
    # flux the driver wrote back to the state the column solved for.
    from gpuwm.core.physics import _prepare_atmosphere

    atmosphere = _prepare_atmosphere(state)
    t1 = cp.asnumpy(atmosphere["temperature"][0])[..., :cfg.nx - 2]
    tg = cp.asnumpy(driver.fields["tgxy"])[..., :cfg.nx - 2]
    hfx = cp.asnumpy(driver.fields["hfx"])[..., :cfg.nx - 2]
    assert float(np.abs(hfx).min()) > 0.0
    np.testing.assert_array_equal(np.sign(hfx), np.sign(tg - t1))
    # With a 303 K ground under a 300 K boundary layer the unstable branch is
    # the one taken, so the flux is upward.
    assert float(hfx.min()) > 0.0


@requires_gpu
def test_a_snowpack_column_runs_the_snow_stack():
    """Without this the snow topology ladder is never entered at all."""
    from gpuwm.core.dycore import step

    state, cfg, driver = _build(nx=6, ny=4, snow_mm=60.0, snow_depth_m=0.30)
    isnow = cp.asnumpy(driver.fields["isnowxy"])[..., :cfg.nx - 2]
    # SNOW_INIT's 0.25 < SNODEP <= 0.45 rung is three layers.
    assert set(np.unique(isnow)) == {-3}
    snice = cp.asnumpy(driver.fields["snicexy"])[..., :, :cfg.nx - 2]
    assert float(snice.sum()) > 0.0
    before = {name: cp.asnumpy(driver.fields[name]).copy()
              for name in ("snicexy", "snliqxy", "tsnoxy", "taussxy")}
    for _ in range(10):
        step(state, cfg)
    for name in ("snicexy", "snliqxy", "tsnoxy", "zsnsoxy"):
        array = cp.asnumpy(driver.fields[name])
        assert np.isfinite(array).all(), name
    snowc = cp.asnumpy(driver.fields["snowc"])[..., :cfg.nx - 2]
    assert float(snowc.min()) > 0.0
    # With a snowpack present SNOW_AGE has something to age, so TAUSSXY is
    # no longer inert -- which is what makes its absence from the snow-free
    # test a property of WRF rather than a hole in the gate.
    for name in ("snicexy", "tsnoxy", "taussxy"):
        got = cp.asnumpy(driver.fields[name])
        moved = got[..., :cfg.nx - 2] != before[name][..., :cfg.nx - 2]
        assert moved.any(), f"{name} did not move under a snowpack"


@requires_gpu
def test_the_parameter_handle_cache_is_not_mutated_by_a_column():
    """The cache is only sound if ``sflx`` treats the handle as read-only.

    ``NoahmpRuntimeParameters.handle`` memoises one ``SflxParameters`` per
    distinct land-use/soil identity, so a 250,000-column nest builds a handful
    of handles instead of 250,000.  If any column mutated one, every later
    column with the same identity would run with the mutation.  This hashes
    every field of every cached handle across twenty steps.
    """
    import hashlib

    from gpuwm.core.dycore import step

    def digest(params):
        h = hashlib.sha256()
        for key in sorted(params._handle_cache):
            handle = params._handle_cache[key]
            h.update(repr(key).encode())
            for name in sorted(handle.__dataclass_fields__):
                value = getattr(handle, name)
                if name == "energy":
                    for sub in sorted(value.__dataclass_fields__):
                        h.update(np.asarray(
                            getattr(value, sub), dtype=object).astype(
                                "U64").tobytes())
                else:
                    h.update(np.asarray(value, dtype=object).astype(
                        "U64").tobytes())
        return h.hexdigest()

    state, cfg, driver = _build(nx=6, ny=4)
    step(state, cfg)
    assert driver.noahmp_params._handle_cache, "no handle was cached"
    before = digest(driver.noahmp_params)
    for _ in range(20):
        step(state, cfg)
    assert digest(driver.noahmp_params) == before


@requires_gpu
def test_a_glacier_column_is_refused_rather_than_run_as_vegetation():
    """The post-static guard names the first active land glacier at init."""
    from gpuwm.core.noahmp_runtime import NoahmpGlacierColumnError

    vegetation = np.full((4, 6), _GRASSLAND)
    vegetation[0, 0] = _ICE
    vegetation[1, 2] = _ICE
    xice = np.zeros((4, 6))
    xice[0, 0] = 1.0  # sea ice is not an active Noah-MP land column
    with pytest.raises(
            NoahmpGlacierColumnError,
            match=r"first offending active land cell is \(j=1, i=2\).*"
                  r"VEGTYP=15=ISICE_TABLE"):
        _build(nx=6, ny=4, vegtyp=vegetation, xice=xice)


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------

@requires_gpu
def test_a_noahmp_restart_round_trips_and_reproduces_the_next_step(tmp_path):
    from gpuwm.core.dycore import step
    from gpuwm.io.restart import restore_restart, write_restart

    state, cfg, driver = _build(nx=6, ny=4)
    for _ in range(6):
        step(state, cfg)

    path = tmp_path / "noahmp.npz"
    write_restart(path, state, cfg)
    fresh_state, fresh_cfg, fresh_driver = _build(nx=6, ny=4)
    restore_restart(path, fresh_state, fresh_cfg)

    for name in _CARRIED:
        np.testing.assert_array_equal(
            cp.asnumpy(fresh_driver.fields[name]),
            cp.asnumpy(driver.fields[name]), err_msg=name)

    # Storage identity is not trajectory identity.  Step both and compare.
    step(state, cfg)
    step(fresh_state, fresh_cfg)
    for name in _CARRIED:
        np.testing.assert_array_equal(
            cp.asnumpy(fresh_driver.fields[name]),
            cp.asnumpy(driver.fields[name]), err_msg=f"after/{name}")
    for name in ("u", "v", "w", "thp"):
        np.testing.assert_array_equal(
            cp.asnumpy(getattr(fresh_state, name)),
            cp.asnumpy(getattr(state, name)), err_msg=f"after/{name}")


_WATCHED = ("tgxy", "tvxy", "tahxy", "eahxy", "chxy", "cmxy", "canliqxy",
            "fwetxy", "alboldxy", "taussxy", "tslb", "smois", "sh2o",
            "tsk", "hfx", "qfx", "lh", "grdflx", "albedo", "emiss")


def _next_step_delta(tmp_path, field, index, *, ulps=0, add=None, mul=None):
    """Perturb one carrier after a restore and report which fields moved."""
    from gpuwm.core.dycore import step
    from gpuwm.io.restart import restore_restart, write_restart

    state, cfg, driver = _build(nx=6, ny=4)
    for _ in range(6):
        step(state, cfg)
    path = tmp_path / "noahmp.npz"
    write_restart(path, state, cfg)
    step(state, cfg)
    reference = {name: cp.asnumpy(driver.fields[name]).copy()
                 for name in _WATCHED}

    fresh_state, fresh_cfg, fresh_driver = _build(nx=6, ny=4)
    restore_restart(path, fresh_state, fresh_cfg)
    nudged = cp.asnumpy(fresh_driver.fields[field])
    if ulps:
        nudged.view(np.uint32)[index] += np.uint32(ulps)
    elif add is not None:
        nudged[index] = np.float32(nudged[index] + add)
    else:
        nudged[index] = np.float32(nudged[index] * mul)
    fresh_driver.fields[field][...] = cp.asarray(nudged)
    step(fresh_state, fresh_cfg)
    return sorted(
        name for name in _WATCHED
        if not np.array_equal(cp.asnumpy(fresh_driver.fields[name]),
                              reference[name]))


@requires_gpu
def test_the_restart_identity_check_can_fail(tmp_path):
    """A test that cannot fail is not evidence.

    One ULP is added to the top soil temperature of one column after the
    restore.  If the round trip above were vacuous -- if the carriers did not
    actually feed the next step -- this would still match.

    TSLB is the perturbation target because it was chosen by measurement, not
    by guesswork.  The first version of this test nudged TGXY and watched
    TGXY/TVXY/HFX/TSLB/TSK, and passed nothing: a one-ULP TGXY nudge does
    propagate, but on this configuration it reaches only LH and QFX in one
    step, because ENERGY's ground temperature is re-solved from STC(1) and the
    entry TG is largely an iteration seed.  A one-ULP TSLB nudge must reach at
    least five of the twenty watched fields, including the downstream sensible
    heat flux.  The exact membership is not pinned: shared surface forcing can
    move an intermediate across a rounding boundary without making this
    falsification control any weaker.
    """
    moved = _next_step_delta(tmp_path, "tslb", (0, 0, 0), ulps=1)
    assert len(moved) >= 5, (
        "a one-ULP perturbation of TSLB moved only "
        f"{moved}; the round-trip comparison proves little")
    assert "hfx" in moved, (
        "the TSLB perturbation did not reach a downstream flux: "
        f"{moved}")


@requires_gpu
@pytest.mark.parametrize("field,index,kwargs", [
    ("tahxy", (0, 0), {"add": 5.0}),
    ("chxy", (0, 0), {"mul": 1.5}),
    ("cmxy", (0, 0), {"mul": 1.5}),
    ("fwetxy", (0, 0), {"add": 0.3}),
])
def test_four_registry_carriers_do_not_reach_the_next_step(
        tmp_path, field, index, kwargs):
    """Measured: four WRF restart carriers whose entry value is inert here.

    This is a measurement pinned so it cannot drift silently, not a
    requirement.  A +5 K canopy-air temperature, a 50%-larger heat or momentum
    exchange coefficient, and a +0.3 wet-canopy fraction all produce a bit
    identical next step on this configuration.  The reasons are in the source:
    ``FWET`` is unconditionally reassigned by PRECIP_HEAT before ENERGY sees
    it, and ``TAH``/``CH``/``CM`` are re-derived inside VEGE_FLUX and SFCDIF1
    by iterations whose entry values are initial guesses.

    Why it matters: a restart test that perturbed only these four would prove
    nothing, and a reader who assumed "it is Registry restart state, so it
    must matter" would draw the wrong conclusion about what a checkpoint has
    to preserve exactly.  If this test starts failing, one of those two
    routines changed and the restart evidence needs re-reading -- it does not
    mean a bug has appeared.
    """
    moved = _next_step_delta(tmp_path, field, index, **kwargs)
    assert moved == [], (
        f"{field} now reaches the next step ({moved}); re-read the restart "
        "evidence rather than assuming this is a regression")


# ---------------------------------------------------------------------------
# output
# ---------------------------------------------------------------------------

@requires_gpu
def test_a_real_wrfout_carries_the_noahmp_state_and_reads_back(tmp_path):
    """Write the file with the production writer, then read it with netCDF4.

    ``state_frame`` returning the names is not the claim; the claim is that a
    file exists on disk whose Noah-MP snow stack sits on its own vertical axis
    and whose values survive the round trip.
    """
    import netCDF4

    from gpuwm.core.dycore import step
    from gpuwm.io.wrfout import WrfoutWriter, state_frame

    state, cfg, driver = _build(nx=6, ny=4, snow_mm=40.0, snow_depth_m=0.20)
    for _ in range(3):
        step(state, cfg)
    frame = state_frame(state)
    # WRF's external names, which are not the Registry symbols: the wrfout
    # spells ``tvxy`` as ``TV`` and ``isnowxy`` as ``ISNOW``, because that
    # is what the Registry's dname column says and what stock WRF writes.
    # gpuwm/io/wrf_output_schema.py is where each one is transcribed.
    for name in ("TV", "TG", "ISNOW", "ZSNSO", "SNICE", "SNLIQ",
                 "TSNO", "TAUSS", "FSA", "ETRAN", "SOILENERGY"):
        assert name in frame, name

    path = tmp_path / "wrfout_d01_2026-07-01_18_00_36"
    writer = WrfoutWriter(path, nx=cfg.nx, ny=cfg.ny, nz=cfg.nz,
                          dx=cfg.dx, dy=cfg.dy, soil_layers=4)
    # ISNOW and PGS are int32 and stay int32: coercing the frame to float32
    # here is what the writer used to do to them on disk.
    host = {name: np.asarray(value) for name, value in frame.items()}
    writer.write_frame("2026-07-01_18:00:36", host)
    writer.close()

    with netCDF4.Dataset(path) as ds:
        assert len(ds.dimensions["snow_layers_stag"]) == NSNOW
        assert len(ds.dimensions["snso_layers_stag"]) == NSNOW + 4
        assert ds.variables["SNICE"].dimensions == (
            "Time", "snow_layers_stag", "south_north", "west_east")
        assert ds.variables["ZSNSO"].dimensions == (
            "Time", "snso_layers_stag", "south_north", "west_east")
        assert ds.variables["ISNOW"].dtype == np.int32
        for name in ("TV", "TG", "ZSNSO", "SNICE", "FSA", "ISNOW"):
            stored = np.asarray(ds.variables[name][0])
            np.testing.assert_array_equal(stored, host[name], err_msg=name)


# ---------------------------------------------------------------------------
# what running it cost
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_column_cost_is_what_the_registry_says():
    """Measure the per-column cost, and fail if it drifts by an order.

    The registry warning quotes measured cost figures (per 360,000-column
    slab call since 2026-07-27; per column in the host era).  A quoted
    figure nobody measures is a figure that becomes false, so this pins the
    order of magnitude at a width every box can afford.  The bracket stays
    0.3 ms to 30 ms per land column deliberately: it has to survive a
    slower box and a faster one, and its job is to catch a 100x
    regression, not to be a benchmark.

    With all seven leaves batched -- THERMOPROP, RADIATION, VEGE_FLUX,
    BARE_FLUX, TSNOSOI, PHASECHANGE and WATER -- this 48-land-column grid
    measures about 0.9 ms/column on the reference RTX 5090 and 0.54-0.58
    ms/column at 352, against 3.0 and 2.65 ms with the same leaves forced on
    the host.  Absolute milliseconds are a property of the machine -- the
    figures in the first half of ``docs/noahmp_device_column_report.md`` came
    off a different box and are three to six times larger -- so the honest
    always-on gate is an order-of-magnitude bracket, and the paired
    measurement lives in ``tests/test_noahmp_device_column_cost.py`` behind
    ``GPUWM_NOAHMP_WIDTH_SWEEP=1``.
    """
    import time

    from gpuwm.core.dycore import step

    state, cfg, driver = _build(nx=8, ny=6)
    step(state, cfg)                       # warm the parameter-transfer cache
    columns = driver.last_noahmp_census["land"]
    cp.cuda.runtime.deviceSynchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        step(state, cfg)
    cp.cuda.runtime.deviceSynchronize()
    per_column = (time.perf_counter() - t0) / (3 * columns)
    assert 3.0e-4 < per_column < 3.0e-2, (
        f"{per_column * 1e3:.2f} ms per land column is outside the published "
        "bracket; update the registry warning and this gate together")


# ---------------------------------------------------------------------------
# the batched driver: marshalling and write-back
# ---------------------------------------------------------------------------
#
# ``_write_back_batch`` writes about sixty carriers for a whole staged batch
# with one advanced-index assignment each, where the driver used to write them
# one column at a time.  The six-step device/host digest above cannot see a
# defect in it, because both modes run the same write-back; these two can.
#
# The wider field set matters here.  ``_CARRIED`` is the 45 arrays a forecast
# carries forward, and the write-back also fills 26 diagnostics -- RS, the two
# column-energy integrals, Q2MV/Q2MB and PONDING among them -- which are
# exactly the branchy parts of :1223-1400 and are in none of the digests above.

_ALL_WRITTEN = tuple(sorted(set(_CARRIED) | set(NOAHMP_DIAGNOSTICS_2D)
                            | {"smstav", "smstot"}))


def _all_written_digest(driver):
    digest = hashlib.sha256()
    for name in _ALL_WRITTEN:
        array = np.ascontiguousarray(cp.asnumpy(driver.fields[name]))
        digest.update(name.encode())
        digest.update(str(array.dtype).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _six_steps(monkeypatch, *, column_batch=None, staged=False,
               slab_chunk=None, **build_kwargs):
    from gpuwm.core.dycore import step
    import gpuwm.core.noahmp_runtime as runtime_module

    if column_batch is not None:
        monkeypatch.setattr(runtime_module, "COLUMN_BATCH", column_batch)
    if slab_chunk is not None:
        monkeypatch.setattr(runtime_module, "SLAB_COLUMN_CHUNK", slab_chunk)
    if staged:
        # The per-column staging is no longer what a forecast runs; the two
        # gates below that exercise _write_back_batch have to say so by name
        # or they exercise nothing.
        monkeypatch.setenv(runtime_module.STAGED_COLUMNS_ENV, "1")
    else:
        monkeypatch.delenv(runtime_module.STAGED_COLUMNS_ENV, raising=False)
    state, cfg, driver = _build(**build_kwargs)
    for _ in range(6):
        step(state, cfg)
    return driver


@requires_gpu
@pytest.mark.parametrize("column_batch", [1, 7, 2048])
def test_the_batched_write_back_does_not_depend_on_the_tiling(column_batch,
                                                              monkeypatch):
    """Any tiling of the column loop must produce the same slab.

    ``COLUMN_BATCH`` decides how many columns are staged before their leaves
    are batched and their results written back, so it decides the shape of
    every advanced-index assignment in ``_write_back_batch``.  Noah-MP has no
    horizontal coupling, so the answer cannot depend on it -- and at
    ``COLUMN_BATCH = 1`` the batched write-back degenerates to one column at a
    time, which is the shape the per-column version had.  A defect in the
    index arrays, the transposes or the masked stores shows up as a tiling
    dependence and nowhere else.

    All 71 written fields are compared, not the 45 carried ones: RS and the
    two column-energy integrals are diagnostics and are where the branches
    are.
    """
    assert len(_ALL_WRITTEN) >= 70, len(_ALL_WRITTEN)
    whole = _six_steps(monkeypatch, column_batch=2048, staged=True,
                       nx=6, ny=4, water_columns=1,
                       snow_mm=45.0, snow_depth_m=0.16)
    tiled = _six_steps(monkeypatch, column_batch=column_batch, staged=True,
                       nx=6, ny=4, water_columns=1,
                       snow_mm=45.0, snow_depth_m=0.16)
    for name in _ALL_WRITTEN:
        np.testing.assert_array_equal(
            cp.asnumpy(tiled.fields[name]), cp.asnumpy(whole.fields[name]),
            err_msg=f"COLUMN_BATCH={column_batch} moved {name}")
    assert _all_written_digest(tiled) == _all_written_digest(whole)


@requires_gpu
def test_the_write_back_comparison_can_see_a_misrouted_column(monkeypatch):
    """Give each column the next column's answer; the slab must move.

    This is the defect a batched write-back produces: the index arrays and the
    value arrays fall out of step by one, every field is the right shape, and
    every column carries its neighbour's surface.  Until that is observed
    being rejected, the tiling gate above is a comparison that has never been
    seen to fail.
    """
    import dataclasses

    import gpuwm.core.noahmp_runtime as runtime_module

    honest = _all_written_digest(
        _six_steps(monkeypatch, staged=True, nx=6, ny=4, water_columns=1))

    real = runtime_module._write_back_batch

    def rolled(host, host_3d, staged, **kwargs):
        shifted = [dataclasses.replace(column, result=staged[
            (index + 1) % len(staged)].result)
            for index, column in enumerate(staged)]
        return real(host, host_3d, shifted, **kwargs)

    monkeypatch.setattr(runtime_module, "_write_back_batch", rolled)
    moved = _all_written_digest(
        _six_steps(monkeypatch, staged=True, nx=6, ny=4, water_columns=1))
    assert moved != honest, (
        "rolling every column's result onto its neighbour left the whole "
        "written slab unchanged; the write-back gate cannot see a misroute")


# ---------------------------------------------------------------------------
# the slab orchestration path, held to the same bars as the staged one
# ---------------------------------------------------------------------------

@requires_gpu
def test_the_slab_and_staged_paths_write_the_same_slab(monkeypatch):
    """The orchestration against the per-column seam, at the forecast level.

    Same device leaves on both sides -- the staged arm keeps the default
    device evaluator -- so the only thing that differs is the orchestration:
    whole-slab CuPy against one generator frame per column.  All 71 written
    fields, bitwise, over six steps on a snow-covered domain, which is wider
    than the 45-array carried digest precisely where the write-back branches
    are.
    """
    slab = _six_steps(monkeypatch, nx=6, ny=4, water_columns=1,
                      snow_mm=45.0, snow_depth_m=0.16)
    staged = _six_steps(monkeypatch, staged=True, nx=6, ny=4, water_columns=1,
                        snow_mm=45.0, snow_depth_m=0.16)
    for name in _ALL_WRITTEN:
        np.testing.assert_array_equal(
            cp.asnumpy(slab.fields[name]), cp.asnumpy(staged.fields[name]),
            err_msg=f"slab/staged {name}")
    assert _all_written_digest(slab) == _all_written_digest(staged)


@requires_gpu
@pytest.mark.parametrize("slab_chunk", [1, 7, 65536])
def test_the_slab_chunk_tiling_is_inert(slab_chunk, monkeypatch):
    """SLAB_COLUMN_CHUNK is a memory knob and must not move the forecast.

    The successor of the COLUMN_BATCH gate above, for the path a forecast
    actually takes: the chunk bound decides every gather, launch width and
    scatter in _lsm_step_slab, and Noah-MP has no horizontal coupling, so a
    chunk dependence would be an indexing defect and nothing else.  At
    chunk=1 the whole orchestration degenerates to one column per launch.
    """
    whole = _six_steps(monkeypatch, slab_chunk=65536,
                       nx=6, ny=4, water_columns=1,
                       snow_mm=45.0, snow_depth_m=0.16)
    tiled = _six_steps(monkeypatch, slab_chunk=slab_chunk,
                       nx=6, ny=4, water_columns=1,
                       snow_mm=45.0, snow_depth_m=0.16)
    for name in _ALL_WRITTEN:
        np.testing.assert_array_equal(
            cp.asnumpy(tiled.fields[name]), cp.asnumpy(whole.fields[name]),
            err_msg=f"SLAB_COLUMN_CHUNK={slab_chunk} moved {name}")
    assert _all_written_digest(tiled) == _all_written_digest(whole)


@requires_gpu
def test_the_slab_write_back_comparison_can_see_a_misrouted_column(
        monkeypatch):
    """Give each column the next column's slab answer; the slab must move.

    The same defect the staged control above produces, in the shape the
    orchestration would actually produce it: every output array rolled one
    lane, every field the right shape, every column carrying its neighbour's
    surface.  Until this is observed being rejected, the chunk gate above is
    a comparison that has never been seen to fail.
    """
    import gpuwm.core.noahmp_runtime as runtime_module

    honest = _all_written_digest(
        _six_steps(monkeypatch, nx=6, ny=4, water_columns=1))

    real = runtime_module._write_back_slab

    def rolled(fields, j, i, r, **kwargs):
        shifted = {name: cp.roll(value, 1, axis=0)
                   for name, value in r.items()}
        return real(fields, j, i, shifted, **kwargs)

    monkeypatch.setattr(runtime_module, "_write_back_slab", rolled)
    moved = _all_written_digest(
        _six_steps(monkeypatch, nx=6, ny=4, water_columns=1))
    assert moved != honest, (
        "rolling every column's slab answer onto its neighbour left the "
        "whole written slab unchanged; the slab write-back gate cannot see "
        "a misroute")
