"""Device integration proofs; run after the real mosaic module is merged."""
from dataclasses import replace

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = pytest.mark.gpu

#: Rounding-sized distances between the two Noah kernels on a one-category
#: column after three steps (see the comment where they are used).
PURE_TSK_BOUND_K = 1.0e-3
PURE_SOIL_BOUND = 1.0e-5


def _domain(cp, *, mosaic):
    # This existing builder uses the real initialize_physics and mixed water.
    from test_restart import _physics_state
    state, cfg, driver = _physics_state(cp, sf_surface_mosaic=int(mosaic), mosaic_cat=3,
                                        cu_physics=0, ra_physics=0)
    if mosaic:
        from gpuwm.core.noah_mosaic_door import attach_noah_mosaic_to_driver
        luf = np.zeros((21, cfg.ny, cfg.nx), np.float32)
        luf[9, :, :-1] = 1.0
        luf[16, :, -1] = 1.0
        luf[9, :, 1] = 0.6
        luf[6, :, 1] = 0.4
        # A landless land column (no fraction at all): WRF divides 0 by 0
        # in its area averages there; the port runs the cell's own category.
        luf[:, :, 2] = 0.0
        attach_noah_mosaic_to_driver(
            driver, cfg, landusef=luf, processed=True,
            landuse_attrs={"MMINLU": "MODIFIED_IGBP_MODIS_NOAH",
                           "ISURBAN": 13, "ISWATER": 17, "ISICE": 15},
            fractional_seaice=False)
    return state, cfg, driver


@requires_gpu
def test_three_compute_steps_water_exact_mixed_land_changes():
    import cupy as cp
    off_state, off_cfg, off = _domain(cp, mosaic=False)
    state, cfg, driver = _domain(cp, mosaic=True)
    for step in range(3):
        off.compute(off_state, off_cfg)
        driver.compute(state, cfg)
        off_state.elapsed_seconds += off_cfg.dt
        state.elapsed_seconds += cfg.dt
    # Open water bypasses the land-tile SFLX loop in both drivers.
    for name in ("tsk", "hfx", "qfx", "tslb", "smois", "sh2o"):
        assert off.fields[name][..., -1].get().tobytes() == driver.fields[name][..., -1].get().tobytes(), name
    # A single-category land column runs the same Noah physics through two
    # different kernels: noah.cu (FMA-contracted, CUDA libm, not bitwise
    # WRF) and noah_mosaic.cu (bitwise WRF lsm_mosaic, graded by
    # tests/test_noah_mosaic_wrf471_parity.py), and the mosaic grid TSK is
    # the emissivity-weighted fourth root of its tiles.  So the statement
    # here is the physical one -- the same surface to a rounding-sized
    # distance -- and the exact one lives in the oracle gate.
    pure_columns = [column for column in range(cfg.nx - 1) if column != 1]
    # The landless column is one of them: it is finite and is its category.
    for name in ("tsk", "znt", "hfx", "qfx", "tslb"):
        assert np.all(np.isfinite(driver.fields[name][..., 2].get())), name
    pure_mosaic = driver.fields["tsk"][:, pure_columns].get()
    pure_off = off.fields["tsk"][:, pure_columns].get()
    assert np.max(np.abs(pure_mosaic - pure_off)) < PURE_TSK_BOUND_K
    for name in ("tslb", "smois", "sh2o"):
        a = driver.fields[name][..., pure_columns].get()
        b = off.fields[name][..., pure_columns].get()
        assert np.max(np.abs(a - b)) < PURE_SOIL_BOUND, name
    # The 60/40 cell is two tiles, and its surface is not the dominant one.
    mixed = np.abs(driver.fields["tsk"][:, 1].get() - off.fields["tsk"][:, 1].get())
    assert np.all(mixed > 0.0)


@requires_gpu
def test_tile_checkpoint_roundtrip_and_setting_mismatch(tmp_path):
    import cupy as cp
    from gpuwm.io.restart import write_restart, restore_restart, read_restart_header, RestartMismatchError
    from gpuwm.core.noah_mosaic import mosaic_array_shapes
    state, cfg, driver = _domain(cp, mosaic=True)
    for _ in range(3):
        driver.compute(state, cfg)
        state.elapsed_seconds += cfg.dt
    path = write_restart(tmp_path / "tiles.npz", state, cfg)
    fresh, fresh_cfg, fresh_driver = _domain(cp, mosaic=True)
    restore_restart(path, fresh, fresh_cfg)
    for name in mosaic_array_shapes(cfg.mosaic_cat, cfg.ny, cfg.nx):
        assert "fields/" + name in read_restart_header(path)["array_manifest"]
        assert fresh_driver.fields[name].get().tobytes() == driver.fields[name].get().tobytes(), name
    with pytest.raises(RestartMismatchError, match="mosaic_cat"):
        restore_restart(path, fresh, replace(fresh_cfg, mosaic_cat=2))


@requires_gpu
def test_missing_tile_state_refuses_at_noah_dispatch():
    import cupy as cp
    state, cfg, driver = _domain(cp, mosaic=False)
    cfg = replace(cfg, sf_surface_mosaic=1)
    with pytest.raises(ValueError, match="carries no LANDUSEF.*dominant category"):
        driver._run_noah({}, cfg, 1)


@requires_gpu
def test_relocation_refuses_old_tile_footprint():
    import cupy as cp
    state, cfg, driver = _domain(cp, mosaic=True)
    with pytest.raises(ValueError, match="old footprint's tiles"):
        driver.recouple_after_relocation(state, cfg)


def _ucm_domain(cp):
    from datetime import datetime
    from test_restart import _physics_state
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.noah_mosaic_door import attach_noah_mosaic_to_driver
    state, base_cfg, _ = _physics_state(cp, cu_physics=0, ra_physics=0)
    cfg = replace(base_cfg, sf_surface_mosaic=1, sf_urban_physics=1)
    shape = (cfg.ny, cfg.nx)
    land = np.ones(shape, np.float32); land[:, -1] = 0
    cat = np.where(land, 10, 17).astype(np.int32); cat[:, 0] = 13
    tsk = np.full(shape, 299., np.float32)
    soil = np.full((4, *shape), .31, np.float32)
    driver = initialize_physics(state, cfg, landmask=land, tsk=tsk,
        soil_temperature=np.full((4, *shape), 296., np.float32),
        soil_moisture=soil, liquid_moisture=soil, ivgtyp=cat,
        isltyp=np.where(land, 6, 14), vegfra=55., tmn=286.,
        swdown=450., glw=310., pblh=700.,
        radiation_start_time=datetime(2000, 6, 1, 12),
        radiation_latitude=np.full(shape, 45., np.float32),
        radiation_longitude=np.full(shape, 10., np.float32))
    fractions = np.zeros((21, *shape), np.float32)
    fractions[9] = land; fractions[16, :, -1] = 1
    fractions[9, :, 0] = .4; fractions[12, :, 0] = .6
    fractions[9, :, 1] = .6; fractions[12, :, 1] = .4
    driver.fields["frc_urb2d"][:, :2] = cp.float32(.5)
    attach_noah_mosaic_to_driver(driver, cfg, landusef=fractions, processed=True,
        landuse_attrs={"MMINLU":"MODIFIED_IGBP_MODIS_NOAH", "ISURBAN":13,
                       "ISWATER":17,"ISICE":15}, fractional_seaice=False)
    return state, cfg, driver


@requires_gpu
def test_ucm_three_compute_steps_single_call_and_tile_checkpoint(tmp_path, monkeypatch):
    import cupy as cp
    from gpuwm.core import urban_ucm
    from gpuwm.core.noah_mosaic import MOSAIC_URBAN_TILE_FIELDS, MOSAIC_URBAN_SOIL_FIELDS
    from gpuwm.io.restart import write_restart, restore_restart, read_restart_header
    state, cfg, driver = _ucm_domain(cp)
    initial = driver.fields["tr_urb2d_mosaic"].copy()
    calls = []
    overrides = urban_ucm.after_surface_diagnostics
    def record_override(*args, **kwargs):
        calls.append("override")
        return overrides(*args, **kwargs)
    def forbidden_grid_blend(*args, **kwargs):
        raise AssertionError("plain grid UCM launched after the mosaic tile UCM")
    monkeypatch.setattr(urban_ucm, "after_lsm", forbidden_grid_blend)
    monkeypatch.setattr(urban_ucm, "after_surface_diagnostics", record_override)
    for _ in range(3):
        driver.compute(state, cfg)
        state.elapsed_seconds += cfg.dt
    # Exercise the public hook too: an incidental caller must also skip it.
    driver.urban_coupler.after_lsm(driver.fields, {}, cfg, dt=cfg.dt, itimestep=3)
    assert calls == ["override"] * 3
    assert not np.array_equal(initial.get().view(np.uint32),
                              driver.fields["tr_urb2d_mosaic"].get().view(np.uint32))
    path = write_restart(tmp_path / "urban-tiles.npz", state, cfg)
    fresh, fresh_cfg, fresh_driver = _ucm_domain(cp)
    restore_restart(path, fresh, fresh_cfg)
    manifest = read_restart_header(path)["array_manifest"]
    for name in (*MOSAIC_URBAN_TILE_FIELDS, *MOSAIC_URBAN_SOIL_FIELDS):
        assert "fields/" + name in manifest
        assert fresh_driver.fields[name].get().tobytes() == driver.fields[name].get().tobytes(), name


def _rule_domain(cp, rule):
    """_ucm_domain's grid with the urban fraction left to the rule: column 0
    is mostly town (dominant ISURBAN, 60%), column 1 is 40% town in a
    mostly grassland cell, the last column is sea."""
    from datetime import datetime
    from test_restart import _physics_state
    from gpuwm.core.physics import initialize_physics
    from gpuwm.core.noah_mosaic_door import attach_noah_mosaic_to_driver
    state, base_cfg, _ = _physics_state(cp, cu_physics=0, ra_physics=0)
    cfg = replace(base_cfg, sf_surface_mosaic=1, sf_urban_physics=1,
                  mosaic_urban_canopy=rule)
    shape = (cfg.ny, cfg.nx)
    land = np.ones(shape, np.float32); land[:, -1] = 0
    cat = np.where(land, 10, 17).astype(np.int32); cat[:, 0] = 13
    soil = np.full((4, *shape), .31, np.float32)
    driver = initialize_physics(state, cfg, landmask=land,
        tsk=np.full(shape, 299., np.float32),
        soil_temperature=np.full((4, *shape), 296., np.float32),
        soil_moisture=soil, liquid_moisture=soil, ivgtyp=cat,
        isltyp=np.where(land, 6, 14), vegfra=55., tmn=286.,
        swdown=450., glw=310., pblh=700.,
        radiation_start_time=datetime(2000, 6, 1, 12),
        radiation_latitude=np.full(shape, 45., np.float32),
        radiation_longitude=np.full(shape, 10., np.float32))
    fractions = np.zeros((21, *shape), np.float32)
    fractions[9] = land; fractions[16, :, -1] = 1
    fractions[9, :, 0] = .4; fractions[12, :, 0] = .6
    fractions[9, :, 1] = .6; fractions[12, :, 1] = .4
    attach_noah_mosaic_to_driver(driver, cfg, landusef=fractions, processed=True,
        landuse_attrs={"MMINLU": "MODIFIED_IGBP_MODIS_NOAH", "ISURBAN": 13,
                       "ISWATER": 17, "ISICE": 15}, fractional_seaice=False)
    return state, cfg, driver


@requires_gpu
def test_the_town_rule_moves_the_town_column_and_nothing_else():
    """mosaic_urban_canopy: WRF's rule leaves the 40% town tile of column 1
    at fraction 0 (it runs as vegetation); the town rule gives it URBPARM's
    0.9 for its type.  Columns are independent in the surface step, so after
    three driver steps every array is byte-identical between the two rules
    everywhere but column 1, and column 1's surface has moved."""
    import cupy as cp
    wrf_state, wrf_cfg, wrf = _rule_domain(cp, "dominant")
    town_state, town_cfg, town = _rule_domain(cp, "every_tile")
    frc_wrf = wrf.fields["frc_urb2d"].get()
    frc_town = town.fields["frc_urb2d"].get()
    table = np.float32(0.9)  # URBPARM.TBL FRC_URB, type 2 (ISURBAN)
    assert np.all(frc_wrf[:, 0] == table) and np.all(frc_town[:, 0] == table)
    assert np.all(frc_wrf[:, 1] == 0.0) and np.all(frc_town[:, 1] == table)
    assert np.all(frc_wrf[:, 2:] == 0.0) and np.all(frc_town[:, 2:] == 0.0)
    # The type map (and with it WRF's 10 m wind override) stays WRF's.
    assert np.array_equal(wrf.fields["utype_urb2d"].get(), town.fields["utype_urb2d"].get())
    for _ in range(3):
        for state, cfg, driver in ((wrf_state, wrf_cfg, wrf), (town_state, town_cfg, town)):
            driver.compute(state, cfg)
            state.elapsed_seconds += cfg.dt
    others = [c for c in range(wrf_cfg.nx) if c != 1]
    compared = 0
    for name, value in wrf.fields.items():
        other = town.fields.get(name)
        if not hasattr(value, "get") or other is None or value.ndim < 2 \
                or value.shape[-2:] != (wrf_cfg.ny, wrf_cfg.nx):
            continue
        a, b = value.get(), other.get()
        if name == "frc_urb2d":
            continue
        assert a[..., others].tobytes() == b[..., others].tobytes(), name
        compared += 1
    assert compared > 50
    for name in ("tsk", "hfx", "ts_urb2d_mosaic"):
        moved = wrf.fields[name][..., 1].get() != town.fields[name][..., 1].get()
        assert np.any(moved), name


@requires_gpu
def test_the_town_rule_rides_the_checkpoint_and_a_rule_change_is_refused(tmp_path):
    """A town-rule checkpoint restores its fractions and tiles word for word,
    and resuming it under WRF's rule (or the reverse) stops naming the key."""
    import cupy as cp
    from gpuwm.io.restart import (RestartMismatchError, read_restart_header,
                                  restore_restart, write_restart)
    state, cfg, driver = _rule_domain(cp, "every_tile")
    for _ in range(2):
        driver.compute(state, cfg)
        state.elapsed_seconds += cfg.dt
    path = write_restart(tmp_path / "town.npz", state, cfg)
    assert read_restart_header(path)["config"]["mosaic_urban_canopy"] == "every_tile"
    fresh, fresh_cfg, fresh_driver = _rule_domain(cp, "every_tile")
    restore_restart(path, fresh, fresh_cfg)
    for name in ("frc_urb2d", "ts_urb2d_mosaic", "tr_urb2d_mosaic", "tsk", "landusef2"):
        assert fresh_driver.fields[name].get().tobytes() == driver.fields[name].get().tobytes(), name
    wrf_state, wrf_cfg, _ = _rule_domain(cp, "dominant")
    with pytest.raises(RestartMismatchError, match="mosaic_urban_canopy"):
        restore_restart(path, wrf_state, wrf_cfg)
    wrf_path = write_restart(tmp_path / "wrf.npz", wrf_state, wrf_cfg)
    assert "mosaic_urban_canopy" not in read_restart_header(wrf_path)["config"]
    town_state, town_cfg, _ = _rule_domain(cp, "every_tile")
    with pytest.raises(RestartMismatchError, match="mosaic_urban_canopy"):
        restore_restart(wrf_path, town_state, town_cfg)


@requires_gpu
@pytest.mark.parametrize("unit,builder,function", [
    ("noah_mosaic_unit", "_mosaic_module", "noah_mosaic_column"),
    ("noah_mosaic_ucm_unit", "_mosaic_ucm_module", "noah_mosaic_ucm_column"),
])
def test_the_mosaic_units_compile_to_the_frames_they_are_priced_at(
        unit, builder, function):
    """The platform check no ``.cu`` enumeration reaches for these units.

    Both launch only as gpuwm/core/noah_mosaic.py's own ``--fmad=false``
    compositions, so they are priced from
    ``preflight.CHAINED_TRANSLATION_UNIT_FRAMES`` and
    ``under_priced_kernel_frames`` cannot see them drift.  THE BREAKAGE
    THIS PREVENTS: the rows were priced from NVRTC 13.4.92 readings (240 B
    and 400 B), and the default install's NVRTC 12.9.86 compiles the
    columns to 688 B and 1,040 B on sm_120, so every mosaic run on such a
    card was admitted with its local-memory reservation under-charged.
    Each unit is compiled through the loader that launches it and its one
    kernel's frame is held to the row on whatever card runs this.
    """
    import cupy as cp
    from gpuwm.core import noah_mosaic as NM
    from gpuwm.core import preflight as pf

    row = pf.CHAINED_TRANSLATION_UNIT_FRAMES[unit]
    module = getattr(NM, builder)()
    frame = int(module.get_function(function).local_size_bytes)
    profile = pf.local_memory_profile_from_device(cp)
    unpriced = (frame - row.max_local_size_bytes) * profile.resident_thread_capacity
    assert frame <= row.max_local_size_bytes, (
        f"{unit} ({function}) compiles to {frame} B per thread on this "
        f"platform ({profile.name}, NVRTC {cp.cuda.nvrtc.getVersion()}) "
        f"against the {row.max_local_size_bytes} B gpuwm/core/preflight.py "
        "CHAINED_TRANSLATION_UNIT_FRAMES prices it at, so every mosaic run "
        f"under-charges the local-memory reservation by up to "
        f"{unpriced / 1024 ** 2:.1f} MiB.  Remedy: move the row to this "
        "reading and record it in CHAINED_UNITS_WITHOUT_A_PER_PLATFORM_ROW "
        "(gpuwm/core/kernel_frame_recordings.py)")
