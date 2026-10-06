"""Allocation-derived admission guards for rank physics and output pools.

Breakage: pricing a rank as a resident MYNN domain, then pricing its
scratch-backed returned rates again as YSU allocations, refused fitting
multi-card forecasts. Missing RUC or atmosphere allocations has the opposite
failure mode: an admitted forecast can exhaust its device pool.
"""

from dataclasses import replace
from datetime import datetime
from math import prod

import pytest

from gpuwm.config import RunConfig
from gpuwm.core import preflight as pf
from gpuwm.core.devices import DeviceOptions
from gpuwm.core.devices_memory import estimate_devices
from gpuwm.core.mynn_pbl_scratch import mynn_pricing_memory
from gpuwm.experiment import experiment_from_run_config


def _config(**kwargs):
    values = dict(nx=320, ny=240, nz=50, dx=3000.0, dy=3000.0,
                  ztop=20000.0, dt=15.0, run_seconds=120.0,
                  moist=True, mp_physics=28, bl_pbl_physics=5,
                  sf_sfclay_physics=5, sf_surface_physics=3,
                  num_soil_layers=9, ra_physics=0, cu_physics=0)
    return RunConfig(**(values | kwargs))


def test_mynn_returned_rates_are_priced_once():
    """Reverting the selector adds nine nonexistent YSU volume arrays."""
    from gpuwm.core.mynn_pbl_scratch import mynn_pbl_tendency_field_shapes

    cfg = _config()
    assert pf.ysu_output_transient_shapes(cfg) == {}
    owned = mynn_pbl_tendency_field_shapes(cfg.nz, cfg.ny, cfg.nx)
    assert len(owned) == 6
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        registry = pf.scratch_slot_registry(cfg, tile_buffer=True)
    assert {name: registry[name] for name in owned} == owned
    assert pf.ysu_output_transient_shapes(replace(cfg, bl_pbl_physics=1))


def test_physics_atmosphere_prices_the_density_allocation():
    """The density returned by _prepare_atmosphere is not a state alias."""
    cfg = _config()
    assert pf.atmosphere_transient_shapes(cfg)["atmosphere/rho"] == (
        cfg.nz, cfg.ny, cfg.nx)


@pytest.mark.parametrize("tile_buffer", [False, True])
def test_mynn_optional_scalar_lifetimes_are_priced_beside_shared_rates(tile_buffer):
    """Old/new coupled rates overlap; the qn solver also owns plain pool arrays."""
    off = _config(bl_mynn_mixscalars=0)
    on = replace(off, bl_mynn_mixscalars=1)
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        assert pf.mynn_mixscalars_memory_items(off, tile_buffer=tile_buffer) == ()
        assert pf.mynn_mixscalars_memory_items(
            replace(on, bl_pbl_physics=1), tile_buffer=tile_buffer) == ()
        items = pf.mynn_mixscalars_memory_items(on, tile_buffer=tile_buffer)
        chunk = min(on.nx * on.ny,
                    pf.mynn_pbl_column_chunk(on, tile_buffer=tile_buffer))
        mass = on.nz * on.ny * on.nx * 4
        by_category = {category: sum(item.nbytes for item in items
                                    if item.category == category)
                       for category in ("physics", "transient")}
        assert by_category == {"physics": 4 * mass,
                               "transient": 9 * mass + chunk * (55 * on.nz + 40) * 4}
        assert pf.ysu_output_transient_shapes(on) == {}
        estimates = [pf.estimate_experiment(
            experiment_from_run_config(cfg, datetime(2026, 1, 1)),
            vram_gib=32, tile_buffer=tile_buffer).domains[0] for cfg in (off, on)]
    for category, amount in by_category.items():
        assert (estimates[1].category_bytes(category)
                - estimates[0].category_bytes(category)) == amount


def test_rank_residents_use_the_tile_workspace_but_loader_does_not():
    """A rank's _tile_buffer marker must reach the same price policy."""
    cfg = _config()
    exp = replace(experiment_from_run_config(cfg, datetime(2026, 1, 1)),
                  devices=DeviceOptions(count=2))
    with mynn_pricing_memory(total_bytes=96 * pf.GIB, free_bytes=96 * pf.GIB):
        price = estimate_devices(exp, vram_gib=96)
        for rank in price["rank_shapes"]:
            ny, nx = rank["compute_shape"]
            local = replace(exp, devices=DeviceOptions(), domains=(
                replace(exp.root, run=replace(cfg, nx=nx, ny=ny)),))
            tile = pf.estimate_experiment(local, vram_gib=96, tile_buffer=True)
            resident = pf.estimate_experiment(local, vram_gib=96)
            assert rank["resident_bytes"] == tile.peak_envelope_bytes
            assert tile.peak_envelope_bytes < resident.peak_envelope_bytes
            delta = resident.resident_bytes - tile.resident_bytes
            expected = sum(prod(s) * 4 for s in pf.mynn_pbl_scratch_slots(
                local.root.run).values()) - sum(prod(s) * 4 for s in
                pf.mynn_pbl_scratch_slots(local.root.run, tile_buffer=True).values())
            assert delta == expected


def test_optional_aerosol_probe_outputs_are_not_forecast_residents():
    """The six unused outputs must not return to the full-domain pool."""
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        registry = pf.scratch_slot_registry(_config())
    absent = ("nwfa_entry_m3", "nifa_entry_m3", "rc_entry", "nc_entry_m3",
              "nu_c_entry", "l_qc_entry")
    assert all("mp_thompson_aero_" + name not in registry for name in absent)
    assert all("mp_thompson_aero_" + name in registry for name in (
        "ncten", "nwfaten", "nifaten", "entry_density", "tau1_density",
        "nwfa_work_m3", "qc_entry", "ni_entry", "condensation_rate"))


def test_itemized_card_budget_refuses_one_byte_short():
    """Correcting alias prices must preserve the hard allocation refusal."""
    from gpuwm.core.devices_memory import devices_gate

    exp = replace(experiment_from_run_config(_config(), datetime(2026, 1, 1)),
                  devices=DeviceOptions(count=2))
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        price = estimate_devices(exp, vram_gib=32)
    budgets = {row["card"]: row["total_bytes"] for row in price["cards"]}
    assert not devices_gate(price, budgets=budgets)["refuse"]
    budgets[1] -= 1
    gate = devices_gate(price, budgets=budgets)
    assert gate["refuse"]
    assert "card 1: REFUSED" in gate["verdict"]


@pytest.mark.parametrize("levels", [6, 9])
def test_ruc_carried_soil_and_snow_fields_keep_their_full_extent(levels):
    """The nine-level soil and snow state must not hide under a six-level price."""
    from gpuwm.core.ruc_runtime import RUC_STATE_3D, RUC_STATE_2D

    cfg = _config(num_soil_layers=levels)
    shapes = pf.physics_array_shapes(cfg)
    for name in ("smois", "tslb", "sh2o", "smcrel", *RUC_STATE_3D):
        assert shapes["fields/" + name] == (levels, cfg.ny, cfg.nx)
    for name in ("snow", "snowh", "snowc", "snoalb", *RUC_STATE_2D):
        assert shapes["fields/" + name] == (cfg.ny, cfg.nx)


@pytest.mark.parametrize("levels", [6, 9])
def test_ruc_solver_pools_are_added_beside_the_carried_fields(levels):
    """Both cached slabs and live result copies were missing from admission."""
    from gpuwm.core.ruc_memory import ruc_runtime_memory_bytes

    cfg = _config(num_soil_layers=levels)
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1))
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        estimate = pf.estimate_domain(exp.root, tile_buffer=True)
    expected = ruc_runtime_memory_bytes(cfg.nx * cfg.ny, levels)
    rows = {item.name.removeprefix("ruc/"): item for item in estimate.items
            if item.name.startswith("ruc/")}
    assert {name: item.nbytes for name, item in rows.items()} == expected
    assert rows["sfctmp_outputs"].category == "transient"
    assert all(rows[name].category == "physics" for name in (
        "driver_workspace", "sfctmp_workspace", "tables"))


@pytest.mark.parametrize("microphysics", [8, 28])
def test_thompson_tables_match_the_upload_contract_once_per_device(microphysics):
    """Missing constant tables hid 380 MB; multiplying by domains adds it twice."""
    from gpuwm.core.thompson_contract import (
        AUXILIARY_TABLE_RECORDS, GENERATED_TABLE_FILES)

    cfg = _config(mp_physics=microphysics)
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1))
    sizes = [record.payload_bytes for group in GENERATED_TABLE_FILES.values()
             for record in group]
    sizes.extend(record.payload_bytes for record in AUXILIARY_TABLE_RECORDS)
    if microphysics == 28:
        from gpuwm.core.thompson_aerosol_contract import (
            CCN_ACTIVATION_VALUES, derived_constant_arrays)
        sizes += [CCN_ACTIVATION_VALUES * 8]
        sizes += [array.nbytes for array in derived_constant_arrays().values()]
    expected = sum((size + 511) // 512 * 512 for size in sizes)
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        one = pf.estimate_experiment(exp, vram_gib=32)
        two = pf.estimate_experiment(replace(exp, domains=(
            exp.root, replace(exp.root, grid_id=2))), vram_gib=32)
    assert one.physics_tables_bytes == two.physics_tables_bytes == expected
    assert one.resident_bytes == one.domains[0].resident_bytes + expected


def test_legacy_shortwave_constants_survive_a_night_call():
    """A zero-column SW call still owns its constructor's three uploads."""
    from gpuwm.core import rrtmg_legacy as radiation
    from gpuwm.core import rrtmg_sw as sw

    table = radiation._sw_tables()
    packed, _ = sw._pack_cuda_tables(table)
    allocations = (packed.nbytes, table.ngb.size * 4, 3 * sw.NBNDSW * 4)
    expected = sum((size + 511) // 512 * 512 for size in allocations)
    common = dict(ncol=10, nz=50, p_top=5000.0, longwave=False,
                  ncol_day=0, resident_threads=0)
    off = radiation.legacy_radiation_vram_bytes(**common, shortwave=False)
    night = radiation.legacy_radiation_vram_bytes(**common, shortwave=True)
    assert night - off == expected


def test_legacy_longwave_constants_survive_the_shortwave_peak(monkeypatch):
    """Taking max(LW+tables, SW) used to lose retained LW coefficients."""
    from gpuwm.core import rrtmg_legacy as radiation

    # Force the SW phase to dominate so an LW-only charge is observable.
    monkeypatch.setattr(radiation._sw, "sw_batched_vram_bytes",
                        lambda *args, **kwargs: 10**12)
    common = dict(ncol=10, nz=50, p_top=5000.0, column_chunk=10,
                  shortwave=True, resident_threads=0)
    shortwave = radiation.legacy_radiation_vram_bytes(**common, longwave=False)
    both = radiation.legacy_radiation_vram_bytes(**common, longwave=True)
    expected = radiation._lw.lw_batched_const_bytes(radiation._lw_coeffs())
    assert both - shortwave == expected


@pytest.mark.parametrize("microphysics", [8, 28])
@pytest.mark.parametrize("levels", [50, 64, 65, 128])
def test_only_reachable_fallout_tiers_reserve_local_memory(microphysics, levels):
    """Loading a 256-level export must not price it for a 64-level launch."""
    exp = experiment_from_run_config(
        _config(mp_physics=microphysics, nz=levels), datetime(2026, 1, 1))
    frames = pf.kernel_local_frame_bytes(exp)
    assert frames["thompson"] == (2816 if levels <= 64 else 11264)
    if microphysics == 28:
        assert frames["thompson_aerosol_sed"] == (2304 if levels <= 64 else 9216)
    else:
        assert "thompson_aerosol_sed" not in frames


@pytest.mark.parametrize("levels", [50, 64, 65, 128])
def test_the_fork_thompson_generation_is_priced_at_its_own_fallout_frames(levels):
    """thompson_version = "wrf_39_noaa" compiles a wider sedimentation frame.

    Breakage prevented: the HRRR configuration recipes select the fork
    generation; priced at the wrf_461 rows, the sedimentation row under-states
    the fork build by 2,048 B per resident thread, and the reservation inherits
    that error as soon as the classic unit's equal frame stops covering it (the
    gpu gate tests/test_kernel_local_bounds.py reads both rows off the
    driver)."""
    exp = experiment_from_run_config(
        _config(mp_physics=28, nz=levels, thompson_version="wrf_39_noaa"),
        datetime(2026, 1, 1))
    frames = pf.kernel_local_frame_bytes(exp)
    assert frames["thompson_aerosol_sed"] == (2816 if levels <= 64 else 11264)
    assert frames["thompson"] == (2816 if levels <= 64 else 11264)
    default = pf.kernel_local_frame_bytes(experiment_from_run_config(
        _config(mp_physics=28, nz=levels), datetime(2026, 1, 1)))
    assert default["thompson_aerosol_sed"] == (2304 if levels <= 64 else 9216)


def test_deep_other_physics_does_not_raise_a_shallow_fallout_tier():
    """A deep non-Thompson domain cannot select Thompson's deep kernels."""
    exp = experiment_from_run_config(_config(), datetime(2026, 1, 1))
    other = replace(exp.root, grid_id=2,
                    run=replace(exp.root.run, nz=128, mp_physics=10))
    mixed = replace(exp, domains=(exp.root, other))
    frames = pf.kernel_local_frame_bytes(mixed)
    assert frames["thompson"] == 2816
    assert frames["thompson_aerosol_sed"] == 2304
    deep = replace(other, run=replace(other.run, mp_physics=28))
    frames = pf.kernel_local_frame_bytes(replace(exp, domains=(exp.root, deep)))
    assert frames["thompson"] == 11264
    assert frames["thompson_aerosol_sed"] == 9216


def test_streamed_buffers_share_one_microphysics_table_owner():
    """Increasing buffer count must not duplicate immutable per-card tables."""
    from gpuwm.core.prepared_tile_memory import PreparedTileMemory

    cfg = _config(ra_physics=4, ra_rrtmg_variant="rrtmg_legacy",
                  wrf_rrtmg_compatibility="wrf-rrtmg-4-4-legacy-v1")
    exp = experiment_from_run_config(cfg, datetime(2026, 1, 1))
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        memory = PreparedTileMemory(exp, pf.MEASURED_LOCAL_MEMORY_PROFILE)
        shape = (200, 160)
        cells = prod(shape) * cfg.nz
        two, three = memory.terms(cells, 2, shape), memory.terms(cells, 3, shape)
    expected = pf.thompson_coefficient_bytes(aerosol=True)
    assert two["fixed/physics_tables_bytes"] == three["fixed/physics_tables_bytes"] == expected
    assert three["pool_bytes"] - two["pool_bytes"] == memory.buffer_bytes(cells, shape)


@pytest.mark.parametrize("transport", ["host", "staged", "auto", "peer"])
@pytest.mark.parametrize("ids", [(0, 1), (0, 0)])
def test_host_staging_prices_auto_and_staged_cross_card_channels(transport, ids):
    """Auto on a non-peer pair must not silently remove its host RAM reserve."""
    from tilestream.ranks import choose_transports

    exp = replace(experiment_from_run_config(_config(), datetime(2026, 1, 1)),
                  devices=DeviceOptions(count=2, ids=ids, transport=transport))
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        price = estimate_devices(exp, vram_gib=32)
        explicit = estimate_devices(
            replace(exp, devices=replace(exp.devices, transport="host")), vram_gib=32)
    expected = explicit["host_staging_bytes"] if transport != "peer" else 0
    assert price["host_staging_bytes"] == expected
    if ids[0] == ids[1]:
        assert expected == 0
    elif transport != "peer":
        assert expected > 0
    if transport == "auto":
        paths = choose_transports(ids, "auto", {})
        assert paths[(ids[0], ids[1])] == ("local" if ids[0] == ids[1] else "staged")


def test_ineligible_full_inventory_does_not_admit_a_small_snapshot_subset(monkeypatch):
    """A small first frame must not enable an unpriced partial snapshot cache."""
    import numpy as np
    from gpuwm.core import devices_memory as memory

    cfg = _config()
    subset = {"state/mup": np.zeros((cfg.ny, cfg.nx), dtype=np.float32)}
    cap = subset["state/mup"].nbytes * 4
    monkeypatch.setattr(memory, "FRAME_SNAPSHOT_LIMIT_BYTES", cap)
    exp = replace(experiment_from_run_config(cfg, datetime(2026, 1, 1)),
                  devices=DeviceOptions(count=2))
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        price = estimate_devices(exp, inventory=subset, vram_gib=32)
        for rank in price["rank_shapes"]:
            ny, nx = rank["compute_shape"]
            assert ny * nx * 4 < cap
            assert memory.frame_snapshot_budget(replace(cfg, nx=nx, ny=ny)) == 0
            assert rank["frame_snapshot_bytes"] == 0


def test_small_rank_keeps_snapshot_overlap_capacity():
    """The large-rank policy preserves buffered output where every carrier fits."""
    from gpuwm.core.devices_memory import frame_snapshot_budget, inventory_shapes

    cfg = _config(nx=80, ny=64)
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        capacity = frame_snapshot_budget(cfg)
        inventory_bytes = sum(prod(shape) * 4 for shape in inventory_shapes(cfg).values())
    assert 0 < inventory_bytes <= capacity


@pytest.mark.parametrize("surface", [2, 3])
def test_ruc_pinned_mirrors_are_added_only_to_the_host_budget(surface):
    from gpuwm.core.ruc_memory import ruc_pinned_host_bytes

    cfg = _config(sf_surface_physics=surface, num_soil_layers=9 if surface == 3 else 4)
    exp = replace(experiment_from_run_config(cfg, datetime(2026, 1, 1)),
                  devices=DeviceOptions(count=2))
    with mynn_pricing_memory(total_bytes=32 * pf.GIB, free_bytes=32 * pf.GIB):
        price = estimate_devices(exp, vram_gib=32)
    expected = sum(ruc_pinned_host_bytes(ny * nx, 9)
                   for rank in price["rank_shapes"]
                   for ny, nx in [rank["compute_shape"]]) if surface == 3 else 0
    assert price["host_physics_bytes"] == expected
    assert price["host_bytes"] == sum(price[key] for key in (
        "host_store_bytes", "host_staging_bytes", "host_boundary_bytes", "host_physics_bytes"))
    for card in price["cards"]:
        assert card["total_bytes"] == sum(card[key] for key in (
            "resident_bytes", "seam_bytes", "template_bytes", "frame_snapshot_bytes"))
