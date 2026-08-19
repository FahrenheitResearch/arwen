"""The soil state must stop carrying the forcing mesh.

The acceptance instrument is the one that measured the defect on a real
European run: a 16-term bicubic fitted INSIDE each source cell.  A field
that holds no information below the source spacing is explained by it at
R2 ~ 1; terrain (0.907) and land use (0.512) measured the same way are not.
Every test here that claims the imprint is gone measures it with that fit,
and :func:`test_instrument_detects_the_defect_it_is_meant_to_detect` proves
the instrument works in both directions on this fixture -- it must trip on
the unfixed field and it must stay quiet on a field that genuinely carries
sub-cell structure.
"""
import pathlib

import numpy as np
import pytest

from gpuwm.core.noah import SOIL_COLS, load_tables, pack_params
from gpuwm.ingest.horiz import _WPS_FULL_CHAIN, wps_masked_field_interpolate
from gpuwm.ingest.soil import preprocess_noah_soil
from gpuwm.ingest.soil_downscale import (
    SOURCE_MESH_ADVISORY_RATIO,
    SoilMeshPlan,
    declared_soil_texture_downscale,
    downscale_deep_soil_temperature,
    downscale_soil_moisture,
    parse_ingest_table,
    soil_texture_bounds,
)


SOURCE_STEP = 0.25
#: Deliberately NOT a divisor of the source spacing.  A target grid whose
#: cells fall on exact sub-multiples of 0.25 degrees samples only ten
#: discrete phases inside a source cell, which leaves most phase bins empty
#: and makes the phase instrument measure the sampling instead of the
#: field.  A real Lambert grid is incommensurate with the lat/lon source
#: mesh, and so is this fixture.
TARGET_STEP_LON = 0.0231
TARGET_STEP_LAT = 0.0187


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------
def bicubic_r2_within_source_cell(field, lat, lon, mask, step=SOURCE_STEP):
    """Fraction of within-source-cell variance a 16-term bicubic explains."""
    key = ((np.floor(lon / step).astype(np.int64) << 20)
           + np.floor(lat / step).astype(np.int64))
    residual = total = 0.0
    for cell in np.unique(key[mask]):
        selected = mask & (key == cell)
        if int(selected.sum()) < 30:
            continue
        x = np.mod(lon[selected], step) / step
        y = np.mod(lat[selected], step) / step
        values = field[selected]
        design = np.stack([(x ** a) * (y ** b)
                           for a in range(4) for b in range(4)], axis=1)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        error = values - design @ coefficients
        residual += float((error ** 2).sum())
        total += float(((values - values.mean()) ** 2).sum())
    assert total > 0.0
    return 1.0 - residual / total


def phase_concentration(field, lon, mask, step=SOURCE_STEP, bins=25):
    """How much of a field's curvature sits on the source-cell boundaries.

    Two readings of the same phase-binned |curvature| profile, because they
    fail differently.  ``range_over_mean`` is the headline number the
    evidence reports (terrain 0.108, land use 0.109); it is a max-minus-min
    and so is sensitive to a thin tail.  ``excess`` is the amplitude-
    weighted first-harmonic resultant minus its uniform-weight null -- an
    average rather than an extremum, and therefore the one an assertion
    should lean on.
    """
    curvature = np.abs(field[:, :-2] - 2 * field[:, 1:-1] + field[:, 2:])
    inside = mask[:, :-2] & mask[:, 1:-1] & mask[:, 2:]
    weights = curvature[inside]
    phase = np.mod(lon[:, 1:-1][inside] / step, 1.0)
    angle = 2.0 * np.pi * phase
    resultant = float(np.hypot((weights * np.cos(angle)).sum(),
                               (weights * np.sin(angle)).sum())
                      / weights.sum())
    null = float(np.hypot(np.cos(angle).sum(), np.sin(angle).sum())
                 / weights.size)
    index = np.minimum((phase * bins).astype(int), bins - 1)
    profile = (np.bincount(index, weights=weights, minlength=bins)
               / np.maximum(np.bincount(index, minlength=bins), 1))
    return {
        "range_over_mean": float(
            (profile.max() - profile.min()) / max(profile.mean(), 1e-30)),
        "excess": resultant - null,
    }


# ---------------------------------------------------------------------------
# fixture: a coarse source carried onto a fine grid by the REAL WPS chain
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def quilted():
    """A soil state with the forcing mesh in it, and a fine soil texture.

    The moisture is a smooth 0.25 degree field carried to a 0.025 degree
    grid by ``wps_masked_field_interpolate`` with metgrid's own chain, so
    the imprint under test is produced by the production interpolator and
    not by a hand-drawn staircase.  The texture is an independent
    fine-scale categorical map, which is what the 30 arc-second soil
    database is.
    """
    source_lat = np.arange(48.0, 56.0 + 1e-9, SOURCE_STEP)
    source_lon = np.arange(0.0, 12.0 + 1e-9, SOURCE_STEP)
    grid_lon, grid_lat = np.meshgrid(
        np.arange(3.0, 6.0, TARGET_STEP_LON),
        np.arange(51.0, 52.6, TARGET_STEP_LAT))

    mesh_lon, mesh_lat = np.meshgrid(source_lon, source_lat)
    # Cell-to-cell roughness, not just a slow analytic surface: a real
    # forcing model's soil moisture differs between neighbouring source
    # cells, and it is that difference the sixteen-point interpolant turns
    # into a curvature break on the cell boundary.  A perfectly smooth
    # source would hand the test a quilt with no creases to remove.
    noise = np.random.default_rng(7).normal(size=mesh_lon.shape)
    coarse = np.clip(0.22
                     + 0.05 * np.sin(mesh_lon * 0.7)
                     + 0.04 * np.cos(mesh_lat * 0.9)
                     + 0.03 * noise, 0.05, 0.45)

    source_valid = np.ones(mesh_lat.shape, dtype=bool)
    target_active = np.ones(grid_lat.shape, dtype=bool)
    moisture = np.stack([
        wps_masked_field_interpolate(
            coarse + 0.01 * layer, source_lat, source_lon,
            grid_lat, grid_lon, source_valid=source_valid,
            target_active=target_active, chain=_WPS_FULL_CHAIN,
            fill_value=np.nan)
        for layer in range(4)])
    assert np.isfinite(moisture).all()

    # Land everywhere except a block of water, so "water is untouched" is
    # a measurement and not a vacuous pass.
    land = np.ones(grid_lat.shape, dtype=bool)
    land[:12, :12] = False

    # A fine categorical texture: real sub-source-cell structure, exactly
    # what the reconstitution injects and what the quilt lacks.  The
    # patches are built by quantizing a smooth random field rather than by
    # tiling a fixed block, because a fixed block size aliases against the
    # 0.25 degree ruler the phase instrument uses and would hand the test
    # a texture with source-mesh phase structure of its own -- which the
    # 30 arc-second soil database does not have.
    rng = np.random.default_rng(20260814)
    rows, columns = grid_lat.shape
    seed_count = 700
    seed_row = rng.uniform(0.0, rows, size=seed_count)
    seed_column = rng.uniform(0.0, columns, size=seed_count)
    seed_category = rng.integers(1, 13, size=seed_count).astype(np.float64)
    row_index = np.arange(rows)[:, None, None]
    column_index = np.arange(columns)[None, :, None]
    distance = ((row_index - seed_row[None, None, :]) ** 2
                + (column_index - seed_column[None, None, :]) ** 2)
    soil_type = seed_category[distance.argmin(axis=2)]
    soil_type[~land] = 14.0  # SOILPARM's WATER row

    deep = (285.0 + 2.0 * np.sin(grid_lon * 0.8)
            + 0.6 * np.sin(grid_lon * 40.0) * np.cos(grid_lat * 40.0))
    plan = SoilMeshPlan.from_grids(source_lat, source_lon, grid_lat, grid_lon)
    return {
        "lat": grid_lat, "lon": grid_lon, "land": land,
        "moisture": moisture, "soil_type": soil_type, "deep": deep,
        "plan": plan, "params": pack_params(load_tables()),
        "source_lat": source_lat, "source_lon": source_lon,
    }


# ---------------------------------------------------------------------------
# both directions
# ---------------------------------------------------------------------------
def test_instrument_detects_the_defect_it_is_meant_to_detect(quilted):
    """The fit must trip on the quilt and stay quiet on real structure."""
    imprinted = bicubic_r2_within_source_cell(
        quilted["moisture"][3], quilted["lat"], quilted["lon"],
        quilted["land"])
    structured = bicubic_r2_within_source_cell(
        quilted["soil_type"], quilted["lat"], quilted["lon"], quilted["land"])
    assert imprinted > 0.999, imprinted
    assert structured < 0.9, structured


def test_downscaling_removes_the_source_mesh_imprint(quilted):
    """The acceptance criterion, in miniature.  FAILS without the fix."""
    before = bicubic_r2_within_source_cell(
        quilted["moisture"][3], quilted["lat"], quilted["lon"],
        quilted["land"])
    result, receipt = downscale_soil_moisture(
        quilted["moisture"], soil_type=quilted["soil_type"],
        terrestrial=quilted["land"], params=quilted["params"],
        plan=quilted["plan"], announce=False)
    after = bicubic_r2_within_source_cell(
        result[3], quilted["lat"], quilted["lon"], quilted["land"])
    assert receipt["applied"] is True
    assert before > 0.999, before
    assert after < 0.99, after
    assert before - after > 0.01

    before_phase = phase_concentration(
        quilted["moisture"][3], quilted["lon"], quilted["land"])
    after_phase = phase_concentration(
        result[3], quilted["lon"], quilted["land"])
    # The floor a texture-carrying field can reach is the TEXTURE's own
    # reading, not the smooth-field null: the reconstituted moisture
    # inherits the soil map's curvature because that curvature is the
    # information being injected.
    texture_phase = phase_concentration(
        quilted["soil_type"], quilted["lon"], quilted["land"])
    assert before_phase["excess"] > 0.2, before_phase
    assert after_phase["excess"] < before_phase["excess"] / 5.0, (
        before_phase, after_phase)
    assert after_phase["range_over_mean"] < (
        before_phase["range_over_mean"] / 3.0), (before_phase, after_phase)
    assert after_phase["range_over_mean"] == pytest.approx(
        texture_phase["range_over_mean"], rel=0.5), (
            after_phase, texture_phase)


def test_result_stays_inside_the_target_texture_bounds(quilted):
    """A downscaling that produces unphysical moisture is worse than a quilt."""
    result, _ = downscale_soil_moisture(
        quilted["moisture"], soil_type=quilted["soil_type"],
        terrestrial=quilted["land"], params=quilted["params"],
        plan=quilted["plan"], announce=False)
    smcdry, smcmax, texture = soil_texture_bounds(
        quilted["soil_type"], quilted["params"])
    usable = quilted["land"] & texture
    for layer in range(result.shape[0]):
        values = result[layer]
        assert (values[usable] >= smcdry[usable] - 1e-12).all()
        assert (values[usable] <= smcmax[usable] + 1e-12).all()


def test_water_cells_are_untouched(quilted):
    result, _ = downscale_soil_moisture(
        quilted["moisture"], soil_type=quilted["soil_type"],
        terrestrial=quilted["land"], params=quilted["params"],
        plan=quilted["plan"], announce=False)
    water = ~quilted["land"]
    assert np.array_equal(result[:, water], quilted["moisture"][:, water])


def test_total_soil_water_does_not_drift(quilted):
    """Affine reconstitution against the cell-mean texture conserves water."""
    _, receipt = downscale_soil_moisture(
        quilted["moisture"], soil_type=quilted["soil_type"],
        terrestrial=quilted["land"], params=quilted["params"],
        plan=quilted["plan"], announce=False)
    assert abs(receipt["soil_water_change_pct"]) < 2.0, receipt


def test_disabled_is_the_identity(quilted):
    """The WRF-comparison switch returns the input array itself."""
    plan = SoilMeshPlan.from_grids(
        quilted["source_lat"], quilted["source_lon"],
        quilted["lat"], quilted["lon"], enabled=False)
    result, receipt = downscale_soil_moisture(
        quilted["moisture"], soil_type=quilted["soil_type"],
        terrestrial=quilted["land"], params=quilted["params"],
        plan=plan, announce=False)
    assert result is quilted["moisture"]
    assert receipt["applied"] is False
    assert receipt["downscale_enabled"] is False


def test_receipt_records_the_source_resolution_and_warns(quilted):
    receipt = downscale_soil_moisture(
        quilted["moisture"], soil_type=quilted["soil_type"],
        terrestrial=quilted["land"], params=quilted["params"],
        plan=quilted["plan"], announce=False)[1]
    assert receipt["source_spacing_deg"]["lat"] == pytest.approx(SOURCE_STEP)
    assert receipt["source_spacing_deg"]["lon"] == pytest.approx(SOURCE_STEP)
    assert receipt["resolution_ratio"] == pytest.approx(
        SOURCE_STEP / TARGET_STEP_LON, rel=1e-3)
    assert receipt["advisory"] is True
    assert receipt["advisory_ratio"] == SOURCE_MESH_ADVISORY_RATIO
    assert "wrf_reference" in receipt


def test_no_advisory_when_the_source_is_not_coarse(quilted):
    plan = SoilMeshPlan(
        source_spacing_deg_lat=0.03, source_spacing_deg_lon=0.03,
        target_spacing_deg_lat=TARGET_STEP_LAT,
        target_spacing_deg_lon=TARGET_STEP_LON)
    assert plan.advisory is False
    assert plan.resolution_ratio == pytest.approx(
        0.03 / TARGET_STEP_LON, rel=1e-6)


def test_a_source_finer_than_the_target_is_left_alone(quilted):
    plan = SoilMeshPlan(
        source_spacing_deg_lat=0.01, source_spacing_deg_lon=0.01,
        target_spacing_deg_lat=TARGET_STEP_LAT,
        target_spacing_deg_lon=TARGET_STEP_LON)
    result, receipt = downscale_soil_moisture(
        quilted["moisture"], soil_type=quilted["soil_type"],
        terrestrial=quilted["land"], params=quilted["params"],
        plan=plan, announce=False)
    assert result is quilted["moisture"]
    assert receipt["applied"] is False


def test_land_without_a_soil_category_is_reported_not_guessed(quilted):
    soil_type = np.array(quilted["soil_type"])
    soil_type[20:24, 20:24] = 14.0  # WATER row on LAND cells
    receipt = downscale_soil_moisture(
        quilted["moisture"], soil_type=soil_type,
        terrestrial=quilted["land"], params=quilted["params"],
        plan=quilted["plan"], announce=False)[1]
    assert receipt["land_without_texture"] >= 16


# ---------------------------------------------------------------------------
# the deep-temperature analogue
# ---------------------------------------------------------------------------
def test_deep_temperature_keeps_every_source_cell_mean(quilted):
    """Anchoring on TMN must add scales the source lacks and nothing else."""
    temperature = np.stack([
        np.full(quilted["lat"].shape, 283.0 + layer) for layer in range(4)])
    result, receipt = downscale_deep_soil_temperature(
        temperature, deep_soil_temperature=quilted["deep"],
        terrestrial=quilted["land"], plan=quilted["plan"], announce=False)
    assert receipt["applied"] is True
    key = ((np.floor(quilted["lon"] / SOURCE_STEP).astype(np.int64) << 20)
           + np.floor(quilted["lat"] / SOURCE_STEP).astype(np.int64))
    for cell in np.unique(key[quilted["land"]]):
        selected = quilted["land"] & (key == cell)
        if int(selected.sum()) < 30:
            continue
        for layer in range(4):
            assert result[layer][selected].mean() == pytest.approx(
                temperature[layer][selected].mean(), abs=0.05)
    # ... and layer 4 must actually have moved, or nothing was downscaled
    assert receipt["fields"]["TSLB_L4"]["rms_change_k"] > 0.05
    assert (receipt["fields"]["TSLB_L4"]["rms_change_k"]
            > 5.0 * receipt["fields"]["TSLB_L1"]["rms_change_k"])


def test_deep_temperature_disabled_is_the_identity(quilted):
    temperature = np.stack([
        np.full(quilted["lat"].shape, 283.0) for _ in range(4)])
    plan = SoilMeshPlan.from_grids(
        quilted["source_lat"], quilted["source_lon"],
        quilted["lat"], quilted["lon"], enabled=False)
    result, receipt = downscale_deep_soil_temperature(
        temperature, deep_soil_temperature=quilted["deep"],
        terrestrial=quilted["land"], plan=plan, announce=False)
    assert result is temperature
    assert receipt["applied"] is False


# ---------------------------------------------------------------------------
# the declaration
# ---------------------------------------------------------------------------
def _config_loading_available():
    """Skip the loader tests on a worktree that is not pip-installed.

    ``load_experiment`` runs the version-identity gate first, which
    refuses a tree whose distribution metadata belongs to another install.
    That refusal is a property of the checkout, not of this change, and it
    is the same one every other config-loading test in the suite hits.
    """
    from gpuwm.provenance_gate import resolve, version_identity_refusal

    return version_identity_refusal(resolve()) is None


def test_silence_means_enabled(tmp_path):
    assert declared_soil_texture_downscale(None) is True
    config = tmp_path / "silent.toml"
    config.write_text("[experiment]\nname = 'x'\n", encoding="utf-8")
    assert declared_soil_texture_downscale(config) is True


def test_declaration_can_turn_it_off_for_wrf_comparison(tmp_path):
    config = tmp_path / "wrf.toml"
    config.write_text(
        "[ingest]\nsoil_texture_downscale = false\n", encoding="utf-8")
    assert declared_soil_texture_downscale(config) is False


def test_a_non_boolean_declaration_is_refused(tmp_path):
    config = tmp_path / "bad.toml"
    config.write_text(
        "[ingest]\nsoil_texture_downscale = 'yes'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be true or false"):
        declared_soil_texture_downscale(config)


def test_an_unknown_ingest_key_is_refused():
    with pytest.raises(ValueError, match="unknown key"):
        parse_ingest_table({"soil_texture_downscsle": False}, source="x.toml")


def test_the_switch_is_reachable_from_a_bare_experiment_config(tmp_path):
    """`gpuwm go` configs carry no ``[case_data]``, so the switch cannot
    live there.  A committed single-domain config must accept ``[ingest]``
    through the experiment loader and keep its meaning."""
    from gpuwm.experiment import load_experiment

    if not _config_loading_available():
        pytest.skip("worktree is not bound to its own distribution metadata")
    template = pathlib.Path("configs") / "gfs_wrf_direct_proof.toml"
    if not template.is_file():
        pytest.skip("no committed experiment config to extend")
    payload = template.read_text(encoding="utf-8")
    assert "[ingest]" not in payload
    config = tmp_path / "with_ingest.toml"
    config.write_text(
        payload + "\n[ingest]\nsoil_texture_downscale = false\n",
        encoding="utf-8")
    load_experiment(config)          # must not refuse the table
    assert declared_soil_texture_downscale(config) is False


def test_a_typo_in_the_ingest_table_is_refused_by_the_loader(tmp_path):
    from gpuwm.experiment import load_experiment

    if not _config_loading_available():
        pytest.skip("worktree is not bound to its own distribution metadata")
    template = pathlib.Path("configs") / "gfs_wrf_direct_proof.toml"
    if not template.is_file():
        pytest.skip("no committed experiment config to extend")
    config = tmp_path / "typo.toml"
    config.write_text(
        template.read_text(encoding="utf-8")
        + "\n[ingest]\nsoil_texture_downscaling = false\n",
        encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_experiment(config)


def test_a_nested_child_inherits_the_declaration_through_the_catalog():
    """A child sees the catalog, not the case data, and must not seam."""
    import dataclasses

    from gpuwm.ingest.preflight import InputCatalog

    assert "soil_texture_downscale" in {
        entry.name for entry in dataclasses.fields(InputCatalog)}


# ---------------------------------------------------------------------------
# end to end, through the function the routes actually call
# ---------------------------------------------------------------------------
def _gfs_soil_fields(quilted):
    shape = quilted["lat"].shape
    fields = {
        "LANDSEA": quilted["land"].astype(np.float64),
        "SKINTEMP": np.full(shape, 288.0),
        "TMN": quilted["deep"],
    }
    for index, name in enumerate(
            ("GFS_ST000010", "GFS_ST010040", "GFS_ST040100", "GFS_ST100200")):
        fields[name] = np.full(shape, 285.0 + index)
    for index, name in enumerate(
            ("GFS_SM000010", "GFS_SM010040", "GFS_SM040100", "GFS_SM100200")):
        fields[name] = quilted["moisture"][index]
    return fields


def test_preprocess_noah_soil_applies_it_by_default(quilted, capsys):
    """The front door: a plan in, the imprint out.  FAILS without the fix."""
    fields = _gfs_soil_fields(quilted)
    plain = preprocess_noah_soil(
        fields, soil_type=quilted["soil_type"],
        landmask=quilted["land"].astype(np.float64),
        water_temperature_policy="wrf_compat")
    fixed = preprocess_noah_soil(
        fields, soil_type=quilted["soil_type"],
        landmask=quilted["land"].astype(np.float64),
        water_temperature_policy="wrf_compat", soil_mesh=quilted["plan"])
    capsys.readouterr()

    before = bicubic_r2_within_source_cell(
        plain.soil_moisture[3], quilted["lat"], quilted["lon"],
        quilted["land"])
    after = bicubic_r2_within_source_cell(
        fixed.soil_moisture[3], quilted["lat"], quilted["lon"],
        quilted["land"])
    assert before > 0.999, before
    assert after < 0.99, after
    assert fixed.soil_texture_downscale["applied"] is True
    assert plain.soil_texture_downscale == {}
    # SH2O is derived FROM the downscaled moisture, not from the quilt
    assert not np.allclose(fixed.liquid_moisture, plain.liquid_moisture)


def test_preprocess_noah_soil_reports_the_source_resolution(quilted):
    fields = _gfs_soil_fields(quilted)
    state = preprocess_noah_soil(
        fields, soil_type=quilted["soil_type"],
        landmask=quilted["land"].astype(np.float64),
        water_temperature_policy="wrf_compat", soil_mesh=quilted["plan"])
    receipt = state.soil_texture_downscale
    assert receipt["source_spacing_deg"]["lon"] == pytest.approx(SOURCE_STEP)
    assert receipt["advisory"] is True
    assert receipt["deep_soil_temperature"]["applied"] is True


def test_preprocess_noah_soil_disabled_is_byte_identical(quilted):
    fields = _gfs_soil_fields(quilted)
    plain = preprocess_noah_soil(
        fields, soil_type=quilted["soil_type"],
        landmask=quilted["land"].astype(np.float64),
        water_temperature_policy="wrf_compat")
    plan = SoilMeshPlan.from_grids(
        quilted["source_lat"], quilted["source_lon"],
        quilted["lat"], quilted["lon"], enabled=False)
    declined = preprocess_noah_soil(
        fields, soil_type=quilted["soil_type"],
        landmask=quilted["land"].astype(np.float64),
        water_temperature_policy="wrf_compat", soil_mesh=plan)
    for name in ("soil_moisture", "soil_temperature", "liquid_moisture",
                 "tsk", "deep_soil_temperature"):
        assert np.array_equal(getattr(plain, name), getattr(declined, name)), \
            name
    assert declined.soil_texture_downscale["applied"] is False


def test_the_smcdry_floor_still_reports_what_the_source_delivered(quilted):
    """The floor's receipt describes the SOURCE, so it runs first."""
    fields = _gfs_soil_fields(quilted)
    fields = dict(fields)
    fields["GFS_SM000010"] = np.zeros_like(fields["GFS_SM000010"])
    state = preprocess_noah_soil(
        fields, soil_type=quilted["soil_type"],
        landmask=quilted["land"].astype(np.float64),
        water_temperature_policy="wrf_compat", soil_mesh=quilted["plan"])
    assert state.moisture_floor["total_floored_cells"] > 0
    smcdry, _, texture = soil_texture_bounds(
        quilted["soil_type"], quilted["params"])
    usable = quilted["land"] & texture
    assert (state.soil_moisture[0][usable] >= smcdry[usable] - 1e-12).all()


# ---------------------------------------------------------------------------
# the router and the nest
# ---------------------------------------------------------------------------
def test_the_router_forwards_the_plan(quilted):
    from gpuwm.ingest.ruc_soil import preprocess_land_surface_soil

    fields = _gfs_soil_fields(quilted)
    routed = preprocess_land_surface_soil(
        fields, sf_surface_physics=2, soil_type=quilted["soil_type"],
        landmask=quilted["land"].astype(np.float64),
        water_temperature_policy="wrf_compat", soil_mesh=quilted["plan"])
    direct = preprocess_noah_soil(
        fields, soil_type=quilted["soil_type"],
        landmask=quilted["land"].astype(np.float64),
        water_temperature_policy="wrf_compat", soil_mesh=quilted["plan"])
    assert np.array_equal(routed.soil_moisture, direct.soil_moisture)
    assert routed.soil_texture_downscale["applied"] is True


def test_a_nest_carries_the_plan_to_its_own_soil_column():
    """A child re-ingests the same source, so it must inherit the remedy."""
    import inspect

    from gpuwm.ingest import nest_init

    assert "soil_mesh" in inspect.signature(
        nest_init.PreparedChildInput.__init__).parameters
    source = inspect.getsource(nest_init.finalize_prepared_child)
    assert "soil_mesh=prepared.soil_mesh" in source


def test_every_production_route_declares_a_source_mesh():
    """No front door may quietly keep the defect."""
    import inspect

    from gpuwm import era5_direct, gfs_direct, mapped_direct, runtime

    for module, function in ((era5_direct, "prepare_era5_wrf"),
                             (gfs_direct, "prepare_gfs_wrf"),
                             (mapped_direct, "prepare_mapped_wrf"),
                             (runtime, "prepare_real_case")):
        source = inspect.getsource(getattr(module, function))
        assert "soil_mesh=" in source, function


def test_smcdry_and_smcmax_come_from_soilparm(quilted):
    """The constants are Noah's own, not numbers invented here."""
    params = quilted["params"]
    smcdry, smcmax, _ = soil_texture_bounds(
        np.array([[1.0, 12.0]]), params)
    assert smcdry[0, 0] == pytest.approx(
        params.soil[0, SOIL_COLS.index("smcdry")])
    assert smcmax[0, 1] == pytest.approx(
        params.soil[11, SOIL_COLS.index("smcmax")])


def test_the_fallback_layer_geometry_cannot_drift_from_noahs():
    """A silent disagreement would put a plausible wrong number in a
    receipt and a plausible wrong depth weight on the deep layers."""
    from gpuwm.ingest import soil, soil_downscale

    soil_downscale._refuse_soil_geometry_drift()
    assert np.array_equal(soil_downscale.NOAH_DEFAULT_LAYER_THICKNESS_M,
                          soil.NOAH_LAYER_THICKNESS_M)
    assert np.array_equal(soil_downscale.NOAH_DEFAULT_LAYER_MIDPOINTS_M,
                          soil.NOAH_LAYER_MIDPOINTS_M)


def test_the_source_cell_low_pass_is_a_true_fractional_box_mean():
    """The footprint is 5.5 by 9.1 cells on the domain this was built for.
    Rounding that to 5 by 9 biases the low-pass by ten percent, and that
    bias lands directly in the reconstituted moisture."""
    from gpuwm.ingest.soil_downscale import _frac_box_mean

    rng = np.random.default_rng(11)
    values = rng.normal(size=(21, 21))
    valid = np.ones_like(values, dtype=bool)
    width = 5.5
    got = _frac_box_mean(values, valid, width, 1.0)
    # brute force, same definition: integer core plus a fractional edge
    half = (width - 1.0) / 2.0
    core, edge = int(np.floor(half)), half - int(np.floor(half))
    row, column = 10, 10
    total = weight = 0.0
    for offset in range(-core - 1, core + 2):
        index = column + offset
        if not 0 <= index < values.shape[1]:
            continue
        w = 1.0 if abs(offset) <= core else edge
        total += w * values[row, index]
        weight += w
    assert got[row, column] == pytest.approx(total / weight)


def test_the_low_pass_ignores_cells_without_a_texture():
    """A coastline must not pull water's (0.0, 1.0) pair into a land
    cell's denominator."""
    from gpuwm.ingest.soil_downscale import _frac_box_mean

    values = np.array([[1.0, 2.0, 99.0, 4.0, 5.0]])
    valid = np.array([[True, True, False, True, True]])
    got = _frac_box_mean(values, valid, 3.0, 1.0)
    assert got[0, 2] == pytest.approx((2.0 + 4.0) / 2.0)
    assert 99.0 not in got


def test_a_window_narrower_than_one_cell_is_the_cell_itself():
    """The two axes are independent: a source can be coarser in latitude
    and finer in longitude than the target.  An un-clamped half-width goes
    negative there and the cumulative-sum window inverts, which returns
    negative 'means' with no error."""
    from gpuwm.ingest.soil_downscale import _frac_box_mean

    values = np.arange(25.0).reshape(5, 5)
    valid = np.ones_like(values, dtype=bool)
    got = _frac_box_mean(values, valid, 0.4, 1.0)
    assert np.allclose(got, values)
    assert np.isfinite(got).all()


def test_a_mixed_footprint_downscales_on_the_coarse_axis_only(quilted):
    """One axis finer than the target, one coarser: the plan is still a
    plan, and the result is still bounded."""
    plan = SoilMeshPlan(
        source_spacing_deg_lat=SOURCE_STEP,
        source_spacing_deg_lon=0.4 * TARGET_STEP_LON,
        target_spacing_deg_lat=TARGET_STEP_LAT,
        target_spacing_deg_lon=TARGET_STEP_LON)
    result, receipt = downscale_soil_moisture(
        quilted["moisture"], soil_type=quilted["soil_type"],
        terrestrial=quilted["land"], params=quilted["params"],
        plan=plan, announce=False)
    assert receipt["applied"] is True
    smcdry, smcmax, texture = soil_texture_bounds(
        quilted["soil_type"], quilted["params"])
    usable = quilted["land"] & texture
    assert (result[0][usable] >= smcdry[usable] - 1e-12).all()
    assert (result[0][usable] <= smcmax[usable] + 1e-12).all()


def test_the_declaration_reaches_a_nest_on_every_hierarchy_route():
    """A child sees the catalog, not the config.  A WRF-comparison run that
    disabled the reconstitution on the parent and silently kept it on every
    nest would seam at the nest boundary -- and would be a comparison that
    is not a comparison."""
    import dataclasses
    import inspect

    from gpuwm import era5_direct, gfs_direct, mapped_direct, source_hierarchy
    from gpuwm.ingest.nest_init import NestedInputCatalog
    from gpuwm.ingest.preflight import InputCatalog

    for catalog in (NestedInputCatalog, InputCatalog):
        names = {entry.name for entry in dataclasses.fields(catalog)}
        assert "soil_texture_downscale" in names, catalog.__name__

    assert "soil_texture_downscale" in inspect.signature(
        source_hierarchy.initialize_and_export_regular_source_hierarchy
    ).parameters
    for module, function in ((era5_direct, "prepare_era5_wrf"),
                             (gfs_direct, "prepare_gfs_wrf"),
                             (mapped_direct, "prepare_mapped_wrf")):
        source = inspect.getsource(getattr(module, function))
        assert "soil_texture_downscale=declared_soil_texture_downscale" in             source, function


def test_a_nested_catalog_defaults_the_declaration_on():
    from gpuwm.ingest.nest_init import NestedInputCatalog
    from gpuwm.ingest.soil_downscale import declared_soil_texture_downscale

    class _Snapshot:
        valid_time = __import__("datetime").datetime(2026, 8, 14, 12)

    catalog = NestedInputCatalog(
        snapshots=(_Snapshot(),), static_catalog=object())
    assert catalog.soil_texture_downscale is True
    assert declared_soil_texture_downscale(catalog) is True
    declined = NestedInputCatalog(
        snapshots=(_Snapshot(),), static_catalog=object(),
        soil_texture_downscale=False)
    assert declared_soil_texture_downscale(declined) is False


def test_a_source_that_cannot_describe_a_mesh_is_not_an_error(quilted):
    """Not every source has a regular mesh to measure -- native HRRR is on
    a Lambert grid, and a stand-in snapshot in a test has no axes at all.
    Reaching into ``.latitude`` at the call site turned "there is nothing
    to measure" into an AttributeError at every front door."""
    from types import SimpleNamespace

    from gpuwm.ingest.soil_downscale import soil_mesh_plan_from_case

    lat, lon = quilted["lat"], quilted["lon"]
    real = SimpleNamespace(latitude=quilted["source_lat"],
                           longitude=quilted["source_lon"])
    # a source that cannot describe a mesh
    assert soil_mesh_plan_from_case(SimpleNamespace(), (lat, lon)) is None
    assert soil_mesh_plan_from_case(
        SimpleNamespace(latitude=None, longitude=None), (lat, lon)) is None
    assert soil_mesh_plan_from_case(
        SimpleNamespace(latitude=lat, longitude=lon), (lat, lon)) is None
    # a TARGET that is not a grid -- the other end, which failed the same
    # way one commit later
    assert soil_mesh_plan_from_case(real, SimpleNamespace()) is None
    assert soil_mesh_plan_from_case(real, None) is None
    assert soil_mesh_plan_from_case(real, (None, None)) is None
    # both ends real, both ways of naming the target
    plan = soil_mesh_plan_from_case(real, (lat, lon))
    assert plan is not None and plan.enabled is True
    assert soil_mesh_plan_from_case(
        real, SimpleNamespace(latlon_mass=lambda: (lat, lon))
    ).footprint_cells == plan.footprint_cells
    assert soil_mesh_plan_from_case(
        real, (lat, lon), enabled=False).enabled is False


def test_no_source_mesh_is_announced_rather_than_silently_skipped(
        quilted, capsys):
    """A route that declares none must say so: without it the forcing mesh
    stays in SMOIS for the whole forecast."""
    from gpuwm.ingest import soil as soil_module

    soil_module._REPORTED_MISSING_SOIL_MESH.clear()
    fields = _gfs_soil_fields(quilted)
    state = preprocess_noah_soil(
        fields, soil_type=quilted["soil_type"],
        landmask=quilted["land"].astype(np.float64),
        water_temperature_policy="wrf_compat", soil_mesh=None)
    captured = capsys.readouterr()
    assert state.soil_texture_downscale == {}
    assert "declared none" in captured.err
    soil_module._REPORTED_MISSING_SOIL_MESH.clear()

