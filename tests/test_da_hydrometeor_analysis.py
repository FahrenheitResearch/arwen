"""The radar precipitation analysis: the device stage and its seam.

``gpuwm.da.hydrometeor_analysis`` ports NOAA's Thompson retrieval and the
final precipitation analysis of the GSD cloud analysis (NOAA-EMC/HRRR
v4.1.21).  NOAA's own code is the oracle for the kernels
(``tools/gsd_precip_oracle``); these tests hold the seam: the stage inside
``assimilate_radar_grid``, its refusals, and a planted twin written through
the radar lane's own writer and applied by the one writer of an analysis.

Device tests are marked ``gpu``.  The CPU ones never open a CUDA context.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path

import numpy as np
import pytest

from gpuwm.da import hydrometeor_analysis as ha
from gpuwm.da import obsop
from gpuwm.da.letkf import Localization
from gpuwm.da.radar_assimilation import (
    PRECIP_ANALYSIS_MEMBER_REFUSAL, PRECIP_ANALYSIS_MODES,
    RadarAssimilationConfig, RadarAssimilationError, assimilate_radar_grid,
    grid_rotation, mass_to_u_faces, mass_to_v_faces, mass_to_w_faces)

NZ, NY, NX = 6, 16, 16
DX_M = 3000.0
TOP_M = 12000.0
MEMBERS = 6
SEED = 20261003
#: Observed echo: columns [4, 8) x [4, 8), levels 1 to 4.
ECHO = (slice(1, 5), slice(4, 8), slice(4, 8))
#: Observed clear air: columns [9, 13) x [4, 12), every level.
CLEAR = (slice(0, NZ), slice(9, 13), slice(4, 12))
#: Rain the radar never sampled: columns [0, 3) x [0, 16), levels 1 to 3.
UNSAMPLED = (slice(1, 4), slice(0, 3), slice(0, NX))
PRECIP = ("qr", "nr", "qs", "qg")


# ---------------------------------------------------------------------------
# refusals and field sets (CPU)
# ---------------------------------------------------------------------------


def _config(**overrides):
    kwargs = dict(solve_device="host",
                  localization=Localization(horizontal_m=15000.0,
                                            vertical_m=4000.0),
                  rtps_alpha=0.9, analysis_fields=("u", "v"))
    kwargs.update(overrides)
    return RadarAssimilationConfig(**kwargs)


def test_precip_analysis_is_off_by_default_and_names_its_modes():
    assert _config().precip_analysis == "off"
    assert PRECIP_ANALYSIS_MODES == ("off", "clear")
    assert ha.MODES == ("clear", "trim-build", "retrieve-all")
    with pytest.raises(RadarAssimilationError, match="precip_analysis must"):
        _config(precip_analysis="retrieve")


def test_mode_identifiers_name_what_they_do_not_a_model():
    # The project law: no case or model name in a generic identifier.  The
    # modes are identifiers users type; NOAA's file and line stay in the
    # docstrings and the receipt, where they are provenance.
    banned = ("hrrr", "rap", "rtma", "gfs", "rrfs")
    for mode in ha.MODES + PRECIP_ANALYSIS_MODES:
        assert not any(word in mode.lower().split("-") for word in banned), mode
    assert not any(word in key for key in ha.NOAA_SETTINGS["limits_kg_kg"]
                   ["value"] for word in banned)
    for old in ("hrrr", "HRRR"):
        with pytest.raises(RadarAssimilationError,
                           match="precip_analysis must"):
            _config(precip_analysis=old, mp_physics=8)
        with pytest.raises(ha.HydrometeorAnalysisError, match="mode must"):
            ha.HydrometeorAnalysisConfig(mode=old)


def test_trim_build_is_refused_for_an_ensemble_analysis():
    # NOAA gives ensemble members the clear step only; this seam analyses
    # members and nothing else, so the deterministic rule is refused here
    # for every scheme, with the breakage named.
    for mp in (8, 28, None, 6):
        with pytest.raises(RadarAssimilationError,
                           match="refused for an ensemble analysis") as err:
            _config(precip_analysis="trim-build", mp_physics=mp)
        assert str(err.value) == PRECIP_ANALYSIS_MEMBER_REFUSAL
    assert "converges on one value" in PRECIP_ANALYSIS_MEMBER_REFUSAL
    assert ("test_trim_build_on_every_member_collapses_rain_spread"
            in PRECIP_ANALYSIS_MEMBER_REFUSAL)
    # clear fits every scheme, stated or not.
    for mp in (None, 6, 10):
        assert _config(precip_analysis="clear",
                       mp_physics=mp).precip_analysis == "clear"


def test_trim_build_is_refused_for_a_scheme_without_thompsons_moments():
    # The module rule, for the deterministic member it is kept for.
    # Thompson and aerosol-aware Thompson carry the rule's moment set.
    for mp in (8, 28):
        ha.require_thompson_rain_number(mp, mode="trim-build")
    for mp in (10, 6, 9):
        with pytest.raises(ha.HydrometeorAnalysisError, match="Use 'clear'"):
            ha.require_thompson_rain_number(mp, mode="trim-build")
    with pytest.raises(ha.HydrometeorAnalysisError, match="State mp_physics"):
        ha.require_thompson_rain_number(None, mode="trim-build")


def test_the_stage_needs_a_radar_file():
    with pytest.raises(RadarAssimilationError, match="reads no radar file"):
        _config(velocity=False, precip_analysis="clear")


def test_clear_takes_each_precipitating_mass_with_its_paired_moments():
    morrison = ("qv", "qc", "nc", "qr", "nr", "qi", "ni", "qs", "ns", "qg",
                "ng", "p", "alt")
    assert ha.precipitating_fields(morrison, mp_physics=10) == (
        "qr", "nr", "qs", "ns", "qg", "ng")
    thompson = ("qv", "qc", "qr", "nr", "qi", "ni", "qs", "qg")
    assert ha.precipitating_fields(thompson, mp_physics=8) == (
        "qr", "nr", "qs", "qg")
    assert ha.precipitating_fields(("qv", "qc", "qr", "qi", "qs", "qg"),
                                   mp_physics=6) == ("qr", "qs", "qg")
    # Detected from the spellings when the scheme is not stated.
    assert ha.precipitating_fields(thompson) == ("qr", "nr", "qs", "qg")


def test_without_a_device_the_stage_refuses_by_name(monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(ha.HydrometeorAnalysisError,
                       match="no host implementation"):
        ha.require_device()


def test_the_driver_flag_reaches_the_configuration():
    from tools.da_cycle_prepared import plan_radar_assimilation

    def args(**updates):
        values = dict(
            horizontal_loc_m=12000.0, vertical_loc_m=3000.0, rtps_alpha=0.9,
            relaxation="rtps", thin_cells=1, err_inflation=1.0,
            z_thin_cells=1, z_err_inflation=1.0, z0_thin_cells=4,
            z0_err_inflation=1.0, cwp_thin_cells=1, cwp_err_inflation=1.0,
            cwp_horizontal_loc_m=None, cwp_vertical_loc_m=None,
            positivity_policy="clip", solve_device="host",
            memory_budget_mib=512.0, hydrometeors=False,
            reflectivity_analysis=False, clear_air_analysis=False,
            goes_cwp=[])
        values.update(updates)
        return types.SimpleNamespace(**values)

    for mode in PRECIP_ANALYSIS_MODES:
        cfg = plan_radar_assimilation(args(precip_analysis=mode), 8,
                                      analysis_fields=("u", "v"), cwp=False)
        assert cfg.precip_analysis == mode
    # A namespace built before the flag existed still plans, with it off.
    assert plan_radar_assimilation(args(), 8, analysis_fields=("u", "v"),
                                   cwp=False).precip_analysis == "off"


def test_the_flag_sits_beside_the_clear_air_flag_with_its_three_values():
    import ast

    source = (Path(__file__).resolve().parents[1] / "tools"
              / "da_cycle_prepared.py").read_text(encoding="utf-8")
    flags = []
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node, ast.Call)
                and getattr(node.func, "attr", None) == "add_argument"
                and node.args and isinstance(node.args[0], ast.Constant)):
            flags.append((node.lineno, node.args[0].value, node))
    flags.sort(key=lambda row: row[0])
    names = [name for _line, name, _node in flags]
    assert names[names.index("--clear-air-analysis") + 1] == "--precip-analysis"
    node = flags[names.index("--precip-analysis")][2]
    keywords = {kw.arg: kw.value for kw in node.keywords}
    assert ast.literal_eval(keywords["choices"]) == PRECIP_ANALYSIS_MODES
    assert ast.literal_eval(keywords["default"]) == "off"


def test_kernel_source_divides_only_through_round_to_nearest_intrinsics():
    # The lane rule behind the literal-division gate: no float division by a
    # constant in kernel code, read by the gate's own scanner; every double
    # quotient in the port is __ddiv_rn.
    from tools.literal_division_scan import literal_divisions

    assert literal_divisions(ha._SOURCE) == []
    assert ha._SOURCE.count("__ddiv_rn") >= 10


# ---------------------------------------------------------------------------
# the planted twin (world building)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def grid():
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.0, dx=DX_M, dy=DX_M, e_we=NX + 1, e_sn=NY + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, TOP_M, NZ + 1),
        name="hydrometeor-analysis-test")


def _smooth(field):
    for _ in range(2):
        for axis in (1, 2):
            field = (np.roll(field, 1, axis) + field
                     + np.roll(field, -1, axis)) / 3.0
    return field


def _member(rng, index):
    shape = (NZ, NY, NX)
    u = 8.0 + _smooth(rng.normal(0.0, 1.5, shape))
    v = -2.0 + _smooth(rng.normal(0.0, 1.5, shape))
    w = _smooth(rng.normal(0.0, 0.3, shape))
    column_p = np.linspace(95000.0, 30000.0, NZ)
    column_t = np.linspace(296.0, 240.0, NZ)
    p = np.broadcast_to(column_p[:, None, None], shape).copy()
    temperature = np.broadcast_to(column_t[:, None, None], shape).copy()
    column_q = np.linspace(8.0e-3, 2.0e-4, NZ) * (1.0 + 0.02 * index)
    qv = np.broadcast_to(column_q[:, None, None], shape).copy()
    alt = 287.0 * temperature * (1.0 + 1.61 * qv) / p
    factor = 1.0 + 0.1 * index
    qr = np.zeros(shape)
    nr = np.zeros(shape)
    qs = np.zeros(shape)
    qg = np.zeros(shape)
    # False precipitation where the radar observed clear air.
    clear = np.zeros(shape, bool)
    clear[CLEAR] = True
    clear[0] = False
    qr[clear] = 1.0e-3 * factor
    nr[clear] = 1.0e4 * factor
    qs[clear] = 4.0e-4 * factor
    qg[clear] = 2.0e-4 * factor
    # Real rain under observed echo.
    qr[ECHO] = 2.0e-3 * factor
    nr[ECHO] = 2.0e4
    qs[ECHO] = 1.0e-4
    qg[ECHO] = 5.0e-4
    # Rain the radar never sampled.
    qr[UNSAMPLED] = 1.5e-3 * factor
    nr[UNSAMPLED] = 1.5e4
    fields = {
        "u": mass_to_u_faces(u), "v": mass_to_v_faces(v),
        "w": mass_to_w_faces(w),
        "thp": _smooth(rng.normal(0.0, 0.5, shape)),
        "qv": qv, "qr": qr, "nr": nr, "qs": qs, "qg": qg, "p": p,
        "alt": alt,
    }
    return {name: np.asarray(value, np.float32)
            for name, value in fields.items()}


@pytest.fixture(scope="module")
def world(grid, tmp_path_factory):
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.superob import GriddedObservations, SuperobParams

    root = tmp_path_factory.mktemp("hydrometeor-analysis")
    rng = np.random.default_rng(SEED)
    checkpoints = {}
    backgrounds = {}
    for index in range(MEMBERS):
        member_dir = root / f"member_{index:03d}"
        member_dir.mkdir()
        fields = _member(rng, index)
        path = member_dir / "gpuwmrst_d01_000600.npz"
        np.savez(path, **{f"state/{name}": value
                          for name, value in fields.items()})
        checkpoints[index] = path
        backgrounds[index] = fields

    shape = (NZ, NY, NX)
    site = obsop.RadarSite(latitude_deg=float(grid.lat[NY // 2, 1]),
                           longitude_deg=float(grid.lon[NY // 2, 1]),
                           altitude_m=350.0, name="AAAA")
    beam = obsop.beam_geometry(obsop.GridGeometry.from_target_grid(grid),
                               site)
    east, north, up = (np.broadcast_to(np.asarray(c, np.float64),
                                       shape).copy()
                       for c in beam.unit_vector_enu())
    vr_mask = np.zeros((1,) + shape, np.int8)
    vr_mask[0, 2:4, 5:11, 5:11] = 1
    sina, cosa = grid_rotation(grid)
    u_e, v_n = obsop.earth_relative_winds(
        np.full(shape, 9.0), np.full(shape, -1.0), sina, cosa)
    vr = (u_e * east + v_n * north)[None]
    z_obs = np.zeros(shape, np.float32)
    z_mask = np.zeros(shape, np.int8)
    z_obs[ECHO] = 30.0
    z_obs[2, 4:8, 4:8] = 42.0
    z_mask[ECHO] = 1
    z0_mask = np.zeros(shape, np.int8)
    z0_mask[CLEAR] = 1
    zeros = np.zeros(shape, np.float32)
    observations = GriddedObservations(
        z_obs=z_obs, z_mask=z_mask,
        z_err=np.where(z_mask == 1, 5.0, 0.0).astype(np.float32),
        z_max=z_obs, z_mean=z_obs, z_count=z_mask.astype(np.int32) * 4,
        vr_obs=vr.astype(np.float32), vr_mask=vr_mask,
        vr_err=np.where(vr_mask == 1, 1.0, 0.0).astype(np.float32),
        vr_count=vr_mask.astype(np.int32),
        vr_rejected=np.zeros((1,) + shape, np.int32),
        vr_beam_east=east[None].astype(np.float32),
        vr_beam_north=north[None].astype(np.float32),
        vr_beam_up=up[None].astype(np.float32),
        radars=[{"id": site.name, "lat_deg": site.latitude_deg,
                 "lon_deg": site.longitude_deg, "alt_m": site.altitude_m,
                 "valid_time": "2026-01-01T00:00:00Z"}],
        counts=[], provenance=[],
        z0_mask=z0_mask, z0_count=z0_mask.astype(np.int32) * 8,
        z0_err=np.where(z0_mask == 1, 7.5, 0.0).astype(np.float32))
    obs_path = root / "obs-radar-grid.nc"
    write_radar_grid(obs_path, observations, grid,
                     valid_time="2026-01-01T00:00:00Z",
                     params=SuperobParams(), overwrite=True)
    return types.SimpleNamespace(root=root, checkpoints=checkpoints,
                                 backgrounds=backgrounds, obs_path=obs_path,
                                 zeros=zeros)


def _apply(world, increments, tag, mp_physics=8):
    from gpuwm.ensemble.increments import apply_increments_to_checkpoint

    analysed = {}
    for index, member in increments.items():
        dest = world.root / f"analysis_{tag}_{index:03d}.npz"
        apply_increments_to_checkpoint(world.checkpoints[index], member,
                                       dest, mp_physics=mp_physics)
        with np.load(dest) as data:
            analysed[index] = {key[len("state/"):]: data[key]
                               for key in data.files}
    return analysed


def _mask(region):
    out = np.zeros((NZ, NY, NX), bool)
    out[region] = True
    return out


def test_the_stage_refuses_without_a_device_inside_the_seam(world, grid,
                                                             monkeypatch):
    monkeypatch.setitem(sys.modules, "cupy", None)
    cfg = _config(precip_analysis="clear", mp_physics=8)
    with pytest.raises(ha.HydrometeorAnalysisError,
                       match="no host implementation"):
        assimilate_radar_grid(world.checkpoints, world.obs_path, grid, cfg)


def test_off_adds_nothing_to_the_analysis(world, grid):
    increments, provenance = assimilate_radar_grid(
        world.checkpoints, world.obs_path, grid, _config())
    assert "precip_analysis" not in provenance
    for member in increments.values():
        assert sorted(member) == ["u", "v"]


# ---------------------------------------------------------------------------
# on the card
# ---------------------------------------------------------------------------


@pytest.mark.gpu
def test_planted_twin_clear_removes_false_rain_and_only_that(world, grid):
    from gpuwm.da.moments import moment_consistency_report

    cfg = _config(precip_analysis="clear", mp_physics=8)
    increments, provenance = assimilate_radar_grid(
        world.checkpoints, world.obs_path, grid, cfg)
    receipt = provenance["precip_analysis"]
    assert receipt["fields"] == list(PRECIP)
    assert receipt["filter_analysed_fields"] == []
    analysed = _apply(world, increments, "clear")
    clear, echo = _mask(CLEAR), _mask(ECHO)
    covered = clear | echo
    for index in range(MEMBERS):
        before = world.backgrounds[index]
        after = analysed[index]
        for name in PRECIP:
            # No rain (or snow, graupel, rain number) left in observed
            # clear air, and nothing removed under observed echo.
            assert np.count_nonzero(after[name][clear]) == 0, name
            assert np.array_equal(after[name][echo].view(np.int32),
                                  before[name][echo].view(np.int32)), name
            # Cells the radar did not sample are bitwise untouched.
            assert np.array_equal(after[name][~covered].view(np.int32),
                                  before[name][~covered].view(np.int32))
            assert np.all(after[name] >= 0.0)
            # The stage's increment is exactly zero there too.
            assert np.all(increments[index][name][~covered] == 0.0)
        report = moment_consistency_report(after, mp_physics=8)
        assert report["consistent"], report
        assert report["stranded_number_cells_total"] == 0, report
        # The receipt's removed water equals what was planted.
        member = receipt["members"][index]["per_field"]
        alt = before["alt"].astype(np.float64)
        dz = TOP_M / NZ
        for name in ("qr", "qs", "qg"):
            planted = before[name][clear].astype(np.float64)
            assert member[name]["removed_sum"] == pytest.approx(
                planted.sum(), rel=1e-12)
            assert member[name]["added_sum"] == 0.0
            planted_kg = (planted * DX_M * DX_M * dz / alt[clear]).sum()
            assert member[name]["removed_kg"] == pytest.approx(
                planted_kg, rel=1e-12)
        assert member["nr"]["removed_sum"] == pytest.approx(
            before["nr"][clear].astype(np.float64).sum(), rel=1e-12)
        cells = receipt["members"][index]["cells"]
        assert cells["cleared"] == int(np.count_nonzero(
            before["qr"][clear]))
        assert cells["changed_without_coverage"] == 0


@pytest.mark.gpu
def test_the_filters_increment_survives_wherever_the_stage_did_not_act(
        world, grid):
    """With the filter analysing the Thompson set, on and off differ only in
    observed clear air, and there the analysis is exactly zero."""
    fields = ("u", "v", "thp", "qv", "qr", "nr", "qs", "qg")
    base = dict(analysis_fields=fields, positivity_policy="clip",
                mp_physics=8)
    off, off_prov = assimilate_radar_grid(
        world.checkpoints, world.obs_path, grid, _config(**base))
    on, on_prov = assimilate_radar_grid(
        world.checkpoints, world.obs_path, grid,
        _config(precip_analysis="clear", **base))
    assert "precip_analysis" not in off_prov
    assert on_prov["precip_analysis"]["filter_analysed_fields"] == list(PRECIP)
    clear = _mask(CLEAR)
    analysed = _apply(world, on, "filter-on")
    for index in range(MEMBERS):
        for name in fields:
            moved = on[index][name] != off[index][name]
            if name not in PRECIP:
                assert not moved.any(), name
                continue
            assert not (moved & ~clear).any(), name
            assert np.all(analysed[index][name][clear] == 0.0), name


def _trim_build(member, reflectivity, *, scope="covered", mode="trim-build"):
    """The module rule on one member's background, on the card."""
    import cupy as cp

    fields = {name: cp.asarray(member[name]) for name in PRECIP}
    pressure = cp.asarray(member["p"])
    temperature = ha.temperature_from_density(
        pressure, cp.asarray(member["alt"]), cp.asarray(member["qv"]))
    cfg = ha.HydrometeorAnalysisConfig(mode=mode, scope=scope)
    if mode == "clear":
        return ha.hydrometeor_analysis(fields, cp.asarray(reflectivity), cfg)
    return ha.hydrometeor_analysis(fields, cp.asarray(reflectivity), cfg,
                                   temperature=temperature,
                                   pressure=pressure)


def _world_reflectivity(world, grid):
    import cupy as cp

    from gpuwm.da.obs_radar import read_document

    document = read_document(world.obs_path, expected_grid=grid,
                             expected_grid_identity=grid.identity_sha256())
    reflectivity, _ = ha.radar_grid_reflectivity(document)
    return cp.asnumpy(reflectivity)


@pytest.mark.gpu
def test_trim_build_keeps_pairs_and_coverage_on_the_planted_world(world,
                                                                   grid):
    from gpuwm.da.moments import moment_consistency_report

    reflectivity = _world_reflectivity(world, grid)
    clear, echo = _mask(CLEAR), _mask(ECHO)
    covered = clear | echo
    for index in range(MEMBERS):
        before = world.backgrounds[index]
        result = _trim_build(before, reflectivity)
        after = {name: value.get() for name, value in result.analysed.items()}
        columns = result.receipt["columns"]
        # 16 echo columns (warm first level: trim or build), 32 clear ones.
        assert (columns["warm_trimmed"]
                + columns["warm_built_at_strongest_echo"]) == 16, columns
        assert columns["no_echo_cleared_per_level"] == 32, columns
        for name in PRECIP:
            assert np.count_nonzero(after[name][clear]) == 0, name
            assert np.array_equal(after[name][~covered].view(np.int32),
                                  before[name][~covered].view(np.int32))
            assert np.all(after[name] >= 0.0)
        # Under echo the background's rain is trimmed toward the retrieval
        # or kept, never raised above it by more than the 3 g/kg build cap.
        assert np.all(after["qr"][echo] <= max(
            float(before["qr"][echo].max()), 3.0e-3) + 1e-9)
        report = moment_consistency_report({**before, **after},
                                           mp_physics=8)
        assert report["consistent"], report


def _probe_column():
    """The review's column: 5 g/kg of rain below the beam, 0.5 g/kg under
    observed 20 to 40 dBZ echo, observed clear air above."""
    nz, ny, nx = 10, 3, 3
    shape = (nz, ny, nx)
    ref = np.full(shape, -99999.0)
    ref[:, 1, 1] = [-99999.0, -99999.0, 30.0, 40.0, 35.0, 30.0, 20.0,
                    -99.0, -99.0, -99.0]
    t = np.empty(shape, np.float32)
    p = np.empty(shape, np.float32)
    for k in range(nz):
        t[k] = 300.0 - 3.0 * k
        p[k] = 95000.0 - 5000.0 * k
    member = {name: np.zeros(shape, np.float32) for name in PRECIP}
    member["qr"][0:2, 1, 1] = 5.0e-3
    member["qr"][2:7, 1, 1] = 0.5e-3
    member["nr"][:, 1, 1] = np.where(member["qr"][:, 1, 1] > 0, 1.0e4, 0.0)
    return member, ref, t, p


@pytest.mark.gpu
def test_covered_trim_takes_its_ratio_from_sampled_levels_only():
    import cupy as cp

    member, ref, t, p = _probe_column()
    rows = {}
    for scope in ("covered", "noaa"):
        cfg = ha.HydrometeorAnalysisConfig(mode="trim-build", scope=scope)
        result = ha.hydrometeor_analysis(
            {name: cp.asarray(member[name]) for name in PRECIP},
            cp.asarray(ref), cfg, temperature=cp.asarray(t),
            pressure=cp.asarray(p))
        rows[scope] = (cp.asnumpy(result.analysed["qr"])[:, 1, 1],
                       result.receipt)
    retrieved, _, _ = ha.thompson_retrieval(cp.asarray(t), cp.asarray(p),
                                            cp.asarray(ref))
    strongest = float(cp.asnumpy(retrieved)[3, 1, 1]) * 1.0e-3
    assert strongest == pytest.approx(0.9135e-3, abs=1e-7)

    qr, receipt = rows["covered"]
    # Below the beam: kept as given.
    assert np.array_equal(qr[0:2], member["qr"][0:2, 1, 1])
    # Sampled levels: the covered background (0.5 g/kg) is under the
    # retrieval, so nothing is trimmed and the retrieval is built at the
    # strongest echo; the other sampled levels keep their rain.
    assert receipt["columns"]["warm_built_at_strongest_echo"] == 1
    assert receipt["columns"]["warm_trimmed"] == 0
    assert qr[3] == pytest.approx(strongest, rel=1e-6)
    for k in (2, 4, 5, 6):
        assert qr[k] == member["qr"][k, 1, 1], k
    assert receipt["columns"]["background_maximum_on_unsampled_level"] == 1
    assert "sampled" in receipt["trim_background_maximum"]
    assert receipt["cells"]["kept_without_coverage"] == 0

    # NOAA's every-level maximum, unchanged: the whole column is scaled by
    # retrieval / 5 g/kg, the levels below the beam included.
    qr, receipt = rows["noaa"]
    ratio = strongest / 5.0e-3
    assert receipt["columns"]["warm_trimmed"] == 1
    assert qr[0] == pytest.approx(5.0e-3 * ratio, rel=1e-6)
    assert qr[3] == pytest.approx(0.5e-3 * ratio, rel=1e-6)
    assert receipt["columns"]["background_maximum_on_unsampled_level"] == 1


@pytest.mark.gpu
def test_trim_build_on_every_member_collapses_rain_spread(world, grid):
    """The measurement behind PRECIP_ANALYSIS_MEMBER_REFUSAL.

    Six members whose rain under the observed echo differs by up to 50 per
    cent.  ``clear`` leaves that spread exactly as it was; the trim and
    build rule, run on every member, scales each column to one retrieval
    maximum and leaves almost none."""
    reflectivity = _world_reflectivity(world, grid)
    echo = _mask(ECHO)
    out = {}
    for mode in ("clear", "trim-build"):
        rain = []
        for index in range(MEMBERS):
            result = _trim_build(world.backgrounds[index], reflectivity,
                                 mode=mode)
            rain.append(result.analysed["qr"].get()[echo].astype(np.float64))
        out[mode] = np.stack(rain)
    before = np.stack([world.backgrounds[index]["qr"][echo]
                       .astype(np.float64) for index in range(MEMBERS)])
    spread_before = float(before.std(axis=0, ddof=1).mean())
    spread_clear = float(out["clear"].std(axis=0, ddof=1).mean())
    spread_trim = float(out["trim-build"].std(axis=0, ddof=1).mean())
    print(f"rain spread under echo (kg/kg): before {spread_before:.6e}, "
          f"clear {spread_clear:.6e}, trim-build {spread_trim:.6e}, "
          f"ratio {spread_trim / spread_before:.4f}")
    assert spread_clear == spread_before
    assert spread_trim < 0.05 * spread_before


@pytest.mark.gpu
def test_clear_kernel_is_exact_and_deterministic():
    import cupy as cp

    rng = np.random.default_rng(7)
    shape = (5, 9, 11)
    ref = rng.choice([45.0, 12.0, 0.0, -5.0, -99.0, -99999.0, -100.0],
                     shape)
    field = rng.uniform(-1e-4, 3e-3, shape).astype(np.float32)
    cfg = ha.HydrometeorAnalysisConfig(mode="clear")
    first = ha.hydrometeor_analysis({"qr": cp.asarray(field)},
                                    cp.asarray(ref), cfg)
    second = ha.hydrometeor_analysis({"qr": cp.asarray(field)},
                                     cp.asarray(ref), cfg)
    out = cp.asnumpy(first.analysed["qr"])
    assert np.array_equal(out.view(np.int32),
                          cp.asnumpy(second.analysed["qr"]).view(np.int32))
    no_echo = (ref <= 0.0) & (ref > -100.0)
    assert np.all(out[no_echo] == 0.0)
    echo = ref > 0.0
    assert np.array_equal(out[echo], np.maximum(field[echo], 0.0))
    uncovered = ref <= -100.0
    assert np.array_equal(out[uncovered].view(np.int32),
                          field[uncovered].view(np.int32))
    increments = cp.asnumpy(first.increments["qr"])
    assert np.all(increments[uncovered] == 0.0)


@pytest.mark.gpu
def test_device_temperatures_match_a_float64_check_to_one_unit():
    import cupy as cp

    from gpuwm.core import constants as c

    rng = np.random.default_rng(11)
    shape = (4, 6, 7)
    thp = rng.normal(0.0, 3.0, shape).astype(np.float32)
    thb = np.linspace(295.0, 340.0, shape[0]).astype(np.float32)
    p = rng.uniform(2.0e4, 1.0e5, shape).astype(np.float32)
    qv = rng.uniform(0.0, 0.02, shape).astype(np.float32)
    alt = rng.uniform(0.8, 2.5, shape).astype(np.float32)
    from_theta = cp.asnumpy(ha.temperature_from_theta(
        cp.asarray(thp), cp.asarray(thb), cp.asarray(p)))
    check = ((thb[:, None, None].astype(np.float64) + thp)
             * (p.astype(np.float64) / c.P0) ** c.RCP).astype(np.float32)
    assert np.max(np.abs(from_theta.view(np.int32).astype(np.int64)
                         - check.view(np.int32).astype(np.int64))) <= 1
    from_density = cp.asnumpy(ha.temperature_from_density(
        cp.asarray(p), cp.asarray(alt), cp.asarray(qv)))
    check = (p.astype(np.float64) * alt / (
        c.RD * (1.0 + c.RVOVRD * qv.astype(np.float64)))).astype(np.float32)
    assert np.max(np.abs(from_density.view(np.int32).astype(np.int64)
                         - check.view(np.int32).astype(np.int64))) <= 1
