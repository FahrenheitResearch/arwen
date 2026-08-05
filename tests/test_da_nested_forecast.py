"""CPU contracts for the radar-DA fine nested free-forecast leg.

The GPU half -- the parent-inertness ratchet and the real
parent-state-derived child -- lives in
``tests/test_da_nested_forecast_gpu.py``.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from fractions import Fraction
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

from gpuwm.config import RunConfig
from gpuwm.da import nested_forecast as nf
from gpuwm.experiment import (DomainConfig, ExperimentConfig,
                              ProjectionConfig, VerticalConfig)
from gpuwm.physics_compat import (WSM6_PROFILE_ID,
                                  single_domain_runtime_switches)


def _nowcast_run(**overrides) -> RunConfig:
    """The shipped nowcast profile at the demo's own geometry."""
    switches = dict(single_domain_runtime_switches(WSM6_PROFILE_ID))
    switches.pop("acknowledgements", None)
    base = dict(
        nx=132, ny=132, nz=49, dx=3000.0, dy=3000.0, ztop=20000.0,
        dt=15.0, run_seconds=900.0, output_interval_s=900.0,
        specified=True, nested=False, grid_id=1,
        spec_bdy_width=5, spec_zone=1, relax_zone=4,
        moist=True, hypsometric_opt=2, map_proj=1,
    )
    base.update({key: value for key, value in switches.items()
                 if key in RunConfig.__dataclass_fields__})
    base.update(overrides)
    return RunConfig(**base)


def _nowcast_experiment(run: RunConfig | None = None) -> ExperimentConfig:
    run = _nowcast_run() if run is None else run
    root = DomainConfig(
        grid_id=1, parent_id=0, i_parent_start=1, j_parent_start=1,
        parent_grid_ratio=1, parent_time_step_ratio=1,
        history_interval_s=900.0, run=run, time_step=15)
    return ExperimentConfig(
        name="nowcast", start_time=datetime(2024, 5, 21, 21),
        run_seconds=900.0,
        vertical=VerticalConfig(tuple(np.linspace(1.0, 0.0, run.nz + 1)),
                                5000.0, 2, 0.2),
        projection=ProjectionConfig("lambert", 35.0, -97.0, 30.0, 60.0,
                                    -97.0),
        restart_interval_s=0.0, domains=(root,))


# ---------------------------------------------------------------------------
# the user-facing configuration surface
# ---------------------------------------------------------------------------

def test_child_dx_and_dt_are_derived_from_the_parent_never_typed():
    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    nested = nf.nested_experiment(exp, child)

    # dx: the parent's, divided by the ratio EXACTLY.
    assert Fraction(child.run.dx) == Fraction(exp.root.run.dx) / 3
    assert child.run.dx == 1000.0
    assert nested.dx_exact(2) == Fraction(1000)
    # dt: WRF's CHAINED single-precision division, not the exact rational.
    assert child.run.dt == float(np.float32(15.0) / np.float32(3))
    assert nested.dt_exact(2) == Fraction(5)
    # The child carries no root clock keys.
    assert child.time_step is None


def test_half_width_km_sizes_the_nest_and_snaps_to_whole_parent_cells():
    exp = _nowcast_experiment()
    # 60 km half width at 1 km spacing wants 120 cells; 120 is already a
    # whole number of 3 km parent cells.
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, half_width_km=60.0))
    assert (child.run.nx, child.run.ny) == (120, 120)
    # A half width that does not land on a parent cell is rounded DOWN.
    ragged = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, half_width_km=61.0))
    assert ragged.run.nx % 3 == 0
    assert ragged.run.nx == 120


def test_nest_is_centred_in_the_parent_by_default():
    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    # 126 child cells = 42 parent cells inside a 132-cell parent.
    assert child.i_parent_start == (132 - 42) // 2 + 1 == 46
    assert child.j_parent_start == 46


def test_explicit_placement_is_honoured():
    exp = _nowcast_experiment()
    child = nf.nest_domain_config(exp, nf.NestGeometry(
        ratio=3, nx=126, ny=126, i_parent_start=30, j_parent_start=41))
    assert (child.i_parent_start, child.j_parent_start) == (30, 41)


def test_geometry_refuses_an_ambiguous_or_half_stated_extent():
    with pytest.raises(nf.NestedForecastRefusal, match="exactly once"):
        nf.NestGeometry(ratio=3, nx=126, ny=126, half_width_km=60.0)
    with pytest.raises(nf.NestedForecastRefusal, match="exactly once"):
        nf.NestGeometry(ratio=3)
    with pytest.raises(nf.NestedForecastRefusal, match="together"):
        nf.NestGeometry(ratio=3, nx=126)
    with pytest.raises(nf.NestedForecastRefusal, match="together"):
        nf.NestGeometry(ratio=3, nx=126, ny=126, i_parent_start=30)


def test_ratio_one_nest_is_refused():
    with pytest.raises(nf.NestedForecastRefusal, match="refines nothing"):
        nf.NestGeometry(ratio=1, nx=126, ny=126)


def test_a_nest_too_small_for_its_own_boundary_frame_is_refused():
    exp = _nowcast_experiment()
    with pytest.raises(nf.NestedForecastRefusal, match="relaxation zone"):
        nf.nest_domain_config(exp, nf.NestGeometry(ratio=3, nx=6, ny=6))


# ---------------------------------------------------------------------------
# admissibility is enforced by the validator, not by convention
# ---------------------------------------------------------------------------

def test_cumulus_is_refused_below_the_convection_permitting_spacing():
    """The wizard only ADVISES here; on this route it is a refusal."""
    from gpuwm.domain_wizard import CUMULUS_CONVECTION_PERMITTING_DX_KM

    exp = _nowcast_experiment(_nowcast_run(cu_physics=1, cudt_minutes=5.0))
    # The derived child pins cu_physics=0 regardless of the parent, so the
    # refusal is proved against a child that tries to keep it.
    child_run = _nowcast_run(
        nx=126, ny=126, dx=1000.0, dy=1000.0, dt=5.0, grid_id=2,
        nested=True, specified=False, cu_physics=1, cudt_minutes=5.0)
    assert 1.0 < CUMULUS_CONVECTION_PERMITTING_DX_KM
    with pytest.raises(nf.NestedForecastRefusal,
                       match="convection-permitting"):
        nf.validate_nest_admissibility(child_run, parent_run=exp.root.run)


def test_the_derived_child_pins_cumulus_off_even_under_a_cumulus_parent():
    exp = _nowcast_experiment(_nowcast_run(cu_physics=1, cudt_minutes=5.0))
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    assert child.run.cu_physics == 0
    assert child.run.cudt_minutes == 0.0


def test_pbl_below_the_gray_zone_is_refused_without_an_acknowledgement():
    from gpuwm.domain_wizard import GRAY_ZONE_DX_KM

    exp = _nowcast_experiment()
    # ratio 9 off 3 km puts the child at 333 m, below the gray-zone floor.
    sub_gray = _nowcast_run(
        nx=126, ny=126, dx=1000.0 / 3.0, dy=1000.0 / 3.0, dt=15.0 / 9.0,
        grid_id=2, nested=True, specified=False, cu_physics=0)
    assert sub_gray.dx / 1000.0 < GRAY_ZONE_DX_KM
    assert sub_gray.bl_pbl_physics != 0
    with pytest.raises(nf.NestedForecastRefusal, match="gray-zone"):
        nf.validate_nest_admissibility(sub_gray, parent_run=exp.root.run)
    # Deliberate acknowledgement is the documented escape hatch.
    record = nf.validate_nest_admissibility(
        sub_gray, parent_run=exp.root.run,
        acknowledgements=("nested-forecast:sub-gray-zone-pbl",))
    assert record["acknowledgements"] == ["nested-forecast:sub-gray-zone-pbl"]


def test_exactly_one_km_does_not_trip_the_gray_zone_boundary():
    """GRAY_ZONE_DX_KM is a STRICT floor; 1.000 km is admissible."""
    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    assert child.run.dx == 1000.0
    assert child.run.bl_pbl_physics == 1
    record = nf.validate_nest_admissibility(
        child.run, parent_run=exp.root.run)
    assert record["child_dx_km"] == 1.0
    assert record["pbl_gray_zone_dx_km"] == 1.0


def test_nested_domain_must_carry_spec_exp_zero():
    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    assert child.run.spec_exp == 0.0
    with pytest.raises(nf.NestedForecastRefusal, match="spec_exp"):
        nf.validate_nest_admissibility(
            replace(child.run, spec_exp=0.33), parent_run=exp.root.run)


def test_nested_flags_and_vertical_and_scheme_identity_are_enforced():
    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    with pytest.raises(nf.NestedForecastRefusal, match="nested=True"):
        nf.validate_nest_admissibility(
            replace(child.run, nested=False, specified=True),
            parent_run=exp.root.run)
    with pytest.raises(nf.NestedForecastRefusal, match="[Vv]ertical nesting"):
        nf.validate_nest_admissibility(
            replace(child.run, nz=25), parent_run=exp.root.run)
    with pytest.raises(nf.NestedForecastRefusal, match="mp_physics"):
        nf.validate_nest_admissibility(
            replace(child.run, mp_physics=8), parent_run=exp.root.run)


# ---------------------------------------------------------------------------
# one-way nesting is mandatory
# ---------------------------------------------------------------------------

def test_the_assembled_experiment_is_one_way():
    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    nested = nf.nested_experiment(exp, child)
    assert nested.feedback == 0
    assert nested.smooth_option == 0
    assert len(nested.domains) == 2
    assert nested.domains[0] is exp.root


def test_two_way_feedback_on_the_parent_is_not_inherited():
    exp = replace(_nowcast_experiment(), feedback=1)
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    assert nf.nested_experiment(exp, child).feedback == 0


def test_a_second_nest_is_refused():
    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    nested = nf.nested_experiment(exp, child)
    with pytest.raises(nf.NestedForecastRefusal, match="already carries"):
        nf.nested_experiment(nested, child)


# ---------------------------------------------------------------------------
# nest-down of the surface: WRF's interp_fcni arithmetic
# ---------------------------------------------------------------------------

def _registration(ratio=3, i_start=5, j_start=7, child_nx=9, child_ny=12,
                  parent_nx=20, parent_ny=20):
    exp = _nowcast_experiment(_nowcast_run(nx=parent_nx, ny=parent_ny))
    child_dc = DomainConfig(
        grid_id=2, parent_id=1, i_parent_start=i_start,
        j_parent_start=j_start, parent_grid_ratio=ratio,
        parent_time_step_ratio=ratio, history_interval_s=900.0,
        run=_nowcast_run(nx=child_nx, ny=child_ny, dx=1000.0, dy=1000.0,
                         dt=5.0, grid_id=2, nested=True, specified=False))
    return nf.donor_registration(child_dc, exp.root.run)


def test_donor_nest_down_reproduces_wrf_interp_fcni_arithmetic():
    """child(n) takes parent(ipos + (n-1)//nri), interp_fcn.F."""
    ratio, i_start, j_start = 3, 5, 7
    reg = _registration(ratio=ratio, i_start=i_start, j_start=j_start,
                        child_nx=9, child_ny=12)
    parent = np.arange(20 * 20, dtype=np.float64).reshape(20, 20)
    child = nf.donor_nest_down(parent, reg)
    assert child.shape == (12, 9)
    for nj in range(1, 13):
        cj = j_start + (nj - 1) // ratio          # 1-based donor cell
        for ni in range(1, 10):
            ci = i_start + (ni - 1) // ratio
            assert child[nj - 1, ni - 1] == parent[cj - 1, ci - 1]


def test_donor_nest_down_never_invents_a_value_not_in_the_parent():
    """A category must survive nest-down as a category."""
    reg = _registration()
    categories = np.random.default_rng(0).integers(
        1, 22, size=(20, 20)).astype(np.float32)
    child = nf.donor_nest_down(categories, reg)
    assert set(np.unique(child)).issubset(set(np.unique(categories)))


def test_donor_nest_down_carries_leading_axes_untouched():
    reg = _registration()
    monthly = np.random.default_rng(1).random((12, 20, 20))
    soil = np.random.default_rng(2).random((4, 20, 20))
    assert nf.donor_nest_down(monthly, reg).shape == (12, 12, 9)
    assert nf.donor_nest_down(soil, reg).shape == (4, 12, 9)
    child = nf.donor_nest_down(monthly, reg)
    np.testing.assert_array_equal(
        child[3], nf.donor_nest_down(monthly[3], reg))


def test_donor_nest_down_refuses_a_mismatched_parent_shape():
    reg = _registration()
    with pytest.raises(ValueError, match="horizontal shape"):
        nf.donor_nest_down(np.zeros((19, 20)), reg)


def test_an_undeclared_surface_field_is_a_refusal_not_a_default():
    reg = _registration()
    with pytest.raises(nf.NestedForecastRefusal,
                       match="no declared nest-down operator"):
        nf.nest_down_mapping({"MYSTERY": np.zeros((20, 20))},
                             nf.SURFACE_NEST_DOWN, reg, what="surface")


def test_map_geometry_is_grid_derived_not_inherited():
    reg = _registration()
    out, dropped = nf.nest_down_mapping(
        {"LU_INDEX": np.zeros((20, 20)), "MAPFAC_M": np.ones((20, 20))},
        nf.STATIC_NEST_DOWN, reg, what="static")
    assert "MAPFAC_M" not in out
    assert dropped == ["MAPFAC_M"]


def test_child_land_inventory_keeps_xland_consistent_with_landmask():
    exp = _nowcast_experiment(_nowcast_run(nx=20, ny=20))
    child_dc = DomainConfig(
        grid_id=2, parent_id=1, i_parent_start=5, j_parent_start=5,
        parent_grid_ratio=3, parent_time_step_ratio=3,
        history_interval_s=900.0,
        run=_nowcast_run(nx=9, ny=9, dx=1000.0, dy=1000.0, dt=5.0,
                         grid_id=2, nested=True, specified=False))
    rng = np.random.default_rng(3)
    landmask = (rng.random((20, 20)) > 0.5).astype(np.float32)
    static = {name: np.zeros((20, 20), np.float32)
              for name in ("HGT_M", "LU_INDEX", "SCT_DOM", "SCB_DOM",
                           "SNOALB", "SOILTEMP", "TMN")}
    static["LANDMASK"] = landmask
    static["LANDUSEF"] = np.zeros((21, 20, 20), np.float32)
    static["SOILCTOP"] = np.zeros((16, 20, 20), np.float32)
    static["SOILCBOT"] = np.zeros((16, 20, 20), np.float32)
    for name in ("GREENFRAC", "LAI12M", "ALBEDO12M"):
        static[name] = np.zeros((12, 20, 20), np.float32)
    surface_fields = {
        "TSK": np.full((20, 20), 290.0, np.float32),
        "TSLB": np.full((4, 20, 20), 288.0, np.float32),
        "SMOIS": np.full((4, 20, 20), 0.3, np.float32),
        "SH2O": np.full((4, 20, 20), 0.3, np.float32),
        "TMN": np.full((20, 20), 285.0, np.float32),
        "SEAICE": np.zeros((20, 20), np.float32),
        "SNOW": np.zeros((20, 20), np.float32),
        "SNOWH": np.zeros((20, 20), np.float32),
        "LANDMASK": landmask,
        "XLAND": np.where(landmask >= 0.5, 1.0, 2.0).astype(np.float32),
    }
    inventory = nf.child_land_inventory(
        static, SimpleNamespace(fields=MappingProxyType(surface_fields)),
        child_dc, exp.root.run)
    child_surface = inventory["surface"].fields
    np.testing.assert_array_equal(
        child_surface["XLAND"],
        np.where(child_surface["LANDMASK"] >= 0.5, 1.0, 2.0))
    # Nest-down of a mask must stay a mask.
    assert set(np.unique(child_surface["LANDMASK"])).issubset({0.0, 1.0})
    assert inventory["receipt"]["terrain_policy"] == nf.TERRAIN_POLICY
    assert inventory["receipt"]["land_policy"] == nf.LAND_POLICY


# ---------------------------------------------------------------------------
# receipt and cost model
# ---------------------------------------------------------------------------

def test_the_nest_is_deliberately_cheap_by_default():
    assert nf.DEFAULT_NEST_MEMBERS == 0
    assert nf.NestGeometry(ratio=3, nx=126, ny=126).members == 0


def test_cost_model_prices_the_nest_against_its_parent():
    exp = _nowcast_experiment()
    child = nf.nest_domain_config(
        exp, nf.NestGeometry(ratio=3, nx=126, ny=126))
    cost = nf.nest_cost_model(nf.nested_experiment(exp, child), child)
    assert cost["parent_columns"] == 132 * 132
    assert cost["child_columns"] == 126 * 126
    assert cost["covered_parent_columns"] == 42 * 42
    assert cost["child_substeps_per_parent_step"] == 3
    # 126^2 / 132^2 columns, three substeps each.
    assert cost["dycore_cost_vs_parent"] == pytest.approx(
        (126 * 126) / (132 * 132) * 3)
    assert cost["parent_fraction_covered"] == pytest.approx(
        (42 * 42) / (132 * 132))


def test_receipt_records_that_the_nest_inherited_the_analysis():
    exp = _nowcast_experiment()
    geometry = nf.NestGeometry(ratio=3, nx=126, ny=126, members=2)
    child = nf.nest_domain_config(exp, geometry)
    nested = nf.nested_experiment(exp, child)
    record = nf.nested_forecast_receipt(
        geometry=geometry, exp=nested, child_dc=child,
        admissibility=nf.validate_nest_admissibility(
            child.run, parent_run=exp.root.run),
        land_receipt={"terrain_policy": nf.TERRAIN_POLICY,
                      "land_policy": nf.LAND_POLICY},
        legs=[6, 7, 8], nest_members=2)
    assert record["schema"] == nf.RECEIPT_SCHEMA
    assert record["one_way"] is True
    assert record["feedback"] == 0
    assert record["initialization"]["inherits_assimilated_state"] is True
    assert record["initialization"]["cold_start_from_analysis_file"] is False
    assert record["initialization"]["source"] == "parent-live-state-sint"
    assert record["nest"]["dx_m"] == 1000.0
    assert record["nest"]["dx_exact_m"] == "1000"
    assert record["nest"]["dt_exact_s"] == "5"
    assert record["ensemble"]["nest_members"] == 2
    assert record["legs"] == [6, 7, 8]


# ---------------------------------------------------------------------------
# the pricing surface
# ---------------------------------------------------------------------------

def test_cost_tool_prices_a_nest_and_reports_a_refusal_as_a_row():
    """The DA route runs no VRAM gate; this is the surface that fills it."""
    import json
    import io
    import contextlib

    from tools.da_nest_cost import main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main(["--json", "--half-width-km", "60",
                     "--parent-leg-seconds", "34.5",
                     "--trajectories", "11", "--free-legs", "6"]) == 0
    result = json.loads(buffer.getvalue())
    assert result["basis"].startswith("computed")
    row, = result["rows"]
    assert (row["child_nx"], row["child_ny"]) == (120, 120)
    assert row["child_dx_m"] == 1000.0
    assert row["child_dt_s"] == 5.0
    # A nest costs VRAM and time; both must be positive and finite.
    assert row["vram_delta_mib"] > 0.0
    assert row["projected"]["nest_seconds_added_per_leg"] > 0.0
    assert result["parent"]["vram"]["domains"] == 1
    assert row["vram"]["domains"] == 2


def test_cost_tool_reports_an_inadmissible_nest_instead_of_raising():
    import json
    import io
    import contextlib

    from tools.da_nest_cost import main

    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        assert main(["--json", "--ratio", "9",
                     "--half-width-km", "20"]) == 0
    row, = json.loads(buffer.getvalue())["rows"]
    assert "gray-zone" in row["refused"]
