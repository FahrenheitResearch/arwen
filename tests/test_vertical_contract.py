from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from gpuwm.vertical_contract import (
    expected_coordinate_shapes,
    explicit_vertical_from_wrf_namelist,
    validate_coordinate_shapes,
    validate_explicit_eta_grid,
)


@pytest.mark.parametrize("nz", [4, 7, 49, 80, 113])
def test_explicit_eta_contract_is_count_agnostic(nz):
    eta = np.linspace(1.0, 0.0, nz + 1)
    actual = validate_explicit_eta_grid(
        eta, nz=nz, p_top=10_000.0, context="test")
    np.testing.assert_array_equal(actual, eta)
    assert actual is not eta


@pytest.mark.parametrize(
    "eta,p_top,match",
    [
        ([1.0, 0.5, 0.0], 10_000.0, "shape"),
        ([1.0, 0.8, np.nan, 0.2, 0.0], 10_000.0, "finite"),
        ([0.9, 0.7, 0.4, 0.2, 0.0], 10_000.0, "exactly"),
        ([1.0, 0.7, 0.7, 0.2, 0.0], 10_000.0, "strictly"),
        ([1.0, 0.7, 0.4, 0.2, 0.0], 0.0, "p_top"),
    ],
)
def test_explicit_eta_contract_rejects_malformed_grids(eta, p_top, match):
    with pytest.raises(ValueError, match=match):
        validate_explicit_eta_grid(
            eta, nz=4, p_top=p_top, context="malformed")


def test_source_top_coverage_fails_closed():
    eta = np.linspace(1.0, 0.0, 9)
    with pytest.raises(ValueError, match="source atmosphere stops"):
        validate_explicit_eta_grid(
            eta, nz=8, p_top=5_000.0, source_top_pressure_pa=10_000.0,
            context="GFS")


@pytest.mark.parametrize("source,requested,top_hpa", [
    ("hrrr-prs", 1500.0, 50),
    ("rap", 1500.0, 50),
    ("icon-eu", 1500.0, 50),
    ("aifs", 1500.0, 50),
    ("20crv3-cf", 5000.0, 100),
    ("era5", 50.0, 1),
    ("gfs", 0.5, 0.01),
])
def test_source_top_refuses_before_inputs_or_memory(
        tmp_path, monkeypatch, source, requested, top_hpa):
    """The real config loaders and check door reject before any paid work."""
    import argparse
    from gpuwm import cli
    from gpuwm.case_data import load_experiment_case_bytes
    from gpuwm.experiment import load_experiment
    from gpuwm.ingest.source_coverage import VerticalLadderRefusal
    from gpuwm.runplan import PLAN_SCHEMA, build_plan, resolve_plan
    from gpuwm.vertical_contract import validate_config_source_top

    recipe = (Path(__file__).resolve().parents[1] / "configs" / "recipes" /
              "conus_hrrr_configuration.toml").read_text()
    text = recipe.replace('source = "hrrr-prs"', f'source = "{source}"').replace(
        "p_top = 5000.0", f"p_top = {requested}")
    config = tmp_path / "experiment.toml"
    config.write_text(text)

    def forbidden(*args, **kwargs):
        pytest.fail("source-top refusal must precede input and memory preflight")

    monkeypatch.setattr(cli, "check_main", forbidden)
    args = argparse.Namespace(config=config, ingest_preflight_handler=forbidden)
    with pytest.raises(VerticalLadderRefusal) as caught:
        cli.combined_check_main(args)
    message = str(caught.value)
    assert f"source '{source}'" in message
    assert f"{top_hpa:g} hPa" in message
    assert f"p_top = {requested:g} Pa" in message
    assert "extrapolation" in message
    with pytest.raises(VerticalLadderRefusal, match="model-top check REFUSED"):
        load_experiment(config)
    with pytest.raises(VerticalLadderRefusal, match="model-top check REFUSED"):
        load_experiment_case_bytes(text.encode(), source=str(config), base_dir=tmp_path)
    plan = build_plan({"schema": PLAN_SCHEMA, "name": "source-top-test",
                       "route": "experiment", "config": {"path": str(config)}},
                      source="test", base_dir=tmp_path, sha256="0" * 64)
    with pytest.raises(VerticalLadderRefusal, match="model-top check REFUSED"):
        resolve_plan(plan, require_inputs=False)
    # Equality is legal, and an extendable source can fetch beyond its
    # certified default ladder without the early check forbidding it.
    assert "PASS" in validate_config_source_top(
        {"fetch": {"source": source}, "shared": {"p_top": top_hpa * 100}},
        context="fixed")
    assert "PASS" in validate_config_source_top(
        {"fetch": {"source": "gfs"}, "shared": {"p_top": 5000}}, context="extended")


def test_coordinate_shape_contract_derives_every_extent():
    shapes = expected_coordinate_shapes(80)
    validate_coordinate_shapes(shapes, nz=80, context="prepared cache")
    bad = dict(shapes)
    bad["znw"] = (50,)
    with pytest.raises(ValueError, match="shape drift"):
        validate_coordinate_shapes(bad, nz=80, context="prepared cache")


@pytest.mark.parametrize("nz", [4, 17, 49, 80, 113])
def test_wrf_namelist_vertical_parser_is_count_agnostic(tmp_path, nz):
    eta = ", ".join(repr(float(value))
                    for value in np.linspace(1.0, 0.0, nz + 1))
    path = tmp_path / "namelist.input"
    path.write_text(
        "&domains\n"
        f" e_vert = {nz + 1}, {nz + 1},\n"
        f" eta_levels = {eta},\n"
        " p_top_requested = 12345, 12345,\n"
        "/\n"
        "&dynamics\n hybrid_opt = 2,\n etac = 0.37,\n/\n",
        encoding="ascii",
    )
    vertical = explicit_vertical_from_wrf_namelist(
        path, expected_nz=nz, context="test")
    assert vertical.mass_level_count == nz
    assert vertical.interface_level_count == nz + 1
    assert vertical.p_top == 12_345.0
    assert vertical.etac == 0.37


def test_wrf_namelist_vertical_parser_rejects_mismatch_and_malformed_eta(
        tmp_path):
    path = tmp_path / "namelist.input"
    path.write_text(
        "&domains\n e_vert = 5, 6,\n"
        " eta_levels = 1, .75, .5, .25, 0,\n"
        " p_top_requested = 10000,\n/\n"
        "&dynamics\n hybrid_opt = 2,\n etac = .2,\n/\n",
        encoding="ascii",
    )
    with pytest.raises(ValueError, match="uniform e_vert"):
        explicit_vertical_from_wrf_namelist(path, context="test")

    malformed = path.read_text().replace(
        "e_vert = 5, 6", "e_vert = 5").replace(".5, .25", ".75, .25")
    path.write_text(malformed, encoding="ascii")
    with pytest.raises(ValueError, match="strictly decreasing"):
        explicit_vertical_from_wrf_namelist(path, context="test")
