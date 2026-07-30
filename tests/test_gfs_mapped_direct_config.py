from __future__ import annotations

import json
from pathlib import Path
import re
import tomllib

from gpuwm.experiment import load_experiment
from gpuwm.native_wrf_contract import validate_native_lambert_contracts


ROOT = Path(__file__).parents[1]


def _toml(name: str) -> dict[str, object]:
    with (ROOT / "configs" / name).open("rb") as stream:
        return tomllib.load(stream)


def _namelist_section(text: str, name: str) -> str:
    match = re.search(rf"&{re.escape(name)}\s*(.*?)\n/", text, re.DOTALL)
    assert match is not None
    return match.group(1).strip()


def test_gfs_direct_proof_changes_source_time_not_target_geometry():
    era5 = _toml("era5_wrf_direct_proof.toml")
    gfs = _toml("gfs_wrf_direct_proof.toml")

    assert gfs["projection"] == era5["projection"]
    assert gfs["shared"] == era5["shared"]
    assert gfs["domain"] == era5["domain"]
    assert str(gfs["experiment"]["start_time"]) == "2026-07-20 00:00:00"
    assert gfs["experiment"]["run_seconds"] == 10800.0

    era5_wps = (ROOT / "configs" / "era5_wrf_direct_proof.namelist.wps").read_text()
    gfs_wps = (ROOT / "configs" / "gfs_wrf_direct_proof.namelist.wps").read_text()
    assert _namelist_section(gfs_wps, "geogrid") == _namelist_section(
        era5_wps, "geogrid"
    )
    assert "interval_seconds = 10800" in gfs_wps
    assert "2026-07-20_03:00:00" in gfs_wps


def test_gfs_stock_wrf_namelist_binds_the_native_eta_grid():
    gfs = _toml("gfs_wrf_direct_proof.toml")
    namelist = (
        ROOT / "configs" / "gfs_wrf_direct_proof.namelist.input"
    ).read_text()
    eta_text = re.search(
        r"eta_levels\s*=(.*?)p_top_requested", namelist, re.DOTALL,
    )
    assert eta_text is not None
    eta = [float(value) for value in re.findall(
        r"(?<![A-Za-z_])(?:0|1)\.\d+", eta_text.group(1)
    )]
    assert eta == gfs["shared"]["eta_levels"]
    assert "start_year                          = 2026" in namelist
    assert "interval_seconds                    = 10800" in namelist


def test_gfs_hierarchy_proof_is_a_valid_mapping_bound_two_domain_layout():
    config = ROOT / "configs" / "gfs_wrf_hierarchy_proof.toml"
    wps = ROOT / "configs" / "gfs_wrf_hierarchy_proof.namelist.wps"
    stock = (
        ROOT / "configs" / "gfs_wrf_hierarchy_proof.namelist.input"
    ).read_text()
    mapping = json.loads((
        ROOT / "configs" / "rw-wps-gfs-pressure-grib2.mapping.json"
    ).read_text())
    exp = load_experiment(config)
    grids = validate_native_lambert_contracts(
        exp, wps, source_name="GFS mapped hierarchy proof",
        source_top_pressure_pa=10_000.0,
    )

    assert len(exp.domains) == len(grids) == 2
    assert mapping["target"]["max_dom"] >= len(exp.domains)
    root, child = exp.domains
    assert (root.run.nx, root.run.ny, root.run.dx) == (250, 200, 12_000.0)
    assert (child.run.nx, child.run.ny, child.run.dx) == (240, 160, 3_000.0)
    assert (
        child.parent_id,
        child.i_parent_start,
        child.j_parent_start,
        child.parent_grid_ratio,
    ) == (1, 90, 80, 4)
    assert "max_dom                             = 2" in stock
    assert "e_we                                = 251, 241" in stock
    assert "specified                           = .true., .false." in stock
    assert "nested                              = .false., .true." in stock

    eta_text = re.search(
        r"eta_levels\s*=(.*?)p_top_requested", stock, re.DOTALL,
    )
    assert eta_text is not None
    eta = [float(value) for value in re.findall(
        r"(?<![A-Za-z_])(?:0|1)\.\d+", eta_text.group(1)
    )]
    assert eta == list(exp.vertical.eta_levels)
