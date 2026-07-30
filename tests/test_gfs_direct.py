from __future__ import annotations

import dataclasses
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pytest

from gpuwm.gfs_direct import (
    INPUT_MANIFEST_SCHEMA,
    _PRESSURE_LEVELS_HPA,
    _geometry_contract,
    _load_bridge_snapshots,
    _read_series,
    _source_coverage_receipt,
    _validate_grid_and_vertical_contract,
    _verify_static_receipt,
    _verify_input_manifest,
)
from gpuwm.experiment import VerticalConfig, load_experiment
from gpuwm.physics_compat import WSM6_PROFILE_ID
from gpuwm.ingest.grib import Era5Snapshot


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gfs_series_requires_f000_and_uniform_cadence(tmp_path):
    for hour in (0, 3, 6):
        (tmp_path / f"f{hour:03}.grib2").write_bytes(b"GRIB")
    series = tmp_path / "series.tsv"
    series.write_text(
        "0\tf000.grib2\n3\tf003.grib2\n6\tf006.grib2\n",
        encoding="utf-8")
    assert [hour for hour, _ in _read_series(series)] == [0, 3, 6]
    series.write_text("0\tf000.grib2\n3\tf003.grib2\n7\tf006.grib2\n")
    with pytest.raises(ValueError, match="uniform"):
        _read_series(series)

    series.write_text("0\tf000.grib2\n385\tf003.grib2\n")
    with pytest.raises(ValueError, match="f384"):
        _read_series(series)

    series.write_text("0\tf000.grib2\n6\tf003.grib2\n")
    with pytest.raises(ValueError, match="exactly 1 or 3 hours"):
        _read_series(series)


def test_gfs_manifest_binds_all_dynamic_grib_roles(tmp_path):
    paths = {}
    for role in (
            "series", "bridge", "static_receipt", "grib-f000", "grib-f003"):
        paths[role] = tmp_path / role
        paths[role].write_text(role, encoding="utf-8")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "schema": INPUT_MANIFEST_SCHEMA,
        "files": {
            role: {"name": path.name, "sha256": _digest(path)}
            for role, path in paths.items()
        },
    }), encoding="utf-8")
    _verify_input_manifest(manifest, _digest(manifest), paths)
    paths["grib-f003"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="grib-f003"):
        _verify_input_manifest(manifest, _digest(manifest), paths)


def test_bridge_loader_keeps_antimeridian_crop_axis_continuous(tmp_path):
    """Worldwide lane: a crop crossing 180E keeps a continuous,
    uniform ascending axis (extending past 180) instead of the
    retired rotate-and-argsort, whose non-uniform axis the
    horizontal interpolator refuses.  Fields stay unpermuted."""
    root = tmp_path / "decoded"
    root.mkdir()
    gate_text = (
        "status\tPASS\n"
        "schema\tgpuwm-gfs-grib2-bridge-v1\n"
        "cycle\t2026-07-20 00:00:00\n"
        "forecast_hours\t0,3\n"
        "pressure_levels_pa\t" + ",".join(
            str(int(level * 100)) for level in _PRESSURE_LEVELS_HPA) + "\n"
        "nx\t4\nny\t2\nlat1\t20\nlon1\t179.75\ndx\t0.25\ndy\t0.25\n"
        "lat2\t20.25\nlon2\t180.5\nnum_data_points\t8\n"
        "scan_mode\t0x40\n"
        "originating_center\t7\nmaster_table_version\t2\n"
        "local_table_version\t1\nforecast_generating_process_ids\t0:81,3:96\n"
        "land_mask_parameter\t0\n"
        "invariant_fields\tSOURCE_OROGRAPHY,LANDSEA\n"
        "invariant_fingerprint_fnv1a64\tSOURCE_OROGRAPHY:abc,LANDSEA:def\n")
    for hour in (0, 3):
        time_root = root / f"f{hour:03}"
        time_root.mkdir()
        base_2d = np.tile(np.arange(4, dtype="<f4"), (2, 1))
        for name in ("GHT", "T", "RH", "U", "V"):
            np.tile(base_2d, (21, 1, 1)).tofile(time_root / f"{name}.f32le")
        for name in (
            "PSFC", "SOURCE_OROGRAPHY", "SKINTEMP", "SNOW", "SNOWH",
            "T2", "RH2", "U10", "V10", "LANDSEA", "XICE",
            "GFS_ST000010", "GFS_ST010040", "GFS_ST040100",
            "GFS_ST100200", "GFS_SM000010", "GFS_SM010040",
            "GFS_SM040100", "GFS_SM100200",
        ):
            base_2d.tofile(time_root / f"{name}.f32le")
    dummy = tmp_path / "dummy.grib2"
    dummy.write_bytes(b"GRIB")
    records = ((0, dummy), (3, dummy))
    inventory = root / "inventory.tsv"
    inventory.write_text("test inventory\n", encoding="utf-8")
    decoded_manifest = root / "decoded-sha256.tsv"
    decoded_lines = ["hour\tvariable\tbytes\tsha256\tfilename"]
    names = (
        "GHT", "T", "RH", "U", "V",
        "PSFC", "SOURCE_OROGRAPHY", "SKINTEMP", "SNOW", "SNOWH",
        "T2", "RH2", "U10", "V10", "LANDSEA", "XICE",
        "GFS_ST000010", "GFS_ST010040", "GFS_ST040100",
        "GFS_ST100200", "GFS_SM000010", "GFS_SM010040",
        "GFS_SM040100", "GFS_SM100200",
    )
    for hour in (0, 3):
        for name in names:
            relative = f"f{hour:03d}/{name}.f32le"
            path = root / relative
            decoded_lines.append(
                f"{hour}\t{name}\t{path.stat().st_size}\t{_digest(path)}\t{relative}")
    decoded_manifest.write_text("\n".join(decoded_lines) + "\n", encoding="utf-8")
    source_digest = _digest(dummy)
    (root / "gate.tsv").write_text(
        gate_text
        + f"source_sha256\t0:{source_digest},3:{source_digest}\n"
        + f"inventory_sha256\t{_digest(inventory)}\n"
        + f"decoded_manifest_sha256\t{_digest(decoded_manifest)}\n",
        encoding="utf-8")
    snapshots = _load_bridge_snapshots(
        root, datetime(2026, 7, 20), records)
    np.testing.assert_array_equal(
        snapshots[0].longitude, [179.75, 180.0, 180.25, 180.5])
    np.testing.assert_array_equal(
        snapshots[0].fields["PSFC"][0], [0.0, 1.0, 2.0, 3.0])
    assert np.all(np.diff(snapshots[0].longitude) == 0.25)


def _matching_wps() -> str:
    return (
        "&share\n max_dom = 1,\n/\n"
        "&geogrid\n"
        " map_proj = 'lambert',\n"
        " e_we = 251,\n e_sn = 201,\n"
        " dx = 12000.0,\n dy = 12000.0,\n"
        " ref_lat = 39.6848,\n ref_lon = -83.9297,\n"
        " truelat1 = 30.0,\n truelat2 = 60.0,\n"
        " stand_lon = -83.9297,\n/\n"
    )


def test_gfs_geometry_and_static_receipt_fail_closed(tmp_path):
    config = Path(__file__).parents[1] / "configs" / "gfs_wrf_direct_proof.toml"
    exp = load_experiment(config)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_matching_wps(), encoding="utf-8")
    grid = _validate_grid_and_vertical_contract(exp, wps)
    static = tmp_path / "static.npz"
    static.write_bytes(b"domain-specific-static")
    receipt = tmp_path / "static-receipt.json"
    receipt.write_text(json.dumps({
        "schema": "gpuwm-native-static-direct-v1",
        "status": "PASS",
        "geometry": _geometry_contract(grid, exp.root.run),
        "cache": {
            "path": static.name,
            "bytes": static.stat().st_size,
            "sha256": _digest(static),
        },
    }), encoding="utf-8")
    _verify_static_receipt(receipt, static, grid, exp.root.run)

    wps.write_text(_matching_wps().replace(
        "ref_lon = -83.9297", "ref_lon = -83.0"), encoding="utf-8")
    with pytest.raises(ValueError, match="geometry mismatch"):
        _validate_grid_and_vertical_contract(exp, wps)

    static.write_bytes(b"same-shape-wrong-domain")
    with pytest.raises(ValueError, match="does not bind"):
        _verify_static_receipt(receipt, static, grid, exp.root.run)


@pytest.mark.parametrize("nz", [7, 49, 80, 113])
def test_gfs_vertical_contract_accepts_any_source_covered_explicit_eta(
        tmp_path, nz):
    config = Path(__file__).parents[1] / "configs" / "gfs_wrf_direct_proof.toml"
    baseline = load_experiment(config)
    run = dataclasses.replace(baseline.root.run, nz=nz)
    domain = dataclasses.replace(baseline.root, run=run)
    vertical = VerticalConfig(
        eta_levels=tuple(float(value)
                         for value in np.linspace(1.0, 0.0, nz + 1)),
        p_top=10_000.0,
        hybrid_opt=2,
        etac=0.37,
    )
    exp = dataclasses.replace(
        baseline, domains=(domain,), vertical=vertical)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_matching_wps(), encoding="utf-8")
    _validate_grid_and_vertical_contract(exp, wps)


def test_gfs_vertical_contract_refuses_model_top_above_source(tmp_path):
    config = Path(__file__).parents[1] / "configs" / "gfs_wrf_direct_proof.toml"
    baseline = load_experiment(config)
    vertical = dataclasses.replace(baseline.vertical, p_top=5_000.0)
    exp = dataclasses.replace(baseline, vertical=vertical)
    wps = tmp_path / "namelist.wps"
    wps.write_text(_matching_wps(), encoding="utf-8")
    with pytest.raises(ValueError, match="source atmosphere stops"):
        _validate_grid_and_vertical_contract(exp, wps)


class _CoverageGrid:
    def __init__(self, mass, u=None, v=None):
        self._mass = mass
        self._u = mass if u is None else u
        self._v = mass if v is None else v

    def latlon_mass(self):
        return self._mass

    def latlon_u(self):
        return self._u

    def latlon_v(self):
        return self._v


def test_gfs_source_coverage_requires_unclipped_parabolic_and_masked_halos():
    axis = np.arange(30, dtype=np.float64)
    longitude, latitude = np.meshgrid([12.0, 13.0], [12.0, 13.0])
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 7, 20),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=axis,
        longitude=axis,
        fields={
            "LANDSEA": np.zeros((30, 30), dtype=np.float64),
            "SKINTEMP": np.full((30, 30), 290.0, dtype=np.float64),
        },
    )
    grid = _CoverageGrid((latitude, longitude))
    lake = np.array([[True, False], [False, False]])
    receipt = _source_coverage_receipt(snapshot, grid, lake)
    # v2 masked coverage: deterministic-stencil box (floor-based [-1, +2]),
    # search reach declared crop-wide, class support reported.
    assert receipt["masked_surface_donor_x"] == [11, 15]
    assert receipt["masked_deterministic_donor_offsets"] == [-1, 2]
    assert "masked_search_radius" not in receipt
    assert receipt["masked_class_support"] == {"land": 0, "water": 900}
    assert receipt["lake_cells"] == 1

    edge_longitude = longitude.copy()
    edge_longitude[:, 0] = 0.25
    edge_grid = _CoverageGrid((latitude, longitude), u=(latitude, edge_longitude))
    with pytest.raises(ValueError, match="parabolic donor halo for u"):
        _source_coverage_receipt(snapshot, edge_grid, lake)

    shallow_longitude = longitude - 11.5
    shallow_grid = _CoverageGrid((latitude, shallow_longitude))
    with pytest.raises(ValueError, match="parabolic donor halo"):
        _source_coverage_receipt(snapshot, shallow_grid, lake)


def test_gfs_lake_coverage_proves_a_global_in_crop_water_donor():
    axis = np.arange(60, dtype=np.float64)
    source_land = np.ones((60, 60), dtype=np.float64)
    source_land[30, 50] = 0.0
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 7, 20),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=axis,
        longitude=axis,
        fields={
            "LANDSEA": source_land,
            "SKINTEMP": np.full((60, 60), 290.0, dtype=np.float64),
        },
    )
    point = (np.array([[30.0]]), np.array([[30.0]]))
    receipt = _source_coverage_receipt(
        snapshot, _CoverageGrid(point), np.array([[True]]))
    assert receipt["lake_source_search"] == "global expanding nearest-water"
    assert receipt["max_lake_source_search_radius_cells"] == 32.0
    assert receipt["max_lake_source_water_distance_cells"] == 20.0


def test_wizard_suggested_margin_passes_the_lake_donor_proof():
    """Ties the wizard's suggested fetch margin to THIS module's
    donor-coverage function: a crop carrying the documented margin
    passes whenever the nearest source water lies within the
    GFS_LAKE_DONOR_MARGIN_DEG allowance, and still fails closed when
    the crop holds no water at all."""
    from gpuwm.fetch import (GFS_SOURCE_RESOLUTION_DEG,
                             gfs_suggested_fetch_margin_deg)

    margin_cells = int(round(gfs_suggested_fetch_margin_deg()
                             / GFS_SOURCE_RESOLUTION_DEG))
    n = 2 * margin_cells + 1
    axis = np.arange(n, dtype=np.float64)
    center = margin_cells
    land = np.ones((n, n), dtype=np.float64)
    # Worst allowed donor: water at exactly the margin allowance -- the
    # crop edge (margin_cells + 1 half-cell steps away) is still
    # provably farther, so the proof passes.
    land[center, n - 1] = 0.0
    snapshot = Era5Snapshot(
        valid_time=datetime(2026, 7, 29),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=axis,
        longitude=axis,
        fields={
            "LANDSEA": land,
            "SKINTEMP": np.full((n, n), 290.0, dtype=np.float64),
        },
    )
    point = (np.array([[float(center)]]), np.array([[float(center)]]))
    receipt = _source_coverage_receipt(
        snapshot, _CoverageGrid(point), np.array([[True]]))
    assert (receipt["max_lake_source_water_distance_cells"]
            == float(margin_cells))

    # No water anywhere in the crop: the same function fails closed.
    dry = Era5Snapshot(
        valid_time=datetime(2026, 7, 29),
        levels_hpa=np.array([1000.0], dtype=np.float64),
        latitude=axis,
        longitude=axis,
        fields={
            "LANDSEA": np.ones((n, n), dtype=np.float64),
            "SKINTEMP": np.full((n, n), 290.0, dtype=np.float64),
        },
    )
    with pytest.raises(ValueError, match="no finite source-water"):
        _source_coverage_receipt(
            dry, _CoverageGrid(point), np.array([[True]]))


def test_git_identity_degrades_honestly_outside_a_checkout(monkeypatch):
    """An installed wheel is neither a git checkout nor a sealed
    runtime: provenance reports unavailable instead of demanding
    GPUWM_NATIVE_DISTRIBUTION_MANIFEST or raising."""
    import subprocess as subprocess_module

    import gpuwm.gfs_direct as gfs_direct

    monkeypatch.delenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST",
                       raising=False)

    def fail(*arguments, **keywords):
        raise subprocess_module.CalledProcessError(128, "git")

    monkeypatch.setattr(gfs_direct.subprocess, "run", fail)
    identity = gfs_direct._git_source_identity()
    assert identity["available"] is False
    assert identity["identity_source"] == "unavailable-installed-runtime"


def test_implementation_hashes_no_longer_demand_the_sealed_manifest(
        monkeypatch, tmp_path):
    """Wheel installs miss the repo-only paths (tools/, Rust sources);
    the inventory records them honestly without requiring the sealed
    archive's distribution manifest."""
    import gpuwm.gfs_direct as gfs_direct

    monkeypatch.delenv("GPUWM_NATIVE_DISTRIBUTION_MANIFEST",
                       raising=False)
    monkeypatch.setattr(
        gfs_direct, "_IMPLEMENTATION_PATHS",
        ("gpuwm/gfs_direct.py", "tools/absent-in-wheel-installs.rs"))
    hashes = gfs_direct._implementation_sha256()
    assert "gpuwm/gfs_direct.py" in hashes
    assert "distribution/repo_only_paths.json" in hashes
    assert "distribution/manifest.json" not in hashes


def test_output_root_parent_is_created_instead_of_tracebacking(tmp_path):
    """PP-10: an absent --output-root parent was a bare traceback.

    The bridge stages its scratch space in the PARENT of --output-root so
    the result can be renamed into place on one filesystem.  An absent
    parent surfaced as `FileNotFoundError: .../gpuwm-gfs-bridge-4g0c7np7`
    about 40 s into the work -- after --dry-run had passed cleanly -- in
    a product that is otherwise exemplary about naming the remedy.
    """
    from gpuwm.gfs_direct import _prepare_output_root_parent

    output_root = tmp_path / "out" / "miami-init"
    assert not output_root.parent.exists()
    parent = _prepare_output_root_parent(output_root)
    assert parent == output_root.parent
    assert parent.is_dir()
    # The output root itself stays create-only: only its parent is made.
    assert not output_root.exists()
    # Idempotent.
    assert _prepare_output_root_parent(output_root) == parent

    # A parent path that is a FILE is a refusal with a stated remedy,
    # not a traceback from tempfile.
    blocked = tmp_path / "afile" / "init"
    blocked.parent.write_text("not a directory", encoding="utf-8")
    with pytest.raises(NotADirectoryError, match="parent"):
        _prepare_output_root_parent(blocked)


def test_the_front_door_prints_the_forecast_command_it_already_knows(tmp_path):
    """B-2: the runner needed six hashes the user extracted by hand.

    `gpuwm fetch --author-front-door-manifest` prints the complete
    rw-wps line with its digest filled in and is the most-praised thing
    in both pilots.  The front door printed nothing at all -- it dumped
    42 KB of proof.json and stopped -- so every user grepped
    `content_sha256` out of the JSON themselves.  Every value below was
    already in this process.
    """
    import json
    from gpuwm.gfs_direct import prepared_forecast_next_command

    root = tmp_path / "miami-init"
    root.mkdir()
    proof = {
        "input_manifest_sha256": "a" * 64,
        "prepared_cache": {"content_sha256": "b" * 64},
    }
    (root / "proof.json").write_text(json.dumps(proof), encoding="utf-8")
    proof_digest = hashlib.sha256(
        (root / "proof.json").read_bytes()).hexdigest()

    config = _wizard_single_domain_config(tmp_path)
    namelist = tmp_path / "x.namelist.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")

    lines = prepared_forecast_next_command(
        proof, output_root=root, experiment_config=config,
        wps_namelist=namelist)
    text = "\n".join(lines)
    assert "tools/prepared_single_domain_forecast.py" in text
    assert f"--proof-sha256 {proof_digest}" in text
    assert f"--source-manifest-sha256 {'a' * 64}" in text
    assert f"--prepared-content-sha256 {'b' * 64}" in text
    assert "--run-seconds" not in text.split("(")[0]  # no longer required
    assert "default to the hash-bound experiment" in text
    # The two values v1.0.0 left the user to work out are filled in.
    assert f"--physics-profile {WSM6_PROFILE_ID}" in text
    assert f"--outdir {root / 'forecast'}" in text
    _assert_pasteable(lines)

    # A multi-domain hierarchy proof has no prepared-cache identity, so
    # it names the OTHER runner -- and prints that runner's WHOLE
    # command rather than a fragment: it binds the experiment config by
    # digest too, and this process can read it.
    hierarchy = prepared_forecast_next_command(
        {"schema": "gpuwm-gfs-native-hierarchy-proof-v1"},
        output_root=root, experiment_config=config, wps_namelist=namelist)
    hierarchy_text = "\n".join(hierarchy)
    assert "prepared_domain_tree_forecast.py" in hierarchy_text
    assert f"--preparation-receipt-sha256 {proof_digest}" in hierarchy_text
    assert f"--experiment-config-sha256 {_digest(config)}" in hierarchy_text
    assert f"--outdir {root / 'forecast'}" in hierarchy_text
    _assert_pasteable(hierarchy)


#: A bare ALL-CAPS token is how v1.0.0 spelled "you work this one out"
#: (`--outdir OUTPUT_DIR`, `--geog-root WPS_GEOG_DIR`).
_PLACEHOLDER_WORD = re.compile(r"^[A-Z][A-Z0-9_]{2,}$")


def _assert_pasteable(lines) -> None:
    """No printed command line may carry an unresolved placeholder.

    Deliberately mechanical: a command line is one that starts a shell
    invocation or continues one, and on such a line an angle bracket or
    a bare ALL-CAPS token means the user must edit before pasting --
    which is the whole failure.  Parenthesised prose is exempt; it is
    not a command and never claimed to be.
    """

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("("):
            continue
        looks_like_command = (
            stripped.startswith(("python ", "rw-wps ", "gpuwm ", "cargo ",
                                 "git ", "bash ", "--"))
            or stripped.endswith("\\"))
        if not looks_like_command:
            continue
        assert "<" not in stripped and ">" not in stripped, (
            f"unresolved placeholder in a printed command: {line!r}")
        for token in stripped.rstrip("\\").split():
            assert not _PLACEHOLDER_WORD.match(token), (
                f"unresolved ALL-CAPS placeholder {token!r} in: {line!r}")


def _wizard_single_domain_config(tmp_path):
    """A config the prepared-forecast runner would actually accept.

    Emitted by the wizard rather than hand-written, because the point of
    resolving the profile is that both sides read the same table.
    """

    from gpuwm.cli import main as cli_main

    out = tmp_path / "wizard" / "case.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert cli_main(["domain", "--point=39.7,-96.6", "--card", "24gb",
                     "--ladder", "12", "--source", "gfs",
                     "--physics-profile", WSM6_PROFILE_ID,
                     "--cycle", "2026-07-29T18", "--out", str(out)]) == 0
    return out


def test_a_config_bound_to_no_shipped_profile_prints_prose_not_a_command(
        tmp_path):
    """The honest gap: say so; never print a command that cannot run."""

    import json
    from gpuwm.gfs_direct import prepared_forecast_next_command

    root = tmp_path / "init"
    root.mkdir()
    proof = {"input_manifest_sha256": "a" * 64,
             "prepared_cache": {"content_sha256": "b" * 64}}
    (root / "proof.json").write_text(json.dumps(proof), encoding="utf-8")

    # A shipped proof descriptor: single-domain and valid, but its
    # physics predate the profiles and match none of them.
    lines = prepared_forecast_next_command(
        proof, output_root=root,
        experiment_config=Path("configs/gfs_wrf_direct_proof.toml"),
        wps_namelist=Path("configs/gfs_wrf_direct_proof.namelist.wps"))
    text = "\n".join(lines)
    assert "matches none of the profiles" in text
    assert "prepared_single_domain_forecast.py" not in text
    _assert_pasteable(lines)


def test_the_front_door_manifest_line_is_pasteable_too(tmp_path):
    """`gpuwm fetch --author-front-door-manifest` prints a command too.

    It carried `--output-root OUTPUT_DIR` and `--geog-root
    WPS_GEOG_DIR`: the same defect, on the line both pilots praised
    most.
    """

    from gpuwm import fetch

    out = tmp_path / "gfs"
    out.mkdir()
    (out / "gfs-series.tsv").write_text("0\tf000.grib2\n", encoding="utf-8")
    (out / "f000.grib2").write_bytes(b"GRIB")
    (out / fetch.FETCH_MANIFEST_NAME).write_text(json.dumps({
        "schema": fetch.FETCH_MANIFEST_SCHEMA,
        "source": "gfs", "cycle": "2026-07-29T18:00:00Z",
        "forecast_hours": [0],
        "files": [{"name": "f000.grib2", "role": "gfs-subset",
                   "forecast_hour": 0}],
    }), encoding="utf-8")
    bridge = tmp_path / "gfs_grib2_bridge"
    bridge.write_bytes(b"stand-in")
    namelist = tmp_path / "x.namelist.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")
    config = tmp_path / "x.toml"
    config.write_text("[experiment]\n", encoding="utf-8")

    said: list[str] = []
    fetch.author_gfs_front_door_manifest(
        out=out, bridge=bridge, wps_namelist=namelist,
        experiment_config=config, progress=said.append)
    printed = [line for text in said for line in text.splitlines()]
    assert any("rw-wps --source gfs" in line for line in printed)
    _assert_pasteable(printed)
