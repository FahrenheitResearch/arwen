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
from gpuwm.bridges import decode_failure_message
from gpuwm.experiment import VerticalConfig, load_experiment
from gpuwm.physics_compat import (
    MORRISON_PROFILE_ID,
    NOAHMP_PROFILE_ID,
    WSM6_PROFILE_ID,
    single_domain_runtime_switches,
)
from gpuwm.ingest.grib import Era5Snapshot


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gfs_series_requires_f000_and_uniform_cadence(tmp_path):
    for hour in (0, 3, 6):
        (tmp_path / f"f{hour:03}.grib2").write_bytes(b"GRIB")
    series = tmp_path / "series.tsv"
    series.write_text(
        "0\tf000.grib2\t81\n"
        "3\tf003.grib2\t96\n"
        "6\tf006.grib2\t96\n",
        encoding="utf-8")
    assert [hour for hour, _ in _read_series(series)] == [0, 3, 6]
    series.write_text("0\tf000.grib2\t96\n3\tf003.grib2\t96\n")
    with pytest.raises(ValueError, match="analysis process ID 81"):
        _read_series(series)
    series.write_text("0\tf000.grib2\t81\n3\tf003.grib2\t82\n")
    with pytest.raises(ValueError, match="uncertified forecast process ID 82"):
        _read_series(series)
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
    # The module form, not a script path under tools/: what this line
    # prints has to be runnable by a reader who pip-installed the wheel
    # and has no checkout.  (tools/ still carries a delegating entry
    # point for the spelling older transcripts use.)
    assert "python -m gpuwm.prepared_single_domain_forecast" in text
    assert "tools/prepared_single_domain_forecast.py" not in text
    assert f"--proof-sha256 {proof_digest}" in text
    assert f"--source-manifest-sha256 {'a' * 64}" in text
    assert f"--prepared-content-sha256 {'b' * 64}" in text
    assert "--run-seconds" not in text.split("(")[0]  # no longer required
    assert "default to the hash-bound experiment" in text
    # The two values v1.0.0 left the user to work out are filled in.
    assert f"--physics-profile {WSM6_PROFILE_ID}" in text
    outdir = root.parent / f"{root.name}-forecast"
    # POSIX display form: the certified runtime is Linux/CUDA, and
    # forward slashes are accepted by every path API on Windows too.
    assert f"--outdir {_printed(outdir)}" in text
    _assert_pasteable(lines)
    _assert_runner_accepts_printed_outdir(text, prepared_root=root)

    # A multi-domain hierarchy proof has no prepared-cache identity, so
    # it names the OTHER runner -- and prints that runner's WHOLE
    # command rather than a fragment: it binds the experiment config by
    # digest too, and this process can read it.
    hierarchy = prepared_forecast_next_command(
        {"schema": "gpuwm-gfs-native-hierarchy-proof-v1"},
        output_root=root, experiment_config=config, wps_namelist=namelist)
    hierarchy_text = "\n".join(hierarchy)
    assert "python -m gpuwm.prepared_domain_tree_forecast" in hierarchy_text
    assert f"--preparation-receipt-sha256 {proof_digest}" in hierarchy_text
    assert f"--experiment-config-sha256 {_digest(config)}" in hierarchy_text
    assert f"--outdir {_printed(outdir)}" in hierarchy_text
    _assert_pasteable(hierarchy)
    _assert_runner_accepts_printed_outdir(
        hierarchy_text, prepared_root=root, config=config)


def _printed(value) -> str:
    """A path as the next-command printer renders it."""

    import shlex

    return shlex.quote(str(value).replace("\\", "/"))


def _assert_runner_accepts_printed_outdir(text, *, prepared_root,
                                          config=None):
    """The printed --outdir must survive the runner's own guard.

    ``_assert_pasteable`` is lexical: it rejects placeholders but never
    asks whether the command would be REFUSED.  That is exactly how the
    node-8 finding shipped green -- the front door suggested
    ``<prepared_root>/forecast`` while both runners declare
    ``--prepared-root`` a protected input and refuse any --outdir that
    overlaps it, so the suggestion produced a traceback.
    """

    from gpuwm import prepared_domain_tree_forecast as module

    printed = [line.split("--outdir ", 1)[1].strip()
               for line in text.splitlines() if "--outdir " in line]
    assert printed, "no --outdir was printed"
    protected = ((Path(prepared_root), Path(config)) if config is not None
                 else (Path(prepared_root),))
    for candidate in printed:
        # The guard itself, not a re-implementation of it.  It creates
        # the directory on success, so release what we just claimed --
        # both printed commands name the same outdir, and the second
        # claim would otherwise fail for the wrong reason.
        claimed = module.claim_output_directory(
            Path(candidate), protected_roots=protected)
        claimed.rmdir()


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

    # A shipped proof descriptor with exactly one selector taken off
    # the profile it otherwise resolves to: single-domain and valid,
    # matching none of the shipped profiles.  Derived from the shipped
    # file rather than hand-written, so the only difference from a
    # config that DOES resolve is the one line this test is about --
    # and `radt` is the specific line, because 12.0 is the value the
    # shipped descriptors carried while they matched no profile at all.
    descriptor = tmp_path / "off-profile.toml"
    shipped = Path("configs/gfs_wrf_direct_proof.toml").read_text(
        encoding="utf-8")
    assert shipped.count("radt = 1.0") == 1
    descriptor.write_text(shipped.replace("radt = 1.0", "radt = 12.0"),
                          encoding="utf-8")

    lines = prepared_forecast_next_command(
        proof, output_root=root,
        experiment_config=descriptor,
        wps_namelist=Path("configs/gfs_wrf_direct_proof.namelist.wps"))
    text = "\n".join(lines)
    assert "matches none of the profiles" in text
    assert "prepared_single_domain_forecast.py" not in text
    _assert_pasteable(lines)


def _wizard_multi_domain_config(tmp_path, profile=None):
    """The wizard's own DEFAULT emission, with a nest: max_dom = 2.

    Emitted rather than hand-written for the same reason the
    single-domain helper is: the claim under test is that the config the
    product tells a user to make passes the door the product tells them
    to use, and a hand-written stand-in can only test a config nobody
    was told to write.  With no ``--physics-profile`` this is the
    product default suite -- Thompson MP8, Kain-Fritsch on d01 and none
    on d02, RTE+RRTMGP -- which is deliberately NOT in the prepared
    single-domain runner's whitelist.
    """

    from gpuwm.cli import main as cli_main

    name = "default" if profile is None else profile
    out = tmp_path / "wizard-nested" / f"{name}.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    command = ["domain", "--point=39.7,-96.6", "--card", "24gb",
               "--ladder", "12-3", "--source", "gfs",
               "--cycle", "2026-07-29T18", "--out", str(out)]
    if profile is not None:
        command.extend(("--physics-profile", profile))
    assert cli_main(command) == 0
    return out


def test_the_wizard_default_multi_domain_config_passes_its_own_front_door(
        tmp_path):
    """F21, the 1.1.0 regression: max_dom = 2 met a single-domain gate.

    v1.1's any-combo work made `prepare_gfs_wrf` call
    `validate_single_domain_physics_profile` for every configuration it
    was given.  That whitelist belongs to the prepared SINGLE-domain
    forecast runner -- the wizard says so in the note it prints beside
    the file -- and the product's own default suite is deliberately not
    in it, so the config the wizard emits BY DEFAULT could not pass the
    GFS front door at all.  A two-domain config that prepared cleanly on
    1.0.1 died with a raw traceback on 1.1.0.
    """

    from gpuwm.gfs_direct import front_door_physics_selection

    exp = load_experiment(_wizard_multi_domain_config(tmp_path))
    assert len(exp.domains) == 2

    # This is exactly what 1.1.0 did to this config, kept here as the
    # falsifier: the whitelist itself is right, and refusing the default
    # suite is what it is FOR.  The defect was asking it at all.
    with pytest.raises(ValueError, match="selected physics differs"):
        from gpuwm.physics_compat import (
            validate_single_domain_physics_profile,
        )
        validate_single_domain_physics_profile(
            WSM6_PROFILE_ID, config=exp.root.run)

    receipt = front_door_physics_selection(exp)
    assert receipt["profile"] is None
    # Every domain of the tree is recorded, not just the root: a child
    # selects its own cumulus and radiation cadence.
    assert sorted(receipt["domains"]) == ["1", "2"]
    assert receipt["domains"]["1"]["selectors"]["cu_physics"] == 1
    assert receipt["domains"]["2"]["selectors"]["cu_physics"] == 0
    assert (receipt["domains"]["1"]["components"]["microphysics"]
            == "thompson-mp8")
    assert receipt["domains"]["1"]["components"]["cumulus"] == "kain-fritsch"
    assert receipt["domains"]["2"]["components"]["cumulus"] == "off"


def test_the_shipped_two_domain_proof_config_still_passes_the_front_door():
    """The 1.0.1-shape max_dom = 2 config prepares through the gate.

    `configs/gfs_wrf_hierarchy_proof.toml` is the committed two-domain
    descriptor the hierarchy route has always used, and it is the
    sharper case of the two: it selects the LEGACY AGGREGATE radiation
    spelling with radiation off (`ra_lw_physics`/`ra_sw_physics` at -1,
    `ra_physics` 0), which the registry has no option for.  So a fix
    that merely swapped the profile whitelist for a registry-resolution
    requirement would still refuse the project's own hierarchy config --
    a second regression wearing the first one's clothes.  Recording is
    not permission: the receipt names what it can and reports the
    blocker for what it cannot, and neither answer refuses the run.
    """

    from gpuwm.gfs_direct import front_door_physics_selection

    config = (Path(__file__).parents[1] / "configs"
              / "gfs_wrf_hierarchy_proof.toml")
    exp = load_experiment(config)
    assert len(exp.domains) == 2
    receipt = front_door_physics_selection(exp)
    assert receipt["schema"] == (
        "gpuwm-front-door-physics-selection-multi-domain-v1")
    assert receipt["domains"]["1"]["selectors"]["mp_physics"] == 6
    assert receipt["domains"]["1"]["components"] is None
    assert "ra_physics" in receipt["domains"]["1"]["registry_blocker"]


def test_the_single_domain_whitelist_still_binds_on_single_domain(tmp_path):
    """Never widen: one domain still meets the profile whitelist.

    The scoping fix must not become an escape hatch.  A single-domain
    config carrying the product default suite is refused by the same
    door that now lets the two-domain one through, and a single-domain
    config bound to a shipped profile still passes.
    """

    from gpuwm.gfs_direct import front_door_physics_selection

    default_suite = load_experiment(_wizard_single_domain_config_default(
        tmp_path))
    assert len(default_suite.domains) == 1
    with pytest.raises(ValueError, match="selected physics differs"):
        front_door_physics_selection(default_suite)

    bound = load_experiment(_wizard_single_domain_config(tmp_path))
    assert len(bound.domains) == 1
    receipt = front_door_physics_selection(bound)
    assert receipt["schema"] == "gpuwm-front-door-physics-selection-v1"
    assert receipt["profile"] == WSM6_PROFILE_ID


def _noahmp_experiment(exp, acknowledgements=()):
    run = dataclasses.replace(
        exp.root.run,
        **single_domain_runtime_switches(NOAHMP_PROFILE_ID),
    )
    root = dataclasses.replace(exp.root, run=run)
    return dataclasses.replace(
        exp, domains=(root,), acknowledgements=tuple(acknowledgements))


@pytest.mark.parametrize(
    ("flag", "toml", "sources"),
    (
        (("noahmp-host-column-throughput-v1",), (), ["--ack"]),
        ((), ("noahmp-host-column-throughput-v1",),
         ["[experiment].acknowledgements"]),
        (("noahmp-host-column-throughput-v1",),
         ("noahmp-host-column-throughput-v1",),
         ["--ack", "[experiment].acknowledgements"]),
    ),
)
def test_ack_delivery_unions_flag_and_toml_with_source_provenance(
        tmp_path, flag, toml, sources):
    from gpuwm.gfs_direct import front_door_physics_selection

    exp = _noahmp_experiment(
        load_experiment(_wizard_single_domain_config(tmp_path)), toml)
    receipt = front_door_physics_selection(
        exp, physics_profile=NOAHMP_PROFILE_ID,
        expert_acknowledgements=flag)

    assert receipt["acknowledgements"] == [
        "noahmp-host-column-throughput-v1"]
    assert receipt["acknowledgement_provenance"] == {
        "noahmp-host-column-throughput-v1": sources}


@pytest.mark.parametrize("delivery", ("flag", "toml"))
def test_wrong_ack_id_refuses_with_the_exact_two_short_forms(
        tmp_path, delivery):
    from gpuwm.gfs_direct import front_door_physics_selection

    toml = ("wrong-id-v2",) if delivery == "toml" else ()
    flag = ("wrong-id-v2",) if delivery == "flag" else ()
    exp = _noahmp_experiment(
        load_experiment(_wizard_single_domain_config(tmp_path)), toml)
    with pytest.raises(ValueError) as caught:
        front_door_physics_selection(
            exp, physics_profile=NOAHMP_PROFILE_ID,
            expert_acknowledgements=flag)
    message = str(caught.value)
    assert "\n" not in message
    assert "d01 resolved physics tuple (" in message
    assert (
        "--ack noahmp-host-column-throughput-v1 or "
        'acknowledgements = ["noahmp-host-column-throughput-v1"]'
        in message
    )


@pytest.mark.parametrize("delivery", ("flag", "toml", "both"))
def test_registry_reachable_tuple_is_unaffected_by_ack_delivery(
        tmp_path, delivery):
    from gpuwm.gfs_direct import front_door_physics_selection

    flag = ("irrelevant-v1",) if delivery in ("flag", "both") else ()
    toml = ("irrelevant-v2",) if delivery in ("toml", "both") else ()
    exp = dataclasses.replace(
        load_experiment(_wizard_single_domain_config(tmp_path)),
        acknowledgements=toml)
    receipt = front_door_physics_selection(
        exp, expert_acknowledgements=flag)
    assert receipt["profile"] == WSM6_PROFILE_ID


def test_an_explicit_profile_is_still_enforced_on_a_domain_tree(tmp_path):
    """A gate the caller ASKED for is never dropped by the scoping fix.

    `--physics-profile` on a multi-domain run is how a user says "bind
    this tree to a shipped suite"; node-8 verified that route works on
    1.1.0 and it must keep working.  It binds the ROOT -- children carry
    their own cumulus and radiation cadence by design, so demanding the
    profile of every domain would refuse the wizard's own nested
    emission of that same profile.
    """

    from gpuwm.gfs_direct import front_door_physics_selection

    exp = load_experiment(_wizard_multi_domain_config(
        tmp_path, profile=MORRISON_PROFILE_ID))
    assert len(exp.domains) == 2
    receipt = front_door_physics_selection(
        exp, physics_profile=MORRISON_PROFILE_ID)
    assert receipt["profile"] == MORRISON_PROFILE_ID

    with pytest.raises(ValueError, match="selected physics differs"):
        front_door_physics_selection(exp, physics_profile=WSM6_PROFILE_ID)


def test_a_front_door_refusal_reaches_the_user_as_a_sentence(tmp_path,
                                                             capsys):
    """Not a traceback.  F21's third consequence, and F19's whole point.

    The gate is entered through `python -m gpuwm.gfs_direct`, whose
    return code `rw-wps` passes straight through, so what the user sees
    is whatever this `main` prints.  On 1.1.0 that was a stack trace.
    """

    from gpuwm.gfs_direct import main as gfs_main

    config = _wizard_single_domain_config_default(tmp_path)
    capsys.readouterr()  # the wizard's own report is not under test
    missing = tmp_path / "series.tsv"
    missing.write_text("0\tf000.grib2\n3\tf003.grib2\n", encoding="utf-8")
    code = gfs_main([
        "--series", str(missing),
        "--cycle", "2026-07-29_18:00:00",
        "--bridge", str(tmp_path / "absent-bridge"),
        "--wps-namelist", str(tmp_path / "absent.namelist.wps"),
        "--experiment-config", str(config),
        "--input-manifest", str(tmp_path / "absent-manifest.json"),
        "--input-manifest-sha256", "0" * 64,
        "--output-root", str(tmp_path / "out"),
    ])
    assert code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    lines = [line for line in captured.err.splitlines() if line.strip()]
    assert len(lines) == 1, captured.err
    assert lines[0].startswith("rw-wps --source gfs: ")
    assert "Traceback" not in captured.err


def _wizard_single_domain_config_default(tmp_path):
    """One domain, product default suite: no shipped profile matches it."""

    from gpuwm.cli import main as cli_main

    out = tmp_path / "wizard-default" / "case.toml"
    out.parent.mkdir(parents=True, exist_ok=True)
    assert cli_main(["domain", "--point=39.7,-96.6", "--card", "24gb",
                     "--ladder", "12", "--source", "gfs",
                     "--cycle", "2026-07-29T18", "--out", str(out)]) == 0
    return out


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


def test_a_bound_refusal_reaches_python_with_the_reason_and_a_remedy():
    """The reported blocker's shape, as a user would now receive it.

    v1.1.1 raised ``GFS Rust bridge failed: <stderr>`` and stopped.  The
    stderr is true and unreadable on its own: a soil-moisture value of
    1.05 means nothing to someone who did not write the bridge, and the
    obvious reading -- that the range is too tight -- is the one thing
    that must not be acted on.
    """

    stderr = (
        "GFS_SM010040 value 1.05 outside [0,1] by 0.05 "
        "(quantization tolerance 0.000001); a bound-kissing value is "
        "clamped, this one is not\n")
    message = decode_failure_message("GFS Rust bridge", stderr)
    lines = message.splitlines()

    # The bridge's own words survive verbatim on the first line.
    assert lines[0] == (
        "GFS Rust bridge failed: " + stderr.strip())
    # And the remedy says what the number means, that bound-kissing is
    # already handled, and what to actually do.
    assert "remedy:" in message
    assert "GFS_SM010040 decoded 1.05" in message
    assert "clamps those" in message
    assert "Re-fetch the cycle and re-run" in message
    # Every remedy line is a comment or a command, so the block survives
    # being pasted whole.
    for line in lines[2:]:
        assert line.strip().startswith("#"), line
    _assert_pasteable(lines)


def test_a_failure_that_is_not_a_bound_refusal_gains_no_quantization_prose():
    """A missing file is not a packing story and must not be told as one."""

    message = decode_failure_message(
        "GFS Rust bridge", "series line 1 references missing \"f000.grb2\"\n")
    assert message == (
        "GFS Rust bridge failed: series line 1 references missing "
        "\"f000.grb2\"")
    assert "quantization" not in message
    assert "remedy" not in message


def test_a_silent_decoder_failure_still_names_itself():
    assert decode_failure_message("GFS Rust bridge", "   ") == (
        "GFS Rust bridge failed")


def test_the_printed_command_survives_a_path_with_a_space_in_it():
    """Not lexical this time: the line is shell-parsed back to argv.

    The existing checks reject angle brackets and ALL-CAPS placeholders,
    which is why a real defect walked past them -- a perfectly valid
    `--outdir` containing a space was printed bare and split into two
    arguments the moment the command was pasted.
    """

    import shlex

    from gpuwm.gfs_direct import _arg

    for awkward in ("plain/path", "a dir/with spaces", "quote's/dir",
                    "dollar$dir/x"):
        rendered = _arg(awkward)
        assert shlex.split(f"--outdir {rendered}") == ["--outdir", awkward]
    # An ordinary path is left alone, so this is invisible except where
    # it matters.
    assert _arg("relative/output") == "relative/output"


@pytest.mark.parametrize("proof", [
    # The single-domain branch (gfs_direct.py ~1233-1242): prints
    # --prepared-root, --experiment-config, --wps-namelist, --outdir.
    {"input_manifest_sha256": "a" * 64,
     "prepared_cache": {"content_sha256": "b" * 64}},
    # The multi-domain hierarchy branch (~1207-1216): a proof with no
    # single prepared-cache identity, printing the OTHER runner's
    # --prepared-root, --experiment-config, --outdir.  Both branches
    # interpolate paths, so both must survive a path with a space.
    {"schema": "gpuwm-gfs-native-hierarchy-proof-v1"},
], ids=["single-domain", "hierarchy"])
def test_every_printed_path_in_the_forecast_command_is_shell_parseable(
        tmp_path, proof):
    import shlex
    import shutil

    from gpuwm.gfs_direct import prepared_forecast_next_command

    root = tmp_path / "a case with spaces" / "prepared"
    root.mkdir(parents=True)
    (root / "proof.json").write_text(json.dumps(proof), encoding="utf-8")
    # The experiment config and namelist live in the spaced directory too,
    # so their printed paths -- not only --outdir -- carry a space and
    # must be quoted to survive the parse.
    config = shutil.copy2(
        _wizard_single_domain_config(tmp_path), root / "case config.toml")
    namelist = root / "namelist with spaces.wps"
    namelist.write_text("&share\n/\n", encoding="utf-8")

    lines = prepared_forecast_next_command(
        proof, output_root=root, experiment_config=config,
        wps_namelist=namelist)
    command = " ".join(
        line.strip().rstrip("\\") for line in lines
        if line.strip().startswith(("python ", "--", "      --")))
    # The line shell-parses at all only because every path with a space
    # in it is quoted; an unquoted one would split here.
    argv = shlex.split(command)
    outdir = argv[argv.index("--outdir") + 1]
    assert " " in outdir, outdir
    assert Path(outdir).name == f"{root.name}-forecast"
    # Every value that IS a path this process wrote to a spaced directory
    # must come back through the parse whole, not split.
    prepared_root = argv[argv.index("--prepared-root") + 1]
    assert prepared_root == str(root).replace("\\", "/"), prepared_root
    config_arg = argv[argv.index("--experiment-config") + 1]
    assert " " in config_arg and Path(config_arg).is_file(), config_arg


# ---------------------------------------------------------------------------
# F2: the export request reaches the adapter from both front doors
# ---------------------------------------------------------------------------

def _gfs_front_door_args(extra: list[str]):
    from gpuwm import source_cli

    return source_cli._parser().parse_args([
        "--source", "gfs",
        "--gfs-series", "series.tsv",
        "--cycle", "2026-07-29_06:00:00",
        "--bridge", "bridge.exe",
        "--wps-namelist", "namelist.wps",
        "--experiment-config", "experiment.toml",
        "--source-manifest", "manifest.json",
        "--source-manifest-sha256", "a" * 64,
        "--output-root", "prep",
        "--geog-root", "geog",
        *extra,
    ])


def test_the_gfs_front_door_forwards_the_export_request_only_when_declined():
    """`rw-wps` passes the flag through; the default prints nothing extra."""

    from gpuwm import source_cli

    default = source_cli._gfs_command(_gfs_front_door_args([]))
    declined = source_cli._gfs_command(
        _gfs_front_door_args(["--no-stock-wrf-export"]))

    assert "--no-stock-wrf-export" not in default
    assert "--no-stock-wrf-export" in declined
    assert source_cli._required_gfs_args(
        _gfs_front_door_args(["--no-stock-wrf-export"])) == []
    # The default has to be FALSY, not True-meaning-"export": every
    # inventory action on this parser refuses to be combined with any
    # namespace entry that is not None/False, so a flag defaulting to True
    # breaks `--validate-hrrr-domain` and its siblings for every caller.
    assert _gfs_front_door_args([]).no_stock_wrf_export is False
    assert source_cli._active_action_arguments(
        _gfs_front_door_args([]),
        allowed=frozenset({"no_stock_wrf_export"})) == \
        source_cli._active_action_arguments(
            _gfs_front_door_args([]), allowed=frozenset())


def test_the_export_request_is_a_gfs_route_flag_and_says_so_elsewhere():
    from gpuwm import source_cli

    args = _gfs_front_door_args(["--no-stock-wrf-export"])
    for validator in (source_cli._required_era5_args,
                      source_cli._required_twentycr_args,
                      source_cli._required_mapped_args,
                      source_cli._required_hrrr_args):
        errors = validator(args)
        assert any("--no-stock-wrf-export" in error for error in errors), \
            validator.__name__


def test_the_gfs_adapter_cli_carries_the_export_request_to_the_preparation(
        monkeypatch, tmp_path, capsys):
    from gpuwm import gfs_direct

    observed = {}

    def prepare(**kwargs):
        observed.update(kwargs)
        return {"schema": "test", "wrf_manifest": {"status": "READY"}}

    monkeypatch.setattr(gfs_direct, "prepare_gfs_wrf", prepare)
    monkeypatch.setattr(
        gfs_direct, "prepared_forecast_next_command",
        lambda *_a, **_k: [])
    base = [
        "--series", "series.tsv", "--cycle", "2026-07-29_06:00:00",
        "--bridge", "bridge.exe", "--wps-namelist", "namelist.wps",
        "--experiment-config", "experiment.toml",
        "--input-manifest", "manifest.json",
        "--input-manifest-sha256", "a" * 64,
        "--output-root", str(tmp_path / "prep"),
    ]

    assert gfs_direct.main(base) == 0
    assert observed["stock_wrf_export"] is True

    assert gfs_direct.main(base + ["--no-stock-wrf-export"]) == 0
    assert observed["stock_wrf_export"] is False
    capsys.readouterr()


def test_the_front_door_says_plainly_when_the_bonus_export_did_not_happen():
    """A user who expected wrf-native-input/ is told, in one sentence."""

    from gpuwm.gfs_direct import stock_wrf_export_notice

    assert stock_wrf_export_notice({"wrf_manifest": {"status": "READY"}}) == []
    assert stock_wrf_export_notice({}) == []

    refused = stock_wrf_export_notice({"wrf_manifest": {
        "status": "REFUSED",
        "reason": "unsupported direct-export configuration: "
                  "{'bl_pbl_physics': (5, 1)}"}})
    assert any("bonus stock-WRF export was refused" in line
               for line in refused)
    assert any("'bl_pbl_physics': (5, 1)" in line for line in refused)
    assert any("run the forecast command below" in line for line in refused)

    skipped = stock_wrf_export_notice(
        {"wrf_manifest": {"status": "NOT_REQUESTED"}})
    assert any("no stock-WRF export was requested" in line
               for line in skipped)
