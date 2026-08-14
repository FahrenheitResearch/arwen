"""CPU-only tests for the T17 product and T18 evidence-pack tools."""

from __future__ import annotations

import os

from datetime import datetime, timedelta
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import matplotlib.image as mpimg
import netCDF4
import numpy as np
import pytest

from gpuwm.verify import nest_gates

# tools/flagship/products.py imports wrf-rust at module scope, so without that
# package this line raises during COLLECTION and takes the whole suite with it
# -- every unrelated test included.  Skip this module instead of the run.
pytest.importorskip(
    "wrf", reason="wrf-rust is required by tools.flagship.products")

from tools.flagship import evidence_pack, products  # noqa: E402


def _variable(ds, name, dims, values, units=""):
    variable = ds.createVariable(name, "f4", dims)
    variable.units = units
    variable[:] = np.asarray(values, dtype=np.float32)
    return variable


def _write_wrfout(path: Path, valid_time: datetime, step: int, *,
                  offset: float = 0.0, finite: bool = True,
                  precip_components: tuple[bool, bool] = (True, True),
                  declared_complete: str | None = None,
                  hydrometeors: bool = True) -> Path:
    ny, nx, nz = 3, 4, 3
    grid = np.arange(ny * nx, dtype=np.float32).reshape(ny, nx)
    path.parent.mkdir(parents=True, exist_ok=True)
    with netCDF4.Dataset(path, "w") as ds:
        ds.TEST_STEP = step
        ds.TEST_OFFSET = offset
        if declared_complete is not None:
            setattr(ds, products._COMPLETE_PRECIP_ATTRIBUTE, declared_complete)
        ds.createDimension("Time", 1)
        ds.createDimension("DateStrLen", 19)
        ds.createDimension("south_north", ny)
        ds.createDimension("west_east", nx)
        ds.createDimension("bottom_top", nz)
        times = ds.createVariable("Times", "S1", ("Time", "DateStrLen"))
        stamp = valid_time.strftime("%Y-%m-%d_%H:%M:%S").encode("ascii")
        times[0] = np.frombuffer(stamp, dtype="S1")

        lat = 34.0 + np.arange(ny, dtype=np.float32)[:, None] * 0.1
        lon = -98.0 + np.arange(nx, dtype=np.float32)[None, :] * 0.1
        _variable(ds, "XLAT", ("Time", "south_north", "west_east"),
                  np.broadcast_to(lat, (1, ny, nx)), "degrees_north")
        _variable(ds, "XLONG", ("Time", "south_north", "west_east"),
                  np.broadcast_to(lon, (1, ny, nx)), "degrees_east")

        refl = np.stack((grid, grid + 10.0, grid + 20.0))[None] + step * 5.0 + offset
        if not finite:
            refl[:] = np.nan
        _variable(ds, "REFL_10CM",
                  ("Time", "bottom_top", "south_north", "west_east"),
                  refl, "dBZ")
        # Deliberately wrong/raw values: flagship science must not consume these.
        _variable(ds, "MSLP", ("Time", "south_north", "west_east"),
                  np.full((1, ny, nx), 777.0), "hPa")
        _variable(ds, "PSFC", ("Time", "south_north", "west_east"),
                  np.full((1, ny, nx), 95000.0), "Pa")
        _variable(ds, "U10", ("Time", "south_north", "west_east"),
                  np.full((1, ny, nx), 3.0), "m s-1")
        _variable(ds, "V10", ("Time", "south_north", "west_east"),
                  np.full((1, ny, nx), 4.0), "m s-1")
        if hydrometeors:
            _variable(ds, "QRAIN",
                      ("Time", "bottom_top", "south_north", "west_east"),
                      np.full((1, nz, ny, nx), 1.0e-4), "kg kg-1")
        if precip_components[0]:
            _variable(ds, "RAINC", ("Time", "south_north", "west_east"),
                      np.full((1, ny, nx), step * 0.25 + offset), "mm")
        if precip_components[1]:
            _variable(ds, "RAINNC", ("Time", "south_north", "west_east"),
                      (step + grid * 0.1 + offset)[None], "mm")
    return path


def _make_run(root: Path, *, domain: str = "d01", steps=(0, 1, 2),
              interval_minutes: int = 60, offset: float = 0.0,
              finite: bool = True, **kwargs) -> list[Path]:
    start = datetime(1999, 5, 3, 12)
    paths = []
    for step in steps:
        valid = start + timedelta(minutes=interval_minutes * step)
        name = f"wrfout_{domain}_{valid.strftime('%Y-%m-%d_%H_%M_%S')}"
        paths.append(_write_wrfout(
            root / name, valid, step, offset=offset, finite=finite, **kwargs))
    return paths


@pytest.fixture
def fake_wrf_rust(monkeypatch):
    """Synthetic wrf-rust boundary with distinct earth-relative answers."""
    calls: list[tuple[str, str, int, str | None]] = []
    interpolation_calls: list[float] = []

    class FakeWrfFile:
        def __init__(self, path):
            self.path = Path(path)

    def fake_getvar(wrf_file, name, timeidx=0, units=None, **kwargs):
        del kwargs
        with netCDF4.Dataset(wrf_file.path, "r") as ds:
            step = int(ds.TEST_STEP)
            offset = float(ds.TEST_OFFSET)
            ny = len(ds.dimensions["south_north"])
            nx = len(ds.dimensions["west_east"])
            nz = len(ds.dimensions["bottom_top"])
            raw_refl = np.asarray(ds.variables["REFL_10CM"][timeidx], dtype=float)
        calls.append((wrf_file.path.name, name, timeidx, units))
        grid = np.arange(ny * nx, dtype=float).reshape(ny, nx)
        if name == "slp":
            return 1000.0 - step + grid * 0.1 + offset
        if name == "uvmet10":
            return np.stack((np.full((ny, nx), 8.0 + step),
                             np.full((ny, nx), -2.0 - offset)))
        if name == "wspd10":
            return np.full((ny, nx), np.hypot(8.0 + step, -2.0 - offset))
        if name == "dbz":
            return raw_refl + 25.0
        if name == "pressure":
            return np.broadcast_to(
                np.array([800.0, 600.0, 400.0])[:, None, None], (nz, ny, nx))
        if name == "height":
            return np.broadcast_to(
                np.array([200.0, 550.0, 750.0])[:, None, None], (nz, ny, nx))
        if name == "temp":
            return np.broadcast_to(
                np.array([-10.0, -20.0, -30.0])[:, None, None], (nz, ny, nx))
        if name == "uvmet":
            u = np.broadcast_to(
                np.array([10.0, 20.0, 30.0])[:, None, None], (nz, ny, nx))
            v = np.broadcast_to(
                np.array([-5.0, 0.0, 5.0])[:, None, None], (nz, ny, nx))
            return np.stack((u, v))
        if name == "uhel":
            return 10.0 + grid + step
        if name == "uhel_max":
            return 20.0 + grid + step * 3.0
        raise AssertionError(f"unexpected getvar request {name!r}")

    def fake_interplevel(field, pressure, target):
        assert field.shape == pressure.shape
        interpolation_calls.append(float(target))
        # The fixture brackets 500 hPa at levels 1 and 2.
        return 0.5 * (field[1] + field[2])

    monkeypatch.setattr(products, "WrfFile", FakeWrfFile)
    monkeypatch.setattr(products, "getvar", fake_getvar)
    monkeypatch.setattr(products, "interplevel", fake_interplevel)
    return {"calls": calls, "getvar": fake_getvar,
            "interpolation_calls": interpolation_calls}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(rung: Path, milestone: str, artifacts: list[Path], *,
                    evaluator_commit: str = "a" * 40,
                    fingerprint: str = "b" * 64) -> None:
    payload = {
        "schema": 3,
        "rung": milestone,
        "evaluator_commit": evaluator_commit,
        "experiment_fingerprint": fingerprint,
        "artifacts": [{
            "relative_path": artifact.relative_to(rung).as_posix(),
            "bytes": artifact.stat().st_size,
            "sha256": _sha(artifact),
        } for artifact in artifacts],
    }
    (rung / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")


def _complete_evidence_root(root: Path) -> None:
    for milestone in nest_gates.MILESTONES:
        if milestone == "P5B":
            continue
        rung = root / milestone
        rung.mkdir(parents=True)
        report = rung / f"{milestone}-report.json"
        report.write_text(json.dumps({
            "schema": 1,
            "rung": milestone,
            "gates": [{"metric": gate.metric, "passed": True}
                      for gate in nest_gates.gates_for(milestone)],
        }), encoding="utf-8")
        _write_manifest(rung, milestone, [report])


def test_products_use_wrf_rust_and_render_outer_inventory(
        tmp_path, fake_wrf_rust):
    run = tmp_path / "run"
    _make_run(run)
    output = tmp_path / "products"

    summary = products.generate_products(run, output, domains=["d01"])

    assert summary["schema"] == 2
    # The version reported is the one INSTALLED here, not a literal: the
    # certified window is a range (>=0.2.35,<0.3), so hard-coding the floor
    # would red this suite on every box running a newer core.
    assert summary["science_core"] == {
        "distribution": "wrf-rust", "version": products.WRF_RUST_VERSION}
    assert products.version_supported(products.WRF_RUST_VERSION)
    assert summary["input_identity"]["root"] == "."
    assert len(summary["input_identity"]["sha256"]) == 64
    assert "run_dir_absolute" in summary["diagnostics"]
    assert summary["domains"]["d01"]["required_products"] == [
        "reflectivity", "slp", "uvmet10", "precipitation", "500hpa"]

    names = set(summary["products"])
    assert any(name.startswith("d01_500hpa_") for name in names)
    assert any("refl_dbz_crosscheck" in name for name in names)
    assert summary["warnings"]
    rendered = sorted(output.glob("*.png"))
    assert len(rendered) == summary["product_count"]
    assert all(path.stat().st_size > 1000 for path in rendered)
    image = mpimg.imread(next(path for path in rendered if "500hpa" in path.name))
    assert np.isfinite(image).all()
    assert float(np.std(image)) > 0.02

    stored = json.loads((output / "run-summary.json").read_text("utf-8"))
    first = stored["domains"]["d01"]["frame_inventory"][0]
    assert first["coordinate_source"] == "XLAT/XLONG"
    assert first["key_fields"]["REFL_10CM_COMPOSITE"] == {
        "finite_count": 12, "max": 31.0, "min": 20.0, "units": "dBZ"}
    assert first["key_fields"]["SLP_WRF_RUST"]["min"] == pytest.approx(1000.0)
    assert first["key_fields"]["UVMET10_U"]["min"] == pytest.approx(8.0)
    assert first["key_fields"]["TEMP_500_WRF_RUST"]["finite_count"] == 12
    assert stored["domains"]["d01"]["frame_inventory"][-1][
        "key_fields"]["TOTAL_PRECIP"]["min"] == pytest.approx(2.5)

    requested = [item[1] for item in fake_wrf_rust["calls"]]
    assert {"slp", "uvmet10", "wspd10", "dbz", "pressure", "height",
            "temp", "uvmet"} <= set(requested)
    assert "U10" not in requested and "V10" not in requested and "PSFC" not in requested
    assert fake_wrf_rust["interpolation_calls"] == [500.0] * 12
    assert all(item["wrf_rust_version"] == products.WRF_RUST_VERSION
               for item in summary["product_artifacts"])
    for artifact in summary["product_artifacts"]:
        path = output / artifact["relative_path"]
        assert artifact["bytes"] == path.stat().st_size
        assert artifact["sha256"] == _sha(path)


def test_index_space_coordinates_fail_before_product_writes(
        tmp_path, fake_wrf_rust, monkeypatch):
    run = tmp_path / "run"
    _make_run(run, steps=(0,))
    output = tmp_path / "index-space-out"

    def index_coordinates(_frame, shape):
        indices = np.indices(shape, dtype=float)
        return indices[1], indices[0]

    monkeypatch.setattr(products, "_coordinates", index_coordinates)
    with pytest.raises(ValueError, match="native XLAT/XLONG source"):
        products.generate_products(run, output, domains=["d01"])
    assert not output.exists()


def test_inner_products_use_uhel_uhel_max_and_uvmet10(
        tmp_path, fake_wrf_rust):
    run = tmp_path / "run"
    _make_run(run, domain="d03", interval_minutes=15)

    summary = products.generate_products(run, tmp_path / "out", domains=["d03"])

    names = set(summary["products"])
    assert "d03_uhel_track_all_times.png" in names
    assert sum("hook_echo" in name for name in names) == 3
    assert summary["domains"]["d03"]["required_products"] == [
        "reflectivity", "slp", "uvmet10", "precipitation",
        "uhel_track", "hook_echo"]
    requested = [item[1] for item in fake_wrf_rust["calls"]]
    assert requested.count("uhel") == 3
    assert requested.count("uhel_max") == 3
    assert requested.count("uvmet10") == 3


def test_reference_requires_exact_nonempty_coverage(tmp_path, fake_wrf_rust):
    run = tmp_path / "run"
    _make_run(run)
    empty = tmp_path / "empty-reference"
    empty.mkdir()
    with pytest.raises(ValueError, match="no WRF output frames"):
        products.generate_products(
            run, tmp_path / "empty-out", domains=["d01"], reference_dir=empty)
    assert not (tmp_path / "empty-out").exists()

    subset = tmp_path / "subset"
    _make_run(subset, steps=(1,), offset=2.0)
    with pytest.raises(ValueError, match="time inventory"):
        products.generate_products(
            run, tmp_path / "subset-out", domains=["d01"], reference_dir=subset)
    assert not (tmp_path / "subset-out").exists()

    reference = tmp_path / "reference"
    _make_run(reference, offset=2.0)
    summary = products.generate_products(
        run, tmp_path / "compared", domains=["d01"], reference_dir=reference)
    assert sum("comparison" in name for name in summary["products"]) == 9
    assert summary["reference_identity"]["root"] == "."


def test_fail_loud_inventory_finiteness_time_and_cadence(tmp_path, fake_wrf_rust):
    missing_domain = tmp_path / "missing-domain"
    _make_run(missing_domain)
    with pytest.raises(ValueError, match="requested domains have no WRF frames"):
        products.generate_products(
            missing_domain, tmp_path / "out1", domains=["d01", "d02"])
    assert not (tmp_path / "out1").exists()

    nonfinite = tmp_path / "nonfinite"
    _make_run(nonfinite, finite=False)
    with pytest.raises(ValueError, match="REFL_10CM.*completely finite"):
        products.generate_products(nonfinite, tmp_path / "out2", domains=["d01"])
    assert not (tmp_path / "out2").exists()

    invalid = tmp_path / "invalid-time"
    path = _make_run(invalid, steps=(0,))[0]
    with netCDF4.Dataset(path, "r+") as ds:
        ds.variables["Times"][0] = np.frombuffer(b"1999-99-99_99:99:99", dtype="S1")
    with pytest.raises(ValueError, match="invalid Times inventory"):
        products.generate_products(invalid, tmp_path / "out3", domains=["d01"])
    assert not (tmp_path / "out3").exists()

    gap = tmp_path / "gap"
    _make_run(gap, steps=(0, 2))
    with pytest.raises(ValueError, match="cadence mismatch"):
        products.generate_products(gap, tmp_path / "out4", domains=["d01"])
    assert not (tmp_path / "out4").exists()


def test_precip_components_complete_declaration_and_decrease(
        tmp_path, fake_wrf_rust):
    one_component = tmp_path / "one-component"
    _make_run(one_component, steps=(0,), precip_components=(True, False))
    with pytest.raises(ValueError, match="requires RAINC and RAINNC"):
        products.generate_products(
            one_component, tmp_path / "bad-out", domains=["d01"])
    assert not (tmp_path / "bad-out").exists()

    declared = tmp_path / "declared"
    _make_run(declared, steps=(0,), precip_components=(True, False),
              declared_complete="RAINC")
    summary = products.generate_products(
        declared, tmp_path / "declared-out", domains=["d01"])
    assert summary["domains"]["d01"]["frame_inventory"][0][
        "precipitation_source"] == "declared-complete:RAINC"

    decreasing = tmp_path / "decreasing"
    paths = _make_run(decreasing, steps=(0, 1))
    with netCDF4.Dataset(paths[1], "r+") as ds:
        ds.variables["RAINC"][:] = -1.0
        ds.variables["RAINNC"][:] = -1.0
    with pytest.raises(ValueError, match="accumulator decrease"):
        products.generate_products(
            decreasing, tmp_path / "decreasing-out", domains=["d01"])
    assert not (tmp_path / "decreasing-out").exists()


def test_wrf_rust_shape_exception_and_version_mismatch_precede_writes(
        tmp_path, fake_wrf_rust, monkeypatch):
    run = tmp_path / "run"
    _make_run(run, steps=(0,))
    base = fake_wrf_rust["getvar"]

    def malformed_uvmet(wrf_file, name, **kwargs):
        if name == "uvmet10":
            return np.zeros((2, 2, 2))
        return base(wrf_file, name, **kwargs)

    monkeypatch.setattr(products, "getvar", malformed_uvmet)
    with pytest.raises(ValueError, match="uvmet10 shape"):
        products.generate_products(run, tmp_path / "shape-out", domains=["d01"])
    assert not (tmp_path / "shape-out").exists()

    def nonfinite_slp(wrf_file, name, **kwargs):
        if name == "slp":
            return np.full((3, 4), np.nan)
        return base(wrf_file, name, **kwargs)

    monkeypatch.setattr(products, "getvar", nonfinite_slp)
    with pytest.raises(ValueError, match="wrf-rust slp.*completely finite"):
        products.generate_products(run, tmp_path / "finite-out", domains=["d01"])
    assert not (tmp_path / "finite-out").exists()

    def explode(wrf_file, name, **kwargs):
        if name == "slp":
            raise RuntimeError("wrf-rust science failure")
        return base(wrf_file, name, **kwargs)

    monkeypatch.setattr(products, "getvar", explode)
    with pytest.raises(RuntimeError, match="wrf-rust science failure"):
        products.generate_products(run, tmp_path / "error-out", domains=["d01"])
    assert not (tmp_path / "error-out").exists()

    monkeypatch.setattr(products.importlib_metadata, "version", lambda name: "0.2.34")
    with pytest.raises(RuntimeError, match="version mismatch"):
        products.generate_products(run, tmp_path / "version-out", domains=["d01"])
    assert not (tmp_path / "version-out").exists()


def test_real_wrf_rust_core_if_fixture_is_available():
    path = Path(os.environ.get("GPUWM_TEST_REAL_WRFOUT",
                               "gpuwm-fixture-unset/real-wrfout"))
    if not path.is_file():
        pytest.skip("set GPUWM_TEST_REAL_WRFOUT to a real wrfout "
                    "NetCDF to run the real wrf-rust core check")
    from wrf import WrfFile as RealWrfFile, getvar as real_getvar

    wrf_file = RealWrfFile(str(path))
    slp = np.asarray(real_getvar(wrf_file, "slp"))
    uvmet10 = np.asarray(real_getvar(wrf_file, "uvmet10"))
    assert slp.shape == (400, 500) and np.isfinite(slp).all()
    assert uvmet10.shape == (2, 400, 500) and np.isfinite(uvmet10).all()


def test_evidence_pack_binds_hashes_pins_ledger_and_portable_paths(tmp_path):
    root = tmp_path / "rungs"
    _complete_evidence_root(root)
    output = tmp_path / "pack"

    pack = evidence_pack.build_evidence_pack(root, output)

    assert pack["schema"] == 2
    assert pack["closeout_ready"] is True
    assert pack["rungs_root"]["relative_path"] == "."
    assert len(pack["rungs_root"]["inventory_sha256"]) == 64
    assert len(pack["pack_fingerprint"]) == 64
    assert pack["manifest_provenance"][0]["evaluator_commit"] == "a" * 40
    assert pack["manifest_provenance"][0]["experiment_fingerprint"] == "b" * 64
    assert all(len(item["sha256"]) == 64 for item in pack["source_artifacts"])
    assert all(item["bytes"] > 0 for item in pack["source_artifacts"])
    assert all(not Path(item["relative_path"]).is_absolute()
               for item in pack["source_artifacts"])
    assert {item["role"] for item in pack["ledger"]["sources"]} == {
        "gate_ledger", "architecture_doc", "plan_doc"}
    assert all(len(item["source_commit"]) == 40
               for item in pack["ledger"]["sources"])
    stored = json.loads((output / "evidence-pack.json").read_text("utf-8"))
    assert stored == pack


def test_evidence_pack_rejects_hash_substitution_escape_and_missing_pin(tmp_path):
    root = tmp_path / "rungs"
    _complete_evidence_root(root)
    report = root / "N3" / "N3-report.json"
    report.write_text(report.read_text("utf-8") + " ", encoding="utf-8")
    with pytest.raises(evidence_pack.EvidencePackError,
                       match="byte/hash verification"):
        evidence_pack.build_evidence_pack(root, tmp_path / "tampered-out")
    assert not (tmp_path / "tampered-out").exists()

    bad = tmp_path / "bad"
    rung = bad / "N3"
    rung.mkdir(parents=True)
    artifact = rung / "N3-report.json"
    artifact.write_text('{"schema": 1, "rung": "N3", "gates": []}',
                        encoding="utf-8")
    (rung / "manifest.json").write_text(json.dumps({
        "schema": 3, "rung": "N3", "evaluator_commit": "a" * 40,
        "artifacts": [{"relative_path": artifact.name,
                       "bytes": artifact.stat().st_size, "sha256": _sha(artifact)}],
    }), encoding="utf-8")
    with pytest.raises(evidence_pack.EvidencePackError,
                       match="experiment_fingerprint or config_digest"):
        evidence_pack.build_evidence_pack(
            bad, tmp_path / "bad-out", diagnostic_incomplete=True)

    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    (rung / "manifest.json").write_text(json.dumps({
        "schema": 3, "rung": "N3", "evaluator_commit": "a" * 40,
        "experiment_fingerprint": "b" * 64,
        "artifacts": [{"relative_path": "../../outside.json",
                       "bytes": outside.stat().st_size, "sha256": _sha(outside)}],
    }), encoding="utf-8")
    with pytest.raises(evidence_pack.EvidencePackError, match="escapes rung"):
        evidence_pack.build_evidence_pack(
            bad, tmp_path / "escape-out", diagnostic_incomplete=True)


def test_evidence_conflicts_and_blockers_never_publish_closeout(tmp_path):
    root = tmp_path / "rungs"
    _complete_evidence_root(root)
    rung = root / "N3"
    gate = nest_gates.gates_for("N3")[0]
    verdict = rung / "N3-verdicts.json"
    verdict.write_text(json.dumps({
        "schema": 1, "rung": "N3", gate.metric: {"passed": False},
    }), encoding="utf-8")
    _write_manifest(rung, "N3", [rung / "N3-report.json", verdict])
    with pytest.raises(evidence_pack.EvidencePackError,
                       match="conflicting passed values"):
        evidence_pack.build_evidence_pack(root, tmp_path / "conflict-out")
    assert not (tmp_path / "conflict-out").exists()

    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    with pytest.raises(evidence_pack.EvidencePackError,
                       match="required rung manifests"):
        evidence_pack.build_evidence_pack(incomplete, tmp_path / "closeout")
    assert not (tmp_path / "closeout").exists()
    diagnostic = evidence_pack.build_evidence_pack(
        incomplete, tmp_path / "diagnostic", diagnostic_incomplete=True)
    assert diagnostic["mode"] == "diagnostic-incomplete"
    assert diagnostic["closeout_ready"] is False
    assert diagnostic["blockers"]
    assert evidence_pack.main([
        "--rungs-root", str(incomplete), "--outdir", str(tmp_path / "cli-diagnostic"),
        "--diagnostic-incomplete",
    ]) == 2


def test_undeclared_conflicting_verdict_never_publishes_closeout(tmp_path):
    root = tmp_path / "rungs"
    _complete_evidence_root(root)
    rung = root / "N3"
    gate = nest_gates.gates_for("N3")[0]
    verdict = rung / "N3-verdicts.json"
    verdict.write_text(json.dumps({
        "schema": 1, "rung": "N3", gate.metric: {"passed": False},
    }), encoding="utf-8")
    output = tmp_path / "undeclared-out"

    result = subprocess.run([
        sys.executable, str(Path(evidence_pack.__file__)),
        "--rungs-root", str(root), "--outdir", str(output),
    ], check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert "undeclared rung artifacts: N3/N3-verdicts.json" in result.stderr
    assert not output.exists()
    assert not (output / "evidence-pack.json").exists()

    diagnostic_output = tmp_path / "undeclared-diagnostic"
    assert evidence_pack.main([
        "--rungs-root", str(root), "--outdir", str(diagnostic_output),
        "--diagnostic-incomplete",
    ]) == 2
    diagnostic = json.loads(
        (diagnostic_output / "evidence-pack.json").read_text("utf-8"))
    assert diagnostic["closeout_ready"] is False
    assert "undeclared rung artifacts: N3/N3-verdicts.json" in diagnostic["blockers"]
