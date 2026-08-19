"""``rw_ensbatch`` / ``rw_obsgrid``: the seam, the contract, the artifacts.

The two binaries that close the render law's remaining holes -- ensemble
products (``gpuwm/da/enprod.py`` said outright there was "no second engine
to switch to") and gridded radar observations.

Like ``tests/test_render_rust.py``, nothing here mocks an engine: the
end-to-end tests drive the real executables this checkout built, and skip
honestly when it has not built them.  What IS stubbed is the subprocess
layer of the resolver's decision table, because a stale-build refusal
cannot be produced without shipping a stale build.
"""

from __future__ import annotations

import datetime
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import rustwx_lanes

_NZ, _NY, _NX = 3, 16, 20
_STAMP = "2026-07-28_20:00:00"


def _built(name: str) -> Path | None:
    from gpuwm.bridges import executable_name

    path = (rustwx_lanes.crate_dir() / "target" / "release"
            / executable_name(name))
    return path if path.is_file() else None


_ENSEMBLE = _built(rustwx_lanes.ENSEMBLE_NAME)
_OBSGRID = _built(rustwx_lanes.OBSGRID_NAME)
_BUILD_HINT = ("this checkout has not built the rustwx workspace (cd "
               "tools/rustwx && cargo build --release --locked --offline)")

needs_ensemble = pytest.mark.skipif(_ENSEMBLE is None, reason=_BUILD_HINT)
needs_obsgrid = pytest.mark.skipif(_OBSGRID is None, reason=_BUILD_HINT)


# ---------------------------------------------------------------------------
# The contract markers, against the real binaries
# ---------------------------------------------------------------------------

@needs_ensemble
def test_the_ensemble_marker_is_the_built_binary_s_own_answer():
    """The Python constant against the Rust writer, not against itself.

    A marker both halves read from one Python string proves nothing.  The
    renderer incident (task #106) was exactly this: two builds with
    different md5s both reported ``verified`` because nothing asked them
    what grammar they spoke.
    """

    probe = subprocess.run([str(_ENSEMBLE), "--abi"], capture_output=True,
                           text=True, errors="replace", timeout=60)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == rustwx_lanes.ENSEMBLE_ABI_MARKER


@needs_obsgrid
def test_the_obsgrid_marker_is_the_built_binary_s_own_answer():
    probe = subprocess.run([str(_OBSGRID), "--abi"], capture_output=True,
                           text=True, errors="replace", timeout=60)
    assert probe.returncode == 0, probe.stderr
    assert probe.stdout.strip() == rustwx_lanes.OBSGRID_ABI_MARKER


def test_no_two_bundled_binaries_pin_the_same_marker():
    """The renderer was the odd one out once; nothing may be again.

    Five binaries now answer ``--abi``.  If two of them pinned the same
    line, one would verify against the other's contract and a bundle
    could ship the wrong pair with every probe green.
    """

    from gpuwm import rustwx, rustwx_fetch
    from gpuwm.obs import nexrad

    markers = {
        "rw_fetch": rustwx_fetch.FETCH_ABI_MARKER,
        "rw_nexrad": nexrad.NEXRAD_ABI_MARKER,
        "rw_wrfbatch": rustwx.RENDERER_ABI_MARKER,
        "rw_ensbatch": rustwx_lanes.ENSEMBLE_ABI_MARKER,
        "rw_obsgrid": rustwx_lanes.OBSGRID_ABI_MARKER,
    }
    for name, marker in markers.items():
        assert isinstance(marker, str) and marker.strip(), name
        assert "\t" in marker, f"{name}: not a tab-separated contract line"
    assert len(set(markers.values())) == len(markers), markers


def test_a_binary_that_predates_the_handshake_is_refused(monkeypatch,
                                                         tmp_path):
    """Exit 2 with no ``--abi`` line: every build older than the contract."""

    from gpuwm import bridges

    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))
    monkeypatch.setattr(
        rustwx_lanes.subprocess, "run",
        lambda *a, **k: SimpleNamespace(returncode=2, stdout="", stderr=""))
    ok, evidence = rustwx_lanes.probe_ensemble_bin(tmp_path / "rw_ensbatch")
    assert ok is False
    assert "--abi does not match the contract" in evidence
    assert "cargo build" in evidence


def test_a_binary_answering_the_right_contract_is_accepted(monkeypatch,
                                                           tmp_path):
    """The negative control: the guard does not fire on the right answer."""

    from gpuwm import bridges

    monkeypatch.setattr(bridges, "launchable", lambda path: (True, "ok"))
    monkeypatch.setattr(
        rustwx_lanes.subprocess, "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stdout=rustwx_lanes.OBSGRID_ABI_MARKER + "\n",
            stderr=""))
    ok, evidence = rustwx_lanes.probe_obsgrid_bin(tmp_path / "rw_obsgrid")
    assert ok is True, evidence


def test_an_override_naming_a_missing_file_is_a_hard_error(monkeypatch,
                                                           tmp_path):
    """Explicit configuration fails loudly, never falls through.

    Falling through would put the panels on matplotlib while the operator
    believed they were on the Rust engine, which is the render law
    violated silently.
    """

    missing = tmp_path / "nope.exe"
    monkeypatch.setenv(rustwx_lanes.ENSEMBLE_ENV, str(missing))
    with pytest.raises(FileNotFoundError, match="names a missing file"):
        rustwx_lanes.find_ensemble_bin()


# ---------------------------------------------------------------------------
# The engine resolver `gpuwm enprod --engine` uses
# ---------------------------------------------------------------------------

def test_an_explicit_rust_request_refuses_where_auto_degrades(monkeypatch):
    """The pair the render-engine defect was invisible to.

    With a stale binary staged, ``auto`` must fall back naming the
    mismatch and ``rust`` must refuse.  Asserting only the refusal would
    leave the fallback free to become a refusal too.
    """

    from gpuwm.da.enprod import resolve_enprod_engine

    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin",
                        lambda: Path("staged-rw_ensbatch"))
    monkeypatch.setattr(rustwx_lanes, "probe_ensemble_bin",
                        lambda path: (False, "abi mismatch"))
    engine, why = resolve_enprod_engine("auto")
    assert engine == "matplotlib"
    assert "abi mismatch" in why
    with pytest.raises(RuntimeError, match="abi mismatch"):
        resolve_enprod_engine("rust")

    # ... and the fallback is never probed at all.
    def explode(path):
        raise AssertionError("--engine matplotlib must not probe the engine")

    monkeypatch.setattr(rustwx_lanes, "probe_ensemble_bin", explode)
    assert resolve_enprod_engine("matplotlib")[0] == "matplotlib"


def test_an_unbuilt_engine_names_the_build_line(monkeypatch):
    from gpuwm.da.enprod import resolve_enprod_engine

    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin", lambda: None)
    engine, why = resolve_enprod_engine("auto")
    assert engine == "matplotlib"
    assert "cargo build" in why


# ---------------------------------------------------------------------------
# End to end, against the real binaries
# ---------------------------------------------------------------------------

def _png_size(path: Path) -> tuple[int, int]:
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    assert data[12:16] == b"IHDR"
    return struct.unpack(">II", data[16:24])


@pytest.fixture(scope="module")
def ensemble_root(tmp_path_factory) -> Path:
    """The project's OWN synthetic ensemble writer, not a local fixture.

    Cross-lane rule: the reader is exercised against the other lane's
    real writer.  ``write_synthetic_ensemble`` is what ``gpuwm enprod
    --make-fixture`` produces, so a fixture that reads here is a fixture
    an operator can produce.
    """

    from gpuwm.da.enprod import write_synthetic_ensemble

    root = tmp_path_factory.mktemp("ens") / "ensemble"
    write_synthetic_ensemble(root, n_members=4)
    return root


@needs_ensemble
def test_the_ensemble_engine_draws_all_five_products(ensemble_root,
                                                     tmp_path):
    from gpuwm.da.enprod import MANIFEST_FILENAME

    out = tmp_path / "png"
    written, failures, skipped, report = rustwx_lanes.run_ensemble_renderer(
        _ENSEMBLE, ensemble_root / MANIFEST_FILENAME,
        store_root=tmp_path / "store", out_dir=out, field="refl",
        products=",".join(rustwx_lanes.ENSEMBLE_PRODUCTS), threshold=35.0,
        neighborhood_km=9.0, width=640, height=480)
    assert failures == [], failures
    assert len(written) == len(rustwx_lanes.ENSEMBLE_PRODUCTS), written
    assert skipped == []
    for path in written:
        assert path.is_file(), path
        assert path.stat().st_size > 5_000, path
        assert _png_size(path) == (640, 480), path

    # The panel stamps its own denominator; the report is that stamp, read
    # off the engine rather than recomputed, so a caption and a
    # provenance record cannot disagree.
    assert "members" in report and "n=4" in report["members"]
    assert "coverage" in report and "coverage=1.0000" in report["coverage"]
    assert "pmm_ties" in report
    # One stable colour per member NUMBER, which is the only label a
    # paintball plot has.
    legend = report["paintball_legend"]
    assert legend.split()[0].startswith("0=#")
    assert len(set(legend.split())) == 4


@needs_ensemble
def test_two_neighborhood_radii_are_two_files(ensemble_root, tmp_path):
    """A neighborhood is part of the product, so it is part of the name.

    MEASURED before this test existed: ``gpuwm enprod --threshold 25
    --neighborhood-km 0,9,27`` reported ``3 panel(s)`` and left ONE file
    on disk.  The CLI takes a LIST of radii and drives the engine once
    per radius, and the engine named all of them
    ``ens_prob_refl_p25dbz`` -- so the 9 km field overwrote the point
    probability and the 27 km field overwrote that.  The matplotlib
    engine has always carried the ``r5km`` token; this is the Rust route
    catching up to it.
    """

    from gpuwm.da.enprod import MANIFEST_FILENAME, radius_slug

    manifest = ensemble_root / MANIFEST_FILENAME
    produced: list[Path] = []
    for radius_km in (0.0, 9.0):
        written, failures, _skipped, _report = (
            rustwx_lanes.run_ensemble_renderer(
                _ENSEMBLE, manifest, store_root=tmp_path / "store",
                out_dir=tmp_path / "png", field="refl", products="prob",
                threshold=35.0, neighborhood_km=radius_km,
                width=320, height=240))
        assert failures == [], failures
        assert len(written) == 1, written
        produced.append(written[0])

    assert produced[0] != produced[1], produced
    assert produced[0].is_file() and produced[1].is_file()
    # And the token is the one the other engine writes, so the two
    # engines' filenames stay readable as the same grammar.
    assert radius_slug(9.0) in produced[1].name, produced[1].name
    assert radius_slug(9.0) not in produced[0].name, produced[0].name


@needs_ensemble
def test_the_render_layout_ruling_is_applied_at_write_time(ensemble_root,
                                                           tmp_path):
    """``<out>/<domain>/<product>/<valid-day>/`` -- never flat."""

    from gpuwm.da.enprod import MANIFEST_FILENAME

    out = tmp_path / "png"
    written, failures, _skipped, _report = (
        rustwx_lanes.run_ensemble_renderer(
            _ENSEMBLE, ensemble_root / MANIFEST_FILENAME,
            store_root=tmp_path / "store", out_dir=out, field="refl",
            products="mean", width=400, height=300))
    assert failures == [], failures
    assert not list(out.glob("*.png")), "the layout ruling was inverted flat"
    relative = written[0].relative_to(out).parts
    assert len(relative) == 4, relative
    assert relative[0].startswith("ens")
    assert relative[1] == "ens_mean_refl"


@needs_ensemble
def test_an_unknown_field_is_refused_with_the_vocabulary(ensemble_root,
                                                         tmp_path):
    from gpuwm.da.enprod import MANIFEST_FILENAME

    result = subprocess.run(
        [str(_ENSEMBLE), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(tmp_path / "png"), "--manifest",
         str(ensemble_root / MANIFEST_FILENAME), "--field", "not_a_field"],
        capture_output=True, text=True, errors="replace", timeout=120)
    assert result.returncode == 2, result.stdout
    assert "not_a_field" in result.stderr
    assert "--list-fields" in result.stderr


@pytest.fixture(scope="module")
def radar_grid(tmp_path_factory) -> Path:
    """One real ``gpuwm-obs.radar-grid.v1`` file from the real writer."""

    pytest.importorskip("netCDF4", reason="the obs writer needs netCDF4")
    from gpuwm.obs.radar_grid import write_radar_grid
    from gpuwm.obs.sweeps import Moment, RadarSite, RadarVolume, Sweep
    from gpuwm.obs.superob import (SuperobParams, merge_contributions,
                                   superob_volume)
    from gpuwm.obs.target_grid import TargetGrid
    from gpuwm.static.lambert import LambertGrid

    projection = LambertGrid(
        ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.2778, dx=3000.0, dy=3000.0, e_we=25, e_sn=21)
    grid = TargetGrid.from_projection(
        projection, z_w=np.linspace(0.0, 10000.0, 9), name="lanes-test")
    gates = 40
    azimuths = np.arange(0.0, 360.0, 4.0, dtype=np.float32)
    sweep = Sweep(
        sweep_index=0, elevation_number=1, elevation_angle_deg=0.5,
        nyquist_velocity_ms=32.0, start_status=3, end_status=2,
        cut_sector=0, complete=True, azimuth_deg=azimuths,
        elevation_deg=np.full(azimuths.size, 0.5, dtype=np.float32),
        moments={
            "REF": Moment("REF", "dBZ", gates, 2125.0, 250.0, np.tile(
                np.linspace(10.0, 55.0, gates, dtype=np.float32)[None, :],
                (azimuths.size, 1))),
            "VEL": Moment("VEL", "m/s", gates, 2125.0, 250.0, np.tile(
                np.linspace(-20.0, 20.0, gates, dtype=np.float32)[None, :],
                (azimuths.size, 1))),
        })
    centre_j, centre_i = grid.ny // 2, grid.nx // 2
    volume = RadarVolume(
        site=RadarSite(id="KTLX", name="test",
                       lat_deg=float(grid.lat[centre_j, centre_i]),
                       lon_deg=float(grid.lon[centre_j, centre_i]),
                       alt_m=380.0, source="test"),
        valid_time="2026-07-28T20:03:16Z", station_id="KTLX",
        volume_file="KTLX20260728_200316_V06", volume_sha256="0" * 64,
        volume_bytes=1, pack_path=Path("test.rdrpack"), pack_sha256="1" * 64,
        params={"moments": ["REF", "VEL"], "max_range_km": 250.0},
        framing={"magic": "AR2V0006", "block_count": 1}, sweeps=(sweep,))
    params = SuperobParams()
    observations = merge_contributions(
        [superob_volume(volume, grid, params=params)], grid, params=params,
        z_reduce="max")
    path = tmp_path_factory.mktemp("obs") / "radar-grid.nc"
    write_radar_grid(path, observations, grid,
                     valid_time="2026-07-28T20:03:16Z", params=params,
                     overwrite=True)
    return path


@needs_obsgrid
def test_the_obs_engine_reads_the_schema_natively(radar_grid, tmp_path):
    """No fake wrfout in the middle.

    Wrapping observations in a wrfout would make the renderer's
    ``TitleProvenance::LocalImport`` state that a model produced them,
    which is a lie on every panel.  This asserts the file is read as what
    it is: four gridded products and the site roster, off one NetCDF.
    """

    out = tmp_path / "png"
    written, failures, skipped, sites = rustwx_lanes.run_obsgrid_renderer(
        _OBSGRID, radar_grid, out_dir=out, products="all", sites=True,
        rings_km="50,100", width=640, height=480)
    assert failures == [], failures
    assert len(written) == 4, (written, skipped)
    for path in written:
        assert path.is_file() and path.stat().st_size > 5_000, path
        assert _png_size(path) == (640, 480), path
    # The site identifier is the file's own, resolved and position-checked
    # against `radar_lat`/`radar_lon` -- not a positional placeholder.
    assert sites == [{"index": 0, "id": "KTLX",
                      "lat": pytest.approx(35.34, abs=0.05),
                      "lon": pytest.approx(-97.27, abs=0.05)}]


@needs_obsgrid
def test_a_file_that_declares_another_schema_is_refused_by_name(tmp_path):
    """An undeclared or foreign layout is refused, never read hopefully.

    Every variable this reader addresses is a fixed name in a fixed
    dimension order, and a file with the right names in another order has
    the right shapes and the wrong field.
    """

    netCDF4 = pytest.importorskip("netCDF4")

    path = tmp_path / "not-a-radar-grid.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as dataset:
        dataset.setncattr("schema", "some-other-thing.v9")
        dataset.createDimension("south_north", 2)
        dataset.createDimension("west_east", 2)
    result = subprocess.run(
        [str(_OBSGRID), "--obs", str(path), "--out-dir", str(tmp_path / "o")],
        capture_output=True, text=True, errors="replace", timeout=120)
    assert result.returncode != 0
    assert "gpuwm-obs.radar-grid.v1" in result.stderr
    assert "some-other-thing.v9" in result.stderr


# ---------------------------------------------------------------------------
# The overlay/annotation seam on the EXISTING renderer
# ---------------------------------------------------------------------------

def _fixture_wrfout(path: Path) -> Path:
    from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs

    rng = np.random.default_rng(11)
    lat = np.tile(np.linspace(38.0, 40.0, _NY)[:, None], (1, _NX))
    lon = np.tile(np.linspace(-98.0, -95.0, _NX)[None, :], (_NY, 1))
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(2026, 7, 28, 20), grid_id=2, parent_id=1,
        i_parent_start=5, j_parent_start=5, parent_grid_ratio=3, dt=6.0)
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                      global_attrs=attrs) as writer:
        writer.write_frame(_STAMP, {
            "T": np.zeros((_NZ, _NY, _NX), np.float32),
            "MU": np.zeros((_NY, _NX), np.float32),
            "REFL_10CM": rng.uniform(-20.0, 65.0,
                                     (_NZ, _NY, _NX)).astype(np.float32),
            "T2": rng.uniform(280.0, 300.0, (_NY, _NX)).astype(np.float32),
            "XLAT": lat.astype(np.float32),
            "XLONG": lon.astype(np.float32),
            "HGT": np.zeros((_NY, _NX), np.float32),
            "SINALPHA": np.zeros((_NY, _NX), np.float32),
            "COSALPHA": np.ones((_NY, _NX), np.float32),
        })
    return path


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_overlays_change_the_pixels_and_their_absence_does_not(tmp_path):
    """Both directions, which is what makes this a gate and not a demo.

    * WITH ``--overlays``, the PNG differs -- the flag is not accepted and
      ignored, which is exactly how an opt-in overlay would silently do
      nothing;
    * WITHOUT it, the PNG is byte-identical to one drawn with no flag at
      all, which is the standing pixel-identity claim for every product
      ``rw_wrfbatch`` already draws.
    """

    import hashlib
    import json

    renderer = _built("rw_wrfbatch")
    wrfout = _fixture_wrfout(tmp_path / "wrfout_d02_2026-07-28_20-00-00.nc")
    overlays = tmp_path / "overlays.json"
    overlays.write_text(json.dumps({
        "lines": [{"points": [[38.3, -97.6], [38.3, -95.6],
                              [39.7, -95.6], [39.7, -97.6]],
                   "closed": True, "color": "#c02020", "width": 3}],
        "points": [{"lat": 39.2, "lon": -96.9, "shape": "circle",
                    "radius_px": 8}],
        "rings": [{"lat": 39.0, "lon": -96.5, "radii_km": [40]}],
    }), encoding="utf-8")

    def render(tag: str, extra: list[str]) -> Path:
        out = tmp_path / f"out_{tag}"
        command = [
            str(renderer), "--store-root", str(tmp_path / f"store_{tag}"),
            "--out-dir", str(out), "--products", "composite_reflectivity",
            "--frames", "0", "--width", "500", "--height", "400",
            "--source-label", "ArWen", *extra, str(wrfout)]
        result = subprocess.run(command, capture_output=True, text=True,
                                errors="replace", timeout=300)
        assert result.returncode == 0, result.stdout + result.stderr
        pngs = sorted(out.rglob("*.png"))
        assert len(pngs) == 1, pngs
        return pngs[0]

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    plain = digest(render("plain", []))
    again = digest(render("again", []))
    marked = digest(render("marked", ["--overlays", str(overlays)]))

    assert plain == again, (
        "two flagless renders of one file differ; the pixel-identity "
        "claim is not even reproducible")
    assert marked != plain, (
        "--overlays was accepted and drew nothing, which is the failure "
        "mode an opt-in overlay flag has")


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_the_driver_forwards_the_overlay_flags(monkeypatch, tmp_path):
    """``gpuwm.rustwx.run_renderer`` reaches the new surface."""

    from gpuwm import rustwx

    seen: list[list[str]] = []

    class Result:
        returncode = 0
        stdout = ""
        stderr = ""

    monkeypatch.setattr(rustwx.subprocess, "run",
                        lambda command, **kwargs: (
                            seen.append([str(part) for part in command])
                            or Result()))
    rustwx.run_renderer(
        tmp_path / "rw_wrfbatch", tmp_path / "wrfout_d02_x.nc",
        store_root=tmp_path / "store", out_dir=tmp_path / "png",
        products="all", frames="all", width=800, height=600,
        overlays=tmp_path / "overlays.json",
        annotate=tmp_path / "annotate.json")
    command = seen[-1]
    assert command[command.index("--overlays") + 1] == str(
        tmp_path / "overlays.json")
    assert command[command.index("--annotate") + 1] == str(
        tmp_path / "annotate.json")

    # ...and omits them entirely when they are not asked for, because a
    # flag that is always present is a flag that always runs its code.
    seen.clear()
    rustwx.run_renderer(
        tmp_path / "rw_wrfbatch", tmp_path / "wrfout_d02_x.nc",
        store_root=tmp_path / "store", out_dir=tmp_path / "png",
        products="all", frames="all", width=800, height=600)
    assert "--overlays" not in seen[-1]
    assert "--annotate" not in seen[-1]


# ---------------------------------------------------------------------------
# P0: the two npz lanes' sibling wrfouts
# ---------------------------------------------------------------------------

def test_a_snapshot_without_coordinates_is_refused_by_name(tmp_path):
    """No guessing a grid from an array shape."""

    from gpuwm.io.surface_wrfout import (SurfaceSnapshotRefusal,
                                         write_surface_wrfout)

    with pytest.raises(SurfaceSnapshotRefusal, match="no latitude/longitude"):
        write_surface_wrfout(tmp_path / "wrfout_x.nc",
                             {"T2": np.zeros((4, 5), np.float32)},
                             time_str=_STAMP, dx=1000.0)


def test_a_field_on_a_different_grid_than_its_coordinates_is_refused(
        tmp_path):
    from gpuwm.io.surface_wrfout import (SurfaceSnapshotRefusal,
                                         write_surface_wrfout)

    snapshot = {
        "XLAT": np.zeros((4, 5), np.float32),
        "XLONG": np.zeros((4, 5), np.float32),
        "T2": np.zeros((5, 4), np.float32),
    }
    with pytest.raises(SurfaceSnapshotRefusal, match=r"T2 is \(5, 4\)"):
        write_surface_wrfout(tmp_path / "wrfout_x.nc", snapshot,
                             time_str=_STAMP, dx=1000.0)


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_a_surface_snapshot_wrfout_renders_through_the_real_renderer(
        tmp_path):
    """The whole point of P0, proven against the artifact.

    Two lanes published ``.npz`` and nothing else, so the production
    renderer could not draw them at all.  This writes the same kind of
    snapshot ``tilestream.bigdomain.snapshot`` produces and puts it
    through the real ``rw_wrfbatch``.
    """

    from gpuwm.io.surface_wrfout import (SURFACE_SNAPSHOT, VERTICAL_ATTR,
                                         write_surface_wrfout)
    from gpuwm.io.wrfout import wrf_global_attrs

    ny, nx = 24, 30
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    core = 62.0 * np.exp(-(((yy - 12) / 4.0) ** 2
                           + ((xx - 15) / 5.0) ** 2)) - 12.0
    snapshot = {
        "LAT": np.tile(np.linspace(34.4, 36.3, ny)[:, None],
                       (1, nx)).astype(np.float32),
        "LON": np.tile(np.linspace(-98.4, -96.1, nx)[None, :],
                       (ny, 1)).astype(np.float32),
        "HT": np.full((ny, nx), 350.0, np.float32),
        "T2": (288.0 + 6.0 * core / 60.0).astype(np.float32),
        "U10": (8.0 * np.sin(xx / 7.0)).astype(np.float32),
        "V10": (8.0 * np.cos(yy / 7.0)).astype(np.float32),
        "PSFC": np.full((ny, nx), 97000.0, np.float32),
        "RAINNC": np.clip(core * 0.6, 0.0, None).astype(np.float32),
        "RAINC": np.zeros((ny, nx), np.float32),
        "REFL_COMPOSITE": core.astype(np.float32),
    }
    grid = SimpleNamespace(truelat1=33.0, truelat2=37.0, stand_lon=-97.2778,
                           ref_lat=35.3331, ref_lon=-97.2778)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(2026, 7, 28, 20), grid_id=1, parent_id=1,
        i_parent_start=1, j_parent_start=1, parent_grid_ratio=1, dt=6.0)
    path = write_surface_wrfout(
        tmp_path / "wrfout_d01_bigdom_f000.nc", snapshot, time_str=_STAMP,
        dx=2000.0, global_attrs=attrs, grid_id=1)

    # The file says what its vertical axis is, so a reader looking for a
    # sounding finds the answer in the file rather than in a wrong plot.
    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(path) as dataset:
        assert getattr(dataset, VERTICAL_ATTR) == SURFACE_SNAPSHOT
        assert dataset.dimensions["bottom_top"].size == 1

    out = tmp_path / "png"
    result = subprocess.run(
        [str(_built("rw_wrfbatch")), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(out), "--products", "all", "--frames", "all",
         "--width", "500", "--height", "400", str(path)],
        capture_output=True, text=True, errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert any("composite_reflectivity" in name for name in produced), produced
    assert any("2m_temperature" in name for name in produced), produced
    assert any("10m_wind_speed" in name for name in produced), produced


# ---------------------------------------------------------------------------
# The run origin: what the CALLER's own invocation writes
# ---------------------------------------------------------------------------
#
# The test above supplies ``wrf_global_attrs``, so it never asked what the
# two shipping callers actually pass.  ``tilestream/run_bigdomain.py``
# passed no global attributes at all, and the frames it wrote were refused
# by the very renderer this writer exists to feed:
#
#     wrfout_bigdom_256_f020.nc has no sound WRF run origin:
#     START_DATE unavailable; SIMULATION_START_DATE unavailable;
#     XTIME fallback failed: XTIME is unavailable
#
# MEASURED on a real 256x256x33 tile-streamed forecast, 2026-08-17.

def test_a_snapshot_with_no_run_origin_is_refused_before_it_is_written(
        tmp_path):
    """A file the renderer will refuse is not written and called a success.

    Concrete breakage: ``rw_wrfbatch`` computes every product's lead time
    from ``START_DATE``/``SIMULATION_START_DATE`` and falls back to
    ``XTIME``.  A surface snapshot carries none of the three, so without
    an origin the writer produces a file that renders NOTHING -- which is
    what the tile-streamed lane shipped.
    """

    from gpuwm.io.surface_wrfout import (SurfaceSnapshotRefusal,
                                         write_surface_wrfout)

    snapshot = {
        "XLAT": np.zeros((4, 5), np.float32),
        "XLONG": np.zeros((4, 5), np.float32),
    }
    target = tmp_path / "wrfout_x.nc"
    with pytest.raises(SurfaceSnapshotRefusal, match="run origin"):
        write_surface_wrfout(target, snapshot, time_str=_STAMP, dx=1000.0)
    assert not target.exists(), "the refused file must not be left on disk"


def test_a_start_time_becomes_the_run_origin_the_renderer_reads(tmp_path):
    from gpuwm.io.surface_wrfout import write_surface_wrfout

    netCDF4 = pytest.importorskip("netCDF4")
    snapshot = {
        "XLAT": np.zeros((4, 5), np.float32),
        "XLONG": np.zeros((4, 5), np.float32),
    }
    path = write_surface_wrfout(
        tmp_path / "wrfout_x.nc", snapshot, time_str="1970-01-01_00:20:00",
        dx=1000.0, start_time=datetime.datetime(1970, 1, 1))
    with netCDF4.Dataset(path) as dataset:
        assert dataset.START_DATE == "1970-01-01_00:00:00"
        assert dataset.SIMULATION_START_DATE == "1970-01-01_00:00:00"


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_the_tile_streamed_lane_s_own_call_renders(tmp_path):
    """The CALLER's invocation, not a fixture's.

    ``tilestream.run_bigdomain._write_snapshot_wrfout`` is driven exactly
    as ``stage_forecast`` drives it, and the frame it writes goes through
    the real ``rw_wrfbatch``.  Before this test the lane's own frames
    carried ``MAP_PROJ=0`` and no run origin and drew zero products.
    """

    pytest.importorskip("netCDF4")
    from gpuwm.static.lambert import LambertGrid
    from tilestream.run_bigdomain import _write_snapshot_wrfout

    ny, nx = 24, 30
    grid = LambertGrid(e_we=nx + 1, e_sn=ny + 1, dx=3000.0, dy=3000.0,
                       ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0,
                       truelat2=37.0, stand_lon=-97.2778)
    lat, lon = grid.latlon_mass()
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    core = 62.0 * np.exp(-(((yy - 12) / 4.0) ** 2
                           + ((xx - 15) / 5.0) ** 2)) - 12.0
    snapshot = {
        "LAT": np.asarray(lat, np.float32),
        "LON": np.asarray(lon, np.float32),
        "HT": np.full((ny, nx), 350.0, np.float32),
        "T2": (288.0 + 6.0 * core / 60.0).astype(np.float32),
        "U10": (8.0 * np.sin(xx / 7.0)).astype(np.float32),
        "V10": (8.0 * np.cos(yy / 7.0)).astype(np.float32),
        "PSFC": np.full((ny, nx), 97000.0, np.float32),
        "RAINNC": np.clip(core * 0.6, 0.0, None).astype(np.float32),
        "RAINC": np.zeros((ny, nx), np.float32),
        "REFL_COMPOSITE": core.astype(np.float32),
        "elapsed_s": 1200.0,
    }
    cfg = SimpleNamespace(dx=3000.0, dy=3000.0, dt=15.0)
    written = _write_snapshot_wrfout(tmp_path / "bigdom_30_f020.npz",
                                     snapshot, cfg, "f020", grid=grid)
    assert written is not None and written.is_file()

    out = tmp_path / "png"
    result = subprocess.run(
        [str(_built("rw_wrfbatch")), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(out), "--products", "all", "--frames", "all",
         "--width", "500", "--height", "400", str(written)],
        capture_output=True, text=True, errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert any("composite_reflectivity" in name for name in produced), produced


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_the_da_cycle_lane_s_own_call_renders(tmp_path):
    """The other npz lane's invocation, on the same terms.

    ``tools/da_cycle_prepared._write_composite_wrfout`` writes the
    sibling wrfout for a DA leg's composite.  It supplies
    ``wrf_global_attrs``, so unlike the tile-streamed lane it always
    carried a run origin -- this pins that, because the two lanes share a
    writer and a change made for one must not silently take the origin
    away from the other.
    """

    pytest.importorskip("netCDF4")
    from gpuwm.static.lambert import LambertGrid
    from tools.da_cycle_prepared import _write_composite_wrfout

    ny, nx = 20, 26
    grid = LambertGrid(e_we=nx + 1, e_sn=ny + 1, dx=3000.0, dy=3000.0,
                       ref_lat=39.79, ref_lon=-104.55, truelat1=30.0,
                       truelat2=60.0, stand_lon=-104.55)
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    composite = (58.0 * np.exp(-(((yy - 10) / 3.5) ** 2
                                 + ((xx - 13) / 4.5) ** 2)) - 10.0)
    cfg = SimpleNamespace(dx=3000.0, dy=3000.0, dt=6.0)
    experiment = SimpleNamespace(
        start_time=datetime.datetime(2021, 12, 30, 17, 0, 0))
    written = _write_composite_wrfout(
        tmp_path / "leg01_composite.npz", composite, grid, cfg, 900.0,
        experiment, label="leg01")
    assert written is not None and written.is_file()

    netCDF4 = pytest.importorskip("netCDF4")
    with netCDF4.Dataset(written) as dataset:
        assert dataset.START_DATE == "2021-12-30_17:00:00"

    out = tmp_path / "png"
    result = subprocess.run(
        [str(_built("rw_wrfbatch")), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(out), "--products", "all", "--frames", "all",
         "--width", "500", "--height", "400", str(written)],
        capture_output=True, text=True, errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert any("composite_reflectivity" in name for name in produced), produced


@pytest.mark.skipif(_built("rw_wrfbatch") is None, reason=_BUILD_HINT)
def test_the_streamed_lane_s_updraft_reaches_a_production_panel(tmp_path):
    """The column extremes draw, under WRF's own names for them.

    ``tilestream/bigdomain_render.py``'s ``wmax`` panel was the last
    weather field of that lane with no production renderer: a surface
    snapshot has no profile, so the column maximum updraft is the only
    way the storm's engine reaches a picture at all.

    The snapshot carries ``WMAX``/``WMIN``; the writer publishes them as
    ``W_UP_MAX``/``W_DN_MAX`` -- the Registry's names, so a reader who
    knows wrfout knows them without being told -- and the importer's
    raw-extras TABLE is what puts them in the store.  That table already
    carries ``WSPD10MAX`` and ``UP_HELI_MAX``, the same family of
    column/period maxima, so this is a row, not a code path.

    The divergence from WRF is the averaging WINDOW, and the file states
    it rather than leaving a reader to assume: WRF's are running maxima
    between history writes; these are the frame's own instant.
    """

    netCDF4 = pytest.importorskip("netCDF4")
    from gpuwm.io.surface_wrfout import W_EXTREME_ATTR
    from gpuwm.static.lambert import LambertGrid
    from tilestream.run_bigdomain import _write_snapshot_wrfout

    ny, nx = 24, 30
    grid = LambertGrid(e_we=nx + 1, e_sn=ny + 1, dx=3000.0, dy=3000.0,
                       ref_lat=35.3331, ref_lon=-97.2778, truelat1=33.0,
                       truelat2=37.0, stand_lon=-97.2778)
    lat, lon = grid.latlon_mass()
    yy, xx = np.mgrid[0:ny, 0:nx].astype(np.float32)
    bell = np.exp(-(((yy - 12) / 4.0) ** 2 + ((xx - 15) / 5.0) ** 2))
    snapshot = {
        "LAT": np.asarray(lat, np.float32),
        "LON": np.asarray(lon, np.float32),
        "REFL_COMPOSITE": (62.0 * bell - 12.0).astype(np.float32),
        "WMAX": (24.0 * bell).astype(np.float32),
        "WMIN": (-9.0 * bell).astype(np.float32),
        "elapsed_s": 1200.0,
    }
    cfg = SimpleNamespace(dx=3000.0, dy=3000.0, dt=15.0)
    written = _write_snapshot_wrfout(tmp_path / "bigdom_30_f020.npz",
                                     snapshot, cfg, "f020", grid=grid)
    assert written is not None and written.is_file()

    with netCDF4.Dataset(written) as dataset:
        assert "W_UP_MAX" in dataset.variables
        assert "W_DN_MAX" in dataset.variables
        # The window divergence is IN the file, not only in a docstring.
        assert "INSTANTANEOUS" in getattr(dataset, W_EXTREME_ATTR)

    out = tmp_path / "png"
    result = subprocess.run(
        [str(_built("rw_wrfbatch")), "--store-root", str(tmp_path / "store"),
         "--out-dir", str(out), "--products", "all", "--frames", "all",
         "--width", "500", "--height", "400", str(written)],
        capture_output=True, text=True, errors="replace", timeout=600)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    produced = sorted(p.name for p in out.rglob("*.png"))
    assert any("w_up_max" in name for name in produced), produced
    assert any("w_dn_max" in name for name in produced), produced
