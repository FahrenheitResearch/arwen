"""The initial PBLH comes from the source's own PBL height when it has one.

Breakage this prevents (2026-10-21, full-CONUS HRRR 06Z): the initial
history wrote PBLH = 0 at every point, so the f00 10 m gust (and every
PBL-depth product) was computed with no boundary layer, while NOAA's f00 --
a cycled state whose wrfinput carries the background PBLH (HRRR wrfnat
HPBL, mean 347 m that hour) -- had one.  WOOF's f00 gust came out 2.6 m/s
weaker than the same wrf-rust gust run on NOAA's own f00 state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.core.landuse import LanduseInitialization
from gpuwm.ingest import hrrr_physics
from gpuwm.ingest.hrrr import load_hrrr_native_window
from gpuwm.ingest.native_supplements import gate_optional_surface_fields
from gpuwm.ingest.prepared_cache import select_prepared_met_fields
from gpuwm.ingest.hrrr import _ATMOSPHERE_2D, _ATMOSPHERE_3D


def _landuse(shape=(2, 3)):
    zeros = np.zeros(shape, dtype=np.float32)
    return LanduseInitialization(
        season=1, landmask=zeros + 1, xland=zeros + 1, lakemask=zeros,
        ivgtyp=np.ones(shape, dtype=np.int32), isltyp=np.ones(shape, dtype=np.int32),
        snowc=zeros, pblh=zeros.copy(), ust=zeros + 1e-4, mavail=zeros + 1,
        z0=zeros + 0.1, znt=zeros + 0.1, albbck=zeros + 0.2, albedo=zeros + 0.2,
        embck=zeros + 0.95, emiss=zeros + 0.95)


def test_a_source_pbl_height_seeds_the_initial_pblh(capsys):
    hrrr_physics._PBLH_LOGGED.clear()
    pblh = np.array([[0.0, 120.5, 347.0], [1500.0, 2.0, 80.0]], dtype=np.float32)
    seeded = hrrr_physics._seeded_initial_pblh(
        _landuse(), SimpleNamespace(fields={"PBLH": pblh}))
    np.testing.assert_array_equal(seeded.pblh, pblh)
    assert seeded.pblh.dtype == np.float32
    assert "seeded from the initial source's own PBL height" in capsys.readouterr().out


def test_a_source_without_one_keeps_the_stock_zero_and_says_so(capsys):
    hrrr_physics._PBLH_LOGGED.clear()
    landuse = _landuse()
    kept = hrrr_physics._seeded_initial_pblh(landuse, SimpleNamespace(fields={}))
    assert kept is landuse
    assert not kept.pblh.any()
    assert "PBLH starts at 0 everywhere (stock WRF cold start)" in capsys.readouterr().out


@pytest.mark.parametrize("bad, match", [
    (np.zeros((3, 2), dtype=np.float32), "shape"),
    (np.full((2, 3), np.nan, dtype=np.float32), "finite"),
    (np.full((2, 3), 2.0e4, dtype=np.float32), "outside 0..10000 m"),
])
def test_a_source_pbl_height_that_cannot_be_right_is_refused(bad, match):
    with pytest.raises(ValueError, match=match):
        hrrr_physics._seeded_initial_pblh(
            _landuse(), SimpleNamespace(fields={"PBLH": bad}))


def test_the_prepared_met_subset_carries_the_source_pblh():
    shape = (2, 3)
    fields = {name: np.ones(shape, dtype=np.float32)
              for name in ("LANDSEA", "SKINTEMP", "T2", "U10", "V10", "SOILT", "SOILW")}
    fields["PBLH"] = np.full(shape, 250.0, dtype=np.float32)
    selected = select_prepared_met_fields(SimpleNamespace(fields=fields))
    np.testing.assert_array_equal(selected.fields["PBLH"], fields["PBLH"])


def test_the_gate_declares_pblh_in_metres_and_nothing_else():
    assert gate_optional_surface_fields({}) == ()
    assert gate_optional_surface_fields({
        "optional_surface_fields": "PBLH",
        "optional_surface_units": "PBLH=m"}) == ("PBLH",)
    for gate in ({"optional_surface_fields": "PBLH", "optional_surface_units": "PBLH=km"},
                 {"optional_surface_fields": "GUST", "optional_surface_units": "GUST=m"},
                 {"optional_surface_fields": "PBLH,PBLH",
                  "optional_surface_units": "PBLH=m,PBLH=m"}):
        with pytest.raises(ValueError, match="PBLH in m"):
            gate_optional_surface_fields(gate)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bridge_with_pblh(root: Path) -> str:
    """A one-point bridge publication whose decoder declared HPBL."""
    root.mkdir()
    (root / "gate.txt").write_text(
        "status\tPASS\n"
        "cycle\t2026-07-18 00:00:00\n"
        "atmosphere_selected_per_time\t562\n"
        "hybrid_levels\t50\n"
        "optional_surface_fields\tPBLH\n"
        "optional_surface_units\tPBLH=m\n"
        "soil_selected_per_time\t18\n"
        "window_zero_based_inclusive\ti=10..10 j=20..20\n"
        "window_shape\t1x1\n"
        "qice_mapping\tPASS discipline=0 category=1 parameter=82 level_type=105\n"
        "cross_time_inventory\tPASS exact selected keys/levels/grid\n")
    for hour in (0, 1):
        atmosphere = root / f"atmosphere-f{hour:02d}"
        soil = root / f"soil-f{hour:02d}"
        atmosphere.mkdir()
        soil.mkdir()
        for name in _ATMOSPHERE_3D:
            np.ones((50, 1, 1), dtype="<f4").tofile(atmosphere / f"{name}.f32le")
        for name in _ATMOSPHERE_2D:
            np.ones((1, 1), dtype="<f4").tofile(atmosphere / f"{name}.f32le")
        np.full((1, 1), 347.0 + hour, dtype="<f4").tofile(atmosphere / "PBLH.f32le")
        for name in ("SOILT", "SOILW"):
            np.ones((9, 1, 1), dtype="<f4").tofile(soil / f"{name}.f32le")
    payloads = sorted(path for path in root.rglob("*") if path.is_file())
    (root / "SHA256SUMS").write_text("\n".join(
        f"{_hash(path)}  ./{path.relative_to(root).as_posix()}" for path in payloads) + "\n")
    return _hash(root / "SHA256SUMS")


def test_the_native_loader_reads_the_declared_pbl_height(tmp_path):
    root = tmp_path / "bridge"
    manifest = _bridge_with_pblh(root)
    snapshot = load_hrrr_native_window(root, 1, expected_manifest_sha256=manifest)
    assert snapshot.fields["PBLH"].shape == (1, 1)
    assert float(snapshot.fields["PBLH"][0, 0]) == 348.0
    # The count is held: a gate declaring HPBL with 561 records is refused.
    gate = root / "gate.txt"
    gate.write_text(gate.read_text().replace("\t562\n", "\t561\n"))
    lines = [f"{_hash(path)}  ./{path.relative_to(root).as_posix()}"
             for path in sorted(p for p in root.rglob("*")
                                if p.is_file() and p.name != "SHA256SUMS")]
    (root / "SHA256SUMS").write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="atmosphere_selected_per_time"):
        load_hrrr_native_window(root, 1, expected_manifest_sha256=_hash(root / "SHA256SUMS"))


def test_the_initial_cldfra_is_the_radiations_own_cloud_fraction(monkeypatch):
    """f00 CLDFRA is seeded with radiation's own cal_cldfra1 on the start.

    Breakage it prevents: the initial history wrote CLDFRA = 0 everywhere
    (the field is first written by the radiation call), so every f00 cloud
    product was blank (2026-10-21 06Z: column-max cloud fraction 0 at every
    point against NOAA's 54 % mean).
    """
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)
    except Exception:  # no usable device
        pytest.skip("no CUDA device")
    from gpuwm.core import physics as physics_module
    from gpuwm.core.rrtmgp import cal_cldfra1

    nz, ny, nx = 3, 2, 2
    temperature = cp.full((nz, ny, nx), 270.0, dtype=cp.float32)
    pressure = cp.full((nz, ny, nx), 80_000.0, dtype=cp.float32)
    monkeypatch.setattr(physics_module, "_prepare_atmosphere",
                        lambda state: {"temperature": temperature, "pressure": pressure})
    qv = cp.full((nz, ny, nx), 3.0e-3, dtype=cp.float32)
    qc = cp.zeros((nz, ny, nx), dtype=cp.float32)
    qc[1] = 2.0e-4  # one cloudy layer
    state = SimpleNamespace(qv=qv, qc=qc, qi=cp.zeros_like(qc), qs=cp.zeros_like(qc))
    driver = SimpleNamespace(cldfra=cp.zeros((nz, ny, nx), dtype=cp.float32))
    cfg = SimpleNamespace(icloud=1, mp_physics=28)
    hrrr_physics._seed_initial_cldfra(driver, state, cfg)
    expected = cal_cldfra1(qv.reshape(nz, -1), qc.reshape(nz, -1), state.qi.reshape(nz, -1),
                           state.qs.reshape(nz, -1), temperature.reshape(nz, -1),
                           pressure.reshape(nz, -1), f_qc=True, f_qi=True, f_qs=True,
                           mp_physics=28).reshape(nz, ny, nx)
    cp.testing.assert_array_equal(driver.cldfra, expected)
    assert float(driver.cldfra[1].min()) > 0.0
    # icloud=0 radiates no cloud: the field stays as the driver made it.
    untouched = SimpleNamespace(cldfra=cp.zeros((nz, ny, nx), dtype=cp.float32))
    hrrr_physics._seed_initial_cldfra(untouched, state, SimpleNamespace(icloud=0, mp_physics=28))
    assert not bool(untouched.cldfra.any())


def _seal_root(tmp_path: Path, *, with_pblh: bool) -> tuple[Path, list[str]]:
    """A one-point two-hour bridge publication as the decoder writes it."""
    root = tmp_path / "bridge"
    root.mkdir()
    gate = [
        "status\tPASS", "forecast_hours\t0,1", "series_count\t2",
        f"atmosphere_selected_per_time\t{562 if with_pblh else 561}",
        "soil_selected_per_time\t18", "window_shape\t1x1",
    ]
    if with_pblh:
        gate += ["optional_surface_fields\tPBLH", "optional_surface_units\tPBLH=m"]
    (root / "gate.txt").write_text("\n".join(gate) + "\n")
    for hour in (0, 1):
        atmosphere = root / f"atmosphere-f{hour:02d}"
        soil = root / f"soil-f{hour:02d}"
        atmosphere.mkdir()
        soil.mkdir()
        for name in _ATMOSPHERE_3D + _ATMOSPHERE_2D + (("PBLH",) if with_pblh else ()):
            np.ones(1, dtype="<f4").tofile(atmosphere / f"{name}.f32le")
        for name in ("SOILT", "SOILW"):
            np.ones(9, dtype="<f4").tofile(soil / f"{name}.f32le")
    for name in ("atmos0", "soil0", "atmos1", "soil1", "decoder", "evidence.json"):
        (tmp_path / name).write_text("{}" if name.endswith(".json") else name)
    series = tmp_path / "series.tsv"
    series.write_text("0\tatmos0\tsoil0\n1\tatmos1\tsoil1\n")
    args = ["--root", str(root), "--receipt", str(tmp_path / "receipt.json"),
            "--source-manifest-sha256", "0" * 64, "--decoder", str(tmp_path / "decoder"),
            "--series", str(series), "--time-evidence", str(tmp_path / "evidence.json"),
            "--expected-window-shape", "1x1"]
    return root, args


@pytest.mark.parametrize("with_pblh", [False, True])
def test_the_bridge_seal_counts_the_declared_pbl_height(tmp_path, monkeypatch, with_pblh):
    """Breakage it prevents (rolling 2.9.4 pass, 2026-10-21 17:11Z, and the
    box-D whole start): a wrfnat publishing HPBL gave 562 selected records
    and one more payload, and the seal refused it ("gate
    atmosphere_selected_per_time='562', expected '561'"), so every native
    HRRR preparation stopped before its first step."""
    import sys
    from tools import seal_hrrr_native_bridge

    root, args = _seal_root(tmp_path, with_pblh=with_pblh)
    monkeypatch.setattr(sys, "argv", ["seal_hrrr_native_bridge.py", *args])
    seal_hrrr_native_bridge.main()
    assert (root / "SHA256SUMS").is_file()


def test_the_initial_cldfra_takes_the_sources_layer_cover(monkeypatch):
    """HRRR's cover is mostly MYNN subgrid cloud a start state cannot hold:
    resolved clouds alone gave a 6 % column-maximum mean against NOAA's
    54 % (2026-10-21 06Z).  With the source's LCDC/MCDC/HCDC each layer's
    cover is the source's: resolved levels take max(own, cover), an empty
    layer puts its cover on its most humid level."""
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)
    except Exception:
        pytest.skip("no CUDA device")
    from gpuwm.core import physics as physics_module

    nz, ny, nx = 6, 1, 2
    pressure = cp.asarray([95000, 80000, 60000, 45000, 30000, 20000],
                          dtype=cp.float32)[:, None, None] * cp.ones((nz, ny, nx), cp.float32)
    temperature = cp.asarray([290, 280, 265, 250, 230, 215],
                             dtype=cp.float32)[:, None, None] * cp.ones((nz, ny, nx), cp.float32)
    monkeypatch.setattr(physics_module, "_prepare_atmosphere",
                        lambda state: {"temperature": temperature, "pressure": pressure})
    qv = cp.full((nz, ny, nx), 1.0e-4, dtype=cp.float32)
    qv[1] = 8.0e-3  # the most humid low level
    zeros = cp.zeros((nz, ny, nx), dtype=cp.float32)
    state = SimpleNamespace(qv=qv, qc=zeros.copy(), qi=zeros.copy(), qs=zeros.copy())
    driver = SimpleNamespace(cldfra=zeros.copy())
    met = SimpleNamespace(fields={
        "LCDC": np.array([[70.0, 0.0]], dtype=np.float32),
        "MCDC": np.array([[0.0, 40.0]], dtype=np.float32),
        "HCDC": np.array([[100.0, 0.0]], dtype=np.float32)})
    hrrr_physics._seed_initial_cldfra(driver, state, SimpleNamespace(icloud=1, mp_physics=28), met)
    seeded = cp.asnumpy(driver.cldfra)[:, 0, :]
    low, mid, high = seeded[:2], seeded[2:4], seeded[4:]
    np.testing.assert_allclose(low.max(axis=0), [0.7, 0.0], atol=1e-6)
    np.testing.assert_allclose(mid.max(axis=0), [0.0, 0.4], atol=1e-6)
    np.testing.assert_allclose(high.max(axis=0), [1.0, 0.0], atol=1e-6)
    assert seeded[1, 0] == pytest.approx(0.7)  # on the most humid low level
