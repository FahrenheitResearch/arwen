"""A delivered product tree contains PRODUCTS, and nothing else.

The render path gives the Rust renderer a working store to write its
intermediate hour files into.  That store used to be created INSIDE the
directory the products are delivered to (``tempfile.mkdtemp(prefix=
".rwstore-", dir=outdir)``), which breaks the delivery two ways, both
seen on 2026-08-20:

* 25 ``.rwstore-*`` directories were still sitting inside a delivered
  product folder, with paths long enough that Windows could not list the
  directory at all.  The removal is best-effort (a memory-mapped hour
  file can hold a handle past process exit), so "it is cleaned up in a
  finally" is not a property of the delivered tree.
* a ``tar`` pull of the tree raced a live render and died:
  ``tar: .rwstore-p_4f7zzy: File removed before we read it``.  Even a
  scratch store that IS removed successfully is inside the tree while it
  exists, so any copy/pull/scan of the delivery can trip over it.

The remedy is structural, not a better cleanup: working scratch lives in
a SIBLING directory outside the delivered tree, so a delivered tree never
contains scratch even mid-render and even when the removal fails.
"""

from __future__ import annotations

import datetime
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm import render as render_mod
from gpuwm.io.wrfout import WrfoutWriter, wrf_global_attrs

_NZ, _NY, _NX = 4, 12, 16
_STAMP = "1974-04-03_18:00:00"

#: What a scratch store is named, wherever it lives.
SCRATCH_GLOBS = (".rwstore-*", "_ens_store", "*.render-scratch")


def _frame(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    shape3 = (_NZ, _NY, _NX)
    shape2 = (_NY, _NX)
    return {
        "T": rng.normal(300.0, 2.0, shape3).astype("f4"),
        "QVAPOR": np.abs(rng.normal(0.005, 0.001, shape3)).astype("f4"),
        "P": rng.normal(50000.0, 500.0, shape3).astype("f4"),
        "PB": np.full(shape3, 50000.0, dtype="f4"),
        "PH": rng.normal(1000.0, 50.0, (_NZ + 1, _NY, _NX)).astype("f4"),
        "PHB": np.full((_NZ + 1, _NY, _NX), 1000.0, dtype="f4"),
        "U": rng.normal(5.0, 1.0, (_NZ, _NY, _NX + 1)).astype("f4"),
        "V": rng.normal(5.0, 1.0, (_NZ, _NY + 1, _NX)).astype("f4"),
        "W": rng.normal(0.0, 0.5, (_NZ + 1, _NY, _NX)).astype("f4"),
        "PSFC": rng.normal(97000.0, 200.0, shape2).astype("f4"),
        "T2": rng.normal(295.0, 2.0, shape2).astype("f4"),
        "Q2": np.abs(rng.normal(0.008, 0.001, shape2)).astype("f4"),
        "U10": rng.normal(4.0, 1.0, shape2).astype("f4"),
        "V10": rng.normal(4.0, 1.0, shape2).astype("f4"),
        "HGT": np.zeros(shape2, dtype="f4"),
        "XLAT": np.linspace(38.0, 39.0, _NY * _NX).reshape(
            shape2).astype("f4"),
        "XLONG": np.linspace(-97.0, -96.0, _NY * _NX).reshape(
            shape2).astype("f4"),
        "REFL_10CM": rng.normal(20.0, 5.0, shape3).astype("f4"),
    }


@pytest.fixture()
def wrfout(tmp_path) -> Path:
    grid = SimpleNamespace(truelat1=38.5, truelat2=39.5, stand_lon=-96.5,
                           ref_lat=39.0, ref_lon=-96.5)
    attrs = wrf_global_attrs(
        grid, datetime.datetime(1974, 4, 3, 18), grid_id=2, parent_id=1,
        i_parent_start=5, j_parent_start=5, parent_grid_ratio=3, dt=6.0)
    path = tmp_path / "wrfout_d02_1974-04-03_18-00-00.nc"
    with WrfoutWriter(path, nx=_NX, ny=_NY, nz=_NZ, dx=1000.0, dy=1000.0,
                      global_attrs=attrs) as writer:
        writer.write_frame(_STAMP, _frame(seed=7))
    return path


def scratch_inside(tree: Path) -> list[Path]:
    """Every scratch directory anywhere inside a delivered tree."""

    found: list[Path] = []
    for pattern in SCRATCH_GLOBS:
        found.extend(sorted(tree.rglob(pattern)))
    return found


# ---------------------------------------------------------------------------
# The location contract, provable without the renderer built
# ---------------------------------------------------------------------------


def test_series_render_scratch_is_never_inside_the_delivered_tree(
        wrfout, tmp_path, monkeypatch):
    """The store handed to the renderer is OUTSIDE the delivery."""

    from gpuwm import rustwx

    outdir = tmp_path / "png"
    seen: dict[str, Path] = {}

    def fake_series(renderer, series, *, store_root, out_dir, **kwargs):
        store_root = Path(store_root)
        store_root.mkdir(parents=True, exist_ok=True)
        (store_root / "hour_000.rwstore").write_bytes(b"scratch")
        seen["store"] = store_root
        seen["mid_render"] = scratch_inside(Path(out_dir))
        return [], [], []

    monkeypatch.setattr(rustwx, "run_renderer_series", fake_series)
    monkeypatch.setattr(render_mod, "require_renderer",
                        lambda: Path("rw_wrfbatch"))
    render_mod.render_series_rust(
        [str(wrfout)], products="refl", timeidx=None, outdir=outdir,
        size=(400, 300))

    store = seen["store"]
    assert not store.is_relative_to(outdir), (
        f"the renderer's working store {store} is inside the delivered "
        f"tree {outdir}")
    assert seen["mid_render"] == [], (
        "the delivered tree carried scratch DURING the render: "
        f"{seen['mid_render']} -- this is what the tar pull raced")
    assert scratch_inside(outdir) == []


def test_per_file_render_scratch_is_never_inside_the_delivered_tree(
        wrfout, tmp_path, monkeypatch):
    """Same contract on the per-file lane (`render_wrfouts_rust`)."""

    from gpuwm import rustwx

    outdir = tmp_path / "png"
    seen: dict[str, Path] = {}

    def fake_one(renderer, path, *, store_root, out_dir, **kwargs):
        store_root = Path(store_root)
        store_root.mkdir(parents=True, exist_ok=True)
        (store_root / "hour_000.rwstore").write_bytes(b"scratch")
        seen["store"] = store_root
        seen["mid_render"] = scratch_inside(Path(out_dir))
        return [], [], []

    monkeypatch.setattr(rustwx, "run_renderer", fake_one)
    monkeypatch.setattr(render_mod, "require_renderer",
                        lambda: Path("rw_wrfbatch"))
    render_mod.render_wrfouts_rust(
        [str(wrfout)], products="refl", timeidx=None, outdir=outdir,
        size=(400, 300))

    store = seen["store"]
    assert not store.is_relative_to(outdir), (
        f"the renderer's working store {store} is inside the delivered "
        f"tree {outdir}")
    assert seen["mid_render"] == []
    assert scratch_inside(outdir) == []


def test_a_scratch_store_that_cannot_be_removed_is_still_outside(
        wrfout, tmp_path, monkeypatch, capsys):
    """The 25-leftover case: removal FAILS and the delivery stays clean.

    A memory-mapped hour file can hold its handle past the renderer's
    exit, so the removal is best-effort by nature.  What must not be
    best-effort is WHERE the leftover sits.
    """

    from gpuwm import rustwx

    outdir = tmp_path / "png"
    seen: dict[str, Path] = {}

    def fake_one(renderer, path, *, store_root, out_dir, **kwargs):
        store_root = Path(store_root)
        store_root.mkdir(parents=True, exist_ok=True)
        (store_root / "hour_000.rwstore").write_bytes(b"scratch")
        seen["store"] = store_root
        return [], [], []

    def refuse_rmtree(path, *args, **kwargs):
        if kwargs.get("ignore_errors"):
            return
        raise OSError(145, "The directory is not empty")

    monkeypatch.setattr(rustwx, "run_renderer", fake_one)
    monkeypatch.setattr(render_mod, "require_renderer",
                        lambda: Path("rw_wrfbatch"))
    monkeypatch.setattr(shutil, "rmtree", refuse_rmtree)
    monkeypatch.setattr(time, "sleep", lambda _seconds: None)

    render_mod.render_wrfouts_rust(
        [str(wrfout)], products="refl", timeidx=None, outdir=outdir,
        size=(400, 300))

    assert seen["store"].is_dir(), "the un-removable store is the premise"
    assert scratch_inside(outdir) == [], (
        "an un-removable scratch store landed inside the delivered tree")


def _renderer_gate() -> tuple[bool, str]:
    """This tree's own renderer, or the reason it may not be used."""

    from gpuwm import rustwx

    renderer = rustwx.find_renderer()
    if renderer is None:
        return False, ("rust renderer not built (cd tools/rustwx && cargo "
                       "build --release --locked --offline)")
    reason = render_mod.renderer_refusal(renderer)
    if reason is None:
        return True, ""
    return False, f"the resolved renderer is not this tree's: {reason}"


_RENDERER_USABLE, _RENDERER_SKIP = _renderer_gate()


@pytest.mark.skipif(not _RENDERER_USABLE, reason=_RENDERER_SKIP)
def test_a_real_render_delivers_a_scratch_free_tree(wrfout, tmp_path):
    """AGAINST THE ARTIFACT: the real renderer, the real front door.

    The mocked tests above fix the location; this one is the proof that
    a bare default ``gpuwm render`` leaves a delivered case folder with
    products in it and nothing else -- including on the box where the
    memory-map handle lag makes the removal fail.
    """

    import gpuwm.cli as cli

    out = tmp_path / "png"
    rc = cli.main(["render", str(wrfout), "--engine", "rust",
                   "--products", "refl", "--out", str(out)])
    assert rc == 0, "the render must still succeed"
    assert list(out.rglob("*.png")), "the render delivered no products"
    assert scratch_inside(out) == [], (
        "the delivered case folder carries scratch: "
        f"{scratch_inside(out)}")


def test_the_ensemble_lane_keeps_its_store_out_of_the_delivery(tmp_path,
                                                               monkeypatch):
    """`rw_ensbatch`'s store is scratch too, and it was never removed."""

    from gpuwm import rustwx_lanes
    from gpuwm.da import enprod

    outdir = tmp_path / "png"
    outdir.mkdir()
    ens_root = tmp_path / "ens"
    ens_root.mkdir()
    (ens_root / enprod.MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
    seen: dict[str, Path] = {}

    def fake_ensemble(engine, manifest, *, store_root, out_dir, **kwargs):
        store_root = Path(store_root)
        store_root.mkdir(parents=True, exist_ok=True)
        (store_root / "member.rwstore").write_bytes(b"scratch")
        seen["store"] = store_root
        seen["mid_render"] = scratch_inside(Path(out_dir))
        panel = Path(out_dir) / "arwen_ens_mean_refl.png"
        panel.write_bytes(bytes.fromhex("89504e470d0a1a0a"))
        return [panel], [], [], {}

    monkeypatch.setattr(rustwx_lanes, "find_ensemble_bin",
                        lambda: Path("rw_ensbatch"))
    monkeypatch.setattr(rustwx_lanes, "run_ensemble_renderer",
                        fake_ensemble)
    rc = enprod.run_suite_rust(
        ens_root, fields=("refl",), products=("mean",), thresholds=None,
        radii=None, timeidx=None, outdir=outdir, source_label=None,
        accept_status=("ok",), nan_policy="mask", tie_rule="flat-index")

    assert rc == 0
    store = seen["store"]
    assert not store.is_relative_to(outdir), (
        f"the ensemble renderer's store {store} is inside the delivered "
        f"tree {outdir}")
    assert seen["mid_render"] == []
    assert scratch_inside(outdir) == []
