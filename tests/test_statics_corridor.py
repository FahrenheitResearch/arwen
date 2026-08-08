"""The statics corridor: sealed child-resolution statics for prepared
moving nests (gpuwm.static.corridor).

THE LOAD-BEARING TEST here is crop-vs-direct bit-identity: a footprint
cropped out of the parent-extent corridor must equal, byte for byte, the
statics built directly for that footprint from the same geography --
the same ``identical source + identical cells = identical bytes``
invariant the relocation machinery asserts at every move.  It runs
against a fully synthetic WPS_GEOG tree that exercises the build's
distinct code paths (grid-cell accumulation on a fine source, the
categorical pixel-count path, the interpolation sequences on a coarse
global source, and the halo'd terrain smoother), and the instrument is
validated by planting a single-ULP perturbation and watching it caught.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import gpuwm.static.corridor as corridor_module
from gpuwm.static.build import GeogSelection, build_static
from gpuwm.native_wrf_contract import NATIVE_STATIC_REQUIRED
from gpuwm.static.corridor import (
    CORRIDOR_BYTES_PER_CELL,
    CORRIDOR_PLANES_PER_CELL,
    CORRIDOR_REBUILT_STATICS,
    ChildStaticsCorridor,
    CorridorRefusal,
    STATICS_CORRIDOR_RECEIPT,
    build_child_statics_corridor,
    corridor_cost,
    corridor_footprint_statics_builder,
    corridor_geometry,
    corridor_grid,
    grid_identity_probes,
    load_child_statics_corridor,
    write_statics_corridor_set,
)
from gpuwm.static.lambert import LambertGrid


# ---------------------------------------------------------------------------
# Synthetic WPS_GEOG: every default dataset the build resolves, tiny
# ---------------------------------------------------------------------------

def _write_index(dirpath: Path, **kv) -> dict:
    lines = [f"{key} = {value}" for key, value in kv.items()
             if value is not None]
    dirpath.mkdir(parents=True, exist_ok=True)
    (dirpath / "index").write_text("\n".join(lines) + "\n",
                                   encoding="utf-8")
    return kv


def _write_tiles(dirpath: Path, data: np.ndarray, kv: dict) -> None:
    nz, nyg, nxg = data.shape
    tx, ty = int(kv["tile_x"]), int(kv["tile_y"])
    signed = str(kv.get("signed", "no")).lower() in ("yes", "true")
    ws = int(kv["wordsize"])
    base = {1: "i1", 2: "i2", 4: "i4"}[ws]
    if not signed:
        base = "u" + base[1:]
    dt = np.dtype(">" + base) if ws > 1 else np.dtype(base)
    for ys in range(1, nyg + 1, ty):
        for xs in range(1, nxg + 1, tx):
            tile = data[:, ys - 1:ys - 1 + ty, xs - 1:xs - 1 + tx]
            name = (f"{xs:05d}-{xs + tx - 1:05d}."
                    f"{ys:05d}-{ys + ty - 1:05d}")
            tile.astype(dt).tofile(dirpath / name)


def _fine_regional(kv_extra=None):
    """445 m pixels centered on the test domain (gcell-eligible)."""
    kv = dict(projection="regular_ll", dx=0.004, dy=0.004,
              known_x=1.0, known_y=1.0, known_lat=33.5, known_lon=-98.5,
              tile_x=250, tile_y=250)
    kv.update(kv_extra or {})
    return kv, 750, 750


def _coarse_global(nz=1, kv_extra=None):
    """1-degree global pixels (interpolation-sequence paths)."""
    kv = dict(projection="regular_ll", dx=1.0, dy=1.0,
              known_x=1.0, known_y=1.0, known_lat=-89.5, known_lon=-179.5,
              tile_x=60, tile_y=60)
    if nz > 1:
        kv["tile_z"] = nz
    kv.update(kv_extra or {})
    return kv, 360, 180


def _synthetic_wps_geog(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    yy_ = lambda nyg, nxg, nz: np.meshgrid(  # noqa: E731
        np.arange(nz), np.arange(1, nyg + 1), np.arange(1, nxg + 1),
        indexing="ij")

    # terrain: fine, continuous, signed int16 metres
    kv, nxg, nyg = _fine_regional(dict(
        type="continuous", signed="yes", wordsize=2, tile_z=1,
        units='"meters MSL"'))
    kv = _write_index(root / "topo_gmted2010_30s", **kv)
    z, y, x = yy_(nyg, nxg, 1)
    _write_tiles(root / "topo_gmted2010_30s",
                 (37 * x + 91 * y) % 1500 + 3 * ((x * y) % 7), kv)

    # landuse: fine, categorical 1..21 with water blocks (category 17)
    kv, nxg, nyg = _fine_regional(dict(
        type="categorical", category_min=1, category_max=21,
        wordsize=1, tile_z=1,
        mminlu='"MODIFIED_IGBP_MODIS_NOAH"', iswater=17, islake=21,
        isice=15, isurban=13))
    kv = _write_index(root / "modis_landuse_20class_30s_with_lakes", **kv)
    z, y, x = yy_(nyg, nxg, 1)
    landuse = 1 + (3 * x + 5 * y) % 21
    water = ((x // 40) + (y // 40)) % 5 == 0
    landuse = np.where(water, 17, landuse)
    _write_tiles(root / "modis_landuse_20class_30s_with_lakes", landuse, kv)

    # soils: coarse global categorical 1..16
    for name in ("soiltype_top_30s", "soiltype_bot_30s"):
        kv, nxg, nyg = _coarse_global(kv_extra=dict(
            type="categorical", category_min=1, category_max=16,
            wordsize=1, tile_z=1))
        kv = _write_index(root / name, **kv)
        z, y, x = yy_(nyg, nxg, 1)
        _write_tiles(root / name, 1 + (2 * x + y) % 16, kv)

    # monthly continuous climatologies, coarse global
    monthly = (
        ("greenfrac_fpar_modis", 12, 0.01, 100),
        ("lai_modis_10m", 12, 0.1, 60),
        ("albedo_modis", 12, 1.0, 30),
        ("maxsnowalb_modis", 1, 1.0, 80),
        ("soiltemp_1deg", 1, 1.0, 300),
    )
    for name, nz, scale, span in monthly:
        kv, nxg, nyg = _coarse_global(nz=nz, kv_extra=dict(
            type="continuous", signed="yes", wordsize=2,
            scale_factor=scale))
        kv = _write_index(root / name, **kv)
        z, y, x = yy_(nyg, nxg, nz)
        _write_tiles(root / name, (x + 2 * y + 5 * z) % span + 1, kv)
    return root


def _parent_grid() -> LambertGrid:
    return LambertGrid(
        ref_lat=35.0, ref_lon=-97.0, truelat1=30.0, truelat2=60.0,
        stand_lon=-97.0, dx=6000.0, dy=6000.0, e_we=11, e_sn=11)


_REF_I = _REF_J = 4
_RATIO = 3
_CHILD_EWE = 10  # nx = 9 child cells


def _child_dc(i_parent_start=_REF_I, j_parent_start=_REF_J):
    return SimpleNamespace(
        grid_id=2, parent_id=1, parent_grid_ratio=_RATIO,
        i_parent_start=i_parent_start, j_parent_start=j_parent_start,
        run=SimpleNamespace(nx=_CHILD_EWE - 1, ny=_CHILD_EWE - 1))


def _parent_run():
    return SimpleNamespace(nx=10, ny=10)


def _reference_grid():
    return _parent_grid().nest(_REF_I, _REF_J, _RATIO,
                               _CHILD_EWE, _CHILD_EWE)


def _catalog(root: Path, tmp_path: Path):
    namelist = tmp_path / "namelist.wps"
    if not namelist.exists():
        namelist.write_text(
            "&share\n max_dom = 2,\n/\n&geogrid\n/\n", encoding="utf-8")
    files = [SimpleNamespace(path=namelist, role="wps_namelist")]
    files.extend(
        SimpleNamespace(path=root / name / "index", role="geog_index")
        for name in ("topo_gmted2010_30s",
                     "modis_landuse_20class_30s_with_lakes"))
    return SimpleNamespace(files=tuple(files))


@pytest.fixture(scope="module")
def geog_root(tmp_path_factory) -> Path:
    return _synthetic_wps_geog(tmp_path_factory.mktemp("wps-geog"))


@pytest.fixture(scope="module")
def corridor_build(geog_root, tmp_path_factory):
    catalog = _catalog(geog_root, tmp_path_factory.mktemp("catalog"))
    return build_child_statics_corridor(
        child_dc=_child_dc(), parent_run=_parent_run(),
        reference_grid=_reference_grid(), static_catalog=catalog)


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def test_corridor_geometry_covers_the_whole_parent():
    geometry = corridor_geometry(_child_dc(), _parent_run())
    assert geometry["corridor_nx"] == 30 and geometry["corridor_ny"] == 30
    assert geometry["origin_translation_child_cells"] == [-9, -9]
    # Every placement that keeps the child inside the parent crops
    # inside the corridor: (ip-1)*ratio + nx <= corridor_nx.
    assert (8 - 1) * 3 + 9 == geometry["corridor_nx"] - 3 + 3


def test_the_quoted_price_equals_what_the_build_actually_costs(
        corridor_build):
    """``corridor_cost`` prices a corridor nobody has built yet.

    ``gpuwm run-plan --estimate`` shows that figure to a caller BEFORE
    the preparation runs, so it has to be the same number the sealed
    receipt reports afterwards -- not close to it.  One side is
    arithmetic on the geometry and the field inventory; the other is
    measured off the built arrays.  They are held equal here against a
    real build rather than against each other.
    """
    quoted = corridor_cost(_child_dc(), _parent_run())
    entry = corridor_build.entry

    assert quoted["cells"] == entry["cells"]
    assert quoted["host_bytes"] == entry["host_bytes"]
    assert quoted["corridor_nx"] == entry["corridor_nx"]
    assert quoted["corridor_ny"] == entry["corridor_ny"]
    # And the per-cell figure is the whole contract, not a subset that
    # happens to sum correctly for this fixture's field list.
    assert set(entry["fields"]) == set(NATIVE_STATIC_REQUIRED)
    assert (quoted["host_bytes"]
            == quoted["cells"] * CORRIDOR_BYTES_PER_CELL)


def test_a_field_added_to_the_contract_reprices_the_corridor(monkeypatch):
    """The price is counted off the contract, never restated.

    A static field added to ``NATIVE_STATIC_REQUIRED`` enlarges every
    corridor.  If the bytes-per-cell figure were a literal, the
    estimate would keep quoting the old size and a caller would size a
    disk for a corridor that no longer fits it.
    """
    import gpuwm.static.corridor as corridor_module

    grown = tuple(NATIVE_STATIC_REQUIRED) + ("SOME_NEW_STATIC",)
    monkeypatch.setattr(corridor_module, "NATIVE_STATIC_REQUIRED", grown)
    assert (corridor_module._planes_per_cell()
            == CORRIDOR_PLANES_PER_CELL + 1)


def test_corridor_grid_is_the_reference_lattice():
    reference = _reference_grid()
    geometry = corridor_geometry(_child_dc(), _parent_run())
    grid = corridor_grid(reference, geometry)
    assert grid.e_we == 31 and grid.e_sn == 31
    # Corridor cell (10, 10) IS reference cell (1, 1): same bytes.
    lat_c, lon_c = grid.ij_to_latlon(10.0, 10.0)
    lat_r, lon_r = reference.ij_to_latlon(1.0, 1.0)
    assert float(lat_c) == float(lat_r) and float(lon_c) == float(lon_r)
    # And the corridor's first cell sits on the parent's first cell --
    # physically: the parent transform is a DIFFERENT float path, so
    # this is an alignment check, not a byte contract (the byte contract
    # is with the child lattice above, which is what statics build on).
    lat_p, lon_p = _parent_grid().ij_to_latlon(0.5 + 0.5 / 3, 0.5 + 0.5 / 3)
    lat_1, lon_1 = grid.ij_to_latlon(1.0, 1.0)
    assert abs(float(lat_1) - float(lat_p)) < 1e-10
    assert abs(float(lon_1) - float(lon_p)) < 1e-10


def test_translated_extent_override_defaults_stay_byte_inert():
    reference = _reference_grid()
    plain = reference.translated(6, 3)
    assert plain.e_we == reference.e_we and plain.e_sn == reference.e_sn
    lat_a, lon_a = plain.ij_to_latlon(2.0, 2.0)
    lat_b, lon_b = reference.ij_to_latlon(8.0, 5.0)
    assert float(lat_a) == float(lat_b) and float(lon_a) == float(lon_b)
    with pytest.raises(ValueError, match="at least one cell"):
        reference.translated(0, 0, e_we=1)


# ---------------------------------------------------------------------------
# THE LOAD-BEARING TEST: crop == direct build, bitwise; instrument armed
# ---------------------------------------------------------------------------

def test_crop_equals_direct_footprint_build_bitwise(
        geog_root, corridor_build, tmp_path):
    receipt = write_statics_corridor_set(
        tmp_path / "statics-corridor", [corridor_build])
    corridor = load_child_statics_corridor(
        tmp_path / "statics-corridor", expected_set_receipt=receipt,
        grid_id=2, child_dc=_child_dc(), parent_run=_parent_run(),
        reference_grid=_reference_grid())

    reference = _reference_grid()
    selection = GeogSelection.fallback(geog_root)
    checked_fields = 0
    for ip, jp in ((_REF_I, _REF_J), (1, 1), (8, 8), (2, 6), (7, 3)):
        shift_i = (ip - _REF_I) * _RATIO
        shift_j = (jp - _REF_J) * _RATIO
        direct = build_static(reference.translated(shift_i, shift_j),
                              geog_root, selection=selection)
        crop = corridor.crop(ip, jp)
        assert sorted(crop) == sorted(direct)
        for name in sorted(direct):
            a = np.asarray(crop[name])
            b = np.asarray(direct[name])
            assert a.dtype == b.dtype and a.shape == b.shape, name
            assert a.tobytes() == b.tobytes(), (
                f"{name} differs at placement ({ip}, {jp})")
            checked_fields += 1
    assert checked_fields == 5 * 14

    # VALIDATE THE INSTRUMENT: one flipped ULP in the corridor must be
    # caught by exactly the comparison above.
    fields = dict(corridor.fields)
    tampered = np.array(fields["HGT_M"], copy=True)
    tampered[12, 17] = np.nextafter(tampered[12, 17], np.inf)
    fields["HGT_M"] = tampered
    perturbed = ChildStaticsCorridor(
        geometry=corridor.geometry, fields=fields,
        cache_sha256=corridor.cache_sha256)
    bad = perturbed.crop(_REF_I, _REF_J)  # covers corridor cell (12, 17)
    good = build_static(reference, geog_root, selection=selection)
    assert bad["HGT_M"].tobytes() != good["HGT_M"].tobytes()
    mismatches = np.flatnonzero(
        bad["HGT_M"].view(np.uint64) != good["HGT_M"].view(np.uint64))
    assert mismatches.size == 1


def test_corridor_emission_is_deterministic(corridor_build, geog_root,
                                            tmp_path):
    first = write_statics_corridor_set(tmp_path / "one", [corridor_build])
    second = write_statics_corridor_set(tmp_path / "two", [corridor_build])
    assert first == second
    assert (first["domains"]["d02"]["cache"]["sha256"]
            == second["domains"]["d02"]["cache"]["sha256"])
    assert ((tmp_path / "one" / "d02.npz").read_bytes()
            == (tmp_path / "two" / "d02.npz").read_bytes())
    assert ((tmp_path / "one" / STATICS_CORRIDOR_RECEIPT).read_bytes()
            == (tmp_path / "two" / STATICS_CORRIDOR_RECEIPT).read_bytes())


# ---------------------------------------------------------------------------
# Refusals: tamper, receipt drift, wrong tree, off-corridor crops
# ---------------------------------------------------------------------------

def _sealed(corridor_build, tmp_path):
    directory = tmp_path / "statics-corridor"
    receipt = write_statics_corridor_set(directory, [corridor_build])
    return directory, receipt


def _load(directory, receipt, **overrides):
    kwargs = dict(
        expected_set_receipt=receipt, grid_id=2, child_dc=_child_dc(),
        parent_run=_parent_run(), reference_grid=_reference_grid())
    kwargs.update(overrides)
    return load_child_statics_corridor(directory, **kwargs)


def test_load_refuses_a_tampered_cache(corridor_build, tmp_path):
    directory, receipt = _sealed(corridor_build, tmp_path)
    cache = directory / "d02.npz"
    blob = bytearray(cache.read_bytes())
    blob[-1] ^= 0x01
    cache.write_bytes(bytes(blob))
    with pytest.raises(CorridorRefusal, match="digest mismatch"):
        _load(directory, receipt)


def test_load_refuses_a_mutated_on_disk_receipt(corridor_build, tmp_path):
    directory, receipt = _sealed(corridor_build, tmp_path)
    on_disk = json.loads(
        (directory / STATICS_CORRIDOR_RECEIPT).read_text(encoding="utf-8"))
    on_disk["domains"]["d02"]["cells"] += 1
    (directory / STATICS_CORRIDOR_RECEIPT).write_text(
        json.dumps(on_disk), encoding="utf-8")
    with pytest.raises(CorridorRefusal, match="differs from the copy"):
        _load(directory, receipt)


def test_load_refuses_a_corridor_from_a_different_tree(corridor_build,
                                                       tmp_path):
    directory, receipt = _sealed(corridor_build, tmp_path)
    with pytest.raises(CorridorRefusal, match="reference_i_parent_start"):
        _load(directory, receipt, child_dc=_child_dc(i_parent_start=5))


def test_load_refuses_an_uncovered_grid_id(corridor_build, tmp_path):
    directory, receipt = _sealed(corridor_build, tmp_path)
    with pytest.raises(CorridorRefusal, match="names d03"):
        _load(directory, receipt, grid_id=3)


def test_crop_refuses_off_corridor_placements(corridor_build, tmp_path):
    directory, receipt = _sealed(corridor_build, tmp_path)
    corridor = _load(directory, receipt)
    for ip, jp in ((0, 4), (4, 0), (9, 4), (4, 9)):
        with pytest.raises(CorridorRefusal, match="outside the statics"):
            corridor.crop(ip, jp)


def test_footprint_statics_builder_contract(corridor_build, tmp_path):
    directory, receipt = _sealed(corridor_build, tmp_path)
    corridor = _load(directory, receipt)
    builder = corridor_footprint_statics_builder(corridor)
    assert builder.static_provenance == CORRIDOR_REBUILT_STATICS
    assert builder.highres_applied is False
    assert corridor.cache_sha256[:12] in builder.source_label
    fields = builder(object(), _child_dc(i_parent_start=6,
                                         j_parent_start=2))
    reference = corridor.crop(6, 2)
    assert sorted(fields) == sorted(reference)
    for name in fields:
        assert fields[name].tobytes() == reference[name].tobytes()
        assert not fields[name].flags.writeable


def test_initializer_seam_requires_exactly_one_statics_source():
    from gpuwm.ingest.relocation_init import real_relocation_initializer

    with pytest.raises(TypeError, match="exactly one statics source"):
        real_relocation_initializer(
            vertical=object(), child_config=_child_dc(),
            reference_grid=_reference_grid(),
            reference_i_parent_start=_REF_I,
            reference_j_parent_start=_REF_J)
    with pytest.raises(TypeError, match="exactly one statics source"):
        real_relocation_initializer(
            catalog=object(), statics_builder=lambda *a: {},
            vertical=object(), child_config=_child_dc(),
            reference_grid=_reference_grid(),
            reference_i_parent_start=_REF_I,
            reference_j_parent_start=_REF_J)


def test_corridor_selection_validation():
    from gpuwm.static.corridor import validated_corridor_selection

    exp = SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, parent_id=0),
        SimpleNamespace(grid_id=2, parent_id=1),
        SimpleNamespace(grid_id=3, parent_id=2),
    ))
    assert validated_corridor_selection(exp, None) == ()
    assert validated_corridor_selection(exp, "all") == (2, 3)
    assert validated_corridor_selection(exp, (3,)) == (3,)
    with pytest.raises(ValueError, match="not child domains"):
        validated_corridor_selection(exp, (1,))
    with pytest.raises(ValueError, match="'all' or child grid ids"):
        validated_corridor_selection(exp, "some")
    single = SimpleNamespace(domains=(
        SimpleNamespace(grid_id=1, parent_id=0),))
    with pytest.raises(ValueError, match="no child domain"):
        validated_corridor_selection(single, "all")


# ---------------------------------------------------------------------------
# Bundle-gated: the same bit-identity against the REAL 30s GEOG tree
# ---------------------------------------------------------------------------

def _real_geog_root():
    root = os.environ.get("GPUWM_CASE_DATA_ROOT")
    if not root or not Path(root).is_dir():
        return None
    direct = Path(root) / "WPS_GEOG"
    if direct.is_dir():
        return direct
    staged = sorted(Path(root).glob("*/static/WPS_GEOG"))
    return staged[0] if staged else None


@pytest.mark.skipif(_real_geog_root() is None,
                    reason="GPUWM_CASE_DATA_ROOT/WPS_GEOG not present")
def test_crop_equals_direct_build_on_the_real_static_source(tmp_path):
    root = _real_geog_root()
    parent = LambertGrid(
        ref_lat=31.2, ref_lon=-89.9, truelat1=21.9, truelat2=41.9,
        stand_lon=-88.8, dx=6000.0, dy=6000.0, e_we=9, e_sn=9)
    child_dc = SimpleNamespace(
        grid_id=2, parent_id=1, parent_grid_ratio=3, i_parent_start=3,
        j_parent_start=3, run=SimpleNamespace(nx=9, ny=9))
    parent_run = SimpleNamespace(nx=8, ny=8)
    reference = parent.nest(3, 3, 3, 10, 10)
    selection = GeogSelection.fallback(root)
    geometry = corridor_geometry(child_dc, parent_run)
    fields = build_static(corridor_grid(reference, geometry), root,
                          selection=selection)
    corridor = ChildStaticsCorridor(
        geometry=geometry, fields=fields, cache_sha256="0" * 64)
    for ip, jp in ((3, 3), (1, 1), (5, 2)):
        direct = build_static(
            reference.translated((ip - 3) * 3, (jp - 3) * 3), root,
            selection=selection)
        crop = corridor.crop(ip, jp)
        for name in sorted(direct):
            assert (np.asarray(crop[name]).tobytes()
                    == np.asarray(direct[name]).tobytes()), (
                f"{name} differs at ({ip}, {jp})")
