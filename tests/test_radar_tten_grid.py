"""The Python half of ``rw_nexrad grid-ref``: descriptor, reader, receipt.

The data path is Rust (``cargo test -p rw-nexrad grid_ref``); what is
pinned here is what Python owns: the grid descriptor it hands the binary,
and the reader that refuses a product that does not prove it is the one
asked for -- wrong grid, wrong shape, short file, altered bytes.
"""
from __future__ import annotations

import hashlib
import json
import subprocess

import numpy as np
import pytest

from gpuwm.obs import radar_tten_grid as rtg
from gpuwm.obs.target_grid import TargetGrid
from gpuwm.static.lambert import LambertGrid

NX, NY, NZ = 12, 9, 4


def _grid(name="analytic", top=8000.0) -> TargetGrid:
    projection = LambertGrid(
        ref_lat=35.3, ref_lon=-97.3, truelat1=33.0, truelat2=37.0,
        stand_lon=-97.3, dx=3000.0, dy=3000.0, e_we=NX + 1, e_sn=NY + 1)
    return TargetGrid.from_projection(
        projection, z_w=np.linspace(300.0, top, NZ + 1), name=name)


def _product(tmp_path, grid, *, field=None, identity=None, shape=None):
    """A receipt + field pair in the binary's layout, written by hand."""
    shape = shape or [grid.nz, grid.ny, grid.nx]
    if field is None:
        field = np.full(shape, rtg.NO_COVERAGE, dtype="<f4")
        field[0, 4, 5] = 31.5
        field[1, 4, 5] = rtg.OBSERVED_NO_ECHO
    prefix = tmp_path / "ref"
    data = tmp_path / "ref.ref.f32"
    np.ascontiguousarray(field, dtype="<f4").tofile(data)
    raw = data.read_bytes()
    receipt = {
        "schema": rtg.REF_SCHEMA, "status": "READY", "shape": shape,
        "data": {"file": data.name, "bytes": len(raw),
                 "sha256": hashlib.sha256(raw).hexdigest()},
        "grid": {"identity_sha256": identity or grid.identity_sha256()},
    }
    (tmp_path / "ref.json").write_text(json.dumps(receipt))
    return prefix, data


def test_the_descriptor_carries_identity_projection_and_the_exact_columns(tmp_path):
    grid = _grid()
    path = tmp_path / "grid" / "grid.json"
    doc = rtg.write_grid_descriptor(grid, path)
    on_disk = json.loads(path.read_text())
    assert on_disk == doc
    assert doc["schema"] == rtg.GRID_SCHEMA
    assert doc["identity_sha256"] == grid.identity_sha256()
    assert (doc["nx"], doc["ny"], doc["nz"]) == (NX, NY, NZ)
    spec = doc["projection"]
    assert spec["kind"] == "lambert"
    assert (spec["e_we"], spec["e_sn"]) == (NX + 1, NY + 1)
    assert (spec["known_x"], spec["known_y"]) == ((NX + 1) / 2.0, (NY + 1) / 2.0)
    z_path = path.parent / doc["z_w"]["file"]
    assert doc["z_w"]["shape"] == [NZ + 1, NY, NX]
    assert doc["z_w"]["sha256"] == hashlib.sha256(z_path.read_bytes()).hexdigest()
    z_w = np.fromfile(z_path, dtype="<f8").reshape(doc["z_w"]["shape"])
    assert np.array_equal(z_w, grid.z_w)
    # Rewriting an unchanged grid rewrites identical bytes.
    before = path.read_bytes()
    rtg.write_grid_descriptor(grid, path)
    assert path.read_bytes() == before


def test_a_matching_product_reads_back_as_float32_in_noaa_convention(tmp_path):
    grid = _grid()
    prefix, _ = _product(tmp_path, grid)
    ref, receipt = rtg.read_reflectivity(prefix, grid=grid)
    assert ref.dtype == np.float32 and ref.shape == (NZ, NY, NX)
    assert ref[0, 4, 5] == np.float32(31.5)
    assert ref[1, 4, 5] == rtg.OBSERVED_NO_ECHO
    assert ref[2, 0, 0] == rtg.NO_COVERAGE
    assert receipt["schema"] == rtg.REF_SCHEMA
    # The identity alone is enough to bind it, too.
    rtg.read_reflectivity(prefix, identity_sha256=grid.identity_sha256())


def test_a_product_for_another_grid_is_refused(tmp_path):
    grid = _grid()
    other = _grid(top=9000.0)            # same horizontal grid, other columns
    assert other.identity_sha256() != grid.identity_sha256()
    prefix, _ = _product(tmp_path, grid, identity=other.identity_sha256())
    with pytest.raises(rtg.RadarTtenGridError, match="not the required grid"):
        rtg.read_reflectivity(prefix, grid=grid)


def test_shape_size_and_digest_disagreements_are_refused(tmp_path):
    grid = _grid()
    prefix, _ = _product(tmp_path, grid, shape=[NZ + 1, NY, NX],
                         field=np.zeros((NZ + 1, NY, NX), dtype="<f4"))
    with pytest.raises(rtg.RadarTtenGridError, match="shape"):
        rtg.read_reflectivity(prefix, grid=grid)

    prefix, data = _product(tmp_path, grid)
    data.write_bytes(data.read_bytes()[:-4])
    with pytest.raises(rtg.RadarTtenGridError, match="bytes"):
        rtg.read_reflectivity(prefix, grid=grid)

    prefix, data = _product(tmp_path, grid)
    raw = bytearray(data.read_bytes())
    raw[0] ^= 1
    data.write_bytes(bytes(raw))
    with pytest.raises(rtg.RadarTtenGridError, match="sha256"):
        rtg.read_reflectivity(prefix, grid=grid)

    prefix, _ = _product(tmp_path, grid)
    receipt = json.loads((tmp_path / "ref.json").read_text())
    receipt["schema"] = "gpuwm-obs.radar-grid.v2"
    (tmp_path / "ref.json").write_text(json.dumps(receipt))
    with pytest.raises(rtg.RadarTtenGridError, match="schema"):
        rtg.read_reflectivity(prefix, grid=grid)


def test_the_reader_follows_the_receipts_data_name(tmp_path):
    grid = _grid()
    prefix, data = _product(tmp_path, grid)
    moved = tmp_path / "ref.f32"
    data.rename(moved)
    receipt = json.loads((tmp_path / "ref.json").read_text())
    receipt["data"]["file"] = moved.name
    (tmp_path / "ref.json").write_text(json.dumps(receipt))
    ref, _ = rtg.read_reflectivity(prefix, grid=grid)
    assert ref.shape == (NZ, NY, NX)


def _grid_ref_binary():
    from gpuwm.obs.nexrad import find_nexrad_bin
    binary = find_nexrad_bin()
    if binary is None:
        pytest.skip("rw_nexrad is not built here")
    usage = subprocess.run([str(binary), "grid-ref", "--help"], capture_output=True,
                           text=True, timeout=30)
    if usage.returncode != 0 or "ref2tten" not in usage.stdout:
        pytest.skip("this rw_nexrad predates grid-ref")
    return binary


def test_the_binary_refuses_a_set_with_no_decodable_volume(tmp_path):
    binary = _grid_ref_binary()
    grid = _grid()
    bogus = tmp_path / "KTLX20261001_195500_V06"
    bogus.write_bytes(b"AR2V0006." + b"\0" * 40)
    with pytest.raises(RuntimeError, match="no volume survived"):
        rtg.grid_reflectivity(grid, [bogus], tmp_path / "out" / "ref", binary=binary)


def test_the_binary_refuses_a_descriptor_whose_columns_were_altered(tmp_path):
    binary = _grid_ref_binary()
    grid = _grid()
    descriptor = tmp_path / "grid.json"
    rtg.write_grid_descriptor(grid, descriptor)
    z_path = tmp_path / "grid.z_w.f64"
    raw = bytearray(z_path.read_bytes())
    raw[-1] ^= 1
    z_path.write_bytes(bytes(raw))
    bogus = tmp_path / "KTLX20261001_195500_V06"
    bogus.write_bytes(b"AR2V0006." + b"\0" * 40)
    with pytest.raises(RuntimeError, match="sha256"):
        rtg.grid_reflectivity(grid, [bogus], tmp_path / "out" / "ref",
                              grid_descriptor=descriptor, binary=binary)
