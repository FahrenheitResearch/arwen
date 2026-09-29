"""Legacy corridors are checked against their own prepared tree."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.static.corridor import (
    ChildStaticsCorridor, CorridorRefusal, corridor_geometry,
    origin_in_frame_cells, write_statics_corridor_set,
)
from test_statics_corridor import (
    corridor_build, geog_root, _child_dc, _load, _parent_run,
)


def _legacy(build, directory, *, changes=None, fields=None):
    entry = dict(build.entry)
    entry.pop("build_contract")
    entry.update(changes or {})
    return write_statics_corridor_set(
        directory, [replace(build, entry=entry,
                            fields=build.fields if fields is None else fields)])


def _sealed_child(build):
    corridor = ChildStaticsCorridor(build.entry, build.fields, "unused")
    return corridor.crop_at(*build.entry["reference_origin_child_cells"])


@pytest.mark.parametrize("overlay", [False, True])
@pytest.mark.parametrize("contract", [None, "older-column-binning"])
def test_legacy_corridor_loads_without_geography_or_builder(
        corridor_build, tmp_path, monkeypatch, overlay, contract):
    import gpuwm.static.build as build

    calls = []

    def forbidden(*args, **kwargs):
        calls.append(args)
        raise AssertionError("legacy loading must not build statics")

    monkeypatch.setattr(build, "build_static", forbidden)
    directory = tmp_path / "corridor"
    changes = {"geog": {"root": str(tmp_path / "missing-geography")}}
    if overlay:
        changes["highres"] = {"status": "APPLIED"}
    if contract is not None:
        changes["build_contract"] = contract
    receipt = _legacy(corridor_build, directory, changes=changes)
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    loaded = _load(directory, receipt,
                   sealed_child_statics=_sealed_child(corridor_build))
    assert calls == []
    assert loaded.highres_applied is overlay
    for name, field in corridor_build.fields.items():
        assert loaded.fields[name].tobytes() == field.tobytes()
        assert not loaded.fields[name].flags.writeable
    assert before == {p.name: p.read_bytes() for p in directory.iterdir()}


@pytest.mark.parametrize("name", ["HGT_M", "LANDUSEF", "ALBEDO12M"])
def test_legacy_reference_mismatch_names_field_and_cell_count(
        corridor_build, tmp_path, name):
    fields = dict(corridor_build.fields)
    fields[name] = fields[name].copy()
    cell = (0,) * (fields[name].ndim - 2) + (9, 9)
    fields[name][cell] += 0.5
    directory = tmp_path / "corridor"
    receipt = _legacy(corridor_build, directory, fields=fields)
    with pytest.raises(CorridorRefusal) as caught:
        _load(directory, receipt,
              sealed_child_statics=_sealed_child(corridor_build))
    message = str(caught.value)
    assert f"'{name}': 1" in message
    assert "prepared placement" in message and "first relocation" in message
    assert "rw-wps" in message and "--statics-corridor" in message


def test_legacy_reference_allows_one_ulp_and_reports_it(
        corridor_build, tmp_path, capsys):
    fields = dict(corridor_build.fields)
    fields["HGT_M"] = fields["HGT_M"].copy()
    fields["HGT_M"][9, 9] = np.nextafter(fields["HGT_M"][9, 9], np.inf)
    directory = tmp_path / "corridor"
    receipt = _legacy(corridor_build, directory, fields=fields)
    _load(directory, receipt,
          sealed_child_statics=_sealed_child(corridor_build))
    assert "within one ULP: {'HGT_M': 1}" in capsys.readouterr().err


def test_legacy_corridor_ignores_changed_builder(
        corridor_build, tmp_path, monkeypatch):
    import gpuwm.static.build as build

    changed = {name: field.copy()
               for name, field in corridor_build.fields.items()}
    changed["HGT_M"] += 0.5
    calls = []

    def newer_builder(*args, **kwargs):
        calls.append(args)
        return changed

    monkeypatch.setattr(build, "build_static", newer_builder)
    directory = tmp_path / "corridor"
    receipt = _legacy(corridor_build, directory)
    _load(directory, receipt,
          sealed_child_statics=_sealed_child(corridor_build))
    assert calls == []


def test_legacy_root_frame_uses_composed_prepared_origin(
        corridor_build, tmp_path):
    child = _child_dc()
    child.grid_id, child.parent_id = 3, 2
    parent = SimpleNamespace(grid_id=2, parent_id=1, parent_grid_ratio=3,
                             i_parent_start=2, j_parent_start=3)
    origin = origin_in_frame_cells({2: parent, 3: child}, 3, 1)
    assert origin == (18, 27)
    frame = dict(frame_run=SimpleNamespace(nx=20, ny=20), frame_grid_id=1,
                 ratio_to_frame=9, reference_origin=origin)
    window = (9, 18, 30, 30)
    geometry = corridor_geometry(child, _parent_run(), **frame, window=window)
    rooted = replace(corridor_build, grid_id=3,
                     entry={**corridor_build.entry, **geometry})
    directory = tmp_path / "corridor"
    receipt = _legacy(rooted, directory)
    loaded = _load(directory, receipt, grid_id=3, child_dc=child,
                   frame_kwargs=frame, required_window=window,
                   sealed_child_statics=_sealed_child(corridor_build))
    expected = _sealed_child(corridor_build)
    for name, value in loaded.crop_at(*origin).items():
        assert value.tobytes() == expected[name].tobytes()
    # A parent-relative crop is outside this root-frame window.
    with pytest.raises(CorridorRefusal, match="outside"):
        loaded.crop(child.i_parent_start, child.j_parent_start)


@pytest.mark.parametrize("missing", [None, "HGT_M"])
def test_legacy_corridor_requires_all_sealed_reference_fields(
        corridor_build, tmp_path, missing):
    directory = tmp_path / "corridor"
    receipt = _legacy(corridor_build, directory)
    sealed = dict(_sealed_child(corridor_build)) if missing else None
    if missing:
        del sealed[missing]
    with pytest.raises(CorridorRefusal, match="sealed child statics.*missing"):
        _load(directory, receipt, sealed_child_statics=sealed)


def test_legacy_corridor_checks_digest_before_reference(corridor_build,
                                                       tmp_path):
    directory = tmp_path / "corridor"
    receipt = _legacy(corridor_build, directory)
    with (directory / "d02.npz").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(CorridorRefusal, match="digest mismatch"):
        _load(directory, receipt)


def test_current_corridor_needs_no_reference_or_geography(corridor_build,
                                                        tmp_path):
    directory = tmp_path / "corridor"
    receipt = write_statics_corridor_set(directory, [replace(
        corridor_build, entry={**corridor_build.entry, "geog": {}})])
    loaded = _load(directory, receipt)
    assert loaded.cache_sha256 == receipt["domains"]["d02"]["cache"]["sha256"]
