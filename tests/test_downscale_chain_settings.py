"""Regression controls for the real standalone-child GUI handoff."""
from dataclasses import asdict
import hashlib
from pathlib import Path

import pytest

from gpuwm import downscale
from gpuwm.config import RunConfig, load_config
from test_downscale_cli import _PARENT_CONFIG, _history, _give_the_parent_a_real_projection
from datetime import datetime


def test_parent_query_uses_history_for_legacy_child(tmp_path):
    """No experiment projection is invented from the legacy child config."""
    cfg = dict(_PARENT_CONFIG, nx=156, ny=156, nz=4, grid_id=2,
               specified=True, nested=False)
    child = tmp_path / 'child.toml'
    child.write_text(downscale._render_child_toml(cfg))
    original = child.read_bytes()
    frame = tmp_path / 'wrfout_d02_1974-04-03_12_00_00'
    _history(frame, datetime(1974, 4, 3, 12), ny=156, nx=156)
    _give_the_parent_a_real_projection(frame, ny=156, nx=156)
    answer = downscale.inspect_downscale_parent(tmp_path, 2)
    assert answer['parent_domain'] == 2
    assert answer['nx'] == answer['ny'] == 156
    grid = downscale._parent_geometry(frame)
    j, i = downscale._nearest_parent_index(grid['xlat'], grid['xlong'], *answer['center_latlon'])
    assert (j, i) == (78, 78)
    placement = downscale._centered_placement(grid, j0=j, i0=i, ratio=3, child_nx=36, child_ny=36)
    assert placement.i_parent_start > 2 and placement.j_parent_start > 2
    assert child.read_bytes() == original


def test_edge_center_refusal_is_geometry_not_physics_or_vram():
    parent = {'nx': 156, 'ny': 156, 'dx': 4000., 'dy': 4000.}
    with pytest.raises(downscale.OfflineChildContractError) as caught:
        downscale._fit_child_size(parent, _PARENT_CONFIG, j0=0, i0=78, ratio=3,
            run_seconds=3600., output_interval_s=3600., vram_gib=24.)
    text = str(caught.value)
    assert 'interpolation margin' in text and 'move the center inward' in text
    assert 'physics' not in text and 'VRAM' not in text


def test_edited_child_roundtrip_keeps_geometry_and_hash(tmp_path):
    cfg = RunConfig(**downscale._derive_child_run_config(_PARENT_CONFIG,
        parent={'nx': 156, 'ny': 156, 'dx': 4000., 'dy': 4000.}, ratio=3,
        child_nx=36, child_ny=36, run_seconds=3600., output_interval_s=900.))
    document = downscale.child_settings_document(cfg)
    assert document['values'] == asdict(cfg)
    fields = {row['name']: row for row in document['fields']}
    assert fields['mp_physics']['editable'] and fields['dt']['editable']
    assert not fields['grid_id']['editable'] and not fields['dx']['editable']
    assert document['physics_components']
    edited = dict(document['values'], time_step_sound=6)
    path = tmp_path / 'edited-child.toml'
    path.write_text(downscale._render_child_toml(edited, tiles_mode='auto'))
    loaded = downscale.resolve_child_run_config(path)
    assert loaded.time_step_sound == 6 and loaded.dx == cfg.dx and loaded.grid_id == cfg.grid_id
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert downscale._verify_child_config_hash(path, digest) == digest
    path.write_text(path.read_text() + '\n# changed after review\n')
    with pytest.raises(downscale.OfflineChildContractError, match='review.*again'):
        downscale._verify_child_config_hash(path, digest)
