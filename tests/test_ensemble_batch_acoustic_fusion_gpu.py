"""Isolated Mu/W pilot against independent unchanged scalar acoustic helpers."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest
from conftest import requires_gpu

pytestmark = [pytest.mark.gpu, requires_gpu]


def _retain_actual_column_image(launch, directory):
    """Save only cache images whose source matches the loaded handle receipt."""
    if not hasattr(launch, 'column_launch'):
        return
    from cupy.cuda import compiler
    receipt = dict(launch.column_launch.binding_receipt)
    directory.mkdir(parents=True, exist_ok=True)
    cache = os.environ.get('CUPY_CACHE_DIR')
    images = []
    if cache:
        for source in Path(cache).glob('*.cubin.cu'):
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != receipt['compiled_source_sha256']:
                continue
            raw = source.with_suffix('').read_bytes()
            length = len(compiler._hash_hexdigest(b''))
            payload = raw[length:]
            assert raw[:length] == compiler._hash_hexdigest(payload).encode()
            (directory / source.name).write_bytes(data)
            (directory / source.with_suffix('').name).write_bytes(payload)
            images.append({'source': source.name, 'source_sha256': hashlib.sha256(data).hexdigest(),
                           'cubin_sha256': hashlib.sha256(payload).hexdigest(), 'bytes': len(payload)})
    (directory / 'actual-image-receipt.json').write_text(json.dumps({
        'binding': receipt, 'images': images,
        'missing_image_reason': None if images else 'enable fresh CUPY_CACHE_SAVE_CUDA_SOURCE=1 before import'
    }, indent=2, default=str) + '\n')


def _assert_words(cp, actual, expected, label):
    a, b = cp.asnumpy(actual).view(np.uint32), cp.asnumpy(expected).view(np.uint32)
    changed = a != b
    if np.any(changed):
        coordinates = np.argwhere(changed)[:12]
        raise AssertionError({'label': label, 'different_words': int(np.count_nonzero(changed)),
                              'first12': [{'coordinate': row.tolist(),
                                           'actual': f'{int(a[tuple(row)]):08x}',
                                           'expected': f'{int(b[tuple(row)]):08x}'} for row in coordinates]})


@pytest.mark.parametrize('members', [1, 4, 20])
@pytest.mark.parametrize('terrain,mapped,nz', [
    (False, False, 5), (True, True, 5),
    (False, False, 50), (True, True, 50)])
@pytest.mark.parametrize('top_lid', [False, True])
def test_column_mu_w_pilot_matches_original_all_state_and_scratch(
        members, terrain, mapped, nz, top_lid, tmp_path, shared_intermediates=False):
    import cupy as cp
    from test_ensemble_batch_acoustic_gpu import _pack_physical
    from gpuwm.core import acoustic as original
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble import batch_acoustic as separate
    from gpuwm.ensemble import batch_acoustic_fusion as candidate
    state, references, cfg, slots = _pack_physical(
        members, terrain=terrain, mapped=mapped, nz=nz)
    cfg = replace(cfg, emdiv=0.0, damp_opt=0, top_lid=top_lid)
    state.cfg = cfg
    dtau = 0.5
    coefficients = separate.prepare_acoustic_coefficients(state, cfg, dtau)
    launch = candidate.prepare_acoustic_substep_launch(state, cfg, dtau, coefficients,
        shared_intermediates=shared_intermediates)
    originals = [original.prepare_acoustic_substep_launch(
        reference, cfg, dtau, original.prepare_acoustic_coefficients(reference, cfg, dtau))
        for reference in references]
    (tmp_path / 'fusion-receipt.json').write_text(json.dumps(launch.fusion_receipt, indent=2, default=str) + '\n')
    (tmp_path / 'compiled-attributes.json').write_text(json.dumps(
        getattr(launch, 'compiled_attributes', {}), indent=2, default=str) + '\n')
    _retain_actual_column_image(launch, tmp_path / 'actual-column')
    for substep, first in enumerate((True, False, False)):
        launch(first=first)
        for scalar in originals:
            scalar(first=first)
        cp.cuda.get_current_stream().synchronize()
        for member, reference in enumerate(references):
            for name in state_array_shapes(cfg):
                _assert_words(cp, state.member_view(name, member), getattr(reference, name),
                              (substep, member, name))
            for slot in slots:
                expected = reference.existing_scratch(slot)
                if expected is not None:
                    _assert_words(cp, state.scratch_member_view(slot, member), expected,
                                  (substep, member, 'scratch:' + slot))
    print('isolated acoustic Mu/W state+scratch exact PASS', members, terrain, mapped,
          nz, top_lid, 'shared' if shared_intermediates else 'simple', flush=True)


@pytest.mark.parametrize('members', [1, 4, 20])
@pytest.mark.parametrize('terrain,mapped,nz', [
    (False, False, 5), (True, True, 5),
    (False, False, 50), (True, True, 50)])
@pytest.mark.parametrize('top_lid', [False, True])
def test_shared_column_mu_w_matches_original_all_state_and_scratch(
        members, terrain, mapped, nz, top_lid, tmp_path):
    test_column_mu_w_pilot_matches_original_all_state_and_scratch(
        members, terrain, mapped, nz, top_lid, tmp_path, shared_intermediates=True)


def test_column_mu_w_crosses_scalar_block_boundary(tmp_path, shared_intermediates=False):
    import cupy as cp
    from test_ensemble_batch_acoustic_gpu import _pack_physical
    from gpuwm.core import acoustic as original
    from gpuwm.core.device_inventory import state_array_shapes
    from gpuwm.ensemble import batch_acoustic as separate
    from gpuwm.ensemble import batch_acoustic_fusion as candidate
    state, references, cfg, slots = _pack_physical(
        4, terrain=True, mapped=True, nz=5, ny=7, nx=37)
    cfg = replace(cfg, emdiv=0.0, damp_opt=0, top_lid=True)
    state.cfg = cfg
    coefficients = separate.prepare_acoustic_coefficients(state, cfg, 0.5)
    launch = candidate.prepare_acoustic_substep_launch(state, cfg, 0.5, coefficients,
        shared_intermediates=shared_intermediates)
    originals = [original.prepare_acoustic_substep_launch(
        reference, cfg, 0.5, original.prepare_acoustic_coefficients(reference, cfg, 0.5))
        for reference in references]
    _retain_actual_column_image(launch, tmp_path / 'actual-column')
    for substep, first in enumerate((True, False, False)):
        launch(first=first)
        for scalar in originals:
            scalar(first=first)
        cp.cuda.get_current_stream().synchronize()
        for member, reference in enumerate(references):
            for name in state_array_shapes(cfg):
                _assert_words(cp, state.member_view(name, member), getattr(reference, name),
                              (substep, member, name))
            for slot in slots:
                expected = reference.existing_scratch(slot)
                if expected is not None:
                    _assert_words(cp, state.scratch_member_view(slot, member), expected,
                                  (substep, member, 'scratch:' + slot))


def test_shared_column_mu_w_crosses_scalar_block_boundary(tmp_path):
    test_column_mu_w_crosses_scalar_block_boundary(tmp_path, shared_intermediates=True)


def test_fused_candidate_submission_adds_no_device_allocation(shared_intermediates=False):
    import cupy as cp
    from test_ensemble_batch_acoustic_gpu import _pack_physical
    from gpuwm.ensemble import batch_acoustic as separate
    from gpuwm.ensemble import batch_acoustic_fusion as candidate
    state, _, cfg, _ = _pack_physical(4, nz=50)
    cfg = replace(cfg, emdiv=0.0, damp_opt=0, top_lid=True)
    state.cfg = cfg
    coefficients = separate.prepare_acoustic_coefficients(state, cfg, 0.5)
    launch = candidate.prepare_acoustic_substep_launch(state, cfg, 0.5, coefficients,
        shared_intermediates=shared_intermediates)
    launch(first=True)
    cp.cuda.get_current_stream().synchronize()

    class Recorder(cp.cuda.MemoryHook):
        def __init__(self):
            self.requests = []

        def malloc_preprocess(self, device_id, size, mem_size):
            self.requests.append((device_id, size, mem_size))

    recorder = Recorder()
    with recorder:
        launch(first=False)
        cp.cuda.get_current_stream().synchronize()
    assert not recorder.requests


def test_shared_fused_candidate_submission_adds_no_device_allocation():
    test_fused_candidate_submission_adds_no_device_allocation(shared_intermediates=True)
