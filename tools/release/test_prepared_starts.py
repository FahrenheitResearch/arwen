"""A different prepared child cannot establish forecast nondeterminism.

The release cut prepared with no card; R11 prepared with CUDA. Comparing
their pictures blamed T2 although the child state differed before step 1.
"""
import json

import numpy as np
import pytest

from tools.release.prepared_starts import compare_starts
from gpuwm.ingest.prepared_cache import _array_sha256


def prepared(root, value, backend):
    folder = root / 'hierarchy-head/domains/d02/prepared-cache'
    folder.mkdir(parents=True)
    array = np.array([value], dtype='float32')
    np.save(folder / 'state.npy', array)
    header = dict(schema='gpuwm-prepared-real-cache-v1', created_utc=backend,
                  metadata={'backend': backend}, arrays={'state/thp': dict(
                      file='state.npy', dtype=str(array.dtype), shape=list(array.shape),
                      nbytes=array.nbytes, sha256=_array_sha256(array))})
    (folder / 'header.json').write_text(json.dumps(header))
    return folder


def test_one_ulp_in_the_prepared_child_refuses_a_repeatability_claim(tmp_path):
    left, right = tmp_path / 'cpu', tmp_path / 'cuda'
    prepared(left, np.float32(300), 'cpu')
    prepared(right, np.nextafter(np.float32(300), np.float32(301)), 'cuda')
    result = compare_starts(left, right)
    assert result['status'] == 'REFUSE'
    assert result['different'] == ['hierarchy-head/domains/d02/prepared-cache/header.json:state/thp']
    assert 'before forecast step 1' in result['reason']
    assert '--prepared-root' in result['remedy']


def test_equal_arrays_allow_different_preparation_metadata(tmp_path):
    left, right = tmp_path / 'cpu', tmp_path / 'cuda'
    prepared(left, 300, 'cpu')
    prepared(right, 300, 'cuda')
    result = compare_starts(left, right)
    assert result['status'] == 'PASS'
    assert result['arrays'] == 1


def test_the_guard_checks_payloads_instead_of_trusting_header_hashes(tmp_path):
    folder = prepared(tmp_path, 300, 'cpu')
    np.save(folder / 'state.npy', np.array([301], dtype='float32'))
    with pytest.raises(ValueError, match='state/thp.*fails its manifest'):
        compare_starts(tmp_path, tmp_path)


def test_missing_starts_are_not_an_empty_pass(tmp_path):
    with pytest.raises(ValueError, match='no prepared arrays'):
        compare_starts(tmp_path, tmp_path)
