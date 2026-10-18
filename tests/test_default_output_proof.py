"""Negative controls for the release's unconditional full-field byte proof."""
from pathlib import Path
import numpy as np
import pytest
from netCDF4 import Dataset
from tools.release.default_output_proof import compare_folders


def _frame(folder, words, extra=False, size=4):
    folder.mkdir()
    with Dataset(folder / 'wrfout_d01_2026-10-03_12:00:00', 'w') as ds:
        ds.createDimension('x', size)
        var = ds.createVariable('T', 'f4', ('x',), fill_value=False)
        var[:] = np.asarray(words, np.uint32).view(np.float32)
        if extra:
            ds.createVariable('NEW', 'i4', ('x',))[:] = np.arange(size)


def test_equal_hash_does_not_hide_changed_nan_payload_or_signed_zero(tmp_path):
    a, b = tmp_path / 'base', tmp_path / 'tip'
    _frame(a, [0, 0x7fc00001, 0x3f800000, 0x7f800000])
    _frame(b, [0x80000000, 0x7fc00002, 0x3f800001, 0x7f800000])
    result = compare_folders(a, b, hash_bytes=lambda _: 'same planted hash')
    row, = result['rows']
    assert row['base_sha256'] == row['tip_sha256']
    assert row['byte_compare_performed'] and not row['byte_equal']
    assert result['values_differing'] == 3 and result['verdict'] == 'DIFFERENT'
    assert row['base_audit'] == dict(values=4, finite=2, nan=1, infinity=1, negative_zero=0)
    assert row['tip_audit']['negative_zero'] == 1
    assert row['examples'][1]['base_word_hex'] == '0100c07f'
    assert row['examples'][1]['tip_word_hex'] == '0200c07f'


def test_added_field_is_a_difference_even_when_every_common_word_matches(tmp_path):
    a, b = tmp_path / 'base', tmp_path / 'tip'
    _frame(a, [0, 1, 2, 3])
    _frame(b, [0, 1, 2, 3], extra=True)
    result = compare_folders(a, b)
    assert result['fields'] == 1 and result['values_differing'] == 0
    assert result['verdict'] == 'DIFFERENT'
    assert result['missing'][0]['field'] == 'NEW'
    assert result['missing'][0]['before'] == 'ABSENT'


def test_empty_arm_and_shape_changes_refuse_to_claim_identity(tmp_path):
    a, b = tmp_path / 'base', tmp_path / 'tip'
    _frame(a, [0, 1, 2, 3])
    b.mkdir()
    with pytest.raises(ValueError, match='history frames'):
        compare_folders(a, b)
    with Dataset(b / 'wrfout_d01_2026-10-03_12:00:00', 'w') as ds:
        ds.createDimension('x', 3)
        ds.createVariable('T', 'f4', ('x',))[:] = np.zeros(3, np.float32)
    with pytest.raises(ValueError, match='shape differs'):
        compare_folders(a, b)
