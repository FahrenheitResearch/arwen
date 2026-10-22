"""A zero reference must remain a portable, unambiguous comparison receipt."""
import json

import numpy as np

from tools.ccpp_column_oracle.compare_drag_suite import _stats


def test_nonzero_tendency_against_zero_has_no_relative_ratio():
    result = _stats(np.zeros((2, 2)), np.ones((2, 2)))
    assert result["rel_l2"] is None
    assert result["max_abs_diff"] == 1
    assert result["columns_differing"] == 2
    assert json.loads(json.dumps(result, allow_nan=False)) == result
