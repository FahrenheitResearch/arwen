import numpy as np
import pytest

from gpuwm.verify_exact.compare import FieldMeta, RunRecord, field_digest
from tools.combo_round4_scalar_probe import validate_reference


def test_recovered_raw_reference_requires_the_recorded_bits():
    values = np.array([1, 2], dtype=np.float32)
    meta = FieldMeta("U", "<f4", (2,))
    ref = RunRecord({"U": meta}, {1: {"U": field_digest(values, meta.dtype, meta.shape)}})
    assert validate_reference({"U": values}, ref) == 1
    changed = values.copy()
    changed[0] = np.nextafter(changed[0], np.float32(2))
    with pytest.raises(ValueError, match="does not match"):
        validate_reference({"U": changed}, ref)
    with pytest.raises(ValueError, match="does not match"):
        validate_reference({}, ref)
    with pytest.raises(ValueError, match="does not match"):
        validate_reference({"U": values}, ref, excluded=["U"])
