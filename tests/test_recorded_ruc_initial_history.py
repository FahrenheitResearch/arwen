"""Grade actual initialized land fields against the native initial frame."""
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("field", ["SH2O", "ALBBCK"])
def test_native_initialized_surface_field_is_not_replaced_by_input_placeholder(field):
    history = os.environ.get("GPUWM_EXACT_HISTORY")
    reference = os.environ.get("GPUWM_EXACT_REFERENCE")
    if not (history and reference):
        pytest.skip("set an actual initial history and native reference")
    from gpuwm.verify_exact.recording import load_run_record
    from gpuwm.verify_exact.replay import read_frame
    from gpuwm.verify_exact.compare import field_digest

    stock = load_run_record(Path(reference))
    actual = read_frame(Path(history))
    metadata = stock.fields[field]
    assert actual[field].shape == metadata.shape
    assert actual[field].dtype == np.dtype(metadata.dtype)
    assert field_digest(actual[field], metadata.dtype, metadata.shape) == stock.digests[0][field]
