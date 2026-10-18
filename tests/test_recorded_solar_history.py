"""Grade the actual radiation-call fields against a recorded native run."""
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("field", ["COSZEN", "SWDOWN", "RTHRATSW"])
def test_native_solar_intrinsic_reaches_complete_history_fields(field):
    history = os.environ.get("GPUWM_EXACT_HISTORY")
    reference = os.environ.get("GPUWM_EXACT_REFERENCE")
    step = os.environ.get("GPUWM_EXACT_STEP")
    if not (history and reference and step):
        pytest.skip("set an actual history, native reference and step")
    from gpuwm.verify_exact.recording import load_run_record
    from gpuwm.verify_exact.replay import read_frame
    from gpuwm.verify_exact.compare import field_digest

    stock = load_run_record(Path(reference))
    actual = read_frame(Path(history))
    metadata = stock.fields[field]
    assert actual[field].shape == metadata.shape
    assert actual[field].dtype == np.dtype(metadata.dtype)
    assert field_digest(actual[field], metadata.dtype, metadata.shape) == stock.digests[int(step)][field]
