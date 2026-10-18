"""Grade complete surface fields in an actual history frame against stock hashes."""
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("field", ["QFX", "Q2"])
def test_recorded_surface_flux_words(field):
    history = os.environ.get("GPUWM_EXACT_HISTORY")
    reference = os.environ.get("GPUWM_EXACT_REFERENCE")
    step = os.environ.get("GPUWM_EXACT_STEP")
    if not (history and reference and step):
        pytest.skip("set the real history, stock reference directory and recorded step")
    from gpuwm.verify_exact.compare import field_digest
    from gpuwm.verify_exact.recording import load_run_record
    from gpuwm.verify_exact.replay import read_frame

    stock = load_run_record(Path(reference))
    expected = stock.digests[int(step)][field]
    assert expected is not None
    candidate = read_frame(Path(history))
    assert field in candidate
    metadata = stock.fields[field]
    assert candidate[field].dtype == np.dtype(metadata.dtype)
    assert candidate[field].shape == metadata.shape
    assert field_digest(candidate[field], metadata.dtype, metadata.shape) == expected
