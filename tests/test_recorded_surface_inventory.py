"""Grade the literal initialized Registry inventory from an actual forecast."""
import os
from pathlib import Path
import numpy as np
import pytest

@pytest.mark.parametrize("field", ["SFROFF", "UDROFF", "ACSNOM", "NOAHRES", "TKE_PBL", "SWNORM"])
def test_initialized_registry_field_is_emitted_with_native_words(field):
    history = os.environ.get("GPUWM_EXACT_HISTORY")
    reference = os.environ.get("GPUWM_EXACT_REFERENCE")
    if not (history and reference):
        pytest.skip("set actual initial history and native reference")
    from gpuwm.verify_exact.recording import load_run_record
    from gpuwm.verify_exact.replay import read_frame
    from gpuwm.verify_exact.compare import field_digest
    stock = load_run_record(Path(reference))
    actual = read_frame(Path(history))
    metadata = stock.fields[field]
    assert field in actual, f"missing native Registry history field {field}"
    assert actual[field].shape == metadata.shape
    assert actual[field].dtype == np.dtype(metadata.dtype)
    assert field_digest(actual[field], metadata.dtype, metadata.shape) == stock.digests[0][field]
