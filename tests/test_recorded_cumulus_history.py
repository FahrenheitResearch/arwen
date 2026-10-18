"""Grade a real emitted first-step GF history against the stock recording."""
import os
from pathlib import Path

import numpy as np
import pytest


@pytest.mark.parametrize("field", ["RUCUTEN", "RVCUTEN"])
def test_emitted_unused_cumulus_momentum_is_the_stock_producer(field):
    history = os.environ.get("GPUWM_EXACT_HISTORY")
    if not history:
        pytest.skip("set GPUWM_EXACT_HISTORY to a real first-step A071 wrfout")
    from gpuwm.verify_exact.recording import load_run_record
    from gpuwm.verify_exact.replay import read_frame

    root = Path(os.environ.get("GPUWM_EXACT_RECORDINGS", "/work/cache/combo-rec-full"))
    stock = load_run_record(root / "scalar" / "iowa-convective" / "A071")
    reference = stock.raw(1)
    assert reference is not None and field in reference
    candidate = read_frame(Path(history))
    assert field in candidate
    assert candidate[field].dtype == reference[field].dtype
    assert candidate[field].shape == reference[field].shape
    np.testing.assert_array_equal(candidate[field].view("u4"), reference[field].view("u4"))

@pytest.mark.parametrize("field", ["RAINC", "RUCUTEN", "RVCUTEN", "RTHCUTEN", "RQVCUTEN", "RQCCUTEN"])
def test_recorded_convective_history_words(field):
    history = os.environ.get("GPUWM_EXACT_HISTORY")
    reference = os.environ.get("GPUWM_EXACT_REFERENCE")
    step = os.environ.get("GPUWM_EXACT_STEP")
    if not (history and reference and step):
        pytest.skip("set an actual history frame, stock reference and step")
    from gpuwm.verify_exact.recording import load_run_record
    from gpuwm.verify_exact.replay import read_frame
    from gpuwm.verify_exact.compare import field_digest

    stock = load_run_record(Path(reference))
    actual = read_frame(Path(history))
    metadata = stock.fields[field]
    assert actual[field].shape == metadata.shape
    assert actual[field].dtype == np.dtype(metadata.dtype)
    assert field_digest(actual[field], metadata.dtype, metadata.shape) == stock.digests[int(step)][field]
