"""The canonical digest refuses foreign byte order, not an explicitly native one.

WOOF 1.1.0's dry bake (2026-10-19): every hex global forecast stopped at its final digest because
'atmosphere/saved/vertical_velocity' arrived as dtype '<f4' (a NetCDF read), which is this machine's order
on x86 but was refused because its byteorder character is '<' rather than '='.
"""
import sys

import numpy as np
import pytest

from gpuwm import state_digest


def test_an_explicit_native_order_is_accepted():
    native = "<" if sys.byteorder == "little" else ">"
    value = np.arange(6, dtype=f"{native}f4")
    assert state_digest._canonical_host("atmosphere/saved/vertical_velocity", value).dtype == np.float32


def test_a_foreign_order_is_still_refused():
    foreign = ">" if sys.byteorder == "little" else "<"
    with pytest.raises(ValueError, match="byte order"):
        state_digest._canonical_host("x", np.arange(6, dtype=f"{foreign}f4"))
