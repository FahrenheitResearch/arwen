"""The immutable eager interpolation literal is checked on the CPU leg."""
import hashlib

from test_smag2d_scalar_speed_identity import EAGER_HELPERS, EAGER_HELPERS_SHA256


def test_original_eager_interpolation_literal_is_pinned_on_cpu():
    assert hashlib.sha256(EAGER_HELPERS.encode()).hexdigest() == EAGER_HELPERS_SHA256
