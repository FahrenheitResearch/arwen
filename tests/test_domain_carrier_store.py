"""A domain-owned carrier (fixed shape) is kept whole in the host stores.

Breakage: with sf_ocean_physics = 2 (the default for runs of 48 h or more),
fields/_pwp_status (shape (1,) int32) passed the manifest as a domain carrier
but the slab loop scattered it by rows: "IndexError: tuple index out of range"
at the start of every store-direct forecast (full CONUS HRRR, --devices 2,
2026-10-21).
"""

import numpy as np
import pytest

from gpuwm.ingest.prepared_store import PreparedStoreError, _hold_domain_carrier
from tilestream.bigdomain import _hold_domain_carrier as _hold_streamed
from tilestream.hoststore import manifest_from_arrays


def test_the_pwp_status_word_has_a_fixed_shape_in_the_manifest():
    values = {'fields/tsk': np.empty((12, 17), np.float32),
              'fields/_pwp_status': np.zeros(1, np.int32)}
    manifest = {spec.name: spec for spec in manifest_from_arrays(values, 16, 12, 17)}
    assert manifest['fields/_pwp_status'].fixed_shape == (1,)
    assert manifest['fields/tsk'].fixed_shape is None


@pytest.mark.parametrize("hold, error", [(_hold_domain_carrier, PreparedStoreError),
                                         (_hold_streamed, ValueError)])
def test_every_slab_keeps_the_word_whole_and_a_disagreeing_slab_is_refused(hold, error):
    dst = np.full(1, -1, np.int32)
    for index in range(3):
        hold(dst, np.zeros(1, np.int32), 'fields/_pwp_status', index)
    assert dst.tolist() == [0]
    with pytest.raises(error, match="slab-invariant"):
        hold(dst, np.ones(1, np.int32), 'fields/_pwp_status', 3)
    with pytest.raises(error, match="fixed shape"):
        hold(dst, np.zeros(2, np.int32), 'fields/_pwp_status', 4)
