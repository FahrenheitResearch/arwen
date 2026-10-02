"""Opt-in native acoustic routine parity against compiled WRF v4.7.1."""
import os

import pytest

from conftest import requires_gpu


@requires_gpu
def test_exact_acoustic_routines_have_no_differing_words():
    if os.environ.get("GPUWM_WRF_EXACT") != "1":
        pytest.skip("requires an exact-mode process")
    library=os.environ.get("GPUWM_SMALLSTEP_ORACLE_LIB")
    if not library:
        pytest.skip("requires the compiled, unmodified WRF oracle")
    from tools.smallstep_wrf_exact.compare import compare, differences
    result=compare(library)
    assert not list(differences(result))
