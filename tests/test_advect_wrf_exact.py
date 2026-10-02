"""Exact advection through real CUDA launchers and compiled WRF outputs."""
import os
import pytest
from conftest import requires_gpu


@requires_gpu
def test_exact_advection_active_specified_cases_match_all_words():
    if os.environ.get("GPUWM_WRF_EXACT_ADVECTION") != "1":
        pytest.skip("requires an exact advection process")
    from tools.advect_wrf_exact.compare import compare
    results=compare()
    from gpuwm.verify.advect_oracle import load_advect_cases
    selected_names={case.name for case in load_advect_cases()
                    if case.metadata["specified"] or not
                    (case.metadata["open_x"] or case.metadata["open_y"])}
    selected={name:row for name,row in results.items() if name in selected_names}
    assert selected, "the real specified fixture must be exercised"
    for name,row in selected.items():
        for routine,metric in row.items():
            assert metric["different_words"]==0,(name,routine,metric)
