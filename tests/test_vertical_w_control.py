"""CPU argument ownership and strict hash join for the independent W control."""
import copy

import numpy as np
import pytest

from tools.wrf_diffusion_oracle.vertical_w_control import coefficient_matched_prepared, match_gpu_receipt


def test_control_changes_only_one_input_and_owns_its_coefficient_copy():
    prepared = {"kmh": np.full((3, 4, 5), 8., dtype=np.float32, order="F"),
                "kmv": np.full((3, 4, 5), .25, dtype=np.float32, order="F"),
                "w": np.arange(60, dtype=np.float32).reshape(3, 4, 5),
                "rho": np.ones((3, 4, 5), dtype=np.float32)}
    original = {key: value.tobytes() for key,value in prepared.items()}
    controlled = coefficient_matched_prepared(prepared)
    assert {key for key in prepared if controlled[key].tobytes() != original[key]} == {"kmh"}
    assert controlled["kmh"].flags.f_contiguous
    assert not np.shares_memory(controlled["kmh"], prepared["kmv"])
    for key in ("kmv", "w", "rho"):
        assert controlled[key] is prepared[key]
    controlled["kmh"].fill(11.)
    assert {key:value.tobytes() for key,value in prepared.items()} == original


def test_native_join_requires_complete_inventory_and_exact_hashes():
    fields = {"case/km2_flux0/w": {"words": 3, "controlled_native_sha256": "a"*64,
                                   "stock_native_sha256": "b"*64}}
    proof = {"fields": fields}
    gpu = {"cases": {"case": {"km2_flux0": {"w": {"words": 3,
              "gpu_sha256": "a"*64, "different_words": 2}}}}}
    joined = match_gpu_receipt(proof, gpu)
    assert joined["all_current_match_declared_native"] is True
    assert joined["current_stock_native_different_words"] == 2
    altered = copy.deepcopy(gpu)
    altered["cases"]["case"]["km2_flux0"]["w"]["gpu_sha256"] = "c"*64
    assert match_gpu_receipt(proof, altered)["all_current_match_declared_native"] is False
    altered["cases"]["case"]["km2_flux0"]["w"]["words"] = 4
    with pytest.raises(ValueError, match="word count differs"):
        match_gpu_receipt(proof, altered)
    with pytest.raises(ValueError, match="inventories differ"):
        match_gpu_receipt(proof, {"cases": {}})
