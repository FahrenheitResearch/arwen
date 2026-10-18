import numpy as np
import pytest
from tools.coupled_trace_receipt import digest, collect


def test_one_word_change_identifies_only_its_block():
    before = np.arange(33, dtype=np.float32)
    after = before.copy()
    after.view(np.uint32)[19] += 1
    a, b = digest(before, 8), digest(after, 8)
    assert a["sha256"] != b["sha256"]
    assert [i for i, (x, y) in enumerate(zip(a["blocks"], b["blocks"])) if x != y] == [2]


def test_signed_zero_difference_is_retained():
    assert digest(np.array([0.0], np.float32))["sha256"] != digest(np.array([-0.0], np.float32))["sha256"]


def test_unknown_field_cannot_produce_an_empty_passing_receipt(tmp_path):
    with pytest.raises(ValueError, match="unknown trace keys"):
        collect(tmp_path, "wrf", (2, 2, 2), 4, False, keys={"absent/field"})


def test_requested_step_does_not_select_the_previous_step(tmp_path):
    np.savez(tmp_path / "0001_s001_rk0_it00_p00_phy_prep.npz",
             x_theta=np.ones((2,2,2), np.float32))
    value = np.full((2,2,2), 3., np.float32)
    np.savez(tmp_path / "0002_s002_rk0_it00_p00_phy_prep.npz", x_theta=value)
    key = "phy_prep/output/theta"
    result = collect(tmp_path, "woof", (2,2,2), 4, False, keys={key}, step=2)
    assert not result["missing"]
    assert result["rows"][key]["sha256"] == digest(value)["sha256"]


def test_inactive_cumulus_cannot_be_requested_as_computed_state(tmp_path):
    with pytest.raises(ValueError, match="no computed cumulus"):
        collect(tmp_path, "woof", (2,2,2), 4, False, cu_physics=0,
                keys={"coupled_cumulus/output/rthcuten"})
