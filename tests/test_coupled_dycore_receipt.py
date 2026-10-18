import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "coupled_dycore_receipt", Path(__file__).parents[1] / "tools/coupled_dycore_receipt.py")
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)


def test_missing_held_dump_is_explicit(tmp_path):
    result = receipt.collect(tmp_path, "woof", (2, 3, 4), ["held_diffusion/u"])
    assert not result["rows"]
    assert result["missing"][0]["key"] == "held_diffusion/u"


def test_wrong_stagger_is_refused(tmp_path):
    np.savez(tmp_path / "0001_s001_rk0_it00_c02_vsmag.npz",
             x_hv_ru=np.zeros((2, 3, 4), dtype=np.float32))
    result = receipt.collect(tmp_path, "woof", (2, 3, 4), ["held_diffusion/u"])
    assert not result["rows"]
    assert "required (2, 3, 5)" in result["missing"][0]["error"]


def test_unknown_selection_is_refused(tmp_path):
    with pytest.raises(ValueError, match="unknown"):
        receipt.collect(tmp_path, "woof", (2, 3, 4), ["not-a-point"])
