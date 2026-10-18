from pathlib import Path
import importlib.util

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location(
    "coupled_gf_receipt", Path(__file__).parents[1] / "tools/coupled_gf_receipt.py")
receipt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(receipt)


def test_missing_launch_is_not_reported_as_empty_exactness(tmp_path):
    with pytest.raises(ValueError, match="no captured"):
        receipt.collect(tmp_path, "woof", (2, 3, 5))


def test_partial_domain_capture_is_rejected(tmp_path):
    np.savez(tmp_path / "gf_launch_0000.npz",
             lvin=np.zeros((14, len(receipt._IN_LEV), 2), dtype=np.float32),
             scin=np.zeros((14, 9), dtype=np.float32),
             iin=np.zeros((14, 3), dtype=np.int32))
    with pytest.raises(ValueError, match="exactly one domain"):
        receipt.collect(tmp_path, "woof", (2, 3, 5))
