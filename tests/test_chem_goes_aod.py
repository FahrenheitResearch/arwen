"""A tiny real GOES-18 AOD crop, decoded and quality-gated in Rust."""
from pathlib import Path
import numpy as np
import pytest
from gpuwm.obs.goes_aod import read_goes_aod

DATA = Path(__file__).parent / "data/goes_aod/aod-crop.hex"


def test_real_aod_crop(tmp_path):
    path = tmp_path / "crop.goespack"
    path.write_bytes(bytes.fromhex(DATA.read_text()))
    pack = read_goes_aod(path)
    assert pack.family == "aod"
    assert pack.plane("aod").shape == (16, 16)
    assert np.isfinite(pack.plane("aod")).sum() == 43
    assert np.all(pack.plane("aod_dqf")[np.isfinite(pack.plane("aod"))] == 0)
    assert pack.has_dqf_planes
    assert pack.meta["wavelength_nm"] == 550


def test_corrupt_aod_payload(tmp_path):
    raw = bytearray(bytes.fromhex(DATA.read_text()))
    raw[-1] ^= 1
    path = tmp_path / "bad.goespack"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match="payload hashes"):
        read_goes_aod(path)
