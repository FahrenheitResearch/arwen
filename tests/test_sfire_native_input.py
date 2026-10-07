"""Native input inventory and exact directional aliases for coupled fire."""
from types import SimpleNamespace

import numpy as np
import pytest

from gpuwm.ingest.wrfinput import active_moisture_inventory, resolve_map_factors


def test_disabled_microphysics_retains_active_vapor_only():
    required, allowed = active_moisture_inventory(SimpleNamespace(moist=True, mp_physics=0))
    assert required == allowed == frozenset({"QVAPOR"})


def test_native_directional_aliases_preserve_selected_words_and_provenance():
    raw = {}
    for point, shape in (("M", (5, 7)), ("U", (5, 8)), ("V", (6, 7))):
        name = "MAPFAC_" + point
        raw[name] = np.zeros(shape, "f4")
        raw[name + "X"] = np.linspace(0.8, 1.2, np.prod(shape), dtype="f4").reshape(shape)
        raw[name + "Y"] = raw[name + "X"].copy()
    receipt = resolve_map_factors(raw)
    for name, row in receipt.items():
        assert raw[name] is raw[row["source"]]
        assert np.array_equal(raw[name].view("u4"), raw[row["equivalent"]].view("u4"))
        assert len(row["original_sha256"]) == len(row["selected_sha256"]) == 64


def test_anisotropic_directional_geometry_cannot_be_collapsed():
    raw = {"MAPFAC_M": np.zeros((2, 3), "f4"), "MAPFAC_MX": np.ones((2, 3), "f4"),
           "MAPFAC_MY": np.full((2, 3), 1.01, "f4")}
    with pytest.raises(ValueError, match="anisotropic metrics"):
        resolve_map_factors(raw)
    assert np.count_nonzero(raw["MAPFAC_M"]) == 0


def test_positive_legacy_factor_remains_its_own_authority():
    value = np.ones((2, 3), "f4")
    raw = {"MAPFAC_M": value, "MAPFAC_MX": np.full((2, 3), 0.9, "f4"),
           "MAPFAC_MY": np.full((2, 3), 1.1, "f4")}
    assert resolve_map_factors(raw) == {}
    assert raw["MAPFAC_M"] is value
