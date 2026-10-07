"""The start state's W is carried from a source that publishes one.

Named breakage: a start from the HRRR (or RAP) native analysis under the
old exact-zero policy dropped every updraft the analysis held, so the
storms it carried collapsed and re-formed in the first hour (2024-05-21
18Z Iowa crop, f01 FSS35 at 27 km 0.30 against HRRR's own 0.42).  These
tests hold the three pieces of the fix: the native mappings declare the
published omega, the join packs it, and the initializer converts and
staggers it the way UPP published it.
"""
import hashlib
import json

import numpy as np
import pytest

from gpuwm.ingest.real import (
    CARRIED_VERTICAL_VELOCITY_POLICY, UPP_OMEGA_D608, UPP_OMEGA_G, UPP_OMEGA_RD,
    WRF_REAL_VERTICAL_VELOCITY_POLICY, interface_vertical_velocity,
    vertical_velocity_from_omega, vertical_velocity_receipt)
from gpuwm.mapped_source import (
    REGULAR_JOIN_DROPPED_FIELDS, VERTICAL_VELOCITY_LEGACY_NAMES, load_mapping)
from gpuwm.source_authorities import packaged_authorities, packaged_profile


NATIVE_PROFILES = ("hrrr-native-grib2-v1", "rap-native-grib2-v1")


@pytest.mark.parametrize("profile", NATIVE_PROFILES)
def test_the_native_mapping_declares_the_published_omega(profile):
    """VVEL (0/2/8) on every hybrid level, in the units the product has."""
    mapping = load_mapping(packaged_authorities(profile)["mapping"])
    omega = mapping["fields"]["pressure_vertical_velocity"]
    selector = omega["selectors"][0]
    assert (selector["discipline"], selector["category"],
            selector["parameter"], selector["level_type"]) == (0, 2, 8, 105)
    assert omega["units"] == {"source": "Pa s-1", "target": "Pa s-1"}
    assert omega["location"] == "mass"
    assert omega["target_axes"] == ["vertical", "y", "x"]
    policy = mapping["target"]["initialization_policies"]["vertical_velocity"]
    assert policy.startswith("carried-from-pressure-vertical-velocity")
    assert "vertical_velocity" in mapping["target"]["policy_controlled_fields"]


@pytest.mark.parametrize("profile", NATIVE_PROFILES)
def test_the_packaged_pin_is_the_mapping_on_disk(profile):
    """A re-pinned digest is what lets the wheel refuse a drifted table."""
    authorities = packaged_authorities(profile)
    pinned = packaged_profile(profile)["sha256"]["mapping"]
    observed = hashlib.sha256(authorities["mapping"].read_bytes()).hexdigest()
    assert pinned == observed


def test_the_join_owns_both_vertical_velocity_spellings():
    assert VERTICAL_VELOCITY_LEGACY_NAMES == {
        "pressure_vertical_velocity": ("OMEGA", "Pa s-1"),
        "vertical_velocity": ("WW", "m s-1"),
    }
    # The drop by name was the guard; it is retired with the defect.
    assert REGULAR_JOIN_DROPPED_FIELDS == ()


def test_omega_inverts_upp_exactly():
    """UPP: omga = -w * p * g / (Rd * T * (1 + 0.608 q)).  Round trip."""
    rng = np.random.default_rng(7)
    shape = (5, 3, 4)
    w = rng.normal(0.0, 8.0, shape)
    temperature = rng.uniform(210.0, 305.0, shape)
    qv = rng.uniform(0.0, 0.02, shape)
    pressure = rng.uniform(15000.0, 101000.0, shape)
    omega = -w * pressure * UPP_OMEGA_G / (
        UPP_OMEGA_RD * temperature * (1.0 + UPP_OMEGA_D608 * qv))
    back = vertical_velocity_from_omega(omega, temperature, qv, pressure)
    assert np.allclose(back, w, rtol=1e-12, atol=1e-12)
    # Sign: rising air is omega < 0 and w > 0.
    assert np.all(np.sign(back) == -np.sign(omega))
    # Order of magnitude at 800 hPa, 280 K: about 0.1 m s-1 per Pa s-1.
    one = vertical_velocity_from_omega(
        np.array([-10.0]), np.array([280.0]), np.array([0.005]),
        np.array([80000.0]))
    assert 0.95 < float(one[0]) < 1.1


def test_interfaces_average_their_bounding_levels_and_pin_the_ends():
    w_mass = np.arange(4 * 2 * 3, dtype=np.float32).reshape(4, 2, 3) - 10.0
    w = interface_vertical_velocity(w_mass)
    assert w.shape == (5, 2, 3)
    assert w.dtype == np.float32
    assert np.all(w[0] == 0.0)            # set_w_surface owns the ground
    assert np.all(w[-1] == 0.0)           # WRF's model-top condition
    assert np.allclose(w[1:4], 0.5 * (w_mass[:-1] + w_mass[1:]))
    with pytest.raises(ValueError):
        interface_vertical_velocity(np.zeros((1, 2, 3), dtype=np.float32))


def test_the_receipt_says_which_field_and_how_strong():
    w_mass = np.zeros((3, 2, 2), dtype=np.float32)
    w_mass[1, 0, 0] = 12.0
    w_mass[2, 1, 1] = -3.0
    w = interface_vertical_velocity(w_mass)
    receipt = vertical_velocity_receipt("OMEGA", w_mass, w)
    assert receipt["policy"] == "carried-from-source"
    assert receipt["source_field"] == "OMEGA"
    assert "UPP" in receipt["conversion"]
    assert receipt["mass_level_w_m_s"] == {
        "minimum": -3.0, "maximum": 12.0,
        "cells_above_1_m_s": 1, "cells_below_minus_1_m_s": 1}
    assert receipt["interface_w_m_s"]["maximum"] == 6.0
    assert receipt["interface_w_m_s"]["nonzero_count"] == 3
    assert set(CARRIED_VERTICAL_VELOCITY_POLICY) <= set(receipt)
    # The zero policy and the carried receipt never share a policy word.
    assert WRF_REAL_VERTICAL_VELOCITY_POLICY["policy"] == "exact-fp32-zero"
    assert json.dumps(receipt)  # JSON-clean for the prepared-cache header
