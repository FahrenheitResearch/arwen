"""WRF v4.6.1 source-oracle checks for the shared surface-forcing seams."""

from __future__ import annotations

import numpy as np


def test_ruc_arw_species_partition_matches_the_wrf_source_probe():
    """Exercise every temperature/frozen-fraction arm and a dry column."""
    from gpuwm.core.ruc import _ruc_arw_precipitation_partition
    from tools.surface_forcing_wrf461_oracle.transcribe_surface_forcing import (
        ruc_arw_precipitation,
    )

    inputs = {
        "rainbl": np.array([2.0, 2.0, 2.0, 0.0, 0.3, 1.0], np.float32),
        "rainncv": np.array([1.0, 1.0, 1.0, 0.0, 0.5, 0.6], np.float32),
        "snowncv": np.array([0.3, 0.4, 0.2, 0.0, 0.2, 0.1], np.float32),
        "graupelncv": np.array([0.2, 0.1, 0.3, 0.0, 0.1, 0.15],
                               np.float32),
        "frzfrac": np.array([0.6, 0.0, 0.7, 0.0, 1.0, 0.5], np.float32),
        "tabs": np.array([270.0, 270.0, 275.0, 280.0, 270.0, 272.999],
                         np.float32),
        "dt": np.float32(12.0),
    }
    expected = ruc_arw_precipitation(**inputs)
    actual = _ruc_arw_precipitation_partition(
        inputs["rainbl"], inputs["rainncv"], inputs["snowncv"],
        inputs["graupelncv"], inputs["frzfrac"], inputs["tabs"], inputs["dt"])
    for name, observed in zip(
            ("prcpms", "newsnms", "snowrat", "grauprat", "icerat", "curat"),
            actual):
        np.testing.assert_array_equal(observed, expected[name], err_msg=name)

    # This is the root of B-R1: changing only the graupel writer must change
    # RUC's constituent fraction instead of disappearing into SR.
    perturbed = dict(inputs)
    perturbed["graupelncv"] = inputs["graupelncv"].copy()
    perturbed["graupelncv"][0] += np.float32(0.1)
    changed = _ruc_arw_precipitation_partition(
        perturbed["rainbl"], perturbed["rainncv"], perturbed["snowncv"],
        perturbed["graupelncv"], perturbed["frzfrac"], perturbed["tabs"],
        perturbed["dt"])
    assert changed[4][0] != actual[4][0]


def test_noahmp_six_rates_match_the_wrf_source_probe_for_multiple_mixtures():
    from gpuwm.core.surface_forcing import (
        SurfacePrecipitationForcing,
        noahmp_six_precipitation_rates,
    )
    from tools.surface_forcing_wrf461_oracle.transcribe_surface_forcing import (
        noahmp_six_rates,
    )

    values = {
        "rainbl": np.array([1.0, 2.0, 0.0, 0.7], np.float32),
        "sr": np.array([0.0, 0.75, 1.0, 0.25], np.float32),
        "rainc": np.array([0.2, 0.4, 0.0, 0.1], np.float32),
        "rainnc": np.array([0.3, 1.1, 0.0, 0.8], np.float32),
        "rainshv": np.array([0.1, 0.0, 0.0, 0.1], np.float32),
        "snow": np.array([0.0, 0.5, 0.0, 0.2], np.float32),
        "graupel": np.array([0.0, 0.2, 0.0, 0.1], np.float32),
        "hail": np.array([0.0, 0.1, 0.0, 0.05], np.float32),
        "dt": np.float32(10.0),
    }
    forcing = SurfacePrecipitationForcing(
        rain_convective=values["rainc"],
        rain_nonconvective=values["rainnc"],
        rain_shallow_convective=values["rainshv"],
        snow_nonconvective=values["snow"],
        graupel_nonconvective=values["graupel"],
        hail_nonconvective=values["hail"],
    )
    expected = noahmp_six_rates(**values)
    actual = noahmp_six_precipitation_rates(
        values["rainbl"], values["sr"], forcing, values["dt"], arrays=np)
    assert actual.keys() == expected.keys()
    for name in actual:
        np.testing.assert_array_equal(actual[name], expected[name],
                                      err_msg=name)


def test_ruc_fractional_ice_blends_match_the_wrf_source_probe():
    from gpuwm.core.ruc_runtime import (
        _ruc_fractional_deblend,
        _ruc_fractional_reblend,
    )
    from tools.surface_forcing_wrf461_oracle.transcribe_surface_forcing import (
        ruc_fractional_post,
        ruc_fractional_pre,
    )

    xice = np.array([0.55, 0.75], np.float32)
    ice_albedo = np.array([0.62, 0.68], np.float32)
    ice_emiss = np.array([0.96, 0.97], np.float32)
    blended_albedo = ruc_fractional_post(
        ice=ice_albedo, sea=np.float32(0.08), xice=xice)
    blended_emiss = ruc_fractional_post(
        ice=ice_emiss, sea=np.float32(0.98), xice=xice)
    expected = ruc_fractional_pre(
        blended_albedo=blended_albedo,
        blended_emiss=blended_emiss,
        xice=xice)
    mask = np.ones_like(xice, dtype=bool)
    actual_albedo = _ruc_fractional_deblend(
        blended_albedo, 0.08, xice, mask, arrays=np)
    actual_emiss = _ruc_fractional_deblend(
        blended_emiss, 0.98, xice, mask, arrays=np)
    np.testing.assert_array_equal(actual_albedo, expected["albedo"])
    np.testing.assert_array_equal(actual_emiss, expected["emiss"])

    sea_flux = np.array([12.0, 18.0], np.float32)
    ice_flux = np.array([-25.0, -31.0], np.float32)
    actual_flux = _ruc_fractional_reblend(
        ice_flux, sea_flux, xice, mask, arrays=np)
    expected_flux = ruc_fractional_post(
        ice=ice_flux, sea=sea_flux, xice=xice)
    np.testing.assert_array_equal(actual_flux, expected_flux)
