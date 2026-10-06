"""Sun-angle albedo and RUC sea-ice seam against compiled fork Fortran."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from gpuwm.core.solar_albedo import MODIS_ISTWE, update_solar_albedo_host

F = np.float32
DATA = Path(__file__).parent / "data" / "solar_albedo_fork.npz"


def _fixture():
    return np.load(DATA, allow_pickle=False)


def _fields(fixture):
    return {key[3:]: fixture[key].copy() for key in fixture.files
            if key.startswith("in_")}


def _words(actual, expected):
    np.testing.assert_array_equal(actual.view(np.uint32),
                                  expected.view(np.uint32))


def test_source_pinned_fortran_fixture_provenance():
    from tools.solar_albedo_oracle.build import COMMIT, PIN
    with _fixture() as fixture:
        receipt = json.loads(str(fixture["receipt"]))
    assert receipt["commit"] == COMMIT
    assert receipt["sources"] == PIN
    assert receipt["blocks"]["radiation_update"] == [1038, 1063]
    assert "-ffp-contract=off" in receipt["flags"]
    assert "-O0" in receipt["flags"]


def test_three_radiation_calls_match_fork_words():
    with _fixture() as fixture:
        fields = _fields(fixture)
        original_background = fields["albbck"].copy()
        for call, coszen in enumerate(fixture["coszen"]):
            if call:
                fields["albedo"][...] = F(0.123)
            update_solar_albedo_host(fields, coszen, initialize=call == 0)
            _words(fields["albsol"], fixture["out_albsol"][call])
            _words(fields["albbcksol"], fixture["out_albbcksol"][call])
        _words(fields["albbck"], original_background)


def test_modis_classes_overhead_sun_and_half_height():
    with _fixture() as fixture:
        overhead = fixture["out_albsol"][2, :21]
        half = fixture["out_albsol"][1, :21]
    _words(overhead, np.full(21, F(0.17), F))
    d = np.where(np.array(MODIS_ISTWE) == 1, F(0.1), F(0.25))
    expected = F(0.17) * ((F(1) + F(2) * d)
                         / (F(1) + (F(2) * d) * F(0.5)))
    _words(half, expected)
    assert [i + 1 for i, cls in enumerate(MODIS_ISTWE) if cls == 1] == [
        1, 2, 3, 4, 5, 8, 9, 15, 18]


def test_cap_is_only_on_albsol_and_masked_cells_keep_first_copy():
    with _fixture() as fixture:
        # Low sun can normalize a high background above the cap. The
        # background passed to RUC keeps that uncapped word.
        assert fixture["out_albsol"][0, 21] == F(0.9)
        assert fixture["out_albbcksol"][0, 21] > F(0.9)
        # Zero sun, negative sun, XLAND==1.5, snow, tiny ice and water.
        for call in range(3):
            _words(fixture["out_albsol"][call, 22:28],
                   np.full(6, F(0.9), F))
            _words(fixture["out_albbcksol"][call, 22:28],
                   fixture["in_albbck"][22:28])


@pytest.mark.parametrize("fractional", [0, 1])
def test_ruc_ice_override_and_fractional_blend_match_fork(fractional):
    from gpuwm.core.ruc_runtime import (
        _ruc_fractional_deblend, _ruc_fractional_reblend,
        _ruc_seaice_albedo_override,
    )
    with _fixture() as fixture:
        xice = fixture["ice_xice"].copy()
        threshold = 0.02 if fractional else 0.5
        background, ice = _ruc_seaice_albedo_override(
            fixture["ice_in_albbcksol"], xice, 0.65, arrays=np,
            xice_threshold=threshold)
        _words(background, fixture[f"ice_{fractional}_pre_albbcksol"])
        albedo = fixture["ice_in_albsol"].copy()
        emiss = fixture["ice_in_emiss"].copy()
        if fractional:
            albedo = _ruc_fractional_deblend(albedo, 0.08, xice, ice,
                                            arrays=np)
            emiss = _ruc_fractional_deblend(emiss, 0.98, xice, ice,
                                           arrays=np)
        _words(albedo, fixture[f"ice_{fractional}_pre_albsol"])
        _words(emiss, fixture[f"ice_{fractional}_pre_emiss"])
        if fractional:
            albedo = _ruc_fractional_reblend(albedo, F(0.08), xice, ice,
                                            arrays=np)
            emiss = _ruc_fractional_reblend(emiss, F(0.98), xice, ice,
                                           arrays=np)
        _words(albedo, fixture[f"ice_{fractional}_post_albsol"])
        _words(emiss, fixture[f"ice_{fractional}_post_emiss"])


def test_flat_coszen_accepts_two_dimensional_surface_fields():
    with _fixture() as fixture:
        fields = {key: array.reshape(32, 64) for key, array in
                  _fields(fixture).items()}
        update_solar_albedo_host(fields, fixture["coszen"][0], initialize=True)
        _words(fields["albsol"].reshape(-1), fixture["out_albsol"][0])


def test_active_class_outside_modis_table_is_refused():
    with _fixture() as fixture:
        fields = _fields(fixture)
        fields["ivgtyp"][0] = 24
        with pytest.raises(ValueError, match="MODIS IVGTYP classes 1 through 21"):
            update_solar_albedo_host(fields, fixture["coszen"][0])
