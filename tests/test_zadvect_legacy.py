"""Pinned earlier WRF implicit advection, including nonzero implicit flux.

The breakage guarded here: selecting the operational earlier algorithm must
not silently use modern WRF's mean-wind splitter or old/new mass convention.
The fixture is native Fortran, not a Python rewrite of the CUDA operator.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from conftest import requires_gpu

FIXTURE = Path(__file__).parent / "data/ieva_wrf_legacy.npz"
COMMIT = "40ee6058c2fc6624cbfbbe8cf1c20c59e6a45827"


def test_native_fixture_pins_the_actual_operator_and_declared_correction():
    with np.load(FIXTURE) as data:
        provenance = json.loads(str(data["provenance"]))
        assert provenance["commit"] == COMMIT
        assert set(provenance["sources"]) == {
            "module_advect_em.F", "module_big_step_utilities_em.F", "module_em.F",
            "solve_em.F"}
        assert len(provenance["extracts"]) == 7
        assert "-ffp-contract=off" in provenance["flags"]
        assert all(np.isfinite(data[name]).all() for name in data.files
                   if name.startswith("wrf_") or name == "original_rw_t")
        assert np.count_nonzero(data["wrf_wwI"]) > 0
        assert np.count_nonzero(data["wrf_wwI_m"]) > 0
        assert np.array_equal(data["mut"], data["wrf_mut_new"])
        assert np.any(data["original_rw_t"] != data["wrf_rw_t"])
        nx, ny, nz = map(int, data["meta"][:3])
        coordinate = json.loads(str(data["coordinate"]))
        assert nz == 50 and len(coordinate["eta_levels"]) == 51
        assert coordinate["eta_levels"][1] == 0.998
        assert coordinate["hybrid_opt"] == 2 and coordinate["etac"] == 0.2
        assert coordinate["p_top"] == 5000.0
        omega = data["wrf_wwI"].reshape(nz + 1, ny, nx)
        lower = (omega[0] + omega[1]) * data["rdn"][1] > 0
        corrected = data["wrf_rw_t"].reshape(nz + 1, ny, nx)
        original = data["original_rw_t"].reshape(nz + 1, ny, nx)
        differs = (corrected.view(np.uint32) != original.view(np.uint32)).any(axis=0)
        assert np.array_equal(differs[1:-1, 1:-1], lower[1:-1, 1:-1])


def test_scalar_split_keeps_variant_without_changing_two_flux_contract():
    from gpuwm.core.ieva import ScalarSplit
    split = ScalarSplit("explicit", "implicit", "wrf_legacy")
    explicit, implicit = split
    assert explicit == split[0] == "explicit"
    assert implicit == split[1] == "implicit"
    assert split.variant == "wrf_legacy"


@requires_gpu
def test_legacy_kernels_match_native_fortran_words():
    import cupy as cp
    from test_zadvect_implicit import _kernel_results, _same_bits

    results = _kernel_results(cp, variant="wrf_legacy", fixture=FIXTURE)
    with np.load(FIXTURE) as data:
        for name, (actual, shape, region) in results.items():
            assert np.isfinite(actual).all(), name
            expected = data[f"wrf_{name}"].reshape(shape)
            different, signed_zero = _same_bits(actual[region], expected[region])
            assert different == 0, f"{name}: {different} words differ from native Fortran"


@requires_gpu
def test_legacy_differs_from_modern_split_on_the_same_input():
    import cupy as cp
    from test_zadvect_implicit import _kernel_results

    old = _kernel_results(cp, variant="wrf_legacy", fixture=FIXTURE)
    modern = _kernel_results(cp, variant="wrf_471", fixture=FIXTURE)
    assert np.any(old["wwI"][0].view(np.uint32) != modern["wwI"][0].view(np.uint32))
    assert np.any(old["rth_t"][0].view(np.uint32) != modern["rth_t"][0].view(np.uint32))
