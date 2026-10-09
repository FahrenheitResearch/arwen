"""km_opt = 1 against compiled WRF v4.6.1: every output word of the package.

The fixtures (tests/data/wrf461_km1) hold twelve 12 x 10 patches of real
columns, all 49 mass levels, cropped from a real.exe CONUS 12 km analysis
(2019-11-26 12Z): high terrain, Pacific, Gulf and Atlantic ocean, a stable
night over the plains, snow cover, a moist convective coast, a mixed coast,
and steep, map-factor-extreme, zero-flow and near-zero-flow edge variants,
open and periodic.  Five arms each: PBL off with K = 75 and prescribed drag
and heat (isfflx 0), PBL off mesoscale K (500, 2) with surface-layer fluxes
(isfflx 1), PBL on (horizontal only, khdif 300, kvdif 1), surface flux only
(K = 0, isfflx 2), and odd namelist words (khdif = 1/3).  The references
are the unmodified WRF v4.6.1 bodies (tools/wrf_diffusion_oracle/
km1_build.py, km1_oracle.py capture): calculate_km_kh -> isotropic_km,
vertical_diffusion_2 and horizontal_diffusion_2 in module_first_rk_step_
part2.F order.  The engine side is the production
``dycore._compute_wrf_smag_tendencies``.

Under the strict build with its diffusion stage (GPUWM_WRF_EXACT=1 and
GPUWM_WRF_EXACT_DIFFUSION=1) every one of the 5,018,280 words is the WRF
word: no tolerance.  Default and GPUWM_WRF_EXACT=1-only arithmetic are
measured, not gated (receipts in the same folder).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
import pytest
from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests/data/wrf461_km1"
TOOLS = ROOT / "tools/wrf_diffusion_oracle"
sys.path.insert(0, str(TOOLS))


def _manifest():
    return json.loads((DATA / "km1-cases.json").read_text())


def _arm(name):
    import km1_oracle
    return next(a for a in km1_oracle.ARMS if a[0] == name)


def _case(entry):
    with np.load(DATA / entry["file"]) as data:
        arrays = {k[3:]: data[k] for k in data.files if k.startswith("in_")}
        meta = json.loads(str(data["meta_json"]))
    with np.load(DATA / entry["reference_file"]) as ref:
        reference = {k: ref[k] for k in ref.files}
    return arrays, meta, reference


def test_km1_fixtures_are_the_compiled_wrf_461_package():
    manifest = _manifest()
    build = manifest["fortran_build"]
    assert build["wrf_release"] == "4.6.1"
    assert build["wrf_commit"] == "d66e442fccc04111067e29274c9f9eaccc3cef28"
    assert build["source_sha256"]["dyn_em/module_diffusion_em.F"] == (
        "a7d4570c97e51c635e86a0dbd628c6846457ac5b93d5a7af798b118c7d8d2d54")
    assert "-ffp-contract=off" in build["flags"] and "-fcheck=bounds" in build["flags"]
    for name in ("isotropic_km", "calculate_km_kh", "horizontal_diffusion_2",
                 "vertical_diffusion_2", "vertical_diffusion_w_2"):
        assert name in build["routines"], name
    for name, digest in build["wrapper_sources_sha256"].items():
        assert hashlib.sha256((TOOLS / name).read_bytes()).hexdigest() == digest, name
    assert len(manifest["cases"]) == 12
    assert sum(c["meta"]["nx"] * c["meta"]["ny"] for c in manifest["cases"]) == 1440
    assert {c["meta"]["bx"] for c in manifest["cases"]} == {0, 1}
    for case in manifest["cases"]:
        for key in ("file", "reference_file"):
            digest = case["sha256" if key == "file" else "reference_sha256"]
            assert hashlib.sha256((DATA / case[key]).read_bytes()).hexdigest() == digest, case[key]
    import km1_oracle
    assert [list(a) for a in km1_oracle.ARMS] == manifest["arms"]


def test_isotropic_constants_are_the_wrf_coefficient_words():
    """isotropic_km: xkmh=khdif, xkmv=kvdif, xkh*=k/prandtl in REAL(4)."""
    from gpuwm.core.dycore import isotropic_km_constants
    import km1_oracle
    manifest = _manifest()
    _arrays, _meta, reference = _case(manifest["cases"][0])
    for arm in km1_oracle.ARMS:
        got = isotropic_km_constants(arm[1], arm[2])
        for word, key in zip(got, ("kmh", "kmv", "khh", "khv")):
            wrf = np.unique(reference[f"{arm[0]}__coef_{key}"])
            assert wrf.size == 1 and wrf[0].view(np.uint32) == np.float32(word).view(np.uint32), (arm[0], key)
    # WRF's division is not 3*k for every word.
    assert isotropic_km_constants(75.0, 75.0)[2] == np.float32(225.0)


def test_constant_k_package_runs_exactly_where_wrf_writes_words():
    from gpuwm.config import (RunConfig, constant_k_mixing_active,
                              wrf_mixing_package_active)
    base = dict(nx=8, ny=8, nz=8, dx=100.0, dy=100.0, ztop=1600.0, dt=1.0,
                run_seconds=0.0)
    on = constant_k_mixing_active
    assert not on(RunConfig(**base))                         # WRF default
    assert on(RunConfig(**base, khdif=1.0))
    assert on(RunConfig(**base, kvdif=1.0, bl_pbl_physics=1))
    # K = 0, PBL off: vertical_diffusion_2 still writes the surface arms.
    assert on(RunConfig(**base, isfflx=0, tke_heat_flux=0.2))
    assert on(RunConfig(**base, isfflx=0, tke_drag_coefficient=1e-3))
    assert not on(RunConfig(**base, isfflx=0))
    assert on(RunConfig(**base, isfflx=1, sf_sfclay_physics=1))
    assert not on(RunConfig(**base, isfflx=1, sf_sfclay_physics=1,
                            bl_pbl_physics=1))
    assert not on(RunConfig(**base, km_opt=4, khdif=1.0))
    assert wrf_mixing_package_active(RunConfig(**base, km_opt=4))
    assert wrf_mixing_package_active(RunConfig(**base, khdif=1.0))


def test_prandtl_is_wrf_fixed_constant_and_no_configuration_key(tmp_path):
    """WRF fixes isotropic_km's prandtl at 1./3.0 (module_model_constants.F)
    with no namelist knob, so the engine does too: the dycore reads it off
    RunConfig as a class constant, which is no field, no TOML key and no
    digest or restart-identity entry.  Before it existed a km_opt = 1 run
    died at step 1 with AttributeError (the shipped constant-K profile)."""
    from dataclasses import fields
    from gpuwm.config import RunConfig, load_config
    from gpuwm.core.dycore import _PRANDTL, isotropic_km_constants
    cfg = RunConfig(nx=8, ny=8, nz=8, dx=100.0, dy=100.0, ztop=1600.0,
                    dt=1.0, run_seconds=0.0, khdif=75.0, kvdif=75.0)
    assert cfg.constant_k_prandtl == _PRANDTL == 1.0 / 3.0
    assert "constant_k_prandtl" not in {f.name for f in fields(RunConfig)}
    assert (isotropic_km_constants(cfg.khdif, cfg.kvdif, cfg.constant_k_prandtl)
            == isotropic_km_constants(cfg.khdif, cfg.kvdif))
    with pytest.raises(TypeError):
        RunConfig(nx=8, ny=8, nz=8, dx=100.0, dy=100.0, ztop=1600.0, dt=1.0,
                  run_seconds=0.0, constant_k_prandtl=1.0)
    toml = tmp_path / "prandtl.toml"
    toml.write_text("[grid]\nnx = 8\nny = 8\nnz = 8\ndx = 100.0\ndy = 100.0\n"
                    "ztop = 1600.0\n[run]\ndt = 1.0\nrun_seconds = 0.0\n"
                    "[dynamics]\nkm_opt = 1\nconstant_k_prandtl = 1.0\n")
    with pytest.raises(ValueError, match="constant_k_prandtl"):
        load_config(toml)


def test_preflight_prices_the_constant_k_package():
    """km_opt = 1 with constant K runs WRF's mixing package (3641a45f7), so
    preflight prices its K pair, vertical pair, carrying tendencies and face
    workspaces; the retired Laplacian's diff_* temporaries are not priced."""
    import gpuwm.core.preflight as pf
    from gpuwm.config import RunConfig
    base = dict(nx=8, ny=8, nz=8, dx=100.0, dy=100.0, ztop=1600.0, dt=1.0,
                run_seconds=0.0)
    slots = set(pf.scratch_slot_registry(RunConfig(**base, khdif=10.0, kvdif=1.0),
                                          n_lbc_intervals=2))
    assert {"smag_km", "smag_kh", "smag_kmv", "smag_khv", "smag_ru", "smag_rv",
            "smag_rw", "smag_rth", "diff6_x", "diff6_y"} <= slots
    assert not {"diff_u", "diff_v", "diff_w", "diff_th"} & slots
    idle = set(pf.scratch_slot_registry(RunConfig(**base), n_lbc_intervals=2))
    assert not {"smag_km", "smag_kmv", "smag_ru"} & idle


def test_guards_of_the_retired_constant_laplacian_are_gone():
    """The refusals that existed only for the periodic, flat, dry
    Laplacian retire with it (3641a45f7): constant K is admitted on open
    boundaries and beside chem."""
    from gpuwm.config import RunConfig, validate_chem_config, validate_run_config
    base = dict(nx=16, ny=16, nz=16, dx=100.0, dy=100.0, ztop=1600.0,
                dt=0.5, run_seconds=0.0, khdif=10.0, kvdif=1.0)
    validate_run_config(RunConfig(**base, open_x=True, open_y=True))
    validate_chem_config(RunConfig(**base, chem_sets="tracer_test"))
    from gpuwm.core import diffusion
    assert not hasattr(diffusion, "add_diffusion_tendencies")


def test_measured_receipts_record_every_arithmetic():
    for name, strict, diffusion in (("km1-compare-strict-diffusion.json", True, True),
                                    ("km1-compare-strict.json", True, False),
                                    ("km1-compare-default.json", False, False)):
        receipt = json.loads((DATA / name).read_text())
        assert (receipt["strict"], receipt["exact_diffusion_substage"]) == (strict, diffusion)
        assert receipt["columns"] == 1440 and receipt["case_arms"] == 60
        assert receipt["total_words"] == 5_018_280
    exact = json.loads((DATA / "km1-compare-strict-diffusion.json").read_text())
    assert exact["total_different_words"] == 0 and exact["max_ulp"] == 0


def _exact_diffusion_selected():
    return (os.environ.get("GPUWM_WRF_EXACT") == "1"
            and os.environ.get("GPUWM_WRF_EXACT_DIFFUSION") == "1")


@pytest.mark.gpu
@requires_gpu
@pytest.mark.skipif(not _exact_diffusion_selected(),
                    reason="the 0-ULP claim is for the strict build with its diffusion stage "
                           "(GPUWM_WRF_EXACT=1 GPUWM_WRF_EXACT_DIFFUSION=1)")
@pytest.mark.parametrize("arm", ["pbl_off_k75", "pbl_off_meso", "pbl_on_horizontal",
                                 "pbl_off_surface_only", "pbl_off_odd_words"])
def test_km1_package_is_every_compiled_wrf_461_word(arm):
    import km1_oracle
    words = 0
    for entry in _manifest()["cases"]:
        arrays, meta, reference = _case(entry)
        got = km1_oracle.woof_outputs(arrays, meta, _arm(arm))
        for key in km1_oracle.OUTPUTS + ("coef_kmh", "coef_kmv", "coef_khh", "coef_khv"):
            want = reference[f"{arm}__{key}"]
            assert got[key].shape == want.shape, (entry["name"], key)
            assert np.array_equal(got[key].view(np.uint32), want.view(np.uint32)), (entry["name"], key)
            words += want.size
    assert words == 5_018_280 // 5


@pytest.mark.gpu
@requires_gpu
def test_open_boundary_ring_keeps_the_surface_fluxes():
    """The defect the oracle found: the mixing package cleared the outer
    ring after the vertical pass, so on an open-boundary PBL-off run the
    lowest-level ring lost its surface momentum, heat and moisture flux,
    which WRF writes there and advances.  Any arithmetic: the ring words
    were exactly zero before the fix and are WRF's nonzero words now."""
    import km1_oracle
    entry = next(c for c in _manifest()["cases"] if c["name"] == "high_terrain")
    arrays, meta, reference = _case(entry)
    assert meta["bx"] and meta["by"]
    arm = _arm("pbl_off_surface_only")
    got = km1_oracle.woof_outputs(arrays, meta, arm)
    for key in ("u", "v", "theta", "qv"):
        want = reference[f"{arm[0]}__{key}"][0]
        ring = ~km1_oracle.interior_mask(want[None].shape, key, meta)[0]
        assert (want[ring] != 0).all(), key
        assert (got[key][0][ring] != 0).all(), key
        if _exact_diffusion_selected():
            assert np.array_equal(got[key][0][ring].view(np.uint32), want[ring].view(np.uint32)), key
