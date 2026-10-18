"""NSSL option 18 against WRF v4.6.1's own driver, column by column.

Fixtures: ``tests/data/oracles/nssl2_column/nssl2-column-oracle-v2.npz``
(dt 12 s, itimestep 2) and ``nssl2-column-oracle-v2-first-step.npz`` (dt
12 s, itimestep 1), packed by
``tools/nssl2_wrf461_column_oracle/make_fixture.py`` from one oracle run: all
96 columns of its twelve regimes, the microphysics-time inputs both engines
started from, WRF's driver slab at the three stage boundaries (after
sedimentation, after GS, after NUCOND; the first-step fixture carries no
stages) and WRF's final outputs.  The WRF side is the pinned official
``module_mp_nssl_2mom.F`` (commit d66e442f) as WRF's own GNU build
preprocesses it, with the four documented WRF defects corrected
(``wrf_bugfix.py``), compiled with WRF's flags plus ``-fno-tree-vectorize``;
the .json files record the provenance.

These proofs explicitly select ``GPUWM_WRF_EXACT=1`` in a fresh process:
the whole production step reproduces WRF's outputs at 0 ULP. Default runs
select the separate legacy sources and are not this bitwise contract.
The gates name the defects they pin:

* NUCOND was a semantic port: its condensation loop used latent/cp where
  WRF uses e1*f1 = (L/cp/pi0)*pk, a re-derived dtabqvs, a re-associated
  vapour update, a clamped ssfilt and a renucleation gate WRF does not have
  (463 values off WRF from WRF's own post-GS slab);
* radardd02 summed every term in REAL where WRF carries REAL*8 for rain,
  graupel and hail, used CUDA's log10f, and a correctly rounded gamma(0.2)
  that is one ulp off WRF's own GAMMA_SP (776 reflectivity values, 169 snow
  radii);
* the surface reducer multiplied each category by dt*rho before summing,
  where WRF sums xfall first (rainncv/sr off by an ulp);
* on the first step the primary-ice target t7 was diagnosed from vapour
  after the first-step number initialisation had moved sub-threshold
  condensate into it; WRF diagnoses it before (cloud ice 1,422 ulp off);
* the GS contact-freezing diffusiophoretic term used 1000 for rw = 461.5,
  NUCOND lacked the frozen half of WRF's moment tail, and GS clamped WRF's
  negative mass residues instead of handing them to that tail.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from wrf_exact_process import wrf_exact_process

_ROOT = Path(__file__).resolve().parents[1]
_DIR = _ROOT / "tests" / "data" / "oracles" / "nssl2_column"
_FIXTURE = _DIR / "nssl2-column-oracle-v2.npz"
_FIRST_STEP = _DIR / "nssl2-column-oracle-v2-first-step.npz"
_MOMENTS = ("qv", "qc", "qr", "qi", "qs", "qg", "qh", "qndrop", "qnr", "qni",
            "qns", "qng", "qnh", "qnn", "qvolg", "qvolh")
def _fixture(path=_FIXTURE):
    return dict(np.load(path, allow_pickle=False))


@pytest.mark.parametrize("path, itimestep, stages",
                         [(_FIXTURE, 2, True), (_FIRST_STEP, 1, False)])
def test_fixture_is_the_recorded_wrf_column_set(path, itimestep, stages):
    provenance = json.loads(path.with_suffix(".json").read_text(
        encoding="utf-8"))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == \
        provenance["fixture_sha256"]
    assert provenance["schema"] == "woof.nssl2.column-oracle-fixture/v2"
    assert "wrf_bugfix.py" in provenance["wrf_reference_build"]
    assert provenance["stages"] is stages
    data = _fixture(path)
    assert data["columns"].tolist() == provenance["columns"] == list(range(96))
    assert len(set(data["regime"].tolist())) == 12
    assert int(data["itimestep"]) == itimestep
    assert data["in_theta"].shape == (96, 50)
    assert data["in_w"].shape == (96, 51)
    assert ("s1_qv" in data) is stages


def _device_stage(data, stage):
    import cupy as cp

    from gpuwm.core.nssl2_driver_support import NSSL2DriverWorkspace

    ncol, nz = data["in_theta"].shape
    shape = (nz, 1, ncol)

    def dev(a):
        return cp.asarray(np.ascontiguousarray(
            np.asarray(a, np.float32).T[:, None, :]))

    workspace = NSSL2DriverWorkspace(
        cp.empty((16, *shape), dtype=cp.float32),
        cp.zeros((5, 1, ncol), dtype=cp.float32), shape,
        ignored_accumulator=cp.zeros((1, ncol), dtype=cp.float32))
    for index, name in enumerate(_MOMENTS):
        workspace.state[index] = dev(data[f"s{stage}_{name}"])
    environment = {name: dev(data[f"in_{name}"])
                   for name in ("rho", "pressure", "exner", "dz", "qv")}
    environment["w"] = dev(data["in_w"])
    return workspace, dev(data[f"s{stage}_theta"]), environment, shape


def _host(array):
    import cupy as cp
    return np.ascontiguousarray(cp.asnumpy(array)[:, 0, :].T)


def _bits(a):
    return np.asarray(a, np.float32).view(np.uint32)


def _run_gs(data):
    import cupy as cp

    from gpuwm.core.nssl2_fused_gs import launch_fused_gs

    workspace, theta, env, shape = _device_stage(data, 1)
    launch_fused_gs(
        workspace, theta, env["rho"], env["pressure"], env["exner"],
        env["w"], cp.empty(shape, dtype=cp.float32),
        cp.empty(shape, dtype=cp.float32), env["dz"], float(data["dt_s"]),
        hail_on=True, loaded_vapor=env["qv"])
    cp.cuda.Stream.null.synchronize()
    return workspace, theta


@pytest.mark.gpu
@wrf_exact_process
def test_nucond_is_bitwise_wrf():
    """From WRF's own post-GS slab through the production NUCOND to WRF's
    post-NUCOND slab: every moment and theta bit for bit under the strict
    build, selected explicitly for this proof."""
    import cupy as cp

    from gpuwm.core.nssl2_nucond import launch_nucond

    data = _fixture()
    workspace, theta, env, shape = _device_stage(data, 2)
    launch_nucond(
        theta, env["rho"], env["pressure"], env["exner"], env["w"],
        *(workspace.field(n) for n in ("qv", "qc", "qr", "qi", "qs",
                                       "qndrop", "qnr", "qni", "qns", "qnn")),
        float(data["dt_s"]),
        supersaturation_scratch=cp.empty(shape, dtype=cp.float32),
        concentration_space=True, predicted_ccn=True, validate_values=False,
        frozen=tuple(workspace.field(n) for n in
                     ("qg", "qh", "qng", "qnh", "qvolg", "qvolh")))
    cp.cuda.Stream.null.synchronize()
    changed = 0
    for name in _MOMENTS:
        got = _host(workspace.field(name))
        want = data[f"s3_{name}"]
        changed += int(np.count_nonzero(_bits(want) != _bits(data[f"s2_{name}"])))
        np.testing.assert_array_equal(_bits(got), _bits(want), err_msg=name)
    np.testing.assert_array_equal(_bits(_host(theta)),
                                  _bits(data["s3_theta"]), err_msg="theta")
    # The gate is live: WRF's NUCOND really moves these fields here.
    assert changed > 100


@pytest.mark.gpu
@wrf_exact_process
def test_gs_is_bitwise_wrf_under_the_strict_build():
    """From WRF's post-sedimentation slab through the production fused GS
    (t7 from the loaded vapour) to WRF's post-GS slab, bit for bit."""
    data = _fixture()
    workspace, theta = _run_gs(data)
    for name in _MOMENTS:
        np.testing.assert_array_equal(_bits(_host(workspace.field(name))),
                                      _bits(data[f"s2_{name}"]), err_msg=name)
    np.testing.assert_array_equal(_bits(_host(theta)), _bits(data["s2_theta"]))


@pytest.mark.gpu
@wrf_exact_process
def test_contact_freezing_new_ice_matches_wrf():
    """New cloud ice in a supercooled cloudy cell that held none, from WRF's
    post-sedimentation slab through the production fused GS, against WRF's
    post-GS slab.  The diffusiophoretic term read 1000 for rw = 461.5 and
    gave 1.30-1.32 times WRF's crystals here."""
    data = _fixture()
    workspace, _ = _run_gs(data)
    new_woof = _host(workspace.field("qni"))
    fresh = ((data["s1_qni"] == 0.0) & (data["s1_qi"] == 0.0)
             & (data["s2_qni"] > 0.0) & (data["s1_qc"] > 1.0e-6))
    assert np.count_nonzero(fresh) >= 20
    ratio = new_woof[fresh].astype(np.float64) / data["s2_qni"][fresh]
    assert np.max(np.abs(ratio - 1.0)) < 1.0e-3, ratio


@pytest.mark.gpu
@wrf_exact_process
def test_gs_hands_wrf_negative_mass_residues_to_nucond():
    """Where WRF's GS leaves a negative hydrometeor mass (donor-limiter
    cancellation), WOOF's GS leaves one too, for NUCOND's tail to vaporize;
    it used to clamp it to zero and drop the residue from vapour."""
    data = _fixture()
    workspace, _ = _run_gs(data)
    residues = 0
    for name in ("qc", "qr", "qi", "qs", "qg", "qh"):
        negative = data[f"s2_{name}"] < 0.0
        residues += int(np.count_nonzero(negative))
        got = _host(workspace.field(name))
        assert np.all(got[negative] < 0.0), name
    assert residues >= 5


@pytest.mark.gpu
@pytest.mark.parametrize("path", [_FIXTURE, _FIRST_STEP],
                         ids=["itimestep2", "first-step"])
@wrf_exact_process
def test_production_step_matches_wrf(path):
    """The whole production step (``microphysics.apply`` on a DomainState)
    against WRF's final outputs: every output bit for bit under the strict
    build (the first-step fixture pins the t7 ordering)."""
    spec = importlib.util.spec_from_file_location(
        "nssl2_column_oracle_runner",
        _ROOT / "tools" / "nssl2_wrf461_column_oracle" / "run_gpu_columns.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    data = _fixture(path)
    src = {name: data[f"in_{name}"] for name in
           ("theta", "pressure", "w", "rainnc", "snownc", "grplnc", "hailnc",
            *_MOMENTS)}
    src.update(rho=data["col_rho"], dz=data["col_dz"])
    out, _, _, _ = runner.production_step(
        src, float(data["dt_s"]), int(data["itimestep"]))
    mismatched = {}
    for name in runner.OUT3 + runner.OUT2:
        n = int(np.count_nonzero(_bits(out[name]) != _bits(data[f"out_{name}"])))
        if n:
            mismatched[name] = n
    assert not mismatched, mismatched
