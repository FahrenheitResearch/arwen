"""RUC LSM on the GPU against WRF v4.6.1, word for word.

``tools/ruc_lsm_gpu_oracle`` drives the production entry point
(:func:`gpuwm.core.ruc_runtime.ruc_lsm_step`, the fused CUDA kernels) and an
unmodified WRF v4.6.1 ``LSMRUC`` + ``SFCDIAGS_RUCLSM`` inside a transcription
of the surface driver's RUC arm from the same ``inputs.bin``.  The fixtures
here are what that WRF build wrote (gfortran 13.3.0, glibc 2.39, WRF's GNU
flags plus ``-finit-local-zero``; ``pack_fixture.py``), so this test reruns
only the WOOF side and grades every compared word after every step:
RUCLSMINIT's four outputs, every LSMRUC state and seam field, the soil
profiles and T2/TH2/Q2.

It fails on either fix this lane made being undone:

* RUC's transcendentals back on float64-rounded-once or CUDA libm words
  (hundreds of words differ by the fourth step: grdflx, hfx, qfx, the
  saturation humidities, rhosnf/snowfallac through the new-snow TANH);
* SFCEVP double counted again, as WRF does (:1095 and :1116).  SFCEVP is
  graded against WRF's entry value plus ``qfx*dt`` once, carried over the
  steps by the Fortran driver.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from conftest import requires_gpu

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools" / "ruc_lsm_gpu_oracle"
FIXTURES = ROOT / "tests" / "data" / "oracles" / "ruc_lsm"
CASES = ("wrf461-nzs9.npz", "wrf461-nzs6.npz", "wrf461-mosaic.npz")


def _tool(name):
    if str(TOOL) not in sys.path:
        sys.path.insert(0, str(TOOL))
    return importlib.import_module(name)


def _fixture(name):
    data = np.load(FIXTURES / name)
    return data, json.loads(bytes(data["receipt"]).decode())


@pytest.mark.parametrize("name", CASES)
def test_fixture_covers_the_regimes_it_claims(name):
    """CPU: the stored case still spans the regimes the report names."""
    data, receipt = _fixture(name)
    regimes = set(receipt["regime"])
    for needed in ("convective", "stable_night", "snow_thick", "snow_thin",
                   "snow_trace", "snow_melt", "snowfall_bare", "frozen_soil",
                   "land_ice", "sea_ice_full", "sea_ice_fractional", "ocean",
                   "high_terrain", "desert", "wet_drainage"):
        assert needed in regimes, needed
    assert receipt["options"]["ncol"] >= 64
    # The single-count SFCEVP really differs from WRF's doubled word on land.
    k = receipt["options"]["nsteps"]
    assert np.any(data[f"step{k}__sfcevp"] != data[f"step{k}__sfcevp_once_acc"])


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("name", CASES)
def test_gpu_step_matches_wrf461_word_for_word(tmp_path, name):
    data, receipt = _fixture(name)
    columns = _tool("columns")
    run_woof = _tool("run_woof")
    layout = _tool("layout")
    o = receipt["options"]
    case = columns.build_case(
        nzs=o["nzs"], nsteps=o["nsteps"], dt=o["dt"],
        fractional_seaice=o["fractional_seaice"], mosaic=bool(o["mosaic"]),
        lakemodel=o["lakemodel"], rdlai2d=bool(o["rdlai2d"]))
    run_woof.run(case, tmp_path)
    digest = hashlib.sha256((tmp_path / "inputs.bin").read_bytes()).hexdigest()
    assert digest == receipt["inputs_sha256"], (
        "WOOF's starting state is not the one the WRF fixture ran from")
    woof = np.load(tmp_path / "woof.npz")
    differing = {}
    for field in layout.INIT_2D + layout.INIT_PROFILES:
        got = woof[f"init__{field}"].view(np.uint32)
        want = data[f"init__{field}"].view(np.uint32)
        if not np.array_equal(got, want):
            differing[("init", field)] = int(np.count_nonzero(got != want))
    from columns import PROFILES
    for k in range(1, o["nsteps"] + 1):
        for field in layout.OUT_2D + PROFILES:
            want = data[f"step{k}__" + (layout.SFCEVP_ONCE_ACC
                                        if field == "sfcevp" else field)]
            got = woof[f"step{k}__{field}"]
            bad = got.view(np.uint32) != want.view(np.uint32)
            if np.any(bad):
                differing[(k, field)] = int(np.count_nonzero(bad))
    assert not differing, f"words differing from WRF v4.6.1: {differing}"


@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("backend", ["fused", "reference", "packed"])
def test_binary_ice_mode_full_field_hashes(tmp_path, monkeypatch, backend):
    """Fractional XICE with fractional_seaice=0 keeps WRF's binary seam."""
    import cupy as cp
    from gpuwm.core import ruc_runtime
    from gpuwm.core.surface_forcing import SurfacePrecipitationForcing
    receipt = json.loads((FIXTURES / "binaryice-hashes.json").read_text())
    o = receipt["options"]
    case = _tool("columns").build_case(nzs=o["nzs"], nsteps=o["nsteps"], dt=o["dt"],
        fractional_seaice=0, round_xice=False)
    bank = None
    if backend == "reference":
        monkeypatch.setattr(ruc_runtime, "ruc_lsm_step", ruc_runtime._ruc_lsm_step_reference)
    elif backend == "packed":
        from gpuwm.ensemble.batch_ruc import PackedRucDriver, workspace_allocations
        from gpuwm.core.ruc_memory import allocation_bytes
        bank = PackedRucDriver(2, (1, o["ncol"]), nzs=o["nzs"],
            available_bytes=allocation_bytes(workspace_allocations(2, (1, o["ncol"]), o["nzs"])))

        def packed(fields, atmosphere, **kw):
            stack = lambda values: {n: cp.stack([a, a]) if isinstance(a, cp.ndarray) else a for n, a in values.items()}
            slab, atmo = stack(fields), stack(atmosphere)
            census = bank.step(slab, atmo, params=kw["params"],
                precipitation=SurfacePrecipitationForcing.from_fields(slab),
                dt=[kw["dt"]] * 2, itimestep=[kw["itimestep"]] * 2,
                soilprop=kw["ruc_soilprop"], irrigation=kw["ruc_irrigation"], snow=kw["ruc_snow"])
            for n, a in fields.items():
                if isinstance(a, cp.ndarray):
                    cp.testing.assert_array_equal(slab[n][0].view("u1"), slab[n][1].view("u1"))
                    a[...] = slab[n][0]
            return census[0]

        monkeypatch.setattr(ruc_runtime, "ruc_lsm_step", packed)
    try:
        _tool("run_woof").run(case, tmp_path)
        with np.load(tmp_path / "woof.npz") as output:
            for field, want in receipt["sha256"].items():
                got = hashlib.sha256(output[field].astype("<f4").tobytes()).hexdigest()
                assert got == want, f"{backend}: {field} differs from the WRF full-field hash"
    finally:
        if bank is not None:
            bank.close()
