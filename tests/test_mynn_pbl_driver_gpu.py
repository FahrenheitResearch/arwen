"""Device driver checks against the pinned CPU and WRF references.

The warm step keeps every historical per-field driver budget. The cold
step keeps them too, except the two wind tendencies, whose budgets are
wider there and also held by absolute caps (``_COLD_VAPOR_ULP``,
``_COLD_VAPOR_ABS``). Ordinary mixing length now reuses the rounded helper
already called by initialization. The other leaves retain their recorded
residuals; these upper bounds are regression limits, not a claim of full
WRF or forecast accuracy.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu
from _toolchain_rows import toolchain_row

import cupy as cp

from gpuwm.core.fp32_ulp import fp32_ulp_distance
from gpuwm.core.mynn_pbl import mynn_bl_driver
from gpuwm.core.mynn_pbl_scratch import MynnPblScratch
from gpuwm.core.mynn_pbl_gpu import (
    _driver_prep_cuda, _driver_surface_cuda, mynn_bl_driver_cuda,
)

from test_mynn_pbl import DRIVER_CASES, _driver_step


#: Measured worst case per field over both fixture steps, inherited from the
#: leaves named in the module docstring.  A tighter number here is a real
#: improvement; a looser one is a regression.
_PROFILE_BUDGET = {
    "rublten": 819, "rvblten": 205, "rthblten": 1258291,
    "rqvblten": 1677721, "rqcblten": 126, "rqiblten": 8,
    "rqsblten": 0, "dozone": 0, "exch_h": 144, "exch_m": 128,
    "qke": 10, "tsq": 283, "qsq": 108, "cov": 208, "el": 101,
    "sh": 48, "sm": 61, "qc_bl": 5, "qi_bl": 2, "cldfra_bl": 32,
}
_COLUMN_BUDGET = {
    "pblh": 1, "rmol": 0, "maxwidth": 0, "maxmf": 2, "ztop_plume": 0,
}

# COLD-STEP OVERRIDES FOR THE TWO WIND TENDENCIES, AND WHY THEY ARE WIDER.
# Both drivers now hand vapor (sqv) to initialization on the cold call, as
# WRF does; both used to hand it total water (sqv+sqc+sqi). The earlier cold
# comparison was CPU against CUDA on that same input, so it was consistent
# and its 819 and 205 ULP budgets (the ordinary row above) were right for
# it: this is NOT a case of an old comparison made against a wrong
# reference. What changed is the input. With vapor, the cloudy cold columns
# start from a different initialized state, and on that state the CUDA
# driver's residue against the CPU driver in these two cancellation-
# sensitive rates is larger: 3276 and 1638 ULP over the original
# first-four-column population, measured on sm_89 / NVRTC 13.4, with the
# same limits passing on RTX PRO 6000 (sm_120), CuPy 14.2.0 / NVRTC 12.9.
# They are a new measured residue on a new cold-cloud population, declared
# as such. Every other bound and the entire warm row stay unchanged. The
# absolute bounds keep these rate comparisons from admitting a larger
# physical error behind a ULP cap.
_COLD_VAPOR_ULP = {"rublten": 3276, "rvblten": 1638}
_COLD_VAPOR_ABS = {
    "rublten": 7.147900760173798e-8,
    "rvblten": 4.452886059880257e-9,
}

#: ``_PROFILE_BUDGET`` per compiler where a compiler reads it differently,
#: keyed on (compute capability, NVRTC major.minor), the pair measured.
#: A146: NVRTC had compiled every float division by a compile-time constant
#: as a multiply by the rounded reciprocal on Blackwell, and the kernels now
#: spell those divisions ``__fdiv_rn``, the IEEE quotient.  On sm_120 that
#: moves rvblten against the CPU driver from 205 to 819 ULP (rublten's
#: budget); every other field reads at or below its budget, most far below
#: (exch_h 5, tsq 10, cov 7 against 144, 283, 208).  MEASURED 2026-09-30 on
#: the RTX 5070 Ti (sm_120, NVRTC 13.4.92) over both fixture steps.
_PROFILE_BUDGET_BY_TOOLCHAIN = {
    ("120", (13, 4)): {**_PROFILE_BUDGET, "rvblten": 819},
}
#: A167: NVRTC 12.9.86, the compiler of the default gpuwm[gpu] extra
#: (cupy-cuda12x), reads the sm_120 row A146 re-recorded under 13.4: every
#: reading this file's tests take, and the device result behind each, is
#: bit-identical under the two compilers.  Before this row 12.9.86 failed
#: here by name (tests/_toolchain_rows.py).  MEASURED 2026-10-01 on node-4's
#: RTX 5070 Ti and node-2's RTX 5090, two processes per compiler, at
#: integrate/2.8 9dbb4a2db.
_PROFILE_BUDGET_BY_TOOLCHAIN[("120", (12, 9))] = (
    _PROFILE_BUDGET_BY_TOOLCHAIN[("120", (13, 4))])


def _worst(device, host) -> int:
    return int(fp32_ulp_distance(
        cp.asnumpy(device).astype(np.float32),
        np.asarray(host, dtype=np.float32),
    ).max())


def _device(values):
    return {name: cp.asarray(np.ascontiguousarray(np.asarray(value)))
            for name, value in values.items()}


@requires_gpu
@pytest.mark.parametrize("step", (1, 2))
@pytest.mark.parametrize("contraction", ("production", "disabled"))
def test_ordinary_mixing_length_uses_the_rounded_column_contract(
        monkeypatch, step, contraction):
    """The ordinary driver must use the same length law as initialization.

    Capture the actual coupled inputs, including the cloud and plume fields,
    then compare the leaf with the independent CPU transcription. The old
    duplicated device body differs by 101 ULP on these inputs.
    """
    from gpuwm.core import mynn_pbl, mynn_pbl_gpu

    original = mynn_pbl_gpu.mynn_mixlength_default_cuda
    seen = []
    if contraction == "disabled":
        from test_mynn_pbl_gpu import _mynn_module

        module = _mynn_module("-fmad=false")
        get_kernel = mynn_pbl_gpu.get_kernel

        def length_kernel(name, function):
            if name == "mynn_pbl" and function == "mynn_mixlength_default_columns":
                return module.get_function(function)
            return get_kernel(name, function)

        monkeypatch.setattr(mynn_pbl_gpu, "get_kernel", length_kernel)

    def capture(values, **kwargs):
        inputs = {name: cp.asnumpy(value).copy()
                  for name, value in values.items()}
        result = original(values, **kwargs)
        actual = {name: cp.asnumpy(getattr(result, name)).copy()
                  for name in ("el", "qkw")}
        seen.append((inputs, actual))
        return result

    monkeypatch.setattr(mynn_pbl_gpu, "mynn_mixlength_default_cuda", capture)
    _, values, initflag, delt = _driver_step(step)
    mynn_pbl_gpu.mynn_bl_driver_cuda(
        _device(values), initflag=initflag, delt=delt, flag_qs=True)
    assert len(seen) == 1
    inputs, actual = seen[0]
    reference = mynn_pbl.mynn_mixlength_default(inputs)
    for name in ("el", "qkw"):
        np.testing.assert_array_equal(actual[name].view(np.uint32),
                                      reference[name].view(np.uint32))
        assert np.isfinite(actual[name]).all()
    assert (actual["el"][:, 0] == 0).all()
    assert (actual["el"][:, 1:] > 0).all()
    assert (actual["qkw"] > 0).all()


@requires_gpu
@pytest.mark.parametrize("step", (1, 2))
def test_device_driver_stays_within_the_measured_leaf_residue(step):
    """Every output, against the CPU driver, at the numbers this repo measures.

    ``rthblten``/``rqvblten`` carry huge ULP counts because they are
    cancellation residues from the assembled CUDA leaves. The CPU driver
    now agrees bitwise with WRF on this option, including cold clouds. The
    count is a regression tripwire, not a claim about accuracy. The integer
    indices have no budget at all: a PBL top or plume top that moves a level
    is a structural disagreement, not rounding.
    """

    _, values, initflag, delt = _driver_step(step)
    # Preserve the historical four-column ULP ratchet unchanged.  The new
    # snow-only column has its own WRF-facing gate below; folding a new
    # population into these measured maxima would redefine them.
    values = {name: np.asarray(value)[:4].copy()
              for name, value in values.items()}
    host = mynn_bl_driver(
        values, initflag=initflag, delt=delt, flag_qs=True,
    )
    device = mynn_bl_driver_cuda(
        _device(values), initflag=initflag, delt=delt, flag_qs=True)

    profile = toolchain_row(_PROFILE_BUDGET_BY_TOOLCHAIN, _PROFILE_BUDGET,
                            "_PROFILE_BUDGET_BY_TOOLCHAIN")
    if step == 1:
        profile = {**profile, **_COLD_VAPOR_ULP}
        for name, maximum in _COLD_VAPOR_ABS.items():
            error = np.abs(cp.asnumpy(device[name]).astype(np.float64)
                           - np.asarray(host[name], dtype=np.float64))
            assert float(error.max()) <= maximum, (name, float(error.max()))
    for name, budget in profile.items():
        worst = _worst(device[name], host[name])
        assert worst <= budget, f"{name}: {worst} ULP (budget {budget})"
    for name, budget in _COLUMN_BUDGET.items():
        worst = _worst(device[name], np.asarray(host[name]).reshape(-1))
        assert worst <= budget, f"{name}: {worst} ULP (budget {budget})"
    for name in ("kpbl", "ktop_plume"):
        np.testing.assert_array_equal(
            cp.asnumpy(device[name]).astype(np.int32).reshape(-1),
            np.asarray(host[name], dtype=np.int32).reshape(-1),
            err_msg=name,
        )


@requires_gpu
def test_device_driver_supplies_snow_within_the_existing_wrf_leaf_budgets():
    """The production driver reads sqs and retains the existing WRF budgets."""

    blocks, values, initflag, delt = _driver_step(2)
    supplied = mynn_bl_driver_cuda(
        _device(values), initflag=initflag, delt=delt, flag_qs=True)
    withheld = mynn_bl_driver_cuda(
        _device(values), initflag=initflag, delt=delt, flag_qs=False)
    index = DRIVER_CASES.index("snow_anvil")
    changed = 0
    for name in ("qc_bl", "qi_bl", "cldfra_bl"):
        want = np.asarray(
            [np.float32(row[name]) for row in blocks[index]], dtype=np.float32)
        assert _worst(supplied[name][index], want) <= (
            toolchain_row(_PROFILE_BUDGET_BY_TOOLCHAIN, _PROFILE_BUDGET,
                          "_PROFILE_BUDGET_BY_TOOLCHAIN")[name])
        got = cp.asnumpy(supplied[name][index])
        without = cp.asnumpy(withheld[name][index])
        changed += int(np.count_nonzero(got != without))
    assert changed > 0


@requires_gpu
def test_the_driver_assembly_itself_is_bitwise():
    """This lane's own kernels contribute nothing to the residue above.

    ``zw``, ``thl``, ``sqw``, ``thetav``, ``qv1`` and the whole surface-flux
    block are what ``mynn_bl_driver_cuda`` adds between the leaf calls.  They
    are required to be bit identical to the CPU driver's own assembly, which
    is what makes the budgets in the previous test attributable to the leaves
    rather than to the assembly.  Written as CuPy array expressions instead of
    kernels, ``thetav`` alone moved ``PBLH`` and the whole condensation chain.
    """

    _, values, _, _ = _driver_step(2)
    ncol, nz = np.asarray(values["dz"]).shape
    layers = _device({name: values[name] for name in (
        "dz", "u", "v", "w", "th", "sqv", "sqc", "sqi", "p", "exner",
        "rho", "tk")})
    scalars = _device({name: np.broadcast_to(
        np.asarray(values[name], dtype=np.float32), (ncol,)).copy()
        for name in ("dx", "xland", "ts", "ps", "ust", "hfx", "qfx",
                     "wspd", "uoce", "voce")})
    # The assembly kernels now write into the declared workspace instead of
    # allocating; a standalone holder is what a caller with no DomainState
    # gets, and it changes nothing about what the kernels compute.
    work = MynnPblScratch.standalone(ncol, nz)
    zw, prep = _driver_prep_cuda(layers, scalars["ust"], ncol, nz, work)
    surface = _driver_surface_cuda(layers, prep["qv1"], scalars, ncol, nz,
                                   work)

    from gpuwm.core.mynn_pbl import (
        CP, GTR, KARMAN, P608, XLVCP, XLSCP, F, _driver_zw,
    )
    host_dz = np.asarray(values["dz"], dtype=np.float32)
    assert _worst(zw, _driver_zw(host_dz, nz)) == 0
    for column in range(ncol):
        for k in range(nz):
            exner = F(values["exner"][column, k])
            sqc = F(values["sqc"][column, k])
            sqi = F(values["sqi"][column, k])
            sqv = F(values["sqv"][column, k])
            th = F(values["th"][column, k])
            assert cp.asnumpy(prep["qv1"])[column, k] \
                == F(sqv / F(F(1.0) - sqv))
            assert cp.asnumpy(prep["sqw"])[column, k] == F(F(sqv + sqc) + sqi)
            assert cp.asnumpy(prep["thl"])[column, k] == F(
                F(th - F(F(XLVCP / exner) * sqc)) - F(F(XLSCP / exner) * sqi))
            assert cp.asnumpy(prep["thetav"])[column, k] == F(
                th * F(F(1.0) + F(P608 * sqv)))
    for column in range(ncol):
        rho0 = F(values["rho"][column, 0])
        exner0 = F(values["exner"][column, 0])
        ust = F(np.broadcast_to(np.asarray(values["ust"]), (ncol,))[column])
        qv1 = F(cp.asnumpy(prep["qv1"])[column, 0])
        cpm = F(CP * F(F(1.0) + F(F(0.84) * qv1)))
        flqv = F(F(np.broadcast_to(np.asarray(values["qfx"]),
                                   (ncol,))[column]) / rho0)
        th_sfc = F(F(np.broadcast_to(np.asarray(values["ts"]),
                                     (ncol,))[column]) / exner0)
        flt = F(F(F(np.broadcast_to(np.asarray(values["hfx"]),
                                    (ncol,))[column]) / F(rho0 * cpm))
                - F(F(XLVCP * F(0.0)) / exner0))
        fltv = F(flt + F(F(flqv * P608) * th_sfc))
        ust3 = F(F(ust * ust) * ust)
        rmol = F(-F(F(F(KARMAN * GTR) * fltv) / max(ust3, F(1.0e-6))))
        assert cp.asnumpy(surface["flqv"])[column] == flqv
        assert cp.asnumpy(surface["th_sfc"])[column] == th_sfc
        assert cp.asnumpy(surface["flt"])[column] == flt
        assert cp.asnumpy(surface["fltv"])[column] == fltv
        assert cp.asnumpy(surface["rmol"])[column] == rmol
        assert cp.asnumpy(surface["flqc"])[column] == F(0.0)

    # :1095-1096.  pmz/phh moved into this kernel when the glibc libm block
    # landed in mynn_pbl.cu; before that they came back to the host at 132 us
    # per column.  Equality here is the whole point of the move -- these two
    # scalars are the surface boundary condition mym_predict integrates, so a
    # single ULP of drift propagates into every profile the driver returns.
    from gpuwm.core.mynn_pbl import mynn_phih, mynn_phim
    host_zet = cp.asnumpy(surface["zet"])
    for column in range(ncol):
        zet = F(host_zet[column])
        assert cp.asnumpy(surface["pmz"])[column] == F(mynn_phim(zet) - zet)
        assert cp.asnumpy(surface["phh"])[column] == mynn_phih(zet)


@requires_gpu
def test_device_driver_state_actually_advances():
    """A negative control: the device driver must not be returning its input.

    A wrapper that forwarded the incoming state would satisfy every budget
    above if the CPU reference were broken the same way.  On the warm step the
    fixture's four columns are all turbulent, so ``qke`` must move on each.
    """

    _, values, initflag, delt = _driver_step(2)
    device = mynn_bl_driver_cuda(
        _device(values), initflag=initflag, delt=delt, flag_qs=True)
    before = np.asarray(values["qke"], dtype=np.float32)
    after = cp.asnumpy(device["qke"])
    assert after.shape == before.shape
    for index, case in enumerate(DRIVER_CASES):
        assert not np.array_equal(after[index], before[index]), case
    assert np.isfinite(after).all()


@requires_gpu
def test_device_driver_refuses_a_nondefault_identity():
    """The device twin must fail closed on the same knobs as the reference."""

    _, values, _, delt = _driver_step(2)
    device_values = _device(values)
    for knob, bad in (
        ("bl_mynn_edmf", 0), ("bl_mynn_output", 1), ("icloud_bl", 0),
        ("tke_budget", 1), ("spp_pbl", 2),
    ):
        with pytest.raises(ValueError, match=knob):
            mynn_bl_driver_cuda(
                device_values, initflag=0, delt=delt, **{knob: bad})
    with pytest.raises(ValueError, match="restart"):
        mynn_bl_driver_cuda(device_values, initflag=0, delt=delt, restart=True)
    mynn_bl_driver_cuda(device_values, initflag=0, delt=delt, flag_qs=True)
    with pytest.raises(TypeError, match="initflag"):
        mynn_bl_driver_cuda(device_values, initflag=0.0, delt=delt)
    missing = dict(device_values)
    del missing["cldfra_bl"]
    with pytest.raises(TypeError, match="cldfra_bl"):
        mynn_bl_driver_cuda(missing, initflag=0, delt=delt)
