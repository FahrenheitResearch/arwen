"""Does the device MYNN driver reproduce the oracle-pinned CPU driver?

``tests/test_mynn_pbl.py`` pins :func:`gpuwm.core.mynn_pbl.mynn_bl_driver`
against ``gpuwm/data/mynn/oracle/driver.csv``, dumped from the byte-unmodified
``module_bl_mynn.F``.  That makes the CPU driver the reference the device twin
has to reproduce.

Running the two side by side measured something this repository had not
recorded: **the CPU and CUDA leaves are not bitwise twins away from their
oracle fixtures.**  Both halves of every leaf pass their own gate at max_ulp 0
against the same CSV, and on the driver fixture's four columns they disagree
by up to 137 ULP.  The reason is visible in the kernel source and is a
generation gap, not a transcription error: ``mynn_pblh_scale_columns``,
``mynn_mixlength_default_columns``, ``mynn_turbulence_default_interfaces`` and
``mynn_condensation_default_columns`` are written with plain C operators and
CUDA's ``powf``/``tanhf``/``atanf``/``expf``, from before the ``MYNN_ADD``/
``MYNN_MUL`` inline-PTX helpers and the glibc shims were introduced for
``DMP_mf``, ``mym_initialize`` and ``mynn_tendencies``.  Plain operators let
NVRTC contract ``a*b+c`` into an FMA and let CuPy's unconditional
``-ftz=true`` flush subnormals; the oracle inputs never separated the two
roundings, and these do.

``mynn_level2_pairs`` was the fifth member of that list and is no longer:
every operator in it and in ``mynn_mym_level2_column`` is now a
round-to-nearest PTX instruction, and both are bit identical to the oracle.
What that bought here is worth recording precisely, because it is small and it
is not all in one direction.  ``rqiblten`` fell 12 -> 8 and ``tsq`` rose
282 -> 283; the other eighteen profile fields and all five column fields did
not move by one ULP.  A leaf becoming exact does not shrink a residue that the
*other* leaves' contraction dominates -- it just moves where their error lands.

The bisection below is the evidence, one leaf at a time, on identical inputs:

===========================  ==========================================
``mynn_driver_*`` assembly   bitwise (zw, thl, sqw, thetav, qv1, fluxes)
``DMP_mf``                   bitwise on all 18 compared outputs
``GET_PBLH``/``SCALE_AWARE`` 1 ULP in ``zi``, 2 in ``psig_shcu``
``mym_condensation``         5 ULP ``qc_bl``, 32 ``cldfra``, 64 ``vt``
``mym_turbulence``           101 ULP ``el``, 137 ``dfm``/``dfq``
===========================  ==========================================

So the budgets here are not slack.  They are the measured cost of the four
pre-discipline kernels, and the two exactness assertions -- the assembly and
``DMP_mf`` -- are what says this lane's own code is not contributing to it.
"""

from __future__ import annotations

import numpy as np
import pytest

from conftest import requires_gpu

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
def test_device_driver_stays_within_the_measured_leaf_residue(step):
    """Every output, against the CPU driver, at the numbers this repo measures.

    ``rthblten``/``rqvblten`` carry huge ULP counts because they are
    cancellation residues -- ``tests/test_mynn_pbl.py`` records a 1.87e9 ULP
    budget for the same field on the same fixture against WRF itself -- so the
    count is a regression tripwire, not a claim about accuracy.  The integer
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

    for name, budget in _PROFILE_BUDGET.items():
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
        assert _worst(supplied[name][index], want) <= _PROFILE_BUDGET[name]
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
        ("tke_budget", 1), ("spp_pbl", 1),
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
