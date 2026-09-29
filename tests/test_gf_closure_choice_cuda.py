"""Single-member Grell-Freitas closures on the device, against the reference.

``clos_choice`` (WRF's ``ichoice``) 1..16 makes ``cup_forcing_ens_3d`` copy
closure member ``ichoice`` into all sixteen slots of ``xf_ens``, so the
ensemble mean the scheme then takes IS that member.  0 averages all
sixteen, and 0 is the arm compared bitwise against WRF v4.6.1
(tests/test_gf_gfdrv_cuda.py, tests/test_gf_driver_parity.py).  No WRF run
of a single-member closure exists, so what this file proves is the next
best thing that can be proved without one: the shipped kernel and the
float32 reference (gpuwm.verify.gf_driver.gfdrv_column, the same
reference that is bitwise with WRF at 0) agree word for word on every
member, over all 216 oracle columns, through the whole driver.

Both sides run with the oracle's own fzu pinned and the WRF-faithful k22
flag, exactly as the ensemble gate does, so any difference here is the
closure path and nothing else.

It also measures what each member does, because a closure that silently
produced no mass flux would pass an identity test: every member must
move the deep tendencies on some column, and every family must differ
from the ensemble mean on some column.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

cp = pytest.importorskip("cupy")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TOOLS = os.path.join(_ROOT, "tools", "gf_wrf461_oracle")
for _p in (_ROOT, _TOOLS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from gpuwm.config import GF_CLOSURE_FAMILIES, GF_CLOSURE_MEMBERS  # noqa: E402
from gpuwm.core.gf import gf_workspace_floats                 # noqa: E402
from gpuwm.core.kernels import load_module                    # noqa: E402
from gpuwm.verify.gf_driver import gfdrv_column               # noqa: E402
from gpuwm.verify.gf_oracle import GF_NZ, load_gf_oracle      # noqa: E402
from gf_field_lists import (                                  # noqa: E402
    DRV_IN_LEV, DRV_ISCA_FIELDS, DRV_LEV_FIELDS, DRV_SCA_FIELDS,
    captured_fzu, drv_scalar_inputs,
)

NZ = GF_NZ
#: Every value clos_choice admits.  0 is the control: the same comparison
#: on the arm already graded against WRF.
CHOICES = tuple(range(GF_CLOSURE_MEMBERS + 1))
_LEV = ["rthcuten", "rqvcuten", "rqccuten", "rqicuten", "dudt_phy",
        "dvdt_phy", "outt", "outq", "outqc", "outu", "outv"]
_SCA = ["raincv", "pratec", "pret"]


@pytest.fixture(scope="module")
def fixture():
    return load_gf_oracle()


@pytest.fixture(scope="module")
def module():
    return load_module("gf")


def _device(module, fixture, ichoice):
    gl = fixture.levels
    gs = fixture.surface
    n = fixture.ncol
    lvin = np.zeros((n, len(DRV_IN_LEV), NZ), dtype=np.float32)
    for j, name in enumerate(DRV_IN_LEV):
        lvin[:, j, :] = gl[name]
    scin = drv_scalar_inputs(fixture, True)
    iin = np.zeros((n, 3), dtype=np.int32)
    iin[:, 0] = gs["kpbl"].astype(np.int32)
    iin[:, 1] = gs["ishallow"].astype(np.int32)
    iin[:, 2] = np.int32(ichoice)
    d_lev = cp.zeros((n, len(DRV_LEV_FIELDS), NZ), dtype=cp.float32)
    d_sca = cp.zeros((n, len(DRV_SCA_FIELDS)), dtype=cp.float32)
    d_isc = cp.zeros((n, len(DRV_ISCA_FIELDS)), dtype=cp.int32)
    d_ws = cp.empty(gf_workspace_floats(NZ, n), dtype=cp.float32)
    fn = module.get_function("gf_gfdrv_stage")
    fn(((n + 63) // 64,), (64,),
       (cp.asarray(np.ascontiguousarray(lvin)),
        cp.asarray(np.ascontiguousarray(scin)),
        cp.asarray(np.ascontiguousarray(iin)),
        d_lev, d_sca, d_isc, d_ws,
        np.int32(1), np.int32(n), np.int32(NZ)))
    cp.cuda.Stream.null.synchronize()
    lev = cp.asnumpy(d_lev)
    sca = cp.asnumpy(d_sca)
    out = {name: lev[:, DRV_LEV_FIELDS.index(name), :] for name in _LEV}
    out.update({name: sca[:, DRV_SCA_FIELDS.index(name)] for name in _SCA})
    return out


def _reference(fixture, ichoice):
    gl = fixture.levels
    gs = fixture.surface
    up, dn, sh = captured_fzu(fixture)
    rows = []
    for ci in range(fixture.ncol):
        rows.append(gfdrv_column(
            u=gl["u"][ci], v=gl["v"][ci], w=gl["w"][ci], t=gl["t"][ci],
            qv=gl["qv"][ci], p=gl["p"][ci], pi=gl["pi"][ci],
            rho=gl["rho"][ci], dz8w=gl["dz8w"][ci], p8w=gl["p8w"][ci],
            rthften=gl["rthften"][ci], rqvften=gl["rqvften"][ci],
            rthraten=gl["rthraten"][ci], rthblten=gl["rthblten"][ci],
            rqvblten=gl["rqvblten"][ci], ht=gs["ht"][ci], hfx=gs["hfx"][ci],
            qfx=gs["qfx"][ci], xland=gs["xland"][ci],
            kpbl=int(gs["kpbl"][ci]), dt=gs["dt"][ci], dx=gs["dx"][ci],
            ishallow=int(gs["ishallow"][ci]), ichoice=ichoice,
            fzu_up=up[ci] if up[ci] > 0 else None,
            fzu_dn=dn[ci] if dn[ci] > 0 else None,
            fzu_sh=sh[ci] if sh[ci] > 0 else None))
    out = {name: np.stack([np.asarray(r[name], dtype=np.float32)
                           for r in rows]) for name in _LEV}
    out.update({name: np.array([np.float32(r[name]) for r in rows],
                               dtype=np.float32) for name in _SCA})
    return out


@pytest.fixture(scope="module")
def runs(module, fixture):
    return {c: (_device(module, fixture, c), _reference(fixture, c))
            for c in CHOICES}


def _differing(a, b):
    aw = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    bw = np.ascontiguousarray(b, dtype=np.float32).view(np.uint32)
    return int(np.count_nonzero(aw != bw))


@pytest.mark.gpu
@pytest.mark.parametrize("ichoice", CHOICES)
def test_the_kernel_equals_the_reference_on_every_member(runs, ichoice):
    got, want = runs[ichoice]
    bad = {name: _differing(got[name], want[name])
           for name in _LEV + _SCA}
    bad = {k: v for k, v in bad.items() if v}
    assert not bad, f"clos_choice={ichoice}: differing words {bad}"


@pytest.mark.gpu
@pytest.mark.parametrize("ichoice", CHOICES[1:])
def test_every_member_produces_deep_convection_somewhere(runs, ichoice):
    """No member is a closure that never fires on this fixture."""
    got, _ = runs[ichoice]
    heating = np.abs(got["outt"]).max(axis=1)
    columns = int(np.count_nonzero(heating > 0))
    raining = int(np.count_nonzero(got["pret"] > 0))
    assert columns > 0 and raining > 0, (
        f"clos_choice={ichoice}: {columns} columns heated, "
        f"{raining} rained")


@pytest.mark.gpu
@pytest.mark.parametrize("members,what", GF_CLOSURE_FAMILIES)
def test_each_family_differs_from_the_ensemble_mean(runs, members, what):
    """The selector reaches the closure: a family's member is not the
    ensemble mean in disguise."""
    ensemble, _ = runs[0]
    got, _ = runs[members[0]]
    assert _differing(got["rthcuten"], ensemble["rthcuten"]) > 0, what
