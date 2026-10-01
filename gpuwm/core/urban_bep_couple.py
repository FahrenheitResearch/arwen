"""The BEP / BEP+BEM surface coupling (``sf_urban_physics`` 2 and 3).

WRF runs the multi-layer urban models after the LSM column loop and then, in
the LSM driver itself, turns their output into what the PBL scheme and the
surface diagnostics read (``module_sf_noahdrv.F:1679-1776`` for Noah,
``module_sf_noahmpdrv.F:3689-3776`` for Noah-MP):

* the BEP source terms are weighted by the urban fraction, ``a_q``/``a_e``
  zeroed, ``vl``/``sf`` blended with the rural (unobstructed) value 1;
* the RURAL surface flux and drag are folded into level 1 of the same terms
  -- on every column, urban or not, because once ``flag_bep`` is on YSU and
  MYJURB take the whole surface flux from these terms;
* on urban columns the skin temperature, emissivity, albedo, ground, heat and
  moisture fluxes and ``ust`` become the frc-weighted urban/rural blend, and
  ``ts/sh/lh/g/rn_urb2d`` are written.

Then ``module_surface_driver.F:3022-3035`` (Noah) / ``:3408-3421`` (Noah-MP)
overwrite T2/TH2/Q2/U10/V10 on urban-category columns with the lowest model
level (:func:`after_surface_diagnostics`).

The BEP lane owns this module; the BEP+BEM model calls :func:`couple` after
its own column exactly as WRF runs the same block after either model.

Rural words.  The block reads HFX_RURAL, QFX_RURAL, GRDFLX_RURAL,
EMISS_RURAL, TSK_RURAL and ALB_RURAL.  Noah assigns each of those the same
word it assigns HFX, QFX, ... (noahdrv.F:1225-1273; :871-876 on water) and
Noah-MP copies them from HFX, QFX, ... on entry (noahmpdrv.F:3363-3372), so
the kernel reads them from the LSM's fields before it overwrites them.  That
is also why it does not read ``UrbanState.rural``: that handoff is written
per urban column, and this block needs the rural words on every column.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np

#: The PBL handoff names (DESIGN 3.1), in kernel argument order.
PBL_TERMS = ("a_u_bep", "a_v_bep", "a_t_bep", "a_q_bep", "a_e_bep",
             "b_u_bep", "b_v_bep", "b_t_bep", "b_q_bep", "b_e_bep",
             "sf_bep", "vl_bep")
#: The rest of UrbanState.pbl_terms, which the block leaves alone.
PBL_PASSIVE = ("dlg_bep", "dl_u_bep")
#: BEP/BEP_BEM column outputs the block reads.
BEP_OUTPUTS = ("rl_up_urb", "rs_abs_urb", "emiss_urb", "grdflx_urb")
#: LSM fields the block reads as rural words and then overwrites.
FIELD_INOUT = ("ust", "tsk", "hfx", "qfx", "lh", "grdflx", "albedo", "emiss")
#: Urban diagnostics written on frc > 0 columns.
URBAN_DIAG = ("ts_urb2d", "sh_urb2d", "lh_urb2d", "g_urb2d", "rn_urb2d")

_TPB = 128
_LSM_NOAH, _LSM_NOAHMP = 2, 4


def _check(name, arr, shape):
    import cupy as cp

    if (not isinstance(arr, cp.ndarray) or arr.dtype != np.float32
            or not arr.flags.c_contiguous or arr.shape != shape):
        raise ValueError(
            f"{name} must be a C-contiguous float32 CuPy array of shape "
            f"{shape}")


def launch_bep_couple(*, lsm: int, frc_urb2d, dz8w, rho, u_phy, v_phy, glw,
                      swdown, bep_out: Mapping, pbl: Mapping,
                      fields: Mapping, diag: Mapping) -> None:
    """Run the coupling block in place over every column.

    ``lsm`` is ``sf_surface_physics`` (2 Noah, 4 Noah-MP).  ``pbl`` holds
    :data:`PBL_TERMS` (``(nz, ny, nx)``, ``sf_bep`` may be ``(nz+1, ny,
    nx)``), ``bep_out`` :data:`BEP_OUTPUTS`, ``fields`` :data:`FIELD_INOUT`
    and ``diag`` :data:`URBAN_DIAG`, all ``(ny, nx)``.
    """
    import cupy as cp

    from gpuwm.core.kernels import get_kernel

    if lsm not in (_LSM_NOAH, _LSM_NOAHMP):
        raise ValueError(
            f"the BEP coupling block exists for Noah (2) and Noah-MP (4) "
            f"only, got sf_surface_physics={lsm}: WRF's RUC and no-LSM "
            "paths never call an urban model")
    nz, ny, nx = dz8w.shape
    for name, arr in (("dz8w", dz8w), ("rho", rho), ("u_phy", u_phy),
                      ("v_phy", v_phy)):
        _check(name, arr, (nz, ny, nx))
    for name in PBL_TERMS:
        arr = pbl[name]
        want = (nz + 1, ny, nx) if name == "sf_bep" else (nz, ny, nx)
        if arr.shape != want and arr.shape != (nz, ny, nx):
            raise ValueError(f"{name} must have shape {want}")
        _check(name, arr, arr.shape)
    for name, arr in (("frc_urb2d", frc_urb2d), ("glw", glw),
                      ("swdown", swdown)):
        _check(name, arr, (ny, nx))
    for group, names in ((bep_out, BEP_OUTPUTS), (fields, FIELD_INOUT),
                         (diag, URBAN_DIAG)):
        for name in names:
            _check(name, group[name], (ny, nx))
    ncol = ny * nx
    if ncol == 0:
        return
    kernel = get_kernel("urban_bep_couple", "urban_bep_couple")
    kernel(((ncol + _TPB - 1) // _TPB,), (_TPB,),
           (frc_urb2d, dz8w, rho, u_phy, v_phy, glw, swdown)
           + tuple(bep_out[n] for n in BEP_OUTPUTS)
           + tuple(pbl[n] for n in PBL_TERMS)
           + tuple(fields[n] for n in FIELD_INOUT)
           + tuple(diag[n] for n in URBAN_DIAG)
           + (np.int32(lsm), np.int32(nz), np.int32(ny), np.int32(nx)))


def launch_bep_sfcdiag(*, utype_urb2d, th_phy, qv, u_phy, v_phy, psfc,
                       t2, th2, q2, u10, v10) -> None:
    """module_surface_driver.F:3022-3035 / :3408-3421, in place."""
    import cupy as cp

    from gpuwm.core.kernels import get_kernel

    ny, nx = utype_urb2d.shape
    if (not isinstance(utype_urb2d, cp.ndarray)
            or utype_urb2d.dtype != np.int32
            or not utype_urb2d.flags.c_contiguous):
        raise ValueError("utype_urb2d must be C-contiguous int32")
    nz = th_phy.shape[0]
    for name, arr in (("th_phy", th_phy), ("qv", qv), ("u_phy", u_phy),
                      ("v_phy", v_phy)):
        _check(name, arr, (nz, ny, nx))
    for name, arr in (("psfc", psfc), ("t2", t2), ("th2", th2), ("q2", q2),
                      ("u10", u10), ("v10", v10)):
        _check(name, arr, (ny, nx))
    ncol = ny * nx
    if ncol == 0:
        return
    kernel = get_kernel("urban_bep_couple", "urban_bep_sfcdiag")
    kernel(((ncol + _TPB - 1) // _TPB,), (_TPB,),
           (utype_urb2d, th_phy, qv, u_phy, v_phy, psfc, t2, th2, q2, u10,
            v10, np.int32(ny), np.int32(nx)))


def couple(state, *, lsm: int, fields: Mapping, atmosphere: Mapping,
           dt: float, bep_out: Mapping | None = None) -> None:
    """DESIGN 3.4 entry: weight and merge after BEP or BEP_BEM ran.

    ``state`` is the domain's ``UrbanState``: ``state.pbl_terms`` (the
    mapping of :data:`PBL_TERMS` + :data:`PBL_PASSIVE`), ``state.frc_urb2d``,
    the BEP (or BEP_BEM) column outputs :data:`BEP_OUTPUTS` -- passed as
    ``bep_out`` or held as ``state.bep_out`` -- and the urban diagnostics
    under their Registry names.  ``fields`` is
    ``PhysicsDriver.fields``; ``atmosphere`` is ``_prepare_atmosphere``'s
    dict (``dz``, ``rho``, ``u``, ``v``).  ``dt`` is unused by the block and
    accepted for the common signature.
    """
    del dt
    launch_bep_couple(
        lsm=int(lsm), frc_urb2d=state.frc_urb2d, dz8w=atmosphere["dz"],
        rho=atmosphere["rho"], u_phy=atmosphere["u"], v_phy=atmosphere["v"],
        glw=fields["glw"], swdown=fields["swdown"],
        bep_out=state.bep_out if bep_out is None else bep_out,
        pbl=state.pbl_terms, fields=fields,
        diag={name: getattr(state, name) for name in URBAN_DIAG})


def after_surface_diagnostics(state, *, lsm: int, fields: Mapping,
                              atmosphere: Mapping, cfg=None) -> None:
    """Options 2/3: T2/TH2/Q2/U10/V10 from level 1 on urban columns."""
    del lsm, cfg
    launch_bep_sfcdiag(
        utype_urb2d=state.utype_urb2d, th_phy=atmosphere["theta"],
        qv=atmosphere["qv"], u_phy=atmosphere["u"], v_phy=atmosphere["v"],
        psfc=atmosphere["p_interface"][0], t2=fields["t2"],
        th2=fields["th2"], q2=fields["q2"], u10=fields["u10"],
        v10=fields["v10"])
