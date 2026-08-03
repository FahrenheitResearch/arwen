"""The MYNN PBL coupling layer: WRF's ``mynnedmf_wrapper_run``, on device.

``module_pbl_driver.F:1660-1721`` does **not** call ``mynn_bl_driver``.  It
calls ``mynnedmf_wrapper_run`` (``module_bl_mynn_wrapper.F``), and that
wrapper owns three things the column solver does not:

* the mixing-ratio to specific-humidity conversion on the way in
  (``:453-475``) and the reverse on the tendencies and the subgrid cloud
  water on the way out (``:587-607``);
* ``initflag = 1`` on ``itimestep == 1`` and 0 afterwards (``:1643-1647``),
  which is what selects ``mym_initialize`` over the carried state; and
* the per-``j``-row packing that gpuwm replaces with a single
  ``(ncol, nz)`` transpose, since every column is independent.

Keeping it in its own module rather than inside :mod:`gpuwm.core.physics`
means the unit conversions live next to the source anchor that justifies
them.  Losing either one is a silent O(qv) bias in the moisture tendency,
not a crash, which is the kind of defect that survives a smoke test.
"""

from __future__ import annotations

from collections.abc import Mapping

import cupy as cp
import numpy as np

from gpuwm.core.kernels import get_kernel
from gpuwm.core.mynn_pbl_gpu import (
    _flag_mask,
    _nonfinite,
    mynn_bl_driver_cuda,
)
from gpuwm.core.mynn_pbl_scratch import (
    MYNN_PBL_COLUMN_CHUNK,
    MYNN_PBL_STAGE_LAYERS,
    MYNN_PBL_TENDENCY_FIELDS,
    MynnPblScratch,
    SLOT_STAGE_DX,
    SLOT_STAGE_LAYER,
    SLOT_ZERO_COLUMN,
    mynn_pbl_tendency_field_shapes,
)
from gpuwm.core.state import DTYPE


#: WRF Registry ``mynnscheme`` prognostic/carried 3-D state, in the spelling
#: ``module_pbl_driver.F`` binds.  ``el_pbl``/``sh3d``/``sm3d`` keep WRF's
#: names rather than the solver's ``el``/``sh``/``sm`` so the restart
#: manifest and wrfout read the same identifiers WRF writes.
MYNN_PBL_STATE_3D = (
    "qke", "tsq", "qsq", "cov", "el_pbl", "sh3d", "sm3d",
    "qc_bl", "qi_bl", "cldfra_bl",
)

#: Solver output name -> the ``PhysicsDriver.fields`` key it persists into.
_STATE_FIELD = {
    "qke": "qke", "tsq": "tsq", "qsq": "qsq", "cov": "cov",
    "el": "el_pbl", "sh": "sh3d", "sm": "sm3d",
    "qc_bl": "qc_bl", "qi_bl": "qi_bl", "cldfra_bl": "cldfra_bl",
}

#: Per-column plume diagnostics the wrapper exports (``:1698-1699``).
MYNN_PBL_DIAGNOSTICS_2D = ("maxwidth", "maxmf", "ztop_plume")

#: Integer per-column diagnostic; kept apart because it is int32.
MYNN_PBL_DIAGNOSTICS_INT_2D = ("ktop_plume",)

_TPB = 128
#: Full MYNN wrapper call requires the lowest five model levels.
VERTICAL_LEVEL_BOUNDS = (5, None)

#: WRF Registry packages whose moist state declares ``F_QS``.  The PBL
#: driver forwards that generated flag at ``module_pbl_driver.F:873-878``.
#:
#: ``28`` (Registry.EM_COMMON:3036, package ``thompsonaero``:
#: ``moist:qv,qc,qr,qi,qs,qg``) belongs here for exactly the reason 8 does,
#: and was missing while mp_physics=28 was being ported: with the flag false
#: ``module_bl_mynn.F:734`` and ``:876`` substitute ``sqs = 0``, so MYNN saw
#: no snow at all under the one Thompson variant whose package declares it,
#: and ``:1324`` skipped ``rqsblten``.  MEASURED on the committed WRF v4.6.1
#: MYNN driver oracle's ``snow_anvil`` column (max sqs 4.08e-05): the
#: substitution drove ``qi_bl`` from 5.4863e-07 to exactly 0 and moved
#: ``qc_bl``, ``cldfra_bl``, ``rqvblten``, ``rthblten`` and ``exch_h`` with
#: it.  ``gpuwm/physics_registry_v2.json`` has published 28 as a
#: flag_qs-true selector since the scheme was registered; the two are bound
#: together by ``tests/test_physics_registry.py::
#: test_the_registry_flag_qs_contract_is_the_one_the_shipped_runtime_applies``.
MYNN_SNOW_MICROPHYSICS = frozenset((6, 8, 10, 18, 28))


def mynn_flag_qs(mp_physics: int) -> bool:
    """Return WRF v4.6.1's Registry-derived ``FLAG_QS`` for ``mp_physics``."""

    return int(mp_physics) in MYNN_SNOW_MICROPHYSICS


#: Layer inputs the driver reads straight out of ``atmosphere``, paired with
#: the key they arrive under.  ``w`` is WRF's staggered vertical velocity
#: indexed kts..kte -- the LOWER interface of each mass layer, not a layer
#: mean (``module_pbl_driver.F:1663`` binds ``w=w``).
_ATMOSPHERE_LAYERS = (
    ("dz", "dz"), ("u", "u"), ("v", "v"), ("th", "theta"),
    ("p", "pressure"), ("exner", "exner"), ("rho", "rho"),
    ("tk", "temperature"), ("qv", "qv"), ("qc", "qc"), ("qi", "qi"),
    ("qs", "qs"),
)
#: Per-column inputs that are contiguous prefixes of a ``(ny, nx)`` field and
#: therefore need no staging buffer at all: a chunk of one is a contiguous
#: slice.
_COLUMN_FIELDS = (
    ("xland", "xland"), ("ts", "tsk"), ("ps", "psfc"), ("ust", "ust"),
    ("hfx", "hfx"), ("qfx", "qfx"), ("wspd", "wspd"), ("pblh", "pblh"),
    ("rmol", "rmol"),
)
#: Solver output -> the returned A-grid tendency name.
_TENDENCY_OUT = (
    ("rublten", "du"), ("rvblten", "dv"), ("rthblten", "dtheta"),
    ("rqvblten", "dqv"), ("rqcblten", "dqc"), ("rqiblten", "dqi"),
)


def mynn_pbl_step(
    atmosphere: Mapping[str, cp.ndarray],
    fields: Mapping[str, cp.ndarray],
    *,
    w: cp.ndarray,
    dx: float,
    delt: float,
    itimestep: int,
    mp_physics: int,
    state=None,
    column_chunk: int | None = None,
    **options,
) -> dict[str, cp.ndarray]:
    """Run one MYNN PBL call and write its state back into ``fields``.

    Returns the A-grid tendency set (``du``/``dv``/``dtheta``/``dqv``/
    ``dqc``/``dqi``) in the same shape and units
    :func:`gpuwm.core.physics.couple_ysu_tendencies` consumes, so the
    mass-coupling and C-grid interpolation stay one implementation.

    ``itimestep`` is WRF's, one-based: the first model step is 1 and selects
    the ``mym_initialize`` cold start.  ``options`` are the MYNN identity
    knobs, forwarded verbatim to the driver so the refusal for an unported
    branch comes from the routine that would have had to implement it.

    The domain is walked in chunks of ``column_chunk`` columns
    (:data:`gpuwm.core.mynn_pbl_scratch.MYNN_PBL_COLUMN_CHUNK` by default).
    Every MYNN kernel gives one CUDA thread one complete column and reads no
    neighbour, so the chunk boundary is not a seam: the split is bitwise
    identical to the single wide call at every width measured, and that is
    asserted in ``tests/test_mynn_pbl_scratch.py`` rather than assumed.  What
    it buys is a working set that no longer grows with ``ny * nx`` -- the
    whole reason a preflight could not price this scheme before.
    """

    if int(itimestep) < 1:
        raise ValueError("MYNN PBL itimestep is one-based and starts at 1")
    theta = atmosphere["theta"]
    nz, ny, nx = theta.shape
    if nz < VERTICAL_LEVEL_BOUNDS[0]:
        raise ValueError(
            f"MYNN PBL requires at least {VERTICAL_LEVEL_BOUNDS[0]} "
            "vertical levels")
    ncol = ny * nx
    if w.shape[0] < nz:
        raise ValueError("MYNN PBL needs w on the lower interface of each "
                         "layer, i.e. at least nz levels")

    chunk = MYNN_PBL_COLUMN_CHUNK if column_chunk is None else int(column_chunk)
    if chunk < 1:
        raise ValueError("MYNN PBL column_chunk must be a positive integer")
    chunk = min(chunk, ncol)

    if state is None:
        work = MynnPblScratch.standalone(chunk, nz)
        tendencies = {name: cp.zeros((nz, ny, nx), dtype=DTYPE)
                      for name in MYNN_PBL_TENDENCY_FIELDS}
    else:
        work = MynnPblScratch.from_state(state, chunk, nz)
        tendencies = {
            name: state.scratch(shape, slot) for name, (slot, shape) in zip(
                MYNN_PBL_TENDENCY_FIELDS,
                mynn_pbl_tendency_field_shapes(nz, ny, nx).items())
        }

    # Flat (nz, ncol) / (ncol,) views.  Reshape on a C-contiguous field is a
    # view, so nothing here is a copy.
    source = {name: atmosphere[key].reshape(nz, ncol)
              for name, key in _ATMOSPHERE_LAYERS}
    source["w"] = cp.ascontiguousarray(w[:nz]).reshape(nz, ncol)
    for solver_name, field_name in _STATE_FIELD.items():
        source[solver_name] = fields[field_name].reshape(nz, ncol)
    flat_columns = {name: fields[key].reshape(ncol)
                    for name, key in _COLUMN_FIELDS}
    flat_kpbl = fields["kpbl"].reshape(ncol)
    out_state = {name: fields[field].reshape(nz, ncol)
                 for name, field in _STATE_FIELD.items()}
    out_state["exch_h"] = fields["exch_h"].reshape(nz, ncol)
    out_state["exch_m"] = fields["exch_m"].reshape(nz, ncol)
    flat_tendencies = {name: array.reshape(nz, ncol)
                       for name, array in tendencies.items()}

    initflag = 1 if int(itimestep) == 1 else 0
    for lo in range(0, ncol, chunk):
        hi = min(lo + chunk, ncol)
        n = hi - lo
        stage = work.group(SLOT_STAGE_LAYER, MYNN_PBL_STAGE_LAYERS, (n, nz))
        for name in source:
            stage[name][...] = source[name][:, lo:hi].T

        # --- module_bl_mynn_wrapper.F:453-475 mixing ratio -> specific -----
        count = n * nz
        blocks = (count + _TPB - 1) // _TPB
        get_kernel("mynn_pbl", "mynn_wrapper_to_specific")(
            (blocks,), (_TPB,),
            (stage["qv"], stage["qc"], stage["qi"], stage["qs"],
             stage["sqv"], stage["sqc"], stage["sqi"], stage["sqs"],
             np.int32(count)),
        )

        spacing = work.one(SLOT_STAGE_DX, (n,))
        spacing[...] = DTYPE(dx)
        zeros_column = work.one(SLOT_ZERO_COLUMN, (n,))
        values = {name: stage[name] for name in
                  ("dz", "u", "v", "w", "th", "sqv", "sqc", "sqi", "sqs",
                   "p", "exner", "rho", "tk")}
        for solver_name in _STATE_FIELD:
            values[solver_name] = stage[solver_name]
        values["dx"] = spacing
        # gpuwm has no ocean-current coupling; WRF's uncoupled path passes
        # the zero-initialised UOCE/VOCE arrays (Registry default 0.0).
        values["uoce"] = zeros_column
        values["voce"] = zeros_column
        for name in flat_columns:
            values[name] = flat_columns[name][lo:hi]
        values["kpbl"] = flat_kpbl[lo:hi]

        out = mynn_bl_driver_cuda(
            values, initflag=initflag, delt=DTYPE(delt), scratch=work,
            flag_qs=mynn_flag_qs(mp_physics), **options,
        )

        # --- module_bl_mynn_wrapper.F:587-607 specific -> mixing ratio -----
        get_kernel("mynn_pbl", "mynn_wrapper_from_specific")(
            (blocks,), (_TPB,),
            (stage["sqv"], out["rqvblten"], out["rqcblten"],
             out["rqiblten"], out["qc_bl"], out["qi_bl"], np.int32(count)),
        )

        for name, target in out_state.items():
            target[:, lo:hi] = out[name].T
        for name in ("pblh", "rmol", "kpbl"):
            fields[name].reshape(ncol)[lo:hi] = out[name]
        for name in (*MYNN_PBL_DIAGNOSTICS_2D, *MYNN_PBL_DIAGNOSTICS_INT_2D):
            fields[name].reshape(ncol)[lo:hi] = out[name]
        for solver_name, tendency_name in _TENDENCY_OUT:
            flat_tendencies[tendency_name][:, lo:hi] = out[solver_name].T

    return dict(tendencies)


#: Validity words for :func:`validate_mynn_tendencies`, which has no
#: workspace to draw from because ``physics.py`` calls it on the returned
#: fields.  Built on first use rather than at import so a CPU-only test run
#: can import this module, and held for the life of the process: 256 bytes,
#: allocated once, never per step.
_VALIDITY_FLAGS: list[cp.ndarray] = []


def _validity_flags() -> cp.ndarray:
    if not _VALIDITY_FLAGS:
        _VALIDITY_FLAGS.append(cp.zeros(len(MYNN_PBL_TENDENCY_FIELDS),
                                        dtype=cp.int32))
    return _VALIDITY_FLAGS[0]


def validate_mynn_tendencies(out: Mapping[str, cp.ndarray]) -> None:
    """Refuse a non-finite MYNN tendency before it reaches RK3.

    The YSU path has the same guard for the same reason: a NaN that enters
    the slow-tendency slot is unattributable one step later.

    ``bool(cp.isfinite(a).all())`` per field allocated a full ``(nz, ny,
    nx)`` boolean temporary and synchronised six times -- 105.8 MiB of
    per-step churn at the d04 nest, on arrays that had just been produced
    without any.  The reduction writes one int32 word per field and the six
    are read together; ``tests/test_mynn_pbl_scratch.py`` pins the two
    spellings against each other rather than assuming they agree.
    """

    names = tuple(out)
    if not names:
        return
    mask = _flag_mask(_nonfinite(), [out[name] for name in names],
                      _validity_flags())
    for name, nonfinite in zip(names, mask):
        if nonfinite:
            raise FloatingPointError(
                f"MYNN PBL produced a non-finite {name} tendency")


__all__ = [
    "MYNN_PBL_DIAGNOSTICS_2D",
    "MYNN_PBL_DIAGNOSTICS_INT_2D",
    "MYNN_SNOW_MICROPHYSICS",
    "MYNN_PBL_STATE_3D",
    "mynn_flag_qs",
    "mynn_pbl_step",
    "validate_mynn_tendencies",
]
