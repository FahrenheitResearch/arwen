"""The Noah-MP coupling layer: WRF's ``noahmplsm`` driver, column by column.

``module_surface_driver.F:3127-3181`` does **not** call ``NOAHMP_SFLX``.  It
calls ``noahmplsm`` (``phys/module_sf_noahmpdrv.F:12-1432``), and that driver
owns everything between WRF's grid arrays and the column solver:

* the ``ITIMESTEP == 1`` open-water / sea-ice soil initialisation (:689-711);
* the sea-ice and open-water ``CYCLE ILOOP`` skips (:715-725), which decide
  which columns the LSM touches at all;
* the 2-D-to-1-D forcing marshalling (:729-796), including
  ``Q_ML = QV3D/(1+QV3D)``, ``P_ML`` from the two lowest interfaces, and
  ``Z_ML = 0.5*DZ8W(1)``;
* the soil-type water override, the urban vegetation remap, the
  ``VEGTYP in {25,26,27}`` bare override and the ``FICEOLD`` reconstruction
  (:912-1044);
* ``TRANSFER_MP_PARAMETERS`` per column (:962); and
* the write-back (:1223-1400), which is where ``TSK = TRAD``,
  ``CANWAT = CANLIQ + CANICE``, ``Q2MV = Q2MV/(1-Q2MV)`` and the
  SOILENERGY/SNOWENERGY integrals live.

Losing any of it is not a crash.  Losing the ``Q_ML`` conversion is an O(qv)
bias in every flux; losing the open-water skip runs a land model on the sea;
losing ``TSK = TRAD`` leaves the radiative temperature out of the surface
layer's next call.  So this module is that driver, with the anchors next to
the arithmetic.

Where the column runs
---------------------
On the forecast path, nowhere in Python per column.  :func:`noahmp_lsm_step`
dispatches to :func:`_lsm_step_slab`, which evaluates the driver prologue as
whole-grid CuPy statements and answers the land columns through
:func:`gpuwm.core.noahmp_column_slab.evaluate_sflx_slab` -- the eight leaf
batches, ENERGY's four composition segments and ``sflx_post``'s two, launched
in the scalar order over chunks of :data:`SLAB_COLUMN_CHUNK` columns.
Measured twice at 360,000 land columns on one RTX 5090, that call costs
0.202--0.227 s against 166--206 s through the staged path below in the same
process; the assembled slab is bitwise against the scalar column and against
the staged path at every width checked.

The per-column seam this module grew up around is **kept, not dead**: it is
the paired second implementation.  :func:`sflx_steps` starts one generator
frame per column, suspends at each physical leaf call that has a device
batch (**eight** -- :data:`DEVICE_LEAF_BATCH` is the authority), and resumes
in the same frame.  It is what runs when ``GPUWM_NOAHMP_STAGED_COLUMNS=1``
selects it by name, when a test binds another evaluator, or when
``GPUWM_NOAHMP_HOST_LEAVES=1`` asks for the CPython leaves -- and the
six-step digest gates in ``tests/test_noahmp_runtime.py`` are what hold the
two paths to the same answer, bit for bit.  Booby traps in
``tests/test_noahmp_device_wiring.py`` prove which path a forecast takes,
in both directions, because "finished, verified, and wired to nothing" has
happened twice in this repository.

The first version of this seam paused the column by *restarting* it: the
vegetated column was rewound to its entry state and replayed the pre-VEGE
prefix once the batch returned.  That was answer-preserving and it was
expensive -- it duplicated every host statement before VEGE_FLUX, which is why
the whole-column cost barely fell.  The generator continuation replaced it,
and the slab orchestration replaces the per-column frame itself.

Noah-MP has no horizontal coupling, so neither the order in which staged calls
are answered nor the tiling in :data:`COLUMN_BATCH` can change any column's
answer; :func:`evaluate_leaf_batch_on_host` is the paired authority that shows
it does not.

The published option identity
-----------------------------
:data:`NOAHMP_OPTION_IDENTITY` in :mod:`gpuwm.config` is the enforced part --
one accepted value per namelist knob, refused by ``validate_run_config``
before a run starts.  :data:`NOAHMP_RUNTIME_RESTRICTIONS` is the rest: every
restriction that the identity schema has no field for.  Both are published
because a restriction that lives only in a docstring is a restriction a user
will meet at hour three of a forecast.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import numpy as np

from gpuwm.core.noahmp import (DEFAULT_VEGETATION_DATASET,
                               load_noahmp_parameters,
                               transfer_mp_parameters)
from gpuwm.core.noahmp_driver import (LandUseIdentity, noahmp_init_column,
                                      snow_init_column)
from gpuwm.core.noahmp_libm import f32
from gpuwm.core.noahmp_sflx import DVEG as SFLX_DVEG
from gpuwm.core.noahmp_sflx import SflxParameters, sflx_steps
from gpuwm.core.noahmp_snow import SnowColumn
from gpuwm.core.noahmp_energy import LeafRequest, answer_on_host
from gpuwm.core.noahmp_bareflux_gpu import evaluate_bare_flux_calls
from gpuwm.core.noahmp_radiation_gpu import evaluate_radiation_calls
from gpuwm.core.noahmp_thermal_gpu import (evaluate_phasechange_calls,
                                           evaluate_thermoprop_calls,
                                           evaluate_tsnosoi_calls)
from gpuwm.core.noahmp_sflx_pre_gpu import evaluate_sflx_pre_calls
from gpuwm.core.noahmp_vegeflux_gpu import evaluate_vege_flux_calls
from gpuwm.core.noahmp_water_gpu import evaluate_water_calls
from gpuwm.core.surface_forcing import (
    SurfacePrecipitationForcing,
    noahmp_six_precipitation_rates,
)

#: How many land columns are staged before their paused leaves are batched.
#: The bound is memory, not arithmetic: every staged column holds one live
#: generator frame and its ``sflx_pre`` products until the batch returns, and
#: Noah-MP has no horizontal coupling, so any tiling gives the same answer.
#: A 360,000-column nest would otherwise hold 360,000 suspended frames at once.
COLUMN_BATCH = 2048


#: Leaf name to its CUDA batch.  Every entry answers the *physical* call the
#: column pauses at, so adding one is a packing job and never a re-transcription.
DEVICE_LEAF_BATCH = {
    "sflx_pre": evaluate_sflx_pre_calls,
    "thermoprop": evaluate_thermoprop_calls,
    "radiation": evaluate_radiation_calls,
    "vege_flux": evaluate_vege_flux_calls,
    "bare_flux": evaluate_bare_flux_calls,
    "tsnosoi": evaluate_tsnosoi_calls,
    "phasechange": evaluate_phasechange_calls,
    "water": evaluate_water_calls,
}


def evaluate_leaf_batch_on_device(leaf: str, calls):
    """Answer one leaf's collected calls with its CUDA batch."""
    try:
        batch = DEVICE_LEAF_BATCH[leaf]
    except KeyError:
        raise ValueError(f"no device batch for Noah-MP leaf {leaf!r}")
    return batch(calls)


def evaluate_leaf_batch_on_host(leaf: str, calls):
    """Answer the same collected calls with the unmodified CPython leaves.

    This is the paired authority for every measurement and every bitwise gate:
    the identical staging, the identical column arithmetic, and the leaf
    evaluated by the routine the WRF fixtures pin.
    """
    return [answer_on_host(LeafRequest(leaf, args, kwargs))
            for args, kwargs in calls]


#: Which of the two the column loop uses.  A module attribute rather than an
#: argument so a test can bind the host authority for a whole run without
#: threading a flag through ``physics``; ``GPUWM_NOAHMP_HOST_LEAVES=1`` does
#: the same thing for a timing run that must not touch the GPU.
LEAF_BATCH_EVALUATOR = evaluate_leaf_batch_on_device

#: Setting this to ``1`` forces the paired host authority for a whole process.
HOST_LEAVES_ENV = "GPUWM_NOAHMP_HOST_LEAVES"

#: Setting this to ``1`` forces the per-column staged path even when the
#: device evaluator is bound -- the pre-orchestration control flow, kept as
#: the paired second implementation for parity and timing runs.  The default
#: forecast path is the whole-slab orchestration below.
STAGED_COLUMNS_ENV = "GPUWM_NOAHMP_STAGED_COLUMNS"

#: Land columns evaluated per :func:`gpuwm.core.noahmp_column_slab
#: .evaluate_sflx_slab` launch.  This is the explicit bound the slab packers'
#: allocation-inventory rows demand: every device transient in the slab
#: composition is ``chunk x stride``, never ``nx*ny x stride``, so VRAM cost
#: is decided here rather than discovered at nest width.
#: ``gpuwm/core/preflight.py`` prices ``SLAB_COLUMN_CHUNK x
#: SLAB_TRANSIENT_BYTES_PER_COLUMN``; change one and the other moves with it.
SLAB_COLUMN_CHUNK = 65536

#: Peak device transient per column of one slab chunk, in bytes.  Measured
#: on the RTX 5090: 2,723 B of peak allocator demand for one
#: evaluate_sflx_slab call at exactly SLAB_COLUMN_CHUNK columns (the gate in
#: tests/test_noahmp_column_slab.py re-measures it and holds this ceiling to
#: within 2x).  A ceiling, not the measurement: the pool allocates in
#: rounded blocks and the composition's lifetime overlaps must be allowed to
#: breathe without moving preflight every commit.
#:
#: Demand, read at the allocator boundary, not CuPy pool *growth*: growth is
#: what the pool had to acquire from the driver, so it collapses whenever a
#: warm pool can serve the transient from blocks it is already holding.  The
#: same call reads 2,751 B/column of growth in a fresh process and 504
#: B/column deep in the GPU suite; only the demand is a property of this
#: composition rather than of what ran before it.
SLAB_TRANSIENT_BYTES_PER_COLUMN = 4096

#: The same call's whole-grid residue, in bytes per nx*ny column: the device
#: prologue intermediates (Q_ML through FICEOLD), the land index arrays and
#: the pool's cross-chunk fragmentation, which scale with the nest and not
#: with the chunk.  Measured at 360,000 columns: 311.8 MiB whole-call pool
#: growth minus the chunk term's 172.0 MiB = 407 B per grid column.
SLAB_GRID_TRANSIENT_BYTES_PER_COLUMN = 512


def _resolve_leaf_evaluator():
    if os.environ.get(HOST_LEAVES_ENV, "") == "1":
        return evaluate_leaf_batch_on_host
    return LEAF_BATCH_EVALUATOR


def _slab_path_selected() -> bool:
    """True when this call takes the whole-slab orchestration.

    The slab path exists only for the shipped device evaluator: any other
    binding -- the host authority, ``GPUWM_NOAHMP_HOST_LEAVES=1``, or a test's
    wrapped evaluator -- means the caller wants the per-column seam, whose
    whole purpose is that leaves can be answered by something else.  Identity
    rather than truthiness, so a wrapper around the device evaluator still
    routes to the path that will actually call it.
    """
    if os.environ.get(STAGED_COLUMNS_ENV, "") == "1":
        return False
    return _resolve_leaf_evaluator() is evaluate_leaf_batch_on_device


def _glacier_evaluate_on_device(inputs, count, dt):
    from gpuwm.core.noahmp_glacier_gpu import evaluate_glacier_columns
    return evaluate_glacier_columns(inputs, count, dt)


def _glacier_evaluate_on_host(inputs, count, dt):
    from gpuwm.core.noahmp_glacier_gpu import (
        evaluate_glacier_columns_on_host)
    return evaluate_glacier_columns_on_host(inputs, count, dt)


#: Which glacier evaluator the dispatch uses -- the same module-attribute
#: doctrine as :data:`LEAF_BATCH_EVALUATOR`: the CUDA batch by default,
#: rebindable by a test, and ``GPUWM_NOAHMP_HOST_LEAVES=1`` selects the
#: paired CPython transcription for a whole process.
GLACIER_BATCH_EVALUATOR = _glacier_evaluate_on_device


def _resolve_glacier_evaluator():
    if os.environ.get(HOST_LEAVES_ENV, "") == "1":
        return _glacier_evaluate_on_host
    return GLACIER_BATCH_EVALUATOR


def _glacier_execution_provenance(evaluator) -> str:
    """The execution-provenance token the census publishes when the
    glacier path runs: which implementation answered the columns."""
    if evaluator is _glacier_evaluate_on_device:
        return ("noahmp-glacier/cuda "
                "(gpuwm/core/kernels/noahmp_glacier.cu)")
    if evaluator is _glacier_evaluate_on_host:
        return "noahmp-glacier/host (gpuwm/core/noahmp_glacier.py)"
    return f"noahmp-glacier/bound:{getattr(evaluator, '__name__', repr(evaluator))}"


def _glacier_lsm_step(fields, atmosphere, glacier_mask, *, params, coszen,
                      dt, dzs, evaluator) -> None:
    """Run NOAHMP_GLACIER over the masked columns and write the driver's
    glacier arm back into ``fields``.

    The marshalling is ``module_sf_noahmpdrv.F``'s glacier arm
    (:765, :1045-1063): the same ``Q_ML``/``P_ML``/``Z_ML`` forcing the
    ordinary column receives, ``PRCP = PRECIP_IN/DT``, and
    ``TBOT = MIN(TBOT, 263.15)``.  The write-back is the arm's own fixed
    assignments (:1065-1150) plus the shared block (:1223-1400) evaluated
    with the glacier outputs -- including the ``undefined_value``
    overwrites of the vegetation carriers, which WRF performs every step
    and which are therefore state, not noise.  ``mutates fields`` only at
    the masked columns; runs after both paths' ordinary write-back, which
    never touches these columns.
    """
    import cupy as cp

    mask_host = cp.asnumpy(glacier_mask) if hasattr(
        glacier_mask, "__cuda_array_interface__") else np.asarray(
        glacier_mask)
    gj, gi = np.nonzero(mask_host)
    count = int(gj.size)
    if count == 0:
        return
    at = (gj, gi)
    dt32 = f32(dt)

    def host2(name):
        return cp.asnumpy(fields[name])[at].astype(np.float32, copy=False)

    def forcing(name, level=0):
        return cp.asnumpy(
            atmosphere[name][level]).astype(np.float32, copy=False)[at]

    temperature = forcing("temperature")
    qv = forcing("qv")
    u = forcing("u")
    v = forcing("v")
    dz1 = forcing("dz")
    p_int0 = forcing("p_interface", 0)
    p_int1 = forcing("p_interface", 1)
    one = np.float32(1.0)
    half = np.float32(0.5)
    q_ml = qv / (one + qv)                       # :758  Q_ML = QV/(1+QV)
    z_ml = half * dz1                            # :756
    p_ml = (p_int1 + p_int0) * half              # :755
    rainbl = host2("rainbl")
    prcp = rainbl / np.float32(dt32)             # :765
    tbot = np.minimum(host2("tmn"), np.float32(263.15))   # :1050
    cosz = cp.asnumpy(cp.asarray(coszen, dtype=cp.float32))[at]

    isnow = cp.asnumpy(fields["isnowxy"])[at].astype(np.int32, copy=False)
    snice = cp.asnumpy(fields["snicexy"])[:, gj, gi].T.astype(
        np.float32, copy=False)
    snliq = cp.asnumpy(fields["snliqxy"])[:, gj, gi].T.astype(
        np.float32, copy=False)
    # FICEOLD (:1026-1029): live slots divide unguarded, the rest stay 0.
    slot = np.arange(NSNOW, dtype=np.int32)[None, :]
    live = slot >= (isnow[:, None] + NSNOW)
    ficeold = np.zeros((count, NSNOW), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(snice, snice + snliq, out=ficeold, where=live)

    tslb = cp.asnumpy(fields["tslb"])[:, gj, gi].T
    tsno = cp.asnumpy(fields["tsnoxy"])[:, gj, gi].T
    inputs = {
        "cosz": cosz, "sfctmp": temperature, "sfcprs": p_ml, "uu": u,
        "vv": v, "q2": q_ml, "soldn": host2("swdown"), "prcp": prcp,
        "lwdn": host2("glw"), "tbot": tbot, "zlvl": z_ml,
        "qsnow": host2("qsnowxy"), "sneqvo": host2("sneqvoxy"),
        "albold": host2("alboldxy"), "cm": host2("cmxy"),
        "ch": host2("chxy"), "sneqv": host2("snow"),
        "snowh": host2("snowh"), "tg": host2("tgxy"),
        "tauss": host2("taussxy"), "qsfc": host2("qsfc"),
        "isnow": isnow, "ficeold": ficeold,
        "smc": cp.asnumpy(fields["smois"])[:, gj, gi].T,
        "sh2o": cp.asnumpy(fields["sh2o"])[:, gj, gi].T,
        "zsnso": cp.asnumpy(fields["zsnsoxy"])[:, gj, gi].T,
        "stc": np.concatenate((tsno, tslb), axis=1),
        "snice": snice, "snliq": snliq,
        "zsoil": zsoil_from_dzs(dzs),
    }
    r = evaluator(inputs, count, float(dt32))

    # ---- write-back: the glacier arm (:1065-1150) + shared (:1223-1400)
    def dev(name):
        return cp.asarray(np.ascontiguousarray(r[name]))

    def put2(carrier, name):
        fields[carrier][at] = dev(name)

    undef = np.float32(f32(UNDEFINED_VALUE))
    zero = np.float32(0.0)
    dt32f = np.float32(dt32)

    put2("tsk", "trad")
    put2("hfx", "fsh")
    put2("grdflx", "ssoil")
    fields["smstav"][at] = zero                              # :1233-1234
    fields["smstot"][at] = zero
    runsf_mm = dev("runsrf") * dt32f                         # :1146 RUNSF*dt
    runsb_mm = dev("runsub") * dt32f                         # :1147 RUNSB*dt
    fields["sfcrunoff"][at] = fields["sfcrunoff"][at] + runsf_mm
    fields["udrunoff"][at] = fields["udrunoff"][at] + runsb_mm
    salb = dev("albedo")
    lit = salb > np.float32(-999.0)                          # :1231-1233
    fields["albedo"][at] = cp.where(lit, salb, fields["albedo"][at])
    fields["snowc"][at] = np.float32(1.0)                    # :1065 FSNO=1
    fields["smois"][:, gj, gi] = dev("smc").T
    fields["sh2o"][:, gj, gi] = dev("sh2o").T
    stc = dev("stc")
    fields["tslb"][:, gj, gi] = stc[:, NSNOW:].T
    fields["tsnoxy"][:, gj, gi] = stc[:, :NSNOW].T
    fields["zsnsoxy"][:, gj, gi] = dev("zsnso").T
    fields["snicexy"][:, gj, gi] = dev("snice").T
    fields["snliqxy"][:, gj, gi] = dev("snliq").T
    put2("snow", "sneqv")
    put2("snowh", "snowh")
    fields["isnowxy"][at] = cp.asarray(r["isnow"].astype(np.int32))
    fields["canwat"][at] = zero            # CANLIQ = CANICE = 0 (:1150-1151)
    fields["acsnow"][at] = (fields["acsnow"][at]
                            + cp.asarray(rainbl) * dev("fpice"))
    ponding = (dev("ponding") + dev("ponding1")) + dev("ponding2")
    fields["acsnom"][at] = (fields["acsnom"][at]
                            + (dev("qmelt") * dt32f + ponding))
    fields["pondingxy"][at] = ponding
    put2("emiss", "emissi")
    put2("qsfc", "qsfc")
    # The vegetation carriers take the arm's own constants every step
    # (:1066-1104): undefined_value where WRF writes it, zero where it
    # writes zero.  QRAINXY and PGSXY are pass-through, exactly as in WRF.
    fields["tvxy"][at] = undef
    put2("tgxy", "tg")
    fields["canliqxy"][at] = zero
    fields["canicexy"][at] = zero
    fields["eahxy"][at] = undef
    fields["tahxy"][at] = undef
    put2("cmxy", "cm")
    put2("chxy", "ch")
    fields["fwetxy"][at] = undef
    put2("sneqvoxy", "sneqvo")
    put2("alboldxy", "albold")
    put2("qsnowxy", "qsnow")
    fields["wslakexy"][at] = undef
    fields["waxy"][at] = undef
    fields["lai"][at] = undef              # XLAIXY = PLAI = undefined
    fields["xsaixy"][at] = undef
    put2("taussxy", "tauss")
    fields["z0"][at] = np.float32(0.002)   # :1137 Z0WRF = 0.002
    fields["znt"][at] = np.float32(0.002)
    fields["t2mvxy"][at] = undef
    put2("t2mbxy", "t2m")
    # :1285-1286 run for every column; for the glacier arm Q2MV is the
    # undefined_value and the quotient is what WRF publishes.
    fields["q2mvxy"][at] = undef / (np.float32(1.0) - undef)
    q2b = dev("q2e")
    fields["q2mbxy"][at] = q2b / (np.float32(1.0) - q2b)
    put2("tradxy", "trad")
    fields["fvegxy"][at] = zero
    fields["runsfxy"][at] = runsf_mm
    fields["runsbxy"][at] = runsb_mm
    fields["ecanxy"][at] = zero
    put2("edirxy", "edir")
    fields["etranxy"][at] = zero
    put2("fsaxy", "fsa")
    put2("firaxy", "fira")
    fields["rssunxy"][at] = undef
    fields["rsshaxy"][at] = undef
    # :1305-1314 with RSSUN = undefined <= 0: the conductance is closed.
    fields["rs"][at] = zero
    fields["chvxy"][at] = undef
    put2("chbxy", "ch")
    fields["canhsxy"][at] = zero
    put2("fpicexy", "fpice")
    put2("qsnbotxy", "qsnbot")
    put2("qmeltxy", "qmelt")
    put2("eflxbxy", "eflxb")               # defined replacement (glacier.py)
    put2("qfx", "edir")                    # :1141 QFX = ESOIL
    put2("lh", "fgev")                     # :1142 LH = FGEV
    # :1381-1394 with the glacier column's own HCPCT (defined replacement
    # for WRF's loop-carried read; gpuwm/core/noahmp_glacier.py docstring).
    hcpct = dev("hcpct")
    zsnso = dev("zsnso")
    isnow_out = cp.asarray(r["isnow"].astype(np.int32))
    soil_energy = cp.zeros(count, dtype=cp.float32)
    snow_energy = cp.zeros(count, dtype=cp.float32)
    nsoil = fields["tslb"].shape[0]
    for k in range(-NSNOW + 1, nsoil + 1):
        slot_k = k + NSNOW - 1
        live_k = k >= isnow_out + 1
        top = k == isnow_out + 1
        above = zsnso[:, slot_k - 1] if slot_k > 0 else zero
        thickness = cp.where(top, -zsnso[:, slot_k],
                             above - zsnso[:, slot_k])
        term = ((thickness * hcpct[:, slot_k])
                * (stc[:, slot_k] - np.float32(273.16))) * np.float32(0.001)
        contribution = cp.where(live_k, term, zero)
        if k >= 1:
            soil_energy = soil_energy + contribution
        else:
            snow_energy = snow_energy + contribution
    fields["soilenergy"][at] = soil_energy
    fields["snowenergy"][at] = snow_energy


@dataclass
class _StagedColumn:
    """One land column suspended at a leaf call, waiting for its batch."""

    j: int
    i: int
    col: SnowColumn
    steps: Any
    request: LeafRequest
    result: Any = field(default=None)


def _drive_staged_columns(staged, evaluator) -> None:
    """Run every staged column to completion, one leaf batch at a time.

    Each round collects the leaf every still-running column is paused at,
    evaluates each distinct leaf's calls together, and resumes the columns.
    With eight batched leaves under the pinned identity that is nine rounds:
    the SFLX prefix, THERMOPROP and RADIATION are on every column's path and
    reach every column together, then a bare column runs one leaf ahead of a
    vegetated one for the rest of the column because it never pauses at
    VEGE_FLUX.  Each round issues at most two batches -- one per distinct leaf
    the active columns are waiting on.  The loop is written generally because
    the number of device leaves is what this seam exists to grow.
    """
    active = list(range(len(staged)))
    rounds = 0
    while active:
        rounds += 1
        if rounds > _MAX_LEAF_ROUNDS:
            raise RuntimeError(
                f"a Noah-MP column yielded more than {_MAX_LEAF_ROUNDS} leaf "
                "requests; the staging loop would not terminate")
        buckets: dict[str, list[int]] = {}
        for index in active:
            buckets.setdefault(staged[index].request.leaf, []).append(index)
        answers: dict[int, Any] = {}
        for leaf, indices in buckets.items():
            calls = [(staged[k].request.args, staged[k].request.kwargs)
                     for k in indices]
            values = evaluator(leaf, calls)
            if len(values) != len(indices):
                raise RuntimeError(
                    f"the {leaf} batch returned {len(values)} results for "
                    f"{len(indices)} calls")
            answers.update(zip(indices, values))
        still_running = []
        for index in active:
            column = staged[index]
            try:
                column.request = column.steps.send(answers[index])
            except StopIteration as stop:
                column.result = stop.value
                column.request = None
            else:
                still_running.append(index)
        active = still_running


#: A column pauses once per device leaf on its path.  Eight leaves are batched
#: today and the deepest schedule takes nine rounds; the bound is deliberately
#: loose and exists only so a malformed generator cannot spin forever inside a
#: forecast.
_MAX_LEAF_ROUNDS = 32

#: WRF fixes the snow stack at three layers (``module_sf_noahmpdrv.F:628``).
NSNOW = 3

#: The only soil-layer count this port is validated at.  Every Noah-MP oracle
#: fixture in the tree is four-layer, and ``LAND_SURFACE_SOIL_LAYERS[4]``
#: already refuses anything else at configuration time; this is the runner's
#: own second line.
NSOIL = 4

#: ``XICE_THRESHOLD``, WRF Registry default 0.5.  This is the DEFAULT, not
#: a pin: :class:`NoahmpRuntimeParameters` accepts ``xice_threshold=`` so a
#: caller whose ice analysis is fractional (the MPAS seam's collaborator
#: runs 0.02) classifies sea ice where their data says it is.  Every
#: consumer -- the two ``noahmp_lsm_step`` bodies, the glacier guard and
#: the post-LSM T2/Q2 ownership -- reads ``params.xice_threshold``.
XICE_THRESHOLD = 0.5

#: ``undefined_value``, ``module_sf_noahmpdrv.F:629``.  Reaches the column as
#: ``QC``, which NOAHMP_SFLX forwards to ENERGY and nothing reads under
#: ``opt_sfc=1``.
UNDEFINED_VALUE = -1.0e36

#: ``FOLN``, :1031 -- "for now, set to nitrogen saturation".
FOLN = 1.0

#: Carried per-column scalars, in WRF's Registry spelling lower-cased so a
#: restart manifest and a wrfout carry the identifiers WRF writes.  Every one
#: is read by :func:`gpuwm.core.noahmp_sflx.sflx` and written back by it; the
#: six carbon pools, ``zwtxy``/``wtxy``/``smcwtdxy``/``deeprechxy``/
#: ``rechxy`` and ``grainxy``/``gddxy`` are deliberately absent, because the
#: pinned identity makes them provably pass-through and ``sflx`` does not
#: take them at all.
NOAHMP_STATE_2D = (
    "tvxy", "tgxy", "canicexy", "canliqxy", "eahxy", "tahxy",
    "cmxy", "chxy", "fwetxy", "sneqvoxy", "alboldxy",
    "qsnowxy", "qrainxy", "wslakexy", "waxy", "xsaixy", "taussxy",
)

#: Carried integer per-column state.  ``pgsxy`` is Registry INOUT and is
#: pass-through under ``opt_crop=0``; it is carried because the column solver
#: takes it and refuses a nonzero crop stage.
NOAHMP_STATE_INT_2D = ("isnowxy", "pgsxy")

#: Carried snow-stack state, ``(NSNOW, ny, nx)``, WRF's ``-NSNOW+1:0``.
NOAHMP_STATE_SNOW_3D = ("tsnoxy", "snicexy", "snliqxy")

#: Carried snow+soil interface depths, ``(NSNOW + NSOIL, ny, nx)``.
NOAHMP_STATE_SNOWSOIL_3D = ("zsnsoxy",)

#: The Noah-MP-only diagnostics this runner publishes.  NOAHMP_SFLX returns
#: about 120 outputs; the ones absent here are dropped rather than stored,
#: which is a restriction and is published as one.
NOAHMP_DIAGNOSTICS_2D = (
    "t2mvxy", "t2mbxy", "q2mvxy", "q2mbxy", "tradxy",
    "fsaxy", "firaxy", "ecanxy", "edirxy", "etranxy",
    "runsfxy", "runsbxy", "fvegxy", "rssunxy", "rsshaxy",
    "chvxy", "chbxy", "fpicexy", "qsnbotxy", "qmeltxy", "pondingxy",
    "canhsxy", "eflxbxy", "soilenergy", "snowenergy", "rs",
)

#: Restrictions the option-identity schema has no field for.  Publishing
#: these is not documentation: it is the difference between a user who knows
#: the remaining runtime restrictions a user must see before a forecast.
#: Each entry is (name, what gpuwm does, why it differs).
NOAHMP_RUNTIME_RESTRICTIONS: tuple[tuple[str, str, str], ...] = (
    (
        "glacier_columns",
        "an active land column whose VEGTYP equals ISICE_TABLE dispatches "
        "to the ported NOAHMP_GLACIER column (gpuwm/core/noahmp_glacier.py "
        "on the host, gpuwm/core/kernels/noahmp_glacier.cu on the device), "
        "never to NOAHMP_SFLX; guard_noahmp_glacier_columns refuses only "
        "when the path is disabled (glacier_path=False)",
        "module_sf_noahmpdrv.F:1045-1150 routes it to NOAHMP_GLACIER "
        "(module_sf_noahmp_glacier.F, 3,080 lines, sha256 bf94f352...), "
        "transcribed at the pinned identity opt_alb=2 opt_snf=1 opt_tbot=2 "
        "opt_stc=1 opt_gla=1.  Two defined replacements for WRF's "
        "undefined reads are documented in that module's docstring "
        "(HCPCT/EFLXB).",
    ),
    (
        "sea_ice_columns",
        "XICE >= xice_threshold columns take WRF's skip (SH2O=1, "
        "XLAI=0.01) and are otherwise untouched",
        "This is what WRF does (:715-723).  It is listed because the skip "
        "means no sea-ice surface energy balance exists in a Noah-MP run: "
        "TSK/HFX/QFX over sea ice keep whatever the surface layer left.",
    ),
    (
        "xice_threshold",
        f"defaults to WRF's Registry {XICE_THRESHOLD}; configurable per "
        "run through NoahmpRuntimeParameters(xice_threshold=...) and the "
        "seams that construct it (initialize_physics, "
        "run_mpas_column_batch)",
        "WRF's XICE_THRESHOLD is a namelist value.  A fractional sea-ice "
        "analysis (the MPAS collaborator's runs at 0.02) classifies "
        "columns the Registry default would hand to the land model; the "
        "chosen value is part of the restart identity and the "
        "classification census reports how many columns each class took.",
    ),
    (
        "diagnostic_inventory",
        "26 of NOAHMP_SFLX's ~120 outputs are stored "
        "(NOAHMP_DIAGNOSTICS_2D)",
        "The rest are computed and discarded.  A consumer that needs "
        "APAR/PSN/SAV/SAG, the eight component fluxes, or the twelve "
        "canopy water-budget terms must add them to that tuple; they are "
        "not silently zero-filled anywhere.",
    ),
    (
        "column_solver_location",
        "the whole column runs on the device: noahmp_column_slab's "
        "orchestration answers every land column with no Python per column, "
        "in chunks of SLAB_COLUMN_CHUNK",
        "The assembled slab is bitwise (max ULP 0) against the scalar "
        "column: on twelve heterogeneous fixture columns field by field, at "
        "a full 65,536-column chunk, and end to end at 360,000 columns "
        "against the staged path with the same device leaves -- 83 written "
        "fields, zero differing bits -- plus the six-step carried-state "
        "digests on a bare and a snowpack domain.  Measured twice on one "
        "RTX 5090, a 360,000-land-column call costs 0.202-0.227 s through "
        "the slab path against 166-206 s through the per-column staged path "
        "in the same process; at dt=1.667 s and bldt=0 that is 7.3-8.2 wall "
        "seconds per simulated minute, against 4,982-5,977 before the "
        "orchestration.  Absolute seconds are a property of the machine and "
        "this box varies up to 30% between harnesses and hours.  The staged "
        "per-column path remains the paired second implementation "
        "(GPUWM_NOAHMP_STAGED_COLUMNS=1), and GPUWM_NOAHMP_HOST_LEAVES=1 "
        "still selects the CPython leaves on it.",
    ),
    (
        "soil_layers",
        "four only",
        "Every Noah-MP oracle fixture in the tree is four-layer.  The "
        "runner refuses any other count rather than extrapolating the "
        "parameter transfer to layers no fixture covers.",
    ),
    (
        "qsfc_units",
        "QSFC is passed to and from the column unconverted",
        "module_sf_noahmpdrv.F:238 documents QSFC as specific humidity and "
        ":808/:1245 pass it straight through, while WRF's surface layer "
        "fills the same Registry field with a mixing ratio.  That "
        "inconsistency is WRF's, it is fully defined, and gpuwm reproduces "
        "it rather than inserting a conversion WRF does not have.",
    ),
    (
        "accumulator_reset",
        "the ten ACC_* carriers are zero on entry to every call",
        "soiltstep=0 makes soil_update_steps=1, and :649-660 zeroes all ten "
        "before the column loop, so no cross-step accumulator exists to "
        "carry.  gpuwm therefore does not allocate them.  Admitting "
        "soiltstep>0 means allocating them AND carrying them through "
        "restart.",
    ),
)


class NoahmpGlacierColumnError(NotImplementedError):
    """A glacier column exists but the ported glacier path is disabled."""


def classify_noahmp_surface(xland, xice, vegtyp, *, xice_threshold, isice):
    """The driver's column partition (``module_sf_noahmpdrv.F:715-725``,
    ``:1045``) as one pure function over numpy or cupy arrays.

    ``sea_ice`` is decided FIRST and by ``XICE`` alone -- a column at or
    above the threshold takes WRF's sea-ice skip whatever XLAND says.
    ``open_water`` is what remains of ``XLAND >= 1.5``.  ``land`` is
    everything else, and ``glacier`` is its ``VEGTYP == ISICE_TABLE``
    subset, which dispatches to NOAHMP_GLACIER rather than NOAHMP_SFLX.
    ``sflx_land`` is the complement that runs the ordinary column.

    This is the seam the native-XLAND contract hangs on: a category-15
    fractional-ice column carrying native ``xland = 1`` with
    ``xice < xice_threshold`` classifies ``land`` + ``glacier`` here (the
    ice-surface physics), while a derived ``xland = 2`` for the same
    column would classify it ``open_water`` and run no surface at all.
    """
    thr = np.float32(xice_threshold)
    sea_ice = xice >= thr
    open_water = ((xland - np.float32(1.5)) >= np.float32(0.0)) & ~sea_ice
    land = ~(sea_ice | open_water)
    glacier = land & (vegtyp == int(isice))
    return {"sea_ice": sea_ice, "open_water": open_water, "land": land,
            "glacier": glacier, "sflx_land": land & ~glacier}


def guard_noahmp_glacier_columns(fields, params: "NoahmpRuntimeParameters"):
    """Admit or refuse Noah-MP glacier columns before state can advance.

    WRF v4.6.1 ``module_sf_noahmpdrv.F:1045`` routes these columns to
    ``NOAHMP_GLACIER``, which IS ported (:mod:`gpuwm.core.noahmp_glacier`)
    and dispatched by both ``noahmp_lsm_step`` bodies.  The refusal
    survives as the backstop for the one state it still owns: glacier
    columns present while ``params.glacier_path`` is False (a test or an
    operator disabling the path deliberately).  Running ``NOAHMP_SFLX``
    on them was never an option in either state.
    """
    import cupy as cp

    xice = cp.asarray(fields["xice"])
    xland = cp.asarray(fields["xland"])
    vegtyp = cp.asarray(fields["ivgtyp"])
    masks = classify_noahmp_surface(
        xland, xice, vegtyp, xice_threshold=params.xice_threshold,
        isice=int(params.land_use.isice))
    offending = masks["glacier"]
    if not bool(cp.any(offending)):
        return
    if params.glacier_path:
        return
    flat = int(cp.argmax(offending.reshape(-1)).item())
    j, i = (int(value) for value in np.unravel_index(
        flat, tuple(int(value) for value in offending.shape)))
    category = int(vegtyp[j, i].item())
    raise NoahmpGlacierColumnError(
        "post-static Noah-MP glacier guard: first offending active land "
        f"cell is (j={j}, i={i}), VEGTYP={category}=ISICE_TABLE. "
        "The NOAHMP_GLACIER path is disabled on these parameters "
        "(glacier_path=False) and NOAHMP_SFLX is not a substitute for it.")


class NoahmpRuntimeParameters:
    """The Noah-MP parameter bundle plus its per-column transfer cache.

    Deliberately not a dataclass.  ``gpuwm/io/restart.py``'s
    ``_packed_parameters_identity`` walks a dataclass field by field, and this
    object's interesting content is three parsed tables and a memo dictionary
    -- neither of which is a restart identity.  What identifies the run is the
    table bytes and the option identity, which :meth:`restart_identity`
    returns as strict JSON.
    """

    def __init__(self, bundle=None, *,
                 dataset_identifier: str = DEFAULT_VEGETATION_DATASET,
                 xice_threshold: float = XICE_THRESHOLD,
                 glacier_path: bool = True):
        # Keyword-only with no **kwargs: a misspelled keyword refuses with
        # its name (house convention), never lands as an ignored attribute.
        threshold = float(xice_threshold)
        if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"xice_threshold must be a finite fraction in [0, 1], got "
                f"{xice_threshold!r} (WRF Registry default "
                f"{XICE_THRESHOLD})")
        #: The sea-ice classification threshold every consumer reads.
        #: WRF-compat default 0.5; a fractional-ice caller (the MPAS seam's
        #: collaborator runs 0.02) sets its own.
        self.xice_threshold = threshold
        #: The ported NOAHMP_GLACIER dispatch, default ON.  False makes
        #: guard_noahmp_glacier_columns refuse glacier columns again --
        #: the red-on-revert state, never a shipping configuration.
        self.glacier_path = bool(glacier_path)
        self.bundle = load_noahmp_parameters() if bundle is None else bundle
        self.dataset_identifier = str(dataset_identifier)
        categories, veg = self.bundle.vegetation_groups(
            self.dataset_identifier)
        self._categories = categories
        self._veg = veg
        self.land_use = LandUseIdentity(
            isice=int(veg.scalar("ISICE")),
            isurban=int(veg.scalar("ISURBAN")),
            iswater=int(veg.scalar("ISWATER")),
            isbarren=int(veg.scalar("ISBARREN")),
            natural=int(veg.scalar("NATURAL")),
            lcz=tuple(int(veg.scalar(f"LCZ_{n}")) for n in range(1, 12)),
        )
        self._transfer_cache: dict[tuple, object] = {}
        self._handle_cache: dict[tuple, SflxParameters] = {}

    # -- TRANSFER_MP_PARAMETERS, memoised on its own argument tuple ---------
    def transferred(self, *, vegtyp: int, soiltyp: tuple[int, ...],
                    slopetype: int, soilcolor: int):
        key = (int(vegtyp), tuple(int(v) for v in soiltyp),
               int(slopetype), int(soilcolor))
        cached = self._transfer_cache.get(key)
        if cached is None:
            cached = transfer_mp_parameters(
                self.bundle, vegtype=key[0], soiltype=key[1],
                slopetype=key[2], soilcolor=key[3], croptype=0,
                dataset_identifier=self.dataset_identifier)
            self._transfer_cache[key] = cached
        return cached

    def handle(self, *, vegtyp: int, soiltyp: tuple[int, ...],
               slopetype: int, soilcolor: int, nsoil: int) -> SflxParameters:
        """The ``SflxParameters`` for one column's parameter identity.

        Memoised on the same tuple ``TRANSFER_MP_PARAMETERS`` is keyed by, so
        a 250,000-column nest builds one handle per distinct land-use/soil
        pair rather than 250,000 handles.  This is only sound because
        ``sflx`` treats the handle as read-only; ``tests/test_noahmp_runtime.py``
        pins that by hashing the handle's fields across a step.
        """
        key = (int(vegtyp), tuple(int(v) for v in soiltyp),
               int(slopetype), int(soilcolor), int(nsoil))
        cached = self._handle_cache.get(key)
        if cached is None:
            cached = SflxParameters.from_transferred(
                self.transferred(vegtyp=key[0], soiltyp=key[1],
                                 slopetype=key[2], soilcolor=key[3]),
                nsoil=key[4])
            self._handle_cache[key] = cached
        return cached

    def slab_row(self, *, vegtyp: int, soiltyp: tuple[int, ...],
                 slopetype: int, soilcolor: int, nsoil: int) -> dict:
        """One flattened parameter row for the slab orchestration.

        Memoised on the same identity tuple as :meth:`handle`, so a
        360,000-column nest builds one row per distinct land-use/soil pair.
        The row is a plain dict of numpy values and is treated as read-only
        by :func:`gpuwm.core.noahmp_column_slab.parameter_fields`.
        """
        from gpuwm.core.noahmp_column_slab import parameter_row

        key = (int(vegtyp), tuple(int(v) for v in soiltyp),
               int(slopetype), int(soilcolor), int(nsoil))
        cache = self.__dict__.setdefault("_slab_row_cache", {})
        row = cache.get(key)
        if row is None:
            row = parameter_row(self.handle(
                vegtyp=key[0], soiltyp=key[1], slopetype=key[2],
                soilcolor=key[3], nsoil=key[4]))
            cache[key] = row
        return row

    def table(self, group: str, name: str):
        return self.bundle.mptable[group].scalar(name)

    @property
    def n_vegetation_classes(self) -> int:
        """How many land-use classes the ACTIVE table covers.

        Read from the loaded table, never hardcoded: MODIFIED_IGBP_MODIS_NOAH
        declares 20 and USGS declares 24, and a guard written for one of them
        is wrong for the other.
        """
        return int(self._categories.scalar("NVEG"))

    def veg_value(self, name: str, vegtyp: int) -> float:
        """``<NAME>_TABLE(VEGTYP)`` out of the active vegetation dataset.

        Only the cold start needs this: ``TRANSFER_MP_PARAMETERS`` already
        indexes every parameter the column solver reads, but ``NOAHMP_INIT``
        reads ``SLA_TABLE`` directly (:2187) before any transfer has happened.

        A class the table does not cover is refused BY NAME. This used to be
        a bare ``entry[vegtyp - 1]``, and both of its failure modes were
        reached by one real run: a 15 km MPAS mesh over the Great Lakes,
        built from ``modis_landuse_20class_30s_with_lakes`` whose lake class
        is 21, put ``ivgtyp = 21`` into 115 cells and the cold start died on
        ``IndexError: tuple index out of range``. And a ``vegtyp`` of 0 does
        not raise at all -- Python reads ``entry[-1]``, the LAST class -- so
        the run completes with a lake given the parameters of whatever class
        happens to sit at the end of the table. The silent one is worse.
        """
        index = int(vegtyp)
        limit = self.n_vegetation_classes
        if index < 1 or index > limit:
            raise ValueError(
                f"land-use class {index} is outside the {limit} classes "
                f"{self.dataset_identifier} covers, so {name.upper()}_TABLE "
                f"has no row for it. Reading one anyway gives this column "
                f"another class's vegetation parameters without saying so. "
                f"Land-use classes must be 1..{limit}; a static built from an "
                f"archive with more classes than the table declares (a "
                f"20-class MODIS table against a with-lakes archive whose "
                f"lake class is 21, for instance) has to be recoded before "
                f"the run, not silently indexed past the end."
            )
        entry = self._veg.values[name.upper()]
        return float(entry[index - 1] if len(entry) > 1 else entry[0])

    def soil_value(self, name: str, soiltyp: int) -> float:
        """``<NAME>_TABLE(SOILTYP)`` out of the pinned SOILPARM section."""
        return float(self.bundle.soil["STAS"].column(name)[int(soiltyp) - 1])

    def restart_identity(self) -> dict:
        """Strict-JSON identity: the table bytes plus the dataset choice."""
        receipt = self.bundle.receipt
        tables = receipt.get("tables", receipt)
        payload = {}
        if isinstance(tables, Mapping):
            for name, entry in sorted(tables.items()):
                if isinstance(entry, Mapping):
                    payload[str(name)] = {
                        "bytes": int(entry.get("bytes", 0)),
                        "sha256": str(entry.get("sha256", "")),
                    }
        return {
            "algorithm": "noahmp-lsm-wrf-v4.6.1-v1",
            "wrf_source": "phys/module_sf_noahmpdrv.F:noahmplsm + "
                          "phys/module_sf_noahmplsm.F:NOAHMP_SFLX",
            "dataset_identifier": self.dataset_identifier,
            "nsnow": int(NSNOW),
            "xice_threshold": float(self.xice_threshold),
            "column_solver": "host-fp32+device-vegeflux-v1",
            # The glacier dispatch is part of the trajectory identity: a
            # restart written with it enabled must not resume with it off,
            # and the threshold above decides which columns it owns.
            "glacier": {
                "wrf_source": "phys/module_sf_noahmp_glacier.F:"
                              "NOAHMP_GLACIER",
                "sha256": "bf94f3522c3b9c2c9cfbb34fa7e485ff58519106db4345"
                          "20968793409a520579",
                "enabled": bool(self.glacier_path),
            },
            "tables": payload,
        }


class NoahmpSolarGeometry:
    """The domain's own solar geometry, which the LSM seam does not carry.

    ``noahmplsm`` reads ``COSZIN``, ``XLAT``, ``JULIAN`` and ``YR``.  gpuwm's
    surface seam has none of them -- the radiation callable owns the clock and
    the grid geometry -- so a Noah-MP run must bind this explicitly.  It is
    required rather than defaulted: a silent ``lat=0, julian=0`` would run
    every column on the equator at New Year and produce a plausible forecast.
    """

    def __init__(self, start_time: datetime, latitude_deg, longitude_deg):
        if not isinstance(start_time, datetime):
            raise TypeError("Noah-MP start_time must be a datetime")
        self.start_time = start_time
        self.latitude_deg = np.ascontiguousarray(
            np.asarray(latitude_deg, dtype=np.float32))
        self.longitude_deg = np.ascontiguousarray(
            np.asarray(longitude_deg, dtype=np.float32))
        if self.latitude_deg.shape != self.longitude_deg.shape:
            raise ValueError(
                "Noah-MP latitude/longitude shapes must match")
        if not np.isfinite(self.latitude_deg).all():
            raise ValueError("Noah-MP latitude must be finite")

    def valid_time(self, elapsed_seconds: float) -> datetime:
        return self.start_time + timedelta(seconds=float(elapsed_seconds))

    def cosz(self, elapsed_seconds: float) -> np.ndarray:
        """COSZ per column, WRF's ``calc_coszen`` with no interval offset."""
        from gpuwm.core.dudhia import wrf_solar_geometry

        mu, _solcon = wrf_solar_geometry(
            self.valid_time(elapsed_seconds),
            self.latitude_deg, self.longitude_deg, hour_offset_seconds=0.0)
        return np.asarray(mu, dtype=np.float32)

    def julian(self, elapsed_seconds: float) -> float:
        """WRF's ``JULIAN_IN``: day-of-year minus one plus the day fraction."""
        valid = self.valid_time(elapsed_seconds)
        hour = (valid.hour + valid.minute / 60.0 + valid.second / 3600.0
                + valid.microsecond / 3.6e9)
        return float(valid.timetuple().tm_yday - 1.0 + hour / 24.0)

    def year(self, elapsed_seconds: float) -> int:
        return int(self.valid_time(elapsed_seconds).year)

    @property
    def restart_identity(self) -> dict:
        return {
            "start_time": self.start_time.isoformat(),
            "latitude_deg_sha256": _array_digest(self.latitude_deg),
            "longitude_deg_sha256": _array_digest(self.longitude_deg),
        }


def _array_digest(array) -> str:
    import hashlib

    contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def year_length(year: int) -> int:
    """``YEARLEN``, module_sf_noahmpdrv.F:675-684 -- Gregorian, spelled out."""
    year = int(year)
    if year % 4 != 0:
        return 365
    if year % 100 != 0:
        return 366
    return 366 if year % 400 == 0 else 365


def zsoil_from_dzs(dzs) -> np.ndarray:
    """``ZSOIL``, :686-689: depth to each soil interface, negative downward."""
    dz = np.asarray(dzs, dtype=np.float32)
    zsoil = np.empty(dz.size, dtype=np.float32)
    zsoil[0] = f32(-dz[0])
    for k in range(1, dz.size):
        zsoil[k] = f32(f32(-dz[k]) + zsoil[k - 1])
    return zsoil


#: Every field the cold start reads or writes, pulled to the host once.
#: Shared by both arms so a carrier cannot be present in one and absent from
#: the other -- which would be a parity failure that only showed up as a
#: forecast seeded from an uninitialised array.
_COLD_START_SLAB = ("snow", "snowh", "canwat", "tslb", "smois", "sh2o",
                    "tsk", "xice", "ivgtyp", "isltyp", "lai",
                    *NOAHMP_STATE_2D, *NOAHMP_STATE_INT_2D,
                    *NOAHMP_STATE_SNOW_3D, *NOAHMP_STATE_SNOWSOIL_3D)

#: Every carrier ``NOAHMP_INIT`` leaves changed.  The rest of the slab above
#: is input (``tsk``, ``xice``, ``ivgtyp``, ``isltyp``) or Registry-zero state
#: the routine has no argument for (``taussxy``, ``pgsxy``, 2114-2153's
#: comment).  ``tests/test_noahmp_cold_start_device.py`` compares exactly this
#: set between the two arms and asserts the set itself, so a carrier added to
#: one arm and not the other cannot slip past the parity gate.
COLD_START_WRITES = (
    "snow", "snowh", "canwat", "tslb", "smois", "sh2o", "lai",
    "isnowxy", "zsnsoxy", "tsnoxy", "snicexy", "snliqxy",
    "tvxy", "tgxy", "canicexy", "canliqxy", "eahxy", "tahxy",
    "cmxy", "chxy", "fwetxy", "sneqvoxy", "alboldxy",
    "qsnowxy", "qrainxy", "wslakexy", "waxy", "xsaixy",
)

#: Setting this to ``1`` forces the paired numpy authority for the one-time
#: cold start -- the same doctrine as :data:`HOST_LEAVES_ENV` and
#: :data:`STAGED_COLUMNS_ENV`, for the routine that runs before the first
#: step rather than inside it.  A parity run sets it to produce the scalar
#: reference the device arm is compared against.
HOST_COLD_START_ENV = "GPUWM_NOAHMP_HOST_COLD_START"


def _resolve_cold_start():
    """Which arm answers the cold start on this call.

    ``HOST_LEAVES_ENV`` is honoured as well as the dedicated switch: a process
    that asked for the CPython authority for the whole column solver and still
    got a device cold start would be publishing host step numbers on top of a
    state the device produced.
    """
    if os.environ.get(HOST_COLD_START_ENV, "") == "1":
        return cold_start_on_host
    if os.environ.get(HOST_LEAVES_ENV, "") == "1":
        return cold_start_on_host
    return COLD_START_EVALUATOR


def noahmp_cold_start(fields, *, params: NoahmpRuntimeParameters,
                      dzs) -> None:
    """``NOAHMP_INIT`` + ``SNOW_INIT`` over the whole slab.

    Runs once, at driver construction, exactly where WRF runs it
    (``phys/module_physics_init.F`` calls NOAHMP_INIT before the first step).
    Every array it writes is a Noah-MP carrier that
    :func:`gpuwm.core.physics.initialize_physics` has just allocated at zero,
    which is *not* a usable Noah-MP state: ``TV = TG = 0 K`` walks straight
    into a negative saturation vapour pressure.

    ``FNDSOILW`` is not an argument to the ported ``NOAHMP_INIT`` because WRF
    declares it and never reads it.  ``FNDSNOWH`` is passed true: gpuwm's
    ingest always carries a physical snow depth, and the false branch
    (``SNOWH = SNOW * 0.005``) would silently replace it.

    The columns are answered by :func:`cold_start_on_device` -- the CUDA
    driver kernel ``gpuwm/core/kernels/noahmp_driver.cu``, which
    ``gpuwm/core/preflight.py`` already prices into every scheme-4 compile.
    :func:`cold_start_on_host` is the paired scalar authority behind
    ``GPUWM_NOAHMP_HOST_COLD_START=1``; the two are byte-for-byte equal and
    ``tests/test_noahmp_cold_start_device.py`` is what says so.
    """
    import cupy as cp

    slab = {name: np.ascontiguousarray(cp.asnumpy(fields[name]))
            for name in _COLD_START_SLAB}
    nsoil = slab["tslb"].shape[0]
    if nsoil != NSOIL:
        raise ValueError(
            f"Noah-MP cold start is four-layer only, got nsoil={nsoil}")

    _resolve_cold_start()(slab, params=params, dzs=dzs)

    for name, array in slab.items():
        fields[name][...] = cp.asarray(array)


def cold_start_on_device(slab, *, params: NoahmpRuntimeParameters,
                         dzs) -> None:
    """``NOAHMP_INIT`` + ``SNOW_INIT`` for the whole slab, on the GPU.

    One thread per column through ``noahmp_driver_noahmp_init``, whose
    bitwise identity with the unmodified WRF v4.6.1 driver over
    ``gpuwm/data/noahmp/oracle/noahmp-driver.csv`` is the gate in
    ``tests/test_noahmp_driver_cuda.py`` (``max_ulp == 0``, no tolerance
    anywhere).  This is the marshalling that was missing: the kernel and its
    wrappers were finished and verified with no production caller, so every
    Noah-MP run compiled ``noahmp_driver.cu`` and then cold-started 360,000
    columns through the CPython transcription at 18.4-38.8 us each.

    Chunked at :data:`SLAB_COLUMN_CHUNK`, the same explicit bound the
    per-timestep slab packers carry: the staging arrays are
    ``chunk x stride`` and never ``nx*ny x stride``, so the one-time cold
    start cannot decide a nest's peak VRAM.
    """
    import cupy as cp

    from gpuwm.core.noahmp_driver_gpu import (NI_IN, NI_IN_STRIDE, NI_IX,
                                              NI_IX_STRIDE, NI_OUT,
                                              run_noahmp_init)

    ny, nx = slab["tsk"].shape
    nsoil = slab["tslb"].shape[0]
    ncol = ny * nx
    identity = params.land_use
    dz = np.asarray(dzs, dtype=np.float32)

    for name in _COLD_START_SLAB:
        if not slab[name].flags["C_CONTIGUOUS"]:
            # The unpack writes through ``reshape(ncol)``, which is a view
            # only while the carrier is contiguous.  On a strided one it
            # would be a copy, every kernel answer would land in a temporary
            # and be discarded, and the forecast would start from the zeros
            # initialize_physics allocated -- TV = TG = 0 K, silently.
            raise ValueError(
                f"Noah-MP cold start: {name} is not C-contiguous")

    vegtyp = np.ascontiguousarray(slab["ivgtyp"]).reshape(ncol).astype(
        np.int32)
    soiltyp = np.ascontiguousarray(slab["isltyp"]).reshape(ncol).astype(
        np.int32)

    # 2047-2057: WRF aborts.  A zero ISLTYP is what an ingest path that forgot
    # the soil category produces, and running with it would index the table at
    # -1.  Scanned before the launch and reported at the first offending
    # column in the host arm's own row-major order, so the two arms refuse the
    # same grid with the same words.
    offending = np.flatnonzero(soiltyp < 1)
    if offending.size:
        j, i = divmod(int(offending[0]), nx)
        raise ValueError(
            f"Noah-MP cold start: ISLTYP={int(soiltyp[offending[0]])} at "
            f"(j={j}, i={i})")

    # BEXP/SMCMAX/PSISAT and SLA are ``<NAME>_TABLE(index)`` reads, one per
    # distinct category rather than one per column -- a 360,000-column nest
    # carries a handful of land-use and soil classes.
    bexp = np.empty(ncol, dtype=np.float32)
    smcmax = np.empty(ncol, dtype=np.float32)
    psisat = np.empty(ncol, dtype=np.float32)
    for category in np.unique(soiltyp):
        rows = soiltyp == category
        bexp[rows] = params.soil_value("BB", int(category))
        smcmax[rows] = params.soil_value("MAXSMC", int(category))
        psisat[rows] = params.soil_value("SATPSI", int(category))
    sla = np.empty(ncol, dtype=np.float32)
    for category in np.unique(vegtyp):
        sla[vegtyp == category] = params.veg_value("SLA", int(category))

    def flat(name, layer=None):
        """One carrier as a flat ``ncol`` view, in the grid's row-major order.

        The same order ``vegtyp`` and ``soiltyp`` above were flattened in and
        the same order the host arm's ``j``/``i`` loop visits, which is what
        makes the ISLTYP refusal above name the same column WRF would.
        """
        array = slab[name] if layer is None else slab[name][layer]
        return np.ascontiguousarray(array).reshape(ncol)

    for start in range(0, ncol, SLAB_COLUMN_CHUNK):
        stop = min(start + SLAB_COLUMN_CHUNK, ncol)
        span = slice(start, stop)
        n = stop - start

        x = np.zeros((n, NI_IN_STRIDE), dtype=np.float32)
        ix = np.zeros((n, NI_IX_STRIDE), dtype=np.int32)

        ix[:, NI_IX["nsoil"]] = nsoil
        ix[:, NI_IX["nsnow"]] = NSNOW
        # FNDSNOWH true, cropcat 0 and sf_urban_physics 0 are the same three
        # constants the host arm passes; the pinned option identity
        # (iopt_run=3, iopt_crop=0, iopt_irr=0, iopt_irrm=0) is what
        # noahmp_driver.cu is compiled for and has no slot to vary.
        ix[:, NI_IX["fndsnowh"]] = 1
        ix[:, NI_IX["vegtyp"]] = vegtyp[span]
        ix[:, NI_IX["cropcat"]] = 0
        ix[:, NI_IX["sf_urban_physics"]] = 0
        ix[:, NI_IX["isice"]] = identity.isice
        ix[:, NI_IX["isurban"]] = identity.isurban
        ix[:, NI_IX["iswater"]] = identity.iswater
        ix[:, NI_IX["isbarren"]] = identity.isbarren
        for slot, category in enumerate(identity.lcz):
            ix[:, NI_IX["lcz"] + slot] = category

        x[:, NI_IN["xice"]] = flat("xice")[span]
        x[:, NI_IN["tsk"]] = flat("tsk")[span]
        x[:, NI_IN["lai"]] = flat("lai")[span]
        x[:, NI_IN["bexp"]] = bexp[span]
        x[:, NI_IN["smcmax"]] = smcmax[span]
        x[:, NI_IN["psisat"]] = psisat[span]
        x[:, NI_IN["sla"]] = sla[span]
        # SLA_TABLE(NATURAL_TABLE) is read only at 2185, on an urban column
        # with sf_urban_physics > 0.  With sf_urban_physics = 0 every urban
        # and LCZ column takes the bare branch at 2164-2178, so the slot is
        # never read; NaN makes a port that read it anyway fail loudly rather
        # than agree with whatever was in the buffer.
        x[:, NI_IN["sla_natural"]] = np.float32("nan")
        x[:, NI_IN["snow"]] = flat("snow")[span]
        x[:, NI_IN["snowh"]] = flat("snowh")[span]
        for k in range(nsoil):
            x[:, NI_IN["dzs"] + k] = dz[k]
            x[:, NI_IN["tslb"] + k] = flat("tslb", k)[span]
            x[:, NI_IN["smois"] + k] = flat("smois", k)[span]
        for k in range(NSNOW + nsoil):
            x[:, NI_IN["zsnsoxy"] + k] = flat("zsnsoxy", k)[span]
        for name in NOAHMP_STATE_SNOW_3D:
            for k in range(NSNOW):
                x[:, NI_IN[name] + k] = flat(name, k)[span]

        y = cp.asnumpy(run_noahmp_init(x, ix))

        for name, out in (("snow", "snow"), ("snowh", "snowh"),
                          ("canwat", "canwat"), ("lai", "lai"),
                          ("tvxy", "tv"), ("tgxy", "tg"),
                          ("canicexy", "canice"), ("canliqxy", "canliq"),
                          ("eahxy", "eah"), ("tahxy", "tah"),
                          ("cmxy", "cm"), ("chxy", "ch"),
                          ("fwetxy", "fwet"), ("sneqvoxy", "sneqvo"),
                          ("alboldxy", "albold"), ("qsnowxy", "qsnow"),
                          ("qrainxy", "qrain"), ("wslakexy", "wslake"),
                          ("waxy", "wa"), ("xsaixy", "xsai")):
            slab[name].reshape(ncol)[span] = y[:, NI_OUT[out]]
        # ISNOWXY is the integer carrier; the kernel returns it as a float
        # over -3..0, which every integer dtype in the tree holds exactly.
        slab["isnowxy"].reshape(ncol)[span] = y[:, NI_OUT["isnow"]]
        for name, out in (("tslb", "tslb"), ("smois", "smois"),
                          ("sh2o", "sh2o")):
            for k in range(nsoil):
                slab[name][k].reshape(ncol)[span] = y[:, NI_OUT[out] + k]
        for k in range(NSNOW + nsoil):
            slab["zsnsoxy"][k].reshape(ncol)[span] = y[:, NI_OUT["zsnso"] + k]
        for name, out in (("tsnoxy", "tsno"), ("snicexy", "snice"),
                          ("snliqxy", "snliq")):
            for k in range(NSNOW):
                slab[name][k].reshape(ncol)[span] = y[:, NI_OUT[out] + k]


def cold_start_on_host(slab, *, params: NoahmpRuntimeParameters,
                       dzs) -> None:
    """The same cold start, column by column, on the unmodified CPython port.

    This is the paired scalar authority: the routine
    ``tests/test_noahmp_driver.py`` pins against the WRF v4.6.1 oracle and the
    reference every parity run compares the device arm to.  It is the shipped
    path only under ``GPUWM_NOAHMP_HOST_COLD_START=1`` (or the process-wide
    ``GPUWM_NOAHMP_HOST_LEAVES=1``); a default forecast takes
    :func:`cold_start_on_device`.
    """
    ny, nx = slab["tsk"].shape
    nsoil = slab["tslb"].shape[0]
    identity = params.land_use

    for j in range(ny):
        for i in range(nx):
            vegtyp = int(slab["ivgtyp"][j, i])
            soiltyp = int(slab["isltyp"][j, i])
            if soiltyp < 1:
                # 2047-2057: WRF aborts.  A zero ISLTYP is what an ingest
                # path that forgot the soil category produces, and running
                # with it would index the table at -1.
                raise ValueError(
                    f"Noah-MP cold start: ISLTYP={soiltyp} at (j={j}, i={i})")
            cold = noahmp_init_column(
                nsoil=nsoil, dzs=dzs, fndsnowh=True, identity=identity,
                vegtyp=vegtyp, soiltyp=soiltyp,
                xice=slab["xice"][j, i], tsk=slab["tsk"][j, i],
                lai=slab["lai"][j, i],
                bexp=params.soil_value("BB", soiltyp),
                smcmax=params.soil_value("MAXSMC", soiltyp),
                psisat=params.soil_value("SATPSI", soiltyp),
                sla=params.veg_value("SLA", vegtyp),
                snow=slab["snow"][j, i], snowh=slab["snowh"][j, i],
                tslb=slab["tslb"][:, j, i], smois=slab["smois"][:, j, i],
                zsnsoxy=slab["zsnsoxy"][:, j, i],
                tsnoxy=slab["tsnoxy"][:, j, i],
                snicexy=slab["snicexy"][:, j, i],
                snliqxy=slab["snliqxy"][:, j, i],
                cropcat=0, nsnow=NSNOW, iopt_run=3, iopt_crop=0,
                iopt_irr=0, iopt_irrm=0, sf_urban_physics=0)
            slab["snow"][j, i] = cold.snow
            slab["snowh"][j, i] = cold.snowh
            slab["canwat"][j, i] = cold.canwat
            slab["tslb"][:, j, i] = cold.tslb
            slab["smois"][:, j, i] = cold.smois
            slab["sh2o"][:, j, i] = cold.sh2o
            slab["lai"][j, i] = cold.lai
            slab["isnowxy"][j, i] = cold.isnow
            slab["zsnsoxy"][:, j, i] = cold.zsnso
            slab["tsnoxy"][:, j, i] = cold.tsno
            slab["snicexy"][:, j, i] = cold.snice
            slab["snliqxy"][:, j, i] = cold.snliq
            slab["tvxy"][j, i] = cold.tv
            slab["tgxy"][j, i] = cold.tg
            slab["canicexy"][j, i] = cold.canice
            slab["canliqxy"][j, i] = cold.canliq
            slab["eahxy"][j, i] = cold.eah
            slab["tahxy"][j, i] = cold.tah
            slab["cmxy"][j, i] = cold.cm
            slab["chxy"][j, i] = cold.ch
            slab["fwetxy"][j, i] = cold.fwet
            slab["sneqvoxy"][j, i] = cold.sneqvo
            slab["alboldxy"][j, i] = cold.albold
            slab["qsnowxy"][j, i] = cold.qsnow
            slab["qrainxy"][j, i] = cold.qrain
            slab["wslakexy"][j, i] = cold.wslake
            slab["waxy"][j, i] = cold.wa
            slab["xsaixy"][j, i] = cold.xsai
            # TAUSSXY and PGSXY are not NOAHMP_INIT arguments; their WRF
            # Registry cold state is 0, which initialize_physics already set.


#: Which of the two arms the cold start uses.  A module attribute rather than
#: an argument -- the same doctrine as :data:`LEAF_BATCH_EVALUATOR` and
#: :data:`GLACIER_BATCH_EVALUATOR` -- so a parity test can bind the scalar
#: authority for a whole build without threading a flag through ``physics``.
COLD_START_EVALUATOR = cold_start_on_device


def noahmp_lsm_step(
    fields,
    atmosphere: Mapping[str, object],
    *,
    params: NoahmpRuntimeParameters,
    geometry: NoahmpSolarGeometry,
    precipitation: SurfacePrecipitationForcing,
    coszen,
    dt: float,
    dx: float,
    dzs,
    itimestep: int,
    elapsed_seconds: float,
    dveg: int,
    opt_run: int,
    opt_crop: int,
    opt_irr: int,
    opt_tdrn: int,
    opt_soil: int,
) -> dict[str, int]:
    """One ``noahmplsm`` call.  Mutates ``fields`` in place.

    Returns a small census -- how many columns ran, were skipped as water,
    and were skipped as sea ice -- because "the LSM ran" and "the LSM ran on
    any land" are different claims and a test must be able to tell them
    apart.
    """
    import cupy as cp

    if int(itimestep) < 1:
        raise ValueError("noahmplsm itimestep is one-based and starts at 1")
    if int(dveg) != SFLX_DVEG:
        # The column solver transcribes only the DVEG == 4 arm of
        # module_sf_noahmplsm.F:857-870, so a different dveg would silently
        # take the FVEG = SHDMAX branch under another option's name.
        raise ValueError(
            f"dveg={dveg}: gpuwm transcribes the dveg={SFLX_DVEG} arm only")
    if int(opt_soil) != 1:
        raise ValueError(
            f"opt_soil={opt_soil}: only the uniform-column soil-type branch "
            "(module_sf_noahmpdrv.F:737-746, iopt_soil=1) is ported")
    if int(opt_crop) != 0 or int(opt_irr) != 0 or int(opt_tdrn) != 0:
        raise ValueError(
            "opt_crop/opt_irr/opt_tdrn must be 0; their branches are proved "
            "dead under the admitted identity, not transcribed")
    if int(opt_run) == 5:
        raise ValueError(
            "opt_run=5 needs GROUNDWATER/SHALLOWWATERTABLE, which is not "
            "ported")

    # ---- the whole-slab orchestration, unless a caller opted out ----------
    # This is the forecast path.  The per-column staging below remains the
    # paired second implementation: it is what runs when a test binds another
    # evaluator (the host authority, a nudge wrapper) or when
    # GPUWM_NOAHMP_STAGED_COLUMNS=1 asks for it by name, and the six-step
    # digest gate in tests/test_noahmp_runtime.py is what says the two paths
    # are the same function.
    if _slab_path_selected():
        census = _lsm_step_slab(
            fields, atmosphere, params=params, geometry=geometry,
            precipitation=precipitation, coszen=coszen,
            dt=dt, dx=dx, dzs=dzs, itimestep=itimestep,
            elapsed_seconds=elapsed_seconds)
        _noahmp_post_lsm_diagnostics(fields, params=params)
        return census

    # ---- host slab -------------------------------------------------------
    names_2d = (
        "xland", "xice", "ivgtyp", "isltyp", "shdmax", "tmn", "swdown",
        "glw", "rainbl", "sr", "qsfc", "lai", "snow", "snowh", "canwat",
        "tsk", "hfx", "qfx", "lh", "grdflx", "albedo", "snowc", "emiss",
        "z0", "znt", "smstav", "smstot", "sfcrunoff", "udrunoff",
        "acsnow", "acsnom",
        *NOAHMP_STATE_2D, *NOAHMP_STATE_INT_2D, *NOAHMP_DIAGNOSTICS_2D)
    host = {name: np.ascontiguousarray(cp.asnumpy(fields[name]))
            for name in names_2d}
    host_3d = {name: np.ascontiguousarray(cp.asnumpy(fields[name]))
               for name in ("tslb", "smois", "sh2o",
                            *NOAHMP_STATE_SNOW_3D,
                            *NOAHMP_STATE_SNOWSOIL_3D)}

    # ``.astype(np.float32)`` is where the per-column ``f32(temperature[j,i])``
    # went.  It is the same rounding, applied once per field instead of once
    # per column, and it is a no-op on a slab that is already binary32.
    def _forcing(name, level=0):
        return np.ascontiguousarray(
            cp.asnumpy(atmosphere[name][level])).astype(np.float32, copy=False)

    temperature = _forcing("temperature")
    qv = _forcing("qv")
    u = _forcing("u")
    v = _forcing("v")
    dz1 = _forcing("dz")
    p_int0 = _forcing("p_interface", 0)
    p_int1 = _forcing("p_interface", 1)

    ny, nx = host["tsk"].shape
    nsoil = host_3d["tslb"].shape[0]
    if nsoil != NSOIL:
        raise ValueError(f"Noah-MP is four-layer only, got nsoil={nsoil}")

    cosz = np.ascontiguousarray(
        cp.asnumpy(cp.asarray(coszen, dtype=cp.float32)))
    if cosz.shape != (ny, nx):
        raise ValueError(
            "Noah-MP solar geometry must match the domain grid, got "
            f"{cosz.shape} for a {(ny, nx)} slab")
    julian32 = f32(geometry.julian(elapsed_seconds))
    yearlen = year_length(geometry.year(elapsed_seconds))
    latitude = np.deg2rad(geometry.latitude_deg).astype(np.float32)
    zsoil = zsoil_from_dzs(dzs)
    identity = params.land_use
    dt32 = f32(dt)
    dx32 = f32(dx)
    foln32 = f32(FOLN)
    undefined32 = f32(UNDEFINED_VALUE)
    zero32 = f32(0.0)
    co2_fraction = f32(params.table("noahmp_global_parameters", "CO2"))
    o2_fraction = f32(params.table("noahmp_global_parameters", "O2"))

    # ---- :689-711, the ITIMESTEP == 1 ocean / sea-ice soil state ----------
    if int(itimestep) == 1:
        water = (host["xland"] - np.float32(1.5)) >= np.float32(0.0)
        ice_full = host["xice"] == np.float32(1.0)
        host["smstav"][water] = np.float32(1.0)
        host["smstot"][water] = np.float32(1.0)
        for k in range(nsoil):
            host_3d["smois"][k][water] = np.float32(1.0)
            host_3d["tslb"][k][water] = np.float32(273.16)
        seaice = ice_full & ~water
        host["smstav"][seaice] = np.float32(1.0)
        host["smstot"][seaice] = np.float32(1.0)
        for k in range(nsoil):
            host_3d["smois"][k][seaice] = np.float32(1.0)

    # ---- :715-725, the two CYCLE ILOOP skips, over the whole slab --------
    # WRF writes these as two ``CYCLE ILOOP`` guards inside DO J / DO I.  The
    # order matters and is kept: a column with XICE >= the configured
    # threshold takes the sea-ice skip whatever XLAND says.
    sea_ice = host["xice"] >= np.float32(params.xice_threshold)
    open_water = ((host["xland"] - np.float32(1.5)) >= np.float32(0.0)) \
        & ~sea_ice
    is_land = ~(sea_ice | open_water)
    host_3d["sh2o"][:, sea_ice] = np.float32(1.0)
    host["lai"][sea_ice] = np.float32(0.01)
    census = {"land": int(is_land.sum()),
              "water": int(open_water.sum()),
              "sea_ice": int(sea_ice.sum())}

    # ---- :729-796, 2-D to 1-D, evaluated once for the slab ---------------
    # Every one of these was a per-column ``f32(a op b)`` in CPython.  NumPy
    # float32 performs the identical IEEE binary32 operation in the identical
    # rounding mode, so this is a change of loop shape and not of arithmetic;
    # the six-step carried-state gate is what says so.
    dt32f = np.float32(dt32)
    q_ml_2d = qv / (np.float32(1.0) + qv)
    z_ml_2d = np.float32(0.5) * dz1
    p_ml_2d = (p_int1 + p_int0) * np.float32(0.5)
    host_precipitation = SurfacePrecipitationForcing(**{
        name: np.ascontiguousarray(cp.asnumpy(cp.asarray(
            getattr(precipitation, name), dtype=cp.float32)))
        for name in SurfacePrecipitationForcing.__dataclass_fields__
    })
    # :776-786, the branch WRF-ARW takes because surface_driver passes all
    # six MP_* accumulations.
    precip_rates = noahmp_six_precipitation_rates(
        host["rainbl"], host["sr"], host_precipitation, dt32f, arrays=np)
    fvgmax_2d = host["shdmax"] / np.float32(100.0)
    # :1030-1031.  CO2_TABLE / O2_TABLE are the MPTABLE global partial-
    # pressure fractions; these products are the only place the driver
    # multiplies them.
    co2air_2d = np.float32(co2_fraction) * p_ml_2d
    o2air_2d = np.float32(o2_fraction) * p_ml_2d

    # :912-918, the water soil category at a land point; :920-933, the urban
    # vegetation remap at sf_urban_physics == 0.  Both are per-column integer
    # decisions with no arithmetic, so both are masks.
    soiltyp_2d = np.where(
        (host["isltyp"] == 14) & (host["xice"] == np.float32(0.0)),
        7, host["isltyp"]).astype(np.int32)
    urban_categories = np.asarray(
        (identity.isurban, *identity.lcz), dtype=np.int64)
    vegtyp_2d = np.where(np.isin(host["ivgtyp"].astype(np.int64),
                                 urban_categories),
                         identity.isurban, host["ivgtyp"]).astype(np.int32)
    # :1036-1042.  PLAI is overwritten by PHENOLOGY, but the statement is
    # transcribed because the identity is an argument here, not an assumption.
    plai_2d = np.where(np.isin(vegtyp_2d, (25, 26, 27)),
                       np.float32(0.0), host["lai"]).astype(np.float32)

    # ---- :1026-1029, FICEOLD, over the whole snow stack ------------------
    # ``DO IZ = ISNOW+1, 0`` visits snow slot ``IZ + NSNOW - 1``, so the live
    # slots are exactly those at or above ``ISNOW + NSNOW``.  WRF divides
    # unguarded; this reproduces that on the live slots and leaves the rest at
    # the zero the array is allocated with, which is what the loop does.
    snow_slot = np.arange(NSNOW, dtype=np.int32)[:, None, None]
    live_snow = snow_slot >= (host["isnowxy"].astype(np.int32) + NSNOW)
    ficeold_3d = np.zeros((NSNOW, ny, nx), dtype=np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(host_3d["snicexy"],
                  host_3d["snicexy"] + host_3d["snliqxy"],
                  out=ficeold_3d, where=live_snow)

    # :1045, the glacier entry point: those columns dispatch to the ported
    # NOAHMP_GLACIER path, run below AFTER the write-back copies (which
    # rewrite whole arrays and would clobber an earlier glacier result).
    # ``census["land"]`` counts the columns NOAHMP_SFLX answers;
    # ``census["glacier"]`` counts NOAHMP_GLACIER's.
    glacier = is_land & (vegtyp_2d == identity.isice)
    n_glacier = int(glacier.sum())
    census["land"] -= n_glacier
    census["glacier"] = n_glacier
    if n_glacier and not params.glacier_path:
        gj, gi = (int(v[0]) for v in np.nonzero(glacier))
        raise NoahmpGlacierColumnError(
            f"column (j={gj}, i={gi}) has VEGTYP={identity.isice} == "
            "ISICE_TABLE, which module_sf_noahmpdrv.F:1045 routes to "
            "NOAHMP_GLACIER; that path is disabled on these parameters "
            "(glacier_path=False) and NOAHMP_SFLX is not a substitute "
            "for it")

    evaluator = _resolve_leaf_evaluator()
    staged: list[_StagedColumn] = []

    def flush() -> None:
        if not staged:
            return
        _drive_staged_columns(staged, evaluator)
        _write_back_batch(host, host_3d, staged, nsoil=nsoil, dt=dt32f)
        staged.clear()

    land_j, land_i = np.nonzero(is_land & ~glacier)
    for j, i in zip(land_j.tolist(), land_i.tolist()):
        vegtyp = int(vegtyp_2d[j, i])
        soiltyp = (int(soiltyp_2d[j, i]),) * nsoil
        handle = params.handle(vegtyp=vegtyp, soiltyp=soiltyp,
                               slopetype=1, soilcolor=4, nsoil=nsoil)

        col = SnowColumn(
            nsnow=NSNOW, nsoil=nsoil, isnow=int(host["isnowxy"][j, i]),
            snowh=host["snowh"][j, i], sneqv=host["snow"][j, i],
            snice=host_3d["snicexy"][:, j, i],
            snliq=host_3d["snliqxy"][:, j, i],
            stc=np.concatenate((host_3d["tsnoxy"][:, j, i],
                                host_3d["tslb"][:, j, i])),
            zsnso=host_3d["zsnsoxy"][:, j, i],
            sh2o=host_3d["sh2o"][:, j, i])

        column_smc = np.ascontiguousarray(host_3d["smois"][:, j, i])
        sflx_kwargs = {
            "ficeold": ficeold_3d[:, j, i].copy(),
            "wa": host["waxy"][j, i],
            "wslake": host["wslakexy"][j, i],
            "lat": latitude[j, i],
            "yearlen": yearlen,
            "julian": julian32,
            "cosz": cosz[j, i],
            "dt": dt32,
            "dx": dx32,
            "dz8w": dz1[j, i],
            "zsoil": zsoil,
            "shdmax": fvgmax_2d[j, i],
            "vegtyp": vegtyp,
            "ice": 0,
            "ist": 1,
            "croptype": 0,
            "sfctmp": temperature[j, i],
            "sfcprs": p_ml_2d[j, i],
            "psfc": p_int0[j, i],
            "uu": u[j, i],
            "vv": v[j, i],
            "q2": q_ml_2d[j, i],
            "qc": undefined32,
            "soldn": host["swdown"][j, i],
            "lwdn": host["glw"][j, i],
            "prcpconv": precip_rates["prcpconv"][j, i],
            "prcpnonc": precip_rates["prcpnonc"][j, i],
            "prcpshcv": precip_rates["prcpshcv"][j, i],
            "prcpsnow": precip_rates["prcpsnow"][j, i],
            "prcpgrpl": precip_rates["prcpgrpl"][j, i],
            "prcphail": precip_rates["prcphail"][j, i],
            "tbot": host["tmn"][j, i],
            "co2air": co2air_2d[j, i],
            "o2air": o2air_2d[j, i],
            "foln": foln32,
            "zlvl": z_ml_2d[j, i],
            "albold": host["alboldxy"][j, i],
            "sneqvo": host["sneqvoxy"][j, i],
            "tah": host["tahxy"][j, i],
            "eah": host["eahxy"][j, i],
            "canliq": host["canliqxy"][j, i],
            "canice": host["canicexy"][j, i],
            "tv": host["tvxy"][j, i],
            "tg": host["tgxy"][j, i],
            "qsfc": host["qsfc"][j, i],
            "lai": plai_2d[j, i],
            "sai": host["xsaixy"][j, i],
            "cm": host["cmxy"][j, i],
            "ch": host["chxy"][j, i],
            "tauss": host["taussxy"][j, i],
            "pgs": int(host["pgsxy"][j, i]),
            "calculate_soil": True,
        }
        # The column runs exactly once.  It suspends at each physical leaf
        # call that has a device batch, is answered there, and continues
        # in the same frame; nothing is replayed and no partially mutated
        # state is ever observed.
        steps = sflx_steps(handle, col, column_smc, **sflx_kwargs)
        try:
            first = steps.send(None)
        except StopIteration:
            raise RuntimeError(
                f"land column (j={j}, i={i}) finished NOAHMP_SFLX without "
                "pausing once, which no land column can do: the SFLX "
                "prefix, RADIATION, BARE_FLUX and WATER are on every "
                "column's path")
        staged.append(_StagedColumn(j=j, i=i, col=col, steps=steps,
                                    request=first))
        if len(staged) >= COLUMN_BATCH:
            flush()

    flush()

    for name in names_2d:
        fields[name][...] = cp.asarray(host[name])
    for name, array in host_3d.items():
        fields[name][...] = cp.asarray(array)
    if n_glacier:
        glacier_evaluator = _resolve_glacier_evaluator()
        _glacier_lsm_step(fields, atmosphere, glacier, params=params,
                          coszen=coszen, dt=dt32, dzs=dzs,
                          evaluator=glacier_evaluator)
        census["glacier_path"] = _glacier_execution_provenance(
            glacier_evaluator)
    _noahmp_post_lsm_diagnostics(fields, params=params)
    return census


def _noahmp_post_lsm_diagnostics(fields, *,
                                  params: NoahmpRuntimeParameters) -> None:
    """WRF v4.6.1 Noah-MP's post-LSM ``T2/TH2/Q2`` ownership.

    ``SFCLAY_mynn`` (and the MM5 surface layers) diagnose the three fields
    before the land-surface call.  WRF does not retain those values under
    ``CASE (NOAHMPSCHEME)``: ``module_surface_driver.F:3333-3370`` first
    invalidates all three and then unconditionally replaces them.

    Water and full sea ice use the guarded flux diagnostic at :3344-3355.
    Urban and partial-ice categories take Noah-MP's bare-ground values at
    :3357-3365.  Every other land category blends Noah-MP's vegetation and
    bare-ground diagnostics with ``FVEGXY`` at :3366-3369.  The urban-model
    overwrites following that block are outside ArWen's admitted
    ``sf_urban_physics=0`` identity.

    This routine deliberately writes only ``t2``, ``th2`` and ``q2``.
    Noah-MP's driver write-back owns land ``TSK/HFX/QFX/LH/QSFC/ZNT`` but has
    no ``UST/CHS/CHS2/CQS2/FLHC/FLQC`` argument
    (``module_surface_driver.F:3127-3181``); those exchange arrays therefore
    remain the surface layer's output exactly as they do in WRF.
    """
    import cupy as cp

    f32 = np.float32
    identity = params.land_use
    vegtyp = fields["ivgtyp"].astype(cp.int32, copy=False)
    xice = fields["xice"]

    water_or_full_ice = (
        (vegtyp == int(identity.iswater))
        | ((vegtyp == int(identity.isice))
           & (xice >= f32(params.xice_threshold)))
    )
    urban_categories = cp.asarray(
        np.asarray((identity.isurban, *identity.lcz), dtype=np.int32))
    urban_or_partial_ice = (
        cp.isin(vegtyp, urban_categories)
        | ((vegtyp == int(identity.isice))
           & (xice < f32(params.xice_threshold)))
    )

    rho = fields["psfc"] / (f32(287.0) * fields["tsk"])
    q_guard = fields["cqs2"] < f32(1.0e-5)
    t_guard = fields["chs2"] < f32(1.0e-5)
    safe_cqs2 = cp.where(q_guard, f32(1.0), fields["cqs2"])
    safe_chs2 = cp.where(t_guard, f32(1.0), fields["chs2"])
    q_flux = cp.where(
        q_guard,
        fields["qsfc"],
        fields["qsfc"] - fields["qfx"] / (rho * safe_cqs2),
    ).astype(cp.float32)
    t_flux = cp.where(
        t_guard,
        fields["tsk"],
        fields["tsk"] - fields["hfx"]
        / (rho * f32(1004.5) * safe_chs2),
    ).astype(cp.float32)

    fveg = fields["fvegxy"]
    t_land = (
        fveg * fields["t2mvxy"]
        + (f32(1.0) - fveg) * fields["t2mbxy"]
    ).astype(cp.float32)
    q_land = (
        fveg * fields["q2mvxy"]
        + (f32(1.0) - fveg) * fields["q2mbxy"]
    ).astype(cp.float32)
    t_nonwater = cp.where(
        urban_or_partial_ice, fields["t2mbxy"], t_land)
    q_nonwater = cp.where(
        urban_or_partial_ice, fields["q2mbxy"], q_land)
    fields["t2"][...] = cp.where(
        water_or_full_ice, t_flux, t_nonwater).astype(cp.float32)
    fields["q2"][...] = cp.where(
        water_or_full_ice, q_flux, q_nonwater).astype(cp.float32)
    fields["th2"][...] = (
        fields["t2"]
        * cp.power(f32(1.0e5) / fields["psfc"], f32(287.0 / 1004.5))
    ).astype(cp.float32)


def _lsm_step_slab(
    fields,
    atmosphere: Mapping[str, object],
    *,
    params: NoahmpRuntimeParameters,
    geometry: NoahmpSolarGeometry,
    precipitation: SurfacePrecipitationForcing,
    coszen,
    dt: float,
    dx: float,
    dzs,
    itimestep: int,
    elapsed_seconds: float,
) -> dict[str, int]:
    """One ``noahmplsm`` call through the whole-slab orchestration.

    The same driver contract as the staged body of :func:`noahmp_lsm_step` --
    the same prologue statements, the same skips, the same write-back
    arithmetic -- with two structural differences and no arithmetic ones:

    * everything stays on the device.  The staged path copies ~96 carriers to
      host NumPy, loops per column, and copies back; here the prologue is the
      identical float32 statements as CuPy ufuncs (IEEE binary32 on both
      sides, which ``tests/test_noahmp_slab_libm.py`` measures rather than
      assumes), the land columns are gathered with index arrays, and
      :func:`gpuwm.core.noahmp_column_slab.evaluate_sflx_slab` answers whole
      chunks;
    * the column count per launch is bounded by :data:`SLAB_COLUMN_CHUNK`,
      which is the explicit bound the slab modules' allocation-inventory rows
      demand and the number ``preflight`` prices.

    The two paths are held equal by the six-step carried-state digest gates
    in ``tests/test_noahmp_runtime.py`` and the chunk-inertness gate, both
    bitwise over every carried array.
    """
    import cupy as cp

    from gpuwm.core.noahmp_column_slab import (evaluate_sflx_slab,
                                               parameter_fields)

    ny, nx = fields["tsk"].shape
    nsoil = fields["tslb"].shape[0]
    if nsoil != NSOIL:
        raise ValueError(f"Noah-MP is four-layer only, got nsoil={nsoil}")

    one = np.float32(1.0)
    half = np.float32(0.5)
    zero32f = np.float32(0.0)
    dt32f = np.float32(f32(dt))
    dx32f = np.float32(f32(dx))

    def _forcing(name, level=0):
        return cp.asarray(atmosphere[name][level], dtype=cp.float32)

    temperature = _forcing("temperature")
    qv = _forcing("qv")
    u = _forcing("u")
    v = _forcing("v")
    dz1 = _forcing("dz")
    p_int0 = _forcing("p_interface", 0)
    p_int1 = _forcing("p_interface", 1)

    cosz = cp.ascontiguousarray(cp.asarray(coszen, dtype=cp.float32))
    if cosz.shape != (ny, nx):
        raise ValueError(
            "Noah-MP solar geometry must match the domain grid, got "
            f"{cosz.shape} for a {(ny, nx)} slab")
    julian32 = f32(geometry.julian(elapsed_seconds))
    yearlen = year_length(geometry.year(elapsed_seconds))
    latitude = cp.asarray(np.deg2rad(geometry.latitude_deg).astype(np.float32))
    zsoil_row = cp.asarray(zsoil_from_dzs(dzs))
    identity = params.land_use
    foln32 = np.float32(f32(FOLN))
    undefined32 = np.float32(f32(UNDEFINED_VALUE))
    co2_fraction = np.float32(f32(params.table("noahmp_global_parameters",
                                               "CO2")))
    o2_fraction = np.float32(f32(params.table("noahmp_global_parameters",
                                              "O2")))

    xland = fields["xland"]
    xice = fields["xice"]

    # ---- :689-711, the ITIMESTEP == 1 ocean / sea-ice soil state ----------
    if int(itimestep) == 1:
        water = (xland - np.float32(1.5)) >= zero32f
        ice_full = xice == one
        fields["smstav"][...] = cp.where(water, one, fields["smstav"])
        fields["smstot"][...] = cp.where(water, one, fields["smstot"])
        fields["smois"][...] = cp.where(water[None], one, fields["smois"])
        fields["tslb"][...] = cp.where(water[None], np.float32(273.16),
                                       fields["tslb"])
        seaice1 = ice_full & ~water
        fields["smstav"][...] = cp.where(seaice1, one, fields["smstav"])
        fields["smstot"][...] = cp.where(seaice1, one, fields["smstot"])
        fields["smois"][...] = cp.where(seaice1[None], one, fields["smois"])

    # ---- :715-725, the two CYCLE ILOOP skips -------------------------------
    sea_ice = xice >= np.float32(params.xice_threshold)
    open_water = ((xland - np.float32(1.5)) >= zero32f) & ~sea_ice
    is_land = ~(sea_ice | open_water)
    fields["sh2o"][...] = cp.where(sea_ice[None], one, fields["sh2o"])
    fields["lai"][...] = cp.where(sea_ice, np.float32(0.01), fields["lai"])
    census = {"land": int(is_land.sum()),
              "water": int(open_water.sum()),
              "sea_ice": int(sea_ice.sum())}

    # ---- :729-796, 2-D to 1-D, evaluated once for the slab -----------------
    q_ml = qv / (one + qv)
    z_ml = half * dz1
    p_ml = (p_int1 + p_int0) * half
    precip_rates = noahmp_six_precipitation_rates(
        fields["rainbl"], fields["sr"], precipitation, dt32f, arrays=cp)
    fvgmax = fields["shdmax"] / np.float32(100.0)
    co2air = co2_fraction * p_ml
    o2air = o2_fraction * p_ml

    soiltyp = cp.where((fields["isltyp"] == 14) & (xice == zero32f),
                       7, fields["isltyp"]).astype(cp.int32)
    urban_categories = cp.asarray(
        np.asarray((identity.isurban, *identity.lcz), dtype=np.int64))
    vegtyp = cp.where(cp.isin(fields["ivgtyp"].astype(cp.int64),
                              urban_categories),
                      identity.isurban, fields["ivgtyp"]).astype(cp.int32)

    # ---- :1026-1029, FICEOLD ------------------------------------------------
    snow_slot = cp.arange(NSNOW, dtype=cp.int32)[:, None, None]
    live_snow = snow_slot >= (fields["isnowxy"].astype(cp.int32) + NSNOW)
    snice = fields["snicexy"]
    ficeold = cp.where(live_snow, snice / (snice + fields["snliqxy"]),
                       zero32f).astype(cp.float32)

    # ---- :1045, the glacier entry point ------------------------------------
    # Glacier columns dispatch to the ported NOAHMP_GLACIER path; the
    # ordinary slab below answers only ``is_land & ~glacier``.  The two
    # column sets are disjoint, so the order between them cannot matter.
    glacier = is_land & (vegtyp == identity.isice)
    n_glacier = int(glacier.sum())
    census["land"] -= n_glacier
    census["glacier"] = n_glacier
    if n_glacier:
        if not params.glacier_path:
            gj, gi = (int(w[0]) for w in np.nonzero(cp.asnumpy(glacier)))
            raise NoahmpGlacierColumnError(
                f"column (j={gj}, i={gi}) has VEGTYP={identity.isice} == "
                "ISICE_TABLE, which module_sf_noahmpdrv.F:1045 routes to "
                "NOAHMP_GLACIER; that path is disabled on these "
                "parameters (glacier_path=False) and NOAHMP_SFLX is not "
                "a substitute for it")
        glacier_evaluator = _resolve_glacier_evaluator()
        _glacier_lsm_step(fields, atmosphere, glacier, params=params,
                          coszen=cosz, dt=float(dt32f), dzs=dzs,
                          evaluator=glacier_evaluator)
        census["glacier_path"] = _glacier_execution_provenance(
            glacier_evaluator)

    land_j, land_i = cp.nonzero(is_land & ~glacier)
    n_land = int(land_j.size)
    if n_land == 0:
        return census

    # ---- one parameter row per distinct (vegetation, soil) identity --------
    veg_land = cp.asnumpy(vegtyp[land_j, land_i]).astype(np.int64)
    soil_land = cp.asnumpy(soiltyp[land_j, land_i]).astype(np.int64)
    keys, inverse = np.unique(veg_land * 1000 + soil_land,
                              return_inverse=True)
    rows = [params.slab_row(vegtyp=int(key // 1000),
                            soiltyp=(int(key % 1000),) * nsoil,
                            slopetype=1, soilcolor=4, nsoil=nsoil)
            for key in keys]

    gather_2d = {
        "canliq": "canliqxy", "canice": "canicexy", "sneqv": "snow",
        "wa": "waxy", "snowh": "snowh", "tv": "tvxy", "tg": "tgxy",
        "lwdn": "glw", "eah": "eahxy", "tah": "tahxy",
        "sneqvo": "sneqvoxy", "albold": "alboldxy", "cm": "cmxy",
        "ch": "chxy", "tauss": "taussxy", "qsfc": "qsfc", "tbot": "tmn",
        "soldn": "swdown", "wslake": "wslakexy",
    }
    gather_local = {
        "lat": latitude, "shdmax": fvgmax, "uu": u, "vv": v,
        "zlvl": z_ml, "co2air": co2air, "o2air": o2air, "dz8w": dz1,
        "psfc": p_int0, "sfcprs": p_ml, "sfctmp": temperature, "q2": q_ml,
        "prcpconv": precip_rates["prcpconv"],
        "prcpnonc": precip_rates["prcpnonc"],
        "prcpshcv": precip_rates["prcpshcv"],
        "prcpsnow": precip_rates["prcpsnow"],
        "prcpgrpl": precip_rates["prcpgrpl"],
        "prcphail": precip_rates["prcphail"],
        "cosz": cosz,
    }
    gather_3d = {
        "smc": fields["smois"], "sh2o": fields["sh2o"],
        "zsnso": fields["zsnsoxy"], "snice": fields["snicexy"],
        "snliq": fields["snliqxy"], "ficeold": ficeold,
    }

    for start in range(0, n_land, SLAB_COLUMN_CHUNK):
        stop = min(start + SLAB_COLUMN_CHUNK, n_land)
        jc = land_j[start:stop]
        ic = land_i[start:stop]
        m = stop - start
        at = (jc, ic)

        chunk = {name: fields[carrier][at]
                 for name, carrier in gather_2d.items()}
        chunk.update({name: array[at]
                      for name, array in gather_local.items()})
        chunk.update({name: array[:, jc, ic].T
                      for name, array in gather_3d.items()})
        chunk["stc"] = cp.concatenate(
            (fields["tsnoxy"][:, jc, ic], fields["tslb"][:, jc, ic]),
            axis=0).T
        chunk["zsoil"] = cp.broadcast_to(zsoil_row[None, :], (m, nsoil))
        chunk["isnow"] = fields["isnowxy"][at].astype(cp.int32)
        chunk["vegtyp"] = vegtyp[at]

        zeros = cp.zeros(m, dtype=cp.float32)
        chunk.update({
            "acc_ssoil": zeros,
            "julian": cp.full(m, julian32, dtype=cp.float32),
            "dt": cp.full(m, dt32f, dtype=cp.float32),
            "dx": cp.full(m, dx32f, dtype=cp.float32),
            "foln": cp.full(m, foln32, dtype=cp.float32),
            "qc": cp.full(m, undefined32, dtype=cp.float32),
            "yearlen": cp.full(m, yearlen, dtype=cp.int32),
            "ist": cp.ones(m, dtype=cp.int32),
            "ice": cp.zeros(m, dtype=cp.int32),
            "croptype": cp.zeros(m, dtype=cp.int32),
        })
        chunk.update(parameter_fields(rows, inverse[start:stop]))

        out = evaluate_sflx_slab(chunk, m)
        _write_back_slab(fields, jc, ic, out, nsoil=nsoil, dt=dt32f)

    return census


def _write_back_slab(fields, j, i, r, *, nsoil, dt) -> None:
    """``module_sf_noahmpdrv.F:1223-1400`` on the device, for one slab chunk.

    The identical statements as :func:`_write_back_batch` -- same operator
    order, same ``where(b > a, b, a)`` spelling of gfortran's ``MAX``, same
    layer-order energy accumulation -- reading the orchestration's output
    bundle instead of per-column ``SflxResult`` objects, writing the CuPy
    carriers instead of host copies.  The one structural difference is the
    dead-layer guard in the energy integrals: the host loop skips a layer no
    staged column has reached, this evaluates it masked to an exact ``+0.0``,
    and the docstring of :func:`_write_back_batch` already argues why that
    contribution is the identity.
    """
    import cupy as cp

    at = (j, i)
    zero = np.float32(0.0)
    one = np.float32(1.0)

    for carrier, name in _WRITE_BACK_DIRECT:
        fields[carrier][at] = r[name]

    fields["qfx"][at] = (r["ecan"] + r["edir"]) + r["etran"]           # :1206
    fields["lh"][at] = (r["fcev"] + r["fgev"]) + r["fctr"]             # :1207
    fields["smstav"][at] = zero                                        # :1227
    fields["smstot"][at] = zero
    fields["sfcrunoff"][at] = fields["sfcrunoff"][at] + r["runsrf"]
    fields["udrunoff"][at] = fields["udrunoff"][at] + r["runsub"]
    lit = r["albedo"] > np.float32(-999.0)                             # :1232
    fields["albedo"][at] = cp.where(lit, r["albedo"], fields["albedo"][at])

    fields["canwat"][at] = r["canliq"] + r["canice"]                   # :1241
    fields["acsnow"][at] = (fields["acsnow"][at]
                            + fields["rainbl"][at] * r["fpice"])
    ponding = (r["ponding"] + r["ponding1"]) + r["ponding2"]
    fields["acsnom"][at] = fields["acsnom"][at] + (r["qmelt"] * dt + ponding)
    fields["pondingxy"][at] = ponding

    # :1285-1286, specific humidity back to mixing ratio.
    fields["q2mvxy"][at] = r["q2v"] / (one - r["q2v"])
    fields["q2mbxy"][at] = r["q2b"] / (one - r["q2b"])

    # ---- the column itself -------------------------------------------------
    fields["smois"][:, j, i] = r["smc"].T
    fields["sh2o"][:, j, i] = r["sh2o"].T
    fields["tslb"][:, j, i] = r["stc"][:, NSNOW:].T
    fields["tsnoxy"][:, j, i] = r["stc"][:, :NSNOW].T
    fields["zsnsoxy"][:, j, i] = r["zsnso"].T
    fields["snicexy"][:, j, i] = r["snice"].T
    fields["snliqxy"][:, j, i] = r["snliq"].T
    fields["snow"][at] = r["sneqv"]
    fields["snowh"][at] = r["snowh"]
    fields["isnowxy"][at] = r["isnow"]

    # ---- :1305-1314, the canopy conductance inverse --------------------------
    laisun = cp.where(zero > r["laisun"], zero, r["laisun"])
    laisha = cp.where(zero > r["laisha"], zero, r["laisha"])
    rb = cp.where(zero > r["rb"], zero, r["rb"])
    closed = ((r["rssun"] <= zero) | (r["rssha"] <= zero)
              | (laisun == zero) | (laisha == zero))
    inverse = ((one / (r["rssun"] + rb)) * laisun
               + (one / (r["rssha"] + rb)) * laisha)
    fields["rs"][at] = cp.where(closed, zero, one / inverse)

    # ---- :1381-1394, the two column-energy integrals -------------------------
    stc = r["stc"]
    zsnso = r["zsnso"]
    hcpct = r["hcpct"]
    isnow = r["isnow"].astype(cp.int32)
    count = int(j.size)
    soil_energy = cp.zeros(count, dtype=cp.float32)
    snow_energy = cp.zeros(count, dtype=cp.float32)
    for k in range(-NSNOW + 1, nsoil + 1):
        slot = k + NSNOW - 1
        live = k >= isnow + 1
        top = k == isnow + 1
        above = zsnso[:, slot - 1] if slot > 0 else zero
        thickness = cp.where(top, -zsnso[:, slot], above - zsnso[:, slot])
        term = ((thickness * hcpct[:, slot])
                * (stc[:, slot] - np.float32(273.16))) * np.float32(0.001)
        contribution = cp.where(live, term, zero)
        if k >= 1:
            soil_energy = soil_energy + contribution
        else:
            snow_energy = snow_energy + contribution
    fields["soilenergy"][at] = soil_energy
    fields["snowenergy"][at] = snow_energy


#: The ``SflxResult`` scalars the write-back reads, in one tuple so they can
#: be lifted off the whole batch in a single pass.  A name here that the
#: solver stops producing is an AttributeError at the first flush, which is
#: the failure mode to want.
_WRITE_BACK_SCALARS = (
    "trad", "fsh", "ecan", "edir", "etran", "fcev", "fgev", "fctr", "ssoil",
    "runsrf", "runsub", "albedo", "fsno", "canliq", "canice", "fpice",
    "qmelt", "ponding", "ponding1", "ponding2", "emissi", "qsfc", "tv", "tg",
    "eah", "tah", "cm", "ch", "fwet", "sneqvo", "albold", "qsnow", "qrain",
    "wslake", "lai", "sai", "tauss", "z0wrf", "t2mv", "t2mb", "q2v", "q2b",
    "fveg", "fsa", "fira", "rssun", "rssha", "chv", "chb", "canhs", "qsnbot",
    "eflxb", "laisun", "laisha", "rb",
)

#: 2-D carrier -> the ``SflxResult`` attribute copied into it unchanged.
#: Everything with arithmetic or a branch is written out below instead.
_WRITE_BACK_DIRECT = (
    ("tsk", "trad"), ("hfx", "fsh"), ("grdflx", "ssoil"),            # :1223
    ("snowc", "fsno"), ("emiss", "emissi"), ("qsfc", "qsfc"),
    ("tvxy", "tv"), ("tgxy", "tg"), ("canliqxy", "canliq"),
    ("canicexy", "canice"), ("eahxy", "eah"), ("tahxy", "tah"),
    ("cmxy", "cm"), ("chxy", "ch"), ("fwetxy", "fwet"),
    ("sneqvoxy", "sneqvo"), ("alboldxy", "albold"), ("qsnowxy", "qsnow"),
    ("qrainxy", "qrain"), ("wslakexy", "wslake"),
    # WAXY is deliberately absent: opt_run=3 kills GROUNDWATER, so nothing in
    # the column assigns WA and SflxResult does not carry it.  Writing back a
    # value the solver never produced is how a pass-through becomes a lie.
    ("lai", "lai"), ("xsaixy", "sai"), ("taussxy", "tauss"),
    ("z0", "z0wrf"), ("znt", "z0wrf"),                               # :1281
    ("t2mvxy", "t2mv"), ("t2mbxy", "t2mb"), ("tradxy", "trad"),
    ("fvegxy", "fveg"), ("runsfxy", "runsrf"), ("runsbxy", "runsub"),
    ("ecanxy", "ecan"), ("edirxy", "edir"), ("etranxy", "etran"),
    ("fsaxy", "fsa"), ("firaxy", "fira"), ("rssunxy", "rssun"),
    ("rsshaxy", "rssha"), ("chvxy", "chv"), ("chbxy", "chb"),
    ("canhsxy", "canhs"), ("fpicexy", "fpice"), ("qsnbotxy", "qsnbot"),
    ("qmeltxy", "qmelt"), ("eflxbxy", "eflxb"),
)


def _write_back_batch(host, host_3d, staged, *, nsoil, dt) -> None:
    """``module_sf_noahmpdrv.F:1223-1400``, for a whole staged batch.

    This was one call per land column and about sixty NumPy scalar stores in
    each, and it was 6% of the measured land-surface call before any physics
    ran.  Every store here is the same IEEE binary32 operation on the same
    values; only the loop shape changed.  Two places need care and get it:

    * ``MAX(x, 0.0)`` at :1305-1307 is gfortran's, and Python's ``max`` and
      ``numpy.maximum`` disagree on it for a negative zero -- ``max(-0.0,
      0.0)`` is ``-0.0`` and ``numpy.maximum(-0.0, 0.0)`` is ``+0.0``.  The
      ``where(b > a, b, a)`` form below is the one that matches ``max``, and
      it is written out rather than called so the disagreement is visible.
    * the two energy integrals at :1381-1394 accumulate **in layer order** and
      FP32 addition is not associative, so the layer loop is still a loop.
      What is vectorised is the column axis inside it.
    """
    import operator

    count = len(staged)
    j = np.fromiter((c.j for c in staged), dtype=np.intp, count=count)
    i = np.fromiter((c.i for c in staged), dtype=np.intp, count=count)
    at = (j, i)
    results = [c.result for c in staged]
    columns = [c.col for c in staged]

    # One attribute pass over the batch, transposed once.
    rows = np.array(
        [operator.attrgetter(*_WRITE_BACK_SCALARS)(x) for x in results],
        dtype=np.float32)
    r = {name: rows[:, k] for k, name in enumerate(_WRITE_BACK_SCALARS)}

    for carrier, name in _WRITE_BACK_DIRECT:
        host[carrier][at] = r[name]

    host["qfx"][at] = (r["ecan"] + r["edir"]) + r["etran"]            # :1206
    host["lh"][at] = (r["fcev"] + r["fgev"]) + r["fctr"]              # :1207
    host["smstav"][at] = np.float32(0.0)                              # :1227
    host["smstot"][at] = np.float32(0.0)
    host["sfcrunoff"][at] = host["sfcrunoff"][at] + r["runsrf"]
    host["udrunoff"][at] = host["udrunoff"][at] + r["runsub"]
    # :1232.  The literal is WRF's -999, not the -999.9 sentinel NOAHMP_SFLX
    # writes at :1076, so the test is strictly wider than the sentinel.
    lit = r["albedo"] > np.float32(-999.0)
    host["albedo"][j[lit], i[lit]] = r["albedo"][lit]

    host["canwat"][at] = r["canliq"] + r["canice"]                    # :1241
    host["acsnow"][at] = host["acsnow"][at] + host["rainbl"][at] * r["fpice"]
    ponding = (r["ponding"] + r["ponding1"]) + r["ponding2"]
    host["acsnom"][at] = host["acsnom"][at] + (r["qmelt"] * dt + ponding)
    host["pondingxy"][at] = ponding

    # :1285-1286, specific humidity back to mixing ratio.
    host["q2mvxy"][at] = r["q2v"] / (np.float32(1.0) - r["q2v"])
    host["q2mbxy"][at] = r["q2b"] / (np.float32(1.0) - r["q2b"])

    # ---- the column itself ------------------------------------------------
    stc = np.array([c.stc for c in columns], dtype=np.float32)
    zsnso = np.array([c.zsnso for c in columns], dtype=np.float32)
    isnow = np.fromiter((c.isnow for c in columns), dtype=np.int32,
                        count=count)
    host_3d["smois"][:, j, i] = np.array([x.smc for x in results],
                                         dtype=np.float32).T
    host_3d["sh2o"][:, j, i] = np.array([c.sh2o for c in columns],
                                        dtype=np.float32).T
    host_3d["tslb"][:, j, i] = stc[:, NSNOW:].T
    host_3d["tsnoxy"][:, j, i] = stc[:, :NSNOW].T
    host_3d["zsnsoxy"][:, j, i] = zsnso.T
    host_3d["snicexy"][:, j, i] = np.array([c.snice for c in columns],
                                           dtype=np.float32).T
    host_3d["snliqxy"][:, j, i] = np.array([c.snliq for c in columns],
                                           dtype=np.float32).T
    host["snow"][at] = np.fromiter((c.sneqv for c in columns),
                                   dtype=np.float32, count=count)
    host["snowh"][at] = np.fromiter((c.snowh for c in columns),
                                    dtype=np.float32, count=count)
    host["isnowxy"][at] = isnow

    # ---- :1305-1314, the canopy conductance inverse -----------------------
    zero = np.float32(0.0)
    laisun = np.where(zero > r["laisun"], zero, r["laisun"])
    laisha = np.where(zero > r["laisha"], zero, r["laisha"])
    rb = np.where(zero > r["rb"], zero, r["rb"])
    closed = ((r["rssun"] <= zero) | (r["rssha"] <= zero)
              | (laisun == zero) | (laisha == zero))
    with np.errstate(divide="ignore", invalid="ignore"):
        inverse = ((np.float32(1.0) / (r["rssun"] + rb)) * laisun
                   + (np.float32(1.0) / (r["rssha"] + rb)) * laisha)
        host["rs"][at] = np.where(closed, zero, np.float32(1.0) / inverse)

    # ---- :1381-1394, the two column-energy integrals ----------------------
    hcpct = np.array([x.hcpct for x in results], dtype=np.float32)
    soil_energy = np.zeros(count, dtype=np.float32)
    snow_energy = np.zeros(count, dtype=np.float32)
    for k in range(-NSNOW + 1, nsoil + 1):
        slot = k + NSNOW - 1
        live = k >= isnow + 1
        if not live.any():
            continue
        top = k == isnow + 1
        above = zsnso[:, slot - 1] if slot > 0 else zero
        thickness = np.where(top, -zsnso[:, slot], above - zsnso[:, slot])
        term = ((thickness * hcpct[:, slot])
                * (stc[:, slot] - np.float32(273.16))) * np.float32(0.001)
        # A column that has not reached ISNOW+1 adds an exact +0.0, which is
        # the identity for every value this accumulator can hold: it starts
        # at +0.0 and the only way to reach -0.0 by addition is from -0.0.
        contribution = np.where(live, term, zero)
        if k >= 1:
            soil_energy = soil_energy + contribution
        else:
            snow_energy = snow_energy + contribution
    host["soilenergy"][at] = soil_energy
    host["snowenergy"][at] = snow_energy


__all__ = [
    "COLD_START_EVALUATOR",
    "COLD_START_WRITES",
    "FOLN",
    "GLACIER_BATCH_EVALUATOR",
    "HOST_COLD_START_ENV",
    "NOAHMP_DIAGNOSTICS_2D",
    "NOAHMP_RUNTIME_RESTRICTIONS",
    "NOAHMP_STATE_2D",
    "NOAHMP_STATE_INT_2D",
    "NOAHMP_STATE_SNOWSOIL_3D",
    "NOAHMP_STATE_SNOW_3D",
    "NSNOW",
    "NSOIL",
    "NoahmpGlacierColumnError",
    "NoahmpRuntimeParameters",
    "NoahmpSolarGeometry",
    "UNDEFINED_VALUE",
    "XICE_THRESHOLD",
    "classify_noahmp_surface",
    "cold_start_on_device",
    "cold_start_on_host",
    "guard_noahmp_glacier_columns",
    "noahmp_cold_start",
    "noahmp_lsm_step",
    "year_length",
    "zsoil_from_dzs",
]
