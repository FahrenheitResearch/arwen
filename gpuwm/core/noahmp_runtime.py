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

#: ``XICE_THRESHOLD``, WRF Registry default 0.5.  gpuwm has no namelist field
#: for it, so it is pinned here and published in
#: :data:`NOAHMP_RUNTIME_RESTRICTIONS` rather than left implicit.
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
        "a post-static pre-forecast scan refuses any active land column "
        "whose VEGTYP equals ISICE_TABLE; the runtime check remains a "
        "backstop",
        "module_sf_noahmpdrv.F:1043-1150 routes it to NOAHMP_GLACIER "
        "(module_sf_noahmp_glacier.F, 3,080 lines), which is not ported.  "
        "Falling through to NOAHMP_SFLX would run a vegetated land model on "
        "an ice sheet.",
    ),
    (
        "sea_ice_columns",
        "XICE >= 0.5 columns take WRF's skip (SH2O=1, XLAI=0.01) and are "
        "otherwise untouched",
        "This is what WRF does (:715-723).  It is listed because the skip "
        "means no sea-ice surface energy balance exists in a Noah-MP run: "
        "TSK/HFX/QFX over sea ice keep whatever the surface layer left.",
    ),
    (
        "xice_threshold",
        f"pinned at {XICE_THRESHOLD}",
        "WRF's XICE_THRESHOLD is a namelist value; gpuwm has no "
        "configuration field for it, so it cannot be varied and the "
        "Registry default is pinned.",
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
    """A column reached the glacier entry point, which is not ported."""


def guard_noahmp_glacier_columns(fields, params: "NoahmpRuntimeParameters"):
    """Refuse Noah-MP glacier columns before forecast state can advance.

    WRF v4.6.1 ``module_sf_noahmpdrv.F:1043`` routes these columns to
    ``NOAHMP_GLACIER``.  That solver is out of scope; running ``NOAHMP_SFLX``
    would be a different surface.  Sea-ice/open-water columns are excluded
    with the same masks as the runtime driver.
    """
    import cupy as cp

    xice = cp.asarray(fields["xice"])
    xland = cp.asarray(fields["xland"])
    vegtyp = cp.asarray(fields["ivgtyp"])
    active_land = ((xice < np.float32(XICE_THRESHOLD))
                   & (xland < np.float32(1.5)))
    offending = active_land & (vegtyp == int(params.land_use.isice))
    if not bool(cp.any(offending)):
        return
    flat = int(cp.argmax(offending.reshape(-1)).item())
    j, i = (int(value) for value in np.unravel_index(
        flat, tuple(int(value) for value in offending.shape)))
    category = int(vegtyp[j, i].item())
    raise NoahmpGlacierColumnError(
        "post-static Noah-MP glacier guard: first offending active land "
        f"cell is (j={j}, i={i}), VEGTYP={category}=ISICE_TABLE. "
        "NOAHMP_GLACIER is not ported; porting it is the follow-up that "
        "removes this guard.")


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
                 dataset_identifier: str = DEFAULT_VEGETATION_DATASET):
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

    def veg_value(self, name: str, vegtyp: int) -> float:
        """``<NAME>_TABLE(VEGTYP)`` out of the active vegetation dataset.

        Only the cold start needs this: ``TRANSFER_MP_PARAMETERS`` already
        indexes every parameter the column solver reads, but ``NOAHMP_INIT``
        reads ``SLA_TABLE`` directly (:2187) before any transfer has happened.
        """
        entry = self._veg.values[name.upper()]
        return float(entry[int(vegtyp) - 1] if len(entry) > 1 else entry[0])

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
            "xice_threshold": float(XICE_THRESHOLD),
            "column_solver": "host-fp32+device-vegeflux-v1",
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


def noahmp_cold_start(fields, *, params: NoahmpRuntimeParameters,
                      dzs) -> None:
    """``NOAHMP_INIT`` + ``SNOW_INIT`` over the whole slab, on the host.

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
    """
    import cupy as cp

    slab = {name: np.ascontiguousarray(cp.asnumpy(fields[name]))
            for name in ("snow", "snowh", "canwat", "tslb", "smois", "sh2o",
                         "tsk", "xice", "ivgtyp", "isltyp", "lai",
                         *NOAHMP_STATE_2D, *NOAHMP_STATE_INT_2D,
                         *NOAHMP_STATE_SNOW_3D, *NOAHMP_STATE_SNOWSOIL_3D)}
    ny, nx = slab["tsk"].shape
    nsoil = slab["tslb"].shape[0]
    if nsoil != NSOIL:
        raise ValueError(
            f"Noah-MP cold start is four-layer only, got nsoil={nsoil}")
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

    for name, array in slab.items():
        fields[name][...] = cp.asarray(array)


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
    # order matters and is kept: a column with XICE >= XICE_THRESHOLD takes
    # the sea-ice skip whatever XLAND says.
    sea_ice = host["xice"] >= np.float32(XICE_THRESHOLD)
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

    # :1043, the glacier entry point.  Reported with its own column index,
    # because "somewhere in this nest" is not a usable message.
    glacier = is_land & (vegtyp_2d == identity.isice)
    if glacier.any():
        gj, gi = (int(v[0]) for v in np.nonzero(glacier))
        raise NoahmpGlacierColumnError(
            f"column (j={gj}, i={gi}) has VEGTYP={identity.isice} == "
            "ISICE_TABLE, which module_sf_noahmpdrv.F:1043 routes to "
            "NOAHMP_GLACIER; that entry point is not ported and "
            "NOAHMP_SFLX is not a substitute for it")

    evaluator = _resolve_leaf_evaluator()
    staged: list[_StagedColumn] = []

    def flush() -> None:
        if not staged:
            return
        _drive_staged_columns(staged, evaluator)
        _write_back_batch(host, host_3d, staged, nsoil=nsoil, dt=dt32f)
        staged.clear()

    land_j, land_i = np.nonzero(is_land)
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
           & (xice >= f32(XICE_THRESHOLD)))
    )
    urban_categories = cp.asarray(
        np.asarray((identity.isurban, *identity.lcz), dtype=np.int32))
    urban_or_partial_ice = (
        cp.isin(vegtyp, urban_categories)
        | ((vegtyp == int(identity.isice))
           & (xice < f32(XICE_THRESHOLD)))
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
    sea_ice = xice >= np.float32(XICE_THRESHOLD)
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

    # ---- :1043, the glacier entry point ------------------------------------
    glacier = is_land & (vegtyp == identity.isice)
    if bool(glacier.any()):
        gj, gi = (int(w[0]) for w in np.nonzero(cp.asnumpy(glacier)))
        raise NoahmpGlacierColumnError(
            f"column (j={gj}, i={gi}) has VEGTYP={identity.isice} == "
            "ISICE_TABLE, which module_sf_noahmpdrv.F:1043 routes to "
            "NOAHMP_GLACIER; that entry point is not ported and "
            "NOAHMP_SFLX is not a substitute for it")

    land_j, land_i = cp.nonzero(is_land)
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
    "FOLN",
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
    "guard_noahmp_glacier_columns",
    "noahmp_cold_start",
    "noahmp_lsm_step",
    "year_length",
    "zsoil_from_dzs",
]
