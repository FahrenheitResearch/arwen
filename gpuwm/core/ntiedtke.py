"""Launcher for New Tiedtke (``cu_physics = 16``).

Owns the ONE launch-geometry descriptor every stage runs under, and is the
only thing that launches ``gpuwm/core/kernels/ntiedtke.cu``.

WHY THE DESCRIPTOR IS A TYPE AND NOT A CONVENTION
-------------------------------------------------
The column workspace is per-block and lane-interleaved -- element ``k`` of
slot ``s`` for lane ``t`` at ``block_base + (s*nz + k)*LANES + t`` -- so a
column's slots belong to one ``(block, lane)`` pair for the WHOLE stage
sequence.  Launch one stage on a different threads-per-block or a different
grid and that column resumes on a different lane, reading another column's
state.  Nothing crashes.  Every number stays finite.

It is also the one guarantee with no analogue in the Fortran, so no
comparison against the reference can catch it -- and this project's culture
is kernel performance work.  ``ntiedtke_cutypen`` is at 91 registers;
someone will want to re-tile it for occupancy, and re-tiling one stage is
precisely the thing that must be impossible.

So:

* :class:`NtLaunchGeometry` is frozen and computed once per step.
* :class:`NtStages` methods take **no grid or block argument at all**.
  There is no per-stage override to reach for -- changing the tile means
  changing the descriptor, which changes every stage together.
* Every kernel additionally REPORTS the geometry it actually observed and
  refuses to compute when it disagrees with what it was promised, so a
  launch routed around this module fails loudly instead of silently
  reading another column.

``tests/test_ntiedtke_launch_geometry.py`` is the gate on the last point.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

#: Every device array in this module is float32, the same DTYPE the
#: rest of gpuwm uses. Imported rather than restated so a change to
#: the model's working precision reaches the scheme.
from gpuwm.core.state import DTYPE

#: Threads per block for every New Tiedtke stage.
#:
#: CORRECTED.  This was documented as "the workspace's lane count, not a
#: tuning knob", on the assumption that the kernels would use a per-block
#: lane-interleaved workspace.  THEY DO NOT: there is no such workspace
#: (see docs/ntiedtke/PORT-RECORD.md §7), and every kernel indexes its arrays by the
#: GLOBAL column `a[k*ncol + i]`.  The kernels are therefore already
#: tile-independent and 32 is a FREE TUNING KNOB, not a correctness
#: constraint.
#:
#: It is still refused below, and the descriptor is still the single place
#: a tile is chosen -- that mechanism is worth keeping and is what a tile
#: cap would be introduced through -- but the reason is now "one deliberate
#: choice per step" rather than "the layout is keyed on it".
#:
#: 32 is one warp per block, which caps occupancy through the
#: blocks-per-SM limit regardless of register count, and cuascn is at 94
#: registers with cutypen at 91.  Revisiting it is a Phase 4 question.
NT_TPB = 32

#: Stage ids, mirroring the ``NT_STAGE_*`` defines in ntiedtke.cu.  The
#: kernel writes its observed geometry into ``geom_report[stage_id]``.
NT_STAGE_PREP = 0
NT_STAGE_CONVERT = 1
NT_STAGE_CUININ = 2
NT_STAGE_CUTYPEN = 3
NT_STAGE_MIDLEVEL = 4
#: cumastrn:500-541, the first-guess cloud-base mass flux.  A
#: PREREQUISITE stage: cuascn, cudlfsn and the closure all read its
#: pmfub, and it can clear ldcum.
NT_STAGE_MFUB = 5
#: cumastrn:620-745, the CAPE closure.  Takes the CLOSURE-TIME ktype,
#: post :566-568, not cuascn's.
NT_STAGE_CLOSURE = 6
#: cu_ntiedtke.F90:1755-2258, the entraining/detraining updraft.  The
#: largest routine in the scheme and the only one carrying a TILE-WIDE
#: flag: llo3 is an OR-reduction over every column, latched true for the
#: rest of the run, so it is a launch argument and not a per-lane value.
NT_STAGE_CUASCN = 7
#: cu_ntiedtke.F90:3064-3148 and :3152-3252 -- the tendencies.  Both
#: ACCUMULATE into their outputs rather than assigning; see
#: docs/ntiedtke/PORT-RECORD.md section 17 for why that is not the same thing.
NT_STAGE_CUDTDQN = 8
NT_STAGE_CUDUDVN = 9
#: cu_ntiedtke.F90:2262-2487 and :2495-2721 -- the downdraft pair.  ALL SIX
#: of cudlfsn's level outputs are class 2, so its kernel must LEAVE the
#: untouched levels alone rather than zero them.
NT_STAGE_CUDLFSN = 10
NT_STAGE_CUDDRAFN = 11
#: cu_ntiedtke.F90:2725-3060 -- the final fluxes.  ktopm2 is NOT one of its
#: arguments: :2877 overwrites it unconditionally before any read, so
#: cumastrn:565's horizontal leak is dead.  See docs/ntiedtke/PORT-RECORD.md section 16.
NT_STAGE_CUFLXN = 12
#: cumastrn:562-590 -- the cloud-depth check, and the FIRST piece of
#: orchestration to gain an owner.  It carries :566-568, the ktype flip
#: that selects scale_fac vs scale_fac2, and :580-588's downdraft zeroing.
#: It runs BETWEEN cuascn and cudlfsn.
NT_STAGE_DEPTH = 13
#: cumastrn:833-919 -- the adjustments block, between cuflxn and cudtdqn.
#: The DOWNDRAFT stability cap, not the momentum rescale at :996-1016;
#: the two share the local name zmfs and were confused once already.
NT_STAGE_ADJUST = 14
#: cumastrn:996-1016 -- the MOMENTUM mass-flux rescale, and the only
#: consumer of zcons (1/(g*dt)) in the scheme.  Everything else caps with
#: zcons2 (3/(g*dt)), one character away.  Produces zmfuus/zmfdus, which
#: is what cududvn consumes.
NT_STAGE_MRESCALE = 15
#: cumastrn:743-819 -- the UPDRAFT rescale, where the CAPE closure's
#: zmfub1 is actually spent, plus the 6.6 and 6.7 cleanups.
NT_STAGE_USCALE = 16
#: cumastrn:927-995 -- the updraft/downdraft momentum profiles.  This is
#: what writes puu/pvu/pud/pvd, which cududvn consumes; the port's eighth
#: wrong claim said nothing between cuinin and cududvn did.
NT_STAGE_MPROFILE = 17
#: cumastrn:1030-1056 -- the KE dissipation, the LAST arithmetic in
#: cumastrn and the only place the momentum tendency feeds back into the
#: heat tendency.
NT_STAGE_KEDIS = 18

#: cu_ntiedtke_post_run.  NOT part of cumastrn and not part of
#: cu_ntiedtke_run -- it is the driver-level step that turns the scheme's
#: updated state into the eight tendency fields WRF consumes, and it is the
#: only stage that reads two vertical conventions at once.
NT_STAGE_POSTRUN = 19

#: cu_ntiedtke_run's post-conversion (:278-320), which turns cumastrn's
#: tendencies into the state post_run differences.  ID 20, but it RUNS
#: BEFORE stage 19 -- ids label the report array, NT_CALL_ORDER is the
#: sequence, and the two have disagreed since cudtdqn and cududvn.
NT_STAGE_POSTCONV = 20
NT_STAGE_COUNT = 21

NT_STAGE_NAMES = {
    NT_STAGE_PREP: "prep",
    NT_STAGE_CONVERT: "convert",
    NT_STAGE_CUININ: "cuinin",
    NT_STAGE_CUTYPEN: "cutypen",
    NT_STAGE_MIDLEVEL: "midlevel",
    NT_STAGE_MFUB: "mfub",
    NT_STAGE_CLOSURE: "closure",
    NT_STAGE_CUASCN: "cuascn",
    NT_STAGE_CUDTDQN: "cudtdqn",
    NT_STAGE_CUDUDVN: "cududvn",
    NT_STAGE_CUDLFSN: "cudlfsn",
    NT_STAGE_CUDDRAFN: "cuddrafn",
    NT_STAGE_CUFLXN: "cuflxn",
    NT_STAGE_DEPTH: "cloud_depth",
    NT_STAGE_ADJUST: "adjust",
    NT_STAGE_MRESCALE: "momentum_rescale",
    NT_STAGE_USCALE: "updraft_scale",
    NT_STAGE_MPROFILE: "momentum_profile",
    NT_STAGE_KEDIS: "ke_dissipation",
    NT_STAGE_POSTRUN: "post_run",
    NT_STAGE_POSTCONV: "post_conversion",
}


@dataclass(frozen=True)
class NtLaunchGeometry:
    """One immutable tile, constructed once per cumulus step.

    Frozen on purpose: a stage that wanted its own tile would have to build
    a second descriptor, which is a visible act rather than an argument
    tweak at one call site.
    """

    ncol: int
    nz: int
    tpb: int = NT_TPB

    def __post_init__(self):
        if self.ncol <= 0 or self.nz <= 0:
            raise ValueError(f"ncol={self.ncol}, nz={self.nz} must be > 0")
        if self.tpb != NT_TPB:
            raise ValueError(
                f"tpb={self.tpb} is not the stage tile width {NT_TPB}. "
                "This is no longer a correctness constraint -- the kernels "
                "index by global column and are tile-independent -- but the "
                "tile is chosen ONCE per step on purpose, so changing it is "
                "an edit here and not an argument at one call site.")

    @property
    def nblocks(self) -> int:
        return (self.ncol + self.tpb - 1) // self.tpb

    @property
    def grid(self) -> tuple[int]:
        return (self.nblocks,)

    @property
    def block(self) -> tuple[int]:
        return (self.tpb,)

    @property
    def observed_stamp(self) -> int:
        """What a correctly-launched kernel writes into ``geom_report``."""
        return (self.nblocks << 16) | (self.tpb & 0xFFFF)


class NtStages:
    """The only launcher for the New Tiedtke kernels.

    Every method launches on ``self.geometry`` and takes no grid or block
    argument.  ``geom_report`` accumulates each stage's OBSERVED geometry;
    :meth:`check_geometry` fails if any stage ran under a different one.
    """

    def __init__(self, geometry: NtLaunchGeometry):
        import cupy as cp

        self.geometry = geometry
        self._cp = cp
        #: -1 means "this stage has not run", which is distinguishable from
        #: any real geometry (they are all non-negative).
        self._geom_report = cp.full(NT_STAGE_COUNT, -1, dtype=cp.int32)
        #: GUARANTEE 4.  geom_report says THAT a stage ran and under what
        #: tile; this says WHEN.  One ticket per launch, drawn device-side
        #: by block 0 thread 0, for the same reason the geometry check is
        #: device-side: a launch routed around this class must still be
        #: caught.  -1 is "did not run".
        self._order_report = cp.full(NT_STAGE_COUNT, -1, dtype=cp.int32)
        self._ticket = cp.zeros(1, dtype=cp.int32)

    @property
    def geom_report(self):
        return self._geom_report

    @property
    def order_report(self):
        return self._order_report

    def _tail(self):
        """The trailing arguments every ntiedtke kernel takes.

        Appended HERE rather than at each call site, so a stage cannot opt
        out of the geometry or ordering checks by forgetting them.
        """
        import numpy as np
        g = self.geometry
        return (np.int32(g.tpb), np.int32(g.nblocks), self._geom_report,
                self._order_report, self._ticket)

    def launch(self, kernel_name: str, args: tuple):
        """Launch one stage under the descriptor.  No geometry parameter."""
        from gpuwm.core.kernels import load_module

        fn = load_module("ntiedtke").get_function(kernel_name)
        fn(self.geometry.grid, self.geometry.block, tuple(args) + self._tail())
        return self

    def check_geometry(self):
        """Every stage that ran must report the descriptor's geometry.

        Raises rather than returning a flag: a mismatch means some column
        has already read another column's workspace, and the run is void.
        """
        report = self._cp.asnumpy(self._geom_report)
        want = self.geometry.observed_stamp
        bad = {NT_STAGE_NAMES.get(i, i): int(v)
               for i, v in enumerate(report) if v != -1 and int(v) != want}
        if bad:
            raise RuntimeError(
                "New Tiedtke stages ran under different launch geometries, "
                f"which means at least one column resumed on a different "
                f"lane and read another column's workspace.  Expected "
                f"{want:#x} (grid {self.geometry.nblocks}, block "
                f"{self.geometry.tpb}); got {bad}.")
        return True

    def check_order(self, expected):
        """Every stage that ran did so in the DECLARED order.

        ``expected`` is the sequence of stage ids the caller says it
        launched, in order.  Declared rather than derived, for the reason
        CALL_SITE_DEBTS is declared: when the orchestration grows a stage,
        the sequence gains a member and someone has to notice, rather than
        the assertion quietly re-deriving itself around the change.

        Raises rather than returning a flag.  An out-of-order launch reads
        a predecessor's array before that predecessor wrote it -- nothing
        crashes, every number stays finite, and the parity suite launches
        the sequence it was written with, so nothing else would see it.
        """
        order = self._cp.asnumpy(self._order_report)
        ran = {i: int(v) for i, v in enumerate(order) if int(v) != -1}
        want = {int(s): n for n, s in enumerate(expected)}
        if ran != want:
            name = NT_STAGE_NAMES.get
            got_seq = [name(s, s) for s, _ in sorted(ran.items(),
                                                     key=lambda kv: kv[1])]
            raise RuntimeError(
                "New Tiedtke stages did not run in the declared order. "
                f"Declared {[name(s, s) for s in expected]}; observed "
                f"{got_seq}. An out-of-order launch reads a predecessor's "
                "column arrays before that predecessor wrote them, and "
                "nothing else would catch it.")
        return True

    def stages_that_ran(self):
        report = self._cp.asnumpy(self._geom_report)
        return {NT_STAGE_NAMES[i] for i, v in enumerate(report)
                if int(v) != -1}


__all__ = [
    "NT_TPB", "NT_STAGE_COUNT", "NT_STAGE_NAMES",
    "NT_STAGE_PREP", "NT_STAGE_CONVERT", "NT_STAGE_CUININ",
    "NT_STAGE_CUTYPEN", "NT_STAGE_MIDLEVEL", "NT_STAGE_MFUB", "NT_STAGE_CLOSURE",
    "NtLaunchGeometry", "NtStages",
]


#: Arrays a stage READS before any stage has WRITTEN them. Every one is
#: either a driver input, an alias of another array under the reference's
#: own dummy name, or a copy the assembler must make -- and each is
#: classified below, because "read before written" with no explanation is
#: precisely the class-1 hazard this port has hit eight times, moved up one
#: level from routines to stages.
NT_SEEDS = {
    # --- the driver's own arrays, WRF order ---------------------------
    "t3d": "driver", "qv3d": "driver", "qc3d": "driver", "qi3d": "driver",
    "u3d": "driver", "v3d": "driver", "pcps": "driver", "p8w": "driver",
    "dz8w": "driver", "rho3d": "driver", "w": "driver", "qvften": "driver",
    "thften": "driver", "xland": "driver", "hfx": "driver", "qfx": "driver",
    "dx": "driver",
    # post_run's reference state: the SAME driver arrays under the names
    # module_cu_ntiedtke.F gives them, so they are ALIASES and share one
    # storage. Classed "driver" until the pipeline was assembled, which
    # allocated each twice and would have handed post_run six arrays of
    # zeros -- every tendency then measured against a reference state of
    # nothing. Neither the dataflow gate nor the workspace tests could see
    # it: both names were legitimately seeds, and only running the walk
    # showed that they had to be the SAME seed.
    #
    # exner is the exception and stays a driver seed: the driver spells it
    # pi3d, but no kernel takes a parameter of that name, so `exner` IS the
    # array rather than a second name for one.
    "exner": "driver (pi3d)",
    "t": "alias of t3d", "qv": "alias of qv3d", "qc": "alias of qc3d",
    "qi": "alias of qi3d", "u": "alias of u3d", "v": "alias of v3d",
    # --- cu_ntiedtke_run -> cumastrn, the call at :283-291 -------------
    "pten": "alias of ztp1", "pqen": "alias of zqp1",
    "puen": "alias of pum1", "pven": "alias of pvm1",
    "pqsen": "alias of zqsat", "pap": "alias of prsl", "paph": "alias of prsi",
    "lndj": "alias of slimsk",
    # --- cumastrn -> its callees ---------------------------------------
    "ztenh": "alias of ptenh", "zqenh": "alias of pqenh",
    "ktype": "alias of ktype_o", "kcbot": "alias of cubot_o",
    "kdpl": "alias of kdpl_o", "wbase": "alias of wbase_o",
    "loddraf": "alias of lddraf",
    "pvom": "alias of ptenu", "pvol": "alias of ptenv",
    "rn": "alias of zprecc",
    # --- copies the assembler makes ------------------------------------
    # ztenu/ztenv are copied from pvom/pvol at :1019-1024, which is BEFORE
    # cududvn writes them -- so what is copied is the ZEROED array from
    # :258-259, not cududvn's output. The chain "ztenu is a copy of pvom,
    # pvom is an alias of ptenu" is therefore wrong in a way that reads
    # correctly: it names the post-cududvn array as the source of a
    # pre-cududvn copy. Found by the alias-target gate below, on the table
    # its own author wrote an hour earlier.
    #
    # MEASURED: ztenu and ztenv are identically zero on all 5,292 fixture
    # rows, which confirms nothing writes pvom/pvol between :259 and
    # :1019. The assembler still COPIES rather than zero-filling -- the
    # shortcut would be right today and is exactly the shape that has been
    # wrong eight times -- but the measurement means a future divergence
    # would be visible rather than silent.
    "ztenu": "zeroed by the assembler (:258-259), copied at :1019-1024",
    "ztenv": "zeroed by the assembler (:258-259), copied at :1019-1024",
    "zqq": "copy of pqte before cumastrn (:274)",
    "ztt": "copy of ptte before cumastrn (:276)",
}

# ===========================================================================
# The workspace
# ===========================================================================
# WHAT THE ASSEMBLER ALLOCATES.  Every array any stage touches, at the
# capped tile, once per chunk.  The names are the kernels' own parameter
# names, which are the reference's dummy names, so the connection graph
# between stages IS the naming -- and where the reference passes one array
# under two names, NT_SEEDS records it rather than the workspace pretending
# they are two arrays.
#
# TWO LEVEL-INDEXING CONVENTIONS.  Some stages walk k = 0..nz-1 and others
# jk = 1..klev.  One (nz+2, ncol) allocation serves both: a 1-based stage
# gets the whole array and a 0-based stage gets ``arr[1:]``, which is a
# contiguous VIEW of the same memory, so the two agree on where level 1
# lives.  Getting the base wrong shifts a whole column by one level with no
# crash, which is the vertical flip's failure mode at a smaller scale.
#
# The base per stage is NOT declared here.  Three static derivations gave
# three different answers (docs/ntiedtke/PORT-RECORD.md section 32), so it is
# ESTABLISHED BY GRADING: the pipeline runs each stage against that stage's
# own captured boundary and a wrong base fails there, by name.

#: 116 names
NT_LEVEL_F = (
    "culu", "cuqu", "cutu", "dmfde", "dmfen", "dz8w", "exner", "ghti", "ghtl", "omg", "p8w",
    "pap", "paph", "pcps", "pcte", "pdmfdp", "pdmfup", "pdpmel", "pgeo", "pgeoh", "plglac",
    "plrain", "plu", "plude", "pmfd", "pmfdde_rate", "pmfdq", "pmfds", "pmflxr",
    "pmflxs", "pmfu", "pmfude_rate", "pmful", "pmfuq", "pmfus", "pqc", "pqd", "pqen",
    "pqenh", "pqi", "pqsen", "pqsenh", "pqte", "pqu", "pqv", "prsi", "prsl", "pt", "ptd", "pten",
    "ptenh", "ptenq", "ptent", "ptenu", "ptenv", "ptte", "ptu", "pu", "pud", "puen", "pum1",
    "puu", "pv", "pvd", "pven", "pverv", "pvm1", "pvol", "pvom", "pvu", "qc", "qc3d", "qcf", "qi",
    "qi3d", "qif", "qv", "qv3d", "qvf", "qvften", "qvftenz", "rho3d", "rqccuten",
    "rqicuten", "rqvcuten", "rthcuten", "rucuten", "rvcuten", "t", "t3d", "tf", "thften",
    "thftenz", "u", "u3d", "uf", "v", "v3d", "vf", "w", "zmfdu", "zmfdus", "zmfdv", "zmfuu",
    "zmfuus", "zmfuv", "zqenh", "zqp1", "zqq", "zqsat", "ztenh", "ztenu", "ztenv", "ztp1",
    "ztt", "zuv2",
)
#: 2 names
NT_LEVEL_I = (
    "culab", "klab",
)
#: 30 names
NT_SURFACE_F = (
    "delt_out", "dx", "hfx", "pmfub", "prain", "pratec", "prfl", "prsfc", "pssfc", "qfx",
    "raincv", "rn", "scale_fac", "scale_fac2", "upbl", "wbase", "wbase_o", "wup",
    "xland", "zcape", "zcape1", "zcape2", "zdhpbl", "zheat", "zmfub", "zmfub1", "zprecc",
    "ztau_o", "ztaubl", "ztauc",
)
#: 19 names
NT_SURFACE_I = (
    "cubot_o", "cutop_o", "ictop0", "idtop", "kcbot", "kctop", "kctop0", "kdpl",
    "kdpl_o", "kdtop", "klwmin", "ktype", "ktype_o", "ldcum", "ldcum_o", "lddraf",
    "lndj", "loddraf", "slimsk",
)


#: ``ntiedtke_cutypen`` takes one float pointer and slices it internally
#: into ptu, pqu, plu, dh, dhen, kup, vptu, vten, zbuo and abuoy, plus one
#: int slab for klab.
#:
#: TEN, not the eleven the kernel's own signature comment claimed. The body
#: uses slots 0-9 and the graded parity test allocates ten, so the comment
#: was stale by one whole level array -- 3.5 MiB at the capped tile, in a
#: campaign where 50 MiB has to earn itself. Corrected in the .cu, and
#: test_ntiedtke_workspace derives the count from the kernel body rather
#: than trusting either number.
NT_CUTYPEN_SCR_SLOTS = 10
NT_CUTYPEN_SCR_I_SLOTS = 1


#: STORAGE IDENTITY, which is not the same question as seeding.
#:
#: NT_SEEDS answers "where does this array's value come from before any
#: stage writes it" -- a class-1 question. This answers "is this a second
#: name for an array that already exists", which is about the ALLOCATION,
#: and the two overlap without coinciding.
#:
#: ``cutu``/``cuqu``/``culu``/``culab`` are the difference and they are why
#: this table exists. They are cutypen's OUTPUT parameters, so they are
#: never read before written and NT_SEEDS has nothing to say about them --
#: but ``cumastrn:490`` passes ``ptu, pqu, ilab, plu``, the very arrays
#: cuinin filled, and cutypen UPDATES THEM IN PLACE. The port renamed them
#: only so one fixture file could record cuinin's answer and cutypen's
#: separately.
#:
#: Allocating them apart would have left every stage after cutypen reading
#: cuinin's plume instead of cutypen's. Found by walking the pipeline: the
#: boundary grade for cutypen failed on three of four fields, and ``culu``
#: passed by luck because plu is zero where it was compared.
#: The OWNER is the name the array is first written under, in call order.
NT_ALIASES = {
    "cutu": "ptu", "cuqu": "pqu", "culu": "plu", "culab": "klab",
    # cutypen writes ldcum and ictop0; mfub UPDATES ldcum in place -- it
    # can clear it at :536 -- and later stages read ictop0. Neither is
    # ever a const read, so the class-1 analysis in NT_SEEDS is silent
    # about them and only storage identity has anything to say.
    "ldcum": "ldcum_o", "ictop0": "cutop_o",
    # cuascn's kctop0 IS ictop0: cumastrn passes cutypen's cloud top in
    # and cuascn may lower it. Another in/out the const scan cannot see --
    # it arrived as 0 against the oracle's 46 at cuascn's boundary.
    "kctop0": "cutop_o",
    # mfub's first-guess mass flux. cumastrn calls it pmfub throughout;
    # the port named mfub's output zmfub so the fixture could record the
    # first guess separately from the closure's rescale.
    #
    # INVISIBLE TO THE DATAFLOW GATE, and that is a gap worth naming: the
    # gate flags a CONST parameter read before any stage writes it, and
    # pmfub is intent(inout) in cuascn -- read AND written -- so it never
    # enters the const scan. Every in/out parameter in the port has the
    # same hole. What caught it was the walk.
    "pmfub": "zmfub",
    # cumastrn:596 passes idtop for cudlfsn's kdtop dummy. Another in/out
    # the const scan cannot see, and the walk found it three stages later:
    # updraft_scale zeroes the downdraft arrays above idtop (:800-813), so
    # an unseeded idtop left pdmfdp non-zero at one level -- 5.5e-07 where
    # WRF has exactly 0, on a stage whose own parity test is green because
    # it does not grade the downdraft arrays it zeroes.
    "idtop": "kdtop",
    # cumastrn:922 passes ptte and pqte for cudtdqn's ptent/ptenq dummies.
    # cudtdqn ACCUMULATES into them (section 17), so allocating them apart
    # meant the convective heating went into an array nothing downstream
    # read: the KE dissipation added to the forcing, the post-conversion
    # differenced the forcing against itself, and post_run returned five
    # of eight tendency fields wrong. Five graded boundaries failed on one
    # missing alias.
    "ptent": "ptte", "ptenq": "pqte",
    # cu_ntiedtke_run is CALLED with pu=uf, pv=vf, pt=tf, pqv=qvf,
    # pqc=qcf, pqi=qif (module_cu_ntiedtke.F:205-226), so the
    # post-conversion updates the very arrays post_run then differences
    # against the driver's untouched state. Six names, one storage each.
    "pt": "tf", "pqv": "qvf", "pqc": "qcf", "pqi": "qif",
    "pu": "uf", "pv": "vf",
}

#: Per-STAGE parameter aliases, for names that mean different arrays in
#: different kernels.
#:
#: ``NT_ALIASES`` above is keyed by parameter name alone, which works only
#: while a name means one thing everywhere. ``pmfu``/``pmfd`` do not: eight
#: stages take the updraft and downdraft mass fluxes and mean the arrays of
#: those names, and ONE -- cududvn -- takes the same dummy names and is
#: handed something else.
#:
#: cumastrn:1026 calls cududvn with ``zmfuus``/``zmfdus``, the pair the
#: momentum rescale at :996-1016 produces by applying its own per-column
#: limiter. The kernel's header says so ("pmfu/pmfd MUST BE THE SCALED
#: PAIR ... the unscaled pair would be wrong on exactly the columns the
#: rescaling touched") and the binding did not, because binding by name
#: silently found the slots literally called pmfu and pmfd. The requirement
#: was written down at the call site and agreed with nothing.
#:
#: MEASURED, not reasoned. On a live 4.5 km column at forecast minute 25
#: the limiter binds at zmfs = 0.33333334 and our convective momentum came
#: out 2.9999996 to 3.0000032 times WRF's across all twelve affected words,
#: with rthcuten 0.1-0.4% off on the same levels because cumastrn:1030-1052
#: builds the KE-dissipation heating out of the momentum increment. On the
#: same 64-column set, a column's momentum differs from WRF IF AND ONLY IF
#: its zmfs < 1, in both directions, and where it differs the ratio is
#: exactly 1/zmfs.
#:
#: WHY THE FIXTURE NEVER SAW IT: the rescale's cap does not bind anywhere in
#: the 18 analytic columns -- the closest reaches 0.5076 of it -- so the
#: stage is graded at max_ulp == 0 with its only consequence never taken.
#: ``tests/test_ntiedtke_dead_workspace_writes.py`` is the structural gate;
#: it fails on the unfixed tree because zmfuus/zmfdus are then written by
#: one stage and read by none.
NT_STAGE_ALIASES = {
    ("ntiedtke_cududvn", "pmfu"): "zmfuus",
    ("ntiedtke_cududvn", "pmfd"): "zmfdus",
}

#: Every workspace slot name, for callers that need to resolve an alias
#: without allocating a workspace.
NT_SLOT_NAMES = frozenset(
    NT_LEVEL_F + NT_LEVEL_I + NT_SURFACE_F + NT_SURFACE_I)


def nt_resolve(name: str, universe=NT_SLOT_NAMES) -> str:
    """The array a kernel parameter name actually denotes.

    Follows ``NT_ALIASES`` and ``NT_SEEDS``' prose ``alias of X`` rows to
    the end of the chain, because an alias may name another alias -- ``rn``
    is ``zprecc``, and nothing stops a future row pointing at ``rn``. Stops
    at the last name that is a real slot, so a dummy naming something the
    workspace does not hold resolves to itself and fails loudly at bind.

    ``universe`` exists so ``NtWorkspace.resolve`` can pass the arrays it
    actually allocated while a static caller passes ``NT_SLOT_NAMES``.
    """
    seen = [name]
    while True:
        direct = NT_ALIASES.get(seen[-1])
        m = _ALIAS_RE.match(NT_SEEDS.get(seen[-1], ""))
        if direct is None and not m:
            return seen[-1]
        nxt = direct if direct is not None else m.group(1)
        if nxt in seen:
            raise ValueError(f"alias cycle: {' -> '.join(seen)} -> {nxt}")
        if nxt not in universe:
            return seen[-1]
        seen.append(nxt)


def nt_tile_columns(ncol: int, multiprocessor_count: int) -> int:
    """Columns held in flight: enough to fill the card, and no more.

    THE SAME COLUMN COUNT GRELL-FREITAS USES, and sourced from GF's own
    constants rather than restated. Section 26 decided the cap on a
    measurement -- uncapped the workspace is a 2.1 GiB allocation on the
    profile domain against a card whose peak is already 10.3 of 15.92;
    capped it is 351 MiB -- and it decided it as GF's tile, 17,920 columns
    on this box.

    THE FIRST VERSION OF THIS GOT HALF THAT, and the error is worth
    keeping. Written as ``SMs * BLOCKS_PER_SM * NT_TPB`` by analogy with
    GF's formula, it gives 8,960, because NT's block is 32 threads and
    GF's is 64. The same ARITHMETIC is not the same TILE. Every VRAM
    figure recorded since section 26 would have halved, and the formula
    would still have matched the allocator exactly -- so the workspace
    self-check would have passed and only the recorded decision would
    have been contradicted.

    A domain smaller than one tile never allocates a whole tile, which is
    what makes this safe to call from the memory gate on any grid. Takes
    the SM count rather than querying the device, so preflight's
    ABSENT-CARD path gets the same answer a live query would.
    """
    from gpuwm.core.gf import GF_BLOCK, GF_TILE_BLOCKS_PER_SM

    tile = int(multiprocessor_count) * GF_TILE_BLOCKS_PER_SM * GF_BLOCK
    return int(min(int(ncol), tile))


def nt_aliased_names() -> frozenset[str]:
    """Names that are second names for an array, so allocate nothing.

    Module-level because :class:`NtWorkspace` and :func:`nt_workspace_bytes`
    must agree exactly: a byte count computed from a different alias set
    than the allocator uses is the census-apart-from-the-thing-it-counts
    failure that put 75 where 89 belonged (docs/ntiedtke/PORT-RECORD.md section 33).
    """
    return frozenset(
        n for n in (tuple(NT_LEVEL_F) + tuple(NT_LEVEL_I)
                    + tuple(NT_SURFACE_F) + tuple(NT_SURFACE_I))
        if _ALIAS_RE.match(NT_SEEDS.get(n, "")) or n in NT_ALIASES)


def nt_workspace_bytes(nz: int, columns: int) -> int:
    """What :class:`NtWorkspace` will allocate, without allocating it.

    Derived from the SAME name tables and the same alias set the
    constructor uses, so the two cannot disagree. Checked against a real
    allocation in tests/test_ntiedtke_workspace.py rather than trusted.
    """
    rows = int(nz) + NtWorkspace.HALO_BELOW + NtWorkspace.HALO_ABOVE
    columns = int(columns)
    aliased = nt_aliased_names()
    level = sum(1 for n in tuple(NT_LEVEL_F) + tuple(NT_LEVEL_I)
                if n not in aliased)
    surface = sum(1 for n in tuple(NT_SURFACE_F) + tuple(NT_SURFACE_I)
                  if n not in aliased)
    scratch = NT_CUTYPEN_SCR_SLOTS + NT_CUTYPEN_SCR_I_SLOTS
    # Every array is 4 bytes wide: float32, or int32 for the label and
    # index carriers.
    return 4 * columns * ((level + scratch) * rows + surface)


class NtWorkspace:
    """Every array a chunk of columns needs, allocated once.

    ``ncol`` is the CHUNK width, not the domain width -- the tile is capped
    at :func:`nt_tile_columns` and a domain is walked in chunks, because the
    uncapped allocation measured +1,042 MiB against Grell-Freitas on the
    profile domain and the capped one measures -238 (section 26).
    """

    #: Level arrays are (nz + 2, ncol): one halo row below so a 0-based
    #: stage can be handed ``arr[1:]``, and one above for the interface
    #: arrays that read index nz.
    HALO_BELOW = 1
    HALO_ABOVE = 1

    def __init__(self, ncol: int, nz: int):
        import cupy as cp

        if ncol <= 0 or nz <= 0:
            raise ValueError(f"ncol and nz must be positive, got {ncol}, {nz}")
        self.ncol = int(ncol)
        self.nz = int(nz)
        rows = self.nz + self.HALO_BELOW + self.HALO_ABOVE
        self._level = {}
        self._surface = {}
        # ALIASES ARE NOT ALLOCATED.  ``pten`` and ``ztp1`` are one array
        # under two of the reference's dummy names, so allocating both
        # would be dead memory AND a second storage that bind() never
        # reaches -- the kind of buffer that is only ever discovered by
        # someone counting bytes. This is guarantee 2 of section 7,
        # landing where it said it would: with the assembler.
        #
        # Copies and zeroed arrays DO get storage: a snapshot is a second
        # array by definition.
        self._aliased = nt_aliased_names()
        for names, dtype in ((NT_LEVEL_F, cp.float32), (NT_LEVEL_I, cp.int32)):
            for n in names:
                if n in self._aliased:
                    continue
                self._level[n] = cp.zeros((rows, self.ncol), dtype=dtype)
        for names, dtype in ((NT_SURFACE_F, cp.float32),
                             (NT_SURFACE_I, cp.int32)):
            for n in names:
                if n in self._aliased:
                    continue
                self._surface[n] = cp.zeros(self.ncol, dtype=dtype)

        # cutypen's multi-slot scratch, and it is NOT small: twelve
        # level-sized arrays, an 11% addition to the whole workspace. It
        # was invisible to the name census because the kernel takes ONE
        # pointer and slices it internally -- exactly the shape that makes
        # a memory figure an undercount, and section 26's 75 was one.
        self.scr = cp.zeros((NT_CUTYPEN_SCR_SLOTS, rows, self.ncol),
                            dtype=cp.float32)
        self.scr_i = cp.zeros((NT_CUTYPEN_SCR_I_SLOTS, rows, self.ncol),
                              dtype=cp.int32)

    def release(self) -> int:
        """Drop every array, returning the bytes handed back.

        A relocation drops the physics driver, but the driver and this
        workspace's adapter reference EACH OTHER (PhysicsDriver holds
        cumulus_callable, NewTiedtke.bind_driver holds the driver back),
        and reference counting cannot free a cycle.  CPython's cyclic
        collector can, but its thresholds count PYTHON objects and a
        433.2 MiB CuPy array is one small object -- so the cycle sits
        uncollected while holding a third of a gigabyte of VRAM.

        MEASURED, matched pair on prepared_nt16_hafs, 30 forecast
        minutes, identical but for the relocation surface: four
        NtWorkspace constructions against the control's two, and the
        pool's live floor ratcheting 4425 -> 6923 MiB against the
        control's 4425 -> 4451.  The live bytes never came back between
        constructions, which is the signature of a retained cycle rather
        than of churn.

        Idempotent, so the release path can run without knowing whether
        an earlier one already fired.
        """
        freed = self.bytes_allocated() if self._level or self._surface else 0
        self._level = {}
        self._surface = {}
        self.scr = None
        self.scr_i = None
        return int(freed)

    # -- resolution ------------------------------------------------------
    def resolve(self, name: str) -> str:
        """Follow NT_SEEDS' alias chain to the array that actually exists.

        The reference passes one array under different dummy names at
        different call sites; the workspace holds ONE array and this is
        where the second name is resolved.

        Delegates so that a caller without an allocated workspace -- the
        structural gate in tests/test_ntiedtke_dead_workspace_writes.py --
        resolves names by the SAME code rather than by a second copy of
        this walk. Two copies of an alias resolver is how the aliases and
        the thing checking them stop agreeing, which is the failure this
        module has hit twice already.
        """
        return nt_resolve(name, self._level.keys() | self._surface.keys())

    def bind(self, name: str, base: int):
        """The array a kernel parameter should receive.

        ``base`` is the stage's level-indexing base: 1 hands over the whole
        allocation, 0 hands over ``arr[1:]``. Surface arrays ignore it --
        they have no level axis and cannot be shifted, which is why passing
        the base to every bind is safe rather than merely convenient.
        """
        target = self.resolve(name)
        if target in self._surface:
            return self._surface[target]
        if target not in self._level:
            raise KeyError(
                f"{name!r} resolves to {target!r}, which is not an array in "
                f"the workspace. Add it to NT_LEVEL_* / NT_SURFACE_*, or "
                f"declare what it aliases in NT_SEEDS.")
        if base == 1:
            return self._level[target]
        if base == 0:
            return self._level[target][self.HALO_BELOW:]
        raise ValueError(f"level base must be 0 or 1, got {base!r}")

    def levels(self, name: str):
        """The nz real levels of an array, 0-based, for comparison."""
        target = self.resolve(name)
        if target in self._surface:
            return self._surface[target]
        return self._level[target][self.HALO_BELOW:self.HALO_BELOW + self.nz]

    def bytes_allocated(self) -> int:
        """What this workspace costs, MEASURED rather than computed.

        Every VRAM figure in this campaign has been a projection. This is
        the first that can be read off an allocation.
        """
        # ``scr``/``scr_i`` are None after release(), so a caller that
        # prices a released workspace reads zero rather than raising --
        # the count is "what this holds", and a released one holds
        # nothing.
        return (sum(a.nbytes for a in self._level.values())
                + sum(a.nbytes for a in self._surface.values())
                + (self.scr.nbytes if self.scr is not None else 0)
                + (self.scr_i.nbytes if self.scr_i is not None else 0))

    def storage_census(self) -> dict:
        """How many DISTINCT buffers, level and surface.

        Section 26 priced the tile decision on "the 75 distinct level
        arrays the assembled scheme holds" -- an estimate made before an
        assembler existed to count them. This is the count.
        """
        return {"level": len(self._level),
                "surface": len(self._surface),
                "scratch_level_equivalents": (NT_CUTYPEN_SCR_SLOTS
                                              + NT_CUTYPEN_SCR_I_SLOTS),
                "aliased_away": len(self._aliased),
                "bytes": self.bytes_allocated()}


#: NT_SEEDS spells an alias in prose so the record and the code cannot
#: disagree about it; this is the one place that prose is parsed.
#:
#: "ALIAS OF" ONLY.  The first version matched "copy of" as well, which
#: collapsed ``zqq`` onto ``pqte`` and ``ztt`` onto ``ptte`` -- and those
#: are SNAPSHOTS, not other names for the same storage. The post-conversion
#: computes ``ptte - ztt``; with the two collapsed it would have computed
#: ``ptte - ptte`` and returned zero convective heating on every column,
#: finite and plausible. Found by review asking an unrelated question
#: about the allocation count.
#:
#: The distinction is the one NT_SEEDS already draws in prose:
#:   alias   the reference passes ONE array under a second dummy name
#:   copy    the assembler takes a snapshot into a SECOND array
#:   zeroed  the assembler initialises it
#: Only the first is a shared storage.
_ALIAS_RE = re.compile(r"alias of (\w+)")


# ===========================================================================
# Every stage's argument list, in order
# ===========================================================================
# GENERATED from ntiedtke.cu and GATED against it -- a stage whose kernel
# gains, loses or reorders a parameter fails
# test_ntiedtke_stage_signature.py rather than being launched with the
# arguments shifted by one, which CUDA would accept and which would read
# another array's memory as a float.
#
# The names are the kernels' own parameter names, so the assembler binds by
# name: NtWorkspace resolves each to a buffer, NT_SEEDS resolves the cases
# where the reference gives one array two names, and the scalars come from
# the run context. Nothing here is a hand-maintained mapping, which is the
# only reason a 700-argument table is safe to have.

NT_STAGE_SIGNATURE = {
    "ntiedtke_adjust": (
        "ldcum", "loddraf", "idtop", "kctop", "kcbot", "paph", "pqen",
        "pmfu", "pmfuq", "pmful", "pmfd", "pmfds", "pmfdq", "plude",
        "pdmfup", "pdmfdp", "pmfdde_rate", "pmfude_rate", "pmflxr",
        "pmflxs", "prsfc", "pssfc", "ncol", "klev", "ztmst", "cp", "rd",
        "rv", "xlv", "xlf", "grav", "expect_tpb", "expect_nblocks",
        "geom_report", "order_report", "ticket"
    ),
    "ntiedtke_closure": (
        "ldcum", "ktype", "kcbot", "kctop", "kdpl", "loddraf", "lndj",
        "wup", "zmfub", "zdhpbl", "scale_fac", "scale_fac2", "pgeoh",
        "paph", "pap", "pgeo", "pten", "pqen", "ptenh", "pqenh", "ptu",
        "pqu", "plu", "pmfu", "ptd", "pqd", "ptte", "pqte", "pmfd", "pmfds",
        "pmfdq", "pdmfdp", "pmfdde_rate", "zheat", "zcape", "zcape1",
        "zcape2", "ztauc", "ztaubl", "ztau_o", "upbl", "zmfub1",
        "tiedtke_closure", "ncol",
        "nz", "dt", "cp", "rd", "rv", "xlv", "xlf", "grav", "expect_tpb",
        "expect_nblocks", "geom_report", "order_report", "ticket"
    ),
    "ntiedtke_cloud_depth": (
        "ldcum", "kcbot", "kctop", "paph", "pdmfup", "ktype", "ictop0",
        "prfl", "pmfd", "pmfds", "pmfdq", "pdmfdp", "pdpmel", "ncol",
        "klev", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
    "ntiedtke_convert": (
        "tf", "qvf", "uf", "vf", "omg", "ghtl", "ghti", "prsl", "qvftenz",
        "thftenz", "ztp1", "zqp1", "zqsat", "pgeo", "pgeoh", "pum1", "pvm1",
        "pverv", "ptte", "pqte", "ncol", "nz", "cp", "rd", "rv", "xlv",
        "xlf", "grav", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
    "ntiedtke_cuascn": (
        "pten", "pqen", "pqsen", "pgeo", "pgeoh", "pap", "paph", "pverv",
        "lndj", "kdpl", "wbase", "ptenh", "pqenh", "ptu", "pqu", "plu",
        "pmfu", "pmfus", "pmfuq", "pmful", "plude", "pdmfup", "plglac",
        "pmfude_rate", "klab", "ldcum", "ktype", "kcbot", "kctop", "kctop0",
        "pmfub", "wup", "ncol", "klev", "ztmst", "llo3", "cp", "rd", "rv",
        "xlv", "xlf", "grav", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
    "ntiedtke_cuddrafn": (
        "lddraf", "ptenh", "pqenh", "pgeo", "pgeoh", "paph", "pmfu", "ptd",
        "pqd", "pmfd", "pmfds", "pmfdq", "pdmfdp", "pmfdde_rate", "prfl",
        "ncol", "klev", "cp", "rd", "rv", "xlv", "xlf", "grav",
        "expect_tpb", "expect_nblocks", "geom_report", "order_report",
        "ticket"
    ),
    "ntiedtke_cudlfsn": (
        "ldcum", "kcbot", "kctop", "ptenh", "pqenh", "pten", "pqsen",
        "pgeo", "pgeoh", "paph", "ptu", "pqu", "pmfub", "ptd", "pqd",
        "pmfd", "pmfds", "pmfdq", "pdmfdp", "prfl", "kdtop", "lddraf",
        "ncol", "klev", "cp", "rd", "rv", "xlv", "xlf", "grav",
        "expect_tpb", "expect_nblocks", "geom_report", "order_report",
        "ticket"
    ),
    "ntiedtke_cudtdqn": (
        "ldcum", "paph", "pten", "plglac", "plude", "pmfus", "pmfds",
        "pmfuq", "pmfdq", "pmful", "pdmfup", "pdmfdp", "pdpmel", "ptent",
        "ptenq", "pcte", "ncol", "klev", "ktopm2", "cp", "rd", "rv", "xlv",
        "xlf", "grav", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
    "ntiedtke_cududvn": (
        "ldcum", "ktype", "kcbot", "paph", "puen", "pven", "pmfu", "pmfd",
        "puu", "pud", "pvu", "pvd", "ptenu", "ptenv", "zmfuu", "zmfuv",
        "zmfdu", "zmfdv", "ncol", "klev", "ktopm2", "cp", "rd", "rv", "xlv",
        "xlf", "grav", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
    "ntiedtke_cuflxn": (
        "kcbot", "kctop", "kdtop", "lndj", "pten", "ptenh", "pqenh", "paph",
        "pap", "pqen", "pgeoh", "ldcum", "lddraf", "ktype", "pmfu", "pmfd",
        "pmfus", "pmfds", "pmfuq", "pmfdq", "pmful", "plude", "plglac",
        "pdmfup", "pdmfdp", "pmfdde_rate", "pqsen", "pdpmel", "pmflxr",
        "pmflxs", "prain", "ncol", "klev", "ztmst", "cp", "rd", "rv", "xlv",
        "xlf", "grav", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
    "ntiedtke_cuinin": (
        "pten", "pqen", "pqsen", "puen", "pven", "pverv", "pgeo", "paph",
        "pgeoh", "ptenh", "pqenh", "pqsenh", "ptu", "pqu", "ptd", "pqd",
        "puu", "pvu", "pud", "pvd", "plu", "klab", "klwmin", "ncol", "nz",
        "cp", "rd", "rv", "xlv", "xlf", "grav", "expect_tpb",
        "expect_nblocks", "geom_report", "order_report", "ticket"
    ),
    "ntiedtke_cutypen": (
        "pqen", "ptenh", "pqenh", "pgeoh", "paph", "pgeo", "pqsen", "pap",
        "pten", "hfx", "qfx", "cutu", "cuqu", "culu", "culab", "scr",
        "scr_i", "ldcum_o", "ktype_o", "cubot_o", "cutop_o", "kdpl_o",
        "wbase_o", "ncol", "nz", "cp", "rd", "rv", "xlv", "xlf", "grav",
        "expect_tpb", "expect_nblocks", "geom_report", "order_report",
        "ticket"
    ),
    "ntiedtke_ke_dissipation": (
        "ldcum", "kctop", "paph", "puen", "pven", "ztenu", "ztenv", "pvom",
        "pvol", "ptte", "zuv2", "ncol", "klev", "cp", "rd", "rv", "xlv",
        "xlf", "grav", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
    "ntiedtke_mfub": (
        "ptte", "pqte", "paph", "puen", "pven", "ptu", "pqu", "plu",
        "ztenh", "zqenh", "ktype", "kcbot", "lndj", "ldcum", "zdhpbl",
        "upbl", "zmfub", "tiedtke_closure",
        "ncol", "nz", "dt", "cp", "rd", "rv", "xlv", "xlf",
        "grav", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
    "ntiedtke_midlevel": (
        "pten", "pqen", "pqsen", "pverv", "pgeo", "pgeoh", "ldcum", "ktype",
        "kcbot", "klab", "plrain", "pmfu", "pmfub", "ptu", "pqu", "plu",
        "pmfus", "pmfuq", "pmful", "pdmfup", "dmfen", "dmfde", "ncol", "nz",
        "cp", "rd", "rv", "xlv", "xlf", "grav", "expect_tpb",
        "expect_nblocks", "geom_report", "order_report", "ticket"
    ),
    "ntiedtke_momentum_profile": (
        "ldcum", "ktype", "kcbot", "kctop", "kdpl", "idtop", "puen", "pven",
        "pmfu", "pmfd", "pmfude_rate", "pmfdde_rate", "puu", "pvu", "pud",
        "pvd", "ncol", "klev", "expect_tpb", "expect_nblocks",
        "geom_report", "order_report", "ticket"
    ),
    "ntiedtke_momentum_rescale": (
        "ldcum", "kctop", "paph", "pmfu", "pmfd", "zmfuus", "zmfdus",
        "ncol", "klev", "ztmst", "cp", "rd", "rv", "xlv", "xlf", "grav",
        "expect_tpb", "expect_nblocks", "geom_report", "order_report",
        "ticket"
    ),
    "ntiedtke_post_conversion": (
        "pcte", "ztp1", "ptte", "ztt", "pqte", "zqq", "zqp1", "qcf", "qif",
        "uf", "vf", "pvom", "pvol", "prsfc", "pssfc", "pqc", "pqi", "pt",
        "pqv", "pu", "pv", "zprecc", "ncol", "klev", "delt", "expect_tpb",
        "expect_nblocks", "geom_report", "order_report", "ticket"
    ),
    "ntiedtke_post_run": (
        "exner", "qv", "qc", "qi", "t", "u", "v", "tf", "qvf", "qcf", "qif",
        "uf", "vf", "rn", "rthcuten", "rqvcuten", "rqccuten", "rqicuten",
        "rucuten", "rvcuten", "raincv", "pratec", "ncol", "klev", "stepcu",
        "dt", "expect_tpb", "expect_nblocks", "geom_report", "order_report",
        "ticket"
    ),
    "ntiedtke_prep": (
        "t3d", "qv3d", "qc3d", "qi3d", "u3d", "v3d", "pcps", "dz8w",
        "rho3d", "p8w", "w", "qvften", "thften", "xland", "hfx", "qfx",
        "dx", "prsl", "ghtl", "omg", "tf", "qvf", "qcf", "qif", "uf", "vf",
        "qvftenz", "thftenz", "prsi", "ghti", "slimsk", "scale_fac",
        "scale_fac2", "delt_out", "ncol", "nz", "dt", "stepcu", "itimestep",
        "grav", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
    "ntiedtke_updraft_scale": (
        "loddraf", "kcbot", "kctop", "zmfub1", "zmfub", "paph", "ldcum",
        "ktype", "idtop", "pmfu", "pmfus", "pmfuq", "pmful", "pdmfup",
        "plude", "pmfude_rate", "pmfd", "pmfds", "pmfdq", "pdmfdp",
        "pmfdde_rate", "ncol", "klev", "ztmst", "cp", "rd", "rv", "xlv",
        "xlf", "grav", "expect_tpb", "expect_nblocks", "geom_report",
        "order_report", "ticket"
    ),
}


#: The descriptor arguments NtStages.launch appends. Asserted to be the
#: last five parameters of every kernel by test_ntiedtke_stage_signature,
#: so args_for can drop them by name without counting.
_GEOMETRY_TAIL = ("expect_tpb", "expect_nblocks", "geom_report",
                  "order_report", "ticket")


# ===========================================================================
# The level-indexing base, as a PRIOR
# ===========================================================================
# NOT a derivation. Four attempts to settle this statically gave four
# answers (docs/ntiedtke/PORT-RECORD.md sections 32 and 35), the fourth because
# instrumenting the graded launches measures the ALLOCATION and not the
# indexing -- ``ntiedtke_convert`` is handed 51 rows and uses 50 of them
# from index 0.
#
# What measures the right thing is which loop variables are used as level
# indices, and that says three stages walk k = 0..nz-1. This table records
# that, and the pipeline's per-stage boundary grade CONFIRMS OR REFUTES it:
# a wrong base fails at that stage by name, which
# test_ntiedtke_workspace.py demonstrates on a real kernel rather than
# asserting.
NT_LEVEL_BASE = {
    "ntiedtke_prep": 0,
    "ntiedtke_convert": 0,
    "ntiedtke_cuinin": 0,
}


def nt_level_base(stage: str) -> int:
    """0 or 1, defaulting to 1 -- the convention 18 of 21 stages use."""
    return NT_LEVEL_BASE.get(stage, 1)


class NtPipeline:
    """One chunk of columns through the whole scheme.

    The domain is walked in chunks of at most :func:`nt_tile_columns`
    columns, because the uncapped allocation measures 2.4 GiB on the
    profile domain against a 15.92 GiB card whose peak is already 11.3
    (sections 26 and 33).

    THE ASSEMBLER'S OWN WORK, as opposed to the stages': three zeroings at
    ``cu_ntiedtke_run:257-259``, two snapshots at ``:274``/``:276``, one
    pair of copies at ``cumastrn:1019-1024``, and the ``llo3`` reduction.
    Each is named in the ownership manifest as "the assembler", so none of
    them can be mistaken for a missing transcription.
    """

    def __init__(self, *, ncol: int, nz: int, dt: float, stepcu: int = 1,
                 itimestep: int = 2, ktopm2: int = 2,
                 cp: float = 1004.5, rd: float = 287.0, rv: float = 461.6,
                 xlv: float = 2.5e6, xlf: float = 3.5e5,
                 grav: float = 9.81, base=None,
                 tiedtke_closure: bool = False):
        import numpy as np

        self.w = NtWorkspace(ncol=ncol, nz=nz)
        self.geometry = NtLaunchGeometry(ncol=ncol, nz=nz)
        self.stages = NtStages(self.geometry)
        self._base = dict(NT_LEVEL_BASE if base is None else base)
        self._delt = float(dt) * int(stepcu)
        self.scalars = {
            "ncol": np.int32(ncol),
            "nz": np.int32(nz),
            "klev": np.int32(nz),
            "stepcu": np.int32(stepcu),
            "itimestep": np.int32(itimestep),
            # ktopm2 = 2 unconditionally at cumastrn:2877 -- established in
            # section 20, and re-derived rather than remembered.
            "ktopm2": np.int32(ktopm2),
            "llo3": np.int32(0),          # set per chunk by _reduce_llo3
            "dt": np.float32(dt),
            # cu_physics = 6 (Tiedtke) inside the New Tiedtke kernels: a
            # FIXED ztau = 2400 s in place of ztauc*scale_fac (modC,
            # module_cu_tiedtke.F:105) and the MOISTURE-CONVERGENCE deep
            # first guess in place of the geometric 0.1*zmfmax (modB,
            # :855-866).  Those two substitutions are the whole difference
            # between the schemes' deep closures.  0 keeps cu16 bit-exact.
            "tiedtke_closure": np.int32(bool(tiedtke_closure)),
            "delt": np.float32(self._delt),
            "ztmst": np.float32(self._delt),
            "cp": np.float32(cp), "rd": np.float32(rd), "rv": np.float32(rv),
            "xlv": np.float32(xlv), "xlf": np.float32(xlf),
            "grav": np.float32(grav),
        }

    def base(self, stage: str) -> int:
        return self._base.get(stage, 1)

    def run_chunk(self) -> None:
        """The whole scheme, once, over this pipeline's columns."""
        _nt_walk(self)

    # -- one stage -------------------------------------------------------
    def args_for(self, stage: str) -> tuple:
        """Bind a kernel's parameters by NAME, in the source's own order.

        NT_STAGE_SIGNATURE is generated from the .cu and regenerated-and-
        compared every run, so an argument cannot silently shift by one --
        which CUDA would accept, and which would read another array's
        memory as a float.
        """
        base = self.base(stage)
        out = []
        for name in NT_STAGE_SIGNATURE[stage]:
            if name in _GEOMETRY_TAIL:
                continue              # NtStages.launch appends these
            if name in self.scalars:
                out.append(self.scalars[name])
            elif name == "scr":
                out.append(self.w.scr)
            elif name == "scr_i":
                out.append(self.w.scr_i)
            else:
                out.append(self.w.bind(
                    NT_STAGE_ALIASES.get((stage, name), name), base))
        return tuple(out)

    def run_stage(self, stage: str) -> None:
        self.stages.launch(stage, self.args_for(stage))

    # -- the assembler's own work ---------------------------------------
    def zero_run_head(self) -> None:
        """``cu_ntiedtke_run:257-259``: pcte, pvom and pvol to zero.

        pvom/pvol matter twice: they make accumulate == replace for
        momentum (section 17), and they are what :1019-1024 copies into
        ztenu/ztenv BEFORE cududvn writes them -- which is why those two
        are identically zero on all 5,292 fixture rows.
        """
        for name in ("pcte", "ptenu", "ptenv"):
            self.w.bind(name, 1).fill(0)

    def snapshot_forcing(self) -> None:
        """``:274`` and ``:276``: zqq = pqte, ztt = ptte.

        SNAPSHOTS, not aliases. The post-conversion computes ``ptte - ztt``
        so that only the CONVECTIVE increment escapes; sharing storage
        would make it identically zero (section 33).
        """
        self.w.bind("zqq", 1)[...] = self.w.bind("pqte", 1)
        self.w.bind("ztt", 1)[...] = self.w.bind("ptte", 1)

    def snapshot_momentum(self) -> None:
        """``cumastrn:1019-1024``: ztenu = pvom, ztenv = pvol, before
        cududvn writes them. One array copy, the caller's work."""
        self.w.bind("ztenu", 1)[...] = self.w.bind("pvom", 1)
        self.w.bind("ztenv", 1)[...] = self.w.bind("pvol", 1)

    def reduce_llo3(self) -> int:
        """The chunk-wide reduction, with the hoist's soundness CHECKED.

        In the reference ``llo3`` is a scalar that latches inside cuascn's
        descent: ``is = sum(klab(:, jk+1))`` each iteration, and
        ``if (is > 0) llo3 = .true.`` with no reset (cu_ntiedtke.F90:1903,
        :1975-2003). The port hoists it to a launch argument, which is
        exact only if it is already true at the FIRST iteration -- section
        12's precondition, measured 48 of 48 TRIGGERING columns on the
        fixture -- 8 per dx across all six, out of 108 columns.

        A chunk is a smaller population than the fixture, so the
        precondition is strictly harder to satisfy and CANNOT be inherited.
        It is therefore re-checked here, at the reduction's own scope: the
        value at the first iteration must equal the value the loop would
        have reached, or the hoist is unsound for this chunk and the
        pipeline refuses rather than passing a wrong scalar.
        """
        import cupy as cp

        klab = self.w.bind("klab", 1)
        nz = self.w.nz
        first = bool((klab[nz] > 0).any())         # klab(:, klev)
        ever = bool((klab[1:nz + 1] > 0).any())    # any level, any column
        if first != ever:
            raise ValueError(
                "llo3's hoist is unsound for this chunk: klab is zero at "
                "klev on every column but non-zero at some level below, so "
                "the reference's llo3 would latch DURING cuascn's descent "
                "and a single launch argument cannot represent it. Section "
                "12's precondition holds on the fixture (48 of 48 triggering "
                "columns) and "
                "does not hold here.")
        cp.cuda.Stream.null.synchronize()
        import numpy as np

        self.scalars["llo3"] = np.int32(1 if first else 0)
        return int(first)


#: cumastrn's call order.  DECLARED, not derived, for the reason
#: CALL_SITE_DEBTS is declared: when the orchestration lands the stages that
#: own :566-568 and :580-588, this sequence gains members and someone has to
#: notice rather than have the assertion re-derive itself around the change.
NT_CALL_ORDER = (
    "ntiedtke_prep",        # cu_ntiedtke_run head
    "ntiedtke_convert",     # the unit conversion
    "ntiedtke_cuinin",      # cumastrn:474
    "ntiedtke_cutypen",     # :490
    "ntiedtke_mfub",        # :500-541, the first-guess mass flux
    "ntiedtke_cuascn",      # :547
    "ntiedtke_cloud_depth",  # :562-590, the ktype flip
    "ntiedtke_cudlfsn",     # :596
    "ntiedtke_cuddrafn",    # :608
    "ntiedtke_closure",     # :620-745
    "ntiedtke_updraft_scale",  # :743-819, where zmfub1 is spent
    "ntiedtke_cuflxn",      # :822
    "ntiedtke_adjust",      # :833-919, the downdraft stability cap
    "ntiedtke_cudtdqn",     # :922
    "ntiedtke_momentum_profile",  # :927-995, writes puu/pvu/pud/pvd
    "ntiedtke_momentum_rescale",  # :996-1016, the only zcons consumer
    "ntiedtke_cududvn",     # :1026
    "ntiedtke_ke_dissipation",  # :1030-1056, the last arithmetic
    # And out of cumastrn entirely.
    "ntiedtke_post_conversion",  # cu_ntiedtke.F90:278-320
    "ntiedtke_post_run",         # module_cu_ntiedtke.F:502-527
)


def _nt_walk(pipeline) -> None:
    """One chunk through all twenty stages, with the assembler's own work.

    The interleaved steps are not stages and are named as such in the
    ownership manifest, so none can be mistaken for a missing
    transcription: llo3's chunk-wide reduction before cuascn, and the
    ztenu/ztenv copy before cududvn.
    """
    pipeline.zero_run_head()
    pipeline.run_stage("ntiedtke_prep")
    pipeline.run_stage("ntiedtke_convert")
    pipeline.snapshot_forcing()
    for stage in NT_CALL_ORDER:
        if stage in ("ntiedtke_prep", "ntiedtke_convert"):
            continue
        if stage == "ntiedtke_cuascn":
            pipeline.reduce_llo3()
        if stage == "ntiedtke_cududvn":
            pipeline.snapshot_momentum()
        pipeline.run_stage(stage)


# ===========================================================================
# The adapter -- cu_physics = 16
# ===========================================================================


class NewTiedtke:
    """Stateless ``cu_physics = 16`` callable for :class:`PhysicsDriver`.

    No ``nca_seconds``: New Tiedtke carries no NCA hold, so the Task-1
    attachment contract applies -- the rates replace the held cumulus
    tendencies wholesale and ``rainc`` is a per-due-call increment consumed
    once.  ``cudt_minutes`` is pinned to 0 by ``validate_run_config`` for
    exactly this reason: its RAINCV is ``rn/stepcu``, a per-call rate with
    no persistence, and a hold would reapply it every step.

    THE FOLD LIVES HERE AND NOWHERE ELSE.  WRF's cumulus driver pre-folds
    the radiative and boundary-layer lanes into the advective one for
    G3SCHEME and NTIEDTKESCHEME only::

        RTHFTEN = (RTHFTEN + RTHRATEN + RTHBLTEN) * pi
        RQVFTEN =  RQVFTEN + RQVBLTEN

    (``module_cumulus_driver.F:867-886``, read rather than inferred.)  Note
    the ``* pi``: the theta lane is multiplied by the Exner function, so
    what the scheme receives is a TEMPERATURE forcing, not a theta one.
    That is why ``cu_ntiedtke_pre_run`` takes ``thften`` into a scheme that
    works in T throughout.  A fold written as a plain sum -- which is how
    every prose description of it in this tree reads, including the
    registry's own warning -- would be wrong by a factor of Exner, roughly
    0.3 to 1.0 through the depth of the column.

    Grell-Freitas is NOT in WRF's fold list; it sums the lanes itself, so
    ArWen's dycore exports PURE ADVECTION and that export must not move.
    Doing the fold in the adapter is what keeps it that way.
    """

    def __init__(self) -> None:
        self._driver = None
        self._pipeline = None

    def bind_driver(self, driver) -> None:
        """Driver hook: gives the adapter the held radiation and PBL rates.

        Same optional-hook precedent as Grell-Freitas.  Without a driver
        the forcing lanes are zero, which is also what a radiation-off,
        PBL-off configuration would feed -- and ``validate_run_config``
        refuses ``cu_physics = 16`` without a PBL scheme, so the silent
        case is only reachable from a directly attached adapter in tests.
        """
        self._driver = driver

    def release(self) -> int:
        """Drop the cached pipeline AND the driver back-reference.

        ``bind_driver`` above completes a cycle -- PhysicsDriver holds
        ``cumulus_callable``, and this holds the driver -- so dropping
        ``state.physics`` at a relocation frees neither, and the 433.2
        MiB workspace behind the pipeline stays resident.  Clearing the
        back-reference is what makes the release deterministic; clearing
        the pipeline alone would still leave the cycle for the cyclic
        collector to find whenever it next runs.

        Returns the workspace bytes handed back, so the relocation
        receipt can carry a measured number rather than a claim.
        Idempotent.
        """
        freed = 0
        pipeline = self._pipeline
        if pipeline is not None:
            workspace = getattr(pipeline[1], "w", None)
            release = getattr(workspace, "release", None)
            if release is not None:
                freed = release()
        self._pipeline = None
        self._driver = None
        return int(freed)

    def _lane(self, name, nz, ny, nx):
        import cupy as cp

        lane = (None if self._driver is None
                else getattr(self._driver, name, None))
        if lane is None:
            return cp.zeros((nz, ny, nx), dtype=DTYPE)
        if lane.shape != (nz, ny, nx):
            raise ValueError(
                f"driver {name} must be [nz, ny, nx]={(nz, ny, nx)}, "
                f"got {lane.shape}")
        return lane

    def _for(self, ncol: int, nz: int, cfg):
        """One pipeline per chunk width, reused across steps.

        Rebuilding the workspace every call would re-allocate ~350 MiB of
        device memory at every cumulus step; the arrays are overwritten by
        the walk, so the same chunk width can reuse one.
        """
        key = (ncol, nz)
        if self._pipeline is None or self._pipeline[0] != key:
            # RELEASE THE OLD ONE FIRST.  Assigning the new tuple over the
            # old held both workspaces live across the constructor, and at
            # 433.2 MiB each that is 857.0 MiB where 433.2 was needed --
            # measured, not reasoned: build 17920, then build 17529 with
            # the first still referenced, and the CuPy pool reads 857.0.
            #
            # It fires EVERY cumulus step on EVERY domain, because the last
            # chunk of a domain is ragged whenever the tile does not divide
            # the column count, and it never does here: d01 is 38,870
            # columns and d02 is 71,289 against a 17,920 tile, so each
            # domain alternates a full width with its own remainder and the
            # single-slot cache thrashes between the two.
            #
            # This is the containment, not the cure.  The remaining cost is
            # a 433.2 MiB free-and-reallocate per step, which the pool
            # absorbs but which is still churn.  The cure is to allocate
            # the workspace ONCE at the tile width and launch over n <= W
            # columns, which needs NtPipeline to stop deriving three things
            # from one ncol -- the workspace extent, the launch geometry
            # and the kernels' own ncol scalar.  Deferred deliberately: it
            # touches the class every Phase 1 parity gate exercises, so it
            # wants its own bitwise proof rather than a ride on this one.
            self._pipeline = None
            self._pipeline = (key, NtPipeline(
                ncol=ncol, nz=nz, dt=float(_nt_model_dt(cfg)),
                stepcu=1, itimestep=2,
                tiedtke_closure=bool(
                    getattr(cfg, "ntiedtke_tiedtke_closure", False))))
        return self._pipeline[1]

    def __call__(self, *, atmosphere, fields, state, cfg):
        import cupy as cp

        from gpuwm.core.physics import CumulusResult

        nz, ny, nx = state.p.shape
        ncol = ny * nx

        def flat(a):
            """(nz, ny, nx) -> (nz, ncol), A VIEW where it can be.

            The kernels index ``a[k*ncol + i]``, level-major, which is what
            a C-contiguous (nz, ny, nx) reshapes to for free.  gf.py pays a
            real transpose here because its kernels are (ncol, nz).
            """
            return cp.ascontiguousarray(a).reshape(nz, ncol)

        exner = atmosphere["exner"]
        # THE FOLD.  module_cumulus_driver.F:879-880, including the * pi.
        # RTHRATEN is the SUM of the two radiative lanes, the same way
        # gf.py assembles it -- the driver holds longwave and shortwave
        # separately and WRF's RTHRATEN is their total.
        rthraten = (self._lane("rthratenlw", nz, ny, nx)
                    + self._lane("rthratensw", nz, ny, nx))
        thften = ((self._lane("gf_rthdynten", nz, ny, nx)
                   + rthraten
                   + self._lane("gf_rthblten", nz, ny, nx)) * exner)
        qvften = (self._lane("gf_rqvdynten", nz, ny, nx)
                  + self._lane("gf_rqvblten", nz, ny, nx))

        w = state.w
        p_int = atmosphere["p_interface"]

        driver_lev = {
            "t3d": atmosphere["temperature"], "qv3d": atmosphere["qv"],
            "qc3d": atmosphere["qc"], "qi3d": atmosphere["qi"],
            "u3d": atmosphere["u"], "v3d": atmosphere["v"],
            "pcps": atmosphere["pressure"], "dz8w": atmosphere["dz"],
            "rho3d": atmosphere["rho"], "exner": exner,
            "qvften": qvften, "thften": thften,
        }
        driver_iface = {"p8w": p_int, "w": w}
        surface = {"xland": fields["xland"], "hfx": fields["hfx"],
                   "qfx": fields["qfx"]}

        out_lev = {n: cp.zeros((nz, ncol), dtype=DTYPE) for n in
                   ("rthcuten", "rqvcuten", "rqccuten", "rqicuten",
                    "rucuten", "rvcuten")}
        out_sfc = {n: cp.zeros(ncol, dtype=DTYPE)
                   for n in ("raincv", "pratec")}

        flat_lev = {k: flat(v) for k, v in driver_lev.items()}
        flat_iface = {k: cp.ascontiguousarray(v).reshape(nz + 1, ncol)
                      for k, v in driver_iface.items()}
        flat_sfc = {k: cp.ascontiguousarray(v).reshape(ncol)
                    for k, v in surface.items()}
        dx_col = cp.full(ncol, DTYPE(cfg.dx), dtype=DTYPE)

        # ONE PIPELINE FOR THE WHOLE RUN, with the short last chunk padded.
        #
        # The domain is walked in chunks of at most nt_tile_columns, and the
        # last chunk of a domain is ragged whenever the tile does not divide
        # the column count -- which it never does here (38,870 and 71,289
        # against 17,920).  Sizing the pipeline to each chunk therefore
        # rebuilt a 433.2 MiB workspace twice per domain per step.  Sizing
        # it to the TILE instead and padding the tail builds it once for the
        # run, holds exactly the same 433.2 MiB resident, and is measured
        # bitwise identical.
        #
        # THE PAD IS A COPY OF THE CHUNK'S FIRST COLUMN, not zeros, and the
        # difference is the soundness argument rather than a preference.
        # Stages are per-column except for ONE cross-column operation --
        # llo3's chunk-wide OR (see NtPipeline.reduce_llo3) -- and OR over a
        # multiset that merely repeats a member it already contains cannot
        # change its value.  So the pad provably cannot move llo3.  Zeros
        # would not carry that argument: a zero column has zero pressure,
        # and whether the entry stages produce klab > 0 or a NaN from it is
        # a question about arithmetic on degenerate input rather than a
        # property of the reduction.  Both measure bitwise identical here;
        # only one of them is an argument.
        width = nt_tile_columns(ncol, _nt_multiprocessor_count())
        p = self._for(width, nz, cfg)
        for lo in range(0, ncol, width):
            hi = min(lo + width, ncol)
            n = hi - lo
            for name, a in flat_lev.items():
                buf = p.w.bind(name, 0)
                buf[:nz, :n] = a[:, lo:hi]
                if n < width:
                    buf[:nz, n:] = a[:, lo:lo + 1]
            for name, a in flat_iface.items():
                buf = p.w.bind(name, 0)
                buf[:nz + 1, :n] = a[:, lo:hi]
                if n < width:
                    buf[:nz + 1, n:] = a[:, lo:lo + 1]
            for name, a in flat_sfc.items():
                buf = p.w.bind(name, 1)
                buf[:n] = a[lo:hi]
                if n < width:
                    buf[n:] = a[lo]
            buf = p.w.bind("dx", 1)
            buf[:n] = dx_col[lo:hi]
            if n < width:
                buf[n:] = dx_col[lo]
            p.run_chunk()
            for name, dest in out_lev.items():
                dest[:, lo:hi] = p.w.bind(name, 0)[:nz, :n]
            for name, dest in out_sfc.items():
                dest[lo:hi] = p.w.bind(name, 1)[:n]

        def grid(a):
            return cp.ascontiguousarray(a.reshape(nz, ny, nx))

        return CumulusResult(
            rthcuten=grid(out_lev["rthcuten"]),
            rqvcuten=grid(out_lev["rqvcuten"]),
            rqccuten=grid(out_lev["rqccuten"]),
            rqicuten=grid(out_lev["rqicuten"]),
            rucuten=grid(out_lev["rucuten"]),
            rvcuten=grid(out_lev["rvcuten"]),
            # RAINCV is rn/stepcu, mm per call, consumed once.
            #
            # PRATEC IS COMPUTED AND DELIBERATELY NOT HANDED UP.  The
            # kernel forms both (module_cu_ntiedtke.F:505-508)::
            #
            #     raincv = rn / stepcu
            #     pratec = rn / (stepcu * dt)
            #
            # so pratec is raincv/dt -- the same quantity as a rate, with
            # no information raincv lacks -- and it stays under oracle
            # parity in the kernel either way.  The driver refuses pratec
            # without nca_seconds (physics.py:2433) because in ITS
            # vocabulary pratec is the HELD rate an NCA hold re-applies
            # across the steps it spans.  New Tiedtke has no hold: the
            # cudt law pins cudt_minutes = 0 for exactly this reason, so
            # stepcu is 1, raincv is rn, and the driver adds it once per
            # step.  There is no rate to hold.
            #
            # Grell-Freitas does the identical thing for the identical
            # reason (gf.py:319-321): its kernel computes pratec, its
            # adapter returns rainc = pratec * dt and nothing else.
            #
            # THE COST, STATED: driver.cu_pratec stays zero, and it is a
            # restart-serialized slot (restart.py:607).  That is a real
            # gap, shared with GF and predating this port -- not an
            # argument that pratec is meaningless.  It reaches no forecast
            # output: PRATEC carries Registry io 'r', restart-only and
            # never history (wrf_output_schema.py:568), so no wrfout field
            # moves.  Closing it means teaching the no-hold branch to
            # store a rate it never re-applies, which is a change to
            # machinery KF and GF both traverse, for a field nothing
            # currently reads.  Deferred on purpose, not overlooked.
            rainc=out_sfc["raincv"].reshape(ny, nx))


def _nt_multiprocessor_count() -> int:
    """SMs on the live device, or the profile's count when there is none."""
    try:
        import cupy as cp

        return int(cp.cuda.Device().attributes["MultiProcessorCount"])
    except Exception:                                          # noqa: BLE001
        from gpuwm.core.preflight import MEASURED_LOCAL_MEMORY_PROFILE

        return int(MEASURED_LOCAL_MEMORY_PROFILE.multiprocessor_count)


def _nt_model_dt(cfg) -> float:
    """WRF hands the scheme its model DT (module_cumulus_driver.F:1028).

    The outer clock when the case integrates internal substeps, which is
    the idiom both the KF and GF adapters use.
    """
    from gpuwm.core.physics import _model_clock_dt

    return _model_clock_dt(cfg)
